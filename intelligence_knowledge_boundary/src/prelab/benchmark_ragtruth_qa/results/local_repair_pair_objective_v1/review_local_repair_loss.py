"""Bounded tensor-only audit: no data, tokenizer, encoder, GPU, or fitting."""
from pathlib import Path
import hashlib,importlib.util,json
import torch
import torch.nn.functional as F

OUT=Path(__file__).resolve().parent
SRC=OUT.parents[1]/'src/local_repair_pair_loss.py'
spec=importlib.util.spec_from_file_location('frozen_pair_objective',SRC)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
loss_fn=module.local_repair_loss

def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    original_check=json.loads((OUT/'CPU_CHECK.json').read_text('utf-8'))
    source_hash=hashlib.sha256(SRC.read_bytes()).hexdigest()
    assert source_hash==original_check['source_sha256']
    x=torch.tensor([-100.,1.,2.,3.,500.],dtype=torch.float32,requires_grad=True)
    y=torch.tensor([700.,-2.,0.,-900.],dtype=torch.float32,requires_grad=True)
    ix=torch.tensor([1,2,3]);iy=torch.tensor([1,2])
    control=loss_fn(x,y,ix,iy,ranking_weight=0.,margin=1.)
    ranked=loss_fn(x,y,ix,iy,ranking_weight=1.,margin=1.)
    expected_bce=.5*(F.softplus(-x[ix]).mean()+F.softplus(y[iy]).mean())
    expected_rank=F.softplus(1.-x[ix].mean()+y[iy].mean())
    assert torch.allclose(control['loss'],expected_bce,atol=1e-7,rtol=0)
    assert torch.equal(control['loss'],control['local_bce'])
    assert torch.equal(ranked['relative_rank'],expected_rank)
    assert torch.equal(ranked['loss'],ranked['local_bce']+ranked['relative_rank'])
    gc=torch.autograd.grad(control['loss'],(x,y),retain_graph=True)
    gr=torch.autograd.grad(ranked['relative_rank'],(x,y),retain_graph=True)
    gt=torch.autograd.grad(ranked['loss'],(x,y))
    expected_gc_x=torch.zeros_like(x);expected_gc_y=torch.zeros_like(y)
    expected_gc_x[ix]=.5*(x[ix].sigmoid()-1.)/len(ix)
    expected_gc_y[iy]=.5*y[iy].sigmoid()/len(iy)
    k=(1.-x[ix].mean()+y[iy].mean()).sigmoid().detach()
    expected_gr_x=torch.zeros_like(x);expected_gr_y=torch.zeros_like(y)
    expected_gr_x[ix]=-k/len(ix);expected_gr_y[iy]=k/len(iy)
    for actual,expected in zip(gc,(expected_gc_x,expected_gc_y)):
        assert torch.allclose(actual,expected,atol=1e-7,rtol=0)
    for actual,expected in zip(gr,(expected_gr_x,expected_gr_y)):
        assert torch.allclose(actual,expected,atol=1e-7,rtol=0)
    for total,c,r in zip(gt,gc,gr):assert torch.allclose(total,c+r,atol=1e-7,rtol=0)
    for gx,gy in (gc,gr,gt):
        assert torch.equal(gx[[0,4]],torch.zeros(2)) and torch.equal(gy[[0,3]],torch.zeros(2))
    assert bool((gt[0][ix]<0).all()) and bool((gt[1][iy]>0).all())
    # Replicating target values changes target counts, not either side's mass.
    x6=x.detach()[ix].repeat(2);y4=y.detach()[iy].repeat(2)
    repeated=loss_fn(x6,y4,torch.arange(6),torch.arange(4),1.,1.)
    assert torch.allclose(repeated['loss'],ranked['loss'],atol=1e-7,rtol=0)
    shift_results=[]
    for shift in (-8.,8.):
        shifted=loss_fn(x.detach()+shift,y.detach()+shift,ix,iy,1.,1.)
        assert torch.equal(shifted['relative_rank'],ranked['relative_rank'])
        assert not torch.equal(shifted['local_bce'],ranked['local_bce'])
        shift_results.append({'common_shift':shift,'rank':float(shifted['relative_rank']),
                              'local_bce':float(shifted['local_bce'])})
    # Off-target logits may differ arbitrarily without affecting either term.
    x2=x.detach().clone();y2=y.detach().clone();x2[[0,4]]=-12345.;y2[[0,3]]=12345.
    changed=loss_fn(x2,y2,ix,iy,1.,1.)
    assert torch.equal(changed['loss'],ranked['loss'])
    report={'status':'passed_tensor_only_independent_formula_review','source_sha256':source_hash,
        'review_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'unequal_targets_3_vs_2_equal_half_BCE_sides':True,
        'per_target_gradients_match_analytic_inverse_lengths':True,
        'duplicate_target_distribution_does_not_change_side_mass':True,
        'off_target_logit_gradient_exact_zero_for_BCE_rank_and_sum':True,
        'off_target_logit_values_do_not_change_loss':True,
        'lambda_zero_exact_paired_BCE':True,'rank_formula_margin1_matches':True,
        'common_shift_cancels_rank_only_not_BCE':True,'common_shift_checks':shift_results,
        'fixture':{'local_bce':float(ranked['local_bce'].detach()),'rank':float(ranked['relative_rank'].detach()),
                   'combined':float(ranked['loss'].detach()),'target_gradient_norm_BCE':float(torch.cat(gc).norm()),
                   'target_gradient_norm_BCE_plus_rank':float(torch.cat(gt).norm())},
        'interpretation':'Local silver-edit preference, not human factual truth. Direct output gradients are masked; shared encoder/context gradients can still change unlabelled positions.',
        'GPU_used':False,'encoder_loaded':False,'tokenizer_loaded':False,'real_data_read':False,'QA_cal_test_read':False,'trained':False}
    assert not torch.cuda.is_initialized()
    (OUT/'INDEPENDENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
