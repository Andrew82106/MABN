"""CPU-only reconstruction of frozen weights,110 epochs and120 alpha choices."""
from pathlib import Path
from collections import defaultdict
import importlib.util
import pickle
import sys
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'src'))
import run30 as run
q=run.q
helper=run.cache.QA/'results/minicheck_tail_all_docs_v3/postrun_audit.py'
spec=importlib.util.spec_from_file_location('r30_independent_metrics',helper)
ind=importlib.util.module_from_spec(spec);spec.loader.exec_module(ind)


def main():
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    run.check_prepared()
    for commit in ('complete.json','fit_complete.json'):
        for name,h in q.read(OUT/commit)['files_sha256'].items():assert q.sha(OUT/name)==h,name
    windows,answers,folds,aw=run.metadata()
    wy=np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows])
    ay=np.asarray([a['gold'] if a['main_eligible'] else -1 for a in answers])
    with np.load(OUT/'designs.npz') as d:raw=np.column_stack((d['hidden'],d['logit'])).astype(np.float32)
    wp={m:np.zeros(len(windows),bool) for m in run.METHODS}
    ap={m:np.zeros(len(answers),bool) for m in run.METHODS}
    wc=np.zeros(len(windows),int);ac=np.zeros(len(answers),int)
    reports=[];epochs_checked=0;alpha_checked=0;max_probability_delta=0.
    with threadpool_limits(limits=4):
        for fold,groups in enumerate(folds):
            pack=pickle.loads((OUT/f'fold_{fold}_prepared.pkl').read_bytes())
            control=pickle.loads((run.OLD/f'fold_{fold}_frozen.pkl').read_bytes())['model']
            frozen=q.read(OUT/f'fold_{fold}_calibration.json');assert pack['groups']==frozen['groups']==groups
            ix=np.asarray([i for i,w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups['fit_groups']])
            assert np.array_equal(ix,pack['window_fit_ix']) and np.array_equal(ix,control['fit_ix'])
            assert np.array_equal(pack['window_y'],wy)
            expected_window=np.zeros(len(windows));expected_window[ix]=control['loss_weights']
            assert np.array_equal(expected_window,pack['window_loss'])
            for name in ('mean_','var_','scale_'):
                assert np.array_equal(getattr(pack['scaler'],name),getattr(control['scaler'],name))
            b=control['base_weights'].astype(np.float32)
            mean=np.average(raw[ix].astype(np.float64),axis=0,weights=b)
            variance=np.average((raw[ix]-mean)**2,axis=0,weights=b)
            assert np.allclose(mean,pack['scaler'].mean_,rtol=0,atol=1e-10)
            assert np.allclose(variance,pack['scaler'].var_,rtol=1e-9,atol=1e-10)
            active_groups=defaultdict(list)
            for i,a in enumerate(answers):
                if a['main_eligible'] and a['group_id'] in groups['fit_groups']:active_groups[a['group_id']].append(i)
            base=np.zeros(len(answers))
            for ids in active_groups.values():base[ids]=1/(len(active_groups)*len(ids))
            factor=np.array([.5/base[ay==c].sum() for c in (0,1)])
            weight=np.zeros(len(answers))
            for ids in active_groups.values():
                local=base[ids]*factor[ay[ids]];weight[ids]=local/local.sum()/len(active_groups)
            assert np.array_equal(base,pack['answer']['base'])
            assert np.array_equal(factor,pack['answer']['class_factors'])
            assert np.allclose(weight,pack['answer']['loss'],rtol=0,atol=1e-16)
            assert np.array_equal(ay,pack['answer']['labels'])
            assert abs(weight.sum()-1)<1e-12 and abs(expected_window.sum()-3854)<1e-8
            for g in groups['fit_groups']:
                ids=[i for i,w in enumerate(windows) if w['group_id']==g]
                assert abs(expected_window[ids].sum()-3854/len(groups['fit_groups']))<1e-8
            refusal=[i for i,a in enumerate(answers) if a['reviewed_safe_refusal'] and a['group_id'] in groups['fit_groups']]
            assert all(weight[i]>0 and not expected_window[aw[answers[i]['item_id']]].any() for i in refusal)
            assert not weight[ay<0].any()
            steps=[]
            for order in pack['orders']:
                assert sorted(order)==sorted(groups['fit_groups'])
                bb=list(run.batches(order,answers,aw));steps.append(len(bb))
                ai=[i for p in bb for i in p['answer_indices']]
                wi=[i for p in bb for i in p['window_indices']]
                assert len(ai)==len(set(ai)) and len(wi)==len(set(wi))
                assert set(ai)=={i for i,a in enumerate(answers) if a['group_id'] in groups['fit_groups']}
                assert set(wi)=={i for i,w in enumerate(windows) if w['group_id'] in groups['fit_groups']}
                assert all(p['window_indices'][lo:hi].tolist()==aw[answers[i]['item_id']]
                           for p in bb for i,(lo,hi) in zip(p['answer_indices'],p['answer_local_bounds']))
            assert steps==[21]*10
            x=torch.from_numpy(pack['scaler'].transform(raw).astype(np.float32))
            ci=np.asarray([i for i,w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups['calibration_groups']])
            cai=np.asarray([i for i,a in enumerate(answers) if a['main_eligible'] and a['group_id'] in groups['calibration_groups']])
            ei=np.asarray([i for i,w in enumerate(windows) if w['group_id'] in groups['evaluation_groups']])
            eai=np.asarray([i for i,a in enumerate(answers) if a['group_id'] in groups['evaluation_groups']])
            wc[ei]+=1;ac[eai]+=1
            mode_rng={};selected={};mode_notes={}
            for mode in run.MODES:
                directory=OUT/f'fold_{fold}_{mode}';complete=q.read(directory/'complete.json')
                keys=[];epoch_notes=[];mode_rng[mode]=[]
                for epoch in range(11):
                    stem=f'epoch_{epoch:02d}'
                    checkpoint=torch.load(directory/(stem+'.pt'),map_location='cpu',weights_only=False)
                    assert checkpoint['epoch']==epoch and checkpoint['mode']==mode and checkpoint['fold']==fold
                    model=run.Probe();model.load_state_dict(checkpoint['model'],strict=True);model.eval()
                    if epoch==0:
                        assert all(torch.equal(v,pack['initial_state'][k]) for k,v in model.state_dict().items())
                        assert torch.equal(checkpoint['RNG'],pack['post_initialization_RNG'])
                    mode_rng[mode].append(checkpoint['RNG'])
                    state=checkpoint['optimizer']
                    assert len(state['param_groups'])==1
                    assert state['param_groups'][0]['lr']==.001 and state['param_groups'][0]['weight_decay']==.01
                    if not epoch:assert not state['state']
                    else:assert all(int(s['step'])==epoch*21 for s in state['state'].values())
                    with torch.no_grad():
                        z=model(x);v=torch.sigmoid(z).double().numpy()
                        target=torch.tensor(wy[ix],dtype=torch.float32)
                        ww=torch.tensor(expected_window[ix],dtype=torch.float32)
                        wl=float((F.binary_cross_entropy_with_logits(z[ix],target,reduction='none')*ww).sum()/3854)
                        ai=pack['answer']['eligible_indices'];az=torch.stack([z[aw[answers[i]['item_id']]].max() for i in ai])
                        at=torch.tensor(ay[ai],dtype=torch.float32)
                        # Preserve frozen loss casting; independent double weights
                        # above already verified before the float32 arithmetic.
                        aa=torch.tensor(pack['answer']['loss'][ai],dtype=torch.float32)
                        al=float((F.binary_cross_entropy_with_logits(az,at,reduction='none')*aa).sum())
                    saved=np.load(directory/(stem+'_probability.npy'));delta=float(np.abs(v-saved).max())
                    assert delta<=1e-12;max_probability_delta=max(max_probability_delta,delta)
                    entry=q.read(directory/(stem+'.json'));assert complete['history'][epoch]==entry
                    assert abs(wl-entry['fit_window_BCE'])<=1e-7 and abs(al-entry['fit_answer_BCE'])<=1e-7
                    assert abs(entry['fit_objective']-(wl+(al if mode=='window_plus_answer' else 0)))<=2e-7
                    av=np.asarray([v[aw[a['item_id']]].max() for a in answers])
                    wt,at=ind.threshold_from_roc(wy[ci],v[ci]),ind.threshold_from_roc(ay[cai],av[cai])
                    assert wt==entry['thresholds']['window']['threshold'] and at==entry['thresholds']['answer']['threshold']
                    wm,am=ind.metrics(wy[ci],v[ci],wt),ind.metrics(ay[cai],av[cai],at)
                    key=[min(wm['f1'],am['f1']),wm['f1'],wm['precision'],-epoch]
                    assert key==entry['key'];keys.append(key)
                    epoch_notes.append({'epoch':epoch,'fit_window_BCE':wl,'fit_answer_BCE':al,
                        'calibration_window_F1':wm['f1'],'calibration_answer_F1':am['f1']})
                    epochs_checked+=1
                best=max(range(1,11),key=lambda e:keys[e])
                assert complete['selected']==complete['history'][best]==frozen['selected_epochs'][mode]
                selected[mode]=best;mode_notes[mode]=epoch_notes
            assert all(torch.equal(a,b) for a,b in zip(mode_rng[run.MODES[0]],mode_rng[run.MODES[1]]))
            with np.load(OUT/f'fold_{fold}_scores.npz') as scores, np.load(run.R26/f'fold_{fold}_scores.npz') as old:
                for name in run.BASES:assert np.array_equal(scores[name],old[name])
                for mode in run.MODES:
                    direct=np.load(OUT/f'fold_{fold}_{mode}'/f'epoch_{selected[mode]:02d}_probability.npy')
                    assert np.array_equal(direct,scores[mode])
                    for prior,fused in zip(run.BASES,run.FUSIONS[mode]):
                        table=frozen['alpha_tables'][fused];keys=[]
                        for j,alpha in enumerate(run.ALPHA):
                            v=scores[fused+'_all_alpha'][j]
                            assert np.array_equal(v,run.previous.combine(scores[prior],direct,alpha))
                            av=np.asarray([v[aw[a['item_id']]].max() for a in answers])
                            wt,at=ind.threshold_from_roc(wy[ci],v[ci]),ind.threshold_from_roc(ay[cai],av[cai])
                            assert wt==table[j]['thresholds']['window']['threshold']
                            assert at==table[j]['thresholds']['answer']['threshold']
                            wm,am=ind.metrics(wy[ci],v[ci],wt),ind.metrics(ay[cai],av[cai],at)
                            key=[min(wm['f1'],am['f1']),wm['f1'],wm['precision'],-alpha]
                            assert key==table[j]['key'];keys.append(key);alpha_checked+=1
                        chosen=max(range(6),key=lambda j:keys[j])
                        assert table[chosen]==frozen['selected_alpha'][fused]
                        assert np.array_equal(scores[fused],scores[fused+'_all_alpha'][chosen])
                for name in run.METHODS:
                    v=scores[name];av=np.asarray([v[aw[a['item_id']]].max() for a in answers]);t=frozen['thresholds'][name]
                    assert ind.threshold_from_roc(wy[ci],v[ci])==t['window']['threshold']
                    assert ind.threshold_from_roc(ay[cai],av[cai])==t['answer']['threshold']
                    wp[name][ei]=v[ei]>=t['window']['threshold'];ap[name][eai]=av[eai]>=t['answer']['threshold']
            reports.append({'fold':fold,'assigned_fit_groups':len(groups['fit_groups']),
                'fit_safe_refusals_answer_supervised':len(refusal),'steps_each_mode':210,
                'all11_epoch_RNG_states_equal_between_modes':True,'selected_epochs':selected,
                'selected_alpha':{k:v['alpha'] for k,v in frozen['selected_alpha'].items()},'learning_curves':mode_notes})
    assert (wc==1).all() and (ac==1).all() and epochs_checked==110 and alpha_checked==120
    summary=q.read(OUT/'summary.json')['methods']
    for name in run.METHODS:
        assert run.cache.transfer.scene.counts(wy[wy>=0],wp[name][wy>=0])==summary[name]['windows']
        assert run.cache.transfer.scene.counts(ay[ay>=0],ap[name][ay>=0])==summary[name]['answers']
        assert sum(bool(ap[name][i]) for i,a in enumerate(answers) if a['reviewed_safe_refusal'])==summary[name]['safe_refusal_false_positives']
    assert not torch.cuda.is_initialized()
    q.save(OUT/'AUDIT.json',{'passed':True,'folds':reports,'checkpoint_epochs_replayed':epochs_checked,
        'alpha_candidates_checked':alpha_checked,'probability_replay_max_abs_difference':max_probability_delta,
        'independent_window_and_answer_weight_reconstruction':True,'fit_only_scaler_identity_checked':True,
        'all_epoch_thresholds_and_selected_epoch_ROC_reconstructed':True,
        'same_initial_state_step_count_and_dropout_RNG_at_every_epoch':True,
        'both_metrics_share_same_epoch_and_alpha_candidate':True,'every_outer_answer_and_window_once':True,
        'no_refit_no_new_selection_no_GPU':True,'human_gold':False,'original_validation_test_read_this_run':False,
        'complete_sha256':q.sha(OUT/'complete.json'),'audit_code_sha256':q.sha(Path(__file__))})
    print('R30_110_CHECKPOINTS_120_ALPHA_FROZEN_AUDIT_PASSED',flush=True)


if __name__=='__main__':main()
