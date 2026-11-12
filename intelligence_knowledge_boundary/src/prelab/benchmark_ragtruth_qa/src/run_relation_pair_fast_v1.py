"""Relation-only FAVA local-pair transfer, sized for one RTX 3070 run.

The single pre-registered candidate uses local BCE plus a 0.25 ranking term.
No command schedules or automatically starts GPU work. Shared QA functions are
called with explicit inputs; no imported module globals or old outputs change.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json,math
from pathlib import Path
import shutil,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from transformers import ModernBertConfig,ModernBertForTokenClassification
import run_full_context_encoder_v2 as base
import run_auxiliary_transfer as shared
from local_repair_pair_loss import local_repair_loss
import local_repair_pair_loss as objective_module

q=base.q;mapping=base.mapping;evaluation=base.evaluation
ROOT=q.ROOT;OUT=ROOT/'results/relation_pair_fast_v1'
DATA=ROOT/'auxiliary_fava_local_pairs_v1';TOKENS=DATA/'tokenization_v1'
VARIANTS={'relation_rank025':.25}
BATCH=8;NQA=3680;QA_EPOCHS=3;PAIR_ORDER_SEED=20261009;MARGIN=1.
RELATION_CANDIDATES=4705
RELATION_ELIGIBLE=4704

def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def fresh_optimizer(model):return torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def require_tokens():
    if not (TOKENS/'complete.json').exists() or not (TOKENS/'CPU_OUTPUT_CHECK.json').exists():
        raise RuntimeError('WAIT: complete pair tokenization and full CPU output check required; no partial prepare or training')
    c=q.read(TOKENS/'complete.json');m=q.read(TOKENS/'manifest.json');checked=q.read(TOKENS/'CPU_OUTPUT_CHECK.json')
    assert c['status']=='CPU_tokenization_complete_not_trained' and c['candidate_pairs']==10040 and c['input_records']==20080
    assert c['manifest_sha256']==q.sha(TOKENS/'manifest.json')==checked['manifest_sha256']
    assert m['status']=='all_candidates_tokenized_not_trained' and checked['status']=='passed' and checked['all_input_records']==20080
    assert not c['exceptions_require_review'] and not m['mapping_or_length_exceptions'],'Preserve exception cases; review before changing protocol'
    assert not m['GPU_used'] and not m['trained'] and not m['QA_fit_cal_test_read']
    assert m['candidate_manifest_sha256']==q.sha(DATA/'manifest.json')
    for n,h in m['files_sha256'].items():assert q.sha(TOKENS/n)==h,n
    freeze=q.read(TOKENS/'token_design_freeze.json')
    assert freeze['protocol_sha256']==q.sha(TOKENS/'token_protocol.json')
    for p,h in freeze['source_sha256'].items():assert q.sha(p)==h,p
    return m

def source_files():
    # Base source snapshot is existing QA fit/cal preparation, never official test.
    snap=base.source_files()
    paths=[Path(__file__),Path(shared.__file__),Path(objective_module.__file__),
           ROOT/'results/local_repair_pair_objective_v1/CPU_CHECK.json',
           ROOT/'results/local_repair_pair_objective_v1/INDEPENDENT_REVIEW.json',
           DATA/'complete.json',DATA/'manifest.json',DATA/'candidate_pairs.jsonl',
           TOKENS/'complete.json',TOKENS/'manifest.json',TOKENS/'CPU_OUTPUT_CHECK.json',
           TOKENS/'paired_index.jsonl',TOKENS/'token_inputs.jsonl',TOKENS/'token_design_freeze.json',
           TOKENS/'token_protocol.json',TOKENS/'nfc_repairs.jsonl',TOKENS/'eligible_material_group_index.jsonl',
           Path(__file__).with_name('prepare_fava_local_pair_tokens_v1.py'),
           Path(__file__).with_name('fava_nfc_character_map.py'),base.OUT/'preparation_complete.json',
           base.OUT/'inputs.jsonl',base.OUT/'training_weights.npz',base.OUT/'answer_orders.npy',base.OUT/'protocol.json',
           ROOT/'results/full_context_fava_transfer_v1/transfer/complete.json',
           ROOT/'results/full_context_aux_transfer_v1/transfer/complete.json']
    snap.update({str(p.resolve()):q.sha(p) for p in paths});return snap

def protocol(cohort):
    n=cohort['eligible_pairs']
    return {'version':'relation-pair-fast-transfer-v1','variants':VARIANTS,'margin':MARGIN,
        'initialization':'Fresh generic ModernBERT-base via original v2.load_model, pretrained revision and seed20261005. No QA-fitted initialization.',
        'paired_data':cohort,'pair_order_seed':PAIR_ORDER_SEED,
        'selection_before_QA':'Use every structurally eligible upstream pair whose frozen FAVA top-level author tag is exactly relation. Entity pairs are excluded. This rule was fixed without reading QA calibration labels, scores or text.',
        'local_supervision':'Each original/preferred side has only its own target_raw_lexical_indices. z1-z0 FP32 -> original answer-character map -> each side raw coordinates. No repaired whole-answer or outside-target zero labels.',
        'loss':'Exactly frozen local_repair_pair_loss: half original local BCE(y=1)+half repaired local BCE(y=0), plus fixed lambda0.25*softplus(1-mean_bad+mean_repaired). Lambda0.25 is pre-registered from the old label-free TinyModernBERT audit where lambda1 made the gradient norm about3.2x BCE. It is not selected on QA results. No new class factors. Silver editing roles are not certified truth.',
        'pair_weights':'Recomputed inside the eligible relation-only cohort: equal material groups -> equal original answers within group -> equal relation pairs per original answer. Both versions are one unit. Sum base weights1.',
        'aux_minibatch':'For batch B_actual complete pairs, sum_i Npairs/B_actual * normalized_pair_weight[i] * pair_loss[i]. Two separate full forwards per pair, retain both graphs until one backward. 8pairs accumulation; final actual pair count used, not8. One fixed auxiliary epoch.',
        'single_candidate':'One relation-only candidate, no lambda grid and no second auxiliary branch. Compare its final common QA metrics with already frozen development results. This does not isolate relation filtering from paired training.',
        'precision':'BF16 CUDA model forward/non-reentrant checkpoint recomputation; FP32 parameters, gradients, optimizer states, both output logits before subtraction, mapping, BCE and rank. CPU FP32. TF32 off. No GradScaler.',
        'optimizer':{'type':'AdamW','lr':1e-5,'weight_decay':.01,'clip':1.,'effective_batch_pairs':BATCH,
                     'scheduler':None,'continuous_across_aux_and_QA':True,'aux_updates':math.ceil(n/BATCH),
                     'last_aux_batch_pairs':n%BATCH or BATCH,'QA_updates_each':460,'updates_total_each':math.ceil(n/BATCH)+1380},
        'QA':{'fit_answers':3680,'fit_groups':615,'calibration_answers':159,'epochs':3,'mass':560300,
              'reuse':'QA inputs, weights and all6 stored orders copied byte-exact; use only original first3 orders, no optimizer/RNG reset at transition.',
              'evaluation':'Unchanged raw4BPE stride1 lexical max; answermax; original labels and refusals. No test opened.',
              'selection':'Only after all3 QA epochs complete, select one epoch by min(windowF1,answerF1),windowF1,windowprecision,earlier epoch. Each epoch uses original separate cal thresholds and precision/higher-cutoff ties. No auxiliary cal evaluation or checkpoint selection.'},
        'artifacts':'Keep auxiliary_final and QA1/2/3 model+optimizer+CPU/CUDA RNG; all QA raw-token predictions, geometry scores, thresholds, logs, logical tokens/time/peak memory. No automatic resume or overwrite.',
        'GPU_gate':'Explicit longest-complete-pair by sum of two encoder lengths (then max length, first index) paired two-graph smoke; repeat each side <=2e-6, nonzero finite gradient, BF16 forward and checkpoint replay, FP32 optimizer. Failure preserved; no truncation or altered precision/budget.',
        'interpretation':'Paired local silver supervision; not human factual gold. Rank is shift-invariant alone, full BCE+rank is not. Adding rank changes gradient strength and shape; do not attribute results exclusively to contrast reasoning. Context/shared-parameter indirect gradients remain.',
        'runtime_basis':'RTX3070 estimate is computed from two completed ModernBERT runs on this machine, not from model FLOPs. It remains an estimate because the paired path has not yet received a real GPU smoke.',
        'scope':'CPU preparation only until separate explicit GPU commands. No baseline, official test, extra grid or score-combination branch.',
        'automatic_GPU_or_training':False,'official_test_opened':False}

class IndexedPairs:
    """Only single-version token records are passed to base.logits."""
    def __init__(self,path,index):self.stream=Path(path).open('rb');self.index=index
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        p=self.index[int(i)];result={'pair_id':p['pair_id'],'source_response_id':p['source_response_id'],'group_id':p['group_id']}
        for side in ('original','preferred'):
            desc=p[side];self.stream.seek(desc['byte_offset']);raw=self.stream.read(desc['byte_length'])
            assert sha_bytes(raw)==desc['record_bytes_sha256']
            row=json.loads(raw)
            assert row['input_id']==desc['input_id'] and row['pair_id']==p['pair_id'] and row['side']==side
            assert row['source_response_id']==p['source_response_id'] and row['group_id']==p['group_id']
            assert row['mapping_complete'] and not row['exceptions'] and row['local_lexical_supervision_available']
            assert 'risk_mask' not in row and 'answer_risk' not in row and not row['full_answer_labels_created']
            assert row['outside_target']=='unknown_no_binary_supervision'
            assert row['target']==desc['target'] and len(row['input_ids'])==desc['input_tokens']<=8192
            assert row['attention_mask']==[1]*len(row['input_ids'])
            ids=row['target_raw_lexical_indices'];n=len(row['response_token_ids'])
            assert n==desc['raw_answer_tokens'] and len(ids)==desc['target_raw_lexical_count']>0
            assert ids==sorted(set(ids)) and min(ids)>=0 and max(ids)<n
            assert all(row['lexical_mask'][j] for j in ids)
            text=row['model_inputs']['response'];target=row['target'];assert text[target['start']:target['end']]==target['text']
            wanted=[j for j,(lo,hi) in enumerate(row['response_token_offsets'])
                    if any(c.isalnum() for c in text[max(lo,target['start']):min(hi,target['end'])])
                    and max(lo,target['start'])<min(hi,target['end'])]
            assert ids==wanted
            # base.logits requires this alias; its input/mapping/precision are untouched.
            row['raw_token_count']=n;result[side]=row
        return result
    def close(self):self.stream.close()

def pair_base_weights(index):
    tree=defaultdict(lambda:defaultdict(list))
    for j,p in enumerate(index):tree[p['group_id']][p['source_response_id']].append(j)
    w=np.zeros(len(index),np.float64)
    for answers in tree.values():
        for ix in answers.values():w[ix]=1/(len(tree)*len(answers)*len(ix))
    assert w.size and (w>0).all() and abs(w.sum()-1)<1e-12
    return w,tree

def prepare():
    assert not torch.cuda.is_initialized();tm=require_tokens()  # No partial output if upstream is incomplete.
    assert not (OUT/'prepare_started.json').exists(),'Do not overwrite an existing prepared run'
    assert shared.base is base and shared.BATCH==BATCH and base.SEED==20261005
    assert q.sha(objective_module.__file__)==q.read(ROOT/'results/local_repair_pair_objective_v1/INDEPENDENT_REVIEW.json')['source_sha256']
    old=base.check_prepared();assert old['answers']==3839
    allpairs=q.lines(TOKENS/'paired_index.jsonl');assert len(allpairs)==10040
    candidates=[p for p in allpairs if p['type']=='relation']
    assert len(candidates)==RELATION_CANDIDATES
    index=[p for p in candidates if p['eligible_for_future_local_pair_training']]
    assert len(index)==RELATION_ELIGIBLE and all(not p['ineligibility_reasons'] for p in index)
    weights,tree=pair_base_weights(index)
    n=len(index);ng=len(tree);na=sum(len(x) for x in tree.values())
    snap=source_files();OUT.mkdir(parents=True,exist_ok=True)
    q.save(OUT/'prepare_started.json',{'source_sha256':snap,'token_manifest_sha256':q.sha(TOKENS/'manifest.json')})
    ds=IndexedPairs(TOKENS/'token_inputs.jsonl',index);lengths=[];raw_total=targets=0
    try:
        for i in range(n):
            pair=ds[i];lens=[]
            for side in ('original','preferred'):
                row=pair[side];lens.append(len(row['input_ids']));raw_total+=row['raw_token_count'];targets+=len(row['target_raw_lexical_indices'])
            lengths.append(lens)
            if (i+1)%2000==0:print('PAIR_TRANSFER_VERIFY',i+1,n,flush=True)
    finally:ds.close()
    longest=max(range(n),key=lambda j:(sum(lengths[j]),max(lengths[j]),-j))
    cohort={'all_upstream_candidate_pairs':10040,'candidate_relation_pairs':len(candidates),'eligible_pairs':n,
            'excluded_relation_pairs_retained_upstream':len(candidates)-n,'entity_pairs_not_selected':10040-len(candidates),
            'original_answers':na,'material_groups':ng,'encoder_input_tokens':sum(map(sum,lengths)),
            'raw_answer_tokens_both_versions':raw_total,'local_target_raw_tokens_both_versions':targets,
            'max_side_input_tokens':max(map(max,lengths)),'longest_pair_index':longest,
            'longest_pair_id':index[longest]['pair_id'],'longest_pair_lengths':lengths[longest],
            'source_isolation_inherited_not_rescanned':True,'token_manifest_sha256':q.sha(TOKENS/'manifest.json')}
    q.save(OUT/'pair_index.json',{'pairs':index,'cohort':cohort,'token_source':str((TOKENS/'token_inputs.jsonl').resolve())})
    np.savez_compressed(OUT/'pair_weights.npz',base=weights,target_mass=np.asarray(1.,np.float64))
    np.save(OUT/'pair_order.npy',np.random.default_rng(PAIR_ORDER_SEED).permutation(n))
    for oldname,newname in [('inputs.jsonl','qa_inputs.jsonl'),('training_weights.npz','qa_training_weights.npz'),('answer_orders.npy','qa_answer_orders.npy')]:
        shutil.copyfile(base.OUT/oldname,OUT/newname);assert q.sha(base.OUT/oldname)==q.sha(OUT/newname)
    qa_rows=q.lines(OUT/'qa_inputs.jsonl');qa_orders=np.load(OUT/'qa_answer_orders.npy');qw=load_npz(OUT/'qa_training_weights.npz')
    assert len(qa_rows)==3839 and qa_orders.shape==(6,NQA) and int(qw['target_mass'])==560300
    qfit=sum(len(r['input_ids']) for r in qa_rows[:NQA]);qeval=sum(len(r['input_ids']) for r in qa_rows)
    oldckpt=base.OUT/'full_finetune/epoch_01.pt';oldsize=oldckpt.stat().st_size if oldckpt.exists() else None
    reference=[]
    for path in (ROOT/'results/full_context_fava_transfer_v1/transfer/complete.json',
                 ROOT/'results/full_context_aux_transfer_v1/transfer/complete.json'):
        value=q.read(path);aux=value['auxiliary'];qa_seconds=sum(e['seconds'] for e in value['all_QA_epochs'])
        reference.append({'path':str(path.resolve()),'sha256':q.sha(path),
                          'aux_tokens_per_second':aux['logical_input_tokens']/aux['seconds'],
                          'QA_three_epochs_seconds':qa_seconds,'total_seconds':value['seconds']})
    slowest=min(x['aux_tokens_per_second'] for x in reference)
    qa_max=max(x['QA_three_epochs_seconds'] for x in reference)
    point_seconds=cohort['encoder_input_tokens']/slowest+qa_max
    conservative_seconds=(cohort['encoder_input_tokens']/slowest)*1.35+qa_max*1.15+600
    resources={'relation_only_single_candidate':True,'QA_three_files_byte_identical':True,
        'auxiliary_forward_calls_each':2*n,'QA_train_forward_calls_each':3*NQA,'QA_evaluation_forward_calls_each':3*3839,
        'logical_training_input_tokens_each':cohort['encoder_input_tokens']+3*qfit,
        'aux_logical_input_tokens_each':cohort['encoder_input_tokens'],'QA_training_input_tokens_each_epoch':qfit,
        'QA_evaluation_input_tokens_each_epoch':qeval,'logical_evaluation_input_tokens_each':3*qeval,
        'aux_optimizer_updates_each':math.ceil(n/BATCH),'QA_updates_each':1380,'total_updates_each':math.ceil(n/BATCH)+1380,
        'longest_complete_pair':cohort['longest_pair_lengths'],'checkpoints':4,'variants':1,
        'existing_same_model_QA_checkpoint_bytes':oldsize,
        'estimated_four_checkpoint_bytes':4*oldsize if oldsize else None,
        'free_disk_bytes_at_prepare':shutil.disk_usage(OUT).free,
        'same_host_completed_run_basis':reference,
        'point_runtime_seconds_from_slowest_observed_token_rate_plus_max_QA3':point_seconds,
        'conservative_runtime_seconds_with_pair_contingency_and_10min_overhead':conservative_seconds,
        'expected_runtime_minutes':[math.ceil(point_seconds/60),math.ceil(conservative_seconds/60)],
        'GPU_estimate_not_measured_for_pair_path':True,
        'GPU_memory':'8GB target, same full base+FP32 Adam but two checkpointed forward graphs alive together. Only explicit longest-pair smoke can confirm actual fit; no memory claim before smoke.',
        'time_limit':'Actual pair-path runtime remains unmeasured. Abort/report if explicit smoke extrapolation exceeds60 minutes; do not silently shrink the cohort or epochs.',
        'gradient_strength_limit':'Lambda0.25 still changes gradient direction and magnitude. It is a pre-registered candidate, not a scale-invariant ablation.',
        'trained':False,'GPU_used':False}
    q.save(OUT/'WEIGHTS_AND_RESOURCE_PLAN.json',resources);q.save(OUT/'protocol.json',protocol(cohort))
    assert snap==source_files();q.save(OUT/'source_snapshot.json',{'files_sha256':snap,'official_test_opened':False})
    names=['pair_index.json','pair_weights.npz','pair_order.npy','qa_inputs.jsonl','qa_training_weights.npz','qa_answer_orders.npy','WEIGHTS_AND_RESOURCE_PLAN.json','protocol.json','source_snapshot.json']
    q.save(OUT/'preparation_complete.json',{'status':'prepared_not_trained','cohort':cohort,'variants':list(VARIANTS),
        'files_sha256':{n:q.sha(OUT/n) for n in names},'GPU_used':False,'official_test_opened':False})
    assert not torch.cuda.is_initialized();print('RELATION_PAIR_FAST_PREPARED',json.dumps(cohort),flush=True)

def check_prepared():
    p=q.read(OUT/'preparation_complete.json');assert p['status']=='prepared_not_trained'
    assert q.read(OUT/'protocol.json')==protocol(p['cohort'])
    for n,h in p['files_sha256'].items():assert q.sha(OUT/n)==h,n
    assert q.read(OUT/'source_snapshot.json')['files_sha256']==source_files()
    return p

def train_pair_epoch(model,optimizer,rows,order,weights,device,variant,progress=True,audit_hook=None):
    """Production auxiliary loop, also exercised with TinyModernBERT on CPU."""
    assert variant in VARIANTS;n=len(order);assert sorted(np.asarray(order).tolist())==list(range(n))
    assert len(rows)==n==len(weights) and np.isfinite(weights).all() and (weights>0).all() and abs(float(np.sum(weights))-1)<1e-12
    model.train();tick=time.perf_counter();steps=logical=raw=target_count=0;online=Counter();batches=[];seen=[]
    for first in range(0,n,BATCH):
        chosen=order[first:first+BATCH];optimizer.zero_grad(set_to_none=True);batch_scale=0.
        for i in chosen:
            i=int(i);p=rows[i];za=base.logits(model,p['original'],device);zb=base.logits(model,p['preferred'],device)
            ia=torch.as_tensor(p['original']['target_raw_lexical_indices'],device=device,dtype=torch.long)
            ib=torch.as_tensor(p['preferred']['target_raw_lexical_indices'],device=device,dtype=torch.long)
            if audit_hook is not None:za.retain_grad();zb.retain_grad()
            value=local_repair_loss(za,zb,ia,ib,ranking_weight=VARIANTS[variant],margin=MARGIN)
            scale=n*float(weights[i])/len(chosen);weighted=value['loss']*scale
            assert weighted.dtype==torch.float32 and torch.isfinite(weighted);weighted.backward()
            for key in ('loss','local_bce','relative_rank'):online[key]+=float(value[key].detach())*scale
            if audit_hook is not None:audit_hook(p,za,zb,ia,ib,scale)
            batch_scale+=scale;seen.append(p['pair_id']);logical+=sum(len(p[s]['input_ids']) for s in ('original','preferred'))
            raw+=len(za)+len(zb);target_count+=len(ia)+len(ib)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm) and norm>0
        optimizer.step();steps+=1
        batches.append({'pairs':len(chosen),'summed_scale':batch_scale,'gradient_norm_before_clip':float(norm)})
        if progress and ((first+len(chosen))%400==0 or first+len(chosen)==n):
            print('LOCAL_PAIR_TRAIN',variant,first+len(chosen),n,round(time.perf_counter()-tick,1),flush=True)
    base.check_fp32_state(model,optimizer)
    return {'phase':'auxiliary','variant':variant,'ranking_weight':VARIANTS[variant],'margin':MARGIN,
        'optimizer_updates':steps,'pair_presentations':n,'forward_calls':2*n,'logical_input_tokens':logical,
        'raw_answer_tokens':raw,'local_target_raw_tokens':target_count,'online_component_sums':dict(online),
        'online_mean_batch_objective':online['loss']/steps,'loss_scope':'Online batch loss, not deterministic full-fit loss',
        'batches':batches,'seen_pair_order_sha256':q.digest(seen),'seconds':time.perf_counter()-tick}

def save_checkpoint(model,optimizer,path,phase,epoch,updates,variant):
    assert not path.exists()
    state={'model_state_dict':evaluation.cpu_state(model.state_dict()),'optimizer_state_dict':evaluation.cpu_state(optimizer.state_dict()),
           'phase':phase,'phase_epoch':epoch,'global_optimizer_updates':updates,'variant':variant,
           'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state().cpu(),
           'preparation_sha256':q.sha(OUT/'preparation_complete.json')}
    torch.save(state,path)

def evaluate_QA(model,rows,weights,answers,tokens,device='cuda'):
    model.eval();probabilities={};fit_bce=0.;logical=0
    with torch.no_grad():
        for i,row in enumerate(rows):
            z=base.logits(model,row,device);assert z.dtype==torch.float32 and torch.isfinite(z).all()
            probabilities[row['response_id']]=torch.sigmoid(z).cpu().numpy();logical+=len(row['input_ids'])
            if i<NQA:
                lo,hi=weights['bounds'][i];y=torch.as_tensor(weights['y'][lo:hi],device=device,dtype=torch.float32)
                w=torch.as_tensor(weights['loss'][lo:hi],device=device,dtype=torch.float32)
                fit_bce+=float((F.binary_cross_entropy_with_logits(z,y,reduction='none')*w).double().sum())
            if (i+1)%400==0:print('LOCAL_PAIR_QA_EVAL',i+1,len(rows),flush=True)
    fit=evaluation.score_geometry(answers,tokens,probabilities,range(NQA))
    cal=evaluation.score_geometry(answers,tokens,probabilities,range(NQA,len(rows)))
    assert len(fit['window_scores'])==653979 and len(cal['window_scores'])==42241 and len(cal['answer_scores'])==159
    ts={'window':q.choose_threshold(cal['window_labels'],cal['window_scores']),
        'answer':q.choose_threshold(cal['answer_labels'],cal['answer_scores'])}
    return probabilities,fit,cal,ts,fit_bce/int(weights['target_mass']),logical

def train(variant):
    p=check_prepared();assert variant in VARIANTS
    gate=q.read(OUT/'GPU_SELFCHECK.json');cpu=q.read(OUT/'CPU_SELFCHECK.json')
    assert gate['passed'] and cpu['passed'] and gate['preparation_sha256']==cpu['preparation_sha256']==q.sha(OUT/'preparation_complete.json')
    directory=OUT/variant;directory.mkdir(exist_ok=True)
    assert not (directory/'started.json').exists(),'No automatic resume/overwrite'
    estimate=q.read(OUT/'WEIGHTS_AND_RESOURCE_PLAN.json')['existing_same_model_QA_checkpoint_bytes']
    if estimate:assert shutil.disk_usage(directory).free>4*estimate+2*2**30
    base.configure_gpu();model=optimizer=ds=None
    try:
        rows=q.lines(OUT/'qa_inputs.jsonl');answers,tokens,_=mapping.metadata()
        assert [r['response_id'] for r in rows]==[a['response_id'] for a in answers]
        qw=load_npz(OUT/'qa_training_weights.npz');pw=load_npz(OUT/'pair_weights.npz')['base']
        orders=np.load(OUT/'qa_answer_orders.npy');po=np.load(OUT/'pair_order.npy');index=q.read(OUT/'pair_index.json')['pairs']
        ds=IndexedPairs(TOKENS/'token_inputs.jsonl',index)
        model=base.load_model().cuda();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=fresh_optimizer(model);tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        q.save(directory/'started.json',{'variant':variant,'preparation_sha256':q.sha(OUT/'preparation_complete.json'),
            'initial_seed':base.SEED,'aux_order_sha256':q.sha(OUT/'pair_order.npy'),'time':time.time(),'no_auxiliary_calibration':True})
        aux=train_pair_epoch(model,optimizer,ds,po,pw,'cuda',variant);ds.close();ds=None
        updates=aux['optimizer_updates'];expected_aux=math.ceil(len(index)/BATCH)
        assert updates==expected_aux==shared.optimizer_steps(optimizer)
        aux['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        save_checkpoint(model,optimizer,directory/'auxiliary_final.pt','auxiliary',1,updates,variant)
        aux.update(checkpoint_sha256=q.sha(directory/'auxiliary_final.pt'),calibration_evaluated=False,checkpoint_selected=False)
        q.save(directory/'auxiliary_final.json',aux);print('LOCAL_PAIR_AUX_COMPLETE',variant,updates,flush=True)
        history=[]
        for epoch in range(1,QA_EPOCHS+1):
            et=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            assert shared.optimizer_steps(optimizer)==expected_aux+(epoch-1)*460
            training=shared.train_epoch(model,optimizer,rows,orders[epoch-1],qw,'cuda',f'{variant}_QA{epoch}')
            updates+=training['optimizer_updates'];assert training['optimizer_updates']==460 and shared.optimizer_steps(optimizer)==updates
            prob,fit,cal,ts,bce,logical=evaluate_QA(model,rows,qw,answers,tokens)
            stem=f'qa_epoch_{epoch:02d}';save_checkpoint(model,optimizer,directory/(stem+'.pt'),'QA',epoch,updates,variant)
            np.savez_compressed(directory/(stem+'_token_predictions.npz'),**prob)
            np.savez_compressed(directory/(stem+'_scores.npz'),**{'fit_'+k:v for k,v in fit.items()},**{'cal_'+k:v for k,v in cal.items()})
            entry={'epoch':epoch,'variant':variant,'phase':'QA','thresholds':ts,
                'selection_key':[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-epoch],
                'fit_at_cal_thresholds':evaluation.metrics_for(fit,ts),'calibration':evaluation.metrics_for(cal,ts),
                'fit_weighted_bce':bce,'training':training,'global_optimizer_updates':updates,
                'evaluation_logical_input_tokens':logical,'seconds':time.perf_counter()-et,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'artifacts_sha256':{ext:q.sha(directory/(stem+ext)) for ext in ('.pt','_token_predictions.npz','_scores.npz')},'official_test_opened':False}
            q.save(directory/(stem+'.json'),entry);history.append(entry)
            print('LOCAL_PAIR_QA_COMPLETE',variant,epoch,entry['calibration']['windows']['f1'],entry['calibration']['answers']['f1'],flush=True)
        assert len(history)==3 and updates==expected_aux+1380
        selected=max(history,key=lambda e:e['selection_key'])
        q.save(directory/'complete.json',{'status':'complete_development_only','variant':variant,'selected':selected,'all_QA_epochs':history,
            'auxiliary':aux,'optimizer_updates':updates,'optimizer_state_continued':True,
            'actual_training_logical_input_tokens':aux['logical_input_tokens']+sum(e['training']['logical_input_tokens'] for e in history),
            'actual_evaluation_logical_input_tokens':sum(e['evaluation_logical_input_tokens'] for e in history),
            'seconds':time.perf_counter()-tick,'preparation_sha256':q.sha(OUT/'preparation_complete.json'),'official_test_opened':False})
    finally:
        if ds is not None:ds.close()
        del model,optimizer;torch.cuda.empty_cache()
    print('LOCAL_PAIR_TRANSFER_COMPLETE_GPU_RELEASED',variant,flush=True)

def gpu_smoke():
    p=check_prepared();assert q.read(OUT/'CPU_SELFCHECK.json')['passed']
    assert not (OUT/'GPU_SELFCHECK.json').exists(),'Preserve completed smoke'
    base.configure_gpu();model=opt=None;handle=None
    try:
        idx=q.read(OUT/'pair_index.json')['pairs'];ds=IndexedPairs(TOKENS/'token_inputs.jsonl',idx)
        try:pair=ds[p['cohort']['longest_pair_index']]
        finally:ds.close()
        assert [len(pair[s]['input_ids']) for s in ('original','preferred')]==p['cohort']['longest_pair_lengths']
        model=base.load_model().cuda();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.eval()
        repeats={}
        with torch.no_grad():
            for side in ('original','preferred'):
                a=base.logits(model,pair[side],'cuda');b=base.logits(model,pair[side],'cuda')
                repeats[side]=float((a-b).abs().max());assert repeats[side]<=2e-6 and torch.isfinite(a).all()
        calls=[]
        handle=model.model.layers[0].register_forward_pre_hook(lambda m,x:calls.append((torch.is_autocast_enabled('cuda'),str(torch.get_autocast_dtype('cuda')))))
        opt=fresh_optimizer(model);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        # Exactly the single production paired two-graph path;
        # no QA labels/scores, no checkpoint/parameter selection in this gate.
        run=train_pair_epoch(model,opt,[pair],np.array([0]),np.array([1.]),'cuda','relation_rank025',progress=False)
        torch.cuda.synchronize();assert shared.optimizer_steps(opt)==1
        assert len(calls)>=4 and all(on and dtype=='torch.bfloat16' for on,dtype in calls)
        dtype=base.check_fp32_state(model,opt)
        q.save(OUT/'GPU_SELFCHECK.json',{'passed':True,'pair_id':pair['pair_id'],'input_tokens':p['cohort']['longest_pair_lengths'],
            'production_two_forward_graphs_backward':True,'ranking_weight':.25,'margin':1.,'no_QA_gold_or_calibration_used':True,
            'repeated_max_abs_difference':repeats,'gradient_norm_before_clip':run['batches'][0]['gradient_norm_before_clip'],
            'step_seconds':time.perf_counter()-tick,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'dtype_check':dtype,'checkpoint_forward_recompute_BF16':True,'preparation_sha256':q.sha(OUT/'preparation_complete.json'),'official_test_opened':False})
    finally:
        if handle is not None:handle.remove()
        del model,opt;torch.cuda.empty_cache()
    print('LOCAL_PAIR_GPU_SMOKE_PASSED_GPU_RELEASED',flush=True)

def cpu_test():
    p=check_prepared();assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    # 11pairs -> 8+3 updates, unequal group/answer/pair counts, unequal masks.
    fixture=[]
    for i in range(11):
        def side(preferred):
            ids=[1,5,7,11,13,17,19+(i%3),23 if preferred else 29,2]
            target=[1] if preferred else ([1,2] if i%2 else [2])
            return {'input_ids':ids,'raw_token_count':4,'mapping':[[0,1,2,3],[4,5,6,7],[1.,1.,1.,1.]],
                    'target_raw_lexical_indices':target}
        fixture.append({'pair_id':f'p{i}','group_id':'g0' if i<8 else 'g1',
                        'source_response_id':'a0' if i<6 else 'a1' if i<8 else 'a2',
                        'original':side(False),'preferred':side(True)})
    weights,tree=pair_base_weights(fixture);assert len(tree)==2
    assert abs(weights[:8].sum()-.5)<1e-12 and abs(weights[8:].sum()-.5)<1e-12
    assert abs(weights[:6].sum()-.25)<1e-12 and abs(weights[6:8].sum()-.25)<1e-12
    config=ModernBertConfig(vocab_size=41,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
        max_position_embeddings=64,pad_token_id=0,num_labels=2,local_attention=16,reference_compile=False)
    config._attn_implementation='sdpa'
    initial=None;reports={};order=np.array([3,10,1,5,0,9,2,8,4,7,6],np.int64)
    qa_rows=[dict(fixture[i]['original'],response_id=f'qa{i}') for i in range(3)]
    qw={'bounds':np.array([[0,4],[4,8],[8,12]]),'y':np.array([0,1,0,1]*3),'loss':np.ones(12),'target_mass':np.array(12)}
    for variant in VARIANTS:
        torch.manual_seed(base.SEED);model=ModernBertForTokenClassification(config).float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if initial is None:initial=state
        else:assert all(torch.equal(initial[k],v) for k,v in state.items())
        opt=fresh_optimizer(model);identity=id(opt);mask_checks=[]
        def hook(pair,za,zb,ia,ib,scale):
            for z,ids,direction in ((za,ia,-1),(zb,ib,1)):
                outside=torch.ones(len(z),dtype=torch.bool);outside[ids]=False
                assert torch.equal(z.grad[outside],torch.zeros_like(z.grad[outside]))
                assert torch.isfinite(z.grad).all() and bool((z.grad[ids]*direction>0).all())
            mask_checks.append(pair['pair_id'])
        aux=train_pair_epoch(model,opt,fixture,order,weights,'cpu',variant,progress=False,audit_hook=hook)
        assert [b['pairs'] for b in aux['batches']]==[8,3] and aux['optimizer_updates']==shared.optimizer_steps(opt)==2
        for b,first in zip(aux['batches'],(0,8)):
            chosen=order[first:first+BATCH];assert abs(b['summed_scale']-11*weights[chosen].sum()/len(chosen))<1e-12
        assert mask_checks==[fixture[i]['pair_id'] for i in order]
        param=next(model.parameters());state_identity=id(opt.state[param]);steps=[2]
        for epoch in range(3):
            result=shared.train_epoch(model,opt,qa_rows,np.array([2,0,1]),qw,'cpu',f'tiny_QA{epoch+1}',progress=False)
            assert result['optimizer_updates']==1 and id(opt)==identity and id(opt.state[param])==state_identity
            steps.append(shared.optimizer_steps(opt));assert steps[-1]==epoch+3
        reports[variant]={'steps_aux_then_QA':steps,'batches':aux['batches'],'pair_order_sha256':aux['seen_pair_order_sha256'],
                          'forward_calls':aux['forward_calls'],'logical_input_tokens':aux['logical_input_tokens'],'local_masks_checked':len(mask_checks)}
    assert list(reports)==['relation_rank025'] and reports['relation_rank025']['forward_calls']==22
    assert not torch.cuda.is_initialized()
    q.save(OUT/'CPU_SELFCHECK.json',{'passed':True,'production_pair_loop_TinyModernBERT':True,'same_initial_states':True,
        'both_local_masks_direct_gradients_checked':True,'outside_target_direct_gradient_zero':True,'groups_answers_pairs_mass_checked':True,
        'eight_plus_three_actual_batch_scaling_checked':True,'single_optimizer_continues_into_three_QA_epochs':True,
        'single_pre_registered_variant':True,'variants':reports,'preparation_sha256':q.sha(OUT/'preparation_complete.json'),
        'real_dataset_training':False,'pretrained_model_loaded':False,'GPU_used':False,'official_test_opened':False})
    print('RELATION_PAIR_FAST_CPU_TEST_PASSED',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','check','cpu-test','gpu-smoke','train'])
    parser.add_argument('--variant',choices=list(VARIANTS));args=parser.parse_args()
    if args.stage=='train' and args.variant is None:parser.error('train requires --variant')
    with threadpool_limits(limits=4):
        try:
            if args.stage=='prepare':prepare()
            elif args.stage=='check':check_prepared();print('RELATION_PAIR_FAST_PREPARED_CHECK_PASSED',flush=True)
            elif args.stage=='cpu-test':cpu_test()
            elif args.stage=='gpu-smoke':gpu_smoke()
            else:train(args.variant)
        except BaseException:
            OUT.mkdir(parents=True,exist_ok=True)
            q.save(OUT/f'FAILURE_{args.stage}_{args.variant or "shared"}_{time.time_ns()}.json',
                   {'stage':args.stage,'variant':args.variant,'traceback':traceback.format_exc(),'automatic_retry':False})
            raise

if __name__=='__main__':main()
