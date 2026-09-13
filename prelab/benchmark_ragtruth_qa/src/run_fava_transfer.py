"""Prepared-only FAVA synthetic auxiliary -> QA transfer experiment.

Explicit CPU prepare/cpu-test, and separately explicit GPU smoke/train commands.
No command chains preparation into training. No official test is opened.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from transformers import ModernBertConfig, ModernBertForTokenClassification
import run_full_context_encoder_v2 as base
import run_auxiliary_transfer as shared
from run_auxiliary_transfer import IndexedAuxiliary, aux_token_weights, train_epoch, optimizer_steps

q=base.q
mapping=base.mapping
evaluation=base.evaluation
ROOT=q.ROOT
OUT=ROOT/'results/full_context_fava_transfer_v1'
AUX=ROOT/'auxiliary_fava_v2'
TOKENS=AUX/'tokenization_v2'
NAUX=7482
NQA=3680
QA_EPOCHS=3
AUX_SEED=20261007
BATCH=8


def protocol():
    return {'version':'fava-synthetic-aux-to-QA-full-context-v1',
        'scope':'One auxiliary train epoch followed by three QA train epochs. CPU preparation is not training; official test stays sealed.',
        'initialization':'Exactly run_full_context_encoder_v2.load_model: generic ModernBERT-base, same pretrained revision, fresh two-class head and initial seed20261005. Never start from a QA-fitted checkpoint.',
        'architecture_and_precision':'Unchanged v2 full-context two-logit classifier and original QA character mapping. Auxiliary-only proven NFC combining-mark ownership comes from frozen tokenization_v2; no answer or label rewrite. CUDA BF16 forward/checkpoint recompute; FP32 parameters, gradients, optimizer, logit difference, mapping and BCE. CPU FP32; TF32 disabled.',
        'input':'Auxiliary: all five original numbered Reference sections + separator + empty question + separator + original corrupted answer. QA: original entire source + separator + original question + separator + complete original answer. Frozen auxiliary token_inputs and byte-identical QA v2 inputs; no truncation, correction/markup input or generator identity feature. Released FAVA checking-prompt metadata is not the semantic model input or a native generation trace.',
        'reuse':'Import only IndexedAuxiliary, aux_token_weights, train_epoch and optimizer_steps from unchanged run_auxiliary_transfer. Do not replace old module globals, OUT or artifacts. New output/checkpoints and phase coordination belong to this runner.',
        'auxiliary':{'answers':NAUX,'sources':7469,'groups':7408,'epochs':1,'order_seed':AUX_SEED,
            'target_loss_mass':854011,'weights':'Equal source-material group, then equal answer within group, then equal lexical raw token within answer. Scale base to lexical mass; binary class factors from auxiliary train base mass only; reequalize each group loss and total mass.',
            'selection':'None. Do not score calibration, choose an auxiliary checkpoint, or evaluate an auxiliary validation set. Save online training loss and the final auxiliary checkpoint only.'},
        'QA':{'fit_answers':NQA,'fit_groups':615,'calibration_answers':159,'calibration_groups':154,
            'epochs':QA_EPOCHS,'weights':'Exact v2 training_weights.npz including its original/added-answer split, class factors and group reweighting; target_mass560300.',
            'orders':'Exact first three rows of v2 answer_orders.npy. No RNG reseed at phase transition.',
            'selection':'Only the three QA epochs: separate calibration-only window/answer F1 thresholds, precision/higher cutoff ties; epoch selection by min(two F1), windowF1, window precision, earlier QA epoch.',
            'metrics':'Unchanged original4rawBPE stride1 lexical risk union/max; full fit3680 and calibration159; calibration42241 windows. Answermax includes all eligible good answers, including refusals.'},
        'optimizer':{'type':'AdamW','lr':1e-5,'weight_decay':.01,'gradient_clip':1.,
            'microbatch_answers':1,'effective_batch_answers':BATCH,'scheduler':'none',
            'continuity':'One optimizer object created before auxiliary epoch; all state, moments and step counters continue through QA epochs. No optimizer reset.',
            'expected_updates':{'auxiliary':936,'each_QA_epoch':460,'total':2316}},
        'loss':'Same v2 weighted BCE scaling per answer: sum(tokenBCE*phase_loss_weight)*phase_answer_count/(actual_accumulated_batch_answers*phase_target_mass). Final short auxiliary batch has2 answers.',
        'artifacts':'Final auxiliary model+optimizer/RNG checkpoint and training loss. Preserve each of all3 QA model+optimizer/RNG checkpoints, complete predictions, thresholds/metrics, actual logical input tokens, updates, wall time and CUDA peaks. No automatic resume or overwrite.',
        'resource_gate':'Future explicit GPU smoke on longest1481-token auxiliary input before training, synthetic zero target only; repeated inference<=2e-6, finite nonzero gradient, original precision and full context retained. Failure is recorded; no silent truncation/fallback.',
        'comparison_limits':'Additional synthetic silver supervision and training schedule together change. FAVA labels may be noisy; only factual-only rows are used, and unmarked positions are not verified true negatives. All selected auxiliary answers contain annotated risk, but the objective is token BCE, not answer classification. 18522 answer presentations vs QA-only6epoch22080 does not establish equal computation: auxiliary inputs are longer. Report actual tokens/time/memory. No isolated architecture or independent-test improvement claim.',
        'token_accounting':'Logical input tokens processed by forward calls; training and full-QA evaluation separated. Checkpoint recomputation is excluded from logical-token counts but included in wall time and memory.',
        'automatic_GPU_or_training':False}


def source_files():
    snap=base.source_files()
    paths=[Path(__file__),Path(base.__file__),Path(shared.__file__),AUX/'DATA_REPORT.md',AUX/'manifest.json',AUX/'complete.json',
        AUX/'candidate_fit.jsonl',TOKENS/'token_input_preparation.json',TOKENS/'token_inputs.jsonl',
        TOKENS/'token_protocol.json',TOKENS/'token_design_freeze.json',TOKENS/'nfc_repairs.jsonl',
        Path(__file__).with_name('prepare_fava_tokens_v2.py'),Path(__file__).with_name('fava_nfc_character_map.py'),
        AUX/'DATA_AUDIT.json',base.OUT/'preparation_complete.json',base.OUT/'inputs.jsonl',
        base.OUT/'training_weights.npz',base.OUT/'answer_orders.npy',base.OUT/'protocol.json']
    snap.update({str(p.resolve()):q.sha(p) for p in paths})
    return snap






def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT/'prepare_started.json').exists(),'No overwrite of prepared run'
    OUT.mkdir(parents=True,exist_ok=True)
    old=base.check_prepared();assert old['answers']==3839
    am=q.read(AUX/'manifest.json');ap=q.read(TOKENS/'token_input_preparation.json')
    assert am['status']=='staged_synthetic_auxiliary_fit_not_trained' and am['answers']==NAUX
    assert ap['status']=='prepared_not_trained' and not ap['exceptions'] and ap['GPU_used'] is False
    assert ap['answers']==ap['expected_answers']==NAUX
    assert ap['manifest_sha256']==q.sha(AUX/'manifest.json')==q.read(AUX/'complete.json')['manifest_sha256']
    assert ap['candidate_sha256']==q.sha(AUX/'candidate_fit.jsonl')
    assert ap['nfc_repairs_sha256']==q.sha(TOKENS/'nfc_repairs.jsonl')
    assert ap['token_design_sha256']==q.sha(TOKENS/'token_design_freeze.json')
    token_freeze=q.read(TOKENS/'token_design_freeze.json')
    assert token_freeze['protocol_sha256']==q.sha(TOKENS/'token_protocol.json')
    for path,h in token_freeze['source_sha256'].items():assert q.sha(Path(path))==h
    assert shared.BATCH==BATCH==8 and shared.base is base
    assert shared.OUT==ROOT/'results/full_context_aux_transfer_v1'
    assert shared.AUX==ROOT/'auxiliary_human_v1' and shared.NAUX==9678
    assert q.read(AUX/'DATA_AUDIT.json')['passed']
    assert q.sha(AUX/'candidate_fit.jsonl')==am['artifacts_sha256']['candidate_fit.jsonl']
    assert q.sha(TOKENS/'token_inputs.jsonl')==ap['token_inputs_sha256']
    snap=source_files();q.save(OUT/'prepare_started.json',{'time':time.time(),'source_sha256':snap})
    qa_rows=q.lines(base.OUT/'inputs.jsonl');answers,tokens,_=mapping.metadata()
    assert [r['response_id'] for r in qa_rows]==[a['response_id'] for a in answers]
    qa_ids={a['response_id'] for a in answers}
    candidate={r['response_id']:(r['source_id'],r['group_id'],r['answer_sha256']) for r in q.lines(AUX/'candidate_fit.jsonl')}
    index=[];offsets=[];aux_ids=set();groups=set();sources=set();cursor=0
    with (TOKENS/'token_inputs.jsonl').open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            r=json.loads(line);assert r['response_id'] not in aux_ids|qa_ids
            assert candidate[r['response_id']]==(r['source_id'],r['group_id'],r['answer_sha256'])
            assert r['task_type']=='FAVA_synthetic' and r['synthetic_not_human_gold'] and r['unmarked_positions_silver_not_verified_negative']
            aux_ids.add(r['response_id']);groups.add(r['group_id']);sources.add(r['source_id'])
            assert len(r['input_ids'])<=8192 and len(r['response_token_ids'])==len(r['risk_mask'])==len(r['lexical_mask'])
            n=len(r['response_token_ids']);lex=int(sum(r['lexical_mask']))
            assert lex and all(not y or l for y,l in zip(r['risk_mask'],r['lexical_mask']))
            index.append({'response_id':r['response_id'],'source_id':r['source_id'],'group_id':r['group_id'],
                          'task_type':r['task_type'],'answer_sha256':r['answer_sha256'],'raw_token_count':n,
                          'lexical_tokens':lex,'input_tokens':len(r['input_ids']),'raw_token_bounds':[cursor,cursor+n]})
            cursor+=n;offsets.append(offset)
    assert len(index)==NAUX and len(sources)==7469 and len(groups)==7408
    assert aux_ids==set(candidate) and cursor==ap['stats']['raw_answer_tokens']==969140
    assert max(r['input_tokens'] for r in index)==1481
    assert sum(r['lexical_tokens'] for r in index)==854011
    weights=aux_token_weights(index,(json.loads(line) for line in (TOKENS/'token_inputs.jsonl').open(encoding='utf-8')))
    assert int(weights['target_mass'])==854011
    assert int(weights['y'].sum())==ap['stats']['risk_answer_tokens']==215922
    np.savez_compressed(OUT/'auxiliary_training_weights.npz',**weights)
    np.save(OUT/'auxiliary_byte_offsets.npy',np.asarray(offsets,np.int64))
    np.save(OUT/'auxiliary_answer_order.npy',np.random.default_rng(AUX_SEED).permutation(NAUX))
    q.save(OUT/'auxiliary_index.json',{'answers':index,'source_path':str((TOKENS/'token_inputs.jsonl').resolve()),
                                    'source_sha256':ap['token_inputs_sha256']})
    shutil.copyfile(base.OUT/'inputs.jsonl',OUT/'qa_inputs.jsonl')
    shutil.copyfile(base.OUT/'training_weights.npz',OUT/'qa_training_weights.npz')
    order=np.load(base.OUT/'answer_orders.npy');assert order.shape==(6,NQA)
    np.save(OUT/'qa_answer_orders.npy',order[:QA_EPOCHS])
    assert q.sha(OUT/'qa_inputs.jsonl')==q.sha(base.OUT/'inputs.jsonl')
    assert q.sha(OUT/'qa_training_weights.npz')==q.sha(base.OUT/'training_weights.npz')
    assert np.array_equal(np.load(OUT/'qa_answer_orders.npy'),order[:3])
    per_group=defaultdict(lambda:[0.,0.,0])
    for row,(lo,hi) in zip(index,weights['bounds']):
        per_group[row['group_id']][0]+=float(weights['base'][lo:hi].sum())
        per_group[row['group_id']][1]+=float(weights['loss'][lo:hi].sum())
        per_group[row['group_id']][2]+=1
    unit=weights['target_mass']/len(groups)
    assert all(abs(b-unit)<1e-8 and abs(l-unit)<1e-8 for b,l,n in per_group.values())
    aux_train_tokens=sum(r['input_tokens'] for r in index)
    assert aux_train_tokens==ap['stats']['encoder_input_tokens']==5853211
    qa_fit_tokens=sum(len(r['input_ids']) for r in qa_rows[:NQA]);qa_eval_tokens=sum(len(r['input_ids']) for r in qa_rows)
    q.save(OUT/'WEIGHT_REUSE_AND_RESOURCE_PLAN.json',{
        'qa_inputs_byte_identical':True,'qa_weight_file_byte_identical':True,'qa_first_three_orders_exact':True,
        'aux_groups':len(groups),'aux_equal_group_base_and_loss_mass':unit,'aux_class_factors':weights['class_factors'].tolist(),
        'aux_class_mass_before_balance':weights['class_mass_before_balance'].tolist(),
        'aux_group_stats':{g:{'base':b,'loss':l,'answers':n} for g,(b,l,n) in per_group.items()},
        'answer_presentations':NAUX+3*NQA,'qa_only_six_epoch_answer_presentations':6*NQA,
        'logical_training_input_tokens':aux_train_tokens+3*qa_fit_tokens,
        'aux_training_input_tokens':aux_train_tokens,'each_qa_training_input_tokens':qa_fit_tokens,
        'each_qa_evaluation_input_tokens':qa_eval_tokens,'total_evaluation_input_tokens':3*qa_eval_tokens,
        'qa_only_six_epoch_training_input_tokens':6*qa_fit_tokens,
        'qa_only_epoch0_plus_six_evaluation_input_tokens':7*qa_eval_tokens,
        'expected_optimizer_updates':936+3*460,'raw_aux_tokens':cursor,'max_aux_input_tokens':1481,
        'max_QA_input_tokens':max(len(r['input_ids']) for r in qa_rows),
        'not_compute_matched':True,'trained':False,'GPU_used':False})
    q.save(OUT/'protocol.json',protocol())
    assert snap==source_files()
    q.save(OUT/'source_snapshot.json',{'files_sha256':snap,'official_test_opened':False})
    files=['auxiliary_training_weights.npz','auxiliary_byte_offsets.npy','auxiliary_answer_order.npy',
           'auxiliary_index.json','qa_inputs.jsonl','qa_training_weights.npz','qa_answer_orders.npy',
           'WEIGHT_REUSE_AND_RESOURCE_PLAN.json','protocol.json','source_snapshot.json']
    q.save(OUT/'preparation_complete.json',{'status':'prepared_not_trained','aux_answers':NAUX,
        'qa_fit_answers':NQA,'calibration_answers':159,'files_sha256':{f:q.sha(OUT/f) for f in files},
        'GPU_used':False,'official_test_opened':False})
    assert not torch.cuda.is_initialized()
    print('FAVA_TRANSFER_PREPARED',NAUX,NQA,159,'training_input_tokens',aux_train_tokens+3*qa_fit_tokens,flush=True)


def check_prepared():
    p=q.read(OUT/'preparation_complete.json')
    assert p['status']=='prepared_not_trained' and p['aux_answers']==NAUX and p['qa_fit_answers']==NQA
    assert q.read(OUT/'protocol.json')==protocol()
    for name,h in p['files_sha256'].items():assert q.sha(OUT/name)==h,name
    assert q.read(OUT/'source_snapshot.json')['files_sha256']==source_files()
    return p


def load_weights(name):
    with np.load(OUT/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}




def save_checkpoint(model,optimizer,path,phase,epoch,updates):
    state={'model_state_dict':evaluation.cpu_state(model.state_dict()),
           'optimizer_state_dict':evaluation.cpu_state(optimizer.state_dict()),
           'phase':phase,'phase_epoch':epoch,'global_optimizer_updates':updates,
           'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state().cpu(),
           'preparation_sha256':q.sha(OUT/'preparation_complete.json')}
    torch.save(state,path)




def train():
    check_prepared();gate=q.read(OUT/'GPU_SELFCHECK.json')
    assert gate['passed'] and gate['preparation_sha256']==q.sha(OUT/'preparation_complete.json')
    assert q.read(OUT/'CPU_SELFCHECK.json')['passed']
    directory=OUT/'transfer';directory.mkdir(exist_ok=True)
    assert not (directory/'started.json').exists(),'No automatic overwrite/resume'
    base.configure_gpu()
    rows=q.lines(OUT/'qa_inputs.jsonl');answers,tokens,_=mapping.metadata()
    qa_weights=load_weights('qa_training_weights.npz');aux_weights=load_weights('auxiliary_training_weights.npz')
    orders=np.load(OUT/'qa_answer_orders.npy');aux_order=np.load(OUT/'auxiliary_answer_order.npy')
    auxiliary=IndexedAuxiliary(TOKENS/'token_inputs.jsonl',np.load(OUT/'auxiliary_byte_offsets.npy'))
    # One fresh model and one optimizer for the entire transfer, with original v2 seed.
    model=base.load_model().cuda();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    q.save(directory/'started.json',{'time':time.time(),'preparation_sha256':q.sha(OUT/'preparation_complete.json'),
                                   'initial_seed':base.SEED,'no_auxiliary_calibration_evaluation':True})
    all_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    aux_entry=train_epoch(model,optimizer,auxiliary,aux_order,aux_weights,'cuda','auxiliary_epoch1')
    auxiliary.close();updates=aux_entry['optimizer_updates'];assert updates==936==optimizer_steps(optimizer)
    aux_entry['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    save_checkpoint(model,optimizer,directory/'auxiliary_final.pt','auxiliary',1,updates)
    aux_entry.update({'checkpoint_sha256':q.sha(directory/'auxiliary_final.pt'),'global_optimizer_updates':updates,
                      'calibration_evaluated':False,'checkpoint_selection':False,'official_test_opened':False})
    q.save(directory/'auxiliary_final.json',aux_entry)
    history=[];best=None
    for epoch in range(1,QA_EPOCHS+1):
        tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        assert optimizer_steps(optimizer)==936+(epoch-1)*460
        training=train_epoch(model,optimizer,rows,orders[epoch-1],qa_weights,'cuda',f'QA_epoch{epoch}')
        updates+=training['optimizer_updates'];assert training['optimizer_updates']==460 and optimizer_steps(optimizer)==updates
        model.eval();probabilities={};fit_bce=0.;eval_tokens=0
        with torch.no_grad():
            for i,row in enumerate(rows):
                z=base.logits(model,row,'cuda');assert torch.isfinite(z).all()
                probabilities[row['response_id']]=torch.sigmoid(z).cpu().numpy();eval_tokens+=len(row['input_ids'])
                if i<NQA:
                    lo,hi=qa_weights['bounds'][i]
                    y=torch.as_tensor(qa_weights['y'][lo:hi],device='cuda',dtype=torch.float32)
                    w=torch.as_tensor(qa_weights['loss'][lo:hi],device='cuda',dtype=torch.float32)
                    fit_bce+=float((F.binary_cross_entropy_with_logits(z,y,reduction='none')*w).double().sum())
                if (i+1)%400==0:print('FAVA_TRANSFER_QA_EVAL',epoch,i+1,len(rows),flush=True)
        fit=evaluation.score_geometry(answers,tokens,probabilities,range(NQA))
        cal=evaluation.score_geometry(answers,tokens,probabilities,range(NQA,len(rows)))
        assert len(fit['window_scores'])==653979 and len(cal['window_scores'])==42241 and len(cal['answer_scores'])==159
        ts={'window':q.choose_threshold(cal['window_labels'],cal['window_scores']),
            'answer':q.choose_threshold(cal['answer_labels'],cal['answer_scores'])}
        key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-epoch]
        stem=f'qa_epoch_{epoch:02d}'
        save_checkpoint(model,optimizer,directory/(stem+'.pt'),'QA',epoch,updates)
        np.savez_compressed(directory/(stem+'_token_predictions.npz'),**probabilities)
        np.savez_compressed(directory/(stem+'_scores.npz'),**{'fit_'+k:v for k,v in fit.items()},**{'cal_'+k:v for k,v in cal.items()})
        entry={'epoch':epoch,'phase':'QA','thresholds':ts,'selection_key':key,
            'fit_at_cal_thresholds':evaluation.metrics_for(fit,ts),'calibration':evaluation.metrics_for(cal,ts),
            'fit_weighted_bce':fit_bce/int(qa_weights['target_mass']),'training':training,
            'global_optimizer_updates':updates,'evaluation_logical_input_tokens':eval_tokens,
            'seconds':time.perf_counter()-tick,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'artifacts_sha256':{ext:q.sha(directory/(stem+ext)) for ext in ('.pt','_token_predictions.npz','_scores.npz')},
            'official_test_opened':False}
        q.save(directory/(stem+'.json'),entry);history.append(entry)
        if best is None or key>best['selection_key']:best=entry
        print('FAVA_TRANSFER_QA_EPOCH_COMPLETE',epoch,entry['calibration']['windows']['f1'],entry['calibration']['answers']['f1'],flush=True)
    assert updates==2316
    q.save(directory/'complete.json',{'status':'complete_development_only','selected':best,'all_QA_epochs':history,
        'auxiliary':aux_entry,'optimizer_state_continued':True,'optimizer_updates':updates,
        'actual_training_logical_input_tokens':aux_entry['logical_input_tokens']+sum(e['training']['logical_input_tokens'] for e in history),
        'actual_evaluation_logical_input_tokens':sum(e['evaluation_logical_input_tokens'] for e in history),
        'seconds':time.perf_counter()-all_start,'not_compute_matched':True,'official_test_opened':False})
    del model,optimizer;torch.cuda.empty_cache()
    print('FAVA_TRANSFER_COMPLETE_GPU_RELEASED',flush=True)


def gpu_smoke():
    """Only a future explicit GPU command; synthetic target on longest full auxiliary input."""
    check_prepared();assert q.read(OUT/'CPU_SELFCHECK.json')['passed']
    assert not (OUT/'GPU_SELFCHECK.json').exists(),'Preserve completed smoke'
    base.configure_gpu();index=q.read(OUT/'auxiliary_index.json')['answers']
    i=max(range(len(index)),key=lambda j:index[j]['input_tokens'])
    ds=IndexedAuxiliary(TOKENS/'token_inputs.jsonl',np.load(OUT/'auxiliary_byte_offsets.npy'));row=ds[i];ds.close()
    assert len(row['input_ids'])==1481
    model=base.load_model().cuda();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.eval()
    with torch.no_grad():
        a=base.logits(model,row,'cuda');b=base.logits(model,row,'cuda')
    difference=float((a-b).abs().max());assert difference<=2e-6 and torch.isfinite(a).all()
    calls=[]
    handle=model.model.layers[0].register_forward_pre_hook(lambda m,x:calls.append((torch.is_autocast_enabled('cuda'),str(torch.get_autocast_dtype('cuda')))))
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    opt.zero_grad(set_to_none=True);z=base.logits(model,row,'cuda')
    loss=F.binary_cross_entropy_with_logits(z,torch.zeros_like(z));assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm) and norm>0;opt.step();torch.cuda.synchronize()
    assert len(calls)>=2 and all(enabled and dtype=='torch.bfloat16' for enabled,dtype in calls)
    handle.remove();dtype_check=base.check_fp32_state(model,opt)
    q.save(OUT/'GPU_SELFCHECK.json',{'passed':True,'response_id':row['response_id'],'input_tokens':len(row['input_ids']),
        'synthetic_zero_target_only':True,'repeated_max_abs_difference':difference,'gradient_norm_before_clip':float(norm),
        'step_seconds':time.perf_counter()-tick,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'dtype_check':dtype_check,'checkpoint_forward_recompute_BF16':True,'no_gold_or_calibration_score_used':True,
        'preparation_sha256':q.sha(OUT/'preparation_complete.json'),'official_test_opened':False})
    del model,opt;torch.cuda.empty_cache();print('FAVA_TRANSFER_GPU_SMOKE_PASSED_GPU_RELEASED',flush=True)


def cpu_test():
    check_prepared();assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    # Nontrivial unequal-group/unequal-answer fixture checks intended weighting.
    fixture=[{'response_id':'a','group_id':'g1','lexical_mask':[1,1,0],'risk_mask':[0,1,0]},
             {'response_id':'b','group_id':'g1','lexical_mask':[1,1,1],'risk_mask':[1,0,0]},
             {'response_id':'c','group_id':'g2','lexical_mask':[1,0,1],'risk_mask':[0,0,1]}]
    index=[dict(r,raw_token_count=len(r['risk_mask'])) for r in fixture]
    w=aux_token_weights(index,fixture);mass=int(w['target_mass']);assert mass==7
    group_sums={}
    for group,ids in [('g1',[0,1]),('g2',[2])]:
        ix=np.concatenate([np.arange(*w['bounds'][i]) for i in ids])
        assert abs(w['base'][ix].sum()-mass/2)<1e-12 and abs(w['loss'][ix].sum()-mass/2)<1e-12
        group_sums[group]={'base':float(w['base'][ix].sum()),'loss':float(w['loss'][ix].sum())}
    assert w['base'][0]==w['base'][1] and w['base'][2]==w['loss'][2]==0
    config=ModernBertConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,max_position_embeddings=64,pad_token_id=0,num_labels=2,local_attention=16,reference_compile=False)
    config._attn_implementation='sdpa';torch.manual_seed(base.SEED)
    model=ModernBertForTokenClassification(config);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    example={'input_ids':[1,3,4,5,6,7,8,2],'raw_token_count':3,'mapping':[[0,0,1,2],[4,5,6,7],[.5,.5,1.,1.]]}
    rr=[dict(example,response_id=r['response_id']) for r in fixture]
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01);opt_id=id(opt)
    autocast=[];h=model.model.layers[0].register_forward_pre_hook(lambda m,x:autocast.append(torch.is_autocast_enabled('cpu')))
    first=train_epoch(model,opt,rr,np.array([2,0,1]),w,'cpu','tiny_aux',False)
    assert first['optimizer_updates']==1 and optimizer_steps(opt)==1
    moment=opt.state[model.classifier.weight]['exp_avg'].clone();assert moment.abs().sum()>0
    phases=[]
    for epoch in range(3):
        phases.append(train_epoch(model,opt,rr,np.array([1,2,0]),w,'cpu',f'tiny_QA{epoch+1}',False))
        assert id(opt)==opt_id and optimizer_steps(opt)==epoch+2
    assert len(autocast)>=8 and not any(autocast);h.remove()
    assert not torch.equal(moment,opt.state[model.classifier.weight]['exp_avg'])
    dtype_check=base.check_fp32_state(model,opt)
    assert model.classifier.weight.grad is not None and model.classifier.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.model.parameters())
    model.eval()
    with torch.no_grad():
        a=base.logits(model,example,'cpu');b=base.logits(model,dict(example,input_ids=[1,13,14,15,6,7,8,2]),'cpu')
    assert not torch.equal(a,b)
    # Exact real-data reuse without re-tokenizing or touching any model outputs.
    assert q.sha(OUT/'qa_inputs.jsonl')==q.sha(base.OUT/'inputs.jsonl')
    assert q.sha(OUT/'qa_training_weights.npz')==q.sha(base.OUT/'training_weights.npz')
    assert np.array_equal(np.load(OUT/'qa_answer_orders.npy'),np.load(base.OUT/'answer_orders.npy')[:3])
    ai=q.read(OUT/'auxiliary_index.json')['answers'];ds=IndexedAuxiliary(TOKENS/'token_inputs.jsonl',np.load(OUT/'auxiliary_byte_offsets.npy'))
    sample=sorted({0,len(ai)-1,max(range(len(ai)),key=lambda j:ai[j]['input_tokens'])})
    for i in sample:
        row=ds[i];assert row['response_id']==ai[i]['response_id'] and len(row['input_ids'])==ai[i]['input_tokens']
        assert row['raw_token_count']==ai[i]['raw_token_count'] and len(row['mapping'][0])==len(row['mapping'][1])==len(row['mapping'][2])
    ds.close();assert not torch.cuda.is_initialized()
    assert shared.OUT==ROOT/'results/full_context_aux_transfer_v1' and shared.AUX==ROOT/'auxiliary_human_v1'
    assert shared.NAUX==9678 and shared.BATCH==8 and shared.base is base
    assert all(f.__module__=='run_auxiliary_transfer' for f in (IndexedAuxiliary,aux_token_weights,train_epoch,optimizer_steps))
    q.save(OUT/'CPU_SELFCHECK.json',{'passed':True,'real_data_qa_inputs_weights_orders_exact':True,
        'unequal_group_answer_lexical_fixture':group_sums,'nonlexical_loss_zero':True,
        'tiny_same_training_loop_aux_then_three_QA_phases':True,'single_optimizer_steps':[1,2,3,4],
        'optimizer_moments_continue':True,'whole_encoder_and_head_have_gradients':True,
        'context_changes_answer_logits':True,'CPU_autocast_forward_and_recompute_disabled':True,
        'dtype_check':dtype_check,'indexed_real_auxiliary_samples':sample,'production_model_or_labels_fitted':False,
        'shared_pure_functions_not_copied_or_globals_modified':True,
        'GPU_used':False,'official_test_opened':False,'preparation_sha256':q.sha(OUT/'preparation_complete.json')})
    print('FAVA_TRANSFER_CPU_CHECK_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','check','cpu-test','gpu-smoke','train']);a=p.parse_args()
    with threadpool_limits(limits=4):
        try:{'prepare':prepare,'check':check_prepared,'cpu-test':cpu_test,'gpu-smoke':gpu_smoke,'train':train}[a.stage]()
        except Exception:
            OUT.mkdir(parents=True,exist_ok=True)
            q.save(OUT/f'FAILURE_{a.stage}_{time.time_ns()}.json',{'stage':a.stage,'traceback':traceback.format_exc()})
            raise
