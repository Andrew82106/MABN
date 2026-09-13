"""Independent frozen direct-window TCN audit: CPU forward only, no training.

Reads only named fit/cal exports and frozen development artifacts. No production
module imports, backward calls, optimizer construction, PCA/scaler fitting, or
official test reads. Run only after the complete marker exists.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=OUT.parent/'development_v1'
REPORT=OUT/'INDEPENDENT_AUDIT_WINDOW_TCN.json'
HELPER=OLD/'audit_coefficients_qa.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='a7a25ec0c346ad32621769762ff97b7f81c7caa6d959658a88508fb4ab5c1dae'
spec=importlib.util.spec_from_file_location('independent_qa_metadata',HELPER)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
read,sha,near=q.read,q.sha,q.near
NFIT,NCAL,N,WIDTH=168123,42241,210364,1089


def save(report):
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def hash_binding(report):
    complete=read(OUT/'complete.json')
    assert complete['status']=='complete_development_only'
    assert complete['test_opened'] is False and complete['gpu_used'] is False
    checked={}
    for name,h in complete['files_sha256'].items():
        path=(OUT/name).resolve();assert path.is_relative_to(OUT)
        assert sha(path)==h,name;checked[str(path)]=h
    assert complete['script_sha256']==sha(ROOT/'src/run_window_sequence.py')
    frozen,source,prep=map(read,(OUT/'freeze.json',OUT/'source_snapshot.json',OUT/'preparation_manifest.json'))
    for key,name in [('protocol_sha256','protocol.json'),('source_snapshot_sha256','source_snapshot.json'),('chain_index_sha256','chain_index.json')]:
        assert frozen[key]==sha(OUT/name)
    for text,h in source['files_sha256'].items():
        path=Path(text).resolve();assert path.is_relative_to(ROOT)
        assert not any(x in str(path).lower() for x in ('sealed','withheld','test.json','test_'))
        assert sha(path)==h;text=str(path);checked[text]=h
    for name,h in prep['files_sha256'].items():
        assert sha(OUT/name)==h;checked[str((OUT/name).resolve())]=h
    assert prep['complete'] and prep['shape']==[N,WIDTH]
    assert prep['fit_batches_per_epoch']==119 and prep['minibatch_constant']==119/NFIT
    assert read(OUT/'started.json')['freeze_sha256']==sha(OUT/'freeze.json')
    assert read(OUT/'CPU_PREFLIGHT.json')['code_sha256']==sha(ROOT/'src/run_window_sequence.py')
    protocol=read(OUT/'protocol.json')
    assert (protocol['seed'],protocol['epochs'],protocol['configurations'],protocol['batch_chains'])==(20260929,30,1,8)
    assert protocol['receptive_field_windows']==7 and protocol['future_windows_used'] is True
    report['hash_binding']={'verified_file_count':len(checked),'sha256':checked}
    report['complete_sha256']=sha(OUT/'complete.json')
    return prep


def geometry(answers,windows,indices,report):
    saved=read(OUT/'chain_index.json')
    assert (saved['window_count'],saved['fit_window_count'],saved['calibration_window_count'])==(N,NFIT,NCAL)
    seqs=[]
    for a in answers:
        ix=indices[a['answer_id']]
        starts=[windows[j]['token_indices'][0] for j in ix]
        assert np.all(np.diff(starts)>0)
        stops=[0]+[j for j in range(1,len(ix)) if starts[j]!=starts[j-1]+1]+[len(ix)]
        for lo,hi in zip(stops,stops[1:]):
            rows=ix[lo:hi]
            assert rows==list(range(rows[0],rows[-1]+1))
            seqs.append({**{k:a[k] for k in ('answer_id','response_id','group_id','partition')},
                         'left':rows[0],'right':rows[-1]+1,'length':len(rows),
                         'raw_start_first':starts[lo],'raw_start_last':starts[hi-1]})
    assert len(seqs)==len(saved['chains'])==1187
    for expected,stored in zip(seqs,saved['chains']):
        assert all(stored[k]==v for k,v in expected.items())
    assert np.array_equal(np.concatenate([np.arange(s['left'],s['right']) for s in seqs]),np.arange(N))
    fit_ids=np.asarray([i for i,s in enumerate(seqs) if s['partition']=='fit'])
    assert len(fit_ids)==952 and all(seqs[i]['right']<=NFIT for i in fit_ids)
    orders=np.load(OUT/'epoch_chain_order.npy',allow_pickle=False)
    assert orders.shape==(30,952)
    rng=np.random.default_rng(20260929)
    for order in orders:
        assert np.array_equal(order,rng.permutation(fit_ids))
        assert np.array_equal(np.sort(order),fit_ids)
        coverage=np.concatenate([np.arange(seqs[i]['left'],seqs[i]['right']) for i in order])
        assert np.array_equal(np.sort(coverage),np.arange(NFIT))
        assert len([order[j:j+8] for j in range(0,len(order),8)])==119
    report['geometry']={'answers':793,'fit_answers':634,'calibration_answers':159,'windows':N,
        'fit_windows':NFIT,'calibration_windows':NCAL,'chains':1187,'fit_chains':952,'calibration_chains':235,
        'raw_start_gaps':394,'all_gaps_split':True,'no_cross_answer_or_partition':True,
        'epochs':30,'fit_batches_per_epoch':119,'all_fit_windows_once_each_epoch':True,'seed_permutations_exact':True}
    return seqs


def matrices_and_weights(answers,windows,prep,report):
    _,independent=q.audit_weights(windows)
    with np.load(OLD/'training_weights.npz',allow_pickle=False) as z:oldw={k:z[k].copy() for k in z.files}
    with np.load(OUT/'training_weights.npz',allow_pickle=False) as z:
        assert set(z.files)==set(oldw)
        assert all(np.array_equal(z[k],oldw[k]) for k in z.files)
    obj=q.unpickle(OLD/'hidden64_lookback_nll_C0.001.pkl')
    sc=q.unpickle(OUT/'scaler.pkl')
    assert obj['fit_only'] and (obj['fit_rows'],obj['fit_groups'],obj['fit_answers'],obj['width'])==(NFIT,615,634,WIDTH)
    assert obj['weights_sha256']==sha(OLD/'training_weights.npz') and obj['fit_keys_sha256']==sha(OLD/'fit_keys.json')
    for k in ('mean_','var_','scale_','n_samples_seen_','n_features_in_','with_mean','with_std'):
        assert np.array_equal(getattr(sc,k),getattr(obj['scaler'],k))
    assert prep['scaler_source_sha256']==sha(OLD/'hidden64_lookback_nll_C0.001.pkl')
    assert prep['PCA_source_sha256']==obj['pca_sha256']==sha(OLD/'hidden_pca.pkl')
    pca=q.unpickle(OLD/'hidden_pca.pkl');fit={a['response_id']:a for a in answers[:634]}
    assert pca['fit_answers']==634 and pca['fit_groups']==615 and pca['components'].shape==(64,4096)
    assert len(pca['sample'])==pca['sample_count']==20288
    for s in pca['sample']:
        a=fit[s['response_id']]
        assert s['group_id']==a['group_id'] and 0<=s['token_index']<a['token_count']
    base=np.load(OLD/'matrices/base.npy',mmap_mode='r',allow_pickle=False)
    hidden=np.load(OLD/'matrices/hidden.npy',mmap_mode='r',allow_pickle=False)
    x=np.load(OUT/'standardized_window_features.npy',mmap_mode='r',allow_pickle=False)
    oldz=np.load(OLD/'matrices/hidden64_lookback_nll_fit_standardized.npy',mmap_mode='r',allow_pickle=False)
    assert base.shape==(N,1025) and hidden.shape==(N,64) and x.shape==(N,WIDTH) and oldz.shape==(NFIT,WIDTH)
    assert all(a.dtype==np.float32 for a in (base,hidden,x,oldz))
    for left in range(0,N,16384):
        right=min(left+16384,N)
        transformed=np.concatenate((base[left:right],hidden[left:right]),axis=1)
        transformed-=sc.mean_;transformed/=sc.scale_
        assert np.isfinite(transformed).all() and np.array_equal(transformed,x[left:right]),('standardized input',left)
        if left<NFIT:
            end=min(right,NFIT)
            assert np.array_equal(transformed[:end-left],oldz[left:end]),('old fit input',left)
    report['input_and_weights']={'standardized_all_210364_rows_exact':True,'old_fit_168123_rows_exact':True,
        'width':WIDTH,'order':'base1025 (LB1024 then NLL) followed by hidden64',
        'scaler_state_exact':True,'PCA_20288_samples_all_fit_only':True,'PCA_refitted':False,
        'all_original_LR_weight_arrays_exact':True,'independent_weight_formula':independent,
        'minibatch_objective_constant':119/NFIT,'runtime_weights_are_original_values_cast_float32':True}
    return x,oldw


def manual_forward(x,mask,state):
    hidden=F.gelu(F.linear(x,state['projection.weight'],state['projection.bias'])).transpose(1,2)*mask[:,None,:]
    for i,dilation in enumerate((1,2)):
        convolved=F.conv1d(hidden,state[f'convolutions.{i}.weight'],state[f'convolutions.{i}.bias'],padding=dilation,dilation=dilation)
        hidden=(hidden+F.gelu(convolved))*mask[:,None,:]
    return F.linear(hidden.transpose(1,2),state['output.weight'],state['output.bias']).squeeze(-1)*mask


def selected_replay(answers,indices,seqs,x,weights,report):
    summary=read(OUT/'summary.json');selected=summary['selected']['window_tcn'];epoch=selected['epoch']
    history=summary['all_epochs']['window_tcn']
    assert len(history)==30 and [x['epoch'] for x in history]==list(range(1,31))
    assert summary['test_opened'] is False and summary['gpu_used'] is False and summary['calibration_results_are_selection_optimistic'] is True
    path=OUT/f'window_tcn/epoch_{epoch:03d}.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['epoch']==epoch and state['seed']==20260929 and state['device']=='cpu'
    assert state['protocol_sha256']==sha(OUT/'protocol.json') and state['preparation_manifest_sha256']==sha(OUT/'preparation_manifest.json')
    params=state['model_state_dict']
    shapes={'projection.weight':(32,1089),'projection.bias':(32,),
        'convolutions.0.weight':(32,32,3),'convolutions.0.bias':(32,),
        'convolutions.1.weight':(32,32,3),'convolutions.1.bias':(32,),
        'output.weight':(1,32),'output.bias':(1,)}
    assert set(params)==set(shapes)
    for k,t in params.items():assert tuple(t.shape)==shapes[k] and t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all()
    assert sum(t.numel() for t in params.values())==41121
    opt=state['optimizer_state_dict']
    assert len(opt['state'])==len(params)
    assert all(float(v['step'])==epoch*119 for v in opt['state'].values())
    assert len(opt['param_groups'])==1 and opt['param_groups'][0]['lr']==.001 and opt['param_groups'][0]['weight_decay']==.01
    logits=np.empty(N,np.float32)
    with torch.inference_mode():
        for left in range(0,len(seqs),8):
            chosen=seqs[left:left+8];length=max(s['length'] for s in chosen)
            xb=torch.zeros(len(chosen),length,WIDTH);mask=torch.zeros(len(chosen),length)
            for j,s in enumerate(chosen):
                xb[j,:s['length']]=torch.from_numpy(np.array(x[s['left']:s['right']],copy=True));mask[j,:s['length']]=1
            out=manual_forward(xb,mask,params).numpy()
            for j,s in enumerate(chosen):logits[s['left']:s['right']]=out[j,:s['length']]
    assert np.isfinite(logits).all()
    scores=torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float64)
    answer_scores=np.asarray([scores[indices[a['answer_id']]].max() for a in answers])
    with np.load(OUT/f'window_tcn/epoch_{epoch:03d}_scores.npz',allow_pickle=False) as z:
        saved={k:z[k].copy() for k in z.files}
    errors={k:near(v,saved[k],k,rtol=3e-6,atol=3e-6) for k,v in [('window_logits',logits),('window_scores',scores),('answer_scores',answer_scores)]}
    assert np.array_equal(np.asarray([saved['window_scores'][indices[a['answer_id']]].max() for a in answers]),saved['answer_scores'])
    z=saved['window_logits'][:NFIT].astype(np.float64)
    bce=np.logaddexp(0,z)-weights['y']*z
    loss=float(weights['loss_weights']@bce/NFIT)
    near(loss,selected['full_fit_weighted_BCE'],'fit BCE',rtol=1e-12,atol=1e-12)
    report['selected_CPU_forward']={'epoch':epoch,'parameters':41121,'optimizer_steps':epoch*119,
        'window_rows':N,'answer_rows':793,'max_abs_errors':errors,
        'exact':{k:bool(np.array_equal(v,saved[k])) for k,v in [('window_logits',logits),('window_scores',scores),('answer_scores',answer_scores)]},
        'full_fit_weighted_BCE':loss,'saved_answermax_exact':True,'production_module_imported':False,
        'method':'Functional linear/GELU/conv1d from frozen state_dict; independent batch construction; no nn.Module, dropout, backward or optimizer execution.'}
    rows=[]
    old=read(OLD/'summary.json')['selected']['hidden64_lookback_nll']
    token=read(OUT.parent/'sequence_v1/summary.json')['selected']['token_tcn']
    for name,e in [('same_input_old_LR',old),('old_token_TCN',token),('direct_window_TCN',selected)]:
        m=e['metrics']['calibration']
        rows.append({'method':name,'C':e.get('C'),'epoch':e.get('epoch'),
                     'calibration_window_F1':m['windows']['f1'],'calibration_answer_F1':m['answers']['f1']})
    report['calibration_comparison']=rows
    return rows


def audit(report):
    assert (OUT/'complete.json').exists(),'Await formal completion; audit must not compete with training'
    assert not torch.cuda.is_initialized()
    prep=hash_binding(report)
    answers,windows,indices=q.metadata()
    seqs=geometry(answers,windows,indices,report);save(report)
    x,weights=matrices_and_weights(answers,windows,prep,report);save(report)
    print('WINDOW_TCN_INPUT_WEIGHTS_AND_COVERAGE_PASSED',flush=True)
    rows=selected_replay(answers,indices,seqs,x,weights,report)
    peer=OUT/'CALIBRATION_AUDIT_WINDOW_TCN.json'
    if peer.exists():
        p=read(peer);assert p.get('passed') is True or p.get('status')=='passed'
        report['independent_30_epoch_calibration_audit']={'path':str(peer.resolve()),'sha256':sha(peer),'passed':True}
    else:
        report['independent_30_epoch_calibration_audit']={'pending':True}
    report['audit_script_sha256']=sha(__file__)
    assert not torch.cuda.is_initialized()
    lines=['# Direct-window TCN 独立结果审计','',
        '输入、旧fit-only scaler/PCA、原LR权重及30轮覆盖均已核对；选中epoch由保存状态做CPU纯前向回放。未训练、调用GPU或读取test。','',
        '| 方法 | C / epoch | cal窗口F1 | cal整答F1 |','|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['C'] if r['C'] is not None else r['epoch']} | {r['calibration_window_F1']:.6f} | {r['calibration_answer_F1']:.6f} |")
    lines+=['','这些校准成绩用于选择参数/epoch和阈值，存在选择乐观；不能当独立最终测试成绩。',
            '与旧tokenTCN相比同时改变输入宽度、网络宽度、窗口覆盖和权重，不能把差异单独归因于直接窗口监督。']
    (OUT/'INDEPENDENT_AUDIT_WINDOW_TCN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    report={'status':'running','utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
        'no_fitting_or_backward':True,'no_GPU':True,'official_test_or_withheld_read':False,'frozen_files_modified':False,
        'limits':['Numerical consistency and artifact hashes do not independently prove unrecorded execution history.',
                  'All30 threshold/epoch selection checks are delegated to the separate independent calibration audit.',
                  'PCA/scaler reuse is checked against frozen prior artifacts and audited fit indices; no refitting.']}
    try:
        with threadpool_limits(limits=4):audit(report)
        report['status']='passed'
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('INDEPENDENT_WINDOW_TCN_AUDIT_PASSED',sha(REPORT),flush=True)
