"""Bounded regularization extension for all four original QA LR families.

Only published official-train fit/cal exports are used. Frozen old candidates
are replayed rather than refitted. No official test content is accessed.
"""
from pathlib import Path
import argparse, pickle, time
from datetime import datetime, timezone
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import run_development as q

OUT=q.ROOT/'results/regularization_v1'
OLD=q.OUT

def now():return datetime.now(timezone.utc).isoformat()

def protocol():
    return {'version':'qa-regularization-v1','methods':list(q.METHODS),
      'new_C':[.000001,.00001,.0001],'reused_C':[.001,.01,.1],
      'hypothesis':'All four previous best Cs hit the smallest tried value. Test stronger shrinkage; not evidence that labels or loss scaling are wrong.',
      'fit':'Same 168123 fit windows, frozen fit-only weighted scalers, exact frozen labels and loss weights; no new PCA or feature fit',
      'solver':'liblinear l2, max_iter2000, seed20260924, four CPU threads',
      'selection':'Same q.selection_key across six Cs separately per family, calibration-only thresholds for windows and answers',
      'answer':'Maximum all eligible windows, no gold-dependent filtering',
      'scope':'634 fit and 159 calibration published QA answers only; cal selection optimistic, official test unopened',
      'fairness':'All four original families receive the same three additional Cs; all old scores replayed exactly',
      'implementation_sha256':q.sha(__file__),'base_complete_sha256':q.sha(OLD/'complete.json'),
      'base_code_sha256':q.sha(q.__file__),'seed':20260924}

def self_test():
    rng=np.random.default_rng(20260924);x=rng.normal(size=(40,5));y=np.arange(40)%2;w=rng.uniform(.5,2,40)
    a=LogisticRegression(C=.01,solver='liblinear',tol=1e-10).fit(x,y,sample_weight=w)
    b=LogisticRegression(C=.0001,solver='liblinear',tol=1e-10).fit(x,y,sample_weight=100*w)
    diff=float(np.max(np.abs(a.coef_-b.coef_)))
    assert diff<1e-10 and np.max(np.abs(a.intercept_-b.intercept_))<1e-10
    return {'weight_mass_C_inverse_equivalence_max_abs':diff,'not_a_dataset_causality_test':True}

def run():
    cfg=q.read(OUT/'protocol.json');assert cfg==protocol()
    assert not (OUT/'started.json').exists(),'Refuse silently restarting training'
    q.save(OUT/'started.json',{'utc':now(),'protocol_sha256':q.sha(OUT/'protocol.json'),'self_test':self_test()})
    start=time.perf_counter();meta=q.metadata();old=q.read(OLD/'summary.json')
    arrays={k:np.load(OLD/'matrices'/(k+'.npy'),mmap_mode='r') for k in ('base','slots','hidden')}
    with np.load(OLD/'training_weights.npz',allow_pickle=False) as z:
        loss=z['loss_weights'].copy();y=z['y'].copy()
    assert np.array_equal(y,[w['label'] for w in meta['windows'][:168123]])
    assert abs(loss.sum()-168123)<1e-6
    mm=q.read(OLD/'matrix_manifest.json')
    for k,h in mm['files_sha256'].items():assert q.sha(OLD/'matrices'/(k+'.npy'))==h
    saved=q.read(OLD/'complete.json')['files_sha256']
    ci=[j for j,a in enumerate(meta['answers']) if a['partition']=='calibration'];lo,hi=meta['bounds']['calibration']
    all_entries={};selected={};outputs=[];replay_errors={}
    for method in q.METHODS:
        original=pickle.loads((OLD/f'{method}_C0.001.pkl').read_bytes());sc=original['scaler']
        x=np.empty((len(meta['windows']),q.WIDTHS[method]),np.float32)
        for left in range(0,len(x),q.BATCH):
            right=min(left+q.BATCH,len(x));x[left:right]=sc.transform(q.raw(arrays,method,left,right)).astype(np.float32)
        entries=[]
        for c in cfg['reused_C']:
            name=f'{method}_C{c:g}'
            for suffix in ('.pkl','_scores.npz','_result.json'):assert q.sha(OLD/(name+suffix))==saved[name+suffix]
            obj=pickle.loads((OLD/(name+'.pkl')).read_bytes())
            for key in ('mean_','scale_','var_'):assert np.array_equal(getattr(sc,key),getattr(obj['scaler'],key))
            with np.load(OLD/(name+'_scores.npz'),allow_pickle=False) as z:scores=z['window_scores'].copy()
            replay=obj['model'].predict_proba(x)[:,1];err=float(np.max(np.abs(scores-replay)))
            assert err<1e-12
            e=q.read(OLD/(name+'_result.json'));assert q.metrics(meta,scores,e['thresholds'])==e['metrics']
            assert list(q.selection_key(e['thresholds'],c))==e['selection_key']
            replay_errors[name]=err;entries.append(dict(e,source='frozen_previous_candidate'))
        for c in cfg['new_C']:
            begin=time.perf_counter();clf=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=cfg['seed'])
            clf.fit(x[:168123],y,sample_weight=loss);assert clf.n_iter_.max()<2000
            scores=clf.predict_proba(x)[:,1];aa=q.answer_scores(meta,scores)
            ts={'window':q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],scores[lo:hi]),
                'answer':q.choose_threshold([meta['answers'][j]['label'] for j in ci],aa[ci])}
            name=f'{method}_C{c:g}';obj=dict(original,model=clf,C=c,thresholds=ts,selection_key=q.selection_key(ts,c),
                protocol_sha256=q.sha(OUT/'protocol.json'),source_scaler_sha256=q.sha(OLD/f'{method}_C0.001.pkl'))
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5))
            np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=scores,answer_scores=aa)
            e={'candidate':name,'C':c,'thresholds':ts,'selection_key':list(q.selection_key(ts,c)),
               'metrics':q.metrics(meta,scores,ts),'seconds':time.perf_counter()-begin,'source':'new_fit',
               'iterations':clf.n_iter_.tolist(),'model_sha256':q.sha(OUT/(name+'.pkl')),'scores_sha256':q.sha(OUT/(name+'_scores.npz'))}
            q.save(OUT/(name+'_result.json'),e);entries.append(e)
            outputs.extend([name+'.pkl',name+'_scores.npz',name+'_result.json'])
            print('QA_REGULARIZATION_FINISHED',name,round(e['seconds'],1),flush=True)
        selected[method]=max(entries,key=lambda e:e['selection_key']);all_entries[method]=entries
        del x
    assert cfg==protocol()
    summary={'selected':selected,'all_candidates':all_entries,'old_replay_max_abs':replay_errors,'new_fits':12,
             'seconds':time.perf_counter()-start,'calibration_selection_optimistic':True,'official_test_opened':False,
             'old_selected':old['selected']}
    q.save(OUT/'summary.json',summary)
    report=['更强正则化对照，仅开发校准数据，不能作为独立测试成绩。','',
            '| 方法 | 选中C | 定位F1 | 整答F1 |','|---|---:|---:|---:|']
    for name,e in selected.items():
        m=e['metrics']['calibration'];report.append(f"| {name} | {e['C']:g} | {m['windows']['f1']:.4f} | {m['answers']['f1']:.4f} |")
    report+=['','四类同样增加3个C；既有12候选逐值回放，不重拟合。训练权重、标签、特征、标准化和阈值规则均未改。',
             '损失权重总和与C存在倒数等价关系，但本轮不把它解释为标签错误或过拟合的已证实原因。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    outputs+=['summary.json','REPORT.md','protocol.json','started.json']
    q.save(OUT/'complete.json',{'utc':now(),'files_sha256':{n:q.sha(OUT/n) for n in outputs},'official_test_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','run']);args=p.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='initialize':
            assert not (OUT/'protocol.json').exists();q.save(OUT/'protocol.json',protocol());print(self_test())
        else:run()
