"""Independent saved-model and metric replay. No fit, GPU, or official test."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
QA = EXP.parent
OLD = EXP/'probe_v1'
NF, NA = 653979, 3680
CS = (1e-5, 1e-4, .001)


def read(p): return json.loads(Path(p).read_text('utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')


def threshold(y, s):
    # Independent ascending unique bins; all observations in a tie move together.
    values, inverse = np.unique(s,return_inverse=True)
    sizes = np.bincount(inverse,minlength=len(values))
    positives = np.bincount(inverse,weights=y,minlength=len(values)).astype(np.int64)
    predicted = np.r_[np.cumsum(sizes[::-1])[::-1],0]
    tp = np.r_[np.cumsum(positives[::-1])[::-1],0]
    ts = np.r_[values,np.nextafter(values[-1],np.inf)]
    f1 = 2*tp/(predicted+int(y.sum()))
    precision = np.divide(tp,predicted,out=np.zeros(len(tp)),where=predicted>0)
    j = int(np.lexsort((ts,precision,f1))[-1])
    return dict(threshold=float(ts[j]), f1=float(f1[j]), precision=float(precision[j]),
                rows=len(y),positive=int(y.sum()))


def metric(y,s,t):
    confusion=np.bincount(2*y+(s>=t).astype(int),minlength=4)
    tn,fp,fn,tp=map(int,confusion)
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        auroc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)))


def key(entry):
    t=entry['thresholds'];w=t['window'];a=t['answer']
    return [min(w['f1'],a['f1']),w['f1'],w['precision'],-entry['C']]


def main():
    start=time.perf_counter()
    assert not (OUT/'INDEPENDENT_AUDIT.json').exists(), 'Preserve completed audit'
    complete=read(OUT/'complete.json');prepared=read(OUT/'preparation_complete.json')
    assert complete['new_LR_fits']==12 and complete['control_refits']==0
    assert complete['official_test_opened'] is False
    for name,h in complete['files_sha256'].items(): assert sha(OUT/name)==h,name
    # The prior six-control replay is inherited only through its frozen hashes.
    for path,h in prepared['source_sha256'].items():
        assert 'official_test' not in str(path).lower()
        assert sha(path)==h,path
    for name,h in prepared['files_sha256'].items(): assert sha(OUT/name)==h,name
    selfcheck=read(OUT/'CPU_SELFCHECK.json');assert selfcheck['passed']
    summary=read(OUT/'summary.json');controls=read(OUT/'CONTROLS.json')
    assert summary['deterministic_binary_controls']==controls
    answers=lines(EXP/'data/answers_fit.jsonl')+lines(QA/'data/answers_calibration.jsonl')
    windows=lines(EXP/'data/windows_k4_fit.jsonl')+lines(QA/'data/windows_k4_calibration.jsonl')
    assert len(answers)==3839 and len(windows)==696220
    assert all(a['partition']=='fit' for a in answers[:NA])
    assert all(a['partition']=='calibration' for a in answers[NA:])
    groups=[{a['group_id'] for a in aa} for aa in (answers[:NA],answers[NA:])]
    assert list(map(len,groups))==[615,154] and groups[0].isdisjoint(groups[1])
    index={a['response_id']:i for i,a in enumerate(answers)}
    assert len(index)==3839
    owner=np.array([index[w['response_id']] for w in windows],int)
    assert np.all(owner[:NF]<NA) and np.all(owner[NF:]>=NA)
    assert np.bincount(owner,minlength=3839).min()>0
    yw=np.array([w['label'] for w in windows],int)
    ya=np.array([a['label'] for a in answers],int)
    assert [int(yw[:NF].sum()),int(yw[NF:].sum()),int(ya[:NA].sum()),int(ya[NA:].sum())]==[58433,5984,1127,100]
    raw=np.load(OLD/'matrices/window65.npy',mmap_mode='r')
    assert raw.shape==(696220,65) and raw.dtype==np.float32
    with np.load(OLD/'expanded3680_weights.npz',allow_pickle=False) as z:
        assert np.array_equal(z['y'],yw[:NF]) and z['loss'].sum()==168123.
    with np.load(OUT/'type_targets_fit.npz',allow_pickle=False) as z:
        targets=z['targets'];assert targets.shape==(NF,2)
        assert np.array_equal(targets.any(1),yw[:NF].astype(bool))
        bits=np.bincount(targets[:,0]+2*targets[:,1],minlength=4).tolist()
        assert bits==selfcheck['fit_window_partition_clean_baseless_conflict_both']
    reports=[];selected={}
    for method,width in (('hidden64',64),('hidden64_risk',65)):
        entries=summary['all_candidates'][method]
        assert [e['C'] for e in entries]==list(CS)
        oldref=pickle.loads((OLD/f'expanded3680_{method}_C1e-05.pkl').read_bytes())
        for e in entries:
            name=e['candidate'];obj=pickle.loads((OUT/(name+'.pkl')).read_bytes())
            assert read(OUT/(name+'_result.json'))==e
            for suffix,h in e['files_sha256'].items(): assert sha(OUT/(name+suffix))==h
            assert obj['C']==e['C'] and obj['fit_rows']==NF and obj['method']==method
            assert obj['head_order']==['baseless','conflict'] and obj['fixed_aggregation']=='max'
            assert obj['thresholds']==e['thresholds']
            assert obj['original_binary_weight_sha256']==sha(OLD/'expanded3680_weights.npz')
            assert obj['targets_sha256']==sha(OUT/'type_targets_fit.npz')
            scaler=obj['scaler']
            for field in ('mean_','var_','scale_','n_samples_seen_'):
                assert np.array_equal(getattr(scaler,field),getattr(oldref['scaler'],field))
            x=np.empty((len(raw),width),np.float32)
            for lo in range(0,len(raw),16384):
                hi=min(lo+16384,len(raw))
                # Match the defined float32 round after each arithmetic operation.
                piece=np.array(raw[lo:hi,:width],np.float32,copy=True)
                piece-=scaler.mean_;piece/=scaler.scale_
                assert np.array_equal(piece,scaler.transform(raw[lo:hi,:width]).astype(np.float32))
                x[lo:hi]=piece
            with np.load(OUT/(name+'_scores.npz'),allow_pickle=False) as z:
                saved={k:z[k].copy() for k in z.files}
            ps=[];errors=[]
            for j,head in enumerate(obj['heads']):
                assert head.C==e['C'] and head.random_state==20260924
                assert head.solver=='liblinear' and head.max_iter==2000
                assert head.penalty=='l2' and head.class_weight is None
                assert np.array_equal(head.classes_,[0,1])
                assert head.coef_.shape==(1,width) and np.isfinite(head.coef_).all()
                assert 0<int(head.n_iter_.max())<2000
                p=head.predict_proba(x)[:,1]
                assert np.array_equal(p,saved['head_scores'][:,j])
                independent=expit((x@head.coef_.T+head.intercept_).ravel())
                err=float(np.abs(independent-p).max());assert err<2e-14
                errors.append(err);ps.append(p)
            stack=np.column_stack(ps);assert np.array_equal(stack,saved['head_scores'])
            win=np.maximum(ps[0],ps[1]);assert np.array_equal(win,saved['window_scores'])
            ans=np.full(3839,-np.inf);np.maximum.at(ans,owner,win)
            assert np.array_equal(ans,saved['answer_scores'])
            ts={'window':threshold(yw[NF:],win[NF:]),'answer':threshold(ya[NA:],ans[NA:])}
            assert ts==e['thresholds']
            metrics={}
            for part,wl,al in (('fit',slice(0,NF),slice(0,NA)),('calibration',slice(NF,None),slice(NA,None))):
                metrics[part]={'windows':metric(yw[wl],win[wl],ts['window']['threshold']),
                               'answers':metric(ya[al],ans[al],ts['answer']['threshold'])}
            assert metrics==e['metrics'] and key(e)==e['selection_key']
            assert e['n_iter']==[h.n_iter_.tolist() for h in obj['heads']]
            reports.append({'candidate':name,'all_head_probabilities_exact':True,
                'manual_coef_probability_max_errors':errors,'scaler_float32_exact':True,
                'window_max_exact':True,'answer_max_exact':True,'thresholds':ts,'metrics':metrics,
                'n_iter':e['n_iter'],'fixed_same_C':True})
            print('INDEPENDENT_DIRECT_TYPE_REPLAY',name,'exact',flush=True)
            del x
        chosen=max(entries,key=key);assert chosen==summary['selected'][method]
        prior=controls[method]['selected']
        assert max(controls[method]['all_candidates'],key=lambda r:r['selection_key'])==prior
        nw=chosen['metrics']['calibration']['windows']['f1'];na=chosen['metrics']['calibration']['answers']['f1']
        ow=prior['metrics']['calibration']['windows']['f1'];oa=prior['metrics']['calibration']['answers']['f1']
        selected[method]={'new':chosen['candidate'],'control':prior['candidate'],
            'C':chosen['C'],'window_f1':nw,'answer_f1':na,'control_window_f1':ow,'control_answer_f1':oa,
            'window_f1_delta':nw-ow,'answer_f1_delta':na-oa}
    report={'passed':True,'scope':'Saved12heads;6candidates;fit/cal only;prior6controls by frozen preparation hashes',
        'complete_sha256':sha(OUT/'complete.json'),'auditor_sha256':sha(__file__),
        'new_fits':0,'GPU_used':False,'official_test_opened':False,
        'rows':{'fit_windows':NF,'calibration_windows':42241,'fit_answers':NA,'calibration_answers':159},
        'source_groups_disjoint':True,'original_binary_weights_reused':True,
        'fit_type_OR_exact':True,'candidates':reports,'selected':selected,
        'seconds':time.perf_counter()-start,
        'limits':'Repeated calibration development;fixed max is not calibrated union;duplicated old head is not independently learned capacity control.'}
    save(OUT/'INDEPENDENT_AUDIT.json',report)
    rows=['# Direct type heads: independent audit','',
          'Passed. All 12 saved heads reproduce all 696,220 window probabilities exactly. Both-head max, 3,839 answer maxima, calibration thresholds, fit/cal counts and the shared-C choices match. No fitting, GPU or official test access.','',
          '| Input | Selected C | Window F1 | Answer F1 | Window delta vs selected binary | Answer delta |',
          '|---|---:|---:|---:|---:|---:|']
    for m,r in selected.items():
        rows.append(f"| {m} | {r['C']:g} | {r['window_f1']:.9f} | {r['answer_f1']:.9f} | {r['window_f1_delta']:+.9f} | {r['answer_f1_delta']:+.9f} |")
    rows+=['','Both input variants lose localization F1 against their selected binary controls. This is an observed result under the fixed labels, shared binary weights and max aggregation, not evidence that type supervision cannot work. The two learned boundaries also change capacity relative to the deterministic duplicated control.','']
    (OUT/'INDEPENDENT_AUDIT.md').write_text('\n'.join(rows),'utf-8')
    print('DIRECT_TYPE_AUDIT_PASSED',json.dumps(selected),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        try: main()
        except BaseException as exc:
            save(OUT/f'AUDIT_FAILURE_{time.time_ns()}.json',{'error':repr(exc),'new_fits':0,'GPU_used':False})
            raise
