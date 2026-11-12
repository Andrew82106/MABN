"""Independent frozen-result CPU audit. Never imports a production runner or model."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RESULTS=ROOT/'results'
BINDINGS={}

def sha(p):
    p=Path(p).resolve()
    assert p.is_relative_to(ROOT)
    rel=str(p.relative_to(ROOT))
    assert not any(x in rel.lower() for x in ('sealed','withheld','test'))
    h=hashlib.sha256(p.read_bytes()).hexdigest();BINDINGS[rel]=h
    return h

def read(p):
    sha(p)
    return json.loads(Path(p).read_text('utf-8'))

def choose(y,s):
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    positive=np.bincount(inv,weights=y,minlength=len(u)).astype(np.int64)
    tp=np.r_[np.cumsum(positive[::-1])[::-1],0]
    rows=np.r_[np.cumsum(n[::-1])[::-1],0]
    thresholds=np.r_[u,np.nextafter(u[-1],np.inf)]
    f1=2*tp/(rows+y.sum())
    precision=np.divide(tp,rows,out=np.zeros(len(rows)),where=rows!=0)
    j=max(range(len(rows)),key=lambda i:(f1[i],precision[i],thresholds[i]))
    return dict(threshold=float(thresholds[j]),f1=float(f1[j]),precision=float(precision[j]),rows=len(y),positive=int(y.sum()))

def metric(y,s,t):
    pred=s>=t;tp=int(y[pred].sum());fp=int(pred.sum())-tp
    fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    pos=np.bincount(inv,weights=y,minlength=len(u));neg=n-pos
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(y.sum()*(len(y)-y.sum()))
    ap=np.sum(pos[::-1]/y.sum()*np.cumsum(pos[::-1])/np.cumsum(n[::-1]))
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
        f1=2*tp/(2*tp+fp+fn),auroc=float(auc),average_precision=float(ap))

def main():
    # Do not inspect training outputs until this terminal completion artifact exists.
    assert (OUT/'complete.json').exists(),'Training has not completed; do not read partial outputs'
    done=read(OUT/'complete.json');assert done['status']=='complete_development_only'
    assert done['test_opened'] is False and done['gpu_used'] is False
    assert sha(ROOT/'src/run_window_sequence.py')==done['script_sha256']
    frozen={k.replace('\\','/'):v for k,v in done['files_sha256'].items()}
    def frozen_read(name):
        assert sha(OUT/name)==frozen[name]
        return read(OUT/name)
    protocol=frozen_read('protocol.json');summary=frozen_read('summary.json')
    assert protocol['epochs']==30 and protocol['configurations']==1
    assert summary['epochs']==30 and summary['new_models']==1 and not summary['test_opened']
    assert summary['calibration_results_are_selection_optimistic']
    prior_done=read(RESULTS/'development_v1/complete.json')
    index_path=RESULTS/'development_v1/score_index.json'
    assert sha(index_path)==prior_done['files_sha256']['score_index.json']
    idx=read(index_path);windows,answers=idx['windows'],idx['answers']
    assert len(windows)==210364 and len(answers)==793
    assert len({w['window_id'] for w in windows})==len(windows)
    assert len({a['answer_id'] for a in answers})==len(answers)
    byanswer=defaultdict(list)
    for j,w in enumerate(windows):byanswer[w['answer_id']].append(j)
    ax=[np.asarray(byanswer[a['answer_id']],int) for a in answers]
    assert all(len(x) for x in ax) and np.array_equal(np.concatenate(ax),np.arange(210364))
    y=np.asarray([w['label'] for w in windows],int);ya=np.asarray([a['label'] for a in answers],int)
    assert set(y)==set(ya)=={0,1}
    bounds={'fit':(0,168123,0,634),'calibration':(168123,210364,634,793)}
    denominators={}
    for part,(l,r,al,ar) in bounds.items():
        assert all(w['partition']==part for w in windows[l:r])
        assert all(a['partition']==part for a in answers[al:ar])
        for a,ix in zip(answers[al:ar],ax[al:ar]):
            assert all(windows[j]['group_id']==a['group_id'] for j in ix)
        denominators[part]=dict(windows=r-l,positive_windows=int(y[l:r].sum()),
            answers=ar-al,positive_answers=int(ya[al:ar].sum()),
            groups=len({a['group_id'] for a in answers[al:ar]}))
    assert not ({a['group_id'] for a in answers[:634]} & {a['group_id'] for a in answers[634:]})
    def load_scores(path,expected):
        assert sha(path)==expected
        with np.load(path,allow_pickle=False) as z:
            w=z['window_scores'].astype(np.float64);a=z['answer_scores'].astype(np.float64)
        assert w.shape==(210364,) and a.shape==(793,) and np.isfinite(w).all() and np.isfinite(a).all()
        assert np.all((w>=0)&(w<=1))
        assert np.array_equal(np.asarray([w[ix].max() for ix in ax]),a)
        return w,a
    error=0.;metric_blocks=0;threshold_count=0;epoch_rows=[];keys=[]
    def check_metrics(entry,w,a):
        nonlocal error,metric_blocks
        for part,(l,r,al,ar) in bounds.items():
            for g,yy,ss in [('windows',y[l:r],w[l:r]),('answers',ya[al:ar],a[al:ar])]:
                got=metric(yy,ss,entry['thresholds']['window' if g=='windows' else 'answer']['threshold'])
                expected=entry['metrics'][part][g];assert got.keys()==expected.keys()
                delta=max(abs(float(got[k])-float(expected[k])) for k in got)
                error=max(error,delta);assert delta<=1e-12,(part,g,delta)
                metric_blocks+=1
    history=summary['all_epochs']['window_tcn'];assert len(history)==30
    for epoch in range(1,31):
        stem=f'window_tcn/epoch_{epoch:03d}'
        entry=frozen_read(stem+'.json');assert entry==history[epoch-1]
        assert entry['epoch']==epoch and entry['method']=='window_tcn'
        assert entry['scores_sha256']==frozen[stem+'_scores.npz']
        assert entry['model_sha256']==frozen[stem+'.pt']
        w,a=load_scores(OUT/(stem+'_scores.npz'),entry['scores_sha256'])
        ts={'window':choose(y[168123:],w[168123:]),'answer':choose(ya[634:],a[634:])}
        assert ts==entry['thresholds'],(epoch,ts,entry['thresholds'])
        threshold_count+=2
        ww,aa=ts['window'],ts['answer'];key=[min(ww['f1'],aa['f1']),ww['f1'],ww['precision'],-epoch]
        assert key==entry['selection_key'];keys.append(key)
        check_metrics(entry,w,a)
        epoch_rows.append(dict(epoch=epoch,cal_window_f1=ww['f1'],cal_answer_f1=aa['f1'],
                               window_threshold=ww['threshold'],answer_threshold=aa['threshold']))
        if epoch%10==0:print('WINDOW_TCN_AUDIT_EPOCH',epoch,flush=True)
    selected_index=max(range(30),key=lambda i:keys[i]);selected=summary['selected']['window_tcn']
    assert selected==history[selected_index]
    comparisons=[dict(method='window_tcn',epoch=selected['epoch'],
        cal_window_f1=selected['metrics']['calibration']['windows']['f1'],
        cal_answer_f1=selected['metrics']['calibration']['answers']['f1'])]
    for folder,method in [('development_v1','hidden64_lookback_nll'),('sequence_v1','token_tcn')]:
        src=RESULTS/folder;source_done=read(src/'complete.json')
        source_hashes={k.replace('\\','/'):v for k,v in source_done['files_sha256'].items()}
        assert sha(src/'summary.json')==source_hashes['summary.json']
        old=read(src/'summary.json')['selected'][method]
        name=f"token_tcn/epoch_{old['epoch']:03d}_scores.npz" if folder=='sequence_v1' else old['candidate']+'_scores.npz'
        w,a=load_scores(src/name,source_hashes[name]);check_metrics(old,w,a)
        assert {'window':choose(y[168123:],w[168123:]),'answer':choose(ya[634:],a[634:])}==old['thresholds']
        comparisons.append(dict(method=method,epoch=old.get('epoch'),C=old.get('C'),
            cal_window_f1=old['metrics']['calibration']['windows']['f1'],
            cal_answer_f1=old['metrics']['calibration']['answers']['f1']))
    helper_hash=sha(Path(__file__))
    report=dict(passed=True,blockers=[],scope='Only frozen fit/cal artifacts; no model/production imports, training, GPU or test reads',
        epochs=30,threshold_count_exact=threshold_count,selection_keys_exact=30,
        selected_epoch=selected['epoch'],selected_epoch_exact=True,metric_blocks=metric_blocks,
        max_metric_abs_error=error,all_30_answermax_exact=True,denominators=denominators,
        comparisons=comparisons,epoch_rows=epoch_rows,calibration_results_are_selection_optimistic=True,
        bindings=BINDINGS,helper_sha256=helper_hash,
        limitations=['Result identity and labels use prior frozen score_index; full source/gold geometry audit is delegated to parent.',
            'No model forward, fitting, PCA/scaler/weight reconstruction or full-fit BCE replay in this audit.',
            'Reported F1 selected on calibration, not independent test; TCN versus LR changes architecture, and versus tokenTCN also changes input and supervision.',
            'Verified hashes cover inspected result/metadata artifacts; large feature matrices and model bytes belong to parent audit.'])
    target=OUT/'CALIBRATION_AUDIT_WINDOW_TCN.json';assert not target.exists()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'passed':True,'report':str(target),'sha256':sha(target),'selected_epoch':selected['epoch'],
        'thresholds_exact':threshold_count,'metric_blocks':metric_blocks,'max_metric_abs_error':error,
        'comparisons':comparisons,'denominators':denominators},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
