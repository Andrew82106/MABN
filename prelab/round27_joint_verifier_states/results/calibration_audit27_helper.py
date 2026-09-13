"""Independent R27 LR calibration and full-answer persistence replay; no fitting."""
from pathlib import Path
from collections import defaultdict
import hashlib
import itertools
import json
import pickle
import sys
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;PRE=ROOT.parent
R23=PRE/'round23b_local_evidence_probe';R22=PRE/'round22_evidence_verification'
FAMILIES=('joint_mean','joint_slots');METHODS=FAMILIES+tuple(m+'_smooth' for m in FAMILIES)
BINDINGS={}

class State:
    def __setstate__(self,s):self.__dict__.update(s)

class InertUnpickler(pickle.Unpickler):
    def find_class(self,m,n):
        if m.startswith('sklearn.'):return State
        if m.startswith('numpy') or m in ('builtins','collections'):return super().find_class(m,n)
        raise pickle.UnpicklingError(f'Unexpected global {m}.{n}')

def sha(p):
    p=Path(p).resolve();assert any(p.is_relative_to(r) for r in (ROOT,R23,R22))
    h=hashlib.sha256(p.read_bytes()).hexdigest();BINDINGS[str(p.relative_to(PRE))]=h
    return h

def read(p):sha(p);return json.loads(Path(p).read_text('utf-8'))
def readl(p):sha(p);return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]

def threshold(y,s):
    assert len(y)==len(s) and set(y)=={0,1} and np.isfinite(s).all()
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    p=np.bincount(inv,weights=y,minlength=len(u)).astype(np.int64)
    tp=np.r_[p.sum(),np.cumsum(p[::-1])[::-1],0]
    count=np.r_[len(y),np.cumsum(n[::-1])[::-1],0]
    ts=np.r_[np.nextafter(u[0],-np.inf),u,np.nextafter(u[-1],np.inf)]
    f1=2*tp/(count+y.sum());precision=np.divide(tp,count,out=np.zeros(len(count)),where=count!=0)
    k=max(range(len(ts)),key=lambda j:(f1[j],precision[j],ts[j]))
    return dict(threshold=float(ts[k]),validation_f1=float(f1[k]),validation_precision=float(precision[k]),
        tokens=len(y),scorable_tokens=len(y),risk_tokens=int(y.sum()))

def compare_thresholds(a,b,tol=1e-12):
    error=0.;exact=0
    for level in ('window','answer'):
        assert a[level].keys()==b[level].keys()
        err=max(abs(float(a[level][k])-float(b[level][k])) for k in a[level])
        assert err<=tol,(level,a[level],b[level],err)
        error=max(error,err);exact+=int(a[level]==b[level])
    return error,exact

def lr_scores(x,obj):
    assert x.dtype==np.float32 and np.array_equal(obj['model'].classes_,[0,1])
    assert obj['scaler'].with_mean and obj['scaler'].with_std
    z=x.copy();z-=obj['scaler'].mean_;z/=obj['scaler'].scale_
    return expit((z@obj['model'].coef_.T+obj['model'].intercept_).ravel())

def posterior(s,chains,stay,temp):
    if stay==.5 and temp==1.:return s.copy()
    q=np.clip(s,1e-12,1-1e-12);p=expit((np.log(q)-np.log1p(-q))/temp)
    out=np.empty_like(s);switch=1-stay
    for ix in chains:
        e=p[ix];n=len(ix);f0=np.empty(n);f1=np.empty(n)
        f0[0]=1-e[0];f1[0]=e[0]
        for j in range(1,n):
            z0=(1-e[j])*(stay*f0[j-1]+switch*f1[j-1])
            z1=e[j]*(switch*f0[j-1]+stay*f1[j-1])
            den=z0+z1;f0[j]=z0/den;f1[j]=z1/den
        b0=b1=1.;out[ix[-1]]=f1[-1]/(f0[-1]+f1[-1])
        for j in range(n-2,-1,-1):
            z0=stay*(1-e[j+1])*b0+switch*e[j+1]*b1
            z1=switch*(1-e[j+1])*b0+stay*e[j+1]*b1
            den=z0+z1;b0=z0/den;b1=z1/den
            out[ix[j]]=f1[j]*b1/(f0[j]*b0+f1[j]*b1)
    return out

def main():
    complete=read(OUT/'complete.json');freeze=read(OUT/'fit_freeze.json');cfg=read(ROOT/'protocol.json')
    assert sha(OUT/'fit_freeze.json')==complete['files_sha256']['fit_freeze.json']
    assert sha(ROOT/'src/run27.py')==freeze['snapshot']['code_sha256']
    assert sha(ROOT/'protocol.json')==freeze['snapshot']['protocol_sha256']
    assert cfg['C']==[.001,.01,.1] and cfg['stay']==[.5,.8,.95,.99] and cfg['temperature']==[.5,1.,2.]
    r23_freeze=read(R23/'results/fit_freeze.json')
    assert sha(R23/'results/fit_freeze.json')==freeze['snapshot']['r23_fit_sha256']
    for name in ('designs.npz','candidate_windows.jsonl'):
        assert sha(R23/'results'/name)==r23_freeze['files_sha256'][name]
    windows=readl(R23/'results/candidate_windows.jsonl');items=readl(R22/'results/answer_index22.jsonl')
    assert len(windows)==12222 and len(items)==602
    assert all(w['split']=='train' for w in windows) and all(a['split']=='train' for a in items)
    assert len({w['window_key'] for w in windows})==len(windows)
    itemmap={a['item_id']:a for a in items};assert len(itemmap)==602
    byitem=defaultdict(list)
    for j,w in enumerate(windows):
        assert len(w['item_ids'])==1;iid=w['item_ids'][0];a=itemmap[iid]
        assert all(w[k]==a[k] for k in ('row_id','group_id','condition'))
        byitem[iid].append(j)
    assert set(byitem)==set(itemmap)
    # R24 geometry: one whole answer chain, sorted by raw start; do not split gaps.
    chains=[np.asarray(sorted(ix,key=lambda j:windows[j]['raw_token_indices'][0]),int) for ix in byitem.values()]
    assert len(chains)==602 and np.array_equal(np.sort(np.concatenate(chains)),np.arange(12222))
    raw_gaps=sum(int(np.sum(np.diff([windows[j]['raw_token_indices'][0] for j in ix])!=1)) for ix in chains)
    assert sum(w['main_eligible'] for w in windows)==9526 and sum(a['main_eligible'] for a in items)==598
    with np.load(R23/'results/designs.npz',allow_pickle=False) as z:base=z['base'].copy();slots=z['slots'].copy()
    assert base.shape==(12222,785) and slots.shape==(12222,3144)
    allgroups={a['group_id'] for a in items};assert len(allgroups)==278
    fold_rows=[];lr_rows=[];smooth_rows=[];partitions=[];selected_reload={}
    lr_error=sm_error=lr_threshold_error=sm_threshold_error=0.
    lr_threshold_exact=sm_threshold_exact=identity_exact=0
    for fold in range(5):
        path=OUT/f'fold_{fold}_frozen.pkl';assert sha(path)==freeze['files_sha256'][path.name]
        with path.open('rb') as f:obj=InertUnpickler(f).load()
        path=OUT/f'fold_{fold}_scores.npz';assert sha(path)==freeze['files_sha256'][path.name]
        with np.load(path,allow_pickle=False) as z:
            local=z['local65'].copy();global_=z['global65'].copy();saved={m:z[m].copy() for m in METHODS}
        assert local.shape==global_.shape==(12222,65) and local.dtype==global_.dtype==np.float32
        assert all(v.shape==(12222,) and np.isfinite(v).all() for v in saved.values())
        g={k:set(v) for k,v in obj['groups'].items()};fg,cg,eg=(g[k] for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert not(fg&cg or fg&eg or cg&eg) and fg|cg|eg==allgroups;partitions.append(g)
        cx=[j for j,w in enumerate(windows) if w['group_id'] in cg]
        wi=np.asarray([j for j in cx if windows[j]['main_eligible']],int)
        ca=[a for a in items if a['group_id'] in cg];ai=[a for a in ca if a['main_eligible']]
        yw=np.asarray([windows[j]['gold'] for j in wi],int);ya=np.asarray([a['gold'] for a in ai],int)
        assert all(set(byitem[a['item_id']]).issubset(set(cx)) for a in ca)
        def thresholds(v):
            every={a['item_id']:v[byitem[a['item_id']]].max() for a in ca}
            aa=np.asarray([every[a['item_id']] for a in ai])
            return {'window':threshold(yw,v[wi]),'answer':threshold(ya,aa)}
        row=dict(fold=fold,cal_groups=len(cg),cal_all_windows=len(cx),cal_main_windows=len(wi),
            cal_window_positive=int(yw.sum()),cal_all_answers=len(ca),cal_main_answers=len(ai),cal_answer_positive=int(ya.sum()),
            cal_safe_refusal_answers=sum(a['reviewed_safe_refusal'] for a in ai),choices={})
        for family,original in [('joint_mean',base),('joint_slots',slots)]:
            x=np.column_stack((original,local,global_)).astype(np.float32)
            chosen=obj['models'][family];candidates=chosen['all_lr_candidates'];table=chosen['calibration_candidates']
            assert len(candidates)==len(table)==3
            keys=[]
            for j,(c,candidate,entry) in enumerate(zip(cfg['C'],candidates,table)):
                assert candidate['C']==candidate['model'].C==entry['C']==c
                v=lr_scores(x,candidate);assert v.shape==(12222,) and np.isfinite(v).all()
                ts=thresholds(v);error,exact=compare_thresholds(ts,entry['thresholds'])
                lr_threshold_error=max(lr_threshold_error,error);lr_threshold_exact+=exact
                w,a=ts['window'],ts['answer'];key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],-c]
                assert key==list(entry['selection_key']);keys.append(key)
                lr_rows.append(dict(fold=fold,family=family,C=c,thresholds=ts,selection_key=key))
                if j==chosen['selected_candidate']:
                    error=float(np.max(np.abs(v-saved[family])));lr_error=max(lr_error,error);assert error<=1e-12
                    assert obj['thresholds'][family]==entry['thresholds']
                    for k in ('coef_','intercept_','classes_'):assert np.array_equal(getattr(chosen['model'],k),getattr(candidate['model'],k))
                    for k in ('mean_','scale_'):assert np.array_equal(getattr(chosen['scaler'],k),getattr(candidate['scaler'],k))
            best=max(range(3),key=lambda j:keys[j]);assert best==chosen['selected_candidate'] and chosen['C']==candidates[best]['C']
            assert thresholds(saved[family])==obj['thresholds'][family]
            name=family+'_smooth';table=obj['smoothing_tables'][name];assert len(table)==12
            keys=[];reconstructed=[]
            for j,(stay,temp) in enumerate(itertools.product(cfg['stay'],cfg['temperature'])):
                entry=table[j];assert entry['stay']==stay and entry['temperature']==temp
                v=posterior(saved[family],chains,stay,temp);ts=thresholds(v)
                error,exact=compare_thresholds(ts,entry['thresholds']);sm_threshold_error=max(sm_threshold_error,error);sm_threshold_exact+=exact
                w,a=ts['window'],ts['answer'];key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],
                    int(stay==.5 and temp==1.),-stay,-abs(float(np.log(temp)))]
                assert key==list(entry['key']);keys.append(key);reconstructed.append(v)
                smooth_rows.append(dict(fold=fold,family=family,stay=stay,temperature=temp,thresholds=ts,key=key))
                if stay==.5 and temp==1.:
                    assert np.array_equal(v,saved[family]) and ts==obj['thresholds'][family];identity_exact+=1
            best_s=max(range(12),key=lambda j:keys[j]);choice=table[best_s]
            assert obj['smoothing_choices'][name]==choice and obj['thresholds'][name]==choice['thresholds']
            error=float(np.max(np.abs(reconstructed[best_s]-saved[name])));sm_error=max(sm_error,error);assert error<=1e-12
            assert thresholds(saved[name])==obj['thresholds'][name]
            row['choices'][family]=dict(C=chosen['C'],stay=choice['stay'],temperature=choice['temperature'],
                cal_window_f1=choice['thresholds']['window']['validation_f1'],cal_answer_f1=choice['thresholds']['answer']['validation_f1'])
        selected_reload[fold]=(saved,obj['thresholds'],eg)
        fold_rows.append(row);print(json.dumps(row),flush=True)
    assert all(partitions[f]['calibration_groups']==partitions[(f+1)%5]['evaluation_groups'] for f in range(5))
    # New methods only: ensure the separate output pass reloaded the chosen scores/cutoffs.
    for name in ('window_scores_oof.jsonl','answer_scores_oof.jsonl'):
        assert sha(OUT/name)==complete['files_sha256'][name]
    oofw=readl(OUT/'window_scores_oof.jsonl');oofa=readl(OUT/'answer_scores_oof.jsonl')
    assert len(oofw)==12222 and len(oofa)==602
    windex={w['window_key']:j for j,w in enumerate(windows)}
    assert {w['window_key'] for w in oofw}==set(windex)
    assert {a['item_id'] for a in oofa}==set(itemmap)
    for w in oofw:
        saved,ts,eg=selected_reload[w['fold']];j=windex[w['window_key']]
        assert w['group_id'] in eg and w['main_eligible']==windows[j]['main_eligible'] and w['gold']==windows[j]['gold']
        for m in METHODS:
            assert w['scores'][m]==float(saved[m][j])
            assert w['predictions'][m]==bool(saved[m][j]>=ts[m]['window']['threshold'])
    for a in oofa:
        saved,ts,eg=selected_reload[a['fold']];original=itemmap[a['item_id']]
        assert a['group_id'] in eg and a['main_eligible']==original['main_eligible'] and a['gold']==original['gold']
        for m in METHODS:
            v=float(saved[m][byitem[a['item_id']]].max())
            assert a['scores'][m]==v and a['predictions'][m]==bool(v>=ts[m]['answer']['threshold'])
    assert sha(OUT/'summary.json')==complete['files_sha256']['summary.json']
    summary=read(OUT/'summary.json');oof_metrics={}
    for m in METHODS:
        oof_metrics[m]={}
        for unit,rows in [('windows',oofw),('answers',oofa)]:
            eligible=[r for r in rows if r['main_eligible']]
            y=np.asarray([r['gold'] for r in eligible],bool)
            pred=np.asarray([r['predictions'][m] for r in eligible],bool)
            tp=int(np.sum(y&pred));fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred));tn=int(np.sum(~y&~pred))
            got=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp),recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn),alert_rate=float(pred.mean()),risk_rate=float(y.mean()),
                **{unit:len(y),'risk_'+unit:int(y.sum())})
            assert got==summary['methods'][m][unit],(m,unit,got,summary['methods'][m][unit])
            oof_metrics[m][unit]=got
    test_started=read(OUT/'test_started.json');assert test_started['freeze_sha256']==sha(OUT/'fit_freeze.json')
    assert 'utc' not in freeze and 'utc' not in test_started
    assert len(lr_rows)==30 and len(smooth_rows)==120 and identity_exact==10
    report=dict(status='passed_with_provenance_limitations',numerical_blockers=[],no_new_training_or_GPU=True,
        original_heldout_or_QA_test_read_by_auditor=False,no_production_imports=True,
        LR_candidates_replayed=30,LR_thresholds_verified=60,LR_threshold_dictionaries_bit_exact=lr_threshold_exact,
        LR_calibration_keys_exact=30,C_selections_exact=10,max_LR_threshold_abs_error=lr_threshold_error,
        selected_raw_score_arrays_compared=10,max_selected_raw_score_abs_error=lr_error,
        smoothing_candidates_reconstructed=120,smoothing_thresholds_verified=240,
        smoothing_threshold_dictionaries_bit_exact=sm_threshold_exact,max_smoothing_threshold_abs_error=sm_threshold_error,
        smoothing_keys_exact=120,smoothing_choices_exact=10,identity_scores_and_thresholds_exact=10,
        selected_smooth_score_arrays_compared=10,max_selected_smooth_score_abs_error=sm_error,
        new_four_methods_selected_thresholds_on_reloaded_scores_exact=40,
        all_new_OOF_scores_predictions_exact=True,OOF_window_score_comparisons=12222*4,OOF_answer_score_comparisons=602*4,
        new_OOF_main_metric_blocks_exact=8,new_OOF_main_metrics=oof_metrics,
        smoothing_geometry=dict(whole_answer_chains=602,raw_start_gaps_not_split=raw_gaps,all_candidate_windows=12222,
            safe_refusal_nonmain_windows_retained=True,answermax_before_main_answer_filter=True),
        folds=fold_rows,LR_candidates=lr_rows,smoothing_candidates=smooth_rows,bindings=BINDINGS,helper_sha256=sha(Path(__file__)),
        limits=['R27 prepare inherits R26 prepare/train_metadata, which parses mixed-split input before selecting actual train; cannot claim heldout input text was never parsed historically.',
            'Freeze and outer-start records have no UTC execution timestamps; hash binding and code gates do not independently prove historical execution order.',
            'Repeated assistant-labeled actual-train development folds, not a fresh final test or human QA evidence.',
            'Used saved local65/global65; source mapping, old PCA fit isolation, weights and scaler derivation are delegated to parent.',
            'Did not replay the17 old baseline sources, bootstrap, ranking or highlight metrics.'])
    target=OUT/'CALIBRATION_AUDIT27.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'status':report['status'],'report':str(target),'sha256':sha(target),
        'LR_thresholds_exact':lr_threshold_exact,'smoothing_thresholds_exact':sm_threshold_exact,
        'max_LR_score_error':lr_error,'max_smoothing_score_error':sm_error,
        'max_LR_threshold_error':lr_threshold_error,'max_smoothing_threshold_error':sm_threshold_error},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
