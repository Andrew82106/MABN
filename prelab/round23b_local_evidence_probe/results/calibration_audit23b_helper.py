"""Independent CPU-only calibration replay; no production modules or model fitting."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import pickle
import sys
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
BINDINGS={}
FAMILIES=('local_probe','local_mean_fusion','local_slots_fusion')

class State:
    def __setstate__(self,state):self.__dict__.update(state)

class ArraysOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('sklearn.'):return State
        if module.startswith('numpy') or module in ('builtins','collections'):
            return super().find_class(module,name)
        raise pickle.UnpicklingError(f'Unexpected executable pickle global {module}.{name}')

def sha(path):
    path=Path(path).resolve();assert path.is_relative_to(ROOT)
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    BINDINGS[str(path.relative_to(ROOT))]=h
    return h

def read(path):
    sha(path);return json.loads(Path(path).read_text('utf-8'))

def readl(path):
    sha(path);return [json.loads(s) for s in Path(path).read_text('utf-8').splitlines() if s]

def threshold(y,s):
    # Ascending unique-score buckets, both all/none endpoints, lexicographic ties.
    assert len(y)==len(s) and set(y)=={0,1} and np.isfinite(s).all()
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    p=np.bincount(inv,weights=y,minlength=len(u)).astype(np.int64)
    positive=np.r_[p.sum(),np.cumsum(p[::-1])[::-1],0]
    counts=np.r_[len(y),np.cumsum(n[::-1])[::-1],0]
    ts=np.r_[np.nextafter(u[0],-np.inf),u,np.nextafter(u[-1],np.inf)]
    f1=2*positive/(counts+y.sum())
    precision=np.divide(positive,counts,out=np.zeros(len(counts)),where=counts!=0)
    k=max(range(len(ts)),key=lambda j:(f1[j],precision[j],ts[j]))
    return dict(threshold=float(ts[k]),validation_f1=float(f1[k]),validation_precision=float(precision[k]),
                tokens=len(y),scorable_tokens=len(y),risk_tokens=int(y.sum()))

def metrics(y,s,t,unit):
    pred=s>=t;tp=int(y[pred].sum());fp=int(pred.sum())-tp
    fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    pos=np.bincount(inv,weights=y,minlength=len(u));neg=n-pos
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(y.sum()*(len(y)-y.sum()))
    ap=np.sum(pos[::-1]/y.sum()*np.cumsum(pos[::-1])/np.cumsum(n[::-1]))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),auroc=float(auc),average_precision=float(ap),
        alert_rate=float(pred.mean()),risk_rate=float(y.mean()),
        **{unit:len(y),'risk_'+unit:int(y.sum())})

def replay(x,obj):
    assert x.dtype==np.float32
    scaler,model=obj['scaler'],obj['model']
    assert scaler.with_mean and scaler.with_std and np.array_equal(model.classes_,[0,1])
    z=x.copy();z-=scaler.mean_;z/=scaler.scale_
    # Preserve original float32 scaler rounding before float64 binary LR dot product.
    return expit((z@model.coef_.T+model.intercept_).ravel())

def main():
    complete=read(OUT/'complete.json');freeze=read(OUT/'fit_freeze.json')
    protocol=read(ROOT/'protocol.json');summary=read(OUT/'summary.json')
    assert protocol['C']==[.001,.01,.1] and protocol['new_lr_fits']==freeze['new_lr_fits']==45
    assert not protocol['original_validation_or_test_used']
    assert sha(ROOT/'src/run23.py')==freeze['snapshot']['code_sha256']
    assert sha(ROOT/'protocol.json')==freeze['snapshot']['protocol_sha256']
    for f in ('fit_freeze.json','summary.json','answer_scores_oof.jsonl'):
        assert sha(OUT/f)==complete['files_sha256'][f]
    for f in ('designs.npz','candidate_windows.jsonl'):
        assert sha(OUT/f)==freeze['files_sha256'][f]
    windows=readl(OUT/'candidate_windows.jsonl')
    # Only metadata keys are retained; OOF predictions are not used to choose anything.
    metadata_fields=('item_id','row_id','group_id','condition','gold','main_eligible','reviewed_safe_refusal')
    items=[{k:a[k] for k in metadata_fields} for a in readl(OUT/'answer_scores_oof.jsonl')]
    assert len(windows)==12222 and len(items)==602
    assert all(w['split']=='train' for w in windows)
    assert len({w['window_key'] for w in windows})==len(windows)
    itemmap={a['item_id']:a for a in items};assert len(itemmap)==602
    byitem=defaultdict(list)
    for j,w in enumerate(windows):
        assert len(w['item_ids'])==1
        iid=w['item_ids'][0];a=itemmap[iid]
        assert all(w[k]==a[k] for k in ('row_id','group_id','condition'))
        byitem[iid].append(j)
    assert set(byitem)==set(itemmap)
    assert sum(a['main_eligible'] for a in items)==598
    assert sum(w['main_eligible'] for w in windows)==9526
    with np.load(OUT/'designs.npz',allow_pickle=False) as z:
        designs={k:z[k].copy() for k in ('base','slots','extra')}
    assert {k:v.shape for k,v in designs.items()}=={'base':(12222,785),'slots':(12222,3144),'extra':(12222,2)}
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in designs.values())
    allgroups={a['group_id'] for a in items};assert len(allgroups)==278
    fold_records=[];candidate_records=[];max_score_error=0.;max_threshold_error=0.;max_metric_error=0.
    threshold_exact=0;metric_blocks=0;group_partitions=[]
    for fold in range(5):
        name=f'fold_{fold}_frozen.pkl';assert sha(OUT/name)==freeze['files_sha256'][name]
        with (OUT/name).open('rb') as f:obj=ArraysOnlyUnpickler(f).load()
        name=f'fold_{fold}_scores.npz';assert sha(OUT/name)==freeze['files_sha256'][name]
        with np.load(OUT/name,allow_pickle=False) as z:
            projected=z['projected_hidden'].copy();saved={m:z[m].copy() for m in FAMILIES}
        assert projected.shape==(12222,64) and projected.dtype==np.float32
        g={k:set(v) for k,v in obj['groups'].items()};fg,cg,eg=(g[k] for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert not (fg&cg or fg&eg or cg&eg) and fg|cg|eg==allgroups
        group_partitions.append(g)
        cx=np.asarray([j for j,w in enumerate(windows) if w['group_id'] in cg],int)
        wi=np.asarray([j for j in cx if windows[j]['main_eligible']],int)
        ca=[a for a in items if a['group_id'] in cg]
        ai=[a for a in ca if a['main_eligible']]
        yw=np.asarray([windows[j]['gold'] for j in wi],int)
        ya=np.asarray([a['gold'] for a in ai],int)
        safe=[a for a in ai if a['reviewed_safe_refusal']]
        assert all(a['gold']==0 for a in safe)
        cal_indices=set(cx.tolist())
        assert all(set(byitem[a['item_id']]).issubset(cal_indices) for a in ca)
        def amax(values):
            # Compute max over ALL windows for every cal answer before eligibility.
            every={a['item_id']:float(np.max(values[byitem[a['item_id']]])) for a in ca}
            return np.asarray([every[a['item_id']] for a in ai])
        local=np.column_stack((projected,designs['extra'])).astype(np.float32)
        matrices={'local_probe':local,'local_mean_fusion':np.column_stack((designs['base'],local)),
                  'local_slots_fusion':np.column_stack((designs['slots'],local))}
        fold_record=dict(fold=fold,cal_groups=len(cg),cal_all_windows=len(cx),cal_main_windows=len(wi),
            cal_window_positive=int(yw.sum()),cal_all_answers=len(ca),cal_main_answers=len(ai),
            cal_answer_positive=int(ya.sum()),cal_safe_refusal_answers=len(safe),
            cal_nonmain_windows_used_in_answermax=sum(not windows[j]['main_eligible'] for j in cx),
            selected_C={})
        for family in FAMILIES:
            chosen=obj['models'][family];candidates=chosen['all_lr_candidates'];table=chosen['calibration_candidates']
            assert len(candidates)==len(table)==3
            entries=[];selected_values=None
            for j,(c,candidate,t) in enumerate(zip(protocol['C'],candidates,table)):
                assert candidate['C']==t['C']==candidate['model'].C==c
                values=replay(matrices[family],candidate)
                assert values.shape==(12222,) and np.isfinite(values).all()
                aa=amax(values)
                ts={'window':threshold(yw,values[wi]),'answer':threshold(ya,aa)}
                for level in ts:
                    assert ts[level].keys()==t['thresholds'][level].keys()
                    delta=max(abs(float(ts[level][k])-float(t['thresholds'][level][k])) for k in ts[level])
                    max_threshold_error=max(max_threshold_error,delta)
                    assert delta<=1e-12,(fold,family,c,level,delta)
                    threshold_exact+=int(ts[level]==t['thresholds'][level])
                w,a=ts['window'],ts['answer']
                key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],-c]
                assert key==list(t['selection_key'])
                entries.append(key)
                candidate_records.append(dict(fold=fold,family=family,C=c,thresholds=ts,selection_key=key))
                if j==chosen['selected_candidate']:
                    selected_values=values;delta=float(np.max(np.abs(values-saved[family])))
                    max_score_error=max(max_score_error,delta);assert delta<=1e-12
                    assert obj['thresholds'][family]==t['thresholds']
                    for field in ('coef_','intercept_','classes_'):
                        assert np.array_equal(getattr(chosen['model'],field),getattr(candidate['model'],field))
                    for field in ('mean_','var_','scale_'):
                        assert np.array_equal(getattr(chosen['scaler'],field),getattr(candidate['scaler'],field))
            best=max(range(3),key=lambda j:entries[j])
            assert best==chosen['selected_candidate'] and chosen['C']==candidates[best]['C']
            fold_record['selected_C'][family]=chosen['C']
            # Frozen selected scores give exact threshold classification independent of dot rounding.
            aa=amax(saved[family]);thresholds=obj['thresholds'][family]
            expected=summary['folds'][str(fold)][family]['calibration']
            for level,yy,ss in [('windows',yw,saved[family][wi]),('answers',ya,aa)]:
                got=metrics(yy,ss,thresholds['window' if level=='windows' else 'answer']['threshold'],level)
                assert got.keys()==expected[level].keys()
                delta=max(abs(float(got[k])-float(expected[level][k])) for k in got)
                max_metric_error=max(max_metric_error,delta);assert delta<=1e-12
                metric_blocks+=1
        fold_records.append(fold_record);print(json.dumps(fold_record),flush=True)
    assert all(group_partitions[f]['calibration_groups']==group_partitions[(f+1)%5]['evaluation_groups'] for f in range(5))
    assert len(candidate_records)==45
    report=dict(passed=True,blockers=[],scope='R16 actual-train frozen R23b artifacts only; no original validation/test or R17 metadata',
        no_fitting=True,no_GPU=True,no_production_imports=True,sklearn_objects_loaded_as_inert_state=True,
        coefficient_candidates_replayed=45,calibration_thresholds_verified=90,threshold_dictionaries_bit_exact=threshold_exact,
        calibration_selection_keys_exact=45,C_selections_exact=15,selected_saved_score_arrays_compared=15,
        max_selected_score_abs_error=max_score_error,max_calibration_threshold_abs_error=max_threshold_error,
        calibration_metric_blocks=metric_blocks,max_calibration_metric_abs_error=max_metric_error,
        answermax_all_candidate_windows_before_main_answer_filter=True,safe_refusal_nonmain_windows_retained_for_answermax=True,
        total_candidate_windows=12222,total_main_windows=9526,total_answers=602,total_main_answers=598,
        total_main_risk_windows=1063,total_main_risk_answers=151,total_safe_refusal_answers=117,
        folds=fold_records,candidates=candidate_records,bindings=BINDINGS,helper_sha256=sha(Path(__file__)),
        limitations=['Uses frozen projected_hidden; PCA reconstruction, fit weights and scaler derivation belong to parent audit.',
            'OOF answer rows supplied identities and gold only; no OOF scores were used for calibration decisions.',
            'Replayed all 45 coefficient objects; only 15 selected candidates have saved full score arrays for direct comparison.',
            'Current frozen hashes checked; historical external chronology and human-label correctness are not established.',
            'No audit of legacy baselines or persistence smoother here.'])
    target=OUT/'CALIBRATION_AUDIT23B.json';assert not target.exists()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'passed':True,'report':str(target),'sha256':sha(target),'thresholds_exact':threshold_exact,
        'thresholds_verified':90,'C_selections_exact':15,'selected_score_max_error':max_score_error,
        'threshold_max_error':max_threshold_error,'metric_max_error':max_metric_error},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
