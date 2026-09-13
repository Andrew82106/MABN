"""Independent signed-score calibration audit. CPU NumPy/stdlib only."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
OLD=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned','slots_base_smooth')
FIXED=('lumina_single_mean_lambda05','lumina_joint_body_lambda05')
SECONDARY='lumina_joint_body_lambda_cal'
METHODS=OLD+FIXED+(SECONDARY,)
SCORE_KEYS={.25:'joint_lambda025',.5:'joint_lambda05',.75:'joint_lambda075'}
BINDINGS={}

def sha(p):
    p=Path(p).resolve();assert p.is_relative_to(ROOT)
    h=hashlib.sha256(p.read_bytes()).hexdigest();BINDINGS[str(p.relative_to(ROOT))]=h
    return h

def read(p):
    sha(p);return json.loads(Path(p).read_text('utf-8'))

def readl(p):
    sha(p);return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]

def choose(y,s):
    assert set(y)=={0,1} and len(y)==len(s) and np.isfinite(s).all()
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    p=np.bincount(inv,weights=y,minlength=len(u)).astype(np.int64)
    positive=np.r_[p.sum(),np.cumsum(p[::-1])[::-1],0]
    counts=np.r_[len(y),np.cumsum(n[::-1])[::-1],0]
    ts=np.r_[np.nextafter(u[0],-np.inf),u,np.nextafter(u[-1],np.inf)]
    f1=2*positive/(counts+y.sum())
    precision=np.divide(positive,counts,out=np.zeros(len(counts)),where=counts!=0)
    j=max(range(len(ts)),key=lambda k:(f1[k],precision[k],ts[k]))
    return dict(threshold=float(ts[j]),validation_f1=float(f1[j]),validation_precision=float(precision[j]),
                tokens=len(y),scorable_tokens=len(y),risk_tokens=int(y.sum()))

def metric(y,s,t,unit):
    pred=s>=t;tp=int(y[pred].sum());fp=int(pred.sum())-tp
    fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True)
    pos=np.bincount(inv,weights=y,minlength=len(u));neg=n-pos
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(y.sum()*(len(y)-y.sum()))
    ap=np.sum(pos[::-1]/y.sum()*np.cumsum(pos[::-1])/np.cumsum(n[::-1]))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),auroc=float(auc),average_precision=float(ap),
        alert_rate=float(pred.mean()),risk_rate=float(y.mean()),**{unit:len(y),'risk_'+unit:int(y.sum())})

def main():
    complete=read(OUT/'complete25.json');freeze=read(OUT/'calibration_freeze25.json')
    assert not complete['original_validation_test_used'] and complete['new_models_fitted']==0
    assert freeze['all_calibration_choices_frozen_before_outer'] and freeze['new_models_fitted']==0
    for name in ('calibration_freeze25.json','summary25.json'):
        assert sha(OUT/name)==complete['files_sha256'][name]
    summary=read(OUT/'summary25.json');protocol=read(ROOT/'scoring_protocol.json')
    assert protocol['secondary_lambda_grid']==[.25,.5,.75] and protocol['fixed_lambda']==.5
    assert tuple(protocol['methods'])==METHODS and protocol['score_transform'].startswith('None')
    assert not protocol['original_validation_test_used']
    def frozen_read(name):
        assert sha(OUT/name)==freeze['files_sha256'][name]
        return read(OUT/name)
    snapshot=frozen_read('source_snapshot25.json')
    for p in (ROOT/'src/run25.py',ROOT/'scoring_protocol.json'):
        assert sha(p)==snapshot['files_sha256'][str(p.resolve())]
    for name in ('candidate_windows25.jsonl','answer_index25.jsonl','formula_window_scores25.npz'):
        assert sha(OUT/name)==freeze['files_sha256'][name]
    windows=readl(OUT/'candidate_windows25.jsonl');items=readl(OUT/'answer_index25.jsonl')
    assert len(windows)==12222 and len(items)==602
    assert len({w['window_key'] for w in windows})==12222 and len({a['item_id'] for a in items})==602
    assert all(a['split']=='train' for a in items) and all(w['split']=='train' for w in windows)
    itemmap={a['item_id']:a for a in items};byitem=defaultdict(list)
    for j,w in enumerate(windows):
        assert len(w['item_ids'])==1
        iid=w['item_ids'][0];a=itemmap[iid]
        assert all(w[k]==a[k] for k in ('row_id','group_id','condition'))
        byitem[iid].append(j)
    assert set(byitem)==set(itemmap)
    assert sum(w['main_eligible'] for w in windows)==9526
    assert sum(a['main_eligible'] for a in items)==598
    assert sum(w['gold']==1 for w in windows)==1063 and sum(a['gold']==1 for a in items)==151
    with np.load(OUT/'formula_window_scores25.npz',allow_pickle=False) as z:formula={k:z[k].copy() for k in z.files}
    assert set(formula)=={'single_lambda05',*SCORE_KEYS.values()}
    for v in formula.values():assert v.shape==(12222,) and v.dtype==np.float64 and np.isfinite(v).all()
    ranges={k:dict(min=float(v.min()),max=float(v.max()),negative=int(np.sum(v<0)),above_one=int(np.sum(v>1))) for k,v in formula.items()}
    # Test the independent threshold helper on signed and tied data as well.
    assert choose(np.array([0,1,0]),np.array([-4.,-2.,-3.]))['threshold']==-2.
    folds=[];candidate_rows=[];thresholds_count={'lambda_candidates':0,'fixed_methods':0,'old_baselines':0}
    metric_blocks=0;max_metric_error=0.;groups=[]
    allgroups={a['group_id'] for a in items};assert len(allgroups)==278
    for fold in range(5):
        frozen=frozen_read(f'fold_{fold}_calibration25.json')
        path=OUT/f'fold_{fold}_scores25.npz';assert sha(path)==freeze['files_sha256'][path.name]
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        assert set(values)==set(METHODS)
        assert all(v.shape==(12222,) and np.isfinite(v).all() for v in values.values())
        assert frozen['fixed_lambda']==.5 and frozen['score_transform']=='none'
        assert frozen['new_models_fitted']==0 and frozen['all_thresholds_use_calibration_only']
        g={k:set(v) for k,v in frozen['groups'].items()};fg,cg,eg=(g[k] for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert not (fg&cg or fg&eg or cg&eg) and fg|cg|eg==allgroups;groups.append(g)
        cx=[j for j,w in enumerate(windows) if w['group_id'] in cg]
        wi=np.array([j for j in cx if windows[j]['main_eligible']],int)
        ca=[a for a in items if a['group_id'] in cg];ai=[a for a in ca if a['main_eligible']]
        yw=np.array([windows[j]['gold'] for j in wi],int);ya=np.array([a['gold'] for a in ai],int)
        safe=[a for a in ai if a['reviewed_safe_refusal']];assert all(a['gold']==0 for a in safe)
        assert all(set(byitem[a['item_id']]).issubset(set(cx)) for a in ca)
        def answermax(v):
            # No zero initial value or window eligibility filter for signed scores.
            every={a['item_id']:float(np.max(v[byitem[a['item_id']]])) for a in ca}
            return np.array([every[a['item_id']] for a in ai])
        def thresholds(v):return {'window':choose(yw,v[wi]),'answer':choose(ya,answermax(v))}
        assert np.array_equal(values[FIXED[0]],formula['single_lambda05'])
        assert np.array_equal(values[FIXED[1]],formula['joint_lambda05'])
        for name in OLD+FIXED:
            assert thresholds(values[name])==frozen['thresholds'][name],(fold,name)
            thresholds_count['old_baselines' if name in OLD else 'fixed_methods']+=2
        table=frozen['secondary_candidates'];assert len(table)==3
        keys=[]
        for lam,t in zip(protocol['secondary_lambda_grid'],table):
            assert t['lambda']==lam
            ts=thresholds(formula[SCORE_KEYS[lam]]);assert ts==t['thresholds'],(fold,lam)
            w,a=ts['window'],ts['answer']
            key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],int(lam==.5),-lam]
            assert key==t['selection_key'];keys.append(key);thresholds_count['lambda_candidates']+=2
            candidate_rows.append(dict(fold=fold,**t))
        best=max(range(3),key=lambda j:keys[j]);lam=table[best]['lambda']
        assert best==frozen['secondary_selected_index'] and lam==frozen['secondary_selected_lambda']
        assert frozen['thresholds'][SECONDARY]==table[best]['thresholds']
        assert np.array_equal(values[SECONDARY],formula[SCORE_KEYS[lam]])
        frow=dict(fold=fold,selected_lambda=lam,selected_index=best,cal_groups=len(cg),
            cal_all_windows=len(cx),cal_main_windows=len(wi),cal_window_positive=int(yw.sum()),
            cal_all_answers=len(ca),cal_main_answers=len(ai),cal_answer_positive=int(ya.sum()),
            cal_safe_refusal_answers=len(safe),cal_nonmain_windows_used_in_answermax=sum(not windows[j]['main_eligible'] for j in cx),
            selected_window_f1=table[best]['thresholds']['window']['validation_f1'],
            selected_answer_f1=table[best]['thresholds']['answer']['validation_f1'])
        for name in METHODS:
            aa=answermax(values[name]);expected=summary['folds'][str(fold)][name]['calibration']
            ts=frozen['thresholds'][name]
            for level,yy,ss in [('windows',yw,values[name][wi]),('answers',ya,aa)]:
                got=metric(yy,ss,ts['window' if level=='windows' else 'answer']['threshold'],level)
                assert got.keys()==expected[level].keys()
                delta=max(abs(float(got[k])-float(expected[level][k])) for k in got)
                max_metric_error=max(max_metric_error,delta);assert delta<=1e-12
                metric_blocks+=1
        folds.append(frow);print(json.dumps(frow),flush=True)
    assert all(groups[f]['calibration_groups']==groups[(f+1)%5]['evaluation_groups'] for f in range(5))
    assert thresholds_count==dict(lambda_candidates=30,fixed_methods=20,old_baselines=70)
    report=dict(passed=True,blockers=[],scope='R25 actual-train frozen artifacts only; no original heldout or QA data',
        no_fit=True,no_GPU=True,no_production_imports=True,thresholds_exact=thresholds_count,
        total_threshold_dictionaries_exact=sum(thresholds_count.values()),lambda_candidates=15,
        lambda_selection_keys_exact=15,selected_lambda_exact=5,selected_lambdas=[x['selected_lambda'] for x in folds],
        signed_scores_unchanged=True,fixed_and_selected_formula_arrays_exact=True,score_ranges=ranges,
        all_candidate_window_answermax_before_main_answer_filter=True,safe_refusal_windows_retained_for_answermax=True,
        calibration_metric_blocks=metric_blocks,max_calibration_metric_abs_error=max_metric_error,
        folds=folds,candidates=candidate_rows,bindings=BINDINGS,helper_sha256=sha(Path(__file__)),
        primary_remains_joint_fixed_lambda05=True,
        selection_rule='Max min(windowF1,answerF1), then windowF1, window precision, prefer lambda .5, then lower lambda',
        limitations=['Per-token formulas and original-window aggregation are delegated to parent/root audit.',
            'Legacy coefficient and smoother replay, OOF metric counts, and complete freeze-before-outer chain are delegated to parent.',
            'This audit verifies calibration only; selected lambda cannot replace the predefined fixed .5 primary.',
            'Development folds have been repeatedly inspected and are not a fresh final test.'])
    target=OUT/'LAMBDA_CALIBRATION_AUDIT25.json';assert not target.exists()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'passed':True,'report':str(target),'sha256':sha(target),
        'selected_lambdas':report['selected_lambdas'],'thresholds_exact':thresholds_count,
        'cal_metric_blocks':metric_blocks,'max_metric_error':max_metric_error},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
