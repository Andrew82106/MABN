"""Independent numerical replay of frozen global/local score fusion; no fitting."""
from pathlib import Path
from collections import defaultdict
import json, hashlib, pickle, time
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OLD=ROOT.parent/'round23b_local_evidence_probe/results'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def threshold(y,s):
    y=np.asarray(y,int);s=np.asarray(s,np.float64);assert set(y)=={0,1} and np.isfinite(s).all()
    ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix]
    end=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.r_[0,np.cumsum(yy)[end]];n=np.r_[0,end+1];cuts=np.r_[np.nextafter(ss[0],np.inf),ss[end]]
    f1=2*tp/(n+y.sum());precision=np.divide(tp,n,out=np.zeros(len(n)),where=n>0)
    j=max(range(len(n)),key=lambda i:(f1[i],precision[i],cuts[i]))
    return {'threshold':float(cuts[j]),'validation_f1':float(f1[j]),'validation_precision':float(precision[j]),'tokens':len(y),'scorable_tokens':len(y),'risk_tokens':int(y.sum())}
def metric(rr,m):
    r=[x for x in rr if x['main_eligible']];y=np.array([x['gold'] for x in r],bool);p=np.array([x['predictions'][m] for x in r],bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'alert_rate':float(p.mean()),'risk_rate':float(y.mean())}
def main():
    t=time.perf_counter();cfg=read(ROOT/'protocol.json');begin=read(OUT/'started.json');freeze=read(OUT/'calibration_freeze.json')
    assert begin['code_sha256']==freeze['code_sha256']==sha(ROOT/'src/run26.py')
    assert begin['protocol_sha256']==sha(ROOT/'protocol.json')
    assert begin['source_complete_sha256']==freeze['source_sha256']==sha(OLD/'complete.json')
    assert read(OUT/'test_started.json')['freeze_sha256']==sha(OUT/'calibration_freeze.json')
    for folder,file in [(OLD,'complete.json'),(OUT,'calibration_freeze.json'),(OUT,'complete.json')]:
        for n,h in read(folder/file)['files_sha256'].items():assert sha(folder/n)==h,n
    windows=rows(OLD/'candidate_windows.jsonl');wa=rows(OUT/'window_scores_oof.jsonl');aa=rows(OUT/'answer_scores_oof.jsonl')
    olda=rows(OLD/'answer_scores_oof.jsonl');oldw=rows(OLD/'window_scores_oof.jsonl')
    assert len(windows)==len(wa)==12222 and len(aa)==len(olda)==602
    byitem=defaultdict(list)
    for i,w in enumerate(windows):byitem[w['item_ids'][0]].append(i)
    wm={w['window_key']:w for w in wa};am={a['item_id']:a for a in aa}
    assert len(wm)==len(windows) and len(am)==602
    for r,prior in zip(wa,oldw):
        assert {k:v for k,v in r.items() if k not in ('scores','predictions')}=={k:v for k,v in prior.items() if k not in ('scores','predictions')}
    for r,prior in zip(aa,olda):
        assert {k:v for k,v in r.items() if k not in ('scores','predictions')}=={k:v for k,v in prior.items() if k not in ('scores','predictions')}
    methods=cfg['methods'];base=methods[:14];local=cfg['local'];new=methods[14:]
    assert len(methods)==17 and cfg['alpha']==[0.,.125,.25,.5,1.,2.]
    nth=0;nc=0;oldvals=0;maxdiff=0.;threshold_delta=0.;choices=[];safecheck=0
    for f in range(5):
        tab=read(OUT/f'fold_{f}_calibration.json');prev=pickle.loads((OLD/f'fold_{f}_frozen.pkl').read_bytes())
        assert tab['groups']==prev['groups']
        fit,cal,ev=(set(tab['groups'][k]) for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert not fit&cal and not fit&ev and not cal&ev and len(fit|cal|ev)==278
        with np.load(OLD/f'fold_{f}_scores.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in base}
        with np.load(OUT/f'fold_{f}_scores.npz',allow_pickle=False) as z:saved={k:z[k].copy() for k in methods}
        for m in base:
            assert np.array_equal(saved[m],old[m]) and tab['thresholds'][m]==prev['thresholds'][m];oldvals+=len(windows)
        calw=np.array([i for i,w in enumerate(windows) if w['group_id'] in cal and w['main_eligible']]);yw=[windows[i]['gold'] for i in calw]
        cala=[a for a in olda if a['group_id'] in cal and a['main_eligible']];ya=[a['gold'] for a in cala]
        for loc,m in zip(local,new):
            table=tab['tables'][m];keys=[];values=[]
            assert len(table)==6
            for alpha,entry in zip(cfg['alpha'],table):
                p=np.clip(old[loc],1e-12,1-1e-12);g=np.clip(old['verifier_learned_broadcast'],1e-12,1-1e-12)
                s=old[loc].copy() if alpha==0 else expit(np.log(p)-np.log1p(-p)+alpha*(np.log(g)-np.log1p(-g)))
                # Independent odds-product expression, stable on this bounded clipped domain.
                if alpha:
                    odds=p/(1-p)*(g/(1-g))**alpha
                    maxdiff=max(maxdiff,float(np.max(np.abs(s-odds/(1+odds)))))
                a=np.array([np.max(s[byitem[x['item_id']]]) for x in cala])
                th={'window':threshold(yw,s[calw]),'answer':threshold(ya,a)}
                assert entry['alpha']==alpha
                for unit in th:
                    assert th[unit]==entry['thresholds'][unit],(f,m,alpha,unit,th[unit],entry['thresholds'][unit])
                    nth+=1
                w,an=th['window'],th['answer'];key=[min(w['validation_f1'],an['validation_f1']),w['validation_f1'],w['validation_precision'],-alpha]
                assert key==entry['key'];keys.append(key);values.append(s);nc+=1
                if alpha==0:assert np.array_equal(s,old[loc]) and th==prev['thresholds'][loc]
            j=max(range(6),key=lambda j:keys[j]);assert table[j]==tab['choices'][m] and tab['thresholds'][m]==table[j]['thresholds']
            assert np.array_equal(values[j],saved[m]);choices.append({'fold':f,'method':m,'alpha':table[j]['alpha']})
        # Global verifier is constant within each answer, including refusal windows.
        for iid,ix in byitem.items():assert np.all(old['verifier_learned_broadcast'][ix]==old['verifier_learned_broadcast'][ix[0]])
        for i,w in enumerate(windows):
            if w['group_id'] not in ev:continue
            r=wm[w['window_key']];assert r['fold']==f
            for m in methods:
                assert r['scores'][m]==float(saved[m][i])
                assert r['predictions'][m]==bool(saved[m][i]>=tab['thresholds'][m]['window']['threshold'])
        for a in olda:
            if a['group_id'] not in ev:continue
            r=am[a['item_id']];ix=byitem[a['item_id']];assert r['fold']==f
            for m in methods:
                s=float(np.max(saved[m][ix])) if ix else None
                assert r['scores'][m]==s
                assert r['predictions'][m]==(bool(s>=tab['thresholds'][m]['answer']['threshold']) if s is not None else None)
                if a['reviewed_safe_refusal']:
                    assert ix and a['gold']==0 and a['main_eligible'] and all(not windows[i]['main_eligible'] for i in ix);safecheck+=1
    summary=read(OUT/'summary.json');previous=read(OLD/'summary.json')
    for m in base:assert summary['methods'][m]==previous['methods'][m]
    for m in methods:
        for unit,rr in [('windows',wa),('answers',aa)]:
            for k,v in metric(rr,m).items():assert v==summary['methods'][m][unit][k],(m,unit,k)
    assert nth==180 and nc==90 and oldvals==855540 and safecheck==1989
    files=[Path(__file__),ROOT/'src/run26.py',ROOT/'protocol.json',OUT/'started.json',OUT/'calibration_freeze.json',OUT/'test_started.json',OUT/'complete.json',OUT/'summary.json',OLD/'complete.json']
    result={'status':'passed_with_provenance_limitations','numerical_blockers':[],'new_training_or_GPU':False,'original_heldout_or_QA_test_read_by_auditor':False,
      'cohort':{'answers':602,'questions':301,'event_groups':278,'candidate_windows':12222,'main_windows':9526,'risk_windows':1063,'main_answers':598,'risk_answers':151,'safe_refusal_answers':117},
      'all_folds_old_group_membership_exact_disjoint':True,'calibration_candidates_exact':nc,'calibration_threshold_dictionaries_exact':nth,'selected_alphas_exact':15,'selected_frozen_scores_exact':True,
      'independent_odds_product_formula_max_abs':maxdiff,'identity_candidates_and_old_thresholds_exact':15,'old14_full_fold_scores_exact':oldvals,
      'old14_entire_method_summary_dicts_exact':True,'all17_OOF_scores_predictions_main_metrics_exact':True,'safe_refusal_all_candidate_max_checks':safecheck,
      'choices':choices,'new_metrics':{m:{'window_f1':summary['methods'][m]['windows']['f1'],'answer_f1':summary['methods'][m]['answers']['f1']} for m in new},
      'freeze_provenance':{'all5fold10files_bound_by_freeze':True,'test_started_binds_exact_freeze_sha':True,'entrypoint_requires_all_fold_freeze_before_scoring':True,'UTC_timestamps_recorded':False,'file_mtime_order_observed':(OUT/'calibration_freeze.json').stat().st_mtime<=(OUT/'test_started.json').stat().st_mtime<=(OUT/'complete.json').stat().st_mtime},
      'limits':['R26 prepare calls inherited train_metadata, which parses the mixed-split input JSONL before filtering train. Only actual-train generations, labels and feature rows enter scoring; cannot claim original heldout input text was never parsed.',
       'No UTC execution timestamps in R26 freeze/test records; hash chain, entrypoint control flow and observed filesystem times support ordering, not independent historical proof.',
       'Repeatedly inspected assistant-labeled R16 development folds; no fresh final test or human-QA evidence.',
       'Only main confusion metrics recomputed independently; new bootstrap and auxiliary ranking/highlight measures were not independently rerun.'],
      'seconds':time.perf_counter()-t,'files_sha256':{str(p.resolve()):sha(p) for p in files}}
    (OUT/'INDEPENDENT_AUDIT26.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','calibration_candidates_exact','calibration_threshold_dictionaries_exact','old14_full_fold_scores_exact','safe_refusal_all_candidate_max_checks','independent_odds_product_formula_max_abs','new_metrics','seconds')}))
if __name__=='__main__':main()
