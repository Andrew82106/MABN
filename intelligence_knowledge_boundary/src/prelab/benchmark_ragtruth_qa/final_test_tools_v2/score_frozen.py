"""Frozen prediction scoring only: no fit, threshold search or model selection."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,math
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import contracts as c

def confusion(y,s,threshold):
    y=np.asarray(y,np.int64);s=np.asarray(s,np.float64)
    assert y.shape==s.shape and len(y)>0 and np.isfinite(s).all() and set(y)<={0,1}
    p=s>=threshold;tp=int(y[p].sum());fp=int(p.sum())-tp;fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        auroc=float(roc_auc_score(y,s)) if len(set(y))==2 else None,average_precision=float(average_precision_score(y,s)) if y.sum() else None)

def group_confusion(y,s,threshold,rowgroups,groups):
    gi={g:i for i,g in enumerate(groups)};table=np.zeros((len(groups),4),np.int64)
    for yy,ss,g in zip(y,s,rowgroups):
        pred=ss>=threshold;column=0 if yy and pred else 1 if not yy and pred else 2 if yy else 3
        table[gi[g],column]+=1
    return table

def bootstrap(table,draws,level):
    counts=np.einsum('rg,gc->rc',draws,table);tp,fp,fn,tn=counts.T
    values={'precision':np.divide(tp,tp+fp,out=np.zeros(len(tp)),where=tp+fp>0),
        'recall':np.divide(tp,tp+fn,out=np.zeros(len(tp)),where=tp+fn>0),
        'f1':np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp)),where=2*tp+fp+fn>0)}
    q=(1-level)/2
    return {k:{'lower':float(np.quantile(v,q,method='linear')),'upper':float(np.quantile(v,1-q,method='linear'))} for k,v in values.items()}

def score_records(answers,windows,methods,predictions,bootstrap_cfg):
    assert answers and windows
    byanswer={a['response_id']:a for a in answers};bywindow={w['window_id']:w for w in windows}
    assert len(byanswer)==len(answers) and len(bywindow)==len(windows)
    aw=defaultdict(list)
    for j,w in enumerate(windows):
        rid=w['response_id'];assert rid in byanswer and w['group_id']==byanswer[rid]['group_id']
        assert w['eligible'] and w['k']==4 and w['stride']==1 and w['lexical_token_indices']
        aw[rid].append(j)
    for a in answers:
        assert a['eligible'] and a['quality']=='good'
        assert len(aw[a['response_id']])==a['eligible_window_count']>0,'Answer lacks eligible windows: stop complete main scoring, never drop or fill0'
    groups=sorted({a['group_id'] for a in answers});B=bootstrap_cfg['replicates'];rng=np.random.default_rng(bootstrap_cfg['seed'])
    draws=np.stack([np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups)) for _ in range(B)])
    assert np.all(draws.sum(1)==len(groups))
    ys=np.asarray([w['label'] for w in windows]);ya=np.asarray([a['label'] for a in answers])
    results=[];output_windows={};output_answers={}
    assert set(predictions)=={m['method_id'] for m in methods}
    for m in methods:
        mid=m['method_id'];rows=predictions[mid];assert len(rows)==len(windows)
        keyed={p['window_id']:p for p in rows};assert len(keyed)==len(rows) and set(keyed)==set(bywindow)
        scores=[]
        for w in windows:
            p=keyed[w['window_id']]
            for k in ('response_id','group_id'):assert p[k]==w[k]
            assert isinstance(p['score'],(int,float)) and not isinstance(p['score'],bool) and math.isfinite(p['score'])
            scores.append(p['score'])
        s=np.asarray(scores,np.float64);a_scores=np.asarray([s[aw[a['response_id']]].max() for a in answers])
        wt,at=m['window_threshold'],m['answer_threshold']
        metrics={'windows':confusion(ys,s,wt),'answers':confusion(ya,a_scores,at)}
        ci={unit:bootstrap(group_confusion(y,p,t,rg,groups),draws,bootstrap_cfg['confidence_level']) for unit,y,p,t,rg in [
            ('windows',ys,s,wt,[w['group_id'] for w in windows]),('answers',ya,a_scores,at,[a['group_id'] for a in answers])]}
        output_windows[mid]=[{k:w[k] for k in ('window_id','response_id','group_id','token_start','token_end','character_intervals')}|
            {'score':float(score),'threshold':wt,'prediction':int(score>=wt),'label':w['label']} for w,score in zip(windows,s)]
        output_answers[mid]=[{'answer_id':a['answer_id'],'response_id':a['response_id'],'group_id':a['group_id'],'score':float(score),'threshold':at,'prediction':int(score>=at),'label':a['label']} for a,score in zip(answers,a_scores)]
        results.append({'method_id':mid,'primary':m['primary'],'thresholds':{'window':wt,'answer':at},'metrics':metrics,'group_bootstrap_CI':ci})
    return {'methods_in_frozen_order':results,'answers':len(answers),'windows':len(windows),'groups':len(groups),'bootstrap':bootstrap_cfg,
        'all_methods_complete':True,'model_selection_on_scored_data':False,'threshold_selection_on_scored_data':False,'answer_rule':'max_all_eligible_windows'},output_windows,output_answers

def run(freeze_path,bundle_path,prediction_manifest_path,out,simulation=False):
    fp,bp,pp,out=map(Path,(freeze_path,bundle_path,prediction_manifest_path,out))
    if simulation:
        assert out.resolve().is_relative_to(c.HERE.resolve()) and bp.resolve().is_relative_to(c.HERE.resolve())
        frozen=c.validate_methods(c.read(fp));assert frozen['purpose']=='calibration_simulation'
    else:
        frozen,_,binding=c.require_release(fp)
        assert bp.resolve()==(c.ROOT/'data/final_test/bundle_manifest.json').resolve()
        assert out.resolve()==(c.ROOT/'results/final_test').resolve()
    bm=c.check_bundle(bp)
    assert bm['mode']==('calibration_simulation' if simulation else 'authorized_official_test')
    if not simulation:assert bm['binding']==binding
    pm=c.read(pp);assert pm['complete'] and pm['mode']==bm['mode']
    assert pm['development_freeze_sha256']==c.sha(fp) and pm['bundle_manifest_sha256']==c.sha(bp)
    assert len(pm['methods'])==len(frozen['methods']) and {m['method_id'] for m in pm['methods']}=={m['method_id'] for m in frozen['methods']}
    predictions={};source_prediction_hashes={}
    for record in pm['methods']:
        m=next(m for m in frozen['methods'] if m['method_id']==record['method_id'])
        assert not record['fit_performed'] and not record['threshold_selection_performed'] and not record['model_selection_performed']
        assert record['source_artifacts_sha256']==m['artifacts_sha256'];c.check_hashes(record['source_artifacts_sha256'])
        c.check_hashes(record['native_prediction_artifacts_sha256'])
        path=Path(record['window_scores_file']);assert c.sha(path)==record['window_scores_sha256']
        predictions[m['method_id']]=c.lines(path);source_prediction_hashes[str(path.resolve())]=c.sha(path)
    partition=bm['partition'];answers=c.lines(bp.parent/f'answers_{partition}.jsonl');windows=c.lines(bp.parent/f'windows_k4_{partition}.jsonl')
    assert len(answers)==bm['counts']['quality_good_answers'] and len(windows)==bm['counts']['eligible_windows']
    result,wp,ap=score_records(answers,windows,frozen['methods'],predictions,frozen['group_bootstrap'])
    assert not out.exists(),'No overwriting or replacing prior final results'
    out.mkdir(parents=True);files={}
    for i,m in enumerate(frozen['methods']):
        mid=m['method_id']
        for stem,value in [('windows',wp[mid]),('answers',ap[mid])]:
            name=f'method_{i:03d}_{stem}.jsonl';c.save_lines(out/name,value);files[name]=c.sha(out/name)
    result.update({'counts_after_official_quality_filter':bm['counts'],'mode':bm['mode'],'official_test_read':not simulation,
        'development_freeze_sha256':c.sha(fp),'bundle_manifest_sha256':c.sha(bp),'prediction_manifest_sha256':c.sha(pp),
        'source_prediction_files_sha256':source_prediction_hashes,'scoring_code_sha256':c.sha(__file__),
        'limitations':['Model inference is a separate frozen adapter; this layer verifies artifact and geometry bindings, not whether an unreviewed adapter actually used its declared weights.',
            'Associated-source groups are not an exhaustive global entity/event separation proof.','Unlabeled quality-good refusals remain eligible negatives; R16 used a different refusal localization policy.']})
    c.save(out/'summary.json',result);files['summary.json']=c.sha(out/'summary.json')
    c.save(out/'complete.json',{'complete':True,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'outputs_sha256':files,'all_methods_complete':True,'fit_or_selection_performed':False,'official_test_read':not simulation})
    print('FROZEN_SCORING_COMPLETE',{'methods':len(frozen['methods']),'answers':len(answers),'windows':len(windows),'simulation':simulation},flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--bundle',required=True);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--simulation',action='store_true');a=p.parse_args()
    run(a.freeze,a.bundle,a.predictions,a.out,a.simulation)
