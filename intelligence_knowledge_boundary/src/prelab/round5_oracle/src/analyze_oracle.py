import json,pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from common5 import R5,R4,readl,writel,save,sha
from run_confirmation import features
from fit import metrics,calibrate,aggregate

MID='abc_base_hidden16_balanced_0.01'
METHODS=['original_attention','direct_B','frozen_internal_probe']
MODES=['sentence','span']

def load_inputs():
    qs=readl(R5/'data/queries.jsonl');b=pickle.loads((R4/'data/unit_inputs.pkl').read_bytes());nets=pickle.loads((R4/'results/checkpoints/unit_models.pkl').read_bytes());x=features(qs,R5/'data/readouts',b['pca'])
    return qs,{q['query_id']:j for j,q in enumerate(qs)},x,nets

def score(qs,x,nets,method,indices):
    if method=='original_attention':return np.array([qs[j]['base_features'][2] for j in indices])
    if method=='direct_B':return x['abc'][indices,1]
    cp=nets[MID];return cp['model'].predict_proba(x[cp['feature']][indices])[:,1]

def paired_ranking(cases,s):
    pairs={}
    for c,v in zip(cases,s):pairs.setdefault(c['pair_id'],{})[c['label']]=v
    return float(np.mean([float(p[1]>p[0])+.5*float(p[1]==p[0]) for p in pairs.values()]))

def main():
    protocol=json.loads((R5/'protocol.json').read_text());ph=sha(R5/'protocol.json')
    for p,h in protocol['frozen_dependencies'].items():assert sha(R4/p)==h
    qs,index,x,nets=load_inputs();cases=readl(R5/'data/cases.jsonl');val=[c for c in cases if c['split']=='val'];test=[c for c in cases if c['split']=='diagnostic'];yv=np.array([c['label'] for c in val]);yt=np.array([c['label'] for c in test]);thresholds=[]
    # Only validation units are scored until the decision thresholds have been written.
    for mode in MODES:
        for method in METHODS:
            ix=[index[c[mode+'_query']] for c in val];s=score(qs,x,nets,method,ix);t,m=calibrate(yv,s)
            thresholds.append({'mode':mode,'method':method,'threshold':t,'validation':m})
    save(R5/'results/unit_thresholds.json',{'thresholds':thresholds,'validation_cases':[c['case_id'] for c in val],'new_probe_fitting':False,'diagnostic_scores_seen':False,'protocol_sha256':ph})
    th=sha(R5/'results/unit_thresholds.json');unit_metrics=[];unit_preds={};length_controls=[];verdicts={}
    for c in thresholds:
        mode,method=c['mode'],c['method'];ix=[index[a[mode+'_query']] for a in test];s=score(qs,x,nets,method,ix);p=s>=c['threshold'];m=metrics(yt,p)
        m.update(mode=mode,method=method,threshold=c['threshold'],auc=float(roc_auc_score(yt,s)),paired_ranking=paired_ranking(test,s));rng=np.random.default_rng(20260916);boot=[]
        pairs=list(dict.fromkeys(a['pair_id'] for a in test));groups=[[i for i,a in enumerate(test) if a['pair_id']==pair] for pair in pairs]
        for _ in range(2000):
            take=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);boot.append(roc_auc_score(yt[take],s[take]))
        m['auc_95pct']=np.quantile(boot,[.025,.975]).tolist();unit_metrics.append(m);unit_preds[mode+'_'+method]={'score':s,'pred':p}
        print('UNIT',mode,method,'AUC',round(m['auc'],3),'F1',round(m['f1'],3),'P/R',round(m['precision'],3),round(m['recall'],3),flush=True)
    for mode in MODES:
        ix=[index[c[mode+'_query']] for c in test];p=x['abc'][ix,:3];verdict=np.argmax(p,axis=1)
        verdicts[mode]={str(label):{k:int(((yt==label)&(verdict==j)).sum()) for j,k in enumerate(['supported','contradicted','insufficient'])} for label in [0,1]}
        lengths=np.array([len(qs[j]['target_text']) for j in ix]);length_controls.append({'mode':mode,'length_only_auc':float(roc_auc_score(yt,lengths)),'mean_error_length':float(lengths[yt==1].mean()),'mean_clean_length':float(lengths[yt==0].mean())})
    save(R5/'results/unit_metrics.json',unit_metrics);save(R5/'results/length_controls.json',length_controls);save(R5/'results/direct_verdict_counts.json',verdicts)
    writel(R5/'results/unit_predictions.jsonl',[{**c,'methods':{n:{'score':float(v['score'][i]),'pred':bool(v['pred'][i])} for n,v in unit_preds.items()}} for i,c in enumerate(test)])
    # Paired end-to-end diagnostic on exactly the same50previously-tested summaries.
    policies=[p for p in readl(R5/'data/policies.jsonl') if p['split']=='diagnostic'];yy=np.array([p['label'] for p in policies]);cfg=json.loads((R4/'confirmation/protocol.json').read_text())['frozen_methods']['sentences_facts'];pp=score(qs,x,nets,'frozen_internal_probe',list(range(len(qs))));pm=[];preds={}
    original={p['id']:p for p in readl(R4/'confirmation/results/predictions.jsonl')}
    for mode in ['automatic','oracle_sentence','oracle_span']:
        s=np.array([aggregate([pp[index[q]] for q in p[mode]],cfg['aggregate']) for p in policies]);pred=s>=cfg['threshold'];m=metrics(yy,pred);m.update(mode=mode,answer_auc=float(roc_auc_score(yy,s)),threshold=cfg['threshold'],mean_calls=float(np.mean([len(p[mode]) for p in policies])));pm.append(m);preds[mode]={'score':s,'pred':pred}
        if mode=='automatic':
            assert np.allclose(s,[original[p['id']]['methods']['sentences_facts']['score'] for p in policies],rtol=1e-7,atol=1e-7)
            assert np.array_equal(pred,[original[p['id']]['methods']['sentences_facts']['pred'] for p in policies])
        else:assert np.array_equal(pred[yy==0],preds['automatic']['pred'][yy==0]) and np.allclose(s[yy==0],preds['automatic']['score'][yy==0],atol=1e-12)
        print('PIPELINE',mode,{k:m[k] for k in ['f1','precision','recall','tp','fp','fn','tn']},flush=True)
    save(R5/'results/pipeline_metrics.json',pm);writel(R5/'results/pipeline_predictions.jsonl',[{'id':p['id'],'label':p['label'],'methods':{n:{'score':float(v['score'][i]),'pred':bool(v['pred'][i])} for n,v in preds.items()}} for i,p in enumerate(policies)])
    changes={}
    for mode in ['oracle_sentence','oracle_span']:
        old=preds['automatic']['pred'];new=preds[mode]['pred'];changes[mode]={'recovered_error_ids':[p['id'] for i,p in enumerate(policies) if yy[i] and not old[i] and new[i]],'lost_error_ids':[p['id'] for i,p in enumerate(policies) if yy[i] and old[i] and not new[i]]}
    save(R5/'results/changes.json',changes)
    assert ph==sha(R5/'protocol.json') and th==sha(R5/'results/unit_thresholds.json')
    save(R5/'results/analysis_manifest.json',{'completed':True,'diagnostic_only':True,'frozen_protocol_sha256':ph,'unit_thresholds_sha256':th,'new_probe_fitting':False,'automatic_predictions_reproduced':True,'n_paired_units':len(test),'n_pipeline_answers':len(policies),'all_positive_unit_F1':2/3,'all_positive_answer_F1':4/7})

if __name__=='__main__':main()
