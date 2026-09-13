import json,pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from common4 import ROOT,readl,writel,save,sha
from fit import aggregate,metrics,answer_scores,included,calibrate,unit_probs

def local_scores(b,pr,i,scope):
    s=np.asarray(b['base']['base_scores'][i]).copy();off=b['base']['offsets'][i]
    for j in b['by_id'][b['base']['rows'][i]['id']]:
        q=b['queries'][j]
        if q['kind']=='whole' or not included(q,scope):continue
        lo,hi=q['target_span'];s[(off[:,0]<hi)&(off[:,1]>lo)]=pr[j]
    return s

def main():
    dest=ROOT/'results';selection=json.loads((dest/'selection.json').read_text());selection_hash=sha(dest/'selection.json')
    b=pickle.loads((ROOT/'data/unit_inputs.pkl').read_bytes());nets=pickle.loads((dest/'checkpoints/unit_models.pkl').read_bytes());base=b['base'];test=base['split_indices']['test'];va=base['split_indices']['val'];yt=np.array([base['rows'][i]['label'] for i in test]);predictions={};localsaved={};ms=[]
    winners={**selection['family_winners'],'selected':selection['selected']}
    for name,c in winners.items():
        if c['model_id'] is None:
            s=np.array([aggregate(base[c['scope']][i],c['aggregate']) for i in test]);ss=[base[c['scope']][i] for i in test];vs=[base[c['scope']][i] for i in va]
        else:
            pr=unit_probs(b,nets,c['model_id'])
            s,_=answer_scores(b,pr,'test',c['scope'],c['aggregate']);ss=[local_scores(b,pr,i,c['scope']) for i in test];vs=[local_scores(b,pr,i,c['scope']) for i in va]
        pred=s>=c['threshold'];m=metrics(yt,pred);m.update(name=name,config=c,answer_auc=float(roc_auc_score(yt,s)))
        ly=np.concatenate([base['truth'][i] for i in test]);lp=np.concatenate(ss);vly=np.concatenate([base['truth'][i] for i in va]);vlp=np.concatenate(vs)
        # Dense token thresholds use a compact validation-only grid, not test labels.
        thresholds=np.unique(np.quantile(vlp,np.linspace(0,1,201)));tm=max(((metrics(vly,vlp>=t),t) for t in thresholds),key=lambda x:(x[0]['f1'],x[0]['precision'],x[1]))
        m.update(local_token_auc=float(roc_auc_score(ly,lp)),within_answer_auc=float(np.mean([roc_auc_score(base['truth'][i],p) for i,p in zip(test,ss) if len(np.unique(base['truth'][i]))==2])),
            token_metrics=metrics(ly,lp>=tm[1]),token_threshold=float(tm[1]))
        predictions[name]={'scores':s,'pred':pred};localsaved[name]=ss;ms.append(m);print(name,{k:round(m[k],4) if isinstance(m[k],float) else m[k] for k in ['f1','precision','recall','false_alarm','tp','fp','fn','tn']},flush=True)
    allpos=metrics(yt,np.ones(len(yt),bool));allpos['name']='always_error';ms.append(allpos)
    main_pred=predictions['selected']['pred'];rng=np.random.default_rng(20260912);boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(yt),len(yt));boot.append(metrics(yt[ix],main_pred[ix])['f1'])
    save(dest/'metrics.json',ms);save(dest/'confidence.json',{'selected_F1_95pct':np.quantile(boot,[.025,.975]).tolist(),'bootstrap':2000,'unit':'source group, each test row has a unique group','fixed_model_and_threshold':True})
    writel(dest/'test_predictions.jsonl',[{'id':base['rows'][i]['id'],'group':base['rows'][i]['group'],'label':int(yt[j]),'offsets':base['offsets'][i].tolist(),
        'methods':{n:{'score':float(v['scores'][j]),'pred':bool(v['pred'][j]),'token_risks':localsaved[n][j].tolist()} for n,v in predictions.items()}} for j,i in enumerate(test)])
    ann={a['id']:a for a in readl(ROOT/'data/annotations.jsonl')};policies={a['id']:a for a in readl(ROOT/'data/policies.jsonl')};coverage={};total_spans=sum(len(ann[base['rows'][i]['id']]['spans']) for i in test)
    for kind in ['single_sentence','sentences3','facts3']:
        covered=0;answers=0;calls=0
        for i in test:
            r=base['rows'][i];queries=[b['queries'][j] for j in b['by_id'][r['id']] if included(b['queries'][j],kind)];calls+=len(queries)
            hits=[any(q['target_span'][0]<s['end'] and q['target_span'][1]>s['start'] for q in queries) for s in ann[r['id']]['spans']];covered+=sum(hits);answers+=any(hits)
        coverage[kind]={'error_answers_covered':answers,'error_answers':int(yt.sum()),'spans_covered':covered,'all_error_spans':total_spans,'mean_calls':calls/len(test)}
    save(dest/'coverage.json',coverage)
    assert selection_hash==sha(dest/'selection.json')
    save(dest/'evaluation_manifest.json',{'completed':True,'n_test':len(test),'primary':'selected answer-level error-class F1','selection_sha256':selection_hash,
        'thresholds_tuned_on_test':False,'selected_primary_target_met':next(m for m in ms if m['name']=='selected')['f1']>.7})
    print('EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
