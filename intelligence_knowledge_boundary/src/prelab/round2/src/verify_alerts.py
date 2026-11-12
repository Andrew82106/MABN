"""Independent rank-sum/global and fixed-threshold alarm checks from saved scores."""
import json
import numpy as np
from scipy.stats import rankdata
from common import ROOT,readl

def rank_auc(y,p):
    y=np.asarray(y,dtype=bool); n=int(y.sum()); z=len(y)-n
    if not n or not z: return None
    return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*z))

def main():
    checks={}
    for name in ['news_small','news_large','rag_large','confirmation']:
        run=ROOT/('data/news_confirmation' if name=='confirmation' else 'data/'+name)
        dest=ROOT/'results'/name
        anns={a['id']:a for a in readl(run/('annotations.jsonl' if name=='confirmation' else 'local_annotations.jsonl'))}
        pred=readl(dest/'predictions.jsonl'); metrics=json.loads((dest/'metrics.json').read_text())
        checked=0; maxdiff=0.
        for m in metrics:
            if m.get('global_token_auc') is None: continue
            truth=[]; scores=[]; maxima=[]; bags=[]
            for r in pred:
                a=anns[r['id']]; ss=np.array(r['predictions'][m['name']]['tokens']); yy=[]; mask=[]
                for lo,hi in r['offsets']:
                    yy.append(any(lo<s['end'] and hi>s['start'] for s in a['spans']))
                    valid=not any(lo<s['end'] and hi>s['start'] for s in a.get('ignore',[]))
                    if a.get('answers'): valid=valid and any(lo<s['end'] and hi>s['start'] and s['status'] in ['correct','candidate_error'] for s in a['answers'])
                    mask.append(valid)
                mask=np.array(mask); truth.extend(np.array(yy)[mask]); scores.extend(ss[mask]); bags.append(r['label'])
                if a.get('answers'):
                    answer_mask=np.array([any(lo<s['end'] and hi>s['start'] for s in a['answers']) for lo,hi in r['offsets']])
                    maxima.append(ss[answer_mask].max())
                else: maxima.append(ss.max())
            y=np.array(truth); p=np.array(scores); bag=np.array(bags); alarm=np.array(maxima)>m['alert_threshold']
            expected={'global_token_auc':rank_auc(y,p),'test_clean_answer_alarm':float(alarm[bag==0].mean()),
                'test_error_answer_recall':float(alarm[bag==1].mean()),
                'alert_token_precision':float(y[p>m['alert_threshold']].mean()) if (p>m['alert_threshold']).any() else 0.,
                'alert_error_token_recall':float(y[p>m['alert_threshold']].sum()/max(1,y.sum()))}
            for key,value in expected.items():
                diff=abs(value-m[key]); maxdiff=max(diff,maxdiff); assert diff<1e-9,(name,m['name'],key,value,m[key])
            assert m['validation_clean_answer_alarm']<=.05+1e-10
            checked+=1
        checks[name]={'method_runs':checked,'max_difference':maxdiff}
    (ROOT/'results/alert_metric_audit.json').write_text(json.dumps({'passed':True,'checked':checks,
        'scope':'independent saved-score metric checks; small validation sample does not guarantee population false alarm <=5%'},indent=2),encoding='utf8')
    print(checks)

if __name__=='__main__': main()
