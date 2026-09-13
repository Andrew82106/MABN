"""No fitting or method selection on newly reserved source-disjoint human news."""
import json
import pickle
from collections import defaultdict
import numpy as np
import torch
from common import ROOT,readl,writel,sha
from probes import load_data,load_features,best_threshold
import evaluate as ev

def main():
    ev.DEVICE='cpu'; torch.set_num_threads(6)
    run=ROOT/'data/news_confirmation'; old=ROOT/'data/news_large'
    dest=ROOT/'results/confirmation'; dest.mkdir(exist_ok=True)
    rows=readl(run/'rows.jsonl'); val=load_data(old)['val']
    assert not {r['group'] for r in rows}&{r['group'] for r in readl(old/'labeled.jsonl')}
    tf,vf=load_features(run,rows),load_features(old,val)
    ann={r['id']:r for r in readl(run/'annotations.jsonl')}
    cpdir=ROOT/'results/news_large/checkpoints'
    names=[f'{m}_{s}.pt' for m in ['before','lookback','local_mlp','local_mlp_alarm'] for s in [42,43,44]]+['supervised_after.pkl','supervised_lookback.pkl']
    assert all((cpdir/n).exists() for n in names)
    (dest/'frozen_models.json').write_text(json.dumps({n:sha(cpdir/n) for n in names},indent=2),encoding='utf8')
    metrics=[]; predictions={}
    for name in names+['nll_topk']:
        if name=='nll_topk':
            ss=[f['nll'] for f in tf]; vs=[f['nll'] for f in vf]; p=ev.aggregate(ss); vp=ev.aggregate(vs)
            cp={'method':name,'threshold':best_threshold([r['label'] for r in val],vp),'supervision':'no trained weights'}
        else:
            path=cpdir/name
            cp=torch.load(path,map_location='cpu',weights_only=False) if path.suffix=='.pt' else pickle.loads(path.read_bytes())
            p,ss=ev.predict(cp,tf,rows); vp,vs=ev.predict(cp,vf,val)
        m,loc=ev.measure(rows,tf,ann,p,ss,cp['threshold'])
        m.update(ev.calibrated_alert(rows,tf,ann,ss,val,vf,vs))
        m.update(name=name.rsplit('.',1)[0],method=cp['method'],seed=cp.get('seed'),threshold=cp['threshold'],supervision=cp['supervision'])
        metrics.append(m); predictions[m['name']]=(p,ss)
        print(m['name'],{k:round(m[k],3) for k in ['local_auc','global_token_auc','test_clean_answer_alarm','test_error_answer_recall']},flush=True)
    groups=defaultdict(list)
    for m in metrics: groups[m['method']].append(m)
    summary=[]
    for method,ms in groups.items():
        item={'method':method,'runs':len(ms)}
        for key in ['bag_auc','local_auc','global_token_auc','top10_precision','test_clean_answer_alarm','test_error_answer_recall','alert_token_precision','alert_error_token_recall']:
            item[key]=ev.average([m.get(key) for m in ms])
        summary.append(item)
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    # Different name deliberately keeps the primary-run report scanner separate.
    (dest/'confirmation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    writel(dest/'predictions.jsonl',[{'id':r['id'],'group':r['group'],'label':r['label'],'offsets':tf[i]['offsets'].tolist(),
        'predictions':{n:{'bag':float(p[i]),'tokens':s[i].tolist()} for n,(p,s) in predictions.items()}} for i,r in enumerate(rows)])
    (dest/'evaluation_manifest.json').write_text(json.dumps({'n':len(rows),'clean':sum(r['label']==0 for r in rows),
        'error':sum(r['label']==1 for r in rows),'source_disjoint':True,'threshold_source':'original news validation only',
        'labels_sha256':sha(run/'rows.jsonl'),'annotation_sha256':sha(run/'annotations.jsonl'),
        'purpose':'confirmatory holdout for exploratory supervised alert controls; no selection on these results'},indent=2),encoding='utf8')

if __name__=='__main__': main()
