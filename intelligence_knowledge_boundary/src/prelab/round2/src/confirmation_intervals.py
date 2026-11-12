"""Paired source bootstrap, fixed checkpoints; not independent retraining uncertainty."""
import json
import numpy as np
from common import ROOT,readl
from verify_alerts import rank_auc

def main():
    dest=ROOT/'results/confirmation'; rows=readl(dest/'predictions.jsonl')
    ann={a['id']:a for a in readl(ROOT/'data/news_confirmation/annotations.jsonl')}
    metrics=json.loads((dest/'metrics.json').read_text())
    methods=list(dict.fromkeys(m['method'] for m in metrics)); data={}; y=np.array([r['label'] for r in rows])
    groups=list(dict.fromkeys(r['group'] for r in rows)); indices=[[i for i,r in enumerate(rows) if r['group']==g] for g in groups]
    for method in methods:
        mm=[m for m in metrics if m['method']==method]; local=[]; alarms=[]
        for r in rows:
            truth=[any(lo<s['end'] and hi>s['start'] for s in ann[r['id']]['spans']) for lo,hi in r['offsets']]
            values=[rank_auc(truth,r['predictions'][m['name']]['tokens']) for m in mm]
            local.append(np.nan if values[0] is None else np.mean(values))
            alarms.append(np.mean([max(r['predictions'][m['name']]['tokens'])>m['alert_threshold'] for m in mm]))
        data[method]=(np.array(local),np.array(alarms))
    rng=np.random.default_rng(250910); draws={m:{'local_auc':[],'clean_alarm':[],'error_recall':[]} for m in methods}
    for _ in range(1000):
        ix=np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))])
        for method,(loc,alarm) in data.items():
            draws[method]['local_auc'].append(float(np.nanmean(loc[ix])))
            draws[method]['clean_alarm'].append(float(alarm[ix][y[ix]==0].mean()))
            draws[method]['error_recall'].append(float(alarm[ix][y[ix]==1].mean()))
    out={'unit':'source group','n_bootstrap':1000,'seed':250910,
        'scope':'fixed-seed-model mean, sampling uncertainty across heldout sources; no parameter selection',
        'intervals':{m:{k:np.quantile(v,[.025,.975]).tolist() for k,v in d.items()} for m,d in draws.items()},'paired_differences':{}}
    for a,b in [('local_mlp','lookback'),('local_mlp_alarm','local_mlp'),('local_mlp','supervised_lookback')]:
        out['paired_differences'][a+'_minus_'+b]={k:np.quantile(np.array(draws[a][k])-draws[b][k],[.025,.975]).tolist() for k in draws[a]}
    (dest/'confidence_intervals.json').write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
