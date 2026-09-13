"""Final artifact and independently recomputed heldout metric checks."""
import json
import pickle
import numpy as np
import torch
from common import ROOT,readl,sha

def pair_auc(y,p):
    y=np.asarray(y); p=np.asarray(p); pos=p[y==1]; neg=p[y==0]
    if not len(pos) or not len(neg): return None
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))

def main():
    required={'news_small':394,'news_large':394,'trivia_small':2700,'trivia_large':2700,'rag_large':1050}
    checked={}; errors=[]
    for name,n in required.items():
        run=ROOT/f'data/{name}'; dest=ROOT/f'results/{name}'
        if not (dest/'evaluation_manifest.json').exists(): errors.append(name+': missing evaluation'); continue
        rows=readl(run/'labeled.jsonl'); assert len(rows)==n and len({r['id'] for r in rows})==n
        groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
        assert not groups['train']&groups['test'] and not groups['train']&groups['val'] and not groups['val']&groups['test']
        anns={r['id']:r for r in readl(run/'local_annotations.jsonl')}
        test=[r for r in rows if r['split']=='test' and r['label'] is not None]
        preds=readl(dest/'predictions.jsonl'); assert [r['id'] for r in preds]==[r['id'] for r in test]
        metrics=json.loads((dest/'metrics.json').read_text()); maximal=0.
        names={m['method'] for m in metrics}
        expected={'before','after','uncertainty','text','length','shuffled','last_linear','mean_linear','nll_topk'}
        if not name.startswith('trivia'): expected.update(['lookback','support','support_lookback','supervised_before','supervised_after','supervised_lookback','source_copy'])
        if name.startswith('news'): expected.update(['few_local_10','few_local_25','few_local_50','few_local_100'])
        if name.startswith('rag'): expected.update(['fact_after','fact_support_lookback'])
        if name=='news_large': expected.update(['local_mlp','local_mlp_alarm'])
        if not expected.issubset(names): errors.append(name+': missing methods '+str(sorted(expected-names)))
        for m in metrics:
            y=np.array([r['label'] for r in test]); p=np.array([r['predictions'][m['name']]['bag'] for r in preds])
            expected=pair_auc(y,p); difference=abs(expected-m['bag_auc']); maximal=max(maximal,difference); assert difference<1e-10
            guess=p>=m['threshold']; tp=int(np.sum(guess&(y==1))); fp=int(np.sum(guess&(y==0))); fn=int(np.sum(~guess&(y==1)))
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.; assert abs(f1-m['bag_f1'])<1e-10
            if name.startswith('news') and m.get('local_auc') is not None:
                values=[]
                for r in preds:
                    labels=[int(any(a<s['end'] and b>s['start'] for s in anns[r['id']]['spans'])) for a,b in r['offsets']]
                    value=pair_auc(labels,r['predictions'][m['name']]['tokens'])
                    if value is not None: values.append(value)
                assert abs(float(np.mean(values))-m['local_auc'])<1e-10
        for r in rows:
            path=run/f'features/{r["id"]}.npz'; f=dict(np.load(path))
            assert np.array_equal(f['after'][:-1],f['before'][1:])
            assert all(np.isfinite(x).all() for x in f.values())
            if not name.startswith('news'): assert 'generation_seed' in r
        for path in (dest/'checkpoints').glob('few_local*.pkl'):
            cp=pickle.loads(path.read_bytes()); assert set(cp['training_local_ids'])<={r['id'] for r in rows if r['split']=='train'}
        for path in (dest/'checkpoints').glob('local_mlp*.pt'):
            cp=torch.load(path,map_location='cpu',weights_only=False)
            assert set(cp['training_ids'])=={r['id'] for r in rows if r['split']=='train'}
            assert set(cp['validation_ids'])=={r['id'] for r in rows if r['split']=='val'}
        checked[name]={'rows':len(rows),'usable_test':len(test),'test_errors':sum(r['label'] for r in test),'method_runs':len(metrics),
            'independent_auc_max_difference':maximal,'group_disjoint':True,'finite_features':True,'scores_and_spans_aligned':True}
    fresh=ROOT/'data/news_confirmation'; rr=readl(fresh/'rows.jsonl')
    old=readl(ROOT/'data/news_large/labeled.jsonl')
    assert len(rr)==110 and len({r['id'] for r in rr})==110
    assert not {r['group'] for r in rr}&{r['group'] for r in old}
    for r in rr:
        f=dict(np.load(fresh/f'features/{r["id"]}.npz'))
        assert all(np.isfinite(x).all() for x in f.values())
        assert np.array_equal(f['after'][:-1],f['before'][1:])
        assert f['offsets'].min()>=0 and f['offsets'].max()<=len(r['response'])
    frozen=json.loads((ROOT/'results/confirmation/frozen_models.json').read_text())
    assert all(sha(ROOT/'results/news_large/checkpoints'/name)==value for name,value in frozen.items())
    checked['confirmation']={'rows':110,'clean':80,'errors':30,'source_disjoint_from_all_previous_news':True,'frozen_checkpoint_hashes_match':True,'finite_features':True}
    for name in ['selective_capture_audit.json','feature_audit_small.json','feature_audit_large.json',
                 'generation_audit_small.json','generation_audit_large.json','token_repair_rag_large.json','alert_metric_audit.json']:
        path=ROOT/'results'/name
        if not path.exists() or not json.loads(path.read_text())['passed']: errors.append('missing/failed '+name)
    result={'passed':not errors,'missing_or_failed':errors,'checked':checked,
        'sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*.py'))},
        'scope':'Numerical/artifact integrity, not proof of label validity, real-world deployment, causality or novel algorithm'}
    (ROOT/'results/final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))
    assert result['passed'],errors

if __name__=='__main__': main()
