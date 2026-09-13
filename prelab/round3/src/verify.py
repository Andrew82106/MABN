"""Independent source, label, policy and metric checks of completed round 3."""
import json,pickle
import numpy as np
import torch
from scipy.stats import rankdata
from shared import ROOT,OLD,PRE,readl,sha,features
from networks import Probe

def auc(y,p):
    y=np.array(y,dtype=bool);n=int(y.sum());m=len(y)-n
    return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None

def main():
    rows=readl(ROOT/'data/rows.jsonl');anns={a['id']:a for a in readl(ROOT/'data/annotations.jsonl')};rr={r['id']:r for r in rows}
    assert len(rows)==385 and len(rr)==385
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    old=readl(OLD/'data/news_large/labeled.jsonl')+readl(OLD/'data/news_confirmation/rows.jsonl')
    assert not groups['test']&{r['group'] for r in old}
    original={r['id']:r for r in readl(PRE/'data/raw/response.jsonl')}
    for r in rows:
        raw=original[r['id']];assert r['response']==raw['response'] and r['label']==int(bool(raw['labels']))
        assert all(s['label_type']=='Evident Conflict' and not s.get('implicit_true') and not s.get('due_to_null') for s in raw['labels'])
        assert [(s['start'],s['end']) for s in anns[r['id']]['spans']]==[(s['start'],s['end']) for s in raw['labels']]
        for s in anns[r['id']]['spans']:assert r['response'][s['start']:s['end']].strip()==s['text'].strip()
        f=dict(np.load(features(r)));assert all(np.isfinite(x).all() for x in f.values());assert np.array_equal(f['after'][:-1],f['before'][1:])
        assert f['offsets'].min()>=0 and f['offsets'].max()<=len(r['response'])
    manifest=json.loads((ROOT/'data/query_manifest.json').read_text());covered=set()
    for fold in manifest['folds']:
        a=set(fold['fit_ids']);b=set(fold['predicted_ids']);assert not a&b
        assert not {rr[i]['group'] for i in a}&{rr[i]['group'] for i in b};covered|=b
    assert covered=={r['id'] for r in rows if r['split']=='train'}
    qs=readl(ROOT/'data/queries.jsonl');qi={q['query_id']:q for q in qs};assert len(qs)==len(qi)
    for q in qs:
        assert not any(k in q for k in ['label','spans','answer','gold','label_status'])
        lo,hi=q['span'];assert rr[q['id']]['response'][lo:hi]==q['statement'];assert q['evidence']==rr[q['id']]['evidence']
    from transformers import AutoTokenizer
    from engine import MODELS,original_token_offsets
    tok=AutoTokenizer.from_pretrained(MODELS['large'],local_files_only=True)
    for arm in ['direct','verify','expand']:
        records=readl(ROOT/f'data/supplement/{arm}.jsonl');assert len(records)==len(qs) and {r['query_id'] for r in records}==set(qi)
        for r in records:
            assert tok.decode(r['generation_token_ids'],skip_special_tokens=True)==r['response']
            if r['generation_token_ids']:original_token_offsets(tok,r['generation_token_ids'],r['response'])
            assert r['generated_tokens']==len(r['generation_token_ids'])<=64
            f=dict(np.load(ROOT/f'data/supplement/features/{arm}_{r["query_id"]}.npz'));assert all(np.isfinite(v).all() for v in f.values())
            assert all(f[k].shape==(4,3584) for k in ['prompt','mean','last'])
    causal={}
    for arch in ['mlp','conv','gru']:
        torch.manual_seed(42);net=Probe(8,architecture=arch).eval();x=torch.randn(2,19,8)
        with torch.no_grad():d=float((net(x)[:,:11]-net(x[:,:11])).abs().max())
        assert d<1e-6;causal[arch]=d
    pred=readl(ROOT/'results/test_predictions.jsonl');vp=readl(ROOT/'results/validation_predictions.jsonl');metrics=json.loads((ROOT/'results/metrics.json').read_text());assert {r['id'] for r in pred}=={r['id'] for r in rows if r['split']=='test'}
    inp=pickle.loads((ROOT/'data/inputs.pkl').read_bytes());vy=np.concatenate(inp['y']['val'])
    assert [r['id'] for r in vp]==[r['id'] for r in inp['rows']['val']]
    selection=json.loads((ROOT/'results/structure_selection.json').read_text());grid=json.loads((ROOT/'configs/structure_grid.json').read_text())
    assert len(grid)==18;initial=[];val_differences=[]
    for config in grid:
        cp=torch.load(ROOT/f'results/checkpoints/{config["id"]}_42.pt',map_location='cpu',weights_only=False)
        assert cp['config']==config and cp['history'][-1]['step']==400
        assert abs(cp['val_global_token_auc']-max(h['val_global_token_auc'] for h in cp['history']))<1e-12
        assert all(torch.isfinite(x).all() for x in cp['state'].values());initial.append(cp)
    assert max(initial,key=lambda c:c['val_global_token_auc'])['config']['id']==selection['overall_seed42_winner']
    for family,entry in selection['family_winners'].items():
        candidates=[c for c in initial if c['config']['feature']+'_'+c['config']['architecture']==family]
        assert max(candidates,key=lambda c:c['val_global_token_auc'])['config']['id']==entry['config_id']
        for seed in [42,43,44]:
            cp=torch.load(ROOT/f'results/checkpoints/{entry["config_id"]}_{seed}.pt',map_location='cpu',weights_only=False)
            assert cp['history'][-1]['step']==400
            actual=auc(vy,np.concatenate([r['predictions'][family+'_'+str(seed)] for r in vp]))
            diff=abs(actual-cp['val_global_token_auc']);assert diff<1e-4,(family,seed,diff);val_differences.append(diff)
    errors=[];maxdiff=0
    for m in metrics:
        name=m['name'];yy=[];pp=[];local=[];labels=[];alarm=[];clean_val=[]
        for r in vp:
            if r['label']==0:clean_val.append(max(r['predictions'][name]))
        threshold=sorted(clean_val)[int(np.ceil(.95*(len(clean_val)-1)))];assert abs(threshold-m['alert_threshold'])<1e-7
        for r in pred:
            y=[int(any(lo<s['end'] and hi>s['start'] for s in anns[r['id']]['spans'])) for lo,hi in r['offsets']];p=r['predictions'][name]
            assert len(y)==len(p) and np.isfinite(p).all()
            yy.extend(y);pp.extend(p);a=auc(y,p)
            if a is not None:local.append(a)
            labels.append(r['label']);alarm.append(max(p)>threshold)
        labels=np.array(labels);alarm=np.array(alarm)
        expected={'global_token_auc':auc(yy,pp),'local_auc':float(np.mean(local)),'clean_alarm':float(alarm[labels==0].mean()),'error_recall':float(alarm[labels==1].mean())}
        for key,v in expected.items():
            d=abs(v-m[key]);maxdiff=max(d,maxdiff)
            assert d<1e-7,(name,key,v,m[key])
    src=ROOT/'src';result={'passed':True,'rows':385,'test_n':80,'supplement_unique_queries':len(qs),
        'source_disjoint':True,'OOF_selection_verified':True,'original_annotation_slices_match':True,'labels_equal_released_dataset':True,
        'structure_grid_configs':len(grid),'structure_fit_runs':30,'validation_choice_verified':True,'validation_cpu_gpu_auc_max_difference':max(val_differences),
        'supplement_decoding_verified':True,'finite_features':True,'causal_network_prefix_errors':causal,
        'independent_metric_max_difference':maxdiff,'methods_checked':len(metrics),
        'code_sha256':{p.name:sha(p) for p in sorted(src.glob('*.py'))},
        'scope':'integrity and metric audit; does not independently reannotate news or establish live self-generation effectiveness'}
    (ROOT/'results/final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
