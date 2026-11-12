"""Independent numerical checks for causal alignment and attention extraction."""
import json
import argparse
import math
import numpy as np
import torch
from common import ROOT,readl
from engine import load_model,text_prefix,LookbackCapture,forward_features

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['small','large'],default='small'); args=ap.parse_args()
    torch.manual_seed(401); q=torch.randn(1,2,9,8,device='cuda'); k=torch.randn_like(q); v=torch.randn_like(q)
    ix=[3,4,5,6,7]; start=4
    original=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    with LookbackCapture(ix,start) as cap:
        captured=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    expected=[]
    for pos in ix:
        z=[]
        for head in range(2):
            weights=torch.softmax(q[0,head,pos]@k[0,head,:pos+1].T/math.sqrt(8),-1).cpu().numpy()
            a=weights[:start].mean(); b=weights[start:].mean() if pos>=start else 0.
            z.append(a/(a+b))
        expected.append(z)
    assert torch.equal(original,captured)
    ratio_difference=float(np.max(np.abs(np.array(expected)-cap.values[0])))
    assert ratio_difference<.001
    tok,model=load_model(args.model)
    row=readl(ROOT/f'data/news_{args.model}/outputs.jsonl')[0]
    prefix=text_prefix(tok,row['prompt']); enc=tok(prefix+row['response'],add_special_tokens=False,return_offsets_mapping=True)
    positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix)]
    ids=torch.tensor([enc['input_ids']],device='cuda'); chosen=positions[min(10,len(positions)-1)]
    full=model.model(ids,use_cache=False,output_hidden_states=True)
    layers=[round(model.config.num_hidden_layers*f) for f in [.25,.5,.75,1.]]
    expected_states=[full.hidden_states[layer][0,chosen-1].float().clone() for layer in layers]
    full_last=full.last_hidden_state.clone(); del full
    with LookbackCapture([chosen-1],positions[0]) as cap:
        hooked=model.model(ids,use_cache=False,output_hidden_states=False)
    assert torch.equal(full_last,hooked.last_hidden_state)
    del full_last,hooked
    short=model.model(ids[:,:chosen],use_cache=False,output_hidden_states=True)
    errors=[]
    for layer,a in zip(layers,expected_states):
        b=short.hidden_states[layer][0,-1].float()
        relative=float(torch.norm(a-b)/(torch.norm(a)+1e-8)); errors.append(relative); assert relative<.01
    del short
    cached=dict(np.load(ROOT/f'data/news_{args.model}/features/{row["id"]}.npz'))
    fresh=forward_features(tok,model,row['prompt'],row['response'])
    assert np.array_equal(cached['token_ids'],fresh['token_ids'])
    cache_error=float(np.linalg.norm(cached['before'].astype(np.float32)-fresh['before'].astype(np.float32))/(np.linalg.norm(cached['before'].astype(np.float32))+1e-8))
    assert cache_error<.001
    assert np.array_equal(cached['after'][:-1],cached['before'][1:])
    assert np.max(np.abs(cached['lookback']-fresh['lookback']))<.001
    datasets={}
    for path in sorted((ROOT/'data').glob('*/outputs.jsonl')):
        rows=readl(path); ids=[r['id'] for r in rows]; assert len(ids)==len(set(ids))
        groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','val','test']}
        assert not groups['train']&groups['test'] and not groups['train']&groups['val'] and not groups['val']&groups['test']
        for r in rows:
            f=dict(np.load(path.parent/f'features/{r["id"]}.npz'))
            for name,x in f.items(): assert np.isfinite(x).all(),(r['id'],name)
            n=len(f['offsets']); assert f['before'].shape[:2]==(n,4)
            assert np.array_equal(f['after'][:-1],f['before'][1:])
            assert f['offsets'].min()>=0 and f['offsets'].max()<=len(r['response'])
            assert f['lookback'].min()>=0 and f['lookback'].max()<=1
        datasets[path.parent.name]={'rows_at_audit':len(rows),'group_disjoint':True,'finite_and_aligned':True}
    report={'passed':True,'sdpa_output_exactly_unchanged':True,'independent_attention_ratio_max_error':ratio_difference,
        'model':args.model,'prefix_causality_relative_errors':errors,'cache_matches_fresh_extraction':True,'cache_relative_error':cache_error,'before_after_shift_exact':True,
        'datasets':datasets,'lookback_adaptation':'All prompt tokens versus prior answer tokens; official code places an extra response delimiter on generated side. Same ratio formula, different prompt boundary and backbone; not exact full reproduction.'}
    (ROOT/f'results/feature_audit_{args.model}.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
