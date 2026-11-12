"""Read-only numerical diagnostic after the failed frozen R23 GPU check."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e23',ROOT/'src/extract23.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

@torch.inference_mode()
def full_batch(model,sequences,a,b,pad):
    n=len(sequences);length=max(map(len,sequences));ids=torch.full((n,length),pad,device='cuda',dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,s in enumerate(sequences):ids[i,:len(s)]=torch.tensor(s,device='cuda');mask[i,:len(s)]=1
    h=model.model(input_ids=ids,attention_mask=mask,position_ids=torch.arange(length,device='cuda')[None].expand(n,-1),use_cache=False).last_hidden_state[torch.arange(n,device='cuda'),torch.tensor(list(map(len,sequences)),device='cuda')-1]
    logits=model.lm_head(h).float();ab=logits[:,[a,b]]
    return {'verifier_hidden':h.float().cpu().numpy(),'ab_logits':ab.cpu().numpy(),'ab_probabilities':ab.softmax(-1).cpu().numpy(),'ab_full_vocab_probabilities':logits.softmax(-1)[:,[a,b]].cpu().numpy()}

def compare(a,b):
    result={k:float(np.abs(a[k]-b[k]).max()) for k in a}
    result['hidden_mean_abs']=float(np.abs(a['verifier_hidden']-b['verifier_hidden']).mean())
    result['hidden_relative_rms']=float(np.linalg.norm(a['verifier_hidden']-b['verifier_hidden'])/np.linalg.norm(b['verifier_hidden']))
    return result

tok,rows,_,sig,plans=e.prepare();_,model=e.base.model7.load_model();results=[]
for row in (rows[0],rows[-1]):
    p=plans[row['row_id']];ix=sorted({0,len(p['windows'])//2,len(p['windows'])-1});ss=[p['windows'][i]['suffix_ids'] for i in ix]
    full=[p['prefix_ids']+s for s in ss];reference=[e.uncached(model,x,sig['A_id'],sig['B_id'],'cuda') for x in full]
    ref={k:np.concatenate([v[k] for v in reference]) for k in reference[0]}
    cache=e.prefix_cache(model,p['prefix_ids'],'cuda');bat=e.branches(model,cache,ss,sig['A_id'],sig['B_id'],tok.pad_token_id,'cuda')
    singles=[e.branches(model,cache,[s],sig['A_id'],sig['B_id'],tok.pad_token_id,'cuda') for s in ss]
    single={k:np.concatenate([v[k] for v in singles]) for k in singles[0]}
    unc=full_batch(model,full,sig['A_id'],sig['B_id'],tok.pad_token_id)
    repeat=full_batch(model,full,sig['A_id'],sig['B_id'],tok.pad_token_id)
    results.append({'row_id':row['row_id'],'cached_batch':compare(bat,ref),'cached_single':compare(single,ref),'uncached_padded_batch':compare(unc,ref),'uncached_batch_repeat_exact':all(np.array_equal(unc[k],repeat[k]) for k in unc)})
    print(json.dumps(results[-1]),flush=True)
e.base.save(ROOT/'results/CACHE_NUMERICAL_DIAGNOSTIC.json',{'failed_extractor_sha256':e.base.sha(ROOT/'src/extract23.py'),'no_feature_rows_saved':not any((ROOT/'data/features').glob('*.npz')),'comparisons':results,'no_labels_read':True})
