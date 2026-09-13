"""Test exact-length unpadded full batches against individual full forwards."""
from pathlib import Path
from collections import defaultdict
import importlib.util,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e23',ROOT/'src/extract23.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

@torch.inference_mode()
def forward(model,seqs,a,b):
    ids=torch.tensor(seqs,device='cuda')
    h=model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
    logits=model.lm_head(h).float();ab=logits[:,[a,b]]
    return {'verifier_hidden':h.float().cpu().numpy(),'ab_logits':ab.cpu().numpy(),'ab_probabilities':ab.softmax(-1).cpu().numpy(),'ab_full_vocab_probabilities':logits.softmax(-1)[:,[a,b]].cpu().numpy()}

if __name__=='__main__':
    tok,rows,_,sig,plans=e.prepare();_,model=e.base.model7.load_model();results=[]
    for row in (rows[0],rows[-1]):
        p=plans[row['row_id']];groups=defaultdict(list)
        for w in p['windows']:groups[w['input_tokens']].append(w)
        ws=max(groups.values(),key=len)[:8];assert len(ws)>1
        seqs=[p['prefix_ids']+w['suffix_ids'] for w in ws]
        got=forward(model,seqs,sig['A_id'],sig['B_id']);rep=forward(model,seqs,sig['A_id'],sig['B_id'])
        refs=[e.uncached(model,s,sig['A_id'],sig['B_id'],'cuda') for s in seqs]
        ref={k:np.concatenate([x[k] for x in refs]) for k in refs[0]}
        diff={k:float(np.abs(got[k]-ref[k]).max()) for k in got}
        result={'row_id':row['row_id'],'batch_size':len(ws),'input_tokens':len(seqs[0]),'max_abs':diff,
                'hidden_mean_abs':float(np.abs(got['verifier_hidden']-ref['verifier_hidden']).mean()),
                'all_arrays_exact_against_individual':all(np.array_equal(got[k],ref[k]) for k in got),
                'repeat_exact':all(np.array_equal(got[k],rep[k]) for k in got)}
        results.append(result);print(json.dumps(result),flush=True)
    e.base.save(ROOT/'results/EQUAL_LENGTH_NUMERICAL_DIAGNOSTIC.json',{'comparisons':results,'no_labels_read':True,'no_feature_rows_saved':not any((ROOT/'data/features').glob('*.npz'))})
