"""Numerically verify captured prompt and generation-prefix states, no labels used."""
import json,random,time
import numpy as np
import torch
from shared import ROOT,readl
from supplement import Capture,prompt
from engine import load_model,text_prefix

@torch.inference_mode()
def main():
    tok,model=load_model('large');queries={q['query_id']:q for q in readl(ROOT/'data/queries.jsonl')};out=[]
    for arm in ['direct','verify','expand']:
        records=readl(ROOT/f'data/supplement/{arm}.jsonl');train=[r for r in records if queries[r['query_id']]['split']=='train'];sample=random.Random(250911).sample(train,3)
        for r in sample:
            q=queries[r['query_id']];ids=tok(text_prefix(tok,prompt(q,arm)),add_special_tokens=False)['input_ids'];inputs={'input_ids':torch.tensor([ids],device='cuda')}
            begin=time.time();plain=model(**inputs,use_cache=False,logits_to_keep=1).logits
            with Capture(model) as cap:hooked=model(**inputs,use_cache=False,logits_to_keep=1).logits
            assert torch.equal(plain,hooked),'capture modified logits'
            fresh=cap.array()[0,0];stored=dict(np.load(ROOT/f'data/supplement/features/{arm}_{q["query_id"]}.npz'))
            relative=float(np.linalg.norm(fresh-stored['prompt'])/(np.linalg.norm(fresh)+1e-8))
            assert relative<.05,(arm,q['query_id'],'prompt mismatch',relative)
            last_relative=None
            if r['generation_token_ids']:
                prefix=ids+r['generation_token_ids'][:-1]
                with Capture(model) as cap:model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=False,logits_to_keep=1)
                z=cap.array()[0,0];last_relative=float(np.linalg.norm(z-stored['last'])/(np.linalg.norm(z)+1e-8))
                assert last_relative<.05,(arm,q['query_id'],'generation-prefix mismatch',last_relative)
            out.append({'arm':arm,'query_id':q['query_id'],'prompt_relative_error':relative,'last_relative_error':last_relative,
                'hook_logits_exactly_unchanged':True,'seconds_two_prompt_forwards_plus_optional_prefix':time.time()-begin})
    result={'passed':True,'sample_seed':250911,'samples':out,'scope':'nine training-query numerical checks; float/quantized batched-versus-single tolerance .05; captured generation states checked by original-token causal replay',
        'prompt_only_ablation':'verification prompt state is available before any extra output; no future generated tokens needed'}
    (ROOT/'results/supplement_state_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
