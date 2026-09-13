"""Replay six training targets with their exact original fresh-forward batches."""
import random
import numpy as np
import torch
from common4 import ROOT,readl,save
from engine import load_model,text_prefix
from readouts import Capture,prompt

@torch.inference_mode()
def main():
    qs={q['query_id']:q for q in readl(ROOT/'data/queries.jsonl')};fresh=[r for r in readl(ROOT/'data/readouts/records.jsonl') if r['origin']=='round4/fresh forward'];groups={};i=0
    while i<len(fresh):
        group=fresh[i:i+fresh[i]['batch_size']];assert len(group)==fresh[i]['batch_size']
        for r in group:groups[r['query_id']]=group
        i+=len(group)
    selected=[]
    for kind in ['whole','sentence','fact']:
        pool=[r for r in fresh if qs[r['query_id']]['split']=='train' and r['kind']==kind];selected+=random.Random(20260913).sample(pool,2)
    tok,model=load_model('large');tok.padding_side='left';tok.pad_token=tok.eos_token;letters=[tok.encode(c,add_special_tokens=False)[0] for c in ['A','B','C']];out=[]
    for r in selected:
        group=groups[r['query_id']];inp=tok([text_prefix(tok,prompt(qs[s['query_id']])) for s in group],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        plain=model(**inp,use_cache=False,logits_to_keep=1).logits
        with Capture(model) as cap:o=model(**inp,use_cache=False,logits_to_keep=1)
        assert torch.equal(plain,o.logits),'hook changed logits';p=o.logits[:,-1,letters].float().softmax(-1).cpu().numpy();h=cap.array();j=[s['query_id'] for s in group].index(r['query_id'])
        with np.load(ROOT/f'data/readouts/{r["query_id"]}.npz') as saved:
            relative=float(np.linalg.norm(h[j]-saved['hidden'])/(np.linalg.norm(h[j])+1e-9));delta=float(np.max(np.abs(p[j]-saved['probs'])))
        assert relative<.01 and delta<.01,(r['query_id'],relative,delta)
        out.append({'query_id':r['query_id'],'kind':r['kind'],'matched_batch_size':len(group),'relative_state_difference':relative,'max_ABC_difference':delta,'hook_logits_unchanged':True})
        del inp,plain,o,cap;torch.cuda.empty_cache()
    save(ROOT/'results/readout_audit.json',{'passed':True,'training_targets':6,'matched_original_batches':True,'checks':out});print('READOUT REPLAY AUDIT PASSED',flush=True)

if __name__=='__main__':main()
