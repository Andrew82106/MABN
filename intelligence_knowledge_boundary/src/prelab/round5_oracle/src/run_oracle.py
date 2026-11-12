import time,json,hashlib
import numpy as np
import torch
from common5 import R5,R4,PRE,readl,save,sha
from readouts import prompt,Capture
from engine import load_model,text_prefix

@torch.inference_mode()
def main():
    protocol=json.loads((R5/'protocol.json').read_text());ph=sha(R5/'protocol.json')
    for p,h in protocol['frozen_dependencies'].items():assert sha(R4/p)==h
    folder=R5/'data/readouts';records=folder/'records.jsonl';done={r['query_id'] for r in readl(records)} if records.exists() else set();qs=readl(R5/'data/queries.jsonl')
    cache={}
    for data in [R4/'data',R4/'confirmation/data']:
        for q in readl(data/'queries.jsonl'):
            path=data/f'readouts/{q["query_id"]}.npz'
            if path.exists():cache[hashlib.sha256(prompt(q).encode()).hexdigest()]=path
    def record(r):
        with records.open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
        done.add(r['query_id'])
    for q in qs:
        if q['query_id'] in done:continue
        key=hashlib.sha256(prompt(q).encode()).hexdigest()
        if key in cache:
            with np.load(cache[key]) as z:np.savez_compressed(folder/f'{q["query_id"]}.npz',hidden=z['hidden'],probs=z['probs'])
            record({'query_id':q['query_id'],'origin':'exact prior prompt cache','path':str(cache[key]),'prompt_sha256':key})
    todo=sorted([q for q in qs if q['query_id'] not in done],key=lambda q:(len(q['evidence'])+len(q['statement']),q['query_id']))
    print('READOUTS',len(done),'reused;',len(todo),'new',flush=True)
    replay=[];start=time.time()
    if todo:
        tok,model=load_model('large');tok.padding_side='left';tok.pad_token=tok.eos_token;ids=[tok.encode(x,add_special_tokens=False) for x in ['A','B','C']];assert all(len(x)==1 for x in ids);letters=[x[0] for x in ids];i=0
        while i<len(todo):
            size=2 if len(todo[i]['evidence'])+len(todo[i]['statement'])<6000 else 1;batch=todo[i:i+size]
            inp=tok([text_prefix(tok,prompt(q)) for q in batch],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');assert inp['input_ids'].shape[1]<=4096
            begin=time.time()
            with Capture(model) as cap:o=model(**inp,use_cache=False,logits_to_keep=1)
            p=o.logits[:,-1,letters].float().softmax(-1).cpu().numpy();h=cap.array();elapsed=time.time()-begin
            if i==0:
                plain=model(**inp,use_cache=False,logits_to_keep=1).logits;assert torch.equal(plain,o.logits);del plain
                with Capture(model) as other:again=model(**inp,use_cache=False,logits_to_keep=1)
                hp=other.array();pp=again.logits[:,-1,letters].float().softmax(-1).cpu().numpy();assert np.array_equal(h,hp) and np.array_equal(p,pp)
                replay=[{'query_id':q['query_id'],'hidden_difference':0.,'ABC_difference':0.,'hook_logits_unchanged':True} for q in batch];del again,other,hp,pp
            for j,q in enumerate(batch):
                assert np.isfinite(h[j]).all() and np.isfinite(p[j]).all() and abs(p[j].sum()-1)<1e-6
                np.savez_compressed(folder/f'{q["query_id"]}.npz',hidden=h[j].astype(np.float16),probs=p[j]);record({'query_id':q['query_id'],'origin':'fresh same-model forward','prompt_sha256':hashlib.sha256(prompt(q).encode()).hexdigest(),'batch_size':len(batch),'seconds':elapsed/len(batch),'generated_tokens':0})
            del inp,o,cap,h;i+=len(batch)
            if i%20<2 or i==len(todo):print('FRESH',i,len(todo),round(time.time()-start,1),'seconds',flush=True)
            if i%100<2:torch.cuda.empty_cache()
    assert len(done)==len(qs) and sha(R5/'protocol.json')==ph
    save(R5/'results/readout_manifest.json',{'complete':True,'queries':len(qs),'fresh_this_run':len(todo),'seconds':time.time()-start,'replay_checks':replay,'protocol_sha256':ph,'external_llm_calls':0,'generated_tokens':0})
    print('READOUTS COMPLETE',flush=True)

if __name__=='__main__':main()
