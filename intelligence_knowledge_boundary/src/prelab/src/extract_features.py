"""Causal replay: score h_t AFTER consuming token t, never future tokens."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prepare_data import ROOT, read_jsonl, sha

LAYERS=[8,16,24]
def load_model():
    torch.set_num_threads(6)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen2.5-0.5B-Instruct',local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    return tok,model

@torch.inference_mode()
def extract(row,tok,model):
    prefix=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
    enc=tok(prefix+row['response'],return_offsets_mapping=True,add_special_tokens=False)
    positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix)]
    assert positions and positions[0]>0 and len(enc['input_ids'])<=3072
    offsets=np.array([(max(0,enc['offset_mapping'][i][0]-len(prefix)),enc['offset_mapping'][i][1]-len(prefix)) for i in positions],dtype=np.int32)
    ids=torch.tensor([enc['input_ids']],device='cuda')
    out=model.model(ids,output_hidden_states=True,use_cache=False)
    hidden=torch.stack([out.hidden_states[layer][0,positions] for layer in LAYERS],dim=1).cpu().numpy()
    nll=[]
    for begin in range(0,len(positions),32):
        ix=positions[begin:begin+32]
        logits=model.lm_head(out.last_hidden_state[0,torch.tensor(ix,device='cuda')-1]).float()
        nll.extend(torch.nn.functional.cross_entropy(logits,ids[0,ix],reduction='none').cpu().tolist())
    return dict(hidden=hidden,offsets=offsets,nll=np.array(nll,dtype=np.float32),
                token_ids=np.array([enc['input_ids'][i] for i in positions],dtype=np.int32))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int); args=ap.parse_args()
    tok,model=load_model(); begin=time.time()
    rows=sum([read_jsonl(ROOT/f'data/processed/{s}.jsonl') for s in ['train','val','test']],[])
    if args.limit: rows=rows[:args.limit]
    dest=ROOT/'data/features'; dest.mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        path=dest/f'{row["id"]}.npz'
        if not path.exists():
            features=extract(row,tok,model)
            assert len(features['hidden'])==row['n_tokens']
            np.savez_compressed(path,**features)
        if i%10==0 or i==len(rows)-1:
            print(f'features {i+1}/{len(rows)} elapsed={time.time()-begin:.1f}s',flush=True)
    manifest={'model':'Qwen2.5-0.5B-Instruct','layers':LAYERS,'dtype':'float16','device':torch.cuda.get_device_name(),
              'alignment':'h_t after consuming response token t; nll_t from previous state h_(t-1)',
              'future_access':False,'n_files':len(list(dest.glob('*.npz'))),'seconds':time.time()-begin,
              'feature_source':'Replayed released Mistral answers; not Qwen spontaneous generations'}
    (ROOT/'results/feature_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(manifest,flush=True)

if __name__=='__main__': main()
