"""Small target-model generation audit on heldout sources, no injected false answers."""
import json
import random
import time
import numpy as np
import torch
from prepare_data import ROOT,read_jsonl,write_jsonl
from extract_features import load_model,extract

def main():
    tok,model=load_model(); torch.manual_seed(20260909)
    candidates=read_jsonl(ROOT/'data/processed/test.jsonl')
    rng=random.Random(20260909); rng.shuffle(candidates); candidates=candidates[:16]
    path=ROOT/'data/own/generated.jsonl'; path.parent.mkdir(exist_ok=True)
    dest=ROOT/'data/own/features'; dest.mkdir(exist_ok=True)
    previous=read_jsonl(path) if path.exists() else []; ids={r['id'] for r in previous}
    for i,r in enumerate(candidates):
        rid='qwen_'+r['id']
        if rid in ids: continue
        prompt='Summarize the following news in at most 100 words. State only facts supported by the article.\n'+r['evidence']+'\nSummary:'
        text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        inputs=tok(text,return_tensors='pt').to('cuda')
        # Per-source seeds make resume reproduce exactly the same generation.
        torch.manual_seed(20260909+int(r['id']))
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=220,do_sample=True,temperature=.7,top_p=.9,pad_token_id=tok.eos_token_id)
        response=tok.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        row={'id':rid,'parent_id':r['id'],'source_id':r['source_id'],'group':r['group'],'prompt':prompt,
             'evidence':r['evidence'],'response':response,'generator':'Qwen2.5-0.5B-Instruct',
             'label':None,'decoding':{'temperature':.7,'top_p':.9,'max_new_tokens':220,'seed':20260909+int(r['id'])},
             'reached_token_limit':output.shape[1]-inputs['input_ids'].shape[1]>=220}
        features=extract(row,tok,model); row['n_tokens']=len(features['nll'])
        np.savez_compressed(dest/f'{rid}.npz',**features)
        previous.append(row); write_jsonl(path,previous)
        print(f'own {i+1}/16 {rid}: {response[:100]}',flush=True)
    # Plain-text annotation packet deliberately excludes probe predictions.
    packet='\n\n'.join(f'ID {r["id"]}\nSOURCE:\n{r["evidence"]}\nRESPONSE:\n{r["response"]}' for r in previous)
    (ROOT/'data/own/annotation_packet.txt').write_text(packet,encoding='utf8')

if __name__=='__main__': main()
