import argparse
import json
import time
import numpy as np
import torch
from common import ROOT,PRELAB,readl,writel,sha
from engine import load_model,extract,generate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['small','large'],required=True)
    ap.add_argument('--task',choices=['news','trivia','rag'],required=True); ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    if args.task=='news':
        rows=sum([readl(PRELAB/f'data/processed/{s}.jsonl') for s in ['train','val','test']],[])
    else: rows=readl(ROOT/f'data/{args.task}_questions.jsonl')
    if args.limit: rows=rows[:args.limit]
    run=ROOT/f'data/{args.task}_{args.model}'; run.mkdir(exist_ok=True); dest=run/'features'; dest.mkdir(exist_ok=True)
    outfile=run/'outputs.jsonl'; existing=readl(outfile) if outfile.exists() else []
    saved={x['id']:x for x in existing}
    tok,model=load_model(args.model); started=time.time()
    for i,row in enumerate(rows):
        path=dest/f'{row["id"]}.npz'
        if row['id'] in saved and path.exists(): continue
        row=dict(row)
        if row['id'] in saved: row=saved[row['id']]
        elif args.task!='news':
            response,limit,trace=generate(tok,model,row['prompt'],20260910+i,48 if args.task=='trivia' else 100,return_trace=True)
            row.update(response=response,reached_token_limit=limit,generation_seed=20260910+i,generation_token_ids=trace)
        features=extract(tok,model,row,contrast=args.task in ['news','rag'])
        if 'generation_token_ids' in row:
            row['token_replay_exact']=np.array_equal(features['token_ids'],row['generation_token_ids'])
        assert np.isfinite(features['before']).all()
        np.savez_compressed(path,**features)
        if row['id'] not in saved:
            with outfile.open('a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            saved[row['id']]=row
        if i%10==0 or i==len(rows)-1:
            print(f'{args.task}/{args.model} {i+1}/{len(rows)} elapsed={time.time()-started:.1f}s',flush=True)
    (run/'manifest.json').write_text(json.dumps({'model':args.model,'task':args.task,'n_outputs':len(saved),
        'layers':features['layers'].tolist() if 'features' in locals() else None,'seconds_this_run':time.time()-started,
        'input_sha256':sha(ROOT/f'data/{args.task}_questions.jsonl') if args.task!='news' else 'see round1 manifest',
        'mode':'same-model natural generation' if args.task!='news' else 'controlled teacher-forced model/timing comparison',
        'gpu':torch.cuda.get_device_name(0),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'features':'before and after token, lookback ratio, entropy, NLL; evidence removal contrast for news/RAG'},indent=2),encoding='utf8')

if __name__=='__main__': main()
