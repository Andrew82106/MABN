"""Restore actual generated tokenization for the traced RAG run, before any fitting."""
import argparse
import json
import numpy as np
from common import ROOT,readl,writel
from engine import load_model,extract,original_token_offsets
from engine import MODELS
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='large'); ap.add_argument('--task',default='rag'); args=ap.parse_args()
    run=ROOT/f'data/{args.task}_{args.model}'; rows=readl(run/'outputs.jsonl'); needs=[]
    assert (run/'manifest.json').exists(),'Wait for generation completion'
    for r in rows:
        if 'generation_token_ids' not in r: continue
        actual=np.load(run/f'features/{r["id"]}.npz')['token_ids']
        if not np.array_equal(actual,r['generation_token_ids']): needs.append(r)
    if needs: tok,model=load_model(args.model)
    else: tok=AutoTokenizer.from_pretrained(MODELS[args.model],local_files_only=True); model=None
    for r in needs:
        f=extract(tok,model,r,contrast=args.task=='rag'); assert np.array_equal(f['token_ids'],r['generation_token_ids'])
        np.savez_compressed(run/f'features/{r["id"]}.npz',**f)
        print('REPAIRED',r['id'],flush=True)
    for r in rows:
        if 'generation_token_ids' not in r: continue
        original_token_offsets(tok,r['generation_token_ids'],r['response'])
        r['canonical_tokenization_matched_generation']=r.get('canonical_tokenization_matched_generation',r['token_replay_exact'])
        r['token_replay_exact']=True
        assert np.array_equal(np.load(run/f'features/{r["id"]}.npz')['token_ids'],r['generation_token_ids'])
    writel(run/'outputs.jsonl',rows)
    report_path=ROOT/f'results/token_repair_{args.task}_{args.model}.json'
    previous=json.loads(report_path.read_text())['repaired'] if report_path.exists() else []
    report={'passed':True,'n':len(rows),'repaired':sorted(set(previous)|{r['id'] for r in needs}),'repaired_this_run':[r['id'] for r in needs],
        'reason':'Decoding and retokenizing can merge adjacent punctuation or use another subword segmentation. Features now replay original emitted token IDs, with UTF-8 byte-aligned character spans.'}
    report_path.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
