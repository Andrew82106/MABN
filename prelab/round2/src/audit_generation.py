"""Check sampled replay tokenization against a seeded regeneration, without labels."""
import argparse
import json
import random
import numpy as np
from common import ROOT,readl
from engine import load_model,generate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); args=ap.parse_args()
    tok,model=load_model(args.model); out=[]
    for task in ['trivia','rag']:
        run=ROOT/f'data/{task}_{args.model}'
        if not (run/'manifest.json').exists(): continue
        rows=readl(run/'outputs.jsonl'); sample=random.Random(8197).sample(rows,min(8,len(rows)))
        for r in sample:
            text,limit,ids=generate(tok,model,r['prompt'],r['generation_seed'],48 if task=='trivia' else 100,return_trace=True)
            f=dict(np.load(run/f'features/{r["id"]}.npz'))
            out.append({'task':task,'id':r['id'],'regenerated_text_exact':text==r['response'],
                'regenerated_ids_equal_replay':np.array_equal(f['token_ids'],ids),
                'n_regenerated_tokens':len(ids),'n_replay_tokens':len(f['token_ids'])})
    passed=bool(out) and all(r['regenerated_text_exact'] and r['regenerated_ids_equal_replay'] for r in out)
    report={'passed':passed,'sample_seed':8197,'samples':out,'scope':'Sampled seeded regeneration, not saved live state traces for every prior response. RAG runs after the trace addition also record actual generation token IDs.'}
    (ROOT/f'results/generation_audit_{args.model}.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
