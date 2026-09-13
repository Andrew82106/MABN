"""Freeze a stratified answer-label audit sample without loading any predictions."""
import argparse
import json
import random
from common import ROOT,readl,writel,sha

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--model',required=True); args=ap.parse_args()
    run=ROOT/f'data/{args.task}_{args.model}'; rows=readl(run/'labeled.jsonl'); anns={a['id']:a for a in readl(run/'local_annotations.jsonl')}
    rng=random.Random(481); sample=[]
    for label in [0,1]:
        pool=[r for r in rows if r['split']=='test' and r['label']==label]
        sample.extend(rng.sample(pool,min(15,len(pool))))
    out=[]
    for r in sample:
        out.append({'id':r['id'],'label':r['label'],'question':r.get('question'),
            'response':r['response'],'aliases':r.get('aliases',[])[:6],
            'evidence':r.get('evidence'),'answers':anns[r['id']]['answers']})
    writel(run/'review_bundle.jsonl',out)
    (run/'review_bundle_manifest.json').write_text(json.dumps({'seed':481,'n':len(out),'stratification':'test 15 per automatic response label; fewer if unavailable','predictions_read':False,'labels_sha256':sha(run/'labeled.jsonl')},indent=2),encoding='utf8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
