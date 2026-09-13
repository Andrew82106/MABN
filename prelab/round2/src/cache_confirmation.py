"""Frozen unseen human-news feature extraction with the same 7B replay protocol."""
import json
import time
import numpy as np
from common import ROOT,readl,sha
from engine import load_model,extract

def main():
    run=ROOT/'data/news_confirmation'; rows=readl(run/'rows.jsonl')
    dest=run/'features'; dest.mkdir(exist_ok=True)
    tok,model=load_model('large'); start=time.time()
    for i,r in enumerate(rows):
        path=dest/f'{r["id"]}.npz'
        if path.exists(): continue
        f=extract(tok,model,r,contrast=True)
        assert all(np.isfinite(v).all() for v in f.values())
        np.savez_compressed(path,**f)
        if i%10==0 or i==len(rows)-1: print(i+1,len(rows),round(time.time()-start,1),flush=True)
    (run/'feature_manifest.json').write_text(json.dumps({'n':len(rows),'model':'large','seconds':time.time()-start,
        'rows_sha256':sha(run/'rows.jsonl'),'mode':'controlled teacher-forced replay, same as original news'},indent=2),encoding='utf8')

if __name__=='__main__': main()
