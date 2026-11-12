import time
from pathlib import Path
import numpy as np
from common4 import ROOT,readl,save
from engine import load_model,extract
def main():
    tok,model=load_model('large');started=time.time();rows=[r for r in readl(ROOT/'data/rows.jsonl') if r['split']=='test']
    for i,r in enumerate(rows):
        path=Path(r['feature_path'])
        if path.exists():continue
        f=extract(tok,model,r,contrast=True);assert all(np.isfinite(v).all() for v in f.values());np.savez_compressed(path,**f)
        if i%5==0 or i==len(rows)-1:print('BASE FEATURES',i+1,len(rows),round(time.time()-started,1),flush=True)
    save(ROOT/'data/feature_manifest.json',{'n':100,'model':'same Qwen7B NF4','seconds':time.time()-started})
if __name__=='__main__':main()
