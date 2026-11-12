import json,time
import numpy as np
from shared import ROOT,data,features
from engine import load_model,extract

def main():
    tok,model=load_model('large');start=time.time();rows=data()['test']
    for i,r in enumerate(rows):
        p=features(r);p.parent.mkdir(exist_ok=True)
        if p.exists():continue
        f=extract(tok,model,r,contrast=True)
        assert all(np.isfinite(x).all() for x in f.values())
        np.savez_compressed(p,**f)
        if i%10==0 or i==len(rows)-1:print(i+1,len(rows),round(time.time()-start,1),flush=True)
    (ROOT/'data/feature_manifest.json').write_text(json.dumps({'n':len(rows),'seconds':time.time()-start,'model':'large','mode':'teacher-forced same model/protocol as old news'},indent=2),encoding='utf8')

if __name__=='__main__':main()
