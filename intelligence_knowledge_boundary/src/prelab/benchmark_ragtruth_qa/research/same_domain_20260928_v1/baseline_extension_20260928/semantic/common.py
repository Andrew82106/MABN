from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE.parent.parent
PROJECT=next(p for p in HERE.parents if p.name=='intelligence_knowledge_boundary')
DATA=RUN/'data'
sys.path.insert(0,str(RUN))
import metrics

def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.pending')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def write_rows(p,rr):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.pending')
    with t.open('w',encoding='utf-8') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    t.replace(p)
def inputs(split):
    aa=rows(DATA/split/'inputs.jsonl');gg={r['response_id']:r for r in rows(DATA/split/'geometry.jsonl')}
    assert len(aa)=={'fit':1292,'validation':271,'test':279}[split]
    assert set(gg)=={r['response_id'] for r in aa}
    return aa,gg
def mapping(text,raw_offsets,offsets):
    owners=[[] for _ in text]
    valid_offsets=[]
    for j,(a,b) in enumerate(offsets):
        if a<0 or a==b:continue
        assert 0<=a<b<=len(text)
        valid_offsets.append((j,a,b))
        for c in range(a,b):owners[c].append(j)
    assert all(owners[c] for c,ch in enumerate(text) if not ch.isspace())
    assert valid_offsets
    ii,jj,ww=[],[],[]
    for i,(a,b) in enumerate(raw_offsets):
        cc=[c for c in range(a,b) if not text[c].isspace()]
        # Mean-window adapters must not get fabricated zero risks for whitespace.
        # Use native coverage when possible; uncovered gaps take nearest native
        # interval, tie earlier. This does not affect lexical-max evaluation.
        if not cc:cc=list(range(a,b))
        weights=Counter()
        for c in cc:
            ids=owners[c]
            if not ids:ids=[min(valid_offsets,key=lambda z:(max(z[1]-c,c-z[2]+1,0),z[1]))[0]]
            for j in ids:weights[j]+=1/(len(cc)*len(ids))
        if not cc:
            weights[min(valid_offsets,key=lambda z:(max(z[1]-a,a-z[2]+1,0),z[1]))[0]]=1.
        for j,w in weights.items():ii.append(i);jj.append(j);ww.append(w)
    return [ii,jj,ww]
def mapped(score,mp,n):
    ii,jj,ww=mp
    return np.bincount(ii,weights=np.asarray(score,dtype=np.float64)[jj]*ww,minlength=n).tolist()
def val_reports(preds):
    scores={r['response_id']:r['token_scores'] for r in preds}
    rr={p:metrics.evaluate_partition('validation',scores,pooling=p) for p in ['mean','max']}
    best=max(rr,key=lambda p:tuple(rr[p]['selection_key']))
    return {'poolings':rr,'selected_pooling':best,'evaluation':rr[best]}
def source_hashes(split):
    return {str(p):sha(p) for p in [DATA/split/'inputs.jsonl',DATA/split/'geometry.jsonl',DATA/'split_assignments.jsonl']}
