import re,pickle,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from common4 import ROOT,readl,writel,save

def sentences(text):
    out=[];start=0
    for m in re.finditer(r'(?<=[.!?])\s+|\n+',text):
        if text[start:m.start()].strip():out.append([start,m.start()])
        start=m.end()
    if text[start:].strip():out.append([start,len(text)])
    return out or [[0,len(text)]]

def topmean(z):return float(np.sort(z)[-(int(len(z)*.1)+1):].mean()) if len(z) else 0.
def values(offsets,scores,span):
    lo,hi=span;z=scores[(offsets[:,0]<hi)&(offsets[:,1]>lo)]
    return [float(z.max()),float(z.mean()),topmean(z)] if len(z) else [0.,0.,0.]

def factual_span(text,sentence,offsets,scores):
    lo,hi=sentence;part=text[lo:hi];candidates={}
    patterns=[('number',r'\b\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?'),
      ('name',r'\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,3}\b'),
      ('role_relation',r'\b(?:actresses|actress|actors|actor|men|women|boys|girls|male|female|killed|injured|died|dead|won|lost|married|divorced|president|minister|chairman|chairwoman|husband|wife|son|daughter|brother|sister|before|after|million|billion|thousand)\b')]
    stop={'the','a','an','in','on','at','of','and','or','to','is','are','was','were','has','have','had','it','its','this','that','he','she','they','his','her','their','with','as','by','for','from'}
    for kind,pattern in patterns:
        for m in re.finditer(pattern,part,re.I if kind=='role_relation' else 0):
            if m.group().lower() in stop:continue
            candidates[(lo+m.start(),lo+m.end())]=kind
    for m in re.finditer(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b",part):
        if m.group().lower() not in stop:candidates.setdefault((lo+m.start(),lo+m.end()),'risk_word')
    if not candidates:return list(sentence),'sentence_fallback'
    def key(span):
        v=values(offsets,scores,span)
        return (v[2],candidates[span]!='risk_word',-(span[1]-span[0]),-span[0])
    best=max(candidates,key=key);return list(best),candidates[best]

def fit(x,y):
    with threadpool_limits(limits=4):
        return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',solver='liblinear',max_iter=700,random_state=42)).fit(np.concatenate(x),np.concatenate(y))

def main():
    if (ROOT/'data/query_manifest.json').exists():print('QUERIES FROZEN');return
    rows=readl(ROOT/'data/rows.jsonl');ann={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')}
    raw=[];support=[];offsets=[];truth=[]
    for r in rows:
        with np.load(r['feature_path']) as f:
            raw.append(f['lookback'].astype(np.float32));support.append(np.column_stack([f['lookback'],f['support']]).astype(np.float32));off=f['offsets'];offsets.append(off)
        truth.append(np.array([int(any(lo<s['end'] and hi>s['start'] for s in ann[r['id']]['spans'])) for lo,hi in off],np.float32))
    ix={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','val','test']};tr=ix['train'];groups=[rows[i]['group'] for i in tr];scores=[None]*len(rows);folds=[]
    for fold,(a,b) in enumerate(GroupKFold(3).split(tr,groups=groups)):
        ti=[tr[j] for j in a];vi=[tr[j] for j in b];net=fit([raw[j] for j in ti],[truth[j] for j in ti])
        for j in vi:scores[j]=net.predict_proba(raw[j])[:,1].astype(np.float32)
        folds.append({'fit_ids':[rows[j]['id'] for j in ti],'predicted_ids':[rows[j]['id'] for j in vi]});print('OOF',fold,flush=True)
    full=fit([raw[j] for j in tr],[truth[j] for j in tr]);strong=fit([support[j] for j in tr],[truth[j] for j in tr])
    for j in ix['val']+ix['test']:scores[j]=full.predict_proba(raw[j])[:,1].astype(np.float32)
    support_scores=[strong.predict_proba(x)[:,1].astype(np.float32) for x in support]
    bundle={'rows':rows,'split_indices':ix,'offsets':offsets,'truth':truth,'base_scores':scores,'support_scores':support_scores}
    (ROOT/'data/base.pkl').write_bytes(pickle.dumps(bundle,protocol=5));(ROOT/'results/checkpoints/base.pkl').write_bytes(pickle.dumps({'attention':full,'support':strong}))
    queries=[];policies=[]
    for i,r in enumerate(rows):
        spans=sentences(r['response']);ranks=sorted(range(len(spans)),key=lambda j:(-values(offsets[i],scores[i],spans[j])[2],j))[:3];chosen=[]
        specs=[('whole',-1,[0,len(r['response'])],[0,len(r['response'])],'whole',0)]
        for rank,j in enumerate(ranks):
            target,kind=factual_span(r['response'],spans[j],offsets[i],scores[i]);chosen.append({'sentence_span':spans[j],'fact_span':target,'candidate_kind':kind})
            specs += [('sentence',j,spans[j],spans[j],'sentence',rank),('fact',j,spans[j],target,kind,rank)]
        for kind,j,sentence,target,subtype,rank in specs:
            qid=hashlib.sha256(f'{r["id"]}|{kind}|{sentence}|{target}'.encode()).hexdigest()[:20]
            queries.append({'query_id':qid,'id':r['id'],'split':r['split'],'kind':kind,'rank':rank,'sentence_index':j,'sentence_span':sentence,
                'target_span':target,'candidate_kind':subtype,'statement':r['response'][sentence[0]:sentence[1]],'target_text':r['response'][target[0]:target[1]],
                'evidence':r['evidence'],'base_features':values(offsets[i],scores[i],target)})
        policies.append({'id':r['id'],'split':r['split'],'sentence_count':len(spans),'selected':chosen})
    writel(ROOT/'data/queries.jsonl',queries);writel(ROOT/'data/policies.jsonl',policies)
    save(ROOT/'data/query_manifest.json',{'queries':len(queries),'source_OOF_folds':folds,'selection_labels_in_prompts':False,'selection':'up to3risk-ranked sentences, one deterministic factual/risk-word span each','training_ids':[rows[j]['id'] for j in tr]})
    print('QUERIES COMPLETE',len(queries),flush=True)

if __name__=='__main__':main()
