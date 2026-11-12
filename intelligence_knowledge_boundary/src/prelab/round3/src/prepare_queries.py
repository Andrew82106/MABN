"""Source-group out-of-fold risk selection. Query text never includes gold labels."""
import json,pickle,re,hashlib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from shared import ROOT,data,features,readl,writel

def sentences(text):
    out=[];start=0
    for m in re.finditer(r'(?<=[.!?])\s+|\n+',text):
        if text[start:m.start()].strip():out.append([start,m.start()])
        start=m.end()
    if text[start:].strip():out.append([start,len(text)])
    return out or [[0,len(text)]]

def fit(xx,yy):
    with threadpool_limits(limits=4):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',solver='liblinear',max_iter=700,random_state=42))
        model.fit(np.concatenate(xx),np.concatenate(yy))
    return model

def main():
    if (ROOT/'data/query_manifest.json').exists():print('QUERIES FROZEN');return
    (ROOT/'results/checkpoints').mkdir(parents=True,exist_ok=True)
    inp=pickle.loads((ROOT/'data/inputs.pkl').read_bytes());dd=inp['rows'];raw={s:[np.load(features(r))['lookback'].astype(np.float32) for r in rr] for s,rr in dd.items()}
    group=[r['group'] for r in dd['train']];scores={s:[None]*len(rr) for s,rr in dd.items()};folds=[]
    for fold,(ti,vi) in enumerate(GroupKFold(3).split(group,groups=group)):
        net=fit([raw['train'][i] for i in ti],[inp['y']['train'][i] for i in ti])
        for i in vi:scores['train'][i]=net.predict_proba(raw['train'][i])[:,1].astype(np.float32)
        folds.append({'fold':fold,'fit_ids':[dd['train'][i]['id'] for i in ti],'predicted_ids':[dd['train'][i]['id'] for i in vi]});print('OOF',fold,flush=True)
    full=fit(raw['train'],inp['y']['train'])
    for s in ['val','test']:
        scores[s]=[full.predict_proba(x)[:,1].astype(np.float32) for x in raw[s]]
    queries=[];policies=[]
    for s,rr in dd.items():
        for i,r in enumerate(rr):
            spans=sentences(r['response']);off=inp['offsets'][s][i];ss=scores[s][i];risk=[]
            for lo,hi in spans:
                z=ss[(off[:,0]<hi)&(off[:,1]>lo)]
                risk.append(float(np.sort(z)[-(int(len(z)*.1)+1):].mean()) if len(z) else -1)
            active=int(np.argmax(risk));seed=int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16);random=int(np.random.default_rng(seed).integers(len(spans)))
            for policy,j in [('active',active),('random',random)]:
                policies.append({'id':r['id'],'split':s,'policy':policy,'query_id':f'{r["id"]}_{j}','sentence_index':j,'span':spans[j],'base_sentence_risk':risk[j]})
            for j in sorted({active,random}):
                lo,hi=spans[j];queries.append({'query_id':f'{r["id"]}_{j}','id':r['id'],'split':s,'sentence_index':j,
                    'span':[lo,hi],'statement':r['response'][lo:hi],'evidence':r['evidence']})
    writel(ROOT/'data/queries.jsonl',queries);writel(ROOT/'data/query_policies.jsonl',policies)
    (ROOT/'data/policy_scores.pkl').write_bytes(pickle.dumps(scores));(ROOT/'results/checkpoints/policy_linear.pkl').write_bytes(pickle.dumps(full))
    (ROOT/'data/query_manifest.json').write_text(json.dumps({'n_queries':len(queries),'n_policies':len(policies),
        'selection':'top10 token mean per sentence; top sentence vs uniformly random sentence; no gold passed to selection',
        'folds':folds,'training_prediction_scope':'strict source-group OOF fitting including scaler',
        'test_labels_used':False},indent=2),encoding='utf8');print('QUERY PREPARATION COMPLETE',len(queries),flush=True)

if __name__=='__main__':main()
