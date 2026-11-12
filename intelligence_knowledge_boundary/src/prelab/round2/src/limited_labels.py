"""Exploratory label-budget diagnostic; only bag labels used for validation selection."""
import argparse
import pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import ROOT,readl
from probes import load_data,load_features,vectors,local_targets,best_threshold

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); args=ap.parse_args()
    run=ROOT/f'data/news_{args.model}'; dest=ROOT/f'results/news_{args.model}/checkpoints'
    data=load_data(run); tr=data['train']; vr=data['val']; train=load_features(run,tr); val=load_features(run,vr)
    anns={a['id']:a for a in readl(run/'local_annotations.jsonl') if a['id'] in {r['id'] for r in tr}}
    zz,mm=local_targets(train,tr,anns); tx=vectors(train,'lookback'); vx=vectors(val,'lookback'); vy=np.array([r['label'] for r in vr])
    for fraction in [.1,.25,.5,1.]:
        for seed in [42,43,44]:
            name=f'few_local_{int(fraction*100)}'; path=dest/f'{name}_{seed}.pkl'
            if path.exists(): continue
            rng=np.random.default_rng(seed); chosen=[]
            for label in [0,1]:
                pool=[i for i,r in enumerate(tr) if r['label']==label]
                chosen.extend(rng.choice(pool,max(1,round(len(pool)*fraction)),replace=False).tolist())
            xx=np.concatenate([tx[i][mm[i]] for i in chosen]); yy=np.concatenate([zz[i][mm[i]] for i in chosen]); best=None
            for c in [.01,.1,1.]:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=700,class_weight='balanced',solver='liblinear',random_state=seed))
                model.fit(xx,yy); scores=[model.predict_proba(x)[:,1] for x in vx]
                p=np.array([np.sort(s)[-(int(len(s)*.1)+1):].mean() for s in scores]); score=roc_auc_score(vy,p)
                if best is None or score>best['val_auc']:
                    best={'method':name,'kind':'lookback','layer':0,'model':model,'val_auc':score,'threshold':best_threshold(vy,p),'seed':seed,
                        'supervision':f'local labels for {len(chosen)}/{len(tr)} training responses; validation uses bag labels only',
                        'C':c,'training_local_ids':[tr[i]['id'] for i in chosen],'n_labeled_tokens':len(yy),'n_error_tokens':int(yy.sum()),
                        'exploratory':'Added after small-news full-supervision diagnostics; all hyperparameter selection uses validation response labels only'}
            path.write_bytes(pickle.dumps(best)); print(name,seed,len(chosen),best['val_auc'],flush=True)

if __name__=='__main__': main()
