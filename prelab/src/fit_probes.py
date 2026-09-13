"""Train only on bag labels. This module never opens span annotations or test data."""
import copy
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prepare_data import ROOT,read_jsonl

sys.path.insert(0,str(ROOT/'references/HaMI'))
from hami import HaMI,MIL_loss
CFG=json.loads((ROOT/'configs/experiment.json').read_text())

def load_split(split):
    rows=read_jsonl(ROOT/f'data/processed/{split}.jsonl')
    feats=[dict(np.load(ROOT/f'data/features/{x["id"]}.npz')) for x in rows]
    return rows,feats,np.array([x['label'] for x in rows])

def pool(scores,method):
    if method=='mil_mean': return scores.mean()
    if method=='mil_max': return scores.max()
    k=min(len(scores),int(len(scores)*CFG['k_ratio'])+1)
    return scores.topk(k).values.mean()

@torch.no_grad()
def network_scores(net,feats,layer_index,method,device='cuda'):
    net.eval(); bags=[]; tokens=[]
    for f in feats:
        x=torch.as_tensor(f['hidden'][:,layer_index,:],dtype=torch.float32,device=device)
        s=net(x).flatten(); tokens.append(s.cpu().numpy()); bags.append(pool(s,method).item())
    return np.array(bags),tokens

def threshold(y,s):
    candidates=np.r_[np.unique(s),np.max(s)+1e-6]
    return float(max(candidates,key=lambda t:f1_score(y,s>=t,zero_division=0)))

def train_net(method,li,seed,trainf,y,valf,vy):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    hami=HaMI(n_features=896,batch_size=CFG['bags_per_class']).cuda()
    net=hami.net
    opt=torch.optim.Adam(net.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    data=[torch.as_tensor(f['hidden'][:,li,:],device='cuda',dtype=torch.float32) for f in trainf]
    ni=np.flatnonzero(y==0); pi=np.flatnonzero(y==1); rng=np.random.default_rng(seed)
    best=-1; state=None; history=[]
    for step in range(1,CFG['fit_steps']+1):
        neg=rng.choice(ni,CFG['bags_per_class'],replace=False); pos=rng.choice(pi,CFG['bags_per_class'],replace=False)
        batch=[data[i] for i in np.r_[neg,pos]]; lengths=[len(x) for x in batch]
        net.train(); scores=net(torch.cat(batch)).flatten()
        if method=='hami_ori':
            loss=MIL_loss(scores,lengths[:len(neg)],lengths[len(neg):],'cuda',CFG['k_ratio'])
        else:
            chunks=scores.split(lengths)
            bag_scores=torch.stack([pool(x,method) for x in chunks])
            labels=torch.tensor([0.]*len(neg)+[1.]*len(pos),device='cuda')
            loss=torch.nn.functional.binary_cross_entropy(bag_scores,labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%CFG['validate_every']==0:
            vs,_=network_scores(net,valf,li,method)
            auc=roc_auc_score(vy,vs); history.append({'step':step,'loss':loss.item(),'val_auc':auc})
            if auc>best:
                best=auc; state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; best_step=step
    net.load_state_dict(state); vs,_=network_scores(net,valf,li,method)
    return {'state':state,'layer_index':li,'layer':CFG['layers'][li],'val_auc':best,'step':best_step,
            'threshold':threshold(vy,vs),'history':history,'method':method,'seed':seed}

def main():
    torch.set_num_threads(6)
    dest=ROOT/'results/checkpoints'; dest.mkdir(parents=True,exist_ok=True)
    tr,tf,y=load_split('train'); va,vf,vy=load_split('val')
    assert not ({r['group'] for r in tr}&{r['group'] for r in va})
    manifest=[]; begin=time.time()
    for method in ['last_linear','mean_linear']:
        path=dest/f'{method}.pkl'
        if path.exists():
            saved=pickle.loads(path.read_bytes()); manifest.append({k:v for k,v in saved.items() if k!='model'}); continue
        best=None
        for li,layer in enumerate(CFG['layers']):
            def vectors(fs):
                return np.stack([f['hidden'][-1,li,:] if method=='last_linear' else f['hidden'][:,li,:].astype(np.float32).mean(axis=0) for f in fs])
            a,b=vectors(tf),vectors(vf)
            for c in [0.01,0.1,1.0]:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,class_weight='balanced',random_state=42))
                model.fit(a,y); vs=model.predict_proba(b)[:,1]; auc=roc_auc_score(vy,vs)
                if best is None or auc>best['val_auc']:
                    best={'method':method,'model':model,'layer_index':li,'layer':layer,'C':c,'val_auc':auc,'threshold':threshold(vy,vs)}
        path.write_bytes(pickle.dumps(best)); manifest.append({k:v for k,v in best.items() if k!='model'})
        print(method,manifest[-1],flush=True)
    # Text-only and length baselines reveal surface/length shortcuts.
    for method in ['response_length','tfidf_text']:
        if method=='response_length':
            a=np.array([[x['n_tokens']] for x in tr]); b=np.array([[x['n_tokens']] for x in va])
            model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced'))
        else:
            a=[x['response'] for x in tr]; b=[x['response'] for x in va]
            model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000),LogisticRegression(C=1,class_weight='balanced',max_iter=1000))
        model.fit(a,y); vs=model.predict_proba(b)[:,1]
        state={'method':method,'model':model,'val_auc':roc_auc_score(vy,vs),'threshold':threshold(vy,vs)}
        (dest/f'{method}.pkl').write_bytes(pickle.dumps(state)); manifest.append({k:v for k,v in state.items() if k!='model'})
    for method in ['mil_mean','mil_max','mil_topk','hami_ori','shuffled_label_mil_topk']:
        seeds=CFG['probe_seeds'] if method!='shuffled_label_mil_topk' else [42]
        for seed in seeds:
            path=dest/f'{method}_{seed}.pt'
            if path.exists():
                best=torch.load(path,weights_only=False); manifest.append({k:v for k,v in best.items() if k not in ['state','history']}); continue
            best=None
            for li in range(len(CFG['layers'])):
                sy=y.copy(); svy=vy.copy()
                if method=='shuffled_label_mil_topk':
                    rng=np.random.default_rng(12345); rng.shuffle(sy); rng.shuffle(svy)
                actual='mil_topk' if method=='shuffled_label_mil_topk' else method
                fit=train_net(actual,li,seed,tf,sy,vf,svy)
                print(f'{method} seed={seed} layer={CFG["layers"][li]} val={fit["val_auc"]:.4f}',flush=True)
                if best is None or fit['val_auc']>best['val_auc']: best=fit
            best['method']=method
            torch.save(best,path); manifest.append({k:v for k,v in best.items() if k not in ['state','history']})
    (ROOT/'results/fit_manifest.json').write_text(json.dumps({'models':manifest,'seconds':time.time()-begin,'span_labels_read':False},indent=2),encoding='utf8')
    print('FIT COMPLETE',time.time()-begin,flush=True)

if __name__=='__main__': main()
