"""Common fitting, with explicit separation of bag-supervised and local-supervised paths."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from common import ROOT,PRELAB,readl

sys.path.insert(0,str(PRELAB/'references/HaMI'))
from hami import HaMI,MIL_loss
CFG=json.loads((ROOT/'configs/protocol.json').read_text())

def load_data(run):
    rows=[r for r in readl(run/'labeled.jsonl') if r['label'] is not None]
    data={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['group'] for r in data[a]}&{r['group'] for r in data[b]}
    return data

def load_features(run,rows):
    feats=[dict(np.load(run/f'features/{r["id"]}.npz')) for r in rows]
    for f,r in zip(feats,rows):
        if 'qas' in r:
            from label_outputs import parse_answers
            # Boundaries come from output syntax and question order, never gold answers.
            answers=parse_answers(r['response'],r['qas']); index=np.full(len(f['offsets']),-1,dtype=np.int32)
            for j,(_,a,b) in enumerate(answers):
                index[[i for i,(lo,hi) in enumerate(f['offsets']) if lo<b and hi>a]]=j
            assert all((index==j).any() for j in range(3)); f['fact_index']=index
    return feats
def vectors(feats,kind,layer=0):
    if kind.startswith('fact_'):
        raw=vectors(feats,kind[5:],layer)
        return [np.stack([x[f['fact_index']==j].mean(0) for j in range(3)]) for f,x in zip(feats,raw)]
    if kind in ['before','after']: return [f[kind][:,layer,:].astype(np.float32) for f in feats]
    if kind=='uncertainty': return [np.column_stack([f['nll'],f['entropy'],f['margin']]) for f in feats]
    if kind=='support_lookback': return [np.column_stack([f['support'],f['lookback']]).astype(np.float32) for f in feats]
    return [f[kind].astype(np.float32) for f in feats]

def topk(s): return s.topk(min(len(s),int(len(s)*.1)+1)).values.mean()
def best_threshold(y,p):
    return float(max(np.r_[np.unique(p),max(p)+1e-5],key=lambda t:f1_score(y,p>=t,zero_division=0)))

@torch.no_grad()
def net_predict(net,xx):
    net.eval(); lengths=[len(x) for x in xx]; scores=[]
    for start in range(0,len(xx),32):
        batch=xx[start:start+32]
        s=net(torch.as_tensor(np.concatenate(batch),device=next(net.parameters()).device,dtype=torch.float32)).flatten()
        scores.extend([x.cpu().numpy() for x in s.split([len(x) for x in batch])])
    return np.array([np.sort(s)[-(int(len(s)*.1)+1):].mean() for s in scores]),scores

def fit_network(train,ty,val,vy,seed,loss_name):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    net=HaMI(train[0].shape[1],16).net.cuda()
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0005)
    neg=np.flatnonzero(ty==0); pos=np.flatnonzero(ty==1)
    assert len(neg)>=5 and len(pos)>=5,('insufficient training class',len(neg),len(pos))
    best=-1; history=[]
    for step in range(1,CFG['probe_steps']+1):
        ni=rng.choice(neg,16,replace=len(neg)<16); pi=rng.choice(pos,16,replace=len(pos)<16)
        xx=[train[i] for i in np.r_[ni,pi]]; lengths=[len(x) for x in xx]
        net.train(); ss=net(torch.as_tensor(np.concatenate(xx),device='cuda',dtype=torch.float32)).flatten()
        if loss_name=='hami': loss=MIL_loss(ss,lengths[:16],lengths[16:],'cuda',.1)
        else:
            pp=torch.stack([topk(s) for s in ss.split(lengths)])
            loss=torch.nn.functional.binary_cross_entropy(pp,torch.tensor([0.]*16+[1.]*16,device='cuda'))
        opt.zero_grad(); loss.backward(); opt.step()
        if step%CFG['validation_interval']==0:
            vp,_=net_predict(net,val); auc=roc_auc_score(vy,vp)
            history.append({'step':step,'loss':loss.item(),'val_auc':auc})
            if auc>best:
                best=auc; state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; bs=step; threshold=best_threshold(vy,vp)
    return {'state':state,'val_auc':best,'step':bs,'threshold':threshold,'history':history}

def local_targets(feats,rows,anns):
    zz=[]; masks=[]
    for f,r in zip(feats,rows):
        a=anns[r['id']]
        zz.append(np.array([int(any(lo<s['end'] and hi>s['start'] for s in a['spans'])) for lo,hi in f['offsets']]))
        masks.append(np.array([not any(lo<s['end'] and hi>s['start'] for s in a.get('ignore',[]))
            and (not a.get('answers') or any(lo<s['end'] and hi>s['start'] and s['status'] in ['correct','candidate_error'] for s in a['answers']))
            for lo,hi in f['offsets']]))
    return zz,masks

def fit_local_candidate(args):
    from threadpoolctl import threadpool_limits
    xx,yy,vx,vz,vm,vy,c=args
    with threadpool_limits(limits=1):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=700,class_weight='balanced',solver='liblinear',random_state=42))
        model.fit(xx,yy); pred=[model.predict_proba(x)[:,1] for x in vx]
        score=roc_auc_score(np.concatenate([z[m] for z,m in zip(vz,vm)]),np.concatenate([p[m] for p,m in zip(pred,vm)]))
        bags=np.array([np.sort(s)[-(int(len(s)*.1)+1):].mean() for s in pred])
    return {'model':model,'val_auc':score,'threshold':best_threshold(vy,bags),'C':c}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--model',required=True)
    ap.add_argument('--supervised',action='store_true'); ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--supervised-kinds',nargs='+',choices=['before','after','lookback'],default=['before','after','lookback']); args=ap.parse_args()
    torch.set_num_threads(6); run=ROOT/f'data/{args.task}_{args.model}'; dest=ROOT/f'results/{args.task}_{args.model}/checkpoints'; dest.mkdir(parents=True,exist_ok=True)
    rows=load_data(run); tr,vr=rows['train'],rows['val']
    train=load_features(run,tr); val=load_features(run,vr)
    ty=np.array([r['label'] for r in tr]); vy=np.array([r['label'] for r in vr]); manifest=[]
    if args.supervised:
        assert args.task!='trivia'
        anns={a['id']:a for a in readl(run/'local_annotations.jsonl')}
        tz,tm=local_targets(train,tr,anns); vz,vm=local_targets(val,vr,anns)
        for kind in args.supervised_kinds:
            if (dest/f'supervised_{kind}.pkl').exists(): continue
            best=None
            for li in (range(4) if kind in ['before','after'] else [0]):
                tx=vectors(train,kind,li); vx=vectors(val,kind,li)
                xx=np.concatenate([x[m] for x,m in zip(tx,tm)]); yy=np.concatenate([z[m] for z,m in zip(tz,tm)])
                rng=np.random.default_rng(42); idx=np.arange(len(yy)); rng.shuffle(idx); idx=idx[:50000]
                xx=xx[idx]; yy=yy[idx]
                jobs=[(xx,yy,vx,vz,vm,vy,c) for c in [.01,.1,1.]]
                if args.workers>1:
                    with ProcessPoolExecutor(max_workers=args.workers) as pool: fitted=list(pool.map(fit_local_candidate,jobs))
                else: fitted=[fit_local_candidate(job) for job in jobs]
                for fit in fitted:
                    print('LOCAL_CANDIDATE',kind,li,fit['C'],fit['val_auc'],flush=True)
                    if best is None or fit['val_auc']>best['val_auc']:
                        best={**fit,'method':'supervised_'+kind,'kind':kind,'layer':li,'supervision':'token labels; diagnostic, not weak supervision'}
            (dest/f'supervised_{kind}.pkl').write_bytes(pickle.dumps(best)); print('SUPERVISED',kind,best['val_auc'],flush=True)
        return
    # This branch never opens local annotations.
    for reduction in ['last','mean']:
        path=dest/f'{reduction}_linear.pkl'
        if path.exists(): continue
        best=None
        for li in range(4):
            tx=vectors(train,'before',li); vx=vectors(val,'before',li)
            a=np.stack([x[-1] if reduction=='last' else x.mean(0) for x in tx]); b=np.stack([x[-1] if reduction=='last' else x.mean(0) for x in vx])
            for c in [.01,.1,1.]:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,class_weight='balanced',solver='liblinear',random_state=42))
                model.fit(a,ty); p=model.predict_proba(b)[:,1]; score=roc_auc_score(vy,p)
                if best is None or score>best['val_auc']: best={'method':reduction+'_linear','kind':'before','layer':li,'model':model,'val_auc':score,'threshold':best_threshold(vy,p),'reduction':reduction}
        path.write_bytes(pickle.dumps(best)); print(reduction,best['val_auc'],flush=True)
    for kind in ['length','text']:
        if kind=='length':
            a=np.array([[len(f['nll'])] for f in train]); b=np.array([[len(f['nll'])] for f in val]); model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced'))
        else:
            a=[r['response'] for r in tr]; b=[r['response'] for r in vr]; model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000),LogisticRegression(class_weight='balanced',max_iter=700))
        model.fit(a,ty); p=model.predict_proba(b)[:,1]
        (dest/f'{kind}.pkl').write_bytes(pickle.dumps({'method':kind,'kind':kind,'model':model,'val_auc':roc_auc_score(vy,p),'threshold':best_threshold(vy,p)}))
    kinds=['before','after','uncertainty']+(['lookback','support','support_lookback'] if args.task!='trivia' else [])
    if args.task=='rag': kinds+=['fact_after','fact_support_lookback']
    for kind in kinds:
        for seed in CFG['seeds']:
            path=dest/f'{kind}_{seed}.pt'
            if path.exists(): continue
            best=None
            for li in (range(4) if kind in ['before','after','fact_after'] else [0]):
                tx=vectors(train,kind,li); vx=vectors(val,kind,li); scaler=None
                if kind not in ['before','after','fact_after']:
                    scaler=StandardScaler().fit(np.concatenate(tx)); tx=[scaler.transform(x).astype(np.float32) for x in tx]; vx=[scaler.transform(x).astype(np.float32) for x in vx]
                fit=fit_network(tx,ty,vx,vy,seed,'hami' if kind in ['before','after','fact_after'] else 'bce')
                print(kind,seed,li,fit['val_auc'],flush=True)
                if best is None or fit['val_auc']>best['val_auc']:
                    best={**fit,'kind':kind,'layer':li,'scaler':scaler,'seed':seed,'method':kind,'supervision':'bag labels only','n_features':tx[0].shape[1]}
            torch.save(best,path)
    # Fixed shuffle diagnostic, including validation labels, never choosing on true validation.
    path=dest/'shuffled_support.pt'
    kind='support' if args.task!='trivia' else 'uncertainty'
    tx=vectors(train,kind); vx=vectors(val,kind); scaler=StandardScaler().fit(np.concatenate(tx)); tx=[scaler.transform(x).astype(np.float32) for x in tx]; vx=[scaler.transform(x).astype(np.float32) for x in vx]
    sy=ty.copy(); sv=vy.copy(); rng=np.random.default_rng(1701); rng.shuffle(sy); rng.shuffle(sv)
    fit=fit_network(tx,sy,vx,sv,42,'bce'); torch.save({**fit,'kind':kind,'layer':0,'scaler':scaler,'method':'shuffled','n_features':tx[0].shape[1]},path)
    print('FIT COMPLETE',args.task,args.model,flush=True)

if __name__=='__main__': main()
