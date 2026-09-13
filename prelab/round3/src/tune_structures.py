"""Validation-only architecture/parameter search followed by fixed-seed refits."""
import json,pickle,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from shared import ROOT
from networks import Probe,batch,predict

def configs():
    out=[]
    for feature in ['attention','hybrid']:
        base={'feature':feature,'architecture':'mlp','width':64,'depth':1,'dropout':.1,'lr':.001,'weight_decay':.001,'loss':'balanced'}
        for changes in [dict(width=64),dict(width=256),dict(width=256,depth=2),dict(width=64,lr=.0003),
             dict(width=256,dropout=.3,weight_decay=.01),dict(architecture='conv'),dict(architecture='gru'),
             dict(loss='natural'),dict(loss='ranking')]:out.append({**base,**changes})
    for i,c in enumerate(out):c['id']=f'c{i:02d}'
    return out

def fit(inp,c,seed):
    path=ROOT/f'results/checkpoints/{c["id"]}_{seed}.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tx=inp['x'][c['feature']]['train'];vx=inp['x'][c['feature']]['val'];yy=inp['y']['train'];vy=np.concatenate(inp['y']['val'])
    by=np.array([r['label'] for r in inp['rows']['train']]);clean=np.flatnonzero(by==0);error=np.flatnonzero(by==1)
    net=Probe(tx[0].shape[1],**{k:c[k] for k in ['architecture','width','depth','dropout']}).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);best=-1;history=[];start=time.time()
    for step in range(1,401):
        ix=np.r_[rng.choice(clean,8,replace=False),rng.choice(error,8,replace=False)]
        x,lengths=batch([tx[i] for i in ix],'cuda');y,_=batch([yy[i] for i in ix],'cuda')
        mask=torch.arange(x.shape[1],device='cuda')[None,:]<torch.tensor(lengths,device='cuda')[:,None]
        net.train();logits=net(x);losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
        if c['loss']=='natural':loss=losses[mask].mean()
        else:
            pos=mask&(y==1);neg=mask&(y==0);loss=.5*(losses[pos].mean()+losses[neg].mean())
            if c['loss']=='ranking':
                ranks=[]
                for j in range(len(ix)):
                    pp=logits[j][pos[j]];nn=logits[j][neg[j]]
                    if len(pp) and len(nn):ranks.append(torch.nn.functional.softplus(nn.mean()-pp.mean()+1.))
                if ranks:loss=loss+.1*torch.stack(ranks).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%25:continue
        ps=predict(net,vx);score=float(roc_auc_score(vy,np.concatenate(ps)))
        history.append({'step':step,'loss':float(loss.item()),'val_global_token_auc':score})
        if score>best:
            best=score;cp={'config':c,'seed':seed,'step':step,'val_global_token_auc':score,
                'state':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},'n_features':tx[0].shape[1],
                'parameters':sum(p.numel() for p in net.parameters())}
    cp.update(history=history,seconds=time.time()-start);torch.save(cp,path)
    print(c['id'],c['feature'],c['architecture'],c['loss'],seed,round(best,4),'step',cp['step'],'seconds',round(cp['seconds'],1),flush=True)
    return cp

def main():
    torch.set_num_threads(6);(ROOT/'results/checkpoints').mkdir(exist_ok=True)
    inp=pickle.loads((ROOT/'data/inputs.pkl').read_bytes());grid=configs()
    (ROOT/'configs/structure_grid.json').write_text(json.dumps(grid,indent=2),encoding='utf8')
    # Local supervised linear baseline; all candidates use identical training tokens.
    for feature in ['attention','hybrid']:
        path=ROOT/f'results/checkpoints/linear_{feature}.pkl'
        if path.exists():continue
        xx=np.concatenate(inp['x'][feature]['train']);yy=np.concatenate(inp['y']['train']);vx=np.concatenate(inp['x'][feature]['val']);vy=np.concatenate(inp['y']['val']);best=None
        with threadpool_limits(limits=4):
            for c in [.01,.1,1.]:
                net=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=700,random_state=42).fit(xx,yy)
                score=float(roc_auc_score(vy,net.predict_proba(vx)[:,1]))
                if best is None or score>best['val_global_token_auc']:best={'model':net,'feature':feature,'C':c,'val_global_token_auc':score,'method':'linear_'+feature}
        path.write_bytes(pickle.dumps(best));print('LINEAR',feature,best['C'],best['val_global_token_auc'],flush=True)
    initial=[fit(inp,c,42) for c in grid];winners={}
    for cp in initial:
        c=cp['config'];family=c['feature']+'_'+c['architecture']
        if family not in winners or cp['val_global_token_auc']>winners[family]['val_global_token_auc']:winners[family]=cp
    for cp in winners.values():
        for seed in [43,44]:fit(inp,cp['config'],seed)
    selection={'selection':'family config chosen by seed42 validation global token AUROC; epochs individually validation-selected for all seeds',
        'family_winners':{k:{'config_id':v['config']['id'],'val_global_token_auc':v['val_global_token_auc']} for k,v in winners.items()},
        'overall_seed42_winner':max(initial,key=lambda c:c['val_global_token_auc'])['config']['id'],
        'train_ids':[r['id'] for r in inp['rows']['train']],'validation_ids':[r['id'] for r in inp['rows']['val']],
        'fresh_test_used_for_selection':False}
    (ROOT/'results/structure_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8');print('STRUCTURE SEARCH COMPLETE',flush=True)

if __name__=='__main__':main()
