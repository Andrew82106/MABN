"""Bounded supervised controls; fresh confirmation data never read during fitting."""
import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import ROOT,readl
from probes import HaMI,load_data,load_features,vectors,local_targets,net_predict,best_threshold

def main():
    torch.set_num_threads(6)
    run=ROOT/'data/news_large'; dest=ROOT/'results/news_large/checkpoints'
    data=load_data(run); tr,vr=data['train'],data['val']
    tf,vf=load_features(run,tr),load_features(run,vr)
    anns={a['id']:a for a in readl(run/'local_annotations.jsonl')}
    tz,tm=local_targets(tf,tr,anns); vz,vm=local_targets(vf,vr,anns)
    tx,vx=vectors(tf,'lookback'),vectors(vf,'lookback')
    scaler=StandardScaler().fit(np.concatenate(tx))
    tx=[scaler.transform(x).astype(np.float32) for x in tx]
    vx=[scaler.transform(x).astype(np.float32) for x in vx]
    ty=np.array([r['label'] for r in tr]); vy=np.array([r['label'] for r in vr])
    clean=np.flatnonzero(ty==0); errors=np.flatnonzero(ty==1)
    for alpha in [0.,.5]:
        method='local_mlp_alarm' if alpha else 'local_mlp'
        for seed in [42,43,44]:
            path=dest/f'{method}_{seed}.pt'
            if path.exists(): continue
            torch.manual_seed(seed); rng=np.random.default_rng(seed)
            net=HaMI(tx[0].shape[1],16).net.cuda()
            opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0005)
            best=(-1.,-1.); history=[]
            for step in range(1,301):
                ix=np.r_[rng.choice(clean,16,replace=len(clean)<16),rng.choice(errors,16,replace=len(errors)<16)]
                lengths=[len(tx[i]) for i in ix]
                net.train(); ss=net(torch.as_tensor(np.concatenate([tx[i] for i in ix]),device='cuda')).flatten()
                yy=torch.as_tensor(np.concatenate([tz[i] for i in ix]),device='cuda',dtype=torch.float32)
                mask=torch.as_tensor(np.concatenate([tm[i] for i in ix]),device='cuda')
                pos=mask&(yy==1); neg=mask&(yy==0)
                loss=.5*torch.nn.functional.binary_cross_entropy(ss[pos],yy[pos])+.5*torch.nn.functional.binary_cross_entropy(ss[neg],yy[neg])
                maxima=torch.stack([s.max() for s in ss.split(lengths)[:16]])
                loss=loss+alpha*torch.nn.functional.binary_cross_entropy(maxima,torch.zeros_like(maxima))
                opt.zero_grad(); loss.backward(); opt.step()
                if step%20: continue
                vp,vs=net_predict(net,vx); mx=np.array([s.max() for s in vs])
                threshold=float(np.quantile(mx[vy==0],.95,method='higher'))
                recall=float((mx[vy==1]>threshold).mean())
                local_auc=float(roc_auc_score(np.concatenate([z[m] for z,m in zip(vz,vm)]),np.concatenate([s[m] for s,m in zip(vs,vm)])))
                key=(recall,local_auc); history.append({'step':step,'validation_error_recall':recall,'validation_global_token_auc':local_auc})
                if key>best:
                    best=key
                    cp={'method':method,'kind':'lookback','seed':seed,'scaler':scaler,'n_features':tx[0].shape[1],
                        'state':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},'step':step,
                        'threshold':best_threshold(vy,vp),'alert_threshold':threshold,'val_auc':float(roc_auc_score(vy,vp)),
                        'alpha':alpha,'selection':'validation recall at <=5% clean-answer false alarm; tie global token AUROC',
                        'supervision':'training token labels, validation bag/local labels; supervised exploratory control',
                        'training_ids':[r['id'] for r in tr],'validation_ids':[r['id'] for r in vr]}
            cp['history']=history; torch.save(cp,path)
            print(method,seed,best,'step',cp['step'],flush=True)
    (ROOT/'results/alarm_control_protocol.json').write_text(json.dumps({'alpha':[0,.5],'seeds':[42,43,44],
        'steps':300,'features':'7B attention ratios','local_loss':'balanced positive/negative token BCE',
        'extra_loss':'BCE of maximum risk in wholly clean training answers',
        'selection':'validation answer recall at <=5% clean-answer false alarm, tie global token AUROC',
        'status':'exploratory extension after original test results; fresh source-disjoint confirmation reserved',
        'novelty_claim':False},indent=2),encoding='utf8')

if __name__=='__main__': main()
