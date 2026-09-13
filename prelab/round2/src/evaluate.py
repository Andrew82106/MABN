"""Held-out evaluation. Local labels never enter weak fitting or its selection."""
import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from common import ROOT,readl,writel,sha,norm
from probes import HaMI,load_data,load_features,vectors,net_predict,best_threshold,local_targets
DEVICE='cuda'

def auc(y,p): return float(roc_auc_score(y,p)) if len(set(y))==2 else None
def average(xs):
    xs=[x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None
def aggregate(scores): return np.array([np.sort(s)[-(int(len(s)*.1)+1):].mean() for s in scores])

def copy_scores(rows,feats,task):
    result=[]
    for r,f in zip(rows,feats):
        evidence=norm(r['evidence']); scores=np.zeros(len(f['offsets']),dtype=np.float32)
        matches=list(re.finditer(r'"(?:[^"\\]|\\.)*"',r['response'])) if task=='rag' else list(re.finditer(r'\b[\w-]+\b',r['response']))
        for match in matches:
            text=match.group()[1:-1] if task=='rag' else match.group()
            value=float(bool(norm(text)) and (' '+norm(text)+' ') not in (' '+evidence+' '))
            for i,(lo,hi) in enumerate(f['offsets']):
                if lo<match.end() and hi>match.start(): scores[i]=max(scores[i],value)
        result.append(scores)
    return result

def predict(checkpoint,feats,rows):
    kind=checkpoint['kind']
    if kind in ['length','text']:
        x=np.array([[len(f['nll'])] for f in feats]) if kind=='length' else [r['response'] for r in rows]
        return checkpoint['model'].predict_proba(x)[:,1],None
    xx=vectors(feats,kind,checkpoint.get('layer',0))
    if 'state' in checkpoint:
        if checkpoint.get('scaler') is not None: xx=[checkpoint['scaler'].transform(x).astype(np.float32) for x in xx]
        net=HaMI(checkpoint['n_features'],16).net.to(DEVICE); net.load_state_dict(checkpoint['state'])
        bags,scores=net_predict(net,xx)
        if kind.startswith('fact_'):
            expanded=[]
            for f,ss in zip(feats,scores):
                risk=np.zeros(len(f['offsets']),dtype=np.float32)
                for j in range(3): risk[f['fact_index']==j]=ss[j]
                expanded.append(risk)
            scores=expanded
        return bags,scores
    model=checkpoint['model']; scores=[model.predict_proba(x)[:,1] for x in xx]
    if 'reduction' in checkpoint:
        x=np.stack([z[-1] if checkpoint['reduction']=='last' else z.mean(0) for z in xx])
        return model.predict_proba(x)[:,1],scores
    return aggregate(scores),scores

def row_local(scores,f,ann,y,mask):
    ss=np.asarray(scores)[mask]; yy=y[mask]
    result={'local_auc':auc(yy,ss)}
    if yy.sum()>0:
        k=max(1,int(np.ceil(len(ss)*.1)))
        # Deterministic random tie breaking avoids granting constant scores a position prior.
        rng=np.random.default_rng(2801); order=np.lexsort((rng.random(len(ss)),-ss))[:k]
        result.update(local_ap=float(average_precision_score(yy,ss)),top10_precision=float(yy[order].mean()),
                      top10_recall=float(yy[order].sum()/yy.sum()),error_coverage=float(yy.mean()))
    if ann.get('answers'):
        fact_y=[]; fact_s=[]
        for a in ann['answers']:
            if a['status'] not in ['correct','candidate_error']: continue
            take=np.array([lo<a['end'] and hi>a['start'] for lo,hi in f['offsets']])
            if not take.any(): continue
            fact_y.append(int(a['status']=='candidate_error')); fact_s.append(float(np.mean(np.asarray(scores)[take])))
        result['fact_auc']=auc(fact_y,fact_s)
        if result['fact_auc'] is not None:
            high=np.flatnonzero(np.asarray(fact_s)==max(fact_s))
            result['fact_top1']=float(np.mean(np.array(fact_y)[high])); result['fact_chance']=float(np.mean(fact_y))
    return result

def measure(rows,feats,ann,p,scores,threshold):
    y=np.array([r['label'] for r in rows]); out={'n':len(y),'errors':int(y.sum()),'bag_auc':auc(y,p),
        'bag_ap':float(average_precision_score(y,p)),'bag_f1':float(f1_score(y,p>=threshold,zero_division=0))}
    local=[]
    if ann is not None and scores is not None:
        zz,mm=local_targets(feats,rows,ann)
        all_y=np.concatenate([z[m] for z,m in zip(zz,mm)]); all_s=np.concatenate([s[m] for s,m in zip(scores,mm)])
        kk=max(1,int(np.ceil(len(all_y)*.1))); order=np.lexsort((np.random.default_rng(2801).random(len(all_s)),-all_s))[:kk]
        out.update(global_token_auc=auc(all_y,all_s),global_top10_precision=float(all_y[order].mean()),
                   global_top10_recall=float(all_y[order].sum()/max(1,all_y.sum())),global_error_coverage=float(all_y.mean()))
        for r,f,s,z,m in zip(rows,feats,scores,zz,mm):
            v=row_local(s,f,ann[r['id']],z,m); v['id']=r['id']; local.append(v)
        for key in ['local_auc','local_ap','top10_precision','top10_recall','error_coverage','fact_auc','fact_top1','fact_chance']:
            out[key]=average([v.get(key) for v in local])
        out['n_local_mixed']=sum(v['local_auc'] is not None for v in local)
        out['n_fact_mixed']=sum(v.get('fact_auc') is not None for v in local)
    return out,local

def calibrated_alert(rows,feats,ann,scores,vr,vf,vs):
    if scores is None or ann is None: return {}
    def maxima(ff,ss):
        return np.array([float(np.max(s[f['fact_index']>=0] if 'fact_index' in f else s)) for f,s in zip(ff,ss)])
    vm=maxima(vf,vs); clean=np.array([r['label']==0 for r in vr])
    threshold=float(np.quantile(vm[clean],.95,method='higher'))
    tm=maxima(feats,scores); y=np.array([r['label'] for r in rows]); alarm=tm>threshold
    zz,mm=local_targets(feats,rows,ann); truth=np.concatenate([z[m] for z,m in zip(zz,mm)]); flags=np.concatenate([s[m]>threshold for s,m in zip(scores,mm)])
    return {'alert_threshold':threshold,'validation_clean_answer_alarm':float(np.mean(vm[clean]>threshold)),
        'test_clean_answer_alarm':float(np.mean(alarm[y==0])),'test_error_answer_recall':float(np.mean(alarm[y==1])),
        'alert_token_precision':float(truth[flags].mean()) if flags.any() else 0.,
        'alert_error_token_recall':float(truth[flags].sum()/max(1,truth.sum()))}

def bootstrap(rows,p,local,base=None,base_local=None,n=1000,baseline_name='hami'):
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['group']].append(i)
    group_list=list(groups.values()); y=np.array([r['label'] for r in rows]); rng=np.random.default_rng(7192)
    loc=np.array([r.get('local_auc') if r.get('local_auc') is not None else np.nan for r in local]) if local else None
    bloc=np.array([r.get('local_auc') if r.get('local_auc') is not None else np.nan for r in base_local]) if base_local else None
    facts=np.array([r.get('fact_auc') if r.get('fact_auc') is not None else np.nan for r in local]) if local else None
    bfacts=np.array([r.get('fact_auc') if r.get('fact_auc') is not None else np.nan for r in base_local]) if base_local else None
    draws=defaultdict(list)
    for _ in range(n):
        ix=np.concatenate([group_list[j] for j in rng.integers(0,len(group_list),len(group_list))])
        if len(set(y[ix]))==2:
            value=auc(y[ix],p[ix]); draws['bag_auc'].append(value)
            if base is not None: draws['bag_auc_difference_vs_'+baseline_name].append(value-auc(y[ix],base[ix]))
        if loc is not None and np.isfinite(loc[ix]).any():
            draws['local_auc'].append(float(np.nanmean(loc[ix])))
            if bloc is not None: draws['local_auc_difference_vs_'+baseline_name].append(float(np.nanmean(loc[ix]-bloc[ix])))
        if facts is not None and np.isfinite(facts[ix]).any():
            draws['fact_auc'].append(float(np.nanmean(facts[ix])))
            if bfacts is not None: draws['fact_auc_difference_vs_'+baseline_name].append(float(np.nanmean(facts[ix]-bfacts[ix])))
    return {key:np.quantile(vals,[.025,.975]).tolist() for key,vals in draws.items() if vals}

def main():
    global DEVICE
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--model',required=True); ap.add_argument('--cpu',action='store_true'); args=ap.parse_args()
    DEVICE='cpu' if args.cpu else 'cuda'
    torch.set_num_threads(6); run=ROOT/f'data/{args.task}_{args.model}'; dest=ROOT/f'results/{args.task}_{args.model}'
    data=load_data(run); vr,tr=data['val'],data['test']; vf=load_features(run,vr); tf=load_features(run,tr)
    anns=None if args.task=='trivia' else {a['id']:a for a in readl(run/'local_annotations.jsonl')}
    metrics=[]; predictions={}; local_results={}
    for path in sorted((dest/'checkpoints').glob('*')):
        cp=torch.load(path,weights_only=False,map_location='cpu') if path.suffix=='.pt' else pickle.loads(path.read_bytes())
        p,s=predict(cp,tf,tr); m,loc=measure(tr,tf,anns,p,s,cp['threshold'])
        if s is not None and anns is not None:
            _,vs=predict(cp,vf,vr); m.update(calibrated_alert(tr,tf,anns,s,vr,vf,vs))
        m.update(name=path.stem,method=cp['method'],seed=cp.get('seed'),val_auc=float(cp['val_auc']),
                 supervision=cp.get('supervision','bag labels only'),layer_index=cp.get('layer'),threshold=float(cp['threshold']))
        metrics.append(m); predictions[path.stem]=(p,s); local_results[path.stem]=loc
        print(path.stem,{k:round(v,3) for k,v in m.items() if k in ['bag_auc','local_auc','fact_auc'] and v is not None},flush=True)
    for name in ['nll_topk','nll_mean']+(['support_delta','source_copy'] if args.task!='trivia' else []):
        if name=='source_copy': ss=copy_scores(tr,tf,args.task); vs=copy_scores(vr,vf,args.task)
        else:
            ss=[f['support'][:,2] if name=='support_delta' else f['nll'] for f in tf]
            vs=[f['support'][:,2] if name=='support_delta' else f['nll'] for f in vf]
        p=np.array([s.mean() for s in ss]) if name=='nll_mean' else aggregate(ss)
        vp=np.array([s.mean() for s in vs]) if name=='nll_mean' else aggregate(vs)
        vy=[r['label'] for r in vr]; threshold=best_threshold(vy,vp)
        m,loc=measure(tr,tf,anns,p,ss,threshold); m.update(name=name,method=name,seed=None,val_auc=auc(vy,vp),supervision='no fitted weights; validation threshold',threshold=threshold)
        m.update(calibrated_alert(tr,tf,anns,ss,vr,vf,vs))
        metrics.append(m); predictions[name]=(p,ss); local_results[name]=loc
    grouped=defaultdict(list)
    for m in metrics: grouped[m['method']].append(m)
    summary=[]
    for method,ms in grouped.items():
        item={'method':method,'runs':len(ms),'supervision':ms[0]['supervision']}
        for key in ['val_auc','bag_auc','bag_ap','bag_f1','local_auc','top10_precision','top10_recall','error_coverage','fact_auc','fact_top1','fact_chance',
                    'global_token_auc','global_top10_precision','global_top10_recall','global_error_coverage','test_clean_answer_alarm','test_error_answer_recall','alert_token_precision','alert_error_token_recall']:
            item[key]=average([m.get(key) for m in ms])
        summary.append(item)
    # Candidate family chosen solely on validation, then average the prespecified three seeds.
    candidates=[s for s in summary if s['method'] in ['before','after','uncertainty','lookback','support','support_lookback','fact_after','fact_support_lookback']]
    selected=max(candidates,key=lambda x:x['val_auc'])['method']
    selected_names=[m['name'] for m in metrics if m['method']==selected]
    hami_names=[m['name'] for m in metrics if m['method']=='before']
    def ensemble(names):
        p=np.mean([predictions[n][0] for n in names],axis=0)
        scores=[np.mean([predictions[n][1][i] for n in names],axis=0) for i in range(len(tr))]
        _,loc=measure(tr,tf,anns,p,scores,.5)
        return p,scores,loc
    sp,ss,sl=ensemble(selected_names); hp,hs,hl=ensemble(hami_names)
    ci={'selected_by_validation':selected,'selection_input':'mean validation response AUROC; no test/local selection',
        'selected_names':selected_names,'selected_ensemble':bootstrap(tr,sp,sl,hp,hl),
        'hami_before_ensemble':bootstrap(tr,hp,hl),'n_bootstrap':1000,'unit':'source/article/question group'}
    for kind in ['before','after','lookback']:
        name='supervised_'+kind
        if name in predictions:
            names=[m['name'] for m in metrics if m['method']==kind]
            bp,bs,bl=ensemble(names)
            ci[name]=bootstrap(tr,predictions[name][0],local_results[name],bp,bl,baseline_name='same_features_weak')
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (dest/'confidence_intervals.json').write_text(json.dumps(ci,indent=2),encoding='utf8')
    keys=list(dict.fromkeys(k for m in metrics for k in m))
    with (dest/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(metrics)
    out=[]
    for i,r in enumerate(tr):
        out.append({'id':r['id'],'group':r['group'],'label':r['label'],'response':r['response'],
            'offsets':tf[i]['offsets'].tolist(),'predictions':{n:{'bag':float(p[i]),'tokens':s[i].tolist() if s is not None else None} for n,(p,s) in predictions.items()}})
    writel(dest/'predictions.jsonl',out)
    (dest/'evaluation_manifest.json').write_text(json.dumps({'test_n':len(tr),'test_errors':sum(r['label'] for r in tr),
        'labels_sha256':sha(run/'labeled.jsonl'),'annotations_sha256':sha(run/'local_annotations.jsonl'),
        'local_label_scope':'RAG answer-content tokens only; news released spans; Trivia has no localization gold',
        'evaluation_device':DEVICE,
        'linear_token_scores':'naive application of bag-trained classifier, not locally trained'},indent=2),encoding='utf8')
    print('EVALUATION COMPLETE',args.task,args.model,'selected',selected,flush=True)

if __name__=='__main__': main()
