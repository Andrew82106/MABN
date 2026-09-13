"""Single final fresh-test evaluation after both selection manifests exist."""
import json,pickle
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from shared import ROOT,OLD,readl,writel,features
from networks import Probe,predict
from fusion import load_bundle,scores as fusion_scores

def auc(y,p):return float(roc_auc_score(y,p)) if len(set(y))==2 else None
def topmean(s):return float(np.sort(s)[-(int(len(s)*.1)+1):].mean())

def measure(rows,truth,ss,vr,vs):
    by=np.array([r['label'] for r in rows]);clean=np.array([r['label']==0 for r in vr]);yy=np.concatenate(truth);ps=np.concatenate(ss)
    local=[auc(z,s) for z,s in zip(truth,ss)];vals=[v for v in local if v is not None]
    mx=np.array([max(s) for s in ss]);vm=np.array([max(s) for s in vs]);threshold=float(np.quantile(vm[clean],.95,method='higher'));alarm=mx>threshold
    bag=np.array([topmean(s) for s in ss]);vb=np.array([topmean(s) for s in vs]);bt=float(np.quantile(vb[clean],.95,method='higher'));ba=bag>bt
    flags=ps>threshold
    out={'n':len(rows),'errors':int(by.sum()),'global_token_auc':auc(yy,ps),'global_token_ap':float(average_precision_score(yy,ps)),
        'local_auc':float(np.mean(vals)) if vals else None,'local_n':len(vals),'bag_auc':auc(by,bag),'alert_threshold':threshold,
        'validation_clean_alarm':float((vm[clean]>threshold).mean()),'clean_alarm':float(alarm[by==0].mean()),'error_recall':float(alarm[by==1].mean()),
        'alert_token_precision':float(yy[flags].mean()) if flags.any() else 0.,'alert_error_token_recall':float(yy[flags].sum()/max(1,yy.sum())),
        'topmean_alert_threshold':bt,'topmean_clean_alarm':float(ba[by==0].mean()),'topmean_error_recall':float(ba[by==1].mean())}
    case=[{'id':r['id'],'group':r['group'],'label':r['label'],'local_auc':a,'alarm':bool(b),'topmean_alarm':bool(c),'bag':float(p)} for r,a,b,c,p in zip(rows,local,alarm,ba,bag)]
    return out,case

def bootstrap(cases,comparisons,truth,predictions):
    first=next(iter(cases.values()));n=len(first);rng=np.random.default_rng(190926);draws=defaultdict(lambda:defaultdict(list));groups=list(dict.fromkeys(r['group'] for r in first));ixs=[[i for i,r in enumerate(first) if r['group']==g] for g in groups]
    arrays={k:{'local':np.array([np.nan if r['local_auc'] is None else r['local_auc'] for r in rr]),'y':np.array([r['label'] for r in rr]),'alarm':np.array([r['alarm'] for r in rr]),'bag':np.array([r['bag'] for r in rr])} for k,rr in cases.items()}
    for _ in range(1000):
        ix=np.concatenate([ixs[j] for j in rng.integers(0,len(groups),len(groups))])
        for k,v in arrays.items():
            y=v['y'][ix];draws[k]['local_auc'].append(float(np.nanmean(v['local'][ix])));draws[k]['clean_alarm'].append(float(v['alarm'][ix][y==0].mean()));draws[k]['error_recall'].append(float(v['alarm'][ix][y==1].mean()));draws[k]['bag_auc'].append(auc(y,v['bag'][ix]))
            draws[k]['global_token_auc'].append(auc(np.concatenate([truth[i] for i in ix]),np.concatenate([predictions[k][i] for i in ix])))
    result={'n_bootstrap':1000,'unit':'source group','seed':190926,'intervals':{k:{m:np.quantile(z,[.025,.975]).tolist() for m,z in v.items()} for k,v in draws.items()},'paired_differences':{}}
    for a,b in comparisons:
        if a not in draws or b not in draws:continue
        result['paired_differences'][a+'_minus_'+b]={m:np.quantile(np.array(draws[a][m])-draws[b][m],[.025,.975]).tolist() for m in draws[a]}
    return result

def main():
    torch.set_num_threads(6);dest=ROOT/'results';selection=json.loads((dest/'structure_selection.json').read_text());fusion_selection=json.loads((dest/'fusion_selection.json').read_text())
    inp=pickle.loads((ROOT/'data/inputs.pkl').read_bytes());b=load_bundle();rr=inp['rows']['test'];vr=inp['rows']['val'];truth=inp['y']['test']
    predictions={};validation={};meta={};groups={}
    for feature in ['attention','hybrid']:
        cp=pickle.loads((dest/f'checkpoints/linear_{feature}.pkl').read_bytes());name='linear_'+feature
        predictions[name]=[cp['model'].predict_proba(x)[:,1] for x in inp['x'][feature]['test']];validation[name]=[cp['model'].predict_proba(x)[:,1] for x in inp['x'][feature]['val']];meta[name]={'family':name,'supervision':'token positions','feature':feature}
    for family,w in selection['family_winners'].items():
        names=[]
        for seed in [42,43,44]:
            cp=torch.load(dest/f'checkpoints/{w["config_id"]}_{seed}.pt',map_location='cpu',weights_only=False);c=cp['config'];net=Probe(cp['n_features'],**{k:c[k] for k in ['architecture','width','depth','dropout']});net.load_state_dict(cp['state'])
            name=family+'_'+str(seed);predictions[name]=predict(net,inp['x'][c['feature']]['test']);validation[name]=predict(net,inp['x'][c['feature']]['val']);meta[name]={'family':family,'seed':seed,'config':c,'supervision':'token positions'};names.append(name)
        en=family+'_ensemble';groups[en]=names
        if w['config_id']==selection['overall_seed42_winner']:groups['architecture_selected']=names
    # Historical weak baseline transferred unchanged to the new summaries.
    import evaluate as old_eval
    old_eval.DEVICE='cpu';ff={s:[dict(np.load(features(r))) for r in inp['rows'][s]] for s in ['val','test']}
    for seed in [42,43,44]:
        cp=torch.load(OLD/f'results/news_large/checkpoints/before_{seed}.pt',map_location='cpu',weights_only=False);name=f'hami_{seed}'
        _,predictions[name]=old_eval.predict(cp,ff['test'],rr);_,validation[name]=old_eval.predict(cp,ff['val'],vr);meta[name]={'family':'hami','seed':seed,'supervision':'bag labels only; original imported ori adaptation'}
    groups['hami_ensemble']=[f'hami_{s}' for s in [42,43,44]]
    for entry in fusion_selection['methods']:
        cp=pickle.loads((dest/f'checkpoints/fusion_{entry["kind"]}.pkl').read_bytes())
        for policy in ['active','random']:
            name='fusion_'+entry['kind']+'_'+policy;predictions[name]=fusion_scores(cp,b,'test',policy);validation[name]=fusion_scores(cp,b,'val',policy);meta[name]={'family':name,'policy':policy,'kind':entry['kind'],'supervision':'original token positions; no generated gold'}
    for name,names in groups.items():
        predictions[name]=[np.mean([predictions[k][i] for k in names],axis=0) for i in range(len(rr))];validation[name]=[np.mean([validation[k][i] for k in names],axis=0) for i in range(len(vr))];meta[name]={'family':name,'ensemble':names,'selection':'frozen validation choice or prespecified family'}
    metrics=[];cases={}
    for name,ss in predictions.items():
        m,cs=measure(rr,truth,ss,vr,validation[name]);m.update(name=name,**meta[name]);metrics.append(m);cases[name]=cs
        print(name,{k:round(m[k],3) for k in ['global_token_auc','local_auc','clean_alarm','error_recall']},flush=True)
    metric_path=dest/'metrics.json';metric_path.write_text(json.dumps(metrics,indent=2),encoding='utf8')
    writel(dest/'test_predictions.jsonl',[{'id':r['id'],'group':r['group'],'label':r['label'],'offsets':inp['offsets']['test'][i].tolist(),
        'predictions':{n:s[i].tolist() for n,s in predictions.items()}} for i,r in enumerate(rr)])
    writel(dest/'validation_predictions.jsonl',[{'id':r['id'],'group':r['group'],'label':r['label'],
        'predictions':{n:s[i].tolist() for n,s in validation.items()}} for i,r in enumerate(vr)])
    comparisons=[('architecture_selected','linear_attention'),('architecture_selected','linear_hybrid'),
        ('fusion_verify_joint_active','fusion_selection_only_active'),('fusion_verify_joint_active','fusion_verify_prompt_active'),
        ('fusion_verify_hidden_active','fusion_verify_prompt_wide_active'),('fusion_verify_joint_active','fusion_verify_prompt_wide_active'),
        ('fusion_verify_joint_active','fusion_verify_text_active'),('fusion_verify_joint_active','fusion_expand_hidden_active'),
        ('fusion_verify_joint_active','fusion_verify_joint_random'),('fusion_direct_hidden_active','fusion_direct_probs_active')]
    keep={k:v for k,v in cases.items() if k in set(sum(([a,b] for a,b in comparisons),[])) or k=='hami_ensemble'}
    (dest/'confidence_intervals.json').write_text(json.dumps(bootstrap(keep,comparisons,truth,predictions),indent=2),encoding='utf8')
    # Include all summaries, not just selected errors, when computing policy coverage and cost.
    anns={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')};cost=[]
    for policy in ['active','random']:
        covered=0;total=0;answers=0;tokens=0;alltokens=0
        for i,r in enumerate(rr):
            p=b['policies'][(r['id'],policy)];lo,hi=p['span'];sp=anns[r['id']]['spans'];covered+=sum(lo<s['end'] and hi>s['start'] for s in sp);total+=len(sp);answers+=int(any(lo<s['end'] and hi>s['start'] for s in sp));off=inp['offsets']['test'][i];tokens+=int(((off[:,0]<hi)&(off[:,1]>lo)).sum());alltokens+=len(off)
        for arm in ['direct','verify','expand']:
            records=[b['records'][arm][b['policies'][(r['id'],policy)]['query_id']] for r in rr]
            cost.append({'policy':policy,'arm':arm,'n_summaries':len(rr),'calls_per_summary':1,'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in records])),
                'mean_amortized_seconds':float(np.mean([r['amortized_seconds'] for r in records])),'truncated_fraction':float(np.mean([r['reached_token_limit'] for r in records])),
                'selected_span_recall':covered/max(1,total),'error_answer_selection_coverage':answers/sum(r['label'] for r in rr),'selected_token_fraction':tokens/alltokens})
    (dest/'policy_costs.json').write_text(json.dumps(cost,indent=2),encoding='utf8')
    (dest/'evaluation_manifest.json').write_text(json.dumps({'n':80,'error':40,'normal':40,'methods':len(metrics),
        'test':'fresh normalized-source-disjoint Llama2-7B summaries; generation domain differs from Mistral training',
        'selection_manifests_required':['structure_selection.json','fusion_selection.json'],
        'supplement':'same Qwen7B NF4; no external judge; original released human labels',
        'probabilities':'scores uncalibrated; validation empirical threshold does not guarantee population FPR',
        'cost':'additional query costs only; base feature extraction cost excluded, amortized batch times not single-user latency'},indent=2),encoding='utf8');print('FINAL EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
