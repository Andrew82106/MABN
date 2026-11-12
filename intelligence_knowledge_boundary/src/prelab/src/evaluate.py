"""Heldout evaluation; span labels are first consumed here, after fitting."""
import argparse
import csv
import html
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,balanced_accuracy_score
from prepare_data import ROOT,read_jsonl,write_jsonl
from fit_probes import HaMI,CFG,load_split,network_scores

def token_labels(offsets,spans):
    return np.array([int(any(a<s['end'] and b>s['start'] for s in spans)) for a,b in offsets])

def metrics(y,bags,tokens,feats,annotations,thresh):
    out={'response_auc':float(roc_auc_score(y,bags)),
         'response_ap':float(average_precision_score(y,bags)),
         'response_f1':float(f1_score(y,bags>=thresh,zero_division=0)),
         'response_balanced_accuracy':float(balanced_accuracy_score(y,bags>=thresh)),
         'response_precision':float(precision_score(y,bags>=thresh,zero_division=0)),
         'response_recall':float(recall_score(y,bags>=thresh,zero_division=0))}
    if tokens is None: return out
    auc=[]; ap=[]; precision=[]; recall=[]; chance=[]; spans_hit=0; nspans=0; all_y=[]; all_s=[]
    for label,s,f,ann in zip(y,tokens,feats,annotations):
        z=token_labels(f['offsets'],ann['spans']); all_y.extend(z); all_s.extend(s)
        if label and z.any() and not z.all():
            auc.append(roc_auc_score(z,s)); ap.append(average_precision_score(z,s)); chance.append(z.mean())
            k=max(1,int(np.ceil(len(s)*CFG['localization_budget'])))
            top=np.argsort(-s,kind='stable')[:k]
            precision.append(z[top].mean()); recall.append(z[top].sum()/z.sum())
            for span in ann['spans']:
                nspans+=1
                spans_hit+=int(any(f['offsets'][j][0]<span['end'] and f['offsets'][j][1]>span['start'] for j in top))
    out.update(token_auc_global=float(roc_auc_score(all_y,all_s)),
               token_ap_global=float(average_precision_score(all_y,all_s)),
               within_error_response_auc=float(np.mean(auc)),within_error_response_ap=float(np.mean(ap)),
               top10_precision=float(np.mean(precision)),top10_recall=float(np.mean(recall)),
               span_hit_at_top10=float(spans_hit/nspans),token_positive_fraction_in_error_responses=float(np.mean(chance)),
               local_n_responses=len(auc),local_n_spans=nspans)
    return out

def predict(path,rows,feats):
    if path.suffix=='.pt':
        ckpt=torch.load(path,weights_only=False)
        net=HaMI(896,8).net.cuda(); net.load_state_dict(ckpt['state'])
        method=ckpt['method']; actual='mil_topk' if method=='shuffled_label_mil_topk' else method
        bags,tokens=network_scores(net,feats,ckpt['layer_index'],actual)
    else:
        ckpt=pickle.loads(path.read_bytes()); method=ckpt['method']; model=ckpt['model']; tokens=None
        if method in ['last_linear','mean_linear']:
            li=ckpt['layer_index']
            x=np.stack([f['hidden'][-1,li,:] if method=='last_linear' else f['hidden'][:,li,:].astype(np.float32).mean(0) for f in feats])
            bags=model.predict_proba(x)[:,1]
            # Deliberately naive local application of a bag-trained linear probe, reported as a baseline only.
            tokens=[model.predict_proba(f['hidden'][:,li,:])[:,1] for f in feats]
        elif method=='response_length':
            bags=model.predict_proba(np.array([[r['n_tokens']] for r in rows]))[:,1]
        else: bags=model.predict_proba([r['response'] for r in rows])[:,1]
    return ckpt,bags,tokens

def bootstrap(y,bags,tokens,feats,anns,n=1000):
    rng=np.random.default_rng(20260909); ba=[]; la=[]
    local=[]
    if tokens is not None:
        for label,s,f,a in zip(y,tokens,feats,anns):
            z=token_labels(f['offsets'],a['spans'])
            if label and z.any() and not z.all(): local.append(roc_auc_score(z,s))
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))==2: ba.append(roc_auc_score(y[ix],bags[ix]))
        if local: la.append(np.mean(rng.choice(local,len(local),replace=True)))
    return {'response_auc_95ci':np.quantile(ba,[.025,.975]).tolist(),
            'within_error_response_auc_95ci':np.quantile(la,[.025,.975]).tolist() if la else None,
            'bootstrap_unit':'source/response, never independent tokens','replicates':n}

def make_html(rows,feats,anns,bags,tokens,path,title):
    sections=[]
    for r,f,a,b,s in zip(rows,feats,anns,bags,tokens):
        chars=r['response']; cr=np.zeros(len(chars)); gt=np.zeros(len(chars),bool)
        for (start,end),score in zip(f['offsets'],s): cr[start:end]=np.maximum(cr[start:end],score)
        for span in a['spans']: gt[span['start']:span['end']]=True
        # Rank colors show within-response location, not calibrated probability.
        ranked=np.searchsorted(np.sort(s),cr)/max(1,len(s))
        text=''.join(f'<span style="background:rgba(239,68,68,{.65*v:.3f});'+('border-bottom:3px solid #2563eb;' if g else '')+f'" title="score={score:.4f}">{html.escape(c)}</span>' for c,v,g,score in zip(chars,ranked,gt,cr))
        sections.append(f'<article><h3>ID {r["id"]} · 整段标签 {r["label"]} · 风险分数 {b:.3f}</h3><p>{text}</p><details><summary>证据与标注说明</summary><pre>{html.escape(r["evidence"])}</pre><pre>{html.escape(json.dumps(a["spans"],ensure_ascii=False,indent=2))}</pre></details></article>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{max-width:1100px;margin:35px auto;font:17px/1.8 system-ui;background:#f8fafc;color:#172033}article{background:white;padding:20px;margin:18px 0;border:1px solid #ddd;border-radius:8px}pre{white-space:pre-wrap;font-size:13px}h1{font-size:27px}p{line-height:2.1}</style><h1>'+html.escape(title)+'</h1><p>红色越深：该回答内相对风险排名越高。蓝色下划线：真实标注错误片段。颜色不是幻觉概率。全部测试样本按原始 ID 展示，未筛选成功案例。</p>'+''.join(sections)+'</html>'
    path.write_text(page,encoding='utf8')

def main():
    torch.set_num_threads(6)
    rows,feats,y=load_split('test')
    amap={x['id']:x for x in read_jsonl(ROOT/'data/annotations/test_spans.jsonl')}; anns=[amap[r['id']] for r in rows]
    results=[]; predictions={}
    for path in sorted((ROOT/'results/checkpoints').glob('*')):
        if path.suffix not in ['.pt','.pkl']: continue
        ck,bags,tokens=predict(path,rows,feats)
        row={'name':path.stem,'method':ck['method'],'seed':ck.get('seed'), 'layer':ck.get('layer'),
             'val_auc':ck['val_auc'],'threshold':ck['threshold'],**metrics(y,bags,tokens,feats,anns,ck['threshold'])}
        results.append(row); predictions[path.stem]=(bags,tokens)
        write_jsonl(ROOT/f'results/predictions/{path.stem}.jsonl',[{'id':r['id'],'bag_score':float(b),'token_scores':s.tolist() if s is not None else None} for r,b,s in zip(rows,bags,tokens if tokens is not None else [None]*len(rows))])
        print(path.stem,row['response_auc'],row.get('within_error_response_auc'),flush=True)
    # Untrained predictive surprisal baseline; threshold chosen on validation bags only.
    from fit_probes import threshold
    vr,vf,vy=load_split('val'); vs=np.array([np.sort(f['nll'])[-(int(len(f['nll'])*.1)+1):].mean() for f in vf])
    tokens=[f['nll'] for f in feats]; bags=np.array([np.sort(s)[-(int(len(s)*.1)+1):].mean() for s in tokens]); th=threshold(vy,vs)
    results.append({'name':'token_surprisal','method':'token_surprisal','val_auc':roc_auc_score(vy,vs),'threshold':th,**metrics(y,bags,tokens,feats,anns,th)})
    predictions['token_surprisal']=(bags,tokens)
    (ROOT/'results/metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    columns=list(dict.fromkeys(k for r in results for k in r))
    with (ROOT/'results/metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(results)
    # Select the method by mean validation response AUROC, not by test/localization results.
    families={m:np.mean([r['val_auc'] for r in results if r['method']==m]) for m in ['mil_mean','mil_max','mil_topk','hami_ori']}
    method=max(families,key=families.get); members=[r for r in results if r['method']==method]
    # Seed chosen in advance for display/CI; all seeds remain in metrics.
    selected=next((r for r in members if r.get('seed')==42),members[0]); bags,tokens=predictions[selected['name']]
    selection={'method':method,'display_checkpoint':selected['name'],'selection':'mean validation bag AUROC; display seed 42',
               'validation_by_method':families,'metrics':selected,**bootstrap(y,bags,tokens,feats,anns)}
    (ROOT/'results/selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    make_html(rows,feats,anns,bags,tokens,ROOT/'results/token_risk_report.html',f'弱监督事实错误定位预实验 · {selected["name"]}')
    print('SELECTED',json.dumps(selection),flush=True)

if __name__=='__main__': main()
