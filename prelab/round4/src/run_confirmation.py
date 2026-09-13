"""No fitting or selection: apply already-frozen round4 models to50new sources."""
import json,pickle,time,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
from common4 import ROOT,PRE,readl,writel,save,sha
from engine import load_model,extract,text_prefix
from make_queries import sentences,values,factual_span
from readouts import Capture,prompt
from fit import aggregate,metrics
from prepare_confirmation import DEST

def features(qs,folder,pca):
    prob=[];hidden=[];extras=[]
    for q in qs:
        with np.load(folder/f'{q["query_id"]}.npz') as z:prob.append(z['probs']);hidden.append(z['hidden'].ravel())
        exact=' '.join(q['target_text'].lower().split()) in ' '.join(q['evidence'].lower().split())
        extras.append(q['base_features']+[float(exact),min(1,len(q['target_text'])/200.)])
    p=np.stack(prob).astype(np.float32);log=np.log(np.clip(p,1e-7,1));kinds=np.array([[float(q['kind']==k) for k in ['whole','sentence','fact']] for q in qs],np.float32)
    abc=np.column_stack([p,log[:,1]-log[:,0],log[:,2]-log[:,0],-(p*log).sum(1),kinds]);z=pca.transform(np.stack(hidden).astype(np.float32))
    return {'abc':abc,'abc_base':np.column_stack([abc,extras]),'abc_base_hidden16':np.column_stack([abc,extras,z])}

@torch.inference_mode()
def main():
    protocol=json.loads((DEST/'protocol.json').read_text());ph=sha(DEST/'protocol.json');rows=readl(DEST/'data/rows.jsonl');anns={a['id']:a for a in readl(DEST/'data/annotations.jsonl')}
    for p,h in protocol['frozen_dependencies'].items():assert sha(ROOT/p)==h
    assert sha(DEST/'data/rows.jsonl')==protocol['rows_sha256'] and sha(DEST/'data/annotations.jsonl')==protocol['annotations_sha256']
    assert len(rows)==len({r['group'] for r in rows})==protocol['n'] and not {r['group'] for r in rows}&{r['group'] for r in readl(ROOT/'data/rows.jsonl')}
    b=pickle.loads((ROOT/'data/unit_inputs.pkl').read_bytes());nets=pickle.loads((ROOT/'results/checkpoints/unit_models.pkl').read_bytes());base=pickle.loads((ROOT/'results/checkpoints/base.pkl').read_bytes())['attention']
    # Validate the reimplemented feature assembly against prior cached validation data.
    check=[i for i,q in enumerate(b['queries']) if q['split']=='val'][:21];verified=features([b['queries'][i] for i in check],ROOT/'data/readouts',b['pca'])
    for k in verified:assert np.allclose(verified[k],b['x'][k][check],rtol=1e-4,atol=1e-4),k
    tok,model=load_model('large');started=time.time();offsets=[];scores=[];truth=[];queries=[]
    for i,r in enumerate(rows):
        path=Path(r['feature_path'])
        if not path.exists():np.savez_compressed(path,**extract(tok,model,r,contrast=True))
        with np.load(path) as z:off=z['offsets'].copy();s=base.predict_proba(z['lookback'].astype(np.float32))[:,1].astype(np.float32)
        offsets.append(off);scores.append(s);truth.append(np.array([int(any(lo<a['end'] and hi>a['start'] for a in anns[r['id']]['spans'])) for lo,hi in off]))
        spans=sentences(r['response']);ranks=sorted(range(len(spans)),key=lambda j:(-values(off,s,spans[j])[2],j))[:3]
        for rank,j in enumerate(ranks):
            target,kind=factual_span(r['response'],spans[j],off,s)
            for mode,span,sub in [('sentence',spans[j],'sentence'),('fact',target,kind)]:
                qid=hashlib.sha256(f'{r["id"]}|{mode}|{spans[j]}|{span}'.encode()).hexdigest()[:20]
                queries.append({'query_id':qid,'id':r['id'],'split':'confirmation','kind':mode,'rank':rank,'sentence_index':j,'sentence_span':spans[j],
                    'target_span':span,'candidate_kind':sub,'statement':r['response'][spans[j][0]:spans[j][1]],'target_text':r['response'][span[0]:span[1]],
                    'evidence':r['evidence'],'base_features':values(off,s,span)})
        if i%10==0:print('CONFIRMATION BASE',i+1,len(rows),flush=True)
    writel(DEST/'data/queries.jsonl',queries);tok.padding_side='left';tok.pad_token=tok.eos_token;letter_ids=[tok.encode(x,add_special_tokens=False) for x in ['A','B','C']];assert all(len(x)==1 for x in letter_ids);letters=[x[0] for x in letter_ids]
    folder=DEST/'data/readouts';todo=sorted([q for q in queries if not (folder/f'{q["query_id"]}.npz').exists()],key=lambda q:(len(q['evidence'])+len(q['statement']),q['query_id']));i=0;records=[];hook_checked=False
    while i<len(todo):
        size=2 if len(todo[i]['evidence'])+len(todo[i]['statement'])<6000 else 1;batch=todo[i:i+size]
        inp=tok([text_prefix(tok,prompt(q)) for q in batch],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');assert inp['input_ids'].shape[1]<=4096
        begin=time.time()
        with Capture(model) as cap:o=model(**inp,use_cache=False,logits_to_keep=1)
        if not hook_checked:
            plain=model(**inp,use_cache=False,logits_to_keep=1).logits;assert torch.equal(plain,o.logits);del plain;hook_checked=True
        p=o.logits[:,-1,letters].float().softmax(-1).cpu().numpy();h=cap.array();elapsed=time.time()-begin
        for j,q in enumerate(batch):
            assert np.isfinite(h[j]).all() and np.isfinite(p[j]).all()
            np.savez_compressed(folder/f'{q["query_id"]}.npz',hidden=h[j].astype(np.float16),probs=p[j]);records.append({'query_id':q['query_id'],'seconds':elapsed/len(batch),'batch_size':len(batch),'generated_tokens':0})
        del inp,o,cap,h;i+=len(batch)
        if i%20<2 or i==len(todo):print('CONFIRMATION READOUTS',i,len(todo),flush=True)
        if i%100<2:torch.cuda.empty_cache()
    if records:
        old=readl(folder/'records.jsonl') if (folder/'records.jsonl').exists() else [];writel(folder/'records.jsonl',old+records)
    x=features(queries,folder,b['pca']);by={r['id']:[j for j,q in enumerate(queries) if q['id']==r['id']] for r in rows};yt=np.array([r['label'] for r in rows]);results=[];predictions={}
    oldmetrics={r['name']:r for r in json.loads((ROOT/'results/metrics.json').read_text())}
    for name,c in protocol['frozen_methods'].items():
        local=[];ss=[]
        if name=='base_scores':ss=np.array([aggregate(s,c['aggregate']) for s in scores]);local=[s.copy() for s in scores]
        else:
            mid=c['model_id'];pr=x['abc'][:,1] if mid=='raw_B' else nets[mid]['model'].predict_proba(x[nets[mid]['feature']])[:,1]
            for i,r in enumerate(rows):
                ids=[j for j in by[r['id']] if name=='sentences_facts' or queries[j]['kind']=='sentence'];ss.append(aggregate(pr[ids],c['aggregate']));loc=scores[i].copy();off=offsets[i]
                for j in ids:
                    lo,hi=queries[j]['target_span'];loc[(off[:,0]<hi)&(off[:,1]>lo)]=pr[j]
                local.append(loc)
            ss=np.array(ss)
        pred=ss>=c['threshold'];m=metrics(yt,pred);assert abs(m['f1']-f1_score(yt,pred))<1e-12 and abs(m['precision']-precision_score(yt,pred))<1e-12 and abs(m['recall']-recall_score(yt,pred))<1e-12
        m.update(name=name,config=c,answer_auc=float(roc_auc_score(yt,ss)));lp=np.concatenate(local);ly=np.concatenate(truth);token_threshold=oldmetrics[name]['token_threshold']
        m.update(token_metrics=metrics(ly,lp>=token_threshold),token_threshold=token_threshold,token_auc=float(roc_auc_score(ly,lp)),within_answer_auc=float(np.mean([roc_auc_score(y,s) for y,s in zip(truth,local) if len(np.unique(y))==2])))
        rng=np.random.default_rng(20260915);boot=[]
        for _ in range(2000):
            ix=rng.integers(0,len(yt),len(yt));boot.append(metrics(yt[ix],pred[ix])['f1'])
        m['F1_95pct']=np.quantile(boot,[.025,.975]).tolist();results.append(m);predictions[name]={'scores':ss,'pred':pred,'local':local};print('CONFIRMATION RESULT',name,{k:m[k] for k in ['f1','precision','recall','tp','fp','fn','tn']},flush=True)
    save(DEST/'results/metrics.json',results);writel(DEST/'results/predictions.jsonl',[{'id':r['id'],'group':r['group'],'label':r['label'],'offsets':offsets[i].tolist(),'methods':{n:{'score':float(p['scores'][i]),'pred':bool(p['pred'][i]),'token_risks':p['local'][i].tolist()} for n,p in predictions.items()}} for i,r in enumerate(rows)])
    coverage={}
    for kind in ['sentence','fact']:
        hits=sum(any(q['target_span'][0]<a['end'] and q['target_span'][1]>a['start'] for j in by[r['id']] for q in [queries[j]] if q['kind']==kind for a in anns[r['id']]['spans']) for r in rows if r['label'])
        coverage[kind]={'covered_error_answers':hits,'all_error_answers':protocol['errors']}
    save(DEST/'results/coverage.json',coverage)
    for p,h in protocol['frozen_dependencies'].items():assert sha(ROOT/p)==h
    assert sha(DEST/'protocol.json')==ph
    primary=next(m for m in results if m['name']=='sentences_facts')
    save(DEST/'results/manifest.json',{'completed':True,'n':protocol['n'],'primary_method':'sentences_facts','primary_F1':primary['f1'],'primary_target_met':primary['f1']>.7,
        'new_sources_exclude_all684':True,'frozen_protocol_sha256':ph,'no_fitting_or_threshold_tuning':True,'feature_assembly_verified_on_old_validation_queries':len(check),'hook_logits_unchanged':hook_checked,'queries':len(queries),'generated_tokens':0,'external_llm_calls':0,'seconds':time.time()-started})
    print('CONFIRMATION COMPLETE',flush=True)

if __name__=='__main__':main()
