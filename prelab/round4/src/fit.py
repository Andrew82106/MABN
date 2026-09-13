import json,pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from common4 import ROOT,readl,writel,save

SCOPES=['single_sentence','sentences3','facts3','whole','whole_sentences','whole_facts','sentences_facts','all_checks']
AGGS=['max','top2mean','noisy_or']

def unit_probs(b,models,mid):
    if mid=='raw_B':return b['x']['abc'][:,1]
    if mid=='raw_B_over_AB':return b['x']['abc'][:,1]/np.maximum(1e-9,b['x']['abc'][:,:2].sum(1))
    cp=models[mid];return cp['model'].predict_proba(b['x'][cp['feature']])[:,1]

def metrics(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool);tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return {'f1':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'false_alarm':fp/max(1,fp+tn),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def calibrate(y,s):
    y=np.asarray(y,int);s=np.asarray(s);thresholds=np.unique(s);pred=s[None,:]>=thresholds[:,None]
    tp=(pred*y).sum(1);n=pred.sum(1);f=2*tp/(n+y.sum());p=tp/np.maximum(1,n)
    j=max(range(len(thresholds)),key=lambda i:(f[i],p[i],thresholds[i]))
    return float(thresholds[j]),metrics(y,pred[j])

def aggregate(s,kind):
    a=np.asarray(s,float)
    if kind=='max':return float(a.max())
    if kind=='top2mean':return float(np.sort(a)[-2:].mean())
    if kind=='top10':return float(np.sort(a)[-(int(len(a)*.1)+1):].mean())
    if kind=='mean':return float(a.mean())
    return float(-np.expm1(np.log1p(-np.clip(a,0,1-1e-8)).sum()))

def included(q,scope):
    if scope=='single_sentence':return q['kind']=='sentence' and q['rank']==0
    if scope=='sentences3':return q['kind']=='sentence'
    if scope=='facts3':return q['kind']=='fact'
    if scope=='whole':return q['kind']=='whole'
    if scope=='whole_sentences':return q['kind'] in ['whole','sentence']
    if scope=='whole_facts':return q['kind'] in ['whole','fact']
    if scope=='sentences_facts':return q['kind'] in ['sentence','fact']
    if scope=='all_checks':return True
    raise ValueError(scope)

def answer_scores(b,unit_probs,split,scope,agg):
    result=[];calls=[]
    for i in b['base']['split_indices'][split]:
        qs=[j for j in b['by_id'][b['base']['rows'][i]['id']] if included(b['queries'][j],scope)]
        result.append(aggregate(unit_probs[qs],agg));calls.append(len(qs))
    return np.array(result),float(np.mean(calls))

def prepare():
    path=ROOT/'data/unit_inputs.pkl'
    if path.exists():return pickle.loads(path.read_bytes())
    qs=readl(ROOT/'data/queries.jsonl');base=pickle.loads((ROOT/'data/base.pkl').read_bytes());ann={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')}
    tr=np.array([q['split']=='train' for q in qs]);val=np.array([q['split']=='val' for q in qs]);hidden=[];prob=[];extras=[];by={};uy=[]
    for j,q in enumerate(qs):
        with np.load(ROOT/f'data/readouts/{q["query_id"]}.npz') as z:hidden.append(z['hidden'].ravel());prob.append(z['probs'])
        by.setdefault(q['id'],[]).append(j);lo,hi=q['target_span'];uy.append(int(any(lo<s['end'] and hi>s['start'] for s in ann[q['id']]['spans'])))
        exact=' '.join(q['target_text'].lower().split()) in ' '.join(q['evidence'].lower().split())
        extras.append(q['base_features']+[float(exact),min(1,len(q['target_text'])/200.)])
    p=np.stack(prob).astype(np.float32);log=np.log(np.clip(p,1e-7,1));kinds=np.array([[float(q['kind']==k) for k in ['whole','sentence','fact']] for q in qs],np.float32)
    abc=np.column_stack([p,log[:,1]-log[:,0],log[:,2]-log[:,0],-(p*log).sum(1),kinds])
    h=np.stack(hidden).astype(np.float32)
    with threadpool_limits(limits=4):
        pca=PCA(n_components=16,svd_solver='randomized',random_state=42).fit(h[tr]);z=pca.transform(h)
    x={'abc':abc,'abc_base':np.column_stack([abc,extras]),'abc_base_hidden16':np.column_stack([abc,extras,z])}
    result={'queries':qs,'base':base,'by_id':by,'x':x,'y':np.array(uy),'train_mask':tr,'val_mask':val,'pca':pca}
    path.write_bytes(pickle.dumps(result,protocol=5));save(ROOT/'results/unit_transform.json',{'fit_query_ids':[q['query_id'] for q in qs if q['split']=='train'],
        'components':16,'variance':float(pca.explained_variance_ratio_.sum()),'dimensions':{k:v.shape[1] for k,v in x.items()},'unit_positive_train':int(np.array(uy)[tr].sum())})
    return result

def main():
    b=prepare();va=b['base']['split_indices']['val'];yv=np.array([b['base']['rows'][i]['label'] for i in va]);tr=b['train_mask'];models={};trials=[]
    for source in ['base_scores','support_scores']:
        for agg in ['max','top10','mean']:
            s=np.array([aggregate(b['base'][source][i],agg) for i in va]);threshold,m=calibrate(yv,s)
            trials.append({'family':source,'model_id':None,'scope':source,'aggregate':agg,'threshold':threshold,'val_metrics':m,'mean_calls':0.})
    for mid in ['raw_B','raw_B_over_AB']:
        pr=np.full(len(b['queries']),np.nan);v=b['x']['abc'][b['val_mask']]
        pr[b['val_mask']]=v[:,1] if mid=='raw_B' else v[:,1]/np.maximum(1e-9,v[:,:2].sum(1))
        for scope in SCOPES:
            for agg in AGGS:
                s,calls=answer_scores(b,pr,'val',scope,agg);threshold,m=calibrate(yv,s)
                trials.append({'family':scope,'model_id':mid,'scope':scope,'aggregate':agg,'threshold':threshold,'val_metrics':m,'mean_calls':calls})
    for feature in ['abc','abc_base','abc_base_hidden16']:
        for weight in [None,'balanced']:
            for c in [.001,.01,.1,1.]:
                mid=f'{feature}_{weight}_{c}';x=b['x'][feature]
                with threadpool_limits(limits=4):
                    net=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight=weight,solver='liblinear',max_iter=1500,random_state=42)).fit(x[tr],b['y'][tr])
                models[mid]={'model':net,'feature':feature,'C':c,'class_weight':weight};pr=np.full(len(x),np.nan)
                pr[b['val_mask']]=net.predict_proba(x[b['val_mask']])[:,1]
                for scope in SCOPES:
                    for agg in AGGS:
                        s,calls=answer_scores(b,pr,'val',scope,agg);threshold,m=calibrate(yv,s)
                        trials.append({'family':scope,'model_id':mid,'scope':scope,'aggregate':agg,'threshold':threshold,'val_metrics':m,'mean_calls':calls})
                print('UNIT FIT',mid,flush=True)
    for j,r in enumerate(trials):r['candidate_index']=j
    key=lambda r:(r['val_metrics']['f1'],r['val_metrics']['precision'],-r['mean_calls'],-r['candidate_index'])
    winners={}
    for r in trials:
        if r['family'] not in winners or key(r)>key(winners[r['family']]):winners[r['family']]=r
    best=max(trials,key=key)
    (ROOT/'results/checkpoints/unit_models.pkl').write_bytes(pickle.dumps(models,protocol=5));writel(ROOT/'results/validation_trials.jsonl',trials)
    save(ROOT/'results/selection.json',{'selected':best,'family_winners':winners,'primary':'answer-level error-class F1, threshold selected solely on validation','comparison':'>= threshold','test_scores_inspected':False,'n_candidates':len(trials)})
    print('FROZEN SELECTION',json.dumps(best),flush=True)

if __name__=='__main__':main()
