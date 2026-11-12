"""Supplementary-information features and token-score fusion with gold-free policies."""
import json,pickle,re,time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from shared import ROOT,readl

KINDS=['baseline','selection_only','direct_probs','direct_hidden','verify_prompt','verify_prompt_wide','verify_text','verify_hidden','verify_joint','expand_hidden']

def content_features(record,query):
    text=record['response']; verdict=re.search(r'verdict\s*:\s*(SUPPORTED|CONTRADICTED|INSUFFICIENT)',text,re.I)
    value=verdict.group(1).upper() if verdict else 'MISSING'
    quote=re.findall(r'["“]([^"”]+)["”]',text)
    source=' '.join(query['evidence'].lower().split())
    found=any(' '.join(q.lower().split()) in source for q in quote if q.strip())
    a=set(re.findall(r'\w+',query['statement'].lower()));b=set(re.findall(r'\w+',text.lower()))
    return np.array([int(value==v) for v in ['SUPPORTED','CONTRADICTED','INSUFFICIENT','MISSING']]+[
        float(found),float(bool(quote)),len(a&b)/max(1,len(a)),record['generated_tokens']/64.,float(record['reached_token_limit'])],np.float32)

def load_bundle():
    return pickle.loads((ROOT/'data/fusion_inputs.pkl').read_bytes())

def prepare():
    path=ROOT/'data/fusion_inputs.pkl'
    if path.exists():return load_bundle()
    inp=pickle.loads((ROOT/'data/inputs.pkl').read_bytes());base=pickle.loads((ROOT/'data/policy_scores.pkl').read_bytes())
    qs=readl(ROOT/'data/queries.jsonl');qids=[q['query_id'] for q in qs];qi={q:i for i,q in enumerate(qids)}
    train=np.array([q['split']=='train' for q in qs]);latent={};raw={};transforms={};records={};changes={}
    for arm in ['direct','verify','expand']:
        records[arm]={r['query_id']:r for r in readl(ROOT/f'data/supplement/{arm}.jsonl')}
        assert set(records[arm])==set(qids)
        ff=[dict(np.load(ROOT/f'data/supplement/features/{arm}_{q}.npz')) for q in qids]
        if arm!='direct':
            a=np.stack([f['prompt'] for f in ff]).astype(np.float32);z=np.stack([f['last'] for f in ff]).astype(np.float32)
            cosine=(a*z).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(z,axis=-1)+1e-8)
            distance=np.linalg.norm(a-z,axis=-1)/(np.linalg.norm(a,axis=-1)+1e-8)
            changes[arm]=np.column_stack([cosine,distance])
        for part in (['prompt'] if arm=='direct' else ['prompt','mean','last']):
            key=arm+'_'+part;xx=np.stack([f[part].ravel() for f in ff]).astype(np.float32)
            with threadpool_limits(limits=4):
                pca=PCA(n_components=32,svd_solver='randomized',random_state=42).fit(xx[train]);zz=pca.transform(xx)
            latent[key]=zz;transforms[key]=pca;print('SUPPLEMENT PCA',key,round(float(pca.explained_variance_ratio_.sum()),3),flush=True)
            if key=='verify_prompt':
                with threadpool_limits(limits=4):
                    wide=PCA(n_components=104,svd_solver='randomized',random_state=42).fit(xx[train]);latent['verify_prompt_wide']=wide.transform(xx)
                transforms['verify_prompt_wide']=wide
                print('SUPPLEMENT PCA verify_prompt_wide',round(float(wide.explained_variance_ratio_.sum()),3),flush=True)
        if arm=='direct':raw['direct_probs']=np.stack([f['verdict_probs'] for f in ff])
    raw['direct_hidden']=latent['direct_prompt'];raw['verify_prompt']=latent['verify_prompt']
    raw['verify_prompt_wide']=latent['verify_prompt_wide']
    raw['verify_text']=np.stack([content_features(records['verify'][q['query_id']],q) for q in qs])
    for arm in ['verify','expand']:raw[arm+'_hidden']=np.column_stack([*[latent[arm+'_'+p] for p in ['prompt','mean','last']],changes[arm]])
    raw['verify_joint']=np.column_stack([raw['verify_hidden'],raw['verify_text']])
    extras={};scalers={}
    for k,xx in raw.items():
        scaler=StandardScaler().fit(xx[train]);scalers[k]=scaler;extras[k]=scaler.transform(xx).astype(np.float32)
    policies={(p['id'],p['policy']):p for p in readl(ROOT/'data/query_policies.jsonl')}
    result={'rows':inp['rows'],'y':inp['y'],'offsets':inp['offsets'],'base':base,'policies':policies,
        'extras':extras,'query_index':qi,'pcas':transforms,'scalers':scalers,'records':records}
    path.write_bytes(pickle.dumps(result,protocol=5))
    (ROOT/'results/fusion_transform.json').write_text(json.dumps({'training_query_ids':[q['query_id'] for q in qs if q['split']=='train'],
        'pca_components':{k:int(v.n_components_) for k,v in transforms.items()},'input_features':{k:v.shape[1] for k,v in extras.items()},
        'scope':'training query embeddings only; test embeddings transformed without fitting'},indent=2),encoding='utf8')
    return result

def matrix(bundle,split,policy,kind):
    out=[]
    for i,r in enumerate(bundle['rows'][split]):
        score=np.clip(bundle['base'][split][i],1e-6,1-1e-6);logit=np.log(score/(1-score)).astype(np.float32)
        p=bundle['policies'][(r['id'],policy)];lo,hi=p['span'];off=bundle['offsets'][split][i]
        mask=((off[:,0]<hi)&(off[:,1]>lo)).astype(np.float32)
        if kind=='baseline':x=logit[:,None]
        elif kind=='selection_only':x=np.column_stack([logit,mask])
        else:
            ex=bundle['extras'][kind][bundle['query_index'][p['query_id']]]
            x=np.column_stack([logit,mask,mask[:,None]*ex[None,:]])
        out.append(x.astype(np.float32))
    return out

def scores(cp,bundle,split,policy):
    return [cp['model'].predict_proba(x)[:,1].astype(np.float32) for x in matrix(bundle,split,policy,cp['kind'])]

def main():
    b=prepare();dest=ROOT/'results/checkpoints';selection=[]
    for kind in KINDS:
        path=dest/f'fusion_{kind}.pkl'
        if path.exists():selection.append(pickle.loads(path.read_bytes())['selection']);continue
        xx=np.concatenate(sum([matrix(b,'train',p,kind) for p in ['active','random']],[]))
        yy=np.tile(np.concatenate(b['y']['train']),2);vv={p:matrix(b,'val',p,kind) for p in ['active','random']};vy=np.concatenate(b['y']['val']);best=None
        with threadpool_limits(limits=4):
            for c in [.01,.1,1.]:
                net=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=700,random_state=42).fit(xx,yy)
                aucs={p:float(roc_auc_score(vy,np.concatenate([net.predict_proba(x)[:,1] for x in xs]))) for p,xs in vv.items()};score=float(np.mean(list(aucs.values())))
                print('FUSION',kind,c,aucs,flush=True)
                if best is None or score>best['val_auc']:best={'kind':kind,'model':net,'C':c,'val_auc':score,'validation_policy_aucs':aucs}
        best['selection']={'kind':kind,'C':best['C'],'validation_global_auc_mean_policies':best['val_auc'],'validation_policy_aucs':best['validation_policy_aucs']}
        path.write_bytes(pickle.dumps(best));selection.append(best['selection'])
    (ROOT/'results/fusion_selection.json').write_text(json.dumps({'methods':selection,'selection':'mean validation global token AUC over active and random policies',
        'training_includes':'both policies equally, all response tokens; original token labels only',
        'unqueried_tokens':'remain in evaluation; supplementary feature values masked to zero; common fitted base-score calibration',
        'test_used_for_selection':False},indent=2),encoding='utf8');print('FUSION FIT COMPLETE',flush=True)

if __name__=='__main__':main()
