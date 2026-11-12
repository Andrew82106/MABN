"""Train-only feature transforms; labels never enter PCA or standardization."""
import pickle,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from shared import ROOT,data,features,readl

def main():
    path=ROOT/'data/inputs.pkl'
    if path.exists():print('INPUTS EXIST');return
    dd=data(); anns={a['id']:a for a in readl(ROOT/'data/annotations.jsonl')}
    ff={s:[dict(np.load(features(r))) for r in rows] for s,rows in dd.items()}
    n=sum(len(f['after']) for f in ff['train']); idx=np.random.default_rng(20260910).choice(n,min(n,8000),replace=False)
    pcas=[]; hybrid={s:[] for s in dd}
    with threadpool_limits(limits=6):
        for layer in range(4):
            xx=np.concatenate([f['after'][:,layer,:] for f in ff['train']]).astype(np.float32)
            pca=PCA(n_components=64,svd_solver='randomized',random_state=42).fit(xx[idx]);pcas.append(pca);del xx
            for s in dd:
                for i,f in enumerate(ff[s]):
                    if layer==0:hybrid[s].append([])
                    hybrid[s][i].append(pca.transform(f['after'][:,layer,:].astype(np.float32)))
            print('PCA',layer,float(pca.explained_variance_ratio_.sum()),flush=True)
    out={};scalers={}
    for kind in ['attention','hybrid']:
        arrays={s:[f['lookback'].astype(np.float32) if kind=='attention' else np.column_stack([f['lookback'],f['support'],*hybrid[s][i]]) for i,f in enumerate(ff[s])] for s in dd}
        scaler=StandardScaler().fit(np.concatenate(arrays['train']));scalers[kind]=scaler
        out[kind]={s:[scaler.transform(x).astype(np.float32) for x in arr] for s,arr in arrays.items()}
    yy={s:[np.array([int(any(lo<a['end'] and hi>a['start'] for a in anns[r['id']]['spans'])) for lo,hi in f['offsets']],np.float32) for r,f in zip(dd[s],ff[s])] for s in dd}
    result={'rows':dd,'x':out,'y':yy,'offsets':{s:[f['offsets'] for f in fs] for s,fs in ff.items()},'pcas':pcas,'scalers':scalers}
    path.write_bytes(pickle.dumps(result,protocol=5))
    (ROOT/'results/input_transform.json').write_text(json.dumps({'training_ids':[r['id'] for r in dd['train']],
        'pca_sample_tokens':len(idx),'pca_variance':[float(p.explained_variance_ratio_.sum()) for p in pcas],
        'features':{k:out[k]['train'][0].shape[1] for k in out},'fit_scope':'training tokens only; PCA no labels; test transformations frozen'},indent=2),encoding='utf8')
    print('INPUTS COMPLETE',flush=True)

if __name__=='__main__':main()
