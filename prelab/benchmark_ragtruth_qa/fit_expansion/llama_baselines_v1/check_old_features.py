"""Bounded CPU checks of unchanged fit engineering samples; no fit or GPU."""
from pathlib import Path
import json,pickle,sys
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT.parent))
import run_llama_baselines as r
q=r.q

def main():
    assert q.read(OUT/'protocol.json')==r.protocol()
    old=q.metadata();plans=q.lines(r.ROOT/'data/feature_preparation/plans.jsonl');by={p['response_id']:p for p in plans}
    ordered=sorted([p for p in plans if p['partition']=='fit'],key=lambda p:(len(p['original']['input_ids']),p['response_id']))
    fm,_=q.feature_snapshot();cm=q.read(r.CONTROL/'feature_manifest.json');cr={x['response_id']:x for x in cm['records']}
    hm=q.read(r.ROOT/'data/harp_manifest.json');hr={x['response_id']:x for x in hm['records']}
    with np.load(r.ROOT/'data/harp_basis.npz') as z:basis=z['components']
    pca=pickle.loads((q.OUT/'hidden_pca.pkl').read_bytes())
    matrices={k:np.load(p,mmap_mode='r') for k,p in {'base':q.OUT/'matrices/base.npy','pca':q.OUT/'matrices/hidden.npy','harp':r.ROOT/'results/harp_development_v1/harp_windows.npy',r.PRE:r.ROOT/'results/lookback_controls_v2/matrices'/(r.PRE+'.npy'),r.POST:r.ROOT/'results/lookback_controls_v2/matrices'/(r.POST+'.npy')}.items()}
    checks=[];files={}
    for p in (ordered[0],ordered[-1]):
        rid=p['response_id'];a=q.load_features(rid,old,fm,by);t=old['by_response'][rid]['tokens']
        cp=r.CONTROL/'features'/(rid+'.npz');assert q.sha(cp)==cr[rid]['npz_sha256'];files[str(cp.resolve())]=q.sha(cp)
        with np.load(cp) as z:
            a[r.PRE]=z[r.PRE];a[r.POST]=z[r.POST];assert np.array_equal(a['lb'],z['lb_source_post_legacy'])
        r.validate(a,t)
        hp=(a['hidden_last'].astype(np.float64)@basis.T).astype(np.float32)
        pp=((a['hidden_last'].astype(np.float64)-pca['mean'])@pca['components'].T).astype(np.float32)
        path=r.ROOT/'data/harp_features'/(rid+'.npz');assert q.sha(path)==hr[rid]['npz_sha256'];files[str(path.resolve())]=q.sha(path)
        with np.load(path) as z:assert np.array_equal(hp,z['harp'])
        for j in old['answer_windows'][rid]:
            ix=old['windows'][j]['token_indices'];got=r.pool(a,ix,hp[:,:64],pp)
            for name,m in matrices.items():assert np.array_equal(got[name],m[j,:64] if name=='harp' else m[j]),(rid,j,name)
        checks.append({'response_id':rid,'selection':'min/max fullinput length among original634 fit, no label selection','raw_first_offset':t['response_token_offsets_raw'][0][0],'input_tokens':len(p['original']['input_ids']),'raw_answer_tokens':t['token_count'],'all_eligible_windows':len(old['answer_windows'][rid]),'five_matrix_blocks_exact':True,'HARP_full256_exact':True,'first_token_retained':True})
    report={'passed':True,'checks':checks,'scope':'Only original fit engineering rows; not a replacement for mandatory all793 exact gate during formal full preparation. No3046 subset processed.','source_files_sha256':files,'code_sha256':q.sha(r.__file__),'helper_sha256':q.sha(__file__),'protocol_sha256':q.sha(OUT/'protocol.json'),'new_fits':0,'GPU_used':False,'test_read':False}
    q.save(OUT/'OLD_FEATURE_GATE_CHECK.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
