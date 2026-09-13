"""CPU-only HARP-inspired bottom singular subspace of actual BF16 output head."""
from pathlib import Path
import importlib.util,time
import numpy as np
import torch
from safetensors import safe_open
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';OUT=DATA/'harp_features'
spec=importlib.util.spec_from_file_location('harp_basis21',ROOT.parent/'round21_semantic_internal_probe/src/extract21.py')
e21=importlib.util.module_from_spec(spec);spec.loader.exec_module(e21)
base=e21.base

def run():
    assert not torch.cuda.is_initialized()
    model=ROOT.parent/'models/Llama-2-7b-chat-hf';idx=base.read(model/'model.safetensors.index.json')
    filename=idx['weight_map']['lm_head.weight'];asset=model/filename
    download=base.read(ROOT/'model_download_manifest.json');record=next(x for x in download['files'] if x['filename']==filename)
    assert download['all_files_source_hash_matched'] and base.sha(asset)==record['actual_sha256']
    fm=base.read(DATA/'feature_manifest.json');load=base.read(DATA/'feature_signature.json')
    assert fm['complete'] and fm['completed_records']==793 and load['load_config']['torch_dtype']=='bfloat16'
    assert 'lm_head' in load['load_config']['llm_int8_skip_modules']
    sig={'version':'qa-harp-bottom256-runtime-bf16-head-v1','code_sha256':base.sha(Path(__file__)),
         'numerical_helpers_sha256':base.sha(e21.ROOT/'src/extract21.py'),
         'source_feature_manifest_sha256':base.sha(DATA/'feature_manifest.json'),
         'head_shard_sha256':record['actual_sha256'],'head_key':'lm_head.weight','head_runtime_cast':'source tensor -> torch.bfloat16 -> float32 -> float64 Gram',
         'rank':256,'projection':'hidden_last float64 @ bottom right singular components.T -> float32',
         'no_data_fit':True,'no_labels_or_test_read':True,'source':'https://arxiv.org/html/2509.11536v2',
         'adaptation':'Projection only; no claim to reproduce complete HARP weak-supervision/training method'}
    frozen=DATA/'harp_signature.json'
    if frozen.exists():assert base.read(frozen)==sig
    else:base.save(frozen,sig)
    bp=DATA/'harp_basis.npz';mp=DATA/'harp_basis.json'
    if mp.exists():
        bm=base.read(mp);assert bm['signature_sha256']==base.model7.digest(sig) and bm['npz_sha256']==base.sha(bp)
        with np.load(bp) as z:c=z['components'].copy()
    else:
        started=time.perf_counter()
        with safe_open(str(asset),framework='pt',device='cpu') as f:
            sliced=f.get_slice('lm_head.weight');assert sliced.get_shape()==[32000,4096]
            gram=e21.gram_from_blocks(lambda a,b:sliced[a:b].to(torch.bfloat16).float().numpy(),32000,4096)
        ev,c,check=e21.bottom_basis(gram,256);assert check['passed']
        base.save_arrays(bp,{'components':c,'eigenvalues':ev})
        bm={'signature_sha256':base.model7.digest(sig),'npz_sha256':base.sha(bp),'check':check,'seconds':time.perf_counter()-started,
            'head_source_dtype':sliced.get_dtype(),'head_projected_dtype':'BF16','cuda_initialized':torch.cuda.is_initialized()}
        base.save(mp,bm);del gram
    assert c.shape==(256,4096) and c.dtype==np.float64
    OUT.mkdir(parents=True,exist_ok=True);records=[];started=time.perf_counter()
    for i,rec in enumerate(fm['records']):
        rid=rec['response_id'];src=DATA/'features'/(rid+'.npz');dst=OUT/(rid+'.npz');side=dst.with_suffix('.json')
        assert base.sha(src)==rec['npz_sha256']
        with np.load(src,allow_pickle=False) as z:
            h=z['hidden_last'];coords={k:z[k].copy() for k in ('token_ids','response_token_offsets','answer_token_positions')}
            projected=(h.astype(np.float64)@c.T).astype(np.float32)
        assert projected.shape==(rec['response_tokens'],256) and np.isfinite(projected).all()
        if side.exists():
            m=base.read(side);assert m['signature_sha256']==base.model7.digest(sig) and m['source_npz_sha256']==rec['npz_sha256']
            assert base.sha(dst)==m['npz_sha256']
            with np.load(dst) as z:assert np.array_equal(z['harp'],projected)
        else:
            base.save_arrays(dst,dict(harp=projected,**coords))
            m={'response_id':rid,'partition':rec['partition'],'source_npz_sha256':rec['npz_sha256'],'npz_sha256':base.sha(dst),
               'basis_sha256':base.sha(bp),'signature_sha256':base.model7.digest(sig),'shape':list(projected.shape),'no_labels_or_test_read':True}
            base.save(side,m)
        records.append({'response_id':rid,'partition':rec['partition'],'npz_sha256':base.sha(dst),'metadata_sha256':base.sha(side),'response_tokens':rec['response_tokens']})
        if (i+1)%100==0:print('QA_HARP',i+1,793,flush=True)
    assert not torch.cuda.is_initialized()
    base.save(DATA/'harp_manifest.json',{'complete':True,'records':records,'record_count':793,'token_count':sum(x['response_tokens'] for x in records),
               'signature_sha256':base.model7.digest(sig),'basis_sha256':base.sha(bp),'basis_seconds':bm['seconds'],'projection_seconds':time.perf_counter()-started,
               'labels_or_test_read':False,'cuda_initialized':False})
    print('QA_HARP_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):torch.set_num_threads(4);run()
