"""Independent fixed-HARP numerical audit; no project imports/fit/GPU/SVD.

Reuses only our independent development audit's metadata and weight oracle.
Only named fit/calibration exports, their derived features and the fixed public
checkpoint output-head tensor are read. All outputs are new audit artifacts.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import traceback
from datetime import datetime, timezone
import numpy as np
from safetensors import safe_open
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT/'data'
PRIOR = OUT.parent/'development_v1'
REPORT = OUT/'INDEPENDENT_NUMERIC_AUDIT_HARP_QA.json'
REUSED = PRIOR/'audit_coefficients_qa.py'
assert hashlib.sha256(REUSED.read_bytes()).hexdigest() == 'a7a25ec0c346ad32621769762ff97b7f81c7caa6d959658a88508fb4ab5c1dae'
spec = importlib.util.spec_from_file_location('independent_qa_audit_helpers',REUSED)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
sha,read,near = q.sha,q.read,q.near
METHODS = ('harp256','harp64_lookback_nll','harp256_lookback_nll')
WIDTH = dict(zip(METHODS,(256,1089,1281)))
CS=(.001,.01,.1)
NFIT,NCAL,BATCH=168123,42241,16384


def digest(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(report):
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def design(base,harp,method,left,right):
    if method=='harp256':return np.asarray(harp[left:right])
    return np.column_stack((base[left:right],harp[left:right,:64 if method=='harp64_lookback_nll' else 256]))


def moments(base,harp,method,weight):
    mean=np.zeros(WIDTH[method]);var=mean.copy();count=0.
    for left in range(0,NFIT,BATCH):
        right=min(left+BATCH,NFIT)
        x=design(base,harp,method,left,right).astype(np.float64)
        w=weight[left:right].astype(np.float32).astype(np.float64)
        n=w.sum();center=w@x/n;delta=x-center;correction=w@delta
        m2=w@(delta*delta)-correction*correction/n
        count=float(np.float32(count));total=count+n
        var=(count*var+m2+(center-mean)**2*count*n/total)/total
        mean=(count*mean+n*center)/total;count=total
    eps=np.finfo(float).eps;constant=var<=count*eps*var+(count*mean*eps)**2
    scale=np.sqrt(var);scale[constant]=1.
    return mean,var,scale,count


def coefficient_audit(base,harp,answers,indices,weight,snapshot,report):
    families={}
    for method in METHODS:
        objects=[q.unpickle(OUT/f'{method}_C{c:g}.pkl') for c in CS]
        first=objects[0]['scaler']
        mean,var,scale,count=moments(base,harp,method,weight)
        result={'width':WIDTH[method],
                'scaler_mean_max_abs':near(mean,first.mean_,method+' mean',atol=2e-11),
                'scaler_var_max_abs':near(var,first.var_,method+' variance',atol=2e-10),
                'scaler_var_max_relative':float(np.max(np.abs(var-first.var_)/np.maximum(np.abs(first.var_),1e-20))),
                'scaler_scale_max_abs':near(scale,first.scale_,method+' scale',atol=2e-11),
                'scaler_sample_weight_count':float(first.n_samples_seen_),
                'scaler_count_max_abs':near(count,first.n_samples_seen_,method+' count'),
                'fit_standardized_rows_exact':True,'same_scaler_all_C':True,'candidates':[]}
        saved_fit=np.load(OUT/(method+'_fit.npy'),mmap_mode='r',allow_pickle=False)
        assert saved_fit.shape==(NFIT,WIDTH[method]) and saved_fit.dtype==np.float32
        scores=[]
        for c,obj in zip(CS,objects):
            assert obj['fit_only'] and obj['C']==c and obj['method']==method and obj['width']==WIDTH[method]
            assert obj['snapshot']==snapshot
            for key in ('mean_','var_','scale_','n_samples_seen_'):
                assert np.array_equal(getattr(first,key),getattr(obj['scaler'],key))
            model=obj['model']
            assert model.coef_.shape==(1,WIDTH[method]) and model.intercept_.shape==(1,)
            assert np.array_equal(model.classes_,[0,1]) and model.C==c and model.solver=='liblinear'
            assert model.penalty=='l2' and model.class_weight is None and model.random_state==20260927
            assert model.max_iter==2000 and model.n_iter_.max()<2000
            scores.append(np.empty(NFIT+NCAL))
        for left in range(0,NFIT+NCAL,BATCH):
            right=min(left+BATCH,NFIT+NCAL)
            z=design(base,harp,method,left,right).copy()
            assert z.dtype==np.float32 and np.isfinite(z).all()
            z-=first.mean_;z/=first.scale_
            if left<NFIT:
                end=min(right,NFIT)
                assert np.array_equal(z[:end-left],saved_fit[left:end]),(method,'standardized')
            for j,obj in enumerate(objects):
                model=obj['model'];scores[j][left:right]=q.sigmoid((z@model.coef_.T+model.intercept_).ravel())
        for c,obj,s in zip(CS,objects,scores):
            with np.load(OUT/f'{method}_C{c:g}_scores.npz',allow_pickle=False) as z:
                old=z['window_scores'];old_a=z['answer_scores']
            assert old.shape==(NFIT+NCAL,) and old_a.shape==(793,)
            error=near(s,old,(method,c,'manual coefficient replay'),rtol=2e-12,atol=2e-12)
            av=np.asarray([max(s[indices[a['answer_id']]]) for a in answers])
            ae=near(av,old_a,(method,c,'all-window answer max'),rtol=2e-12,atol=2e-12)
            result['candidates'].append({'C':c,'iterations':obj['model'].n_iter_.tolist(),
                'window_scores_replayed':NFIT+NCAL,'window_max_abs':error,'answer_scores_replayed':793,'answer_max_abs':ae})
        families[method]=result;report['families']=families;write(report)
        print('HARP_QA_COEFFICIENT_FAMILY_PASSED',method,flush=True)


def basis_audit(hm):
    sig=read(DATA/'harp_signature.json');meta=read(DATA/'harp_basis.json')
    path=DATA/'harp_basis.npz'
    assert digest(sig)==hm['signature_sha256']==meta['signature_sha256']
    assert sha(path)==hm['basis_sha256']==meta['npz_sha256']
    assert sig['no_data_fit'] and sig['no_labels_or_test_read'] and sig['rank']==256
    assert sha(ROOT/'src/extract_harp.py')==sig['code_sha256']
    assert sha(ROOT.parent/'round21_semantic_internal_probe/src/extract21.py')==sig['numerical_helpers_sha256']
    assert sha(DATA/'feature_manifest.json')==sig['source_feature_manifest_sha256']
    with np.load(path,allow_pickle=False) as z:c=z['components'].copy();ev=z['eigenvalues'].copy()
    assert c.shape==(256,4096) and c.dtype==np.float64 and ev.shape==(256,)
    assert np.all(np.diff(ev)>=0) and ev[0]>0
    orth=near(c@c.T,np.eye(256),'HARP direction orthogonality',atol=1e-11)
    signs=c[np.arange(256),np.argmax(np.abs(c),axis=1)]
    assert np.all(signs>0)
    modeldir=ROOT.parent/'models/Llama-2-7b-chat-hf'
    idx=read(modeldir/'model.safetensors.index.json')
    headpath=modeldir/idx['weight_map']['lm_head.weight']
    assert sha(headpath)==sig['head_shard_sha256']
    load=read(DATA/'feature_signature.json')['load_config']
    assert load['torch_dtype']=='bfloat16' and 'lm_head' in load['llm_int8_skip_modules']
    # Independent uint-bit round-to-nearest-even BF16 cast, without torch/GPU.
    # Reconstruct W^T W solely to verify STORED eigenpairs, never compute a basis.
    gram=np.zeros((4096,4096),np.float64)
    with safe_open(str(headpath),framework='np') as handle:
        sl=handle.get_slice('lm_head.weight');assert sl.get_shape()==[32000,4096]
        for left in range(0,32000,1024):
            x=np.asarray(sl[left:min(left+1024,32000)],np.float32)
            assert np.isfinite(x).all()
            bits=x.view(np.uint32)
            rounded=(bits+np.uint32(0x7fff)+((bits>>16)&np.uint32(1)))&np.uint32(0xffff0000)
            bf=rounded.view(np.float32).astype(np.float64)
            gram+=bf.T@bf
    residual=gram@c.T-c.T*ev
    rel=float(np.max(np.linalg.norm(residual,axis=0)/np.linalg.norm(gram,'fro')))
    assert rel<1e-10
    trace=float(np.trace(gram))
    trace_error=near(trace,meta['check']['gram_trace_total_head_energy'],'output-head trace',atol=1e-7)
    energy=float(ev.sum()/trace)
    result={'direction_data_scope':'No fit/calibration data or labels: fixed runtime-BF16 output head W only.',
            'definition':'Ascending smallest right singular directions of W; stored C shape256x4096, h@C.T without centering/L2 normalization. First64 rows are bottom64.',
            'ascending_eigenvalues':True,'canonical_signs_positive':True,'orthogonality_max_abs':orth,
            'fixed_eigenpair_residual_relative_to_gram_frobenius':rel,'head_energy_trace_max_abs':trace_error,
            'bottom256_eigenvalue_range':[float(ev[0]),float(ev[-1])],
            'bottom64_eigenvalue_range':[float(ev[0]),float(ev[63])],'selected_energy_fraction':energy,
            'head_shard_sha256':sig['head_shard_sha256'],'basis_sha256':hm['basis_sha256'],
            'new_direction_or_svd_fit':False,
            'selection_evidence':'Extractor source calls eigh with subset_by_index=[0,255] and keeps ascending vectors; audit checks frozen eigenpairs and order without recomputing eigendecomposition.'}
    print('HARP_QA_FIXED_HEAD_EIGENPAIRS_PASSED',flush=True)
    return c,result


def feature_audit(c,hm,answers,windows,indices,harp):
    original=read(DATA/'feature_manifest.json')
    records={r['response_id']:r for r in original['records']}
    hrecords={r['response_id']:r for r in hm['records']}
    assert len(hrecords)==len(hm['records'])==793 and set(hrecords)==set(records)=={a['response_id'] for a in answers}
    maximum,window_error,total_tokens=0.,0.,0
    exact,window_exact=True,True
    for number,a in enumerate(answers,1):
        rid=a['response_id'];rec=records[rid];hr=hrecords[rid]
        assert rec['partition']==hr['partition']==a['partition']
        assert rec['response_tokens']==hr['response_tokens']==a['token_count']
        src=DATA/'features'/(rid+'.npz');dst=DATA/'harp_features'/(rid+'.npz');side=dst.with_suffix('.json')
        assert sha(src)==rec['npz_sha256'] and sha(dst)==hr['npz_sha256'] and sha(side)==hr['metadata_sha256']
        sm=read(side)
        assert sm['response_id']==rid and sm['partition']==a['partition']
        assert sm['source_npz_sha256']==rec['npz_sha256'] and sm['npz_sha256']==hr['npz_sha256']
        assert sm['basis_sha256']==hm['basis_sha256'] and sm['signature_sha256']==hm['signature_sha256']
        assert sm['no_labels_or_test_read'] is True
        with np.load(src,allow_pickle=False) as z:
            hidden=z['hidden_last'];coords={k:z[k].copy() for k in ('token_ids','response_token_offsets','answer_token_positions')}
        with np.load(dst,allow_pickle=False) as z:
            h=z['harp']
            for k,v in coords.items():assert np.array_equal(z[k],v),(rid,k)
        projected=(hidden.astype(np.float64)@c.T).astype(np.float32)
        assert h.dtype==np.float32 and h.shape==(a['token_count'],256)
        err=near(projected,h,rid+' fixed HARP',rtol=1e-6,atol=2e-6)
        maximum=max(maximum,err);exact=exact and np.array_equal(projected,h)
        jj=indices[a['answer_id']]
        means=np.asarray([h[windows[j]['token_indices']].mean(0) for j in jj],np.float32)
        we=near(means,harp[jj],rid+' HARP window mean',rtol=0,atol=0)
        window_error=max(window_error,we);window_exact=window_exact and np.array_equal(means,harp[jj])
        total_tokens+=len(h)
        if number%100==0:print('HARP_QA_FIXED_FORWARD',number,793,flush=True)
    return {'answers':793,'raw_tokens':total_tokens,'eligible_windows':NFIT+NCAL,
            'all_token_features_exact':exact,'all_token_projection_max_abs':maximum,
            'all_window_means_exact':window_exact,'all_window_mean_max_abs':window_error,
            'source_partition_ids_offsets_positions_and_basis_bound':True}


def audit(report):
    complete=read(OUT/'complete.json');snapshot=complete['snapshot']
    assert complete['status']=='complete_development_only' and complete['test_opened'] is False
    assert snapshot==read(OUT/'started.json')
    assert snapshot['code_sha256']==sha(ROOT/'src/run_harp_development.py')
    assert snapshot['protocol_sha256']==sha(ROOT/'harp_development_protocol.json')
    assert snapshot['harp_manifest_sha256']==sha(DATA/'harp_manifest.json')
    assert snapshot['prior_complete_sha256']==sha(PRIOR/'complete.json')
    for name,h in complete['files_sha256'].items():assert sha(OUT/name)==h,name
    prior=read(PRIOR/'complete.json')
    for name,h in prior['files_sha256'].items():assert sha(PRIOR/name)==h,name
    mm=read(PRIOR/'matrix_manifest.json');assert sha(PRIOR/'matrices/base.npy')==mm['files_sha256']['base']
    previous_audit=read(PRIOR/'COEFFICIENT_AUDIT_QA.json');assert previous_audit['status']=='passed'
    assert previous_audit['source_sha256'][str((PRIOR/'complete.json').resolve())]==sha(PRIOR/'complete.json')
    answers,windows,indices=q.metadata()
    weight,report['weights']=q.audit_weights(windows)
    report['fit_keys_exact_prior_reuse']=True
    base=np.load(PRIOR/'matrices/base.npy',mmap_mode='r',allow_pickle=False)
    harp=np.load(OUT/'harp_windows.npy',mmap_mode='r',allow_pickle=False)
    assert base.shape==(NFIT+NCAL,1025) and harp.shape==(NFIT+NCAL,256)
    coefficient_audit(base,harp,answers,indices,weight,snapshot,report)
    hm=read(DATA/'harp_manifest.json')
    assert hm['complete'] and hm['record_count']==793 and hm['token_count']==213159
    assert hm['labels_or_test_read'] is False and hm['cuda_initialized'] is False
    c,report['directions']=basis_audit(hm)
    report['feature_replay']=feature_audit(c,hm,answers,windows,indices,harp)
    report['totals']={'LR_candidates':9,'window_scores_replayed':9*(NFIT+NCAL),'answer_scores_replayed':9*793,'new_models_fitted':0}
    report['source_sha256']={str(p.resolve()):sha(p) for p in (Path(__file__),REUSED,ROOT/'src/run_harp_development.py',ROOT/'harp_development_protocol.json',ROOT/'src/extract_harp.py',DATA/'harp_signature.json',DATA/'harp_manifest.json',DATA/'harp_basis.npz',OUT/'complete.json',PRIOR/'complete.json',PRIOR/'training_weights.npz',PRIOR/'fit_keys.json')}
    cal=OUT/'CALIBRATION_AUDIT_HARP_QA.json'
    if cal.exists():
        cr=read(cal)
        assert cr['status']=='passed'
        report['independent_calibration_audit']={'path':str(cal.resolve()),'sha256':sha(cal),'status':'passed'}


if __name__=='__main__':
    report={'status':'running','utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
            'no_new_fit':True,'no_gpu':True,'official_test_or_withheld_read':False,
            'method':'Inert sklearn pickle state; independent weight and weighted-moment formulas, float32 standardization and coefficient/sigmoid replay. Fixed HARP eigenpair/forward audit without eigendecomposition.',
            'limits':['Stored-coefficient replay does not rerun optimization or independently certify unrecorded external execution history.',
                      'Direction basis is fixed-head-derived, not fitted from training examples. Fit-only checks apply to weights/scalers/LR; HARP is projection-only adaptation.',
                      'Results remain calibration-selected development outcomes, not fresh official-test performance.']}
    try:
        with threadpool_limits(limits=4):audit(report)
        report['status']='passed'
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());write(report);raise
    write(report)
    print('HARP_QA_NUMERIC_AUDIT_PASSED',sha(REPORT),flush=True)
