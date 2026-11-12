"""Four matched expanded Llama replay baselines; strict full-cohort CPU gate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,pickle,shutil,sys,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
import run_development as q
import run_probe_expansion as expansion
OUT=HERE/'llama_baselines_v1';FEATURE=HERE/'llama_features_v3';CONTROL=ROOT/'data/lookback_controls_v2'
METHODS=('prefix_pre_header','legacy_lb_nll','prefix_post_header','harp64_legacy_lb_nll')
CS=(1e-5,1e-4,.001,.01,.1);WIDTHS=(1024,1025,1024,1089);BATCH=16384
PRE='lb_prefix_pre_header';POST='lb_prefix_post_header'
NFIT,NCAL,NTOTAL=653979,42241,696220

def now():return datetime.now(timezone.utc).isoformat()
def freeze(p,v):
    if p.exists():assert q.read(p)==v,('Frozen expanded baseline changed',str(p))
    else:q.save(p,v)
def fixed_paths():
    return [Path(__file__),Path(q.__file__),Path(expansion.__file__),HERE/'data/export_freeze.json',
        HERE/'data/new_token_plans.jsonl',HERE/'data/answers_fit.jsonl',HERE/'data/tokens_fit.jsonl',HERE/'data/windows_k4_fit.jsonl',
        FEATURE/'signature.json',FEATURE/'protocol.json',FEATURE/'preparation_freeze.json',
        ROOT/'data/feature_manifest.json',ROOT/'data/feature_signature.json',ROOT/'data/feature_preparation/plans.jsonl',
        CONTROL/'feature_manifest.json',CONTROL/'signature.json',CONTROL/'layouts.jsonl',ROOT/'data/gold_manifest.json',
        ROOT/'data/harp_basis.npz',ROOT/'data/harp_basis.json',ROOT/'data/harp_manifest.json',ROOT/'data/harp_signature.json',
        q.OUT/'hidden_pca.pkl',q.OUT/'complete.json',ROOT/'results/harp_development_v1/complete.json',
        ROOT/'results/lookback_controls_v2/complete.json',expansion.OUT/'complete.json',expansion.OUT/'expanded3680_weights.npz']
def protocol():
    return {'version':'qa-expanded-llama-four-baselines-v1','methods':list(METHODS),'widths':dict(zip(METHODS,WIDTHS)),
        'C':list(CS),'seed':20260924,'fit_count':20,'frozen_old_comparisons_not_refitted':True,
        'cohort':{'fit_answers':3680,'native_fit_answers':634,'aux_fit_answers':3046,'fit_source_groups':615,'fit_windows':NFIT,'calibration_answers':159,'calibration_groups':154,'calibration_windows':NCAL},
        'features':{'prefix_pre_header':'Mean1024 LB across all raw tokens in each eligible4BPE window; full input prefix, prediction-before-current, real chat header.',
            'legacy_lb_nll':'Original source-only post-read no-header LB1024 window mean + NLL1 window mean.',
            'prefix_post_header':'Mean1024 LB; full input prefix, read-current state, same real chat header.',
            'harp64_legacy_lb_nll':'Same legacyLB+NLL1025 followed by mean frozen HARP bottom64 projection; dimension1089.'},
        'shared_projection':'Reuse frozen HARP bottom256 head basis, compute all256 in original order then keep first64. Also preserve frozen original fit-PCA64 projections as a nonpredictor cache for future controls. Neither projection is refit.',
        'PCA_used_in_these_four_classifiers':False,'harp_adaptation':'Projection only, not the original complete HARP weak-supervision method.',
        'cohort_gate':'Require frozen Llama v3 full3046 complete manifest, exact identities/plans/layouts/signature, per-array/JSON hashes and all raw coordinates before any formal preparation or fit. Partial cohorts are never fitted/reported.',
        'old_reuse':'Original634 fit and159 calibration cached LB/NLL/hidden and four Lookback definitions are read-only. Every original210364 window and HARP/PCA representation must match old caches exactly.',
        'base_weights':'Reuse audited probe_v1 expanded3680 weights: equal615 groups; within group native/aux eachhalf; within stratum equal answers then windows.',
        'loss_weights':'Reuse exact class factors then group-renormalized loss; total168123, identical to original634 training. No sample-count-induced C change.',
        'scaler':'Separate fit-only base-weighted StandardScaler per family; partial_fit blocks16384; transformed float32; one scaler shared by all five Cs.',
        'LR':'liblinear L2 max_iter2000 random_state20260924; four CPU threads; no class_weight beyond frozen explicit sample_weight.',
        'threshold_and_selection':'Same q calibration-only separate risk F1 thresholds, tieprecision then higherthreshold; select C by maxmin(windowF1,answerF1),windowF1,windowprecision,smallerC.',
        'geometry':'All653979 fit and42241 calibration windows. Mean all4 raw BPE tokens including punctuation; N<4 only actual tokens; unchanged human labels, QA safe-refusal policy. Answer=max every eligible window.',
        'provenance':'Original793 are published Llama-family responses. Auxiliary3046 answers from other generators are teacher-forced through one fixed Llama NF4 checkpoint; not their native internal trajectories.',
        'test_opened':False,'new_generation':False,'GPU_used_by_this_runner':False,'refit_PCA_or_head_basis':False,
        'development_limit':'Families and C budget chosen after earlier calibration experiments. This is development, not new independent test performance. Expanded3680 reuses615 source groups, not3680 independent questions.',
        'resource_estimate':{'matrix_raw_gib':NTOTAL*(1025+1024+1024+64)*4/2**30,'fit_standardized_gib':NFIT*sum(WIDTHS)*4/2**30,'projection_token_caches_gib':708506*128*4/2**30,'reserve_free_gib':30,'CPU_minutes_planning':[15,60],'not_measured':True},
        'files_sha256':{str(p.resolve()):q.sha(p) for p in fixed_paths()}}

def source(require_new=True):
    assert q.read(OUT/'protocol.json')==protocol()
    for section in ('source_files_sha256','output_files_sha256'):
        for p,h in q.read(HERE/'data/export_freeze.json')[section].items():assert q.sha(p)==h,p
    sig=q.read(FEATURE/'signature.json')
    for p,h in sig['files_sha256'].items():assert q.sha(p)==h,p
    manifest=FEATURE/'feature_manifest.json'
    if not manifest.exists():
        assert not require_new,'Full3046 Llama feature manifest missing: no subset preparation/fit'
        return None
    fm=q.read(manifest);assert fm['complete'] and fm['completed_count']==3046 and len(fm['records'])==3046
    assert fm['signature_sha256']==q.digest(sig) and not fm['labels_used'] and not fm['test_read']
    plans=q.lines(HERE/'data/new_token_plans.jsonl');layouts=q.lines(FEATURE/'layouts.jsonl')
    assert [p['response_id'] for p in plans]==[r['response_id'] for r in fm['records']]==sig['response_ids_in_order']
    assert len({r['response_id'] for r in fm['records']})==3046
    assert len(layouts)==3046
    for p,lo,r in zip(plans,layouts,fm['records']):
        rid=p['response_id'];path=(FEATURE/'features'/(rid+'.npz')).resolve();side=path.with_suffix('.json')
        assert Path(r['npz']).resolve()==path and Path(r['json']).resolve()==side
        assert q.sha(path)==r['npz_sha256']==r['arrays_sha256'] and q.sha(side)==r['json_sha256']
        m=q.read(side)
        for k in ('response_id','source_id','group_id','partition'):assert m[k]==r[k]==p[k]
        assert m['partition']=='fit' and m['complete'] and not m['labels_used'] and not m['test_read']
        assert m['signature_sha256']==r['signature_sha256']==q.digest(sig)
        assert m['plan_sha256']==r['plan_sha256']==q.digest(p) and m['layout_sha256']==r['layout_sha256']==q.digest(lo)
        assert m['arrays_sha256']==r['npz_sha256']
    oldfm,_=q.feature_snapshot();cm=q.read(CONTROL/'feature_manifest.json');cs=q.read(CONTROL/'signature.json')
    assert cm['complete'] and cm['completed_count']==793 and cm['signature_sha256']==q.digest(cs)
    assert cm['all_old_anchors_exact'] and not cm['labels_used'] and not cm['test_read']
    paths=fixed_paths()+[manifest,OUT/'protocol.json']
    snapshot={'files_sha256':{str(p.resolve()):q.sha(p) for p in paths},'new_manifest_complete':True,'old793_unchanged':True,'official_test_opened':False}
    return plans,fm,oldfm,cm,snapshot

def validate(a,t):
    n=t['token_count']
    for k,w in [('lb',1024),('nll',None),('hidden_last',4096),(PRE,1024),(POST,1024)]:
        assert a[k].shape==((n,) if w is None else (n,w)) and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
        if k in ('lb',PRE,POST):assert np.all((a[k]>=0)&(a[k]<=1))
    assert np.all(a['nll']>=0)
    for k in ('token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw'):assert a[k].tolist()==t[k]

def pool(a,ix,harp,pca):
    return {'base':np.r_[a['lb'][ix].mean(0),a['nll'][ix].mean()],PRE:a[PRE][ix].mean(0),POST:a[POST][ix].mean(0),'harp':harp[ix].mean(0),'pca':pca[ix].mean(0)}

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    old,meta=expansion.metadata()
    with np.load(expansion.OUT/'expanded3680_weights.npz') as z:
        assert len(z['y'])==NFIT and abs(z['loss'].sum()-168123)<1e-6
        assert np.array_equal(z['y'],[w['label'] for w in meta['windows'][:NFIT]])
    # Coordinate-free arithmetic checks do not load any new real feature subset.
    a={'lb':np.arange(6*1024,dtype=np.float32).reshape(6,1024),'nll':np.arange(6,dtype=np.float32)}
    a[PRE]=a['lb']+1;a[POST]=a['lb']+2;h=np.arange(6*64,dtype=np.float32).reshape(6,64)
    for ix in ([0,1,2,3],[0,1]):
        v=pool(a,ix,h,h);assert np.array_equal(v['base'][:1024],sum(a['lb'][j] for j in ix)/len(ix))
        assert v['base'][-1]==sum(a['nll'][j] for j in ix)/len(ix)
        assert np.array_equal(v['harp'],sum(h[j] for j in ix)/len(ix))
    cfg=protocol();freeze(OUT/'protocol.json',cfg)
    freeze(OUT/'PREFLIGHT.json',{'passed':True,'full_metadata_geometry_verified':True,'weights_from_audited_expansion':True,'raw4_punctuation_slots_and_short_window_checked':True,'real_features_prepared':False,'new_fits':0,'GPU_used':False,'test_opened':False,'code_sha256':q.sha(__file__),'protocol_sha256':q.sha(OUT/'protocol.json')})
    print('LLAMA_BASELINES_PROTOCOL_READY_NO_TRAINING',flush=True)

def prepare():
    upstream=source(require_new=False)
    if upstream is None:
        print('WAITING_FOR_COMPLETE_3046_LLAMA_FEATURES_NO_PREP_OR_FIT',flush=True);return False
    plans,fm,oldfm,cm,snap=upstream
    if (OUT/'preparation_complete.json').exists():
        done=q.read(OUT/'preparation_complete.json');assert done['source_snapshot']==snap
        for n,h in done['files_sha256'].items():assert q.sha(OUT/n)==h
        return True
    assert not (OUT/'preparation_started.json').exists(),'Interrupted preparation requires separate diagnosed revision; never use partial matrices'
    assert shutil.disk_usage(OUT).free>30*2**30
    freeze(OUT/'preparation_started.json',{'utc':now(),'source_snapshot':snap})
    original,meta=expansion.metadata();nr={r['response_id']:r for r in fm['records']};cr={r['response_id']:r for r in cm['records']}
    oldplans={p['response_id']:p for p in q.lines(ROOT/'data/feature_preparation/plans.jsonl')};newplans={p['response_id']:p for p in plans}
    cl={p['response_id']:p for p in q.lines(CONTROL/'layouts.jsonl')}
    hp=q.read(ROOT/'data/harp_manifest.json');hr={r['response_id']:r for r in hp['records']}
    with np.load(ROOT/'data/harp_basis.npz') as z:basis=z['components'].copy()
    pca=pickle.loads((q.OUT/'hidden_pca.pkl').read_bytes());assert basis.shape==(256,4096) and pca['components'].shape==(64,4096)
    assert {s['response_id'] for s in pca['sample']}=={a['response_id'] for a in original['answers'][:634]}
    folder=OUT/'matrices';folder.mkdir(exist_ok=True)
    specs={'base':1025,PRE:1024,POST:1024,'harp':64}
    mats={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=(NTOTAL,d)) for k,d in specs.items()}
    tokharp=np.lib.format.open_memmap(folder/'token_harp64.npy',mode='w+',dtype=np.float32,shape=(708506,64))
    tokpca=np.lib.format.open_memmap(folder/'token_pca64.npy',mode='w+',dtype=np.float32,shape=(708506,64))
    oldbase=np.load(q.OUT/'matrices/base.npy',mmap_mode='r');oldpca=np.load(q.OUT/'matrices/hidden.npy',mmap_mode='r')
    oldharp=np.load(ROOT/'results/harp_development_v1/harp_windows.npy',mmap_mode='r')
    oldpre=np.load(ROOT/'results/lookback_controls_v2/matrices'/(PRE+'.npy'),mmap_mode='r');oldpost=np.load(ROOT/'results/lookback_controls_v2/matrices'/(POST+'.npy'),mmap_mode='r')
    index=[];cursor=0;oldwindows=0
    for i,(answer,t) in enumerate(zip(meta['answers'],meta['tokens'])):
        rid=answer['response_id'];n=t['token_count']
        if rid in oldplans:
            a=q.load_features(rid,original,oldfm,oldplans)
            path=CONTROL/'features'/(rid+'.npz');rec=cr[rid];side=q.read(path.with_suffix('.json'))
            assert q.sha(path)==rec['npz_sha256']==side['npz_sha256'] and q.sha(path.with_suffix('.json'))==rec['json_sha256']
            assert side['signature_sha256']==cm['signature_sha256'] and side['plan_sha256']==q.digest(oldplans[rid]) and side['layout_sha256']==q.digest(cl[rid])
            with np.load(path) as z:
                for k in ('token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw'):assert np.array_equal(z[k],a[k])
                assert np.array_equal(z['lb_source_post_legacy'],a['lb']);a[PRE]=z[PRE].copy();a[POST]=z[POST].copy()
        else:
            path=FEATURE/'features'/(rid+'.npz');rec=nr[rid];assert q.sha(path)==rec['npz_sha256']
            with np.load(path) as z:a={k:z[k].copy() for k in z.files}
            assert newplans[rid]['source_id']==answer['source_id'] and newplans[rid]['group_id']==answer['group_id']
        validate(a,t)
        harp=(a['hidden_last'].astype(np.float64)@basis.T).astype(np.float32)[:,:64]
        projected=((a['hidden_last'].astype(np.float64)-pca['mean'])@pca['components'].T).astype(np.float32)
        if rid in oldplans:
            path=ROOT/'data/harp_features'/(rid+'.npz');assert q.sha(path)==hr[rid]['npz_sha256']
            with np.load(path) as z:assert np.array_equal(harp,z['harp'][:,:64])
        tokharp[cursor:cursor+n]=harp;tokpca[cursor:cursor+n]=projected
        for j in meta['answer_windows'][rid]:
            v=pool(a,meta['windows'][j]['token_indices'],harp,projected)
            for k in specs:mats[k][j]=v[k]
            if rid in oldplans:
                oldj=j if j<168123 else j-NFIT+168123
                assert np.array_equal(v['base'],oldbase[oldj]) and np.array_equal(v['pca'],oldpca[oldj])
                assert np.array_equal(v[PRE],oldpre[oldj]) and np.array_equal(v[POST],oldpost[oldj]) and np.array_equal(v['harp'],oldharp[oldj,:64]);oldwindows+=1
        index.append({'response_id':rid,'partition':answer['partition'],'group_id':answer['group_id'],'left':cursor,'right':cursor+n});cursor+=n
        if (i+1)%100==0:print('LLAMA_BASELINE_MATRICES',i+1,3839,flush=True)
    assert cursor==708506 and oldwindows==210364
    for a in (*mats.values(),tokharp,tokpca):a.flush()
    q.save(OUT/'token_index.json',{'answers':index})
    assert source(require_new=True)[-1]==snap
    files=['token_index.json']+[str(p.relative_to(OUT)) for p in folder.glob('*.npy')]
    freeze(OUT/'preparation_complete.json',{'source_snapshot':snap,'files_sha256':{n:q.sha(OUT/n) for n in files},'old793_all_windows_exact':True,'total_tokens':cursor,'total_windows':NTOTAL,'official_test_opened':False})
    print('LLAMA_BASELINE_PREPARATION_COMPLETE',flush=True);return True

def fit():
    snap=source(require_new=True)[-1];prep=q.read(OUT/'preparation_complete.json');assert prep['source_snapshot']==snap
    for n,h in prep['files_sha256'].items():assert q.sha(OUT/n)==h,n
    assert not (OUT/'fit_started.json').exists()
    freeze(OUT/'fit_started.json',{'utc':now(),'preparation_sha256':q.sha(OUT/'preparation_complete.json'),'protocol_sha256':q.sha(OUT/'protocol.json')})
    original,meta=expansion.metadata();folder=OUT/'matrices';base=np.load(folder/'base.npy',mmap_mode='r');harp=np.load(folder/'harp.npy',mmap_mode='r')
    pre=np.load(folder/(PRE+'.npy'),mmap_mode='r');post=np.load(folder/(POST+'.npy'),mmap_mode='r')
    with np.load(expansion.OUT/'expanded3680_weights.npz') as z:w={k:z[k].copy() for k in z.files}
    assert np.array_equal(w['y'],[r['label'] for r in meta['windows'][:NFIT]]) and abs(w['loss'].sum()-168123)<1e-6
    assert np.array_equal(w['base'],expansion.expanded_weights(meta)[0])
    (OUT/'training_weights.npz').write_bytes((expansion.OUT/'expanded3680_weights.npz').read_bytes())
    files=['training_weights.npz'];families={};selected={};allstart=time.perf_counter()
    for method,width in zip(METHODS,WIDTHS):
        def raw(l,r):
            if method=='prefix_pre_header':return np.asarray(pre[l:r])
            if method=='prefix_post_header':return np.asarray(post[l:r])
            if method=='legacy_lb_nll':return np.asarray(base[l:r])
            return np.column_stack((base[l:r],harp[l:r]))
        sc=StandardScaler()
        for l in range(0,NFIT,BATCH):r=min(l+BATCH,NFIT);sc.partial_fit(raw(l,r),sample_weight=w['base'][l:r])
        zp=folder/(method+'_fit_standardized.npy');zfit=np.lib.format.open_memmap(zp,mode='w+',dtype=np.float32,shape=(NFIT,width))
        for l in range(0,NFIT,BATCH):r=min(l+BATCH,NFIT);zfit[l:r]=sc.transform(raw(l,r)).astype(np.float32)
        zfit.flush();zhash=q.sha(zp);files.append(str(zp.relative_to(OUT)));entries=[]
        for c in CS:
            begin=time.perf_counter();model=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=20260924)
            model.fit(zfit,w['y'],sample_weight=w['loss']);assert model.n_iter_.max()<2000
            scores=np.empty(NTOTAL,np.float64)
            for l in range(0,NTOTAL,BATCH):r=min(l+BATCH,NTOTAL);scores[l:r]=model.predict_proba(sc.transform(raw(l,r)).astype(np.float32))[:,1]
            answers=q.answer_scores(meta,scores)
            ts={'window':q.choose_threshold([r['label'] for r in meta['windows'][NFIT:]],scores[NFIT:]),'answer':q.choose_threshold([r['label'] for r in meta['answers'][3680:]],answers[3680:])}
            name=f'{method}_C{c:g}';obj={'model':model,'scaler':sc,'method':method,'width':width,'C':c,'thresholds':ts,'selection_key':q.selection_key(ts,c),'fit_only':True,'fit_rows':NFIT,'fit_answers':3680,'fit_groups':615,'weights_sha256':q.sha(OUT/'training_weights.npz'),'fit_matrix_sha256':zhash,'protocol_sha256':q.sha(OUT/'protocol.json')}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=scores,answer_scores=answers)
            entry={'candidate':name,'method':method,'C':c,'thresholds':ts,'selection_key':list(q.selection_key(ts,c)),'metrics':q.metrics(meta,scores,ts),'model_sha256':q.sha(OUT/(name+'.pkl')),'scores_sha256':q.sha(OUT/(name+'_scores.npz')),'iterations':model.n_iter_.tolist(),'seconds':time.perf_counter()-begin}
            q.save(OUT/(name+'_result.json'),entry);entries.append(entry);files.extend([name+'.pkl',name+'_scores.npz',name+'_result.json']);print('LLAMA_EXPANDED_FIT',name,round(entry['seconds'],1),flush=True)
        families[method]=entries;selected[method]=max(entries,key=lambda e:e['selection_key']);del zfit;gc.collect()
    assert source(require_new=True)[-1]==snap
    q.save(OUT/'summary.json',{'selected':selected,'all_candidates':families,'fit_count':20,'fit_answers':3680,'calibration_answers':159,'official_test_opened':False,'development_selection_optimistic':True,'seconds':time.perf_counter()-allstart})
    files+=['summary.json','protocol.json','PREFLIGHT.json','preparation_complete.json','fit_started.json']
    freeze(OUT/'complete.json',{'utc':now(),'files_sha256':{n:q.sha(OUT/n) for n in files},'fit_count':20,'official_test_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','prepare','fit','all']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':initialize()
        elif a.stage=='prepare':prepare()
        elif a.stage=='fit':fit()
        elif prepare():fit()
