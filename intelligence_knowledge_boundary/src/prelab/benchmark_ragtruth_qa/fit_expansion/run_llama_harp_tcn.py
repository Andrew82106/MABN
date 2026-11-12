"""Expanded HARP64 + legacy LB/NLL token TCN; CPU-only staged entry.

initialize: metadata, synthetic CPU checks and immutable protocol only.
prepare/train/all require the complete3046 Llama manifest AND the numerical
preparation committed by run_llama_baselines. They never start that producer.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import argparse,gc,hashlib,json,pickle,shutil,sys,time
import importlib.metadata
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
import run_sequence_full as full
import run_llama_baselines as producer

q=producer.q;expansion=producer.expansion
OUT=HERE/'harp_tcn_v1';METHOD='expanded_full_lb_harp64_tcn'
SEED,EPOCHS,BATCH,THREADS,WIDTH=20260926,30,8,4,1089
FIT_ANSWERS,CAL_ANSWERS,FIT_RAW,TOTAL_RAW=3680,159,665708,708506
FIT_LEXICAL,FIT_RISK,FIT_GROUPS,LOSS_MASS=560300,47398,615,139518
FIT_WINDOWS,CAL_WINDOWS=653979,42241
sha,read,save,digest=q.sha,q.read,q.save,q.digest

def now():return datetime.now(timezone.utc).isoformat()
def freeze_json(path,value):
    assert json.loads(json.dumps(value,ensure_ascii=False))==value
    if path.exists():assert read(path)==value,('Frozen expanded TCN artifact changed',str(path))
    else:save(path,value)
def cpu_only():
    assert not torch.cuda.is_initialized(),'This entry never initializes CUDA'
def array_digest(a):
    a=np.asarray(a);h=hashlib.sha256()
    h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()
def foundation_paths():
    return [Path(__file__),Path(full.__file__),Path(full.v1.__file__),Path(q.__file__),
      Path(producer.__file__),Path(expansion.__file__),HERE/'EXPANDED_HARP_TCN_PLAN.md',
      producer.OUT/'protocol.json',producer.OUT/'PREFLIGHT.json',
      HERE/'data/export_freeze.json',HERE/'data/new_token_plans.jsonl',
      HERE/'data/answers_fit.jsonl',HERE/'data/tokens_fit.jsonl',HERE/'data/windows_k4_fit.jsonl',
      ROOT/'data/gold_manifest.json',ROOT/'data/harp_basis.npz',ROOT/'data/harp_basis.json',
      ROOT/'data/harp_manifest.json',ROOT/'data/harp_signature.json',
      full.OUT/'protocol.json',full.OUT/'complete.json',full.OUT/'preparation_manifest.json',
      full.OLD/'fit_token_weights.npz']
def protocol():
    return {'version':'qa-expanded-harp-token-tcn-v1','method':METHOD,
      'authorization':'Root approved implementation and a future30epoch expanded CPU training run; initialize performs no formal preparation or training.',
      'input':'Every original raw Llama token: legacy source-post no-header LB1024, NLL1, frozen HARP bottom64;1089 dimensions in this order.',
      'HARP':'Consume numerical-preparation token_harp64 from run_llama_baselines: original bottom256 head projection, then first64 in frozen order. No PCA input or basis refit.',
      'reconstruction':'Auxiliary3046 released answers from five other generators are replayed through one fixed Llama NF4 model; these are not their native internal trajectories.',
      'cohort':{'fit_answers':FIT_ANSWERS,'native_answers':634,'auxiliary_answers':3046,'fit_groups':FIT_GROUPS,
        'fit_raw_tokens':FIT_RAW,'fit_lexical_tokens':FIT_LEXICAL,'fit_risk_tokens':FIT_RISK,
        'calibration_answers':CAL_ANSWERS,'calibration_groups':154,'total_raw_tokens':TOTAL_RAW,
        'fit_windows':FIT_WINDOWS,'calibration_windows':CAL_WINDOWS},
      'network':'Reuse run_sequence_full.FullTCN and its unchanged TokenTCN.forward:1089->64 GELU; two single Conv1d64 blocks k3 dilation1/2 with residual,GELU,dropout.2; output1.',
      'parameters':94529,'receptive_field_raw_tokens':7,'offline_future_tokens':True,
      'padding':'Each answer independent. All raw punctuation positions remain valid contextual input. Projection and each residual block mask padded states to0; punctuation and padding loss weights0.',
      'base_weights':'Each source-connected group equal. Within a group, native634 and auxiliary3046 strata each half when both exist; existing stratum gets all mass if alone. Then uniform answers and lexical tokens. Normalize total base to139518.',
      'loss_weights':'Fit-only class factors from lexical base weights; re-equalize615 group losses then normalize total139518. No label-based sample selection.',
      'minibatch_objective':'sum(loss_weight*BCE_logits)*3680/(actual_batch_answers*139518). Exactly460 batches of8 per epoch; no within-batch weight renormalization.',
      'scaler':'Fit-only lexical-base-weighted StandardScaler partial_fit over first665708 raw rows, fixed16384 blocks; punctuation base0. Transform all raw rows tofloat32. Separate expanded scaler, no calibration fitting.',
      'seed':SEED,'epochs':EPOCHS,'batch_answers':BATCH,'cpu_threads':THREADS,
      'shuffle':'One numpy default_rng(20260926),30 consecutive permutations of all3680 fit answer indices; every raw and lexical token covered once per epoch.',
      'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'all_parameters_including_bias':True},
      'update_budget':{'batches_per_epoch':460,'expanded_steps':13800,'old_fit_answers':634,'old_batches_per_epoch':80,'old_steps':2400,
        'interpretation':'Same30epochs, not equal total computation; no post-result schedule selection.'},
      'aggregation':'Unchanged4rawBPE stride1 windows. Max token probability only over lexical positions within each eligible window; then max all eligible windows for answer. N<4 uses actual raw positions.',
      'selection':'Separate calibration window/answer F1 thresholds via original q.choose_threshold. Epoch key=min(F1w,F1a),F1w,windowprecision,earlier epoch. Keep all30 epochs; no early stop.',
      'save':'Each epoch full fit/cal raw-token logits/probabilities, window/answer scores, model+optimizer+torch RNG, thresholds/metrics/coverage; no overwritten old results.',
      'strict_gate':'Before any numerical matrices or fitting, require producer full3046 feature manifest and committed numerical preparation, verify all source and output hashes. Never consume or report a partial cohort; never call producer.prepare.',
      'stages':'initialize is metadata and synthetic CPU checks only. prepare requires committed upstream; train requires own complete preparation; all returns WAITING without either when upstream missing.',
      'frozen_reference':'Old sequence_full_v2 full_lb_harp64_tcn stays unchanged; compare calibration only and disclose different fit size/update count.',
      'no_official_test':True,'no_new_generation':True,'GPU_used':False,'early_stop':False,'PCA_or_HARP_refit':False,
      'limitations':['One seed and repeated calibration checkpoint selection; not sealed-test evidence.',
        'Expanded3680 answers reuse615 groups; they are not3680 independent questions.',
        'RF7 uses neighboring future tokens and is offline.',
        'The earlier plan document said pending; this turn explicitly approved the fixed30epoch/13800step implementation.'],
      'resource_estimate':{'raw_plus_standardized_GiB':2*TOTAL_RAW*WIDTH*4/2**30,'reserve_free_GiB':12,'no_runtime_measurement_yet':True},
      'versions':{'torch':torch.__version__,'numpy':np.__version__,'sklearn':importlib.metadata.version('scikit-learn')},
      'files_sha256':{str(p.resolve()):sha(p) for p in foundation_paths()}}

def make_index(meta):
    index=[];cursor=0
    for a,t in zip(meta['answers'],meta['tokens']):
        assert a['response_id']==t['response_id'] and a['partition']==t['partition']
        n=t['token_count'];assert n==len(t['token_ids'])==len(t['lexical_mask'])==len(t['risk_mask'])
        assert n>0 and any(t['lexical_mask'])
        index.append({k:a[k] for k in ('response_id','answer_id','source_id','group_id','partition')}|
          {'left':cursor,'right':cursor+n,'token_count':n});cursor+=n
    return index

def token_weights(meta,index,native_ids,target_mass=LOSS_MASS):
    fit=[(e,t) for e,t in zip(index,meta['tokens']) if e['partition']=='fit']
    fit_raw=sum(e['token_count'] for e,t in fit)
    lexical=np.concatenate([np.asarray(t['lexical_mask'],bool) for e,t in fit])
    y=np.concatenate([np.asarray(t['risk_mask'],np.int64) for e,t in fit])
    assert set(np.unique(y))<={0,1} and not np.any(y[~lexical])
    tree=defaultdict(lambda:defaultdict(list));b=np.zeros(fit_raw,np.float64)
    for e,t in fit:
        assert e['right']<=fit_raw
        tree[e['group_id']][int(e['response_id'] in native_ids)].append(e)
    for strata in tree.values():
        for answers in strata.values():
            for e in answers:
                ix=np.arange(e['left'],e['right']);ix=ix[lexical[ix]];assert len(ix)>0
                b[ix]=1/(len(strata)*len(answers)*len(ix))
    b*=target_mass/b.sum();mass=np.bincount(y,weights=b,minlength=2);assert np.all(mass>0)
    factors=mass.sum()/(2*mass);loss=b*factors[y]
    for strata in tree.values():
        ix=np.concatenate([np.arange(e['left'],e['right']) for aa in strata.values() for e in aa])
        loss[ix]*=(target_mass/len(tree))/loss[ix].sum()
    loss*=target_mass/loss.sum()
    assert np.array_equal(b>0,lexical) and np.array_equal(loss>0,lexical)
    return {'base':b,'loss':loss,'class_factors':factors,'y':y,'lexical':lexical}

def metadata_state():
    cpu_only();original,meta=expansion.metadata();index=make_index(meta)
    assert len(index)==FIT_ANSWERS+CAL_ANSWERS and index[-1]['right']==TOTAL_RAW
    assert index[FIT_ANSWERS-1]['right']==FIT_RAW
    assert all(e['partition']=='fit' for e in index[:FIT_ANSWERS]) and all(e['partition']=='calibration' for e in index[FIT_ANSWERS:])
    native={a['response_id'] for a in original['answers'][:634]};assert native=={e['response_id'] for e in index[:634]}
    w=token_weights(meta,index,native)
    assert len(w['y'])==FIT_RAW and int(w['lexical'].sum())==FIT_LEXICAL and int(w['y'].sum())==FIT_RISK
    assert len(meta['windows'])==FIT_WINDOWS+CAL_WINDOWS and meta['bounds']=={'fit':[0,FIT_WINDOWS],'calibration':[FIT_WINDOWS,FIT_WINDOWS+CAL_WINDOWS]}
    groups=defaultdict(list)
    for e in index[:FIT_ANSWERS]:groups[e['group_id']].append(e)
    assert len(groups)==FIT_GROUPS and len({e['group_id'] for e in index[FIT_ANSWERS:]})==154
    assert not set(groups)&{e['group_id'] for e in index[FIT_ANSWERS:]}
    native_mass=0.;base_error=loss_error=stratum_error=0.
    for es in groups.values():
        mass=sum(w['base'][e['left']:e['right']].sum() for e in es)
        lm=sum(w['loss'][e['left']:e['right']].sum() for e in es)
        nm=sum(w['base'][e['left']:e['right']].sum() for e in es if e['response_id'] in native)
        base_error=max(base_error,abs(mass-LOSS_MASS/FIT_GROUPS));loss_error=max(loss_error,abs(lm-LOSS_MASS/FIT_GROUPS))
        stratum_error=max(stratum_error,abs(nm-mass/2));native_mass+=nm
    assert max(base_error,loss_error,stratum_error)<1e-8 and abs(native_mass/w['base'].sum()-.5)<1e-10
    rng=np.random.default_rng(SEED);order=np.stack([rng.permutation(FIT_ANSWERS) for _ in range(EPOCHS)])
    for row in order:assert np.array_equal(np.sort(row),np.arange(FIT_ANSWERS))
    assert order.shape==(30,3680) and FIT_ANSWERS%BATCH==0
    report={'fit_answers':FIT_ANSWERS,'calibration_answers':CAL_ANSWERS,'fit_groups':FIT_GROUPS,
      'fit_raw_tokens':FIT_RAW,'fit_lexical_tokens':FIT_LEXICAL,'fit_risk_tokens':FIT_RISK,'all_raw_tokens':TOTAL_RAW,
      'base_mass':float(w['base'].sum()),'loss_mass':float(w['loss'].sum()),'native_base_fraction':float(native_mass/w['base'].sum()),
      'base_group_mass_max_abs':float(base_error),'loss_group_mass_max_abs':float(loss_error),'half_stratum_mass_max_abs':float(stratum_error),
      'punctuation_weight_zero':True,'permutations':30,'batches_per_epoch':460,'planned_steps':13800,
      'weights_array_sha256':{k:array_digest(v) for k,v in w.items()},'index_sha256':digest(index),'order_array_sha256':array_digest(order)}
    return original,meta,index,w,order,report

def new_model():
    assert (full.SEED,full.EPOCHS,full.BATCH,full.WIDTH)==(SEED,EPOCHS,BATCH,WIDTH)
    model=full.new_model()
    assert isinstance(model,full.FullTCN) and sum(p.numel() for p in model.parameters())==94529
    return model
def objective(logits,y,w,fit_answers=FIT_ANSWERS,target_mass=LOSS_MASS):
    assert logits.shape==y.shape==w.shape and len(logits)>0
    return (F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()*(fit_answers/(len(logits)*target_mass))
def batch(chosen,x,index,weights=None):
    assert len(chosen)>0;length=max(index[int(i)]['token_count'] for i in chosen)
    xb=torch.zeros(len(chosen),length,WIDTH,dtype=torch.float32);mask=torch.zeros(len(chosen),length)
    yy=torch.zeros(len(chosen),length);ww=torch.zeros(len(chosen),length)
    for j,i in enumerate(chosen):
        e=index[int(i)];l,r=e['left'],e['right'];n=r-l
        assert n==e['token_count']
        xb[j,:n]=torch.from_numpy(np.array(x[l:r],dtype=np.float32,copy=True));mask[j,:n]=1
        if weights is not None:
            assert e['partition']=='fit' and r<=len(weights['y'])
            yy[j,:n]=torch.from_numpy(weights['y'][l:r].astype(np.float32));ww[j,:n]=torch.from_numpy(weights['loss'][l:r].astype(np.float32))
    return xb,mask,yy,ww

def synthetic_checks():
    cpu_only();model=new_model().eval();other=new_model().eval()
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    gen=torch.Generator().manual_seed(SEED+1);x=torch.randn(2,16,WIDTH,generator=gen);mask=torch.ones(2,16);mask[0,9:]=0
    with torch.no_grad():
        out=model(x,mask);changed=x.clone();changed[0,9:]=1000
        assert torch.allclose(model(changed,mask)[0,:9],out[0,:9],atol=2e-6,rtol=1e-6)
        assert torch.allclose(model(x[:1,:9],torch.ones(1,9))[0],out[0,:9],atol=2e-6,rtol=1e-6)
        changed=x.clone();changed[1]=1000;assert torch.equal(model(changed,mask)[0],out[0])
        changed=x.clone();changed[1,:4]+=100;assert torch.equal(model(changed,mask)[1,8],out[1,8])
        assert torch.equal(out[0,9:],torch.zeros(7))
    # Unequal raw lengths/lexical counts, two strata and two groups.
    toy={'answers':[],'tokens':[]};settings=[('n0','g0',[1,0,1],[0,0,1]),('n1','g1',[1,1],[0,1]),
      ('a0','g0',[1,0,0,1],[1,0,0,0]),('a1','g0',[1],[0]),('a2','g1',[1,0,1],[1,0,0])]
    for rid,g,lex,risk in settings:
        a={'response_id':rid,'answer_id':rid,'source_id':rid,'group_id':g,'partition':'fit'}
        toy['answers'].append(a);toy['tokens'].append({**a,'token_count':len(lex),'token_ids':list(range(len(lex))),'lexical_mask':lex,'risk_mask':risk})
    ti=make_index(toy);tw=token_weights(toy,ti,{'n0','n1'},target_mass=10)
    expected=np.zeros(len(tw['base']))
    for e,t in zip(ti,toy['tokens']):
        strata_answers=2 if e['group_id']=='g0' and e['response_id'].startswith('a') else 1
        local=np.flatnonzero(t['lexical_mask'])+e['left'];expected[local]=2.5/(strata_answers*len(local))
    assert np.allclose(tw['base'],expected,atol=1e-14,rtol=0)
    for g in ('g0','g1'):
        ix=np.concatenate([np.arange(e['left'],e['right']) for e in ti if e['group_id']==g])
        assert abs(tw['loss'][ix].sum()-5)<1e-12
    changed=json.loads(json.dumps(toy));changed['tokens'][0]['risk_mask']=[1,0,0]
    assert np.array_equal(token_weights(changed,ti,{'n0','n1'},10)['base'],tw['base'])
    synthetic=np.arange(ti[-1]['right']*WIDTH,dtype=np.float32).reshape(-1,WIDTH)/10000
    xx,valid,yy,ww=batch([0,2],synthetic,ti,tw)
    assert valid[0].tolist()==[1,1,1,0] and ww[0,1]==0 and ww[0,3]==0
    logits=torch.randn(2,4,generator=gen,requires_grad=True)
    got=objective(logits,yy,ww,fit_answers=5,target_mass=10)
    want=(ww*F.binary_cross_entropy_with_logits(logits,yy,reduction='none')).sum()*.25
    assert torch.equal(got,want);got.backward();assert (logits.grad[ww==0]==0).all()
    # Original aggregation helper is count-generic. Punctuation can carry
    # high probability but must not replace lexical-window maxima.
    tm={'answers':[{'response_id':'x'},{'response_id':'y'}],
      'windows':[{'response_id':'x','token_indices':[0,1,2,3]},{'response_id':'x','token_indices':[1,2,3,4]},{'response_id':'y','token_indices':[0,1]}],
      'by_response':{'x':{'tokens':{'lexical_mask':[1,0,1,0,1]}},'y':{'tokens':{'lexical_mask':[1,0]}}},
      'answer_windows':{'x':[0,1],'y':[2]}}
    ws,ans=full.v1.aggregate(tm,[{'response_id':'x','left':0},{'response_id':'y','left':5}],np.array([.1,.99,.2,.98,.7,.3,.97]))
    assert np.array_equal(ws,[.2,.7,.3]) and np.array_equal(ans,[.7,.3])
    del model,other;gc.collect()
    return {'passed':True,'network_class_reused':True,'initial_parameters_exact':True,'parameters':94529,
      'padding_raw_positions_answer_isolation_RF7_passed':True,'group_stratum_answer_lexical_weights_passed':True,
      'base_independent_of_risk_labels':True,'batch_objective_and_zero_punctuation_padding_gradient_passed':True,
      'raw_window_lexical_max_and_answer_max_passed':True,'formal_fit_or_model_optimizer_steps':0,'CUDA_initialized':torch.cuda.is_initialized()}

def initialize():
    cpu_only();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Already frozen; do not overwrite'
    cfg=protocol();checks=synthetic_checks();_,_,index,w,order,counts=metadata_state()
    freeze_json(OUT/'protocol.json',cfg)
    freeze_json(OUT/'token_index.json',{'answers':index,'all_raw_tokens':TOTAL_RAW,'fit_raw_tokens':FIT_RAW,'fit_lexical_tokens':FIT_LEXICAL})
    for p in (OUT/'fit_token_weights.npz',OUT/'epoch_answer_order.npy'):assert not p.exists()
    np.savez_compressed(OUT/'fit_token_weights.npz',**w);np.save(OUT/'epoch_answer_order.npy',order,allow_pickle=False)
    freeze_json(OUT/'metadata_manifest.json',{'counts':counts,
      'files_sha256':{n:sha(OUT/n) for n in ('token_index.json','fit_token_weights.npz','epoch_answer_order.npy')},
      'protocol_sha256':sha(OUT/'protocol.json'),'only_metadata_no_feature_matrices':True,'test_opened':False})
    freeze_json(OUT/'PREFLIGHT.json',{'passed':True,'synthetic':checks,'actual_metadata':counts,
      'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__),'formal_feature_preparation':False,'formal_training':False,'GPU_used':False})
    freeze_json(OUT/'freeze.json',{'utc':now(),'protocol_sha256':sha(OUT/'protocol.json'),'preflight_sha256':sha(OUT/'PREFLIGHT.json'),
      'metadata_manifest_sha256':sha(OUT/'metadata_manifest.json'),'test_opened':False,'formal_training_started':False})
    print('EXPANDED_HARP_TCN_FROZEN_CPU_ONLY',json.dumps(counts),flush=True)

def frozen_metadata():
    cpu_only();assert read(OUT/'protocol.json')==protocol()
    frozen=read(OUT/'freeze.json');assert frozen['protocol_sha256']==sha(OUT/'protocol.json') and frozen['preflight_sha256']==sha(OUT/'PREFLIGHT.json')
    assert frozen['metadata_manifest_sha256']==sha(OUT/'metadata_manifest.json') and read(OUT/'PREFLIGHT.json')['passed']
    manifest=read(OUT/'metadata_manifest.json')
    for n,h in manifest['files_sha256'].items():assert sha(OUT/n)==h
    original,meta,index,w,order,counts=metadata_state();assert counts==manifest['counts']
    assert read(OUT/'token_index.json')['answers']==index
    with np.load(OUT/'fit_token_weights.npz',allow_pickle=False) as z:assert set(z.files)==set(w) and all(np.array_equal(z[k],w[k]) for k in w)
    assert np.array_equal(np.load(OUT/'epoch_answer_order.npy',allow_pickle=False),order)
    return original,meta,index,w,order

def upstream(require=False):
    cpu_only()
    state=producer.source(require_new=False)
    if state is None or not (producer.OUT/'preparation_complete.json').exists():
        if require:raise AssertionError('Complete3046 Llama manifest and producer numerical preparation required')
        return None
    done=read(producer.OUT/'preparation_complete.json')
    assert done['source_snapshot']==state[-1] and done['old793_all_windows_exact'] and done['total_tokens']==TOTAL_RAW
    assert done['total_windows']==FIT_WINDOWS+CAL_WINDOWS and not done['official_test_opened']
    for n,h in done['files_sha256'].items():assert sha(producer.OUT/n)==h
    path=producer.OUT/'preparation_started.json';assert read(path)['source_snapshot']==state[-1]
    return state,done,{'producer_source_snapshot':state[-1],'producer_preparation_sha256':sha(producer.OUT/'preparation_complete.json'),
      'producer_token_index_sha256':sha(producer.OUT/'token_index.json'),'producer_harp64_sha256':sha(producer.OUT/'matrices/token_harp64.npy'),
      'TCN_protocol_sha256':sha(OUT/'protocol.json'),'metadata_manifest_sha256':sha(OUT/'metadata_manifest.json')}

def prepare():
    cpu_only();assert read(OUT/'protocol.json')==protocol()
    ready=upstream(False)
    if ready is None:
        print('WAITING_FOR_COMPLETE_3046_AND_LLAMA_BASELINE_NUMERICAL_PREPARATION_NO_TCN_PREP_OR_TRAIN',flush=True);return False
    state,_,snap=ready
    if (OUT/'preparation_manifest.json').exists():
        done=read(OUT/'preparation_manifest.json');assert done['source_snapshot']==snap and done['complete']
        for n,h in done['files_sha256'].items():assert sha(OUT/n)==h
        return True
    assert not (OUT/'preparation_started.json').exists(),'Interrupted numerical preparation needs a separately reviewed revision'
    assert shutil.disk_usage(OUT).free>12*2**30
    original,meta,index,w,order=frozen_metadata()
    freeze_json(OUT/'preparation_started.json',{'utc':now(),'source_snapshot':snap})
    plans,newfm,oldfm,_,_=state;newrec={r['response_id']:r for r in newfm['records']}
    oldplans={p['response_id']:p for p in q.lines(ROOT/'data/feature_preparation/plans.jsonl')}
    pindex=read(producer.OUT/'token_index.json')['answers'];assert [e['response_id'] for e in pindex]==[e['response_id'] for e in index]
    for x,y in zip(pindex,index):
        for k in ('response_id','partition','group_id','left','right'):assert x[k]==y[k]
    harp=np.load(producer.OUT/'matrices/token_harp64.npy',mmap_mode='r',allow_pickle=False);assert harp.shape==(TOTAL_RAW,64) and harp.dtype==np.float32
    oldprep=read(full.OUT/'preparation_manifest.json')
    oldcommon=full.OUT/'matrices/common1025.npy';oldharp=full.OUT/'matrices/harp64.npy'
    for p in (oldcommon,oldharp):assert sha(p)==oldprep['files_sha256'][str(p.relative_to(full.OUT))]
    oldc=np.load(oldcommon,mmap_mode='r',allow_pickle=False);oldh=np.load(oldharp,mmap_mode='r',allow_pickle=False)
    oi={e['response_id']:e for e in read(full.OLD/'token_index.json')['answers']}
    directory=OUT/'matrices';directory.mkdir(exist_ok=True)
    raw=np.lib.format.open_memmap(directory/'raw1089.npy',mode='w+',dtype=np.float32,shape=(TOTAL_RAW,WIDTH))
    old_count=0
    for i,(e,t) in enumerate(zip(index,meta['tokens'])):
        rid=e['response_id'];l,r=e['left'],e['right'];n=r-l
        if rid in oldplans:a=q.load_features(rid,original,oldfm,oldplans)
        else:
            p=producer.FEATURE/'features'/(rid+'.npz');assert sha(p)==newrec[rid]['npz_sha256']
            with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in ('lb','nll','token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw')}
        assert a['lb'].shape==(n,1024) and a['nll'].shape==(n,) and a['lb'].dtype==a['nll'].dtype==np.float32
        assert np.isfinite(a['lb']).all() and np.isfinite(a['nll']).all() and np.all((a['lb']>=0)&(a['lb']<=1)) and np.all(a['nll']>=0)
        for k in ('token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw'):assert np.array_equal(a[k],t[k])
        raw[l:r]=np.column_stack((a['lb'],a['nll'],harp[l:r]))
        if rid in oi:
            o=oi[rid];assert n==o['right']-o['left']
            assert np.array_equal(raw[l:r,:1025],oldc[o['left']:o['right']]) and np.array_equal(raw[l:r,1025:],oldh[o['left']:o['right']]);old_count+=1
        if (i+1)%250==0:print('EXPANDED_HARP_TCN_RAW',i+1,3839,flush=True)
    assert old_count==793;raw.flush()
    scaler=StandardScaler()
    for l in range(0,FIT_RAW,16384):
        r=min(l+16384,FIT_RAW);assert w['base'][l:r].sum()>0
        scaler.partial_fit(raw[l:r],sample_weight=w['base'][l:r])
    (OUT/'scaler.pkl').write_bytes(pickle.dumps(scaler,protocol=5))
    normalized=np.lib.format.open_memmap(directory/'standardized1089.npy',mode='w+',dtype=np.float32,shape=(TOTAL_RAW,WIDTH))
    for l in range(0,TOTAL_RAW,16384):
        r=min(l+16384,TOTAL_RAW);normalized[l:r]=scaler.transform(raw[l:r]).astype(np.float32)
        assert np.isfinite(normalized[l:r]).all()
    normalized.flush();assert upstream(True)[-1]==snap
    names=['matrices/raw1089.npy','matrices/standardized1089.npy','scaler.pkl','token_index.json','fit_token_weights.npz','epoch_answer_order.npy']
    freeze_json(OUT/'preparation_manifest.json',{'complete':True,'source_snapshot':snap,'files_sha256':{n:sha(OUT/n) for n in names},
      'shape':[TOTAL_RAW,WIDTH],'fit_raw_tokens':FIT_RAW,'fit_lexical_tokens':FIT_LEXICAL,'loss_mass':float(w['loss'].sum()),
      'all793_original_raw_common_and_HARP_exact':True,'scaler_fit_only':True,'GPU_used':False,'official_test_opened':False})
    print('EXPANDED_HARP_TCN_NUMERICAL_PREPARATION_COMPLETE',flush=True);return True

@torch.no_grad()
def predict(model,x,index):
    model.eval();logits=np.empty(len(x),np.float32)
    for l in range(0,len(index),BATCH):
        chosen=list(range(l,min(l+BATCH,len(index))));xx,mask,_,_=batch(chosen,x,index);values=model(xx,mask).numpy()
        for j,i in enumerate(chosen):
            e=index[i];logits[e['left']:e['right']]=values[j,:e['token_count']]
    assert np.isfinite(logits).all()
    return logits,torch.sigmoid(torch.from_numpy(logits)).numpy()
def evaluate(meta,index,logits,probabilities,w):
    assert logits.shape==probabilities.shape==(TOTAL_RAW,) and np.isfinite(probabilities).all()
    ws,ans=full.v1.aggregate(meta,index,probabilities)
    assert ws.shape==(FIT_WINDOWS+CAL_WINDOWS,) and ans.shape==(FIT_ANSWERS+CAL_ANSWERS,)
    ts={'window':q.choose_threshold([x['label'] for x in meta['windows'][FIT_WINDOWS:]],ws[FIT_WINDOWS:]),
        'answer':q.choose_threshold([x['label'] for x in meta['answers'][FIT_ANSWERS:]],ans[FIT_ANSWERS:])}
    z=logits[:FIT_RAW].astype(np.float64);bce=np.logaddexp(0,z)-w['y']*z
    return {'window_scores':ws,'answer_scores':ans,'token_scores':probabilities,'token_logits':logits},ts,q.metrics(meta,ws,ts),float(w['loss']@bce/LOSS_MASS)

def train():
    cpu_only();ready=upstream(True);assert (OUT/'preparation_manifest.json').exists()
    prep=read(OUT/'preparation_manifest.json');assert prep['complete'] and prep['source_snapshot']==ready[-1]
    for n,h in prep['files_sha256'].items():assert sha(OUT/n)==h
    assert not (OUT/'started.json').exists(),'No silent restart of formal training'
    _,meta,index,w,order=frozen_metadata()
    freeze_json(OUT/'started.json',{'utc':now(),'freeze_sha256':sha(OUT/'freeze.json'),'preparation_manifest_sha256':sha(OUT/'preparation_manifest.json'),
      'epochs':30,'planned_steps':13800,'official_test_opened':False})
    x=np.load(OUT/'matrices/standardized1089.npy',mmap_mode='r',allow_pickle=False)
    assert x.shape==(TOTAL_RAW,WIDTH) and x.dtype==np.float32
    model=new_model();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    directory=OUT/METHOD;directory.mkdir(exist_ok=True);history=[];selected=None;files=[];steps=0;start=time.perf_counter()
    for epoch in range(EPOCHS):
        tick=time.perf_counter();model.train();permutation=order[epoch]
        assert np.array_equal(np.sort(permutation),np.arange(FIT_ANSWERS))
        batches=raw_seen=lex_seen=0
        for l in range(0,FIT_ANSWERS,BATCH):
            chosen=permutation[l:l+BATCH];assert len(chosen)==BATCH
            xx,valid,yy,ww=batch(chosen,x,index,w)
            optimizer.zero_grad(set_to_none=True);loss=objective(model(xx,valid),yy,ww)
            assert torch.isfinite(loss);loss.backward();optimizer.step()
            steps+=1;batches+=1;raw_seen+=int(valid.sum().item());lex_seen+=int((ww>0).sum().item())
        assert batches==460 and raw_seen==FIT_RAW and lex_seen==FIT_LEXICAL and steps==(epoch+1)*460
        fit_seconds=time.perf_counter()-tick;logits,probabilities=predict(model,x,index)
        arrays,ts,metrics,fit_loss=evaluate(meta,index,logits,probabilities,w)
        name=f'epoch_{epoch+1:03d}';sp=directory/(name+'_scores.npz');mp=directory/(name+'.pt')
        np.savez_compressed(sp,**arrays)
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
          'method':METHOD,'epoch':epoch+1,'seed':SEED,'device':'cpu','optimizer_steps':steps,
          'protocol_sha256':sha(OUT/'protocol.json'),'preparation_manifest_sha256':sha(OUT/'preparation_manifest.json'),
          'epoch_answer_order_sha256':sha(OUT/'epoch_answer_order.npy')},mp)
        wth,ath=ts['window'],ts['answer'];key=[min(wth['f1'],ath['f1']),wth['f1'],wth['precision'],-(epoch+1)]
        entry={'method':METHOD,'epoch':epoch+1,'parameters':94529,'thresholds':ts,'selection_key':key,'metrics':metrics,'full_fit_weighted_BCE':fit_loss,
          'epoch_fit_seconds':fit_seconds,'epoch_total_seconds':time.perf_counter()-tick,'optimizer_steps':steps,
          'coverage':{'fit_answers':FIT_ANSWERS,'batches':batches,'raw_tokens':raw_seen,'lexical_tokens':lex_seen,'each_answer_once':True},
          'model_sha256':sha(mp),'scores_sha256':sha(sp),'official_test_opened':False}
        ep=directory/(name+'.json');save(ep,entry);history.append(entry)
        if selected is None or key>selected['selection_key']:selected=entry
        files.extend(str(p.relative_to(OUT)) for p in (sp,mp,ep))
        save(OUT/'progress.json',{'completed_epochs':epoch+1,'selected':selected,'optimizer_steps':steps,'elapsed_seconds':time.perf_counter()-start})
        print('EXPANDED_HARP_TCN_EPOCH',epoch+1,'cal_window',round(wth['f1'],6),'cal_answer',round(ath['f1'],6),'steps',steps,flush=True)
    assert steps==13800 and len(history)==30 and upstream(True)[-1]==prep['source_snapshot'];cpu_only()
    previous=read(full.OUT/'summary.json')['selected']['full_lb_harp64_tcn']
    oldcomplete=read(full.OUT/'complete.json');assert sha(full.OUT/'summary.json')==oldcomplete['files_sha256']['summary.json']
    summary={'selected':{METHOD:selected},'all_epochs':{METHOD:history},'epochs':30,'optimizer_steps':steps,
      'old_frozen_reference':{'method':'full_lb_harp64_tcn','epoch':previous['epoch'],'calibration':previous['metrics']['calibration'],
        'source_summary_sha256':sha(full.OUT/'summary.json'),'old_fit_answers':634,'old_steps':2400},
      'fit_answers':FIT_ANSWERS,'calibration_answers':CAL_ANSWERS,'calibration_selection_optimistic':True,'equal_total_computation':False,
      'official_test_opened':False,'GPU_used':False,'seconds':time.perf_counter()-start}
    save(OUT/'summary.json',summary)
    report=['扩充HARP64+LB/NLL TCN已完成30轮、13800步；使用3680训练回答/615组。以下仅为用于选轮和阈值的开发校准成绩。','',
      f"选中epoch {selected['epoch']}：cal窗口F1={selected['metrics']['calibration']['windows']['f1']:.6f}，cal整答F1={selected['metrics']['calibration']['answers']['f1']:.6f}。",
      '原模型仍冻结；原30轮约2400步，本次不属于相同总计算量。所有30轮状态和fit/cal分数保留。未打开官方test，未运行GPU或重新生成答案。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    files+=['summary.json','REPORT.md','protocol.json','PREFLIGHT.json','freeze.json','metadata_manifest.json','token_index.json',
      'fit_token_weights.npz','epoch_answer_order.npy','preparation_manifest.json','started.json']
    freeze_json(OUT/'complete.json',{'status':'complete_development_only','utc':now(),'files_sha256':{n:sha(OUT/n) for n in files},
      'code_sha256':sha(__file__),'epochs':30,'optimizer_steps':steps,'official_test_opened':False,'GPU_used':False})
    del model,optimizer,x;gc.collect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','prepare','train','all']);args=parser.parse_args()
    torch.set_num_threads(THREADS);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=THREADS):
        if args.stage=='initialize':initialize()
        elif args.stage=='prepare':prepare()
        elif args.stage=='train':train()
        elif prepare():train()

