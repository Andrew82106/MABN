"""Prepared audit only; run after explicit dispatch and all20 results complete.

No production imports, fitting, GPU, test, or token-feature regeneration.
Uses the earlier pinned audit's inert pickle reader and independent metrics.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import gc, hashlib, importlib.util, json, traceback
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'llama_baselines_v1';PROBE=HERE/'probe_v1'
METHODS=('prefix_pre_header','legacy_lb_nll','prefix_post_header','harp64_legacy_lb_nll')
WIDTHS=(1024,1025,1024,1089);GRID=(1e-5,1e-4,.001,.01,.1)
NFIT,NCAL,N,AFIT,ACAL,BATCH,MASS=653979,42241,696220,3680,159,16384,168123
REPORT=OUT/'INDEPENDENT_NUMERIC_AUDIT_LLAMA_BASELINES.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def safe(p):
    p=Path(p).resolve();p.relative_to(ROOT)
    assert not any(x.lower() in ('raw','final_test','sealed','withheld') for x in p.parts)
    return p
def helpers():
    p=ROOT/'results/lookback_controls_v2/audit_lookback_controls_v2.py'
    assert sha(p)=='dc5f32fed2666604907e2474fdb6a9bec46b30f4953b1109b43112c6494414eb'
    spec=importlib.util.spec_from_file_location('previous_independent_controls_audit',p)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    return a

def metadata():
    # Retain identities/labels only, not text or per-token matrices.
    answers=[]
    for part,path,count in [('fit',HERE/'data/answers_fit.jsonl',AFIT),
                            ('calibration',ROOT/'data/answers_calibration.jsonl',ACAL)]:
        start=len(answers)
        for a in rows(path):
            assert a['partition']==part and a['eligible'] and a['quality']=='good'
            assert a['label']==int(bool(a['original_labels']))
            answers.append({k:a[k] for k in ('response_id','answer_id','source_id','group_id','partition','label','eligible_window_count')})
        assert len(answers)-start==count
    ids={a['response_id']:i for i,a in enumerate(answers)};assert len(ids)==AFIT+ACAL
    old=list(rows(ROOT/'data/answers_fit.jsonl'));assert len(old)==634
    for a,b in zip(answers[:634],old):assert all(a[k]==b[k] for k in a)
    groups=[{a['group_id'] for a in answers if a['partition']==p} for p in ('fit','calibration')]
    assert tuple(map(len,groups))==(615,154) and not groups[0]&groups[1]
    y=np.empty(N,np.int64);owner=np.empty(N,np.int32);keys=[];j=0
    for part,path,count in [('fit',HERE/'data/windows_k4_fit.jsonl',NFIT),
                            ('calibration',ROOT/'data/windows_k4_calibration.jsonl',NCAL)]:
        start=j
        for w in rows(path):
            assert w['partition']==part and w['eligible'] and w['k']==4 and w['stride']==1
            assert w['lexical_token_indices'] and w['label'] in (0,1)
            ai=ids[w['response_id']];a=answers[ai]
            assert (w['group_id'],w['answer_id'],part)==(a['group_id'],a['answer_id'],a['partition'])
            y[j]=w['label'];owner[j]=ai;keys.append(w['window_id']);j+=1
        assert j-start==count
    assert j==N and len(set(keys))==N
    counts=np.bincount(owner,minlength=len(answers))
    assert np.array_equal(counts,[a['eligible_window_count'] for a in answers]) and np.all(counts>0)
    assert np.array_equal(owner,np.repeat(np.arange(len(answers)),counts))
    starts=np.r_[0,np.cumsum(counts)[:-1]];assert starts[AFIT]==NFIT
    # Preserve exact original window identities/gold at both old partitions.
    oldj=0
    for w in rows(ROOT/'data/windows_k4_fit.jsonl'):
        assert keys[oldj]==w['window_id'] and y[oldj]==w['label'];oldj+=1
    assert oldj==168123
    ay=np.asarray([a['label'] for a in answers],np.int64)
    return answers,y,ay,owner,counts,starts

def weights(answers,y,owner,counts,near):
    tree=defaultdict(lambda:defaultdict(list))
    for i,a in enumerate(answers[:AFIT]):tree[a['group_id']][int(i<634)].append(i)
    assert len(tree)==615 and all(set(v)=={0,1} for v in tree.values())
    answer_base=np.empty(AFIT,np.float64)
    for pools in tree.values():
        for aa in pools.values():
            for i in aa:answer_base[i]=(MASS/615)/(len(pools)*len(aa)*counts[i])
    base=answer_base[owner[:NFIT]]
    class_mass=np.array([base[y[:NFIT]==k].sum() for k in (0,1)])
    factors=base.sum()/(2*class_mass);loss=base*factors[y[:NFIT]]
    group_ids=np.array([a['group_id'] for a in answers[:AFIT]],object)[owner[:NFIT]]
    group_err=pool_err=0.
    for g in tree:
        ix=np.flatnonzero(group_ids==g);loss[ix]*=(MASS/615)/loss[ix].sum()
        group_err=max(group_err,abs(base[ix].sum()-MASS/615))
        for native in (0,1):
            ii=ix[(owner[ix]<634)==bool(native)];pool_err=max(pool_err,abs(base[ii].sum()-MASS/1230))
    assert sha(OUT/'training_weights.npz')==sha(PROBE/'expanded3680_weights.npz')
    with np.load(OUT/'training_weights.npz',allow_pickle=False) as z:w={k:z[k].copy() for k in z.files}
    assert set(w)=={'base','loss','y','class_factors'} and np.array_equal(w['y'],y[:NFIT])
    report={'base_max_abs':near(base,w['base'],'base',rtol=0,atol=1e-10),
      'loss_max_abs':near(loss,w['loss'],'loss',rtol=0,atol=1e-9),
      'class_factors_max_abs':near(factors,w['class_factors'],'class factors',rtol=0,atol=1e-10),
      'base_mass':float(w['base'].sum()),'loss_mass':float(w['loss'].sum()),
      'equal_group_base_max_abs':group_err,'native_aux_half_group_max_abs':pool_err,
      'frozen_probe_weights_byte_identical':True,'fit_only_y_exact':True}
    assert abs(w['base'].sum()-MASS)<1e-7 and abs(w['loss'].sum()-MASS)<1e-7
    report['group_loss_max_abs']=max(abs(w['loss'][group_ids==g].sum()-MASS/615) for g in tree)
    assert report['group_loss_max_abs']<1e-7
    del group_ids;return w,report

def moments(raw,width,base):
    # Weighted Chan moments over fit only. No scaler.fit or partial_fit.
    # This producer passes float32 matrices to sklearn1.6.1: each partial_fit
    # casts both sample weights and prior weighted count to input float32.
    total=0.;mean=np.zeros(width);var=np.zeros(width)
    for l in range(0,NFIT,BATCH):
        r=min(l+BATCH,NFIT);x=raw(l,r).astype(np.float64)
        b=base[l:r].astype(np.float32).astype(np.float64);mass=b.sum()
        mu=b@x/mass;center=x-mu;correction=b@center
        batch_m2=b@(center*center)-correction*correction/mass
        total=float(np.float32(total))
        delta=mu-mean;combined=total+mass
        var=(total*var+batch_m2+delta*delta*(total*mass/combined))/combined
        mean=(total*mean+mass*mu)/combined;total=combined
    eps=np.finfo(np.float64).eps
    constant=var<=total*eps*var+(total*mean*eps)**2
    scale=np.sqrt(var);scale[constant]=1.
    return mean,var,scale,total

def audit():
    # Gate before helper imports, score/weight reads or report writes.
    assert (OUT/'complete.json').exists(),'WAITING: all20 candidates must be complete; no partial audit'
    complete=read(OUT/'complete.json');cfg=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
    assert complete['fit_count']==summary['fit_count']==cfg['fit_count']==20
    assert not complete['official_test_opened'] and not summary['official_test_opened']
    assert cfg['methods']==list(METHODS) and cfg['C']==list(GRID) and cfg['widths']==dict(zip(METHODS,WIDTHS))
    assert summary['fit_answers']==AFIT and summary['calibration_answers']==ACAL
    assert set(summary['all_candidates'])==set(summary['selected'])==set(METHODS)
    assert all(len(summary['all_candidates'][m])==5 for m in METHODS)
    assert len({e['candidate'] for es in summary['all_candidates'].values() for e in es})==20
    a=helpers();q=a.q;near=a.near;verified={}
    def verify(p,h):
        p=safe(p);actual=sha(p);assert actual==h,(str(p),actual,h);verified[str(p)]=actual
    for n,h in complete['files_sha256'].items():verify(OUT/n,h)
    for p,h in cfg['files_sha256'].items():verify(p,h)
    prep=read(OUT/'preparation_complete.json');snap=prep['source_snapshot']
    assert prep['old793_all_windows_exact'] and prep['total_tokens']==708506 and prep['total_windows']==N
    assert not prep['official_test_opened'] and snap['new_manifest_complete'] and snap['old793_unchanged']
    assert read(OUT/'preparation_started.json')['source_snapshot']==snap
    for p,h in snap['files_sha256'].items():verify(p,h)
    for n,h in prep['files_sha256'].items():verify(OUT/n,h)
    started=read(OUT/'fit_started.json')
    assert started['preparation_sha256']==sha(OUT/'preparation_complete.json') and started['protocol_sha256']==sha(OUT/'protocol.json')
    assert datetime.fromisoformat(started['utc'])<=datetime.fromisoformat(complete['utc'])
    fm=read(HERE/'llama_features_v3/feature_manifest.json');sig=read(HERE/'llama_features_v3/signature.json')
    assert fm['complete'] and fm['completed_count']==len(fm['records'])==3046 and not fm['labels_used'] and not fm['test_read']
    assert [r['response_id'] for r in fm['records']]==sig['response_ids_in_order'] and len(set(sig['response_ids_in_order']))==3046
    answers,y,ay,owner,counts,starts=metadata()
    # Producer joins new caches by response_id; extraction order need not equal
    # the numerical matrix order, which is checked against token_index below.
    assert set(sig['response_ids_in_order'])=={a['response_id'] for a in answers[634:AFIT]}
    index=read(OUT/'token_index.json')['answers'];assert len(index)==len(answers)
    assert all((e['response_id'],e['group_id'],e['partition'])==(a['response_id'],a['group_id'],a['partition']) for e,a in zip(index,answers))
    assert index[0]['left']==0 and index[-1]['right']==708506 and index[AFIT-1]['right']==665708
    assert all(a['right']==b['left'] for a,b in zip(index,index[1:]))
    w,wr=weights(answers,y,owner,counts,near)
    mats={k:np.load(OUT/'matrices'/f'{k}.npy',mmap_mode='r',allow_pickle=False) for k in ('base','harp','lb_prefix_pre_header','lb_prefix_post_header')}
    for k,d in [('base',1025),('harp',64),('lb_prefix_pre_header',1024),('lb_prefix_post_header',1024)]:assert mats[k].shape==(N,d) and mats[k].dtype==np.float32
    families={};metric_error=0.;threshold_count=metric_count=0
    for method,width in zip(METHODS,WIDTHS):
        def raw(l,r):
            if method=='prefix_pre_header':return np.asarray(mats['lb_prefix_pre_header'][l:r])
            if method=='prefix_post_header':return np.asarray(mats['lb_prefix_post_header'][l:r])
            if method=='legacy_lb_nll':return np.asarray(mats['base'][l:r])
            return np.column_stack((mats['base'][l:r],mats['harp'][l:r]))
        objs=[q.unpickle(OUT/f'{method}_C{c:g}.pkl') for c in GRID];sc=objs[0]['scaler']
        mu,var,scale,total=moments(raw,width,w['base'])
        sr={'mean_max_abs':near(mu,sc.mean_,method+' mean',rtol=1e-10,atol=2e-10),
            'variance_max_abs':near(var,sc.var_,method+' variance',rtol=1e-10,atol=2e-9),
            'scale_max_abs':near(scale,sc.scale_,method+' scale',rtol=1e-10,atol=2e-10),
            'weighted_count_max_abs':near(total,sc.n_samples_seen_,method+' count',atol=1e-7)}
        zp=OUT/'matrices'/f'{method}_fit_standardized.npy';zh=sha(zp)
        standard=np.load(zp,mmap_mode='r',allow_pickle=False);assert standard.shape==(NFIT,width) and standard.dtype==np.float32
        stored=[];entries=[];replay=[np.empty(N) for c in GRID]
        for c,obj in zip(GRID,objs):
            name=f'{method}_C{c:g}';entry=read(OUT/f'{name}_result.json');model=obj['model']
            assert (obj['method'],obj['C'],obj['width'],obj['fit_rows'],obj['fit_answers'],obj['fit_groups'])==(method,c,width,NFIT,AFIT,615) and obj['fit_only']
            assert obj['weights_sha256']==sha(OUT/'training_weights.npz') and obj['fit_matrix_sha256']==zh and obj['protocol_sha256']==sha(OUT/'protocol.json')
            for k in ('mean_','var_','scale_','n_samples_seen_','n_features_in_','with_mean','with_std'):assert np.array_equal(getattr(sc,k),getattr(obj['scaler'],k))
            assert sc.n_features_in_==width and sc.with_mean and sc.with_std
            assert model.coef_.shape==(1,width) and model.intercept_.shape==(1,) and np.array_equal(model.classes_,[0,1])
            assert model.C==c and model.solver=='liblinear' and model.penalty=='l2' and model.class_weight is None
            assert model.max_iter==2000 and model.n_iter_.max()<2000 and model.random_state==20260924
            assert entry['candidate']==name and entry['method']==method and entry['C']==c
            assert entry['model_sha256']==sha(OUT/f'{name}.pkl') and entry['scores_sha256']==sha(OUT/f'{name}_scores.npz')
            with np.load(OUT/f'{name}_scores.npz',allow_pickle=False) as z:s=z['window_scores'].copy();av=z['answer_scores'].copy()
            assert s.shape==(N,) and av.shape==(AFIT+ACAL,) and np.isfinite(s).all() and np.isfinite(av).all()
            assert np.array_equal(av,np.maximum.reduceat(s,starts))
            ts={'window':a.choose(y[NFIT:],s[NFIT:]),'answer':a.choose(ay[AFIT:],av[AFIT:])}
            assert ts==entry['thresholds']==obj['thresholds'];threshold_count+=2
            key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c]
            assert key==entry['selection_key'] and tuple(key)==tuple(obj['selection_key'])
            for part,wl,wh,al,ah in [('fit',0,NFIT,0,AFIT),('calibration',NFIT,N,AFIT,AFIT+ACAL)]:
                for unit,yy,ss,t in [('windows',y[wl:wh],s[wl:wh],ts['window']['threshold']),('answers',ay[al:ah],av[al:ah],ts['answer']['threshold'])]:
                    got=a.metric(yy,ss,t);expected=entry['metrics'][part][unit];assert got.keys()==expected.keys()
                    for k,v in got.items():
                        if isinstance(v,int):assert v==expected[k]
                        else:metric_error=max(metric_error,near(v,expected[k],(name,part,unit,k),rtol=0,atol=1e-12))
                    metric_count+=1
            stored.append((s,av));entries.append(entry)
        for l in range(0,N,BATCH):
            r=min(l+BATCH,N);z=raw(l,r).copy();assert np.isfinite(z).all();z-=sc.mean_;z/=sc.scale_
            if l<NFIT:assert np.array_equal(z[:min(r,NFIT)-l],standard[l:min(r,NFIT)])
            for j,obj in enumerate(objs):replay[j][l:r]=q.sigmoid((z@obj['model'].coef_.T+obj['model'].intercept_).ravel())
        checks=[]
        for c,s,(expected,av) in zip(GRID,replay,stored):
            checks.append({'C':c,'window_coef_max_abs':near(s,expected,(method,c,'manual coefficient'),rtol=0,atol=2e-12),
              'answermax_max_abs':near(np.maximum.reduceat(s,starts),av,(method,c,'manual answer'),rtol=0,atol=2e-12)})
        assert summary['all_candidates'][method]==entries
        selected=max(entries,key=lambda e:e['selection_key']);assert selected==summary['selected'][method]
        families[method]={'width':width,'scaler_fit_only':sr,'all_five_scalers_identical':True,'standardized_fit_matrix_exact':True,
          'candidate_replays':checks,'selected_C':selected['C'],'cal_window_F1':selected['metrics']['calibration']['windows']['f1'],
          'cal_answer_F1':selected['metrics']['calibration']['answers']['f1']}
        print('LLAMA_BASELINE_AUDIT_FAMILY_PASSED',method,flush=True)
        del objs,stored,entries,replay,standard,z;gc.collect()
    assert threshold_count==40 and metric_count==80
    assert sha(OUT/'protocol.json')==started['protocol_sha256'] and sha(OUT/'preparation_complete.json')==started['preparation_sha256']
    report={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
      'weights':wr,'families':families,'coefficient_candidates':20,'full_window_replays':20*N,'full_answermax_checks':20*(AFIT+ACAL),
      'thresholds_exact':threshold_count,'selection_keys_exact':20,'selected_Cs_exact':4,'metric_blocks':metric_count,'metric_max_abs':metric_error,
      'complete_preparation_started_fit_snapshot_bindings_verified':True,'verified_hashes':verified,
      'denominators':{'fit':{'answers':AFIT,'positive_answers':int(ay[:AFIT].sum()),'groups':615,'windows':NFIT,'positive_windows':int(y[:NFIT].sum())},
        'calibration':{'answers':ACAL,'positive_answers':int(ay[AFIT:].sum()),'groups':154,'windows':NCAL,'positive_windows':int(y[NFIT:].sum())}},
      'complete_sha256':sha(OUT/'complete.json'),'helper_sha256':sha(__file__),
      'no_fit_or_GPU':True,'official_test_read':False,'production_imports':False,'frozen_files_modified':False,
      'limits':['Audit of committed full matrices, fit weights/scalers,20 coefficient objects and calibration selection; no token-feature extraction or HARP/PCA basis recalculation.',
        'Source and full3046 feature manifests are hash-bound; individual17GiB token arrays are not replayed by this bounded numerical helper.',
        'Auxiliary answers share615 groups and were replayed by one Llama model; calibration selection is optimistic development evidence.',
        'Recorded snapshot/UTC relationships are consistency checks, not independent timestamp attestations.']}
    del mats,w,owner;gc.collect();return report

if __name__=='__main__':
    if not (OUT/'complete.json').exists():raise SystemExit('WAITING_FOR_ALL20_COMPLETED_NO_SCORES_READ_NO_REPORT')
    assert not REPORT.exists(),'Do not overwrite an existing independent audit'
    try:
        with threadpool_limits(limits=4):report=audit()
    except Exception:
        failure={'status':'failed','utc':datetime.now(timezone.utc).isoformat(),'traceback':traceback.format_exc(),'helper_sha256':sha(__file__)}
        (OUT/'INDEPENDENT_NUMERIC_AUDIT_LLAMA_BASELINES.failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
        raise
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=['扩充 Llama 四族20候选独立数值审计通过。fit-only scaler、原辅均权和总loss168123、全部系数回放、40阈值及4个C选择一致。','',
      '| 方法 | C | 校准窗口F1 | 校准整答F1 |','|---|---:|---:|---:|']
    for name,f in report['families'].items():md.append(f"| {name} | {f['selected_C']:g} | {f['cal_window_F1']:.6f} | {f['cal_answer_F1']:.6f} |")
    md+=['','分母为3680训练答/653979窗、159校准答/42241窗。仅完整冻结产物审计，不拟合、不使用GPU、不读取test；以上为用于选参的开发校准成绩。原token→特征及投影全链不在本次有界重算范围。']
    REPORT.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('LLAMA_BASELINE_INDEPENDENT_AUDIT_PASSED',sha(REPORT),flush=True)
