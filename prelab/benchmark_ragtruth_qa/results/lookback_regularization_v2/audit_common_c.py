"""Independent completed common-C audit; no fit/GPU/test/production imports.
Reuses the just-completed793-token-source audit; here checks frozen matrices,
scalers/weights, all25 coefficients,17 byte copies,50 thresholds and selections.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,traceback,gc
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
CTRL=OUT.parent/'lookback_controls_v2';REG=OUT.parent/'regularization_v1';DEV=OUT.parent/'development_v1'
H=CTRL/'audit_lookback_controls_v2.py'
assert hashlib.sha256(H.read_bytes()).hexdigest()=='dc5f32fed2666604907e2474fdb6a9bec46b30f4953b1109b43112c6494414eb'
sp=importlib.util.spec_from_file_location('prior_independent_controls_audit',H);a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
q=a.q;sha,read,near,choose,metric=a.sha,a.read,a.near,a.choose,a.metric
METHODS=a.METHODS;GRID=(1e-5,1e-4,.001,.01,.1);ADDED=GRID[:2];NFIT=168123;N=210364;BATCH=16384
REPORT=OUT/'INDEPENDENT_AUDIT_COMMON_C.json'
def save(r):REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def safe(p):
    p=Path(p).resolve();p.relative_to(ROOT.resolve());return p
def original_source(method,c):
    return (REG,f'lookback_mean_C{c:g}') if method==METHODS[-1] and c in ADDED else (CTRL,f'{method}_C{c:g}')
def scaler_equal(x,y):
    for k in ('mean_','var_','scale_','n_samples_seen_','n_features_in_','with_mean','with_std'):
        assert np.array_equal(getattr(x,k),getattr(y,k)),k

def audit(report):
    assert (OUT/'complete.json').exists(),'No partial cohort audit'
    complete=read(OUT/'complete.json');cfg=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
    assert sha(OUT/'complete.json')=='232bbf28dd956902ff8e1ff7df6594444d3c428dcfdd84ba960aa22f817a512e'
    assert complete['new_fits']==8 and complete['frozen_replays']==17 and not complete['official_test_opened']
    assert summary['new_fits']==8 and summary['frozen_replays']==17 and summary['candidate_count']==25 and not summary['official_test_opened']
    assert cfg['methods']==list(METHODS) and cfg['C']==list(GRID)==summary['all_common_C'] and cfg['added_C']==list(ADDED)
    assert cfg['code_sha256']==sha(ROOT/'src/run_lookback_regularization_v2.py')
    assert cfg['controls_runner_sha256']==sha(ROOT/'src/run_lookback_controls_v2.py')
    assert cfg['controls_complete_sha256']==sha(CTRL/'complete.json') and cfg['regularization_complete_sha256']==sha(REG/'complete.json')
    assert cfg['gold_manifest_sha256']==sha(ROOT/'data/gold_manifest.json') and cfg['no_PCA_or_scaler_fit'] and cfg['no_GPU'] and cfg['no_official_test']
    count=0
    for name,h in complete['files_sha256'].items():assert sha(safe(OUT/name))==h;count+=1
    snapshot=read(OUT/'source_snapshot.json');started=read(OUT/'started.json')
    assert started['source_snapshot_sha256']==sha(OUT/'source_snapshot.json') and started['protocol_sha256']==sha(OUT/'protocol.json')
    assert datetime.fromisoformat(started['utc'])<=datetime.fromisoformat(complete['utc'])
    for p,h in snapshot['files_sha256'].items():assert sha(safe(p))==h;count+=1
    for m in METHODS[:-1]:
        zp=(CTRL/'matrices'/(m+'_fit_standardized.npy')).resolve();assert str(zp) in snapshot['files_sha256']
    report['snapshot']={'started_protocol_and_snapshot_hashes_exact':True,'all_four_standardized_fit_matrices_in_started_snapshot':True,
      'all_source_hashes_unchanged_at_audit':True,'started_sha256':sha(OUT/'started.json'),'source_snapshot_sha256':sha(OUT/'source_snapshot.json'),
      'recorded_start_utc':started['utc'],'recorded_complete_utc':complete['utc'],'limits':'Recorded ordering, not an external timestamp attestation.'}
    prior=CTRL/'INDEPENDENT_AUDIT_LOOKBACK_CONTROLS_V2.json'
    assert sha(prior)=='c3390303f3d4693f3512b43b0d37c53343246cd66085b08051277867b0735b06' and read(prior)['status']=='passed'
    assert read(prior)['complete_sha256']==sha(CTRL/'complete.json')
    regaudit=REG/'INDEPENDENT_AUDIT_REGULARIZATION.json'
    assert sha(regaudit)=='3898dc3c226e4c144b6975638d586344c952872f73c2273e20ea7425ba84e26d' and read(regaudit)['status']=='passed'
    report['prior_chain_evidence']={'complete793_token_to_five_matrix_audit_sha256':sha(prior),'regularization_old_anchor_audit_sha256':sha(regaudit),
      'boundaries':'Prior exact source/matrix/PCA checks retained; no redundant token-array loading or model fitting.'}
    answers,windows,indices=q.metadata();y=np.asarray([w['label'] for w in windows],int);ay=np.asarray([x['label'] for x in answers],int)
    b,wr=q.audit_weights(windows);report['weights']=wr
    with np.load(CTRL/'training_weights.npz',allow_pickle=False) as z:cw={k:z[k] for k in z.files}
    with np.load(DEV/'training_weights.npz',allow_pickle=False) as z:dw={k:z[k] for k in z.files}
    assert cw.keys()==dw.keys() and all(np.array_equal(cw[k],dw[k]) for k in dw)
    assert read(CTRL/'fit_keys.json')==read(DEV/'fit_keys.json') and read(CTRL/'score_index.json')==read(DEV/'score_index.json')
    assert abs(cw['loss_weights'].sum()-168123)<1e-9 and np.array_equal(cw['y'],y[:NFIT])
    mm=read(CTRL/'matrix_manifest.json');families={};thresholds=0;metric_blocks=0;metric_error=0.;reused=0;new=0
    for method in METHODS:
        xp=CTRL/'matrices'/(method+'.npy');assert sha(xp)==mm['files_sha256'][method]
        x=np.load(xp,mmap_mode='r',allow_pickle=False);assert x.shape==(N,1024) and x.dtype==np.float32
        refp=CTRL/f'{method}_C0.001.pkl';ref=q.unpickle(refp);sc=ref['scaler']
        assert ref['fit_only'] and (ref['fit_rows'],ref['fit_groups'],ref['fit_answers'],ref['width'])==(NFIT,615,634,1024)
        assert ref['weights_sha256']==sha(CTRL/'training_weights.npz') and ref['fit_keys_sha256']==sha(CTRL/'fit_keys.json')
        standard=None
        if method!=METHODS[-1]:
            zp=CTRL/'matrices'/(method+'_fit_standardized.npy');standard=np.load(zp,mmap_mode='r',allow_pickle=False)
            assert standard.shape==(NFIT,1024) and standard.dtype==np.float32 and sha(zp)==snapshot['files_sha256'][str(zp.resolve())]
        objs=[];scores=[];entries=[];replays=[np.empty(N) for _ in GRID];cr=[]
        for c in GRID:
            name=f'{method}_C{c:g}';mp=OUT/(name+'.pkl');sf=OUT/(name+'_scores.npz');obj=q.unpickle(mp);entry=read(OUT/(name+'_result.json'))
            assert obj['C']==c and entry['C']==c and entry['method']==method
            model=obj['model'];assert model.C==c and model.coef_.shape==(1,1024) and model.intercept_.shape==(1,) and np.array_equal(model.classes_,[0,1])
            assert model.solver=='liblinear' and model.penalty=='l2' and model.class_weight is None and model.max_iter==2000 and model.n_iter_.max()<2000 and model.random_state==20260924
            scaler_equal(sc,obj['scaler'])
            assert entry['model_sha256']==sha(mp) and entry['scores_sha256']==sha(sf)
            isnew=method!=METHODS[-1] and c in ADDED
            if isnew:
                new+=1;assert obj['method']==method and obj['mode']==entry['mode']=='new_fit' and obj['fit_only']
                for k in ('fit_rows','fit_groups','fit_answers','width','weights_sha256','fit_keys_sha256','feature_signature_sha256'):assert obj[k]==ref[k]
                assert obj['source_scaler_model_sha256']==sha(refp)==entry['provenance']['source_scaler_model_sha256']
                assert Path(entry['provenance']['source_scaler_model']).resolve()==refp.resolve()
                assert obj['protocol_sha256']==sha(OUT/'protocol.json') and obj['fit_standardized_sha256']==snapshot['files_sha256'][str(zp.resolve())]
            else:
                reused+=1;folder,oldname=original_source(method,c);op=folder/(oldname+'.pkl');os=folder/(oldname+'_scores.npz')
                assert mp.read_bytes()==op.read_bytes() and sf.read_bytes()==os.read_bytes()
                assert entry['mode']=='frozen_replay'
                pv=entry['provenance'];assert Path(pv['model']).resolve()==op.resolve() and Path(pv['scores']).resolve()==os.resolve()
                assert pv['model_sha256']==sha(op) and pv['scores_sha256']==sha(os)
                oe=read(folder/(oldname+'_result.json'))
                assert entry['thresholds']==oe['thresholds'] and entry['metrics']==oe['metrics'] and entry['selection_key']==oe['selection_key']
            with np.load(sf,allow_pickle=False) as z:s=z['window_scores'].copy();av=z['answer_scores'].copy()
            assert s.shape==(N,) and av.shape==(793,) and np.isfinite(s).all()
            assert np.array_equal(av,np.asarray([s[indices[aa['answer_id']]].max() for aa in answers]))
            ts={'window':choose(y[NFIT:],s[NFIT:]),'answer':choose(ay[634:],av[634:])}
            assert ts==entry['thresholds']==obj['thresholds'];thresholds+=2
            key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c]
            assert key==entry['selection_key'] and tuple(key)==tuple(obj['selection_key'])
            for part,wl,wr_,al,ar in [('fit',0,NFIT,0,634),('calibration',NFIT,N,634,793)]:
                for unit,yy,ss,cut in [('windows',y[wl:wr_],s[wl:wr_],ts['window']['threshold']),('answers',ay[al:ar],av[al:ar],ts['answer']['threshold'])]:
                    got=metric(yy,ss,cut);oldm=entry['metrics'][part][unit];assert got.keys()==oldm.keys()
                    for k,v in got.items():
                        if isinstance(v,int):assert v==oldm[k]
                        else:metric_error=max(metric_error,near(v,oldm[k],(method,c,part,unit,k),atol=1e-12))
                    metric_blocks+=1
            objs.append(obj);scores.append((s,av));entries.append(entry)
            cr.append({'C':c,'new_fit':isnew,'old_model_and_score_bytes_exact':not isnew})
        for l in range(0,N,BATCH):
            r=min(l+BATCH,N);z=np.asarray(x[l:r]).copy();assert np.isfinite(z).all();z-=sc.mean_;z/=sc.scale_
            if standard is not None and l<NFIT:assert np.array_equal(z[:min(r,NFIT)-l],standard[l:min(r,NFIT)])
            for j,obj in enumerate(objs):replays[j][l:r]=q.sigmoid((z@obj['model'].coef_.T+obj['model'].intercept_).ravel())
        for j,(c,got,(s,av)) in enumerate(zip(GRID,replays,scores)):
            cr[j]['coefficient_max_abs']=near(got,s,(method,c,'manual coefficient'),rtol=0,atol=2e-12)
            ga=np.asarray([got[indices[aa['answer_id']]].max() for aa in answers]);cr[j]['manual_answermax_max_abs']=near(ga,av,(method,c,'manual answer max'),rtol=0,atol=2e-12)
        assert summary['all_candidates'][method]==entries
        selected=max(entries,key=lambda e:e['selection_key']);assert selected==summary['selected'][method]
        oldselected=read(CTRL/'summary.json')['selected'][method]
        families[method]={'candidates':cr,'all_scaler_state_exact':True,'frozen_fit_matrix_exact':method!=METHODS[-1],
          'selected_C':selected['C'],'cal_window_F1':selected['metrics']['calibration']['windows']['f1'],'cal_answer_F1':selected['metrics']['calibration']['answers']['f1'],
          'old_selected_C':oldselected['C'],'old_cal_window_F1':oldselected['metrics']['calibration']['windows']['f1'],'old_cal_answer_F1':oldselected['metrics']['calibration']['answers']['f1']}
        report['families']=families;save(report);print('COMMON_C_FAMILY_PASSED',method,flush=True)
        del x,z,standard,objs,scores,replays;gc.collect()
    assert new==8 and reused==17 and thresholds==50 and metric_blocks==100
    checks={x['candidate']:x for x in summary['old_replay_checks']}
    expected={f'{m}_C{c:g}' for m in METHODS for c in GRID if m==METHODS[-1] or c not in ADDED}
    assert set(checks)==expected and len(checks)==17 and all(c['stored_scores_thresholds_metrics_exact'] and c['max_coefficient_replay_abs']<=1e-12 for c in checks.values())
    assert read(OUT/'source_snapshot.json')==snapshot and sha(OUT/'source_snapshot.json')==started['source_snapshot_sha256']
    report.update(status='passed',blockers=[],new_fitted_candidates_checked=new,frozen_model_and_score_byte_copies_checked=reused,
      all_four_fit_standardized_matrices_exact=True,all25_scalers_equal_original_fit_only_state=True,original_weights_and_loss_mass_unchanged=True,
      full_window_score_replays=25*N,full_answermax_checks=25*793,max_coefficient_abs=max(c['coefficient_max_abs'] for f in families.values() for c in f['candidates']),
      thresholds_exact=thresholds,selection_keys_exact=25,selected_Cs_exact=5,metric_blocks=metric_blocks,max_metric_abs=metric_error,
      verified_artifact_and_source_hashes=count,complete_sha256=sha(OUT/'complete.json'),audit_script_sha256=sha(__file__),
      denominators={'fit':{'answers':634,'positive_answers':328,'groups':615,'windows':NFIT,'positive_windows':21477},
                    'calibration':{'answers':159,'positive_answers':100,'groups':154,'windows':42241,'positive_windows':5984}})
    md=['共同C独立数值审计通过，无阻断。25候选全部手工系数回放；17份旧模型与分数逐字节一致。四个原标准化fit矩阵、全部旧scaler、原权重及loss总量168123保持，started与完整source snapshot绑定正确。','',
      '50个阈值、25个选择key、5个最终C全部一致；每答由全部原可评窗口取max，分母仍fit634/cal159、168123/42241窗。','',
      '| 定义 | 新选C | cal窗口F1 原→扩展 | cal整答F1 原→扩展 |','|---|---:|---:|---:|']
    for m,f in families.items():md.append(f"| {m} | {f['selected_C']:g} | {f['old_cal_window_F1']:.6f} → {f['cal_window_F1']:.6f} | {f['old_cal_answer_F1']:.6f} → {f['cal_answer_F1']:.6f} |")
    md+=['','仅审新增共同网格；上一轮793词元到五组矩阵的完整核验以哈希绑定，不重复提取。未拟合、用GPU或读取官方test/withheld。新增网格是在观察开发结果后提出，以上仍是用于选参的校准成绩，不能视为独立泛化结果。']
    (OUT/'INDEPENDENT_AUDIT_COMMON_C.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

if __name__=='__main__':
    report={'status':'running','audit_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
      'no_new_fit_or_GPU':True,'official_test_or_withheld_read':False,'production_imports':False,'frozen_files_modified':False,
      'limits':['Only the completed common-C extension is numerically replayed. Prior full793 token-feature mapping review is hash-pinned rather than repeated.',
      'Saved fitted objects become inert state holders. Manual float32 scaling and coefficient sigmoid are used; no fit/transform/predict production methods called.',
      'The expanded grid was proposed after earlier development observations; calibration selection is optimistic and not sealed-test evidence.']}
    try:
        with threadpool_limits(limits=4):audit(report)
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('COMMON_C_INDEPENDENT_AUDIT_PASSED',sha(REPORT),flush=True)

