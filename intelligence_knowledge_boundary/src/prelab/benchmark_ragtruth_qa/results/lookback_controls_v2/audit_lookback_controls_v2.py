"""Bounded independent audit of completed five-definition QA controls.
No fitting, production imports, GPU or official-test/withheld content.
Small blocks and one answer feature cache at a time; frozen files read-only.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import importlib.util,json,hashlib,traceback,gc
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=OUT.parent/'development_v1';DATA=ROOT/'data/lookback_controls_v2'
HELPER=OLD/'audit_coefficients_qa.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='a7a25ec0c346ad32621769762ff97b7f81c7caa6d959658a88508fb4ab5c1dae'
spec=importlib.util.spec_from_file_location('independent_previous_qa_audit',HELPER);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
REPORT=OUT/'INDEPENDENT_AUDIT_LOOKBACK_CONTROLS_V2.json'
METHODS=('lb_source_pre_header','lb_prefix_pre_header','lb_source_post_header','lb_prefix_post_header','lb_source_post_legacy')
CS=(.001,.01,.1);NFIT=168123;NCAL=42241;N=NFIT+NCAL;BATCH=16384
sha,read,near=q.sha,q.read,q.near

def save(r):REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def choose(y,s):
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True);p=np.bincount(inv,weights=y,minlength=len(u)).astype(int)
    tp=np.r_[np.cumsum(p[::-1])[::-1],0];count=np.r_[np.cumsum(n[::-1])[::-1],0]
    cuts=np.r_[u,np.nextafter(u[-1],np.inf)];f1=2*tp/(count+y.sum());precision=np.divide(tp,count,out=np.zeros(len(count)),where=count>0)
    j=max(range(len(count)),key=lambda i:(f1[i],precision[i],cuts[i]))
    return {'threshold':float(cuts[j]),'f1':float(f1[j]),'precision':float(precision[j]),'rows':len(y),'positive':int(y.sum())}
def metric(y,s,t):
    pred=s>=t;tp=int(y[pred].sum());fp=int(pred.sum())-tp;fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True);pos=np.bincount(inv,weights=y,minlength=len(u));neg=n-pos
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(y.sum()*(len(y)-y.sum()))
    ap=np.sum(pos[::-1]/y.sum()*np.cumsum(pos[::-1])/np.cumsum(n[::-1]))
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,
      'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'auroc':float(auc),'average_precision':float(ap)}
def keysafe(p,root):
    p=Path(p).resolve();p.relative_to(root.resolve());return p
def exact_obj(a,b):
    for k in ('mean_','var_','scale_','n_samples_seen_','n_features_in_','with_mean','with_std'):
        assert np.array_equal(getattr(a['scaler'],k),getattr(b['scaler'],k)),k
    for k in ('coef_','intercept_','classes_','n_iter_'):
        assert np.array_equal(getattr(a['model'],k),getattr(b['model'],k)),k

def audit(report):
    assert (OUT/'complete.json').exists(),'Do not audit partial candidate results'
    complete=read(OUT/'complete.json');cfg=read(ROOT/'lookback_controls_scoring_protocol_v2.json');summary=read(OUT/'summary.json')
    assert complete['status']=='complete_development_only' and complete['new_fits']==12 and complete['legacy_replays']==3
    assert not complete['official_test_opened'] and not summary['official_test_opened']
    assert complete['code_sha256']==sha(ROOT/'src/run_lookback_controls_v2.py')
    assert complete['protocol_sha256']==sha(ROOT/'lookback_controls_scoring_protocol_v2.json')
    assert cfg['methods']==list(METHODS) and cfg['C']==list(CS) and cfg['width']==1024
    assert cfg['new_LR_fits']==12 and cfg['legacy_frozen_model_replays']==3 and cfg['no_NLL_PCA_or_hidden']
    verified=0
    for name,h in complete['files_sha256'].items():
        assert sha(keysafe(OUT/name,OUT))==h;verified+=1
    snap=read(OUT/'source_snapshot.json')
    assert read(OUT/'started.json')['source_snapshot_sha256']==sha(OUT/'source_snapshot.json')
    assert not snap['test_read']
    for p,h in snap['files_sha256'].items():
        p=keysafe(p,ROOT);assert sha(p)==h;verified+=1
    mm=read(OUT/'matrix_manifest.json')
    assert mm['rows']==N and mm['fit_rows']==NFIT and mm['calibration_rows']==NCAL and mm['width']==1024
    assert mm['source_snapshot_sha256']==sha(OUT/'source_snapshot.json') and mm['all_anchor_windows_exact']
    # Existing development geometry/weights are independently recomputed once;
    # new exports must equal them exactly.
    answers,windows,indices=q.metadata();y=np.asarray([w['label'] for w in windows],int);ay=np.asarray([a['label'] for a in answers],int)
    b,wr=q.audit_weights(windows);report['weights']=wr
    with np.load(OLD/'training_weights.npz',allow_pickle=False) as z:oldw={k:z[k] for k in z.files}
    with np.load(OUT/'training_weights.npz',allow_pickle=False) as z:neww={k:z[k] for k in z.files}
    assert oldw.keys()==neww.keys() and all(np.array_equal(oldw[k],neww[k]) for k in oldw)
    assert read(OUT/'fit_keys.json')==read(OLD/'fit_keys.json')
    assert read(OUT/'score_index.json')==read(OLD/'score_index.json')
    sig=read(DATA/'signature.json');fm=read(DATA/'feature_manifest.json')
    assert fm['complete'] and fm['completed_count']==793 and len(fm['records'])==793 and fm['all_old_anchors_exact']
    assert not fm['test_read'] and not fm['labels_used']
    assert fm['signature_sha256']==digest(sig)==snap['feature_signature_sha256']
    assert fm['variants']==sig['variants']==list(METHODS)
    assert [r['response_id'] for r in fm['records']]==sig['response_ids_in_order']==[a['response_id'] for a in answers]
    report['complete_feature_gate']={'rows':793,'fit':634,'calibration':159,'order_exact':True,'signature_exact':True}
    matrices={}
    for method in METHODS:
        p=OUT/'matrices'/(method+'.npy');assert sha(p)==mm['files_sha256'][method];verified+=1
        x=np.load(p,mmap_mode='r',allow_pickle=False);assert x.shape==(N,1024) and x.dtype==np.float32;matrices[method]=x
    oldbase=np.load(OLD/'matrices/base.npy',mmap_mode='r',allow_pickle=False)
    for l in range(0,N,BATCH):assert np.array_equal(matrices[METHODS[-1]][l:l+BATCH],oldbase[l:l+BATCH,:1024])
    del oldbase
    families={};metric_error=0.;thresholds=0;metric_blocks=0
    for method in METHODS:
        x=matrices[method];objects=[q.unpickle(OUT/f'{method}_C{c:g}.pkl') for c in CS]
        mean,var,scale,count=q.moments({'base':x},'lookback_mean',b);sc=objects[0]['scaler']
        family={'fit_only_scaler':{'mean_max_abs':near(mean,sc.mean_,method+' mean',atol=2e-11),
          'variance_max_abs':near(var,sc.var_,method+' variance',atol=2e-10),
          'scale_max_abs':near(scale,sc.scale_,method+' scale',atol=2e-11),
          'weight_count_max_abs':near(count,sc.n_samples_seen_,method+' mass')},'candidates':[],'fit_standardized_exact':True}
        standard=np.load((OLD/'matrices/lookback_mean_fit_standardized.npy') if method==METHODS[-1] else (OUT/'matrices'/(method+'_fit_standardized.npy')),mmap_mode='r',allow_pickle=False)
        assert standard.shape==(NFIT,1024) and standard.dtype==np.float32
        replays=[np.empty(N) for _ in CS];saved=[];entries=[]
        for c,obj in zip(CS,objects):
            assert obj['method']==method and obj['C']==c and obj['width']==1024 and obj['fit_only']
            assert (obj['fit_rows'],obj['fit_groups'],obj['fit_answers'])==(NFIT,615,634)
            assert obj['weights_sha256']==sha(OUT/'training_weights.npz') and obj['fit_keys_sha256']==sha(OUT/'fit_keys.json')
            assert obj['feature_signature_sha256']==digest(sig)
            assert obj['mode']==('frozen_legacy_replay' if method==METHODS[-1] else 'new_fit')
            for k in ('mean_','var_','scale_','n_samples_seen_','n_features_in_'):
                assert np.array_equal(getattr(obj['scaler'],k),getattr(sc,k))
            model=obj['model'];assert model.coef_.shape==(1,1024) and model.intercept_.shape==(1,) and np.array_equal(model.classes_,[0,1])
            assert model.solver=='liblinear' and model.penalty=='l2' and model.C==c and model.random_state==20260924
            assert model.class_weight is None and model.max_iter==2000 and model.n_iter_.max()<2000
            entry=read(OUT/f'{method}_C{c:g}_result.json')
            assert entry['model_sha256']==sha(OUT/f'{method}_C{c:g}.pkl') and entry['scores_sha256']==sha(OUT/f'{method}_C{c:g}_scores.npz')
            with np.load(OUT/f'{method}_C{c:g}_scores.npz',allow_pickle=False) as z:s=z['window_scores'].copy();a=z['answer_scores'].copy()
            assert s.shape==(N,) and a.shape==(793,) and np.isfinite(s).all()
            assert np.array_equal(a,np.asarray([s[indices[aa['answer_id']]].max() for aa in answers]))
            ts={'window':choose(y[NFIT:],s[NFIT:]),'answer':choose(ay[634:],a[634:])}
            assert ts==entry['thresholds']==obj['thresholds'];thresholds+=2
            key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c]
            assert key==entry['selection_key'] and tuple(key)==tuple(obj['selection_key'])
            for part,wl,wr_,al,ar in [('fit',0,NFIT,0,634),('calibration',NFIT,N,634,793)]:
                for unit,yy,ss,cut in [('windows',y[wl:wr_],s[wl:wr_],ts['window']['threshold']),('answers',ay[al:ar],a[al:ar],ts['answer']['threshold'])]:
                    got=metric(yy,ss,cut);reference=entry['metrics'][part][unit];assert got.keys()==reference.keys()
                    for k,v in got.items():
                        if isinstance(v,int):assert v==reference[k]
                        else:metric_error=max(metric_error,near(v,reference[k],(method,c,part,unit,k),atol=1e-12))
                    metric_blocks+=1
            if method==METHODS[-1]:
                old=OLD/f'lookback_mean_C{c:g}.pkl';prior=q.unpickle(old);exact_obj(obj,prior)
                assert obj['source_original_model_sha256']==sha(old)
                with np.load(OLD/f'lookback_mean_C{c:g}_scores.npz',allow_pickle=False) as z:
                    assert np.array_equal(s,z['window_scores']) and np.array_equal(a,z['answer_scores'])
                oe=read(OLD/f'lookback_mean_C{c:g}_result.json')
                assert ts==oe['thresholds'] and entry['metrics']==oe['metrics'] and key==oe['selection_key']
            saved.append((s,a));entries.append(entry)
        for l in range(0,N,BATCH):
            r=min(l+BATCH,N);z=np.asarray(x[l:r]).copy();assert np.isfinite(z).all();z-=sc.mean_;z/=sc.scale_
            if l<NFIT:assert np.array_equal(z[:min(r,NFIT)-l],standard[l:min(r,NFIT)])
            for j,obj in enumerate(objects):replays[j][l:r]=q.sigmoid((z@obj['model'].coef_.T+obj['model'].intercept_).ravel())
        for c,obj,got,(s,a) in zip(CS,objects,replays,saved):
            e=near(got,s,(method,c,'coef replay'),rtol=0,atol=2e-12)
            av=np.asarray([got[indices[aa['answer_id']]].max() for aa in answers])
            ae=near(av,a,(method,c,'manual answermax'),rtol=0,atol=2e-12)
            family['candidates'].append({'C':c,'mode':obj['mode'],'coefficient_max_abs':e,'answermax_max_abs':ae})
        assert summary['all_candidates'][method]==entries
        selected=max(entries,key=lambda e:e['selection_key']);assert summary['selected'][method]==selected
        family.update(selected_C=selected['C'],cal_window_F1=selected['metrics']['calibration']['windows']['f1'],cal_answer_F1=selected['metrics']['calibration']['answers']['f1'])
        families[method]=family;report['families']=families;save(report)
        print('LB_V2_COEFFICIENT_FAMILY_PASSED',method,flush=True)
        del objects,replays,saved,standard,mean,var,scale,z;gc.collect()
    report.update(thresholds_exact=thresholds,selection_keys_exact=15,selected_Cs_exact=5,metric_blocks=metric_blocks,max_metric_abs=metric_error,
      manual_window_score_count=15*N,manual_answermax_count=15*793,
      max_coefficient_score_abs=max(c['coefficient_max_abs'] for f in families.values() for c in f['candidates']),
      old_three_models_scalers_scores_thresholds_metrics_exact=True,all_original_weights_and_fit_keys_exact=True)
    save(report)
    # Phase 2. Stream one answer's token arrays. Compare every raw-window mean
    # against all five stored matrices, including punctuation and byte fallback.
    toks={}
    keep=['response_id','source_id','group_id','partition','token_count','token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw','lexical_mask']
    for part in ('fit','calibration'):
        for raw in lines(ROOT/'data'/f'tokens_{part}.jsonl'):
            t={k:raw[k] for k in keep};del raw;assert t['partition']==part;toks[t['response_id']]=t
    plan={p['response_id']:{'hash':digest(p),'source_id':p['source_id'],'group_id':p['group_id'],'partition':p['partition']} for p in lines(ROOT/'data/feature_preparation/plans.jsonl')}
    layout={lo['response_id']:digest(lo) for lo in lines(DATA/'layouts.jsonl')}
    records={r['response_id']:r for r in fm['records']}
    assert set(plan)==set(layout)==set(records)==set(toks)=={a['response_id'] for a in answers}
    counts=Counter()
    for ai,a in enumerate(answers):
        rid=a['response_id'];t=toks[rid];rec=records[rid];p=(DATA/'features'/f'{rid}.npz').resolve();side=read(p.with_suffix('.json'))
        assert Path(rec['npz']).resolve()==p and Path(rec['json']).resolve()==p.with_suffix('.json')
        assert sha(p)==rec['npz_sha256']==side['npz_sha256'];assert sha(p.with_suffix('.json'))==rec['json_sha256']
        assert rec['plan_sha256']==side['plan_sha256']==plan[rid]['hash'] and rec['layout_sha256']==side['layout_sha256']==layout[rid]
        assert side['signature_sha256']==digest(sig) and side['complete'] and side['legacy_anchor_exact']
        for k in ('response_id','source_id','group_id','partition'):assert side[k]==rec[k]==t[k]
        for k in ('source_id','group_id','partition'):assert plan[rid][k]==t[k]
        with np.load(p,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        assert set(raw)==set(METHODS)|{'token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw'}
        for k in ('token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw'):assert np.array_equal(raw[k],t[k])
        oldrec=sig['old_records'][rid];oldpath=ROOT/'data/features'/f'{rid}.npz'
        assert sha(oldpath)==oldrec['npz_sha256']
        with np.load(oldpath,allow_pickle=False) as z:assert np.array_equal(raw[METHODS[-1]],z['lb'])
        for m in METHODS:
            assert raw[m].shape==(t['token_count'],1024) and raw[m].dtype==np.float32
            assert np.isfinite(raw[m]).all() and np.all((raw[m]>=0)&(raw[m]<=1))
        for wi in indices[a['answer_id']]:
            ix=windows[wi]['token_indices'];assert ix==list(range(ix[0],ix[0]+min(4,t['token_count'])))
            assert any(t['lexical_mask'][j] for j in ix)
            counts['windows_with_nonlexical_raw_slots']+=any(not t['lexical_mask'][j] for j in ix)
            counts['short_windows']+=len(ix)<4
            for m in METHODS:assert np.array_equal(raw[m][ix].mean(axis=0),matrices[m][wi]),(rid,m,wi)
            counts['windows']+=1
        counts['answers']+=1;counts['raw_tokens']+=t['token_count']
        counts['negative_raw_first_offset_answers']+=t['response_token_offsets_raw'][0][0]<0
        counts['duplicate_adjacent_offsets']+=sum(t['response_token_offsets'][j]==t['response_token_offsets'][j-1] for j in range(1,t['token_count']))
        if (ai+1)%100==0 or ai==792:print('LB_V2_RAW_WINDOW_BINDING',ai+1,flush=True)
        del raw
    assert counts['answers']==793 and counts['raw_tokens']==213159 and counts['windows']==N
    assert complete==read(OUT/'complete.json')
    report.update(status='passed',blockers=[],feature_geometry={'all793_source_plan_layout_cache_identities_exact':True,
      'all_five_raw_window_mean_arrays_exact':True,'all793_legacy_token_LB_arrays_exact':True,
      'window_vectors_compared':5*N,'window_scalar_elements_compared':5*N*1024,'counts':dict(counts)},
      verified_top_level_files=verified,complete_sha256=sha(OUT/'complete.json'),audit_script_sha256=sha(__file__),
      denominators={'fit':{'answers':634,'positive_answers':328,'groups':615,'windows':NFIT,'positive_windows':21477},
                    'calibration':{'answers':159,'positive_answers':100,'groups':154,'windows':NCAL,'positive_windows':5984}})
    md=['通过，无数值阻断。四个新定义的12个LR与旧锚点3个冻结模型完成手工系数回放；30个校准阈值、15个选择key和5个C选择一致。原fit权重/索引不变，五组scaler均仅由fit数据重算验证。','',
      '793个缓存逐条绑定来源、计划、坐标与哈希；全部210364个窗口的五组raw BPE均值逐项精确一致，全部旧锚点token、窗口及三模型输出亦精确。','',
      '| 定义 | C | cal窗口F1 | cal整答F1 |','|---|---:|---:|---:|']
    for m,f in families.items():md.append(f"| {m} | {f['selected_C']:g} | {f['cal_window_F1']:.6f} | {f['cal_answer_F1']:.6f} |")
    md+=['','fit634/cal159，168123/42241可评窗口。上述cal同时用于选C和阈值，不能作为新测试成绩。未重新拟合、运行GPU或读取官方test/withheld；未重审全部旧提取链，也不称原生trace精确复现。']
    (OUT/'INDEPENDENT_AUDIT_LOOKBACK_CONTROLS_V2.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

if __name__=='__main__':
    report={'status':'running','audit_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
      'no_new_fit_or_GPU':True,'official_test_or_withheld_content_read':False,'production_imports':False,'frozen_files_modified':False,
      'limits':['New five-definition scoring audit only; previous geometry helper is hash-pinned and its independent formulas reused.',
        'No fitting, tokenizer/model replay or redundant complete old-source review. Calibration selects C and thresholds; outcomes remain selection optimistic.']}
    try:
        with threadpool_limits(limits=4):audit(report)
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('LB_V2_INDEPENDENT_AUDIT_PASSED',sha(REPORT),flush=True)

