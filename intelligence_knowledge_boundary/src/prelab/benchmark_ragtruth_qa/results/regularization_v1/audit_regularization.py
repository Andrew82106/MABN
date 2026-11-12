"""Bounded independent QA regularization audit; no fitting/GPU/test.

Reuses the prior audited fit-only scaler/PCA, and checks all unchanged state and
weight bindings. Manual float32 scaling/coefficient sigmoid replays all24
candidates in small blocks without creating a full transformed design.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,traceback
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=OUT.parent/'development_v1'
HELPER=OLD/'audit_coefficients_qa.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='a7a25ec0c346ad32621769762ff97b7f81c7caa6d959658a88508fb4ab5c1dae'
spec=importlib.util.spec_from_file_location('independent_old_qa_audit',HELPER);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
read,sha,near=q.read,q.sha,q.near
REPORT=OUT/'INDEPENDENT_AUDIT_REGULARIZATION.json'
METHODS=('lookback_mean','lookback_nll','layerband_slots','hidden64_lookback_nll')
NEW=(1e-6,1e-5,1e-4);REUSED=(.001,.01,.1);N=210364;NFIT=168123

def save(r):REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def choose(y,s):
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True);p=np.bincount(inv,weights=y,minlength=len(u)).astype(int)
    tp=np.r_[np.cumsum(p[::-1])[::-1],0];count=np.r_[np.cumsum(n[::-1])[::-1],0]
    cutoffs=np.r_[u,np.nextafter(u[-1],np.inf)];f1=2*tp/(count+y.sum());precision=np.divide(tp,count,out=np.zeros(len(count)),where=count>0)
    j=max(range(len(count)),key=lambda i:(f1[i],precision[i],cutoffs[i]))
    return {'threshold':float(cutoffs[j]),'f1':float(f1[j]),'precision':float(precision[j]),'rows':len(y),'positive':int(y.sum())}
def metric(y,s,t):
    pred=s>=t;tp=int(y[pred].sum());fp=int(pred.sum())-tp;fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    u,inv,n=np.unique(s,return_inverse=True,return_counts=True);pos=np.bincount(inv,weights=y,minlength=len(u));neg=n-pos
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(y.sum()*(len(y)-y.sum()))
    ap=np.sum(pos[::-1]/y.sum()*np.cumsum(pos[::-1])/np.cumsum(n[::-1]))
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'auroc':float(auc),'average_precision':float(ap)}


def audit(report):
    complete=read(OUT/'complete.json');cfg=read(OUT/'protocol.json');summary=read(OUT/'summary.json');oldcomplete=read(OLD/'complete.json')
    assert complete['official_test_opened'] is False and summary['official_test_opened'] is False and summary['new_fits']==12
    assert cfg['methods']==list(METHODS) and cfg['new_C']==list(NEW) and cfg['reused_C']==list(REUSED)
    assert cfg['implementation_sha256']==sha(ROOT/'src/run_regularization.py') and cfg['base_code_sha256']==sha(ROOT/'src/run_development.py')
    assert cfg['base_complete_sha256']==sha(OLD/'complete.json') and read(OUT/'started.json')['protocol_sha256']==sha(OUT/'protocol.json')
    for name,h in complete['files_sha256'].items():assert sha(OUT/name)==h
    prior_audit=OLD/'COEFFICIENT_AUDIT_QA.json'
    assert sha(prior_audit)=='b3d3de1eafb985dcddbbe7f9a44cf0b91ff4a8b1d2c144cd8863f0c7e9878a85'
    assert read(prior_audit)['status']=='passed'
    report['reused_fit_only_scaler_PCA_evidence']={'prior_audit_sha256':sha(prior_audit),'unchanged_state_rechecked':True,
        'limit':'No new PCA/scaler fits; reuse verified exactly against prior fitted objects already independently audited.'}
    answers,windows,indices=q.metadata();y=np.asarray([w['label'] for w in windows],int);ay=np.asarray([a['label'] for a in answers],int)
    _,weights_report=q.audit_weights(windows)
    assert sha(OLD/'training_weights.npz')==oldcomplete['files_sha256']['training_weights.npz']
    assert sha(OLD/'fit_keys.json')==oldcomplete['files_sha256']['fit_keys.json']
    arrays={k:np.load(OLD/f'matrices/{k}.npy',mmap_mode='r',allow_pickle=False) for k in ('base','slots','hidden')}
    mm=read(OLD/'matrix_manifest.json');assert sha(OLD/'matrix_manifest.json')==oldcomplete['files_sha256']['matrix_manifest.json']
    for k,h in mm['files_sha256'].items():assert sha(OLD/f'matrices/{k}.npy')==h
    assert sha(OLD/'hidden_pca.pkl')==mm['pca_sha256']
    report['weights']=weights_report;families={};metric_error=0.;threshold_count=0;metric_blocks=0;old_replayed=0;new_replayed=0
    for method in METHODS:
        origin=q.unpickle(OLD/f'{method}_C0.001.pkl');sc=origin['scaler'];objects=[];saved_scores=[];entries=[]
        expected_order=REUSED+NEW;table=summary['all_candidates'][method]
        assert [e['C'] for e in table]==list(expected_order)
        assert origin['fit_only'] and (origin['fit_rows'],origin['fit_groups'],origin['fit_answers'])==(NFIT,615,634)
        assert origin['weights_sha256']==sha(OLD/'training_weights.npz') and origin['fit_keys_sha256']==sha(OLD/'fit_keys.json')
        for j,c in enumerate(expected_order):
            folder=OLD if c in REUSED else OUT;name=f'{method}_C{c:g}';obj=q.unpickle(folder/(name+'.pkl'));entry=read(folder/(name+'_result.json'))
            for suffix in ('.pkl','_scores.npz','_result.json'):
                assert sha(folder/(name+suffix))==(oldcomplete if c in REUSED else complete)['files_sha256'][name+suffix]
            if c in REUSED:assert table[j]==dict(entry,source='frozen_previous_candidate');old_replayed+=1
            else:
                assert table[j]==entry and entry['source']=='new_fit';new_replayed+=1
                assert obj['protocol_sha256']==sha(OUT/'protocol.json') and obj['source_scaler_sha256']==sha(OLD/f'{method}_C0.001.pkl')
            for key in ('fit_only','fit_rows','fit_groups','fit_answers','width','weights_sha256','fit_keys_sha256','pca_sha256'):
                assert obj[key]==origin[key]
            for key in ('mean_','var_','scale_','n_samples_seen_','n_features_in_','with_mean','with_std'):
                assert np.array_equal(getattr(obj['scaler'],key),getattr(sc,key))
            model=obj['model'];assert obj['C']==model.C==c and model.solver=='liblinear' and model.penalty=='l2'
            assert model.random_state==20260924 and model.class_weight is None and model.max_iter==2000 and model.n_iter_.max()<2000
            assert model.coef_.shape==(1,q.WIDTH[method]) and model.intercept_.shape==(1,) and np.array_equal(model.classes_,[0,1])
            with np.load(folder/(name+'_scores.npz'),allow_pickle=False) as z:s=z['window_scores'].copy();a=z['answer_scores'].copy()
            assert s.shape==(N,) and a.shape==(793,) and np.isfinite(s).all()
            assert np.array_equal(np.asarray([s[indices[x['answer_id']]].max() for x in answers]),a)
            ts={'window':choose(y[NFIT:],s[NFIT:]),'answer':choose(ay[634:],a[634:])}
            assert ts==entry['thresholds']==obj['thresholds'];threshold_count+=2
            key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c]
            assert key==entry['selection_key'] and tuple(key)==tuple(obj['selection_key'])
            for part,wl,wr,al,ar in [('fit',0,NFIT,0,634),('calibration',NFIT,N,634,793)]:
                for unit,yy,ss,cut in [('windows',y[wl:wr],s[wl:wr],ts['window']['threshold']),('answers',ay[al:ar],a[al:ar],ts['answer']['threshold'])]:
                    got=metric(yy,ss,cut);reference=entry['metrics'][part][unit];assert got.keys()==reference.keys()
                    for k,v in got.items():
                        if isinstance(v,int):assert v==reference[k]
                        else:metric_error=max(metric_error,near(v,reference[k],(method,c,part,unit,k),atol=1e-12))
                    metric_blocks+=1
            objects.append(obj);saved_scores.append(s);entries.append(entry)
        # Bound memory: one16384row float32 design, six score buffers.
        replay=[np.empty(N) for _ in objects];standard=np.load(OLD/f'matrices/{method}_fit_standardized.npy',mmap_mode='r',allow_pickle=False)
        for left in range(0,N,16384):
            right=min(left+16384,N);z=q.design(arrays,method,left,right).copy();z-=sc.mean_;z/=sc.scale_
            if left<NFIT:
                end=min(right,NFIT);assert np.array_equal(z[:end-left],standard[left:end])
            for j,obj in enumerate(objects):replay[j][left:right]=q.sigmoid(z@obj['model'].coef_[0]+obj['model'].intercept_[0])
        errors=[near(got,saved,(method,c,'coef full replay'),rtol=0,atol=2e-12) for c,got,saved in zip(expected_order,replay,saved_scores)]
        selected=max(range(6),key=lambda j:entries[j]['selection_key']);assert summary['selected'][method]==table[selected]
        prior=summary['old_selected'][method];assert prior==read(OLD/'summary.json')['selected'][method]
        chosen=entries[selected]
        families[method]={'all6_scaler_and_fit_provenance_exact':True,'all_fit_standardized_rows_exact':True,
            'selected_C':chosen['C'],'selected_candidate_index':selected,'full_score_replay_max_abs':max(errors),
            'candidate_replays':[{'C':c,'new_fit':c in NEW,'max_abs':e} for c,e in zip(expected_order,errors)],
            'old_selected_C':prior['C'],'old_cal_window_F1':prior['metrics']['calibration']['windows']['f1'],
            'old_cal_answer_F1':prior['metrics']['calibration']['answers']['f1'],
            'new_cal_window_F1':chosen['metrics']['calibration']['windows']['f1'],
            'new_cal_answer_F1':chosen['metrics']['calibration']['answers']['f1']}
        report['families']=families;save(report);print('REGULARIZATION_FAMILY_PASSED',method,flush=True)
    assert old_replayed==new_replayed==12
    report.update(status='passed',blockers=[],new_fitted_candidates_checked=12,old_frozen_candidates_replayed=12,
        full_window_score_replays=24*N,full_answermax_checks=24*793,max_coef_score_abs=max(f['full_score_replay_max_abs'] for f in families.values()),
        thresholds_exact=threshold_count,selection_keys_exact=24,selected_Cs_exact=4,metric_blocks=metric_blocks,max_metric_abs=metric_error,
        denominators={'fit':{'answers':634,'positive_answers':328,'groups':615,'windows':NFIT,'positive_windows':21477},
                      'calibration':{'answers':159,'positive_answers':100,'groups':154,'windows':42241,'positive_windows':5984}},
        no_calibration_refit=True,old_candidate_metadata_and_scores_unchanged=True,complete_sha256=sha(OUT/'complete.json'),audit_script_sha256=sha(__file__))
    lines=['# QA 更强正则化独立审计','',
        '通过，无数值阻断。12个新LR与12个冻结旧候选全部手工系数回放；旧fit-only scaler/PCA、标签和loss权重保持，标准化fit矩阵逐行一致。48个校准阈值及4个最终C选择一致。','',
        '| 方法 | 选中C | cal定位F1 原→扩展 | cal整答F1 原→扩展 |','|---|---:|---:|---:|']
    for m,f in families.items():lines.append(f"| {m} | {f['selected_C']:g} | {f['old_cal_window_F1']:.6f} → {f['new_cal_window_F1']:.6f} | {f['old_cal_answer_F1']:.6f} → {f['new_cal_answer_F1']:.6f} |")
    lines+=['','两分区仍为fit634/cal159回答、168123/42241可评窗口。每答由全部可评窗口取max，没有按正标签筛选窗口。',
        '本次审计未训练、使用GPU或读取test。既有PCA/scaler的fit-only证据来自已核实旧审计与本次逐项复用核验，未重新拟合；校准用于选C/阈值，结果有选择乐观，不能证明泛化提升或标签问题。']
    (OUT/'INDEPENDENT_AUDIT_REGULARIZATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    report={'status':'running','utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
        'no_new_fit_or_GPU':True,'official_QA_test_or_heldout_read':False,'production_imports':False,'frozen_files_modified':False,
        'limits':['Only bounded extension; old PCA/scaler fit audit is referenced and exact reused state is checked, without a redundant full old-source audit.',
                  'Same calibration chooses C and thresholds, hence selection-optimistic; no new-test evidence.',
                  'The synthetic inverse C/weight equivalence self-test was not re-fit during this audit.']}
    try:
        with threadpool_limits(limits=4):audit(report)
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('QA_REGULARIZATION_AUDIT_PASSED',sha(REPORT),flush=True)
