"""Common stronger-C budget for five frozen Lookback definitions; CPU only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,pickle,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import run_lookback_controls_v2 as ctrl

q=ctrl.original;ROOT=q.ROOT;OLD=ctrl.OUT;REG=ROOT/'results/regularization_v1'
OUT=ROOT/'results/lookback_regularization_v1';METHODS=ctrl.METHODS;NEW_C=(1e-5,1e-4);GRID=NEW_C+(.001,.01,.1)
def now():return datetime.now(timezone.utc).isoformat()

def protocol():
    return {'version':'qa-lookback-common-stronger-regularization-v1','methods':list(METHODS),'C':list(GRID),'added_C':list(NEW_C),
     'new_LR_fits':8,'frozen_model_replays':17,'total_candidates':25,
     'scope':'Unchanged634 fit /159 calibration,168123/42241 raw4BPE windows. Official test remains sealed.',
     'source_matrices':'Frozen lookback_controls_v2 five1024-dimensional window matrices; no new pooling, feature extraction or scaler fitting.',
     'fit':'Four control definitions each fit two extra Cs. Same original fit-only scaler, standardized fit matrix, labels and loss weights; total loss mass168123 unchanged.',
     'anchor':'Five original source-post-no-header models reused. Added Cs1e-5/1e-4 come from regularization_v1; old three from lookback_controls_v2. No anchor refitting.',
     'replays':'Existing15 control candidates plus two extra anchor candidates copied as frozen model/score artifacts; exact scores/thresholds/metrics retained. Independent coefficient replay tolerance1e-12 does not replace stored scores.',
     'solver':'Same liblinear L2 max_iter2000 random_state20260924, four CPU threads; no class_weight outside frozen sample weights.',
     'selection':'Same separate calibration thresholds and q.selection_key; max min(windowF1,answerF1), then windowF1,precision,smaller C. All five definitions share exactly five Cs.',
     'answer':'Maximum all unchanged eligible windows. Same human-QA refusal policy and original labels.',
     'primary_closer_official':'lb_prefix_pre_header','no_PCA_or_scaler_fit':True,'no_GPU':True,'no_official_test':True,
     'limitation':'Stronger grid added after observing prior development results; repeated calibration selection is optimistic, not independent final performance.',
     'code_sha256':q.sha(__file__),'controls_runner_sha256':q.sha(ctrl.__file__),
     'controls_complete_sha256':q.sha(OLD/'complete.json'),'regularization_complete_sha256':q.sha(REG/'complete.json'),
     'gold_manifest_sha256':q.sha(ROOT/'data/gold_manifest.json')}

def source_for(method,c):
    if method==ctrl.ANCHOR and c in NEW_C:return REG,f'lookback_mean_C{c:g}'
    return OLD,f'{method}_C{c:g}'

def verify_sources():
    files={}
    for folder in (OLD,REG):
        for n,h in q.read(folder/'complete.json')['files_sha256'].items():
            p=folder/n;assert q.sha(p)==h,p;files[str(p.resolve())]=h
    mm=q.read(OLD/'matrix_manifest.json')
    for method,h in mm['files_sha256'].items():
        p=OLD/'matrices'/(method+'.npy');assert q.sha(p)==h;files[str(p.resolve())]=h
    for p in (Path(__file__),OUT/'protocol.json',OLD/'complete.json',REG/'complete.json'):
        files[str(p.resolve())]=q.sha(p)
    return {'files_sha256':files}

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    assert len(METHODS)*len(GRID)==25 and (len(METHODS)-1)*len(NEW_C)==8
    q.save(OUT/'protocol.json',protocol())
    q.save(OUT/'PREFLIGHT.json',{'passed':True,'common_grid':list(GRID),'new_fits':8,'replays':17,'no_synthetic_or_real_model_fits':True,'GPU_used':False})

def run():
    cfg=q.read(OUT/'protocol.json');assert cfg==protocol();assert not (OUT/'started.json').exists()
    snapshot=verify_sources();meta,(_,loss,_,y),keys=ctrl.geometry_and_weights()
    with np.load(OLD/'training_weights.npz',allow_pickle=False) as z:
        assert np.array_equal(loss,z['loss_weights']) and np.array_equal(y,z['y'])
    assert keys==q.read(OLD/'fit_keys.json') and abs(loss.sum()-168123)<1e-6
    q.save(OUT/'source_snapshot.json',snapshot)
    q.save(OUT/'started.json',{'utc':now(),'protocol_sha256':q.sha(OUT/'protocol.json'),'source_snapshot_sha256':q.sha(OUT/'source_snapshot.json')})
    start=time.perf_counter();families={};selected={};artifacts=[];replay_checks=[];newfits=0
    for method in METHODS:
        matrix=np.load(OLD/'matrices'/(method+'.npy'),mmap_mode='r');assert matrix.shape==(210364,1024)
        reference_path=OLD/f'{method}_C0.001.pkl';reference=pickle.loads(reference_path.read_bytes());sc=reference['scaler']
        zfit=None
        if method!=ctrl.ANCHOR:
            zp=OLD/'matrices'/(method+'_fit_standardized.npy');zfit=np.load(zp,mmap_mode='r')
            assert zfit.shape==(168123,1024) and zfit.dtype==np.float32
            for left in range(0,168123,ctrl.BATCH):
                right=min(left+ctrl.BATCH,168123)
                assert np.array_equal(sc.transform(np.asarray(matrix[left:right])).astype(np.float32),zfit[left:right])
            snapshot['files_sha256'][str(zp.resolve())]=q.sha(zp)
        family=[]
        for c in GRID:
            name=f'{method}_C{c:g}';mp=OUT/(name+'.pkl');sp=OUT/(name+'_scores.npz');begin=time.perf_counter()
            if c not in NEW_C or method==ctrl.ANCHOR:
                folder,oldname=source_for(method,c);source_model=folder/(oldname+'.pkl');source_scores=folder/(oldname+'_scores.npz')
                obj=pickle.loads(source_model.read_bytes());prior=q.read(folder/(oldname+'_result.json'))
                assert obj['C']==c
                for k in ('mean_','var_','scale_','n_samples_seen_'):assert np.array_equal(getattr(sc,k),getattr(obj['scaler'],k))
                with np.load(source_scores,allow_pickle=False) as z:scores=z['window_scores'].copy();answer=z['answer_scores'].copy()
                replay,ra=ctrl.score(meta,matrix,obj);error=float(np.max(np.abs(replay-scores)))
                assert error<=1e-12 and float(np.max(np.abs(ra-answer)))<=1e-12
                ts=ctrl.thresholds(meta,scores,answer);assert ts==prior['thresholds']
                assert q.metrics(meta,scores,ts)==prior['metrics']
                assert np.array_equal(q.answer_scores(meta,scores),answer)
                mp.write_bytes(source_model.read_bytes());sp.write_bytes(source_scores.read_bytes())
                assert q.sha(mp)==q.sha(source_model) and q.sha(sp)==q.sha(source_scores)
                mode='frozen_replay';provenance={'model':str(source_model.resolve()),'scores':str(source_scores.resolve()),'model_sha256':q.sha(source_model),'scores_sha256':q.sha(source_scores)}
                replay_checks.append({'candidate':name,'max_coefficient_replay_abs':error,'stored_scores_thresholds_metrics_exact':True})
            else:
                model=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=20260924)
                model.fit(zfit,y,sample_weight=loss);assert model.n_iter_.max()<2000
                obj={**reference,'model':model,'C':c,'mode':'new_fit','source_scaler_model_sha256':q.sha(reference_path),
                     'protocol_sha256':q.sha(OUT/'protocol.json'),'fit_standardized_sha256':q.sha(zp)}
                scores,answer=ctrl.score(meta,matrix,obj);ts=ctrl.thresholds(meta,scores,answer)
                obj['thresholds']=ts;obj['selection_key']=q.selection_key(ts,c)
                mp.write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(sp,window_scores=scores,answer_scores=answer)
                mode='new_fit';provenance={'source_scaler_model':str(reference_path.resolve()),'source_scaler_model_sha256':q.sha(reference_path)};newfits+=1
            entry={'candidate':name,'method':method,'C':c,'mode':mode,'thresholds':ts,'selection_key':list(q.selection_key(ts,c)),
             'metrics':q.metrics(meta,scores,ts),'model_sha256':q.sha(mp),'scores_sha256':q.sha(sp),'provenance':provenance,
             'iterations':obj['model'].n_iter_.tolist(),'seconds':time.perf_counter()-begin}
            q.save(OUT/(name+'_result.json'),entry);family.append(entry);artifacts.extend([name+'.pkl',name+'_scores.npz',name+'_result.json'])
            print('LOOKBACK_COMMON_C',name,mode,round(entry['seconds'],2),flush=True)
        families[method]=family;selected[method]=max(family,key=lambda e:e['selection_key'])
        del matrix,zfit;gc.collect()
    assert newfits==8 and len(replay_checks)==17 and cfg==protocol()
    for p,h in snapshot['files_sha256'].items():assert q.sha(p)==h,p
    q.save(OUT/'source_snapshot.json',snapshot)
    summary={'selected':selected,'all_candidates':families,'old_replay_checks':replay_checks,'new_fits':8,'frozen_replays':17,'candidate_count':25,
      'seconds':time.perf_counter()-start,'official_test_opened':False,'calibration_selection_optimistic':True,'all_common_C':list(GRID),'primary_closer_official':ctrl.METHODS[1]}
    q.save(OUT/'summary.json',summary)
    report=['五组 Lookback 定义使用同一更强正则化范围，仅开发校准成绩。','', '| 定义 | C | 窗口 F1 | 整答 F1 |','|---|---:|---:|---:|']
    for method in METHODS:
        e=selected[method];m=e['metrics']['calibration'];report.append(f"| {method} | {e['C']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    report+=['','新增8次拟合，17个既有候选保留原模型和预测；全部标签、矩阵、scaler与损失总质量不变。',
             '每族共同比较 C=1e-5、1e-4、.001、.01、.1。较接近官方的主对照仍为 prefix_pre_header；不能见成绩后改称其他定义为主方法。',
             '新增网格是在观察既有开发结果后提出的，因此校准选型仍乐观，封存测试未打开。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    artifacts+=['protocol.json','PREFLIGHT.json','started.json','source_snapshot.json','summary.json','REPORT.md']
    q.save(OUT/'complete.json',{'utc':now(),'files_sha256':{n:q.sha(OUT/n) for n in artifacts},'new_fits':8,'frozen_replays':17,'official_test_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','run']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':initialize()
        else:run()
