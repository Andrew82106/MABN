"""Matched14/15-column LR after frozen heading-source features are complete.

prepare/check never fit a detector; run exits before loading data/writing started
if the independently scheduled GPU producer has not completed all1161 pairs.
"""
from pathlib import Path
import argparse
import os
import pickle
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run_development as q
import run_completed_score_fusion as upstream
import build_citation_heading_semantic as producer

OUT=q.ROOT/'results/citation_heading_semantic_lr_v1'
SEMANTIC=producer.OUT
SCOPE=producer.SCOPE
LEX=q.ROOT/'results/citation_alignment_v1'
CS=(.001,.01,.1)
MODES=('any_source_control','inherited_source_gap')


def protocol():
    p=producer.training_protocol()
    assert p['planned_fits']==18 and p['C']==list(CS) and p['modes']==list(MODES)
    assert p['peers']==list(upstream.PEERS)
    return {'version':'heading-semantic-lr-runner-v1','frozen_producer_training_protocol':p,
        'new_input_names':producer.NAMES,'producer':'citation_heading_semantic_v1',
        'input_widths':dict(zip(MODES,(14,15))),
        'feature_gate':'Must have features_complete statuscomplete/no_testtrue/trainedfalse, exact two names/shape/float32 and bound1161pair inference/387claims; all file hashes, original window order and no-scope zero checked before any fit.',
        'prior_reference':'Keep completed13-column heading_scope selected and all3C entries with scores/model/scaler hashes; no refit. They are historical references, not the matched14-column common-control family.',
        'all_candidates_retained':True,'threads':4,'GPU_used':False,'official_test_opened':False}


def matrices(pair,lexical,scope,semantic):
    n=len(pair)
    assert pair.shape==(n,2) and lexical.shape==(n,8) and scope.shape==(n,3) and semantic.shape==(n,2)
    assert all(np.isfinite(x).all() for x in (pair,lexical,scope,semantic))
    assert ((semantic>=0)&(semantic<=1)).all()
    assert not semantic[scope[:,0]==0].any()
    common=np.column_stack((pair,lexical,scope,semantic[:,0]))
    treatment=np.column_stack((common,semantic[:,1]))
    assert common.shape==(n,14) and treatment.shape==(n,15)
    assert np.array_equal(treatment[:,:14],common)
    return {MODES[0]:common,MODES[1]:treatment}


def tiny():
    rng=np.random.default_rng(20261010);n=32
    pair=rng.uniform(.1,.9,(n,2));lexical=rng.uniform(0,1,(n,8))
    scope=np.zeros((n,3));scope[::2]=rng.uniform(.1,1,(16,3))
    semantic=np.zeros((n,2));semantic[::2]=rng.uniform(0,1,(16,2))
    x=matrices(pair,lexical,scope,semantic)
    assert np.array_equal(x[MODES[0]][:,:2],pair)
    assert np.array_equal(x[MODES[0]][:,2:10],lexical)
    assert np.array_equal(x[MODES[0]][:,10:13],scope)
    assert np.array_equal(x[MODES[0]][:,13],semantic[:,0])
    assert np.array_equal(x[MODES[1]][:,14],semantic[:,1])
    weights=rng.uniform(.1,2,24)
    for m in MODES:
        a=StandardScaler().fit(x[m][:24],sample_weight=weights)
        altered=x[m].copy();altered[24:]=np.nan
        b=StandardScaler().fit(altered[:24],sample_weight=weights)
        assert np.array_equal(a.mean_,b.mean_) and np.array_equal(a.var_,b.var_)
    # Both levels must be determined from the same final window score vector.
    toy={'answer_windows':{'a':[0,1,2],'b':[3,4]},'answers':[{'response_id':'a'},{'response_id':'b'}]}
    assert np.array_equal(q.answer_scores(toy,np.array([.2,.8,.4,.3,.1])),[.8,.3])
    return {'passed':True,'common14_exact_prefix_of_treatment15':True,'original13_columns_preserved':True,
        'same_any_source_support_in_both_modes':True,'gap_is_only_additional_column':True,
        'no_scope_new_semantic_zero':True,'fit_only_scaler_outside_fit_NaN_test':True,
        'answer_max_same_windows_checked':True,'synthetic_only':True,'real_LR_fits':0,'GPU_used':False}


def source_hashes():
    paths=[Path(__file__),Path(q.__file__),Path(upstream.__file__),Path(producer.__file__),
        SEMANTIC/'preparation_freeze.json',SEMANTIC/'training_protocol.json',
        SCOPE/'complete.json',SCOPE/'summary.json',SCOPE/'features_complete.json',
        LEX/'complete.json',upstream.OUT/'complete.json',q.DATA/'gold_manifest.json']
    return {str(p.resolve()):q.sha(p) for p in paths}


def previous_references():
    complete=q.read(SCOPE/'complete.json');summary=q.read(SCOPE/'summary.json')
    assert q.sha(SCOPE/'summary.json')==complete['summary_sha256']
    refs={}
    for peer in upstream.PEERS:
        entries=summary['all_candidates'][peer]
        assert len(entries)==3 and {e['C'] for e in entries}==set(CS)
        for e in entries:
            family=e['candidate'].split('__C')[0]
            for file,key in [(e['candidate']+'.pkl','model_sha256'),(e['candidate']+'_scores.npz','scores_sha256'),
                             (family+'_scaler.pkl','scaler_sha256')]:
                assert q.sha(SCOPE/file)==e[key],file
        refs[peer]={'selected':summary['selected'][peer],'all_candidates':entries}
    return {'source_directory':str(SCOPE),'complete_sha256':q.sha(SCOPE/'complete.json'),
            'new_refits':0,'role':'Historical13-column reference, not new matched14-column control','peers':refs}


def prepare():
    assert not (OUT/'design_freeze.json').exists()
    OUT.mkdir(parents=True,exist_ok=True);producer.check()
    assert q.read(SEMANTIC/'training_protocol.json')==producer.training_protocol()
    q.save(OUT/'protocol.json',protocol());q.save(OUT/'CPU_SELFCHECK.json',tiny())
    q.save(OUT/'PREVIOUS_SCOPE_REFERENCES.json',previous_references())
    q.save(OUT/'design_freeze.json',{'protocol_sha256':q.sha(OUT/'protocol.json'),
        'source_sha256':source_hashes(),'cpu_selfcheck_sha256':q.sha(OUT/'CPU_SELFCHECK.json'),
        'previous_references_sha256':q.sha(OUT/'PREVIOUS_SCOPE_REFERENCES.json'),
        'trained':False,'official_test_opened':False})
    print('HEADING_SEMANTIC_LR_PREPARED_NO_FIT',os.getpid(),flush=True)


def check():
    f=q.read(OUT/'design_freeze.json');assert q.read(OUT/'protocol.json')==protocol()
    assert f['protocol_sha256']==q.sha(OUT/'protocol.json') and f['source_sha256']==source_hashes()
    assert f['cpu_selfcheck_sha256']==q.sha(OUT/'CPU_SELFCHECK.json')
    assert f['previous_references_sha256']==q.sha(OUT/'PREVIOUS_SCOPE_REFERENCES.json')
    producer.check()
    return f


def load_features(meta):
    for directory,name in [(LEX,'complete.json'),(SCOPE,'features_complete.json'),(SEMANTIC,'features_complete.json')]:
        done=q.read(directory/name)
        if directory!=LEX:assert done['status']=='complete' and done['no_test'] and not done['trained']
        for n,h in done['files_sha256'].items():assert q.sha(directory/n)==h,(str(directory),n)
    done=q.read(SEMANTIC/'features_complete.json')
    assert (done['rows'],done['columns'],done['dtype'])==(210364,2,'float32')
    assert q.read(SEMANTIC/'feature_names.json')==producer.NAMES
    inf=q.read(SEMANTIC/'inference_complete.json')
    assert inf['status']=='complete' and inf['answers']==37 and inf['pairs']==1161
    assert inf['preparation_freeze_sha256']==q.sha(SEMANTIC/'preparation_freeze.json')
    assert inf['numeric_agreement_sha256']==q.sha(SEMANTIC/'numeric_agreement.json')
    assert q.read(SEMANTIC/'numeric_agreement.json')['passed']
    for rec in inf['records']:assert q.sha(SEMANTIC/rec['path'])==rec['sha256']
    assert q.read(SEMANTIC/'preparation_statistics.json')['scored_claims']==387
    order=q.digest([w['window_id'] for w in meta['windows']])
    for directory in (LEX,SCOPE,SEMANTIC):assert q.read(directory/'geometry.json')['window_order_sha256']==order
    assert q.read(SCOPE/'geometry.json')['old8_features_sha256']==q.sha(LEX/'window_features.npy')
    assert q.read(SEMANTIC/'geometry.json')['old_scope_features_sha256']==q.sha(SCOPE/'window_features.npy')
    x=[np.load(d/'window_features.npy',allow_pickle=False) for d in (LEX,SCOPE,SEMANTIC)]
    assert [a.shape for a in x]==[(210364,8),(210364,3),(210364,2)]
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in x)
    assert not x[2][x[1][:,0]==0].any()
    return x


def run():
    check()
    if not (SEMANTIC/'features_complete.json').exists():
        assert not (OUT/'started.json').exists()
        print('WAIT_HEADING_SEMANTIC_COMPLETE_NO_FIT_NO_GPU',flush=True)
        return
    assert not (OUT/'started.json').exists()
    meta=q.metadata();lexical,scope,semantic=load_features(meta)
    bw,lw,_,y=q.base_weights(meta);assert len(y)==168123 and np.isclose(lw.sum(),168123)
    assert meta['bounds']=={'fit':[0,168123],'calibration':[168123,210364]}
    references=previous_references();assert references==q.read(OUT/'PREVIOUS_SCOPE_REFERENCES.json')
    src=q.read(upstream.OUT/'summary.json');assert q.sha(upstream.OUT/'summary.json')==q.read(upstream.OUT/'complete.json')['summary_sha256']
    q.save(OUT/'started.json',{'pid':os.getpid(),'time':time.time(),'design_freeze_sha256':q.sha(OUT/'design_freeze.json'),
        'feature_complete_sha256':q.sha(SEMANTIC/'features_complete.json'),'GPU_used':False,'official_test_opened':False})
    selected={};all_candidates={};start=time.perf_counter()
    for peer in upstream.PEERS:
        pair=[]
        for alpha in (0.,1.):
            e=next(e for e in src['all_candidates'][peer] if e['tail_weight']==alpha)
            p=upstream.OUT/(e['candidate']+'_scores.npz');assert q.sha(p)==e['scores_sha256']
            with np.load(p) as z:pair.append(z['window_scores'].copy())
        designs=matrices(np.column_stack(pair),lexical,scope,semantic)
        for mode in MODES:
            x=designs[mode];family=peer+'__'+mode
            scaler=StandardScaler().fit(x[:len(y)],sample_weight=bw);scaled=scaler.transform(x)
            sp=OUT/(family+'_scaler.pkl');sp.write_bytes(pickle.dumps(scaler,protocol=5))
            entries=[]
            for c in CS:
                tick=time.perf_counter();model=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=20261010)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(scaled[:len(y)],y,sample_weight=lw)
                score=model.predict_proba(scaled)[:,1];answer=q.answer_scores(meta,score)
                ts={'window':q.choose_threshold([w['label'] for w in meta['windows'][168123:]],score[168123:]),
                    'answer':q.choose_threshold([a['label'] for a in meta['answers'][634:]],answer[634:])}
                metrics=q.metrics(meta,score,ts);name=family+f'__C{c:g}'
                mp=OUT/(name+'.pkl');pp=OUT/(name+'_scores.npz')
                mp.write_bytes(pickle.dumps(model,protocol=5));np.savez_compressed(pp,window_scores=score,answer_scores=answer)
                e={'candidate':name,'peer':peer,'mode':mode,'C':c,'thresholds':ts,'metrics':metrics,
                    'selection_key':list(q.selection_key(ts,c)),'input_width':x.shape[1],
                    'model_sha256':q.sha(mp),'scaler_sha256':q.sha(sp),'scores_sha256':q.sha(pp),
                    'iterations':model.n_iter_.tolist(),'seconds':time.perf_counter()-tick}
                q.save(OUT/(name+'.json'),e);entries.append(e)
                print('HEADING_SEMANTIC_LR_FIT',name,metrics['calibration']['windows']['f1'],metrics['calibration']['answers']['f1'],flush=True)
            all_candidates[family]=entries;selected[family]=max(entries,key=lambda e:e['selection_key'])
    assert sum(map(len,all_candidates.values()))==18
    check()
    q.save(OUT/'summary.json',{'selected':selected,'all_candidates':all_candidates,'previous_scope_references':references,
        'fits_completed':18,'reference_refits':0,'seconds':time.perf_counter()-start,'GPU_used':False,'official_test_opened':False})
    report=['# 标题继承正文的语义来源：匹配LR对照','','两模式共享完全相同的1161对新推理、原两分数/8词面/3scope及任意来源支持度；仅15列候选新增继承来源支持差。',
        '原634fit/159cal、全部793答/210364窗口、权重与两级阈值规则保持。全18候选保留；原13列scope控制只保存为历史参考，不重训。','',
        '| peer | 模式 | C | 窗口F1 | 整答F1 |','|---|---|---:|---:|---:|']
    for e in selected.values():
        m=e['metrics']['calibration'];report.append(f"| {e['peer']} | {e['mode']} | {e['C']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    report+=['','该输入只补冻结标题继承正文的来源支持，不解决步骤编号、时间顺序或句尾括号引用。',
        '额外语义核查模型与生成白盒融合，上游fit分数非交叉拟合。校准反复用于开发，不是独立测试或SOTA。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    q.save(OUT/'complete.json',{'status':'complete','summary_sha256':q.sha(OUT/'summary.json'),
        'fits_completed':18,'reference_refits':0,'GPU_used':False,'official_test_opened':False,'pid':os.getpid()})
    print('HEADING_SEMANTIC_LR_COMPLETE',os.getpid(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','check','run'))
    with threadpool_limits(limits=4):
        stage=parser.parse_args().stage;globals()[stage]()
        if stage=='check':print('HEADING_SEMANTIC_LR_CHECK_PASSED_NO_FIT',flush=True)
