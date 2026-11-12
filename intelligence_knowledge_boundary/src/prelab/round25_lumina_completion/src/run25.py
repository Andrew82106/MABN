"""CPU-only frozen-formula LUMINA evaluation; no new LR/PCA/model generation.

All calibration choices are saved before any outer-fold metrics. Original
R16 validation/test content is never parsed. The extraction protocol stays
immutable; scoring has a separate, predeclared protocol.
"""
from pathlib import Path
from collections import Counter
import argparse
import importlib.util
import pickle
import time

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
_s=importlib.util.spec_from_file_location('r25_score_source24',ROOT.parent/'round24_sequence_persistence/src/run24.py')
r24=importlib.util.module_from_spec(_s);_s.loader.exec_module(r24)
r22,r18,r17,r10=r24.r22,r24.r18,r24.r17,r24.r10
_e=importlib.util.spec_from_file_location('r25_extractor',ROOT/'src/extract25.py')
e25=importlib.util.module_from_spec(_e);_e.loader.exec_module(e25)
read,readl,save,savel,sha=r22.read,r22.readl,r22.save,r22.savel,r22.sha
BASES=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned')
OLD=BASES+('slots_base_smooth',)
FIXED=('lumina_single_mean_lambda05','lumina_joint_body_lambda05')
SECONDARY='lumina_joint_body_lambda_cal'
METHODS=OLD+FIXED+(SECONDARY,)
LAMBDAS=(.25,.5,.75)


def protocol():
    return {'version':'r25-signed-lumina-calibration-v1',
      'scope':'Exploratory same R16 actual-train 301 questions/278 event groups/602 answers; no original validation/test',
      'methods':list(METHODS),'primary_method':FIXED[1],'additional_fixed_method':FIXED[0],
      'fixed_lambda':.5,'secondary_method':SECONDARY,'secondary_lambda_grid':list(LAMBDAS),
      'secondary_selection':'Calibration max min(window F1,answer F1), then window F1, window precision, prefer lambda=.5, then lower lambda',
      'formula':'lambda*IPR-(1-lambda)*MMD, computed per raw response token in unchanged float32 arithmetic',
      'single_mean':'Both original R19 single-source MMD columns are averaged equally per token before fixed formula',
      'joint_body':'Independent original-versus-joint-two-body intervention MMD; informative source titles remain unchanged',
      'window':'Original 4 raw BPE/stride1 geometry including punctuation positions; arithmetic float32 mean of token formula scores, exported as float64; original short windows retained',
      'answer':'Maximum over ALL output-defined candidate windows for each original answer, without gold eligibility filtering',
      'score_transform':'None; signed raw scores retained. Helpers support actual min/max nextafter endpoints. Scores are not calibrated probabilities.',
      'threshold':'Calibration-only risk F1, then precision, then higher actual score cutoff; separate window and answer thresholds',
      'folds':'Existing fixed event assignment: evaluation f, calibration (f+1)%5, remaining3 listed as legacy-fit only; no parameters fitted for new methods',
      'gold':'Unchanged R16 train assistant-reviewed labels and R22 geometry; 598 evaluable answers/151 risk, 9526 evaluable windows/1063 risk',
      'safe_refusals':'Reviewed safe refusal is answer label0; its candidate windows still contribute to answer max but are excluded from localization denominator',
      'unresolved':'Excluded according to original eligibility, not mapped to negative; all feature rows must be present',
      'old_baselines':'Manual frozen-coefficient/formula replay of six R22 baselines; R24 slots_base_smooth recomputed from its frozen stay/temperature; scores/cutoffs/pooled summaries must match exactly',
      'new_lr_fits':0,'new_pca_fits':0,'new_generation':False,'gpu_used_by_runner':False,
      'freeze':'All five calibration tables/lambda selections/thresholds and full score arrays freeze before separate outer entrypoint',
      'bootstrap':{'draws':2000,'seed':20260925,'unit':'event group','fixed_models_and_thresholds':True},
      'human_gold':False,'original_validation_test_used':False,
      'limits':['Development OOF has been repeatedly inspected, not a fresh final test',
        'Pinned LUMINA formula with NF4 Qwen/body-token perturbation and window-max answer aggregation is an adaptation',
        'Secondary calibrated lambda cannot replace the fixed lambda=.5 primary after results are seen']}


def window_means(token_scores,windows):
    result=[]
    for w in windows:
        values=np.asarray(token_scores[w['row_id']],np.float32)
        ix=np.asarray(w['raw_token_indices'],int)
        assert len(ix)==w['actual_width'] and len(ix)>0
        assert ix.min()>=0 and ix.max()<len(values) and np.all(np.diff(ix)==1)
        result.append(float(values[ix].mean()))
    result=np.asarray(result,np.float64)
    assert result.shape==(len(windows),) and np.isfinite(result).all()
    return result


def lambda_key(thresholds,lam):
    w,a=thresholds['window'],thresholds['answer']
    return [min(w['validation_f1'],a['validation_f1']),w['validation_f1'],
            w['validation_precision'],int(lam==.5),-float(lam)]


def manual_lr(obj,x):
    z=x.copy();z-=obj['scaler'].mean_;z/=obj['scaler'].scale_
    return expit((z.astype(np.float32)@obj['model'].coef_.T+obj['model'].intercept_).ravel())


def verify_prior_files():
    files={}
    for directory,completion,freeze in [(r22.ROOT,'complete22.json','fit_freeze22.json'),
                                         (r24.ROOT,'complete.json','calibration_freeze.json')]:
        for manifest in [completion,freeze]:
            path=directory/'results'/manifest;files[str(path.resolve())]=sha(path)
            for name,h in read(path)['files_sha256'].items():
                p=directory/'results'/name;assert sha(p)==h,(p,h);files[str(p.resolve())]=h
    audit=read(r22.ROOT/'results/INDEPENDENT_AUDIT22.json')
    assert audit['status']=='passed' and audit['complete22_sha256']==sha(r22.ROOT/'results/complete22.json')
    files[str((r22.ROOT/'results/INDEPENDENT_AUDIT22.json').resolve())]=sha(r22.ROOT/'results/INDEPENDENT_AUDIT22.json')
    for name,h in read(r22.ROOT/'results/source_snapshot22.json')['files_sha256'].items():
        assert sha(Path(name))==h,name;files[str(Path(name).resolve())]=h
    return files


def geometry():
    # r19.context selects actual split metadata before parsing a source row.
    # Avoid r18.train_metadata, which parses all original input rows first.
    rows,records,_=e25.r19.context()
    meta_items=[]
    for row in rows:
        g,_=records[row['row_id']];assert len(g['items'])==1
        meta_items.append(dict(g['items'][0],**{k:row[k] for k in
            ('row_id','question_id','group_id','split','condition','category')}))
    items,tokens,regions,coverage=r10.cohort(r18.NEW,'train',(rows,meta_items,records))
    assert items==readl(r22.ROOT/'results/answer_index22.jsonl')
    windows=readl(r22.ROOT/'results/candidate_windows22.jsonl')
    assert len(items)==602 and len(windows)==12222
    byitem={r['item_id']:r for r in items};bytoken={r['token_key']:r for r in tokens}
    for w in windows:
        assert w['split']=='train' and len(w['item_ids'])==1
        item=byitem[w['item_ids'][0]];g,_=records[w['row_id']]
        assert w['row_id']==item['row_id'] and w['group_id']==item['group_id']
        assert g['response'][w['start']:w['end']]==w['text']
        tt=[bytoken[w['row_id']+f'__token{i}'] for i in w['raw_token_indices']]
        lexical=[t for t in tt if t['lexical']]
        assert [t['token_key'] for t in lexical]==w['token_keys']
        eligible=item['asserted_eligible'] and item['localization_status']=='resolved'
        assert w['main_eligible']==eligible
        assert w['gold']==(int(any(t['gold']==1 for t in lexical)) if eligible else None)
    assert len({w['window_key'] for w in windows})==len(windows)
    coverage.update(questions=301,candidate_windows=12222,eligible_windows=sum(w['main_eligible'] for w in windows),
                    risk_windows=sum(w['gold']==1 for w in windows))
    assert coverage['item_evaluable']==598 and coverage['risk_answers']==151
    assert coverage['eligible_windows']==9526 and coverage['risk_windows']==1063
    pack={'items':items,'tokens':tokens,'regions':regions,'windows':windows,'coverage':coverage}
    return pack,rows,records


def old_designs():
    path=r22.R21/'results/designs.npz'
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('base','slots','r19_all','harp','delta','ecs','pks')}


def old_scores(fold,d,windows):
    previous=pickle.loads((r22.R21/'results'/f'fold_{fold}_frozen.pkl').read_bytes())
    f=pickle.loads((r22.ROOT/'results'/f'fold_{fold}_frozen22.pkl').read_bytes())
    values={name:manual_lr(previous['models'][name],d[key]) for name,key in
        [('base','base'),('slots_base','slots'),('r19_all','r19_all')]}
    pc=previous['projections']['delta']
    delta=((d['delta'].astype(np.float64)-pc['mean'])@pc['components'].T).astype(np.float32)
    values['base_harp_delta']=manual_lr(previous['models']['base_harp_delta'],np.column_stack((d['base'],d['harp'],delta)))
    values['lookback_tuned']=manual_lr(previous['models']['lookback_tuned']['candidate'],d['base'][:,:784])
    c=previous['models']['redeep_tuned']['candidate']
    pk=d['pks'][:,c['layers']].astype(np.float64).sum(1);ec=d['ecs'][:,c['heads']].astype(np.float64).sum(1)
    transformed=(np.column_stack((pk,ec))-c['minimum'])/c['scale']
    values['redeep_tuned']=transformed[:,0]-c['beta']*transformed[:,1]
    with np.load(r22.ROOT/'results'/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:
        for name in BASES:
            assert np.array_equal(values[name],z[name]),('Old score mismatch',fold,name,float(np.abs(values[name]-z[name]).max()))
            assert f['thresholds'][name]==previous['thresholds'][name]
    smooth=read(r24.ROOT/'results'/f'fold_{fold}_calibration.json')
    assert smooth['groups']=={k:f[k] for k in ('fit_groups','calibration_groups','evaluation_groups')}
    selected=smooth['choices']['slots_base_smooth']
    values['slots_base_smooth']=r24.whole(values['slots_base'],r24.geometry(windows),selected['stay'],selected['temperature'])
    with np.load(r24.ROOT/'results'/f'fold_{fold}_scores.npz',allow_pickle=False) as z:
        assert np.array_equal(values['slots_base_smooth'],z['slots_base_smooth'])
    thresholds={name:f['thresholds'][name] for name in BASES}
    thresholds['slots_base_smooth']=smooth['thresholds']['slots_base_smooth']
    return f,values,thresholds


def source_files():
    files=verify_prior_files()
    for p in [Path(__file__),ROOT/'scoring_protocol.json',ROOT/'src/extract25.py',ROOT/'protocol.json',
              ROOT/'data/extraction_signature.json',ROOT/'data/joint_plans.jsonl',
              ROOT/'data/preparation_freeze.json',ROOT/'data/cpu_selfcheck.json',
              r18.NEW/'data/annotations_train.jsonl',r18.NEW/'data/question_label_policy.json',
              r18.NEW/'data/safe_refusals_train.json',r18.NEW/'data/windows_k4_train.jsonl',
              e25.R19/'data/feature_manifest.json',e25.R19/'data/fold_assignment.json',
              Path(r24.__file__),Path(r22.__file__),Path(r18.__file__),Path(r17.__file__),Path(r10.__file__),
              Path(r10.metric.__file__)]:
        files[str(p.resolve())]=sha(p)
    return files


def feature_values(pack,rows,records):
    path=ROOT/'data/feature_manifest.json';fm=read(path)
    assert fm['complete'] and fm['completed_count']==602, 'R25 features incomplete; do not score a partial cohort'
    assert set(fm['records'])=={r['row_id'] for r in rows}
    sig=read(ROOT/'data/extraction_signature.json')
    assert sig==fm['extraction_signature'] and e25.base.model7.digest(sig)==fm['extraction_signature_sha256']
    assert sig['protocol_sha256']==sha(ROOT/'protocol.json')
    assert sig['joint_plans_sha256']==sha(ROOT/'data/joint_plans.jsonl')
    for f,h in sig['code_sha256'].items():assert sha(Path(f))==h
    plans={p['row_id']:p for p in readl(ROOT/'data/joint_plans.jsonl')}
    old=read(e25.R19/'data/feature_manifest.json')
    sources={str(path.resolve()):sha(path)}
    raw_single={};raw_joint={lam:{} for lam in LAMBDAS}
    for row in rows:
        rid=row['row_id'];g,h=records[rid];entry=fm['records'][rid]
        npz,side=ROOT/entry['npz'],ROOT/entry['json']
        assert entry['source_generation_sha256']==h and entry['split']=='train'
        assert sha(npz)==entry['npz_sha256'] and sha(side)==entry['json_sha256']
        oldentry=old['records'][rid];oldpath=e25.R19/oldentry['npz'];oldside=e25.R19/oldentry['json']
        assert sha(oldpath)==oldentry['npz_sha256'] and sha(oldside)==oldentry['json_sha256']
        with np.load(oldpath,allow_pickle=False) as z:single=z['token_lumina_mmd'].copy()
        value=e25.cached(row,g,h,plans[rid],single,oldentry,sig);assert value is not None
        arrays,meta=value
        assert meta['labels_or_detector_scores_read'] is False and meta['response_regenerated'] is False
        raw_single[rid]=arrays['token_lumina_single_mean']
        for lam in LAMBDAS:
            raw_joint[lam][rid]=e25.mix_scores(arrays['token_ipr'],arrays['token_mmd_joint_body'],lam)
        assert np.array_equal(raw_joint[.5][rid],arrays['token_lumina_joint_body'])
        for p in [npz,side,oldpath,oldside]:sources[str(p.resolve())]=sha(p)
    candidates={lam:window_means(scores,pack['windows']) for lam,scores in raw_joint.items()}
    fixed={FIXED[0]:window_means(raw_single,pack['windows']),FIXED[1]:candidates[.5]}
    return fixed,candidates,sources


def verify_snapshot(snapshot):
    for filename,h in snapshot['files_sha256'].items():assert sha(Path(filename))==h,filename


def selfcheck():
    # Signed scores and actual-score endpoints; no synthetic model fitting.
    t=r10.threshold_search([0,1,0],np.array([-4.,-2.,-3.]))
    assert t['threshold']==-2 and t['validation_f1']==1
    p={'items':[{'item_id':'a','main_eligible':True},{'item_id':'b','main_eligible':True}],
       'windows':[{'item_ids':['a'],'main_eligible':False},{'item_ids':['a'],'main_eligible':True},{'item_ids':['b'],'main_eligible':False}]}
    assert np.array_equal(r17.answer_scores(p,[-.1,-.5,-.7]),[-.1,-.7])
    w=[{'row_id':'a','raw_token_indices':[0,1,2,3],'actual_width':4},
       {'row_id':'a','raw_token_indices':[1,2,3,4],'actual_width':4},
       {'row_id':'b','raw_token_indices':[0,1],'actual_width':2}]
    got=window_means({'a':np.array([-4.,0.,2.,10.,8.],np.float32),'b':np.array([-2.,-6.],np.float32)},w)
    assert np.array_equal(got,[2.,5.,-4.])
    dummy={'window':{'validation_f1':.7,'validation_precision':.8},'answer':{'validation_f1':.6}}
    assert lambda_key(dummy,.5)>lambda_key(dummy,.25)>lambda_key(dummy,.75)
    return {'status':'passed','signed_score_threshold':True,'all_window_answer_max_including_ineligible':True,
            'raw_four_bpe_mean_and_short_window':True,'secondary_lambda_tie_rule':True,
            'new_models_fitted':0,'gpu_used':False,'run25_sha256':sha(Path(__file__))}


def baseline_check():
    cfg=read(ROOT/'scoring_protocol.json');assert cfg==protocol()
    files=source_files();pack,_,_=geometry();d=old_designs();folds=[]
    for fold in range(5):
        f,values,thresholds=old_scores(fold,d,pack['windows'])
        cal,cx=r18.subset(pack,f['calibration_groups'])
        for method in OLD:assert r22.thresholds(cal,values[method][cx])==thresholds[method]
        folds.append({'fold':fold,'all_seven_old_scores_exact':True,'all_old_calibration_thresholds_exact':True})
    result={'status':'passed','coverage':pack['coverage'],'folds':folds,'source_files_sha256':files,
            'run25_sha256':sha(Path(__file__)),'new_models_fitted':0,'gpu_used':False,'new_feature_scores_read':False}
    save(ROOT/'results/baseline_replay_check25.json',result)
    print('R25_SEVEN_BASELINES_EXACT',flush=True)


def calibrate():
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'calibration_started25.json').exists(), 'Do not silently rerun calibration'
    cfg=read(ROOT/'scoring_protocol.json');assert cfg==protocol();selfcheck()
    files=source_files();pack,rows,records=geometry();fixed,candidates,feature_files=feature_values(pack,rows,records)
    files.update(feature_files);snapshot={'files_sha256':files};verify_snapshot(snapshot)
    save(out/'source_snapshot25.json',snapshot)
    save(out/'calibration_started25.json',{'utc':r10.utc(),'source_snapshot_sha256':sha(out/'source_snapshot25.json'),'new_models_fitted':0})
    savel(out/'candidate_windows25.jsonl',pack['windows']);savel(out/'answer_index25.jsonl',pack['items'])
    np.savez_compressed(out/'formula_window_scores25.npz',single_lambda05=fixed[FIXED[0]],
                        joint_lambda025=candidates[.25],joint_lambda05=candidates[.5],joint_lambda075=candidates[.75])
    d=old_designs();assignment=read(e25.R19/'data/fold_assignment.json')['groups'];names=[];started=time.perf_counter()
    assert set(assignment)=={i['group_id'] for i in pack['items']}
    for fold in range(5):
        previous,values,thresholds=old_scores(fold,d,pack['windows'])
        eg=sorted(g for g,v in assignment.items() if v==fold);cg=sorted(g for g,v in assignment.items() if v==(fold+1)%5)
        fg=sorted(set(assignment)-set(eg)-set(cg))
        assert previous['evaluation_groups']==eg and previous['calibration_groups']==cg and previous['fit_groups']==fg
        assert not(set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        cal,cx=r18.subset(pack,cg)
        for name in OLD:assert r22.thresholds(cal,values[name][cx])==thresholds[name]
        for name in FIXED:
            values[name]=fixed[name];thresholds[name]=r22.thresholds(cal,values[name][cx])
        table=[]
        for lam in cfg['secondary_lambda_grid']:
            ts=r22.thresholds(cal,candidates[lam][cx])
            table.append({'lambda':lam,'thresholds':ts,'selection_key':lambda_key(ts,lam)})
        chosen=max(range(len(table)),key=lambda j:table[j]['selection_key']);selection=table[chosen]
        values[SECONDARY]=candidates[selection['lambda']];thresholds[SECONDARY]=selection['thresholds']
        assert set(values)==set(METHODS) and all(v.shape==(12222,) and np.isfinite(v).all() for v in values.values())
        frozen={'groups':{'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg},'thresholds':thresholds,
            'fixed_lambda':.5,'secondary_candidates':table,'secondary_selected_index':chosen,'secondary_selected_lambda':selection['lambda'],
            'new_models_fitted':0,'score_transform':'none','all_thresholds_use_calibration_only':True}
        save(out/f'fold_{fold}_calibration25.json',frozen)
        np.savez_compressed(out/f'fold_{fold}_scores25.npz',**values)
        names += [f'fold_{fold}_calibration25.json',f'fold_{fold}_scores25.npz']
        print('R25_CALIBRATION_FROZEN',fold,'SECONDARY_LAMBDA',selection['lambda'],flush=True)
    names += ['source_snapshot25.json','candidate_windows25.jsonl','answer_index25.jsonl','formula_window_scores25.npz']
    verify_snapshot(snapshot)
    save(out/'calibration_freeze25.json',{'utc':r10.utc(),'seconds':time.perf_counter()-started,
        'files_sha256':{n:sha(out/n) for n in names},'new_models_fitted':0,'all_calibration_choices_frozen_before_outer':True})


def test():
    out=ROOT/'results';assert not (out/'test_started25.json').exists(), 'Outer results are immutable'
    frozen=read(out/'calibration_freeze25.json')
    for name,h in frozen['files_sha256'].items():assert sha(out/name)==h,name
    snapshot=read(out/'source_snapshot25.json');verify_snapshot(snapshot)
    assert read(ROOT/'scoring_protocol.json')==protocol()
    pack,_,_=geometry()
    assert pack['windows']==readl(out/'candidate_windows25.jsonl') and pack['items']==readl(out/'answer_index25.jsonl')
    save(out/'test_started25.json',{'utc':r10.utc(),'calibration_freeze_sha256':sha(out/'calibration_freeze25.json'),'retuning_allowed':False})
    allw=[];alla=[];details={};wseen=[];aseen=[]
    for fold in range(5):
        f=read(out/f'fold_{fold}_calibration25.json');ev,ex=r18.subset(pack,f['groups']['evaluation_groups'])
        ca,cx=r18.subset(pack,f['groups']['calibration_groups'])
        with np.load(out/f'fold_{fold}_scores25.npz',allow_pickle=False) as z:values={m:z[m].copy() for m in METHODS}
        wr=None;ar=None;detail={}
        for method in METHODS:
            w,a=r18.scored_records(ev,values[method][ex],f['thresholds'][method],method)
            if wr is None:
                wr=[dict(r,scores={},predictions={},fold=fold) for r in w]
                ar=[dict(r,scores={},predictions={},fold=fold) for r in a]
            for target,row in zip(wr,w):target['scores'].update(row['scores']);target['predictions'].update(row['predictions'])
            for target,row in zip(ar,a):target['scores'].update(row['scores']);target['predictions'].update(row['predictions'])
            detail[method]={'calibration':r17.metrics(ca,values[method][cx],f['thresholds'][method]),
                            'evaluation':r17.metrics(ev,values[method][ex],f['thresholds'][method])}
        allw.extend(wr);alla.extend(ar);details[str(fold)]=detail
        wseen.extend(r['window_key'] for r in wr);aseen.extend(r['item_id'] for r in ar)
    assert Counter(wseen)==Counter(w['window_key'] for w in pack['windows'])
    assert Counter(aseen)==Counter(a['item_id'] for a in pack['items'])
    prior_methods=r18.METHODS
    try:
        r18.METHODS=METHODS
        result=r18.pooled(allw,alla,pack)
        previous=read(r22.ROOT/'results/summary22.json')['methods']
        for name in BASES:assert result[name]==previous[name],name
        assert result['slots_base_smooth']==read(r24.ROOT/'results/summary.json')['methods']['slots_base_smooth']
        contrasts={'joint_fixed_vs_single_fixed':[FIXED[1],FIXED[0]],
                   'joint_fixed_vs_slots_smooth':[FIXED[1],'slots_base_smooth'],
                   'joint_calibrated_vs_joint_fixed':[SECONDARY,FIXED[1]]}
        boot={u:r18.bootstrap(rr,protocol()['bootstrap'],contrasts) for u,rr in [('windows',allw),('answers',alla)]}
    finally:r18.METHODS=prior_methods
    verify_snapshot(snapshot)
    summary={'scope':protocol()['scope'],'coverage':pack['coverage'],'primary_method':FIXED[1],
        'methods':result,'folds':details,'paired_bootstrap':boot,'all_seven_old_baselines_exact':True,
        'new_lr_fits':0,'new_pca_fits':0,'signed_raw_formula_scores':True,'score_transform':'none',
        'original_validation_test_used':False,'human_gold':False,'gold_or_window_geometry_changed':False,
        'all_calibration_choices_frozen_before_outer':True,'limits':protocol()['limits']}
    savel(out/'window_scores_oof25.jsonl',allw);savel(out/'answer_scores_oof25.jsonl',alla);save(out/'summary25.json',summary)
    names=['calibration_freeze25.json','summary25.json','window_scores_oof25.jsonl','answer_scores_oof25.jsonl']
    save(out/'complete25.json',{'utc':r10.utc(),'files_sha256':{n:sha(out/n) for n in names},'new_models_fitted':0,'original_validation_test_used':False})
    for method,x in result.items():print(method,round(x['windows']['f1'],6),round(x['answers']['f1'],6),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','selfcheck','baseline-check','status','calibrate','fit','test']);a=p.parse_args()
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):
        if a.stage=='initialize':
            e25.immutable_json(ROOT/'scoring_protocol.json',protocol())
            save(ROOT/'results/scoring_cpu_selfcheck25.json',selfcheck());print('R25_SCORING_PROTOCOL_FROZEN_NO_FIT')
        elif a.stage=='selfcheck':print(selfcheck())
        elif a.stage=='baseline-check':baseline_check()
        elif a.stage=='status':
            fm=read(ROOT/'data/feature_manifest.json');print({'complete':fm['complete'],'completed_count':fm['completed_count'],'expected':602})
        elif a.stage in ('calibrate','fit'):calibrate()
        else:test()
