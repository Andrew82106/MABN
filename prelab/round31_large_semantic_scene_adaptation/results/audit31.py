"""No fitting/GPU: independent R31 coefficient, calibration and count audit."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import pickle
import time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
DATA = ROOT / 'data'
PRELAB = ROOT.parent
OLD = PRELAB / 'round29_full_hidden_scene_adaptation/results'
SCENE = PRELAB / 'benchmark_ragtruth_qa/control_scene_transfer_v1'
R26 = PRELAB / 'round26_global_local_fusion/results'
BASES = ('lookback_tuned', 'local_slots_fusion_smooth_global')
FUSED = ('lookback_plus_semantic', 'r26_plus_semantic')
METHODS = BASES + ('semantic_probe',) + FUSED
ALPHA = (0., .2, .4, .6, .8, 1.)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def check_hashes(root, values):
    for name, digest in values.items(): assert sha(root/name) == digest, str(root/name)


def count(y,p):
    y,p=np.asarray(y,int),np.asarray(p,bool)
    tp=int(np.count_nonzero((y==1)&p));fp=int(np.count_nonzero((y==0)&p))
    fn=int(np.count_nonzero((y==1)&~p));tn=int(np.count_nonzero((y==0)&~p))
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def cutoff(y,s):
    # Independent ascending-score grouped cumulative counts, including endpoints.
    y,s=np.asarray(y,int),np.asarray(s,float)
    assert len(y)==len(s) and np.isfinite(s).all() and set(y)=={0,1}
    order=np.argsort(s,kind='stable');sv=s[order];yv=y[order]
    unique,start=np.unique(sv,return_index=True)
    before=np.r_[0,np.cumsum(yv)]
    tp=y.sum()-before[start];pred=len(y)-start
    f1=2*tp/(pred+y.sum());precision=tp/pred
    candidates=[(float(f),float(p),float(t)) for f,p,t in zip(f1,precision,unique)]
    candidates += [(0.,0.,math.nextafter(float(sv[-1]),math.inf)),
                   (float(f1[0]),float(precision[0]),math.nextafter(float(sv[0]),-math.inf))]
    f,p,t=max(candidates)
    return {'threshold':t,'validation_f1':f,'validation_precision':p,
            'tokens':len(y),'scorable_tokens':len(y),'risk_tokens':int(y.sum())}


def weights(rr):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for i,r in enumerate(rr): groups[r['group_id']][r['condition']][r['item_ids'][0]].append(i)
    b=np.zeros(len(rr),float)
    for conditions in groups.values():
        for answers in conditions.values():
            for ix in answers.values(): b[ix]=1/(len(conditions)*len(answers)*len(ix))
    b/=b.mean();y=np.asarray([r['gold'] for r in rr],int)
    cm=np.bincount(y,weights=b,minlength=2);f=b.sum()/(2*cm);w=b*f[y]
    for conditions in groups.values():
        ix=[i for answers in conditions.values() for indices in answers.values() for i in indices]
        w[ix]*=(len(rr)/len(groups))/w[ix].sum()
    w*=3854/w.sum()
    return b,w,f


def replay_model(frozen,x,windows,groups,width):
    model=frozen['model'];ix=np.asarray([i for i,r in enumerate(windows)
        if r['main_eligible'] and r['group_id'] in groups['fit_groups']])
    assert frozen['groups']==groups and frozen['PCA'] is None
    assert model['width']==width and model['C']==.01 and model['loss_mass']==3854
    assert np.array_equal(model['fit_ix'],ix)
    rr=[windows[i] for i in ix];y=np.asarray([r['gold'] for r in rr],int)
    assert np.array_equal(y,model['fit_y'])
    assert model['fit_keys']==[r['window_key'] for r in rr]
    assert model['fit_groups']==sorted({r['group_id'] for r in rr})
    b,w,f=weights(rr)
    for actual,expected in ((model['base_weights'],b),(model['loss_weights'],w),(model['class_factors'],f)):
        assert np.allclose(actual,expected,rtol=1e-12,atol=1e-12)
    # StandardScaler validates sample_weight in X.dtype (float32 here).
    # Reproduce that documented implementation conversion, not ideal float64 b.
    scaler_weights=b.astype(x.dtype)
    raw=x[ix].astype(np.float64);mean=np.average(raw,axis=0,weights=scaler_weights)
    variance=np.average((raw-mean)**2,axis=0,weights=scaler_weights)
    scaler=model['scaler']
    assert np.allclose(scaler.mean_,mean,rtol=1e-10,atol=1e-10)
    assert np.allclose(scaler.var_,variance,rtol=1e-10,atol=1e-10)
    assert abs(float(scaler.n_samples_seen_)-scaler_weights.astype(np.float64).sum()) < 1e-7
    lr=model['model']
    assert lr.C==.01 and lr.solver=='liblinear' and lr.random_state==20261008 and lr.max_iter==2000
    assert int(lr.n_iter_.max())<2000 and np.array_equal(lr.classes_,[0,1])
    # sklearn's float32 in-place subtract/divide; then explicit coefficient math.
    transformed=x.copy();transformed-=scaler.mean_;transformed/=scaler.scale_
    values=expit((transformed @ lr.coef_.T + lr.intercept_).ravel())
    return values, {'fit_windows':len(ix),'width':width,'base_mass':float(b.sum()),
        'loss_mass':float(w.sum()),'weighted_scaler_and_group_class_weights_verified':True}


def geometry():
    windows,answers,folds=lines(SCENE/'windows.jsonl'),lines(SCENE/'answers.jsonl'),read(SCENE/'folds.json')
    assert len(windows)==12222 and len(answers)==602 and len(folds)==5
    assert all(w['split']=='train' for w in windows)
    assert sum(w['main_eligible'] for w in windows)==9526
    assert sum(a['main_eligible'] for a in answers)==598
    assert sum(a['reviewed_safe_refusal'] for a in answers)==117
    assert all(a['gold']==0 for a in answers if a['reviewed_safe_refusal'])
    assert all(w['gold'] is None for w in windows if not w['main_eligible'])
    assert all(a['gold'] is None for a in answers if not a['main_eligible'])
    by=defaultdict(list)
    for i,w in enumerate(windows): by[w['item_ids'][0]].append(i)
    assert len(by)==602 and all(by[a['item_id']] for a in answers)
    groups={a['group_id'] for a in answers};assert len(groups)==278
    qg=defaultdict(set)
    for a in answers:qg[a['question_id']].add(a['group_id'])
    assert len(qg)==301 and all(len(g)==1 for g in qg.values())
    seen=[]
    for fold,g in enumerate(folds):
        sets=[set(g[k]) for k in ('fit_groups','calibration_groups','evaluation_groups')]
        assert set.union(*sets)==groups and all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        assert g==read(OLD/f'fold_{fold}_calibration.json')['groups']==read(R26/f'fold_{fold}_calibration.json')['groups']
        seen+=g['evaluation_groups']
    assert len(seen)==len(set(seen))==278
    return windows,answers,folds,by


def answer_max(values,answers,by):
    return np.asarray([np.max(values[by[a['item_id']]]) for a in answers])


def thresholds(values,groups,windows,answers,by):
    wi=[i for i,w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups]
    ai=[i for i,a in enumerate(answers) if a['main_eligible'] and a['group_id'] in groups]
    av=answer_max(values,answers,by)
    return {'window':cutoff([windows[i]['gold'] for i in wi],values[wi]),
            'answer':cutoff([answers[i]['gold'] for i in ai],av[ai])}


def selection(ts,alpha):
    return [min(ts['window']['validation_f1'],ts['answer']['validation_f1']),
            ts['window']['validation_f1'],ts['window']['validation_precision'],-alpha]


def prepare():
    check_hashes(ROOT,read(OUT/'DESIGN_FREEZE.json')['files_sha256'])
    p=read(ROOT/'protocol.json')
    assert p['human_gold'] is False and p['probe']['width']==1025 and p['probe']['LR_fits']==5
    assert p['probe']['C']==.01 and p['fusion_alpha']==list(ALPHA)
    assert 'QA-only versus auxiliary->QA' in p['limits']
    for n in ('complete.json','fit_complete.json'):check_hashes(OLD,read(OLD/n)['files_sha256'])
    assert read(OLD/'AUDIT.json')['complete_sha256']==sha(OLD/'complete.json')
    windows,answers,folds,by=geometry()
    # Validate this auditor on one saved R29 fold; no refitting or new scores.
    with np.load(OLD/'designs.npz') as d:x=np.column_stack((d['hidden'],d['logit'])).astype(np.float32)
    saved=pickle.loads((OLD/'fold_0_frozen.pkl').read_bytes())
    values,detail=replay_model(saved,x,windows,folds[0],769)
    with np.load(OLD/'fold_0_scores.npz') as z:diff=float(np.max(np.abs(values-z['semantic_probe'])))
    assert diff <= 1e-12
    assert thresholds(values,folds[0]['calibration_groups'],windows,answers,by)==saved['thresholds']['semantic_probe']
    assert cutoff([0,1,1,0],[.1,.2,.2,.3])['threshold']==.2
    assert count([1,1,0,0],[1,0,1,0])['f1']==.5
    save(OUT/'AUDIT_PREPARATION.json',{'passed':True,'R31_new_artifacts_read':False,
        'old_R29_fold0_manual_replay_maxdiff':diff,'old_R29_fold0_thresholds_exact':True,
        'old_R29_weight_scaler_check':detail,'groups_and_original_denominators_verified':True,
        'source_difference_disclosure_verified':True,'new_model_fits':0,'GPU_used':False,
        'source_sha256':sha(Path(__file__)),'R31_design_freeze_sha256':sha(OUT/'DESIGN_FREEZE.json'),
        'R29_complete_sha256':sha(OLD/'complete.json')})
    print('R31_AUDITOR_CPU_PREPARED_NO_NEW_RESULTS',flush=True)


def run():
    required=[DATA/'complete.json',OUT/'fit_complete.json',OUT/'complete.json']
    if not all(p.exists() for p in required):
        print('WAIT_R31_CACHE_FIT_EVALUATION_COMPLETE_NO_MODEL_LOAD',flush=True);return
    start=time.perf_counter()
    assert read(OUT/'AUDIT_PREPARATION.json')['source_sha256']==sha(Path(__file__))
    check_hashes(ROOT,read(OUT/'DESIGN_FREEZE.json')['files_sha256'])
    check_hashes(DATA,read(DATA/'complete.json')['files_sha256'])
    prep=read(DATA/'preparation_complete.json')
    for p,digest in prep['source_sha256'].items():assert sha(p)==digest
    binding=read(DATA/'selected_binding.json');checkpoint=Path(binding['checkpoint'])
    upstream=read(checkpoint.parent/'complete.json')
    assert [e['epoch'] for e in upstream['all_epochs']]==list(range(7))
    assert upstream['selected']==max(upstream['all_epochs'][1:],key=lambda e:e['selection_key'])
    assert binding['epoch']==upstream['selected']['epoch'] and not binding['selected_by_R16']
    assert sha(checkpoint)==binding['checkpoint_sha256']==upstream['selected']['artifacts_sha256']['.pt']
    assert sha(checkpoint.parent/'complete.json')==binding['complete_sha256']
    saved_probability=checkpoint.parent/f"epoch_{binding['epoch']:02d}_token_predictions.npz"
    assert sha(saved_probability)==binding['selected_probability_sha256']==upstream['selected']['artifacts_sha256']['_token_predictions.npz']
    assert read(DATA/'QA_ANCHOR.json')['max_abs']<=2e-6
    for r in read(DATA/'SCENE_ANCHORS.json'):
        for key in ('direct_logit','repeat_logit','repeat_hidden'):assert r[key]['max_abs']<=2e-6
    fit=read(OUT/'fit_complete.json')
    assert fit['status']=='all5fold_fit_calibration_frozen_before_outer' and fit['LR_fits']==5
    check_hashes(OUT,fit['files_sha256']);check_hashes(OUT,read(OUT/'complete.json')['files_sha256'])
    for p,digest in fit['source_sha256'].items():assert sha(p)==digest
    prepared=read(OUT/'preparation_complete.json');assert prepared['source_sha256']==fit['source_sha256']
    assert sha(OUT/'designs.npz')==prepared['designs_sha256']
    assert sha(OUT/'R29_CONTROL.json')==prepared['control_sha256']
    control=read(OUT/'R29_CONTROL.json')
    assert control['complete_sha256']==sha(OLD/'complete.json') and control['models_refitted']==0
    check_hashes(OLD,read(OLD/'complete.json')['files_sha256'])
    check_hashes(OLD,read(OLD/'fit_complete.json')['files_sha256'])
    windows,answers,folds,by=geometry()
    with np.load(OUT/'designs.npz') as d:hwin,zwin=d['hidden'],d['logit']
    assert hwin.shape==(12222,1024) and zwin.shape==(12222,)
    h=np.load(DATA/'hidden.npy',mmap_mode='r');z=np.load(DATA/'logit.npy',mmap_mode='r')
    bounds=np.load(DATA/'bounds.npy');rows=lines(SCENE/'inputs.jsonl')
    assert h.shape==(14968,1024) and z.shape==(14968,) and len(rows)==len(bounds)==602
    byrow={r['response_id']:b for r,b in zip(rows,bounds)}
    for i,w in enumerate(windows):
        lo,hi=byrow[w['row_id']]
        assert np.array_equal(hwin[i],h[lo+np.asarray(w['raw_token_indices'])].mean(axis=0,dtype=np.float64).astype(np.float32))
        assert zwin[i]==z[lo+np.asarray(w['lexical_token_indices'])].max()
    x=np.column_stack((hwin,zwin)).astype(np.float32)
    ws={m:np.full(len(windows),np.nan) for m in METHODS};aps={m:np.full(len(answers),np.nan) for m in METHODS}
    wp={m:np.zeros(len(windows),bool) for m in METHODS};ap={m:np.zeros(len(answers),bool) for m in METHODS}
    wc=np.zeros(len(windows),int);ac=np.zeros(len(answers),int);checks=[]
    for f,g in enumerate(folds):
        frozen=pickle.loads((OUT/f'fold_{f}_frozen.pkl').read_bytes())
        assert frozen['R29_control_model_sha256']==sha(OLD/f'fold_{f}_frozen.pkl')
        values,detail=replay_model(frozen,x,windows,g,1025)
        prior=pickle.loads((OLD/f'fold_{f}_frozen.pkl').read_bytes())
        for key in ('fit_ix','fit_y','base_weights','loss_weights','class_factors'):
            assert np.array_equal(frozen['model'][key],prior['model'][key])
        scores={'semantic_probe':values};expected_ts={'semantic_probe':thresholds(values,g['calibration_groups'],windows,answers,by)}
        calibration=read(OUT/f'fold_{f}_calibration.json')
        assert calibration['groups']==g
        with np.load(R26/f'fold_{f}_scores.npz') as old:
            for name in BASES:scores[name]=old[name].copy()
        for name in BASES:
            expected_ts[name]=thresholds(scores[name],g['calibration_groups'],windows,answers,by)
        with np.load(OUT/f'fold_{f}_scores.npz') as saved:
            for base,fused in zip(BASES,FUSED):
                table=[];candidates=[]
                for j,alpha in enumerate(ALPHA):
                    if alpha==0:v=scores[base].copy()
                    else:
                        a,b=np.clip(scores[base],1e-12,1-1e-12),np.clip(values,1e-12,1-1e-12)
                        v=expit(np.log(a)-np.log1p(-a)+alpha*(np.log(b)-np.log1p(-b)))
                    assert np.allclose(v,saved[fused+'_all_alpha'][j],rtol=0,atol=1e-12)
                    ts=thresholds(v,g['calibration_groups'],windows,answers,by)
                    table.append({'alpha':alpha,'thresholds':ts,'key':selection(ts,alpha)});candidates.append(v)
                assert table==frozen['alpha_tables'][fused]==calibration['alpha_tables'][fused]
                best=max(range(6),key=lambda j:table[j]['key'])
                assert frozen['selected_alpha'][fused]==calibration['selected_alpha'][fused]==table[best]
                scores[fused]=candidates[best];expected_ts[fused]=table[best]['thresholds']
            assert expected_ts==frozen['thresholds']==calibration['thresholds']
            wi=np.asarray([i for i,w in enumerate(windows) if w['group_id'] in g['evaluation_groups']])
            ai=np.asarray([i for i,a in enumerate(answers) if a['group_id'] in g['evaluation_groups']])
            wc[wi]+=1;ac[ai]+=1
            maxdiff=0.
            for name,v in scores.items():
                maxdiff=max(maxdiff,float(np.max(np.abs(v-saved[name]))));assert maxdiff<=1e-12
                av=answer_max(v,answers,by);ws[name][wi]=v[wi];aps[name][ai]=av[ai]
                wp[name][wi]=v[wi]>=expected_ts[name]['window']['threshold']
                ap[name][ai]=av[ai]>=expected_ts[name]['answer']['threshold']
        checks.append({'fold':f,**detail,'manual_probability_maxdiff':maxdiff,'all12_calibration_candidates_verified':True})
    assert (wc==1).all() and (ac==1).all()
    wr,ar=lines(OUT/'window_scores_oof.jsonl'),lines(OUT/'answer_scores_oof.jsonl')
    assert [r['window_key'] for r in wr]==[r['window_key'] for r in windows]
    assert [r['item_id'] for r in ar]==[r['item_id'] for r in answers]
    summary=read(OUT/'summary.json');assert summary['human_gold'] is False
    counts={}
    for name in METHODS:
        for records,s,p in ((wr,ws[name],wp[name]),(ar,aps[name],ap[name])):
            assert np.allclose(s,[r['scores'][name] for r in records],rtol=0,atol=1e-12)
            assert np.array_equal(p,[r['predictions'][name] for r in records])
        wm=np.asarray([w['main_eligible'] for w in windows]);am=np.asarray([a['main_eligible'] for a in answers])
        counts[name]={'windows':count([w['gold'] for w in windows if w['main_eligible']],wp[name][wm]),
            'answers':count([a['gold'] for a in answers if a['main_eligible']],ap[name][am]),
            'safe_refusal_false_positives':sum(bool(ap[name][i]) for i,a in enumerate(answers) if a['reviewed_safe_refusal'])}
        assert counts[name]==summary['methods'][name]
    save(OUT/'INDEPENDENT_AUDIT31.json',{'passed':True,'counts':counts,'folds':checks,
        'all12222_hidden_means_and_lexical_logit_maxima_exact':True,
        'original_R29_controls_not_refitted_and_unchanged':True,'all60_alpha_candidates_verified':True,
        'all_outer_windows_answers_assigned_once':True,'new_model_fits':0,'GPU_used':False,'human_gold':False,
        'original_validation_test_opened':False,'seconds':time.perf_counter()-start,
        'source_differences':'Backbone, width and prior QA-only versus auxiliary->QA supervision differ together.',
        'complete_sha256':sha(OUT/'complete.json'),'audit_code_sha256':sha(Path(__file__))})
    print('R31_INDEPENDENT_AUDIT_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));stage=p.parse_args().stage
    with threadpool_limits(limits=4):
        try:{'prepare':prepare,'run':run}[stage]()
        except BaseException as exc:
            save(OUT/f'AUDIT_FAILURE_{stage}_{time.time_ns()}.json',{'error':repr(exc),'frozen_artifacts_changed':False})
            raise
