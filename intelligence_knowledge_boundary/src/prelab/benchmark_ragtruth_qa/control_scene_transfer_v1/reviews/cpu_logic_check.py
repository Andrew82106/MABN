"""Bounded CPU tests of control-scene transfer; no GPU, fitting or heldouts.

Real inference is only called while both upstream completed models are absent,
with GPU/model loaders patched to fail. Score runs only on a temporary synthetic
fixture, separately counted without production metric/threshold functions.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np

OUT=Path(__file__).resolve().parent
QA=OUT.parents[1]
sys.path.insert(0,str(QA/'src'))
import run_control_scene_transfer as runner
import torch

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def savel(p,xx):Path(p).write_text(''.join(json.dumps(x)+'\n' for x in xx),encoding='utf-8')

def counts(y,p):
    pairs=list(zip(y,p));assert len(pairs)==len(y)==len(p)
    tp=sum(int(a==1 and b) for a,b in pairs);fp=sum(int(a==0 and b) for a,b in pairs)
    fn=sum(int(a==1 and not b) for a,b in pairs);tn=sum(int(a==0 and not b) for a,b in pairs)
    assert tp+fp+fn+tn==len(y)
    return {'n':len(y),'positive':sum(y),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def threshold(y,s):
    candidates=sorted(set(s))+[float(np.nextafter(max(s),np.inf))]
    best=max(candidates,key=lambda t:(counts(y,[x>=t for x in s])['f1'],counts(y,[x>=t for x in s])['precision'],t))
    m=counts(y,[x>=best for x in s])
    return {'threshold':best,'f1':m['f1'],'precision':m['precision'],'rows':len(y),'positive':sum(y)}

def missing_upstream_check():
    report={}
    for method,(directory,template) in runner.SOURCES.items():
        path=directory/'complete.json'
        if path.exists():
            c=read(path);e=c['selected'];ck=directory/template.format(epoch=e['epoch'])
            assert ck.exists() and sha(ck)==e['artifacts_sha256']['.pt']
            state=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)
            assert 'model_state_dict' in state and 'classifier.weight' in state['model_state_dict']
            assert state['model_state_dict']['classifier.weight'].shape[0]==2
            report[method]={'complete_exists':True,'actual_checkpoint_keys_read_CPU':True,
                            'checkpoint_path':str(ck),'model_tensor_count':len(state['model_state_dict'])}
            del state
            continue
        existing=(runner.OUT/method/'started.json').exists()
        buffer=io.StringIO()
        with patch.object(runner.base,'configure_gpu',side_effect=AssertionError('GPU forbidden')),\
             patch.object(runner.base,'load_model',side_effect=AssertionError('model load forbidden')),\
             patch.object(runner.torch,'load',side_effect=AssertionError('checkpoint load forbidden')),\
             redirect_stdout(buffer):
            runner.infer(method)
        assert 'WAIT_UPSTREAM_QA_SELECTION_NO_INFERENCE' in buffer.getvalue()
        assert (runner.OUT/method/'started.json').exists()==existing
        report[method]={'complete_exists':False,'actual_missing_upstream_infer_call_returned':True,
            'GPU_model_checkpoint_loads':0,'actual_trained_checkpoint_inspection':'not available; producer keys/templates checked from source only'}
    assert not torch.cuda.is_initialized()
    return report

def actual_geometry_check():
    windows=lines(runner.OUT/'windows.jsonl');answers=lines(runner.OUT/'answers.jsonl');folds=read(runner.OUT/'folds.json')
    token_rows=lines(runner.scene.R16/'data/tokens_train.jsonl')
    native={}
    for r in token_rows:
        assert r['split']=='train';native[(r['row_id'],r['token_index'])]=r
    for w in windows:
        assert w['split']=='train' and len(w['item_ids'])==1
        raw=w['raw_token_indices'];assert raw==list(range(raw[0],raw[-1]+1)) and len(raw)<=4
        ix=[j for j in raw if native[(w['row_id'],j)]['lexical']]
        assert ix and ix==w['lexical_token_indices']
        assert (w['gold'] in (0,1)) if w['main_eligible'] else w['gold'] is None
        if w['main_eligible']:assert w['gold']==int(any(native[(w['row_id'],j)]['gold']==1 for j in ix))
    refusal=[a for a in answers if a['reviewed_safe_refusal']]
    assert all(a['main_eligible'] and a['gold']==0 for a in refusal)
    unresolved=[a for a in answers if not a['main_eligible']]
    assert all(a['gold'] is None for a in unresolved)
    group=set(a['group_id'] for a in answers);ev=[]
    for fold in folds:
        fit,cal,test=(set(fold[k]) for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert not(fit&cal or fit&test or cal&test) and fit|cal|test==group
        ev.extend(test)
    assert len(ev)==len(set(ev))==278
    result={'candidate_windows':len(windows),'eligible_windows':sum(w['main_eligible'] for w in windows),
            'risk_windows':sum(w['gold']==1 for w in windows),'answers':len(answers),
            'eligible_answers':sum(a['main_eligible'] for a in answers),'risk_answers':sum(a['gold']==1 for a in answers),
            'safe_refusal_negatives':len(refusal),'unresolved_excluded_answers':len(unresolved),
            'all_original_raw_windows_lexical_indices_and_gold_exact':True,'fivefold_groups_disjoint_and_evaluated_once':True}
    assert result['eligible_windows']==9526 and result['risk_windows']==1063
    assert result['eligible_answers']==598 and result['risk_answers']==151 and len(refusal)==117 and len(unresolved)==4
    old=read(runner.scene.R26/'summary.json')['methods']
    for method in ('lookback_tuned','redeep_tuned','local_slots_fusion_smooth_global'):
        assert all(k in old[method][unit] for unit in ('windows','answers') for k in ('precision','recall','f1'))
    return result

def synthetic_score_check():
    answers=[];windows=[];predictions={};lex={};gold_tokens={}
    for g in range(5):
        for case in range(4):
            rid=f'group{g}_case{case}';iid=rid+'__1';group=f'g{g}';shift=(g-2)*.04
            scores=np.asarray([[.75,.99,.30,.50,.98,.10],[.15,.99,.20,.15,.98,.30],
                               [.70,.99,.75,.60,.99,.80],[.86,.99,.84,.88,.99,.90]][case],np.float64)
            scores[[0,2,3,5]]+=shift
            predictions[rid]=scores;lex[rid]={0,2,3,5};gold_tokens[rid]=({0} if g%2 else {2}) if case==0 else set()
            answers.append({'item_id':iid,'row_id':rid,'group_id':group,'main_eligible':case!=3,
                'gold':int(case==0) if case!=3 else None,'reviewed_safe_refusal':case==2})
            for start in range(3):
                raw=list(range(start,start+4));li=[j for j in raw if j in lex[rid]]
                windows.append({'window_key':rid+f'_w{start}','row_id':rid,'item_ids':[iid],'group_id':group,
                    'raw_token_indices':raw,'lexical_token_indices':li,'main_eligible':case in (0,1),
                    'gold':int(bool(set(li)&gold_tokens[rid])) if case in (0,1) else None})
    rng=np.random.default_rng(20260912);rng.shuffle(windows);rng.shuffle(answers)
    # Independent max computed from original raw indices and the lexical mask;
    # answer order deliberately differs from window and NPZ order.
    wv=np.asarray([max(predictions[w['row_id']][j] for j in w['raw_token_indices'] if j in lex[w['row_id']]) for w in windows])
    av=np.asarray([max(predictions[a['row_id']][j] for j in lex[a['row_id']]) for a in answers])
    actual_w,actual_a=runner.aggregate(predictions,windows,answers)
    assert np.array_equal(wv,actual_w) and np.array_equal(av,actual_a)
    assert max(wv)<1 and any(predictions[rid][1]>max(predictions[rid][j] for j in lex[rid]) for rid in predictions)
    folds=[{'calibration_groups':[f'g{(f+1)%5}'],'evaluation_groups':[f'g{f}'],
            'fit_groups':[f'g{x}' for x in range(5) if x not in (f,(f+1)%5)]} for f in range(5)]
    wm=[w['main_eligible'] for w in windows];am=[a['main_eligible'] for a in answers]
    qa_ts={'window':{'threshold':.55},'answer':{'threshold':.65}}
    direct={'windows':counts([w['gold'] for w in windows if w['main_eligible']],[v>=.55 for w,v in zip(windows,wv) if w['main_eligible']]),
            'answers':counts([a['gold'] for a in answers if a['main_eligible']],[v>=.65 for a,v in zip(answers,av) if a['main_eligible']])}
    assert direct['answers']['fp']==5,'The five high-scoring safe refusals must count as false positives'
    expected_wp=np.zeros(len(windows),bool);expected_ap=np.zeros(len(answers),bool);expected_folds=[]
    for f,fold in enumerate(folds):
        cw=[i for i,w in enumerate(windows) if w['main_eligible'] and w['group_id'] in fold['calibration_groups']]
        ca=[i for i,a in enumerate(answers) if a['main_eligible'] and a['group_id'] in fold['calibration_groups']]
        wt=threshold([windows[i]['gold'] for i in cw],wv[cw].tolist());at=threshold([answers[i]['gold'] for i in ca],av[ca].tolist())
        ew=[i for i,w in enumerate(windows) if w['group_id'] in fold['evaluation_groups']]
        ea=[i for i,a in enumerate(answers) if a['group_id'] in fold['evaluation_groups']]
        for i in ew:expected_wp[i]=wv[i]>=wt['threshold']
        for i in ea:expected_ap[i]=av[i]>=at['threshold']
        expected_folds.append({'fold':f,'thresholds':{'window':wt,'answer':at},
            'windows':counts([windows[i]['gold'] for i in ew if wm[i]],[bool(expected_wp[i]) for i in ew if wm[i]]),
            'answers':counts([answers[i]['gold'] for i in ea if am[i]],[bool(expected_ap[i]) for i in ea if am[i]]),'model_fitted_on_R16':False})
    calibrated={'windows':counts([w['gold'] for w in windows if w['main_eligible']],expected_wp[wm].tolist()),
                'answers':counts([a['gold'] for a in answers if a['main_eligible']],expected_ap[am].tolist())}
    with tempfile.TemporaryDirectory(prefix='synthetic_score_',dir=OUT) as temporary:
        tmp=Path(temporary);methoddir=tmp/'qa_only';methoddir.mkdir();old=tmp/'reference';old.mkdir()
        for name,r in [('windows.jsonl',windows),('answers.jsonl',answers)]:savel(tmp/name,r)
        save(tmp/'folds.json',folds);np.savez_compressed(methoddir/'token_predictions.npz',**predictions)
        save(methoddir/'inference_complete.json',{'token_predictions_sha256':sha(methoddir/'token_predictions.npz'),'source_freeze':{'QA_thresholds':qa_ts}})
        save(old/'summary.json',{'methods':{name:{unit:{k:0. for k in ('precision','recall','f1')} for unit in ('windows','answers')} for name in ('lookback_tuned','redeep_tuned','local_slots_fusion_smooth_global')}})
        with patch.object(runner,'OUT',tmp),patch.object(runner,'check_prepared',return_value={}),patch.object(runner.scene,'R26',old),redirect_stdout(io.StringIO()):
            runner.score('qa_only')
        result=read(methoddir/'summary.json')
        assert result['unchanged_QA_thresholds']==direct
        assert result['folds']==expected_folds
        assert result['existing_R16_fivefold_threshold_calibration']==calibrated
        with np.load(methoddir/'scores.npz',allow_pickle=False) as z:
            assert np.array_equal(z['window_scores'],wv) and np.array_equal(z['answer_scores'],av)
            assert np.array_equal(z['fivefold_window_predictions'],expected_wp)
            assert np.array_equal(z['fivefold_answer_predictions'],expected_ap)
    return {'passed':True,'scope':'Actual score() executed only on synthetic fixture; no real-model score',
        'candidate_windows':len(windows),'answers':len(answers),'punctuation_high_scores_ignored':True,
        'overlapping_windows_preserved_not_deduplicated':True,'shuffled_answer_and_window_orders_matched_by_item':True,
        'high_score_safe_refusals_count_as_answer_FP':5,'unresolved_answers_excluded_not_zero_imputed':5,
        'every_cal_threshold_bruteforce_exact':True,'fixed_QA_thresholds_direct_counts':direct,
        'independent_fivefold_counts':calibrated,'all_prediction_arrays_exact':True}

def main():
    assert not torch.cuda.is_initialized();OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(runner.__file__),Path(runner.scene.__file__),Path(runner.base.__file__)]
    snapshot={str(p.resolve()):sha(p) for p in paths}
    report={'status':'running','source_sha256':snapshot,'script_sha256':sha(__file__)}
    try:
        report['upstream_gate']=missing_upstream_check()
        report['real_prepared_geometry']=actual_geometry_check()
        report['synthetic_score_integration']=synthetic_score_check()
        assert snapshot=={str(p.resolve()):sha(p) for p in paths}
        assert not torch.cuda.is_initialized()
        report.update(status='passed_no_blocking_logic_issue',GPU_used=False,model_fitting=False,heldout_files_opened=False,
            checkpoint_schema_note='Both producer sources save model_state_dict; selected.epoch, thresholds, artifacts_sha256[.pt] and template names agree. Actual completed checkpoint inspection is only claimed where upstream_gate records it.',
            interpretation='Fixed-QA thresholds and fold-calibrated thresholds are separate analyses. Fivefold R16 numbers are repeated development with threshold labels, not untouched transfer-test accuracy.')
    except Exception as e:
        report.update(status='failed',failure=repr(e));raise
    finally:save(OUT/'CPU_LOGIC_REVIEW.json',report)
    print(json.dumps({'status':report['status'],'real_counts':report['real_prepared_geometry'],'GPU_used':False},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
