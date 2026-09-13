"""Full admitted calibration round-trip; never reads official raw test rows."""
from pathlib import Path
from datetime import datetime,timezone
import math
import numpy as np
import contracts as c
import export_qa
import score_frozen as scoring

def main():
    assert not c.AUTHORIZATION.exists(),'Simulation stage expects real release to remain unauthorized'
    out=c.HERE/'simulation_calibration';export_qa.simulate(out)
    bundle=out/'bundle_manifest.json';bm=c.check_bundle(bundle)
    old=c.ROOT/'results/development_v1';name='lookback_mean_C0.001'
    entry=c.read(old/(name+'_result.json'))
    with np.load(old/(name+'_scores.npz'),allow_pickle=False) as z:s=z['window_scores'][168123:].copy();a=z['answer_scores'][634:].copy()
    windows=c.lines(out/'windows_k4_calibration.jsonl');answers=c.lines(out/'answers_calibration.jsonl')
    assert len(windows)==len(s)==42241 and len(answers)==len(a)==159
    artifacts={str((old/(name+ext)).resolve()):c.sha(old/(name+ext)) for ext in ('.pkl','_result.json')}
    method={'method_id':'fixed_old_calibration_fixture','primary':True,'prediction_unit':'raw_window','window_score_interface':'all_eligible_original_k4_windows','score_direction':'larger_is_risk','answer_aggregation':'max_all_eligible_windows',
        'window_threshold':entry['thresholds']['window']['threshold'],'answer_threshold':entry['thresholds']['answer']['threshold'],'artifacts_sha256':artifacts,'selection_record_sha256':c.sha(old/(name+'_result.json'))}
    frozen={'complete':True,'status':'frozen_before_test_release','purpose':'calibration_simulation','selection_data':'fit_and_calibration_only','test_content_used_for_selection':False,
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'development_files_sha256':{str((c.ROOT/'data/gold_manifest.json').resolve()):c.sha(c.ROOT/'data/gold_manifest.json')},'methods':[method],
        'group_bootstrap':{'unit':'group_id','algorithm':'pooled_confusion_percentile','quantile_method':'linear','replicates':5000,'seed':20261001,'confidence_level':.95,'zero_denominator_value':0}}
    fp=out/'simulation_development_freeze.json';c.save(fp,frozen)
    predictions=[{'window_id':w['window_id'],'response_id':w['response_id'],'group_id':w['group_id'],'score':float(score)} for w,score in zip(windows,s)]
    predpath=out/'fixed_old_window_predictions.jsonl';c.save_lines(predpath,predictions)
    pm={'complete':True,'mode':'calibration_simulation','development_freeze_sha256':c.sha(fp),'bundle_manifest_sha256':c.sha(bundle),
        'methods':[{'method_id':method['method_id'],'window_scores_file':str(predpath.resolve()),'window_scores_sha256':c.sha(predpath),'source_artifacts_sha256':artifacts,
            'native_prediction_artifacts_sha256':{str((old/(name+'_scores.npz')).resolve()):c.sha(old/(name+'_scores.npz'))},'fit_performed':False,'threshold_selection_performed':False,'model_selection_performed':False}]}
    pmp=out/'prediction_manifest.json';c.save(pmp,pm)
    result=scoring.run(fp,bundle,pmp,out/'scores',True)
    measured=result['methods_in_frozen_order'][0]['metrics'];reference=entry['metrics']['calibration'];maxerr=0.
    for unit in ('windows','answers'):
        assert measured[unit].keys()==reference[unit].keys()
        for k,v in reference[unit].items():
            if isinstance(v,float):
                err=abs(measured[unit][k]-v);assert err<1e-12;maxerr=max(maxerr,err)
            else:assert measured[unit][k]==v
    computed=c.lines(out/'scores/method_000_answers.jsonl');assert np.array_equal([r['score'] for r in computed],a)
    # Focused strictness checks reuse these admitted rows, not real test data.
    rejected=[];smallboot={**frozen['group_bootstrap'],'replicates':2}
    for label,bad in [('missing_window',predictions[:-1]),('duplicate_window',predictions[:-1]+[predictions[0]]),('nan_score',[{**predictions[0],'score':float('nan')}]+predictions[1:])]:
        try:scoring.score_records(answers,windows,[method],{method['method_id']:bad},smallboot)
        except AssertionError:rejected.append(label)
        else:raise AssertionError('Invalid scores unexpectedly accepted: '+label)
    empty={**answers[0],'response_id':'synthetic_missing_windows','answer_id':'synthetic_missing_windows','eligible_window_count':0}
    try:scoring.score_records(answers+[empty],windows,[method],{method['method_id']:predictions},smallboot)
    except AssertionError:rejected.append('answer_without_eligible_window')
    else:raise AssertionError('Unscored answer silently dropped')
    try:c.require_release(c.HERE/'nonexistent_development_freeze.json')
    except AssertionError:rejected.append('missing_explicit_root_authorization')
    else:raise AssertionError('Real release gate unexpectedly passed')
    table=scoring.group_confusion(np.array([1,1,0]),np.array([.9,.8,.8]),.5,['a','a','b'],['a','b'])
    assert np.array_equal(table,[[2,0,0,0],[0,1,0,0]])
    assert np.array_equal(np.array([[2,0],[0,2],[1,1]])@table,[[4,0,0,0],[0,2,0,0],[2,1,0,0]])
    report={'passed':True,'actual_source':'Existing calibration exports only, including3 already-known quality exclusions; fixed named old model, no candidate selection.',
        'layout_and_gold_all159_records_exact':True,'quality_eligible159_excluded3':True,'raw_tokens':bm['counts']['raw_tokens'],'windows':len(windows),'risk_windows':int(sum(w['label'] for w in windows)),
        'answers':len(answers),'risk_answers':int(sum(a['label'] for a in answers)),'point_metrics_max_abs':maxerr,'all159_answer_max_exact':True,
        'bootstrap_units_are_whole_groups':True,'simulation_bootstrap':frozen['group_bootstrap'],'strict_rejections':rejected,
        'fit_threshold_or_model_selection_performed':False,'official_test_answers_quality_labels_read':False,'real_release_performed':False,'GPU_used':False,
        'bundle_sha256':c.sha(bundle),'score_complete_sha256':c.sha(out/'scores/complete.json'),'helper_sha256':c.sha(__file__),
        'remaining_gate':'Root must freeze all final methods/artifacts/thresholds/inference adapters/bootstrap and explicitly authorize before real raw150 content is read.'}
    c.save(out/'SELFTEST_COMPLETE.json',report)
    print('FINAL_TEST_TOOLS_CALIBRATION_SELFTEST_PASS',report,flush=True)

if __name__=='__main__':main()
