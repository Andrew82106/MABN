"""Freeze stable exporter/scorer code and incomplete examples, no raw test I/O."""
import contracts as c

def main():
    identity=c.read(c.IDENTITY)
    assert identity['quality_good_count'] is None and identity['test_risk_counts'] is None
    names=('contracts.py','gold_rows.py','export_qa.py','score_frozen.py','simulate_calibration.py','initialize_tools.py')
    paths=[c.HERE/n for n in names]+[c.ROOT/'src/build_gold.py',c.ROOT/'src/feature_qa.py',
        c.ROOT/'data/development_manifest.json',c.ROOT/'data/feature_preparation/manifest.json',
        c.ROOT/'data/feature_preparation/plans.jsonl']
    protocol={'version':'ragtruth-qa-final-data-metrics-interface-v2','real_test_released':False,'final_models_specified':False,
        'identity_manifest_sha256':c.sha(c.IDENTITY),'source_index_sha256':c.sha(c.ROOT/'data/source_index.jsonl'),
        'annotation_protocol_sha256':c.sha(c.ROOT/'ANNOTATION_PROTOCOL.md'),'old_gold_manifest_sha256':c.sha(c.ROOT/'data/gold_manifest.json'),
        'tool_files_sha256':{str(p.resolve()):c.sha(p) for p in paths},'identity':'Fixed150 official test QA sources/147 associated groups, specified original llama-2-7b-chat response per source. Withheld linked31 train sources and other model responses excluded by identity.',
        'raw_access_gate':'Explicit root authorization at data/FINAL_TEST_RELEASE_AUTHORIZATION.json must bind complete final development freeze, these tools and fixed identity. No authorize or model-fit command is provided.',
        'quality':'Exactly official quality==good. Preserve all150 records and non-good exclusion reasons after authorization. Good count unknown before release.',
        'canonical_layout':'Same frozen Llama tokenizer and literal wrapper, full-string offsets with boundary-crossing first token; max4096. Preserve genuine test metadata; old development-only model adapters are not bypassed.',
        'gold':'Reuse old character helpers; all four human span types and implicit_true/due_to_null preserved. Lexical=covered isalnum; positive token=overlapping risk isalnum;4rawBPE stride1 window any risk lexical token. N<4 one actual window. Exclude only no-lexical windows; no automatic refusal exclusions.',
        'edge_policy':'Report zero/nonalnum spans without relabeling. Invalid geometry, lost content, empty tokenized answer or missing inference prevent complete main scoring. A good answer with no eligible window remains in gold and stops max-window scoring, never dropped or scored0.',
        'scoring':'No fitting or threshold/candidate selection. Require all frozen methods and every eligible window exactly once. Larger score meansrisk, fixed per-method thresholds, answer=max all eligible windows.',
        'bootstrap':'Actual parameters must be supplied by root development freeze before release. Interface supports group_id pooled-confusion percentile CIs with common group draws across all methods, linear quantiles, zero denominators0. Example5000 seed20261001 level.95 is a proposal, not selection from test.',
        'model_adapter_boundary':'These tools do not invent or run final models. A separately frozen adapter must produce complete canonical window predictions and bind actual model/transform/code/native prediction artifacts. Raw-token scores only exist for methods whose native output defines them; no fake token probabilities are assigned to window classifiers.',
        'validation':'Round-trip only already-open159 calibration, known3 non-good calibration rows, and one fixed old LR prediction/threshold artifact. Official test answer/quality/labels remain unread.',
        'test_exposure':'Once final results are inspected, subsequent model changes require a new independent held-out test; do not overwrite this release/freeze/result directory.'}
    c.save(c.HERE/'protocol.json',protocol)
    example={'complete':False,'status':'NOT_READY_EXAMPLE_ONLY','purpose':'official_test_release','selection_data':'fit_and_calibration_only','test_content_used_for_selection':False,
        'frozen_at_utc':None,'identity_manifest_sha256':c.sha(c.IDENTITY),'tools_protocol_sha256':c.sha(c.HERE/'protocol.json'),
        'development_files_sha256':{},'methods':[],'group_bootstrap':{'unit':'group_id','algorithm':'pooled_confusion_percentile','quantile_method':'linear','replicates':5000,'seed':20261001,'confidence_level':.95,'zero_denominator_value':0},
        'method_schema':{'method_id':'ROOT_MUST_CHOOSE','primary':True,'prediction_unit':'raw_window','window_score_interface':'all_eligible_original_k4_windows','score_direction':'larger_is_risk','answer_aggregation':'max_all_eligible_windows','window_threshold':None,'answer_threshold':None,'artifacts_sha256':{},'selection_record_sha256':None}}
    c.save(c.HERE/'development_freeze.example.json',example)
    c.save(c.HERE/'release_authorization.example.json',{'authorized':False,'authorization':'NOT_AUTHORIZED_EXAMPLE_ONLY','scope':'ragtruth_qa_official_test_150_llama2_only',
        'authorization_utc':None,'development_freeze_path':None,'development_freeze_sha256':None,'identity_manifest_sha256':c.sha(c.IDENTITY),'tools_protocol_sha256':c.sha(c.HERE/'protocol.json')})
    print('FINAL_TEST_INTERFACES_FROZEN_NOT_RELEASED',c.sha(c.HERE/'protocol.json'),flush=True)

if __name__=='__main__':main()
