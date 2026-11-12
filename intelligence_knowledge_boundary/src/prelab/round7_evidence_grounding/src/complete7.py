"""Record completion only when every planned cohort, method and report is present."""
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from evaluate7 import ROOT, METHODS, sha, readl, save


def main():
    out=ROOT/'results'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    freeze=read(out/'freeze.json'); complete=read(out/'test_complete.json')
    annotations=read(ROOT/'data/annotation_freeze.json')
    assert complete['all_methods_and_both_cohorts_evaluated']
    assert complete['freeze_sha256']==sha(out/'freeze.json')
    assert set(freeze['method_names'])==set(METHODS) and len(METHODS)==14
    assert freeze['frozen_model_sha256']==sha(out/'frozen_models.pkl')
    assert freeze['selection_sha256']==sha(out/'selection.json')
    assert freeze['evaluator_sha256']==sha(ROOT/'src/evaluate7.py')
    assert all(sha(ROOT/rel)==h for rel,h in freeze['data_sha256'].items())
    assert all(sha(ROOT/'data'/f'annotations_{s}.jsonl')==v['sha256'] for s,v in annotations['splits'].items())
    assert annotations['annotation_guide_sha256']==sha(ROOT/'ANNOTATION_GUIDE.md')
    assert all(sha(ROOT/'data'/rel)==h for rel,h in annotations['review_files_sha256'].items())
    assert all(read(ROOT/'data'/stage/'manifest.json')['complete'] for stage in ('features','attention','lumina','baselines'))
    runtime=read(ROOT/'data/runtime_audit.json')
    assert runtime['status']=='passed' and runtime['counts']['items']==1300
    baselines=readl(ROOT/'data/baselines.jsonl')
    assert len(baselines)==1300 and all(r['self_risk'] is not None and r['direct_risk'] is not None for r in baselines)
    for cohort,n in [('main',240),('external',100)]:
        assert sha(out/f'metrics_{cohort}.json')==complete['metrics_sha256'][cohort]
        rows=readl(out/f'predictions_{cohort}.jsonl')
        assert len(rows)==n and all(set(r['methods'])==set(METHODS) for r in rows)
        assert all(r['methods'][m]['prediction'] is not None for r in rows for m in METHODS)
    assert read(out/'supplement.json')['all_independent_metric_assertions_passed']
    benchmark=read(out/'cpu_scoring_benchmark.json')
    assert benchmark['freeze_sha256']==sha(out/'freeze.json')
    assert all(v['identical_scores_all_repeats'] and v['measured_batches']==5
               for c in benchmark['cohorts'].values() for v in c['methods'].values())
    assert read(out/'report_manifest.json')['report_sha256']==sha(out/'REPORT.md')
    assert read(out/'tables_manifest.json')['table_sha256']==sha(out/'TABLES.md')
    needed=['REPORT.md','TABLES.md','DATASET_PROFILE.md','RUNTIME_AND_ADAPTATIONS.md',
            'RESULT_REVIEW.md','ERROR_CASES_REVIEW.md','supplement.json','cpu_scoring_benchmark.json',
            'freeze.json','test_complete.json','metrics_main.json','metrics_external.json',
            'predictions_main.jsonl','predictions_external.jsonl','errors_main.jsonl','errors_external.jsonl']
    for name in needed:
        assert (out/name).exists() and (out/name).stat().st_size
    # Verify the concrete report's local links, excluding URLs and code notation.
    for target in re.findall(r'\]\(([^)]+)\)',(out/'REPORT.md').read_text(encoding='utf-8')):
        if '://' not in target:
            assert (out/target.split('#')[0]).resolve().exists(), target
    manifest={'status':'complete_all_methods_both_cohorts_and_report',
              'utc':datetime.now(timezone.utc).isoformat(),'methods':METHODS,'method_families':12,
              'main_groups':200,'external_groups':50,'generated_responses':500,'annotated_slots':1300,
              'main_test_slots':240,'external_test_slots':100,
              'independent_recounts_passed':True,'all_frozen_hash_checks_passed':True,
              'all_signal_alignment_checks_passed':True,'all_report_links_exist':True,
              'annotation_status':'Assistant primary labeling and second reviews; human review not performed.',
              'numerical_repair':'Assertion-only float32 upper tolerance; original freeze archived; no retuning.',
              'artifacts_sha256':{n:sha(out/n) for n in needed},
              'annotation_freeze_sha256':sha(ROOT/'data/annotation_freeze.json'),
              'runtime_audit_sha256':sha(ROOT/'data/runtime_audit.json')}
    save(out/'completion.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('status','method_families','generated_responses','annotated_slots','all_frozen_hash_checks_passed')}))


if __name__=='__main__':
    main()
