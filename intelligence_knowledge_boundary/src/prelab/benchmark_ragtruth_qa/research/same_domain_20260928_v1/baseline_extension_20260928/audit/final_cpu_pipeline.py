"""Wait for ReDeEP completion, then audit/freeze/evaluate/report on CPU only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
RUN = EXT.parent
PROJECT = next(p for p in RUN.parents if p.name == 'intelligence_knowledge_boundary')
WORKSPACE = PROJECT.parent
STATUS = HERE / 'FINAL_CPU_STATUS.json'
COMPLETE = HERE / 'FINAL_CPU_COMPLETE.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def state(stage, status='running', **kwargs):
    save(STATUS, {'stage': stage, 'status': status, 'pid': os.getpid(), 'updated_at': time.time(), **kwargs})


def run_stage(name, script, *arguments):
    state(name)
    command = [sys.executable, '-u', '-B', '-X', 'utf8', str(EXT / 'run_stage.py'),
               name, str(HERE / script), *map(str, arguments)]
    with (HERE / f'{name}_driver.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        state(name, child_pid=process.pid, command=command)
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} failed with exit {code}; inspect {HERE / (name + "_driver.log")}')
    receipt = read(EXT / 'execution' / name / 'LATEST.json')
    assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
    assert receipt['entry_sha256'] == receipt['entry_sha256_after'] == sha(HERE / script)
    print('CPU_STAGE_COMPLETE', name, flush=True)
    return receipt


def final_delivery_audit(original_documents, stage_receipts):
    result = read(HERE / 'SIX_BASELINES_RESULTS.json')
    audit = read(HERE / 'SIX_BASELINES_AUDIT.json')
    frozen = read(HERE / 'EXTENSION_FROZEN.json')
    outputs = read(EXT / 'REPORT_OUTPUTS.json')
    report_paths = list(map(Path, outputs['reports']))
    assert len(report_paths) == 2 and all(p.exists() for p in report_paths)
    assert sha(report_paths[0]) == sha(report_paths[1]) == outputs['report_sha256']
    for path, digest in original_documents.items():
        assert sha(path) == digest, f'Protected original/formal document changed: {path}'
    assert audit['status'] == 'PASS' and audit['result_sha256'] == sha(HERE / 'SIX_BASELINES_RESULTS.json')
    assert result['manifest_sha256'] == sha(HERE / 'EXTENSION_FROZEN.json')
    assert result['primary_candidate'] == 'Fusion-window-HGB'
    mains = [m for m in frozen['methods'] if not m.get('supplementary_only')]
    baselines = [m for m in mains if m['role'] == 'baseline']
    assert len(mains) == 7 and len(baselines) == 6 and len(result['methods']) == 8
    assert result['test_answers_primary'] == 155 and result['test_answers_all'] == 279
    assert result['test_groups_primary'] == 75 and result['bootstrap_samples'] == 2000
    from generate_report import BASELINES, LABELS, PRIMARY, interval_description, strongest_conclusion
    assert {m['name'] for m in baselines} == set(BASELINES)
    text = report_paths[0].read_text(encoding='utf-8')
    parsed_tables = {}
    for heading, cohort in [('## 主测试：155 条', 'no_hint'), ('## 全量补充：279 条', 'all')]:
        section = text.split(heading, 1)[1].split('\n## ', 1)[0]
        rows = [line for line in section.splitlines() if line.startswith('| ')][1:]
        assert len(rows) == 7
        for line, name in zip(rows, [*BASELINES, PRIMARY]):
            cells = [c.strip() for c in line.strip('|').split('|')]
            assert cells[0] == LABELS[name]
            scores = result['methods'][name]['cohorts'][cohort]
            expected = [f'{scores[level][k]:.4f}' for level in ['answer', 'window']
                        for k in ['precision', 'recall', 'f1', 'auroc']]
            assert cells[1:] == expected
        parsed_tables[cohort] = {'rows': 7, 'all_printed_values_match_result_json': True}
    for level in ['answer', 'window']:
        conclusion, winners = strongest_conclusion(result, level)
        assert conclusion in text
        for name in BASELINES:
            delta = result['paired_f1_differences'][PRIMARY + ' minus ' + name][level]
            assert interval_description(delta['paired_95ci']) in text
    assert 'SOTA' not in text and '显著' not in text
    summary = {name: {lev: result['methods'][name]['cohorts']['no_hint'][lev]['f1']
                     for lev in ['answer', 'window']} for name in [*BASELINES, PRIMARY]}
    record = {'status': 'PASS', 'independent_baselines': 6, 'primary_candidates': 1,
        'default_lookback_only_supplementary': True, 'report_copies_sha256_identical': outputs['report_sha256'],
        'tables': parsed_tables, 'test_answers_primary': 155, 'test_answers_all': 279,
        'bootstrap_samples': 2000, 'strongest_baseline_and_ci_text_recomputed': True,
        'protected_document_hashes_unchanged': original_documents,
        'main_f1': summary, 'reports': outputs['reports'],
        'stage_receipts': stage_receipts, 'GPU_used': False,
        'finished_at': time.time()}
    save(HERE / 'FINAL_DELIVERY_AUDIT.json', record)
    return record


def main():
    assert not COMPLETE.exists(), 'Final CPU pipeline already completed; inspect outputs rather than restart'
    assert not (HERE / 'EXTENSION_FROZEN.json').exists(), 'Existing extension freeze requires an explicit replay path'
    original_documents = {str(p): sha(p) for p in (PROJECT / 'reports/paper').glob('*.docx')}
    assert len(original_documents) >= 2
    existing_semantic_hashes = {}
    for folder in ['lettuce_v1_run', 'minicheck_run']:
        for part in ['validation', 'test']:
            path = EXT / 'semantic' / folder / f'{part}_predictions.jsonl'
            existing_semantic_hashes[str(path)] = sha(path)
    save(HERE / 'FINAL_CPU_START_BINDINGS.json', {'documents': original_documents,
        'semantic_predictions': existing_semantic_hashes, 'started_at': time.time(),
        'new_test_evaluation_authorized_by_root': True, 'poll_seconds': 50})
    marker = EXT / 'whitebox/REDEEP_GPU_QUEUE_COMPLETE.json'
    print('CPU_PIPELINE_WAITING_FOR_REDEEP', flush=True)
    while not marker.exists():
        upstream = EXT / 'whitebox/REDEEP_GPU_QUEUE_STATUS.json'
        status = read(upstream) if upstream.exists() else {}
        assert status.get('status') != 'failed', status
        for stage in ['redeep_smoke', 'redeep_extract', 'redeep_score']:
            path = EXT / 'execution' / stage / 'LATEST.json'
            if path.exists():
                value = read(path)
                assert value.get('status') != 'failed', value
        progress = EXT / 'whitebox/redeep/PROGRESS.json'
        state('waiting_for_redeep', status='waiting', poll_seconds=50,
              upstream_progress=read(progress) if progress.exists() else {})
        time.sleep(50)
    completion = read(marker)
    assert completion['status'] == 'complete'
    assert completion['all_gpu_children_exited'] and completion['GPU_released']
    assert completion['raw_cache_counts'] == {'fit': 1292, 'validation': 271, 'test': 279}
    assert completion['prediction_counts'] == {'validation': 271, 'test': 279}
    for path, digest in existing_semantic_hashes.items():
        assert sha(path) == digest, path
    receipts = []
    for name, script, arguments in [
        ('final_input_audit', 'audit_whitebox.py', []),
        ('final_lettuce_audit', 'audit_lettuce.py', []),
        ('final_minicheck_audit', 'audit_minicheck.py', []),
        ('prepare_six_baselines', 'prepare_new_config.py', []),
        ('freeze_six_baselines', 'evaluate_extensions.py', ['freeze-new', '--config', HERE / 'new_methods.json']),
        ('evaluate_six_baselines', 'evaluate_extensions.py', ['evaluate']),
        ('report_six_baselines', 'generate_report.py', []),
    ]:
        receipts.append(run_stage(name, script, *arguments))
    state('final_delivery_audit')
    record = final_delivery_audit(original_documents, receipts)
    save(COMPLETE, {'status': 'complete', 'reports': record['reports'], 'main_f1': record['main_f1'],
        'delivery_audit_sha256': sha(HERE / 'FINAL_DELIVERY_AUDIT.json'), 'GPU_used': False,
        'finished_at': time.time()})
    state('complete', status='complete')
    print(json.dumps({'status': 'complete', 'main_f1': record['main_f1'], 'reports': record['reports']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        state('failed', status='failed', error=traceback.format_exc())
        raise
