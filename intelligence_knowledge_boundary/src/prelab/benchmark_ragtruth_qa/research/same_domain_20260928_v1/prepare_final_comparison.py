"""Assemble completed runs and fixed-average controls without opening test labels."""
from pathlib import Path
import hashlib
import json
import numpy as np

import metrics
from train_fusion import predictions, write_predictions

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_hash(path, expected):
    assert sha(path) == expected, ('Prediction provenance hash mismatch', str(path))


def verify_sources(recorded, expected_paths):
    actual = {Path(p).resolve(): digest for p, digest in recorded.items()}
    assert set(actual) == {Path(p).resolve() for p in expected_paths}, 'Unexpected prediction input files'
    for path, digest in actual.items():
        verify_hash(path, digest)


def verify_semantic(run, test_dir):
    completion = read(run / 'complete.json')
    verify_hash(run / 'best.pt', completion['checkpoint_sha256'])
    # The validation prediction copy must still belong to the selected epoch.
    epoch = completion['best']['epoch']
    verify_hash(run / 'best_validation_predictions.jsonl', sha(run / f'epoch_{epoch:02d}_validation_predictions.jsonl'))
    receipt_path = test_dir / 'complete.json'
    receipt, prepared = read(receipt_path), read(test_dir / 'prepared.json')
    assert receipt['test_labels_read'] is False and receipt['evaluation_run'] is False
    verify_hash(run / 'best.pt', receipt['checkpoint_sha256'])
    verify_hash(test_dir / 'predictions.jsonl', receipt['predictions_sha256'])
    verify_hash(test_dir / 'inputs.jsonl', prepared['input_sha256'])
    verify_sources(prepared['source_sha256'], [HERE / 'data/test/inputs.jsonl',
                   HERE / 'data/test/geometry.jsonl', HERE / 'data/split_assignments.jsonl'])
    assert receipt['answers'] == prepared['answers']
    return [run / 'complete.json', receipt_path, test_dir / 'prepared.json']


def verify_whitebox(run, prediction, partition):
    record = read(run / 'RUN.json')
    completion = read(run / 'COMPLETE.json')
    verify_hash(run / 'best.pt', completion['checkpoint_sha256'])
    verify_hash(run / 'preprocessor.pkl', record['preprocessor_sha256'])
    receipt_path = prediction.with_suffix('.manifest.json')
    receipt = read(receipt_path)
    assert receipt['partition'] == partition and receipt['labels_read'] is False
    assert receipt['training_member_count'] == 0 and receipt['all_predictions_out_of_fit']
    verify_hash(prediction, receipt['prediction_sha256'])
    verify_hash(run / 'best.pt', receipt['checkpoint_sha256'])
    verify_hash(run / 'RUN.json', receipt['training_run_sha256'])
    verify_hash(run / 'FIT_IDS.json', receipt['fit_ids_file_sha256'])
    assert receipt['signal_source']['source_sha256'] == record['signal_source']['source_sha256']
    geometry = {r['response_id']: r for r in metrics.readl(HERE / 'data' / partition / 'geometry.jsonl')}
    # Fold predictions can cover a subset; final freeze requires the complete test.
    sources = receipt['signal_source']['receipts']
    assert set(sources) <= set(geometry) and receipt['responses'] == len(sources)
    for rid, source in sources.items():
        assert source['input_sha256'] == geometry[rid]['input_sha256']
        assert source['source_sha256'] == record['signal_source']['source_sha256']
    if partition == 'test':
        assert set(sources) == set(geometry) and not receipt['fixed_final']
        assert receipt['thresholds'] == completion['best']['evaluation']['thresholds']
    return [run / 'COMPLETE.json', receipt_path]


def verify_baseline(method, run, directory, partition, thresholds, pooling, checkpoint=None):
    receipt_path = directory / 'PREDICTION_RECEIPT.json'
    receipt = read(receipt_path)
    assert receipt['method'] == method and receipt['partition'] == partition
    assert receipt['test_labels_opened'] is False and receipt['annotations_opened'] is False
    assert receipt['fit_called'] is False and receipt['threshold_selection_called'] is False
    assert receipt['pooling'] == pooling and receipt['thresholds'] == thresholds
    verify_hash(directory / 'predictions.jsonl', receipt['prediction_sha256'])
    verify_sources(receipt['input_files'], [HERE / 'data' / partition / f'{name}.jsonl'
                                          for name in ('inputs', 'geometry')])
    models = receipt['models']
    if method == 'lookback':
        assert set(models) == {'model.pkl', 'PROTOCOL.json', 'VALIDATION.json', 'CACHE_BINDINGS.json'}
        for name, digest in models.items():
            verify_hash(run / name, digest)
        extra = [run / 'CACHE_BINDINGS.json']
    else:
        assert method == 'ragognizer' and checkpoint is not None
        verify_hash(run / 'SELECTED.json', models['selected'])
        verify_hash(run / 'RUN_BINDING.json', models['training_binding'])
        assert models['prepared'] == read(run / 'RUN_BINDING.json')['prepared_sha256']
        files = {str(p.relative_to(checkpoint)): p for p in checkpoint.rglob('*')
                 if p.is_file() and p.suffix in {'.safetensors', '.json', '.pt', '.bin'}}
        assert files.keys() == models['checkpoint'].keys()
        for name, path in files.items():
            verify_hash(path, models['checkpoint'][name])
        extra = []
    return [receipt_path, *extra]


def verify_fusion(run, semantic_prediction, whitebox_prediction, granularity):
    selected = read(run / 'SELECTED.json')
    verify_hash(run / 'best.pkl', selected['model_sha256'])
    verify_hash(HERE / 'fusion/OOF_CONFIG.json', selected['config_sha256'])
    receipt_path = run / 'test_predictions.receipt.json'
    receipt = read(receipt_path)
    assert receipt['granularity'] == selected['granularity'] == granularity
    assert receipt['test_labels_read'] is False
    for path, field in [(run / 'best.pkl', 'model_sha256'),
                        (run / 'test_predictions.jsonl', 'predictions_sha256'),
                        (semantic_prediction, 'semantic_sha256'), (whitebox_prediction, 'whitebox_sha256')]:
        verify_hash(path, receipt[field])
    return [receipt_path]


def constant_risk_scores(geometry):
    assert len(geometry) == len({r['response_id'] for r in geometry})
    return {r['response_id']: np.ones(len(r['lexical_mask']), np.float64) for r in geometry}


def main():
    assert not (HERE / 'TEST_RELEASE.json').exists(), 'Comparison must be specified before test release'
    assert read(HERE / 'prediction_pipeline/CPU_COMPLETE.json')['all_cpu_children_exited']
    assert (HERE / 'RAG_PIPELINE_COMPLETE.json').exists()
    sem_selection = read(HERE / 'model_design/SEMANTIC_SELECTION.json')
    fusion_completion = read(HERE / 'fusion/PIPELINE_COMPLETE.json')
    fusion_selection = fusion_completion['selected_fusion']
    assert fusion_completion['semantic_backbone'] == sem_selection['chosen']
    learned_choices = [dict(name='Fusion-' + row['granularity'],
                           key=row['validation']['selection_key'], is_linear=True,
                           is_window=row['granularity'] == 'window')
                       for row in fusion_completion['both_versions']]
    nonlinear = HERE / 'fusion/nonlinear_window_v1'
    nonlinear_selection = read(nonlinear / 'SELECTED.json')
    learned_choices.append(dict(name='Fusion-window-HGB', key=nonlinear_selection['report']['selection_key'],
                                is_linear=False, is_window=True))
    # Before any test release: same shared validation key; exact ties retain
    # linear fusion, then its directly trained window version.
    primary = max(learned_choices, key=lambda r: tuple(r['key']) + (int(r['is_linear']), int(r['is_window'])))
    sem = HERE / f"model_design/modernbert_{sem_selection['chosen']}_v1"
    sem_test = HERE / f"model_design/modernbert_{sem_selection['chosen']}_test_v1/predictions.jsonl"
    wb = HERE / 'whitebox_model'
    shared_branch_artifacts = [sem / 'best.pt', sem / 'prepared.json',
                               wb / 'best.pt', wb / 'preprocessor.pkl', wb / 'RUN.json']
    semantic_receipts = {}
    for capacity in ['base', 'large']:
        semantic_receipts[capacity] = verify_semantic(HERE / f'model_design/modernbert_{capacity}_v1',
                                                     HERE / f'model_design/modernbert_{capacity}_test_v1')
    whitebox_receipts = verify_whitebox(wb, wb / 'token_predictions_test.jsonl', 'test')
    shared_branch_artifacts += semantic_receipts[sem_selection['chosen']] + whitebox_receipts
    methods = []

    def add(name, role, val, test, thresholds, artifacts, *, windows=False, pooling='max'):
        assert all(Path(p).is_file() for p in [val, test, *artifacts]), name
        methods.append({'name': name, 'role': role,
            'validation_predictions': str(val), 'test_predictions': str(test),
            'score_field': 'window_scores' if windows else 'token_scores',
            'scores_are_windows': windows, 'pooling': pooling, 'thresholds': thresholds,
            'model_artifacts': list(dict.fromkeys(str(p) for p in artifacts))})

    for variant, run in [('default', HERE / 'baseline_audit/lookback_run_v2'),
                         ('tuned', HERE / 'baseline_audit/lookback_run_v2/tuned')]:
        record = read(run / 'VALIDATION.json')
        receipts = []
        for split in ['validation', 'test']:
            receipts += verify_baseline('lookback', run,
                HERE / f'baseline_audit/predictions/lookback_{variant}_{split}', split,
                record['thresholds'], record['selected_pooling'])
        add('Lookback-' + variant, 'baseline',
            HERE / f'baseline_audit/predictions/lookback_{variant}_validation/predictions.jsonl',
            HERE / f'baseline_audit/predictions/lookback_{variant}_test/predictions.jsonl',
            record['thresholds'], [run / p for p in ['model.pkl', 'PROTOCOL.json', 'VALIDATION.json']] + receipts,
            pooling=record['selected_pooling'])

    rag = HERE / 'baseline_audit/ragognizer_run_v1'
    selected = read(rag / 'SELECTED.json')
    checkpoint = Path(selected['checkpoint'])
    checkpoint.relative_to(rag)
    artifacts = sorted(p for p in checkpoint.rglob('*') if p.is_file() and p.suffix in {'.safetensors', '.json', '.pt', '.bin'})
    for split in ['validation', 'test']:
        artifacts += verify_baseline('ragognizer', rag,
            HERE / f'baseline_audit/ragognizer_{split}_predictions_v1', split,
            selected['thresholds'], selected['pooling'], checkpoint)
    add('RAGognizer-retrained-NF4', 'baseline',
        HERE / 'baseline_audit/ragognizer_validation_predictions_v1/predictions.jsonl',
        HERE / 'baseline_audit/ragognizer_test_predictions_v1/predictions.jsonl',
        selected['thresholds'], artifacts + [rag / 'SELECTED.json', rag / 'RUN_BINDING.json'],
        pooling=selected['pooling'])

    for capacity in ['base', 'large']:
        run = HERE / f'model_design/modernbert_{capacity}_v1'
        add('Context-' + capacity, 'control', run / 'best_validation_predictions.jsonl',
            HERE / f'model_design/modernbert_{capacity}_test_v1/predictions.jsonl',
            read(run / 'complete.json')['best']['thresholds'],
            [run / 'best.pt', run / 'prepared.json'] + semantic_receipts[capacity])
    add('Whitebox', 'control', wb / 'best_validation_token_predictions.jsonl',
        wb / 'token_predictions_test.jsonl', read(wb / 'best_validation_metrics.json')['evaluation']['thresholds'],
        [wb / p for p in ['best.pt', 'preprocessor.pkl', 'RUN.json']] + whitebox_receipts)

    for kind in ['token', 'window']:
        run = HERE / f'fusion/{kind}_v1'
        selected = read(run / 'SELECTED.json')
        receipts = verify_fusion(run, sem_test, wb / 'token_predictions_test.jsonl', kind)
        add('Fusion-' + kind, 'candidate' if 'Fusion-' + kind == primary['name'] else 'control',
            run / 'best_validation_predictions.jsonl', run / 'test_predictions.jsonl',
            selected['thresholds'], shared_branch_artifacts + receipts + [run / 'best.pkl', run / 'SELECTED.json', HERE / 'fusion/OOF_CONFIG.json'],
            windows=kind == 'window')

    nonlinear_receipts = verify_fusion(nonlinear, sem_test, wb / 'token_predictions_test.jsonl', 'window')
    add('Fusion-window-HGB', 'candidate' if primary['name'] == 'Fusion-window-HGB' else 'control',
        nonlinear / 'best_validation_predictions.jsonl', nonlinear / 'test_predictions.jsonl',
        nonlinear_selection['thresholds'], shared_branch_artifacts + nonlinear_receipts +
        [nonlinear / 'best.pkl', nonlinear / 'SELECTED.json', HERE / 'fusion/OOF_CONFIG.json'], windows=True)

    # These are the two controls already specified and evaluated during fusion training.
    a, b = predictions(sem_test), predictions(wb / 'token_predictions_test.jsonl')
    assert a.keys() == b.keys()
    geometry = {r['response_id']: r for r in metrics.readl(HERE / 'data/test/geometry.jsonl')}
    assert a.keys() == geometry.keys()
    for kind in ['token', 'window']:
        run = HERE / f'fusion/{kind}_v1'
        if kind == 'token':
            score = {rid: (a[rid] + b[rid]) * .5 for rid in a}
        else:
            score = {rid: (metrics.pool_windows(a[rid], geometry[rid]['lexical_mask']) +
                           metrics.pool_windows(b[rid], geometry[rid]['lexical_mask'])) * .5 for rid in a}
        assert all(np.isfinite(v).all() for v in score.values())
        target = run / 'fixed_average_test.jsonl'
        write_predictions(target, score, kind)
        receipt_path = target.with_suffix('.receipt.json')
        save(receipt_path, {'granularity': kind, 'semantic_sha256': sha(sem_test),
             'whitebox_sha256': sha(wb / 'token_predictions_test.jsonl'),
             'predictions_sha256': sha(target), 'test_labels_read': False})
        add('Average-' + kind, 'control', run / 'fixed_average_validation.jsonl', target,
            read(run / 'fixed_average.json')['thresholds'], shared_branch_artifacts + [receipt_path, run / 'fixed_average.json'],
            windows=kind == 'window')

    # Predeclared prevalence reference; never participates in candidate selection.
    constant_dir = HERE / 'prediction_pipeline/always_risk'
    constant_dir.mkdir(parents=True, exist_ok=True)
    constant_receipt = {'mechanism': 'Every original-response BPE token receives risk score 1.0; no fitted model',
                        'role': 'control', 'used_for_model_selection': False, 'test_labels_read': False,
                        'threshold_source': 'shared validation no-hint threshold selection', 'partitions': {}}
    for partition in ['validation', 'test']:
        geometry_path = HERE / 'data' / partition / 'geometry.jsonl'
        values = constant_risk_scores(metrics.readl(geometry_path))
        target = constant_dir / f'{partition}.jsonl'
        write_predictions(target, values)
        constant_receipt['partitions'][partition] = {
            'geometry_sha256': sha(geometry_path), 'predictions_sha256': sha(target), 'responses': len(values)}
        if partition == 'validation':
            constant_thresholds = metrics.evaluate_partition('validation', values)['thresholds']
    assert constant_thresholds == {'answer': 1.0, 'window': 1.0}
    constant_receipt['thresholds'] = constant_thresholds
    receipt_path = constant_dir / 'RECEIPT.json'
    save(receipt_path, constant_receipt)
    add('Always-risk', 'control', constant_dir / 'validation.jsonl', constant_dir / 'test.jsonl',
        constant_thresholds, [receipt_path])

    config = {'methods': methods, 'primary_candidate': primary['name'],
              'candidate_selection': 'shared validation key across linear token/window and bounded nonlinear window trial; exact ties prefer linear, then window',
              'candidate_validation_keys': learned_choices,
              'scope': 'same source-group train/validation/test split; internal split of official train',
              'test_labels_read': False}
    target = HERE / 'FINAL_COMPARISON.json'
    save(target, config)
    print(json.dumps({'config': str(target), 'methods': len(methods),
                      'primary_candidate': config['primary_candidate'], 'test_labels_read': False}))


if __name__ == '__main__':
    main()
