"""Freeze every prediction and threshold before a single common test release."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return metrics.readl(path)


def read_scores(entry, partition):
    rr = rows(entry[f'{partition}_predictions'])
    assert len(rr) == len({r['response_id'] for r in rr})
    return {r['response_id']: np.asarray(r[entry['score_field']], np.float64) for r in rr}


def freeze(args):
    assert not (HERE / 'TEST_RELEASE.json').exists(), 'Test already released'
    config = load(args.config)
    expected = {r['response_id']: r for r in rows(HERE / 'data/test/geometry.jsonl')}
    assert config['methods'] and len({e['name'] for e in config['methods']}) == len(config['methods'])
    manifest = {'config_sha256': sha(args.config), 'data_manifest_sha256': sha(HERE / 'data/MANIFEST.json'),
                'created_at_unix': time.time(), 'test_labels_read': False, 'methods': []}
    data_manifest = load(HERE / 'data/MANIFEST.json')
    bound_data = ['test/geometry.jsonl', 'test/inputs.jsonl', 'split_assignments.jsonl',
                  'validation/answers.jsonl', 'validation/sequences.jsonl']
    manifest['evaluation_files_sha256'] = {str(HERE / p): sha(HERE / p)
                                          for p in ['metrics.py', 'evaluate_final.py']}
    for name in bound_data:
        path = HERE / 'data' / name
        digest = sha(path)
        assert digest == data_manifest['files'][name]['sha256'], name
        manifest['evaluation_files_sha256'][str(path)] = digest
    for entry in config['methods']:
        score = read_scores(entry, 'test')
        assert score.keys() == expected.keys(), (entry['name'], 'IDs')
        for rid, p in score.items():
            assert len(p) and np.isfinite(p).all()
            seq = expected[rid]
            count = (len(metrics.pool_windows(np.zeros(len(seq['lexical_mask'])), seq['lexical_mask']))
                     if entry['scores_are_windows'] else len(seq['lexical_mask']))
            assert p.shape == (count,), (entry['name'], rid)
        validation = metrics.evaluate_partition('validation', read_scores(entry, 'validation'),
                    pooling=entry['pooling'], scores_are_windows=entry['scores_are_windows'])
        assert validation['thresholds'] == entry['thresholds'], (entry['name'], 'threshold provenance')
        bound = dict(entry)
        bound['test_predictions_sha256'] = sha(entry['test_predictions'])
        bound['validation_predictions_sha256'] = sha(entry['validation_predictions'])
        bound['artifact_hashes'] = {p: sha(p) for p in entry['model_artifacts']}
        bound['validation'] = validation
        manifest['methods'].append(bound)
    save(HERE / 'FROZEN_PREDICTIONS.json', manifest)
    print(json.dumps({'frozen': True, 'methods': [r['name'] for r in manifest['methods']], 'test_answers': len(expected)}))


def confusion(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    return np.array([(y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()], np.int64)


def f1_counts(x):
    return 2 * x[..., 0] / np.maximum(2 * x[..., 0] + x[..., 1] + x[..., 2], 1)


def evaluate(args):
    frozen_path = HERE / 'FROZEN_PREDICTIONS.json'
    frozen = load(frozen_path)
    assert frozen['data_manifest_sha256'] == sha(HERE / 'data/MANIFEST.json')
    for path, digest in frozen['evaluation_files_sha256'].items(): assert sha(path) == digest
    for entry in frozen['methods']:
        assert sha(entry['test_predictions']) == entry['test_predictions_sha256']
        assert sha(entry['validation_predictions']) == entry['validation_predictions_sha256']
        for path, digest in entry['artifact_hashes'].items(): assert sha(path) == digest
    release = HERE / 'TEST_RELEASE.json'
    if release.exists():
        assert load(release)['frozen_predictions_sha256'] == sha(frozen_path)
    else:
        assert args.release, 'Explicit --release required for first evaluation'
        save(release, {'frozen_predictions_sha256': sha(frozen_path), 'released_at_unix': time.time(),
                       'policy': 'all model checkpoints, predictions and thresholds frozen before labels opened'})
    label_path = HERE / 'data/test/sealed/labels.jsonl'
    data_manifest = load(HERE / 'data/MANIFEST.json')
    assert sha(label_path) == data_manifest['files']['test/sealed/labels.jsonl']['sha256']
    label_rows = rows(label_path)
    labels = {r['response_id']: r for r in label_rows}
    assert len(labels) == len(label_rows), 'Duplicate test label ID'
    geometry = {r['response_id']: r for r in rows(HERE / 'data/test/geometry.jsonl')}
    members = {r['response_id']: r for r in rows(HERE / 'data/split_assignments.jsonl') if r['split'] == 'test'}
    assert labels.keys() == geometry.keys() == members.keys(), 'Test labels, geometry and membership differ'
    primary = [rid for rid, r in members.items() if not r['unanswerability_hint_setting']]
    groups = sorted({members[rid]['group_id'] for rid in primary})
    gi = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, len(groups), size=(2000, len(groups)))
    reports, boots = {}, {}
    for entry in frozen['methods']:
        raw = read_scores(entry, 'test')
        report = metrics.evaluate_partition('test', raw, pooling=entry['pooling'], thresholds=entry['thresholds'],
                   scores_are_windows=entry['scores_are_windows'], allow_test=True)
        group_counts = {level: np.zeros((len(groups), 4), np.int64) for level in ['answer', 'window']}
        for rid in primary:
            p = raw[rid] if entry['scores_are_windows'] else metrics.pool_windows(raw[rid], geometry[rid]['lexical_mask'], entry['pooling'])
            y = [int(v) for v in labels[rid]['window_labels'] if v is not None]
            g = gi[members[rid]['group_id']]
            group_counts['answer'][g] += confusion([labels[rid]['answer_label']], [max(p) >= entry['thresholds']['answer']])
            group_counts['window'][g] += confusion(y, p >= entry['thresholds']['window'])
        bootstrap = {level: f1_counts(v[indices].sum(axis=1)) for level, v in group_counts.items()}
        report['primary_cluster_bootstrap_95ci'] = {level: np.quantile(v, [.025, .975]).tolist() for level, v in bootstrap.items()}
        report['bootstrap_groups'] = len(groups)
        reports[entry['name']], boots[entry['name']] = report, bootstrap
    differences = {}
    for a in frozen['methods']:
        if a['role'] != 'candidate': continue
        for b in frozen['methods']:
            if b['role'] != 'baseline': continue
            differences[a['name'] + ' minus ' + b['name']] = {
                level: {'difference': reports[a['name']]['cohorts']['no_hint'][level]['f1'] - reports[b['name']]['cohorts']['no_hint'][level]['f1'],
                        'paired_95ci': np.quantile(boots[a['name']][level] - boots[b['name']][level], [.025, .975]).tolist()}
                for level in ['answer', 'window']}
    output = {'primary': 'no_hint', 'test_answers_all': len(members), 'test_answers_primary': len(primary),
              'test_groups_primary': len(groups), 'methods': reports, 'paired_f1_differences': differences,
              'frozen_predictions_sha256': sha(frozen_path), 'bootstrap_samples': 2000,
              'scope': 'internal source-group split of known official train; automatic span labels'}
    save(HERE / 'results/TEST_RESULTS.json', output)
    lines = ['# 同域重训测试结果', '',
             '全部方法共用来源分组后的训练、验证和测试划分；测试前固定模型、阈值和预测。', '',
             f'主评测：原始提示无资料不足暗示，{len(primary)} 条回答、{len(groups)} 个来源组。', '',
             '| 方法 | 回答 F1 | 回答精确率 | 回答召回率 | 4词元窗口 F1 | 窗口精确率 | 窗口召回率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in reports.items():
        a, w = r['cohorts']['no_hint']['answer'], r['cohorts']['no_hint']['window']
        lines.append(f"| {name} | {a['f1']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {w['f1']:.4f} | {w['precision']:.4f} | {w['recall']:.4f} |")
    lines.extend(['', '逐方法全部样本结果、AUROC、来源组重采样区间和配对差值见 TEST_RESULTS.json。', '',
                  '本表使用已知官方训练数据的内部重新划分和自动标注，不是新的外部人工测试集。'])
    (HERE / 'results/TEST_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'evaluate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    {'freeze': freeze, 'evaluate': evaluate}[args.stage](args)
