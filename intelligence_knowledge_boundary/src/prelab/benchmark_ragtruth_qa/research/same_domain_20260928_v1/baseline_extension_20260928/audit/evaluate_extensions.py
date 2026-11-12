"""CPU-only, hash-bound common evaluation of six baseline families.

No original experiment artifact is written. New thresholds/adapters are selected
from validation only; test was already released before this extension.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent
HERE = OUT.parent.parent
sys.path.insert(0, str(HERE))
import metrics
from evaluate_final import sha, save, load, read_scores, confusion, f1_counts

EXISTING = ('Lookback-default', 'Lookback-tuned', 'RAGognizer-retrained-NF4', 'Fusion-window-HGB')
PRIMARY = 'Fusion-window-HGB'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def equivalent(a, b):
    """Allow only rounding-level library differences in floating metrics."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= 1e-12
    return a == b


def indexed(path):
    rows = metrics.readl(path)
    result = {r['response_id']: r for r in rows}
    assert len(result) == len(rows), f'duplicate response_id: {path}'
    return result


def geometry(partition):
    return indexed(HERE / 'data' / partition / 'geometry.jsonl')


def bound_files(include_test_labels=False):
    result = {str(HERE / n): sha(HERE / n) for n in
              ['metrics.py', 'evaluate_final.py', 'FROZEN_PREDICTIONS.json', 'TEST_RELEASE.json']}
    names = ['MANIFEST.json', 'split_assignments.jsonl', 'GROUP_AUDIT.json', 'GROUPS.json']
    for p in ['fit', 'validation', 'test']:
        names += [p + '/geometry.jsonl', p + '/inputs.jsonl']
    names += ['validation/answers.jsonl', 'validation/sequences.jsonl']
    if include_test_labels:
        names += ['test/sealed/labels.jsonl']
    manifest = load(HERE / 'data/MANIFEST.json')
    for name in names:
        path = HERE / 'data' / name
        digest = sha(path)
        if name in manifest['files']:
            assert manifest['files'][name]['sha256'] == digest, name
        result[str(path)] = digest
    result[str(OUT / 'evaluate_extensions.py')] = sha(OUT / 'evaluate_extensions.py')
    return result


def data_audit():
    """Independently check membership, source groups, offsets and lexical mask."""
    hashes = bound_files()
    assignments = metrics.readl(HERE / 'data/split_assignments.jsonl')
    by_id = {r['response_id']: r for r in assignments}
    assert len(assignments) == len(by_id) == 1842
    counts, groups_by_split, ids_by_split = {}, {}, {}
    identity_splits = {}
    for row in assignments:
        for key, value in row.items():
            if key.endswith('_sha256') and value:
                identity_splits.setdefault((key, value), set()).add(row['split'])
    collisions = {k: v for k, v in identity_splits.items() if len(v) > 1}
    assert not collisions, f'cross-split identity collisions: {list(collisions)[:3]}'
    for part, expected in [('fit', 1292), ('validation', 271), ('test', 279)]:
        seqs, inputs = geometry(part), indexed(HERE / 'data' / part / 'inputs.jsonl')
        ids = {r['response_id'] for r in assignments if r['split'] == part}
        assert len(ids) == expected and ids == seqs.keys() == inputs.keys()
        groups = {by_id[rid]['group_id'] for rid in ids}
        lexical_count = window_count = 0
        for rid in ids:
            seq, inp = seqs[rid], inputs[rid]
            text = inp['original_response']
            offsets, mask = seq['response_token_offsets'], seq['lexical_mask']
            assert len(offsets) == len(mask) == len(seq['response_token_ids'])
            assert hashlib.sha256(text.encode('utf-8')).hexdigest() == seq['response_sha256']
            previous = 0
            covered = np.zeros(len(text), dtype=bool)
            for i, (a, b) in enumerate(offsets):
                assert 0 <= a <= b <= len(text) and a >= previous, (rid, i, a, b)
                previous = a
                covered[a:b] = True
                assert bool(mask[i]) == any(c.isalnum() for c in text[a:b]), (rid, i, 'lexical mask')
            assert all(covered[i] for i, c in enumerate(text) if c.isalnum()), (rid, 'uncovered lexical char')
            assert seq['full_sequence_token_count'] == len(seq['full_input_ids'])
            assert seq['response_token_ids'] == [seq['full_input_ids'][i] for i in seq['answer_token_positions']]
            lexical_count += sum(mask)
            window_count += len(metrics.pool_windows(np.zeros(len(mask)), mask))
        counts[part] = {'answers': len(ids), 'groups': len(groups), 'lexical_tokens': lexical_count,
                        'eligible_windows': window_count,
                        'no_hint_answers': sum(not by_id[rid]['unanswerability_hint_setting'] for rid in ids)}
        ids_by_split[part], groups_by_split[part] = ids, groups
    for a, b in [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')]:
        assert not ids_by_split[a] & ids_by_split[b]
        assert not groups_by_split[a] & groups_by_split[b]
    assert counts['validation']['no_hint_answers'] == 135
    assert counts['test']['no_hint_answers'] == 155
    return {'status': 'PASS', 'counts': counts, 'files_sha256': hashes,
            'cross_split_identity_collisions': 0, 'cross_split_source_group_collisions': 0,
            'all_offsets_and_lexical_masks_checked': True,
            'scope': 'split/data integrity; does not alone establish absence of pretrained-data overlap'}


def bind_predictions(entry, partition):
    records = indexed(entry[f'{partition}_predictions'])
    seqs = geometry(partition)
    assert records.keys() == seqs.keys(), (entry['name'], partition, 'IDs')
    row_bindings = []
    field = entry['score_field']
    for rid in sorted(records):
        record, seq = records[rid], seqs[rid]
        values = np.asarray(record[field], dtype=np.float64)
        expected = (len(metrics.pool_windows(np.zeros(len(seq['lexical_mask'])), seq['lexical_mask']))
                    if entry['scores_are_windows'] else len(seq['lexical_mask']))
        assert values.shape == (expected,) and expected and np.isfinite(values).all(), (entry['name'], rid)
        # Optional offset/ID metadata is checked, never merely carried through.
        for key in ('response_token_offsets', 'response_token_ids', 'response_sha256',
                    'rendered_input_sha256', 'lexical_mask'):
            if key in record:
                assert record[key] == seq[key], (entry['name'], rid, key)
        row_bindings.append({'response_id': rid, 'score_count': expected,
            'scores_sha256': canonical_sha(values.tolist()),
            'geometry_sha256': canonical_sha(seq)})
    return {'path': entry[f'{partition}_predictions'],
            'file_sha256': sha(entry[f'{partition}_predictions']),
            'response_count': len(records), 'rows_sha256': canonical_sha(row_bindings),
            'rows': row_bindings}


def existing_entries():
    frozen = load(HERE / 'FROZEN_PREDICTIONS.json')
    assert frozen['data_manifest_sha256'] == sha(HERE / 'data/MANIFEST.json')
    assert load(HERE / 'TEST_RELEASE.json')['frozen_predictions_sha256'] == sha(HERE / 'FROZEN_PREDICTIONS.json')
    for path, digest in frozen['evaluation_files_sha256'].items():
        assert sha(path) == digest, path
    entries = []
    for entry in frozen['methods']:
        if entry['name'] not in EXISTING:
            continue
        for part in ('validation', 'test'):
            assert sha(entry[f'{part}_predictions']) == entry[f'{part}_predictions_sha256']
        for path, digest in entry['artifact_hashes'].items():
            assert sha(path) == digest, path
        validation = metrics.evaluate_partition('validation', read_scores(entry, 'validation'),
            pooling=entry['pooling'], scores_are_windows=entry['scores_are_windows'])
        assert validation['thresholds'] == entry['thresholds'], (entry['name'], 'thresholds')
        assert equivalent(validation, entry['validation']), (entry['name'], 'validation re-score')
        result = dict(entry)
        result['role'] = 'candidate' if entry['name'] == PRIMARY else 'baseline'
        result['supplementary_only'] = entry['name'] == 'Lookback-default'
        result['reused_from_original_frozen_run'] = True
        result['prediction_bindings'] = {part: bind_predictions(entry, part) for part in ['validation', 'test']}
        entries.append(result)
    assert len(entries) == len(EXISTING)
    return entries


def freeze_new(config_path):
    config = load(config_path)
    assert len(config['methods']) == 4, 'Six-family comparison requires four completed new methods plus two existing families'
    audit = data_audit()
    existing = existing_entries()
    new_entries = []
    for source in config['methods']:
        entry = dict(source)
        assert entry['name'] not in EXISTING
        entry.setdefault('role', 'baseline')
        entry.setdefault('score_field', 'token_scores')
        entry.setdefault('scores_are_windows', False)
        candidates = entry.get('pooling_candidates', ['mean', 'max'])
        assert candidates and all(p in ['mean', 'max'] for p in candidates)
        validation_scores = read_scores(entry, 'validation')
        alternatives = {p: metrics.evaluate_partition('validation', validation_scores,
            pooling=p, scores_are_windows=entry['scores_are_windows']) for p in candidates}
        selected = max(candidates, key=lambda p: (alternatives[p]['selection_key'], p == 'mean'))
        entry.update(pooling=selected, thresholds=alternatives[selected]['thresholds'],
                     validation=alternatives[selected], adapter_validation_candidates=alternatives)
        entry['prediction_bindings'] = {part: bind_predictions(entry, part) for part in ['validation', 'test']}
        entry['test_predictions_sha256'] = sha(entry['test_predictions'])
        entry['validation_predictions_sha256'] = sha(entry['validation_predictions'])
        assert entry['model_artifacts'], (entry['name'], 'model/adapter provenance required')
        entry['artifact_hashes'] = {str(p): sha(p) for p in entry['model_artifacts']}
        new_entries.append(entry)
    names = [e['name'] for e in existing + new_entries]
    assert len(names) == len(set(names))
    manifest = {'created_at_unix': time.time(), 'configuration_sha256': sha(config_path),
        'primary_candidate': PRIMARY, 'test_previously_released': True,
        'new_freeze_reads_test_labels': False, 'threshold_source': 'validation/no_hint only',
        'data_audit': audit, 'methods': existing + new_entries}
    destination = OUT / 'EXTENSION_FROZEN.json'
    # Replacing a frozen comparison requires explicitly removing its old file.
    assert not destination.exists(), f'Already frozen: {destination}'
    save(destination, manifest)
    print(json.dumps({'frozen': str(destination), 'methods': names}, ensure_ascii=False))


def independent_metrics(labels, scores, threshold):
    y, p = np.asarray(labels, bool), np.asarray(scores, float)
    tp = sum(bool(a) and b >= threshold for a, b in zip(y, p))
    fp = sum(not bool(a) and b >= threshold for a, b in zip(y, p))
    fn, tn = int(y.sum()) - tp, int((~y).sum()) - fp
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
        'f1': 2 * tp / max(2 * tp + fp + fn, 1),
        'auroc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
        'threshold': float(threshold)}


def evaluate(entries, data_checks, manifest_path=None, output_prefix='SIX_BASELINES'):
    # Full-test labels are opened only after all prediction bindings are checked.
    for path, digest in data_checks['files_sha256'].items():
        assert sha(path) == digest, path
    for entry in entries:
        for partition in ['validation', 'test']:
            binding = bind_predictions(entry, partition)
            assert binding == entry['prediction_bindings'][partition], (entry['name'], partition)
        for path, digest in entry['artifact_hashes'].items():
            assert sha(path) == digest, path
    label_path = HERE / 'data/test/sealed/labels.jsonl'
    assert sha(label_path) == load(HERE / 'data/MANIFEST.json')['files']['test/sealed/labels.jsonl']['sha256']
    labels, seqs = indexed(label_path), geometry('test')
    members = {r['response_id']: r for r in metrics.readl(HERE / 'data/split_assignments.jsonl') if r['split'] == 'test'}
    assert labels.keys() == seqs.keys() == members.keys()
    primary_ids = [rid for rid, r in members.items() if not r['unanswerability_hint_setting']]
    groups = sorted({members[rid]['group_id'] for rid in primary_ids})
    group_index = {g: i for i, g in enumerate(groups)}
    assert len(primary_ids) == 155 and len(groups) == 75
    draws = np.random.default_rng(20260928).integers(0, len(groups), size=(2000, len(groups)))
    reports, boots, audit = {}, {}, {}
    previous = load(HERE / 'results/TEST_RESULTS.json')
    for entry in entries:
        name = entry['name']
        raw = read_scores(entry, 'test')
        report = metrics.evaluate_partition('test', raw, pooling=entry['pooling'],
            thresholds=entry['thresholds'], scores_are_windows=entry['scores_are_windows'], allow_test=True)
        windows = {rid: raw[rid] if entry['scores_are_windows'] else
                   metrics.pool_windows(raw[rid], seqs[rid]['lexical_mask'], entry['pooling']) for rid in members}
        group_counts = {level: np.zeros((len(groups), 4), np.int64) for level in ['answer', 'window']}
        for cohort in ['all', 'no_hint']:
            ids = list(members) if cohort == 'all' else primary_ids
            a = independent_metrics([labels[r]['answer_label'] for r in ids],
                                    [max(windows[r]) for r in ids], entry['thresholds']['answer'])
            w = independent_metrics(np.concatenate([[v for v in labels[r]['window_labels'] if v is not None] for r in ids]),
                                    np.concatenate([windows[r] for r in ids]), entry['thresholds']['window'])
            assert a == report['cohorts'][cohort]['answer'], (name, cohort, 'answer metrics')
            assert w == report['cohorts'][cohort]['window'], (name, cohort, 'window metrics')
        for rid in primary_ids:
            g, p = group_index[members[rid]['group_id']], windows[rid]
            y = [int(v) for v in labels[rid]['window_labels'] if v is not None]
            group_counts['answer'][g] += confusion([labels[rid]['answer_label']], [max(p) >= entry['thresholds']['answer']])
            group_counts['window'][g] += confusion(y, p >= entry['thresholds']['window'])
        bootstrap = {level: f1_counts(v[draws].sum(axis=1)) for level, v in group_counts.items()}
        report['primary_cluster_bootstrap_95ci'] = {level: np.quantile(v, [.025, .975]).tolist() for level, v in bootstrap.items()}
        report['bootstrap_groups'] = len(groups)
        if name in EXISTING:
            assert equivalent(report, previous['methods'][name]), (name, 'old test scores/CI changed')
        reports[name], boots[name] = report, bootstrap
        audit[name] = {'status': 'PASS', 'independent_counts_and_metrics_match': True,
            'validation_thresholds_reproduced': True, 'all_prediction_rows_geometry_hash_bound': True,
            'old_frozen_result_reproduced_with_float_tolerance_1e_12': name in EXISTING,
            'test_predictions_sha256': sha(entry['test_predictions']),
            'validation_predictions_sha256': sha(entry['validation_predictions']),
            'native_granularity': entry.get('native_granularity', 'original frozen record')}
    differences = {}
    for entry in entries:
        if entry['role'] != 'baseline':
            continue
        name = entry['name']
        differences[PRIMARY + ' minus ' + name] = {level: {
            'difference': reports[PRIMARY]['cohorts']['no_hint'][level]['f1'] - reports[name]['cohorts']['no_hint'][level]['f1'],
            'paired_95ci': np.quantile(boots[PRIMARY][level] - boots[name][level], [.025, .975]).tolist()}
            for level in ['answer', 'window']}
    result = {'primary': 'no_hint', 'primary_candidate': PRIMARY, 'test_answers_all': 279,
        'test_answers_primary': 155, 'test_groups_primary': 75, 'methods': reports,
        'paired_f1_differences': differences, 'bootstrap_samples': 2000, 'bootstrap_seed': 20260928,
        'test_previously_released': True,
        'scope': 'fixed-protocol extension on the already-released internal source-group test; automatic span labels',
        'manifest_sha256': sha(manifest_path) if manifest_path else None}
    save(OUT / (output_prefix + '_RESULTS.json'), result)
    save(OUT / (output_prefix + '_AUDIT.json'), {'status': 'PASS', 'data': data_checks, 'methods': audit,
        'test_labels_sha256': sha(label_path), 'primary_candidate_unchanged': PRIMARY,
        'result_sha256': sha(OUT / (output_prefix + '_RESULTS.json'))})
    lines = ['# 共同评测结果', '', '主测试：无暗示 155 条 / 75 来源组；4 BPE，步长 1。', '',
        '| 方法 | 回答 F1 | 回答 P | 回答 R | 回答 AUROC | 窗口 F1 | 窗口 P | 窗口 R | 窗口 AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for entry in entries:
        name = entry['name']
        a, w = (reports[name]['cohorts']['no_hint'][lev] for lev in ['answer', 'window'])
        values = [a[k] for k in ['f1', 'precision', 'recall', 'auroc']] + [w[k] for k in ['f1', 'precision', 'recall', 'auroc']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '全量 279 条结果与来源组配对重采样区间见 JSON。Lookback-default 仅补充，不计作独立第七种方法。',
              '本轮是已解封测试集上的固定协议补充比较，新增方法的阈值和窗口适配仅由验证集确定。']
    (OUT / (output_prefix + '_RESULTS.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit-existing', 'freeze-new', 'evaluate'])
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'audit-existing':
        check = data_audit()
        entries = existing_entries()
        save(OUT / 'EXISTING_REUSE.json', {'data_audit': check, 'methods': entries})
        evaluate(entries, check, output_prefix='EXISTING')
    elif args.stage == 'freeze-new':
        assert args.config is not None
        freeze_new(args.config)
    else:
        path = OUT / 'EXTENSION_FROZEN.json'
        manifest = load(path)
        evaluate(manifest['methods'], manifest['data_audit'], path)


if __name__ == '__main__':
    main()
