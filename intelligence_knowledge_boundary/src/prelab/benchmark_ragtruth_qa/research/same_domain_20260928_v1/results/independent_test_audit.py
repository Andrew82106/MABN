"""Independent frozen-test audit; target access requires completed release gate.

No evaluation/training implementation is imported. Only this audit JSON is written.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

RUN = Path(__file__).resolve().parents[1]
TARGET_ACCESS_ALLOWED = False
EXPECTED_METHODS = {'Lookback-default', 'Lookback-tuned', 'RAGognizer-retrained-NF4',
                    'Context-base', 'Context-large', 'Whitebox', 'Fusion-token',
                    'Fusion-window', 'Fusion-window-HGB', 'Average-token',
                    'Average-window', 'Always-risk'}


def guard(path):
    if '/data/test/sealed/' in Path(path).resolve().as_posix().lower():
        assert TARGET_ACCESS_ALLOWED, 'Test targets cannot be opened before the completed release gate'


def sha(path):
    guard(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    guard(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def records(path):
    guard(path)
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line]


def unique(rows):
    indexed = {row['response_id']: row for row in rows}
    assert len(indexed) == len(rows), 'Duplicate response ID'
    return indexed


def gated_inputs(wait):
    global TARGET_ACCESS_ALLOWED
    completion_path = RUN / 'final_evaluation_pipeline/COMPLETE.json'
    announced = False
    while not completion_path.exists():
        status_path = RUN / 'final_evaluation_pipeline/STATUS.json'
        if status_path.exists():
            assert read(status_path)['stage'] != 'failed', 'Final evaluation pipeline failed; labels remain unopened'
        if not wait:
            raise RuntimeError('Final evaluation is not complete; labels remain unopened')
        if not announced:
            print('WAITING_FOR_COMPLETED_FINAL_EVALUATION; no test targets opened', flush=True)
            announced = True
        time.sleep(50)
    completion = read(completion_path)
    assert completion['status'] == 'complete' and completion['all_children_exited']
    release = read(RUN / 'TEST_RELEASE.json')
    frozen = read(RUN / 'FROZEN_PREDICTIONS.json')
    frozen_hash = sha(RUN / 'FROZEN_PREDICTIONS.json')
    assert completion['frozen_predictions_sha256'] == release['frozen_predictions_sha256'] == frozen_hash
    assert frozen['created_at_unix'] <= release['released_at_unix'] <= completion['finished_at']
    manifest = read(RUN / 'data/MANIFEST.json')
    assert frozen['data_manifest_sha256'] == sha(RUN / 'data/MANIFEST.json')
    assert frozen['config_sha256'] == sha(RUN / 'FINAL_COMPARISON.json')
    for path, digest in frozen['evaluation_files_sha256'].items():
        assert sha(path) == digest, path
    # This is the sole transition that authorizes sealed-file reads, after the gate.
    TARGET_ACCESS_ALLOWED = True
    label_path = RUN / 'data/test/sealed/labels.jsonl'
    assert sha(label_path) == manifest['files']['test/sealed/labels.jsonl']['sha256']
    return completion, release, frozen, manifest, unique(records(label_path))


def intervals(lexical):
    n = len(lexical)
    width = min(4, n)
    return [(start, start + width) for start in range(n - width + 1)
            if any(lexical[start:start + width])]


def prepare_test_targets(labels, geometry):
    targets = {}
    for rid, row in labels.items():
        g = geometry[rid]
        assert row['response_sha256'] == g['response_sha256']
        lexical, risk = g['lexical_mask'], row['risk_mask']
        assert len(lexical) == len(risk) and all(not r or l for r, l in zip(risk, lexical))
        spans = intervals(lexical)
        derived = [any(bool(risk[j]) and bool(lexical[j]) for j in range(a, b)) for a, b in spans]
        assert derived == [bool(y) for y in row['window_labels'] if y is not None]
        assert bool(row['answer_label']) == any(h.get('valid', False) for h in row['released_hallucinations'])
        targets[rid] = {'answer': bool(row['answer_label']), 'windows': derived,
                        'lexical': lexical, 'intervals': spans}
    return targets


def window_scores(raw, target, entry):
    values = [float(value) for value in raw]
    assert values and np.isfinite(values).all()
    if entry['scores_are_windows']:
        assert len(values) == len(target['intervals'])
        return values
    assert len(values) == len(target['lexical'])
    output = []
    for start, end in target['intervals']:
        if entry['pooling'] == 'mean':
            # np.mean matches the defined floating-point native mean adapter.
            score = float(np.mean(values[start:end], dtype=np.float64))
        else:
            assert entry['pooling'] == 'max'
            score = max(values[j] for j in range(start, end) if target['lexical'][j])
        output.append(score)
    return output


def counts(labels, predicted):
    result = [0, 0, 0, 0]  # TP, FP, FN, TN
    assert len(labels) == len(predicted)
    for truth, decision in zip(labels, predicted):
        result[0 if truth and decision else 1 if decision else 2 if truth else 3] += 1
    return np.asarray(result, np.int64)


def summarize(c):
    tp, fp, fn, tn = map(int, c)
    return {'n': tp + fp + fn + tn, 'positive': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def f1_matrix(c):
    numerator = 2 * c[:, 0]
    denominator = numerator + c[:, 1] + c[:, 2]
    return np.divide(numerator, denominator, out=np.zeros(len(c), float), where=denominator != 0)


def close(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-12), (a, b)


def evaluate_fixed(entry, prediction_rows, targets, members):
    predictions = unique(prediction_rows)
    assert predictions.keys() == targets.keys() == members.keys()
    per_answer = {}
    for rid, target in targets.items():
        p = window_scores(predictions[rid][entry['score_field']], target, entry)
        assert len(p) == len(target['windows']) and p
        per_answer[rid] = {
            'answer': counts([target['answer']], [max(p) >= entry['thresholds']['answer']]),
            'window': counts(target['windows'], [score >= entry['thresholds']['window'] for score in p])}
    cohorts = {}
    for name in ['all', 'no_hint']:
        ids = [rid for rid in members if name == 'all' or not members[rid]['unanswerability_hint_setting']]
        cohorts[name] = {level: summarize(sum((per_answer[rid][level] for rid in ids), np.zeros(4, np.int64)))
                        for level in ['answer', 'window']}
        cohorts[name]['groups'] = len({members[rid]['group_id'] for rid in ids})
    return cohorts, per_answer


def audit(wait):
    started = time.perf_counter()
    completion, release, frozen, manifest, labels = gated_inputs(wait)
    config = read(RUN / 'FINAL_COMPARISON.json')
    result_path = RUN / 'results/TEST_RESULTS.json'
    published = read(result_path)
    assert published['frozen_predictions_sha256'] == release['frozen_predictions_sha256']
    methods = {entry['name']: entry for entry in frozen['methods']}
    assert len(methods) == len(frozen['methods']) == 12 and set(methods) == EXPECTED_METHODS == set(published['methods'])
    assignment_rows = records(RUN / 'data/split_assignments.jsonl')
    assignments = unique(assignment_rows)
    split_groups = {split: {r['group_id'] for r in assignment_rows if r['split'] == split}
                    for split in ['fit', 'validation', 'test']}
    assert all(split_groups[a].isdisjoint(split_groups[b]) for a, b in
               [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')])
    members = {rid: row for rid, row in assignments.items() if row['split'] == 'test'}
    geometry = unique(records(RUN / 'data/test/geometry.jsonl'))
    assert members.keys() == geometry.keys() == labels.keys() and len(members) == 279
    primary_ids = [rid for rid, row in members.items() if not row['unanswerability_hint_setting']]
    groups = sorted({members[rid]['group_id'] for rid in primary_ids})
    assert len(primary_ids) == 155 and len(groups) == 75 and len(split_groups['test']) == 133
    assert published['primary'] == 'no_hint' and published['test_answers_all'] == 279
    assert published['test_answers_primary'] == 155 and published['test_groups_primary'] == 75
    targets = prepare_test_targets(labels, geometry)
    validation_answers = unique(records(RUN / 'data/validation/answers.jsonl'))
    validation_sequences = unique(records(RUN / 'data/validation/sequences.jsonl'))
    validation_targets = {}
    for rid, row in validation_sequences.items():
        ranges = intervals(row['lexical_mask'])
        validation_targets[rid] = {'answer': bool(validation_answers[rid]['answer_risk']),
            'windows': [any(row['risk_mask'][j] and row['lexical_mask'][j] for j in range(a, b)) for a, b in ranges],
            'intervals': ranges, 'lexical': row['lexical_mask']}
    all_reports, per_response, val_keys = {}, {}, {}
    verified_artifacts = {}
    for name, entry in methods.items():
        for partition in ['test', 'validation']:
            assert sha(entry[f'{partition}_predictions']) == entry[f'{partition}_predictions_sha256']
        for path, digest in entry['artifact_hashes'].items():
            if path not in verified_artifacts:
                verified_artifacts[path] = sha(path)
            assert verified_artifacts[path] == digest
        actual, rows = evaluate_fixed(entry, records(entry['test_predictions']), targets, members)
        expected = published['methods'][name]
        assert expected['thresholds'] == entry['thresholds'] and expected['pooling'] == entry['pooling']
        for cohort in ['all', 'no_hint']:
            assert actual[cohort]['groups'] == expected['cohorts'][cohort]['groups']
            for level in ['answer', 'window']:
                for key, value in actual[cohort][level].items():
                    close(value, expected['cohorts'][cohort][level][key])
        all_reports[name], per_response[name] = actual, rows
        if name in ['Fusion-token', 'Fusion-window', 'Fusion-window-HGB']:
            val, _ = evaluate_fixed(entry, records(entry['validation_predictions']), validation_targets, validation_answers)
            a, w = val['no_hint']['answer'], val['no_hint']['window']
            val_keys[name] = [min(a['f1'], w['f1']), w['f1'], w['precision']]
            close(val_keys[name], entry['validation']['selection_key'])
    primary = max(val_keys, key=lambda name: tuple(val_keys[name]) +
                  (int(name != 'Fusion-window-HGB'), int(name != 'Fusion-token')))
    assert primary == config['primary_candidate'] == completion['primary_candidate'] == 'Fusion-window-HGB'
    assert [name for name, row in methods.items() if row['role'] == 'candidate'] == [primary]
    # Independent cluster resampling: group multiplicities times confusion counts.
    draws = np.random.default_rng(20260928).integers(0, len(groups), (2000, len(groups)))
    multiplicities = np.array([np.bincount(draw, minlength=len(groups)) for draw in draws])
    boots, cis = {}, {}
    for name in methods:
        boots[name], cis[name] = {}, {}
        for level in ['answer', 'window']:
            grouped = np.array([sum((per_response[name][rid][level] for rid in primary_ids
                                    if members[rid]['group_id'] == group), np.zeros(4, np.int64)) for group in groups])
            boots[name][level] = f1_matrix(multiplicities @ grouped)
            cis[name][level] = np.quantile(boots[name][level], [0.025, 0.975]).tolist()
            close(cis[name][level], published['methods'][name]['primary_cluster_bootstrap_95ci'][level])
        assert published['methods'][name]['bootstrap_groups'] == len(groups)
    baseline_names = [name for name, row in methods.items() if row['role'] == 'baseline']
    paired = {}
    for baseline in baseline_names:
        pair_name = primary + ' minus ' + baseline
        paired[pair_name] = {}
        for level in ['answer', 'window']:
            difference = all_reports[primary]['no_hint'][level]['f1'] - all_reports[baseline]['no_hint'][level]['f1']
            ci = np.quantile(boots[primary][level] - boots[baseline][level], [0.025, 0.975]).tolist()
            close(difference, published['paired_f1_differences'][pair_name][level]['difference'])
            close(ci, published['paired_f1_differences'][pair_name][level]['paired_95ci'])
            paired[pair_name][level] = {'difference': difference, 'paired_95ci': ci}
    strongest = {}
    leaders = {}
    for level in ['answer', 'window']:
        baseline = max(baseline_names, key=lambda name: all_reports[name]['no_hint'][level]['f1'])
        strongest[level] = {'baseline': baseline, **paired[primary + ' minus ' + baseline][level]}
        best_score = max(all_reports[name]['no_hint'][level]['f1'] for name in methods)
        leaders[level] = {'methods': [name for name in methods if all_reports[name]['no_hint'][level]['f1'] == best_score],
                          'f1': best_score, 'descriptive_only_primary_unchanged': True}
    output = {'status': 'PASS', 'method_count': len(methods), 'primary': primary,
        'primary_selected_only_by_independently_recomputed_validation_key': val_keys,
        'all_test_answers': 279, 'primary_answers': 155, 'all_test_groups': 133, 'primary_groups': 75,
        'all_no_hint_confusions_precision_recall_f1_match': True,
        'all_cluster_intervals_and_baseline_paired_intervals_match': True,
        'bootstrap_seed': 20260928, 'bootstrap_samples': 2000,
        'counts_and_f1': all_reports, 'cluster_95ci': cis, 'paired_differences': paired,
        'primary_minus_strongest_observed_baseline_each_level': strongest,
        'observed_test_leaders_each_level': leaders,
        'labels_opened_only_after_completed_release_gate': True,
        'sealed_label_sha256_matches_frozen_manifest': True,
        'frozen_predictions_sha256': release['frozen_predictions_sha256'],
        'test_results_sha256': sha(result_path), 'audit_script_sha256': sha(__file__),
        'audit_seconds_including_gate_wait': time.perf_counter() - started,
        'scope': 'known official-train internal split; automatic span labels; no test-based reselection'}
    destination = RUN / 'results/INDEPENDENT_TEST_AUDIT.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': output['status'], 'primary': primary, 'methods': len(methods),
        'primary_vs_strongest_baseline': strongest, 'descriptive_test_leaders': leaders,
        'audit': str(destination)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Check the completion gate every 50 seconds')
    audit(parser.parse_args().wait)
