"""Independent R24 audit: log-domain recursion, thresholds and OOF accounting.

Uses frozen development scores and labels only. No source model imports, GPU,
model fitting, new grid, or changes to experimental artifacts.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import itertools
import json
import pickle
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OLD = ROOT.parent / 'round22_evidence_verification/results'


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def readl(path):
    return [json.loads(line) for line in Path(path).read_text('utf-8').splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_fb(scores, stay, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    if stay == .5 and temperature == 1:
        return scores.copy()  # Explicit raw-identity convention in the protocol.
    q = np.clip(scores, 1e-12, 1 - 1e-12)
    z = (np.log(q) - np.log1p(-q)) / temperature
    emission = np.column_stack((-np.logaddexp(0, z), -np.logaddexp(0, -z)))
    same, other = np.log(stay), np.log1p(-stay)
    forward = np.empty_like(emission); backward = np.empty_like(emission)
    forward[0] = np.log(.5) + emission[0]
    for i in range(1, len(scores)):
        a, b = forward[i - 1]
        forward[i] = emission[i] + [np.logaddexp(a + same, b + other),
                                     np.logaddexp(a + other, b + same)]
    backward[-1] = 0
    for i in range(len(scores) - 2, -1, -1):
        a, b = emission[i + 1] + backward[i + 1]
        backward[i] = [np.logaddexp(same + a, other + b), np.logaddexp(other + a, same + b)]
    marginal = forward + backward
    return np.exp(marginal[:, 1] - np.logaddexp(marginal[:, 0], marginal[:, 1]))


def choose_threshold(y, scores):
    y = np.asarray(y, dtype=int); scores = np.asarray(scores, dtype=np.float64)
    assert set(y) == {0, 1} and np.isfinite(scores).all()
    order = np.argsort(-scores, kind='stable'); ss, yy = scores[order], y[order]
    endpoints = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.cumsum(yy)[endpoints]; count = endpoints + 1; positives = int(y.sum())
    f = 2 * tp / (count + positives); precision = tp / count
    candidates = [(0., 0., float(np.nextafter(ss[0], np.inf)))]
    candidates.extend((float(a), float(b), float(c)) for a, b, c in zip(f, precision, ss[endpoints]))
    candidates.append((float(f[-1]), float(precision[-1]), float(np.nextafter(ss[-1], -np.inf))))
    f1, p, threshold = max(candidates)
    return {'threshold': threshold, 'validation_f1': f1, 'validation_precision': p,
            'tokens': len(y), 'scorable_tokens': len(y), 'risk_tokens': positives}


def confusion(y, predicted):
    y = np.asarray(y, int); predicted = np.asarray(predicted, bool)
    tp = int(((y == 1) & predicted).sum()); fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum()); tn = int(((y == 0) & ~predicted).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def same_threshold(actual, saved):
    for key in actual:
        if key == 'threshold':
            assert abs(actual[key] - saved[key]) < 5e-12, (key, actual, saved)
        else:
            assert actual[key] == saved[key], (key, actual, saved)


def main():
    started = time.perf_counter()
    config = read(ROOT / 'protocol.json'); complete = read(OUT / 'complete.json')
    freeze = read(OUT / 'calibration_freeze.json'); before = read(OUT / 'calibration_started.json')
    testing = read(OUT / 'test_started.json'); old_complete = read(OLD / 'complete22.json')
    assert before['utc'] <= freeze['utc'] < testing['utc'] <= complete['utc']
    assert before['snapshot'] == freeze['snapshot']
    assert freeze['snapshot']['code_sha256'] == sha(ROOT / 'src/run24.py')
    assert freeze['snapshot']['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert freeze['snapshot']['source_complete_sha256'] == sha(OLD / 'complete22.json')
    assert testing['freeze_sha256'] == sha(OUT / 'calibration_freeze.json')
    for manifest, folder in ((freeze, OUT), (complete, OUT), (old_complete, OLD)):
        for name, expected in manifest['files_sha256'].items():
            assert sha(folder / name) == expected
    bases = config['base_methods']; methods = config['methods']
    assert config['future_windows_used'] and not config['online_detector']
    assert config['stay'] == [.5, .8, .95, .99] and config['temperature'] == [.5, 1., 2.]
    windows = readl(OLD / 'candidate_windows22.jsonl'); answers = readl(OLD / 'answer_index22.jsonl')
    woof = readl(OUT / 'window_scores_oof.jsonl'); aoof = readl(OUT / 'answer_scores_oof.jsonl')
    summary = read(OUT / 'summary.json'); old_summary = read(OLD / 'summary22.json')
    assert len(windows) == len(woof) == 12222 and len(answers) == len(aoof) == 602
    assert len({a['row_id'] for a in answers}) == 602
    item_map = {a['item_id']: a for a in answers}; window_map = {w['window_key']: i for i, w in enumerate(windows)}
    assert len(item_map) == 602 and len(window_map) == len(windows)
    assert all(a['split'] == 'train' for a in answers)
    by_answer = defaultdict(list)
    for index, window in enumerate(windows):
        assert len(window['item_ids']) == 1
        iid = window['item_ids'][0]; item = item_map[iid]
        assert all(window[key] == item[key] for key in ('row_id', 'question_id', 'group_id', 'split'))
        assert window['raw_token_indices'] == list(range(window['raw_token_indices'][0], window['raw_token_indices'][0] + window['actual_width']))
        by_answer[iid].append(index)
    seqs = {}
    gaps = Counter()
    for iid, indices in by_answer.items():
        ix = np.asarray(sorted(indices, key=lambda i: windows[i]['raw_token_indices'][0]))
        starts = [windows[i]['raw_token_indices'][0] for i in ix]
        assert len(set(starts)) == len(starts)
        gaps.update(int(gap) for gap in np.diff(starts)); seqs[iid] = ix
    assert sorted(np.concatenate(list(seqs.values())).tolist()) == list(range(len(windows)))
    assert set(seqs) == set(item_map)
    evaluable_window = np.asarray([w['main_eligible'] for w in windows], bool)
    y = np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows], int)
    all_groups = {a['group_id'] for a in answers}; evaluation_groups = []
    folds = []; frozen_scores = {}; frozen_cal = {}

    for fold in range(5):
        calibration = read(OUT / f'fold_{fold}_calibration.json')
        source = pickle.loads((OLD / f'fold_{fold}_frozen22.pkl').read_bytes())
        groups = calibration['groups']
        for key in groups:
            assert groups[key] == source[key]
        fit, cal, ev = (set(groups[key]) for key in ('fit_groups', 'calibration_groups', 'evaluation_groups'))
        assert not (fit & cal or fit & ev or cal & ev) and fit | cal | ev == all_groups
        evaluation_groups.extend(ev)
        with np.load(OUT / f'fold_{fold}_scores.npz', allow_pickle=False) as z:
            scores = {key: z[key].copy() for key in z.files}
        with np.load(OLD / f'fold_{fold}_scores22.npz', allow_pickle=False) as z:
            for method in bases:
                assert np.array_equal(scores[method], z[method])
                assert calibration['thresholds'][method] == source['thresholds'][method]
                assert summary['methods'][method] == old_summary['methods'][method]
        assert all(score.shape == (len(windows),) and np.isfinite(score).all() for score in scores.values())
        cal_indices = np.flatnonzero([w['group_id'] in cal and w['main_eligible'] for w in windows])
        cal_answers = [a for a in answers if a['group_id'] in cal and a['main_eligible']]
        cal_seqs = {iid: ix for iid, ix in seqs.items() if item_map[iid]['group_id'] in cal}

        def thresholds(value):
            return {'window': choose_threshold(y[cal_indices], value[cal_indices]),
                    'answer': choose_threshold([a['gold'] for a in cal_answers],
                                               [value[seqs[a['item_id']]].max() for a in cal_answers])}

        details = []
        for method in bases:
            selected = calibration['choices'][method + '_smooth']
            independent = np.empty(len(windows))
            for iid, ix in seqs.items():
                independent[ix] = log_fb(scores[method][ix], selected['stay'], selected['temperature'])
            error = float(np.max(np.abs(independent - scores[method + '_smooth'])))
            assert error < 5e-12, (fold, method, error)
            rebuilt_thresholds = thresholds(independent)
            for level in ('window', 'answer'):
                same_threshold(rebuilt_thresholds[level], selected['thresholds'][level])
                assert selected['thresholds'][level] == calibration['thresholds'][method + '_smooth'][level]
            # Recompute only the already-prespecified calibration grid. This
            # verifies the saved choice; no new setting or outer-label choice.
            entries = []
            table = calibration['tables'][method + '_smooth']
            for index, (stay, temp) in enumerate(itertools.product(config['stay'], config['temperature'])):
                candidate = np.full(len(windows), np.nan)
                for iid, ix in cal_seqs.items():
                    candidate[ix] = log_fb(scores[method][ix], stay, temp)
                ts = thresholds(candidate)
                for level in ('window', 'answer'):
                    same_threshold(ts[level], table[index]['thresholds'][level])
                w, a = ts['window'], ts['answer']
                key = [min(w['validation_f1'], a['validation_f1']), w['validation_f1'], w['validation_precision'],
                       int(stay == .5 and temp == 1), -stay, -abs(float(np.log(temp)))]
                assert key == table[index]['key']
                assert stay == table[index]['stay'] and temp == table[index]['temperature']
                entries.append((key, stay, temp))
            chosen = max(entries, key=lambda entry: entry[0])
            assert (chosen[1], chosen[2]) == (selected['stay'], selected['temperature'])
            details.append({'method': method, 'stay': selected['stay'], 'temperature': selected['temperature'],
                            'all_window_log_domain_max_absolute_error': error,
                            'selected_thresholds_recomputed': True, 'calibration_grid_selection_recomputed': True})
        frozen_scores[fold] = scores; frozen_cal[fold] = calibration
        folds.append({'fold': fold, 'groups': {key: len(value) for key, value in groups.items()},
                      'raw_baselines_and_original_thresholds_exact': True, 'methods': details})
        print('AUDIT24_FOLD_PASSED', fold, flush=True)
    assert Counter(evaluation_groups) == Counter({group: 1 for group in all_groups})

    # Scores are verified per answer rather than trusting flattened array order.
    assert {w['window_key'] for w in woof} == set(window_map)
    for row in woof:
        index = window_map[row['window_key']]; original = windows[index]; fold = row['fold']
        assert all(row[key] == value for key, value in original.items())
        assert row['group_id'] in frozen_cal[fold]['groups']['evaluation_groups']
        for method in methods:
            value = float(frozen_scores[fold][method][index])
            assert row['scores'][method] == value
            assert row['predictions'][method] == (value >= frozen_cal[fold]['thresholds'][method]['window']['threshold'])
    assert {a['item_id'] for a in aoof} == set(item_map)
    for row in aoof:
        original = item_map[row['item_id']]; fold = row['fold']; ix = seqs[row['item_id']]
        assert all(row[key] == original[key] for key in ('row_id', 'question_id', 'group_id', 'gold', 'main_eligible'))
        assert row['group_id'] in frozen_cal[fold]['groups']['evaluation_groups']
        for method in methods:
            value = float(frozen_scores[fold][method][ix].max())
            assert row['scores'][method] == value
            assert row['predictions'][method] == (value >= frozen_cal[fold]['thresholds'][method]['answer']['threshold'])
    metrics = {}
    for method in methods:
        metrics[method] = {}
        for unit, rows in (('windows', woof), ('answers', aoof)):
            eligible = [row for row in rows if row['main_eligible']]
            result = confusion([row['gold'] for row in eligible], [row['predictions'][method] for row in eligible])
            for key, value in result.items():
                assert summary['methods'][method][unit][key] == value
            metrics[method][unit] = result
    assert sha(ROOT / 'src/run24.py') == freeze['snapshot']['code_sha256']
    report = {'status': 'passed', 'auditor': 'plan_review', 'audit_code_sha256': sha(Path(__file__)),
        'complete_sha256': sha(OUT / 'complete.json'), 'protocol_sha256': sha(ROOT / 'protocol.json'),
        'read_only_fixed_result_audit': True, 'gpu_used': False, 'new_parameters_or_training': False,
        'original_validation_or_test_read': False, 'source_train_answers': len(answers),
        'source_train_questions': len({a['question_id'] for a in answers}), 'source_train_event_groups': len(all_groups),
        'geometry': {'sequences': len(seqs), 'all_output_defined_windows': len(windows),
                     'evaluable_windows': int(evaluable_window.sum()),
                     'ineligible_windows_preserved_during_smoothing': int((~evaluable_window).sum()),
                     'raw_start_gap_counts': dict(sorted(gaps.items())),
                     'no_answer_or_group_crossing': True, 'gold_independent_sequence_membership': True},
        'folds': folds, 'all_20_selected_posteriors_recomputed_in_log_domain': True,
        'all_240_prespecified_calibration_candidates_recomputed_without_outer_labels': True,
        'threshold_tolerance_absolute': 5e-12, 'calibration_F1_precision_and_grid_selection_exact': True,
        'all_OOF_scores_predictions_answer_max_and_denominators_exact': True,
        'all_four_raw_baselines_scores_thresholds_full_metrics_exact_to_R22': True,
        'recomputed_metrics': metrics, 'seconds': time.perf_counter() - started,
        'limitations': ['Backward recursion uses future windows: offline localization, not a streaming guarantee.',
                       'Overlapping windows are not independent samples; no causal interpretation of improved smoothing.',
                       'All results use repeatedly examined R16 training-group development folds; this is not a fresh-test performance estimate.',
                       'This audit checks computation and calibration isolation, not bootstrap multiple-comparison correction or generalization.']}
    (OUT / 'INDEPENDENT_AUDIT24.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print('INDEPENDENT_AUDIT24_PASSED', round(report['seconds'], 3), flush=True)


if __name__ == '__main__':
    main()
