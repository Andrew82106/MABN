"""Independent read-only reconstruction; no producer imports, training, or GPU."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEATURE = ROOT / 'results/lumina_qa_features_v1'
SCORE = ROOT / 'results/lumina_qa_scoring_v1'
PREP = ROOT / 'results/lumina_qa_preparation_v1'
METHODS = ('lumina', 'ipr', 'negative_mmd')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def distribution(y, scores):
    # Ascending unique score bins, independent of producer's descending stable sort.
    values, inverse, total = np.unique(scores, return_inverse=True, return_counts=True)
    positive = np.bincount(inverse, weights=y, minlength=len(values)).astype(np.int64)
    negative = total - positive
    return values, total, positive, negative


def threshold(y, scores):
    values, total, positive, _ = distribution(y, scores)
    tp = positive[::-1].cumsum()[::-1]
    predicted = total[::-1].cumsum()[::-1]
    cut = np.r_[values, np.nextafter(values[-1], np.inf)]
    tp = np.r_[tp, 0]
    predicted = np.r_[predicted, 0]
    f1 = 2 * tp / (predicted + y.sum())
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), float), where=predicted > 0)
    i = np.lexsort((cut, precision, f1))[-1]
    return {'threshold': float(cut[i]), 'f1': float(f1[i]),
            'precision': float(precision[i]), 'rows': len(y), 'positive': int(y.sum())}


def metrics(y, scores, cut):
    pred = scores >= cut
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    _, total, pos, neg = distribution(y, scores)
    # Mann-Whitney pair count with half credit for tied positive-negative pairs.
    negatives_below = neg.cumsum() - neg
    auc = float(np.sum(pos * (negatives_below + .5 * neg)) / (pos.sum() * neg.sum()))
    # Stepwise precision-recall area, grouping every tied score before a decision.
    cumulative_pos = pos[::-1].cumsum()
    cumulative_total = total[::-1].cumsum()
    ap = float(np.sum(pos[::-1] / pos.sum() * (cumulative_pos / cumulative_total)))
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'auroc': auc, 'average_precision': ap}


def numeric_equal(actual, expected, tolerance=2e-15):
    assert set(actual) == set(expected)
    difference = {}
    for key in actual:
        if isinstance(actual[key], int):
            assert actual[key] == expected[key], (key, actual[key], expected[key])
        else:
            delta = abs(actual[key] - expected[key])
            assert delta <= tolerance, (key, actual[key], expected[key], delta)
            difference[key] = delta
    return difference


def main():
    started = time.perf_counter()
    assert not (OUT / 'NUMERIC_AUDIT.json').exists(), 'Do not overwrite completed audit'
    hashes = {}

    def verify(path, expected):
        path = Path(path)
        actual = sha(path)
        assert actual == expected, (str(path), actual, expected)
        hashes[str(path.resolve())] = actual

    finished = read(FEATURE / 'features_complete.json')
    manifest = read(FEATURE / 'feature_manifest.json')
    signature = read(FEATURE / 'signature.json')
    scored = read(SCORE / 'complete.json')
    snapshot = read(SCORE / 'source_snapshot.json')
    assert finished['status'] == manifest['status'] == scored['status'] == 'complete'
    assert finished['records'] == manifest['records'] == len(manifest['entries']) == 3839
    assert finished['raw_answer_tokens'] == manifest['raw_answer_tokens'] == 708506
    assert finished['all_records_validated'] and manifest['all_records_validated']
    assert not signature['labels_used'] and not signature['trained'] and not signature['test_opened']
    assert not manifest['labels_used'] and not manifest['trained'] and not manifest['test_opened']
    assert scored['new_fits'] == 0 and not scored['GPU_used'] and not scored['official_test_opened']
    sig_hash = digest(signature)
    assert sig_hash == finished['signature_sha256'] == manifest['signature_sha256']
    assert sig_hash == 'b11a164806a1e5f6dd549374155f2a5b0112ce42fc86b92286053005164b6dc9'
    verify(FEATURE / 'features_complete.json', scored['feature_complete_sha256'])
    for name, expected in finished['files_sha256'].items():
        verify(FEATURE / name, expected)
    for name, expected in scored['files_sha256'].items():
        verify(SCORE / name, expected)
    score_complete_before = sha(SCORE / 'complete.json')
    for path, expected in snapshot['files_sha256'].items():
        verify(path, expected)
    for section in ('code_sha256', 'source_sha256'):
        for path, expected in signature[section].items():
            verify(path, expected)
    model_dir = ROOT.parent / 'models/Llama-2-7b-chat-hf'
    model_bytes = 0
    for name, entry in signature['model']['assets'].items():
        path = model_dir / name
        assert path.stat().st_size == entry['bytes']
        verify(path, entry['sha256'])
        model_bytes += entry['bytes']
    print('AUDIT_MODEL_SOURCE_HASHES_PASSED', model_bytes, flush=True)

    answers = lines(ROOT / 'fit_expansion/data/answers_fit.jsonl') + lines(ROOT / 'data/answers_calibration.jsonl')
    tokens = lines(ROOT / 'fit_expansion/data/tokens_fit.jsonl') + lines(ROOT / 'data/tokens_calibration.jsonl')
    windows = lines(ROOT / 'fit_expansion/data/windows_k4_fit.jsonl') + lines(ROOT / 'data/windows_k4_calibration.jsonl')
    rows = lines(PREP / 'feature_inputs.jsonl')
    assert len(answers) == len(tokens) == len(rows) == 3839 and len(windows) == 696220
    assert [a['response_id'] for a in answers] == [r['response_id'] for r in rows] == manifest['answer_order']
    groups = {part: {a['group_id'] for a in answers if a['partition'] == part} for part in ('fit', 'calibration')}
    assert len(groups['fit']) == 615 and len(groups['calibration']) == 154
    assert not groups['fit'] & groups['calibration']
    aw = defaultdict(list)
    for j, w in enumerate(windows):
        aw[w['response_id']].append(j)
    rebuilt = np.empty((696220, 3), np.float64)
    maximum = np.empty((3839, 3), np.float64)
    mean = np.empty_like(maximum)
    raw_count = 0
    punctuation_slots = 0
    short_windows = 0
    raw_outside_windows = 0
    negative_first_offsets = 0
    per_record = []
    for i, (row, answer, tok, entry) in enumerate(zip(rows, answers, tokens, manifest['entries'])):
        rid = answer['response_id']
        assert row['response_id'] == tok['response_id'] == entry['response_id'] == rid
        assert entry['record_index'] == str(i)
        assert entry['record_sha256'] == digest(row)
        assert entry['signature_sha256'] == sig_hash
        assert row['source_id'] == answer['source_id'] == tok['source_id'] == entry['source_id']
        assert row['group_id'] == answer['group_id'] == tok['group_id'] == entry['group_id']
        assert row['partition'] == answer['partition'] == tok['partition'] == ('fit' if i < 3680 else 'calibration')
        assert row['answer_sha256'] == answer['answer_sha256'] == tok['answer_sha256'] == entry['answer_sha256']
        assert row['answer_token_ids'] == tok['token_ids']
        assert row['response_token_offsets'] == tok['response_token_offsets']
        assert row['response_token_offsets_raw'] == tok['response_token_offsets_raw']
        assert row['original_answer_positions'] == tok['answer_token_positions']
        assert answer['quality'] == 'good' and answer['eligible']
        assert answer['label'] == int(bool(answer['original_labels']))
        path = FEATURE / 'features' / f'{i:05d}.npz'
        assert entry['file'] == path.name and read(path.with_suffix('.json')) == entry
        verify(path, entry['npz_sha256'])
        with np.load(path, allow_pickle=False) as z:
            assert str(z['signature_sha256']) == sig_hash
            assert str(z['record_sha256']) == digest(row)
            assert str(z['record_index']) == str(i) and str(z['response_id']) == rid
            assert z['token_ids'].tolist() == tok['token_ids']
            assert np.column_stack((z['token_start'], z['token_end'])).tolist() == tok['response_token_offsets']
            assert np.column_stack((z['token_start_raw'], z['token_end_raw'])).tolist() == tok['response_token_offsets_raw']
            for side in ('original', 'random'):
                assert z[side + '_answer_positions'].tolist() == row[side + '_answer_positions']
                assert np.array_equal(z[side + '_predictor_positions'], z[side + '_answer_positions'] - 1)
            arr = z['lumina_features'].copy()
        assert arr.dtype == np.float32 and arr.shape == (tok['token_count'], 7) and np.isfinite(arr).all()
        primary = np.subtract(np.multiply(arr[:, 0], np.float32(.5)),
                              np.multiply(arr[:, 1], np.float32(.5)))
        assert np.array_equal(primary, arr[:, 2]), rid
        ts = np.stack((primary, arr[:, 0], -arr[:, 1]), axis=1).astype(np.float64)
        n = len(ts)
        lexical = np.asarray(tok['lexical_mask'], bool)
        risk = np.asarray(tok['risk_mask'], bool)
        assert len(lexical) == len(risk) == n and n > 0
        # Independently enumerate every eligible raw sliding window, without a risk-label filter.
        starts = range(n - 3) if n >= 4 else [0]
        expected = [list(range(start, min(start + 4, n))) for start in starts
                    if lexical[start:min(start + 4, n)].any()]
        wi = aw[rid]
        assert [windows[j]['token_indices'] for j in wi] == expected
        covered = set()
        for j, ix in zip(wi, expected):
            w = windows[j]
            assert w['eligible'] and w['partition'] == answer['partition']
            assert w['label'] == int(risk[ix].any())
            assert w['token_ids'] == [tok['token_ids'][k] for k in ix]
            rebuilt[j] = np.sum(ts[ix], axis=0, dtype=np.float64) / len(ix)
            punctuation_slots += int((~lexical[ix]).sum())
            short_windows += int(len(ix) < 4)
            covered.update(ix)
        assert wi
        maximum[i] = np.maximum.reduce(rebuilt[wi], axis=0)
        mean[i] = np.sum(ts, axis=0, dtype=np.float64) / n
        raw_outside_windows += n - len(covered)
        negative_first_offsets += int(tok['response_token_offsets_raw'][0][0] < 0)
        raw_count += n
        per_record.append({'record_index': i, 'response_id': rid, 'npz_sha256': entry['npz_sha256'],
                           'record_sha256': entry['record_sha256'], 'tokens': n, 'windows': len(wi),
                           'formula_exact': True, 'all_original_axes_exact': True,
                           'window_geometry_exact': True})
        if (i + 1) % 500 == 0:
            print('AUDIT_RECONSTRUCTED', i + 1, flush=True)
    assert raw_count == 708506
    assert np.isfinite(rebuilt).all() and np.isfinite(maximum).all() and np.isfinite(mean).all()
    wy = np.asarray([w['label'] for w in windows], int)
    ay = np.asarray([a['label'] for a in answers], int)
    assert sum(a['partition'] == 'calibration' for a in answers) == 159
    assert sum(w['partition'] == 'calibration' for w in windows) == 42241
    assert int(ay[3680:].sum()) == 100 and int(wy[653979:].sum()) == 5984
    summary = read(SCORE / 'summary.json')
    assert summary['primary'] == 'lumina' and not summary['method_selection_performed']
    assert summary['window_order_sha256'] == digest([w['window_id'] for w in windows])
    assert summary['answer_order_sha256'] == digest([a['response_id'] for a in answers])
    checks = {}
    for col, name in enumerate(METHODS):
        saved_result = read(SCORE / f'{name}_result.json')
        assert summary['all_fixed_readouts'][name] == saved_result
        with np.load(SCORE / f'{name}_scores.npz', allow_pickle=False) as z:
            exact = {}
            for key, independent in (('window_scores', rebuilt[:, col]),
                                     ('answermax_scores', maximum[:, col]),
                                     ('paper_answer_mean_scores', mean[:, col])):
                assert np.array_equal(z[key], independent), (name, key, np.max(abs(z[key] - independent)))
                exact[key] = {'values': len(independent), 'exact': True, 'max_abs_error': 0.}
        t_window = threshold(wy[653979:], rebuilt[653979:, col])
        t_answer = threshold(ay[3680:], maximum[3680:, col])
        t_mean = threshold(ay[3680:], mean[3680:, col])
        assert t_window == saved_result['thresholds']['window']
        assert t_answer == saved_result['thresholds']['answer']
        assert t_mean == saved_result['paper_answer_mean']['threshold']
        wm = metrics(wy[653979:], rebuilt[653979:, col], t_window['threshold'])
        am = metrics(ay[3680:], maximum[3680:, col], t_answer['threshold'])
        pm = metrics(ay[3680:], mean[3680:, col], t_mean['threshold'])
        errors = {
            'window': numeric_equal(wm, saved_result['window_then_answermax_metrics']['calibration']['windows']),
            'answermax': numeric_equal(am, saved_result['window_then_answermax_metrics']['calibration']['answers']),
            'paper_answer_mean': numeric_equal(pm, saved_result['paper_answer_mean']['metrics']['calibration'])}
        checks[name] = {'score_arrays': exact, 'thresholds_exact': True,
                        'thresholds': {'window': t_window, 'answermax': t_answer, 'paper_answer_mean': t_mean},
                        'calibration_metrics': {'window': wm, 'answermax': am, 'paper_answer_mean': pm},
                        'metric_absolute_errors': errors}
    # Recheck completed scorer files after reconstruction; audit never writes them.
    assert sha(SCORE / 'complete.json') == score_complete_before
    for name, expected in scored['files_sha256'].items():
        assert sha(SCORE / name) == expected
    assert sha(FEATURE / 'features_complete.json') == scored['feature_complete_sha256']
    report = {
        'status': 'passed', 'audit_kind': 'independent numeric reconstruction from every raw feature NPZ',
        'producer_aggregation_or_metric_functions_called': False, 'producer_modules_imported': False,
        'new_fits': 0, 'GPU_used': False, 'official_test_opened': False,
        'signature_sha256': sig_hash, 'model_files_actual_full_SHA_verified': True,
        'model_asset_bytes_hashed': model_bytes, 'feature_records_actual_SHA_verified': 3839,
        'counts': {'answers': 3839, 'fit_answers': 3680, 'calibration_answers': 159,
                   'fit_groups': 615, 'calibration_groups': 154, 'raw_tokens': raw_count,
                   'windows': len(windows), 'fit_windows': 653979, 'calibration_windows': 42241,
                   'calibration_positive_answers': 100, 'calibration_positive_windows': 5984,
                   'punctuation_slot_occurrences_in_windows': punctuation_slots,
                   'short_windows': short_windows, 'raw_tokens_outside_all_eligible_windows': raw_outside_windows,
                   'negative_first_raw_offsets_preserved': negative_first_offsets},
        'all_708506_float32_token_formulas_exact': True,
        'all_eligible_geometry_reenumerated_without_gold_filter': True,
        'all_3839_answer_labels_preserved_including_unmarked_refusals': True,
        'calibration_thresholds_recomputed': 9,
        'metrics_algorithm': 'Exact confusion counts; unique-bin threshold search; independent tied-pair AUC and stepwise precision-recall AP.',
        'methods': checks, 'all_original_scoring_artifacts_unchanged_after_audit': True,
        'seconds': time.perf_counter() - started,
        'limitations': ['Read-only score/geometry audit; no GPU model replay or independent world-truth relabeling.',
                        'Four-raw-BPE mean and answermax are the unified evaluation adaptation; raw whole-answer mean is separate.',
                        'All reported metrics are repeatedly exposed calibration results, not official sealed-test performance.',
                        'IPR and negative-MMD readouts are components/ablations, not alternatives selected as the formal LUMINA result.']}
    (OUT / 'PER_RECORD_CHECKS.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in per_record), encoding='utf-8')
    (OUT / 'VERIFIED_INPUT_HASHES.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    report['audit_script_sha256'] = sha(__file__)
    report['per_record_checks_sha256'] = sha(OUT / 'PER_RECORD_CHECKS.jsonl')
    report['verified_input_hashes_sha256'] = sha(OUT / 'VERIFIED_INPUT_HASHES.json')
    (OUT / 'NUMERIC_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('INDEPENDENT_NUMERIC_AUDIT_PASSED', round(report['seconds'], 2), flush=True)
    for name, r in checks.items():
        m = r['calibration_metrics']
        print(name, m['window']['f1'], m['answermax']['f1'], m['paper_answer_mean']['f1'], flush=True)


if __name__ == '__main__':
    main()
