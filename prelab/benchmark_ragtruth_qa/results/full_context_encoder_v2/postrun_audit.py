"""Independent CPU saved-score audit; no encoder loading and no test data."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q

HELPER = ROOT / 'results/minicheck_tail_all_docs_v3/postrun_audit.py'
spec = importlib.util.spec_from_file_location('independent_saved_score_metrics', HELPER)
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


def main():
    directory = OUT / 'full_finetune'
    complete = q.read(directory / 'complete.json')
    assert complete['status'] == 'complete_development_only'
    assert complete['official_test_opened'] is False
    windows = q.lines(ROOT / 'data/windows_k4_calibration.jsonl')
    answers = q.lines(ROOT / 'data/answers_calibration.jsonl')
    canonical_w = np.asarray([w['label'] for w in windows])
    canonical_a = np.asarray([a['label'] for a in answers])
    assert len(windows) == 42241 and len(answers) == 159
    orders = np.load(OUT / 'answer_orders.npy', allow_pickle=False)
    assert orders.shape == (6, 3680)
    assert all(np.array_equal(np.sort(order), np.arange(3680)) for order in orders)
    records = []
    for epoch in range(7):
        stem = f'epoch_{epoch:02d}'
        record = q.read(directory / (stem + '.json'))
        assert record['epoch'] == epoch and record['official_test_opened'] is False
        for suffix, expected in record['artifacts_sha256'].items():
            assert q.sha(directory / (stem + suffix)) == expected
        assert np.isfinite(record['fit_weighted_bce']) and record['fit_weighted_bce'] > 0
        assert np.isfinite(record['online_objective_sum'])
        with np.load(directory / (stem + '_scores.npz'), allow_pickle=False) as z:
            assert np.array_equal(z['cal_window_labels'], canonical_w)
            assert np.array_equal(z['cal_answer_labels'], canonical_a)
            assert len(z['fit_window_labels']) == 653979 and len(z['fit_answer_labels']) == 3680
            with np.load(directory / (stem + '_token_predictions.npz'), allow_pickle=False) as p:
                assert len(p.files) == 3839
                probabilities = {a['response_id']: p[a['response_id']] for a in answers}
                window_score = np.asarray([max(probabilities[w['response_id']][w['lexical_token_indices']]) for w in windows])
                per_answer = {}
                for w, s in zip(windows, window_score):
                    rid = w['response_id']
                    per_answer[rid] = max(float(s), per_answer.get(rid, -np.inf))
                answer_score = np.asarray([per_answer[a['response_id']] for a in answers])
                assert np.array_equal(window_score, z['cal_window_scores'])
                assert np.array_equal(answer_score, z['cal_answer_scores'])
            cal = {}
            for unit, plural in (('window', 'windows'), ('answer', 'answers')):
                y, s = z[f'cal_{unit}_labels'], z[f'cal_{unit}_scores']
                assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
                cut = independent.threshold_from_roc(y, s)
                assert cut == record['thresholds'][unit]['threshold']
                cal[plural] = independent.metrics(y, s, cut)
                independent.same_metrics(cal[plural], record['calibration'][plural])
                fit = independent.metrics(z[f'fit_{unit}_labels'], z[f'fit_{unit}_scores'], cut)
                independent.same_metrics(fit, record['fit_at_cal_thresholds'][plural])
            key = [min(cal['windows']['f1'], cal['answers']['f1']), cal['windows']['f1'], cal['windows']['precision'], -epoch]
            assert key == record['selection_key']
        records.append(record)
    assert records == complete['all_epochs']
    selected = max(records[1:], key=lambda x: x['selection_key'])
    assert selected == complete['selected']
    q.save(OUT / 'POSTRUN_SCORE_AUDIT.json', {
        'passed': True, 'audit': 'CPU saved scores, independent sklearn ROC thresholds and original lexical-window aggregation',
        'epochs_including_unselectable_initialization': 7, 'selected_epoch': selected['epoch'],
        'selected_calibration': selected['calibration'], 'selected_fit_at_cal_thresholds': selected['fit_at_cal_thresholds'],
        'fit_weighted_bce_by_epoch': [r['fit_weighted_bce'] for r in records],
        'window_f1_by_epoch': [r['calibration']['windows']['f1'] for r in records],
        'answer_f1_by_epoch': [r['calibration']['answers']['f1'] for r in records],
        'epoch_seconds': [r['seconds'] for r in records], 'training_loop_seconds': complete['seconds'],
        'peak_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in records),
        'calibration_labels_equal_original': True, 'all_epoch_calibration_scores_rebuilt_from_saved_raw_token_probabilities': True,
        'thresholds_metrics_selection_and_artifact_hashes_verified': True,
        'orders_are_six_full_fit_permutations': True, 'optimizer_updates_by_frozen_loop_definition': 2760,
        'neural_model_loaded': False, 'GPU_used': False, 'official_test_opened': False,
        'complete_sha256': q.sha(directory / 'complete.json'), 'auditor_sha256': q.sha(Path(__file__)),
        'independent_metric_helper_sha256': q.sha(HELPER)})
    print('FULL_CONTEXT_V2_POSTRUN_SCORE_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
