"""Independent saved-score audit after the fixed auxiliary/QA run completes."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q

HELPER = ROOT / 'results/minicheck_tail_all_docs_v3/postrun_audit.py'
spec = importlib.util.spec_from_file_location('independent_saved_metrics', HELPER)
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


def main():
    directory = OUT / 'transfer'
    done = q.read(directory / 'complete.json')
    assert done['status'] == 'complete_development_only' and not done['official_test_opened']
    assert done['optimizer_state_continued'] and done['optimizer_updates'] == 2590
    aux = done['auxiliary']
    assert aux['optimizer_updates'] == aux['global_optimizer_updates'] == 1210
    assert aux['answer_presentations'] == 9678 and not aux['calibration_evaluated']
    assert aux['checkpoint_sha256'] == q.sha(directory / 'auxiliary_final.pt')
    assert np.isfinite(aux['online_objective_sum'])
    orders = np.load(OUT / 'qa_answer_orders.npy')
    original = np.load(ROOT / 'results/full_context_encoder_v2/answer_orders.npy')
    assert np.array_equal(orders, original[:3])
    aux_order = np.load(OUT / 'auxiliary_answer_order.npy')
    assert np.array_equal(np.sort(aux_order), np.arange(9678))
    windows = q.lines(ROOT / 'data/windows_k4_calibration.jsonl')
    answers = q.lines(ROOT / 'data/answers_calibration.jsonl')
    wy = np.asarray([w['label'] for w in windows])
    ay = np.asarray([a['label'] for a in answers])
    assert len(windows) == 42241 and len(answers) == 159
    records = []
    for epoch in range(1, 4):
        stem = f'qa_epoch_{epoch:02d}'
        record = q.read(directory / (stem + '.json'))
        assert record['epoch'] == epoch and not record['official_test_opened']
        assert record['global_optimizer_updates'] == 1210 + 460 * epoch
        assert record['training']['optimizer_updates'] == 460
        assert record['training']['answer_presentations'] == 3680
        assert np.isfinite(record['fit_weighted_bce']) and record['fit_weighted_bce'] > 0
        assert np.isfinite(record['training']['online_objective_sum'])
        for suffix, expected in record['artifacts_sha256'].items():
            assert q.sha(directory / (stem + suffix)) == expected
        with np.load(directory / (stem + '_scores.npz'), allow_pickle=False) as scores:
            assert np.array_equal(wy, scores['cal_window_labels'])
            assert np.array_equal(ay, scores['cal_answer_labels'])
            assert len(scores['fit_window_labels']) == 653979
            assert len(scores['fit_answer_labels']) == 3680
            with np.load(directory / (stem + '_token_predictions.npz'), allow_pickle=False) as saved:
                assert len(saved.files) == 3839
                p = {a['response_id']: saved[a['response_id']] for a in answers}
                wscore = np.asarray([max(p[w['response_id']][w['lexical_token_indices']]) for w in windows])
                by_answer = {}
                for w, value in zip(windows, wscore):
                    rid = w['response_id']
                    by_answer[rid] = max(by_answer.get(rid, -np.inf), float(value))
                ascore = np.asarray([by_answer[a['response_id']] for a in answers])
                assert np.array_equal(wscore, scores['cal_window_scores'])
                assert np.array_equal(ascore, scores['cal_answer_scores'])
            cal = {}
            for unit, plural in (('window', 'windows'), ('answer', 'answers')):
                y, s = scores[f'cal_{unit}_labels'], scores[f'cal_{unit}_scores']
                assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
                cut = independent.threshold_from_roc(y, s)
                assert cut == record['thresholds'][unit]['threshold']
                cal[plural] = independent.metrics(y, s, cut)
                independent.same_metrics(cal[plural], record['calibration'][plural])
                fit = independent.metrics(scores[f'fit_{unit}_labels'], scores[f'fit_{unit}_scores'], cut)
                independent.same_metrics(fit, record['fit_at_cal_thresholds'][plural])
            key = [min(cal['windows']['f1'], cal['answers']['f1']), cal['windows']['f1'], cal['windows']['precision'], -epoch]
            assert key == record['selection_key']
        records.append(record)
    assert records == done['all_QA_epochs']
    selected = max(records, key=lambda x: x['selection_key'])
    assert selected == done['selected']
    plan = q.read(OUT / 'WEIGHT_REUSE_AND_RESOURCE_PLAN.json')
    assert done['actual_training_logical_input_tokens'] == plan['logical_training_input_tokens']
    assert done['actual_evaluation_logical_input_tokens'] == plan['total_evaluation_input_tokens']
    q.save(OUT / 'POSTRUN_SCORE_AUDIT.json', {
        'passed': True, 'selected_epoch': selected['epoch'], 'selected_calibration': selected['calibration'],
        'selected_fit_at_cal_thresholds': selected['fit_at_cal_thresholds'],
        'window_f1_by_QA_epoch': [e['calibration']['windows']['f1'] for e in records],
        'answer_f1_by_QA_epoch': [e['calibration']['answers']['f1'] for e in records],
        'fit_weighted_bce_by_QA_epoch': [e['fit_weighted_bce'] for e in records],
        'optimizer_updates': 2590, 'auxiliary_no_calibration_evaluation': True,
        'raw_probability_aggregation_thresholds_metrics_selection_hashes_verified': True,
        'original_QA_first_three_orders_exact': True, 'actual_input_token_budget_matches_fixed_plan': True,
        'seconds': done['seconds'], 'peak_cuda_allocated_bytes': max([aux['peak_cuda_allocated_bytes']] + [e['peak_cuda_allocated_bytes'] for e in records]),
        'neural_model_loaded': False, 'GPU_used': False, 'official_test_opened': False,
        'complete_sha256': q.sha(directory / 'complete.json'), 'auditor_sha256': q.sha(Path(__file__)),
        'independent_metric_helper_sha256': q.sha(HELPER)})
    print('AUX_TRANSFER_POSTRUN_SCORE_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
