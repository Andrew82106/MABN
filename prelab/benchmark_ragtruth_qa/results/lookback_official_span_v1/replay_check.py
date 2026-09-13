"""Saved LR matrix/probability and fit-threshold replay; no model fitting."""
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / 'src'))
import run_development as q


def run():
    start = time.perf_counter()
    complete = q.read(OUT / 'complete.json')
    assert complete['status'] == 'complete' and complete['real_fits'] == 1 and not complete['convergence_warning']
    for name, digest in complete['files_sha256'].items(): assert q.sha(OUT / name) == digest
    summary = q.read(OUT / 'summary.json'); obj = pickle.loads((OUT / 'lookback_lr.pkl').read_bytes())
    model = obj['model']; assert model.get_params() == LogisticRegression(max_iter=1000).get_params()
    assert model.n_iter_.tolist() == summary['n_iter'] == [334]
    assert obj['scaler'] is None and obj['sample_weight'] is None
    x = np.load(OUT / 'span_features.npy', mmap_mode='r')
    with np.load(OUT / 'span_geometry.npz') as z: y, offsets = z['labels'], z['answer_window_offsets']
    with np.load(OUT / 'scores.npz') as z:
        scores, answer, wp, ap = z['window_scores'], z['answer_scores'], z['window_predictions'], z['answer_predictions']
    delta = 0.
    for lo in range(0, len(x), 16384):
        hi = min(len(x), lo+16384)
        wanted = expit(x[lo:hi] @ model.coef_.T + model.intercept_).ravel()
        delta = max(delta, float(np.max(np.abs(wanted - scores[lo:hi]))))
        assert np.array_equal(wanted, scores[lo:hi])
    a = np.full(len(offsets), np.nan)
    for i, (lo, hi) in enumerate(offsets):
        if hi > lo: a[i] = np.max(scores[lo:hi])
    assert np.array_equal(a, answer, equal_nan=True)
    labels = np.asarray([r['answer_label'] for r in q.read(OUT / 'answer_order.json')])
    fit_ai = np.flatnonzero(np.isfinite(a[:3680])); cal_ai = np.arange(3680, 3839)
    ts = summary['thresholds_fit_only']
    assert ts['window'] == q.choose_threshold(y[:639955], scores[:639955])
    assert ts['answer'] == q.choose_threshold(labels[fit_ai], a[fit_ai])
    assert np.array_equal(wp, scores >= ts['window']['threshold'])
    assert np.array_equal(ap, np.where(np.isfinite(a), a >= ts['answer']['threshold'], -1))
    for part, lo, hi, ai in [('fit', 0, 639955, fit_ai), ('calibration', 639955, len(scores), cal_ai)]:
        assert q.count(y[lo:hi], scores[lo:hi], ts['window']['threshold']) == summary['metrics'][part]['spans8']
        assert q.count(labels[ai], a[ai], ts['answer']['threshold']) == summary['metrics'][part]['answers_extra_adaptation']
    q.save(OUT / 'CPU_REPLAY.json', {'status': 'passed', 'all681640_matrix_probabilities_exact': True,
        'max_absolute_difference': delta, 'all_answermax_exact_including_NA': True,
        'fit_only_thresholds_and_all_counts_exact': True, 'default_classifier_parameters_exact': True,
        'source_sha256': q.sha(Path(__file__)), 'complete_sha256': q.sha(OUT / 'complete.json'),
        'model_sha256': q.sha(OUT / 'lookback_lr.pkl'), 'seconds': time.perf_counter() - start,
        'new_fits': 0, 'GPU_used': False, 'test_opened': False})
    print('LOOKBACK8_CPU_REPLAY_PASSED', delta, time.perf_counter() - start, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): run()
