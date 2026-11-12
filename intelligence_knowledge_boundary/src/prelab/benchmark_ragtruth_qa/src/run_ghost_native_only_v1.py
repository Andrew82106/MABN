"""One fixed native-only training ablation; existing expanded model is reused."""
from pathlib import Path
import argparse
import pickle
import time
import traceback
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_ghost_nll_nonlinear_v1 as original

q = original.q
OUT = original.ROOT / 'results/ghost_native_only_v1'
NNATIVE = 168123


def fit():
    assert not (OUT / 'started.json').exists(), 'Preserve prior attempts.'
    meta, features, weights = original.check()
    completed = q.read(original.OUT / 'complete.json')
    assert completed['status'] == 'complete' and completed['fixed_fits'] == 3
    paths = [original.OUT / ('nll_ghost' + suffix) for suffix in ('_result.json', '_scores.npz')]
    for p in paths:
        assert q.sha(p) == completed['files_sha256'][p.name]
    baseline = q.read(paths[0])
    with np.load(paths[1], allow_pickle=False) as z:
        old_scores = z['window_scores'].copy()
        old_answers = z['answer_scores'].copy()
    assert np.array_equal(old_answers, q.answer_scores(meta, old_scores))
    assert q.metrics(meta, old_scores, baseline['thresholds']) == baseline['metrics']
    native_indices = np.concatenate([meta['answer_windows'][a['response_id']] for a in meta['answers'][:634]])
    assert np.array_equal(native_indices, np.arange(NNATIVE))
    assert sum(weights['y'][:NNATIVE]) == 21477
    native_weights = weights['loss'][:NNATIVE].astype(np.float64, copy=True)
    scale = float(weights['loss'].sum() / native_weights.sum())
    native_weights *= scale
    assert np.isfinite(native_weights).all() and (native_weights > 0).all()
    assert abs(native_weights.sum() - weights['loss'].sum()) < 1e-6
    parameters = original.PARAMS.copy()
    assert HistGradientBoostingClassifier(**parameters).get_params() == baseline['effective_parameters']
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'version': 'ghost-native-only-data-ablation-v1', 'new_fits': 1,
        'parameters': parameters, 'features': original.NAMES,
        'training': {'answers': 634, 'windows': NNATIVE, 'groups': 615,
            'weight_rule': 'Retain original native loss ratios, rescale their total to original expanded total168123. No class-factor refit.',
            'weight_scale': scale, 'loss_mass': float(native_weights.sum()),
            'positive_loss_mass': float(native_weights[weights['y'][:NNATIVE] == 1].sum())},
        'control': 'Completed expanded nll_ghost shallow tree, reused without refitting.',
        'evaluation': 'Same original cal159/42241windows, all labels/refusals/4BPE geometry retained; same independent window and all-window answermax threshold selection rule.',
        'selection': 'Single fixed data ablation, no parameter or cross-method search.',
        'limits': ['Designed after generator distribution diagnostics; reused calibration is development evidence.',
            'Removing added data changes training population, score distribution, feature binning and the relative min_samples_leaf constraint; not unique identification of the causal effect of replay.',
            'Weight normalization matches total loss scale but not the original expanded label prior or all per-group shares.',
            'This is not a native-only-from-scratch weighting pipeline: inherited relative class/group weights were formed by the earlier expanded fit-only procedure.'],
        'no_test': True, 'GPU_used': False}
    q.save(OUT / 'protocol.json', protocol)
    q.save(OUT / 'started.json', {'source_sha256': q.sha(Path(__file__)),
        'upstream_complete_sha256': q.sha(original.OUT / 'complete.json'),
        'upstream_preparation_sha256': q.sha(original.OUT / 'preparation_complete.json'),
        'protocol_sha256': q.sha(OUT / 'protocol.json'), 'new_fits': 1, 'GPU_used': False})
    np.save(OUT / 'native_loss_weights.npy', native_weights)
    tick = time.perf_counter()
    model = HistGradientBoostingClassifier(**parameters)
    model.fit(np.asarray(features[:NNATIVE]), weights['y'][:NNATIVE], sample_weight=native_weights)
    assert model.n_iter_ == 100 and not model.do_early_stopping_
    assert model.classes_.tolist() == [0, 1] and model.n_features_in_ == 5
    scores = model.predict_proba(features)[:, 1]
    assert scores.shape == (original.NTOTAL,) and np.isfinite(scores).all()
    answers = q.answer_scores(meta, scores)
    ts = {
        'window': q.choose_threshold([w['label'] for w in meta['windows'][original.NFIT:]], scores[original.NFIT:]),
        'answer': q.choose_threshold([a['label'] for a in meta['answers'][3680:]], answers[3680:])}
    wy = np.asarray([w['label'] for w in meta['windows']])
    ay = np.asarray([a['label'] for a in meta['answers']])
    metrics = {}
    for name, wi, ai in [('trained_native', slice(0, NNATIVE), slice(0, 634)),
                         ('untrained_added_same_sources', slice(NNATIVE, original.NFIT), slice(634, 3680)),
                         ('calibration', slice(original.NFIT, None), slice(3680, None))]:
        metrics[name] = {'windows': q.count(wy[wi], scores[wi], ts['window']['threshold']),
                         'answers': q.count(ay[ai], answers[ai], ts['answer']['threshold'])}
    np.savez_compressed(OUT / 'scores.npz', window_scores=scores, answer_scores=answers)
    (OUT / 'model.pkl').write_bytes(pickle.dumps({'model': model, 'thresholds': ts, 'feature_names': original.NAMES}, protocol=5))
    result = {'thresholds': ts, 'metrics': metrics,
        'expanded_control': {'thresholds': baseline['thresholds'], 'calibration': baseline['metrics']['calibration'],
            'common_native': {
                'windows': q.count(wy[:NNATIVE], old_scores[:NNATIVE], baseline['thresholds']['window']['threshold']),
                'answers': q.count(ay[:634], old_answers[:634], baseline['thresholds']['answer']['threshold'])}},
        'calibration_f1_difference_native_minus_expanded': {
            k: metrics['calibration'][k]['f1'] - baseline['metrics']['calibration'][k]['f1'] for k in ('windows', 'answers')},
        'seconds': time.perf_counter() - tick, 'iterations': model.n_iter_,
        'new_fits': 1, 'no_test': True, 'GPU_used': False}
    q.save(OUT / 'result.json', result)
    q.save(OUT / 'complete.json', {'status': 'complete', 'new_fits': 1,
        'files_sha256': {n: q.sha(OUT / n) for n in ('protocol.json', 'started.json', 'native_loss_weights.npy',
            'model.pkl', 'scores.npz', 'result.json')}, 'no_test': True, 'GPU_used': False})
    print('NATIVE_ONLY_COMPLETE', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['fit'])
    parser.parse_args()
    try:
        with threadpool_limits(limits=4):
            fit()
    except Exception:
        if OUT.exists():
            q.save(OUT / f'FAILURE_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
        raise
