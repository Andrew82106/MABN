"""Four predeclared CPU-only histogram-boosted window fusion candidates."""
from pathlib import Path
import argparse
from collections import Counter
import json
import pickle
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import train_fusion as fusion
from metrics import evaluate_partition, readl

HERE = Path(__file__).resolve().parent
FIXED = dict(learning_rate=0.05, max_iter=100, min_samples_leaf=512,
             early_stopping=False, random_state=20260928)
GRID = [(leaves, l2) for leaves in (3, 7) for l2 in (1., 10.)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main(args):
    started = time.perf_counter()
    assert sklearn.__version__ == '1.6.1', 'Use prelab/.venv'
    assert not (HERE / 'TEST_RELEASE.json').exists(), 'No new training after test release'
    assert not args.out.exists(), 'Use the predeclared fresh output directory'
    args.out.mkdir(parents=True)
    config = read(args.config)
    sa, ra = fusion.assemble_oof(config, 'semantic')
    sb, rb = fusion.assemble_oof(config, 'whitebox')
    answers = readl(HERE / 'data/fit/answers.jsonl')
    sequences = {r['response_id']: r for r in readl(HERE / 'data/fit/sequences.jsonl')}
    assert sa.keys() == sb.keys() == sequences.keys() == {r['response_id'] for r in answers}
    assert len(answers) == 1292
    va, vb = (fusion.predictions(config[branch]['validation']) for branch in ['semantic', 'whitebox'])
    assert len(va) == len(vb) == 271 and sa.keys().isdisjoint(va) and sb.keys().isdisjoint(vb)
    fit_features = fusion.matrices(sa, sb, 'window')
    val_features = fusion.matrices(va, vb, 'window')
    group_counts = Counter(r['group_id'] for r in answers)
    assert len(group_counts) == 617
    features, targets, weights, ids = [], [], [], []
    for answer in answers:
        rid = answer['response_id']; row = sequences[rid]
        indices = fusion.window_indices(row['lexical_mask'])
        risk = np.asarray(row['risk_mask'], bool)
        y = np.array([risk[index].any() for index in indices])
        x = fit_features[rid]
        assert x.shape == (len(y), 17) and len(y)
        features.append(x); targets.extend(y); ids.append(rid)
        weights.extend(np.full(len(y), 1 / (group_counts[answer['group_id']] * len(y))))
    x, y, w = np.concatenate(features), np.asarray(targets, bool), np.asarray(weights)
    assert abs(w.sum() - 617) < 1e-8
    # Match the existing window preprocessing exactly before fitting a new model.
    scaler = StandardScaler().fit(x, sample_weight=w)
    linear_path = HERE / 'fusion/window_v1/best.pkl'
    with linear_path.open('rb') as stream:
        linear = pickle.load(stream)
    assert linear['granularity'] == 'window' and linear['features'] == fusion.ALL_FEATURES
    for field in ['mean_', 'var_', 'scale_', 'n_samples_seen_']:
        assert np.array_equal(getattr(scaler, field), getattr(linear['scaler'], field)), field
    linear_selected = read(HERE / 'fusion/window_v1/SELECTED.json')
    assert linear_selected['balanced'] is True
    masses = np.array([w[~y].sum(), w[y].sum()])
    assert masses.min() > 0
    fit_weights = w * np.where(y, w.sum() / (2 * masses[1]), w.sum() / (2 * masses[0]))
    assert abs(fit_weights.sum() - 617) < 1e-8
    assert abs(fit_weights[y].sum() - fit_weights[~y].sum()) < 1e-8
    xx = scaler.transform(x)
    vv = {rid: scaler.transform(value) for rid, value in val_features.items()}
    fusion.save(args.out / 'FIT_IDS.json', ids)
    training_evidence = {
        'algorithm': 'HistGradientBoostingClassifier', 'fixed_parameters': FIXED,
        'grid': [{'max_leaf_nodes': leaves, 'l2_regularization': l2} for leaves, l2 in GRID],
        'candidate_count': 4, 'granularity': 'window', 'feature_names': fusion.ALL_FEATURES,
        'training_answers': len(ids), 'training_groups': len(group_counts), 'training_windows': len(y),
        'base_weight_sum': float(w.sum()), 'balanced_weight_sum': float(fit_weights.sum()),
        'fit_only_class_masses_before_balance': masses.tolist(),
        'balanced_class_masses': [float(fit_weights[~y].sum()), float(fit_weights[y].sum())],
        'fit_ids_sha256': fusion.sha(args.out / 'FIT_IDS.json'),
        'oof_receipts': {'semantic': ra, 'whitebox': rb}, 'config_sha256': fusion.sha(args.config),
        'scaler_fit_only': True, 'scaler_exactly_matches_linear_window': True,
        'linear_reference_sha256': fusion.sha(linear_path),
        'validation_targets_used_for_fit': False, 'validation_targets_used_for_selection': True,
        'test_labels_read': False, 'test_release_absent_before_training': True,
        'gpu_used': False, 'cpu_threads': 4, 'sklearn_version': sklearn.__version__,
        'selection': 'shared validation no_hint key; exact tie prefers fewer leaves then larger L2',
        'files_sha256': {str(path): fusion.sha(path) for path in [Path(__file__), HERE / 'train_fusion.py',
            HERE / 'metrics.py', HERE / 'evidence_features.py', HERE / 'PROTOCOL.md', HERE / 'data/MANIFEST.json',
            HERE / 'data/fit/answers.jsonl', HERE / 'data/fit/sequences.jsonl',
            Path(config['semantic']['validation']), Path(config['whitebox']['validation'])]}}
    fusion.save(args.out / 'RUN.json', training_evidence)
    print(json.dumps({'stage': 'training_four_candidates', 'fit_windows': len(y),
                      'weight_sum': float(fit_weights.sum()), 'preparation_seconds': time.perf_counter() - started}), flush=True)
    history, best = [], None
    for leaves, l2 in GRID:
        tick = time.perf_counter()
        model = HistGradientBoostingClassifier(**FIXED, max_leaf_nodes=leaves, l2_regularization=l2)
        # Only the 1292 fit answers supply features, labels, and class weights here.
        model.fit(xx, y, sample_weight=fit_weights)
        assert model.n_iter_ == 100 and model.classes_.tolist() == [False, True]
        assert not model.do_early_stopping_
        scores = {rid: model.predict_proba(value)[:, 1] for rid, value in vv.items()}
        report = evaluate_partition('validation', scores, scores_are_windows=True)
        name = f'hist_leaf{leaves}_l2_{l2:g}'
        entry = {'name': name, 'max_leaf_nodes': leaves, 'l2_regularization': l2,
                 'fixed_parameters': FIXED, 'balanced': True, 'feature_count': 17, 'granularity': 'window',
                 'report': report, 'thresholds': report['thresholds'], 'iterations': int(model.n_iter_),
                 'seconds': time.perf_counter() - tick, 'test_labels_read': False}
        entry['selection_key'] = report['selection_key'] + [-leaves, l2]
        history.append(entry)
        fusion.save(args.out / f'{name}.json', entry)
        fusion.write_predictions(args.out / f'{name}_validation_predictions.jsonl', scores, 'window')
        if best is None or entry['selection_key'] > best['selection_key']:
            best = dict(entry)
            with (args.out / 'best.pkl').open('wb') as stream:
                pickle.dump({'model': model, 'scaler': scaler, 'features': fusion.ALL_FEATURES,
                             'width': 17, 'granularity': 'window'}, stream)
            fusion.write_predictions(args.out / 'best_validation_predictions.jsonl', scores, 'window')
        fusion.save(args.out / 'HISTORY.json', history)
        primary = report['cohorts']['no_hint']
        print(json.dumps({'candidate': name, 'seconds': entry['seconds'],
                          'answer_f1': primary['answer']['f1'], 'window_f1': primary['window']['f1']}), flush=True)
    best.update({'feature_names': fusion.ALL_FEATURES, 'config_sha256': fusion.sha(args.config),
                 'model_sha256': fusion.sha(args.out / 'best.pkl'), 'oof_receipts': training_evidence['oof_receipts'],
                 'training_answers': len(ids), 'training_groups': len(group_counts),
                 'training_evidence_sha256': fusion.sha(args.out / 'RUN.json')})
    fusion.save(args.out / 'SELECTED.json', best)
    fusion.save(args.out / 'COMPLETE.json', {'candidate_count': len(history), 'selected': best['name'],
        'seconds': time.perf_counter() - started, 'model_sha256': best['model_sha256'],
        'test_labels_read': False, 'gpu_used': False, 'validation_used_only_for_selection': True})
    print(json.dumps({'complete': True, 'selected': best['name'], 'seconds': time.perf_counter() - started}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'fusion/OOF_CONFIG.json')
    parser.add_argument('--out', type=Path, default=HERE / 'fusion/nonlinear_window_v1')
    with threadpool_limits(limits=4):
        main(parser.parse_args())
