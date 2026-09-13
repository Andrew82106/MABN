"""Training-only, grouped diagnostic CV. All variants reported; no outer selection."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'round12_conditional_fusion'
spec = importlib.util.spec_from_file_location('r13_reuses_r12', PREVIOUS/'src/run12.py')
r12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)
r10, np, SOURCE = r12.r10, r12.np, r12.SOURCE
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS = ('lb_c01', 'lb_c001', 'lb_c0001', 'pca16', 'pca64', 'layer_mean28', 'fit24', 'fit48')
CS = dict(zip(METHODS, (.1, .01, .001, .1, .1, .1, .1, .1)))
SEED = 20260913


def snapshot():
    prior = r12.snapshot()
    assert prior == json.loads((PREVIOUS/'results/source_snapshot12.json').read_text('utf-8'))
    protocol = json.loads((ROOT/'protocol.json').read_text('utf-8'))
    assert protocol['methods'] == list(METHODS) and protocol['C'] == CS and protocol['seed'] == SEED
    paths = [ROOT/'PLAN.md', ROOT/'protocol.json', *sorted((ROOT/'src').glob('*.py')),
             ROOT/'data/blind_label_sample.jsonl', ROOT/'data/label_sample_key.jsonl']
    return {'prior': prior, 'local_files_sha256': {p.relative_to(ROOT).as_posix(): r10.sha(p) for p in paths},
            'outer_folds_sha256': r10.sha(PREVIOUS/'results/fold_assignment.json')}


def transform(model, x):
    raw = x.reshape(-1, 28, 28).mean(axis=2) if model['feature'] == 'layer_mean28' else x
    z = model['scaler'].transform(raw).astype(np.float32)
    if model['components'] is not None:
        z = (z @ model['components'].T).astype(np.float32)
    return z


def weights(rows, target_mass):
    y = np.asarray([r['gold'] for r in rows], int)
    b = r10.base_weights(rows, 'token'); w, factors = r10.loss_weights(rows, y, b)
    w *= target_mass / w.sum()
    return b, w, factors


def fit_model(x, rows, feature, c, target_mass, pca_cache):
    b, w, factors = weights(rows, target_mass)
    raw = x.reshape(-1, 28, 28).mean(axis=2) if feature == 'layer_mean28' else x
    scaler = r10.StandardScaler().fit(raw, sample_weight=b)
    components = None; explained = None
    if feature in ('pca16', 'pca64'):
        if 'components' not in pca_cache:
            z = scaler.transform(raw).astype(np.float32).astype(np.float64)
            covariance = (z.T * (b/b.sum())) @ z
            vals, vecs = np.linalg.eigh(covariance)
            order = np.argsort(vals)[::-1]
            pca_cache['components'] = vecs[:, order[:64]].T
            pca_cache['variance'] = np.maximum(vals[order], 0)
        dim = int(feature[3:]); components = pca_cache['components'][:dim].copy()
        explained = float(pca_cache['variance'][:dim].sum()/pca_cache['variance'].sum())
    model = {'scaler': scaler, 'components': components, 'feature': feature, 'C': c,
             'base_weights': b, 'loss_weights': w, 'class_factors': factors,
             'target_loss_mass': target_mass, 'pca_explained_variance': explained}
    y = np.asarray([r['gold'] for r in rows], int)
    classifier = r10.LogisticRegression(C=c, penalty='l2', solver='liblinear', max_iter=2000, random_state=SEED)
    classifier.fit(transform(model, x), y, sample_weight=w)
    model['model'] = classifier
    assert int(classifier.n_iter_.max()) < 2000
    return model


def probability(model, x):
    return model['model'].predict_proba(transform(model, x))[:, 1]


def count(y, scores, threshold=None, predictions=None):
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    pred = scores >= threshold if predictions is None else np.asarray(predictions, bool)
    tp, fp = int(np.sum((y == 1) & pred)), int(np.sum((y == 0) & pred))
    fn, tn = int(np.sum((y == 1) & ~pred)), int(np.sum((y == 0) & ~pred))
    both = len(set(y)) == 2
    return {'tokens': len(y), 'risk_tokens': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if tp+fn else None,
            'auroc': float(roc_auc_score(y, scores)) if both else None,
            'average_precision': float(average_precision_score(y, scores)) if y.sum() else None,
            'alert_rate': float(pred.mean()), 'risk_rate': float(y.mean())}


def diagnostics(rows):
    by_answer = defaultdict(list)
    for row in rows:
        assert len(row['item_ids']) == 1
        by_answer[row['item_ids'][0]].append(row)
    result = {}
    for method in METHODS:
        ranks = []; fp_categories = Counter(); categories = defaultdict(list)
        for iid, rr in by_answer.items():
            rr.sort(key=lambda r: r['token_index'])
            risk_indices = [j for j, row in enumerate(rr) if row['gold'] == 1]
            y = np.asarray([r['gold'] for r in rr])
            if len(set(y)) == 2:
                scores = np.asarray([r['scores'][method] for r in rr])
                ranks.append({'item_id': iid, 'auroc': float(roc_auc_score(y, scores)),
                              'average_precision': float(average_precision_score(y, scores)), 'tokens': len(rr)})
            for j, row in enumerate(rr):
                categories[row['attribute_group']].append(row)
                if row['predictions'][method] and row['gold'] == 0:
                    if not risk_indices:
                        tag = 'FP_in_fully_nonrisk_answer'
                    elif min(abs(j-k) for k in risk_indices) == 1:
                        tag = 'FP_one_eligible_token_from_gold_risk'
                    else:
                        tag = 'FP_farther_inside_risk_answer'
                    fp_categories[tag] += 1
        result[method] = {'within_answer_ranking': {'answers_with_both_labels': len(ranks),
                            'macro_auroc': float(np.mean([r['auroc'] for r in ranks])),
                            'macro_average_precision': float(np.mean([r['average_precision'] for r in ranks])),
                            'per_answer': ranks},
                          'false_positive_locations': dict(fp_categories),
                          'strata': {k: count([r['gold'] for r in rr], [r['scores'][method] for r in rr],
                                              predictions=[r['predictions'][method] for r in rr]) for k, rr in categories.items()}}
    return result


def bootstrap(rows):
    groups = sorted({r['group_id'] for r in rows}); gi = {g: j for j, g in enumerate(groups)}
    counts = np.zeros((len(groups), len(METHODS), 3), np.int64)
    for row in rows:
        for j, name in enumerate(METHODS):
            y, pred = row['gold'], row['predictions'][name]
            counts[gi[row['group_id']], j] += [int(y == 1 and pred), int(y == 0 and pred), int(y == 1 and not pred)]
    samples = np.random.default_rng(SEED).integers(0, len(groups), (2000, len(groups)))
    frequencies = np.stack([np.bincount(s, minlength=len(groups)) for s in samples])
    total = np.einsum('bg,gmc->bmc', frequencies, counts, optimize=True)
    tp, fp, fn = (total[:, :, j] for j in range(3)); f1 = 2*tp/(2*tp+fp+fn)
    return {'groups': len(groups), 'draws': 2000, 'seed': SEED,
            'note': 'Fixed fitted CV models and fold-specific calibration thresholds; does not include refitting uncertainty or correct selection among variants.',
            'f1': {name: r10.ci(f1[:, j]) for j, name in enumerate(METHODS)},
            'f1_difference_from_lb_c01': {name: r10.ci(f1[:, j]-f1[:, 0]) for j, name in enumerate(METHODS) if j}}


def run():
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    assert not (out/'complete13.json').exists(), 'Diagnostic already completed; do not retune'
    snap = snapshot(); r10.save(out/'source_snapshot13.json', snap)
    r10.save(out/'started13.json', {'utc': r10.utc(), 'source_snapshot_sha256': r10.sha(out/'source_snapshot13.json'),
                                  'original_val_or_test_labels_used': False})
    meta = r10.metadata(SOURCE); _, tokens, _, coverage = r10.cohort(SOURCE, 'train', meta)
    rows = [r for r in tokens if r['main_eligible']]
    assert len(rows) == 3854
    folds = json.loads((PREVIOUS/'results/fold_assignment.json').read_text('utf-8'))
    bank = r12.r11.Bank(SOURCE, meta[2]); x = bank.token_matrix(rows, 'lb')
    for rid in {r['row_id'] for r in rows}:
        side = json.loads((SOURCE/'data/features'/(rid+'.json')).read_text('utf-8'))
        assert side['lookback_axes'] == 'token x layer-major/head-minor'
    y = np.asarray([r['gold'] for r in rows], int)
    ids = np.asarray([r['group_id'] for r in rows])
    all_models, fold_metrics, output = {}, {}, []
    assigned = np.zeros(len(rows), int); begin = time.perf_counter()
    for fold in range(5):
        eval_groups = sorted(g for g, k in folds.items() if k == fold)
        other = sorted(set(folds)-set(eval_groups))
        ordered = [str(g) for g in np.random.default_rng(SEED+fold).permutation(other)]
        cal_groups, fit_groups = ordered[:24], ordered[24:]
        assert [len(eval_groups), len(cal_groups), len(fit_groups)] == [24, 24, 72]
        assert not set(eval_groups) & (set(cal_groups) | set(fit_groups))
        ev = np.flatnonzero(np.isin(ids, eval_groups)); ca = np.flatnonzero(np.isin(ids, cal_groups))
        full = np.flatnonzero(np.isin(ids, fit_groups)); assigned[ev] += 1
        model_set = {}; detail = {}; pca_cache = {}
        predicted = {int(j): {**{k: rows[j][k] for k in ['token_key', 'token_index', 'token_id', 'row_id', 'question_id',
                                                       'group_id', 'condition', 'item_ids', 'start', 'end', 'text',
                                                       'gold', 'attribute_group']},
                              'fold': fold, 'scores': {}, 'predictions': {}} for j in ev}
        for method in METHODS:
            selected_groups = fit_groups[:24] if method == 'fit24' else fit_groups[:48] if method == 'fit48' else fit_groups
            ix = np.flatnonzero(np.isin(ids, selected_groups))
            tr = [rows[j] for j in ix]
            feature = method if method in ('pca16', 'pca64', 'layer_mean28') else 'lb'
            model = fit_model(x[ix], tr, feature, CS[method], len(full), pca_cache)
            train_scores, cal_scores, eval_scores = [probability(model, x[indices]) for indices in (ix, ca, ev)]
            chosen = r10.threshold_search(y[ca], cal_scores); threshold = chosen['threshold']
            assert threshold is not None
            oracle = r10.threshold_search(y[ev], eval_scores)
            model.update(fit_indices=ix, threshold=threshold, threshold_selection=chosen)
            model_set[method] = model
            detail[method] = {'fit_groups': len(selected_groups), 'C': CS[method], 'features': model['model'].coef_.shape[1],
                              'pca_explained_variance': model['pca_explained_variance'], 'threshold': threshold,
                              'train': count(y[ix], train_scores, threshold),
                              'calibration': count(y[ca], cal_scores, threshold),
                              'outer': count(y[ev], eval_scores, threshold),
                              'posthoc_outer_oracle_not_deployable': {**oracle, 'metrics': count(y[ev], eval_scores, oracle['threshold'])}}
            for j, score in zip(ev, eval_scores):
                predicted[int(j)]['scores'][method] = float(score)
                predicted[int(j)]['predictions'][method] = bool(score >= threshold)
        all_models[fold] = {'fit_groups': fit_groups, 'calibration_groups': cal_groups, 'evaluation_groups': eval_groups,
                            'calibration_indices': ca, 'evaluation_indices': ev, 'models': model_set}
        fold_metrics[str(fold)] = detail
        output.extend(predicted.values())
        print(f'FOLD {fold+1}/5 complete', flush=True)
    assert (assigned == 1).all() and len(output) == len(rows)
    output.sort(key=lambda r: r['token_key'])
    pooled, fold_average = {}, {}
    for method in METHODS:
        pooled[method] = count([r['gold'] for r in output], [r['scores'][method] for r in output],
                               predictions=[r['predictions'][method] for r in output])
        fold_average[method] = {stage: {metric: float(np.mean([fold_metrics[str(f)][method][stage][metric] for f in range(5)]))
                                       for metric in ['f1', 'auroc', 'average_precision', 'alert_rate', 'risk_rate']}
                                for stage in ['train', 'calibration', 'outer']}
    elapsed = time.perf_counter()-begin
    result = {'schema': 'round13-training-source-diagnostics-v1', 'coverage': coverage, 'scope': 'train-only diagnostic CV',
              'methods_fixed_before_run': METHODS, 'folds': fold_metrics, 'pooled': pooled,
              'fold_mean': fold_average, 'ranking_note': 'Prefer foldwise AUROC/AP when comparing variants; pooled ranking mixes five fitted score scales.',
              'localization_diagnostics': diagnostics(output), 'paired_bootstrap': bootstrap(output),
              'fit_and_score_seconds': elapsed}
    assert snapshot() == snap
    (out/'fitted_folds.pkl').write_bytes(pickle.dumps(all_models, protocol=5))
    r10.savel(out/'oof_scores.jsonl', output); r10.save(out/'summary.json', result)
    r10.save(out/'complete13.json', {'utc': r10.utc(), 'original_val_or_test_labels_used': False,
        'fits': 40, 'test_retuning': False,
        'files_sha256': {name: r10.sha(out/name) for name in ['source_snapshot13.json', 'fitted_folds.pkl', 'oof_scores.jsonl', 'summary.json']}})
    print('ROUND13_DIAGNOSTIC_COMPLETE', flush=True)


def smoke():
    rows = [{'group_id': 'g'+str(g), 'condition': 'c'+str(c), 'item_ids': [f'i{g}{c}'], 'gold': int(j == 1)}
            for g in range(3) for c in range(2) for j in range(3)]
    b, w, _ = weights(rows, 123.)
    assert np.isclose(w.sum(), 123.)
    for g in range(3):
        assert np.isclose(w[[r['group_id'] == 'g'+str(g) for r in rows]].sum(), 41.)
    fake = np.arange(2*784, dtype=np.float32).reshape(2, 784)
    assert np.allclose(fake.reshape(2, 28, 28).mean(2)[0], np.arange(28)*28+13.5)
    m = count([0, 1, 0, 1], [.1, .2, .6, .9], .5)
    assert [m[k] for k in ['tp', 'fp', 'fn', 'tn']] == [1, 1, 1, 1]
    print('ROUND13_SMOKE_PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['smoke', 'run'])
    args = parser.parse_args()
    with r10.threadpool_limits(limits=4):
        {'smoke': smoke, 'run': run}[args.stage]()
