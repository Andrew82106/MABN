"""Fixed four-token windows: bounded, training-source overfitting experiments."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / 'round14_window_detection'
spec = importlib.util.spec_from_file_location('r15_reuses_r14', PREVIOUS / 'src/run14.py')
r14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r14)
r13, r10, np, SOURCE = r14.r13, r14.r10, r14.np, r14.SOURCE
METHODS = ('baseline_c01', 'ridge_c003', 'ridge_c001', 'heads64_c01',
           'heads128_c01', 'ensemble_c01', 'ensemble_c003')


def snapshot():
    prior = r14.snapshot()
    assert prior == json.loads((PREVIOUS / 'results/source_snapshot14.json').read_text('utf-8'))
    paths = [ROOT / 'PLAN.md', ROOT / 'protocol.json', *sorted((ROOT / 'src').glob('*.py'))]
    old = ['complete14.json', 'fitted_folds.pkl', 'window_catalog_k4.jsonl', 'window_scores_k4.jsonl']
    return {'prior': prior,
            'prior_results_sha256': {n: r10.sha(PREVIOUS / 'results' / n) for n in old},
            'local_files_sha256': {p.relative_to(ROOT).as_posix(): r10.sha(p) for p in paths}}


def head_order(x, rows):
    b = r10.base_weights(rows, 'token'); b = b / b.sum()
    y = np.asarray([r['gold'] for r in rows], float)
    scaler = r10.StandardScaler().fit(x, sample_weight=b)
    z = scaler.transform(x).astype(np.float64)
    yc = y - np.dot(b, y)
    assert np.dot(b, yc * yc) > 0
    corr = (z.T @ (b * yc)) / np.sqrt(np.dot(b, yc * yc))
    corr[scaler.var_ <= np.finfo(float).eps] = 0.0
    assert np.isfinite(corr).all()
    return np.lexsort((np.arange(x.shape[1]), -np.abs(corr))), corr


def probability(wrapper, x):
    if wrapper['kind'] == 'ensemble':
        return np.mean([probability(m, x) for m in wrapper['members']], axis=0)
    return r13.probability(wrapper['model'], x[:, wrapper['columns']])


def fit_single(matrix, rows, ix, columns, c, target_mass):
    model = r13.fit_model(matrix[ix][:, columns], [rows[j] for j in ix], 'lb', c, target_mass, {})
    return {'kind': 'single', 'columns': np.asarray(columns, int), 'model': model,
            'fit_indices': np.asarray(ix, int), 'fit_groups': sorted({rows[j]['group_id'] for j in ix})}


def bootstrap(rows, config):
    groups = sorted({r['group_id'] for r in rows}); lookup = {g: j for j, g in enumerate(groups)}
    counts = np.zeros((len(groups), len(METHODS), 3), np.int64)
    for row in rows:
        for j, method in enumerate(METHODS):
            p, y = row['predictions'][method], row['gold']
            counts[lookup[row['group_id']], j] += [int(p and y), int(p and not y), int(not p and y)]
    samples = np.random.default_rng(config['seed']).integers(0, len(groups), (config['draws'], len(groups)))
    weights = np.stack([np.bincount(s, minlength=len(groups)) for s in samples])
    total = np.einsum('bg,gmc->bmc', weights, counts, optimize=True)
    tp, fp, fn = (total[:, :, j] for j in range(3)); f1 = 2 * tp / (2 * tp + fp + fn)
    return {'groups': len(groups), **config,
            'note': 'Fixed fitted models and thresholds. No refitting uncertainty or multiple-comparison correction.',
            'f1': {name: r10.ci(f1[:, j]) for j, name in enumerate(METHODS)},
            'difference_from_baseline_f1': {name: r10.ci(f1[:, j] - f1[:, 0]) for j, name in enumerate(METHODS) if j}}


def run():
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    assert not (out / 'complete15.json').exists(), 'Completed experiment is frozen'
    config = json.loads((ROOT / 'protocol.json').read_text('utf-8'))
    assert config['methods'] == list(METHODS) and config['width'] == 4
    assert config['classifier_seed'] == r13.SEED
    snap = snapshot(); r10.save(out / 'source_snapshot15.json', snap)
    r10.save(out / 'started15.json', {'utc': r10.utc(), 'source_snapshot_sha256': r10.sha(out / 'source_snapshot15.json')})
    meta = r10.metadata(SOURCE)
    items, tokens, regions, coverage = r10.cohort(SOURCE, 'train', meta)
    bank = r13.r12.r11.Bank(SOURCE, meta[2])
    catalogs, matrices = r14.catalogs(items, tokens, meta[2], bank)
    rows, matrix = catalogs[4], matrices[4]
    old_catalog = [json.loads(s) for s in (PREVIOUS / 'results/window_catalog_k4.jsonl').read_text('utf-8').splitlines()]
    assert rows == old_catalog
    old_scores = {r['window_key']: r for r in map(json.loads, (PREVIOUS / 'results/window_scores_k4.jsonl').read_text('utf-8').splitlines())}
    y = np.asarray([r['gold'] for r in rows], int)
    ids = np.asarray([r['group_id'] for r in rows])
    old_folds = pickle.loads((PREVIOUS / 'results/fitted_folds.pkl').read_bytes())
    all_models, fold_details, predictions = {}, {}, []
    assigned = np.zeros(len(rows), int); nfit = 0
    begin = time.perf_counter()
    for fold in range(5):
        prior = old_folds[fold]; baseline = prior['models'][4]
        ix, ca, ev = [np.flatnonzero(np.isin(ids, prior[k])) for k in ('fit_groups', 'calibration_groups', 'evaluation_groups')]
        assert all(len(prior[k]) == n for k, n in zip(('fit_groups', 'calibration_groups', 'evaluation_groups'), (72, 24, 24)))
        assert not set(prior['fit_groups']) & (set(prior['calibration_groups']) | set(prior['evaluation_groups']))
        assert not set(prior['calibration_groups']) & set(prior['evaluation_groups'])
        for key, value in [('fit_indices', ix), ('calibration_indices', ca), ('evaluation_indices', ev)]:
            assert np.array_equal(baseline[key], value)
        assigned[ev] += 1
        target_mass = sum(r['group_id'] in prior['fit_groups'] for r in catalogs[1])
        assert baseline['target_loss_mass'] == target_mass
        ranked_heads, correlations = head_order(matrix[ix], [rows[j] for j in ix])
        rng = np.random.default_rng(config['ensemble_seed'] + fold)
        subsets = [sorted(str(g) for g in rng.choice(sorted(prior['fit_groups']), config['ensemble_fit_groups'], replace=False))
                   for _ in range(config['ensemble_members'])]
        subset_indices = [np.flatnonzero(np.isin(ids, ss)) for ss in subsets]
        models, detail, scores, thresholds = {}, {}, {}, {}
        for method in METHODS:
            c = config['C'][method]
            member_scores = None
            if method == 'baseline_c01':
                wrapper = {'kind': 'single', 'columns': np.arange(784), 'model': baseline,
                           'fit_indices': ix, 'fit_groups': sorted(prior['fit_groups'])}
            elif method.startswith('ensemble'):
                members = [fit_single(matrix, rows, sub_ix, np.arange(784), c, target_mass) for sub_ix in subset_indices]
                wrapper = {'kind': 'ensemble', 'members': members, 'C': c}
                member_scores = np.asarray([probability(m, matrix) for m in members])
                nfit += len(members)
            else:
                columns = ranked_heads[:int(method.split('_')[0][5:])] if method.startswith('heads') else np.arange(784)
                wrapper = fit_single(matrix, rows, ix, columns, c, target_mass); nfit += 1
                if method.startswith('heads'):
                    wrapper['selection_correlation'] = correlations
            value = member_scores.mean(axis=0) if member_scores is not None else probability(wrapper, matrix)
            selected = r10.threshold_search(y[ca], value[ca]); threshold = selected['threshold']
            if method == 'baseline_c01':
                assert selected == baseline['thresholds']['window_lr']
                assert all(abs(value[j] - old_scores[rows[j]['window_key']]['scores']['window_lr']) < 1e-12 for j in ev)
            wrapper['threshold'] = threshold; wrapper['threshold_selection'] = selected
            models[method] = wrapper; scores[method] = value; thresholds[method] = threshold
            detail[method] = {'threshold': selected,
                              'fit_pool': r14.count(y[ix], value[ix], threshold),
                              'calibration': r14.count(y[ca], value[ca], threshold),
                              'evaluation': r14.count(y[ev], value[ev], threshold)}
            if member_scores is not None:
                detail[method]['member_metrics_at_ensemble_threshold'] = [
                    {'fit': r14.count(y[sub_ix], member_scores[j, sub_ix], threshold),
                     'evaluation': r14.count(y[ev], member_scores[j, ev], threshold)}
                    for j, sub_ix in enumerate(subset_indices)]
                detail[method]['fit_pool_note'] = 'Contains both in-bag and out-of-bag member predictions; not pure resubstitution.'
        for j in ev:
            predictions.append({**rows[int(j)], 'fold': fold,
                'scores': {name: float(value[j]) for name, value in scores.items()},
                'predictions': {name: bool(value[j] >= thresholds[name]) for name, value in scores.items()}})
        all_models[fold] = {**{k: prior[k] for k in ('fit_groups', 'calibration_groups', 'evaluation_groups')},
                           'models': models, 'head_ranking': ranked_heads, 'head_correlations': correlations,
                           'subsets': subsets, 'target_loss_mass': target_mass}
        fold_details[str(fold)] = detail
        print(f'FOLD {fold+1}/5 complete; new fits={nfit}', flush=True)
    assert np.all(assigned == 1) and len(predictions) == len(rows)
    r14.METHODS = METHODS
    coarse, highlights = r14.highlights(predictions, items, tokens, regions)
    yp = [r['gold'] for r in predictions]
    pooled = {name: r14.count(yp, [r['scores'][name] for r in predictions],
                             predictions=[r['predictions'][name] for r in predictions]) for name in METHODS}
    means = {name: {split: {metric: float(np.mean([fold_details[str(f)][name][split][metric] for f in range(5)]))
                                 for metric in ('f1', 'average_precision', 'auroc')}
                    for split in ('fit_pool', 'calibration', 'evaluation')} for name in METHODS}
    for name in METHODS:
        means[name]['fit_minus_evaluation_f1'] = means[name]['fit_pool']['f1'] - means[name]['evaluation']['f1']
    strata = {cat: {name: r14.count([r['gold'] for r in predictions if r['category'] == cat],
                                   [r['scores'][name] for r in predictions if r['category'] == cat],
                                   predictions=[r['predictions'][name] for r in predictions if r['category'] == cat]) for name in METHODS}
              for cat in sorted({r['category'] for r in predictions})}
    summary = {'schema': config['schema'], 'scope': config['scope'], 'width': 4,
               'coverage': coverage, 'window_count': len(rows), 'positive_windows': int(y.sum()),
               'methods': pooled, 'fold_mean': means, 'folds': fold_details, 'categories': strata,
               'coarse_highlight': coarse, 'paired_bootstrap': bootstrap(predictions, config['bootstrap']),
               'fit_and_score_seconds': time.perf_counter() - begin, 'new_fits': nfit, 'reused_models': 5,
               'original_validation_or_test_labels_used': False}
    assert snapshot() == snap
    (out / 'fitted_folds.pkl').write_bytes(pickle.dumps(all_models, protocol=5))
    r10.savel(out / 'window_scores_k4.jsonl', predictions)
    r10.savel(out / 'highlight_regions.jsonl', highlights)
    r10.save(out / 'summary.json', summary)
    files = ('source_snapshot15.json', 'fitted_folds.pkl', 'window_scores_k4.jsonl', 'highlight_regions.jsonl', 'summary.json')
    r10.save(out / 'complete15.json', {'utc': r10.utc(), 'new_fits': nfit, 'reused_models': 5,
                                      'original_validation_or_test_labels_used': False,
                                      'files_sha256': {n: r10.sha(out / n) for n in files}})
    print('ROUND15_COMPLETE', flush=True)


if __name__ == '__main__':
    with r10.threadpool_limits(limits=4):
        run()
