"""Frozen, group-cross-fitted score/NLL interaction; exploratory R10 retest."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'round11_logprob'
spec = importlib.util.spec_from_file_location('r12_reuses_r11', OLD/'src/run11.py')
r11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r11)
r10, np = r11.r10, r11.np
SOURCE, STATUS = r11.SOURCE, r11.STATUS
METHODS = ('lb', 'lb_nll', 'score_only', 'additive', 'interaction')
FUSION = METHODS[2:]
CONTRASTS = {'interaction_minus_additive': ('interaction', 'additive'),
             'interaction_minus_lb': ('interaction', 'lb'),
             'interaction_minus_lb_nll': ('interaction', 'lb_nll'),
             'additive_minus_score_only': ('additive', 'score_only')}


@contextmanager
def method_scope(methods=METHODS):
    names = ('FEATURES', 'ITEM_METHODS', 'TOKEN_METHODS', 'BROADCAST', 'LOCALIZATION_METHODS')
    old = {k: getattr(r10, k) for k in names}
    r10.FEATURES = tuple(methods)
    r10.ITEM_METHODS = tuple(m+'__item' for m in methods)
    r10.TOKEN_METHODS = tuple(m+'__token' for m in methods)
    r10.BROADCAST = {}; r10.LOCALIZATION_METHODS = r10.TOKEN_METHODS
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(r10, k, v)


def snapshot():
    protocol = json.loads((ROOT/'protocol.json').read_text('utf-8'))
    assert protocol['methods'] == list(METHODS)
    assert protocol['C_candidates'] == list(r10.LR_C)
    assert protocol['base_C'] == .1 and protocol['folds'] == 5 and protocol['seed'] == r10.SEED
    assert protocol['contrasts'] == list(CONTRASTS)
    files = [ROOT/'PLAN.md', ROOT/'protocol.json', *sorted((ROOT/'src').glob('*.py'))]
    previous = r11.source_snapshot()
    assert previous == json.loads((OLD/'results/source_snapshot11.json').read_text('utf-8'))
    return {'round11_frozen_graph': previous,
            'old_results_sha256': {p.name: r10.sha(p) for p in (OLD/'results').glob('*')
                                  if p.name in ('freeze11.json', 'selection.json', 'frozen_models.pkl',
                                                'answer_scores_test.jsonl', 'token_scores_test.jsonl')},
            'local_files_sha256': {p.relative_to(ROOT).as_posix(): r10.sha(p) for p in files}}


def split_groups(items):
    groups = sorted({row['group_id'] for row in items})
    assert len(groups) == 120
    shuffled = np.random.default_rng(r10.SEED).permutation(groups)
    return {str(g): k for k, chunk in enumerate(np.array_split(shuffled, 5)) for g in chunk}


def lr_fit(x, rows, granularity, c):
    y = np.asarray([row['gold'] for row in rows], int)
    b = r10.base_weights(rows, granularity)
    w, factors = r10.loss_weights(rows, y, b)
    scaler = r10.StandardScaler().fit(x, sample_weight=b)
    classifier = r10.LogisticRegression(C=c, penalty='l2', solver='liblinear',
                                        max_iter=2000, random_state=r10.SEED)
    classifier.fit(scaler.transform(x).astype(np.float32), y, sample_weight=w)
    return {'scaler': scaler, 'model': classifier, 'kind': 'lr', 'feature': 'lb',
            'granularity': granularity, 'C': c}, b, w, factors


def base_logit(model, x):
    return model['model'].decision_function(model['scaler'].transform(x).astype(np.float32))


def fusion_x(feature, scaler, product_scaler, a, n):
    z = scaler.transform(np.column_stack((a, n)))
    if feature == 'score_only':
        return z[:, :1]
    if feature == 'additive':
        return z
    assert feature == 'interaction'
    product = product_scaler.transform((z[:, 0]*z[:, 1]).reshape(-1, 1))
    return np.column_stack((z, product))


def moments(a, weights):
    weights = weights / weights.sum()
    mean = float(np.sum(a*weights))
    return {'weighted_mean': mean, 'weighted_sd': float(np.sqrt(np.sum(weights*(a-mean)**2))),
            'min': float(a.min()), 'max': float(a.max())}


def fit_fusion(bank, train_items, train_tokens, val_items, val_tokens, folds, out):
    models, thresholds, choices, audit, all_fold_models, drift = {}, {}, {}, {}, {}, {}
    for gran, train, val in [('item', train_items, val_items), ('token', train_tokens, val_tokens)]:
        tr = [row for row in train if row['main_eligible']]
        va = [row for row in val if row['main_eligible']]
        matrix = bank.item_matrix if gran == 'item' else bank.token_matrix
        x, n = matrix(tr, 'lb'), matrix(tr, 'nll')[:, 0]
        vx, vn = matrix(va, 'lb'), matrix(va, 'nll')[:, 0]
        y = np.asarray([row['gold'] for row in tr], int)
        vy = np.asarray([row['gold'] for row in va], int)
        fids = np.asarray([folds[row['group_id']] for row in tr], int)
        a = np.full(len(tr), np.nan); seen = np.zeros(len(tr), int)
        fold_models = []; fold_audit = []
        for k in range(5):
            fit_ix, held_ix = np.flatnonzero(fids != k), np.flatnonzero(fids == k)
            fit_rows = [tr[j] for j in fit_ix]
            base, fb, fw, factors = lr_fit(x[fit_ix], fit_rows, gran, .1)
            a[held_ix] = base_logit(base, x[held_ix]); seen[held_ix] += 1
            train_groups = sorted({row['group_id'] for row in fit_rows})
            held_groups = sorted({tr[j]['group_id'] for j in held_ix})
            assert not set(train_groups) & set(held_groups)
            fold_models.append(base)
            fold_audit.append({'fold': k, 'training_groups': train_groups, 'heldout_groups': held_groups,
                               'training_rows': len(fit_ix), 'heldout_rows': len(held_ix),
                               'training_class_factors': factors})
        assert (seen == 1).all() and np.isfinite(a).all()
        full_base, b, w, factors = lr_fit(x, tr, gran, .1)
        av = base_logit(full_base, vx)
        vweights = r10.base_weights(va, gran)
        scaler = r10.StandardScaler().fit(np.column_stack((a, n)), sample_weight=b)
        z = scaler.transform(np.column_stack((a, n)))
        product_scaler = r10.StandardScaler().fit((z[:, 0]*z[:, 1]).reshape(-1, 1), sample_weight=b)
        all_fold_models[gran] = fold_models
        ids = [row['item_id'] if gran == 'item' else row['token_key'] for row in tr]
        assert len(set(ids)) == len(ids)
        np.savez_compressed(out/f'oof_{gran}.npz', a=a, n=n, y=y, fold_ids=fids,
                            base_weights=b, loss_weights=w, row_ids=np.asarray(ids),
                            group_ids=np.asarray([row['group_id'] for row in tr]))
        audit[gran] = {'folds': fold_audit, 'rows': len(tr), 'groups': len(set(folds)),
                       'each_row_heldout_once': True, 'class_factors_full_train': factors,
                       'base_weight_sum': float(b.sum()),
                       'group_loss_mass': {g: float(w[[r['group_id'] == g for r in tr]].sum()) for g in sorted(folds)},
                       'scaler_uses': 'OOF train features with label-independent base weights',
                       'meta_loss_uses': 'full-training class and group weights'}
        drift[gran] = {'train_oof': moments(a, b), 'train_in_fit_descriptive_only': moments(base_logit(full_base, x), b),
                       'validation_full_refit': moments(av, vweights),
                       'validation_individual_fold_models': [moments(base_logit(m, vx), vweights) for m in fold_models],
                       'note': 'No train-in-fit logits used to train fusion; drift is descriptive, not used to retune deployment.'}
        for feature in FUSION:
            name = feature+'__'+gran
            xx = fusion_x(feature, scaler, product_scaler, a, n)
            vv = fusion_x(feature, scaler, product_scaler, av, vn)
            choices[name] = []; best = None
            for c in r10.LR_C:
                classifier = r10.LogisticRegression(C=c, penalty='l2', solver='liblinear',
                                                    max_iter=2000, random_state=r10.SEED)
                classifier.fit(xx, y, sample_weight=w)
                entry = r10.threshold_search(vy, classifier.predict_proba(vv)[:, 1])
                entry.update(C=c, validation_unit=gran)
                choices[name].append(entry)
                key = (entry['validation_f1'], entry['validation_precision'], -c)
                model = {'kind': 'conditional', 'feature': feature, 'granularity': gran, 'C': c,
                         'base': full_base, 'scaler': scaler, 'product_scaler': product_scaler,
                         'model': classifier}
                if best is None or key > best[0]:
                    best = key, model, entry
            models[name], thresholds[name] = best[1:]
    return models, thresholds, choices, audit, all_fold_models, drift


def predict(bank, items, tokens, models):
    for name, model in models.items():
        rows = items if model['granularity'] == 'item' else tokens
        matrix = bank.item_matrix if model['granularity'] == 'item' else bank.token_matrix
        if model['kind'] == 'lr':
            values = r10.score_array(model, matrix(rows, model['feature']))
        else:
            x, n = matrix(rows, 'lb'), matrix(rows, 'nll')[:, 0]
            good = np.isfinite(x).all(axis=1) & np.isfinite(n)
            values = np.full(len(rows), np.nan)
            a = base_logit(model['base'], x[good])
            xx = fusion_x(model['feature'], model['scaler'], model['product_scaler'], a, n[good])
            values[good] = model['model'].predict_proba(xx)[:, 1]
        for row, value in zip(rows, values):
            row['scores'][name] = float(value) if np.isfinite(value) else None


def metrics(items, tokens, regions, thresholds):
    with method_scope():
        result = r10.evaluate(items, tokens, regions, thresholds, with_bootstrap=False)
    result['primary_method_fixed_before_test'] = 'interaction'
    result['primary_comparator_fixed_before_test'] = 'additive'
    result.pop('whitebox_increment_baseline', None)
    result['comparison_status'] = STATUS
    for value in result['localization_methods'].values():
        rank = value['within_answer_ranking']
        rank['per_answer'] = {rid: {k: m[k] for k in ('auroc', 'average_precision', 'ranking_tokens')}
                              for rid, m in rank['per_answer'].items()}
    return result


def bootstrap(items, tokens, thresholds):
    groups = sorted({r['group_id'] for r in items}); gi = {g: j for j, g in enumerate(groups)}
    samples = np.random.default_rng(r10.SEED).integers(0, len(groups), (2000, len(groups)))
    weights = np.stack([np.bincount(s, minlength=len(groups)) for s in samples])
    out = {'groups': len(groups), 'draws': 2000, 'seed': r10.SEED,
           'unit': 'whole question group, both conditions and all tokens', 'subsets': {}}
    for subset, rows, suffix in [('answer_items', items, '__item'), ('all_resolved_items', tokens, '__token')]:
        names = [m+suffix for m in METHODS]; counts = np.zeros((len(groups), len(names), 3), np.int64)
        for row in rows:
            if not row['main_eligible']:
                continue
            for j, name in enumerate(names):
                score = row['scores'][name]; assert r10.metric.finite(score)
                pred, y = score >= thresholds[name]['threshold'], row['gold']
                counts[gi[row['group_id']], j] += [int(pred and y == 1), int(pred and y == 0), int(not pred and y == 1)]
        summed = np.einsum('bg,gmc->bmc', weights, counts, optimize=True)
        tp, fp, fn = (summed[:, :, k] for k in range(3))
        def divide(a, b, ok):
            return np.divide(a, b, out=np.full(a.shape, np.nan, float), where=ok)
        arrays = {'precision': divide(tp, tp+fp, tp+fp > 0), 'recall': divide(tp, tp+fn, tp+fn > 0),
                  'f1': divide(2*tp, 2*tp+fp+fn, tp+fn > 0)}
        out['subsets'][subset] = {
            'methods': {name: {k: r10.ci(v[:, j]) for k, v in arrays.items()} for j, name in enumerate(names)},
            'contrasts': {label: {k: r10.ci(v[:, METHODS.index(a)]-v[:, METHODS.index(b)]) for k, v in arrays.items()}
                          for label, (a, b) in CONTRASTS.items()}}
    return out


def fit():
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    assert not (out/'freeze12.json').exists(), 'Do not overwrite a frozen fit'
    snap = snapshot(); r10.save(out/'source_snapshot12.json', snap)
    r10.save(out/'fit_started12.json', {'utc': r10.utc(), 'source_snapshot_sha256': r10.sha(out/'source_snapshot12.json')})
    meta = r10.metadata(SOURCE)
    ti, tt, _, tc = r10.cohort(SOURCE, 'train', meta)
    vi, vt, vr, vc = r10.cohort(SOURCE, 'validation', meta)
    folds = split_groups(ti); r10.save(out/'fold_assignment.json', folds)
    bank = r11.Bank(SOURCE, meta[2]); begin = time.perf_counter()
    with method_scope(METHODS[:2]):
        models, thresholds, candidates, old_weights = r10.fit_models(bank, ti, tt, vi, vt)
    mm, th, cc, aa, fold_models, drift = fit_fusion(bank, ti, tt, vi, vt, folds, out)
    models.update(mm); thresholds.update(th); candidates.update(cc)
    predict(bank, vi, vt, models)
    elapsed = time.perf_counter()-begin
    assert snapshot() == snap
    (out/'frozen_models.pkl').write_bytes(pickle.dumps(models, protocol=5))
    (out/'fold_models.pkl').write_bytes(pickle.dumps(fold_models, protocol=5))
    r10.save(out/'selection.json', candidates)
    r10.save(out/'training_weights.json', {'baselines': old_weights, 'fusion': aa})
    r10.save(out/'distribution_diagnostic.json', drift)
    r10.save(out/'validation_metrics.json', {'coverage': vc, **metrics(vi, vt, vr, thresholds)})
    r10.savel(out/'answer_scores_validation.jsonl', vi); r10.savel(out/'token_scores_validation.jsonl', vt)
    files = ['source_snapshot12.json', 'frozen_models.pkl', 'fold_models.pkl', 'selection.json',
             'training_weights.json', 'fold_assignment.json', 'oof_item.npz', 'oof_token.npz',
             'distribution_diagnostic.json', 'validation_metrics.json',
             'answer_scores_validation.jsonl', 'token_scores_validation.jsonl']
    r10.save(out/'freeze12.json', {'schema': 'round12-conditional-fusion-v1', 'utc': r10.utc(),
        'comparison_status': STATUS, 'test_labels_used_in_fit': False, 'primary_method': 'interaction',
        'primary_comparator': 'additive', 'thresholds': thresholds, 'methods': METHODS,
        'train_coverage': tc, 'validation_coverage': vc, 'fit_and_validation_predict_seconds': elapsed,
        'files_sha256': {name: r10.sha(out/name) for name in files}})
    print('ROUND12_FIT_FROZEN', flush=True)


def test():
    out = ROOT/'results'
    assert not (out/'test_complete12.json').exists(), 'Test scoring already completed'
    frozen = json.loads((out/'freeze12.json').read_text('utf-8'))
    for name, digest in frozen['files_sha256'].items():
        assert r10.sha(out/name) == digest, name
    snap = json.loads((out/'source_snapshot12.json').read_text('utf-8'))
    assert snapshot() == snap
    start = {'freeze12_sha256': r10.sha(out/'freeze12.json'), 'comparison_status': STATUS, 'retuning_allowed': False}
    if (out/'test_started12.json').exists():
        assert json.loads((out/'test_started12.json').read_text('utf-8')) == start
    else:
        r10.save(out/'test_started12.json', start)
    meta = r10.metadata(SOURCE)
    items, tokens, regions, coverage = r10.cohort(SOURCE, 'test', meta)
    models = pickle.loads((out/'frozen_models.pkl').read_bytes())
    bank = r11.Bank(SOURCE, meta[2]); begin = time.perf_counter()
    predict(bank, items, tokens, models)
    elapsed = time.perf_counter()-begin
    result = {'schema': 'round12-conditional-fusion-v1', **start, 'coverage': coverage,
              'test_load_and_predict_seconds': elapsed, **metrics(items, tokens, regions, frozen['thresholds']),
              'group_bootstrap': bootstrap(items, tokens, frozen['thresholds'])}
    assert snapshot() == snap
    r10.save(out/'metrics_test.json', result)
    r10.savel(out/'answer_scores_test.jsonl', items); r10.savel(out/'token_scores_test.jsonl', tokens)
    r10.save(out/'test_complete12.json', {**start, 'utc': r10.utc(), 'models': len(models),
        'files_sha256': {name: r10.sha(out/name) for name in ['metrics_test.json', 'answer_scores_test.jsonl', 'token_scores_test.jsonl']}})
    print('ROUND12_EXPLORATORY_TEST_COMPLETE', flush=True)


def smoke():
    a, n, weights = np.array([-3., -1., 2., 5.]), np.array([1., 4., 0., 2.]), np.array([1., 2., 3., 4.])
    scaler = r10.StandardScaler().fit(np.column_stack((a, n)), sample_weight=weights)
    z = scaler.transform(np.column_stack((a, n)))
    ps = r10.StandardScaler().fit((z[:, 0]*z[:, 1]).reshape(-1, 1), sample_weight=weights)
    features = fusion_x('interaction', scaler, ps, a, n)
    assert features.shape == (4, 3)
    assert np.allclose(np.average(features, axis=0, weights=weights), 0, atol=1e-12)
    assert np.allclose(np.average(features**2, axis=0, weights=weights), 1, atol=1e-12)
    assert np.array_equal(fusion_x('additive', scaler, ps, a, n), features[:, :2])
    assert np.array_equal(fusion_x('score_only', scaler, ps, a, n), features[:, :1])
    original = r10.FEATURES
    with method_scope():
        assert r10.FEATURES == METHODS
    assert r10.FEATURES == original
    print('ROUND12_SMOKE_PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['smoke', 'fit', 'test'])
    args = parser.parse_args()
    with r10.threadpool_limits(limits=4):
        {'smoke': smoke, 'fit': fit, 'test': test}[args.stage]()
