"""Independent read-only R32 model/calibration/outer replay; no fitting or GPU."""
from pathlib import Path
import importlib.util
import pickle
import time
import traceback
from collections import defaultdict
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'
PRELAB = ROOT.parent
OLD = {'base769': PRELAB/'round29_full_hidden_scene_adaptation/results',
       'large1025': PRELAB/'round31_large_semantic_scene_adaptation/results'}
WIDTH = {'base769': 769, 'large1025': 1025}
CS = (.001, .01, .1)
ALPHAS = (0., .2, .4, .6, .8, 1.)
helper_path = OLD['large1025']/'audit31.py'
spec = importlib.util.spec_from_file_location('r32_independent_geometry_metrics', helper_path)
ind = importlib.util.module_from_spec(spec); spec.loader.exec_module(ind)
BASES, FUSED, METHODS = ind.BASES, ind.FUSED, ind.METHODS


def combine(prior, semantic, alpha):
    if alpha == 0:
        return np.asarray(prior, np.float64).copy()
    p, s = np.clip(prior, 1e-12, 1-1e-12), np.clip(semantic, 1e-12, 1-1e-12)
    return expit(np.log(p)-np.log1p(-p)+alpha*(np.log(s)-np.log1p(-s)))


def choice_key(thresholds, c, alpha=None):
    w, a = thresholds['window'], thresholds['answer']
    result = [min(w['validation_f1'], a['validation_f1']), w['validation_f1'], w['validation_precision']]
    if alpha is not None:
        result.append(-alpha)
    return result+[-c]


def replay(model, x, windows, groups, c):
    ix = np.asarray([i for i, w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups['fit_groups']])
    rows = [windows[i] for i in ix]
    assert model['C'] == c and model['width'] == x.shape[1] and model['loss_mass'] == 3854
    assert np.array_equal(ix, model['fit_ix']) and np.array_equal(model['fit_y'], [w['gold'] for w in rows])
    assert model['fit_keys'] == [w['window_key'] for w in rows]
    assert model['fit_groups'] == sorted({w['group_id'] for w in rows})
    assert not set(model['fit_groups']) & (set(groups['calibration_groups']) | set(groups['evaluation_groups']))
    b, loss, factors = ind.weights(rows)
    for key, expected in [('base_weights', b), ('loss_weights', loss), ('class_factors', factors)]:
        np.testing.assert_allclose(model[key], expected, rtol=1e-12, atol=1e-12)
    scaler = model['scaler']; weights = b.astype(np.float32)
    mean = np.average(x[ix].astype(np.float64), axis=0, weights=weights)
    variance = np.average((x[ix].astype(np.float64)-mean)**2, axis=0, weights=weights)
    np.testing.assert_allclose(scaler.mean_, mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scaler.var_, variance, rtol=1e-10, atol=1e-10)
    assert abs(float(scaler.n_samples_seen_)-weights.astype(np.float64).sum()) < 1e-7
    lr = model['model']
    assert lr.C == c and lr.solver == 'liblinear' and lr.random_state == 20261008 and lr.max_iter == 2000
    assert lr.penalty == 'l2' and np.array_equal(lr.classes_, [0, 1]) and int(lr.n_iter_.max()) < 2000
    assert np.isfinite(lr.coef_).all() and np.isfinite(lr.intercept_).all()
    transformed = x.copy(); transformed -= scaler.mean_; transformed /= scaler.scale_
    actual = expit(transformed @ lr.coef_[0]+lr.intercept_[0])
    library = lr.predict_proba(scaler.transform(x).astype(np.float32))[:, 1]
    assert np.array_equal(actual, library)
    return actual, {'fit_windows': len(ix), 'base_mass': float(b.sum()), 'loss_mass': float(loss.sum()),
                    'width': x.shape[1], 'C': c, 'iterations': int(lr.n_iter_.max()),
                    'fit_only_groups_weights_scaler_checked': True}


def run():
    assert (OUT/'complete.json').exists() and (OUT/'fit_complete.json').exists()
    assert not (OUT/'INDEPENDENT_AUDIT32.json').exists()
    started = time.perf_counter()
    complete = ind.read(OUT/'complete.json'); fit = ind.read(OUT/'fit_complete.json')
    assert not complete['human_gold'] and not complete['original_validation_test_opened'] and not complete['public_QA_test_opened']
    assert fit['status'] == 'all10_family_folds_frozen_before_outer' and fit['new_LR_fits'] == 20 and fit['reused_LR_models'] == 10
    ind.check_hashes(OUT, complete['files_sha256']); ind.check_hashes(OUT, fit['files_sha256'])
    for path, expected in fit['source_sha256'].items():
        assert ind.sha(path) == expected
    prepared = ind.read(OUT/'preparation_complete.json')
    assert prepared['source_sha256'] == fit['source_sha256']
    assert prepared['check_sha256'] == ind.sha(OUT/'CPU_SELFCHECK.json')
    windows, answers, folds, by = ind.geometry()
    wy = np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows])
    ay = np.asarray([a['gold'] if a['main_eligible'] else -1 for a in answers])
    assert (int((wy>=0).sum()), int((ay>=0).sum())) == (9526, 598)
    summary = ind.read(OUT/'summary.json'); checks, recomputed, differences = [], {}, {}
    all_tables = 0; reused = 0; new = 0
    for family, previous in OLD.items():
        directory = OUT/family
        ind.check_hashes(directory, ind.read(directory/'complete.json')['files_sha256'])
        ind.check_hashes(directory, ind.read(directory/'fit_complete.json')['files_sha256'])
        with np.load(previous/'designs.npz', allow_pickle=False) as z:
            x = np.column_stack((z['hidden'], z['logit'])).astype(np.float32)
        assert x.shape == (12222, WIDTH[family]) and np.isfinite(x).all()
        wp = {m: np.zeros(len(windows), bool) for m in METHODS}; ap = {m: np.zeros(len(answers), bool) for m in METHODS}
        ws = {m: np.full(len(windows), np.nan) for m in METHODS}; aa = {m: np.full(len(answers), np.nan) for m in METHODS}
        wc, ac = np.zeros(len(windows), int), np.zeros(len(answers), int)
        for fold, groups in enumerate(folds):
            path = directory/f'fold_{fold}_frozen.pkl'
            frozen = pickle.loads(path.read_bytes())
            old = pickle.loads((previous/f'fold_{fold}_frozen.pkl').read_bytes())
            assert frozen['groups'] == old['groups'] == groups and not frozen['human_gold']
            assert frozen['old_control_sha256'] == ind.sha(previous/f'fold_{fold}_frozen.pkl')
            assert set(frozen['models']) == set(CS)
            side = ind.read(directory/f'fold_{fold}_calibration.json')
            assert side == {k: frozen[k] for k in ('groups', 'thresholds', 'calibration_tables', 'selected', 'human_gold', 'old_control_sha256')}
            with np.load(directory/f'fold_{fold}_scores.npz', allow_pickle=False) as z:
                scores = {key: z[key].copy() for key in z.files}
            with np.load(previous/f'fold_{fold}_scores.npz', allow_pickle=False) as z:
                old_scores = {key: z[key].copy() for key in z.files}
            semantic = []
            for j, c in enumerate(CS):
                model = frozen['models'][c]
                actual, detail = replay(model, x, windows, groups, c)
                assert np.array_equal(actual, scores['semantic_all_C'][j])
                if c == .01:
                    reused += 1
                    assert np.array_equal(actual, old_scores['semantic_probe'])
                    for name in ('fit_ix', 'fit_y', 'base_weights', 'loss_weights', 'class_factors'):
                        assert np.array_equal(model[name], old['model'][name])
                    for name in ('coef_', 'intercept_', 'n_iter_'):
                        assert np.array_equal(getattr(model['model'], name), getattr(old['model']['model'], name))
                    for name in ('mean_', 'var_', 'scale_'):
                        assert np.array_equal(getattr(model['scaler'], name), getattr(old['model']['scaler'], name))
                else:
                    new += 1
                ts = ind.thresholds(actual, groups['calibration_groups'], windows, answers, by)
                row = {'C': c, 'thresholds': ts, 'key': choice_key(ts, c)}
                assert row == frozen['calibration_tables']['semantic_probe'][j]
                if c == .01:
                    assert ts == old['thresholds']['semantic_probe']
                semantic.append(actual); all_tables += 1
                checks.append({'family': family, 'fold': fold, 'C': c, 'reused': c==.01,
                               'probability_max_abs_diff': 0., **detail})
            sem_table = frozen['calibration_tables']['semantic_probe']
            best = max(sem_table, key=lambda row: row['key'])
            assert best == frozen['selected']['semantic_probe']
            assert np.array_equal(scores['semantic_probe'], semantic[CS.index(best['C'])])
            assert frozen['thresholds']['semantic_probe'] == best['thresholds']
            for prior, fused in zip(BASES, FUSED):
                assert np.array_equal(scores[prior], old_scores[prior])
                assert frozen['thresholds'][prior] == old['thresholds'][prior]
                table = frozen['calibration_tables'][fused]
                assert len(table) == 18 and scores[fused+'_all_C_alpha'].shape == (18, 12222)
                for ci, c in enumerate(CS):
                    for ai, alpha in enumerate(ALPHAS):
                        j = ci*6+ai
                        values = combine(scores[prior], semantic[ci], alpha)
                        assert np.array_equal(values, scores[fused+'_all_C_alpha'][j])
                        if c == .01:
                            assert np.array_equal(values, old_scores[fused+'_all_alpha'][ai])
                        if alpha == 0:
                            assert np.array_equal(values, scores[prior])
                        ts = ind.thresholds(values, groups['calibration_groups'], windows, answers, by)
                        row = {'C': c, 'alpha': alpha, 'thresholds': ts, 'key': choice_key(ts, c, alpha)}
                        assert row == table[j]; all_tables += 1
                        if c == .01:
                            assert ts == old['alpha_tables'][fused][ai]['thresholds']
                j = max(range(18), key=lambda j: table[j]['key'])
                assert frozen['selected'][fused] == table[j]
                assert frozen['thresholds'][fused] == table[j]['thresholds']
                assert np.array_equal(scores[fused], scores[fused+'_all_C_alpha'][j])
            evaluation = set(groups['evaluation_groups'])
            wi = np.asarray([i for i, w in enumerate(windows) if w['group_id'] in evaluation])
            ai = np.asarray([i for i, a in enumerate(answers) if a['group_id'] in evaluation])
            wc[wi] += 1; ac[ai] += 1
            for method in METHODS:
                av = ind.answer_max(scores[method], answers, by)
                ws[method][wi] = scores[method][wi]; aa[method][ai] = av[ai]
                wp[method][wi] = scores[method][wi] >= frozen['thresholds'][method]['window']['threshold']
                ap[method][ai] = av[ai] >= frozen['thresholds'][method]['answer']['threshold']
        assert np.all(wc==1) and np.all(ac==1)
        window_records = ind.lines(directory/'window_scores_oof.jsonl')
        answer_records = ind.lines(directory/'answer_scores_oof.jsonl')
        assert len(window_records)==12222 and len(answer_records)==602
        final = {}
        for method in METHODS:
            assert np.isfinite(ws[method]).all() and np.isfinite(aa[method]).all()
            assert np.array_equal(ws[method], [w['scores'][method] for w in window_records])
            assert np.array_equal(wp[method], [w['predictions'][method] for w in window_records])
            assert np.array_equal(aa[method], [a['scores'][method] for a in answer_records])
            assert np.array_equal(ap[method], [a['predictions'][method] for a in answer_records])
            final[method] = {'windows': ind.count(wy[wy>=0], wp[method][wy>=0]),
                'answers': ind.count(ay[ay>=0], ap[method][ay>=0]),
                'safe_refusal_false_positives': sum(bool(ap[method][i]) for i, a in enumerate(answers) if a['reviewed_safe_refusal'])}
        assert final == ind.read(directory/'summary.json')['methods'] == summary['methods'][family]
        recomputed[family] = final
        control = ind.read(previous/'summary.json')['methods']
        delta = {method: {unit: {'fixed_C001': control[method][unit]['f1'], 'C_selected': final[method][unit]['f1'],
                 'difference': final[method][unit]['f1']-control[method][unit]['f1']}
                 for unit in ('windows', 'answers')} for method in METHODS}
        assert delta == summary['fixed_C001_comparison'][family]
        differences[family] = delta
    assert (new, reused, all_tables) == (20, 10, 390)
    result = {'passed': True, 'new_models_replayed': new, 'old_models_reused_and_replayed': reused,
        'calibration_entries_independently_checked': all_tables, 'model_checks': checks,
        'all_probabilities_exact': True, 'all390_thresholds_and_choices_exact': True,
        'all_C001_control_probabilities_weights_scalers_and120_fusions_exact': True,
        'outer_each_candidate_window_and_answer_once': True, 'answer_max_uses_all12222_candidates': True,
        'eligible_windows': 9526, 'eligible_answers': 598, 'safe_refusals': 117, 'unresolved_answers_excluded': 4,
        'recomputed_methods': recomputed, 'fixed_C001_comparison': differences,
        'human_gold': False, 'new_fits': 0, 'GPU_used': False,
        'original_validation_test_opened': False, 'public_QA_test_opened': False,
        'complete_sha256': ind.sha(OUT/'complete.json'), 'fit_complete_sha256': ind.sha(OUT/'fit_complete.json'),
        'audit_code_sha256': ind.sha(Path(__file__)), 'independent_helper_sha256': ind.sha(helper_path),
        'seconds': time.perf_counter()-started,
        'limits': 'Repeated R16 training-source development, not human-gold public QA or independent testing. Cross-representation changes are not an isolated size effect.'}
    ind.save(OUT/'INDEPENDENT_AUDIT32.json', result)
    lines = ['# R32 independent audit', '', '20 new and10 reused models replay exactly. All390 calibration entries, selections,9526-window/598-answer counts and fixed-C differences pass. No refitting or GPU.', '',
             '| Input | Method | Window F1 | Answer F1 | Window Δ vs fixed C | Answer Δ vs fixed C |', '|---|---|---:|---:|---:|---:|']
    for family, methods in recomputed.items():
        for method in ('semantic_probe', *FUSED):
            m, d = methods[method], differences[family][method]
            lines.append(f"| {family} | {method} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} | {d['windows']['difference']:+.6f} | {d['answers']['difference']:+.6f} |")
    lines += ['', 'All602 answers remain in candidate geometry;117 reviewed safe refusals retain answer-negative supervision, and4 unresolved answers are excluded only from scored answers. Unknown/refusal windows were not zero-filled. This is repeatedly developed assistant-labelled R16 data, not public human-gold QA.']
    (OUT/'INDEPENDENT_AUDIT32.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('R32_INDEPENDENT_AUDIT_PASSED_20_NEW_10_OLD_390_CHOICES', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        try:
            run()
        except Exception:
            ind.save(OUT/f'AUDIT_FAILURE_{time.time_ns()}.json', {'traceback': traceback.format_exc(), 'new_fits': 0, 'GPU_used': False})
            raise
