"""Bounded C grid on two frozen scene representations; no GPU or new data."""
from pathlib import Path
import argparse
import importlib.util
import pickle
import sys
import time
import warnings
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
R29 = ROOT.parent / 'round29_full_hidden_scene_adaptation'
R31 = ROOT.parent / 'round31_large_semantic_scene_adaptation'
sys.path.insert(0, str(R29 / 'src'))
import run29 as old29
r = old29.original
q, core = r.q, r.core
SOURCES = {'base769': R29 / 'results', 'large1025': R31 / 'results'}
WIDTHS = {'base769': 769, 'large1025': 1025}
CS = (.001, .01, .1)
BASES, FUSED, METHODS, ALPHA, SEED = r.BASES, r.FUSED, r.METHODS, r.ALPHA, r.SEED


def frozen_write(path, value):
    if path.exists():
        assert q.read(path) == value, str(path)
    else:
        q.save(path, value)


def protocol():
    return {'version': 'r32-scene-regularization-v1', 'seed': SEED,
        'representations': WIDTHS, 'C': list(CS), 'alpha': list(ALPHA),
        'new_LR_fits': 20, 'reused_LR_models': 10, 'PCA_fits': 0,
        'fit': 'Same eligible fit windows, fit-only base-weighted StandardScaler, original group/class weights and mass3854, liblinear max_iter2000; ConvergenceWarning stops run. No refit on fit+cal.',
        'folds': 'Exact R29/R31 five event-group fit/calibration/evaluation sets; only original R16 train metadata.',
        'choice': 'Semantic: C by min(windowF1,answerF1), windowF1, windowPrecision, smallerC. Each prior: joint C/alpha by same first3 criteria then smallerAlpha then smallerC. One candidate shared by two thresholds.',
        'fusion': 'Exact R28 sigmoid(logit(prior)+alpha*logit(semantic)); clip1e-12; alpha0 exact identity; corresponding R26 fold probabilities.',
        'candidate_counts': {'semantic_per_family_fold': 3, 'per_prior_family_fold': 18,
            'total_calibration_entries': 390, 'new_fusion_scores': 240,
            'reused_fusion_scores': 120},
        'control': 'C=.01 models never refitted; full12222 probability vectors, both thresholds, all6alpha vectors and selected controls must exactly replay.',
        'metrics': '9526 eligible windows and598 eligible answers; answermax ALL12222 candidate windows;117 safe refusals remain answer0;4 unresolved excluded. Refusal windows not zero-filled.',
        'freeze': 'All10 family-fold candidate/choice sets complete before outer metrics. All30 models retained (10 reused), all390 calibration entries retained. No outer selection.',
        'scope': 'Repeated R16 train602/301questions/278groups; human_gold=false. Not independent human-gold verification or proof the target is solved.',
        'limits': 'Prior folds/representations were exposed. Base auxiliary->QA2 vs large QA-only changes backbone, upstream supervision and width. C search adds calibration selection; no generalization guarantee.',
        'GPU_used': False, 'human_gold': False, 'original_validation_test_opened': False,
        'public_QA_test_opened': False}


def source_hashes():
    paths = [Path(__file__), ROOT/'PLAN.md', ROOT/'protocol.json', Path(r.__file__),
        Path(old29.__file__), R31/'src/run31.py', Path(core.__file__),
        Path(core.r13.__file__), Path(core.r10.__file__), Path(r.cache.__file__),
        r.SCENE/'windows.jsonl', r.SCENE/'answers.jsonl', r.SCENE/'folds.json',
        r.R26/'calibration_freeze.json']
    for directory in SOURCES.values():
        paths += [directory/n for n in ('complete.json', 'fit_complete.json', 'summary.json',
            'preparation_complete.json', 'designs.npz')]
        paths += [directory/f'fold_{f}_{suffix}' for f in range(5)
                  for suffix in ('frozen.pkl', 'scores.npz', 'calibration.json')]
    paths += [R29/'results/AUDIT.json', R31/'results/INDEPENDENT_AUDIT31.json']
    paths += [r.R26/f'fold_{f}_{suffix}' for f in range(5) for suffix in ('scores.npz', 'calibration.json')]
    return {str(p.resolve()): q.sha(p) for p in paths}


def design(family):
    with np.load(SOURCES[family]/'designs.npz', allow_pickle=False) as data:
        x = np.column_stack((data['hidden'], data['logit'])).astype(np.float32)
    assert x.shape == (12222, WIDTHS[family]) and np.isfinite(x).all()
    return x


def key(thresholds, c, alpha=None):
    ans = r.selection_key(thresholds, 0 if alpha is None else alpha)
    return ans[:3] + ([] if alpha is None else [ans[3]]) + [-c]


def manual_replay(model, x):
    z = x.copy()
    z -= model['scaler'].mean_
    z /= model['scaler'].scale_
    return expit(z @ model['model'].coef_[0] + model['model'].intercept_[0])


def control(family, fold, x, windows, answers, groups, aw):
    directory = SOURCES[family]
    saved = pickle.loads((directory/f'fold_{fold}_frozen.pkl').read_bytes())
    model = saved['model']
    assert saved['groups'] == groups and model['C'] == .01 and model['width'] == WIDTHS[family]
    ix = np.asarray([i for i, w in enumerate(windows)
                     if w['main_eligible'] and w['group_id'] in set(groups['fit_groups'])])
    assert np.array_equal(ix, model['fit_ix'])
    assert np.array_equal([windows[i]['gold'] for i in ix], model['fit_y'])
    expected = core.r13.weights([windows[i] for i in ix], 3854)
    for name, value in zip(('base_weights', 'loss_weights', 'class_factors'), expected):
        assert np.array_equal(model[name], value), name
    semantic = manual_replay(model, x)
    cal_groups = groups['calibration_groups']
    with np.load(directory/f'fold_{fold}_scores.npz', allow_pickle=False) as data:
        values = {name: data[name].copy() for name in METHODS}
        assert np.array_equal(semantic, values['semantic_probe']), (family, fold, 'probability')
        assert r.thresholds(semantic, cal_groups, windows, answers, aw) == saved['thresholds']['semantic_probe']
        oldcal = q.read(r.R26/f'fold_{fold}_calibration.json')
        with np.load(r.R26/f'fold_{fold}_scores.npz', allow_pickle=False) as baseline:
            for name in BASES:
                assert np.array_equal(values[name], baseline[name])
                assert saved['thresholds'][name] == oldcal['thresholds'][name]
        for prior, fused in zip(BASES, FUSED):
            table = []
            for j, alpha in enumerate(ALPHA):
                v = r.combine(values[prior], semantic, alpha)
                assert np.array_equal(v, data[fused+'_all_alpha'][j])
                ts = r.thresholds(v, cal_groups, windows, answers, aw)
                table.append({'alpha': alpha, 'thresholds': ts, 'key': r.selection_key(ts, alpha)})
            assert table == saved['alpha_tables'][fused]
            chosen = max(table, key=lambda row: row['key'])
            assert chosen == saved['selected_alpha'][fused]
            assert np.array_equal(r.combine(values[prior], semantic, chosen['alpha']), values[fused])
    return saved, values


def prepare():
    assert not (OUT/'preparation_complete.json').exists(), 'Preserve existing preparation'
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_write(ROOT/'protocol.json', protocol())
    snap = source_hashes()
    windows, answers, folds, aw = r.metadata()
    groups = r.group_check(windows, answers, folds)
    # Only scaler fitting on synthetic data; no LR or real-label fitting.
    rng = np.random.default_rng(SEED)
    toy = rng.normal(size=(30, 6)).astype(np.float32)
    weight = np.linspace(.2, 2., 20)
    a = StandardScaler().fit(toy[:20], sample_weight=weight)
    changed = toy.copy(); changed[20:] = np.nan
    b = StandardScaler().fit(changed[:20], sample_weight=weight)
    assert np.array_equal(a.mean_, b.mean_) and np.array_equal(a.var_, b.var_)
    ts = {'window': {'validation_f1': .5, 'validation_precision': .6},
          'answer': {'validation_f1': .7}}
    assert key(ts, .001) > key(ts, .01)
    assert key(ts, .1, 0) > key(ts, .001, .2)
    assert key(ts, .001, .2) > key(ts, .01, .2)
    checks = {}
    for family, directory in SOURCES.items():
        for manifest_name in ('complete.json', 'fit_complete.json'):
            manifest = q.read(directory/manifest_name)
            for name, digest in manifest['files_sha256'].items(): assert q.sha(directory/name) == digest
        prep = q.read(directory/'preparation_complete.json')
        assert prep['designs_sha256'] == q.sha(directory/'designs.npz')
        for path, digest in prep['source_sha256'].items(): assert q.sha(Path(path)) == digest
        x = design(family); checks[family] = []
        for fold, split in enumerate(folds):
            saved, values = control(family, fold, x, windows, answers, split, aw)
            checks[family].append({'fold': fold, 'probability_maxdiff': 0.,
                'all12222_scores_exact': True, 'all6alpha_two_priors_exact': True,
                'calibration_and_choices_exact': True, 'fit_weights_exact': True,
                'model_sha256': q.sha(directory/f'fold_{fold}_frozen.pkl')})
        print('R32_CONTROL_REPLAY_PASSED', family, '5models', flush=True)
    assert snap == source_hashes()
    q.save(OUT/'CPU_SELFCHECK.json', {'passed': True, 'controls': checks,
        'group_counts': groups, 'fit_only_scaler_toy': True, 'C_alpha_tie_rules': True,
        'old_models_refitted': 0, 'new_models_fitted': 0, 'GPU_used': False,
        'window_order_sha256': q.digest([w['window_key'] for w in windows])})
    q.save(OUT/'preparation_complete.json', {'status': 'prepared_not_fitted',
        'source_sha256': snap, 'check_sha256': q.sha(OUT/'CPU_SELFCHECK.json'),
        'real_LR_fits': 0, 'requested_future_new_fits': 20, 'reused_models': 10,
        'human_gold': False, 'GPU_used': False})
    print('R32_PREPARED_NO_NEW_FIT', flush=True)


def check_prepared():
    p = q.read(OUT/'preparation_complete.json')
    assert p['source_sha256'] == source_hashes()
    assert p['check_sha256'] == q.sha(OUT/'CPU_SELFCHECK.json')
    return p


def fit():
    prepared = check_prepared()
    assert not (OUT/'fit_started.json').exists(), 'Preserve prior fit or failure'
    q.save(OUT/'fit_started.json', {'preparation_sha256': q.sha(OUT/'preparation_complete.json')})
    windows, answers, folds, aw = r.metadata()
    start = time.perf_counter(); files = []; nfit = 0
    for family in SOURCES:
        directory = OUT/family; directory.mkdir()
        x = design(family)
        for fold, groups in enumerate(folds):
            old, prior_values = control(family, fold, x, windows, answers, groups, aw)
            models, semantic_scores, semantic_table = {}, {}, []
            ix = old['model']['fit_ix']
            for c in CS:
                if c == .01:
                    model, values = old['model'], prior_values['semantic_probe']
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        model, values = core.fit(x, windows, ix, c, {'loss_mass': 3854, 'seed': SEED})
                    nfit += 1
                    for name in ('fit_ix', 'fit_y', 'base_weights', 'loss_weights', 'class_factors'):
                        assert np.array_equal(model[name], old['model'][name]), name
                    for name in ('mean_', 'var_', 'scale_'):
                        assert np.array_equal(getattr(model['scaler'], name), getattr(old['model']['scaler'], name)), name
                models[c] = model; semantic_scores[c] = values
                ts = r.thresholds(values, groups['calibration_groups'], windows, answers, aw)
                semantic_table.append({'C': c, 'thresholds': ts, 'key': key(ts, c)})
            sem_chosen = max(semantic_table, key=lambda row: row['key'])
            selected = {'semantic_probe': sem_chosen}
            table = {'semantic_probe': semantic_table}
            scores = {'semantic_probe': semantic_scores[sem_chosen['C']]}
            thresholds = {'semantic_probe': sem_chosen['thresholds']}
            candidate_values = {'semantic_all_C': np.asarray([semantic_scores[c] for c in CS])}
            for prior, fused in zip(BASES, FUSED):
                scores[prior] = prior_values[prior]; thresholds[prior] = old['thresholds'][prior]
                rows, vals = [], []
                for c in CS:
                    for alpha in ALPHA:
                        v = r.combine(scores[prior], semantic_scores[c], alpha)
                        ts = r.thresholds(v, groups['calibration_groups'], windows, answers, aw)
                        if alpha == 0: assert np.array_equal(v, scores[prior]) and ts == thresholds[prior]
                        rows.append({'C': c, 'alpha': alpha, 'thresholds': ts, 'key': key(ts, c, alpha)})
                        vals.append(v)
                chosen = max(range(len(rows)), key=lambda j: rows[j]['key'])
                selected[fused], table[fused] = rows[chosen], rows
                scores[fused], thresholds[fused] = vals[chosen], rows[chosen]['thresholds']
                candidate_values[fused+'_all_C_alpha'] = np.asarray(vals)
            obj = {'groups': groups, 'models': models, 'thresholds': thresholds,
                'calibration_tables': table, 'selected': selected, 'human_gold': False,
                'old_control_sha256': q.sha(SOURCES[family]/f'fold_{fold}_frozen.pkl')}
            (directory/f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(obj, protocol=5))
            np.savez_compressed(directory/f'fold_{fold}_scores.npz', **scores, **candidate_values)
            q.save(directory/f'fold_{fold}_calibration.json', {k: obj[k] for k in
                ('groups', 'thresholds', 'calibration_tables', 'selected', 'human_gold', 'old_control_sha256')})
            files += [f'{family}/fold_{fold}_{s}' for s in ('frozen.pkl', 'scores.npz', 'calibration.json')]
            print('R32_FOLD_FROZEN', family, fold,
                  {m: (z['C'], z.get('alpha')) for m, z in selected.items()}, flush=True)
    assert nfit == 20 and prepared == check_prepared()
    q.save(OUT/'fit_complete.json', {'status': 'all10_family_folds_frozen_before_outer',
        'new_LR_fits': nfit, 'reused_LR_models': 10, 'seconds': time.perf_counter()-start,
        'files_sha256': {n: q.sha(OUT/n) for n in files}, 'source_sha256': source_hashes(),
        'human_gold': False, 'GPU_used': False})
    print('R32_ALL_FIT_CAL_CHOICES_FROZEN', flush=True)


def evaluate():
    assert not (OUT/'complete.json').exists(), 'Preserve completed evaluation'
    frozen = q.read(OUT/'fit_complete.json')
    assert frozen['status'] == 'all10_family_folds_frozen_before_outer'
    assert frozen['source_sha256'] == source_hashes()
    for name, digest in frozen['files_sha256'].items(): assert q.sha(OUT/name) == digest
    results, comparisons = {}, {}
    # Original evaluator's namespace is isolated; its denominator/max rules stay exact.
    for family, old in SOURCES.items():
        directory = OUT/family
        local_files = {Path(n).name: d for n, d in frozen['files_sha256'].items() if n.startswith(family+'/')}
        q.save(directory/'fit_complete.json', dict(frozen, files_sha256=local_files))
        spec = importlib.util.spec_from_file_location('_r32_eval_'+family, r.__file__)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.OUT = directory; module.source_hashes = source_hashes
        module.evaluate()
        results[family] = q.read(directory/'summary.json')['methods']
        previous = q.read(old/'summary.json')['methods']
        comparisons[family] = {m: {unit: {'fixed_C001': previous[m][unit]['f1'],
            'C_selected': results[family][m][unit]['f1'],
            'difference': results[family][m][unit]['f1']-previous[m][unit]['f1']}
            for unit in ('windows', 'answers')} for m in METHODS}
    q.save(OUT/'summary.json', {'methods': results, 'fixed_C001_comparison': comparisons,
        'human_gold': False, 'all10_family_folds_frozen_before_outer': True,
        'scope': protocol()['scope'], 'limits': protocol()['limits']})
    names = ['summary.json', 'fit_complete.json'] + [f'{family}/complete.json' for family in SOURCES]
    q.save(OUT/'complete.json', {'status': 'complete_development_only',
        'files_sha256': {n: q.sha(OUT/n) for n in names}, 'human_gold': False,
        'original_validation_test_opened': False, 'public_QA_test_opened': False})
    print('R32_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'check', 'fit', 'evaluate'))
    stage = parser.parse_args().stage
    with threadpool_limits(limits=4):
        try:
            {'prepare': prepare, 'check': check_prepared, 'fit': fit, 'evaluate': evaluate}[stage]()
        except BaseException as exc:
            OUT.mkdir(parents=True, exist_ok=True)
            q.save(OUT/f'FAILURE_{stage}_{time.time_ns()}.json', {'error': repr(exc),
                'GPU_used': False, 'original_validation_test_opened': False})
            raise
