"""Full hidden768+z matched against frozen R28 PCA32+z; CPU development only."""
from pathlib import Path
import sys
import argparse
import importlib.util
import pickle
import shutil
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
R28 = ROOT.parent / 'round28_semantic_scene_adaptation'
sys.path.insert(0, str(R28 / 'src'))
import run28 as original

q, cache, core = original.q, original.cache, original.core
OUT = ROOT / 'results'
OLD = R28 / 'results'
SCENE, R26 = original.SCENE, original.R26
BASES, FUSED, METHODS, ALPHA, SEED = original.BASES, original.FUSED, original.METHODS, original.ALPHA, original.SEED
metadata, group_check = original.metadata, original.group_check
combine, thresholds, answer_scores, selection_key = original.combine, original.thresholds, original.answer_scores, original.selection_key


def protocol():
    return {'version': 'r29-full768-semantic-scene-adaptation-v1', 'seed': SEED,
        'scope': 'Same repeatedly developed original R16 train602/301questions/278groups; human_gold=false. Not a fresh test or replacement for public human-gold QA.',
        'encoder': 'Identical frozen auxiliary->QA2 model and existing R28 cache. No new encoder, data, GPU pass or checkpoint selection.',
        'input': 'Exact R28 designs.npz: mean hidden768 over each unchanged four-raw-BPE candidate window including punctuation/whitespace, plus maximum semantic logit over its original lexical tokens. No PCA. Raw769 float32.',
        'control': 'Reuse completed R28 PCA32+z models, fold scores, choices and OOF metrics under fixed hashes; never refit controls. Only representation changes from33 to769.',
        'probe': {'width': 769, 'C': .01, 'solver': 'liblinear', 'max_iter': 2000,
            'seed': SEED, 'LR_fits': 5, 'PCA_fits': 0, 'loss_mass': 3854,
            'scaler': 'Unchanged fit-only original-base-weighted StandardScaler; transformfloat32.',
            'weights': 'Unchanged R20/R13 group/condition/answer/window weights and fit-only class factors, group reequalization, total3854. Assert identical to corresponding R28 frozen fit.',
            'nonconvergence': 'Preserve failure and stop; no budget/C/solver changes.'},
        'folds': 'Exact original five event-group fit/calibration/evaluation sets. Only eligible fit windows fit scaler/LR. All five models and calibration choices committed before outer metrics.',
        'baselines': list(BASES), 'fusion_alpha': list(ALPHA),
        'fusion': 'Identical R28 sigmoid(logit(prior)+alpha*logit(same semantic probe)), clip1e-12 and alpha0 exact identity. Use each corresponding R26 fold_f_scores.npz, never concatenated OOF training scores.',
        'selection': 'One alpha per baseline and fold shared by both units. Calibration min(windowF1,answerF1), windowF1, windowPrecision, smalleralpha; unchanged separate thresholds and all60 candidates retained.',
        'eligibility': 'Only original eligible asserted windows train/score localization. Unknown/refusal windows are not set0. Answer max uses all candidate windows;117 reviewed safe refusals remain original negative answers;4 unresolved answers excluded.',
        'limits': 'Higher dimensionality also changes the number of fitted coefficients and the L2 geometry; sameC does not identify PCA information loss alone. No causal or universal learnability claim from this one comparison.',
        'historical_limit': 'Old R16 validation/test were exposed in earlier rounds. This run does not read them. Public QA official_test150 remains sealed.',
        'GPU_used': False, 'human_gold': False, 'original_validation_test_opened': False,
        'public_QA_test_opened': False}


def verify_control():
    done = q.read(OLD / 'complete.json')
    assert done['status'] == 'complete_development_only' and done['human_gold'] is False
    for name, digest in done['files_sha256'].items(): assert q.sha(OLD / name) == digest
    fits = q.read(OLD / 'fit_complete.json')
    for name, digest in fits['files_sha256'].items(): assert q.sha(OLD / name) == digest
    original.check_prepared()
    audit = q.read(OLD / 'AUDIT.json')
    assert audit['passed'] and audit['complete_sha256'] == q.sha(OLD / 'complete.json')
    return {'complete_sha256': q.sha(OLD / 'complete.json'), 'audit_sha256': q.sha(OLD / 'AUDIT.json'),
        'frozen_files_sha256': fits['files_sha256'], 'models_refitted': 0,
        'methods': q.read(OLD / 'summary.json')['methods']}


def design():
    OUT.mkdir(parents=True, exist_ok=True)
    cache.freeze(ROOT / 'protocol.json', protocol())
    windows, answers, folds, _ = metadata()
    stats = group_check(windows, answers, folds)
    assert all(0 < len(f['fit_groups']) < 278 for f in folds)
    rng = np.random.default_rng(SEED)
    toy = rng.normal(size=(40, 769)).astype(np.float32)
    ix = np.arange(30); weights = np.linspace(.2, 2., len(ix))
    a = StandardScaler().fit(toy[ix], sample_weight=weights)
    changed = toy.copy(); changed[30:] = np.nan
    b = StandardScaler().fit(changed[ix], sample_weight=weights)
    assert np.array_equal(a.mean_, b.mean_) and np.array_equal(a.var_, b.var_)
    p = np.asarray([.01, .4, .9]); assert np.array_equal(combine(p, p, 0), p)
    cache.freeze(OUT / 'CPU_SELFCHECK.json', {'passed': True, 'group_counts': stats,
        'fit_only_scaler_outside_fit_NaN_test': True, 'question_pairs_stay_together': True,
        'alpha_zero_exact_identity': True, 'raw_design_width': 769,
        'window_order_sha256': q.digest([w['window_key'] for w in windows]),
        'source_sha256': q.sha(Path(__file__)), 'real_models_fitted': False, 'GPU_used': False})
    print('R29_DESIGN_CPU_CHECK_PASSED', flush=True)


def source_hashes():
    names = [Path(__file__), Path(original.__file__), Path(cache.__file__), Path(core.__file__),
        Path(core.r13.__file__), Path(core.r10.__file__), ROOT / 'protocol.json', OUT / 'CPU_SELFCHECK.json',
        OLD / 'complete.json', OLD / 'fit_complete.json', OLD / 'summary.json', OLD / 'AUDIT.json',
        OLD / 'preparation_complete.json', OLD / 'designs.npz',
        SCENE / 'windows.jsonl', SCENE / 'answers.jsonl', SCENE / 'folds.json', R26 / 'calibration_freeze.json']
    names += [directory / f'fold_{f}_{suffix}' for directory in (OLD, R26) for f in range(5)
              for suffix in ('calibration.json', 'scores.npz')]
    names += [OLD / f'fold_{f}_frozen.pkl' for f in range(5)]
    return {str(p.resolve()): q.sha(p) for p in names}


def prepare():
    assert not (OUT / 'preparation_complete.json').exists()
    design(); control = verify_control(); snap = source_hashes()
    shutil.copyfile(OLD / 'designs.npz', OUT / 'designs.npz')
    assert q.sha(OLD / 'designs.npz') == q.sha(OUT / 'designs.npz')
    with np.load(OUT / 'designs.npz') as data:
        h, z = data['hidden'], data['logit']
        assert h.shape == (12222, 768) and z.shape == (12222,)
        assert h.dtype == z.dtype == np.float32 and np.isfinite(h).all() and np.isfinite(z).all()
        x = np.column_stack((h, z)); assert x.shape == (12222, 769) and x.dtype == np.float32
    q.save(OUT / 'PCA32_CONTROL.json', control)
    assert snap == source_hashes()
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_not_fitted', 'source_sha256': snap,
        'designs_sha256': q.sha(OUT / 'designs.npz'), 'control_sha256': q.sha(OUT / 'PCA32_CONTROL.json'),
        'windows': 12222, 'raw_width': 769, 'human_gold': False, 'GPU_used': False,
        'original_validation_test_opened': False})
    print('R29_FULL_HIDDEN_DESIGNS_PREPARED', x.shape, flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['source_sha256'] == source_hashes()
    assert p['designs_sha256'] == q.sha(OUT / 'designs.npz')
    assert p['control_sha256'] == q.sha(OUT / 'PCA32_CONTROL.json')
    return p


def fit():
    prepared = check_prepared(); assert not (OUT / 'fit_started.json').exists()
    q.save(OUT / 'fit_started.json', {'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    windows, answers, folds, aw = metadata(); start = time.perf_counter(); names = []
    with np.load(OUT / 'designs.npz') as data:
        x = np.column_stack((data['hidden'], data['logit'])).astype(np.float32)
    with threadpool_limits(limits=4):
        for fold, groups in enumerate(folds):
            fit_groups = set(groups['fit_groups'])
            ix = np.asarray([i for i, w in enumerate(windows) if w['main_eligible'] and w['group_id'] in fit_groups])
            model, semantic = core.fit(x, windows, ix, .01, {'loss_mass': 3854, 'seed': SEED})
            assert model['width'] == 769
            control = pickle.loads((OLD / f'fold_{fold}_frozen.pkl').read_bytes())
            assert control['groups'] == groups and control['model']['width'] == 33
            for key in ('fit_ix', 'fit_y', 'base_weights', 'loss_weights', 'class_factors'):
                assert np.array_equal(model[key], control['model'][key]), key
            for key in ('fit_keys', 'fit_groups', 'C', 'loss_mass'):
                assert model[key] == control['model'][key], key
            values = {'semantic_probe': semantic}
            ts = {'semantic_probe': thresholds(semantic, groups['calibration_groups'], windows, answers, aw)}
            old = q.read(R26 / f'fold_{fold}_calibration.json')
            with np.load(R26 / f'fold_{fold}_scores.npz', allow_pickle=False) as saved:
                for name in BASES:
                    values[name] = saved[name].copy(); assert values[name].shape == (12222,)
                    ts[name] = old['thresholds'][name]
                    assert thresholds(values[name], groups['calibration_groups'], windows, answers, aw) == ts[name]
            tables, candidates, chosen = {}, {}, {}
            for prior, fused in zip(BASES, FUSED):
                table, scores = [], []
                for alpha in ALPHA:
                    v = combine(values[prior], semantic, alpha)
                    t = thresholds(v, groups['calibration_groups'], windows, answers, aw)
                    if alpha == 0: assert np.array_equal(v, values[prior]) and t == ts[prior]
                    table.append({'alpha': alpha, 'thresholds': t, 'key': selection_key(t, alpha)})
                    scores.append(v)
                j = max(range(len(table)), key=lambda i: table[i]['key'])
                tables[fused], candidates[fused], chosen[fused] = table, np.asarray(scores), table[j]
                values[fused], ts[fused] = scores[j], table[j]['thresholds']
            frozen = {'groups': groups, 'PCA': None, 'model': model, 'thresholds': ts,
                'alpha_tables': tables, 'selected_alpha': chosen, 'human_gold': False,
                'PCA32_control_model_sha256': q.sha(OLD / f'fold_{fold}_frozen.pkl')}
            (OUT / f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(frozen, protocol=5))
            np.savez_compressed(OUT / f'fold_{fold}_scores.npz', **values,
                **{k + '_all_alpha': v for k, v in candidates.items()})
            q.save(OUT / f'fold_{fold}_calibration.json', {k: frozen[k] for k in
                ('groups', 'thresholds', 'alpha_tables', 'selected_alpha', 'human_gold', 'PCA32_control_model_sha256')})
            names += [f'fold_{fold}_{suffix}' for suffix in ('frozen.pkl', 'scores.npz', 'calibration.json')]
            print('R29_FOLD_FROZEN', fold, 'LRiterations', model['model'].n_iter_.tolist(),
                  {k: v['alpha'] for k, v in chosen.items()}, flush=True)
    assert prepared == check_prepared()
    q.save(OUT / 'fit_complete.json', {'status': 'all5fold_fit_calibration_frozen_before_outer',
        'seconds': time.perf_counter()-start, 'PCA_fits': 0, 'LR_fits': 5,
        'files_sha256': {n: q.sha(OUT / n) for n in names}, 'source_sha256': source_hashes(),
        'human_gold': False, 'original_validation_test_opened': False})


def evaluate():
    # Execute the exact original aggregation, eligibility and outer-score code
    # under independent globals; never alter original source or its output path.
    spec = importlib.util.spec_from_file_location('_r29_same_evaluation', original.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.OUT = OUT
    module.source_hashes = source_hashes
    module.evaluate()
    old = q.read(OUT / 'PCA32_CONTROL.json')['methods']
    new = q.read(OUT / 'summary.json')['methods']
    comparisons = {name: {unit: {'PCA32': old[name][unit]['f1'], 'full768': new[name][unit]['f1'],
        'difference': new[name][unit]['f1']-old[name][unit]['f1']} for unit in ('windows', 'answers')}
        for name in METHODS}
    q.save(OUT / 'PCA32_COMPARISON.json', {'methods': comparisons,
        'PCA_control_refits': 0, 'control_manifest_sha256': q.sha(OUT / 'PCA32_CONTROL.json'),
        'human_gold': False, 'source': 'Repeated R16 train only; not public QA.',
        'limits': protocol()['limits']})
    print('R29_FULL_VS_FROZEN_PCA_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('design', 'prepare', 'fit', 'evaluate'))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        try: {'design': design, 'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[args.stage]()
        except BaseException as exc:
            OUT.mkdir(parents=True, exist_ok=True)
            q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json',
                   {'error': repr(exc), 'original_validation_test_opened': False, 'GPU_used': False})
            raise
