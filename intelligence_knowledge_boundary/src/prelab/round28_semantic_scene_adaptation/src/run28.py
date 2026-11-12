"""Fixed event-group adaptation; no original R16 heldouts are accessed."""
from pathlib import Path
import argparse
import importlib.util
import pickle
import time
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
import extract28 as cache

q, ROOT, SCENE = cache.q, cache.ROOT, cache.SCENE
OUT = ROOT / 'results'
R26 = ROOT.parent / 'round26_global_local_fusion/results'
spec = importlib.util.spec_from_file_location('r28_original_fit', ROOT.parent / 'round20_localization_optimization/src/run20.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
BASES = ('lookback_tuned', 'local_slots_fusion_smooth_global')
FUSED = ('lookback_plus_semantic', 'r26_plus_semantic')
METHODS = BASES + ('semantic_probe',) + FUSED
ALPHA = (0., .2, .4, .6, .8, 1.)
SEED = 20261008


def metadata():
    windows, answers = q.lines(SCENE / 'windows.jsonl'), q.lines(SCENE / 'answers.jsonl')
    folds = q.read(SCENE / 'folds.json')
    aw = defaultdict(list)
    for i, w in enumerate(windows):
        assert w['split'] == 'train' and len(w['item_ids']) == 1
        aw[w['item_ids'][0]].append(i)
    assert len(windows) == 12222 and len(answers) == len(aw) == 602
    assert sum(w['main_eligible'] for w in windows) == 9526
    assert sum(a['main_eligible'] for a in answers) == 598
    assert sum(a['reviewed_safe_refusal'] for a in answers) == 117
    assert all(aw[a['item_id']] for a in answers)
    return windows, answers, folds, aw


def group_check(windows, answers, folds):
    qg = defaultdict(set)
    for a in answers:
        qg[a['question_id']].add(a['group_id'])
    assert len(qg) == 301 and all(len(v) == 1 for v in qg.values())
    all_groups = {a['group_id'] for a in answers}; assert len(all_groups) == 278
    seen = []; report = []
    for fold, groups in enumerate(folds):
        ss = [set(groups[k]) for k in ('fit_groups', 'calibration_groups', 'evaluation_groups')]
        assert set.union(*ss) == all_groups and all(not ss[i] & ss[j] for i in range(3) for j in range(i))
        assert groups == q.read(R26 / f'fold_{fold}_calibration.json')['groups']
        seen += groups['evaluation_groups']; record = {'fold': fold}
        for name, s in zip(('fit', 'calibration', 'evaluation'), ss):
            aa = [a for a in answers if a['group_id'] in s]
            ww = [w for w in windows if w['group_id'] in s]
            record[name] = {'groups': len(s), 'answers': len(aa), 'resolved_answers': sum(a['main_eligible'] for a in aa),
                'candidate_windows': len(ww), 'eligible_windows': sum(w['main_eligible'] for w in ww),
                'groups_with_eligible_windows': len({w['group_id'] for w in ww if w['main_eligible']}),
                'positive_windows': sum(w['main_eligible'] and w['gold'] == 1 for w in ww)}
        report.append(record)
    assert len(seen) == len(set(seen)) == 278
    return report


def combine(prior, semantic, alpha):
    if alpha == 0:
        return np.asarray(prior, np.float64).copy()
    a, b = np.clip(prior, 1e-12, 1-1e-12), np.clip(semantic, 1e-12, 1-1e-12)
    return expit(np.log(a)-np.log1p(-a) + alpha*(np.log(b)-np.log1p(-b)))


def projection(raw, fit_ix, weights, fold):
    weights = np.asarray(weights, np.float64); weights = weights / weights.sum()
    fit = np.asarray(raw[fit_ix], np.float64); mean = weights @ fit
    _, singular, components = randomized_svd((fit-mean)*np.sqrt(weights[:, None]),
        n_components=32, n_iter=3, random_state=SEED+fold, flip_sign=True)
    total = float(np.einsum('ij,i,ij->', fit-mean, weights, fit-mean))
    projected = ((np.asarray(raw, np.float64)-mean) @ components.T).astype(np.float32)
    obj = {'mean': mean, 'components': components, 'singular_values': singular,
        'explained_variance_ratio_sum': float((singular**2).sum()/total), 'weighted_total_variance': total,
        'fit_ix': fit_ix, 'normalized_base_weights': weights, 'seed': SEED+fold, 'whiten': False}
    return obj, projected


def design():
    OUT.mkdir(parents=True, exist_ok=True)
    windows, answers, folds, _ = metadata()
    stats = group_check(windows, answers, folds)
    protocol = {'version': 'r28-frozen-semantic-scene-adaptation-v1', 'seed': SEED,
        'scope': 'Repeated original R16 train602/301questions/278groups only; human_gold=false. Not a replacement for public human-gold QA results.',
        'encoder': 'Frozen auxiliary->QA2 selected previously using QA only. Whole-input bidirectional semantic checker; extra model, not native Qwen white-box state.',
        'semantic_input': 'For each unchanged candidate4rawBPE window, mean hidden768 over raw_token_indices INCLUDING punctuation/whitespace positions; concatenate max actual semantic logit over original lexical_token_indices. No gold in feature construction.',
        'projection': {'components': 32, 'n_iter': 3, 'seed': '20261008+fold', 'whiten': False,
            'fit_scope': 'Only main_eligible windows in original fold fit groups; original group/condition/answer/window BASE weights, no class weighting in PCA.'},
        'probe': {'input_dimension': 33, 'C': .01, 'solver': 'liblinear', 'max_iter': 2000,
            'scaler': 'Original fit-only base-weighted StandardScaler; float32 transformed design.',
            'weights': 'Reuse original R20/R13 group/condition/answer/window weights, fit-only class factors then equal group loss, total3854.',
            'LR_fits': 5, 'PCA_fits': 5, 'no_C_search_or_refit_on_fit_plus_cal': True},
        'baselines': list(BASES), 'fusion_alpha': list(ALPHA),
        'fusion': 'sigmoid(logit(prior)+alpha*logit(same semantic probe)); clip1e-12, alpha0 exact identity. Each baseline receives the SAME semantic branch and six-alpha budget.',
        'selection': 'For each baseline in each fold, select one alpha shared by window/answer scoring using calibration min(windowF1,answerF1), then windowF1, windowPrecision, smaller alpha. Separate original two thresholds; preserve all12 candidates per fold.',
        'folds': 'Original outer=f, calibration=(f+1)%5, other3 fit. No question/condition/answer/token/window split across these event groups.',
        'prior_scores': 'Only corresponding R26 fold_f_scores.npz in original candidate window order. Never train with concatenated OOF feature scores.',
        'eligibility': 'Only original eligible asserted windows train/score localization. Unknown/refusal windows never relabeled0. Answer max uses ALL candidate windows;117 reviewed-safe-refusal answers retain existing negative label;4 unresolved answers excluded.',
        'evaluation': 'Freeze all5 fitted probes/calibration choices before outer metrics. No selection on outer results; alpha0 does not guarantee unseen-fold improvement.',
        'human_gold': False, 'original_validation_test_opened': False, 'public_QA_test_opened': False,
        'historical_limit': 'Old R16 validation/test were exposed in earlier rounds; only no reads in THIS run are promised. Public QA official_test150 remains sealed.'}
    cache.freeze(ROOT / 'protocol.json', protocol)
    p = np.asarray([.01, .4, .9]); assert np.array_equal(combine(p, p, 0), p)
    assert np.allclose(combine(p, np.full(3, .5), 1), p, rtol=0, atol=2e-16)
    rng = np.random.default_rng(SEED); toy = rng.normal(size=(70, 40)).astype(np.float32)
    fit_ix = np.arange(50); b = np.linspace(.1, 2, 50)
    a, z = projection(toy, fit_ix, b, 0)
    changed = toy.copy(); changed[50:] += 100
    aa, _ = projection(changed, fit_ix, b, 0)
    assert np.array_equal(a['mean'], aa['mean']) and np.array_equal(a['components'], aa['components'])
    assert z.shape == (70, 32)
    check = {'passed': True, 'group_counts': stats, 'question_pairs_stay_together': True,
        'PCA_fit_only_toy_passed': True, 'alpha_zero_exact_identity': True,
        'window_order_sha256': q.digest([w['window_key'] for w in windows]),
        'source_sha256': q.sha(Path(__file__)), 'GPU_used': False, 'real_models_fitted': False}
    cache.freeze(OUT / 'CPU_SELFCHECK.json', check)
    print('R28_DESIGN_AND_CPU_CHECK_PASSED', flush=True)


def source_hashes():
    names = [Path(__file__), Path(cache.__file__), Path(core.__file__), Path(core.r13.__file__),
        Path(core.r10.__file__), ROOT / 'protocol.json', OUT / 'CPU_SELFCHECK.json', cache.OUT / 'complete.json',
        SCENE / 'windows.jsonl', SCENE / 'answers.jsonl', SCENE / 'folds.json', R26 / 'calibration_freeze.json']
    names += [R26 / f'fold_{f}_{suffix}' for f in range(5) for suffix in ('calibration.json', 'scores.npz')]
    return {str(p.resolve()): q.sha(p) for p in names}


def prepare():
    assert not (OUT / 'preparation_complete.json').exists(), 'Preserve completed preparation'
    design(); cache.check_complete()
    windows, answers, folds, _ = metadata()
    rows = q.lines(SCENE / 'inputs.jsonl'); lookup = {r['response_id']: i for i, r in enumerate(rows)}
    bounds = np.load(cache.OUT / 'bounds.npy')
    hidden = np.load(cache.OUT / 'hidden.npy', mmap_mode='r'); logit = np.load(cache.OUT / 'logit.npy')
    hh, zz = [], []
    for w in windows:
        lo, hi = bounds[lookup[w['row_id']]]
        hh.append(np.asarray(hidden[lo + np.asarray(w['raw_token_indices'])], np.float64).mean(axis=0))
        zz.append(float(logit[lo + np.asarray(w['lexical_token_indices'])].max()))
    h, z = np.asarray(hh, np.float32), np.asarray(zz, np.float32)
    assert h.shape == (12222, 768) and z.shape == (12222,) and np.isfinite(h).all() and np.isfinite(z).all()
    old = q.read(R26 / 'calibration_freeze.json')
    for name, value in old['files_sha256'].items():
        assert q.sha(R26 / name) == value
    np.savez_compressed(OUT / 'designs.npz', hidden=h, logit=z)
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_not_fitted', 'source_sha256': source_hashes(),
        'designs_sha256': q.sha(OUT / 'designs.npz'), 'windows': 12222, 'human_gold': False,
        'GPU_used': False, 'original_validation_test_opened': False})
    print('R28_DESIGNS_PREPARED', h.shape, flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['source_sha256'] == source_hashes() and p['designs_sha256'] == q.sha(OUT / 'designs.npz')
    return p


def answer_scores(values, answers, aw):
    return np.asarray([max(values[aw[a['item_id']]]) for a in answers], np.float64)


def thresholds(values, groups, windows, answers, aw):
    groups = set(groups)
    wi = [i for i, w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups]
    ai = [i for i, a in enumerate(answers) if a['main_eligible'] and a['group_id'] in groups]
    av = answer_scores(values, answers, aw)
    return {'window': core.r10.threshold_search([windows[i]['gold'] for i in wi], values[wi]),
        'answer': core.r10.threshold_search([answers[i]['gold'] for i in ai], av[ai])}


def selection_key(ts, alpha):
    w, a = ts['window'], ts['answer']
    return [min(w['validation_f1'], a['validation_f1']), w['validation_f1'], w['validation_precision'], -alpha]


def fit():
    prepared = check_prepared(); assert not (OUT / 'fit_started.json').exists()
    q.save(OUT / 'fit_started.json', {'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    windows, answers, folds, aw = metadata(); start = time.perf_counter(); names = []
    with np.load(OUT / 'designs.npz') as z:
        hidden, logits = z['hidden'].copy(), z['logit'].copy()
    with threadpool_limits(limits=4):
        for fold, groups in enumerate(folds):
            fit_groups = set(groups['fit_groups'])
            ix = np.asarray([i for i, w in enumerate(windows) if w['main_eligible'] and w['group_id'] in fit_groups])
            rr = [windows[i] for i in ix]; b = core.r10.base_weights(rr, 'token')
            pc, projected = projection(hidden, ix, b, fold)
            used_groups = sorted({w['group_id'] for w in rr})
            pc['fit_keys'] = [w['window_key'] for w in rr]; pc['fit_groups'] = used_groups
            pc['assigned_fit_groups'] = sorted(fit_groups)
            x = np.column_stack((projected, logits)).astype(np.float32)
            model, semantic = core.fit(x, windows, ix, .01, {'loss_mass': 3854, 'seed': SEED})
            assert model['width'] == 33 and model['fit_groups'] == used_groups
            values = {'semantic_probe': semantic}; ts = {'semantic_probe': thresholds(semantic, groups['calibration_groups'], windows, answers, aw)}
            old = q.read(R26 / f'fold_{fold}_calibration.json')
            with np.load(R26 / f'fold_{fold}_scores.npz', allow_pickle=False) as saved:
                for name in BASES:
                    values[name] = saved[name].copy(); assert values[name].shape == (12222,)
                    ts[name] = old['thresholds'][name]
                    assert thresholds(values[name], groups['calibration_groups'], windows, answers, aw) == ts[name]
            tables, candidates, chosen = {}, {}, {}
            for prior, fused in zip(BASES, FUSED):
                table = []; scores = []
                for alpha in ALPHA:
                    v = combine(values[prior], semantic, alpha)
                    t = thresholds(v, groups['calibration_groups'], windows, answers, aw)
                    if alpha == 0:
                        assert np.array_equal(v, values[prior]) and t == ts[prior]
                    table.append({'alpha': alpha, 'thresholds': t, 'key': selection_key(t, alpha)})
                    scores.append(v)
                j = max(range(len(table)), key=lambda i: table[i]['key'])
                tables[fused] = table; candidates[fused] = np.asarray(scores); chosen[fused] = table[j]
                values[fused], ts[fused] = scores[j], table[j]['thresholds']
            frozen = {'groups': groups, 'PCA': pc, 'model': model, 'thresholds': ts,
                'alpha_tables': tables, 'selected_alpha': chosen, 'human_gold': False}
            (OUT / f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(frozen, protocol=5))
            np.savez_compressed(OUT / f'fold_{fold}_scores.npz', **values,
                **{k + '_all_alpha': v for k, v in candidates.items()})
            q.save(OUT / f'fold_{fold}_calibration.json', {k: frozen[k] for k in ('groups', 'thresholds', 'alpha_tables', 'selected_alpha', 'human_gold')})
            names += [f'fold_{fold}_{suffix}' for suffix in ('frozen.pkl', 'scores.npz', 'calibration.json')]
            print('R28_FOLD_FROZEN', fold, {k: v['alpha'] for k, v in chosen.items()}, flush=True)
    assert prepared == check_prepared()
    q.save(OUT / 'fit_complete.json', {'status': 'all5fold_fit_calibration_frozen_before_outer', 'seconds': time.perf_counter()-start,
        'PCA_fits': 5, 'LR_fits': 5, 'files_sha256': {n: q.sha(OUT / n) for n in names},
        'source_sha256': source_hashes(), 'human_gold': False, 'original_validation_test_opened': False})


def evaluate():
    assert not (OUT / 'complete.json').exists()
    frozen = q.read(OUT / 'fit_complete.json'); assert frozen['source_sha256'] == source_hashes()
    for n, value in frozen['files_sha256'].items():
        assert q.sha(OUT / n) == value
    windows, answers, folds, aw = metadata()
    wp = {m: np.zeros(len(windows), bool) for m in METHODS}; ap = {m: np.zeros(len(answers), bool) for m in METHODS}
    ws = {m: np.full(len(windows), np.nan) for m in METHODS}; ass = {m: np.full(len(answers), np.nan) for m in METHODS}
    wc, ac = np.zeros(len(windows), int), np.zeros(len(answers), int)
    for fold, groups in enumerate(folds):
        f = q.read(OUT / f'fold_{fold}_calibration.json'); assert f['groups'] == groups
        eg = set(groups['evaluation_groups'])
        wi = np.asarray([i for i, w in enumerate(windows) if w['group_id'] in eg])
        ai = np.asarray([i for i, a in enumerate(answers) if a['group_id'] in eg])
        wc[wi] += 1; ac[ai] += 1
        with np.load(OUT / f'fold_{fold}_scores.npz', allow_pickle=False) as z:
            for m in METHODS:
                v = z[m]; av = answer_scores(v, answers, aw)
                ws[m][wi] = v[wi]; ass[m][ai] = av[ai]
                wp[m][wi] = v[wi] >= f['thresholds'][m]['window']['threshold']
                ap[m][ai] = av[ai] >= f['thresholds'][m]['answer']['threshold']
    assert (wc == 1).all() and (ac == 1).all()
    wy = np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows]); wm = wy >= 0
    ay = np.asarray([a['gold'] if a['main_eligible'] else -1 for a in answers]); am = ay >= 0
    result = {}
    for m in METHODS:
        assert np.isfinite(ws[m]).all() and np.isfinite(ass[m]).all()
        result[m] = {'windows': cache.transfer.scene.counts(wy[wm], wp[m][wm]),
            'answers': cache.transfer.scene.counts(ay[am], ap[m][am]),
            'safe_refusal_false_positives': sum(bool(ap[m][i]) for i, a in enumerate(answers) if a['reviewed_safe_refusal'])}
    original = q.read(R26 / 'summary.json')['methods']
    for m in BASES:
        for unit in ('windows', 'answers'):
            for key in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1'):
                assert result[m][unit][key] == original[m][unit][key]
    q.savel(OUT / 'window_scores_oof.jsonl', [dict(w, scores={m: float(ws[m][i]) for m in METHODS},
        predictions={m: bool(wp[m][i]) for m in METHODS}) for i, w in enumerate(windows)])
    q.savel(OUT / 'answer_scores_oof.jsonl', [dict(a, scores={m: float(ass[m][i]) for m in METHODS},
        predictions={m: bool(ap[m][i]) for m in METHODS}) for i, a in enumerate(answers)])
    q.save(OUT / 'summary.json', {'methods': result, 'old_baselines_exact': True, 'human_gold': False,
        'scope': 'Repeated R16 train development, not a fresh test or replacement for public human-gold QA.',
        'all5_frozen_before_outer_metrics': True, 'original_validation_test_opened': False})
    q.save(OUT / 'complete.json', {'status': 'complete_development_only', 'human_gold': False,
        'files_sha256': {n: q.sha(OUT / n) for n in ('fit_complete.json', 'summary.json', 'window_scores_oof.jsonl', 'answer_scores_oof.jsonl')},
        'original_validation_test_opened': False})
    for m, r in result.items():
        print('R28_RESULT', m, r['windows']['f1'], r['answers']['f1'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('design', 'prepare', 'fit', 'evaluate'))
    stage = parser.parse_args().stage
    try:
        {'design': design, 'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[stage]()
    except BaseException as exc:
        q.save(OUT / f'FAILURE_{stage}_{time.time_ns()}.json', {'error': repr(exc), 'original_validation_test_opened': False})
        raise
