"""Three fixed shallow-tree feature comparisons. CPU prepare/check/explicit fit."""
from pathlib import Path
import argparse
import importlib.metadata
import pickle
import shutil
import sys
import time
import traceback

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fit_expansion'))
import run_llama_baselines as old

q, expansion = old.q, old.expansion
OUT = ROOT / 'results/ghost_nll_nonlinear_v1'
GHOST = ROOT / 'results/ghost_matched_lr_v1'
UPSTREAM = ROOT / 'results/ghost_geometry_features_v1'
NFIT, NCAL, NTOTAL = 653979, 42241, 696220
METHODS = {'nll_only': [0], 'ghost_only': [1, 2, 3, 4], 'nll_ghost': [0, 1, 2, 3, 4]}
NAMES = ['mean_actual_token_nll', 'adjacent_layer_cosine_change',
         'layer_to_final_cosine', 'top10_normalized_entropy',
         'top10_unweighted_embedding_divergence']
PARAMS = dict(max_iter=100, max_depth=2, max_leaf_nodes=4, learning_rate=.05,
              l2_regularization=1., min_samples_leaf=20, early_stopping=False,
              random_state=20261008, monotonic_cst=None)


def protocol():
    return {'version': 'ghost-nll-three-fixed-shallow-trees-v1',
            'methods': METHODS, 'feature_names': NAMES, 'new_fits': 3,
            'model': 'HistGradientBoostingClassifier', 'parameters': PARAMS,
            'effective_parameters': HistGradientBoostingClassifier(**PARAMS).get_params(),
            'CPU_threads': 4, 'cohort': {'fit_answers': 3680, 'calibration_answers': 159,
            'fit_groups': 615, 'calibration_groups': 154, 'fit_windows': NFIT,
            'calibration_windows': NCAL, 'all_windows': NTOTAL},
            'NLL': 'Unchanged raw base.npy[:,1024]: mean -log p(actual current token | preceding state), across every raw BPE slot including punctuation. Higher means less likely text, not known factual error. No sign inversion or monotonic constraint.',
            'GHOST': 'The four completed raw-window means, with punctuation and short-window actual slots retained. No fitted score is an input.',
            'preprocessing': 'No scaler, projection or learned compression. The tree binning is fitted on fit rows only.',
            'weights': 'Byte-identical expanded3680 training_weights.npz loss; group/native-aux/answer/window base weighting, fit class factors then group renormalization; loss mass168123. No new class_weight.',
            'fit': 'All653979 fit windows; calibration is never passed to fit, validation, early stopping or feature construction. Exactly100 boosting iterations per method.',
            'thresholds': 'Original q.choose_threshold on cal only, independently window and answer: F1, precision, then higher threshold; >= predicts risk.',
            'answer_score': 'Maximum over every original eligible window; original labels, refusal policy and4rawBPE geometry unchanged.',
            'selection': 'One fixed model per method. Report all three; no model/parameter search or cross-method selected winner.',
            'limits': ['Exploratory development designed after observed linear GHOST results; not independent test evidence.',
                       'A local shallow boosted-tree comparison, not the paper GHOST random-forest answer-level reproduction.',
                       'Extra3046 published answers use common Llama replay, not their generators historical native states.',
                       'Fit/cal F1 or ranking differences alone do not establish absence of overfitting.'],
            'resource_estimate': {'three_fit_minutes_planning': [1, 6], 'not_measured': True,
                'basis': 'Prior fixed depth2/100-stage2-feature CPU combiner max6.283s on168123 rows; conservatively scale rows, up to5 features and3fits, plus metadata/metrics overhead.',
                'raw_design_MiB': NTOTAL * 5 * 4 / 2**20, 'disk_reserve_MiB': 100},
            'official_test_opened': False, 'GPU_used': False}


def source_paths():
    paths = [Path(__file__), Path(old.__file__), Path(q.__file__), Path(expansion.__file__),
             ROOT / 'src/feature_qa.py', old.OUT / 'preparation_complete.json',
             old.OUT / 'training_weights.npz', expansion.OUT / 'expanded3680_weights.npz',
             GHOST / 'feature_preparation_complete.json', GHOST / 'ghost_window_features.npy',
             UPSTREAM / 'features_complete.json', ROOT / 'data/gold_manifest.json',
             ROOT / 'fit_expansion/data/export_freeze.json']
    for part in ('fit', 'calibration'):
        paths += [ROOT / f'data/{kind}_{part}.jsonl' for kind in ('answers', 'tokens', 'windows_k4')]
    paths += [ROOT / f'fit_expansion/data/{kind}_fit.jsonl' for kind in ('answers', 'tokens', 'windows_k4')]
    return paths


def metadata():
    _, meta = expansion.metadata()
    assert len(meta['answers']) == 3839 and len(meta['windows']) == NTOTAL
    assert meta['bounds'] == {'fit': [0, NFIT], 'calibration': [NFIT, NTOTAL]}
    groups = [{a['group_id'] for a in meta['answers'][lo:hi]} for lo, hi in ((0, 3680), (3680, 3839))]
    assert [len(g) for g in groups] == [615, 154] and not groups[0] & groups[1]
    assert len({a['response_id'] for a in meta['answers']}) == 3839
    assert all(meta['answer_windows'][a['response_id']] for a in meta['answers'])
    return meta


def tiny_check():
    rng = np.random.default_rng(20261008)
    x = rng.normal(size=(256, 5)).astype(np.float32)
    y = ((x[:, 0] * x[:, 1] > .15) | (x[:, 4] > 1.)).astype(int)
    weights = np.linspace(.2, 1.8, len(y), dtype=np.float64)
    for name, columns in METHODS.items():
        model = HistGradientBoostingClassifier(**PARAMS)
        model.fit(x[:, columns], y, sample_weight=weights)
        scores = model.predict_proba(x[:, columns])[:, 1]
        assert model.classes_.tolist() == [0, 1] and model.n_iter_ == 100
        assert not model.do_early_stopping_ and model.n_features_in_ == len(columns)
        assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    # Independent brute-force tie/endpoint check, and all-window answer max.
    yy = np.asarray([0, 1, 1, 0, 1]); ss = np.asarray([.2, .4, .4, .7, .9])
    t = q.choose_threshold(yy, ss)
    keys = []
    for z in [np.nextafter(ss.max(), np.inf), *np.unique(ss)]:
        m = q.count(yy, ss, z); keys.append((m['f1'], m['precision'], z))
    assert (t['f1'], t['precision'], t['threshold']) == max(keys)
    toy = {'answers': [{'response_id': 'a'}, {'response_id': 'b'}],
           'answer_windows': {'a': [0, 1, 2], 'b': [3, 4]}}
    assert q.answer_scores(toy, ss).tolist() == [.4, .9]
    return {'status': 'passed', 'synthetic_rows': 256, 'synthetic_models': 3,
            'real_data_fits': 0, 'iterations_each': 100, 'no_early_stopping': True,
            'weighted_fit_finite_probability_and_class_axis': True,
            'threshold_bruteforce_and_all_window_answermax': True, 'GPU_used': False}


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Preparation is immutable; do not overwrite.'
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol())
    tick = time.perf_counter()
    q.save(OUT / 'CPU_SELFCHECK.json', tiny_check())
    fp = q.read(GHOST / 'feature_preparation_complete.json')
    up = q.read(UPSTREAM / 'features_complete.json')
    assert fp['status'] == up['status'] == 'complete' and fp['records'] == up['records'] == 3839
    assert fp['all_coordinates_verified'] and fp['all_raw_punctuation_slots_pooled'] and not fp['test_opened']
    assert up['raw_answer_tokens'] == 708506 and up['no_test'] and not up['trained']
    assert fp['upstream_complete_sha256'] == q.sha(UPSTREAM / 'features_complete.json')
    assert fp['matrix_sha256'] == q.sha(GHOST / 'ghost_window_features.npy')
    meta = metadata()
    assert q.digest(meta['windows']) == fp['window_order_sha256']
    with np.load(old.OUT / 'training_weights.npz') as z:
        weights = {k: z[k].copy() for k in z.files}
    assert q.sha(old.OUT / 'training_weights.npz') == q.sha(expansion.OUT / 'expanded3680_weights.npz')
    assert np.array_equal(weights['y'], [w['label'] for w in meta['windows'][:NFIT]])
    assert all(weights[k].shape == (NFIT,) and np.isfinite(weights[k]).all() and np.all(weights[k] > 0) for k in ('base', 'loss'))
    assert abs(weights['loss'].sum() - 168123) < 1e-6
    raw_path = old.OUT / 'matrices/base.npy'
    expected = {k.replace('\\', '/'): v for k, v in q.read(old.OUT / 'preparation_complete.json')['files_sha256'].items()}
    raw_sha = q.sha(raw_path)
    assert raw_sha == expected['matrices/base.npy']
    raw = np.load(raw_path, mmap_mode='r'); ghost = np.load(GHOST / 'ghost_window_features.npy', mmap_mode='r')
    assert raw.shape == (NTOTAL, 1025) and ghost.shape == (NTOTAL, 4)
    assert raw.dtype == ghost.dtype == np.float32
    design = np.lib.format.open_memmap(OUT / 'window_features.npy', mode='w+', dtype=np.float32, shape=(NTOTAL, 5))
    for lo in range(0, NTOTAL, 16384):
        hi = min(NTOTAL, lo + 16384)
        design[lo:hi, 0] = raw[lo:hi, 1024]
        design[lo:hi, 1:] = ghost[lo:hi]
        assert np.isfinite(design[lo:hi]).all() and np.all(design[lo:hi, 0] >= 0)
        assert np.array_equal(design[lo:hi, 0], raw[lo:hi, -1])
        assert np.array_equal(design[lo:hi, 1:], ghost[lo:hi])
    design.flush(); del design
    shutil.copyfile(old.OUT / 'training_weights.npz', OUT / 'training_weights.npz')
    geometry = {'window_order_sha256': fp['window_order_sha256'],
                'answer_ids': [a['response_id'] for a in meta['answers']], 'bounds': meta['bounds'],
                'fit_positive_windows': int(weights['y'].sum()),
                'cal_positive_windows': sum(w['label'] for w in meta['windows'][NFIT:]),
                'fit_positive_answers': sum(a['label'] for a in meta['answers'][:3680]),
                'cal_positive_answers': sum(a['label'] for a in meta['answers'][3680:]),
                'fit_groups': 615, 'cal_groups': 154, 'groups_disjoint': True}
    assert [geometry[k] for k in ('fit_positive_windows', 'cal_positive_windows', 'fit_positive_answers', 'cal_positive_answers')] == [58433, 5984, 1127, 100]
    q.save(OUT / 'geometry.json', geometry)
    snapshot = {str(p.resolve()): q.sha(p) for p in source_paths()}
    q.save(OUT / 'source_binding.json', {'files_sha256': snapshot,
        'raw_base_verified_and_exact_NLL_copy': {'path': str(raw_path.resolve()), 'sha256': raw_sha, 'zero_based_column': 1024},
        'software': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')},
        'no_test': True})
    q.save(OUT / 'preparation_complete.json', {'status': 'CPU_ready_not_fitted',
        'files_sha256': {name: q.sha(OUT / name) for name in ('protocol.json', 'CPU_SELFCHECK.json',
            'source_binding.json', 'geometry.json', 'window_features.npy', 'training_weights.npz')},
        'shape': [NTOTAL, 5], 'real_fits': 0, 'seconds': time.perf_counter() - tick,
        'no_test': True, 'GPU_used': False})
    (OUT / 'REPORT.md').write_text(
        '# 固定三组浅树对照：仅完成 CPU 准备\n\n'
        'NLL 1维、GHOST 4维及两者5维，各固定100轮、深度2、最多4叶；无搜索。原3680训练答、159校准答、653979/42241窗口与质量168123的损失权重精确复用。\n\n'
        'NLL来自旧原始base.npy第1025列，是含标点的窗口实际词元平均负对数概率；feature_qa.py以位置前一状态预测当前词元。高值表示低语言概率，不自动等于事实错误。三组均不使用旧分类器分数、PCA或拟合缩放。\n\n'
        '完成全量输入顺序/标签/权重断言与3个256行合成小模型检查，未拟合真实数据、未用GPU或测试。准备矩阵约13.3MiB；三次真实拟合保守预计1–6分钟，尚未实测。\n\n'
        '该方案在看到线性结果后制定，是开发探索；不能称原论文随机森林复现，也不能仅由fit低于cal推断无过拟合。三个模型全部报告，不跨方法选赢家。\n', encoding='utf-8')
    print('CPU_PREPARATION_COMPLETE', q.read(OUT / 'preparation_complete.json'), flush=True)


def check():
    done = q.read(OUT / 'preparation_complete.json')
    assert done['status'] == 'CPU_ready_not_fitted' and done['real_fits'] == 0
    assert q.read(OUT / 'protocol.json') == protocol()
    for name, sha in done['files_sha256'].items():
        assert q.sha(OUT / name) == sha, name
    src = q.read(OUT / 'source_binding.json')
    for path, sha in src['files_sha256'].items():
        assert q.sha(path) == sha, path
    assert src['software'] == {k: importlib.metadata.version(k) for k in src['software']}
    meta = metadata(); geo = q.read(OUT / 'geometry.json')
    assert q.digest(meta['windows']) == geo['window_order_sha256']
    assert geo['answer_ids'] == [a['response_id'] for a in meta['answers']]
    x = np.load(OUT / 'window_features.npy', mmap_mode='r')
    assert x.shape == (NTOTAL, 5) and x.dtype == np.float32 and np.isfinite(x).all()
    with np.load(OUT / 'training_weights.npz') as z:
        weights = {k: z[k].copy() for k in z.files}
    assert np.array_equal(weights['y'], [w['label'] for w in meta['windows'][:NFIT]])
    assert q.sha(OUT / 'training_weights.npz') == q.sha(old.OUT / 'training_weights.npz')
    print('CPU_CHECK_PASSED_NO_REAL_FIT', flush=True)
    return meta, x, weights


def fit():
    meta, x, weights = check()
    assert not (OUT / 'started.json').exists(), 'Never overwrite a prior fit attempt.'
    q.save(OUT / 'started.json', {'code_sha256': q.sha(Path(__file__)),
        'preparation_sha256': q.sha(OUT / 'preparation_complete.json'), 'fixed_fits': 3, 'GPU_used': False})
    start = time.perf_counter(); entries = {}
    for name, columns in METHODS.items():
        tick = time.perf_counter(); values = np.asarray(x[:, columns], dtype=np.float32)
        model = HistGradientBoostingClassifier(**PARAMS)
        model.fit(values[:NFIT], weights['y'], sample_weight=weights['loss'])
        assert model.n_iter_ == 100 and not model.do_early_stopping_
        assert model.classes_.tolist() == [0, 1] and model.n_features_in_ == len(columns)
        scores = model.predict_proba(values)[:, 1]
        assert scores.shape == (NTOTAL,) and np.isfinite(scores).all()
        answer = q.answer_scores(meta, scores)
        ts = {'window': q.choose_threshold([w['label'] for w in meta['windows'][NFIT:]], scores[NFIT:]),
              'answer': q.choose_threshold([a['label'] for a in meta['answers'][3680:]], answer[3680:])}
        metric = q.metrics(meta, scores, ts)
        obj = {'model': model, 'columns': columns, 'feature_names': [NAMES[j] for j in columns],
               'thresholds': ts, 'fit_rows': NFIT, 'weights_sha256': q.sha(OUT / 'training_weights.npz')}
        (OUT / (name + '.pkl')).write_bytes(pickle.dumps(obj, protocol=5))
        np.savez_compressed(OUT / (name + '_scores.npz'), window_scores=scores, answer_scores=answer,
            window_predictions=(scores >= ts['window']['threshold']).astype(np.int8),
            answer_predictions=(answer >= ts['answer']['threshold']).astype(np.int8))
        entry = {'method': name, 'columns': columns, 'thresholds': ts, 'metrics': metric,
            'iterations': model.n_iter_, 'effective_parameters': model.get_params(),
            'seconds': time.perf_counter() - tick, 'model_sha256': q.sha(OUT / (name + '.pkl')),
            'scores_sha256': q.sha(OUT / (name + '_scores.npz'))}
        q.save(OUT / (name + '_result.json'), entry); entries[name] = entry
        print('FIXED_TREE_COMPLETE', name, metric['calibration']['windows']['f1'], metric['calibration']['answers']['f1'], entry['seconds'], flush=True)
    q.save(OUT / 'summary.json', {'methods': entries, 'fixed_fits': 3, 'no_method_selection': True,
        'seconds': time.perf_counter() - start, 'official_test_opened': False, 'GPU_used': False})
    q.save(OUT / 'complete.json', {'status': 'complete', 'fixed_fits': 3,
        'files_sha256': {name: q.sha(OUT / name) for name in ['summary.json', 'started.json',
            *[m + s for m in METHODS for s in ('.pkl', '_scores.npz', '_result.json')]]},
        'official_test_opened': False, 'GPU_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'check', 'fit'])
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=4):
            {'prepare': prepare, 'check': check, 'fit': fit}[args.stage]()
    except Exception:
        if OUT.exists():
            q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json',
                   {'stage': args.stage, 'traceback': traceback.format_exc(), 'no_test': True, 'GPU_used': False})
        raise
