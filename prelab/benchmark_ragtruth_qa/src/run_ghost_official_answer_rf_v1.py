"""Paper-structure GHOST answer RF transfer. Explicit CPU prepare/check/fit."""
from pathlib import Path
import argparse
import importlib.metadata
import pickle
import time
import traceback

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
import run_development as q

ROOT = q.ROOT
OUT = ROOT / 'results/ghost_official_answer_rf_v1'
FEATURES = ROOT / 'results/ghost_geometry_features_v1'
INPUT = ROOT / 'results/ghost_geometry_preparation_v1/feature_inputs.jsonl'
NAMES = ['adjacent_layer_cosine_change', 'layer_to_final_cosine',
         'top10_normalized_entropy', 'top10_unweighted_embedding_divergence']
PARAMS = dict(n_estimators=750, max_depth=40, min_samples_split=8,
              min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample',
              random_state=42, n_jobs=4)


def protocol():
    return {'version': 'ghost-paper-structure-answer-rf-v1',
        'classification_unit': 'Whole original answer; windows are N/A and never predicted or broadcast.',
        'features': NAMES, 'aggregation': 'Unweighted arithmetic mean over ALL raw answer BPE tokens, including punctuation and boundary/header-overlap token. Float32 mean; no lexical/gold filtering.',
        'shape': [3839, 4], 'fit_answers': 3680, 'calibration_answers': 159,
        'fit_groups': 615, 'calibration_groups': 154,
        'label': 'Original whole-answer binary label from nonempty official human span list. Not token/window OR; keep all good answers and safe refusals.',
        'classifier': 'RandomForestClassifier', 'parameters': PARAMS,
        'effective_parameters': RandomForestClassifier(**PARAMS).get_params(),
        'sample_weight': None, 'scaler': None, 'projection': None, 'new_real_fits': 1,
        'class_balance': 'Only paper balanced_subsample; no external source/answer/window loss weights.',
        'fit_scope': 'First3680 answers only; no calibration passed to fit, no hyperparameter search, no early stopping.',
        'threshold': 'One answer threshold from original cal159 choose_threshold: maximum F1, then precision, then higher threshold; >= means risk. AUROC/AP and confusion also reported.',
        'model_selection': 'One fixed RF; no alternative seeds, hyperparameters, window versions, smoothing or further postprocessing.',
        'method_identity': 'GHOST paper classifier structure and final RF hyperparameters transferred to current public-QA cohort. Author official code was unavailable; do not describe this local file as author code.',
        'seed_convention': 'The paper did not disclose a numeric final seed;42 is a fixed local transfer convention, not a recovered author seed.',
        'feature_transfer_limits': ['Existing NousResearch/Llama-2-7b-chat-hf checkpoint and NF4 unified teacher-forced replay are retained; not original released generators trajectories.',
            'Existing features use prediction-before-current state at answer_token_position-1, with states3..29 and fixed arithmetic; backbone/reading point/precision equivalence to the authors run is not established.',
            'Additional3046 answers came from other generators and share source groups with native fit answers; human answer labels remain unchanged.',
            'Repeated development calibration; no independent-test or bitwise-reproduction claim. Window F1 is not applicable.'],
        'prior_work': 'Earlier GHOST window LR/shallow-tree variants remain separate self-developed local adaptations and are not substituted for this formal answer baseline.',
        'GPU_used': False, 'official_test_opened': False,
        'resource_estimate': {'answer_matrix_KiB': 3839 * 4 * 4 / 1024,
                              'real_fit_minutes_planning': [0.1, 3], 'not_measured': True}}


def data_files():
    return [ROOT / 'fit_expansion/data/answers_fit.jsonl', ROOT / 'fit_expansion/data/tokens_fit.jsonl',
            ROOT / 'data/answers_calibration.jsonl', ROOT / 'data/tokens_calibration.jsonl']


def answer_metadata():
    af, tf, ac, tc = [q.lines(p) for p in data_files()]
    assert len(af) == len(tf) == 3680 and len(ac) == len(tc) == 159
    answers, tokens = af + ac, tf + tc
    assert len({a['response_id'] for a in answers}) == 3839
    for i, (a, t) in enumerate(zip(answers, tokens)):
        assert a['partition'] == t['partition'] == ('fit' if i < 3680 else 'calibration')
        assert a['eligible'] and a['quality'] == 'good'
        assert a['response_id'] == t['response_id'] and a['answer_sha256'] == t['answer_sha256']
        assert a['original_labels'] == t['original_labels']
        assert a['label'] == t['answer_risk'] == int(bool(a['original_labels']))
    groups = [{a['group_id'] for a in rows} for rows in (af, ac)]
    assert [len(g) for g in groups] == [615, 154] and not groups[0] & groups[1]
    y = np.asarray([a['label'] for a in answers], dtype=np.int8)
    assert int(y[:3680].sum()) == 1127 and int(y[3680:].sum()) == 100
    return answers, tokens, y


def aggregate_all(answers, tokens):
    done = q.read(FEATURES / 'features_complete.json')
    assert done['status'] == 'complete' and done['records'] == 3839 and done['raw_answer_tokens'] == 708506
    assert done['no_test'] and not done['trained'] and done['all_records_validated']
    for name, sha in done['files_sha256'].items(): assert q.sha(FEATURES / name) == sha
    manifest = q.read(FEATURES / 'feature_manifest.json'); sig = q.read(FEATURES / 'signature.json')
    assert done['signature_sha256'] == manifest['signature_sha256'] == q.digest(sig)
    assert manifest['records'] == len(manifest['entries']) == 3839
    assert manifest['feature_names'] == sig['features'] == NAMES
    assert not manifest['labels_used'] and not manifest['test_opened']
    assert q.sha(INPUT) == sig['source_sha256'][str(INPUT.resolve())]
    rows = q.lines(INPUT)
    assert len(rows) == 3839
    ids = [a['response_id'] for a in answers]
    assert ids == manifest['answer_order'] == [r['response_id'] for r in rows]
    result = np.empty((3839, 4), dtype=np.float32); count = 0
    for i, (r, a, t, m) in enumerate(zip(rows, answers, tokens, manifest['entries'])):
        rid = a['response_id']; n = t['token_count']
        assert r['answer_sha256'] == a['answer_sha256']
        assert r['group_id'] == a['group_id'] == m['group_id']
        assert r['source_id'] == a['source_id'] == m['source_id']
        assert r['partition'] == a['partition'] == m['partition']
        assert r['answer_token_ids'] == t['token_ids'] and len(r['answer_token_ids']) == n > 0
        assert r['answer_token_positions'] == t['answer_token_positions']
        assert r['response_token_offsets'] == t['response_token_offsets']
        assert r['response_token_offsets_raw'] == t['response_token_offsets_raw']
        path = FEATURES / 'features' / f'{i:05d}.npz'
        assert m['file'] == path.name and q.sha(path) == m['npz_sha256']
        assert q.read(path.with_suffix('.json')) == m
        pos = np.asarray(r['answer_token_positions'], dtype=np.int64)
        off = np.asarray(r['response_token_offsets'], dtype=np.int64)
        raw = np.asarray(r['response_token_offsets_raw'], dtype=np.int64)
        axes = {'token_ids': np.asarray(r['answer_token_ids'], dtype=np.int64), 'answer_token_positions': pos,
                'predictor_positions': pos - 1, 'token_start': off[:, 0], 'token_end': off[:, 1],
                'token_start_raw': raw[:, 0], 'token_end_raw': raw[:, 1]}
        with np.load(path, allow_pickle=False) as z:
            assert str(z['response_id'].item()) == rid and str(z['record_index'].item()) == str(i)
            assert str(z['record_sha256'].item()) == q.digest(r)
            assert str(z['signature_sha256'].item()) == done['signature_sha256']
            for key, expected in axes.items():
                assert z[key].dtype == np.int64 and np.array_equal(z[key], expected)
            values = z['ghost_features']
            assert values.dtype == np.float32 and values.shape == (n, 4) and np.isfinite(values).all()
            result[i] = values.mean(axis=0, dtype=np.float32)
        count += n
    assert count == 708506 and np.isfinite(result).all()
    return result


def tiny_check():
    values = np.asarray([[0., 0., 0., 0.], [40., 80., 120., 160.],
                         [0., 0., 0., 0.], [0., 0., 0., 0.]], dtype=np.float32)
    assert np.array_equal(values.mean(0), [10., 20., 30., 40.])
    assert np.array_equal(values[:2].mean(0), [20., 40., 60., 80.])
    rng = np.random.default_rng(42)
    x = rng.normal(size=(192, 4)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] > .2).astype(np.int8)
    model = RandomForestClassifier(**PARAMS)
    model.fit(x, y)  # Synthetic only; deliberately no sample_weight argument.
    p = model.predict_proba(x)[:, 1]
    assert model.classes_.tolist() == [0, 1] and model.n_features_in_ == 4
    assert len(model.estimators_) == 750 and model.class_weight == 'balanced_subsample'
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    yy, pp = np.asarray([0, 1, 0, 1]), np.asarray([.1, .4, .4, .9])
    ts = q.choose_threshold(yy, pp)
    options = [np.nextafter(pp.max(), np.inf), *np.unique(pp)]
    expected = []
    for threshold in options:
        m = q.count(yy, pp, threshold); expected.append((m['f1'], m['precision'], threshold))
    assert (ts['f1'], ts['precision'], ts['threshold']) == max(expected)
    return {'status': 'passed', 'synthetic_rows': 192, 'synthetic_classes': 2,
            'synthetic_trees': 750, 'all_BPE_unweighted_mean_including_punctuation': True,
            'short_answer_uses_all_actual_tokens': True, 'no_external_sample_weights': True,
            'probability_axis_and_threshold_bruteforce': True, 'real_fits': 0, 'GPU_used': False}


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Preserve prior preparation.'
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol())
    start = time.perf_counter()
    q.save(OUT / 'CPU_SELFCHECK.json', tiny_check())
    answers, tokens, y = answer_metadata()
    x = aggregate_all(answers, tokens)
    np.save(OUT / 'answer_features.npy', x); np.save(OUT / 'answer_labels.npy', y)
    q.save(OUT / 'answer_order.json', [{'response_id': a['response_id'], 'source_id': a['source_id'],
        'group_id': a['group_id'], 'partition': a['partition'], 'raw_token_count': t['token_count']}
        for a, t in zip(answers, tokens)])
    paths = [Path(__file__), Path(q.__file__), INPUT, *data_files(), FEATURES / 'features_complete.json',
             *[FEATURES / n for n in q.read(FEATURES / 'features_complete.json')['files_sha256']]]
    q.save(OUT / 'source_binding.json', {'files_sha256': {str(p.resolve()): q.sha(p) for p in paths},
        'all_3839_record_NPZ_and_JSON_verified': True, 'all_token_axes_and_means_verified': True,
        'software': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')},
        'no_test': True})
    q.save(OUT / 'preparation_complete.json', {'status': 'CPU_ready_not_fitted', 'shape': [3839, 4],
        'raw_tokens': 708506, 'fit_positive_answers': int(y[:3680].sum()), 'cal_positive_answers': int(y[3680:].sum()),
        'files_sha256': {n: q.sha(OUT / n) for n in ('protocol.json', 'CPU_SELFCHECK.json', 'source_binding.json',
            'answer_features.npy', 'answer_labels.npy', 'answer_order.json')},
        'real_fits': 0, 'seconds': time.perf_counter() - start, 'GPU_used': False, 'no_test': True})
    (OUT / 'REPORT.md').write_text('# GHOST论文结构迁移：整答随机森林\n\n'
        '仅CPU准备已完成：3839答、708506个原回答BPE逐项核对；每答对全部四维特征作无权均值，矩阵3839×4。原3680训练/159校准整答标签不变。\n\n'
        '固定RF为750棵树、深度40、split8、leaf2、sqrt、balanced_subsample、seed42、4线程。无外部sample_weight、缩放、搜索或后处理；校准仅选一个整答F1阈值。192条合成两类RF和均值/阈值检查通过，尚未真实fit。窗口指标明确N/A，不广播整答结果。\n\n'
        '作者未提供可用官方实现，论文未公开最终数值seed；42是本地迁移约定。复用当前Llama NF4统一重放、原读取位置的特征，因此是保持论文分类结构的迁移，不是逐数值复现。旧窗口LR/浅树属于单独自研实验。\n', encoding='utf-8')
    print('ANSWER_RF_PREPARE_COMPLETE', q.read(OUT / 'preparation_complete.json'), flush=True)


def check():
    done = q.read(OUT / 'preparation_complete.json')
    assert done['status'] == 'CPU_ready_not_fitted' and done['real_fits'] == 0
    assert q.read(OUT / 'protocol.json') == protocol()
    for name, sha in done['files_sha256'].items(): assert q.sha(OUT / name) == sha
    binding = q.read(OUT / 'source_binding.json')
    for path, sha in binding['files_sha256'].items(): assert q.sha(path) == sha
    assert binding['software'] == {k: importlib.metadata.version(k) for k in binding['software']}
    answers, tokens, y = answer_metadata()
    x = np.load(OUT / 'answer_features.npy'); saved_y = np.load(OUT / 'answer_labels.npy')
    assert x.dtype == np.float32 and x.shape == (3839, 4) and np.isfinite(x).all()
    assert np.array_equal(saved_y, y)
    assert [r['response_id'] for r in q.read(OUT / 'answer_order.json')] == [a['response_id'] for a in answers]
    print('ANSWER_RF_CHECK_PASSED_NO_FIT', flush=True)
    return x, y


def fit():
    x, y = check()
    assert not (OUT / 'started.json').exists(), 'No overwrite/refit.'
    q.save(OUT / 'started.json', {'code_sha256': q.sha(Path(__file__)),
        'preparation_sha256': q.sha(OUT / 'preparation_complete.json'), 'real_fits': 1, 'GPU_used': False})
    start = time.perf_counter()
    model = RandomForestClassifier(**PARAMS)
    model.fit(x[:3680], y[:3680])
    assert model.classes_.tolist() == [0, 1] and len(model.estimators_) == 750
    scores = model.predict_proba(x)[:, 1]
    assert scores.shape == (3839,) and np.isfinite(scores).all()
    threshold = q.choose_threshold(y[3680:], scores[3680:])
    metrics = {part: q.count(y[lo:hi], scores[lo:hi], threshold['threshold'])
               for part, lo, hi in [('fit', 0, 3680), ('calibration', 3680, 3839)]}
    obj = {'model': model, 'answer_threshold': threshold, 'feature_names': NAMES,
           'fit_rows': 3680, 'sample_weight': None, 'scaler': None}
    (OUT / 'answer_rf.pkl').write_bytes(pickle.dumps(obj, protocol=5))
    np.savez_compressed(OUT / 'answer_scores.npz', answer_scores=scores,
        answer_predictions=(scores >= threshold['threshold']).astype(np.int8))
    result = {'method': 'ghost_paper_structure_answer_rf', 'threshold': threshold, 'answer_metrics': metrics,
        'window_metrics': None, 'window_metric_reason': 'N/A: whole-answer model; no broadcast or local model.',
        'parameters': model.get_params(), 'real_fits': 1, 'sample_weight': None, 'scaler': None,
        'seconds': time.perf_counter() - start, 'model_sha256': q.sha(OUT / 'answer_rf.pkl'),
        'scores_sha256': q.sha(OUT / 'answer_scores.npz'), 'no_test': True, 'GPU_used': False}
    q.save(OUT / 'summary.json', result)
    q.save(OUT / 'complete.json', {'status': 'complete', 'real_fits': 1, 'windows': 'N/A',
        'files_sha256': {n: q.sha(OUT / n) for n in ('started.json', 'answer_rf.pkl', 'answer_scores.npz', 'summary.json')},
        'no_test': True, 'GPU_used': False})
    print('ANSWER_RF_COMPLETE', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'check', 'fit'])
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=4):
            {'prepare': prepare, 'check': check, 'fit': fit}[args.stage]()
    except Exception:
        if OUT.exists(): q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
        raise
