"""Lookback Lens author-structure8-BPE LR transfer; CPU only, no test."""
from pathlib import Path
import argparse
import importlib.metadata
import pickle
import time
import traceback
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import run_development as q

ROOT = q.ROOT
OUT = ROOT / 'results/lookback_official_span_v1'
OLD = ROOT / 'data/lookback_controls_v2'
NEW = ROOT / 'fit_expansion/llama_features_v3'
KEY = 'lb_prefix_pre_header'
K = 8


def protocol():
    return {'version': 'lookback-author-structure-sliding8-v1', 'author_revision': 'e0a1fa3a898fbf6512af7be5567dea8ffe7a6620',
        'paper': 'https://aclanthology.org/2024.emnlp-main.84.pdf',
        'author_code': 'https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step03_lookback_lens.py',
        'feature': 'Existing lb_prefix_pre_header raw token32layers x32heads=1024; concatenate layer-major/head-major then unweighted mean over8 raw BPE slots.',
        'window': 'Exactly8 native raw answer BPE, stride1, complete windows only as author convert_to_token_level; no gold-dependent segmentation, punctuation slots remain. Answers shorter than8 retain metadata but have no span score.',
        'label': 'Risk1 iff any frozen gold lexical risk token in the8 slots; otherwise0. This reverses author factual1 convention without changing binary classifier structure.',
        'classifier_call': 'sklearn.linear_model.LogisticRegression(max_iter=1000)',
        'effective_parameters': LogisticRegression(max_iter=1000).get_params(),
        'sample_weight': None, 'scaler': None, 'C_search': False, 'extra_features': [],
        'no_smoothing_or_fusion': True, 'new_real_fits': 1, 'CPU_threads': 4,
        'data_protocol_adaptation': 'Reuse current3680 fit answers and159 calibration answers with615/154 disjoint material groups instead of author within-span random/two-fold splits; original human labels unchanged. Sealed test150 never opened.',
        'primary_metric': 'Calibration sliding-span AUROC for fixed8-BPE windows (paperTable2 metric); not predefined-span or original4-BPE F1.',
        'extra_window_F1': 'For project interface only: threshold selected once using in-sample FIT window scores/labels, maximizing F1 then precision then higher threshold. Calibration never selects a threshold.',
        'extra_answer_F1': 'Additional interface adaptation, not author output: max score over all complete8-BPE windows per answer; its separate threshold selected only on FIT answers with windows by the same rule. No-window answers are recorded N/A, never filled with0.',
        'convergence': 'Keep max_iter1000 fixed; record any warning and n_iter, never enlarge budget or change model after seeing scores.',
        'limits': ['Frozen NF4 teacher-forced Llama replay and actual chat/header boundary differ from the original author generation/version environment; do not claim bitwise reproduction.',
            'Added3046 other-generator responses are replayed by the same Llama checkpoint, not native trajectories of those generators.',
            'Fit-only thresholds use in-sample scores and can overfit. The calibration set has previously been repeatedly used in development; it is not an untouched test.',
            'No head selection, feature scaling, weighting, NLL/HARP, local smoothing or answer-score broadcast is added.'],
        'GPU_used': False, 'official_test_opened': False}


def data_paths():
    return [ROOT / 'fit_expansion/data/answers_fit.jsonl', ROOT / 'fit_expansion/data/tokens_fit.jsonl',
            ROOT / 'data/answers_calibration.jsonl', ROOT / 'data/tokens_calibration.jsonl']


def metadata():
    af, tf, ac, tc = [q.lines(p) for p in data_paths()]
    assert len(af) == len(tf) == 3680 and len(ac) == len(tc) == 159
    answers, tokens = af + ac, tf + tc
    assert len({a['response_id'] for a in answers}) == 3839
    for i, (a, t) in enumerate(zip(answers, tokens)):
        assert a['response_id'] == t['response_id'] and a['answer_sha256'] == t['answer_sha256']
        assert a['partition'] == t['partition'] == ('fit' if i < 3680 else 'calibration')
        assert a['eligible'] and a['quality'] == 'good'
        assert a['label'] == t['answer_risk'] == int(bool(a['original_labels']))
        assert len(t['token_ids']) == len(t['lexical_mask']) == len(t['risk_mask']) == t['token_count']
        assert np.all(~np.asarray(t['risk_mask'], bool) | np.asarray(t['lexical_mask'], bool))
    groups = [{a['group_id'] for a in aa} for aa in (af, ac)]
    assert [len(g) for g in groups] == [615, 154] and not groups[0] & groups[1]
    return answers, tokens


def mean8(x):
    if len(x) < K: return np.empty((0, x.shape[1]), dtype=np.float32)
    return np.stack([x[j:j+K].mean(axis=0, dtype=np.float32) for j in range(len(x)-K+1)])


def tiny_check():
    x = np.zeros((10, 1024), np.float32); x[1] = 80
    result = mean8(x)
    assert result.shape == (3, 1024) and np.all(result[:2] == 10) and np.all(result[2] == 0)
    assert mean8(x[:7]).shape == (0, 1024)
    mask = np.zeros(10, bool); mask[8] = True
    labels = np.asarray([mask[j:j+K].any() for j in range(3)])
    assert labels.tolist() == [False, True, True]
    rng = np.random.default_rng(42)
    xx = rng.uniform(0, 1, (96, 1024)).astype(np.float32)
    yy = np.tile([0, 1], 48)
    model = LogisticRegression(max_iter=1000)
    model.fit(xx, yy)
    assert model.classes_.tolist() == [0, 1] and model.n_features_in_ == 1024
    p = model.predict_proba(xx)[:, 1]
    assert np.isfinite(p).all() and model.class_weight is None
    fit_y, fit_p = [0, 1, 0, 1], [.1, .4, .4, .9]
    threshold = q.choose_threshold(fit_y, fit_p)
    options = [np.nextafter(.9, np.inf), .1, .4, .9]
    brute = [(q.count(fit_y, fit_p, t)['f1'], q.count(fit_y, fit_p, t)['precision'], t) for t in options]
    assert (threshold['f1'], threshold['precision'], threshold['threshold']) == max(brute)
    return {'status': 'passed', 'synthetic_fit_rows': 96, 'complete8_stride1_mean': True,
            'punctuation_slot_retained': True, 'any_risk_OR': True, 'short_answer_no_fabricated_window': True,
            'default_LR_only_max_iter_changed': True, 'fit_threshold_bruteforce': True, 'real_fits': 0, 'GPU_used': False}


def sources():
    result = {}
    for folder, count in [(OLD, 793), (NEW, 3046)]:
        manifest = q.read(folder / 'feature_manifest.json'); sig = q.read(folder / 'signature.json')
        assert manifest['complete'] and manifest['completed_count'] == len(manifest['records']) == count
        assert not manifest['test_read'] and not manifest['labels_used']
        assert manifest['signature_sha256'] == q.digest(sig)
        layouts = {r['response_id']: r for r in q.lines(folder / 'layouts.jsonl')}
        assert len(layouts) == count
        if folder == OLD: assert sig['layouts_sha256'] == q.sha(folder / 'layouts.jsonl')
        else: assert sig['files_sha256'][str((folder / 'layouts.jsonl').resolve())] == q.sha(folder / 'layouts.jsonl')
        for record in manifest['records']:
            rid = record['response_id']; assert rid not in result
            assert record['signature_sha256'] == manifest['signature_sha256']
            assert record['layout_sha256'] == q.digest(layouts[rid])
            result[rid] = (folder, record, layouts[rid])
    return result


def consumed():
    return [Path(__file__), Path(q.__file__), *data_paths(), OUT / 'AUTHOR_SOURCE.json',
            OUT / 'author_step03_lookback_lens.py',
            *[p / n for p in (OLD, NEW) for n in ('feature_manifest.json', 'signature.json', 'layouts.jsonl')]]


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Preserve existing preparation.'
    OUT.mkdir(parents=True, exist_ok=True)
    author = q.read(OUT / 'AUTHOR_SOURCE.json')
    assert q.sha(OUT / author['local_source']) == author['sha256']
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'CPU_SELFCHECK.json', tiny_check())
    start = time.perf_counter(); answers, tokens = metadata(); source = sources()
    assert set(source) == {a['response_id'] for a in answers}
    counts = [max(0, t['token_count'] - K + 1) for t in tokens]
    nfit, ncal = sum(counts[:3680]), sum(counts[3680:])
    assert (nfit, ncal) == (639955, 41685)
    total = nfit + ncal
    x = np.lib.format.open_memmap(OUT / 'span_features.npy', mode='w+', dtype=np.float32, shape=(total, 1024))
    y = np.empty(total, np.int8); starts = np.empty(total, np.int32); ai = np.empty(total, np.int32)
    offsets = np.empty((3839, 2), np.int64); cursor = raw_count = pure_punctuation = 0
    ledger = []
    for i, (a, t, count) in enumerate(zip(answers, tokens, counts)):
        rid = a['response_id']; folder, rec, layout = source[rid]
        assert rec['partition'] == a['partition'] and rec['group_id'] == a['group_id'] and rec['source_id'] == a['source_id']
        pos = np.asarray(t['answer_token_positions'], np.int64)
        assert layout['answer_token_positions'] == pos.tolist()
        assert layout['pre_query_positions'] == (pos - 1).tolist()
        assert layout['header_token_positions'][-1] == pos[0] - 1
        assert layout['prefix_context_token_positions'] == list(range(layout['header_token_positions'][0]))
        path = folder / 'features' / (rid + '.npz'); side = path.with_suffix('.json')
        assert Path(rec['npz']).resolve() == path.resolve() and Path(rec['json']).resolve() == side.resolve()
        assert q.sha(path) == rec['npz_sha256'] and q.sha(side) == rec['json_sha256']
        meta = q.read(side)
        for key in ('response_id', 'source_id', 'group_id', 'partition', 'signature_sha256', 'plan_sha256', 'layout_sha256'):
            assert meta[key] == rec[key]
        assert not meta['labels_used'] and not meta['test_read']
        with np.load(path, allow_pickle=False) as z:
            for key in ('token_ids', 'answer_token_positions', 'response_token_offsets', 'response_token_offsets_raw'):
                assert np.array_equal(z[key], t[key])
            features = z[KEY]
            assert features.shape == (t['token_count'], 1024) and features.dtype == np.float32
            assert np.isfinite(features).all() and np.all((features >= 0) & (features <= 1))
            values = mean8(features)
            assert values.shape == (count, 1024)
            if count:
                assert np.array_equal(values[0], features[:8].mean(0)) and np.array_equal(values[-1], features[-8:].mean(0))
            x[cursor:cursor+count] = values
        risk = np.asarray(t['risk_mask'], bool); lex = np.asarray(t['lexical_mask'], bool)
        labels = np.asarray([risk[j:j+K].any() for j in range(count)], dtype=np.int8)
        y[cursor:cursor+count] = labels; starts[cursor:cursor+count] = np.arange(count); ai[cursor:cursor+count] = i
        offsets[i] = [cursor, cursor + count]
        pure_punctuation += sum(not lex[j:j+K].any() for j in range(count))
        ledger.append({'response_id': rid, 'source_id': a['source_id'], 'group_id': a['group_id'],
            'partition': a['partition'], 'answer_label': a['label'], 'raw_tokens': t['token_count'],
            'window_left': cursor, 'window_right': cursor + count, 'windows': count,
            'no_span_reason': 'shorter_than8_raw_BPE' if not count else None,
            'source_npz_sha256': rec['npz_sha256'], 'source_json_sha256': rec['json_sha256']})
        cursor += count; raw_count += t['token_count']
        if (i+1) % 250 == 0: print('LOOKBACK8_PREPARE', i+1, 3839, flush=True)
    assert cursor == total and raw_count == 708506
    x.flush(); del x
    np.savez_compressed(OUT / 'span_geometry.npz', labels=y, answer_index=ai, token_start=starts,
                        token_end=starts + K, answer_window_offsets=offsets)
    q.save(OUT / 'answer_order.json', ledger)
    q.save(OUT / 'source_binding.json', {'files_sha256': {str(p.resolve()): q.sha(p) for p in consumed()},
        'software': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')},
        'all3839_NPZ_JSON_hashes_and_raw_token_axes_verified': True, 'feature': KEY, 'no_test': True})
    q.save(OUT / 'preparation_complete.json', {'status': 'CPU_ready_not_fitted', 'answers': 3839,
        'fit_answers': 3680, 'cal_answers': 159, 'fit_scored_answers': sum(c > 0 for c in counts[:3680]),
        'cal_scored_answers': sum(c > 0 for c in counts[3680:]), 'fit_windows': nfit, 'cal_windows': ncal,
        'fit_positive_windows': int(y[:nfit].sum()), 'cal_positive_windows': int(y[nfit:].sum()),
        'raw_tokens': raw_count, 'pure_nonlexical_windows_retained': pure_punctuation,
        'no_window_answers': [r for r in ledger if r['windows'] == 0],
        'files_sha256': {n: q.sha(OUT / n) for n in ('protocol.json', 'CPU_SELFCHECK.json', 'source_binding.json',
            'span_features.npy', 'span_geometry.npz', 'answer_order.json')},
        'seconds': time.perf_counter() - start, 'real_fits': 0, 'GPU_used': False, 'no_test': True})
    print('LOOKBACK8_PREPARATION_COMPLETE', q.read(OUT / 'preparation_complete.json'), flush=True)


def check():
    done = q.read(OUT / 'preparation_complete.json')
    assert done['status'] == 'CPU_ready_not_fitted' and done['real_fits'] == 0
    assert q.read(OUT / 'protocol.json') == protocol()
    for name, sha in done['files_sha256'].items(): assert q.sha(OUT / name) == sha
    binding = q.read(OUT / 'source_binding.json')
    for path, sha in binding['files_sha256'].items(): assert q.sha(path) == sha
    assert binding['software'] == {k: importlib.metadata.version(k) for k in binding['software']}
    answers, tokens = metadata(); ledger = q.read(OUT / 'answer_order.json')
    assert [r['response_id'] for r in ledger] == [a['response_id'] for a in answers]
    x = np.load(OUT / 'span_features.npy', mmap_mode='r')
    with np.load(OUT / 'span_geometry.npz', allow_pickle=False) as z: geo = {k: z[k].copy() for k in z.files}
    assert x.shape == (681640, 1024) and x.dtype == np.float32
    assert len(geo['labels']) == len(x) and geo['answer_window_offsets'].shape == (3839, 2)
    for i, (r, t) in enumerate(zip(ledger, tokens)):
        lo, hi = geo['answer_window_offsets'][i]
        assert [int(lo), int(hi)] == [r['window_left'], r['window_right']]
        assert hi - lo == max(0, t['token_count'] - K + 1)
        assert np.all(geo['answer_index'][lo:hi] == i)
        assert np.array_equal(geo['token_start'][lo:hi], np.arange(hi-lo))
        assert np.array_equal(geo['token_end'][lo:hi], np.arange(hi-lo)+K)
        expected = [int(any(t['risk_mask'][j:j+K])) for j in range(hi-lo)]
        assert np.array_equal(geo['labels'][lo:hi], expected)
    for lo in range(0, len(x), 16384): assert np.isfinite(x[lo:lo+16384]).all()
    result = {'status': 'passed', 'real_fits': 0, 'all_labels_geometry_and_frozen_source_exact': True,
              'n_features': 1024, 'fit_windows': 639955, 'cal_windows': 41685, 'no_test': True, 'GPU_used': False}
    q.save(OUT / 'CPU_OUTPUT_CHECK.json', result)
    print('LOOKBACK8_CHECK_PASSED_NO_REAL_FIT', flush=True)
    return x, geo, ledger


def answer_max(scores, offsets):
    result = np.full(len(offsets), np.nan, dtype=np.float64)
    for i, (lo, hi) in enumerate(offsets):
        if hi > lo: result[i] = np.max(scores[lo:hi])
    return result


def fit():
    x, geo, ledger = check(); y = geo['labels']; nfit = 639955
    assert not (OUT / 'started.json').exists(), 'Never overwrite a real fit attempt.'
    q.save(OUT / 'started.json', {'code_sha256': q.sha(Path(__file__)), 'preparation_sha256': q.sha(OUT / 'preparation_complete.json'),
        'new_real_fits': 1, 'threshold_scope': 'fit only', 'GPU_used': False})
    start = time.perf_counter()
    classifier = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier.fit(x[:nfit], y[:nfit])
    warning_log = [{'type': type(w.message).__name__, 'message': str(w.message)} for w in caught]
    assert classifier.classes_.tolist() == [0, 1] and classifier.n_features_in_ == 1024
    fit_seconds = time.perf_counter() - start
    scores = np.empty(len(x), np.float64)
    for lo in range(0, len(x), 16384): scores[lo:lo+16384] = classifier.predict_proba(x[lo:lo+16384])[:, 1]
    assert np.isfinite(scores).all()
    ay = np.asarray([r['answer_label'] for r in ledger]); a_score = answer_max(scores, geo['answer_window_offsets'])
    fit_ai = np.flatnonzero(np.isfinite(a_score[:3680])); cal_ai = np.arange(3680, 3839)
    assert len(fit_ai) == 3678 and np.isfinite(a_score[cal_ai]).all()
    thresholds = {'window': q.choose_threshold(y[:nfit], scores[:nfit]),
                  'answer': q.choose_threshold(ay[fit_ai], a_score[fit_ai])}
    # Freeze threshold record before any calibration metric is calculated.
    q.save(OUT / 'fit_only_thresholds.json', {'thresholds': thresholds, 'fit_window_rows': nfit,
        'fit_answer_rows': len(fit_ai), 'calibration_used': False})
    metrics = {}
    for part, lo, hi, aix in [('fit', 0, nfit, fit_ai), ('calibration', nfit, len(x), cal_ai)]:
        metrics[part] = {'spans8': q.count(y[lo:hi], scores[lo:hi], thresholds['window']['threshold']),
                         'answers_extra_adaptation': q.count(ay[aix], a_score[aix], thresholds['answer']['threshold'])}
    obj = {'model': classifier, 'feature': KEY, 'window_size': K, 'thresholds_fit_only': thresholds,
           'sample_weight': None, 'scaler': None, 'fit_windows': nfit}
    (OUT / 'lookback_lr.pkl').write_bytes(pickle.dumps(obj, protocol=5))
    np.savez_compressed(OUT / 'scores.npz', window_scores=scores, window_predictions=(scores >= thresholds['window']['threshold']).astype(np.int8),
                        answer_scores=a_score, answer_has_score=np.isfinite(a_score),
                        answer_predictions=np.where(np.isfinite(a_score), a_score >= thresholds['answer']['threshold'], -1).astype(np.int8))
    summary = {'method': 'lookback_lens_author_structure_sliding8', 'primary_metric': 'calibration8-BPE sliding-span AUROC',
        'primary_AUROC': metrics['calibration']['spans8']['auroc'], 'metrics': metrics,
        'thresholds_fit_only': thresholds, 'classifier_parameters': classifier.get_params(),
        'n_iter': classifier.n_iter_.tolist(), 'warnings': warning_log, 'fit_seconds': fit_seconds,
        'total_seconds': time.perf_counter() - start, 'real_fits': 1, 'no_test': True, 'GPU_used': False,
        'model_sha256': q.sha(OUT / 'lookback_lr.pkl'), 'scores_sha256': q.sha(OUT / 'scores.npz'),
        'answer_F1_is_extra_adaptation': True, 'window_F1_uses_fit_only_threshold': True,
        'short_fit_answers_NA': [r['response_id'] for r in ledger if not r['windows']]}
    q.save(OUT / 'summary.json', summary)
    lines = ['# Lookback Lens：作者结构8-BPE迁移', '',
        '按作者代码固定8个原生BPE滑窗、stride1，均值1024维；唯一分类器为`LogisticRegression(max_iter=1000)`，其余默认。没有scaler、sample_weight、C搜索、NLL/HARP或平滑。', '',
        '| 划分 | 8-BPE窗 | span AUROC（主指标） | span AP | 窗口F1（额外） | 整答F1（额外） |', '|---|---:|---:|---:|---:|---:|']
    for part, m in metrics.items():
        w, a = m['spans8'], m['answers_extra_adaptation']
        lines.append(f"| {part} | {w['n']} | {w['auroc']:.6f} | {w['average_precision']:.6f} | {w['f1']:.6f} | {a['f1']:.6f} |")
    w = metrics['calibration']['spans8']; a = metrics['calibration']['answers_extra_adaptation']
    lines += ['', f"校准窗口TP/FP/FN/TN={w['tp']}/{w['fp']}/{w['fn']}/{w['tn']}；整答={a['tp']}/{a['fp']}/{a['fn']}/{a['tn']}。",
        f"两个阈值只由fit内分数确定：窗口{thresholds['window']['threshold']:.9f}，整答{thresholds['answer']['threshold']:.9f}；未用cal选阈值或参数。整答采用全部完整8-BPE窗max，是项目接口适配，不是作者原输出。",
        '保留3680/159答的615/154材料组隔离；2条fit短答（12818、14641）不足8BPE，按作者完整窗口规则无span，记录N/A，未补0。cal159全部有窗。主指标为8-BPE sliding-span AUROC，不能把这里的窗口F1当作原4-BPE成绩。',
        f"真实拟合{fit_seconds:.2f}秒，迭代{classifier.n_iter_.tolist()}；警告{warning_log}。数据采用已有Llama NF4统一重放、真实header及pre-read特征，存在骨干/软件/读取环境迁移差异，不宣称逐数值复现。",
        '材料组拆分、人标风险OR及fit-only F1阈值均为明确的数据/报告适配；fit阈值来自训练内分数，可能过拟合，cal也已反复开发。sealed test150未读。', '',
        '[原论文§2.1、Table2、AppendixC.1](https://aclanthology.org/2024.emnlp-main.84.pdf)；[锁定作者代码](https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step03_lookback_lens.py)。本目录AUTHOR_SOURCE.json记录下载hash；没有下载模型或新数据。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'status': 'complete', 'real_fits': 1, 'no_test': True, 'GPU_used': False,
        'files_sha256': {n: q.sha(OUT / n) for n in ('started.json', 'lookback_lr.pkl', 'fit_only_thresholds.json',
            'scores.npz', 'summary.json', 'REPORT.md')}, 'convergence_warning': any(w['type'] == 'ConvergenceWarning' for w in warning_log)})
    print('LOOKBACK8_FIT_COMPLETE', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'check', 'dry-run', 'fit'])
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=4):
            {'prepare': prepare, 'check': check, 'dry-run': check, 'fit': fit}[args.stage]()
    except Exception:
        if OUT.exists(): q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
        raise
