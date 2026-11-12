"""CPU-only five-definition Lookback development; official QA test stays sealed.

Four new definitions x three C values are fitted. The exact legacy anchor uses
the three original frozen models, with bit-exact matrix/score/calibration gates.
Never fit or report a partial feature cohort.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_development as original

ROOT = original.ROOT
DATA = ROOT / 'data/lookback_controls_v1'
OUT = ROOT / 'results/lookback_controls_v1'
OLD = ROOT / 'results/development_v1'
PROTOCOL = ROOT / 'lookback_controls_scoring_protocol.json'
METHODS = ('lb_source_pre_header', 'lb_prefix_pre_header', 'lb_source_post_header',
           'lb_prefix_post_header', 'lb_source_post_legacy')
ANCHOR = 'lb_source_post_legacy'
BATCH = original.BATCH
FIT_N, CAL_N, N = 168123, 42241, 210364


def cfg():
    reference = original.read(ROOT / 'development_protocol.json')
    assert reference == original.protocol()
    return {'version': 'qa-lookback-definition-scoring-v1',
        'scope': 'Frozen official-train fit634/calibration159 only; no sealed official-test or withheld content',
        'methods': list(METHODS), 'primary_closer_official': 'lb_prefix_pre_header',
        'width': 1024, 'C': [.001, .01, .1], 'seed': reference['seed'],
        'candidate_count': 15, 'new_LR_fits': 12, 'legacy_frozen_model_replays': 3,
        'extract_protocol_sha256': original.sha(ROOT / 'lookback_controls_protocol.json'),
        'original_protocol_sha256': original.sha(ROOT / 'development_protocol.json'),
        'original_runner_sha256': original.sha(Path(original.__file__)),
        'gold_manifest_sha256': original.sha(ROOT / 'data/gold_manifest.json'),
        'geometry': 'Every frozen eligible 4 raw BPE window, stride1, mean all four raw token LB vectors. N<4 mean all actual raw tokens. No lexical compression, future smoothing, or positive-window selection.',
        'counts': {'fit_answers': 634, 'calibration_answers': 159, 'fit_windows': FIT_N, 'calibration_windows': CAL_N},
        'base_weight': reference['base_weight'], 'loss_weight': reference['loss_weight'],
        'standardization': reference['standardization'], 'lr': reference['lr'],
        'threshold': reference['threshold'], 'selection': reference['selection'],
        'answer_score': reference['answer_score'], 'answer_label': reference['answer_label'],
        'safe_refusals': reference['safe_refusals'],
        'no_NLL_PCA_or_hidden': True, 'no_calibration_refit': True,
        'legacy_gate': 'All new anchor token arrays already equal old cached LB. Require all210364 window vectors, standardized fit matrix, all three old-model score arrays, calibration thresholds, selection keys and every metric to match original lookback_mean exactly.',
        'legacy_fit_policy': 'Reuse three original frozen models rather than re-fit identical data; preserve original weights/scaler and list them as replays, not new training runs.',
        'eligibility_gate': 'Require complete793 feature manifest with exact original identities and per-file/coordinate/layout signature checks BEFORE any fit. Partial feature cohorts are errors, never evaluation subsets.',
        'resource_estimate': {'raw_matrix_gib': N*1024*4*5/2**30, 'standardized_fit_matrix_gib': FIT_N*1024*4*4/2**30,
                              'reserve_free_disk_gib': 9, 'cpu_minutes_planning': [12, 30], 'not_measured': True},
        'reporting': 'All15 candidate fit/cal results, five within-family calibration-selected C values. Report adaptations and original-code mismatch. Calibration selection remains optimistic, not final test.'}


def freeze(path, value):
    if path.exists():
        assert original.read(path) == value, ('Frozen scoring file changed', str(path))
    else:
        original.save(path, value)


def legacy_snapshot():
    complete = original.read(OLD / 'complete.json')
    assert complete['status'] == 'complete_development_only' and not complete['official_test_opened']
    assert complete['script_sha256'] == original.sha(Path(original.__file__))
    names = ['summary.json', 'source_snapshot.json', 'training_weights.npz', 'fit_keys.json',
             'matrix_manifest.json', 'score_index.json']
    names += [f'lookback_mean_C{c:g}{s}' for c in (.001, .01, .1) for s in ('.pkl', '_scores.npz', '_result.json')]
    for name in names:
        assert original.sha(OLD / name) == complete['files_sha256'][name]
    matrix_manifest = original.read(OLD / 'matrix_manifest.json')
    assert original.sha(OLD / 'matrices/base.npy') == matrix_manifest['files_sha256']['base']
    return {'complete_sha256': original.sha(OLD / 'complete.json'),
            'base_matrix_sha256': original.sha(OLD / 'matrices/base.npy'),
            'files_sha256': {name: original.sha(OLD / name) for name in names}}


def feature_snapshot():
    path = DATA / 'feature_manifest.json'
    assert path.exists(), 'No complete new features: do not fit or score a partial cohort'
    fm = original.read(path); sig = original.read(DATA / 'signature.json')
    assert fm['complete'] and fm['completed_count'] == 793 and len(fm['records']) == 793
    assert fm['signature_sha256'] == original.digest(sig)
    assert fm['variants'] == sig['variants'] == list(METHODS)
    assert fm['all_old_anchors_exact'] and not fm['test_read'] and not fm['labels_used']
    for path, h in sig['code_sha256'].items():
        assert original.sha(path) == h
    for name, key in [('lookback_controls_protocol.json', 'protocol_sha256'), ('data/feature_preparation/plans.jsonl', 'plans_sha256'),
                      ('data/feature_manifest.json', 'old_feature_manifest_sha256'), ('data/feature_signature.json', 'old_feature_signature_sha256'),
                      ('data/gold_manifest.json', 'gold_manifest_sha256')]:
        assert original.sha(ROOT / name) == sig[key]
    assert original.sha(DATA / 'layouts.jsonl') == sig['layouts_sha256']
    assert [r['response_id'] for r in fm['records']] == sig['response_ids_in_order']
    assert len(set(sig['response_ids_in_order'])) == 793
    return fm, sig


def source_snapshot(fm, sig):
    paths = [Path(__file__), PROTOCOL, ROOT / 'lookback_controls_protocol.json', DATA / 'signature.json',
             DATA / 'feature_manifest.json', DATA / 'layouts.jsonl', ROOT / 'data/gold_manifest.json',
             ROOT / 'development_protocol.json', Path(original.__file__)]
    return {'files_sha256': {str(p.resolve()): original.sha(p) for p in paths},
            'feature_signature_sha256': original.digest(sig), 'original': legacy_snapshot(), 'test_read': False}


def geometry_and_weights():
    meta = original.metadata()
    assert len(meta['windows']) == N and len(meta['answers']) == 793
    b, loss, factors, y = original.base_weights(meta)
    with np.load(OLD / 'training_weights.npz', allow_pickle=False) as z:
        assert all(np.array_equal(a, z[k]) for a, k in [(b, 'base_weights'), (loss, 'loss_weights'), (factors, 'class_factors'), (y, 'y')])
    keys = {'window_ids': [w['window_id'] for w in meta['windows'][:FIT_N]],
            'group_ids': [w['group_id'] for w in meta['windows'][:FIT_N]],
            'answer_ids': [w['answer_id'] for w in meta['windows'][:FIT_N]]}
    assert keys == original.read(OLD / 'fit_keys.json')
    return meta, (b, loss, factors, y), keys


def new_feature_data(meta, fm, sig):
    plans = {p['response_id']: p for p in original.lines(ROOT / 'data/feature_preparation/plans.jsonl')}
    layouts = {p['response_id']: p for p in original.lines(DATA / 'layouts.jsonl')}
    records = {r['response_id']: r for r in fm['records']}
    assert set(plans) == set(layouts) == set(records) == set(meta['by_response'])
    for answer in meta['answers']:
        rid = answer['response_id']; rec = records[rid]
        expected = (DATA / 'features' / f'{rid}.npz').resolve()
        assert Path(rec['npz']).resolve() == expected and Path(rec['json']).resolve() == expected.with_suffix('.json')
        side = original.read(expected.with_suffix('.json'))
        assert original.sha(expected) == rec['npz_sha256'] == side['npz_sha256']
        assert original.sha(expected.with_suffix('.json')) == rec['json_sha256']
        assert rec['plan_sha256'] == side['plan_sha256'] == original.digest(plans[rid])
        assert rec['layout_sha256'] == side['layout_sha256'] == original.digest(layouts[rid])
        assert side['complete'] and side['signature_sha256'] == original.digest(sig) and side['legacy_anchor_exact']
        for k in ('response_id', 'partition', 'source_id', 'group_id'):
            assert side[k] == rec[k] == answer[k]
        token = meta['by_response'][rid]['tokens']
        with np.load(expected, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        assert set(arrays) == set(METHODS) | {'token_ids', 'answer_token_positions', 'response_token_offsets', 'response_token_offsets_raw'}
        for k in ('token_ids', 'answer_token_positions', 'response_token_offsets', 'response_token_offsets_raw'):
            assert np.array_equal(arrays[k], token[k])
        n = token['token_count']
        for key in METHODS:
            a = arrays[key]; assert a.shape == (n, 1024) and a.dtype == np.float32
            assert np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
        yield rid, arrays


def build_matrices(meta, fm, sig):
    folder = OUT / 'matrices'; folder.mkdir(parents=True, exist_ok=True)
    arrays = {m: np.lib.format.open_memmap(folder / (m+'.npy'), mode='w+', dtype=np.float32, shape=(N, 1024)) for m in METHODS}
    old = np.load(OLD / 'matrices/base.npy', mmap_mode='r')
    for i, (rid, raw) in enumerate(new_feature_data(meta, fm, sig)):
        for j in meta['answer_windows'][rid]:
            ix = meta['windows'][j]['token_indices']
            for method in METHODS:
                arrays[method][j] = raw[method][ix].mean(0)
            assert np.array_equal(arrays[ANCHOR][j], old[j, :1024]), ('Window anchor drift', rid, j)
        if (i+1) % 100 == 0:
            print('CONTROL_MATRIX', i+1, 793, flush=True)
    for a in arrays.values():
        a.flush()
    original.save(OUT / 'matrix_manifest.json', {'rows': N, 'fit_rows': FIT_N, 'calibration_rows': CAL_N,
        'width': 1024, 'files_sha256': {m: original.sha(folder/(m+'.npy')) for m in METHODS},
        'all_anchor_windows_exact': True, 'source_snapshot_sha256': original.sha(OUT/'source_snapshot.json')})
    return arrays


def score(meta, x, obj):
    scores = np.empty(N, np.float64)
    for left in range(0, N, BATCH):
        right = min(left+BATCH, N)
        zz = obj['scaler'].transform(np.asarray(x[left:right])).astype(np.float32)
        scores[left:right] = obj['model'].predict_proba(zz)[:, 1]
    assert np.isfinite(scores).all()
    return scores, original.answer_scores(meta, scores)


def thresholds(meta, scores, answers):
    lo, hi = meta['bounds']['calibration']
    ci = [i for i, a in enumerate(meta['answers']) if a['partition'] == 'calibration']
    return {'window': original.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], scores[lo:hi]),
            'answer': original.choose_threshold([meta['answers'][i]['label'] for i in ci], answers[ci])}


def replay_anchor(meta, matrix, c):
    name = f'lookback_mean_C{c:g}'
    obj = pickle.loads((OLD/(name+'.pkl')).read_bytes())
    assert obj['method'] == 'lookback_mean' and obj['C'] == c and obj['width'] == 1024
    assert obj['fit_only'] and obj['fit_rows'] == FIT_N
    oldz = np.load(OLD/'matrices/lookback_mean_fit_standardized.npy', mmap_mode='r')
    for left in range(0, FIT_N, BATCH):
        right = min(left+BATCH, FIT_N)
        assert np.array_equal(obj['scaler'].transform(np.asarray(matrix[left:right])).astype(np.float32), oldz[left:right])
    scores, answers = score(meta, matrix, obj)
    with np.load(OLD/(name+'_scores.npz'), allow_pickle=False) as z:
        assert np.array_equal(scores, z['window_scores'])
        assert np.array_equal(answers, z['answer_scores'])
    ts = thresholds(meta, scores, answers); entry = original.read(OLD/(name+'_result.json'))
    assert ts == obj['thresholds'] == entry['thresholds']
    assert original.metrics(meta, scores, ts) == entry['metrics']
    assert list(original.selection_key(ts, c)) == entry['selection_key']
    return obj, scores, answers, entry


def baseline_check():
    legacy_snapshot(); meta, _, _ = geometry_and_weights()
    matrix = np.load(OLD/'matrices/base.npy', mmap_mode='r')[:, :1024]
    checked = []
    for c in (.001, .01, .1):
        _, scores, answers, _ = replay_anchor(meta, matrix, c)
        checked.append({'C': c, 'window_scores': len(scores), 'answer_scores': len(answers),
            'standardized_fit_exact': True, 'frozen_scores_exact': True, 'thresholds_metrics_exact': True})
    report = {'passed': True, 'code_sha256': original.sha(__file__), 'protocol_sha256': original.sha(PROTOCOL),
              'checks': checked, 'new_features_used': False, 'new_fit': False, 'GPU_used': False, 'test_read': False,
              'limit': 'Old-model replay wiring passes now. Actual new feature anchor matrices must also pass at formal fit time.'}
    freeze(OUT/'baseline_cpu_check.json', report)
    print(json.dumps(report), flush=True)


def synthetic_tests():
    # Punctuation consumes a raw slot; no token mask compresses geometry.
    x = np.arange(6*1024, dtype=np.float32).reshape(6, 1024)
    ix = [1, 2, 3, 4]
    assert np.array_equal(x[ix].mean(0), (x[1]+x[2]+x[3]+x[4])/4)
    assert np.array_equal(x[:2].mean(0), (x[0]+x[1])/2)
    y = [0, 1, 1, 0, 1]; s = np.array([.2, .4, .4, .7, .9])
    got = original.choose_threshold(y, s)
    candidates = [np.nextafter(s.max(), np.inf), *np.unique(s)]
    want = max((original.count(y, s, t)['f1'], original.count(y, s, t)['precision'], t) for t in candidates)
    assert (got['f1'], got['precision'], got['threshold']) == want
    meta = {'answers': [{'response_id': 'a'}, {'response_id': 'b'}], 'answer_windows': {'a': [0, 1], 'b': [2, 3, 4]}}
    assert np.array_equal(original.answer_scores(meta, s), [.4, .9])
    value = {'passed': True, 'code_sha256': original.sha(__file__), 'protocol_sha256': original.sha(PROTOCOL),
        'checks': ['raw four-token mean including punctuation slots', 'short actual-token mean',
                   'calibration threshold brute-force tie rule', 'answer max does not cross answers'],
        'fit_called': False, 'GPU_used': False, 'test_read': False}
    freeze(OUT/'synthetic_selfcheck.json', value)
    print(json.dumps(value), flush=True)


def fit():
    import shutil
    # Gating happens before started.json and before any new fit.
    assert original.read(PROTOCOL) == cfg()
    fm, sig = feature_snapshot()
    meta, (b, loss, factors, y), keys = geometry_and_weights()
    assert [a['response_id'] for a in meta['answers']] == sig['response_ids_in_order']
    snap = source_snapshot(fm, sig)
    assert not (OUT/'started.json').exists(), 'Do not silently restart a formal fit'
    assert shutil.disk_usage(OUT).free > 9*2**30
    original.save(OUT/'source_snapshot.json', snap)
    original.save(OUT/'started.json', {'source_snapshot_sha256': original.sha(OUT/'source_snapshot.json'), 'test_read': False})
    started = time.perf_counter(); matrices = build_matrices(meta, fm, sig)
    np.savez_compressed(OUT/'training_weights.npz', base_weights=b, loss_weights=loss, class_factors=factors, y=y)
    original.save(OUT/'fit_keys.json', keys)
    entries = {}; selected = {}; artifacts = []
    # Anchor first: block all new fitting if actual newly extracted features
    # cannot reproduce the previous reference exactly.
    for method in (ANCHOR,) + METHODS[:-1]:
        family = []
        if method != ANCHOR:
            sc = StandardScaler()
            for left in range(0, FIT_N, BATCH):
                right = min(left+BATCH, FIT_N)
                sc.partial_fit(np.asarray(matrices[method][left:right]), sample_weight=b[left:right])
            zpath = OUT/'matrices'/(method+'_fit_standardized.npy')
            zfit = np.lib.format.open_memmap(zpath, mode='w+', dtype=np.float32, shape=(FIT_N, 1024))
            for left in range(0, FIT_N, BATCH):
                right = min(left+BATCH, FIT_N)
                zfit[left:right] = sc.transform(np.asarray(matrices[method][left:right])).astype(np.float32)
            zfit.flush()
        for c in (.001, .01, .1):
            begin = time.perf_counter(); name = f'{method}_C{c:g}'
            if method == ANCHOR:
                obj, scores, answer, prior = replay_anchor(meta, matrices[method], c)
                ts = prior['thresholds']; model = obj['model']
                source_model = OLD/f'lookback_mean_C{c:g}.pkl'
                mode = 'frozen_legacy_replay'
            else:
                model = LogisticRegression(C=c, solver='liblinear', penalty='l2', max_iter=2000, random_state=20260924)
                model.fit(zfit, y, sample_weight=loss)
                assert model.n_iter_.max() < 2000
                obj = {'model': model, 'scaler': sc}; scores, answer = score(meta, matrices[method], obj)
                ts = thresholds(meta, scores, answer); source_model = None; mode = 'new_fit'
            key = original.selection_key(ts, c)
            wrapped = {**obj, 'method': method, 'width': 1024, 'C': c, 'thresholds': ts, 'selection_key': key,
                'fit_only': True, 'fit_rows': FIT_N, 'fit_groups': 615, 'fit_answers': 634,
                'weights_sha256': original.sha(OUT/'training_weights.npz'), 'fit_keys_sha256': original.sha(OUT/'fit_keys.json'),
                'feature_signature_sha256': original.digest(sig), 'mode': mode,
                'source_original_model_sha256': original.sha(source_model) if source_model else None}
            modelpath = OUT/(name+'.pkl'); modelpath.write_bytes(pickle.dumps(wrapped, protocol=5))
            scorepath = OUT/(name+'_scores.npz'); np.savez_compressed(scorepath, window_scores=scores, answer_scores=answer)
            entry = {'candidate': name, 'method': method, 'C': c, 'mode': mode, 'thresholds': ts,
                'selection_key': list(key), 'metrics': original.metrics(meta, scores, ts),
                'iterations': model.n_iter_.tolist(), 'seconds': time.perf_counter()-begin,
                'model_sha256': original.sha(modelpath), 'scores_sha256': original.sha(scorepath),
                'legacy_exact': method == ANCHOR}
            original.save(OUT/(name+'_result.json'), entry)
            family.append(entry); artifacts += [name+'.pkl', name+'_scores.npz', name+'_result.json']
            print('CONTROL_CANDIDATE', name, mode, round(entry['seconds'], 2), flush=True)
        entries[method] = family; selected[method] = max(family, key=lambda e: e['selection_key'])
        if method != ANCHOR:
            del zfit; gc.collect()
    fm2, sig2 = feature_snapshot(); assert snap == source_snapshot(fm2, sig2)
    original_summary = original.read(OLD/'summary.json')
    anchor = selected[ANCHOR]
    assert anchor['C'] == original_summary['selected']['lookback_mean']['C']
    assert anchor['metrics'] == original_summary['selected']['lookback_mean']['metrics']
    index = {'windows': [{k:w[k] for k in ('window_id','answer_id','group_id','partition','label')} for w in meta['windows']],
             'answers': [{k:a[k] for k in ('answer_id','group_id','partition','label')} for a in meta['answers']]}
    assert index == original.read(OLD/'score_index.json')
    original.save(OUT/'score_index.json', index)
    summary = {'scope': cfg()['scope'], 'coverage': meta['gold']['counts'], 'candidates': 15, 'new_fits': 12,
        'legacy_replays': 3, 'selected': selected, 'all_candidates': entries, 'all_legacy_checks_exact': True,
        'calibration_results_are_selection_optimistic': True, 'official_test_opened': False,
        'primary_closer_official': 'lb_prefix_pre_header', 'seconds': time.perf_counter()-started,
        'not_exact_original_experiment': 'Different published chat wrapper, NF4 reconstructed states, original released response text; see LOOKBACK_CONTROLS_PLAN.md'}
    original.save(OUT/'summary.json', summary)
    report = ['五组 Lookback 定义对照均使用同一人工 QA 开发集。以下是用于选 C 和阈值的校准成绩，不能当最终测试成绩。','',
              '| 定义 | C | cal 窗口 F1 | cal 整答 F1 |','|---|---:|---:|---:|']
    for m in METHODS:
        e = selected[m]; report.append(f"| {m} | {e['C']} | {e['metrics']['calibration']['windows']['f1']:.3f} | {e['metrics']['calibration']['answers']['f1']:.3f} |")
    report += ['', '四格各3次新拟合，共12次；旧锚点3个模型直接重放，全部窗口矩阵、分数、阈值与原 lookback_mean 精确一致。',
               'prefix_pre_header 的范围和时刻较接近官方代码，但 chat 头、量化和原始 trace 条件仍不同；不称完全复现原版实验。',
               'fit634/cal159，168123/42241 可评窗口；未使用 NLL/PCA/hidden，未打开 test，未重拟合 fit+cal。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', 'utf-8')
    artifacts += ['source_snapshot.json','training_weights.npz','fit_keys.json','matrix_manifest.json','score_index.json','summary.json','REPORT.md']
    original.save(OUT/'complete.json', {'status':'complete_development_only','new_fits':12,'legacy_replays':3,
        'files_sha256': {name:original.sha(OUT/name) for name in artifacts}, 'code_sha256':original.sha(__file__),
        'protocol_sha256':original.sha(PROTOCOL), 'official_test_opened':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize','self-test','baseline-check','fit'])
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage == 'initialize':
            freeze(PROTOCOL, cfg()); print('LOOKBACK_SCORING_PROTOCOL_READY_NO_FIT')
        elif args.stage == 'self-test':
            assert original.read(PROTOCOL) == cfg(); synthetic_tests()
        elif args.stage == 'baseline-check':
            assert original.read(PROTOCOL) == cfg(); baseline_check()
        else:
            fit()
