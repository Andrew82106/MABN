"""Five matched local linear GHOST adaptations; no GPU or automatic waiting.

prepare-design is CPU preparation. prepare-features requires the complete3839
feature manifest. fit is explicit, gated, and never trains a partial cohort.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fit_expansion'))
import run_llama_baselines as old

q, expansion = old.q, old.expansion
OUT = ROOT / 'results/ghost_matched_lr_v1'
GHOST = ROOT / 'results/ghost_geometry_features_v1'
INPUT = ROOT / 'results/ghost_geometry_preparation_v1/feature_inputs.jsonl'
NFIT, NCAL, NTOTAL, BATCH = old.NFIT, old.NCAL, old.NTOTAL, old.BATCH
METHODS = (*old.METHODS, 'ghost_only')
WIDTHS = dict(zip(METHODS, [1028, 1029, 1028, 1093, 4]))
CS = (1e-5, 1e-4, .001, .01, .1)
NAMES = ['adjacent_layer_cosine_change', 'layer_to_final_cosine',
         'top10_normalized_entropy', 'top10_unweighted_embedding_divergence']
MATRIX_NAMES = ['base.npy', old.PRE + '.npy', old.POST + '.npy', 'harp.npy']


def freeze(path, value):
    if path.exists():
        assert q.read(path) == value, ('Frozen definition changed', str(path))
    else:
        q.save(path, value)


def consumed_paths():
    # Bind consumed raw window matrices, not old standardized designs or huge
    # unused per-token attention/hidden/replay caches.
    files = [Path(__file__), Path(old.__file__), Path(q.__file__), Path(expansion.__file__),
             INPUT, GHOST / 'signature.json', GHOST / 'protocol.json', GHOST / 'CPU_CHECK.json',
             old.OUT / 'protocol.json', old.OUT / 'preparation_complete.json', old.OUT / 'complete.json',
             old.OUT / 'summary.json', old.OUT / 'training_weights.npz', old.OUT / 'token_index.json',
             expansion.OUT / 'expanded3680_weights.npz', ROOT / 'data/gold_manifest.json',
             ROOT / 'fit_expansion/data/export_freeze.json']
    for part in ('fit', 'calibration'):
        files += [ROOT / f'data/{kind}_{part}.jsonl' for kind in ('answers', 'tokens', 'windows_k4')]
    files += [ROOT / f'fit_expansion/data/{kind}_fit.jsonl' for kind in ('answers', 'tokens', 'windows_k4')]
    files += [old.OUT / 'matrices' / name for name in MATRIX_NAMES]
    for method in old.METHODS:
        for c in CS:
            name = f'{method}_C{c:g}'
            files += [old.OUT / (name + suffix) for suffix in ('.pkl', '_scores.npz', '_result.json')]
    return files


def snapshot():
    return {'files_sha256': {str(p.resolve()): q.sha(p) for p in consumed_paths()},
            'software': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')},
            'official_test_opened': False}


def protocol():
    return {'version': 'ghost-matched-local-lr-v1', 'methods': list(METHODS), 'widths': WIDTHS,
            'C': list(CS), 'new_fits': 25, 'reused_old_fits': 20, 'seed': 20260924,
            'cohort': {'fit_answers': 3680, 'calibration_answers': 159, 'fit_groups': 615,
                       'calibration_groups': 154, 'fit_windows': NFIT, 'calibration_windows': NCAL,
                       'all_windows': NTOTAL, 'raw_answer_tokens': 708506},
            'append': 'Each original raw family followed by the same four mean GHOST columns; standalone four columns also reported.',
            'feature_names': NAMES, 'pool': 'Mean all actual raw BPE token_indices in the frozen4BPE window, INCLUDING punctuation; short window uses actual slots only.',
            'base_features': {'prefix_pre_header': 'original pre-header prefix LB1024',
                              'legacy_lb_nll': 'original source-only post-read legacy LB1024 then NLL1',
                              'prefix_post_header': 'original post-read prefix LB1024',
                              'harp64_legacy_lb_nll': 'original legacy LB1024 then NLL1 then frozen HARP64'},
            'weights': 'Exact expanded3680 base/loss arrays, group/native-aux/answer/window weighting and original class factors; total loss168123; no new class_weight.',
            'scaler': 'Separate per family; fit-only base-weighted partial_fit in original16384-row blocks; all transformed inputs float32; same scaler across five Cs.',
            'LR': {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 2000, 'random_state': 20260924,
                   'cpu_threads': 4, 'convergence_warning': 'fail and preserve, do not increase budget'},
            'threshold': 'Original q.choose_threshold: calibration only, each scale separately; >= threshold, ties F1 then precision then higher threshold.',
            'C_selection': 'Original q.selection_key: min(windowF1,answerF1), windowF1, windowprecision, smallerC. One candidate serves both scales.',
            'answer_score': 'Maximum of ALL original eligible windows per answer, unchanged answer gold/refusal policy.',
            'gates': 'No feature matrix or fitting from partial3839 input; features_complete must bind every record, identities, raw coordinates and ordered manifest. fit requires completed preparation.',
            'old_controls': 'Read-only original20 models/scores/results; verify hashes, saved scores answermax, thresholds, metrics and selections. Never refit.',
            'storage': 'Keep25 models/scores/results/scalers; raw four-column matrix. One temporary standardized float32 family memmap, hash recorded, deleted after that family. Original raw matrices stay in place.',
            'adaptation': 'Local4rawBPE mean and linear supervised probe; NOT original GHOST RF answer-level reproduction or its paper scores.',
            'provenance': 'Existing3680 replay includes3046 other-generator answers; common Llama reconstruction, not their historical native trajectories.',
            'limitation': 'Repeated public-QA fit/cal development, no independent final-test claim; labels/denominators and existing experiments unchanged.',
            'official_test_opened': False, 'GPU_used': False, 'new_generation': False}


def metadata_and_geometry():
    _, meta = expansion.metadata()
    assert (len(meta['answers']), len(meta['windows'])) == (3839, NTOTAL)
    assert meta['bounds'] == {'fit': [0, NFIT], 'calibration': [NFIT, NTOTAL]}
    inputs = q.lines(INPUT)
    assert [r['response_id'] for r in inputs] == [a['response_id'] for a in meta['answers']]
    token_index = q.read(old.OUT / 'token_index.json')['answers']
    cursor = punctuation_slots = 0
    for i, (r, a, t, index) in enumerate(zip(inputs, meta['answers'], meta['tokens'], token_index)):
        assert r['partition'] == a['partition'] == ('fit' if i < 3680 else 'calibration')
        assert r['group_id'] == a['group_id'] == index['group_id']
        assert r['source_id'] == a['source_id'] and r['answer_sha256'] == t['answer_sha256']
        assert r['response_id'] == t['response_id'] == index['response_id']
        assert r['answer_token_ids'] == t['token_ids']
        for k in ('answer_token_positions', 'response_token_offsets', 'response_token_offsets_raw'):
            assert r[k] == t[k]
        assert len(t['token_ids']) == t['token_count'] == index['right'] - index['left']
        assert index['left'] == cursor
        cursor = index['right']
    for w in meta['windows']:
        t = meta['by_response'][w['response_id']]['tokens']
        ix = w['token_indices']
        assert w['eligible'] and ix == list(range(w['token_start'], w['token_end']))
        assert len(ix) == min(4, t['token_count']) and 0 <= ix[0] <= ix[-1] < t['token_count']
        assert any(t['lexical_mask'][j] for j in ix)
        assert w['label'] == int(any(t['risk_mask'][j] for j in ix))
        punctuation_slots += sum(not t['lexical_mask'][j] for j in ix)
    assert cursor == 708506
    with np.load(old.OUT / 'training_weights.npz', allow_pickle=False) as z:
        weights = {k: z[k].copy() for k in z.files}
    assert q.sha(old.OUT / 'training_weights.npz') == q.sha(expansion.OUT / 'expanded3680_weights.npz')
    b, loss, y, factors = expansion.expanded_weights(meta)
    for name, wanted in [('base', b), ('loss', loss), ('y', y), ('class_factors', factors)]:
        assert np.array_equal(weights[name], wanted), name
    assert abs(weights['loss'].sum() - 168123) < 1e-6
    return meta, inputs, weights, {'raw_tokens': cursor, 'punctuation_slot_occurrences_retained': punctuation_slots,
                                 'fit_positive_windows': int(y.sum()),
                                 'cal_positive_windows': sum(w['label'] for w in meta['windows'][NFIT:]),
                                 'fit_positive_answers': sum(a['label'] for a in meta['answers'][:3680]),
                                 'cal_positive_answers': sum(a['label'] for a in meta['answers'][3680:])}


def thresholds(meta, scores):
    answer = q.answer_scores(meta, scores)
    return {'window': q.choose_threshold([w['label'] for w in meta['windows'][NFIT:]], scores[NFIT:]),
            'answer': q.choose_threshold([a['label'] for a in meta['answers'][3680:]], answer[3680:])}


def old_controls(meta):
    summary = q.read(old.OUT / 'summary.json')
    complete = q.read(old.OUT / 'complete.json')
    assert complete['fit_count'] == summary['fit_count'] == 20 and not complete['official_test_opened']
    assert q.sha(old.OUT / 'summary.json') == complete['files_sha256']['summary.json']
    controls = {}
    for method in old.METHODS:
        entries = []
        for c in CS:
            name = f'{method}_C{c:g}'
            entry = q.read(old.OUT / (name + '_result.json'))
            for suffix in ('.pkl', '_scores.npz', '_result.json'):
                assert q.sha(old.OUT / (name + suffix)) == complete['files_sha256'][name + suffix]
            assert entry in summary['all_candidates'][method]
            assert q.sha(old.OUT / (name + '.pkl')) == entry['model_sha256']
            assert q.sha(old.OUT / (name + '_scores.npz')) == entry['scores_sha256']
            obj = pickle.loads((old.OUT / (name + '.pkl')).read_bytes())
            assert obj['C'] == c and obj['method'] == method and obj['fit_only'] and obj['fit_rows'] == NFIT
            assert obj['weights_sha256'] == q.sha(old.OUT / 'training_weights.npz')
            with np.load(old.OUT / (name + '_scores.npz'), allow_pickle=False) as z:
                scores, answer = z['window_scores'], z['answer_scores']
                assert scores.shape == (NTOTAL,) and answer.shape == (3839,)
                assert np.array_equal(answer, q.answer_scores(meta, scores))
                ts = thresholds(meta, scores)
                assert ts == entry['thresholds'] == obj['thresholds']
                assert q.metrics(meta, scores, ts) == entry['metrics']
                assert list(q.selection_key(ts, c)) == entry['selection_key']
            entries.append(entry)
        best = max(entries, key=lambda e: e['selection_key'])
        assert best == summary['selected'][method]
        controls[method] = {'selected': best, 'all_candidates': entries}
        print('OLD_CONTROL_VERIFIED', method, flush=True)
    return {'status': 'passed', '20_model_score_hashes_verified': True, 'old_fit_count': 20,
            'old_refits': 0, 'answermax_threshold_metrics_selection_exact': True,
            'controls': controls, 'old20_actual_seconds': summary['seconds']}


def load_old_matrices():
    return {name: np.load(old.OUT / 'matrices' / name, mmap_mode='r') for name in MATRIX_NAMES}


def raw_features(method, matrices, ghost, left, right):
    extra = np.asarray(ghost[left:right])
    if method == 'ghost_only':
        return extra
    if method == 'prefix_pre_header':
        base = matrices[old.PRE + '.npy'][left:right]
    elif method == 'prefix_post_header':
        base = matrices[old.POST + '.npy'][left:right]
    elif method == 'legacy_lb_nll':
        base = matrices['base.npy'][left:right]
    else:
        assert method == 'harp64_legacy_lb_nll'
        base = np.column_stack((matrices['base.npy'][left:right], matrices['harp.npy'][left:right]))
    return np.column_stack((base, extra)).astype(np.float32, copy=False)


def pool(features, token_indices):
    return features[token_indices].mean(axis=0)


def synthetic_check():
    v = np.asarray([[0, 0, 0, 0], [40, 80, 120, 160], [0, 0, 0, 0], [0, 0, 0, 0]], np.float32)
    # Slot1 represents punctuation: its values must affect the mean.
    assert np.array_equal(pool(v, [0, 1, 2, 3]), np.asarray([10, 20, 30, 40], np.float32))
    assert np.array_equal(pool(v, [0, 1]), np.asarray([20, 40, 60, 80], np.float32))
    arrays = {name: np.full((8, width), j+1, np.float32) for j, (name, width) in enumerate(zip(MATRIX_NAMES, (1025, 1024, 1024, 64)))}
    extra = np.arange(32, dtype=np.float32).reshape(8, 4)
    for method in METHODS:
        raw = raw_features(method, arrays, extra, 0, 8)
        assert raw.shape == (8, WIDTHS[method]) and np.array_equal(raw[:, -4:], extra)
        if method != 'ghost_only':
            expected = (arrays[old.PRE+'.npy'] if method == 'prefix_pre_header' else
                        arrays[old.POST+'.npy'] if method == 'prefix_post_header' else
                        arrays['base.npy'] if method == 'legacy_lb_nll' else
                        np.column_stack((arrays['base.npy'], arrays['harp.npy'])))
            assert np.array_equal(raw[:, :-4], expected)
    x = np.asarray([[0, 1], [1, 0], [2, 1], [3, 0], [1000, 1000]], np.float32)
    weights = np.asarray([1, 2, 3, 4], float)
    sc = StandardScaler().fit(x[:4], sample_weight=weights)
    assert np.allclose(sc.mean_, np.average(x[:4], axis=0, weights=weights), atol=1e-15, rtol=0)
    transformed = sc.transform(x).astype(np.float32)
    # A synthetic-only LR exercises the same weighted float32/liblinear API.
    toy = LogisticRegression(C=.001, solver='liblinear', penalty='l2', random_state=20260924, max_iter=2000)
    toy.fit(transformed[:4], [0, 0, 1, 1], sample_weight=weights)
    assert np.isfinite(toy.predict_proba(transformed)).all()
    return {'status': 'passed', 'raw_prefix_unchanged_all5': True, 'punctuation_and_short_window_mean': True,
            'fit_only_weighted_scaler': True, 'synthetic_LR_only': 1, 'real_new_fits': 0, 'GPU_used': False}


def prepare_design():
    OUT.mkdir(parents=True, exist_ok=True)
    assert CS == old.CS
    meta, inputs, weights, geometry = metadata_and_geometry()
    prep = q.read(old.OUT / 'preparation_complete.json')
    expected = {str((old.OUT / name).resolve()): digest for name, digest in prep['files_sha256'].items()}
    matrices = load_old_matrices()
    for name, width in zip(MATRIX_NAMES, (1025, 1024, 1024, 64)):
        path = old.OUT / 'matrices' / name
        assert matrices[name].shape == (NTOTAL, width) and matrices[name].dtype == np.float32
        assert q.sha(path) == expected[str(path.resolve())]
    controls = old_controls(meta)
    tests = synthetic_check()
    frozen = snapshot()
    cfg = protocol()
    freeze(OUT / 'protocol.json', cfg)
    freeze(OUT / 'source_snapshot.json', frozen)
    freeze(OUT / 'OLD_CONTROL_CHECK.json', controls)
    freeze(OUT / 'CPU_SELFCHECK.json', tests)
    freeze(OUT / 'geometry.json', {'answer_ids': [r['response_id'] for r in inputs],
                                 'window_order_sha256': q.digest(meta['windows']), **geometry})
    max_width = max(WIDTHS.values())
    resource = {'reused_old_raw_matrices_GiB': sum(m.nbytes for m in matrices.values())/2**30,
                'new_ghost_window_MiB': NTOTAL*4*4/2**20,
                'one_largest_standardized_scratch_GiB': NFIT*max_width*4/2**30,
                'all25_scores_MiB_uncompressed': 25*(NTOTAL+3839)*8/2**20,
                'new_disk_reserve_GiB': 5,
                'old20_observed_CPU_seconds': controls['old20_actual_seconds'],
                'new25_estimated_CPU_minutes': [60, 110],
                'estimate_not_measurement': True, 'threads': 4,
                'note': 'Four high-dimensional families are almost the old20 fits; five4D fits add little. Timing varies with convergence and CPU contention.'}
    freeze(OUT / 'RESOURCE_PLAN.json', resource)
    done = {'status': 'CPU_design_prepared_no_real_fit', 'source_snapshot_sha256': q.sha(OUT/'source_snapshot.json'),
            'protocol_sha256': q.sha(OUT/'protocol.json'), 'all_old_controls_checked': True,
            'full_old_geometry_checked': True, 'all_old_weights_exact': True,
            'new_fits': 0, 'GPU_used': False, 'test_opened': False,
            'files_sha256': {n: q.sha(OUT/n) for n in ('OLD_CONTROL_CHECK.json', 'CPU_SELFCHECK.json', 'geometry.json', 'RESOURCE_PLAN.json')}}
    freeze(OUT / 'design_complete.json', done)
    print(json.dumps(done), flush=True)


def verify_design():
    done = q.read(OUT / 'design_complete.json')
    assert done['status'] == 'CPU_design_prepared_no_real_fit'
    assert q.read(OUT / 'protocol.json') == protocol()
    assert q.sha(OUT/'protocol.json') == done['protocol_sha256']
    assert q.sha(OUT/'source_snapshot.json') == done['source_snapshot_sha256']
    for name, digest in done['files_sha256'].items():
        assert q.sha(OUT/name) == digest
    assert snapshot() == q.read(OUT / 'source_snapshot.json')


def require_feature_complete():
    path = GHOST / 'features_complete.json'
    if not path.exists():
        raise RuntimeError('Complete3839 GHOST features missing; no partial preparation or fit')
    complete = q.read(path)
    assert complete['status'] == 'complete' and complete['records'] == 3839 and complete['raw_answer_tokens'] == 708506
    assert complete['no_test'] and not complete['trained'] and complete['all_records_validated']
    for name, digest in complete['files_sha256'].items():
        target = (GHOST/name).resolve()
        assert target.parent == GHOST.resolve() and q.sha(target) == digest
    sig = q.read(GHOST/'signature.json')
    assert complete['signature_sha256'] == q.digest(sig)
    fm = q.read(GHOST/'feature_manifest.json')
    assert fm['status'] == 'complete' and fm['records'] == len(fm['entries']) == 3839
    assert fm['raw_answer_tokens'] == 708506 and fm['all_records_validated']
    assert fm['feature_names'] == NAMES and fm['signature_sha256'] == complete['signature_sha256']
    assert not fm['labels_used'] and not fm['test_opened'] and not fm['trained']
    return fm, complete


def prepare_features():
    fm, complete = require_feature_complete()  # Gate BEFORE any output/matrix creation.
    verify_design()
    meta, inputs, weights, _ = metadata_and_geometry()
    assert fm['answer_order'] == [r['response_id'] for r in inputs]
    if (OUT/'feature_preparation_complete.json').exists():
        result = q.read(OUT/'feature_preparation_complete.json')
        assert result['upstream_complete_sha256'] == q.sha(GHOST/'features_complete.json')
        assert result['matrix_sha256'] == q.sha(OUT/'ghost_window_features.npy')
        return True
    assert not (OUT/'feature_preparation_started.json').exists(), 'Prior interrupted preparation preserved; diagnose before a separate retry revision'
    q.save(OUT/'feature_preparation_started.json', {'source_complete_sha256': q.sha(GHOST/'features_complete.json')})
    path = OUT/'ghost_window_features.npy'
    matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(NTOTAL, 4))
    seen = np.zeros(NTOTAL, bool)
    for i, (row, record) in enumerate(zip(inputs, fm['entries'])):
        assert record['record_index'] == str(i) and record['response_id'] == str(row['response_id'])
        assert record['file'] == f'{i:05d}.npz' and record['record_sha256'] == q.digest(row)
        assert record['signature_sha256'] == complete['signature_sha256']
        p = GHOST/'features'/record['file']
        assert q.sha(p) == record['npz_sha256'] and q.read(p.with_suffix('.json')) == record
        with np.load(p, allow_pickle=False) as z:
            assert str(z['record_sha256'].item()) == q.digest(row)
            assert str(z['response_id'].item()) == str(row['response_id']) and str(z['record_index'].item()) == str(i)
            assert str(z['signature_sha256'].item()) == complete['signature_sha256']
            for k, expected in [('token_ids', row['answer_token_ids']), ('answer_token_positions', row['answer_token_positions']),
                                ('predictor_positions', np.asarray(row['answer_token_positions'])-1)]:
                assert z[k].dtype == np.int64 and np.array_equal(z[k], expected)
            for k, expected in [('token_start', np.asarray(row['response_token_offsets'])[:,0]),
                                ('token_end', np.asarray(row['response_token_offsets'])[:,1]),
                                ('token_start_raw', np.asarray(row['response_token_offsets_raw'])[:,0]),
                                ('token_end_raw', np.asarray(row['response_token_offsets_raw'])[:,1])]:
                assert z[k].dtype == np.int64 and np.array_equal(z[k], expected)
            features = z['ghost_features']
            assert features.dtype == np.float32 and features.shape == (len(row['answer_token_ids']), 4) and np.isfinite(features).all()
            for j in meta['answer_windows'][row['response_id']]:
                assert not seen[j]
                matrix[j] = pool(features, meta['windows'][j]['token_indices'])
                seen[j] = True
        if (i+1) % 500 == 0:
            print('GHOST_WINDOW_POOL', i+1,3839, flush=True)
    assert seen.all() and np.isfinite(matrix).all()
    matrix.flush()
    result = {'status': 'complete', 'records': 3839, 'shape': [NTOTAL,4], 'dtype': 'float32',
              'matrix_sha256': q.sha(path), 'upstream_complete_sha256': q.sha(GHOST/'features_complete.json'),
              'source_manifest_sha256': q.sha(GHOST/'feature_manifest.json'),
              'design_complete_sha256': q.sha(OUT/'design_complete.json'),
              'window_order_sha256': q.digest(meta['windows']), 'all_coordinates_verified': True,
              'all_raw_punctuation_slots_pooled': True, 'new_fits': 0, 'test_opened': False}
    freeze(OUT/'feature_preparation_complete.json', result)
    print(json.dumps(result), flush=True)
    return True


def fit():
    require_feature_complete()  # Missing features stops before started/model construction.
    verify_design()
    prep = q.read(OUT/'feature_preparation_complete.json')
    assert prep['status'] == 'complete' and prep['shape'] == [NTOTAL, 4]
    assert prep['upstream_complete_sha256'] == q.sha(GHOST/'features_complete.json')
    assert prep['source_manifest_sha256'] == q.sha(GHOST/'feature_manifest.json')
    assert prep['design_complete_sha256'] == q.sha(OUT/'design_complete.json')
    assert prep['matrix_sha256'] == q.sha(OUT/'ghost_window_features.npy')
    assert not (OUT/'fit_started.json').exists(), 'Preserve any previous started run; no automatic refit'
    assert shutil.disk_usage(OUT).free > 5*2**30
    meta, _, weights, _ = metadata_and_geometry()
    assert prep['window_order_sha256'] == q.digest(meta['windows'])
    matrices = load_old_matrices()
    ghost = np.load(OUT/'ghost_window_features.npy', mmap_mode='r')
    assert ghost.shape == (NTOTAL,4) and ghost.dtype == np.float32 and np.isfinite(ghost).all()
    q.save(OUT/'fit_started.json', {'preparation_sha256': q.sha(OUT/'feature_preparation_complete.json'),
                                  'protocol_sha256': q.sha(OUT/'protocol.json'), 'new_fit_budget': 25})
    (OUT/'training_weights.npz').write_bytes((old.OUT/'training_weights.npz').read_bytes())
    all_start = time.perf_counter()
    candidates, selected = {}, {}
    files = ['training_weights.npz', 'fit_started.json']
    for method in METHODS:
        sc = StandardScaler()
        for l in range(0,NFIT,BATCH):
            r = min(l+BATCH,NFIT)
            sc.partial_fit(raw_features(method,matrices,ghost,l,r), sample_weight=weights['base'][l:r])
        scratch = OUT / 'current_fit_standardized.npy'
        zfit = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float32, shape=(NFIT, WIDTHS[method]))
        for l in range(0,NFIT,BATCH):
            r = min(l+BATCH,NFIT)
            zfit[l:r] = sc.transform(raw_features(method,matrices,ghost,l,r)).astype(np.float32)
        zfit.flush()
        matrix_hash = q.sha(scratch)
        entries = []
        for c in CS:
            started = time.perf_counter()
            classifier = LogisticRegression(C=c, solver='liblinear', penalty='l2', max_iter=2000, random_state=20260924)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                classifier.fit(zfit, weights['y'], sample_weight=weights['loss'])
            assert classifier.n_iter_.max() < 2000
            scores = np.empty(NTOTAL, np.float64)
            for l in range(0,NTOTAL,BATCH):
                r = min(l+BATCH,NTOTAL)
                x = sc.transform(raw_features(method,matrices,ghost,l,r)).astype(np.float32)
                scores[l:r] = classifier.predict_proba(x)[:,1]
            answer = q.answer_scores(meta,scores)
            ts = thresholds(meta,scores)
            name = f'{method}__ghost_C{c:g}'
            obj = {'model': classifier, 'scaler': sc, 'method': method, 'width': WIDTHS[method], 'C': c,
                   'thresholds': ts, 'selection_key': q.selection_key(ts,c), 'fit_only': True,
                   'fit_rows': NFIT, 'fit_answers': 3680, 'fit_groups': 615,
                   'weights_sha256': q.sha(OUT/'training_weights.npz'), 'fit_matrix_sha256': matrix_hash,
                   'feature_preparation_sha256': q.sha(OUT/'feature_preparation_complete.json'),
                   'protocol_sha256': q.sha(OUT/'protocol.json')}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj, protocol=5))
            np.savez_compressed(OUT/(name+'_scores.npz'), window_scores=scores, answer_scores=answer)
            entry = {'candidate': name, 'method': method, 'C': c, 'thresholds': ts,
                     'selection_key': list(q.selection_key(ts,c)), 'metrics': q.metrics(meta,scores,ts),
                     'model_sha256': q.sha(OUT/(name+'.pkl')), 'scores_sha256': q.sha(OUT/(name+'_scores.npz')),
                     'iterations': classifier.n_iter_.tolist(), 'seconds': time.perf_counter()-started}
            q.save(OUT/(name+'_result.json'),entry)
            entries.append(entry)
            files += [name+'.pkl',name+'_scores.npz',name+'_result.json']
            print('GHOST_MATCHED_FIT',name,round(entry['seconds'],1),flush=True)
        candidates[method] = entries
        selected[method] = max(entries,key=lambda e:e['selection_key'])
        del zfit
        gc.collect()
        scratch.unlink()  # Only this new runner's fixed temporary file; source matrices remain untouched.
    controls = q.read(OUT/'OLD_CONTROL_CHECK.json')['controls']
    deltas = {method: {level: selected[method]['metrics']['calibration'][level]['f1'] -
                              controls[method]['selected']['metrics']['calibration'][level]['f1']
                      for level in ('windows','answers')} for method in old.METHODS}
    summary = {'selected': selected, 'all_candidates': candidates, 'old_controls': controls,
               'selected_minus_matched_old_cal_F1': deltas, 'new_fits': 25, 'old_refits': 0,
               'fit_answers': 3680, 'calibration_answers': 159, 'fit_windows': NFIT, 'calibration_windows': NCAL,
               'seconds': time.perf_counter()-all_start, 'test_opened': False,
               'development_selection_optimistic': True, 'original_GHOST_RF_reproduction': False}
    q.save(OUT/'summary.json',summary)
    files += ['summary.json','protocol.json','source_snapshot.json','design_complete.json','feature_preparation_complete.json',
              'OLD_CONTROL_CHECK.json','ghost_window_features.npy']
    freeze(OUT/'complete.json', {'status': 'complete', 'new_fits': 25, 'old_refits': 0,
                               'files_sha256': {n:q.sha(OUT/n) for n in files}, 'test_opened': False, 'GPU_used': False})
    print('GHOST_MATCHED_COMPLETE25',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-design','prepare-features','fit'))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.command == 'prepare-design':
            prepare_design()
        elif args.command == 'prepare-features':
            prepare_features()
        else:
            fit()


if __name__ == '__main__':
    main()
