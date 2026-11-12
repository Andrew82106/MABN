"""Fixed CPU probes over frozen MiniCheck claim states plus Llama replay signals.

No model generation, checker fine-tuning, CUDA use, or official-test access.
Preparation is explicit; it does not wait for unfinished upstream extraction.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

import run_development as qa

ROOT = qa.ROOT; OLD = qa.OUT
SEM = ROOT / 'semantic_baseline/cuda_variant'
OUT = ROOT / 'results/semantic_hidden_v1'
METHODS = ('minicheck_hidden64', 'minicheck_hidden64_lookback_nll', 'minicheck_hidden64_lookback_nll_risk')
WIDTHS = (64, 1089, 1090)
BATCH = 16384


def protocol():
    return {'version': 'qa-minicheck-state-fusion-v1', 'scope': 'Only frozen fit634/calibration159; no official test',
        'methods': list(METHODS), 'widths': dict(zip(METHODS, WIDTHS)), 'C': [.001, .01, .1], 'seed': 20260924,
        'extra_model': 'Frozen MiniCheck-RoBERTa-large revision74c8919647e61ed0f71bc177d94f10930f090068, float32 CUDA variant',
        'native_generator_whitebox_only': False, 'checker_finetuned': False,
        'claim_trace': 'Last RoBERTa encoder hidden1024, selected document block = maximum official support for each automatically split claim; first tie; no gold',
        'hidden_alignment': 'Use original-answer half-open character offsets. For each nonwhitespace character average all MiniCheck token states covering it, then average those character states within each original Llama raw BPE. This prevents duplicated Unicode offsets from overweighting characters. Pure-whitespace raw BPE gets zero1024.',
        'coverage_gate': 'Every original-answer nonwhitespace character and every lexical raw BPE must be covered; any missing coverage stops preparation, never removes a row.',
        'risk_alignment': 'For each lexical raw BPE, maximum claim risk among claims intersecting at least one actual alphanumeric character; risk =1-selected support in float64. Nonlexical token risk is0.',
        'risk_logit': 'Clip risk to[1e-6,1-1e-6] then log(p/(1-p)). LR scalar uses max lexical token risk in original4raw window before logit; token scalar retained for separately frozen later TCN.',
        'PCA': {'components': 64, 'n_iter': 3, 'seed': 20260924, 'whiten': False,
            'sample': 'Exact existing PCA sample identities: each fit answer floor(linspace(0,N-1,min(N,32))) raw BPE positions.',
            'weight': 'Reuse frozen source-group/answer/sample-token balanced weights; sum1. Fit only on mapped fit states, never calibration.',
            'apply': '(raw hidden1024-fit mean)@components.T in float64, castfloat32; window mean over four actual original raw BPE tokens, including punctuation.'},
        'training': 'All168123 fit windows, same frozen source/group/answer/window base weights and class-balanced group-renormalized loss mass168123 as development_v1.',
        'standardization': 'Fit-only base-weighted StandardScaler partial_fit in16384-row blocks; float32 transform, same scaler across three Cs.',
        'LR': {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 2000, 'cpu_threads': 4},
        'threshold_selection': 'Separate calibration risk F1 thresholds for window/answer, then precision/higher threshold; family C selected maxmin F1, windowF1, windowprecision, lowerC.',
        'geometry_and_labels': 'Original human span labels, lexical masks, all210364 eligible4raw windows, stride1 and original answer labels unchanged. Answer score=max all eligible windows.',
        'reporting': 'All9 candidates plus frozen baselines. No test or final-generalization claim.',
        'future_information': 'Offline whole claim and retrieved document block are visible to bidirectional MiniCheck. Earlier token states can use later claim words.',
        'next_stage': 'Only after LR complete, a separately frozen single width32 RF7 TCN on LB1024+NLL1+mappedPCA64+per-token risk logit1090; not implemented or run in this stage.',
        'limitations': ['Additional checker parameters, semantic computation and selected-block conditioning change available information.',
            'PCA64 may discard useful low-variance state directions; this fixed stage does not establish full-state information absence.',
            'Repeated calibration selection is optimistic; these development numbers are not sealed-test results.']}


def align(text, offsets, claim_arrays, claims):
    """Text/coordinate-only mapping; no token/span labels accepted by this API."""
    hidden = claim_arrays['hidden_last']; left = claim_arrays['token_start']; right = claim_arrays['token_end']
    assert hidden.ndim == 2 and hidden.shape[1] == 1024 and hidden.dtype == np.float32 and np.isfinite(hidden).all()
    assert left.shape == right.shape == (len(hidden),)
    owners = [[] for _ in text]
    for j, (a, b) in enumerate(zip(left, right)):
        assert 0 <= a < b <= len(text)
        for c in range(int(a), int(b)):
            if not text[c].isspace(): owners[c].append(j)
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and not owners[i]]
    assert not missing, ('Missing MiniCheck nonwhitespace character coverage', missing[:20])
    # Every character has unit mass even if byte-level tokens duplicate its offset.
    char_hidden = np.zeros((len(text), 1024), np.float64)
    for c, indices in enumerate(owners):
        if indices: char_hidden[c] = hidden[indices].astype(np.float64).mean(0)
    support = np.asarray(claim_arrays['selected_support_per_claim'], np.float64)
    assert support.shape == (len(claims),) and np.isfinite(support).all() and ((support >= 0) & (support <= 1)).all()
    risk = 1 - support
    for claim in claims: assert text[claim['start']:claim['end']] == claim['text']
    output = np.zeros((len(offsets), 1024), np.float32); token_risk = np.zeros(len(offsets), np.float64)
    lexical = np.zeros(len(offsets), bool); whitespace = 0
    for j, (a, b) in enumerate(offsets):
        assert 0 <= a <= b <= len(text)
        chars = [c for c in range(int(a), int(b)) if not text[c].isspace()]
        if chars: output[j] = char_hidden[chars].mean(0).astype(np.float32)
        else: whitespace += 1
        alnum = [c for c in range(int(a), int(b)) if text[c].isalnum()]; lexical[j] = bool(alnum)
        if alnum:
            assert all(owners[c] for c in alnum)
            candidate = [ci for ci, claim in enumerate(claims) if any(claim['start'] <= c < claim['end'] for c in alnum)]
            assert candidate, ('Lexical raw BPE missing claim risk', j, a, b)
            token_risk[j] = risk[candidate].max()
    return output, token_risk, lexical, {'nonwhitespace_chars': sum(not c.isspace() for c in text),
        'missing_nonwhitespace_chars': 0, 'lexical_tokens': int(lexical.sum()), 'whitespace_raw_tokens_zeroed': whitespace,
        'duplicated_character_coverage': sum(len(v) > 1 for v in owners)}


def logit(risk):
    p = np.clip(np.asarray(risk, np.float64), 1e-6, 1-1e-6)
    return np.log(p / (1-p))


def synthetic():
    text = 'A中! B'; claims = [{'start': 0, 'end': 3, 'text': 'A中!'}, {'start': 4, 'end': 5, 'text': 'B'}]
    values = np.asarray([2, 4, 8, 10, 12], np.float32)
    a = {'hidden_last': np.repeat(values[:, None], 1024, axis=1),
         'token_start': np.asarray([0, 1, 1, 2, 4]), 'token_end': np.asarray([1, 2, 2, 3, 5]),
         'selected_support_per_claim': np.asarray([.75, .25], np.float32)}
    offsets = np.asarray([[0, 2], [2, 3], [3, 4], [4, 5], [0, 5]])
    h, r, lex, audit = align(text, offsets, a, claims)
    assert h[:, 0].tolist() == [4., 10., 0., 12., 7.5]  # duplicate-char state=(4+8)/2, then equal chars
    assert r.tolist() == [.25, 0., 0., .75, .75] and lex.tolist() == [True, False, False, True, True]
    assert audit['duplicated_character_coverage'] == 1 and audit['missing_nonwhitespace_chars'] == 0
    bad = dict(a)
    for key in ('hidden_last', 'token_start', 'token_end'): bad[key] = a[key][:-1]
    try: align(text, offsets, bad, claims)
    except AssertionError: pass
    else: raise AssertionError('Missing coverage must stop, not silently fill a lexical token')
    assert np.isfinite(logit([0, 1])).all() and np.allclose(logit([.25, .75]), [-np.log(3), np.log(3)])
    return {'passed': True, 'coordinate_unicode_duplicate_character_weighting': True,
        'punctuation_hidden_preserved': True, 'pure_whitespace_zeroed': True,
        'lexical_risk_max_and_missing_coverage_gate': True, 'real_data_fitted': False, 'GPU_used': False}


def initialize():
    assert not (OUT / 'protocol.json').exists()
    check = synthetic(); qa.save(OUT / 'PREFLIGHT.json', check); qa.save(OUT / 'protocol.json', protocol())
    qa.save(OUT / 'freeze.json', {'code_sha256': qa.sha(Path(__file__)), 'baseline_code_sha256': qa.sha(ROOT / 'src/run_development.py'),
        'protocol_sha256': qa.sha(OUT / 'protocol.json'), 'preflight_sha256': qa.sha(OUT / 'PREFLIGHT.json'),
        'upstream_complete_required_before_prepare': True, 'test_opened': False})
    print('QA_SEMANTIC_HIDDEN_PROTOCOL_FROZEN_NO_FIT', flush=True)


def verify_freeze():
    cfg = qa.read(OUT / 'protocol.json'); f = qa.read(OUT / 'freeze.json'); assert cfg == protocol()
    assert qa.sha(Path(__file__)) == f['code_sha256'] and qa.sha(ROOT / 'src/run_development.py') == f['baseline_code_sha256']
    assert qa.sha(OUT / 'protocol.json') == f['protocol_sha256']


def source():
    verify_freeze(); cm = qa.read(SEM / 'claim_feature_manifest.json'); done = qa.read(SEM / 'inference_complete.json')
    assert cm['status'] == done['status'] == 'complete' and cm['answers'] == done['rows'] == 793
    assert cm['hidden_dimension'] == 1024 and not cm['test_opened'] and not done['test_opened']
    assert done['claim_feature_manifest_sha256'] == qa.sha(SEM / 'claim_feature_manifest.json')
    assert cm['source_snapshot_sha256'] == done['source_snapshot_sha256'] == qa.digest(qa.read(SEM / 'inference_source_snapshot.json'))
    meta = qa.metadata(); plans = {p['response_id']: p for p in qa.lines(SEM / 'plans.jsonl')}
    records = {r['response_id']: r for r in cm['records']}
    assert set(plans) == set(records) == {a['response_id'] for a in meta['answers']}
    paths = [Path(__file__), ROOT / 'src/run_development.py', OUT / 'protocol.json', OUT / 'freeze.json',
        SEM / 'claim_feature_manifest.json', SEM / 'inference_complete.json', SEM / 'inference_source_snapshot.json',
        SEM / 'protocol.json', SEM / 'plans.jsonl', ROOT / 'data/gold_manifest.json', OLD / 'complete.json',
        OLD / 'hidden_pca.pkl', OLD / 'training_weights.npz', OLD / 'fit_keys.json', OLD / 'matrix_manifest.json', OLD / 'matrices/base.npy']
    old = qa.read(OLD / 'complete.json')
    for name in ('hidden_pca.pkl', 'training_weights.npz', 'fit_keys.json', 'matrix_manifest.json'):
        assert qa.sha(OLD / name) == old['files_sha256'][name]
    assert qa.sha(OLD / 'matrices/base.npy') == qa.read(OLD / 'matrix_manifest.json')['files_sha256']['base']
    snap = {'files_sha256': {str(p.resolve()): qa.sha(p) for p in paths}, 'test_opened': False, 'checker_finetuned': False}
    return meta, plans, records, snap


def load_claim(answer, plan, rec):
    rid = answer['response_id']; path = SEM / 'claim_features' / (rid + '.npz'); side = qa.read(path.with_suffix('.json'))
    assert qa.sha(path) == rec['npz_sha256'] == side['npz_sha256'] and qa.sha(path.with_suffix('.json')) == rec['metadata_sha256']
    assert side['plan_sha256'] == qa.digest(plan) and side['partition'] == answer['partition'] == plan['partition']
    assert side['original_answer_sha256'] == plan['answer_sha256'] == answer['answer_sha256']
    assert side['claims'] == plan['claims']
    score_path = SEM / 'scores' / (rid + '.json'); assert qa.sha(score_path) == side['score_row_sha256']
    score = qa.read(score_path)
    assert score['plan_sha256'] == side['plan_sha256'] and score['source_snapshot_sha256'] == side['source_snapshot_sha256']
    assert score['partition'] == answer['partition'] and score['device'] == 'cuda:0'
    with np.load(path, allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
    matrix = np.asarray(score['support_by_claim_document'], np.float32); choice = matrix.argmax(1)
    assert np.array_equal(choice, a['selected_document_per_claim'])
    assert np.array_equal(matrix[np.arange(len(choice)), choice], a['selected_support_per_claim'])
    assert np.array_equal(1-a['selected_support_per_claim'].astype(np.float64), np.asarray(score['claim_risk']))
    for ci, claim in enumerate(plan['claims']):
        mask = a['claim_index'] == ci; assert mask.any()
        assert np.all(a['document_index'][mask] == choice[ci])
        assert np.all(a['token_start'][mask] >= claim['start']) and np.all(a['token_end'][mask] <= claim['end'])
    return a


def prepare():
    meta, plans, records, snap = source(); assert not (OUT / 'preparation_started.json').exists()
    qa.save(OUT / 'source_snapshot.json', snap); qa.save(OUT / 'preparation_started.json', {'utc': time.time(), 'source_snapshot_sha256': qa.sha(OUT / 'source_snapshot.json')})
    started = time.perf_counter(); directory = OUT / 'matrices'; directory.mkdir(exist_ok=True)
    hidden = np.lib.format.open_memmap(directory / 'mapped_hidden1024.npy', mode='w+', dtype=np.float32, shape=(213159, 1024))
    risk = np.zeros(213159, np.float64); token_logit = np.zeros(213159, np.float32); index = []; coverage = []; cursor = 0
    for i, answer in enumerate(meta['answers']):
        rid = answer['response_id']; tokens = meta['by_response'][rid]['tokens']; arr = load_claim(answer, plans[rid], records[rid])
        mapped, r, lexical, check = align(answer['original_response'], tokens['response_token_offsets'], arr, plans[rid]['claims'])
        assert lexical.tolist() == tokens['lexical_mask']; n = tokens['token_count']; assert len(mapped) == n
        hidden[cursor:cursor+n] = mapped; risk[cursor:cursor+n] = r; token_logit[cursor:cursor+n] = logit(r).astype(np.float32)
        index.append({'response_id': rid, 'partition': answer['partition'], 'group_id': answer['group_id'], 'left': cursor, 'right': cursor+n, 'token_count': n})
        coverage.append({'response_id': rid, **check}); cursor += n
        if (i+1) % 100 == 0: print('QA_SEMANTIC_ALIGNMENT', i+1, 793, flush=True)
    assert cursor == 213159; hidden.flush(); np.save(directory / 'token_risk.npy', risk); np.save(directory / 'token_risk_logit.npy', token_logit)
    qa.save(OUT / 'token_index.json', {'answers': index}); qa.save(OUT / 'coverage.json', {'records': coverage, 'missing_nonwhitespace_chars': 0, 'no_rows_removed': True})
    old_pca = pickle.loads((OLD / 'hidden_pca.pkl').read_bytes()); by_rid = {a['response_id']: a for a in index}
    sample = []; identities = []; groups = defaultdict(int)
    for answer in index[:634]: groups[answer['group_id']] += 1
    sample_weights = []
    for answer in index[:634]:
        ix = qa.sample_positions(answer['token_count']); sample.append(hidden[answer['left'] + ix])
        sample_weights.extend([1/(groups[answer['group_id']] * len(ix))] * len(ix))
        identities.extend({'response_id': answer['response_id'], 'group_id': answer['group_id'], 'token_index': int(j)} for j in ix)
    assert identities == old_pca['sample']; w = np.asarray(sample_weights, np.float64); w /= w.sum(); assert np.array_equal(w, old_pca['sample_weights'])
    raw = np.concatenate(sample).astype(np.float64); mean = w @ raw; centered = raw - mean
    _, s, components = randomized_svd(centered * np.sqrt(w[:, None]), n_components=64, n_iter=3, random_state=20260924, flip_sign=True)
    trace = float(np.einsum('ij,i,ij->', centered, w, centered)); assert np.isfinite(components).all() and trace > 0
    pca = {'mean': mean, 'components': components, 'singular_values': s, 'explained_variance_ratio_sum': float((s*s).sum()/trace),
        'sample': identities, 'sample_weights': w, 'sample_count': len(w), 'fit_answers': 634, 'fit_groups': 615, 'seed': 20260924, 'n_iter': 3, 'whiten': False}
    (OUT / 'hidden_pca.pkl').write_bytes(pickle.dumps(pca, protocol=5)); del raw, centered, sample; gc.collect()
    projected = np.lib.format.open_memmap(directory / 'token_hidden64.npy', mode='w+', dtype=np.float32, shape=(213159, 64))
    for left in range(0, 213159, BATCH):
        right = min(left+BATCH, 213159); projected[left:right] = ((hidden[left:right].astype(np.float64)-mean)@components.T).astype(np.float32)
    projected.flush(); winhidden = np.zeros((210364, 64), np.float32); winlogit = np.zeros(210364, np.float32)
    for j, window in enumerate(meta['windows']):
        start = by_rid[window['response_id']]['left']; ix = np.asarray(window['token_indices']); tokens = meta['by_response'][window['response_id']]['tokens']
        lexical = np.asarray(tokens['lexical_mask'], bool)[ix]; winhidden[j] = projected[start+ix].mean(0)
        winlogit[j] = logit(risk[start+ix[lexical]].max())
    np.save(directory / 'window_hidden64.npy', winhidden); np.save(directory / 'window_risk_logit.npy', winlogit)
    assert source()[-1] == snap
    names = ['source_snapshot.json', 'token_index.json', 'coverage.json', 'hidden_pca.pkl'] + [str(p.relative_to(OUT)) for p in directory.glob('*.npy')]
    qa.save(OUT / 'preparation_complete.json', {'status': 'complete', 'source_snapshot_sha256': qa.sha(OUT / 'source_snapshot.json'),
        'files_sha256': {name: qa.sha(OUT / name) for name in names}, 'PCA_sample_count': len(w), 'PCA_explained_variance_ratio_sum': pca['explained_variance_ratio_sum'],
        'seconds': time.perf_counter()-started, 'test_opened': False, 'GPU_used': False})
    print('QA_SEMANTIC_HIDDEN_PREPARATION_COMPLETE', flush=True)


def fit():
    meta, _, _, snap = source(); assert snap == qa.read(OUT / 'source_snapshot.json') and not (OUT / 'fit_started.json').exists()
    prep = qa.read(OUT / 'preparation_complete.json'); assert prep['status'] == 'complete'
    for name, expected in prep['files_sha256'].items(): assert qa.sha(OUT / name) == expected
    qa.save(OUT / 'fit_started.json', {'utc': time.time(), 'preparation_sha256': qa.sha(OUT / 'preparation_complete.json')})
    base = np.load(OLD / 'matrices/base.npy', mmap_mode='r'); h = np.load(OUT / 'matrices/window_hidden64.npy', mmap_mode='r')
    r = np.load(OUT / 'matrices/window_risk_logit.npy', mmap_mode='r')
    with np.load(OLD / 'training_weights.npz', allow_pickle=False) as z: weights = {k: z[k].copy() for k in z.files}
    b, loss, y = (weights[k] for k in ('base_weights', 'loss_weights', 'y'))
    assert np.array_equal(y, np.asarray([w['label'] for w in meta['windows'][:168123]])) and np.isclose(loss.sum(),168123)
    families = {}; selected = {}; files = []; start = time.perf_counter()
    for method, width in zip(METHODS, WIDTHS):
        def raw(left, right):
            if method == METHODS[0]: return np.asarray(h[left:right])
            parts = [base[left:right], h[left:right]]
            if method == METHODS[2]: parts.append(r[left:right, None])
            return np.column_stack(parts)
        scaler = StandardScaler()
        for left in range(0, 168123, BATCH):
            right = min(left+BATCH, 168123); scaler.partial_fit(raw(left,right), sample_weight=b[left:right])
        fitpath = OUT / (method + '_fit.npy'); zfit = np.lib.format.open_memmap(fitpath, mode='w+', dtype=np.float32, shape=(168123,width))
        for left in range(0, 168123, BATCH):
            right = min(left+BATCH,168123); zfit[left:right] = scaler.transform(raw(left,right)).astype(np.float32)
        zfit.flush(); entries = []
        for c in protocol()['C']:
            begin = time.perf_counter(); model = LogisticRegression(C=c, solver='liblinear', penalty='l2', max_iter=2000, random_state=20260924)
            model.fit(zfit, y, sample_weight=loss); assert model.n_iter_.max() < 2000
            scores = np.empty(210364,np.float64)
            for left in range(0,len(scores),BATCH):
                right = min(left+BATCH,len(scores)); scores[left:right] = model.predict_proba(scaler.transform(raw(left,right)).astype(np.float32))[:,1]
            answers = qa.answer_scores(meta,scores); ts = {'window':qa.choose_threshold([w['label'] for w in meta['windows'][168123:]],scores[168123:]),
                'answer':qa.choose_threshold([a['label'] for a in meta['answers'][634:]],answers[634:])}
            key = qa.selection_key(ts,c); name = f'{method}_C{c:g}'
            modelpath = OUT / (name+'.pkl'); modelpath.write_bytes(pickle.dumps({'model':model,'scaler':scaler,'C':c,'method':method,'width':width,
                'thresholds':ts,'selection_key':key,'source_snapshot':snap,'weights_file_sha256':qa.sha(OLD/'training_weights.npz'),
                'fit_keys_sha256':qa.sha(OLD/'fit_keys.json'),'pca_sha256':qa.sha(OUT/'hidden_pca.pkl'),'extra_checker':True,'fit_only':True}, protocol=5))
            np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=scores,answer_scores=answers)
            entry = {'candidate':name,'method':method,'C':c,'thresholds':ts,'selection_key':list(key),'metrics':qa.metrics(meta,scores,ts),
                'iterations':model.n_iter_.tolist(),'seconds':time.perf_counter()-begin}
            qa.save(OUT/(name+'_result.json'),entry); entries.append(entry); files.extend([name+'.pkl',name+'_scores.npz',name+'_result.json'])
            print('QA_SEMANTIC_HIDDEN_LR',name,round(entry['seconds'],1),flush=True)
        families[method] = entries; selected[method] = max(entries,key=lambda e:e['selection_key']); del zfit; gc.collect()
    assert source()[-1] == snap
    qa.save(OUT/'summary.json',{'selected':selected,'all_candidates':families,'seconds':time.perf_counter()-start,'test_opened':False,
        'calibration_results_are_selection_optimistic':True,'extra_checker_model':True,'checker_finetuned':False})
    files.extend(['summary.json','fit_started.json','preparation_complete.json','protocol.json','freeze.json'])
    qa.save(OUT/'complete.json',{'status':'complete_development_only','files_sha256':{name:qa.sha(OUT/name) for name in files},
        'test_opened':False,'GPU_used':False,'extra_checker_model':True})
    print('QA_SEMANTIC_HIDDEN_LR_COMPLETE',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['synthetic','initialize','prepare','fit']); args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage == 'synthetic': print(synthetic())
        else: {'initialize':initialize,'prepare':prepare,'fit':fit}[args.stage]()
