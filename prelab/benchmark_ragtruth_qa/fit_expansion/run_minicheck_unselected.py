"""Third-stage complement cache: CPU design/binding, externally scheduled GPU infer.

Never imports a model during CPU preparation. No labels, training, or test access.
Existing selected caches are read-only. All writes stay in minicheck_unselected.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
import numpy as np

HERE = Path(__file__).resolve().parent
QA = HERE.parent
DEST = HERE/'minicheck_unselected'
NEW = HERE/'minicheck'
OLD = QA/'semantic_baseline/cuda_variant'


def sha(path):
    path = Path(path).resolve()
    assert path.is_relative_to(QA)
    assert not any(s in str(path.relative_to(QA)).lower() for s in ('sealed', 'withheld', 'test'))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text('utf-8'))


def readl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_bytes(path, content, frozen=False):
    path = Path(path).resolve(); assert path.is_relative_to(DEST.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        assert path.read_bytes() == content, f'Frozen output differs: {path}'
        return
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_bytes(content); tmp.replace(path)


def write(path, value, frozen=False):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode(), frozen)


def writel(path, values):
    write_bytes(path, ''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values).encode(), True)


def save_npz(path, arrays):
    path = Path(path).resolve(); assert path.is_relative_to(DEST.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f: np.savez(f, **arrays)
    tmp.replace(path)


def protocol():
    return {
        'version': 'qa-unselected-documents-encoder22-and-claim-last-v1',
        'scope': 'New3046 fit answers plus original fit634/cal159; no official test, no labels or fitting',
        'stages': 'freeze-design CPU now; bind CPU only after stage1 inference_complete; infer only when root schedules GPU after stages1/2',
        'checkpoint': 'lytang/MiniCheck-RoBERTa-Large',
        'revision': '74c8919647e61ed0f71bc177d94f10930f090068',
        'expected': {'new_answers': 3046, 'original_answers': 793, 'new_pairs': 5918,
                     'original_pairs': 2308, 'total_pairs': 8226, 'fit_answers': 3680, 'calibration_answers': 159},
        'selection': 'Copy frozen stage1 selected-document metadata for new answers and old final-state cache decisions for original793. Verify against saved all-doc scores. Enumerate every (claim_index,document_index) except the one selected document; no new selection or filtering.',
        'binding': 'Require stage1 complete and its score/claim/encoder22 manifests; bind original plans, scores, selected metadata and used selected claim arrays by SHA256. New choices unavailable before completion are never guessed.',
        'ordering': 'New answer plan order then original answer plan order. Within answer: claim index ascending, then unselected document index ascending. Both sequence indices may repeat.',
        'input': 'Exact original document chunk + tokenizer eos_token + exact claim text; no question, generator identity or annotation input',
        'device': 'cuda:0', 'dtype': 'float32', 'batch_size': 4, 'TF32': False, 'autocast': False,
        'encoder22_schema': 'hidden22[sum_valid_input_tokens,1024]; input_ids/attention_mask/answer_token_start/answer_token_end[sum_valid_input_tokens]; sequence_offsets[n_unselected_pairs+1]; claim_index/document_index/batch_padding_length/original_all_doc_batch_padding_length[n_unselected_pairs]. Include all valid document/special/claim tokens; answer coordinates are absolute original-response positions, otherwise -1.',
        'claim_last_schema': 'hidden_last[sum_claim_tokens,1024]; token_ids/token_start/token_end/input_token_index/token_claim_index/token_document_index[sum_claim_tokens]; sequence_offsets[n_unselected_pairs+1]; claim_index/document_index[n_unselected_pairs]. Actual repeated claim/doc IDs; input_token_index belongs to this document input, not the old selected document.',
        'state': 'encoder.layer[21] output[0] and model.roberta last_hidden_state on claim tokens; shared frozen read-only observer implementation',
        'scores': 'Save logits[2] and class1 support for every unselected pair, plus true claim/doc indices; no per-document risk labels or threshold predictions',
        'agreement': {'logit_max_abs_tolerance': 2e-4, 'support_max_abs_tolerance': 2e-5,
            'same_claim_token_ids_and_original_answer_offsets_vs_old_selected': 'exact',
            'input_positions': 'checked within own full input; never compared to selected document positions',
            'first_two_new_answer_forwards_observer_logits_support_last_state': 'exact'},
        'downstream_label_rule': 'Answer/claim/window risk is defined only after aggregating all retrieved documents. A single document lacking evidence is not an answer-level risk label. No global labels are loaded or assigned in this cache stage.',
        'zero_complement': 'One-document answers have no missing pairs; explicit zero-pair manifest record and no forward/cache arrays',
        'cache_scope': 'Dedicated minicheck_unselected directory, keyed by frozen bound plan and source snapshot hashes; never overwrite or mix selected caches',
        'training': False, 'official_test_opened': False,
    }


def source_paths():
    paths = [Path(__file__), HERE/'run_minicheck_expansion.py', HERE/'protocol.json',
             HERE/'data/export_freeze.json', NEW/'new_plans.jsonl', OLD/'plans.jsonl',
             OLD/'protocol.json', OLD/'inference_complete.json', OLD/'claim_feature_manifest.json',
             QA/'semantic_baseline/run_semantic.py', QA/'semantic_baseline/run_semantic_cuda.py',
             QA/'semantic_baseline/download_manifest.json']
    return {str(p.resolve()): sha(p) for p in paths}


def plans():
    new, old = readl(NEW/'new_plans.jsonl'), readl(OLD/'plans.jsonl')
    assert len(new) == 3046 and len(old) == 793
    assert Counter(p['partition'] for p in new) == {'fit': 3046}
    assert Counter(p['partition'] for p in old) == {'fit': 634, 'calibration': 159}
    allplans = new+old
    assert len({p['response_id'] for p in allplans}) == 3839
    assert not ({p['group_id'] for p in allplans if p['partition'] == 'fit'} &
                {p['group_id'] for p in allplans if p['partition'] == 'calibration'})
    for p in allplans:
        c, d = len(p['claims']), len(p['document_chunks'])
        assert c > 0 and d > 0 and len(p['pair_lengths']) == c*d
        assert min(p['pair_lengths']) > 0 and max(p['pair_lengths']) <= 512
        assert p['truncated_input_tokens'] == p['uncovered_nonwhitespace_chars'] == 0
    return [('new3046', new, NEW), ('original793', old, OLD)]


def complement(p, chosen):
    assert len(chosen) == len(p['claims'])
    d = len(p['document_chunks'])
    assert all(isinstance(x, int) and 0 <= x < d for x in chosen)
    return [(ci, di) for ci in range(len(chosen)) for di in range(d) if di != chosen[ci]]


def freeze_design():
    assert not (DEST/'design_freeze.json').exists(), 'Design already frozen'
    sources = source_paths(); estimates = {}; oldrefs = {
        r['response_id']: r for r in read(OLD/'claim_feature_manifest.json')['records']}
    for origin, pp, directory in plans():
        lo = hi = pairs = last = active = 0
        for p in pp:
            c, d = len(p['claims']), len(p['document_chunks']); active += d > 1
            lengths = np.asarray(p['pair_lengths'], np.int64).reshape(c, d)
            pairs += c*(d-1); last += sum(x['token_count'] for x in p['claims'])*(d-1)
            if origin == 'original793':
                path = directory/'claim_features'/f"{p['response_id']}.json"
                assert sha(path) == oldrefs[p['response_id']]['metadata_sha256']
                m = read(path); assert m['plan_sha256'] == digest(p)
                chosen = m['selected_document_per_claim']; complement(p, chosen)
                remaining = int(lengths.sum()-sum(lengths[i, di] for i, di in enumerate(chosen)))
                lo += remaining; hi += remaining
            else:
                lo += int((lengths.sum(1)-lengths.max(1)).sum())
                hi += int((lengths.sum(1)-lengths.min(1)).sum())
        estimates[origin] = {'answers': len(pp), 'answers_with_missing_pairs': active, 'unselected_pairs': pairs,
            'valid_input_tokens_lower': lo, 'valid_input_tokens_upper': hi,
            'encoder22_float32_bytes_lower': lo*4096, 'encoder22_float32_bytes_upper': hi*4096,
            'claim_last_tokens_estimate_from_plan_counts': last, 'claim_last_float32_bytes_estimate': last*4096}
    assert estimates['new3046']['unselected_pairs'] == 5918 and estimates['original793']['unselected_pairs'] == 2308
    lower = sum(e['encoder22_float32_bytes_lower']+e['claim_last_float32_bytes_estimate'] for e in estimates.values())
    upper = sum(e['encoder22_float32_bytes_upper']+e['claim_last_float32_bytes_estimate'] for e in estimates.values())
    estimate = {'status': 'design_bounds_not_new_selection_binding', 'origins': estimates, 'total_pairs': 8226,
        'hidden_bytes_lower': lower, 'hidden_bytes_upper': upper,
        'hidden_GiB_lower': lower/2**30, 'hidden_GiB_upper': upper/2**30,
        'coordinates_ids_json_npz_headers_additional': True, 'claim_last_count_is_estimate_until_bind': True,
        'disk_free_bytes': shutil.disk_usage(HERE).free,
        'new_choices_used': False, 'old_choices': 'Frozen original selected metadata only; old extra input count exact'}
    write(DEST/'protocol.json', protocol(), True); write(DEST/'design_estimate.json', estimate, True)
    assert source_paths() == sources
    write(DEST/'design_freeze.json', {'status': 'frozen_design_cpu_only_waiting_for_stage1_complete',
        'utc': datetime.now(timezone.utc).isoformat(), 'source_files_sha256': sources,
        'files_sha256': {n: sha(DEST/n) for n in ('protocol.json', 'design_estimate.json')},
        'GPU_initialized': False, 'models_run': False, 'labels_read': False, 'test_opened': False}, True)
    print('UNSELECTED_DESIGN_FROZEN', json.dumps(estimate), flush=True)


def verify_design():
    frozen = read(DEST/'design_freeze.json')
    assert read(DEST/'protocol.json') == protocol()
    for p, h in frozen['source_files_sha256'].items(): assert sha(p) == h, p
    for p, h in frozen['files_sha256'].items(): assert sha(DEST/p) == h, p
    return frozen


def bind():
    verify_design(); assert not (DEST/'binding_freeze.json').exists(), 'Already bound'
    assert (NEW/'inference_complete.json').exists(), 'Stage1 must complete before binding any new choices'
    done = read(NEW/'inference_complete.json')
    assert done['status'] == 'complete' and done['rows'] == 3046
    assert sha(NEW/'claim_feature_manifest.json') == done['claim_feature_manifest_sha256']
    assert sha(NEW/'encoder22_feature_manifest.json') == done['encoder22_feature_manifest_sha256']
    score_refs = {r['response_id']: r['file_sha256'] for r in done['records']}
    lowrefs = {r['response_id']: r for r in read(NEW/'encoder22_feature_manifest.json')['records']}
    bindings = {}; bound = []; counts = {}; seen = set()
    def remember(path, expected=None):
        value = sha(path)
        if expected is not None: assert value == expected, path
        bindings[str(Path(path).resolve())] = value; return value
    for p in [NEW/'inference_complete.json', NEW/'claim_feature_manifest.json',
              NEW/'encoder22_feature_manifest.json', NEW/'inference_source_snapshot.json']:
        remember(p)
    for origin, pp, directory in plans():
        manifest = directory/'claim_feature_manifest.json'; remember(manifest)
        refs = {r['response_id']: r for r in read(manifest)['records']}
        totals = Counter()
        for p in pp:
            rid = p['response_id']; assert rid not in seen; seen.add(rid)
            scorepath = directory/'scores'/f'{rid}.json'
            metapath = directory/'claim_features'/f'{rid}.json'; arraypath = metapath.with_suffix('.npz')
            remember(metapath, refs[rid]['metadata_sha256']); m = read(metapath)
            remember(scorepath, m['score_row_sha256']); s = read(scorepath)
            assert m['plan_sha256'] == s['plan_sha256'] == digest(p)
            assert m['original_answer_sha256'] == p['answer_sha256']
            assert m['partition'] == s['partition'] == p['partition']
            chosen = list(map(int, m['selected_document_per_claim']))
            probs = np.asarray(s['support_by_claim_document'], np.float32)
            logits = np.asarray(s['logits'], np.float32)
            c, d = len(p['claims']), len(p['document_chunks'])
            assert probs.shape == (c, d) and logits.shape == (c, d, 2)
            assert np.isfinite(probs).all() and np.isfinite(logits).all()
            assert chosen == probs.argmax(1).tolist()
            if origin == 'new3046':
                assert sha(scorepath) == score_refs[rid]
                lowpath = NEW/'encoder22_features'/f'{rid}.json'
                remember(lowpath, lowrefs[rid]['metadata_sha256']); low = read(lowpath)
                assert low['plan_sha256'] == digest(p) and low['score_row_sha256'] == sha(scorepath)
                assert low['selected_document_per_claim'] == chosen
            pairs = complement(p, chosen); claim_counts = np.zeros(c, np.int64)
            assert m['npz_sha256'] == refs[rid]['npz_sha256']
            if pairs:
                remember(arraypath, refs[rid]['npz_sha256'])
                with np.load(arraypath, allow_pickle=False) as z:
                    assert z['selected_document_per_claim'].astype(int).tolist() == chosen
                    claim_counts = np.bincount(z['claim_index'], minlength=c)
                    assert len(claim_counts) == c and (claim_counts > 0).all()
            spec = []
            for ci, di in pairs:
                flat = ci*d+di; start = flat//4*4
                spec.append({'claim_index': ci, 'document_index': di, 'input_length': p['pair_lengths'][flat],
                    'claim_token_count': int(claim_counts[ci]), 'reference_logits': logits[ci, di].tolist(),
                    'reference_support': float(probs[ci, di]),
                    'original_all_doc_batch_padding_length': max(p['pair_lengths'][start:start+4])})
            row = {'origin': origin, 'response_id': rid, 'partition': p['partition'], 'group_id': p['group_id'],
                'plan': p, 'original_plan_sha256': digest(p), 'selected_document_per_claim': chosen,
                'unselected_pairs': spec, 'reference_score_path': str(scorepath.resolve()),
                'reference_score_sha256': sha(scorepath), 'selected_claim_npz_path': str(arraypath.resolve()),
                'selected_claim_npz_sha256': refs[rid]['npz_sha256'],
                'selected_claim_metadata_sha256': sha(metapath)}
            bound.append(row)
            totals.update({'answers': 1, 'answers_with_missing_pairs': int(bool(spec)), 'pairs': len(spec),
                'input_tokens': sum(x['input_length'] for x in spec),
                'claim_tokens': sum(x['claim_token_count'] for x in spec)})
        counts[origin] = dict(totals)
    assert counts['new3046']['pairs'] == 5918 and counts['original793']['pairs'] == 2308
    assert len(bound) == 3839 and Counter(r['partition'] for r in bound) == {'fit': 3680, 'calibration': 159}
    exactbytes = 4096*sum(r['input_tokens']+r['claim_tokens'] for r in counts.values())
    stats = {'origins': counts, 'pairs': 8226, 'hidden_float32_bytes_exact': exactbytes,
             'coordinates_and_metadata_additional': True, 'disk_free_bytes': shutil.disk_usage(HERE).free}
    assert stats['disk_free_bytes'] > exactbytes*1.2, 'Insufficient disk margin for unselected caches'
    for p, h in bindings.items(): assert sha(p) == h
    verify_design(); writel(DEST/'bound_plans.jsonl', bound); write(DEST/'bound_statistics.json', stats, True)
    write(DEST/'binding_freeze.json', {'status': 'bound_cpu_only_gpu_not_run', 'utc': datetime.now(timezone.utc).isoformat(),
        'design_freeze_sha256': sha(DEST/'design_freeze.json'), 'source_files_sha256': bindings,
        'files_sha256': {p: sha(DEST/p) for p in ('bound_plans.jsonl', 'bound_statistics.json')},
        'pairs': 8226, 'labels_read': False, 'test_opened': False}, True)
    print('UNSELECTED_BOUND', json.dumps(stats), flush=True)


def verify_binding():
    verify_design(); frozen = read(DEST/'binding_freeze.json')
    assert frozen['design_freeze_sha256'] == sha(DEST/'design_freeze.json')
    for p, h in frozen['source_files_sha256'].items(): assert sha(p) == h, p
    for p, h in frozen['files_sha256'].items(): assert sha(DEST/p) == h, p
    bound = readl(DEST/'bound_plans.jsonl')
    assert len(bound) == 3839 and sum(len(r['unselected_pairs']) for r in bound) == 8226
    for r in bound:
        assert digest(r['plan']) == r['original_plan_sha256']
        assert [(x['claim_index'], x['document_index']) for x in r['unselected_pairs']] == complement(r['plan'], r['selected_document_per_claim'])
    return frozen, bound


def inputs(row, tok):
    texts = []; metadata = []
    for s in row['unselected_pairs']:
        ci, di = s['claim_index'], s['document_index']; c = row['plan']['claims'][ci]
        prefix = row['plan']['document_chunks'][di]['text']+tok.eos_token
        texts.append(prefix+c['text'])
        metadata.append({'claim': c, 'claim_index': ci, 'document_index': di, 'input_claim_start': len(prefix)})
    assert [len(tok.encode(t)) for t in texts] == [s['input_length'] for s in row['unselected_pairs']]
    return texts, metadata


def check_traces(row, traces):
    assert len(traces) == len(row['unselected_pairs'])
    with np.load(row['selected_claim_npz_path'], allow_pickle=False) as z:
        selected_claim = z['claim_index']; selected_ids = z['token_ids']
        selected_start, selected_end = z['token_start'], z['token_end']
        for pair, t in zip(row['unselected_pairs'], traces):
            ci, di = pair['claim_index'], pair['document_index']; mask = selected_claim == ci
            assert np.array_equal(t['token_ids'], selected_ids[mask])
            assert np.array_equal(t['token_start'], selected_start[mask]) and np.array_equal(t['token_end'], selected_end[mask])
            assert len(t['token_ids']) == pair['claim_token_count']
            assert np.all(t['claim_index'] == ci) and np.all(t['document_index'] == di)
            assert len(t['full_input_ids']) == pair['input_length'] and np.all(t['full_attention_mask'] == 1)
            keep = t['input_token_index']
            assert np.array_equal(t['full_input_ids'][keep], t['token_ids'])
            assert np.array_equal(t['full_answer_token_start'][keep], t['token_start'])
            assert np.array_equal(t['full_answer_token_end'][keep], t['token_end'])
            assert np.array_equal(np.flatnonzero(t['full_answer_token_start'] >= 0), keep)


def save_answer(row, z, prob, traces, agreement, snapshot):
    rid = row['response_id']; pairs = row['unselected_pairs']
    ci = np.asarray([s['claim_index'] for s in pairs], np.int32)
    di = np.asarray([s['document_index'] for s in pairs], np.int32)
    full = {'hidden22': np.concatenate([t['full_hidden22'] for t in traces]),
        'input_ids': np.concatenate([t['full_input_ids'] for t in traces]),
        'attention_mask': np.concatenate([t['full_attention_mask'] for t in traces]).astype(np.int8),
        'answer_token_start': np.concatenate([t['full_answer_token_start'] for t in traces]),
        'answer_token_end': np.concatenate([t['full_answer_token_end'] for t in traces]),
        'sequence_offsets': np.r_[0, np.cumsum([len(t['full_input_ids']) for t in traces])].astype(np.int64),
        'claim_index': ci, 'document_index': di,
        'batch_padding_length': np.asarray([t['batch_padding_length'] for t in traces], np.int32),
        'original_all_doc_batch_padding_length': np.asarray([s['original_all_doc_batch_padding_length'] for s in pairs], np.int32)}
    last = {k: np.concatenate([t[k] for t in traces]) for k in
            ('hidden_last', 'token_ids', 'token_start', 'token_end', 'input_token_index')}
    last.update(sequence_offsets=np.r_[0, np.cumsum([len(t['token_ids']) for t in traces])].astype(np.int64),
        claim_index=ci, document_index=di,
        token_claim_index=np.concatenate([t['claim_index'] for t in traces]),
        token_document_index=np.concatenate([t['document_index'] for t in traces]))
    assert full['hidden22'].shape == (int(full['sequence_offsets'][-1]), 1024)
    assert last['hidden_last'].shape == (int(last['sequence_offsets'][-1]), 1024)
    for array in (full['hidden22'], last['hidden_last']): assert array.dtype == np.float32 and np.isfinite(array).all()
    scorepath = DEST/'scores'/f'{rid}.json'
    record = {'response_id': rid, 'partition': row['partition'], 'group_id': row['group_id'],
        'bound_plan_sha256': digest(row), 'source_snapshot_sha256': digest(snapshot),
        'original_plan_sha256': row['original_plan_sha256'], 'original_answer_sha256': row['plan']['answer_sha256'],
        'reference_score_path': row['reference_score_path'], 'reference_score_sha256': row['reference_score_sha256'],
        'pairs': [{'claim_index': int(c), 'document_index': int(d), 'logits': logits.tolist(), 'support': float(p)}
                  for c, d, logits, p in zip(ci, di, z, prob)],
        'numeric_agreement': agreement, 'global_risk_labels_assigned': False}
    write(scorepath, record)
    files = {str(scorepath.relative_to(DEST)): sha(scorepath)}
    for directory, arrays in [('encoder22_features', full), ('claim_features', last)]:
        path = DEST/directory/f'{rid}.npz'; save_npz(path, arrays)
        meta = {'response_id': rid, 'partition': row['partition'], 'group_id': row['group_id'],
            'npz_sha256': sha(path), 'bound_plan_sha256': digest(row), 'source_snapshot_sha256': digest(snapshot),
            'original_answer_sha256': row['plan']['answer_sha256'], 'score_row_sha256': sha(scorepath),
            'original_plan_sha256': row['original_plan_sha256'],
            'reference_score_path': row['reference_score_path'], 'reference_score_sha256': row['reference_score_sha256'],
            'sequences': len(pairs), 'claim_index': ci.tolist(), 'document_index': di.tolist(),
            'valid_tokens': int(arrays['sequence_offsets'][-1]), 'hidden_dimension': 1024, 'dtype': 'float32',
            'cache_kind': directory, 'selection': 'Frozen complement only; no selected pairs',
            'coordinates': 'Original-response claim offsets; own-document local input indices; full-input nonclaim offsets -1'}
        write(path.with_suffix('.json'), meta)
        for f in (path, path.with_suffix('.json')): files[str(f.relative_to(DEST))] = sha(f)
    return {'response_id': rid, 'partition': row['partition'], 'group_id': row['group_id'], 'pairs': len(pairs),
        'input_tokens': int(full['sequence_offsets'][-1]), 'claim_tokens': int(last['sequence_offsets'][-1]),
        'bound_plan_sha256': digest(row), 'source_snapshot_sha256': digest(snapshot),
        'files_sha256': files, 'numeric_agreement': agreement}


def infer():
    frozen, bound = verify_binding()
    assert not (DEST/'inference_complete.json').exists(), 'Complete; do not repeat inference'
    snapshot = {'design_freeze_sha256': sha(DEST/'design_freeze.json'),
        'binding_freeze_sha256': sha(DEST/'binding_freeze.json'), 'runner_sha256': sha(Path(__file__)),
        'device': 'cuda:0', 'dtype': 'float32'}
    write(DEST/'inference_source_snapshot.json', snapshot, True)
    for relative, expected in read(QA/'semantic_baseline/download_manifest.json')['files_sha256'].items():
        assert sha(OLD/relative) == expected
    spec = importlib.util.spec_from_file_location('frozen_unselected_forward', HERE/'run_minicheck_expansion.py')
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
    model, tok = exp.load_model(); records = []; count = 0; started = time.perf_counter()
    for i, row in enumerate(bound):
        rid = row['response_id']; commit = DEST/'rows'/f'{rid}.json'
        if not row['unselected_pairs']:
            records.append({'response_id': rid, 'partition': row['partition'], 'group_id': row['group_id'],
                            'pairs': 0, 'input_tokens': 0, 'claim_tokens': 0, 'reason': 'No unselected document'})
            continue
        if commit.exists():
            r = read(commit)
            assert r['bound_plan_sha256'] == digest(row) and r['source_snapshot_sha256'] == digest(snapshot)
            for name, expected in r['files_sha256'].items(): assert sha(DEST/name) == expected
        else:
            texts, metadata = inputs(row, tok)
            z, prob, traces = exp.probabilities_with22(model, tok, texts, metadata)
            observer = count < 2
            if observer:
                plainz, plainp, plain = exp.cuda.probabilities(model, tok, texts, metadata)
                assert np.array_equal(z, plainz) and np.array_equal(prob, plainp)
                assert all(np.array_equal(t['hidden_last'], p['hidden_last']) for t, p in zip(traces, plain))
            expected_z = np.asarray([s['reference_logits'] for s in row['unselected_pairs']], np.float32)
            expected_p = np.asarray([s['reference_support'] for s in row['unselected_pairs']], np.float32)
            dz, dp = float(abs(z-expected_z).max()), float(abs(prob-expected_p).max())
            assert dz <= 2e-4 and dp <= 2e-5, (rid, dz, dp)
            check_traces(row, traces)
            agreement = {'logit_max_abs': dz, 'support_max_abs': dp, 'claim_token_ids_and_offsets_exact': True,
                'own_document_input_positions_checked': True, 'observer_exact_checked': observer,
                'old_selected_final_states_not_compared_across_documents': True}
            r = save_answer(row, z, prob, traces, agreement, snapshot); write(commit, r)
            count += 1
        records.append(r)
        if (i+1) % 25 == 0: print('UNSELECTED_PROGRESS', i+1, len(bound), round(time.perf_counter()-started, 1), flush=True)
    assert sum(r['pairs'] for r in records) == 8226
    for path, expected in frozen['source_files_sha256'].items(): assert sha(path) == expected
    verify_design()
    write(DEST/'feature_manifest.json', {'status': 'complete', 'records': records, 'answers': len(records),
        'pairs': 8226, 'input_tokens': sum(r['input_tokens'] for r in records),
        'claim_tokens': sum(r['claim_tokens'] for r in records), 'partitions': {'fit': 3680, 'calibration': 159},
        'source_snapshot_sha256': digest(snapshot), 'global_risk_labels_assigned': False, 'test_opened': False})
    write(DEST/'inference_complete.json', {'status': 'complete', 'feature_manifest_sha256': sha(DEST/'feature_manifest.json'),
        'source_snapshot_sha256': digest(snapshot), 'pairs': 8226, 'this_invocation_seconds': time.perf_counter()-started,
        'no_training': True, 'labels_read': False, 'test_opened': False})
    del model
    import torch
    torch.cuda.empty_cache()
    print('UNSELECTED_COMPLETE_GPU_RELEASED', flush=True)


def selfcheck():
    p = {'claims': [{}, {}], 'document_chunks': [{}, {}, {}]}
    assert complement(p, [1, 0]) == [(0, 0), (0, 2), (1, 1), (1, 2)]
    assert complement({'claims': [{}, {}], 'document_chunks': [{}]}, [0, 0]) == []
    for choices in ([0, 0], [1, 2], [2, 1]):
        missing = set(complement(p, choices)); selected = set(enumerate(choices))
        assert not missing & selected and missing | selected == {(i, d) for i in range(2) for d in range(3)}
    print('UNSELECTED_CPU_RULES_SELFCHECK_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'freeze-design', 'bind', 'verify', 'infer'])
    args = parser.parse_args()
    {'selfcheck': selfcheck, 'freeze-design': freeze_design, 'bind': bind,
     'verify': verify_binding, 'infer': infer}[args.stage]()
