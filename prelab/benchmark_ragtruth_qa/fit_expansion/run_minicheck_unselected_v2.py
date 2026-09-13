"""Original all-document batch geometry; save only the frozen unselected pairs.

CPU freeze/verify do not import torch or load a model. GPU scheduling belongs to
root. The v1 code and artifacts are read-only; this wrapper writes a new directory.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import importlib.util
import time
import numpy as np

HERE = Path(__file__).resolve().parent
DEST = HERE/'minicheck_unselected_v2'
V1 = HERE/'minicheck_unselected'
spec = importlib.util.spec_from_file_location('frozen_unselected_v1_helpers', HERE/'run_minicheck_unselected.py')
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
v1.DEST = DEST
sha, read, digest, write = v1.sha, v1.read, v1.digest, v1.write


def protocol():
    return {'version': 'qa-unselected-original-all-doc-batches-v2',
        'scope': 'Exact frozen v1 8226 unselected pairs: new3046 + original793; no labels/training/test',
        'forward': 'For each answer with a nonempty complement, original complete claim-major/document-major pairs, original batch4 dynamic padding; then retain only frozen unselected rows. Zero-complement answers skipped.',
        'maximum_documents': 2, 'device': 'cuda:0', 'dtype': 'float32', 'batch_size': 4,
        'TF32': False, 'autocast': False,
        'selection': 'Read-only v1 bound complement and old/new selected decisions; never select again',
        'geometry': 'flat=claim_index*D+document_index; original_batch_index=flat//4; original_batch_row=flat%4; original_batch_size=min(4,C*D-4*(flat//4)); original_batch_padding_length=max(original pair_lengths of that batch). Actual batch_padding_length must equal original for every complete-forward pair.',
        'agreement': {'all_doc_logits_max_abs': 2e-4, 'all_doc_support_max_abs': 2e-5,
            'selected_last_hidden_max_abs': 2e-4,
            'all_selected_token_ids_offsets_claim_doc_input_indices': 'exact',
            'unselected_claim_ids_and_offsets_vs_selected_same_claim': 'exact',
            'first_two_fresh_active_answer_observers': 'exact logits/support/last states'},
        'schema': 'Same v1 two NPZ schemas and metadata. Both encoder22_features and claim_features NPZ additionally contain sequence-level original_batch_index/original_batch_row/original_batch_size/original_batch_padding_length and actual batch_padding_length. All four original fields have length n_unselected_sequences.',
        'input_indices': 'Own document input position; absolute original-answer claim offsets; full-input nonclaim offsets=-1',
        'storage': 'Dedicated minicheck_unselected_v2; never write v1 or selected caches',
        'labels': 'None read or assigned. A missing single-document support is not a global risk label; aggregate all documents downstream.',
        'verification': 'CPU verify hashes frozen v1 binding/plans/code and recomputes complement/batch geometry; infer additionally rechecks every v1 bound source hash before loading the model.',
        'GPU_run_by_prepare': False, 'test_opened': False}


def source_paths():
    pp = [Path(__file__), HERE/'run_minicheck_unselected.py', HERE/'run_minicheck_expansion.py',
          V1/'design_freeze.json', V1/'binding_freeze.json', V1/'bound_plans.jsonl',
          V1/'bound_statistics.json', V1/'protocol.json',
          v1.QA/'semantic_baseline/run_semantic.py', v1.QA/'semantic_baseline/run_semantic_cuda.py',
          v1.QA/'semantic_baseline/download_manifest.json']
    return {str(p.resolve()): sha(p) for p in pp}


def original_geometry(plan, pairs):
    c, d = len(plan['claims']), len(plan['document_chunks']); total = c*d
    assert 1 <= d <= 2 and len(plan['pair_lengths']) == total
    flat = np.asarray([ci*d+di for ci, di in pairs], np.int64)
    start = flat//4*4
    return {'original_batch_index': (flat//4).astype(np.int32),
        'original_batch_row': (flat%4).astype(np.int32),
        'original_batch_size': np.minimum(4, total-start).astype(np.int32),
        'original_batch_padding_length': np.asarray(
            [max(plan['pair_lengths'][int(a):int(a)+4]) for a in start], np.int32)}


def bound_plans():
    binding = read(V1/'binding_freeze.json'); design = read(V1/'design_freeze.json')
    assert binding['design_freeze_sha256'] == sha(V1/'design_freeze.json')
    assert design['source_files_sha256'][str((HERE/'run_minicheck_unselected.py').resolve())] == sha(HERE/'run_minicheck_unselected.py')
    for name, expected in binding['files_sha256'].items(): assert sha(V1/name) == expected
    bound = v1.readl(V1/'bound_plans.jsonl'); totals = Counter(); origins = {}
    assert len(bound) == 3839 and len({r['response_id'] for r in bound}) == 3839
    assert Counter(r['partition'] for r in bound) == {'fit': 3680, 'calibration': 159}
    for r in bound:
        p = r['plan']; assert digest(p) == r['original_plan_sha256']
        c, d = len(p['claims']), len(p['document_chunks']); assert 1 <= d <= 2
        assert (r['response_id'], r['partition'], r['group_id']) == (p['response_id'], p['partition'], p['group_id'])
        expected = v1.complement(p, r['selected_document_per_claim'])
        pairs = [(s['claim_index'], s['document_index']) for s in r['unselected_pairs']]
        assert pairs == expected and len(set(pairs)) == len(pairs)
        g = original_geometry(p, pairs)
        assert g['original_batch_padding_length'].tolist() == [s['original_all_doc_batch_padding_length'] for s in r['unselected_pairs']]
        t = Counter(answers=1, active_answers=int(bool(pairs)), unselected_pairs=len(pairs),
            full_forward_pairs=c*d if pairs else 0,
            full_forward_input_tokens=sum(p['pair_lengths']) if pairs else 0,
            saved_input_tokens=sum(s['input_length'] for s in r['unselected_pairs']),
            saved_claim_tokens=sum(s['claim_token_count'] for s in r['unselected_pairs']))
        totals.update(t); origins.setdefault(r['origin'], Counter()).update(t)
    assert totals['unselected_pairs'] == 8226 and totals['full_forward_pairs'] == 16452
    assert totals['active_answers'] == 956
    stats = {'totals': dict(totals), 'origins': {k: dict(v) for k,v in origins.items()},
        'hidden_float32_bytes_exact': 4096*(totals['saved_input_tokens']+totals['saved_claim_tokens']),
        'extra_geometry_bytes_both_caches': totals['unselected_pairs']*4*4*2,
        'other_coordinates_and_metadata_additional': True}
    return binding, bound, stats


def freeze():
    assert not (DEST/'preparation_freeze.json').exists(), 'Already frozen'
    snap = source_paths(); _, _, stats = bound_plans()
    write(DEST/'protocol.json', protocol(), True); write(DEST/'preparation_statistics.json', stats, True)
    assert source_paths() == snap
    write(DEST/'preparation_freeze.json', {'status': 'frozen_cpu_only',
        'utc': datetime.now(timezone.utc).isoformat(), 'source_files_sha256': snap,
        'files_sha256': {n: sha(DEST/n) for n in ('protocol.json', 'preparation_statistics.json')},
        'v1_binding_freeze_sha256': sha(V1/'binding_freeze.json'),
        'labels_read': False, 'GPU_initialized': False, 'test_opened': False}, True)
    print('UNSELECTED_V2_CPU_FROZEN', stats, flush=True)


def verify(deep=False):
    f = read(DEST/'preparation_freeze.json'); assert read(DEST/'protocol.json') == protocol()
    for p,h in f['source_files_sha256'].items(): assert sha(p) == h, p
    for p,h in f['files_sha256'].items(): assert sha(DEST/p) == h, p
    binding, bound, stats = bound_plans()
    assert stats == read(DEST/'preparation_statistics.json')
    assert f['v1_binding_freeze_sha256'] == sha(V1/'binding_freeze.json')
    if deep:
        for p,h in binding['source_files_sha256'].items(): assert sha(p) == h, p
        for p,h in read(V1/'design_freeze.json')['source_files_sha256'].items(): assert sha(p) == h, p
    return f, bound


def check_selected(row, traces):
    d = len(row['plan']['document_chunks'])
    selected = [traces[ci*d+di] for ci,di in enumerate(row['selected_document_per_claim'])]
    with np.load(row['selected_claim_npz_path'], allow_pickle=False) as old:
        for key in ('token_ids', 'token_start', 'token_end', 'claim_index', 'document_index', 'input_token_index'):
            assert np.array_equal(np.concatenate([t[key] for t in selected]), old[key]), (row['response_id'], key)
        diff = float(np.max(np.abs(np.concatenate([t['hidden_last'] for t in selected])-old['hidden_last'])))
    assert diff <= 2e-4, (row['response_id'], 'selected_last_hidden', diff)
    return diff


def save_answer(row, z, prob, traces, agreement, snapshot):
    pairs = [(s['claim_index'], s['document_index']) for s in row['unselected_pairs']]
    geometry = original_geometry(row['plan'], pairs)
    actual = np.asarray([t['batch_padding_length'] for t in traces], np.int32)
    assert np.array_equal(actual, geometry['original_batch_padding_length'])
    original_save = v1.save_npz
    def with_geometry(path, arrays):
        arrays.update(geometry); arrays['batch_padding_length'] = actual
        original_save(path, arrays)
    v1.save_npz = with_geometry
    try: record = v1.save_answer(row, z, prob, traces, agreement, snapshot)
    finally: v1.save_npz = original_save
    # Only small metadata is updated; large NPZ arrays are written exactly once.
    for folder in ('encoder22_features', 'claim_features'):
        path = DEST/folder/f"{row['response_id']}.json"; m = read(path)
        m['forward_geometry'] = 'Original complete all-doc claim-major/doc-major batch4; save unselected only'
        m['sequence_geometry_fields'] = list(geometry)
        m['selected_last_state_max_abs'] = agreement['selected_last_hidden_max_abs']
        write(path, m); record['files_sha256'][str(path.relative_to(DEST))] = sha(path)
    return record


def infer():
    frozen, bound = verify(deep=True)
    assert not (DEST/'inference_complete.json').exists(), 'Already complete'
    snapshot = {'preparation_freeze_sha256': sha(DEST/'preparation_freeze.json'),
        'v1_binding_freeze_sha256': sha(V1/'binding_freeze.json'), 'runner_sha256': sha(Path(__file__)),
        'device': 'cuda:0', 'dtype': 'float32', 'forward_geometry': 'Original all-doc batch4'}
    write(DEST/'inference_source_snapshot.json', snapshot, True)
    for p,h in read(v1.QA/'semantic_baseline/download_manifest.json')['files_sha256'].items():
        assert sha(v1.OLD/p) == h
    spec = importlib.util.spec_from_file_location('original_batch_unselected_forward', HERE/'run_minicheck_expansion.py')
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
    model, tok = exp.load_model(); records = []; fresh = 0; active = 0; start = time.perf_counter()
    for row in bound:
        rid = row['response_id']; pairs = row['unselected_pairs']; commit = DEST/'rows'/f'{rid}.json'
        if not pairs:
            records.append({'response_id': rid, 'partition': row['partition'], 'group_id': row['group_id'],
                'pairs': 0, 'input_tokens': 0, 'claim_tokens': 0, 'reason': 'No unselected document'})
            continue
        active += 1
        if commit.exists():
            r = read(commit)
            assert r['bound_plan_sha256'] == digest(row) and r['source_snapshot_sha256'] == digest(snapshot)
            for name,h in r['files_sha256'].items(): assert sha(DEST/name) == h
        else:
            p = row['plan']; c, d = len(p['claims']), len(p['document_chunks'])
            texts, metadata = exp.cuda.pairs(p, tok)
            assert len(texts) == c*d and [len(tok.encode(t)) for t in texts] == p['pair_lengths']
            assert [(m['claim_index'],m['document_index']) for m in metadata] == [(ci,di) for ci in range(c) for di in range(d)]
            z, prob, full = exp.probabilities_with22(model, tok, texts, metadata)
            full_geometry = original_geometry(p, [(ci,di) for ci in range(c) for di in range(d)])
            assert np.array_equal([t['batch_padding_length'] for t in full], full_geometry['original_batch_padding_length'])
            assert [len(t['full_input_ids']) for t in full] == p['pair_lengths']
            observer = fresh < 2
            if observer:
                plainz, plainp, plain = exp.cuda.probabilities(model, tok, texts, metadata)
                assert np.array_equal(z, plainz) and np.array_equal(prob, plainp)
                assert all(np.array_equal(t['hidden_last'],u['hidden_last']) for t,u in zip(full,plain))
            oldscore = read(row['reference_score_path'])
            oz = np.asarray(oldscore['logits'], np.float32).reshape(-1,2)
            op = np.asarray(oldscore['support_by_claim_document'], np.float32).ravel()
            dz, dp = float(abs(z-oz).max()), float(abs(prob-op).max())
            assert dz <= 2e-4 and dp <= 2e-5, (rid, dz, dp)
            dh = check_selected(row, full)
            flat = np.asarray([s['claim_index']*d+s['document_index'] for s in pairs], np.int64)
            traces = [full[int(j)] for j in flat]; v1.check_traces(row, traces)
            agreement = {'logit_max_abs': dz, 'support_max_abs': dp, 'selected_last_hidden_max_abs': dh,
                'claim_token_ids_and_offsets_exact': True, 'selected_all_coordinates_exact': True,
                'own_document_input_positions_checked': True, 'all_original_batch_padding_exact': True,
                'observer_exact_checked': observer, 'old_selected_final_states_not_compared_across_documents': True}
            r = save_answer(row, z[flat], prob[flat], traces, agreement, snapshot)
            write(commit, r); fresh += 1
        records.append(r)
        if active % 25 == 0: print('UNSELECTED_V2_PROGRESS', active, 956, round(time.perf_counter()-start,1), flush=True)
    assert len(records) == 3839 and sum(r['pairs'] for r in records) == 8226
    verify(deep=True)
    checks = [r['numeric_agreement'] for r in records if r['pairs']]
    agreement = {'status': 'passed', 'active_answers': len(checks),
        'max_abs_logit_difference': max(r['logit_max_abs'] for r in checks),
        'max_abs_support_difference': max(r['support_max_abs'] for r in checks),
        'max_abs_selected_last_hidden_difference': max(r['selected_last_hidden_max_abs'] for r in checks),
        'all_original_batch_geometry_exact': True}
    write(DEST/'numeric_agreement.json', agreement)
    write(DEST/'feature_manifest.json', {'status': 'complete', 'records': records, 'answers': len(records),
        'pairs': 8226, 'input_tokens': sum(r['input_tokens'] for r in records),
        'claim_tokens': sum(r['claim_tokens'] for r in records), 'partitions': {'fit':3680,'calibration':159},
        'source_snapshot_sha256': digest(snapshot), 'forward_geometry': 'Original all-doc batch4',
        'numeric_agreement_sha256': sha(DEST/'numeric_agreement.json'),
        'global_risk_labels_assigned': False, 'test_opened': False})
    write(DEST/'inference_complete.json', {'status':'complete','pairs':8226,
        'feature_manifest_sha256':sha(DEST/'feature_manifest.json'),
        'numeric_agreement_sha256':sha(DEST/'numeric_agreement.json'),
        'source_snapshot_sha256':digest(snapshot),'this_invocation_seconds':time.perf_counter()-start,
        'no_training':True,'labels_read':False,'test_opened':False})
    del model
    import torch
    torch.cuda.empty_cache()
    print('UNSELECTED_V2_COMPLETE_GPU_RELEASED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['freeze','verify','infer']); args = p.parse_args()
    if args.stage == 'freeze': freeze()
    elif args.stage == 'verify':
        _, rows = verify(); print('UNSELECTED_V2_CPU_VERIFY_PASSED', len(rows), flush=True)
    else: infer()
