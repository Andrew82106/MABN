"""Separate, frozen encoder22 backfill for ORIGINAL fit634/cal159 only.

Read-only old selected-document decisions; no new document competition, model
training, label inspection, or official-test access. GPU scheduling is external.
"""
from pathlib import Path
import importlib.util
import argparse
import json
import time
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
QA = HERE.parent
OLD = QA/'semantic_baseline/cuda_variant'
DEST = HERE/'minicheck_backfill'
spec = importlib.util.spec_from_file_location('frozen_expansion_cuda', HERE/'run_minicheck_expansion.py')
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
base = exp.base


def source_paths():
    names = [Path(__file__), HERE/'run_minicheck_expansion.py', HERE/'protocol.json', HERE/'data/export_freeze.json',
             QA/'semantic_baseline/run_semantic.py', QA/'semantic_baseline/run_semantic_cuda.py',
             QA/'semantic_baseline/download_manifest.json', OLD/'protocol.json', OLD/'plans.jsonl',
             OLD/'inference_complete.json', OLD/'claim_feature_manifest.json']
    return {str(p.resolve()): base.sha(p) for p in names}


def prepare():
    assert not (DEST/'preparation_freeze.json').exists(), 'Backfill preparation already frozen'
    DEST.mkdir(exist_ok=True)
    snap = source_paths()
    original = base.readl(OLD/'plans.jsonl')
    assert len(original) == 793
    assert sum(p['partition'] == 'fit' for p in original) == 634
    assert sum(p['partition'] == 'calibration' for p in original) == 159
    assert not ({p['group_id'] for p in original if p['partition'] == 'fit'} &
                {p['group_id'] for p in original if p['partition'] == 'calibration'})
    refs = {r['response_id']: r for r in base.read(OLD/'claim_feature_manifest.json')['records']}
    plans = []; tokens = 0; claims = 0; partitions = {}
    for p in original:
        rid = p['response_id']; score = OLD/'scores'/f'{rid}.json'
        metadata = OLD/'claim_features'/f'{rid}.json'; arr = metadata.with_suffix('.npz')
        assert base.sha(metadata) == refs[rid]['metadata_sha256'] and base.sha(arr) == refs[rid]['npz_sha256']
        m = base.read(metadata); r = base.read(score)
        assert m['plan_sha256'] == r['plan_sha256'] == base.digest(p)
        assert m['original_answer_sha256'] == p['answer_sha256'] and base.sha(score) == m['score_row_sha256']
        with np.load(arr) as z:
            chosen = z['selected_document_per_claim'].astype(int).tolist()
        probs = np.asarray(r['support_by_claim_document'], dtype=np.float32)
        assert probs.shape == (len(p['claims']), len(p['document_chunks']))
        assert chosen == probs.argmax(1).tolist() == m['selected_document_per_claim']
        lengths = np.asarray(p['pair_lengths']).reshape(probs.shape)
        selected_lengths = [int(lengths[i,d]) for i,d in enumerate(chosen)]
        original_padding = []
        for i, d in enumerate(chosen):
            flat = i*probs.shape[1]+d; a = flat//4*4
            original_padding.append(max(p['pair_lengths'][a:a+4]))
        plans.append({**p, 'original_plan_sha256': base.digest(p),
            'selected_document_per_claim': chosen, 'selected_input_lengths': selected_lengths,
            'original_batch_padding_lengths': original_padding,
            'old_score_path': str(score.resolve()), 'old_score_sha256': base.sha(score),
            'old_claim_npz_path': str(arr.resolve()), 'old_claim_npz_sha256': base.sha(arr),
            'old_claim_metadata_path': str(metadata.resolve()), 'old_claim_metadata_sha256': base.sha(metadata)})
        tokens += sum(selected_lengths); claims += len(chosen)
        partitions[p['partition']] = partitions.get(p['partition'], 0)+1
    protocol = {'version': 'qa-original-encoder22-selected-document-backfill-v1',
        'scope': 'Original fit634 and calibration159 only, explicit development-feature backfill; official test untouched',
        'checkpoint': 'lytang/MiniCheck-RoBERTa-Large', 'revision': '74c8919647e61ed0f71bc177d94f10930f090068',
        'selection': 'Use exact selected_document_per_claim from old final-state cache; do not recompute document selection',
        'forward': 'One input per original claim, selected document + eos + exact original claim; all valid input tokens',
        'device': 'cuda:0', 'dtype': 'float32', 'batch_size': 4, 'TF32': False, 'autocast': False,
        'state': 'encoder.layer[21] output[0], full valid input, flattened by answer; after layer22 before last two layers',
        'padding': 'Dynamic padding over only selected inputs, so batch geometry may differ from old all-document inference; store new and original padding lengths',
        'agreement': {'all793_old_claim_coordinates_and_token_ids_exact': True,
                      'old_selected_logit_max_abs_tolerance': 2e-4, 'old_selected_support_max_abs_tolerance': 2e-5,
                      'old_claim_final_hidden_max_abs_tolerance': 2e-4,
                      'first_two_forward_observer_vs_unobserved_logits_and_last_state': 'exact'},
        'baseline': 'Old support/risk/final-state arrays remain unchanged; new logits only serve numeric comparison',
        'no_training_or_thresholds': True, 'generator_identity_not_input': True, 'official_test_opened': False}
    base.write(DEST/'protocol.json', protocol)
    (DEST/'plans.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in plans), 'utf-8')
    storage = {'answers': len(plans), 'partitions': partitions, 'claims': claims, 'selected_input_tokens_exact': tokens,
               'hidden22_float32_bytes': tokens*1024*4, 'additional_coordinates_and_metadata': True,
               'disk_free_bytes_at_preparation': shutil.disk_usage(HERE).free,
               'new3046_not_included': True, 'old_original_files_modified': False}
    assert storage['disk_free_bytes_at_preparation'] > storage['hidden22_float32_bytes']*1.2
    base.write(DEST/'storage_estimate.json', storage)
    assert source_paths() == snap
    base.write(DEST/'preparation_freeze.json', {'status': 'prepared_waiting_for_root_gpu_authorization',
        'source_files_sha256': snap, 'files_sha256': {name: base.sha(DEST/name)
            for name in ('protocol.json', 'plans.jsonl', 'storage_estimate.json')},
        'answers': 793, 'partitions': partitions, 'no_label_reading': True, 'test_opened': False,
        'gpu_initialized': False})
    print('BACKFILL_CPU_PREPARED', json.dumps(storage), flush=True)


def verify():
    freeze = base.read(DEST/'preparation_freeze.json')
    for p, h in freeze['source_files_sha256'].items(): assert base.sha(Path(p)) == h, p
    for p, h in freeze['files_sha256'].items(): assert base.sha(DEST/p) == h, p
    plans = base.readl(DEST/'plans.jsonl'); assert len(plans) == 793
    for p in plans:
        for stem in ('old_score', 'old_claim_npz', 'old_claim_metadata'):
            assert base.sha(Path(p[stem+'_path'])) == p[stem+'_sha256']
    for rel, expected in base.read(QA/'semantic_baseline/download_manifest.json')['files_sha256'].items():
        assert base.sha(OLD/rel) == expected
    return freeze, plans


def selected_inputs(plan, tok):
    texts, metadata = [], []
    for ci, di in enumerate(plan['selected_document_per_claim']):
        c = plan['claims'][ci]; d = plan['document_chunks'][di]; prefix = d['text']+tok.eos_token
        texts.append(prefix+c['text'])
        metadata.append({'claim': c, 'claim_index': ci, 'document_index': di, 'input_claim_start': len(prefix)})
    assert [len(tok.encode(t)) for t in texts] == plan['selected_input_lengths']
    return texts, metadata


def save(plan, traces, agreement, source_digest):
    lengths = [len(t['full_input_ids']) for t in traces]
    arrays = {'hidden22': np.concatenate([t['full_hidden22'] for t in traces]),
        'input_ids': np.concatenate([t['full_input_ids'] for t in traces]),
        'attention_mask': np.concatenate([t['full_attention_mask'] for t in traces]).astype(np.int8),
        'sequence_offsets': np.r_[0, np.cumsum(lengths)].astype(np.int64),
        'claim_index': np.arange(len(traces), dtype=np.int32),
        'document_index': np.asarray(plan['selected_document_per_claim'], dtype=np.int32),
        'answer_token_start': np.concatenate([t['full_answer_token_start'] for t in traces]),
        'answer_token_end': np.concatenate([t['full_answer_token_end'] for t in traces]),
        'batch_padding_length': np.asarray([t['batch_padding_length'] for t in traces], dtype=np.int32),
        'original_selection_batch_padding_length': np.asarray(plan['original_batch_padding_lengths'], dtype=np.int32)}
    assert lengths == plan['selected_input_lengths'] and arrays['hidden22'].shape == (sum(lengths), 1024)
    path = DEST/'encoder22_features'/f"{plan['response_id']}.npz"; path.parent.mkdir(exist_ok=True)
    np.savez(path, **arrays)
    base.write(path.with_suffix('.json'), {'response_id': plan['response_id'], 'partition': plan['partition'],
        'npz_sha256': base.sha(path), 'plan_sha256': base.digest(plan),
        'original_plan_sha256': plan['original_plan_sha256'], 'source_snapshot_sha256': source_digest,
        'score_row_sha256': plan['old_score_sha256'], 'original_score_path': plan['old_score_path'],
        'original_answer_sha256': plan['answer_sha256'], 'old_claim_npz_sha256': plan['old_claim_npz_sha256'],
        'layer': 'encoder.layer[21] output[0], after22 before last2', 'dtype': 'float32', 'hidden_dimension': 1024,
        'sequences': len(traces), 'valid_input_tokens': sum(lengths), 'sequence_lengths': lengths,
        'selected_document_per_claim': plan['selected_document_per_claim'], 'selection': 'Copied old frozen decisions',
        'old_last_state_agreement': agreement, 'original_files_changed': False,
        'padding': 'Valid tokens only, actual selected-only and old all-doc padding lengths both recorded',
        'coordinates': 'Absolute original response claim offsets, otherwise -1'})
    return {'response_id': plan['response_id'], 'partition': plan['partition'],
            'valid_input_tokens': sum(lengths), 'sequences': len(traces),
            'npz_sha256': base.sha(path), 'metadata_sha256': base.sha(path.with_suffix('.json'))}


def infer():
    freeze, plans = verify()
    assert not (DEST/'inference_complete.json').exists(), 'Backfill already complete'
    snapshot = {'preparation_freeze_sha256': base.sha(DEST/'preparation_freeze.json'),
                'runner_sha256': base.sha(Path(__file__)), 'dtype': 'float32', 'device': 'cuda:0'}
    if (DEST/'inference_source_snapshot.json').exists():
        assert base.read(DEST/'inference_source_snapshot.json') == snapshot
    else: base.write(DEST/'inference_source_snapshot.json', snapshot)
    model, tok = exp.load_model(); records, checks = [], []; start = time.perf_counter()
    for i, p in enumerate(plans):
        path = DEST/'encoder22_features'/f"{p['response_id']}.npz"
        if path.exists() and path.with_suffix('.json').exists():
            m = base.read(path.with_suffix('.json'))
            assert m['npz_sha256'] == base.sha(path) and m['plan_sha256'] == base.digest(p)
            assert m['source_snapshot_sha256'] == base.digest(snapshot)
            agreement = m['old_last_state_agreement']
            record = {'response_id': p['response_id'], 'partition': p['partition'],
                      'valid_input_tokens': m['valid_input_tokens'], 'sequences': m['sequences'],
                      'npz_sha256': m['npz_sha256'], 'metadata_sha256': base.sha(path.with_suffix('.json'))}
        else:
            texts, meta = selected_inputs(p, tok)
            z, prob, traces = exp.probabilities_with22(model, tok, texts, meta)
            if i < 2:
                plainz, plainp, oldhook = exp.cuda.probabilities(model, tok, texts, meta)
                assert np.array_equal(z, plainz) and np.array_equal(prob, plainp)
                assert all(np.array_equal(t['hidden_last'], u['hidden_last']) for t, u in zip(traces, oldhook))
            old = base.read(Path(p['old_score_path'])); choices = p['selected_document_per_claim']
            oz = np.asarray(old['logits'], np.float32)[np.arange(len(choices)), choices]
            op = np.asarray(old['support_by_claim_document'], np.float32)[np.arange(len(choices)), choices]
            dz, dp = float(abs(z-oz).max()), float(abs(prob-op).max())
            with np.load(p['old_claim_npz_path']) as a:
                for key in ('token_ids', 'token_start', 'token_end', 'claim_index', 'document_index', 'input_token_index'):
                    assert np.array_equal(np.concatenate([t[key] for t in traces]), a[key]), (p['response_id'], key)
                dh = float(abs(np.concatenate([t['hidden_last'] for t in traces])-a['hidden_last']).max())
            assert dz <= 2e-4 and dp <= 2e-5 and dh <= 2e-4, (p['response_id'], dz, dp, dh)
            agreement = {'response_id': p['response_id'], 'partition': p['partition'],
                'max_abs_logit_difference': dz, 'max_abs_support_difference': dp,
                'max_abs_final_claim_hidden_difference': dh, 'claim_tokens_and_coordinates_exact': True,
                'new_forward_has_different_batch_geometry': True, 'old_scores_and_states_not_replaced': True,
                'read_only_observers_exact_checked': i < 2}
            record = save(p, traces, agreement, base.digest(snapshot))
        records.append(record); checks.append(agreement)
        if (i+1) % 25 == 0: print('ENCODER22_BACKFILL_PROGRESS', i+1, 793, round(time.perf_counter()-start, 1), flush=True)
    assert source_paths() == freeze['source_files_sha256']
    base.write(DEST/'numeric_agreement.json', {'status': 'passed', 'records': checks,
        'max_abs_logit_difference': max(r['max_abs_logit_difference'] for r in checks),
        'max_abs_support_difference': max(r['max_abs_support_difference'] for r in checks),
        'max_abs_final_claim_hidden_difference': max(r['max_abs_final_claim_hidden_difference'] for r in checks),
        'all793_coordinates_exact': True})
    base.write(DEST/'encoder22_feature_manifest.json', {'status': 'complete', 'records': records, 'answers': len(records),
        'partitions': {'fit': 634, 'calibration': 159}, 'valid_input_tokens': sum(r['valid_input_tokens'] for r in records),
        'hidden_dimension': 1024, 'dtype': 'float32', 'source_snapshot_sha256': base.digest(snapshot),
        'selection': 'Frozen old selected-document decisions, not rescored selection',
        'numeric_agreement_sha256': base.sha(DEST/'numeric_agreement.json'), 'original_files_changed': False,
        'test_opened': False})
    base.write(DEST/'inference_complete.json', {'status': 'complete', 'answers': len(records),
        'this_invocation_seconds': time.perf_counter()-start,
        'feature_manifest_sha256': base.sha(DEST/'encoder22_feature_manifest.json'),
        'numeric_agreement_sha256': base.sha(DEST/'numeric_agreement.json'),
        'no_training': True, 'test_opened': False})
    del model
    import torch
    torch.cuda.empty_cache()
    print('ENCODER22_BACKFILL_COMPLETE_GPU_RELEASED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'verify', 'infer']); a = p.parse_args()
    with base.threadpool_limits(limits=4):
        if a.stage == 'prepare': prepare()
        elif a.stage == 'verify':
            _, plans = verify(); print('BACKFILL_VERIFY_PASSED', len(plans))
        else: infer()
