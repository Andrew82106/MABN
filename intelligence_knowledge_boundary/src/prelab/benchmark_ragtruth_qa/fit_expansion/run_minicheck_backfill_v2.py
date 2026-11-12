"""Geometry-preserving original793 recovery; old failure directory is immutable."""
from pathlib import Path
import importlib.util
import argparse
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
QA = HERE.parent
OLD = QA/'semantic_baseline/cuda_variant'
DEST = HERE/'minicheck_backfill_v2'
spec = importlib.util.spec_from_file_location('backfill_v1_helpers', HERE/'run_minicheck_backfill.py')
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
exp, base = bf.exp, bf.base
bf.DEST = DEST  # Only this new process's helper output directory; no old writes.


def source_paths():
    names = [Path(__file__), HERE/'run_minicheck_backfill.py', HERE/'run_minicheck_expansion.py',
        HERE/'minicheck_backfill/plans.jsonl', HERE/'minicheck_backfill/preparation_freeze.json',
        HERE/'minicheck_backfill/FAILURE_REPORT.json', OLD/'plans.jsonl', OLD/'protocol.json',
        OLD/'claim_feature_manifest.json', OLD/'inference_complete.json',
        QA/'semantic_baseline/run_semantic.py', QA/'semantic_baseline/run_semantic_cuda.py',
        QA/'semantic_baseline/download_manifest.json']
    return {str(p.resolve()): base.sha(p) for p in names}


def prepare():
    assert not (DEST/'preparation_freeze.json').exists(), 'Already frozen'
    DEST.mkdir(exist_ok=True); source = source_paths()
    source_plans = HERE/'minicheck_backfill/plans.jsonl'
    oldfreeze = base.read(HERE/'minicheck_backfill/preparation_freeze.json')
    assert base.sha(source_plans) == oldfreeze['files_sha256']['plans.jsonl']
    plans = base.readl(source_plans)
    assert len(plans) == 793 and sum(p['partition'] == 'fit' for p in plans) == 634
    assert sum(p['partition'] == 'calibration' for p in plans) == 159
    assert {'13953', '16023', '16029'} <= {p['response_id'] for p in plans}
    protocol = {'version': 'qa-original793-encoder22-all-doc-geometry-recovery-v2',
        'scope': 'Original634fit+159cal; preserve old failed425 and rebuild all793 in separate v2 directory',
        'change': 'Exact original full all-document claim-major/document-major batch4 forward; save only old selected-document rows',
        'reason': 'Original selected-only backfill failed hidden absolute tolerance on13953; no tolerance relaxation',
        'checkpoint': 'lytang/MiniCheck-RoBERTa-Large', 'revision': '74c8919647e61ed0f71bc177d94f10930f090068',
        'device': 'cuda:0', 'dtype': 'float32', 'batch_size': 4, 'TF32': False, 'autocast': False,
        'geometry': 'flat=claim_index*D+document_index; batch_index=flat//4; row=flat%4; batch_size=min(4,C*D-batch_index*4); padding=max original lengths in that exact batch',
        'numeric_limits': {'logits_max_abs': 2e-4, 'support_max_abs': 2e-5, 'final_claim_hidden_max_abs': 2e-4},
        'diagnosis_before_rebuild': ['13953', '16023', '16029'],
        'diagnostic_comparison': 'Record original all-doc and former selected-only variants against immutable old states. Only full all-doc must pass before rebuilding; selected-only failure is diagnostic, never used as a feature.',
        'observers': 'First three anchors compare added layer22 hook to original last-only hook: logits/support/last state exact',
        'reuse_old425': False, 'selection': 'Unchanged frozen old selected_document_per_claim; do not select from recomputed scores',
        'output': 'Same flattened full-input encoder22 schema, plus original batch index/row/size/padding per sequence',
        'original_files_changed': False, 'model_trained': False, 'test_opened': False}
    (DEST/'plans.jsonl').write_bytes(source_plans.read_bytes())
    base.write(DEST/'protocol.json', protocol)
    assert source_paths() == source
    base.write(DEST/'preparation_freeze.json', {'status': 'frozen_recovery_before_gpu',
        'source_files_sha256': source, 'files_sha256': {n: base.sha(DEST/n) for n in ('plans.jsonl', 'protocol.json')},
        'answers': 793, 'partitions': {'fit': 634, 'calibration': 159}, 'no_tolerance_relaxation': True})
    print('BACKFILL_V2_CPU_FROZEN', base.sha(DEST/'preparation_freeze.json'), flush=True)


def verify():
    frozen = base.read(DEST/'preparation_freeze.json')
    for path, h in frozen['source_files_sha256'].items(): assert base.sha(Path(path)) == h, path
    for name, h in frozen['files_sha256'].items(): assert base.sha(DEST/name) == h, name
    plans = base.readl(DEST/'plans.jsonl'); assert len(plans) == 793
    for p in plans:
        for k in ('old_score', 'old_claim_npz', 'old_claim_metadata'):
            assert base.sha(Path(p[k+'_path'])) == p[k+'_sha256']
    for rel, h in base.read(QA/'semantic_baseline/download_manifest.json')['files_sha256'].items():
        assert base.sha(OLD/rel) == h
    return frozen, plans


def compare(p, zs, ps, selected):
    old = base.read(Path(p['old_score_path'])); d = p['selected_document_per_claim']
    oz = np.asarray(old['logits'], np.float32)[np.arange(len(d)), d]
    op = np.asarray(old['support_by_claim_document'], np.float32)[np.arange(len(d)), d]
    dz, dp = float(abs(zs-oz).max()), float(abs(ps-op).max())
    with np.load(p['old_claim_npz_path']) as a:
        for k in ('token_ids', 'token_start', 'token_end', 'claim_index', 'document_index', 'input_token_index'):
            assert np.array_equal(np.concatenate([t[k] for t in selected]), a[k]), (p['response_id'], k)
        dh = float(abs(np.concatenate([t['hidden_last'] for t in selected])-a['hidden_last']).max())
    return {'response_id': p['response_id'], 'partition': p['partition'], 'max_abs_logit_difference': dz,
        'max_abs_support_difference': dp, 'max_abs_final_claim_hidden_difference': dh,
        'claim_tokens_and_coordinates_exact': True,
        'within_original_limits': dz <= 2e-4 and dp <= 2e-5 and dh <= 2e-4}


def full_forward(model, tok, p, observer=False):
    texts, meta = exp.cuda.pairs(p, tok)
    assert [len(tok.encode(t)) for t in texts] == p['pair_lengths']
    z, prob, traces = exp.probabilities_with22(model, tok, texts, meta)
    if observer:
        rz, rp, rt = exp.cuda.probabilities(model, tok, texts, meta)
        assert np.array_equal(z, rz) and np.array_equal(prob, rp)
        assert all(np.array_equal(t['hidden_last'], u['hidden_last']) for t, u in zip(traces, rt))
    C, D = len(p['claims']), len(p['document_chunks']); assert len(traces) == C*D
    for flat, t in enumerate(traces):
        bs = flat//4*4
        assert t['batch_padding_length'] == max(p['pair_lengths'][bs:bs+4])
        assert len(t['full_input_ids']) == p['pair_lengths'][flat]
    choices = p['selected_document_per_claim']; indices = [ci*D+di for ci, di in enumerate(choices)]
    selected = [traces[j] for j in indices]
    agreement = compare(p, z[indices], prob[indices], selected)
    agreement.update({'replay_geometry': 'original_all_doc_batch4_exact_order_and_padding',
                      'old_scores_and_states_not_replaced': True, 'read_only_observers_exact_checked': observer})
    assert agreement['within_original_limits'], agreement
    return selected, agreement


def diagnose(model, tok, plans):
    by = {p['response_id']: p for p in plans}; checks = []
    for rid in ('13953', '16023', '16029'):
        p = by[rid]
        texts, meta = bf.selected_inputs(p, tok)
        z, prob, ts = exp.probabilities_with22(model, tok, texts, meta)
        former = compare(p, z, prob, ts)
        selected, full = full_forward(model, tok, p, observer=True)
        checks.append({'response_id': rid, 'former_selected_only': former, 'original_all_doc': full,
            'former_padding': [t['batch_padding_length'] for t in ts],
            'original_padding': [t['batch_padding_length'] for t in selected]})
    result = {'status': 'passed_all_doc_anchors_before_rebuild', 'records': checks,
        'failure_case_selected_only_exceeds_original_gate': not checks[0]['former_selected_only']['within_original_limits'],
        'original_limits_unchanged': True, 'no_prediction_performance_or_gold_labels_used': True}
    base.write(DEST/'GEOMETRY_DIAGNOSIS.json', result)
    print('BACKFILL_V2_GEOMETRY_GATE_PASSED', json.dumps(result), flush=True)


def save(p, traces, agreement, sd):
    path = DEST/'encoder22_features'/f"{p['response_id']}.npz"
    path.parent.mkdir(exist_ok=True)
    lengths = [len(t['full_input_ids']) for t in traces]
    arrays = {'hidden22': np.concatenate([t['full_hidden22'] for t in traces]),
        'input_ids': np.concatenate([t['full_input_ids'] for t in traces]),
        'attention_mask': np.concatenate([t['full_attention_mask'] for t in traces]).astype(np.int8),
        'sequence_offsets': np.r_[0, np.cumsum(lengths)].astype(np.int64),
        'claim_index': np.arange(len(traces), dtype=np.int32),
        'document_index': np.asarray(p['selected_document_per_claim'], dtype=np.int32),
        'answer_token_start': np.concatenate([t['full_answer_token_start'] for t in traces]),
        'answer_token_end': np.concatenate([t['full_answer_token_end'] for t in traces]),
        'batch_padding_length': np.asarray([t['batch_padding_length'] for t in traces], dtype=np.int32),
        'original_selection_batch_padding_length': np.asarray(p['original_batch_padding_lengths'], dtype=np.int32)}
    C, D = len(p['claims']), len(p['document_chunks'])
    flat = np.arange(C)*D+np.asarray(p['selected_document_per_claim'])
    arrays['original_batch_index'] = (flat//4).astype(np.int32)
    arrays['original_batch_row'] = (flat%4).astype(np.int32)
    arrays['original_batch_size'] = np.minimum(4, C*D-(flat//4)*4).astype(np.int32)
    arrays['original_batch_padding_length'] = np.asarray(p['original_batch_padding_lengths'], np.int32)
    assert np.array_equal(arrays['batch_padding_length'], arrays['original_batch_padding_length'])
    assert lengths == p['selected_input_lengths'] and arrays['hidden22'].shape == (sum(lengths), 1024)
    np.savez(path, **arrays)
    meta = {'response_id': p['response_id'], 'partition': p['partition'], 'npz_sha256': base.sha(path),
        'plan_sha256': base.digest(p), 'original_plan_sha256': p['original_plan_sha256'],
        'source_snapshot_sha256': sd, 'score_row_sha256': p['old_score_sha256'],
        'original_score_path': p['old_score_path'], 'original_answer_sha256': p['answer_sha256'],
        'old_claim_npz_sha256': p['old_claim_npz_sha256'], 'dtype': 'float32', 'hidden_dimension': 1024,
        'layer': 'encoder.layer[21] output[0], after22 before last2', 'sequences': len(traces),
        'valid_input_tokens': sum(lengths), 'sequence_lengths': lengths,
        'selected_document_per_claim': p['selected_document_per_claim'],
        'selection': 'Copied old frozen decisions', 'old_last_state_agreement': agreement,
        'padding': 'Exact original all-doc batch4 geometry; selected rows saved only',
        'coordinates': 'Absolute original response claim offsets, otherwise -1',
        'recovery_version': 'v2_all_doc_geometry', 'old_failed_directory_preserved': True}
    base.write(path.with_suffix('.json'), meta)
    return {'response_id': p['response_id'], 'partition': p['partition'], 'valid_input_tokens': sum(lengths),
            'sequences': len(traces), 'npz_sha256': meta['npz_sha256'], 'metadata_sha256': base.sha(path.with_suffix('.json'))}


def infer():
    frozen, plans = verify()
    assert not (DEST/'inference_complete.json').exists(), 'Already complete'
    snap = {'preparation_freeze_sha256': base.sha(DEST/'preparation_freeze.json'),
            'runner_sha256': base.sha(Path(__file__)), 'device': 'cuda:0', 'dtype': 'float32'}
    assert not (DEST/'inference_source_snapshot.json').exists(), 'No silent retry of recovery'
    base.write(DEST/'inference_source_snapshot.json', snap)
    model, tok = exp.load_model(); diagnose(model, tok, plans)
    records, checks = [], []; start = time.perf_counter()
    for i, p in enumerate(plans):
        traces, agreement = full_forward(model, tok, p)
        records.append(save(p, traces, agreement, base.digest(snap))); checks.append(agreement)
        if (i+1) % 25 == 0: print('BACKFILL_V2_PROGRESS', i+1, 793, round(time.perf_counter()-start, 1), flush=True)
    assert source_paths() == frozen['source_files_sha256']
    numeric = {'status': 'passed', 'records': checks, 'all793_coordinates_exact': True,
        'max_abs_logit_difference': max(r['max_abs_logit_difference'] for r in checks),
        'max_abs_support_difference': max(r['max_abs_support_difference'] for r in checks),
        'max_abs_final_claim_hidden_difference': max(r['max_abs_final_claim_hidden_difference'] for r in checks)}
    base.write(DEST/'numeric_agreement.json', numeric)
    base.write(DEST/'encoder22_feature_manifest.json', {'status': 'complete', 'records': records,
        'answers': 793, 'partitions': {'fit': 634, 'calibration': 159},
        'valid_input_tokens': sum(r['valid_input_tokens'] for r in records), 'dtype': 'float32', 'hidden_dimension': 1024,
        'source_snapshot_sha256': base.digest(snap), 'geometry': 'Original all-doc order/batch4/padding; selected rows only',
        'numeric_agreement_sha256': base.sha(DEST/'numeric_agreement.json'), 'original_files_changed': False,
        'test_opened': False})
    base.write(DEST/'inference_complete.json', {'status': 'complete', 'answers': 793,
        'this_invocation_seconds': time.perf_counter()-start,
        'feature_manifest_sha256': base.sha(DEST/'encoder22_feature_manifest.json'),
        'numeric_agreement_sha256': base.sha(DEST/'numeric_agreement.json'),
        'geometry_diagnosis_sha256': base.sha(DEST/'GEOMETRY_DIAGNOSIS.json'), 'model_trained': False,
        'test_opened': False, 'original_failed_directory_unchanged': True})
    del model
    import torch
    torch.cuda.empty_cache()
    print('BACKFILL_V2_COMPLETE_GPU_RELEASED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'verify', 'infer']); args = parser.parse_args()
    with base.threadpool_limits(limits=4):
        if args.stage == 'prepare': prepare()
        elif args.stage == 'verify': verify(); print('BACKFILL_V2_VERIFY_PASSED')
        else: infer()
