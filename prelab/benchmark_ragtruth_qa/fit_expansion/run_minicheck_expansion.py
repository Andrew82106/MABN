"""Infer new deduplicated fit answers only, after explicit root GPU scheduling.

No training/evaluation CLI exists. Frozen original MiniCheck functions are reused
without writing the original directory; new files live below fit_expansion.
"""
from pathlib import Path
import argparse
import importlib.util
import os
import time
import json
import numpy as np

HERE = Path(__file__).resolve().parent
QA = HERE.parent
DEST = HERE/'minicheck'
OLD = QA/'semantic_baseline/cuda_variant'
spec = importlib.util.spec_from_file_location('expansion_cuda_reference', QA/'semantic_baseline/run_semantic_cuda.py')
cuda = importlib.util.module_from_spec(spec); spec.loader.exec_module(cuda)
base = cuda.base
# The only mutable global points the helper's NEW output to this directory.
# Model/tokenizer assets still come from the unchanged old CUDA directory.
cuda.DEST = DEST
base.ROOT = OLD


def verify_freeze():
    freeze = base.read(HERE/'data/export_freeze.json')
    assert freeze['status'] == 'frozen_cpu_export_waiting_for_gpu_authorization'
    for section in ('source_files_sha256', 'output_files_sha256'):
        for name, expected in freeze[section].items(): assert base.sha(Path(name)) == expected, name
    for rel, expected in base.read(QA/'semantic_baseline/download_manifest.json')['files_sha256'].items():
        assert base.sha(OLD/rel) == expected, rel
    plans = base.readl(DEST/'new_plans.jsonl')
    assert len(plans) == freeze['new_answers'] and len({p['response_id'] for p in plans}) == len(plans)
    assert all(p['partition'] == 'fit' and p['truncated_input_tokens'] == 0 for p in plans)
    reuse = base.read(DEST/'reuse_manifest.json')
    assert reuse['answers'] == len(reuse['records']) == 634
    for r in reuse['records']:
        for kind in ('metadata', 'npz', 'score'):
            assert base.sha(Path(r[kind+'_path'])) == r[kind+'_sha256']
    return freeze, plans, reuse


def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    model = AutoModelForSequenceClassification.from_pretrained(OLD/'model', local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager').to('cuda:0').eval()
    assert model.config.hidden_size == 1024 and model.config.num_hidden_layers == 24
    assert all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters())
    return model, base.tokenizer()


def probabilities_with22(model, tok, texts, metadata):
    """Identical forward; two read-only observers, no output modification."""
    import torch
    captured = {}; logits = []; traces = []
    def last_hook(module, args, output): captured['last'] = output.last_hidden_state.detach()
    def layer_hook(module, args, output): captured['layer22'] = output[0].detach()
    h1 = model.roberta.register_forward_hook(last_hook)
    h2 = model.roberta.encoder.layer[21].register_forward_hook(layer_hook)
    try:
        for start in range(0, len(texts), 4):
            inp = tok(texts[start:start+4], padding=True, truncation=False,
                      return_offsets_mapping=True, return_tensors='pt')
            offsets = inp.pop('offset_mapping').numpy(); ids = inp['input_ids'].numpy().copy()
            masks = inp['attention_mask'].numpy().copy()
            padded = inp['input_ids'].shape[1]; assert padded <= 512
            gpu = {k: v.to('cuda:0') for k, v in inp.items()}
            with torch.inference_mode(): out = model(**gpu).logits
            logits.append(out.detach().cpu().numpy())
            last, lower = captured.pop('last'), captured.pop('layer22')
            assert last.shape == lower.shape and last.shape[-1] == 1024
            for j in range(out.shape[0]):
                meta = metadata[start+j]; prefix = meta['input_claim_start']; claim = meta['claim']
                end = prefix+len(claim['text'])
                keep = np.flatnonzero((offsets[j,:,1] > prefix)&(offsets[j,:,0] < end)&
                                     (offsets[j,:,1] > offsets[j,:,0]))
                valid = np.flatnonzero(masks[j] == 1)
                assert np.array_equal(valid, np.arange(len(valid))) and len(keep) > 0
                a = np.maximum(offsets[j,keep,0], prefix)-prefix+claim['start']
                b = np.minimum(offsets[j,keep,1], end)-prefix+claim['start']
                assert np.all((a >= claim['start'])&(b <= claim['end'])&(b > a))
                fulla = np.full(len(valid), -1, np.int64); fullb = fulla.copy()
                fulla[keep] = a; fullb[keep] = b
                top = last[j,torch.as_tensor(keep,device='cuda:0')].detach().cpu().numpy().copy()
                low = lower[j,torch.as_tensor(valid,device='cuda:0')].detach().cpu().numpy().copy()
                assert top.dtype == low.dtype == np.float32 and np.isfinite(low).all() and np.isfinite(top).all()
                traces.append({'hidden_last': top, 'token_ids': ids[j,keep], 'token_start': a, 'token_end': b,
                    'input_token_index': keep, 'claim_index': np.full(len(keep), meta['claim_index'], np.int32),
                    'document_index': np.full(len(keep), meta['document_index'], np.int32),
                    'full_hidden22': low, 'full_input_ids': ids[j,valid], 'full_attention_mask': masks[j,valid],
                    'full_answer_token_start': fulla, 'full_answer_token_end': fullb,
                    'batch_padding_length': padded})
    finally:
        h1.remove(); h2.remove()
    z = np.concatenate(logits); shifted = z-z.max(1,keepdims=True); ex = np.exp(shifted)
    return z, (ex/ex.sum(1,keepdims=True))[:,1], traces


def save22(plan, matrix, traces, score_path, digest):
    choice = matrix.argmax(1)
    selected = [traces[i*matrix.shape[1]+int(d)] for i, d in enumerate(choice)]
    lengths = [len(t['full_input_ids']) for t in selected]
    arrays = {'hidden22': np.concatenate([t['full_hidden22'] for t in selected]),
              'input_ids': np.concatenate([t['full_input_ids'] for t in selected]),
              'attention_mask': np.concatenate([t['full_attention_mask'] for t in selected]).astype(np.int8),
              'sequence_offsets': np.r_[0, np.cumsum(lengths)].astype(np.int64),
              'claim_index': np.arange(len(choice), dtype=np.int32), 'document_index': choice.astype(np.int32),
              'answer_token_start': np.concatenate([t['full_answer_token_start'] for t in selected]),
              'answer_token_end': np.concatenate([t['full_answer_token_end'] for t in selected]),
              'batch_padding_length': np.asarray([t['batch_padding_length'] for t in selected], np.int32)}
    assert arrays['hidden22'].shape == (sum(lengths), 1024) and np.all(arrays['attention_mask'] == 1)
    path = DEST/'encoder22_features'/f"{plan['response_id']}.npz"; path.parent.mkdir(exist_ok=True)
    np.savez(path, **arrays)
    base.write(path.with_suffix('.json'), {'response_id': plan['response_id'], 'partition': 'fit',
        'npz_sha256': base.sha(path), 'plan_sha256': base.digest(plan), 'source_snapshot_sha256': digest,
        'score_row_sha256': base.sha(score_path), 'original_answer_sha256': plan['answer_sha256'],
        'layer': 'encoder.layer[21] output[0], after 22nd encoder block', 'dtype': 'float32',
        'hidden_dimension': 1024, 'sequences': len(lengths), 'valid_input_tokens': sum(lengths),
        'sequence_lengths': lengths, 'selected_document_per_claim': choice.tolist(),
        'selection': 'Frozen original maximum support, first on ties; no labels',
        'padding': 'Store only valid input tokens; per-sequence original batch padding length retained',
        'coordinates': 'Absolute original response coordinates for claim-intersecting tokens, otherwise -1',
        'original_fit_backfilled': False})
    return {'response_id': plan['response_id'], 'valid_input_tokens': sum(lengths), 'sequences': len(lengths),
            'npz_sha256': base.sha(path), 'metadata_sha256': base.sha(path.with_suffix('.json'))}


def original_oracle(model, tok):
    # Select by original fit order, then filter the old mixed development plan
    # before decoding it. No calibration answer or labels are accessed.
    import re
    reuse = base.read(DEST/'reuse_manifest.json')['records'][:2]
    ids = {r['response_id'] for r in reuse}
    pat = re.compile(r'"response_id"\s*:\s*"([^"\\]+)"')
    plans = {}
    with (OLD/'plans.jsonl').open(encoding='utf-8') as f:
        for line in f:
            match = pat.findall(line)
            if len(match) != 1 or match[0] not in ids: continue
            p = json.loads(line); assert p['partition'] == 'fit'; plans[p['response_id']] = p
    assert set(plans) == ids
    checks = []
    for rid in [r['response_id'] for r in reuse]:
        p = plans[rid]; texts, meta = cuda.pairs(p, tok)
        z, prob, _ = cuda.probabilities(model, tok, texts)
        hz, hp, traces = cuda.probabilities(model, tok, texts, meta)
        assert np.array_equal(z, hz) and np.array_equal(prob, hp), 'Observer changed logits'
        nz, npb, both = probabilities_with22(model, tok, texts, meta)
        assert np.array_equal(z, nz) and np.array_equal(prob, npb), 'Encoder22 observer changed logits'
        assert all(np.array_equal(t['hidden_last'], u['hidden_last']) for t, u in zip(traces, both)), 'Encoder22 observer changed final states'
        old = base.read(OLD/'scores'/f'{rid}.json')
        oz = np.asarray(old['logits'], np.float32).reshape(-1, 2)
        op = np.asarray(old['support_by_claim_document'], np.float32).ravel()
        dz, dp = float(abs(z-oz).max()), float(abs(prob-op).max())
        assert dz <= 2e-4 and dp <= 2e-5
        matrix = prob.reshape(len(p['claims']), len(p['document_chunks']))
        chosen = matrix.argmax(1)
        arr = np.load(OLD/'claim_features'/f'{rid}.npz')
        assert np.array_equal(chosen, arr['selected_document_per_claim'])
        selected = [traces[i*matrix.shape[1]+int(d)] for i, d in enumerate(chosen)]
        for key in ('token_ids', 'token_start', 'token_end', 'claim_index', 'document_index', 'input_token_index'):
            assert np.array_equal(np.concatenate([t[key] for t in selected]), arr[key]), key
        dh = float(abs(np.concatenate([t['hidden_last'] for t in selected])-arr['hidden_last']).max())
        checks.append({'response_id': rid, 'max_abs_logit_difference': dz,
                       'max_abs_support_difference': dp, 'max_abs_hidden_difference_descriptive': dh,
                       'all_coordinates_and_selected_document_exact': True, 'observer_logits_exact': True,
                       'encoder22_observer_logits_and_last_states_exact': True})
    base.write(DEST/'numeric_agreement.json', {'status': 'passed', 'records': checks, 'labels_used': False,
               'old_prediction_rows_reused_not_replaced': True})


def infer():
    freeze, plans, reuse = verify_freeze()
    assert not (DEST/'inference_complete.json').exists(), 'Already complete; no repeat inference'
    snapshot = {'export_freeze_sha256': base.sha(HERE/'data/export_freeze.json'),
                'runner_sha256': base.sha(Path(__file__)), 'protocol_sha256': base.sha(HERE/'protocol.json'),
                'new_plans_sha256': base.sha(DEST/'new_plans.jsonl'),
                'old_reuse_manifest_sha256': base.sha(DEST/'reuse_manifest.json'),
                'device': 'cuda:0', 'dtype': 'float32', 'no_training_or_test': True}
    if (DEST/'inference_source_snapshot.json').exists():
        assert base.read(DEST/'inference_source_snapshot.json') == snapshot
    else: base.write(DEST/'inference_source_snapshot.json', snapshot)
    model, tok = load_model(); original_oracle(model, tok)
    (DEST/'scores').mkdir(exist_ok=True)
    records, feature_records, encoder22_records = [], [], []
    start = time.perf_counter()
    for i, p in enumerate(plans):
        path = DEST/'scores'/f"{p['response_id']}.json"
        target = DEST/'claim_features'/f"{p['response_id']}.npz"
        lower = DEST/'encoder22_features'/f"{p['response_id']}.npz"
        if all(x.exists() for x in (path, target, target.with_suffix('.json'), lower, lower.with_suffix('.json'))):
            r = base.read(path); fm = base.read(target.with_suffix('.json'))
            assert r['plan_sha256'] == base.digest(p) and r['source_snapshot_sha256'] == base.digest(snapshot)
            assert r['device'] == 'cuda:0' and fm['npz_sha256'] == base.sha(target)
            assert fm['score_row_sha256'] == base.sha(path)
            feature_records.append({'response_id': p['response_id'], 'tokens': fm['tokens'],
                                    'npz_sha256': fm['npz_sha256'], 'metadata_sha256': base.sha(target.with_suffix('.json'))})
            lm = base.read(lower.with_suffix('.json'))
            assert lm['npz_sha256'] == base.sha(lower) and lm['score_row_sha256'] == base.sha(path)
            encoder22_records.append({'response_id': p['response_id'], 'valid_input_tokens': lm['valid_input_tokens'],
                'sequences': lm['sequences'], 'npz_sha256': lm['npz_sha256'], 'metadata_sha256': base.sha(lower.with_suffix('.json'))})
        else:
            texts, meta = cuda.pairs(p, tok)
            tick = time.perf_counter(); z, prob, traces = probabilities_with22(model, tok, texts, meta)
            matrix = prob.reshape(len(p['claims']), len(p['document_chunks']))
            r = {'response_id': p['response_id'], 'partition': 'fit', 'plan_sha256': base.digest(p),
                 'source_snapshot_sha256': base.digest(snapshot),
                 'logits': z.reshape(len(p['claims']), len(p['document_chunks']), 2).tolist(),
                 'support_by_claim_document': matrix.tolist(), 'max_support': matrix.max(1).tolist(),
                 'claim_risk': (1-matrix.max(1).astype(np.float64)).tolist(),
                 'seconds': time.perf_counter()-tick, 'truncated_input_tokens': 0, 'device': 'cuda:0'}
            base.write(path, r)
            top = [{k: t[k] for k in ('hidden_last', 'token_ids', 'token_start', 'token_end',
                                      'input_token_index', 'claim_index', 'document_index')} for t in traces]
            feature_records.append(cuda.save_trace(p, matrix, top, path, base.digest(snapshot)))
            encoder22_records.append(save22(p, matrix, traces, path, base.digest(snapshot)))
        records.append({'response_id': p['response_id'], 'file_sha256': base.sha(path)})
        if (i+1) % 50 == 0:
            print('EXPANDED_MINICHECK_PROGRESS', i+1, len(plans), 'seconds', round(time.perf_counter()-start, 1), flush=True)
    for name, expected in freeze['source_files_sha256'].items(): assert base.sha(Path(name)) == expected
    base.write(DEST/'claim_feature_manifest.json', {'status': 'complete', 'records': feature_records,
        'answers': len(feature_records), 'tokens': sum(r['tokens'] for r in feature_records),
        'hidden_dimension': 1024, 'dtype': 'float32', 'source_snapshot_sha256': base.digest(snapshot),
        'selection': 'Official maximum-support document, first on ties; no labels', 'uncovered_nonwhitespace_chars': 0,
        'old_fit_reuse_manifest_sha256': base.sha(DEST/'reuse_manifest.json'),
        'old_fit_reused_answers': 634, 'combined_fit_answers': 634+len(feature_records), 'test_opened': False})
    base.write(DEST/'encoder22_feature_manifest.json', {'status': 'complete', 'records': encoder22_records,
        'answers': len(encoder22_records), 'valid_input_tokens': sum(r['valid_input_tokens'] for r in encoder22_records),
        'hidden_dimension': 1024, 'dtype': 'float32', 'source_snapshot_sha256': base.digest(snapshot),
        'selection': 'Original support-max document per claim', 'old_fit_backfilled': False,
        'no_model_training': True, 'test_opened': False})
    base.write(DEST/'inference_complete.json', {'status': 'complete', 'rows': len(records), 'records': records,
        'this_invocation_seconds': time.perf_counter()-start, 'source_snapshot_sha256': base.digest(snapshot),
        'sum_saved_inference_seconds': sum(base.read(DEST/'scores'/f"{p['response_id']}.json")['seconds'] for p in plans),
        'numeric_agreement_sha256': base.sha(DEST/'numeric_agreement.json'),
        'claim_feature_manifest_sha256': base.sha(DEST/'claim_feature_manifest.json'),
        'encoder22_feature_manifest_sha256': base.sha(DEST/'encoder22_feature_manifest.json'),
        'old_fit_reused': 634, 'all_new_rows_cuda_float32': True, 'model_trained': False,
        'test_opened': False, 'calibration_untouched': True})
    del model
    import torch
    torch.cuda.empty_cache()
    print('EXPANDED_MINICHECK_COMPLETE_GPU_RELEASED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['verify', 'infer'])
    args = parser.parse_args()
    with base.threadpool_limits(limits=4):
        if args.stage == 'verify':
            _, plans, reuse = verify_freeze()
            print('CPU_VERIFY_PASSED', len(plans), 'new;', len(reuse['records']), 'reused')
        else: infer()
