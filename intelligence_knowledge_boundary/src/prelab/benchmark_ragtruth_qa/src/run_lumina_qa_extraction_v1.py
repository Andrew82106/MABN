"""Explicit public-QA LUMINA extraction. check is CPU-only; no automatic GPU queue."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback
import numpy as np
import torch

import feature_qa as q
import run_feature_qa as loader
import lumina_qa_signals as core

ROOT = q.ROOT
PREP = ROOT / 'results/lumina_qa_preparation_v1'
OUT = ROOT / 'results/lumina_qa_features_v1'
COMMON_SIGNATURE = ROOT / 'results/ghost_geometry_features_v1/signature.json'
CORE_CHECK = ROOT / 'results/lumina_qa_signals_v1/CPU_SELFCHECK.json'
INPUT = PREP / 'feature_inputs.jsonl'
VERSION = 'lumina-public-qa-seven-column-features-v1'
NAMES = core.NAMES
EXPECTED, TOKENS, ATOL = 3839, 708506, 8e-6
INPUT_KEYS = set('response_id source_id group_id partition official_split old_plan_sha256 original_prompt '
    'random_prompt original_response answer_sha256 original_prompt_sha256 random_prompt_sha256 material_sha256 '
    'donor_source_id donor_group_id donor_material_sha256 original_reference_range random_reference_range '
    'original_prefix_ids random_prefix_ids answer_token_ids original_input_ids random_input_ids '
    'original_input_ids_sha256 random_input_ids_sha256 original_answer_positions random_answer_positions '
    'original_predictor_positions random_predictor_positions response_token_offsets response_token_offsets_raw '
    'position_shift suffix_tokens_after_answer labels_used exact_original_generation_trace'.split())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def freeze(path, value):
    if path.exists():
        assert q.read(path) == value, ('Frozen definition changed', str(path))
    else:
        atomic_json(path, value)


def inputs_and_bindings():
    complete = q.read(PREP / 'preparation_complete.json')
    assert complete['status'] == 'CPU_ready_not_extracted'
    assert (complete['answers'], complete['fit'], complete['calibration']) == (EXPECTED, 3680, 159)
    assert not complete['GPU_used'] and not complete['gold_labels_read'] and not complete['official_test_opened']
    for name, digest in complete['artifacts_sha256'].items():
        assert Path(name).name == name and q.sha(PREP / name) == digest
    for path, digest in complete['source_sha256'].items():
        assert q.sha(path) == digest, path
    review = q.read(PREP / 'CPU_OUTPUT_CHECK.json')
    assert review['status'] == 'passed' and review['answers'] == EXPECTED and not review['GPU_used']
    assert review['preparation_complete_sha256'] == q.sha(PREP / 'preparation_complete.json')
    rows = [json.loads(s) for s in INPUT.read_text(encoding='utf-8').splitlines() if s]
    assert len(rows) == len({r['response_id'] for r in rows}) == EXPECTED
    assert q.digest([r['response_id'] for r in rows]) == complete['response_order_sha256']
    groups = [{r['group_id'] for r in part} for part in (rows[:3680], rows[3680:])]
    fit_sources = {r['source_id'] for r in rows[:3680]}
    assert len(groups[0]) == 615 and len(groups[1]) == 154 and not groups[0] & groups[1]
    assigned = {}
    for i, r in enumerate(rows):
        assert set(r) == INPUT_KEYS, ('Unexpected input schema', i, set(r) ^ INPUT_KEYS)
        assert r['partition'] == ('fit' if i < 3680 else 'calibration')
        assert r['official_split'] == 'train' and r['labels_used'] is False
        assert r['exact_original_generation_trace'] is False and r['suffix_tokens_after_answer'] == 0
        core.validate_input(r)
        assert q.digest(r['original_response']) == r['answer_sha256']
        assert r['donor_source_id'] in fit_sources and r['donor_source_id'] != r['source_id']
        assert r['donor_group_id'] in groups[0] and r['donor_group_id'] != r['group_id']
        assert r['donor_material_sha256'] != r['material_sha256']
        donor = (r['donor_source_id'], r['donor_group_id'], r['donor_material_sha256'])
        assert assigned.setdefault(r['source_id'], donor) == donor
        for side in ('original', 'random'):
            ids, pos = r[side + '_input_ids'], r[side + '_answer_positions']
            assert 0 < len(ids) <= 4096 and all(isinstance(x, int) and 0 <= x < 32000 for x in ids)
            assert q.digest(ids) == r[side + '_input_ids_sha256']
            assert r[side + '_predictor_positions'] == [p - 1 for p in pos]
            assert q.digest(r[side + '_prompt']) == r[side + '_prompt_sha256']
        ol, oh = r['original_reference_range']; rl, rh = r['random_reference_range']
        op, rp = r['original_prompt'], r['random_prompt']
        assert 0 <= ol < oh <= len(op) and 0 <= rl < rh <= len(rp)
        assert op[:ol] == rp[:rl] and op[oh:] == rp[rh:]
        assert q.digest(op[ol:oh]) == r['material_sha256']
        assert q.digest(rp[rl:rh]) == r['donor_material_sha256']
        assert r['position_shift'] == len(r['random_prefix_ids']) - len(r['original_prefix_ids'])
        assert all(0 <= a < b <= len(r['original_response']) for a, b in r['response_token_offsets'])
        assert all(a < b for a, b in r['response_token_offsets_raw'])
    assert sum(len(r['answer_token_ids']) for r in rows) == TOKENS
    assert sum(len(r['original_input_ids']) for r in rows) == complete['counts']['original_input_tokens'] == 2408466
    assert sum(len(r['random_input_ids']) for r in rows) == complete['counts']['random_input_tokens']
    assert sum(r['response_token_offsets_raw'][0][0] < 0 for r in rows) == complete['counts']['boundary_crossing_answers']
    return rows, complete


def signature(rows):
    # Reuse only already-verified common identity, never GHOST feature definitions or values.
    common = q.read(COMMON_SIGNATURE)
    keys = ('model', 'load_config', 'seed', 'threads', 'software', 'python', 'torch_cuda_runtime')
    shared = {k: common[k] for k in keys}
    assert shared['model']['repo'] == loader.REPO and shared['model']['revision'] == loader.REVISION
    assert shared['load_config'] == loader.LOAD_CONFIG and shared['seed'] == loader.SEED
    assert shared['threads'] == loader.THREADS and shared['python'] == platform.python_version()
    assert shared['torch_cuda_runtime'] == torch.version.cuda
    for name, version in shared['software'].items():
        assert importlib.metadata.version(name) == version, name
    download = q.read(ROOT / 'model_download_manifest.json')
    assert download['status'] == 'complete'
    for entry in download['files']:
        path = loader.MODEL / entry['filename']; asset = shared['model']['assets'][entry['filename']]
        assert path.parent.resolve() == loader.MODEL.resolve()
        assert entry['status'] == 'verified' and entry['source_hash_match']
        assert path.stat().st_size == asset['bytes'] == entry['actual_bytes']
        assert entry['actual_sha256'] == asset['sha256']
    math_check = q.read(CORE_CHECK)
    assert math_check['passed'] and not math_check['GPU_used']
    assert q.sha(core.__file__) == math_check['source_sha256']
    assert q.sha(core.PORT_PATH) == math_check['pinned_port_sha256']
    code = {str(Path(p).resolve()): q.sha(p) for p in (__file__, core.__file__, core.PORT_PATH, q.__file__, loader.__file__)}
    for path, digest in common['code_sha256'].items():
        # The model implementation and BNB arithmetic are common; do not import BNB on CPU.
        if 'site-packages' in path or 'bitsandbytes' in path:
            assert q.sha(path) == digest
            code[path] = digest
    sources = [INPUT, PREP / 'preparation_complete.json', PREP / 'CPU_OUTPUT_CHECK.json',
               PREP / 'protocol.json', PREP / 'donor_assignments.jsonl', PREP / 'source_materials.jsonl',
               CORE_CHECK, COMMON_SIGNATURE, ROOT / 'model_download_manifest.json']
    return {'version': VERSION, **shared, 'code_sha256': code,
        'source_sha256': {str(p.resolve()): q.sha(p) for p in sources},
        'model_asset_verification': 'CPU check reuses prior full-file SHA identities and download manifest and verifies current sizes. Explicit GPU commands independently verify all actual asset SHA values before load_nf4; the loader itself does not perform that hash check.',
        'input_order_sha256': q.digest([r['response_id'] for r in rows]),
        'raw_token_order_sha256': q.digest([[r['response_id'], r['answer_token_ids'], r['response_token_offsets_raw']] for r in rows]),
        'feature_names': NAMES, 'dtype': 'float32', 'top_k': 100, 'chunk_size': 16, 'lambda': .5,
        'official_commit': core.port.OFFICIAL_COMMIT, 'formula_version': core.port.FORMULA_VERSION,
        'target_timing': 'Each side answer position minus1; actual same answer IDs in both contexts.',
        'oracle_atol': ATOL, 'oracle_rtol': 0, 'same_path_repeat_exact': True,
        'labels_used': False, 'trained': False, 'test_opened': False, 'exact_original_generation_trace': False}


def axes(r):
    clipped, raw = (np.asarray(r[k], np.int64) for k in ('response_token_offsets', 'response_token_offsets_raw'))
    result = {'token_ids': np.asarray(r['answer_token_ids'], np.int64),
        'token_start': clipped[:, 0], 'token_end': clipped[:, 1],
        'token_start_raw': raw[:, 0], 'token_end_raw': raw[:, 1]}
    for side in ('original', 'random'):
        for field in ('answer_positions', 'predictor_positions'):
            result[side + '_' + field] = np.asarray(r[side + '_' + field], np.int64)
    return result


def identity(i, r, sig_hash):
    return {'signature_sha256': sig_hash, 'record_sha256': q.digest(r),
            'response_id': str(r['response_id']), 'record_index': str(i)}


def validate_values(f, length):
    assert f.dtype == np.float32 and f.shape == (length, 7) and np.isfinite(f).all()
    assert (f[:, :2] >= 0).all() and ((f[:, 3:] >= 0) & (f[:, 3:] <= 1)).all()
    assert (f[:, 3] <= f[:, 4]).all() and (f[:, 5] <= f[:, 6]).all()
    assert np.array_equal(f[:, 2], .5 * f[:, 0] - .5 * f[:, 1])


def validate_record(folder, i, r, sig_hash, repair=False):
    path = folder / f'{i:05d}.npz'
    if not path.exists():
        assert not path.with_suffix('.json').exists(), ('Orphan sidecar', str(path))
        return None
    wanted, ident = axes(r), identity(i, r, sig_hash)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == set(wanted) | set(ident) | {'lumina_features'}
        for k, v in ident.items():
            assert str(z[k].item()) == v, (path.name, k)
        for k, v in wanted.items():
            assert z[k].dtype == np.int64 and np.array_equal(z[k], v), (path.name, k)
        f = z['lumina_features']; validate_values(f, len(r['answer_token_ids']))
        minima, maxima = f.min(0).tolist(), f.max(0).tolist()
    meta = {**ident, 'file': path.name, 'npz_sha256': q.sha(path), 'feature_names': NAMES,
        'source_id': r['source_id'], 'group_id': r['group_id'], 'partition': r['partition'],
        'donor_source_id': r['donor_source_id'], 'donor_group_id': r['donor_group_id'],
        'material_sha256': r['material_sha256'], 'donor_material_sha256': r['donor_material_sha256'],
        'old_plan_sha256': r['old_plan_sha256'], 'answer_sha256': r['answer_sha256'],
        'original_input_ids_sha256': q.digest(r['original_input_ids']),
        'random_input_ids_sha256': q.digest(r['random_input_ids']),
        'original_input_tokens': len(r['original_input_ids']), 'random_input_tokens': len(r['random_input_ids']),
        'raw_answer_tokens': len(r['answer_token_ids']), 'feature_min': minima, 'feature_max': maxima,
        'labels_used': False, 'trained': False, 'test_opened': False}
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        assert q.read(sidecar) == meta, ('Changed sidecar', str(sidecar))
    elif repair:
        atomic_json(sidecar, meta)
    return meta


def save_record(folder, i, r, f, sig_hash):
    path = folder / f'{i:05d}.npz'; assert not path.exists(), ('No overwrite', str(path))
    f = np.asarray(f); validate_values(f, len(r['answer_token_ids']))
    arrays = {**axes(r), **{k: np.asarray(v) for k, v in identity(i, r, sig_hash).items()}, 'lumina_features': f}
    folder.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.npz.pending').open('wb') as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    path.with_suffix('.npz.pending').replace(path)
    return validate_record(folder, i, r, sig_hash, True)


def storage_check():
    r = {'response_id': 'tiny', 'source_id': 's', 'group_id': 'g', 'partition': 'fit',
         'donor_source_id': 'd', 'donor_group_id': 'dg', 'material_sha256': 'm', 'donor_material_sha256': 'dm',
         'old_plan_sha256': 'p', 'answer_sha256': 'a', 'answer_token_ids': [8, 9],
         'original_input_ids': [1, 2, 8, 9], 'random_input_ids': [1, 3, 4, 8, 9],
         'original_answer_positions': [2, 3], 'random_answer_positions': [3, 4],
         'original_predictor_positions': [1, 2], 'random_predictor_positions': [2, 3],
         'response_token_offsets': [[0, 1], [1, 2]], 'response_token_offsets_raw': [[-1, 1], [1, 2]]}
    f = np.array([[.4, .2, 0, .05, .5, .03, .4], [.2, .4, 0, .1, .4, .2, .3]], np.float32)
    f[:, 2] = .5 * f[:, 0] - .5 * f[:, 1]
    with tempfile.TemporaryDirectory(dir=OUT, prefix='cpu_storage_') as td:
        folder = Path(td); assert validate_record(folder, 0, r, 'sig') is None
        first = save_record(folder, 0, r, f, 'sig')
        assert first == validate_record(folder, 0, r, 'sig')
        folder.joinpath('00000.json').unlink()
        assert first == validate_record(folder, 0, r, 'sig', True)
        try:
            validate_record(folder, 0, r, 'wrong')
            raise RuntimeError('Wrong signature accepted')
        except AssertionError:
            pass
        try:
            save_record(folder, 0, r, f, 'sig')
            raise RuntimeError('Overwrite accepted')
        except AssertionError:
            pass
    return {'atomic_NPZ_axes_and_identity_roundtrip': True, 'missing_sidecar_recovery': True,
            'wrong_signature_rejected': True, 'overwrite_rejected': True,
            'two_different_prefix_axes_and_negative_boundary_preserved': True}


def runner_axis_check():
    # Exercise only this new dense-axis adapter, including16+1 projection blocks.
    # The independent kernel/IPR mathematics is covered by root's existing check.
    from transformers import LlamaConfig, LlamaForCausalLM
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4); torch.manual_seed(20261011)
    config = LlamaConfig(vocab_size=192, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).float().eval()
    answer = list(range(12, 29))
    r = {'answer_token_ids': answer, 'original_prefix_ids': [1, 5, 7, 9],
         'random_prefix_ids': [1, 41, 32, 26, 17],
         'response_token_offsets': [[i, i + 1] for i in range(17)],
         'response_token_offsets_raw': [[i, i + 1] for i in range(17)]}
    for side in ('original', 'random'):
        r[side + '_input_ids'] = r[side + '_prefix_ids'] + answer
        r[side + '_answer_positions'] = list(range(len(r[side + '_prefix_ids']), len(r[side + '_input_ids'])))
    actual, _ = core.extract(model, r); expected = dense_reference(model, r)
    error = float((actual - expected).abs().max())
    assert torch.allclose(actual, expected, atol=ATOL, rtol=0)
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'new_dense_axis_adapter': True, 'answer_tokens': 17,
            'projection_chunks': [16, 1], 'dense_max_abs_error': error,
            'shared_pinned_mathematics_not_new_independent_math_audit': True,
            'pretrained_weights_loaded': False, 'trained': False, 'GPU_used': False}


def check():
    assert not torch.cuda.is_initialized()
    rows, prepared = inputs_and_bindings(); sig = signature(rows); sh = q.digest(sig)
    longest = max(range(EXPECTED), key=lambda i: (sum(len(rows[i][s + '_input_ids']) for s in ('original', 'random')), -i))
    assert len(rows[0]['answer_token_ids']) >= 2
    probes = list(dict.fromkeys([0, longest]))
    protocol = {'version': VERSION, 'signature_sha256': sh, 'commands': ['check', 'gpu-smoke', 'extract'],
        'explicit_GPU_only': True, 'automatic_queue': False, 'smoke_indices': probes,
        'smoke_response_ids': [rows[i]['response_id'] for i in probes], 'longest_record_index': longest,
        'longest_selection': 'Maximum original_input_length + random_input_length, tie first index; no labels.',
        'counts': prepared['counts'], 'max_original_input_tokens': max(len(r['original_input_ids']) for r in rows),
        'max_random_input_tokens': max(len(r['random_input_ids']) for r in rows),
        'longest_record_lengths': [len(rows[longest][s + '_input_ids']) for s in ('original', 'random')],
        'oracle': 'Dense HF hidden_states[1:] predictor slices from complete independent original/random forwards; same pinned mathematics and16-token norm/head projections. Core mathematics already checked separately on TinyLlama.',
        'oracle_atol': ATOL, 'oracle_rtol': 0, 'repeat_exact': True,
        'causality': 'Change answer suffix in both contexts at same total shape; all7 columns for unchanged earlier tokens must remain exact. Changed token itself is excluded because selected-token likelihood explicitly depends on its identity.',
        'features': NAMES, 'feature_array': 'lumina_features float32[N,7]', 'diagnostic_only_columns': NAMES[3:],
        'baseline_formula_unchanged': 'Only .5IPR-.5MMD is official combined score; last4 likelihood columns do not alter baseline.',
        'all_tokens': 'All frozen raw answer BPE including punctuation, first boundary crossing and last token; no label/lexical filtering.',
        'resume': 'Atomic individually committed NPZ with signature/row/axes; validate all existing records, skip only valid; mismatch fails, never overwrites.',
        'storage_estimate': {'raw_7column_bytes': TOKENS * 7 * 4, 'raw_axis_bytes': TOKENS * 9 * 8,
                             'reserve_disk_GiB': 1, 'full_vocab_or_all_layer_hidden_persisted': False},
        'runtime': '7678 backbone forwards plus32*708506=22672192 layer-token vocabulary projections; actual input token total from frozen donor plans. QA runtime not yet measured; no scaling promise from Qwen R25.',
        'limitations': ['Whole fit-only donor material selection is a local fixed policy, not guaranteed semantically unrelated.',
                       'NF4/BF16 backbone and FP32 reductions; independent dense tolerance is not original fullprecision numeric identity.',
                       'Common Llama replay includes other-generator responses; not their original native internal traces.',
                       'This stage exports token signals only. Any later4BPE/window-max evaluation differs from paper response-token mean.',
                       'No original-paper dataset/accuracy reproduction or new classifier training is claimed.'],
        'GPU_used_during_check': False, 'trained': False, 'test_opened': False}
    OUT.mkdir(parents=True, exist_ok=True)
    storage = storage_check(); adapter = runner_axis_check()
    report = {'status': 'passed', 'records': EXPECTED, 'raw_answer_tokens': TOKENS,
        'signature_sha256': sh, 'plan_identity_and_two_prefix_axes_checked': True,
        'old_core_CPU_oracle_reused_not_rerun': q.read(CORE_CHECK), 'storage': storage,
        'new_runner_dense_axis_adapter': adapter,
        'model_loaded': False, 'GPU_used': False, 'trained': False, 'test_opened': False}
    freeze(OUT / 'signature.json', sig); freeze(OUT / 'protocol.json', protocol); freeze(OUT / 'CPU_CHECK.json', report)
    assert not torch.cuda.is_initialized()
    return rows, sig, protocol


@torch.inference_mode()
def dense_reference(model, row):
    """Independent hidden-axis access, identical16-row projection geometry; no hooks."""
    device = model.get_input_embeddings().weight.device
    base = core.port._backbone(model); stats = {}; ipr = None
    for side in ('original', 'random'):
        ids = torch.tensor([row[side + '_input_ids']], dtype=torch.long, device=device)
        full = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    output_attentions=False, output_hidden_states=True, return_dict=True)
        pos = torch.tensor(row[side + '_answer_positions'], device=device) - 1
        final = full.last_hidden_state[0, pos].detach().to('cpu', copy=True)
        layers = [x[0, pos].detach().to('cpu', copy=True) for x in full.hidden_states[1:]] if side == 'original' else []
        del full, ids, pos
        stats[side] = core.port.final_statistics(model, final, row['answer_token_ids'], 16)
        if side == 'original':
            ipr = core.port.ipr_from_hidden(model, layers, stats[side], 16)
        del final, layers
    a, b = stats['original'], stats['random']
    mmd = core.port.cosine_mmd_from_topk(a['top_probs'], a['top_ids'], b['top_probs'], b['top_ids'], model.get_input_embeddings(), 16)
    return torch.stack([ipr, mmd, .5 * ipr - .5 * mmd, a['answer_probs'], a['max_probs'], b['answer_probs'], b['max_probs']], 1)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def verify_runtime_assets(sig):
    # No model/CUDA use. GPU entry invokes this before loading trusted weights.
    for name, asset in sig['model']['assets'].items():
        path = loader.MODEL / name
        assert path.parent.resolve() == loader.MODEL.resolve()
        assert path.stat().st_size == asset['bytes'] and q.sha(path) == asset['sha256'], name


def gpu_smoke(rows, sig, protocol):
    path = OUT / 'GPU_SMOKE.json'; sh = q.digest(sig)
    if path.exists():
        saved = q.read(path); assert saved['status'] == 'passed' and saved['signature_sha256'] == sh
        print('Exact signature GPU smoke already passed; no model loaded.', flush=True); return
    verify_runtime_assets(sig)
    model, loaded = loader.load_nf4(); results = []
    try:
        assert len(core.port._backbone(model).layers) == 32 and model.config.hidden_size == 4096
        for i in protocol['smoke_indices']:
            r = rows[i]; before = [tuple(x._forward_hooks) for x in model.model.layers]
            tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            actual, _ = core.extract(model, r); repeated, _ = core.extract(model, r)
            assert torch.equal(actual, repeated), ('Repeat failed', i)
            reference = dense_reference(model, r); error = (actual - reference).abs().amax(0).tolist()
            assert torch.allclose(actual, reference, atol=ATOL, rtol=0), ('Dense-axis oracle failed', i, error)
            causal = None
            if i == 0:
                keep = max(1, len(r['answer_token_ids']) // 2)
                changed = dict(r, answer_token_ids=r['answer_token_ids'][:keep] +
                               [(x + 7) % model.config.vocab_size for x in r['answer_token_ids'][keep:]])
                for side in ('original', 'random'):
                    changed[side + '_input_ids'] = changed[side + '_prefix_ids'] + changed['answer_token_ids']
                other, _ = core.extract(model, changed)
                causal = float((actual[:keep] - other[:keep]).abs().max())
                assert torch.equal(actual[:keep], other[:keep]), ('Future suffix causality failed', causal)
                del other
            assert before == [tuple(x._forward_hooks) for x in model.model.layers]
            torch.cuda.synchronize()
            info = {'record_index': i, 'response_id': r['response_id'], 'record_sha256': q.digest(r),
                'original_input_tokens': len(r['original_input_ids']), 'random_input_tokens': len(r['random_input_ids']),
                'raw_answer_tokens': len(r['answer_token_ids']), 'repeat_exact': True,
                'dense_max_abs_error_by_column': error, 'future_suffix_max_abs_error': causal,
                'full_inputs_used': True, 'hooks_restored': True, 'seconds': time.perf_counter() - tick,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
            results.append(info); print(json.dumps({'LUMINA_GPU_SMOKE': info}), flush=True)
            del actual, repeated, reference
        freeze(path, {'status': 'passed', 'signature_sha256': sh, 'loaded': loaded, 'records': results,
                      'GPU_used': True, 'trained': False, 'test_opened': False})
    finally:
        del model; clear_gpu()


def all_records(rows, sh, repair=False):
    folder = OUT / 'features'; folder.mkdir(exist_ok=True)
    wanted = {f'{i:05d}.npz' for i in range(EXPECTED)}
    assert not {p.name for p in folder.glob('*.npz')} - wanted
    return [validate_record(folder, i, r, sh, repair) for i, r in enumerate(rows)]


def finish(rows, sh, metadata):
    assert len(metadata) == EXPECTED and all(x is not None for x in metadata)
    assert sum(x['raw_answer_tokens'] for x in metadata) == TOKENS
    manifest = {'status': 'complete', 'version': VERSION, 'signature_sha256': sh,
        'records': EXPECTED, 'raw_answer_tokens': TOKENS, 'feature_names': NAMES,
        'answer_order': [r['response_id'] for r in rows], 'entries': metadata,
        'all_records_validated': True, 'labels_used': False, 'trained': False, 'test_opened': False}
    freeze(OUT / 'feature_manifest.json', manifest)
    freeze(OUT / 'features_complete.json', {'status': 'complete', 'records': EXPECTED,
        'raw_answer_tokens': TOKENS, 'signature_sha256': sh, 'feature_names': NAMES,
        'all_records_validated': True, 'preparation_complete_sha256': q.sha(PREP / 'preparation_complete.json'),
        'files_sha256': {name: q.sha(OUT / name) for name in
            ('feature_manifest.json', 'signature.json', 'protocol.json', 'CPU_CHECK.json', 'GPU_SMOKE.json')},
        'no_test': True, 'trained': False})


def extract(rows, sig):
    sh = q.digest(sig); smoke = q.read(OUT / 'GPU_SMOKE.json')
    assert smoke['status'] == 'passed' and smoke['signature_sha256'] == sh
    metadata = all_records(rows, sh, True); missing = [i for i, x in enumerate(metadata) if x is None]
    if not missing:
        finish(rows, sh, metadata); print('All3839 valid; no GPU loaded.', flush=True); return
    assert not (OUT / 'features_complete.json').exists(), 'Completed cache has missing records'
    verify_runtime_assets(sig)
    model, loaded = loader.load_nf4(); tick = time.perf_counter(); timings = []
    try:
        for done, i in enumerate(missing, 1):
            r = rows[i]; before = [tuple(x._forward_hooks) for x in model.model.layers]
            values, timing = core.extract(model, r)
            assert before == [tuple(x._forward_hooks) for x in model.model.layers]
            metadata[i] = save_record(OUT / 'features', i, r, values.numpy(), sh)
            timings.append({'record_index': i, **timing}); del values
            if done == 1 or done % 25 == 0 or done == len(missing):
                progress = {'status': 'running', 'signature_sha256': sh, 'pid': os.getpid(),
                    'completed_count': EXPECTED - len(missing) + done, 'new_this_run': done,
                    'last_record_index': i, 'seconds': time.perf_counter() - tick}
                atomic_json(OUT / 'progress.json', progress); print(json.dumps(progress), flush=True)
        metadata = all_records(rows, sh, False); finish(rows, sh, metadata)
        atomic_json(OUT / 'last_run.json', {'status': 'complete', 'pid': os.getpid(),
            'resumed_verified_records': EXPECTED - len(missing), 'new_records': len(missing),
            'seconds': time.perf_counter() - tick, 'timings': timings, 'loaded': loaded,
            'GPU_used': True, 'trained': False, 'test_opened': False})
        print('LUMINA_PUBLIC_QA_COMPLETE3839', round(time.perf_counter() - tick, 2), flush=True)
    finally:
        del model; clear_gpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'gpu-smoke', 'extract')); args = parser.parse_args()
    try:
        rows, sig, protocol = check()
        if args.command == 'check':
            print(json.dumps(q.read(OUT / 'CPU_CHECK.json')), flush=True)
        elif args.command == 'gpu-smoke':
            gpu_smoke(rows, sig, protocol)
        else:
            extract(rows, sig)
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / f'FAILURE_{args.command}_{time.time_ns()}_{os.getpid()}.json',
            {'status': 'failed', 'command': args.command, 'exception': repr(exc), 'traceback': traceback.format_exc(),
             'tolerance_changed': False, 'automatic_retry': False, 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
