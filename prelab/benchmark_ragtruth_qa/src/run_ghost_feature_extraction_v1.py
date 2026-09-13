"""Explicit GHOST-adaptation extraction: check is CPU-only; GPU is never queued.

Only the frozen 3839 label-free feature-input records are consumed. Production
uses the existing NF4 loader and the unchanged ghost_geometry primitives.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import feature_qa as q
import ghost_geometry as ghost
import run_feature_qa as loader

ROOT = q.ROOT
PREP = ROOT / 'results/ghost_geometry_preparation_v1'
OUT = ROOT / 'results/ghost_geometry_features_v1'
INPUT = PREP / 'feature_inputs.jsonl'
VERSION = 'ghost-geometry-features-v1'
FEATURES = ['adjacent_layer_cosine_change', 'layer_to_final_cosine',
            'top10_normalized_entropy', 'top10_unweighted_embedding_divergence']
FIRST, LAST, LOGIT_BATCH, EXPECTED_ROWS, EXPECTED_TOKENS = 3, 29, 16, 3839, 708506
# Both implementations use FP32 cosine/probability reductions over the SAME
# complete BF16 forward and lm_head chunk geometry. No tolerance is fitted.
ORACLE_ATOL = 8e-6
INPUT_KEYS = {'response_id', 'source_id', 'group_id', 'partition', 'input_ids',
              'answer_token_positions', 'answer_token_ids', 'response_token_offsets',
              'response_token_offsets_raw', 'old_plan_sha256', 'answer_sha256'}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def frozen_json(path, value):
    if path.exists():
        assert q.read(path) == value, ('Frozen definition changed', str(path))
    else:
        atomic_json(path, value)


def inputs_and_bindings():
    complete = q.read(PREP / 'preparation_complete.json')
    assert complete['status'] == 'CPU_ready_not_GPU_extracted'
    assert complete['answers'] == EXPECTED_ROWS and not complete['official_test_opened']
    for name, digest in complete['source_sha256'].items():
        assert q.sha(Path(name)) == digest, ('Preparation binding changed', name)
    review = q.read(PREP / 'INDEPENDENT_REVIEW.json')
    assert review['status'] == 'passed' and not review['GPU_used']
    for name, digest in review['source_sha256'].items():
        assert q.sha(Path(name)) == digest
    rows = [json.loads(s) for s in INPUT.read_text(encoding='utf-8').splitlines() if s]
    assert len(rows) == EXPECTED_ROWS and len({str(r['response_id']) for r in rows}) == EXPECTED_ROWS
    for i, r in enumerate(rows):
        assert set(r) == INPUT_KEYS
        assert r['partition'] == ('fit' if i < 3680 else 'calibration')
        ids, pos, answer = r['input_ids'], r['answer_token_positions'], r['answer_token_ids']
        assert 0 < len(ids) <= 4096 and all(isinstance(x, int) and 0 <= x < 32000 for x in ids)
        assert len(pos) > 0 and pos == list(range(pos[0], pos[-1] + 1)) and 0 < pos[0] <= pos[-1] < len(ids)
        assert [ids[j] for j in pos] == answer
        clipped, raw = r['response_token_offsets'], r['response_token_offsets_raw']
        assert len(pos) == len(clipped) == len(raw)
        assert all(0 <= a < b for a, b in clipped)
        assert all(a < b for a, b in raw)
    fit = {r['group_id'] for r in rows[:3680]}
    cal = {r['group_id'] for r in rows[3680:]}
    assert len(fit) == 615 and len(cal) == 154 and fit.isdisjoint(cal)
    assert sum(len(r['answer_token_ids']) for r in rows) == EXPECTED_TOKENS
    assert max(len(r['input_ids']) for r in rows) == 1232
    return rows


def signature(rows):
    # Read only local assets; a checkpoint is never loaded by check.
    download = q.read(ROOT / 'model_download_manifest.json')
    assert download['status'] == 'complete'
    assert download['repo_id'] == loader.REPO and download['revision'] == loader.REVISION
    assets = {}
    for entry in download['files']:
        path = loader.MODEL / entry['filename']
        assert path.parent.resolve() == loader.MODEL.resolve()
        assert entry['status'] == 'verified' and entry['source_hash_match']
        assert path.stat().st_size == entry['actual_bytes'] == entry['expected_bytes']
        digest = q.sha(path)
        assert digest == entry['actual_sha256']
        if entry.get('expected_lfs_sha256'):
            assert digest == entry['expected_lfs_sha256']
        assets[entry['filename']] = {'bytes': path.stat().st_size, 'sha256': digest}
    from transformers.models.llama import modeling_llama
    # Importing bitsandbytes itself can initialize CUDA during backend probing.
    # Hash installed source by distribution metadata without executing it.
    bnb_distribution = importlib.metadata.distribution('bitsandbytes')
    code = [Path(__file__), Path(ghost.__file__), Path(loader.__file__), Path(q.__file__),
            Path(inspect.getfile(modeling_llama)),
            Path(bnb_distribution.locate_file('bitsandbytes/nn/modules.py')),
            Path(bnb_distribution.locate_file('bitsandbytes/functional.py'))]
    bindings = [INPUT, PREP / 'preparation_complete.json', PREP / 'INDEPENDENT_REVIEW.json',
                PREP / 'protocol.json', ROOT / 'model_download_manifest.json']
    return {'version': VERSION, 'code_sha256': {str(p.resolve()): q.sha(p) for p in code},
            'source_sha256': {str(p.resolve()): q.sha(p) for p in bindings},
            'model': {'repo': loader.REPO, 'revision': loader.REVISION, 'assets': assets},
            'load_config': loader.LOAD_CONFIG, 'seed': loader.SEED, 'threads': loader.THREADS,
            'software': {k: importlib.metadata.version(k) for k in
                         ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'numpy', 'safetensors', 'tokenizers')},
            'python': platform.python_version(), 'torch_cuda_runtime': torch.version.cuda,
            'input_order_sha256': q.digest([r['response_id'] for r in rows]),
            'raw_token_order_sha256': q.digest([[r['response_id'], r['answer_token_ids'], r['response_token_offsets_raw']] for r in rows]),
            'features': FEATURES, 'first_state': FIRST, 'last_state': LAST, 'logit_batch': LOGIT_BATCH,
            'dtype': 'float32', 'top_k': 10, 'target_timing': 'answer_token_position - 1',
            'oracle_atol_fixed_before_GPU': ORACLE_ATOL, 'same_path_repeat_exact': True,
            'labels_used': False, 'test_opened': False, 'trained': False,
            'exact_original_generation_trace': False}


def axes(r):
    clipped, raw = np.asarray(r['response_token_offsets'], dtype=np.int64), np.asarray(r['response_token_offsets_raw'], dtype=np.int64)
    return {'token_ids': np.asarray(r['answer_token_ids'], dtype=np.int64),
            'answer_token_positions': np.asarray(r['answer_token_positions'], dtype=np.int64),
            'predictor_positions': np.asarray(r['answer_token_positions'], dtype=np.int64) - 1,
            'token_start': clipped[:, 0], 'token_end': clipped[:, 1],
            'token_start_raw': raw[:, 0], 'token_end_raw': raw[:, 1]}


def identity(i, r, sig_hash):
    return {'signature_sha256': sig_hash, 'record_sha256': q.digest(r),
            'response_id': str(r['response_id']), 'record_index': str(i)}


def save_record(folder, i, r, features, sig_hash):
    path = folder / f'{i:05d}.npz'
    assert not path.exists(), ('Refuse to overwrite an existing record', str(path))
    arrays = axes(r)
    arrays.update({k: np.asarray(v) for k, v in identity(i, r, sig_hash).items()})
    arrays['ghost_features'] = np.asarray(features)
    assert arrays['ghost_features'].dtype == np.float32
    assert arrays['ghost_features'].shape == (len(r['answer_token_ids']), 4)
    assert np.isfinite(arrays['ghost_features']).all()
    folder.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.npz.pending')
    with pending.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    return validate_record(folder, i, r, sig_hash, repair_sidecar=True)


def validate_record(folder, i, r, sig_hash, repair_sidecar=False):
    path = folder / f'{i:05d}.npz'
    if not path.exists():
        assert not path.with_suffix('.json').exists(), ('Sidecar exists without NPZ', str(path))
        return None
    expected_axes, expected_identity = axes(r), identity(i, r, sig_hash)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(expected_axes) | set(expected_identity) | {'ghost_features'}
        for name, wanted in expected_identity.items():
            assert str(saved[name].item()) == wanted, (path.name, name)
        for name, wanted in expected_axes.items():
            assert saved[name].dtype == np.int64 and np.array_equal(saved[name], wanted), (path.name, name)
        f = saved['ghost_features']
        assert f.dtype == np.float32 and f.shape == (len(r['answer_token_ids']), 4) and np.isfinite(f).all()
        minima, maxima = f.min(0).tolist(), f.max(0).tolist()
    meta = {**expected_identity, 'file': path.name, 'npz_sha256': q.sha(path),
            'input_ids_sha256': q.digest(r['input_ids']), 'answer_sha256': r['answer_sha256'],
            'old_plan_sha256': r['old_plan_sha256'], 'partition': r['partition'],
            'source_id': r['source_id'], 'group_id': r['group_id'],
            'input_tokens': len(r['input_ids']), 'raw_answer_tokens': len(r['answer_token_ids']),
            'feature_names': FEATURES, 'feature_min': minima, 'feature_max': maxima,
            'labels_used': False, 'trained': False}
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        assert q.read(sidecar) == meta, ('Sidecar changed', str(sidecar))
    elif repair_sidecar:
        # A complete NPZ contains its own signature, row identity and all axes,
        # so an interruption between NPZ and sidecar commits is recoverable.
        atomic_json(sidecar, meta)
    return meta


def storage_selfcheck():
    row = {'response_id': 'synthetic', 'source_id': 'synthetic', 'group_id': 'synthetic', 'partition': 'fit',
           'input_ids': [1, 4, 9, 6], 'answer_token_ids': [9, 6], 'answer_token_positions': [2, 3],
           'response_token_offsets': [[0, 1], [1, 2]], 'response_token_offsets_raw': [[-1, 1], [1, 2]],
           'answer_sha256': 'synthetic', 'old_plan_sha256': 'synthetic'}
    with tempfile.TemporaryDirectory(prefix='storage_check_', dir=OUT) as directory:
        folder = Path(directory)
        f = np.asarray([[.1, .2, .3, .4], [.4, .3, .2, .1]], dtype=np.float32)
        assert validate_record(folder, 0, row, 'synthetic') is None
        first = save_record(folder, 0, row, f, 'synthetic')
        assert validate_record(folder, 0, row, 'synthetic') == first
        folder.joinpath('00000.json').unlink()
        assert validate_record(folder, 0, row, 'synthetic', True) == first
        rejected = False
        try:
            validate_record(folder, 0, row, 'different-signature')
        except AssertionError:
            rejected = True
        assert rejected
    return {'atomic_roundtrip_exact': True, 'sidecar_interruption_recovery': True,
            'wrong_signature_rejected': True, 'punctuation_and_negative_raw_boundary_preserved': True}


def check():
    assert not torch.cuda.is_initialized()
    rows = inputs_and_bindings()
    sig = signature(rows)
    sig_hash = q.digest(sig)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / 'signature.json', sig)
    shortest = min(range(3680), key=lambda i: (len(rows[i]['input_ids']), i))
    longest = max(range(len(rows)), key=lambda i: (len(rows[i]['input_ids']), -i))
    smoke_indices = list(dict.fromkeys([0, shortest, longest]))
    protocol = {'status': 'prepared_not_GPU_run', 'signature_sha256': sig_hash,
                'commands': {'check': 'CPU validation only', 'gpu-smoke': 'explicit GPU only: fixed first/shortest fit and longest full input',
                             'extract': 'explicit GPU only after gpu-smoke; resume all frozen 3839 rows'},
                'smoke_indices': smoke_indices, 'smoke_response_ids': [rows[i]['response_id'] for i in smoke_indices],
                'longest_input_tokens': len(rows[longest]['input_ids']), 'longest_record_index': longest,
                'oracle': 'Independent dense hidden_states from a complete causal decoder forward; exact original full input, all-ones mask; independent layer/pairwise reductions and identical 16-row lm_head geometry.',
                'head_geometry': 'No whole-sequence-vs-chunked BF16 GEMM identity claim; both reference and production explicitly use chunks of16 predictor states.',
                'oracle_atol': ORACLE_ATOL, 'oracle_rtol': 0, 'repeat_exact': True,
                'causality': 'Same-shape intervention at a target and every later token must leave all earlier pre-read features exact.',
                'no_truncation_or_regeneration': True, 'raw_axis_includes_punctuation': True,
                'no_label_or_lexical_filter': True, 'train_classifier': False,
                'feature_shape': [EXPECTED_TOKENS, 4], 'format': 'per-record NPZ and JSON, ordered manifest on completion',
                'resume': 'Skip only signature/identity/axes/finite values/hash-verified records; preserve invalid files and fail; recover missing JSON from self-identifying valid NPZ.',
                'failure': 'Append a unique failure report; do not change frozen source, tolerances, input or prior records.',
                'GPU_scheduling': 'No wait, automatic queue, background launch or GPU use in check; explicit user/root authorization required externally.',
                'scope': 'Existing public-QA development3839 only, no test and no fitting.'}
    frozen_json(OUT / 'protocol.json', protocol)
    storage = storage_selfcheck()
    assert not torch.cuda.is_initialized()
    report = {'status': 'CPU_ready', 'signature_sha256': sig_hash,
              'records': len(rows), 'raw_answer_tokens': EXPECTED_TOKENS, 'fit': 3680, 'calibration': 159,
              'smoke_indices': smoke_indices, 'maximum_input_tokens': 1232, 'storage': storage,
              'checkpoint_assets_hash_verified': True, 'checkpoint_loaded': False, 'GPU_used': False,
              'test_opened': False, 'new_fits': 0}
    frozen_json(OUT / 'CPU_CHECK.json', report)
    return rows, sig, protocol


@torch.no_grad()
def dense_oracle(model, input_ids, positions):
    # Actual complete decoder forward. Hidden-state indexing is independent
    # from ghost hooks, and off-diagonal embedding distances are explicit.
    dense = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        use_cache=False, output_hidden_states=True, output_attentions=False)
    p = positions - 1
    states = torch.stack([x[0, p].float() for x in dense.hidden_states[FIRST:LAST+1]])
    final = dense.last_hidden_state[0, p]
    turbulence = (1 - F.cosine_similarity(states[:-1], states[1:], dim=-1)).mean(0)
    stubbornness = F.cosine_similarity(states, final.float().unsqueeze(0), dim=-1).mean(0)
    distribution = []
    for start in range(0, len(final), LOGIT_BATCH):
        logits = model.lm_head(final[start:start+LOGIT_BATCH]).float()
        values, top_ids = logits.topk(10, dim=-1)
        probabilities = values.softmax(-1)
        entropy = -(probabilities * probabilities.log()).sum(-1)
        embeddings = model.get_input_embeddings().weight[top_ids].float()
        pairs = F.cosine_similarity(embeddings[:, :, None, :], embeddings[:, None, :, :], dim=-1)
        mask = ~torch.eye(10, device=pairs.device, dtype=torch.bool)
        dispersion = (1 - pairs[:, mask]).mean(-1)
        distribution.append(torch.stack((entropy, dispersion), dim=-1))
    output = torch.cat((torch.stack((turbulence, stubbornness), -1), torch.cat(distribution)), -1).float()
    assert torch.isfinite(output).all()
    return output


def tensors(model, row):
    device = model.get_input_embeddings().weight.device
    return (torch.tensor([row['input_ids']], dtype=torch.long, device=device),
            torch.tensor(row['answer_token_positions'], dtype=torch.long, device=device))


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_smoke(rows, sig, protocol):
    destination = OUT / 'GPU_SMOKE.json'
    if destination.exists():
        old = q.read(destination)
        assert old['status'] == 'passed' and old['signature_sha256'] == q.digest(sig)
        print('GPU smoke already passed for this exact frozen signature; no GPU loaded.', flush=True)
        return
    model, loaded = loader.load_nf4()
    assert len(model.model.layers) == 32 and model.config.hidden_size == 4096
    results = []
    try:
        for i in protocol['smoke_indices']:
            row = rows[i]
            ids, positions = tensors(model, row)
            before = [tuple(block._forward_hooks) for block in model.model.layers]
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            actual = ghost.extract(model, ids, positions, FIRST, LAST, LOGIT_BATCH)
            repeated = ghost.extract(model, ids, positions, FIRST, LAST, LOGIT_BATCH)
            assert torch.equal(actual, repeated), ('Same-path repeat failed', row['response_id'])
            expected = dense_oracle(model, ids, positions)
            difference = (actual - expected).abs().amax(0).tolist()
            assert torch.allclose(actual, expected, atol=ORACLE_ATOL, rtol=0), ('Dense oracle failed', i, difference)
            causal = None
            if i == protocol['smoke_indices'][0]:
                middle = len(positions) // 2
                changed = ids.clone()
                changed[:, int(positions[middle]):] = (changed[:, int(positions[middle]):] + 7) % model.config.vocab_size
                intervened = ghost.extract(model, changed, positions, FIRST, LAST, LOGIT_BATCH)
                causal = float((actual[:middle+1] - intervened[:middle+1]).abs().max())
                assert torch.equal(actual[:middle+1], intervened[:middle+1]), ('Causal intervention failed', causal)
                del changed, intervened
            assert before == [tuple(block._forward_hooks) for block in model.model.layers]
            torch.cuda.synchronize()
            info = {'record_index': i, 'response_id': row['response_id'], 'record_sha256': q.digest(row),
                    'input_tokens': len(row['input_ids']), 'raw_answer_tokens': len(positions),
                    'full_input_used': True, 'finite': True, 'repeat_exact': True,
                    'dense_max_abs_error_by_feature': difference, 'causal_max_abs_error': causal,
                    'hooks_restored': True, 'seconds': time.perf_counter() - start,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
            results.append(info)
            print(json.dumps({'GPU_SMOKE_RECORD': info}), flush=True)
            del actual, repeated, expected, ids, positions
        atomic_json(destination, {'status': 'passed', 'signature_sha256': q.digest(sig),
                                  'loaded': loaded, 'checks': results,
                                  'longest_input_checked': protocol['longest_record_index'],
                                  'GPU_used': True, 'trained': False, 'test_opened': False})
    finally:
        del model
        clear_cuda_cache()


def all_records(rows, sig_hash, repair=False):
    folder = OUT / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {f'{i:05d}.npz' for i in range(len(rows))}
    assert not {p.name for p in folder.glob('*.npz')} - wanted, 'Unexpected NPZ in output directory'
    return [validate_record(folder, i, row, sig_hash, repair) for i, row in enumerate(rows)]


def write_complete(rows, sig_hash, metadata):
    assert len(metadata) == EXPECTED_ROWS and all(m is not None for m in metadata)
    assert sum(m['raw_answer_tokens'] for m in metadata) == EXPECTED_TOKENS
    manifest = {'status': 'complete', 'version': VERSION, 'signature_sha256': sig_hash,
                'records': EXPECTED_ROWS, 'raw_answer_tokens': EXPECTED_TOKENS, 'feature_names': FEATURES,
                'answer_order': [r['response_id'] for r in rows], 'entries': metadata,
                'labels_used': False, 'trained': False, 'test_opened': False,
                'all_records_validated': True}
    frozen_json(OUT / 'feature_manifest.json', manifest)
    completion = {'status': 'complete', 'records': EXPECTED_ROWS, 'raw_answer_tokens': EXPECTED_TOKENS,
                  'signature_sha256': sig_hash, 'all_records_validated': True,
                  'files_sha256': {name: q.sha(OUT / name) for name in
                                   ('feature_manifest.json', 'signature.json', 'protocol.json', 'CPU_CHECK.json', 'GPU_SMOKE.json')},
                  'no_test': True, 'trained': False}
    frozen_json(OUT / 'features_complete.json', completion)


def extract(rows, sig):
    sig_hash = q.digest(sig)
    smoke = q.read(OUT / 'GPU_SMOKE.json')
    assert smoke['status'] == 'passed' and smoke['signature_sha256'] == sig_hash
    metadata = all_records(rows, sig_hash, repair=True)
    missing = [i for i, m in enumerate(metadata) if m is None]
    if not missing:
        write_complete(rows, sig_hash, metadata)
        print('All3839 records verified; no GPU loaded.', flush=True)
        return
    assert not (OUT / 'features_complete.json').exists(), 'Completion exists but records are missing'
    # Resumability is by individually committed, self-identifying NPZ files.
    model, loaded = loader.load_nf4()
    started = time.perf_counter()
    try:
        for done, i in enumerate(missing, 1):
            row = rows[i]
            ids, positions = tensors(model, row)
            before = [tuple(block._forward_hooks) for block in model.model.layers]
            values = ghost.extract(model, ids, positions, FIRST, LAST, LOGIT_BATCH)
            assert before == [tuple(block._forward_hooks) for block in model.model.layers]
            array = values.cpu().numpy()
            metadata[i] = save_record(OUT / 'features', i, row, array, sig_hash)
            del ids, positions, values, array
            if done == 1 or done % 25 == 0 or done == len(missing):
                progress = {'status': 'running', 'signature_sha256': sig_hash,
                            'completed_count': len(rows) - len(missing) + done,
                            'new_this_run': done, 'pid': os.getpid(), 'last_record_index': i,
                            'seconds': time.perf_counter() - started, 'loaded': loaded}
                atomic_json(OUT / 'progress.json', progress)
                print(json.dumps({k: progress[k] for k in ('completed_count', 'new_this_run', 'pid', 'last_record_index', 'seconds')}), flush=True)
        # Reopen every NPZ and verify both hashes and exact original token axes.
        metadata = all_records(rows, sig_hash, repair=False)
        write_complete(rows, sig_hash, metadata)
        atomic_json(OUT / 'last_run.json', {'status': 'complete', 'pid': os.getpid(),
                                           'seconds': time.perf_counter() - started,
                                           'resumed_verified_records': len(rows) - len(missing),
                                           'new_records': len(missing), 'loaded': loaded,
                                           'GPU_used': True, 'trained': False})
        print(json.dumps({'status': 'complete', 'records': len(rows), 'new_records': len(missing),
                          'seconds': time.perf_counter()-started}), flush=True)
    finally:
        del model
        clear_cuda_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'gpu-smoke', 'extract'])
    args = parser.parse_args()
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
        failure = OUT / f'FAILURE_{args.command}_{time.time_ns()}_{os.getpid()}.json'
        atomic_json(failure, {'status': 'failed', 'command': args.command, 'pid': os.getpid(),
                              'exception': repr(exc), 'traceback': traceback.format_exc(),
                              'frozen_source_changed': False, 'tolerance_changed': False})
        raise


if __name__ == '__main__':
    main()
