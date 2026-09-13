"""R21 frozen train-only HARP256 and evidence/no-evidence hidden differences.

CPU actions never load the LLM or initialize CUDA. GPU replay requires an
explicit --selfcheck-only or --run action. No labels, scores, fitted PCA, new
generation, reference answers, or original held-out contents are consumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
R18 = ROOT.parent / 'round18_input_information_diagnostics'
R17 = ROOT.parent / 'round17_expanded_retraining'
sys.path.insert(0, str(R17 / 'src'))
import extract17 as base

SOURCE = base.SOURCE
VERSION = 'round21-bottom256-float64-empty-evidence-v1'
EXPECTED, WIDTH, RANK, BLOCK, THREADS = 602, 3584, 256, 1024, 4
MARKER = '\n\nSearch results:\n'
SPLIT_FIELD = re.compile(r'(?<!\\)"split"\s*:\s*"([^"\\]+)"')
COORD_KEYS = ('token_ids', 'response_token_offsets', 'token_start', 'token_end')
SCHEMA = {
    'hidden_noctx_28': 'float32[T,3584], final RMSNorm after reading answer token i at P0+i',
    'hidden_delta_28': 'float32[T,3584], original R18 hidden_28 minus hidden_noctx_28; no L2 normalization',
    'harp_256': 'float32[T,256], original R18 hidden_28 projected onto bottom256 right singular vectors of lm_head',
    'token_ids': 'int64[T], exact frozen generated response IDs',
    'response_token_offsets': 'int32[T,2], exact frozen full-response character offsets',
    'token_start': 'int32[T]', 'token_end': 'int32[T]',
}


def cpu_only():
    assert not torch.cuda.is_initialized(), 'A CPU-only action must not initialize CUDA'


def train_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = list(SPLIT_FIELD.finditer(line))
            assert len(fields) == 1, 'Each line needs one explicit split metadata field'
            if fields[0].group(1) != 'train':
                continue  # Do not parse original held-out contents.
            row = json.loads(line)
            assert row['split'] == 'train'
            rows.append(row)
    return rows


def context():
    rows = train_rows(SOURCE / 'data/inputs.jsonl')
    assert len(rows) == EXPECTED and len({r['row_id'] for r in rows}) == EXPECTED
    assert len({r['question_id'] for r in rows}) == 301
    assert len({r['group_id'] for r in rows}) == 278
    assert all(len(r['questions']) == 1 and len(r['passages']) == 2 for r in rows)
    source = base.read(SOURCE / 'data/generation_manifest.json')
    assert source['complete'] and source['generated'] == 800
    r18 = base.read(R18 / 'data/feature_manifest.json')
    assert r18['complete'] and r18['completed_count'] == EXPECTED
    assert set(r18['records']) == {r['row_id'] for r in rows}
    original = r18['extraction_signature']['base_replay_signature']
    assert base.sha(SOURCE / 'data/inputs.jsonl') == original['source_inputs_sha256']
    assert base.sha(SOURCE / 'data/input_freeze.json') == original['source_input_freeze_sha256']
    assert base.sha(SOURCE / 'data/generation_manifest.json') == original['source_generation_manifest_sha256']
    records = {}
    for row in rows:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', row['row_id'])
        records[row['row_id']] = base.generation(row, source)
    for directory in ('features', 'harp_features'):
        for pattern in ('*.json', '*.npz'):
            for path in (ROOT / 'data' / directory).glob(pattern):
                assert path.stem in records, ('Non-training cache', str(path))
    return rows, records, r18


def cached_reference(row, generated, generation_sha, r18):
    rec = r18['records'][row['row_id']]
    path, side = R18 / rec['npz'], R18 / rec['json']
    assert base.sha(path) == rec['npz_sha256'] and base.sha(side) == rec['json_sha256']
    meta = base.read(side)
    assert rec['source_generation_sha256'] == meta['source_generation_sha256'] == generation_sha
    assert meta['input_row_sha256'] == base.model7.digest(row)
    with np.load(path, allow_pickle=False) as old:
        hidden = old['hidden_28']
        coords = {key: old[key] for key in COORD_KEYS}
    assert hidden.shape == (len(generated['response_token_ids']), WIDTH)
    assert hidden.dtype == np.float32 and np.isfinite(hidden).all()
    validate_coords(coords, generated)
    return hidden, coords, {'r18_npz_sha256': rec['npz_sha256'], 'r18_json_sha256': rec['json_sha256']}


def validate_coords(arrays, generated):
    n = len(generated['response_token_ids'])
    assert arrays['token_ids'].dtype == np.int64 and arrays['token_ids'].shape == (n,)
    assert arrays['token_ids'].tolist() == generated['response_token_ids']
    offsets = arrays['response_token_offsets']
    assert offsets.dtype == np.int32 and offsets.shape == (n, 2)
    assert offsets.tolist() == generated['response_token_offsets']
    for key, column in [('token_start', 0), ('token_end', 1)]:
        assert arrays[key].dtype == np.int32 and np.array_equal(arrays[key], offsets[:, column])


def make_plan(tokenizer, row, generated):
    assert row['split'] == 'train' and row['prompt'].count(MARKER) == 1
    prefix = base.model7.chat_ids(tokenizer, row['prompt'], row['system'])
    assert prefix == generated['input_token_ids'], ('Original chat template changed', row['row_id'])
    question_part, removed = row['prompt'].split(MARKER, 1)
    assert removed.strip() and all(p['title'] in removed and p['text'] in removed for p in row['passages'])
    noctx_prompt = question_part + MARKER
    assert noctx_prompt == row['prompt'][:len(noctx_prompt)]
    noctx_ids = base.model7.chat_ids(tokenizer, noctx_prompt, row['system'])
    assert len(noctx_ids) < len(prefix)
    answer = list(generated['response_token_ids'])
    offsets = base.attention7._response_offsets(tokenizer, answer, generated['response'])
    assert offsets.tolist() == generated['response_token_offsets']
    return {'row_id': row['row_id'], 'actual_split': 'train',
        'system': row['system'], 'original_prompt': row['prompt'], 'noctx_prompt': noctx_prompt,
        'original_prefix_token_ids': prefix, 'noctx_prefix_token_ids': noctx_ids,
        'original_prefix_tokens': len(prefix), 'noctx_prefix_tokens': len(noctx_ids),
        'removed_reference_text_sha256': base.model7.digest(removed),
        'removed_reference_characters': len(removed),
        'original_prompt_sha256': base.model7.digest({'system': row['system'], 'prompt': row['prompt']}),
        'noctx_prompt_sha256': base.model7.digest({'system': row['system'], 'prompt': noctx_prompt}),
        'original_prefix_ids_sha256': base.model7.digest(prefix),
        'noctx_prefix_ids_sha256': base.model7.digest(noctx_ids),
        'response_ids_sha256': base.model7.digest(answer),
        'response_offsets_sha256': base.model7.digest(generated['response_token_offsets']),
        'response_tokens': len(answer),
        'construction': 'Keep original instruction/question and empty Search results marker; delete all following titles and bodies',
        'new_refusal_instruction': False, 'response_regenerated': False}


def signature():
    return {'version': VERSION,
        'code_sha256': {str(p.resolve()): base.sha(p) for p in
                       (Path(__file__), Path(base.__file__), Path(base.model7.__file__), Path(base.attention7.__file__))},
        'base_model_and_source_signature': base.signature(),
        'r18_feature_manifest_sha256': base.sha(R18 / 'data/feature_manifest.json'),
        'rank': RANK, 'width': WIDTH, 'gram_dtype': 'float64', 'gram_row_block': BLOCK,
        'projection_accumulation_dtype': 'float64; final features float32',
        'cpu_threads': THREADS, 'schema': SCHEMA,
        'position': 'post-read final RMSNorm, original P+i and noctx P0+i',
        'normalization': 'No additional L2 normalization; subtract float32 final-RMSNorm outputs',
        'actual_split': 'train', 'expected_rows': EXPECTED}


def write_frozen_json(path, value):
    if path.exists():
        assert base.read(path) == value, ('Frozen artifact changed', str(path))
    else:
        base.save(path, value)


def prepare():
    cpu_only()
    rows, records, r18 = context()
    tok = AutoTokenizer.from_pretrained(base.model7.MODEL, local_files_only=True)
    plans = {}
    for row in rows:
        generated, digest = records[row['row_id']]
        plan = make_plan(tok, row, generated)
        hidden, _, reference = cached_reference(row, generated, digest, r18)
        del hidden
        plan.update(source_generation_sha256=digest, input_row_sha256=base.model7.digest(row), **reference)
        plans[row['row_id']] = plan
    sig = signature()
    write_frozen_json(ROOT / 'data/extraction_signature.json', sig)
    write_frozen_json(ROOT / 'data/noctx_plans.json', plans)
    write_frozen_json(ROOT / 'data/preparation_check.json', {
        'passed': True, 'version': VERSION, 'records': len(rows), 'questions': 301, 'event_groups': 278,
        'response_tokens': sum(p['response_tokens'] for p in plans.values()),
        'prefix_tokens_range_original': [min(p['original_prefix_tokens'] for p in plans.values()), max(p['original_prefix_tokens'] for p in plans.values())],
        'prefix_tokens_range_noctx': [min(p['noctx_prefix_tokens'] for p in plans.values()), max(p['noctx_prefix_tokens'] for p in plans.values())],
        'plans_sha256': base.sha(ROOT / 'data/noctx_plans.json'),
        'extraction_signature_sha256': base.model7.digest(sig),
        'all_original_prefix_ids_rebuilt_exactly': True, 'all_answer_ids_and_offsets_exact': True,
        'all_r18_hidden_and_coordinate_shapes_checked': True, 'all_title_and_body_contents_deleted': True,
        'labels_or_scores_read': False, 'original_heldout_content_parsed': False,
        'gpu_initialized': torch.cuda.is_initialized()})
    cpu_only()
    print('PREPARED', len(rows), 'train records; CUDA not initialized', flush=True)
    return rows, records, r18, plans, sig


def head_asset():
    config = base.read(base.model7.MODEL / 'config.json')
    assert config['hidden_size'] == WIDTH and not config.get('tie_word_embeddings', False)
    found = []
    for path in sorted(base.model7.MODEL.glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            if 'lm_head.weight' in handle.keys():
                part = handle.get_slice('lm_head.weight')
                found.append({'path': str(path.resolve()), 'key': 'lm_head.weight',
                              'shape': list(part.get_shape()), 'dtype': part.get_dtype()})
    assert len(found) == 1, 'Need exactly one dense output head in local safetensors'
    asset = found[0]
    assert asset['shape'] == [config['vocab_size'], WIDTH]
    assert asset['dtype'] in ('BF16', 'F16', 'F32', 'F64'), 'Do not interpret packed/quantized bytes as a dense head'
    return asset


def gram_from_blocks(read_block, nrows, width, block=BLOCK):
    """Accumulate only lower triangle in float64 BLAS; no vocabulary-square U."""
    from scipy.linalg.blas import dsyrk
    gram = np.zeros((width, width), dtype=np.float64, order='F')
    with threadpool_limits(limits=THREADS):
        for begin in range(0, nrows, block):
            chunk = np.asarray(read_block(begin, min(begin + block, nrows)), dtype=np.float64)
            assert chunk.shape[1] == width and np.isfinite(chunk).all()
            gram = dsyrk(1.0, np.asfortranarray(chunk), beta=1.0, c=gram,
                         trans=1, lower=1, overwrite_c=1)
        gram = gram + np.tril(gram, -1).T
    assert np.array_equal(gram, gram.T) and np.isfinite(gram).all()
    return gram


def bottom_basis(gram, rank):
    from scipy.linalg import eigh
    assert gram.dtype == np.float64 and 0 < rank < gram.shape[0]
    with threadpool_limits(limits=THREADS):
        values, vectors = eigh(gram, subset_by_index=[0, rank - 1], driver='evr',
                               check_finite=False, overwrite_a=False)
        # Deterministic sign convention; a repeated eigenvalue still defines a subspace.
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(rank)])
        vectors *= np.where(signs == 0, 1.0, signs)[None]
        residual = gram @ vectors - vectors * values[None]
        scale = max(float(np.linalg.norm(gram, 'fro')), np.finfo(np.float64).tiny)
        relative = np.linalg.norm(residual, axis=0) / scale
        orthogonal = float(np.max(np.abs(vectors.T @ vectors - np.eye(rank))))
        trace = float(np.trace(gram))
        negative_tolerance = 100 * np.finfo(np.float64).eps * scale
    check = {'orthogonality_max_absolute_error': orthogonal,
        'eigen_residual_max_relative_to_gram_frobenius': float(relative.max()),
        'eigenvalue_min': float(values.min()), 'eigenvalue_max_selected': float(values.max()),
        'negative_eigenvalue_tolerance': negative_tolerance, 'gram_trace_total_head_energy': trace,
        'selected_head_energy_fraction': float(values.sum() / trace),
        'passed': bool(orthogonal < 1e-8 and relative.max() < 1e-10 and values.min() >= -negative_tolerance)}
    assert check['passed'], ('Numerically unreliable bottom subspace', check)
    return values, vectors.T.copy(), check


def build_basis(sig):
    cpu_only()
    path = ROOT / 'data/harp_basis.npz'; meta_path = path.with_suffix('.json')
    asset = head_asset()
    expected = {'extraction_signature_sha256': base.model7.digest(sig), 'asset': asset,
                'rank': RANK, 'gram_dtype': 'float64'}
    if meta_path.exists():
        meta = base.read(meta_path)
        assert all(meta[k] == v for k, v in expected.items())
        assert base.sha(path) == meta['arrays_sha256'] and meta['numerical_check']['passed']
        with np.load(path, allow_pickle=False) as arrays:
            components = arrays['components']
        assert components.shape == (RANK, WIDTH) and components.dtype == np.float64
        return components, meta
    started = time.perf_counter()
    with safe_open(asset['path'], framework='pt', device='cpu') as handle:
        sliced = handle.get_slice(asset['key'])
        def reader(begin, end):
            return sliced[begin:end, :].to(dtype=torch.float64, device='cpu').numpy()
        gram = gram_from_blocks(reader, asset['shape'][0], WIDTH)
    gram_seconds = time.perf_counter() - started
    values, components, check = bottom_basis(gram, RANK)
    del gram
    assert signature() == sig, 'Model/source/code changed during CPU decomposition'
    base.save_arrays(path, {'components': components, 'eigenvalues': values,
                           'singular_values': np.sqrt(np.maximum(values, 0.0))})
    meta = {**expected, 'version': VERSION, 'arrays_sha256': base.sha(path),
        'numerical_check': check, 'gram_seconds': gram_seconds, 'seconds': time.perf_counter() - started,
        'basis_order': 'ascending eigenvalues of W.T W; components shape [256,3584]',
        'normalization': 'orthonormal basis; no whitening; deterministic largest-absolute component positive',
        'weight_file_sha256': sig['base_model_and_source_signature']['model_and_tokenizer_files_sha256'][Path(asset['path']).name],
        'static_model_parameter_basis': True, 'fitted_on_labels_or_examples': False,
        'full_vocabulary_square_U_constructed': False, 'gpu_initialized': torch.cuda.is_initialized()}
    base.save(meta_path, meta)
    cpu_only()
    print('HARP_BASIS_COMPLETE', round(meta['seconds'], 3), check, flush=True)
    return components, meta


def harp_features(rows, records, r18, components, basis_meta, sig):
    cpu_only(); entries = {}
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        rid = row['row_id']; generated, digest = records[rid]
        path = ROOT / 'data/harp_features' / (rid + '.npz'); side = path.with_suffix('.json')
        expected = {'row_id': rid, 'source_generation_sha256': digest,
            'input_row_sha256': base.model7.digest(row), 'basis_arrays_sha256': basis_meta['arrays_sha256'],
            'extraction_signature_sha256': base.model7.digest(sig),
            'r18_npz_sha256': r18['records'][rid]['npz_sha256']}
        if side.exists():
            meta = base.read(side)
            assert all(meta[k] == v for k, v in expected.items())
            assert base.sha(path) == meta['arrays_sha256']
            with np.load(path, allow_pickle=False) as arrays:
                value = {key: arrays[key] for key in arrays.files}
        else:
            hidden, coords, reference = cached_reference(row, generated, digest, r18)
            with threadpool_limits(limits=THREADS):
                projected = (hidden.astype(np.float64) @ components.T).astype(np.float32)
            value = {'harp_256': projected, **coords}
            base.save_arrays(path, value)
            meta = {**expected, **reference, 'version': VERSION, 'arrays_sha256': base.sha(path),
                'response_tokens': len(generated['response_token_ids']), 'actual_split': 'train',
                'projection': 'float64 accumulation h_ref @ components.T; cast result to float32',
                'labels_or_scores_read': False}
            base.save(side, meta)
        validate_coords(value, generated)
        assert value['harp_256'].shape == (len(generated['response_token_ids']), RANK)
        assert value['harp_256'].dtype == np.float32 and np.isfinite(value['harp_256']).all()
        entries[rid] = {'npz': str(path.relative_to(ROOT)), 'npz_sha256': meta['arrays_sha256'],
            'json': str(side.relative_to(ROOT)), 'json_sha256': base.sha(side),
            'source_generation_sha256': digest, 'response_tokens': len(generated['response_token_ids'])}
        if index % 100 == 0:
            print('HARP_PROJECTED', index, EXPECTED, flush=True)
    assert signature() == sig
    manifest = {'version': VERSION, 'complete': len(entries) == EXPECTED, 'completed_count': len(entries),
        'expected_rows': EXPECTED, 'records': entries, 'basis_arrays_sha256': basis_meta['arrays_sha256'],
        'extraction_signature_sha256': base.model7.digest(sig), 'actual_split': 'train',
        'selected_questions': 301, 'selected_event_groups': 278, 'labels_or_scores_read': False}
    write_frozen_json(ROOT / 'data/harp_manifest.json', manifest)
    print('HARP_PROJECTION_COMPLETE', len(entries), 'seconds', round(time.perf_counter() - started, 3), flush=True)
    return manifest


def load_harp(rid, generated, manifest):
    rec = manifest['records'][rid]; path = ROOT / rec['npz']
    assert base.sha(path) == rec['npz_sha256']
    with np.load(path, allow_pickle=False) as arrays:
        value = {key: arrays[key] for key in arrays.files}
    validate_coords(value, generated)
    return value['harp_256']


@torch.inference_mode()
def replay_hidden(model, prefix, answer):
    assert not model.training and getattr(model, 'is_loaded_in_4bit', False)
    assert model.config.hidden_size == WIDTH and len(model.model.layers) == 28
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([list(prefix) + list(answer)], dtype=torch.long, device=device)
    last = model.model(input_ids=ids, use_cache=False, output_attentions=False,
                       output_hidden_states=False).last_hidden_state[0]
    value = last[len(prefix):len(prefix) + len(answer)].float().cpu().numpy()
    assert value.shape == (len(answer), WIDTH) and np.isfinite(value).all()
    return value


def extract(model, row, generated, digest, plan, r18, harp_manifest):
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    hidden, coords, reference = cached_reference(row, generated, digest, r18)
    noctx = replay_hidden(model, plan['noctx_prefix_token_ids'], generated['response_token_ids'])
    delta = hidden - noctx  # Both operands are float32 final-RMSNorm states.
    arrays = {'hidden_noctx_28': noctx, 'hidden_delta_28': delta,
              'harp_256': load_harp(row['row_id'], generated, harp_manifest), **coords}
    validate(arrays, generated)
    torch.cuda.synchronize()
    meta = {'version': VERSION, 'row_id': row['row_id'], 'actual_split': 'train', **reference,
        'intervention_plan': plan, 'noctx_prompt_ids': plan['noctx_prefix_token_ids'],
        'original_prompt_ids': plan['original_prefix_token_ids'],
        'response_tokens': len(generated['response_token_ids']),
        'timing': 'After reading token i: original P+i versus noctx P0+i; final RMSNorm',
        'normalization': 'No extra L2 normalization; float32 original-minus-noctx difference',
        'harp_basis_arrays_sha256': harp_manifest['basis_arrays_sha256'],
        'harp_projection_cache_sha256': harp_manifest['records'][row['row_id']]['npz_sha256'],
        'seconds': time.perf_counter() - started, 'peak_allocated_gib': float(torch.cuda.max_memory_allocated() / 2**30),
        'forward_passes': 1, 'labels_or_scores_read': False, 'response_regenerated': False,
        'original_heldout_content_parsed': False, 'schema': SCHEMA,
        'interpretation': 'Offline same-answer replay; removal changes context lengths/positions, not a pure causal truth test'}
    return arrays, meta


def validate(arrays, generated):
    assert set(arrays) == set(SCHEMA)
    validate_coords(arrays, generated)
    for key, width in [('hidden_noctx_28', WIDTH), ('hidden_delta_28', WIDTH), ('harp_256', RANK)]:
        assert arrays[key].shape == (len(generated['response_token_ids']), width)
        assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()


def expected(row, digest, plan, sig):
    return {'row_id': row['row_id'], 'source_generation_sha256': digest,
        'input_row_sha256': base.model7.digest(row), 'plan_sha256': base.model7.digest(plan),
        'extraction_signature_sha256': base.model7.digest(sig)}


def cached(row, generated, digest, plan, sig):
    path = ROOT / 'data/features' / (row['row_id'] + '.json')
    if not path.exists():
        return None
    meta = base.read(path)
    assert all(meta[k] == v for k, v in expected(row, digest, plan, sig).items())
    assert base.sha(path.with_suffix('.npz')) == meta['arrays_sha256']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate(arrays, generated)
    return arrays, meta


def store(row, generated, digest, plan, arrays, meta, sig):
    assert signature() == sig, 'Frozen source or code changed during extraction'
    assert base.sha(SOURCE / 'data/generation_records' / (row['row_id'] + '.json')) == digest
    path = ROOT / 'data/features' / (row['row_id'] + '.npz')
    base.save_arrays(path, arrays)
    meta.update(expected(row, digest, plan, sig), arrays_sha256=base.sha(path), extraction_signature=sig)
    base.save(path.with_suffix('.json'), meta)


def feature_manifest(rows, records, plans, sig):
    entries, missing = {}, []
    for row in rows:
        rid = row['row_id']; generated, digest = records[rid]
        value = cached(row, generated, digest, plans[rid], sig)
        if value is None:
            missing.append(rid); continue
        _, meta = value; relative = 'data/features/' + rid
        entries[rid] = {'npz': relative + '.npz', 'npz_sha256': meta['arrays_sha256'],
            'json': relative + '.json', 'json_sha256': base.sha(ROOT / (relative + '.json')),
            'source_generation_sha256': digest, 'response_tokens': meta['response_tokens'], 'split': 'train'}
    result = {'version': VERSION, 'complete': not missing, 'expected_rows': EXPECTED,
        'completed_count': len(entries), 'records': entries, 'missing_rows': missing, 'schema': SCHEMA,
        'total_response_tokens_completed': sum(r['response_tokens'] for r in entries.values()),
        'selected_questions': 301, 'selected_group_ids': 278, 'actual_split': 'train',
        'validation_or_test_rows_extracted': 0, 'labels_or_scores_read': False, 'response_regenerated': False,
        'all_training_outputs_including_refusals_and_parse_failures': True,
        'extraction_signature': sig, 'extraction_signature_sha256': base.model7.digest(sig),
        'harp_manifest_sha256': base.sha(ROOT / 'data/harp_manifest.json'),
        'limitations': ['Empty-source replay changes context length, relative positions, attention and numerical shape.',
                       'Frozen answer prefixes need not be the no-evidence model natural generation.',
                       'No future-token smoothing; this is offline replay, not a tested realtime implementation.',
                       'Static HARP directions are low output-head sensitivity directions, not verified truth coordinates.']}
    base.save(ROOT / 'data/feature_manifest.json', result)
    return result


def gpu_selfcheck(model, rows, records, plans, r18, harp_manifest, sig):
    checks = []
    for row in (rows[0], rows[-1]):
        rid = row['row_id']; generated, digest = records[rid]; plan = plans[rid]
        original, _, _ = cached_reference(row, generated, digest, r18)
        identity = replay_hidden(model, plan['original_prefix_token_ids'], generated['response_token_ids'])
        difference = float(np.max(np.abs(identity - original)))
        assert np.array_equal(identity, original), ('Original hidden replay changed; do not relax silently', rid, difference)
        arrays, meta = extract(model, row, generated, digest, plan, r18, harp_manifest)
        repeat = replay_hidden(model, plan['noctx_prefix_token_ids'], generated['response_token_ids'])
        assert np.array_equal(arrays['hidden_noctx_28'], repeat), ('Noctx repeat changed', rid)
        assert np.array_equal(arrays['hidden_delta_28'], original - repeat)
        meta['gpu_selfcheck'] = {'identity_r18_max_abs_difference': difference,
            'identity_delta_max_abs': float(np.abs(original - identity).max()),
            'noctx_repeat_max_abs_difference': float(np.abs(repeat - arrays['hidden_noctx_28']).max())}
        store(row, generated, digest, plan, arrays, meta, sig)
        checks.append({'row_id': rid, **meta['gpu_selfcheck'],
                       'shapes': {key: list(value.shape) for key, value in arrays.items()}})
    base.save(ROOT / 'data/extraction_selfcheck.json', {'passed': True, 'checks': checks,
        'extraction_signature_sha256': base.model7.digest(sig),
        'selection': 'first and last actual train rows in original order, no labels',
        'strict_online_prefix_numerical_identity_claimed': False})
    print('GPU_SELFCHECK_PASSED', checks, flush=True)


def cpu_selfcheck():
    """Independent dense-SVD oracle on an ill-conditioned synthetic head."""
    cpu_only(); torch.set_num_threads(THREADS)
    rng = np.random.default_rng(20260921)
    u, _ = np.linalg.qr(rng.normal(size=(71, 12)))
    v, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    weight = u @ np.diag(np.geomspace(1.0, 0.001, 12)) @ v.T
    gram = gram_from_blocks(lambda a, b: weight[a:b], 71, 12, block=9)
    assert np.allclose(gram, weight.T @ weight, rtol=1e-12, atol=1e-14)
    values, components, numerical = bottom_basis(gram, 4)
    _, singular, vt = np.linalg.svd(weight, full_matrices=False)
    projector_error = float(np.max(np.abs(components.T @ components - vt[-4:].T @ vt[-4:])))
    assert projector_error < 1e-8
    assert np.allclose(values, singular[-4:][::-1] ** 2, rtol=1e-8, atol=1e-14)
    source = np.arange(60, dtype=np.float32).reshape(5, 12) / 11
    noctx = source / 2
    assert np.array_equal(source - noctx, source - source / 2)
    projected = (source.astype(np.float64) @ components.T).astype(np.float32)
    assert projected.shape == (5, 4) and np.isfinite(projected).all()
    result = {'passed': True, 'version': VERSION, 'synthetic_shape': [71, 12],
        'bottom_rank': 4, 'gram_max_abs_error': float(np.abs(gram - weight.T @ weight).max()),
        'dense_svd_projector_max_abs_error': projector_error, 'numerical_check': numerical,
        'safe_local_head_metadata': head_asset(), 'gpu_initialized': torch.cuda.is_initialized()}
    base.save(ROOT / 'data/cpu_selfcheck.json', result)
    cpu_only(); print('CPU_SELFCHECK_PASSED', json.dumps(result), flush=True)


def run(action):
    torch.set_num_threads(THREADS)
    if action == 'cpu_selfcheck':
        cpu_selfcheck(); return
    rows, records, r18, plans, sig = prepare()
    if action == 'prepare':
        return
    if action == 'audit':
        result = feature_manifest(rows, records, plans, sig)
        print('AUDIT', result['completed_count'], EXPECTED, result['complete'], flush=True); return
    # All static head decomposition/projection is CPU, before loading Qwen.
    components, basis_meta = build_basis(sig)
    harp_manifest = harp_features(rows, records, r18, components, basis_meta, sig)
    del components
    if action == 'harp':
        cpu_only(); return
    assert action in ('selfcheck', 'run')
    model = None
    check_path = ROOT / 'data/extraction_selfcheck.json'
    if check_path.exists():
        check = base.read(check_path)
        assert check['passed'] and check['extraction_signature_sha256'] == base.model7.digest(sig)
    else:
        _, model = base.model7.load_model(); torch.set_num_threads(THREADS)
        gpu_selfcheck(model, rows, records, plans, r18, harp_manifest, sig)
    if action == 'selfcheck':
        feature_manifest(rows, records, plans, sig); return
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        rid = row['row_id']; generated, digest = records[rid]; plan = plans[rid]
        if cached(row, generated, digest, plan, sig) is not None:
            continue
        if model is None:
            _, model = base.model7.load_model(); torch.set_num_threads(THREADS)
        arrays, meta = extract(model, row, generated, digest, plan, r18, harp_manifest)
        store(row, generated, digest, plan, arrays, meta, sig)
        print('DONE', index, EXPECTED, rid, meta['response_tokens'], round(meta['seconds'], 3), flush=True)
        if index % 100 == 0:
            feature_manifest(rows, records, plans, sig)
    assert signature() == sig
    result = feature_manifest(rows, records, plans, sig)
    print('COMPLETE', result['completed_count'], EXPECTED, result['complete'],
          'elapsed_seconds', round(time.perf_counter() - started, 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for flag, action in [('--prepare-only', 'prepare'), ('--cpu-selfcheck', 'cpu_selfcheck'),
                         ('--harp-only', 'harp'), ('--selfcheck-only', 'selfcheck'),
                         ('--run', 'run'), ('--audit', 'audit')]:
        actions.add_argument(flag, dest='action', action='store_const', const=action)
    run(parser.parse_args().action)
