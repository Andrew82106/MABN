"""Replay every frozen R16 response: 784 Lookback ratios and selected-token NLL.

No generation, labels, question routing, filtering, fitted transforms or hidden
state export. Attention is post-read P+j; NLL predicts token j from P+j-1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'round16_dataset_expansion'
R7 = ROOT.parent / 'round7_evidence_grounding'
R10 = ROOT.parent / 'round10_dual_granularity'
for directory in (R7/'src', R10/'src'):
    sys.path.insert(0, str(directory))
import model7
import attention7

VERSION = 'round17-lookback-nll-v1'
QUERY_BATCH, LOGIT_BATCH = 8, 16
_MODEL_ASSETS = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    pending.replace(path)


def save_arrays(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('wb') as f:
        np.savez_compressed(f, **arrays)
    pending.replace(path)


def source_rows():
    rows = [json.loads(s) for s in (SOURCE/'data/inputs.jsonl').read_text('utf-8').splitlines() if s.strip()]
    frozen, manifest = read(SOURCE/'data/input_freeze.json'), read(SOURCE/'data/generation_manifest.json')
    assert len(rows) == 800 and len({r['row_id'] for r in rows}) == 800
    assert manifest['complete'] and manifest['generated'] == len(rows)
    for name, value in frozen['files_sha256'].items():
        assert sha(SOURCE/name) == value, ('Frozen source changed', name)
    for row in rows:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', row['row_id'])
        assert row['expected_items'] == 1
    return rows, manifest


def signature():
    global _MODEL_ASSETS
    paths = sorted({p for pattern in ('*.safetensors', '*.json', 'merges.txt', 'vocab.*', 'tokenizer.*')
                    for p in model7.MODEL.glob(pattern) if p.is_file()})
    stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    if _MODEL_ASSETS is None:
        _MODEL_ASSETS = {'stat': stats, 'sha256': {p.name: sha(p) for p in paths}}
    assert stats == _MODEL_ASSETS['stat'], 'Local model/tokenizer files changed during replay'
    files = [Path(__file__), Path(model7.__file__), Path(attention7.__file__)]
    return {'version': VERSION,
            'code_sha256': {str(p.resolve()): sha(p) for p in files},
            'source_inputs_sha256': sha(SOURCE/'data/inputs.jsonl'),
            'source_input_freeze_sha256': sha(SOURCE/'data/input_freeze.json'),
            'source_generation_manifest_sha256': sha(SOURCE/'data/generation_manifest.json'),
            'model_path': str(model7.MODEL.resolve()),
            'model_config_sha256': sha(model7.MODEL/'config.json'),
            'tokenizer_config_sha256': sha(model7.MODEL/'tokenizer_config.json'),
            'model_and_tokenizer_files_sha256': _MODEL_ASSETS['sha256'],
            'model7_generation_config': model7.CONFIG,
            'torch_version': torch.__version__,
            'transformers_version': importlib.metadata.version('transformers'),
            'bitsandbytes_version': importlib.metadata.version('bitsandbytes')}


def visible(row):
    return {'system': row['system'], 'prompt': row['prompt'],
            'questions': list(row['questions']),
            'passages': [{'title': p['title'], 'text': p['text']} for p in row['passages']]}


def generation(row, source_manifest):
    path = SOURCE/'data/generation_records'/(row['row_id']+'.json')
    digest = sha(path)
    assert digest == source_manifest['record_sha256'][row['row_id']]
    g = read(path)
    assert g['row_id'] == row['row_id'] and g['input_row_sha256'] == model7.digest(row)
    assert g['prompt_hash'] == model7.prompt_hash(row)
    assert g['signature']['model_config_sha256'] == sha(model7.MODEL/'config.json')
    assert g['signature']['tokenizer_config_sha256'] == sha(model7.MODEL/'tokenizer_config.json')
    assert g['signature']['generation_config'] == model7.CONFIG
    assert g['signature']['model7_sha256'] == sha(Path(model7.__file__))
    # Deliberately do not inspect g['items'] or any annotation; malformed answers
    # such as a bare year still have ordinary response tokens to replay.
    assert g['response_token_ids'] and not g.get('unexpected_special_token_ids')
    return g, digest


class LookbackHooks:
    def __init__(self, model, context, start, count):
        self.model, self.start, self.count = model, start, count
        self.heads, self.layers = model.config.num_attention_heads, len(model.model.layers)
        self.device = model.model.embed_tokens.weight.device
        self.positions = torch.arange(start, start+count, device=self.device)
        self.context = torch.as_tensor(context, device=self.device)
        self.values = np.empty((count, self.layers, self.heads), dtype=np.float32)
        self.pending, self.handles, self.seen = {}, [], set()

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            attn = block.self_attn
            def before(module, args, kwargs, layer=layer):
                if kwargs.get('past_key_value') is not None:
                    raise ValueError('Replay must disable KV cache')
                self.pending[layer] = {'rope': kwargs['position_embeddings']}
            def query(module, args, output, layer=layer):
                self.pending[layer]['q'] = output
            def key(module, args, output, layer=layer, attn=attn):
                self.attention(layer, attn, output)
            self.handles.extend([attn.register_forward_pre_hook(before, with_kwargs=True),
                                 attn.q_proj.register_forward_hook(query),
                                 attn.k_proj.register_forward_hook(key)])
        return self

    def __exit__(self, *args):
        for h in self.handles: h.remove()
        self.pending.clear()

    def attention(self, layer, module, key_projection):
        state = self.pending.pop(layer)
        dim = module.head_dim
        q_all = state['q'].view(1, -1, self.heads, dim).transpose(1, 2)
        k = key_projection.view(1, -1, self.model.config.num_key_value_heads, dim).transpose(1, 2)
        cos, sin = state['rope']
        k = repeat_kv(k*cos[:, None] + rotate_half(k)*sin[:, None], module.num_key_value_groups)[0]
        for begin in range(0, self.count, QUERY_BATCH):
            end = min(begin+QUERY_BATCH, self.count); pos = self.positions[begin:end]
            q = q_all[0].index_select(1, pos)
            qc, qs = cos[0].index_select(0, pos), sin[0].index_select(0, pos)
            q = q*qc[None] + rotate_half(q)*qs[None]
            weights = attention7.causal_attention_rows(q, k, pos, module.scaling)
            self.values[begin:end, layer] = attention7.lookback_from_attention(
                weights, self.context, self.start, pos).T.cpu().numpy()
        self.seen.add(layer)


@torch.inference_mode()
def extract(tokenizer, model, row, generated):
    started = time.perf_counter()
    assert not model.training and model.config.model_type == 'qwen2'
    assert len(model.model.layers) == model.config.num_attention_heads == 28
    assert not getattr(model.config, 'use_sliding_window', False)
    prefix, response = generated['input_token_ids'], generated['response_token_ids']
    offsets = attention7._response_offsets(tokenizer, response, generated['response'])
    assert np.array_equal(offsets, generated['response_token_offsets'])
    context, context_meta = attention7.visible_context_positions(tokenizer, visible(row), prefix)
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([prefix+response], device=device)
    torch.cuda.synchronize(device)
    with LookbackHooks(model, context, len(prefix), len(response)) as hooks:
        last = model.model(input_ids=ids, use_cache=False, output_attentions=False,
                           output_hidden_states=False).last_hidden_state[0]
    assert len(hooks.seen) == 28
    nll = []
    positions = torch.arange(len(prefix), len(prefix)+len(response), device=device)
    for pos in positions.split(LOGIT_BATCH):
        logp = model.lm_head(last.index_select(0, pos-1)).float().log_softmax(-1)
        nll.extend((-logp.gather(1, ids[0, pos, None])).squeeze(-1).cpu().tolist())
    arrays = {'lb': hooks.values.reshape(len(response), 784),
              'nll': np.asarray(nll, dtype=np.float32),
              'token_ids': np.asarray(response, dtype=np.int64),
              'response_token_offsets': offsets.astype(np.int32),
              'token_start': offsets[:, 0].astype(np.int32),
              'token_end': offsets[:, 1].astype(np.int32)}
    validate(arrays, generated)
    torch.cuda.synchronize(device)
    meta = {'version': VERSION, 'row_id': row['row_id'],
            'input_tokens': len(prefix), 'response_tokens': len(response),
            'input_token_ids_sha256': model7.digest(prefix),
            'response_token_ids_sha256': model7.digest(response),
            'response_sha256': model7.digest(generated['response']),
            'lb_shape': [len(response), 784], 'nll_shape': [len(response)],
            'feature_axes': 'token x (layer 1..28 major, head 0..27 minor)',
            'lookback_formula': 'mean_attention_to_sources / (mean_attention_to_sources + mean_attention_to_generated_prefix)',
            'attention_timing': 'post-read query P+j; response side includes current token j',
            'nll_timing': '-log p(saved response token j | saved input + response[:j]); logits at P+j-1',
            'query_batch': QUERY_BATCH, 'logit_batch': LOGIT_BATCH, 'causal_passes': 1,
            'backbone_attention_implementation': model.config._attn_implementation,
            'labels_read': False, 'response_regenerated': False, 'hidden_states_saved': False,
            'seconds': time.perf_counter()-started,
            'peak_allocated_gib': float(torch.cuda.max_memory_allocated(device)/2**30),
            **context_meta}
    return arrays, meta


def validate(arrays, generated):
    n = len(generated['response_token_ids'])
    for name, shape in [('lb', (n, 784)), ('nll', (n,))]:
        assert arrays[name].shape == shape and arrays[name].dtype == np.float32
        assert np.isfinite(arrays[name]).all(), name
    assert np.all((arrays['lb'] >= 0) & (arrays['lb'] <= 1))
    assert np.all(arrays['nll'] >= 0)
    assert arrays['token_ids'].tolist() == generated['response_token_ids']
    assert np.array_equal(arrays['response_token_offsets'], generated['response_token_offsets'])
    assert np.array_equal(arrays['token_start'], arrays['response_token_offsets'][:, 0])
    assert np.array_equal(arrays['token_end'], arrays['response_token_offsets'][:, 1])


def expected(row, source_digest, sig):
    return {'input_row_sha256': model7.digest(row), 'source_generation_sha256': source_digest,
            'extraction_signature_sha256': model7.digest(sig)}


def cached(row, generated, source_digest, sig):
    path = ROOT/'data/features'/(row['row_id']+'.json')
    if not path.exists(): return None
    meta = read(path)
    for k, v in expected(row, source_digest, sig).items(): assert meta[k] == v, (row['row_id'], k)
    assert sha(path.with_suffix('.npz')) == meta['arrays_sha256']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as h:
        arrays = {k: h[k] for k in h.files}
    validate(arrays, generated)
    return arrays, meta


def manifest(rows, source_manifest, sig):
    records, missing = {}, []
    for row in rows:
        g, digest = generation(row, source_manifest)
        value = cached(row, g, digest, sig)
        if value is None: missing.append(row['row_id']); continue
        _, meta = value; relative = 'data/features/'+row['row_id']
        records[row['row_id']] = {'source_generation_sha256': digest,
            'json': relative+'.json', 'json_sha256': sha(ROOT/(relative+'.json')),
            'npz': relative+'.npz', 'npz_sha256': meta['arrays_sha256'],
            'response_tokens': meta['response_tokens']}
    out = {'version': VERSION, 'expected_rows': len(rows), 'completed_rows': len(records), 'completed_count': len(records),
           'complete': not missing, 'missing_rows': missing, 'records': records,
           'total_response_tokens_completed': sum(r['response_tokens'] for r in records.values()),
           'source_root': str(SOURCE.resolve()), 'extraction_signature': sig,
           'extraction_signature_sha256': model7.digest(sig),
           'schema': {'lb': 'float32[N,784], layer-major/head-minor', 'nll': 'float32[N], natural log',
                      'token_ids': 'int64[N]', 'response_token_offsets': 'int32[N,2]',
                      'token_start': 'int32[N]', 'token_end': 'int32[N]'},
           'all_response_tokens_including_refusals_and_parse_failures': True,
           'labels_read': False, 'response_regenerated': False, 'hidden_states_saved': False}
    save(ROOT/'data/feature_manifest.json', out)
    return out


def selfcheck(tok, model, row, g, sig):
    import feature10
    arrays, _ = extract(tok, model, row, g)
    old, _ = feature10.extract_features(tok, model, visible(row), g)
    differences = {'lb': float(np.max(np.abs(arrays['lb']-old['lookback_features']))),
                   'nll': float(np.max(np.abs(arrays['nll']-old['token_nll'])))}
    assert all(v <= 1e-6 for v in differences.values()), differences
    prefix_checks, fixed_shape_checks = [], []
    for count in sorted({1, max(1, len(g['response_token_ids'])//2)}):
        short = dict(g)
        short['response_token_ids'] = g['response_token_ids'][:count]
        # Decode only exact generated prefix IDs; this is verification, not generation.
        short['response'] = tok.decode(short['response_token_ids'], clean_up_tokenization_spaces=False)
        short['response_token_offsets'] = attention7._response_offsets(tok, short['response_token_ids'], short['response']).tolist()
        local, _ = extract(tok, model, row, short)
        delta = {name: float(np.max(np.abs(local[name]-arrays[name][:count]))) for name in ('lb', 'nll')}
        prefix_checks.append({'prefix_tokens': count, 'max_absolute_differences': delta})
        # Sequence-length changes can change BF16 matrix-kernel rounding. Test
        # dependence on future CONTENT separately, with identical tensor shapes.
        altered = dict(g)
        altered['response_token_ids'] = (g['response_token_ids'][:count] +
            [g['response_token_ids'][0]] * (len(g['response_token_ids'])-count))
        altered['response'] = tok.decode(altered['response_token_ids'], clean_up_tokenization_spaces=False)
        altered['response_token_offsets'] = attention7._response_offsets(
            tok, altered['response_token_ids'], altered['response']).tolist()
        local, _ = extract(tok, model, row, altered)
        delta = {name: float(np.max(np.abs(local[name][:count]-arrays[name][:count]))) for name in ('lb', 'nll')}
        assert all(v <= 1e-6 for v in delta.values()), ('Future content affected prefix', count, delta)
        fixed_shape_checks.append({'prefix_tokens': count, 'max_absolute_differences': delta})
    old_rows = [json.loads(s) for s in (R10/'data/inputs.jsonl').read_text('utf-8').splitlines() if s.strip()]
    legacy_row = next(r for r in old_rows if r['split'] == 'train')
    gp = R10/'data/generation_records'/(legacy_row['row_id']+'.json'); legacy_g = read(gp)
    old_path = R10/'data/soft_features'/(legacy_row['row_id']+'.npz')
    old_meta = read(old_path.with_suffix('.json'))
    assert old_meta['source_generation_sha256'] == sha(gp) and old_meta['arrays_sha256'] == sha(old_path)
    legacy_arrays, _ = extract(tok, model, legacy_row, legacy_g)
    with np.load(old_path, allow_pickle=False) as old_arrays:
        legacy_delta = {'lb': float(np.max(np.abs(legacy_arrays['lb']-old_arrays['lookback_features']))),
                        'nll': float(np.max(np.abs(legacy_arrays['nll']-old_arrays['token_nll'])))}
        assert np.array_equal(legacy_arrays['token_ids'], old_arrays['token_ids'])
    assert all(v <= 1e-6 for v in legacy_delta.values()), ('Historical R10 compatibility', legacy_delta)
    save(ROOT/'data/extraction_selfcheck.json', {'passed': True, 'row_id': row['row_id'],
        'extraction_signature_sha256': model7.digest(sig), 'reference_feature10_sha256': sha(Path(feature10.__file__)),
        'reference_max_absolute_differences': differences, 'prefix_length_diagnostics': prefix_checks,
        'fixed_shape_future_content_checks': fixed_shape_checks,
        'legacy_cache_check': {'row_id': legacy_row['row_id'], 'source_generation_sha256': sha(gp),
                               'cache_npz_sha256': sha(old_path), 'max_absolute_differences': legacy_delta},
        'numerical_limitation': 'Truncating to a different sequence/query/logit batch shape can change BF16 numerical results. Strict online-prefix numerical identity is not claimed; fixed-shape future-content intervention and historical-cache compatibility are tested separately.',
        'labels_read': False, 'response_regenerated': False})
    print('SELFCHECK_PASSED', differences, 'FIXED_SHAPE', fixed_shape_checks, 'LEGACY', legacy_delta,
          'LENGTH_DIAGNOSTICS', prefix_checks, flush=True)


def run(limit=None, audit_only=False):
    rows, source_manifest = source_rows(); sig = signature()
    if audit_only:
        result = manifest(rows, source_manifest, sig)
        print('AUDIT', result['completed_rows'], result['expected_rows'], result['complete'], flush=True)
        return
    work = rows if limit is None else rows[:limit]
    tok = model = None
    for i, row in enumerate(work, 1):
        g, digest = generation(row, source_manifest)
        if cached(row, g, digest, sig) is not None: continue
        if model is None:
            tok, model = model7.load_model()
            assert getattr(model, 'is_loaded_in_4bit', False)
            check = ROOT/'data/extraction_selfcheck.json'
            if check.exists():
                assert read(check)['extraction_signature_sha256'] == model7.digest(sig)
            else: selfcheck(tok, model, row, g, sig)
        torch.cuda.reset_peak_memory_stats()
        arrays, meta = extract(tok, model, row, g)
        assert signature() == sig, 'Source or extraction code changed during replay'
        assert sha(SOURCE/'data/generation_records'/(row['row_id']+'.json')) == digest
        path = ROOT/'data/features'/(row['row_id']+'.npz'); save_arrays(path, arrays)
        meta.update(expected(row, digest, sig), arrays_sha256=sha(path), extraction_signature=sig)
        save(path.with_suffix('.json'), meta)
        print('DONE', i, len(work), row['row_id'], len(g['response_token_ids']), round(meta['seconds'], 3), flush=True)
        if i % 50 == 0: manifest(rows, source_manifest, sig)
    source_rows()
    result = manifest(rows, source_manifest, sig)
    print('COMPLETE', result['completed_rows'], result['expected_rows'], result['complete'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int); p.add_argument('--audit', action='store_true')
    args = p.parse_args(); run(args.limit, args.audit)
