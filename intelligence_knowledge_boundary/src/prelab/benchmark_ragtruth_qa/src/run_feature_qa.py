"""Explicit, verified Llama2 NF4 runner: two development GPU selfchecks only.

No all-data extraction command exists in this initial runner. Test/withheld data
are never opened. Base tokenization/features remain in the frozen feature_qa API.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

import feature_qa as feature

ROOT, MODEL = feature.ROOT, feature.MODEL
OUT = ROOT / 'data/replay_selfcheck'
VERSION = 'ragtruth-qa-llama2-nf4-gpu-selfcheck-v1'
REPO = 'NousResearch/Llama-2-7b-chat-hf'
REVISION = '351844e75ed0bcbbe3f10671b3c808d2b83894ee'
SEED, THREADS = 20260911, 4
# Prespecified FP32 reduction tolerances for the independent library oracle.
# Same-path repeated features, hidden states and coordinates require exact equality.
LB_ATOL, LB_RTOL, NLL_ATOL, NLL_RTOL = 2e-6, 1e-5, 1e-6, 1e-6
LOAD_CONFIG = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'bfloat16',
    'bnb_4bit_quant_storage': 'uint8', 'llm_int8_skip_modules': ['lm_head'],
    'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa', 'device_map': {'': 'cuda:0'},
    'local_files_only': True, 'trust_remote_code': False,
    'padding': 'one unpadded sequence and explicit all-ones mask; generation_config is not used',
    'tf32_allowed': False}


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def frozen_json(path, value):
    if path.exists():
        assert feature.read(path) == value, ('Frozen selfcheck definition changed', str(path))
    else:
        feature.save(path, value)


def prepare_signature():
    downloads = feature.read(ROOT / 'model_download_manifest.json')
    assert downloads['status'] == 'complete'
    assert downloads['repo_id'] == REPO and downloads['revision'] == REVISION
    assets = {}
    for entry in downloads['files']:
        assert entry['status'] == 'verified' and entry['source_hash_match']
        path = MODEL / entry['filename']
        assert path.parent.resolve() == MODEL.resolve()
        assert path.stat().st_size == entry['expected_bytes'] == entry['actual_bytes']
        actual = feature.sha(path)
        assert actual == entry['actual_sha256'], ('Downloaded bytes changed', entry['filename'])
        if entry.get('expected_lfs_sha256'):
            assert actual == entry['expected_lfs_sha256']
        assets[entry['filename']] = {'sha256': actual, 'bytes': path.stat().st_size}
    prepared = feature.read(ROOT / 'data/feature_preparation/manifest.json')
    assert prepared['records'] == 793 and prepared['partitions'] == feature.EXPECTED
    assert prepared['signature']['code_sha256'] == feature.sha(Path(feature.__file__))
    assert feature.sha(ROOT / 'data/feature_preparation/plans.jsonl') == prepared['plans_jsonl_sha256']
    for partition, expected in prepared['signature']['development_data_sha256'].items():
        assert partition in feature.PARTITIONS
        assert feature.sha(ROOT / 'data' / (partition + '.jsonl')) == expected
    for name, expected in prepared['signature']['tokenizer']['files_sha256'].items():
        assert assets[name]['sha256'] == expected
    plans = [json.loads(line) for line in (ROOT / 'data/feature_preparation/plans.jsonl').read_text('utf-8').splitlines() if line]
    assert len(plans) == 793 and all(p['partition'] in feature.PARTITIONS and p['official_split'] == 'train' for p in plans)
    chosen = [plans[0], plans[-1]]
    assert chosen[0]['response_id'] != chosen[1]['response_id']
    config = feature.read(MODEL / 'config.json')
    assert (config['model_type'], config['num_hidden_layers'], config['num_attention_heads'], config['hidden_size']) == ('llama', 32, 32, 4096)
    signature = {'version': VERSION, 'repo_id': REPO, 'revision': REVISION, 'checkpoint_assets': assets,
        'model_download_manifest_sha256': feature.sha(ROOT / 'model_download_manifest.json'),
        'preparation_manifest_sha256': feature.sha(ROOT / 'data/feature_preparation/manifest.json'),
        'plans_sha256': prepared['plans_jsonl_sha256'], 'tokenizer': prepared['signature']['tokenizer'],
        'source_development_files_sha256': prepared['signature']['development_data_sha256'],
        'code_sha256': {str(p.resolve()): feature.sha(p) for p in
                       (Path(__file__), Path(feature.__file__), Path(inspect.getfile(eager_attention_forward)))},
        'load_config': LOAD_CONFIG, 'seed': SEED, 'cpu_threads': THREADS,
        'software': {k: importlib.metadata.version(k) for k in
                     ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'numpy', 'safetensors', 'tokenizers')},
        'python': platform.python_version(), 'torch_cuda_runtime': torch.version.cuda,
        'selected_response_ids': [p['response_id'] for p in chosen],
        'selection': 'first and last of frozen fit-then-calibration plans; no answer or label based selection',
        'oracle_tolerances_fixed_before_run': {'lb_atol': LB_ATOL, 'lb_rtol': LB_RTOL,
            'nll_atol': NLL_ATOL, 'nll_rtol': NLL_RTOL, 'repeat_exact': True, 'hidden_exact': True},
        'test_or_withheld_content_read': False, 'labels_used': False, 'exact_original_generation_trace': False}
    frozen_json(OUT / 'signature.json', signature)
    return plans, chosen, prepared, signature


def load_nf4():
    assert torch.cuda.is_available()
    torch.set_num_threads(THREADS); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=['lm_head'])
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        trust_remote_code=False, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert getattr(model, 'is_loaded_in_4bit', False)
    assert feature.architecture(model)['model_type'] == 'llama'
    assert model.model.embed_tokens.weight.device.type == 'cuda'
    assert model.model.embed_tokens.weight.dtype == model.lm_head.weight.dtype == torch.bfloat16
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    quantization = model.config.quantization_config
    if hasattr(quantization, 'to_dict'):
        quantization = quantization.to_dict()
    first_linear = model.model.layers[0].self_attn.q_proj
    assert first_linear.weight.quant_state.quant_type == 'nf4'
    meta = {'seconds': time.perf_counter() - started, 'device': str(model.model.embed_tokens.weight.device),
        'gpu_name': properties.name, 'gpu_capability': list(torch.cuda.get_device_capability(0)),
        'total_gpu_bytes': properties.total_memory,
        'allocated_after_load_gib': torch.cuda.memory_allocated() / 2**30,
        'quantization_config': quantization,
        'quantized_linear': {'class': type(first_linear).__name__, 'weight_dtype': str(first_linear.weight.dtype),
            'compute_dtype': str(first_linear.compute_dtype), 'quant_type': first_linear.weight.quant_state.quant_type,
            'nested': bool(first_linear.weight.quant_state.nested)},
        'embedding_dtype': str(model.model.embed_tokens.weight.dtype), 'lm_head_dtype': str(model.lm_head.weight.dtype),
        'attention_backend': model.config._attn_implementation,
        'architecture': feature.architecture(model)}
    print('NF4_LOADED', json.dumps(meta), flush=True)
    return model, meta


def selected_chunks(count):
    starts = sorted({0, ((count // 2) // feature.QUERY_BATCH) * feature.QUERY_BATCH,
                     ((count - 1) // feature.QUERY_BATCH) * feature.QUERY_BATCH})
    return [list(range(start, min(start + feature.QUERY_BATCH, count))) for start in starts]


class LibraryAttentionOracle:
    """Official eager-attention rows from actual Q/K/V; no full S*S allocation.

    The library casts attention to model dtype after float32 softmax. Capture
    that precast tensor so the primary comparison matches declared feature
    precision; also report native-BF16 cast differences separately.
    """
    def __init__(self, model, plan):
        self.model, self.spec = model, feature.architecture(model)
        self.plan = plan; self.chunks = selected_chunks(len(plan['original']['answer_token_positions']))
        self.local_indices = [i for chunk in self.chunks for i in chunk]
        self.values = np.empty((len(self.local_indices), self.spec['layers'], self.spec['heads']), dtype=np.float32)
        self.native = np.empty_like(self.values)
        self.handles, self.pending, self.seen = [], {}, set()
        self.hidden = None

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            attention = block.self_attn
            def before(module, args, kwargs, layer=layer):
                assert kwargs.get('past_key_value') is None
                self.pending[layer] = {'rope': kwargs['position_embeddings']}
            def query(module, args, output, layer=layer):
                self.pending[layer]['q'] = output
            def key(module, args, output, layer=layer):
                self.pending[layer]['k'] = output
            def value(module, args, output, layer=layer, attention=attention):
                self.reference(layer, attention, output)
            self.handles.extend((attention.register_forward_pre_hook(before, with_kwargs=True),
                attention.q_proj.register_forward_hook(query), attention.k_proj.register_forward_hook(key),
                attention.v_proj.register_forward_hook(value)))
        def final_norm(module, args, output):
            positions = torch.tensor(self.plan['original']['answer_token_positions'], device=output.device)
            self.hidden = output[0].index_select(0, positions).float().cpu().numpy()
        self.handles.append(self.model.model.norm.register_forward_hook(final_norm))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def reference(self, layer, module, values):
        state = self.pending.pop(layer)
        heads, kv, dim = self.spec['heads'], self.spec['kv_heads'], self.spec['head_dim']
        query = state['q'].view(1, -1, heads, dim).transpose(1, 2)
        key = state['k'].view(1, -1, kv, dim).transpose(1, 2)
        value = values.view(1, -1, kv, dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *state['rope'])
        answer_positions = torch.tensor(self.plan['original']['answer_token_positions'], device=query.device)
        source_positions = torch.tensor(self.plan['original']['context_token_positions'], device=query.device)
        key_positions = torch.arange(key.shape[-2], device=query.device)
        offset = 0
        for indices in self.chunks:
            local = torch.tensor(indices, device=query.device)
            pos = answer_positions.index_select(0, local)
            q = query.index_select(2, pos)
            mask = torch.zeros((1, 1, len(pos), key.shape[-2]), dtype=q.dtype, device=q.device)
            mask.masked_fill_(key_positions[None, None, None] > pos[None, None, :, None], float('-inf'))
            captured = []
            original_softmax = F.softmax
            def capture(*args, **kwargs):
                result = original_softmax(*args, **kwargs)
                captured.append(result)
                return result
            with patch.object(F, 'softmax', capture):
                _, native_attention = eager_attention_forward(module, q, key, value, mask,
                                                               scaling=module.scaling, dropout=0.0)
            assert len(captured) == 1 and captured[0].dtype == torch.float32
            def ratio(attention):
                w = attention[0].float()
                means = w.index_select(-1, source_positions).mean(-1)
                answer = w.index_select(-1, answer_positions)
                previous = answer_positions[None] <= pos[:, None]
                previous_mean = (answer * previous[None]).sum(-1) / previous.sum(-1)[None]
                return (means / (means + previous_mean)).T.cpu().numpy()
            self.values[offset:offset + len(pos), layer] = ratio(captured[0])
            self.native[offset:offset + len(pos), layer] = ratio(native_attention)
            offset += len(pos)
        self.seen.add(layer)


@torch.inference_mode()
def independent_reference(model, plan):
    started = time.perf_counter()
    ids, mask, positions = feature.tensor_input(model, plan['original'])
    # Exact first LOGIT_BATCH-sized block, matching the declared batch size;
    # use the model's causal-LM forward rather than reimplementing its lm_head path.
    nll_local = np.arange(min(len(positions), feature.LOGIT_BATCH))
    chosen_positions = positions[:len(nll_local)]
    with LibraryAttentionOracle(model, plan) as oracle:
        output = model(input_ids=ids, attention_mask=mask, use_cache=False,
            output_attentions=False, output_hidden_states=False,
            logits_to_keep=chosen_positions - 1, return_dict=True)
    assert len(oracle.seen) == feature.architecture(model)['layers'] and oracle.hidden is not None
    logp = output.logits[0].float().log_softmax(-1)
    nll = -logp.gather(1, ids[0, chosen_positions, None]).squeeze(-1).cpu().numpy()
    torch.cuda.synchronize()
    return {'lb': oracle.values.reshape(len(oracle.local_indices), -1),
        'native_cast_lb': oracle.native.reshape(len(oracle.local_indices), -1),
        'lb_local_indices': np.asarray(oracle.local_indices, dtype=np.int64),
        'nll': nll, 'nll_local_indices': nll_local,
        'hidden_last': oracle.hidden, 'seconds': time.perf_counter() - started}


@torch.inference_mode()
def noctx_hidden(model, plan):
    assert plan['no_context']['answer_token_ids'] == plan['original']['answer_token_ids']
    ids, mask, positions = feature.tensor_input(model, plan['no_context'])
    torch.cuda.synchronize(); started = time.perf_counter()
    final = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                         output_attentions=False, output_hidden_states=False).last_hidden_state[0]
    result = final.index_select(0, positions).float().cpu().numpy()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def check_one(model, plan, signature):
    assert plan['partition'] in feature.PARTITIONS
    rid = plan['response_id']; print('SELFCHECK_START', rid, plan['partition'], flush=True)
    arrays, meta = feature.extract_features(model, plan)
    repeat, repeat_meta = feature.extract_features(model, plan)
    repeated = {key: float(np.max(np.abs(arrays[key].astype(np.float64) - repeat[key].astype(np.float64))))
                for key in arrays}
    assert all(np.array_equal(arrays[key], repeat[key]) for key in arrays), ('Same-input repeat differs', rid, repeated)
    reference = independent_reference(model, plan)
    lb = arrays['lb'][reference['lb_local_indices']]
    nll = arrays['nll'][reference['nll_local_indices']]
    lb_difference = float(np.max(np.abs(lb - reference['lb'])))
    nll_difference = float(np.max(np.abs(nll - reference['nll'])))
    hidden_difference = float(np.max(np.abs(arrays['hidden_last'] - reference['hidden_last'])))
    assert np.allclose(lb, reference['lb'], atol=LB_ATOL, rtol=LB_RTOL), ('Library LB mismatch', rid, lb_difference)
    assert np.allclose(nll, reference['nll'], atol=NLL_ATOL, rtol=NLL_RTOL), ('Causal-LM NLL mismatch', rid, nll_difference)
    assert np.array_equal(arrays['hidden_last'], reference['hidden_last']), ('Final norm mismatch', rid, hidden_difference)
    h0, noctx_seconds = noctx_hidden(model, plan)
    h0_repeat, noctx_repeat_seconds = noctx_hidden(model, plan)
    assert np.array_equal(h0, h0_repeat), ('No-context hidden repeat differs', rid)
    arrays['hidden_no_context'] = h0
    arrays['hidden_delta'] = arrays['hidden_last'] - h0
    assert np.isfinite(arrays['hidden_delta']).all()
    checks = {'passed': True, 'response_id': rid, 'partition': plan['partition'],
        'signature_sha256': feature.digest(signature), 'plan_sha256': feature.digest(plan),
        'base_feature_metadata': meta, 'repeat_base_feature_seconds': repeat_meta['seconds'],
        'repeat_max_absolute_differences': repeated,
        'library_oracle': {'lb_max_absolute_difference': lb_difference,
            'nll_max_absolute_difference': nll_difference, 'hidden_max_absolute_difference': hidden_difference,
            'lb_query_local_indices': reference['lb_local_indices'].tolist(),
            'nll_query_local_indices': reference['nll_local_indices'].tolist(),
            'native_bf16_cast_lb_max_difference': float(np.abs(reference['native_cast_lb'] - reference['lb']).max()),
            'softmax_precision_policy': 'Compare float32 softmax before official eager casts to BF16; native cast effect reported separately',
            'seconds': reference['seconds']},
        'no_context_hidden_repeat_exact': True, 'no_context_seconds': noctx_seconds,
        'no_context_repeat_seconds': noctx_repeat_seconds,
        'answer_ids_and_clipped_offsets_same_both_views': plan['original']['answer_token_ids'] == plan['no_context']['answer_token_ids'] and plan['original']['response_token_offsets'] == plan['no_context']['response_token_offsets'],
        'boundary_crossing_positions_retained': plan['original']['prefix_boundary_crossing_token_positions'],
        'raw_first_answer_offset': plan['original']['response_token_offsets_raw'][0],
        'clipped_first_answer_offset': plan['original']['response_token_offsets'][0],
        'labels_used': False, 'test_content_read': False, 'original_trace_claimed': False}
    path = OUT / (rid + '.npz'); save_npz(path, arrays)
    save_npz(OUT / (rid + '_library_oracle.npz'), {k: v for k, v in reference.items() if isinstance(v, np.ndarray)})
    checks['arrays_sha256'] = feature.sha(path); checks['arrays_bytes'] = path.stat().st_size
    checks['raw_array_bytes'] = sum(value.nbytes for value in arrays.values())
    feature.save(OUT / (rid + '.json'), checks)
    print('SELFCHECK_PASSED', rid, 'lb', lb_difference, 'nll', nll_difference,
          'hidden', hidden_difference, 'base_seconds', round(meta['seconds'], 3),
          'noctx_seconds', round(noctx_seconds, 3), flush=True)
    return checks


def budget(plans, prepared, checks):
    total_tokens = sum(len(plan['original']['answer_token_positions']) for plan in plans)
    total_rows = len(plans); width, lb_width, rank = 4096, 1024, 256
    measured_tokens = sum(item['base_feature_metadata']['response_tokens'] for item in checks)
    base_seconds = sum(item['base_feature_metadata']['seconds'] for item in checks)
    noctx_seconds = sum(item['no_context_seconds'] for item in checks)
    raw = {'lb': total_tokens * lb_width * 4, 'nll': total_tokens * 4,
        'hidden_last': total_tokens * width * 4, 'hidden_no_context': total_tokens * width * 4,
        'hidden_delta': total_tokens * width * 4, 'harp256': total_tokens * rank * 4,
        'coordinate_arrays': total_tokens * (8 + 8 + 8 + 8 + 4 + 4)}
    base_token_scaled = base_seconds * total_tokens / measured_tokens
    base_row_scaled = base_seconds * total_rows / len(checks)
    noctx_token_scaled = noctx_seconds * total_tokens / measured_tokens
    noctx_row_scaled = noctx_seconds * total_rows / len(checks)
    return {'development_rows': total_rows, 'development_response_tokens': total_tokens,
        'measured_rows': len(checks), 'measured_response_tokens': measured_tokens,
        'base_forward_feature_seconds_measured': base_seconds, 'noctx_forward_seconds_measured': noctx_seconds,
        'base_point_estimate_seconds_by_response_tokens': base_token_scaled,
        'base_point_estimate_seconds_by_rows': base_row_scaled,
        'base_planning_range_seconds': [0.75 * min(base_token_scaled, base_row_scaled), 1.5 * max(base_token_scaled, base_row_scaled)],
        'additional_noctx_point_estimate_seconds_by_response_tokens': noctx_token_scaled,
        'additional_noctx_point_estimate_seconds_by_rows': noctx_row_scaled,
        'additional_noctx_planning_range_seconds': [0.75 * min(noctx_token_scaled, noctx_row_scaled), 1.5 * max(noctx_token_scaled, noctx_row_scaled)],
        'range_is_not_a_statistical_confidence_interval': True,
        'timing_excludes': ['checkpoint verification/loading', 'most disk serialization/hashing', 'HARP basis/projection', 'classifier training'],
        'raw_bytes_by_block': raw,
        'raw_base_total_bytes': sum(raw[k] for k in ('lb', 'nll', 'hidden_last', 'coordinate_arrays')),
        'raw_base_plus_delta_harp_total_bytes': sum(raw.values()),
        'two_row_npz_raw_ratio_with_delta_no_harp': sum(c['arrays_bytes'] for c in checks) / sum(c['raw_array_bytes'] for c in checks),
        'limitations': 'Two selected examples do not span all sequence lengths. Attention work depends on prompt and answer length; compression and GPU timing vary. These are planning estimates, not measured all-data costs.'}


def main():
    started = time.perf_counter(); model = None
    plans, chosen, prepared, signature = prepare_signature()
    try:
        model, loading = load_nf4()
        checks = [check_one(model, plan, signature) for plan in chosen]
        # Revalidate fixed source/code signatures without another expensive weight scan.
        for path, expected in signature['code_sha256'].items():
            assert feature.sha(Path(path)) == expected
        assert feature.sha(ROOT / 'data/feature_preparation/plans.jsonl') == signature['plans_sha256']
        result = {'passed': True, 'version': VERSION, 'signature_sha256': feature.digest(signature),
            'model_loading': loading, 'checks': checks, 'budget': budget(plans, prepared, checks),
            'wall_seconds_including_verification_loading_and_all_selfchecks': time.perf_counter() - started,
            'peak_allocated_gpu_gib_base_checks': max(c['base_feature_metadata']['peak_allocated_gpu_gib'] for c in checks),
            'all_793_backbone_features_extracted': False, 'gpu_rows_selected': len(chosen),
            'test_or_withheld_content_read': False, 'labels_used': False,
            'released_text_rewritten': False, 'exact_original_generation_trace': False}
        feature.save(OUT / 'manifest.json', result)
        print('QA_GPU_SELFCHECK_COMPLETE', json.dumps({'passed': True,
            'response_ids': signature['selected_response_ids'], 'loading_seconds': loading['seconds'],
            'wall_seconds': result['wall_seconds_including_verification_loading_and_all_selfchecks'],
            'peak_base_gib': result['peak_allocated_gpu_gib_base_checks'], 'budget': result['budget']}), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selfcheck-only', action='store_true', required=True)
    parser.parse_args()
    main()
