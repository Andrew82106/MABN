"""Development-only RAGTruth QA tokenization and architecture-aware replay API.

CLI actions are CPU-only preparation/checks. No pretrained model loader or GPU
runner is exposed here. Later orchestration may pass a separately verified NF4
Llama model to extract_features. This is disclosed teacher-forced reconstruction,
never the original published generation trace. Labels never enter this module's
model-visible records or tokenization decisions; official test is not opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / 'models/Llama-2-7b-chat-hf'
VERSION = 'ragtruth-qa-reconstruction-layout-v1'
PARTITIONS = ('fit', 'calibration')
EXPECTED = {'fit': 634, 'calibration': 159}
QUERY_BATCH, LOGIT_BATCH, THREADS = 8, 16, 4
WRAPPER_LEFT, WRAPPER_RIGHT = '<s>[INST] ', ' [/INST] '
VISIBLE_FIELDS = ('source_id', 'group_id', 'partition', 'official_split', 'response_id',
    'model', 'question', 'retrieved_passages', 'released_prompt', 'original_response',
    'answer_sha256', 'prompt_sha256')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def assert_cpu_only():
    assert not torch.cuda.is_initialized(), 'Preparation/check commands must not initialize CUDA'


def development_rows():
    """Read only exported fit/calibration; strip labels before handing rows on."""
    manifest = read(ROOT / 'data/development_manifest.json')
    rows = []
    for partition in PARTITIONS:
        path = ROOT / 'data' / (partition + '.jsonl')
        expected_hash = manifest['files_sha256'].get('data\\' + path.name,
                                                    manifest['files_sha256'].get('data/' + path.name))
        assert expected_hash is not None and sha(path) == expected_hash
        part = []
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                # Development exports contain annotations, but their values are
                # never inspected, passed to planning, or used to select tokens.
                raw = json.loads(line)
                assert raw['partition'] == partition and raw['official_split'] == 'train'
                assert raw['quality'] == 'good', 'Do not redefine locked quality eligibility'
                visible = {key: raw[key] for key in VISIBLE_FIELDS}
                assert visible['model'] == 'llama-2-7b-chat'
                assert digest(visible['released_prompt']) == visible['prompt_sha256']
                assert digest(visible['original_response']) == visible['answer_sha256']
                part.append(visible)
        assert len(part) == EXPECTED[partition]
        rows.extend(part)
    assert len({r['response_id'] for r in rows}) == sum(EXPECTED.values())
    assert not ({r['group_id'] for r in rows if r['partition'] == 'fit'} &
                {r['group_id'] for r in rows if r['partition'] == 'calibration'})
    return rows, manifest


def tokenizer_signature(tokenizer):
    names = ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
             'special_tokens_map.json', 'config.json')
    return {'tokenizer_class': type(tokenizer).__name__, 'is_fast': tokenizer.is_fast,
        'vocab_size': tokenizer.vocab_size, 'length_with_added_tokens': len(tokenizer),
        'files_sha256': {name: sha(MODEL / name) for name in names},
        'transformers_version': importlib.metadata.version('transformers'),
        'tokenizers_version': importlib.metadata.version('tokenizers'),
        'explicit_wrapper': WRAPPER_LEFT + '{released_prompt}' + WRAPPER_RIGHT + '{original_response}',
        'add_special_tokens': False, 'add_eos': False, 'padding': False, 'truncation': False,
        'historical_original_token_ids_available': False}


def encode_view(tokenizer, prompt, response, reference_range=None):
    """Encode the complete text once, preserving answer-boundary crossing tokens.

    raw answer offsets may begin at -1 because the separator space shares the
    first answer token. Clipped offsets retain exactly its answer intersection.
    Duplicate/overlapping offsets (e.g. byte fallback) are not discarded.
    """
    assert tokenizer.is_fast and isinstance(prompt, str) and isinstance(response, str) and response
    prefix = WRAPPER_LEFT + prompt + WRAPPER_RIGHT
    text = prefix + response
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        return_attention_mask=True, padding=False, truncation=False)
    ids = list(map(int, encoded['input_ids']))
    offsets = np.asarray(encoded['offset_mapping'], dtype=np.int32)
    assert len(ids) == len(offsets) and offsets.shape == (len(ids), 2)
    assert ids[0] == tokenizer.bos_token_id, 'One literal BOS is part of the released wrapper'
    assert len(ids) > 1 and ids[1] != tokenizer.bos_token_id, 'Unexpected duplicated leading BOS'
    assert list(encoded['attention_mask']) == [1] * len(ids), 'Single unpadded sequence required'
    assert np.all(offsets[:, 0] >= 0) and np.all(offsets[:, 1] <= len(text))
    assert np.all(offsets[:, 1] >= offsets[:, 0])
    start, end = len(prefix), len(text)
    answer_positions = np.flatnonzero((offsets[:, 1] > start) &
                                     (offsets[:, 0] < end) & (offsets[:, 1] > offsets[:, 0]))
    assert len(answer_positions) > 0 and answer_positions[0] > 0
    raw_offsets = offsets[answer_positions] - start
    clipped = np.column_stack((np.maximum(raw_offsets[:, 0], 0),
                               np.minimum(raw_offsets[:, 1], len(response)))).astype(np.int32)
    assert np.all(clipped[:, 1] > clipped[:, 0])
    covered = np.zeros(len(response), dtype=bool)
    for left, right in clipped:
        covered[left:right] = True
    nonspace = np.fromiter((not char.isspace() for char in response), dtype=bool)
    assert np.all(covered | ~nonspace), 'Tokenizer offsets lost non-whitespace answer characters'
    context_positions = []
    rendered_reference = None
    if reference_range is not None:
        begin, finish = reference_range
        assert 0 <= begin < finish <= len(prompt)
        rendered_reference = [len(WRAPPER_LEFT) + begin, len(WRAPPER_LEFT) + finish]
        context_positions = np.flatnonzero((offsets[:, 1] > rendered_reference[0]) &
            (offsets[:, 0] < rendered_reference[1]) & (offsets[:, 1] > offsets[:, 0])).tolist()
        assert context_positions and max(context_positions) < int(answer_positions[0])
    return {'input_ids': ids, 'attention_mask': list(encoded['attention_mask']),
        'input_token_offsets': offsets.tolist(), 'answer_token_positions': answer_positions.tolist(),
        'answer_token_ids': [ids[i] for i in answer_positions],
        'response_token_offsets_raw': raw_offsets.tolist(),
        'response_token_offsets': clipped.tolist(),
        'context_token_positions': context_positions,
        'rendered_reference_character_range': rendered_reference,
        'answer_character_range': [start, end], 'prefix_character_length': start,
        'prefix_boundary_crossing_token_positions': answer_positions[raw_offsets[:, 0] < 0].tolist(),
        'uncovered_whitespace_characters': int((~covered & ~nonspace).sum()),
        'rendered_text_sha256': digest(text), 'input_ids_sha256': digest(ids),
        'response_text_sha256': digest(response),
        'offset_policy': 'Full-string tokenizer offsets intersected with original answer; no text normalization or independent answer tokenization'}


def prepare_row(tokenizer, row, include_no_context=True):
    assert row['partition'] in PARTITIONS and row['official_split'] == 'train'
    prompt, response, passages = row['released_prompt'], row['original_response'], row['retrieved_passages']
    assert passages and prompt.count(passages) == 1, 'Released three-passage string must occur uniquely'
    begin = prompt.index(passages); end = begin + len(passages)
    original = encode_view(tokenizer, prompt, response, reference_range=(begin, end))
    noctx = None
    if include_no_context:
        # Keep every instruction, including the official refusal instruction;
        # only remove the exact released passage block, never rewrite the query.
        noctx_prompt = prompt[:begin] + prompt[end:]
        noctx = encode_view(tokenizer, noctx_prompt, response)
        assert noctx['answer_token_ids'] == original['answer_token_ids'], 'Counterfactual answer segmentation changed'
        assert noctx['response_token_offsets'] == original['response_token_offsets']
        noctx['released_prompt_without_passages'] = noctx_prompt
        noctx['removed_passages_sha256'] = digest(passages)
    return {'version': VERSION, 'response_id': row['response_id'], 'source_id': row['source_id'],
        'group_id': row['group_id'], 'partition': row['partition'], 'official_split': 'train',
        'released_prompt': prompt, 'original_response': response,
        'prompt_sha256': digest(prompt), 'answer_sha256': digest(response),
        'original': original, 'no_context': noctx,
        'labels_used': False, 'exact_original_generation_trace': False,
        'replay_definition': 'Fixed wrapper/tokenizer/checkpoint/precision teacher-forced reconstruction of published text'}


def architecture(model):
    config = model.config
    assert config.model_type in ('llama', 'qwen2'), 'Only audited rotary Llama/Qwen2 adapters are supported'
    layers, heads = len(model.model.layers), int(config.num_attention_heads)
    kv_heads = int(getattr(config, 'num_key_value_heads', heads))
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, 'head_dim', None) or hidden // heads)
    assert layers == config.num_hidden_layers and heads % kv_heads == 0
    assert head_dim % 2 == 0 and hidden == heads * head_dim
    assert not getattr(config, 'use_sliding_window', False), 'Sliding-window attention needs a separate adapter'
    for block in model.model.layers:
        attention = block.self_attn
        assert attention.head_dim == head_dim
        assert attention.num_key_value_groups == heads // kv_heads
        assert not hasattr(attention, 'q_norm') and not hasattr(attention, 'k_norm')
    return {'model_type': config.model_type, 'layers': layers, 'heads': heads,
        'kv_heads': kv_heads, 'head_dim': head_dim, 'hidden_size': hidden,
        'lb_width': layers * heads, 'vocab_size': int(config.vocab_size)}


def rotate_half(value):
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class LookbackHooks:
    """Bounded query-row recomputation, dynamic layers/heads/grouped KV heads."""
    def __init__(self, model, plan, query_batch=QUERY_BATCH):
        self.model, self.spec, self.query_batch = model, architecture(model), query_batch
        self.device = model.model.embed_tokens.weight.device
        view = plan['original']
        self.positions = torch.tensor(view['answer_token_positions'], device=self.device, dtype=torch.long)
        self.context = torch.tensor(view['context_token_positions'], device=self.device, dtype=torch.long)
        assert self.context.numel() > 0
        self.values = np.empty((len(self.positions), self.spec['layers'], self.spec['heads']), dtype=np.float32)
        self.handles, self.pending, self.seen = [], {}, set()

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            attention = block.self_attn
            def before(module, args, kwargs, layer=layer):
                assert kwargs.get('past_key_value') is None, 'Replay must disable KV cache'
                assert 'position_embeddings' in kwargs, 'Unexpected Transformers rotary API'
                self.pending[layer] = {'rope': kwargs['position_embeddings']}
            def query(module, args, output, layer=layer):
                self.pending[layer]['q'] = output
            def key(module, args, output, layer=layer, attention=attention):
                self.capture(layer, attention, output)
            self.handles.extend((attention.register_forward_pre_hook(before, with_kwargs=True),
                                 attention.q_proj.register_forward_hook(query),
                                 attention.k_proj.register_forward_hook(key)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def capture(self, layer, module, keys):
        state = self.pending.pop(layer)
        heads, kv, dim = self.spec['heads'], self.spec['kv_heads'], self.spec['head_dim']
        assert keys.shape[0] == state['q'].shape[0] == 1
        query = state['q'].view(1, -1, heads, dim).transpose(1, 2)
        key = keys.view(1, -1, kv, dim).transpose(1, 2)
        cos, sin = state['rope']
        key = key * cos[:, None] + rotate_half(key) * sin[:, None]
        key = key.repeat_interleave(heads // kv, dim=1)[0]
        absolute = torch.arange(key.shape[-2], device=self.device)
        for begin in range(0, len(self.positions), self.query_batch):
            end = min(begin + self.query_batch, len(self.positions)); pos = self.positions[begin:end]
            q = query[0].index_select(1, pos)
            qc, qs = cos[0].index_select(0, pos), sin[0].index_select(0, pos)
            q = q * qc[None] + rotate_half(q) * qs[None]
            logits = (q @ key.transpose(-2, -1)) * module.scaling
            logits.masked_fill_(absolute[None, None] > pos[None, :, None], float('-inf'))
            weights = logits.float().softmax(-1)
            source_mean = weights.index_select(-1, self.context).mean(-1)
            # Explicit answer positions handle a boundary-crossing token and
            # any non-contiguous offsets without silently treating padding as text.
            answer_attention = weights.index_select(-1, self.positions)
            previous = self.positions[None, :] <= pos[:, None]
            answer_mean = (answer_attention * previous[None]).sum(-1) / previous.sum(-1)[None]
            ratios = source_mean / (source_mean + answer_mean).clamp_min(torch.finfo(torch.float32).tiny)
            self.values[begin:end, layer] = ratios.T.cpu().numpy()
        self.seen.add(layer)


def tensor_input(model, view):
    ids = torch.tensor([view['input_ids']], device=model.model.embed_tokens.weight.device, dtype=torch.long)
    assert int(ids.max()) < model.model.embed_tokens.num_embeddings and int(ids.min()) >= 0
    mask = torch.ones_like(ids)
    assert view['attention_mask'] == [1] * ids.shape[1]
    positions = torch.tensor(view['answer_token_positions'], device=ids.device, dtype=torch.long)
    assert ids[0, positions].tolist() == view['answer_token_ids']
    assert ids.shape[1] <= model.config.max_position_embeddings, 'Do not silently truncate evidence or answer'
    return ids, mask, positions


@torch.inference_mode()
def extract_features(model, plan, harp_components=None, include_delta=False):
    """Return unpooled raw answer-token arrays and model/timing metadata.

    Caller must supply a verified, frozen model. Optional components are
    [rank, hidden_size] from THIS model head, not the Qwen3584 basis.
    """
    assert plan['partition'] in PARTITIONS and plan['official_split'] == 'train'
    assert not model.training
    spec = architecture(model); started = time.perf_counter()
    ids, mask, positions = tensor_input(model, plan['original'])
    if ids.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(ids.device)
    with LookbackHooks(model, plan) as hooks:
        final = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                            output_attentions=False, output_hidden_states=False).last_hidden_state[0]
    assert len(hooks.seen) == spec['layers']
    hidden = final.index_select(0, positions).float().cpu().numpy()
    nll = []
    for pos in positions.split(LOGIT_BATCH):
        assert torch.all(pos > 0)
        logp = model.lm_head(final.index_select(0, pos - 1)).float().log_softmax(-1)
        nll.extend((-logp.gather(1, ids[0, pos, None])).squeeze(-1).cpu().tolist())
    offsets = np.asarray(plan['original']['response_token_offsets'], dtype=np.int32)
    arrays = {'lb': hooks.values.reshape(len(positions), spec['lb_width']),
        'nll': np.asarray(nll, dtype=np.float32), 'hidden_last': hidden,
        'token_ids': np.asarray(plan['original']['answer_token_ids'], dtype=np.int64),
        'answer_token_positions': np.asarray(plan['original']['answer_token_positions'], dtype=np.int64),
        'response_token_offsets': offsets,
        'response_token_offsets_raw': np.asarray(plan['original']['response_token_offsets_raw'], dtype=np.int32),
        'token_start': offsets[:, 0].copy(), 'token_end': offsets[:, 1].copy()}
    del final
    if harp_components is not None:
        components = np.asarray(harp_components)
        assert components.ndim == 2 and components.shape[1] == spec['hidden_size']
        assert 0 < components.shape[0] < spec['hidden_size'] and np.isfinite(components).all()
        with threadpool_limits(limits=THREADS):
            arrays['harp'] = (hidden.astype(np.float64) @ components.astype(np.float64).T).astype(np.float32)
    if include_delta:
        assert plan['no_context'] is not None
        noctx_ids, noctx_mask, noctx_positions = tensor_input(model, plan['no_context'])
        assert plan['no_context']['answer_token_ids'] == plan['original']['answer_token_ids']
        noctx_final = model.model(input_ids=noctx_ids, attention_mask=noctx_mask, use_cache=False,
            output_attentions=False, output_hidden_states=False).last_hidden_state[0]
        noctx = noctx_final.index_select(0, noctx_positions).float().cpu().numpy()
        arrays['hidden_no_context'] = noctx
        arrays['hidden_delta'] = hidden - noctx
        del noctx_final
    assert arrays['lb'].shape[0] == hidden.shape[0] == len(positions)
    assert hidden.shape[1] == spec['hidden_size']
    for key in ('lb', 'nll', 'hidden_last', 'harp', 'hidden_no_context', 'hidden_delta'):
        if key in arrays:
            assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all(), key
    assert np.all((arrays['lb'] >= 0) & (arrays['lb'] <= 1)) and np.all(arrays['nll'] >= 0)
    if ids.device.type == 'cuda':
        torch.cuda.synchronize(ids.device)
    meta = {'version': VERSION, 'response_id': plan['response_id'], 'partition': plan['partition'],
        'architecture': spec, 'plan_sha256': digest(plan), 'total_input_tokens': ids.shape[1],
        'response_tokens': len(positions), 'schema_shapes': {key: list(value.shape) for key, value in arrays.items()},
        'lb_timing': 'post-read actual answer token position; source attention mean divided by source+answer attention means, current answer token included',
        'nll_timing': 'pre-read position i-1 predicts actual reconstructed token at absolute position i',
        'hidden_timing': 'post-read final model RMSNorm output',
        'precision': {'model_parameter_dtype': str(next(model.parameters()).dtype),
            'is_loaded_in_4bit': bool(getattr(model, 'is_loaded_in_4bit', False)),
            'attention_softmax_for_lb': 'float32', 'feature_storage': 'float32'},
        'harp_components_float64_sha256': hashlib.sha256(np.asarray(harp_components, dtype=np.float64).tobytes(order='C')).hexdigest() if harp_components is not None else None,
        'harp_basis_model_binding_required_from_caller': harp_components is not None,
        'padding': 'one unpadded example; explicit all-ones attention mask, no inherited generation pad id',
        'seconds': time.perf_counter() - started,
        'peak_allocated_gpu_gib': float(torch.cuda.max_memory_allocated(ids.device) / 2**30) if ids.device.type == 'cuda' else None,
        'labels_used': False, 'response_regenerated': False, 'exact_original_generation_trace': False,
        'optional_delta_changes_context_length_and_positions': bool(include_delta),
        'future_token_smoothing_used': False}
    return arrays, meta


def prepare_development():
    assert_cpu_only(); rows, source_manifest = development_rows()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True, trust_remote_code=False)
    signature = {'version': VERSION, 'code_sha256': sha(Path(__file__)),
        'development_manifest_sha256': sha(ROOT / 'data/development_manifest.json'),
        'development_data_sha256': {p: sha(ROOT / 'data' / (p + '.jsonl')) for p in PARTITIONS},
        'tokenizer': tokenizer_signature(tokenizer), 'backbone_weights_frozen': False,
        'test_content_read': False, 'labels_used_for_features': False}
    plans = [prepare_row(tokenizer, row) for row in rows]
    output = ROOT / 'data/feature_preparation'; output.mkdir(parents=True, exist_ok=True)
    plans_path = output / 'plans.jsonl'
    plans_path.write_text(''.join(json.dumps(plan, ensure_ascii=False) + '\n' for plan in plans), encoding='utf-8')
    report = {'status': 'tokenization_prepared_no_pretrained_model_features', 'version': VERSION,
        'records': len(plans), 'partitions': EXPECTED,
        'response_tokens': sum(len(p['original']['answer_token_positions']) for p in plans),
        'max_input_tokens': max(len(p['original']['input_ids']) for p in plans),
        'boundary_crossing_records': sum(bool(p['original']['prefix_boundary_crossing_token_positions']) for p in plans),
        'noctx_answer_ids_and_clipped_offsets_match_all': True,
        'raw_negative_offsets_preserved_clipped_eval_offsets_exported': True,
        'no_nonwhitespace_answer_character_lost': True, 'no_test_or_withheld_rows_read': True,
        'no_pretrained_model_loaded': True, 'gpu_initialized': torch.cuda.is_initialized(),
        'plans_jsonl_sha256': sha(plans_path), 'signature': signature,
        'official_quality_filter_not_redefined': True,
        'released_prompt_including_original_refusal_instruction_preserved': True,
        'remaining_gate': 'Downloaded weights, NF4 loading configuration, GPU selfcheck and replay signature must be frozen separately before feature extraction'}
    save(output / 'manifest.json', report)
    assert_cpu_only(); print(json.dumps({k: v for k, v in report.items() if k != 'signature'}, ensure_ascii=False), flush=True)


def cpu_shape_check():
    """Tiny random CPU Llama and Qwen2 vs dense eager-attention/NLL oracles."""
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    assert_cpu_only(); torch.set_num_threads(THREADS); torch.manual_seed(20260911)
    reports = []
    for family, config_class, model_class, width, heads, kv, layers in (
        ('llama', LlamaConfig, LlamaForCausalLM, 48, 6, 2, 3),
        ('qwen2', Qwen2Config, Qwen2ForCausalLM, 64, 8, 2, 2)):
        config = config_class(vocab_size=128, hidden_size=width, intermediate_size=96,
            num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
            max_position_embeddings=128, attention_dropout=0.0,
            use_sliding_window=False, sliding_window=None)
        config._attn_implementation = 'eager'
        model = model_class(config).cpu().eval()
        ids = [1, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        pos = [8, 9, 10, 11, 12]; context = [2, 3, 4]
        offsets = [[i, i + 1] for i in range(5)]
        original = {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'answer_token_positions': pos, 'answer_token_ids': [ids[i] for i in pos],
            'response_token_offsets': offsets, 'response_token_offsets_raw': [[-1, 1]] + offsets[1:],
            'context_token_positions': context}
        noctx_ids = ids[:2] + ids[5:]
        noctx = {**original, 'input_ids': noctx_ids, 'attention_mask': [1] * len(noctx_ids),
                 'answer_token_positions': [i - 3 for i in pos], 'context_token_positions': []}
        plan = {'partition': 'fit', 'official_split': 'train', 'response_id': 'synthetic_' + family,
                'original': original, 'no_context': noctx}
        components = np.eye(width, dtype=np.float64)[:7]
        arrays, meta = extract_features(model, plan, harp_components=components, include_delta=True)
        with torch.inference_mode():
            full = torch.tensor([ids]); positions = torch.tensor(pos)
            dense = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False,
                          output_attentions=True, output_hidden_states=True)
            oracle = []
            for attention in dense.attentions:
                w = attention[0].index_select(1, positions)
                source_mean = w[:, :, context].mean(-1)
                mask = positions[None, :] <= positions[:, None]
                answer_mean = (w[:, :, positions] * mask[None]).sum(-1) / mask.sum(-1)[None]
                oracle.append((source_mean / (source_mean + answer_mean)).T.cpu().numpy())
            expected_lb = np.stack(oracle, axis=1).reshape(5, layers * heads)
            logp = dense.logits[0, positions - 1].float().log_softmax(-1)
            expected_nll = -logp.gather(1, full[0, positions, None]).squeeze(-1).numpy()
        lb_error = float(np.abs(expected_lb - arrays['lb']).max())
        nll_error = float(np.abs(expected_nll - arrays['nll']).max())
        assert np.allclose(expected_lb, arrays['lb'], atol=2e-7, rtol=1e-6), lb_error
        assert np.allclose(expected_nll, arrays['nll'], atol=1e-6, rtol=1e-6), nll_error
        assert np.array_equal(dense.hidden_states[-1][0, positions].numpy(), arrays['hidden_last'])
        assert np.array_equal(arrays['hidden_delta'], arrays['hidden_last'] - arrays['hidden_no_context'])
        assert np.array_equal(arrays['harp'], arrays['hidden_last'][:, :7])
        reports.append({'family': family, 'architecture': meta['architecture'],
            'shapes': meta['schema_shapes'], 'dense_attention_lb_max_abs_error': lb_error,
            'dense_nll_max_abs_error': nll_error, 'final_hidden_exact': True,
            'delta_exact': True, 'harp_projection_exact': True})
        del model, dense
    value = {'passed': True, 'version': VERSION, 'checks': reports,
        'tiny_random_cpu_models_only': True, 'pretrained_backbone_loaded': False,
        'gpu_initialized': torch.cuda.is_initialized(), 'test_data_read': False}
    save(ROOT / 'data/feature_preparation/cpu_shape_check.json', value)
    assert_cpu_only(); print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-only', action='store_true')
    actions.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_shape_check()
    else:
        prepare_development()
