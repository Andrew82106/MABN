"""Shared frozen Llama replay for the new, group-separated in-domain experiment.

Reads model-visible inputs only. No annotations, fitting or threshold selection.
The native Lookback signal is preserved for its baseline; additional signals
are separate candidate features. Historical generation traces are unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
PROJECT = next(p for p in HERE.parents if p.name == 'intelligence_knowledge_boundary')
sys.path.insert(0, str(BENCH / 'src'))
import feature_qa as feature
from feature_lookback_controls_v2 import ControlHooks
from run_feature_qa import load_nf4

EXTERNAL = PROJECT / 'reports/paper/tmp/submission_20260927_v2/external_eval'
spec = importlib.util.spec_from_file_location('same_domain_native_mapping', EXTERNAL / 'baseline_adapters.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class BufferedHooks(feature.LookbackHooks):
    """The same batch8 arithmetic, keeping readouts on GPU until forward ends.

    Only the two exported definitions are calculated. No CPU synchronization
    per attention head/chunk, and no changes to model attention or logits.
    """
    def __init__(self, model, plan, view):
        super().__init__(model, plan, query_batch=8)
        self.prefix = torch.tensor(view['prefix_context_token_positions'], device=self.device, dtype=torch.long)
        self.header = torch.tensor(view['header_token_positions'], device=self.device, dtype=torch.long)
        self.new = torch.cat((self.header, self.positions))
        assert view['header_token_positions'][-1] < plan['original']['answer_token_positions'][0]
        assert view['header_token_positions']
        self.controls = {k: torch.empty((len(self.positions), self.spec['layers'], self.spec['heads']),
                         device=self.device, dtype=torch.float32)
                         for k in ['lb_prefix_pre_header', 'lb_source_post_legacy']}

    def capture(self, layer, module, keys):
        state = self.pending.pop(layer)
        heads, kv, dim = self.spec['heads'], self.spec['kv_heads'], self.spec['head_dim']
        query = state['q'].view(1, -1, heads, dim).transpose(1, 2)
        key = keys.view(1, -1, kv, dim).transpose(1, 2)
        cos, sin = state['rope']
        key = key * cos[:, None] + feature.rotate_half(key) * sin[:, None]
        key = key.repeat_interleave(heads // kv, dim=1)[0]
        absolute = torch.arange(key.shape[-2], device=self.device)
        for timing in ['post', 'pre']:
            for begin in range(0, len(self.positions), self.query_batch):
                end = min(begin + self.query_batch, len(self.positions))
                pos = self.positions[begin:end] - int(timing == 'pre')
                q = query[0].index_select(1, pos)
                qc, qs = cos[0].index_select(0, pos), sin[0].index_select(0, pos)
                q = q * qc[None] + feature.rotate_half(q) * qs[None]
                logits = (q @ key.transpose(-2, -1)) * module.scaling
                logits.masked_fill_(absolute[None, None] > pos[None, :, None], float('-inf'))
                weights = logits.float().softmax(-1)
                if timing == 'pre':
                    causal_new = self.new[None, :] <= pos[:, None]
                    new_mean = (weights.index_select(-1, self.new) * causal_new[None]).sum(-1) / causal_new.sum(-1)[None]
                    context_mean = weights.index_select(-1, self.prefix).mean(-1)
                    ratio = context_mean / (context_mean + new_mean).clamp_min(torch.finfo(torch.float32).tiny)
                    self.controls['lb_prefix_pre_header'][begin:end, layer] = ratio.T
                else:
                    source_mean = weights.index_select(-1, self.context).mean(-1)
                    attention = weights.index_select(-1, self.positions)
                    previous = self.positions[None, :] <= pos[:, None]
                    answer_mean = (attention * previous[None]).sum(-1) / previous.sum(-1)[None]
                    ratio = source_mean / (source_mean + answer_mean).clamp_min(torch.finfo(torch.float32).tiny)
                    self.controls['lb_source_post_legacy'][begin:end, layer] = ratio.T
        self.seen.add(layer)


def readl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


@torch.inference_mode()
def extract(model, row, geometry):
    ids, mask, positions = feature.tensor_input(model, geometry)
    layer = min(15, len(model.model.layers) - 1)
    captured = {}

    def capture_mid(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured['hidden_mid'] = hidden[0].index_select(0, positions).float().cpu().numpy()

    handle = model.model.layers[layer].register_forward_hook(capture_mid)
    try:
        with BufferedHooks(model, {'original': geometry}, native.lookback_layout(row, geometry)) as hooks:
            final = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                output_hidden_states=False).last_hidden_state[0]
        assert len(hooks.seen) == len(model.model.layers)
        nll, entropy, margin = [], [], []
        for pos in positions.split(16):
            logp = model.lm_head(final.index_select(0, pos - 1)).float().log_softmax(-1)
            selected = logp.gather(1, ids[0, pos, None]).squeeze(-1)
            nll.extend((-selected).cpu().tolist())
            entropy.extend((-(logp.exp() * logp).sum(-1)).cpu().tolist())
            margin.extend((selected - logp.max(-1).values).cpu().tolist())
        arrays = {
            'lb_prefix_pre_header': hooks.controls['lb_prefix_pre_header'].reshape(len(positions), -1).cpu().numpy(),
            'lb_source_post_legacy': hooks.controls['lb_source_post_legacy'].reshape(len(positions), -1).cpu().numpy(),
            'nll': np.asarray(nll, np.float32),
            'entropy': np.asarray(entropy, np.float32),
            'selected_vs_best_logp': np.asarray(margin, np.float32),
            'hidden_mid': captured['hidden_mid'],
            'hidden_last': final.index_select(0, positions).float().cpu().numpy(),
            'token_ids': np.asarray(geometry['answer_token_ids'], np.int64),
            'answer_token_positions': np.asarray(geometry['answer_token_positions'], np.int64),
            'response_token_offsets': np.asarray(geometry['response_token_offsets'], np.int32),
            'lexical_mask': np.asarray(geometry['lexical_mask'], bool),
        }
        assert all(len(v) == len(positions) for v in arrays.values())
        assert all(np.isfinite(v).all() for v in arrays.values())
        return arrays
    finally:
        handle.remove()


def cpu_check():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(20260928)
    torch.set_num_threads(2)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    cfg._attn_implementation = 'eager'
    model = LlamaForCausalLM(cfg).eval()
    ids = [1, 3, 6, 9, 12, 15, 18, 21, 24]
    g = {'input_ids': ids, 'attention_mask': [1] * 9,
         'answer_token_positions': [6, 7, 8], 'answer_token_ids': ids[6:],
         'context_token_positions': [1, 2, 3],
         'response_token_offsets': [[0, 1], [1, 2], [2, 3]],
         'input_token_offsets': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 6], [6, 13],
                                 [13, 14], [14, 15], [15, 16]],
         'lexical_mask': [True, True, True]}
    # A two-token assistant header ends immediately before the answer.
    row = {'prompt_prefix': 'abcd [/INST] '}
    arrays = extract(model, row, g)
    with torch.inference_mode(), ControlHooks(model, {'original': g}, native.lookback_layout(row, g)) as reference_hooks:
        model.model(input_ids=torch.tensor([ids]), use_cache=False)
    for name in ['lb_prefix_pre_header', 'lb_source_post_legacy']:
        assert np.array_equal(arrays[name], reference_hooks.controls[name].reshape(len(g['answer_token_positions']), -1))
    with torch.inference_mode():
        reference = model(torch.tensor([ids]), output_attentions=True)
        direct = -reference.logits[0, 5:8].float().log_softmax(-1).gather(
            1, torch.tensor(ids[6:])[:, None]).squeeze(-1).numpy()
    assert np.allclose(arrays['nll'], direct, rtol=1e-6, atol=1e-6)
    expected = []
    for position in [5, 6, 7]:
        per_layer = []
        for attention in reference.attentions:
            w = attention[0, :, position]
            context = w[:, :4].mean(-1)
            response = w[:, 4:position + 1].mean(-1)
            per_layer.extend((context / (context + response)).tolist())
        expected.append(per_layer)
    assert np.allclose(arrays['lb_prefix_pre_header'], expected, rtol=1e-6, atol=1e-6)
    assert (arrays['selected_vs_best_logp'] <= 1e-6).all()
    assert (arrays['entropy'] >= 0).all()
    publish(HERE / 'signals/CPU_CHECK.json', {'passed': True,
            'nll_matches_direct_next_token_prediction': True,
            'lookback_matches_direct_eager_attention': True,
            'test_labels_read': False})
    print('CPU_CHECK_PASSED', flush=True)


def run(args):
    out = HERE / 'signals'
    out.mkdir(parents=True, exist_ok=True)
    assert (out / 'CPU_CHECK.json').exists()
    tok = native.tokenizer()
    records = []
    for partition in args.partitions:
        rows = readl(HERE / 'data' / partition / 'inputs.jsonl')
        records.extend((partition, row) for row in rows)
    expected_count = len(records)
    if args.limit:
        records = records[:args.limit]
    source_files = [Path(__file__), Path(feature.__file__),
                    BENCH / 'src/feature_lookback_controls_v2.py', BENCH / 'src/run_feature_qa.py',
                    EXTERNAL / 'baseline_adapters.py', HERE / 'data/MANIFEST.json']
    source_hash = digest({str(p): sha(p) for p in source_files})
    model, loading = load_nf4()
    torch.set_num_threads(4)
    publish(out / 'runtime.json', {'loading': loading, 'source_sha256': source_hash,
            'checkpoint': 'NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee',
            'hidden_mid_after_zero_based_block': 15, 'test_labels_read': False})
    started = time.perf_counter()
    for index, (partition, row) in enumerate(records, 1):
        rid = row['response_id']
        assert rid and not any(c in rid for c in '/\\:')
        path = out / f'{rid}.npz'
        receipt = out / f'{rid}.json'
        identity = digest(row)
        if args.resume and path.exists() and receipt.exists():
            old = json.loads(receipt.read_text(encoding='utf-8'))
            assert old['input_sha256'] == identity and old['source_sha256'] == source_hash
            assert old['npz_sha256'] == sha(path)
            continue
        assert not path.exists() and not receipt.exists(), f'Use --resume for {rid}'
        geometry = native.geometry(tok, row)
        if len(geometry['input_ids']) > 4096:
            raise ValueError(f'No truncation allowed: {rid}')
        arrays = extract(model, row, geometry)
        if index == 1 and (HERE / 'signals_slow_reference' / f'{rid}.npz').exists():
            with np.load(HERE / 'signals_slow_reference' / f'{rid}.npz', allow_pickle=False) as old:
                errors = {k: float(np.max(np.abs(arrays[k].astype(float) - old[k].astype(float))))
                          for k in arrays}
                assert all(v == 0 for v in errors.values()), errors
            publish(out / 'BUFFERED_EQUIVALENCE.json', {'first_response_id': rid,
                    'max_abs_errors': errors, 'all_arrays_bitwise_equal': True,
                    'reference_npz_sha256': sha(HERE / 'signals_slow_reference' / f'{rid}.npz')})
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        publish(receipt, {'response_id': rid, 'partition': partition,
                'input_sha256': identity, 'source_sha256': source_hash,
                'npz_sha256': sha(path), 'tokens': len(arrays['nll'])})
        if index % 10 == 0 or index == len(records):
            progress = {'pid': os.getpid(), 'completed': index, 'total': len(records),
                        'elapsed_seconds': time.perf_counter() - started,
                        'last_response_id': rid, 'test_labels_read': False}
            publish(out / 'PROGRESS.json', progress)
            print(json.dumps(progress), flush=True)
    completion = 'COMPLETE_' if len(records) == expected_count else 'PARTIAL_'
    publish(out / (completion + '_'.join(args.partitions) + '.json'),
            {'count': len(records), 'expected_count': expected_count, 'source_sha256': source_hash,
             'seconds': time.perf_counter() - started, 'test_labels_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['cpu-check', 'extract'])
    parser.add_argument('--partitions', nargs='+', choices=['fit', 'validation', 'test'],
                        default=['fit', 'validation'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.action == 'cpu-check':
        cpu_check()
    else:
        run(args)
