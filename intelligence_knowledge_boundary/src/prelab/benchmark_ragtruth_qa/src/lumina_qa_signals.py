"""Thin QA adapter for the already pinned LUMINA formula port; no data loading.

Original and donor prefixes score the same answer IDs at predictor positions.
Saved likelihoods are diagnostic primitives, not additional fitted scores.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import torch

PORT_PATH = Path(__file__).resolve().parents[2] / 'round7_evidence_grounding/src/lumina7.py'
_spec = importlib.util.spec_from_file_location('pinned_lumina_formula_for_public_qa', PORT_PATH)
port = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(port)

NAMES = ['ipr', 'mmd', 'lumina', 'original_answer_probability',
         'original_max_probability', 'random_answer_probability', 'random_max_probability']
CHUNK = 16


def validate_input(row):
    answer = row['answer_token_ids']
    assert len(answer) > 0 and all(isinstance(x, int) for x in answer)
    for side in ('original', 'random'):
        prefix = row[side + '_prefix_ids']
        assert prefix and row[side + '_input_ids'] == prefix + answer
        assert row[side + '_answer_positions'] == list(range(len(prefix), len(prefix) + len(answer)))
    assert len(row['response_token_offsets']) == len(row['response_token_offsets_raw']) == len(answer)


@torch.inference_mode()
def extract(model, row, *, return_statistics=False):
    """Two backbone calls, original all-layer IPR, final top100 cosine MMD."""
    validate_input(row)
    assert not model.training
    answer = row['answer_token_ids']
    limit = model.config.max_position_embeddings
    assert max(len(row[s + '_input_ids']) for s in ('original', 'random')) <= limit
    device = model.get_input_embeddings().weight.device
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    sync(); tick = time.perf_counter()
    original, layers = port.collect_response_hidden(model, row['original_prefix_ids'], answer, True)
    assert len(layers) == len(port._backbone(model).layers)
    stats = port.final_statistics(model, original, answer, CHUNK)
    ipr = port.ipr_from_hidden(model, layers, stats, CHUNK)
    del layers, original
    sync(); first_seconds = time.perf_counter() - tick
    random_hidden, _ = port.collect_response_hidden(model, row['random_prefix_ids'], answer, False)
    other = port.final_statistics(model, random_hidden, answer, CHUNK)
    del random_hidden
    mmd = port.cosine_mmd_from_topk(stats['top_probs'], stats['top_ids'],
                                  other['top_probs'], other['top_ids'],
                                  model.get_input_embeddings(), CHUNK)
    score = port.LAMBDA * ipr - (1 - port.LAMBDA) * mmd
    values = torch.stack([ipr, mmd, score, stats['answer_probs'], stats['max_probs'],
                          other['answer_probs'], other['max_probs']], dim=1)
    assert values.shape == (len(answer), len(NAMES)) and values.dtype == torch.float32
    assert torch.isfinite(values).all() and (values[:, :2] >= 0).all()
    assert ((values[:, 3:] >= 0) & (values[:, 3:] <= 1)).all()
    assert torch.equal(values[:, 2], .5 * values[:, 0] - .5 * values[:, 1])
    sync()
    meta = {'original_and_IPR_seconds': first_seconds, 'seconds': time.perf_counter() - tick,
            'backbone_forwards': 2, 'all_layer_projections': len(port._backbone(model).layers) * len(answer),
            'same_answer_both_passes': True, 'all_answer_positions_retained': True,
            'formula_version': port.FORMULA_VERSION, 'chunk_size': CHUNK,
            'official_commit': port.OFFICIAL_COMMIT}
    if return_statistics:
        return values, meta, stats, other
    return values, meta


def cpu_selfcheck():
    """Small real causal decoder, alignment and a dense formula oracle."""
    from transformers import LlamaConfig, LlamaForCausalLM
    import torch.nn.functional as F
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4); torch.manual_seed(20261011)
    config = LlamaConfig(vocab_size=192, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval().float()
    answer = [12, 19, 51, 77, 44]
    row = {'original_prefix_ids': [1, 5, 7, 9], 'random_prefix_ids': [1, 41, 32, 26, 17],
           'answer_token_ids': answer, 'response_token_offsets': [[i, i+1] for i in range(5)],
           'response_token_offsets_raw': [[i, i+1] for i in range(5)]}
    def complete_axes(r):
        for s in ('original', 'random'):
            r[s + '_input_ids'] = r[s + '_prefix_ids'] + r['answer_token_ids']
            r[s + '_answer_positions'] = list(range(len(r[s + '_prefix_ids']), len(r[s + '_input_ids'])))
        return r
    complete_axes(row)
    hooks_before = [tuple(x._forward_hooks) for x in model.model.layers]
    a, _, ps, qs = extract(model, row, return_statistics=True)
    b, _ = extract(model, row)
    assert torch.equal(a, b)
    with torch.no_grad():
        full = model.model(input_ids=torch.tensor([row['original_input_ids']]),
                           use_cache=False, output_hidden_states=True, return_dict=True)
        pos = torch.tensor(row['original_answer_positions']) - 1
        final = F.softmax(model.lm_head(full.last_hidden_state[0, pos]).float(), -1)
        assert torch.allclose(final[torch.arange(len(answer)), torch.tensor(answer)],
                              ps['answer_probs'], atol=1e-7, rtol=0)
        numerator = torch.zeros(len(answer)); denominator = torch.zeros(len(answer))
        for depth, hidden in enumerate(full.hidden_states[1:], 1):
            probs = F.softmax(model.lm_head(model.model.norm(hidden[0, pos])).float(), -1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1)
            selected = probs.gather(-1, ps['max_ids'].unsqueeze(-1)).squeeze(-1)
            numerator += depth * (1 - (selected / ps['max_probs']).clamp(max=1.))
            denominator += depth / (entropy + 1e-8)
        dense_ipr = numerator / denominator * ps['answer_probs'] / ps['max_probs']
        e = model.get_input_embeddings()
        ep = F.normalize(e(ps['top_ids']).float(), dim=-1, eps=1e-8)
        eq = F.normalize(e(qs['top_ids']).float(), dim=-1, eps=1e-8)
        p, q = ps['top_probs'], qs['top_probs']
        pp = .5 * (1 + torch.einsum('nkd,njd->nkj', ep, ep))
        qq = .5 * (1 + torch.einsum('nkd,njd->nkj', eq, eq))
        pq = .5 * (1 + torch.einsum('nkd,njd->nkj', ep, eq))
        dense_mmd = (pp*p[:,:,None]*p[:,None,:]).sum((1,2)) + (qq*q[:,:,None]*q[:,None,:]).sum((1,2)) - 2*(pq*p[:,:,None]*q[:,None,:]).sum((1,2))
        assert torch.allclose(a[:, 0], dense_ipr, atol=1e-6, rtol=0)
        assert torch.allclose(a[:, 1], dense_mmd, atol=1e-6, rtol=0)
    changed = dict(row, answer_token_ids=answer[:3]+[84, 100])
    complete_axes(changed)
    c, _ = extract(model, changed)
    assert torch.equal(a[:3], c[:3]), 'Future answer suffix changed earlier predictor signals'
    same = dict(row, random_prefix_ids=row['original_prefix_ids'])
    complete_axes(same)
    d, _ = extract(model, same)
    assert torch.equal(d[:, 1], torch.zeros(len(answer))), 'Identical references must give exact zero MMD'
    assert hooks_before == [tuple(x._forward_hooks) for x in model.model.layers]
    assert not torch.cuda.is_initialized()
    return {'passed': True, 'real_tiny_Llama_cpu': True, 'rows': len(answer),
            'all_layers_including_final_second_norm': True, 'target_index_minus_one_dense_checked': True,
            'original_random_lengths': [len(row[s + '_input_ids']) for s in ('original', 'random')],
            'same_path_repeat_exact': True, 'future_suffix_causality_exact': True,
            'identical_context_MMD_zero_exact': True, 'hooks_restored': True,
            'dense_max_abs_error_IPR': float((a[:, 0]-dense_ipr).abs().max()),
            'dense_max_abs_error_MMD': float((a[:, 1]-dense_mmd).abs().max()),
            'no_data_read': True, 'pretrained_weights_loaded': False, 'GPU_used': False}
