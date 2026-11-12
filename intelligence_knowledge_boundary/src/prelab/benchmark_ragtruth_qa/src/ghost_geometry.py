"""Streaming GHOST-inspired primitives; CPU selfcheck, no pretrained loader.

The paper reports answer-level detection. This helper preserves token scores
for a potential local-window adaptation; it makes no accuracy claim.
"""
from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F


class LayerGeometry:
    """Keep two [answer_positions, hidden] arrays rather than the whole stack."""

    def __init__(self):
        self.count = 0
        self.previous = None
        self.unit_sum = None
        self.transition_sum = None

    def update(self, hidden):
        assert hidden.ndim == 2 and torch.isfinite(hidden).all()
        unit = F.normalize(hidden.float(), dim=-1, eps=1e-12)
        if self.count == 0:
            self.unit_sum = unit.clone()
            self.transition_sum = torch.zeros(len(unit), device=unit.device, dtype=torch.float32)
        else:
            assert self.previous.shape == unit.shape
            self.transition_sum.add_(1 - (self.previous * unit).sum(-1))
            self.unit_sum.add_(unit)
        self.previous = unit
        self.count += 1

    def finish(self, final_hidden):
        assert self.count >= 2 and final_hidden.shape == self.previous.shape
        final_unit = F.normalize(final_hidden.float(), dim=-1, eps=1e-12)
        turbulence = self.transition_sum / (self.count - 1)
        stubbornness = (self.unit_sum * final_unit).sum(-1) / self.count
        return torch.stack((turbulence, stubbornness), dim=-1)


def distribution_features(logits, input_embeddings, k=10):
    """Top-k renormalized entropy and unweighted input-embedding dispersion."""
    assert logits.ndim == 2 and input_embeddings.ndim == 2
    assert logits.shape[1] == input_embeddings.shape[0] and 1 < k <= logits.shape[1]
    values, ids = logits.float().topk(k, dim=-1)
    log_prob = values.log_softmax(-1)
    entropy = -(log_prob.exp() * log_prob).sum(-1)
    unit = F.normalize(input_embeddings[ids].float(), dim=-1, eps=1e-12)
    # Sum over i != j without materializing a [positions,k,k] matrix.
    pair_cosine_sum = unit.sum(1).square().sum(-1) - unit.square().sum((-1, -2))
    dispersion = 1 - pair_cosine_sum / (k * (k - 1))
    return torch.stack((entropy, dispersion), dim=-1)


@torch.no_grad()
def extract(model, input_ids, answer_positions, first_state, last_state, logit_batch=16):
    """One teacher-forced forward, features before reading each target token.

    State index j means HF hidden_states[j], i.e. after decoder block j,
    with hidden_states[0] being embeddings. The selected interval must omit
    the final normalized state; that state is used only by stubbornness.
    This explicit pre-read alignment is a local adaptation, not a claim about
    the original paper's unspecified token/logit indexing convention.
    """
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    assert answer_positions.ndim == 1 and len(answer_positions) > 0
    layers = model.model.layers
    assert 1 <= first_state < last_state < len(layers)
    assert bool(torch.all(answer_positions > 0))
    assert int(answer_positions[-1]) < input_ids.shape[1]
    predictor_positions = answer_positions.to(input_ids.device) - 1
    geometry = LayerGeometry()
    observed, handles = [], []

    def hook_for(state_index):
        def capture(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            geometry.update(hidden[0].index_select(0, predictor_positions))
            observed.append(state_index)
        return capture

    try:
        for state in range(first_state, last_state + 1):
            handles.append(layers[state - 1].register_forward_hook(hook_for(state)))
        hidden = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             use_cache=False, output_attentions=False,
                             output_hidden_states=False).last_hidden_state[0]
    finally:
        for handle in handles:
            handle.remove()
    assert observed == list(range(first_state, last_state + 1))
    selected = hidden.index_select(0, predictor_positions)
    geom = geometry.finish(selected)
    chunks = []
    for start in range(0, len(selected), logit_batch):
        logits = model.lm_head(selected[start:start + logit_batch])
        chunks.append(distribution_features(logits, model.get_input_embeddings().weight))
    result = torch.cat((geom, torch.cat(chunks)), dim=-1).float()
    assert result.shape == (len(answer_positions), 4) and torch.isfinite(result).all()
    return result


def selfcheck():
    from transformers import LlamaConfig, LlamaForCausalLM
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    ids = torch.tensor([[1, 7, 11, 5, 13, 9, 17, 14, 3, 22, 27, 30]])
    positions = torch.arange(6, 12)
    with torch.no_grad():
        actual = extract(model, ids, positions, first_state=1, last_state=3, logit_batch=2)
        dense = model(ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
        selected = torch.stack([s[0, positions - 1].float() for s in dense.hidden_states[1:4]])
        unit = F.normalize(selected, dim=-1)
        final = F.normalize(dense.hidden_states[-1][0, positions - 1].float(), dim=-1)
        turbulence = (1 - (unit[:-1] * unit[1:]).sum(-1)).mean(0)
        stubbornness = (unit * final.unsqueeze(0)).sum(-1).mean(0)
        logits = dense.logits[0, positions - 1].float()
        values, top_ids = logits.topk(10, dim=-1)
        p = values.softmax(-1)
        entropy = -(p * p.log()).sum(-1)
        emb = F.normalize(model.get_input_embeddings().weight[top_ids].float(), dim=-1)
        pair = 1 - torch.einsum('nkh,njh->nkj', emb, emb)
        mask = ~torch.eye(10, dtype=torch.bool)
        dispersion = pair[:, mask].mean(-1)
        expected = torch.stack((turbulence, stubbornness, entropy, dispersion), dim=-1)
        maximum_error = float((actual - expected).abs().max())
        assert torch.allclose(actual, expected, atol=6e-7, rtol=0)
        assert torch.equal(actual, extract(model, ids, positions, 1, 3, 2))
        # Causal prefix test: no future answer token influences earlier features.
        prefix = extract(model, ids[:, :9], torch.arange(6, 9), 1, 3, 2)
        prefix_error = float((prefix - actual[:3]).abs().max())
        assert torch.allclose(prefix, actual[:3], atol=6e-7, rtol=0)
        # Permuting vocabulary entries with their corresponding embeddings is invariant.
        perm = torch.randperm(config.vocab_size)
        direct = distribution_features(logits, model.get_input_embeddings().weight)
        permuted = distribution_features(logits[:, perm], model.get_input_embeddings().weight[perm])
        assert torch.allclose(direct, permuted, atol=3e-7, rtol=0)
        uniform = distribution_features(torch.zeros(2, 10), torch.ones(10, 5))
        assert torch.allclose(uniform[:, 0], torch.full((2,), math.log(10)), atol=3e-7, rtol=0)
        assert torch.allclose(uniform[:, 1], torch.zeros(2), atol=3e-7, rtol=0)
    assert all(not block._forward_hooks for block in model.model.layers)
    assert not torch.cuda.is_initialized()
    out = Path(__file__).resolve().parents[1] / 'results/ghost_geometry_preparation_v1'
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'cpu_primitives_checked_not_extracted_on_real_data',
              'dense_layer_and_pairwise_oracle_max_abs_error': maximum_error,
              'causal_prefix_max_abs_error': prefix_error, 'repeat_exact': True,
              'hook_cleanup': True, 'pretrained_model_loaded': False, 'GPU_used': False,
              'real_data_read': False, 'new_fits': 0, 'output_dimensions': 4,
              'features': ['adjacent_layer_cosine_change', 'layer_to_final_cosine',
                           'top10_normalized_entropy', 'top10_unweighted_embedding_divergence'],
              'timing': 'Before reading the target answer token; all four features aligned to predictor position.',
              'paper_scope': 'Original GHOST aggregates token features to answer means; local window training has not been run.'}
    (out / 'CPU_SELFCHECK.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    selfcheck()
