"""Causal, bounded-memory Lookback Lens / ReDeEP-token features for Qwen2.

The API reads only visible prompts and the model's own exact generated token IDs.
It does not select heads, fit transforms, read labels, or emit calibrated risks.
See references/redeep_adaptation.md for the deliberate protocol adaptations.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half


ATTENTION7_VERSION = "round7-attention-v1"
QUERY_BATCH = 8
LOGIT_BATCH = 8
TOP_CONTEXT_FRACTION = 0.10
SEARCH_MARKER = "\n\nSearch results:\n"


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _response_offsets(tok, ids, text):
    """Offsets of generated BPE pieces, without re-tokenizing their concatenation."""
    decoder = {value: key for key, value in bytes_to_unicode().items()}
    pieces = [bytes(decoder[c] for c in tok.convert_ids_to_tokens(int(i))) for i in ids]
    if b"".join(pieces) != text.encode("utf-8"):
        raise ValueError("Exact response token bytes do not match saved text")
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    offsets, cursor = [], 0
    for piece in pieces:
        end = cursor + len(piece)
        offsets.append((bisect.bisect_right(boundaries, cursor) - 1,
                        bisect.bisect_left(boundaries, end)))
        cursor = end
    return np.asarray(offsets, dtype=np.int32).reshape(-1, 2)


def visible_context_positions(tok, row, input_token_ids):
    """Locate the actual search-result text in the exact rendered chat template.

    The source interval contains titles, passage numbering and separators. It
    excludes system instructions, questions and assistant-template tokens.
    Cross-boundary tokens are assigned by nonempty character overlap.
    """
    prompt = row["prompt"]
    marker = prompt.find(SEARCH_MARKER)
    if marker < 0:
        raise ValueError("Normal prompt must contain the Search results heading")
    source_start = marker + len(SEARCH_MARKER)
    if not prompt[source_start:].strip():
        raise ValueError("No visible source text to measure attention against")
    messages = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": prompt}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    user_start = rendered.find(prompt)
    if user_start < 0 or rendered.find(prompt, user_start + 1) >= 0:
        raise ValueError("User prompt cannot be located uniquely in chat template")
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(input_token_ids):
        raise ValueError("Saved input token IDs differ from rendered visible prompt")
    start, end = user_start + source_start, user_start + len(prompt)
    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
               if b > start and a < end and b > a]
    if not indices:
        raise ValueError("No source tokens after exact prompt alignment")
    # Validate visible source bodies without consulting any evidence/gold keys.
    for passage in row.get("passages", []):
        if passage["text"] not in prompt[source_start:]:
            raise ValueError("A supplied source body is absent from model-visible prompt")
    return np.asarray(indices, dtype=np.int64), {
        "rendered_source_character_interval": [start, end],
        "user_source_character_interval": [source_start, len(prompt)],
        "context_token_indices": indices,
        "context_definition": "Actual Search results body, including source titles and separators",
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def causal_attention_rows(query, key, query_positions, scaling, key_positions=None):
    """Compute selected causal attention rows; shapes H,Q,D and H,K,D.

    Inputs are already rotary-embedded and grouped KV heads are expanded. The
    softmax is float32, as in Transformers eager attention. No full SxS tensor.
    """
    if key_positions is None:
        key_positions = torch.arange(key.shape[-2], device=key.device)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    future = key_positions[None, :] > query_positions[:, None]
    scores = scores.masked_fill(future[None], float("-inf"))
    return scores.float().softmax(-1)


def lookback_from_attention(attention, context_positions, response_start, query_positions):
    """Original ratio of two attention means, not the ratio of summed masses."""
    context_mean = attention.index_select(-1, context_positions).mean(-1)
    positions = torch.arange(attention.shape[-1], device=attention.device)
    response_mask = ((positions[None, :] >= response_start) &
                     (positions[None, :] <= query_positions[:, None]))
    counts = response_mask.sum(-1)
    if torch.any(counts == 0):
        raise ValueError("Post-read response query must include a generated token")
    response_mean = (attention * response_mask[None]).sum(-1) / counts[None]
    return context_mean / (context_mean + response_mean).clamp_min(torch.finfo(torch.float32).tiny)


def external_context_score(attention, context_positions, context_hidden,
                           response_hidden, top_fraction=TOP_CONTEXT_FRACTION):
    """ReDeEP token ECS: cosine(final state, mean top-10%-attended source states).

    Pooling by a small selector matrix avoids allocating H*Q*K*hidden_size.
    A minimum of one token is needed for very short synthetic/test contexts.
    """
    source_attention = attention.index_select(-1, context_positions)
    context_count = source_attention.shape[-1]
    top_count = max(1, int(context_count * top_fraction))
    top_ids = source_attention.topk(top_count, dim=-1, sorted=False).indices
    heads, queries = top_ids.shape[:2]
    # Using model dtype keeps the large matmul on the same precision as the
    # backbone. Cosine and output use float32; this does not quantize features.
    selector = torch.zeros(heads * queries, context_count,
                           device=context_hidden.device, dtype=context_hidden.dtype)
    selector.scatter_(1, top_ids.reshape(heads * queries, top_count), 1.0 / top_count)
    pooled = (selector @ context_hidden).reshape(heads, queries, -1)
    return F.cosine_similarity(pooled.float(), response_hidden[None].float(), dim=-1)


def standard_jsd_from_logits(before, after):
    """Standard JSD(P,Q)=0.5 KL(P||M)+0.5 KL(Q||M), bounded by ln(2).

    Upstream ReDeEP's F.kl_div(logP,M) computes reverse KL; we intentionally
    implement the paper formula, not that bug. Values are not arbitrarily scaled.
    """
    logp, logq = before.float().log_softmax(-1), after.float().log_softmax(-1)
    logm = torch.logaddexp(logp, logq) - math.log(2)
    return 0.5 * ((logp.exp() * (logp - logm)).sum(-1) +
                  (logq.exp() * (logq - logm)).sum(-1))


class _FeatureHooks:
    def __init__(self, model, final_hidden, response_start, response_count,
                 context_positions, query_batch=QUERY_BATCH, logit_batch=LOGIT_BATCH):
        self.model = model
        self.final_hidden = final_hidden
        self.device = final_hidden.device
        self.response_start = response_start
        self.response_count = response_count
        self.positions = torch.arange(response_start, response_start + response_count,
                                      device=self.device)
        self.context_positions = context_positions
        self.context_hidden = final_hidden.index_select(0, context_positions)
        self.response_hidden = final_hidden[response_start:response_start + response_count]
        self.query_batch, self.logit_batch = query_batch, logit_batch
        self.layers = len(model.model.layers)
        self.heads = model.config.num_attention_heads
        self.lookback = np.empty((response_count, self.layers, self.heads), dtype=np.float32)
        self.ecs = np.empty_like(self.lookback)
        self.pks = np.empty((response_count, self.layers), dtype=np.float32)
        self.handles, self.pending = [], {}
        self.seen_attention, self.seen_ffn = set(), set()

    def __enter__(self):
        for layer_id, block in enumerate(self.model.model.layers):
            attn = block.self_attn

            def before_attention(module, args, kwargs, layer=layer_id):
                if kwargs.get("past_key_value") is not None:
                    raise ValueError("Feature replay must disable KV caching")
                self.pending[layer] = {"rope": kwargs["position_embeddings"]}

            def query_projection(module, args, output, layer=layer_id):
                self.pending[layer]["q"] = output

            def key_projection(module, args, output, layer=layer_id, attention=attn):
                self._attention(layer, attention, output)

            def before_ffn_norm(module, args, layer=layer_id):
                # Input is the residual stream after attention, before FFN norm.
                self.pending[layer]["pre_ffn"] = args[0][0].index_select(0, self.positions)

            def after_block(module, args, output, layer=layer_id):
                hidden = output[0] if isinstance(output, tuple) else output
                self._pks(layer, hidden[0].index_select(0, self.positions))

            self.handles.extend([
                attn.register_forward_pre_hook(before_attention, with_kwargs=True),
                attn.q_proj.register_forward_hook(query_projection),
                attn.k_proj.register_forward_hook(key_projection),
                block.post_attention_layernorm.register_forward_pre_hook(before_ffn_norm),
                block.register_forward_hook(after_block),
            ])
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def _attention(self, layer, module, key_projection):
        state = self.pending[layer]
        dim = module.head_dim
        q_all = state.pop("q").view(1, -1, self.heads, dim).transpose(1, 2)
        k = key_projection.view(1, -1, self.model.config.num_key_value_heads, dim).transpose(1, 2)
        cos, sin = state.pop("rope")
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        k = repeat_kv(k, module.num_key_value_groups)[0]
        for start in range(0, self.response_count, self.query_batch):
            stop = min(start + self.query_batch, self.response_count)
            positions = self.positions[start:stop]
            q = q_all[0].index_select(1, positions)
            qcos, qsin = cos[0].index_select(0, positions), sin[0].index_select(0, positions)
            q = q * qcos[None] + rotate_half(q) * qsin[None]
            attention = causal_attention_rows(q, k, positions, module.scaling)
            lookback = lookback_from_attention(attention, self.context_positions,
                                              self.response_start, positions)
            ecs = external_context_score(attention, self.context_positions,
                                         self.context_hidden, self.response_hidden[start:stop])
            self.lookback[start:stop, layer] = lookback.T.cpu().numpy()
            self.ecs[start:stop, layer] = ecs.T.cpu().numpy()
        self.seen_attention.add(layer)

    def _pks(self, layer, post_ffn):
        before = self.pending[layer].pop("pre_ffn")
        for start in range(0, self.response_count, self.logit_batch):
            stop = min(start + self.logit_batch, self.response_count)
            # Every intermediate residual uses the same final RMSNorm + head.
            # In particular, final-layer residual is normalized only once.
            left = self.model.lm_head(self.model.model.norm(before[start:stop]))
            right = self.model.lm_head(self.model.model.norm(post_ffn[start:stop]))
            self.pks[start:stop, layer] = standard_jsd_from_logits(left, right).cpu().numpy()
        del self.pending[layer]
        self.seen_ffn.add(layer)


def _item_aggregate(token_values, indices):
    shape = (len(indices),) + token_values.shape[1:]
    if not indices:
        return np.empty(shape, dtype=np.float32)
    return np.asarray([token_values[ii].mean(0) for ii in indices], dtype=np.float32)


@torch.inference_mode()
def extract_attention(tok, model, row, generated) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return raw item and token features; compatible with the Round 6 record schema.

    Two SDPA causal passes: final token states, then selected query rows plus FFN
    changes. Full vocabulary logits exist only for LOGIT_BATCH response positions
    at one layer. The model's attention implementation and parameters are intact.
    """
    if model.training:
        raise ValueError("Frozen evaluation model required")
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("This adapter has been checked only for Qwen2 architecture")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("Sliding-window mask is not supported by this adapter")
    prefix = [int(x) for x in generated["input_token_ids"]]
    response_ids = [int(x) for x in generated["response_token_ids"]]
    if not response_ids:
        raise ValueError("Empty response: no attention feature can be extracted")
    if generated.get("unexpected_special_token_ids"):
        raise ValueError("Unexpected interior special tokens invalidate exact replay")
    offsets = _response_offsets(tok, response_ids, generated["response"])
    if "response_token_offsets" in generated and not np.array_equal(offsets, generated["response_token_offsets"]):
        raise ValueError("Saved response offsets disagree with exact token bytes")
    context_ids, context_meta = visible_context_positions(tok, row, prefix)
    items, item_indices = [], []
    for item in generated["items"]:
        if not item.get("parse_ok") or not item.get("text"):
            continue
        start, end = item["start"], item["end"]
        if generated["response"][start:end] != item["text"]:
            raise ValueError("Item offsets disagree with response text")
        indices = np.flatnonzero((offsets[:, 1] > start) & (offsets[:, 0] < end))
        if not len(indices):
            raise ValueError("Parsed item contains no response token")
        items.append({"item_id": item["item_id"], "item_index": item["item_index"],
                      "start": start, "end": end, "response_token_indices": indices.tolist(),
                      "last_absolute_query_position": len(prefix) + int(indices[-1])})
        item_indices.append(indices)
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([prefix + response_ids], device=device)
    context_positions = torch.as_tensor(context_ids, device=device)
    _synchronize(device)
    start = time.perf_counter()
    final = model.model(input_ids=ids, use_cache=False, output_attentions=False,
                        output_hidden_states=False).last_hidden_state[0]
    _synchronize(device)
    first_seconds = time.perf_counter() - start
    with _FeatureHooks(model, final, len(prefix), len(response_ids), context_positions) as hooks:
        second = model.model(input_ids=ids, use_cache=False, output_attentions=False,
                             output_hidden_states=False)
        if len(hooks.seen_attention) != hooks.layers or len(hooks.seen_ffn) != hooks.layers:
            raise RuntimeError("Not every decoder layer emitted the required signals")
    _synchronize(device)
    seconds = time.perf_counter() - start
    # Row-major flattening is layer then attention head, matching original Lens.
    token_lb = hooks.lookback.reshape(len(response_ids), -1)
    token_ecs = hooks.ecs.reshape(len(response_ids), -1)
    values = {
        "item_ids": np.asarray([item["item_id"] for item in items], dtype=str),
        "lookback_features": _item_aggregate(token_lb, item_indices),
        "redeep_ecs": _item_aggregate(token_ecs, item_indices),
        "redeep_pks": _item_aggregate(hooks.pks, item_indices),
        "token_lookback": token_lb,
        "token_redeep_ecs": token_ecs,
        "token_redeep_pks": hooks.pks,
        "response_token_offsets": offsets,
        "layer_numbers": np.arange(1, hooks.layers + 1, dtype=np.int32),
        "head_numbers": np.arange(hooks.heads, dtype=np.int32),
    }
    if not all(np.isfinite(value).all() for key, value in values.items() if key != "item_ids"):
        raise FloatingPointError("Non-finite attention features; do not replace with zeros")
    if np.any(hooks.pks < -1e-6) or np.any(hooks.pks > math.log(2) + 1e-5):
        raise FloatingPointError("Standard JSD violated its bounds")
    metadata = {
        "version": ATTENTION7_VERSION, "row_id": row["row_id"], "items": items,
        "input_tokens": len(prefix), "response_tokens": len(response_ids),
        "seconds": seconds, "final_hidden_pass_seconds": first_seconds,
        "feature_pass_seconds": seconds - first_seconds, "causal_passes": 2,
        "query_batch": QUERY_BATCH, "logit_batch": LOGIT_BATCH,
        "top_context_fraction": TOP_CONTEXT_FRACTION,
        "top_context_tokens": max(1, int(len(context_ids) * TOP_CONTEXT_FRACTION)),
        "feature_timing": "Post-read query P+j; response side includes tokens 0..j; item means use only item tokens",
        "lookback_formula": "mean_attention_to_sources / (mean_attention_to_sources + mean_attention_to_generated_prefix)",
        "redeep_ecs_formula": "cosine(final_current_state, mean_final_state_of_top_10_percent_attended_source_tokens)",
        "redeep_pks_formula": "standard_JSD(softmax(head(final_norm(pre_FFN_residual))), softmax(head(final_norm(post_FFN_residual))))",
        "pks_bug_fix": "Standard forward KL to mixture; upstream reverse-KL implementation is not reproduced",
        "backbone_attention_implementation": model.config._attn_implementation,
        "feature_axes": "Head features: token_or_item x (layer_major_then_head); PKS: token_or_item x layer",
        "adaptations": ["Qwen2 GQA/RoPE and NF4 backbone", "Source-only context located in exact chat prompt",
                        "Post-read instead of original Lookback next-token prediction timing",
                        "All heads/layers emitted; head/layer selection and scaling happen only in fitter",
                        "Answer-item means instead of final full-answer aggregation", "Standard JSD correction"],
        **context_meta,
    }
    del final, second
    return values, metadata
