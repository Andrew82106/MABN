"""Memory-bounded port of the pinned official LUMINA detector.

This is an item-level adaptation of the official implementation, not a claim
to reproduce every experiment in the paper. See references/lumina_port.md.
No labels are consumed. Original answer IDs remain identical in both passes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


OFFICIAL_COMMIT = "c43ff41d872b05f659dcb3ad3a6dd78226954319"
OFFICIAL_SHA256 = "1b5ecd8c08982b40dae59f299e580219074ee83673768feb53cf15612b7d8a5f"
FORMULA_VERSION = "official-c43ff41-all-output-layers-top100-unnormalized-cosine-ipr-v1"
SCHEMA_VERSION = "round7-lumina-v1"
DEFAULT_CHUNK_SIZE = 16
TOP_K = 100
LAMBDA = 0.5
MARKER = "\n\nSearch results:\n"


def _sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _chat_ids(tok, prompt, system):
    return list(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True))


def _render_passages(passages):
    return "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages))


def build_random_prompt(row, random_row):
    """Only replace source material; never import another row's question."""
    prompt = row["prompt"]
    if prompt.count(MARKER) != 1:
        raise ValueError("Expected exactly one Search results marker; no implicit prompt repair")
    head, original_evidence = prompt.split(MARKER, 1)
    if original_evidence != _render_passages(row["passages"]):
        raise ValueError("Original source formatting differs from numbered-passage protocol")
    passages = random_row["passages"]
    if not passages or len(passages) != len(row["passages"]):
        raise ValueError("Random source must provide the same nonzero number of passages")
    if row.get("question_id") and row.get("question_id") == random_row.get("question_id"):
        raise ValueError("Random source cannot be the same question group")
    if {p["title"] for p in passages} & {p["title"] for p in row["passages"]}:
        raise ValueError("Random source overlaps a target source title")
    if {p["text"] for p in passages} & {p["text"] for p in row["passages"]}:
        raise ValueError("Random source overlaps a target source text")
    return head + MARKER + _render_passages(passages)


def _check_offsets(generated):
    offsets = np.asarray(generated["response_token_offsets"], dtype=np.int32)
    answer_ids = generated["response_token_ids"]
    if not answer_ids and offsets.size == 0:
        offsets = offsets.reshape(0, 2)
    if offsets.shape != (len(answer_ids), 2):
        raise ValueError("Token offsets and answer IDs differ in length")
    if offsets.size and (offsets.min() < 0 or offsets.max() > len(generated["response"])):
        raise ValueError("Token offset outside response")
    return offsets


def _backbone(model):
    base = model.model
    if hasattr(base, "language_model"):
        base = base.language_model
    if not hasattr(base, "layers") or not hasattr(base, "norm"):
        raise TypeError("Port supports decoder backbones exposing layers and final norm")
    return base


@torch.inference_mode()
def collect_response_hidden(model, prefix_ids, answer_ids, capture_layers=True):
    """Capture pre-token states at response positions; move only those to CPU.

    Hugging Face hidden_states[1:] contains decoder outputs 1..L-1 and
    the final-normalized last layer. The official code uses all of these.
    This function deliberately reproduces that convention, without retaining
    full-prompt tensors from every layer or full-vocabulary logits.
    """
    if not prefix_ids or not answer_ids:
        raise ValueError("Nonempty prefix and response required")
    base = _backbone(model)
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([list(prefix_ids) + list(answer_ids)], device=device, dtype=torch.long)
    first = len(prefix_ids) - 1
    stop = first + len(answer_ids)
    captures = {}
    handles = []
    if capture_layers:
        for index, layer in enumerate(base.layers[:-1]):
            def capture(_module, _args, output, key=index):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[key] = hidden[0, first:stop].detach().to("cpu", copy=True)
            handles.append(layer.register_forward_hook(capture))
    try:
        output = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                      output_hidden_states=False, return_dict=True)
        final_hidden = output.last_hidden_state[0, first:stop].detach().to("cpu", copy=True)
    finally:
        for handle in handles:
            handle.remove()
    del output, ids
    if not capture_layers:
        return final_hidden, []
    if len(captures) != len(base.layers) - 1:
        raise RuntimeError("Not every requested decoder layer produced hidden states")
    layer_hidden = [captures[i] for i in range(len(base.layers)-1)] + [final_hidden]
    return final_hidden, layer_hidden


@torch.inference_mode()
def final_statistics(model, hidden, answer_ids, chunk_size=DEFAULT_CHUNK_SIZE, top_k=TOP_K):
    """Keep top-k probabilities and two scalar probabilities, not full logits."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    device = model.get_input_embeddings().weight.device
    result = {key: [] for key in ("top_probs", "top_ids", "max_ids", "max_probs", "answer_probs")}
    for start in range(0, len(answer_ids), chunk_size):
        end = min(start+chunk_size, len(answer_ids))
        logits = model.lm_head(hidden[start:end].to(device)).float()
        probabilities = F.softmax(logits, dim=-1)
        if top_k > probabilities.shape[-1]:
            raise ValueError("top_k exceeds vocabulary; never silently reduce official top-100")
        top_probs, top_ids = torch.topk(probabilities, top_k, dim=-1)
        targets = torch.tensor(answer_ids[start:end], device=device, dtype=torch.long)
        batch = torch.arange(end-start, device=device)
        max_probs, max_ids = probabilities.max(-1)
        answer_probs = probabilities[batch, targets]
        for key, value in (("top_probs", top_probs), ("top_ids", top_ids),
                           ("max_ids", max_ids), ("max_probs", max_probs),
                           ("answer_probs", answer_probs)):
            result[key].append(value.detach().cpu())
        del logits, probabilities, top_probs, top_ids, max_probs, max_ids, answer_probs
    return {key: torch.cat(values) for key, values in result.items()}


@torch.inference_mode()
def cosine_mmd_from_topk(p_probs, p_ids, q_probs, q_ids, embedding,
                         chunk_size=DEFAULT_CHUNK_SIZE):
    """Exact algebraic reduction of the official cosine-kernel top-k MMD.

    The top-k masses are NOT renormalized. The mass-difference term therefore
    cannot be dropped. This is the same kernel sum, without k x k matrices:
      .5 * ((sum p - sum q)^2 + ||sum p normalize(Ep)-sum q normalize(Eq)||^2).
    Only the official top-100 vocabulary truncation approximates the original
    full-vocabulary MMD; this reduction introduces no additional approximation.
    """
    if p_probs.shape != p_ids.shape or q_probs.shape != q_ids.shape or p_probs.shape != q_probs.shape:
        raise ValueError("MMD top-k arrays have incompatible shapes")
    device = embedding.weight.device
    scores = []
    for start in range(0, len(p_probs), chunk_size):
        end = min(start+chunk_size, len(p_probs))
        p = p_probs[start:end].to(device, dtype=torch.float32)
        q = q_probs[start:end].to(device, dtype=torch.float32)
        pe = F.normalize(embedding(p_ids[start:end].to(device)).float(), p=2, dim=-1, eps=1e-8)
        qe = F.normalize(embedding(q_ids[start:end].to(device)).float(), p=2, dim=-1, eps=1e-8)
        difference = (p.unsqueeze(-1)*pe).sum(1) - (q.unsqueeze(-1)*qe).sum(1)
        score = 0.5 * ((p.sum(-1)-q.sum(-1)).square() + difference.square().sum(-1))
        scores.append(score.cpu())
        del p, q, pe, qe, difference, score
    return torch.cat(scores)


@torch.inference_mode()
def ipr_from_hidden(model, layer_hidden, final_stats, chunk_size=DEFAULT_CHUNK_SIZE):
    """Official IPR over all output layers, with streaming float32 sums."""
    base = _backbone(model)
    device = model.get_input_embeddings().weight.device
    length = len(final_stats["max_probs"])
    numerator = torch.zeros(length, dtype=torch.float32)
    denominator = torch.zeros(length, dtype=torch.float32)
    for layer_index, hidden in enumerate(layer_hidden, start=1):
        if hidden.shape[0] != length:
            raise ValueError("Layer and output probabilities differ in response length")
        for start in range(0, length, chunk_size):
            end = min(start+chunk_size, length)
            # Deliberately includes a second final norm for hidden_states[-1],
            # exactly as the pinned repository. See formula-version metadata.
            logits = model.lm_head(base.norm(hidden[start:end].to(device))).float()
            probabilities = F.softmax(logits, dim=-1)
            entropy = -(probabilities*torch.log(probabilities.clamp(min=1e-8))).sum(-1)
            max_ids = final_stats["max_ids"][start:end].to(device)
            max_probs = final_stats["max_probs"][start:end].to(device)
            selected = probabilities.gather(-1, max_ids.unsqueeze(-1)).squeeze(-1)
            ratios = 1 - torch.clamp(selected/max_probs, max=1.0)
            numerator[start:end] += (layer_index*ratios).cpu()
            denominator[start:end] += (layer_index/(entropy+1e-8)).cpu()
            del logits, probabilities, entropy, selected, ratios, max_ids, max_probs
    return numerator/denominator * (final_stats["answer_probs"]/final_stats["max_probs"])


def aggregate_items(generated, token_values):
    """All declared items survive; a parser failure yields NaN plus a reason."""
    offsets = _check_offsets(generated)
    item_ids, indices, details = [], [], []
    for item in generated["items"]:
        item_ids.append(item["item_id"])
        ok = bool(item.get("parse_ok", False) and item.get("text") and
                  item.get("start") is not None and item.get("end") is not None)
        selected = (np.flatnonzero((offsets[:, 1] > item["start"]) & (offsets[:, 0] < item["end"]))
                    if ok else np.asarray([], dtype=np.int64))
        if ok and not len(selected):
            ok = False
        if ok and not (0 <= item["start"] < item["end"] <= len(generated["response"])):
            raise ValueError("Parsed item range is outside the original response")
        indices.append(selected)
        details.append({"item_id": item["item_id"], "score_available": ok,
                        "reason": "ok" if ok else item.get("parse_reason", "unscorable_item"),
                        "response_token_indices": selected.tolist(),
                        "start": item.get("start"), "end": item.get("end"),
                        "score_available_after_response_token_index": int(selected[-1]) if ok else None})
    values = {"item_ids": np.asarray(item_ids, dtype=str)}
    for name, scores in token_values.items():
        vector = np.asarray(scores, dtype=np.float32)
        if vector.shape != (len(offsets),):
            raise ValueError("Token score shape differs from generated response")
        values[name] = np.asarray([float(np.mean(vector[index])) if len(index) else np.nan
                                   for index in indices], dtype=np.float32)
        values["token_"+name] = vector
    values["response_token_offsets"] = offsets
    return values, details


@torch.inference_mode()
def extract_lumina(tok, model, row, generated, random_row, *, chunk_size=DEFAULT_CHUNK_SIZE):
    """Two causal forward passes; item means of official-code token scores.

    random_row must come from a fixed independent, reviewed distractor pool.
    Structural overlap is rejected here; semantic independence is a data audit
    responsibility and must be recorded by the caller before scoring.
    """
    if model.training:
        raise ValueError("LUMINA extraction requires model.eval()")
    if generated.get("row_id", row["row_id"]) != row["row_id"]:
        raise ValueError("Generation belongs to a different row")
    prefix = _chat_ids(tok, row["prompt"], row["system"])
    if prefix != list(generated["input_token_ids"]):
        raise ValueError("Original chat prefix is not the exact frozen generation prefix")
    answer_ids = list(generated["response_token_ids"])
    decoded = tok.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != generated["response"]:
        raise ValueError("Answer IDs do not decode to the frozen response")
    _check_offsets(generated)
    random_prompt = build_random_prompt(row, random_row)
    random_prefix = _chat_ids(tok, random_prompt, row["system"])
    limit = int(getattr(model.config, "max_position_embeddings", 32768))
    if max(len(prefix), len(random_prefix))+len(answer_ids) > limit:
        raise ValueError("Scoring input exceeds model context limit; no silent truncation")
    device = model.get_input_embeddings().weight.device
    _sync(device)
    started = time.perf_counter()
    if not answer_ids:
        vectors = {key: np.asarray([], dtype=np.float32) for key in
                   ("lumina_mmd", "lumina_ipr", "lumina_score")}
        values, details = aggregate_items(generated, vectors)
        return values, {"row_id": row["row_id"], "formula_version": FORMULA_VERSION,
                        "items": details, "seconds": time.perf_counter()-started,
                        "status": "empty_generation", "forward_passes": 0}
    initial_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_hidden, layer_hidden = collect_response_hidden(model, prefix, answer_ids, True)
    _sync(device)
    original_pass_seconds = time.perf_counter()-started
    p_stats = final_statistics(model, final_hidden, answer_ids, chunk_size)
    ipr = ipr_from_hidden(model, layer_hidden, p_stats, chunk_size)
    del layer_hidden, final_hidden
    _sync(device)
    original_compute_seconds = time.perf_counter()-started-original_pass_seconds
    random_start = time.perf_counter()
    random_hidden, _ = collect_response_hidden(model, random_prefix, answer_ids, False)
    q_stats = final_statistics(model, random_hidden, answer_ids, chunk_size)
    del random_hidden
    mmd = cosine_mmd_from_topk(p_stats["top_probs"], p_stats["top_ids"],
                               q_stats["top_probs"], q_stats["top_ids"],
                               model.get_input_embeddings(), chunk_size)
    score = LAMBDA*ipr - (1-LAMBDA)*mmd
    _sync(device)
    values, details = aggregate_items(generated, {"lumina_mmd": mmd.numpy(),
                                                "lumina_ipr": ipr.numpy(),
                                                "lumina_score": score.numpy()})
    metadata = {
        "schema_version": SCHEMA_VERSION, "formula_version": FORMULA_VERSION,
        "official_commit": OFFICIAL_COMMIT, "official_code_sha256": OFFICIAL_SHA256,
        "port_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "row_id": row["row_id"], "random_row_id": random_row.get("row_id"),
        "random_question_id": random_row.get("question_id"),
        "original_prompt_sha256": _sha({"system": row["system"], "prompt": row["prompt"]}),
        "random_prompt_sha256": _sha({"system": row["system"], "prompt": random_prompt}),
        "random_source_sha256": _sha(random_row["passages"]),
        "original_prefix_ids_sha256": _sha(prefix), "random_prefix_ids_sha256": _sha(random_prefix),
        "response_token_ids_sha256": _sha(answer_ids), "same_response_ids_both_passes": True,
        "original_input_tokens": len(prefix), "random_input_tokens": len(random_prefix),
        "response_tokens": len(answer_ids), "forward_passes": 2,
        "layer_count": len(_backbone(model).layers), "top_k": TOP_K, "lambda": LAMBDA,
        "kernel": "(1 + cosine) / 2", "top_k_renormalized": False,
        "mmd_implementation": "exact cosine-kernel weighted-embedding algebra including unequal top-k masses",
        "chunk_size": chunk_size, "probability_and_reduction_dtype": "float32",
        "hidden_state_dtype": str(next(_backbone(model).norm.parameters()).dtype),
        "layer_convention": "official hidden_states[1:]: post blocks 1..L-1; post-final-norm L; norm again for every logit lens",
        "paper_difference": "paper Eq.8 sums 1..L-1; pinned code includes L and reapplies final norm; this port follows code",
        "prompt_difference": "exact frozen prompt and generated IDs; no official extra space, retokenization, or 12000-character truncation",
        "timing": "token t uses causal position prefix_length+t-1; item mean available after its final overlapping token",
        "aggregation": "arithmetic mean of tokens overlapping each parsed item; no future item score used",
        "items": details, "original_forward_seconds": original_pass_seconds,
        "original_statistics_seconds": original_compute_seconds,
        "random_forward_and_comparison_seconds": time.perf_counter()-random_start,
        "seconds": time.perf_counter()-started, "initial_gpu_allocated_bytes": initial_memory,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "status": "ok",
    }
    return values, metadata
