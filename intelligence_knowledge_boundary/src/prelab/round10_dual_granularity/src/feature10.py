"""Soft question/evidence alignment, with an explicit text-only control block.

No hard entity/attribute gate, hidden metadata, fitted encoder, or future answer
selection. Input-embedding MaxSim is a relevance heuristic, NOT entailment.
All attention/hidden features use the post-read query P+j; probability features
use P+j-1. The original question precedes evidence in the actual causal prompt.

Fixed new block: 4 layer bands x 4 channels = 16; text-only controls: 8.
One backbone replay; no vocabulary logits except 16-token NLL/entropy batches.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half

ROOT = Path(__file__).resolve().parents[1]
R9_SRC = ROOT.parent / "round9_evidence_binding" / "src"
if str(R9_SRC) not in sys.path:
    sys.path.insert(0, str(R9_SRC))
from binding9 import visible_only, digest, _sentence_spans, causal_question_routes
from attention7 import causal_attention_rows, lookback_from_attention, _response_offsets

VERSION = "soft-evidence10-v1"
TEMPERATURE = 0.1
BANDS = ((1, 7), (8, 14), (15, 21), (22, 28))
CHANNELS = (
    "soft_relevance_lookback", "question_attention_overlap",
    "attended_embedding_copy", "copied_source_relative_relevance",
)
NEW_FEATURE_NAMES = [f"layers_{a}_{b}__{c}" for a, b in BANDS for c in CHANNELS]
SURFACE_FEATURE_NAMES = [
    "log_body_tokens", "log_sentence_count", "query_word_coverage",
    "max_lexical_sentence_relevance", "lexical_top_gap",
    "current_piece_present_in_body", "current_piece_best_lexical_relevance",
    "current_piece_present_in_question",
]
# Function-word removal only: no attribute dictionary or supplied entity parser.
STOPWORDS = frozenset("""a an the in on at to of for from how what which who whom
when where why can could would please tell specify identify name list give
explain describe according during as is was were are be been being did does do
has have had provide provided based using and or with by about determine state
mention its their his her it they he she this that these those then than into
under over after before between through per""".split())
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


def _words(text):
    return [normalize(m.group()) for m in WORD.finditer(text)]


def _overlap_ids(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]


def visible_layout(tok, visible, input_token_ids):
    """Only align actual displayed text; do not invoke the Round9 matcher."""
    vis = visible_only(visible)
    if not vis["questions"] or not vis["passages"]:
        raise ValueError("Nonempty visible questions and evidence required")
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": vis["system"]},
         {"role": "user", "content": vis["prompt"]}],
        tokenize=False, add_generation_prompt=True)
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(input_token_ids):
        raise ValueError("Exact input IDs disagree with rendered prompt")
    offsets = encoded["offset_mapping"]
    user = rendered.find(vis["prompt"])
    if user < 0 or rendered.find(vis["prompt"], user + 1) >= 0:
        raise ValueError("Visible prompt must occur uniquely")
    qm, sm = "\n\nQuestions:\n", "\n\nSearch results:\n"
    qstart, sstart = vis["prompt"].find(qm), vis["prompt"].find(sm)
    expected_q = "\n".join(f"{i+1}. {q}" for i, q in enumerate(vis["questions"]))
    expected_s = "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}"
                              for i, p in enumerate(vis["passages"]))
    if qstart < 0 or sstart < 0 or vis["prompt"][qstart+len(qm):sstart] != expected_q:
        raise ValueError("Question list differs from real Questions section")
    if vis["prompt"][sstart+len(sm):] != expected_s:
        raise ValueError("Passage list differs from real Search results section")
    start, end = user + sstart + len(sm), user + len(vis["prompt"])
    context = _overlap_ids(offsets, start, end)
    units, cursor = [], start
    for pi, p in enumerate(vis["passages"]):
        header = f"[{pi+1}] {p['title']}\n"
        if rendered[cursor:cursor+len(header)] != header:
            raise ValueError("Passage order/header mismatch")
        title_start = cursor + len(f"[{pi+1}] ")
        title_ids = _overlap_ids(offsets, title_start, title_start+len(p["title"]))
        body_start = cursor + len(header)
        for a, b in _sentence_spans(p["text"]):
            units.append({"unit_index": len(units), "passage_index": pi,
                          "title": p["title"], "text": p["text"][a:b],
                          "start": body_start+a, "end": body_start+b,
                          "title_token_indices": title_ids, "token_indices": []})
        cursor = body_start + len(p["text"]) + 2
    # Assign straddling tokens once by largest nonspace character intersection.
    for i in context:
        a, b = offsets[i]
        choices = [(sum(not c.isspace() for c in rendered[max(a,u["start"]):min(b,u["end"])]), -j)
                   for j, u in enumerate(units) if b > u["start"] and a < u["end"]]
        choices = [x for x in choices if x[0] > 0]
        if choices:
            units[-max(choices)[1]]["token_indices"].append(i)
    units = [u for u in units if u["token_indices"]]
    if not units:
        raise ValueError("No tokenized evidence sentences; cannot fabricate evidence")
    for i, u in enumerate(units):
        u["unit_index"] = i
        u["representation_token_indices"] = sorted(set(u["token_indices"] + u["title_token_indices"]))
    body = sorted(i for u in units for i in u["token_indices"])
    unit_words = [set(_words(u["title"] + " " + u["text"])) for u in units]
    question_plans, qcursor = [], user + qstart + len(qm)
    for qi, q in enumerate(vis["questions"]):
        begin = qcursor + len(f"{qi+1}. ")
        words = [(normalize(m.group()), begin+m.start(), begin+m.end())
                 for m in WORD.finditer(q) if normalize(m.group()) not in STOPWORDS]
        weight_by_token = {}
        weighted_words = []
        for word, a, b in words:
            ids = _overlap_ids(offsets, a, b)
            if not ids:
                continue
            idf = 1.0 + math.log((1+len(units)) / (1+sum(word in uw for uw in unit_words)))
            weighted_words.append((word, idf))
            for i in ids:
                weight_by_token[i] = weight_by_token.get(i, 0.0) + idf / len(ids)
        positions = sorted(weight_by_token)
        weights = np.asarray([weight_by_token[i] for i in positions], dtype=np.float32)
        if len(weights):
            weights /= weights.sum()
        denom = sum(w for _, w in weighted_words)
        lex = [sum(w for word, w in weighted_words if word in uw) / denom if denom else 0.0
               for uw in unit_words]
        coverage = sum(w for word, w in weighted_words if any(word in uw for uw in unit_words)) / denom if denom else 0.0
        question_plans.append({"question_index": qi, "question_text": q,
                               "content_token_indices": positions, "content_token_weights": weights.tolist(),
                               "content_words": [w for w, _ in weighted_words],
                               "all_question_token_indices": _overlap_ids(offsets, begin, begin+len(q)),
                               "lexical_sentence_relevance": lex, "query_word_coverage": coverage,
                               "query_content_missing": not bool(positions)})
        qcursor = begin + len(q) + 1
    return {"version": VERSION, "input_tokens": len(input_token_ids),
            "rendered_prompt_sha256": digest(rendered), "visible_input_sha256": digest(vis),
            "context_token_indices": context, "body_token_indices": body,
            "sentence_units": units, "question_plans": question_plans}


def semantic_sentence_priors(embedding, plan):
    """IDF-weighted MaxSim of question BPE embeddings to sentence+visible title.

    Input embeddings are frozen/static; the question has not read later sources.
    Every real sentence gets nonzero soft weight; an empty query is explicitly
    uniform, not 'sufficient evidence'. No threshold, top-k gate, or fitted term.
    """
    e = F.normalize(embedding.float(), dim=-1)
    priors, scores = [], []
    for q in plan["question_plans"]:
        if q["content_token_indices"]:
            qe = e[q["content_token_indices"]]
            w = torch.as_tensor(q["content_token_weights"], device=e.device)
            vals = [(torch.clamp(qe @ e[u["representation_token_indices"]].T, 0, 1).max(-1).values * w).sum()
                    for u in plan["sentence_units"]]
            values = torch.stack(vals)
        else:
            values = torch.zeros(len(plan["sentence_units"]), device=e.device)
        scores.append(values)
        priors.append((values / TEMPERATURE).softmax(-1))
    return torch.stack(priors), torch.stack(scores)


def _piece(tok, token_id):
    p = normalize(tok.decode([int(token_id)], clean_up_tokenization_spaces=False)).strip()
    return p if "\ufffd" not in p and any(c.isalnum() for c in p) else ""


def surface_controls(tok, plan, prefix, response, routes, route_valid):
    """Visible text only; uses current token piece, never the completed word."""
    pieces = [_piece(tok, i) for i in prefix]
    body_pieces = {pieces[i] for i in plan["body_token_indices"]} - {""}
    unit_piece_sets = [{pieces[i] for i in u["token_indices"]} - {""} for u in plan["sentence_units"]]
    output = np.zeros((len(response), len(SURFACE_FEATURE_NAMES)), dtype=np.float32)
    for j, token_id in enumerate(response):
        output[j, :2] = [math.log1p(len(plan["body_token_indices"])), math.log1p(len(plan["sentence_units"]))]
        if not route_valid[j]:
            continue
        q = plan["question_plans"][int(routes[j])]
        lex = q["lexical_sentence_relevance"]
        ordered = sorted(lex, reverse=True)
        p = _piece(tok, token_id)
        qpieces = {pieces[i] for i in q["all_question_token_indices"]} - {""}
        output[j, 2:] = [q["query_word_coverage"], ordered[0],
                         ordered[0]-(ordered[1] if len(ordered)>1 else 0),
                         float(bool(p) and p in body_pieces),
                         max([lex[i] for i, us in enumerate(unit_piece_sets) if p and p in us] or [0.0]),
                         float(bool(p) and p in qpieces)]
    return output


def soft_alignment_from_attention(attention, body_indices, unit_assignment,
                                  sentence_priors, copy_similarity,
                                  response_start, positions, route_valid):
    """H x Q x K -> H x Q x 4, with exact hand-testable formulas.

    b(k)=a(k)/sum_body a; q(s) is soft query relevance; n_s body tokens.
    r(k)=q(s(k))/n_s, so sum_body r=1 (uniform token prior -> ordinary body LB).
    Channels: <a,r>/(<a,r>+mean_response a); BC(sum_k_in_s b,q);
    <b,copy>; g/(1+g), g=C*<b*copy,r>/<b,copy> (0 if no copy).
    Copy is relu cosine of this token's input embedding to source token embedding.
    """
    source = attention.index_select(-1, body_indices).float()
    tiny = torch.finfo(torch.float32).tiny
    count = source.shape[-1]
    units = unit_assignment.shape[-1]
    lengths = unit_assignment.sum(0).clamp_min(1)
    token_prior = (sentence_priors / lengths[None]) @ unit_assignment.T
    relevance_density = (source * token_prior[None]).sum(-1)
    keypos = torch.arange(attention.shape[-1], device=attention.device)
    response_mask = ((keypos[None] >= response_start) & (keypos[None] <= positions[:, None]))
    response_density = (attention * response_mask[None]).sum(-1) / response_mask.sum(-1)[None].clamp_min(1)
    relevant_lb = relevance_density / (relevance_density + response_density).clamp_min(tiny)
    body_conditional = source / source.sum(-1, keepdim=True).clamp_min(tiny)
    sentence_attention = body_conditional @ unit_assignment
    overlap = torch.sqrt((sentence_attention * sentence_priors[None]).clamp_min(0)).sum(-1)
    weighted_copy = body_conditional * copy_similarity[None]
    copy_mass = weighted_copy.sum(-1)
    relative = count * (weighted_copy * token_prior[None]).sum(-1) / copy_mass.clamp_min(tiny)
    copied_relevance = relative / (1+relative)
    features = torch.stack((relevant_lb, overlap, copy_mass, copied_relevance), -1)
    features[:, ~route_valid, :] = 0
    return features


class _Hooks:
    def __init__(self, model, plan, priors, copy, routes, route_valid, response_count):
        self.model, self.plan, self.priors, self.copy = model, plan, priors, copy
        self.device = model.model.embed_tokens.weight.device
        self.start, self.count = plan["input_tokens"], response_count
        self.heads, self.layers = model.config.num_attention_heads, len(model.model.layers)
        self.positions = torch.arange(self.start, self.start+self.count, device=self.device)
        self.context = torch.as_tensor(plan["context_token_indices"], device=self.device)
        self.body = torch.as_tensor(plan["body_token_indices"], device=self.device)
        self.routes = torch.as_tensor(routes, device=self.device)
        self.valid = torch.as_tensor(route_valid, device=self.device)
        index = {int(k): i for i, k in enumerate(plan["body_token_indices"])}
        self.assignment = torch.zeros(len(self.body), len(plan["sentence_units"]), device=self.device)
        for si, u in enumerate(plan["sentence_units"]):
            self.assignment[[index[k] for k in u["token_indices"]], si] = 1
        self.lookback = np.empty((self.count, self.layers, self.heads), dtype=np.float32)
        self.features = np.zeros((self.count, 4, len(CHANNELS)), dtype=np.float32)
        self.pending, self.handles, self.seen, self.hidden21 = {}, [], set(), None

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            attn = block.self_attn
            def before(module, args, kwargs, layer=layer):
                if kwargs.get("past_key_value") is not None:
                    raise ValueError("Feature replay must disable KV cache")
                self.pending[layer] = {"rope": kwargs["position_embeddings"]}
            def query(module, args, output, layer=layer):
                self.pending[layer]["q"] = output
            def key(module, args, output, layer=layer, attn=attn):
                self._attention(layer, attn, output)
            self.handles.extend([attn.register_forward_pre_hook(before, with_kwargs=True),
                                 attn.q_proj.register_forward_hook(query),
                                 attn.k_proj.register_forward_hook(key)])
        def h21(module, args, result):
            value = result[0] if isinstance(result, tuple) else result
            self.hidden21 = value[0, self.positions].float().cpu().numpy()
        self.handles.append(self.model.model.layers[20].register_forward_hook(h21))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def _attention(self, layer, module, projection):
        state = self.pending.pop(layer)
        dim = module.head_dim
        q_all = state["q"].view(1, -1, self.heads, dim).transpose(1, 2)
        k = projection.view(1, -1, self.model.config.num_key_value_heads, dim).transpose(1, 2)
        cos, sin = state["rope"]
        k = repeat_kv(k*cos[:, None] + rotate_half(k)*sin[:, None], module.num_key_value_groups)[0]
        for begin in range(0, self.count, 8):
            end = min(begin+8, self.count)
            pos = self.positions[begin:end]
            q = q_all[0].index_select(1, pos)
            qc, qs = cos[0].index_select(0, pos), sin[0].index_select(0, pos)
            q = q*qc[None] + rotate_half(q)*qs[None]
            attention = causal_attention_rows(q, k, pos, module.scaling)
            self.lookback[begin:end, layer] = lookback_from_attention(
                attention, self.context, self.start, pos).T.cpu().numpy()
            prior = self.priors.index_select(0, self.routes[begin:end].clamp_min(0))
            features = soft_alignment_from_attention(
                attention, self.body, self.assignment, prior, self.copy[begin:end],
                self.start, pos, self.valid[begin:end])
            self.features[begin:end, layer//7] += features.mean(0).cpu().numpy()/7.0
        self.seen.add(layer)


@torch.inference_mode()
def extract_features(tokenizer, model, visible, generated):
    """Exact causal token features, no labels, scores, standard answers, or fit."""
    started = time.perf_counter()
    tok, vis = tokenizer, visible_only(visible)
    if model.training or model.config.model_type != "qwen2" or len(model.model.layers) != 28:
        raise ValueError("Frozen 28-layer Qwen2 model required")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("Sliding-window masks unsupported")
    prefix = [int(i) for i in generated["input_token_ids"]]
    response = [int(i) for i in generated["response_token_ids"]]
    if not response or generated.get("unexpected_special_token_ids"):
        raise ValueError("Nonempty exact ordinary response tokens required")
    offsets = _response_offsets(tok, response, generated["response"])
    if "response_token_offsets" in generated and not np.array_equal(offsets, generated["response_token_offsets"]):
        raise ValueError("Saved offsets disagree with exact generated bytes")
    plan = visible_layout(tok, vis, prefix)
    routes, valid = causal_question_routes(tok, response, len(vis["questions"]))
    surface = surface_controls(tok, plan, prefix, response, routes, valid)
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([prefix+response], device=device)
    embeddings = F.normalize(model.model.embed_tokens(ids)[0].float(), dim=-1)
    priors, raw_scores = semantic_sentence_priors(embeddings[:len(prefix)], plan)
    # Every row depends only on that token's identity; no across-response pooling.
    copy = (embeddings[len(prefix):] @ embeddings[plan["body_token_indices"]].T).clamp(0, 1)
    del embeddings
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-started
    replay_start = time.perf_counter()
    with _Hooks(model, plan, priors, copy, routes, valid, len(response)) as hooks:
        output = model.model(input_ids=ids, use_cache=False, output_attentions=False, output_hidden_states=False)
    if len(hooks.seen) != 28 or hooks.hidden21 is None:
        raise RuntimeError("Not all feature hooks ran")
    last = output.last_hidden_state[0]
    pos = torch.arange(len(prefix), len(prefix)+len(response), device=device)
    nll, entropy = [], []
    for pp in pos.split(16):
        lp = model.lm_head(last.index_select(0, pp-1)).float().log_softmax(-1)
        nll.extend((-lp.gather(1, ids[0, pp, None])).squeeze(-1).cpu().tolist())
        entropy.extend((-(lp.exp()*lp).sum(-1)).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    arrays = {"new_features": hooks.features.reshape(len(response), -1),
              "surface_features": surface, "lookback_features": hooks.lookback.reshape(len(response), -1),
              "hidden_28": last.index_select(0, pos).float().cpu().numpy(), "hidden_21": hooks.hidden21,
              "token_nll": np.asarray(nll, dtype=np.float32), "token_entropy": np.asarray(entropy, dtype=np.float32),
              "token_ids": np.asarray(response, dtype=np.int64), "token_start": offsets[:, 0], "token_end": offsets[:, 1],
              "response_token_offsets": offsets, "token_question_index": routes, "token_route_valid": valid}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite features; do not silently fill")
    meta = {"version": VERSION, "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "visible_input_sha256": digest(vis), "input_token_ids_sha256": digest(prefix),
            "response_token_ids_sha256": digest(response), "response_sha256": digest(generated["response"]),
            "layout": plan, "semantic_sentence_prior": priors.cpu().tolist(), "semantic_sentence_raw_score": raw_scores.cpu().tolist(),
            "new_feature_names": NEW_FEATURE_NAMES, "surface_feature_names": SURFACE_FEATURE_NAMES,
            "new_dimensions": 16, "surface_dimensions": 8, "semantic_temperature": TEMPERATURE,
            "query_representation": "frozen static input embeddings; question has not integrated later evidence",
            "candidate_policy": "all real sentences kept with positive soft weight; no entity/attribute AND gate",
            "new_feature_timing": "post-read P+j; no future response selection or completed-word matching",
            "probability_timing": "predict current token from P+j-1", "causal_passes": 1,
            "query_batch": 8, "logit_batch": 16, "route_invalid_tokens": int((~valid).sum()),
            "query_content_missing": sum(q["query_content_missing"] for q in plan["question_plans"]),
            "labels_or_detector_scores_read": False, "hidden_metadata_used": False,
            "full_future_output_used_for_candidate_selection": False,
            "preparation_seconds": preparation_seconds, "replay_seconds": time.perf_counter()-replay_start,
            "seconds": time.perf_counter()-started,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device)/2**30) if device.type == "cuda" else None}
    return arrays, meta
