"""Executable GPU feature runner for forced-evidence quote probe v2.

The runner consumes only the frozen label-free manifest.  It exposes explicit
``gpu-smoke`` and ``extract`` stages, but both are review-gated and are never
called by the CPU selfcheck.  P3 greedily generates with KV cache, discards the
cache, and then obtains hidden/attention only from a full ``use_cache=False``
replay.  Cached logits are the sole source for generated-token statistics;
replay logits are descriptive numerical QA only.  Cross-path numerical gates
run only on the frozen eight-claim smoke.  P2 tokenizes the complete
prompt+mechanical quote once and uses a teacher-forced no-cache forward.  Every
formal claim is committed atomically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
from functools import lru_cache
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import tempfile
import time
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

import feature_qa as feature_qa
import run_forced_evidence_quote_probe_v1 as cpu
import run_retrieved_evidence_nli_v1 as bm25_source
import run_exact_subset_attribution_v2 as wddm_gate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V1_OUT = ROOT / "results/forced_evidence_quote_probe_v1"
V1_RESEARCH = ROOT / "research/forced_evidence_quote_probe_v1"
OUT = ROOT / "results/forced_evidence_quote_probe_v2"
RESEARCH = ROOT / "research/forced_evidence_quote_probe_v2"
MODEL = ROOT.parent / "models/Llama-2-7b-chat-hf"
MODEL_MANIFEST = ROOT / "model_download_manifest.json"
PCA_PATH = ROOT / "results/development_v1/hidden_pca.pkl"
PLAN_PATH = V1_RESEARCH / "PLAN.json"
PROTOCOL_PATH = V1_RESEARCH / "PROTOCOL.md"
NUMERICAL_PROTOCOL_PATH = RESEARCH / "NUMERICAL_PROTOCOL.md"
PREPARATION_PATH = V1_OUT / "PREPARATION.json"
INPUT_PATH = V1_OUT / "label_free_inputs.jsonl"
MANIFEST_PATH = V1_OUT / "MANIFEST.json"
CPU_SELFCHECK_PATH = OUT / "GPU_RUNNER_CPU_SELFCHECK.json"
RUNNER_REVIEW_PATH = RESEARCH / "GPU_RUNNER_INDEPENDENT_REVIEW.json"
SMOKE_REVIEW_PATH = RESEARCH / "GPU_SMOKE_INDEPENDENT_REVIEW.json"
SMOKE_PATH = OUT / "GPU_SMOKE.json"
RECORD_DIR = OUT / "gpu_claim_records"
EXTRACTION_COMPLETE = OUT / "GPU_EXTRACTION_COMPLETE.json"
GLOBAL_GPU_LOCK = ROOT / "results/.exclusive_gpu_runner.lock"

VERSION = "forced-evidence-quote-probe-gpu-runner-v2"
EXPECTED_PROTOCOL_SHA256 = "78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa"
EXPECTED_PLAN_SHA256 = "6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c"
EXPECTED_NUMERICAL_PROTOCOL_SHA256 = "ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1"
EXPECTED_INPUT_SHA256 = "ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777"
FIXED_SMOKE_INDICES = [2826, 994, 1533, 1695, 1299, 1143, 3352, 3752]
FIXED_SMOKE_PROMPT_LENGTHS = [255, 354, 385, 435, 508, 568, 634, 799]
SMOKE_ABSOLUTE_TOLERANCES = {
    "selected_logprob": 0.0625,
    "vocab_entropy": 0.0625,
    "signed_margin": 0.5,
}
NEAR_TIE_MARGIN = 0.5
REPEAT_ATOL = 1e-6
REPO = "NousResearch/Llama-2-7b-chat-hf"
REVISION = "351844e75ed0bcbbe3f10671b3c808d2b83894ee"
SEED = 20_261_023
THREADS = 4
EXPECTED_ROWS = 3_776
EXPECTED_ANSWERS = 256
MAX_NEW_TOKENS = 160
MAX_CONTEXT = 4_096
QUERY_CHUNK = 8
LOGIT_BATCH = 16
MIN_FREE_GPU_BYTES = int(6.0 * 1024 ** 3)
LAYERS = 32
HEADS = 32
HIDDEN = 4_096
PCA_WIDTH = 64
ATTENTION_WIDTH = 256
SURFACE_WIDTH = 21
TOKEN_WIDTH = 11
RELATION_WIDTH = 5
HIDDEN_SUMMARY_WIDTH = 256
WHITEBOX_WIDTH = 549
CHOICE_IDS = np.asarray([319, 350, 315], dtype=np.int64)
BYTE_FALLBACK_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>\Z")
EXPECTED_TOKENIZER_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}

MODEL_LOAD = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_storage": "uint8",
    "llm_int8_skip_modules": ["lm_head"],
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "device_map": {"": "cuda:0"},
    "local_files_only": True,
    "trust_remote_code": False,
    "tf32_allowed": False,
    "batch_size": 1,
}

FLOAT_ARRAY_KEYS = frozenset({
    "p1_features",
    "p2_surface21", "p2_token_summary11", "p2_relation5",
    "p2_hidden_summary256", "p2_attention_summary256", "p2_features",
    "p2_selected_logprob", "p2_vocab_entropy", "p2_signed_margin",
    "p2_hidden64", "p2_ratio_token_mean", "p2_ratio_token_min",
    "p2_ratio_token_max", "p2_passage_mass_mean", "p2_claim_mass_mean",
    "p3_surface21", "p3_token_summary11", "p3_relation5",
    "p3_hidden_summary256", "p3_attention_summary256", "p3_features",
    "p3_generated_selected_logprob", "p3_generated_vocab_entropy",
    "p3_generated_signed_margin", "p3_replay_selected_logprob",
    "p3_replay_vocab_entropy", "p3_replay_signed_margin",
    "p3_selected_logprob", "p3_vocab_entropy", "p3_signed_margin",
    "p3_hidden64", "p3_ratio_token_mean", "p3_ratio_token_min",
    "p3_ratio_token_max", "p3_passage_mass_mean", "p3_claim_mass_mean",
})
INT_ARRAY_KEYS = frozenset({
    "p2_content_token_ids", "p2_content_absolute_positions",
    "p3_generated_token_ids", "p3_replay_argmax_token_ids",
    "p3_generated_char_offsets", "p3_content_generated_indices",
    "p3_content_token_ids", "p3_content_absolute_positions",
})
ARRAY_KEYS = FLOAT_ARRAY_KEYS | INT_ARRAY_KEYS


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "CPU stage initialized CUDA"


def atomic_json(path: Path, value, *, refuse_overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite:
        assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def atomic_npz(path: Path, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def load_inputs():
    rows, manifest = cpu.load_sanitized_rows()
    assert len(rows) == EXPECTED_ROWS
    assert len({row["response_id"] for row in rows}) == EXPECTED_ANSWERS
    assert Path(cpu.INPUT_PATH).resolve() == INPUT_PATH.resolve()
    assert Path(cpu.MANIFEST_PATH).resolve() == MANIFEST_PATH.resolve()
    assert sha(INPUT_PATH) == EXPECTED_INPUT_SHA256
    return rows, manifest


def load_plan_and_check():
    plan = read_json(PLAN_PATH)
    preparation = read_json(PREPARATION_PATH)
    assert sha(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256
    assert sha(PLAN_PATH) == EXPECTED_PLAN_SHA256
    assert sha(NUMERICAL_PROTOCOL_PATH) == EXPECTED_NUMERICAL_PROTOCOL_SHA256
    assert preparation["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert preparation["plan_sha256"] == sha(PLAN_PATH)
    assert plan["scope"]["selected_atomic_claims"] == EXPECTED_ROWS
    assert plan["decoding"]["max_new_tokens_including_close"] == MAX_NEW_TOKENS
    assert plan["features"]["P2_total_dim"] == WHITEBOX_WIDTH
    assert plan["features"]["P3_total_dim"] == WHITEBOX_WIDTH
    assert plan["features"]["attention_classifier_view_dim"] == ATTENTION_WIDTH
    assert plan["whitebox_execution"]["P3_authoritative_whitebox_replay"]["use_cache"] is False
    assert plan["execution_gates"]["existing_qk_hook_may_run_with_kv_cache"] is False
    assert plan["relation_readout"]["choice_token_ids"] == {"A": 319, "B": 350, "C": 315}
    return plan


def tokenizer_load():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    assert tokenizer.is_fast
    assert [tokenizer.encode(letter, add_special_tokens=False)
            for letter in ("A", "B", "C")] == [[319], [350], [315]]
    return tokenizer


def sentence_candidates(row: dict):
    """Return the frozen global mechanical quote and every scored sentence."""
    claim = row["claim_prompt_text"]
    every, tops = [], []
    for passage_id in (1, 2, 3):
        body = row[f"passage_{passage_id}"]
        spans = bm25_source.sentence_spans(body)
        sentences = []
        for sentence_id, (left, right) in enumerate(spans):
            text = body[left:right]
            sentences.append({
                "sentence_id": sentence_id, "text": text,
                "text_sha256": digest(text),
            })
        ranked = bm25_source.bm25_rank(claim, sentences)
        by_id = {item["sentence_id"]: item for item in ranked}
        for sentence in sentences:
            score = by_id[sentence["sentence_id"]]
            every.append({
                "passage_id": passage_id,
                "sentence_id": sentence["sentence_id"],
                "text": sentence["text"],
                "text_sha256": sentence["text_sha256"],
                "bm25": float(score["bm25"]),
            })
        top = ranked[0]
        sentence = sentences[top["sentence_id"]]
        tops.append({
            "passage_id": passage_id,
            "sentence_id": sentence["sentence_id"],
            "text": sentence["text"],
            "text_sha256": sentence["text_sha256"],
            "bm25": float(top["bm25"]),
        })
    mechanical = min(
        tops, key=lambda item: (-item["bm25"], item["passage_id"],
                                item["sentence_id"], item["text_sha256"]))
    return mechanical, every


def prompt_ids_and_regions(tokenizer, plan: dict, row: dict):
    prompt, char_regions = cpu.render_prompt_with_regions(
        plan["quote_prompt"]["utf8_template"], row)
    encoded = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True)
    ids = list(map(int, encoded["input_ids"]))
    assert ids and ids[0] == tokenizer.bos_token_id
    assert ids.count(tokenizer.bos_token_id) == 1
    assert len(ids) <= plan["decoding"]["observed_max_input_tokens"]
    regions = cpu.region_token_positions(
        prompt, encoded["offset_mapping"], char_regions)
    return prompt, ids, regions


def fixed_smoke_indices(rows, tokenizer, plan):
    lengths = []
    for row in rows:
        prompt, _regions = cpu.render_prompt_with_regions(
            plan["quote_prompt"]["utf8_template"], row)
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    order = sorted(range(len(rows)), key=lambda index:
                   (lengths[index], rows[index]["microclaim_id"]))
    points = [0, len(order) // 7, 2 * len(order) // 7,
              3 * len(order) // 7, 4 * len(order) // 7,
              5 * len(order) // 7, 6 * len(order) // 7, len(order) - 1]
    chosen = [order[position] for position in points]
    assert len(set(chosen)) == 8
    assert chosen == FIXED_SMOKE_INDICES
    assert [lengths[index] for index in chosen] == FIXED_SMOKE_PROMPT_LENGTHS
    return chosen, lengths


def _sorted_unique_positions(values, sequence_length: int, name: str):
    result = sorted({int(value) for value in values})
    assert result and result[0] >= 0 and result[-1] < sequence_length, name
    assert len(result) == len(values), f"Duplicate {name} token position"
    return result


def summarize_attention(ratio: np.ndarray, passage_mass: np.ndarray,
                        claim_mass: np.ndarray):
    """Create frozen raw summaries and the exact 256-column classifier view."""
    assert ratio.ndim == 3
    tokens = ratio.shape[0]
    layers, heads = ratio.shape[1:]
    assert tokens > 0
    assert passage_mass.shape == (tokens, layers, heads, 3)
    assert claim_mass.shape == (tokens, layers, heads)
    assert all(np.isfinite(value).all()
               for value in (ratio, passage_mass, claim_mass))
    assert np.all((ratio >= 0) & (ratio <= 1))
    assert np.all((passage_mass >= 0) & (passage_mass <= 1 + 1e-5))
    assert np.all((claim_mass >= 0) & (claim_mass <= 1 + 1e-5))

    ratio_mean = ratio.mean(axis=0, dtype=np.float32)
    ratio_min = ratio.min(axis=0)
    ratio_max = ratio.max(axis=0)
    passage_mean = passage_mass.mean(axis=(0, 2), dtype=np.float32).T
    claim_mean = claim_mass.mean(axis=(0, 2), dtype=np.float32)
    view = np.concatenate([
        ratio.mean(axis=(0, 2), dtype=np.float32),
        ratio.mean(axis=2, dtype=np.float32).min(axis=0),
        ratio.mean(axis=2, dtype=np.float32).max(axis=0),
        ratio.mean(axis=0, dtype=np.float32).std(axis=1, ddof=0),
        passage_mean.reshape(-1),
        claim_mean,
    ]).astype(np.float32)
    assert view.shape == (layers * 8,) and np.isfinite(view).all()
    raw = {
        "ratio_token_mean": ratio_mean.astype(np.float16),
        "ratio_token_min": ratio_min.astype(np.float16),
        "ratio_token_max": ratio_max.astype(np.float16),
        "passage_mass_mean": passage_mean.astype(np.float16),
        "claim_mass_mean": claim_mean.astype(np.float16),
    }
    assert raw["ratio_token_mean"].shape == (layers, heads)
    assert raw["passage_mass_mean"].shape == (3, layers)
    return raw, view


def zero_attention():
    raw = {
        "ratio_token_mean": np.zeros((LAYERS, HEADS), dtype=np.float16),
        "ratio_token_min": np.zeros((LAYERS, HEADS), dtype=np.float16),
        "ratio_token_max": np.zeros((LAYERS, HEADS), dtype=np.float16),
        "passage_mass_mean": np.zeros((3, LAYERS), dtype=np.float16),
        "claim_mass_mean": np.zeros(LAYERS, dtype=np.float16),
    }
    return raw, np.zeros(ATTENTION_WIDTH, dtype=np.float32)


class QuoteAttentionHooks:
    """Bounded Q/K reconstruction for the frozen quote-attention features."""

    def __init__(self, model, query_positions, passage_positions,
                 claim_positions, sequence_length, query_chunk=QUERY_CHUNK):
        self.model = model
        self.spec = feature_qa.architecture(model)
        self.layers = self.spec["layers"]
        self.heads = self.spec["heads"]
        self.sequence_length = int(sequence_length)
        self.device = model.model.embed_tokens.weight.device
        self.query_positions = _sorted_unique_positions(
            query_positions, self.sequence_length, "quote query")
        assert len(passage_positions) == 3
        self.passage_positions = [
            _sorted_unique_positions(values, self.sequence_length,
                                     f"passage_{index}")
            for index, values in enumerate(passage_positions, 1)
        ]
        self.claim_positions = _sorted_unique_positions(
            claim_positions, self.sequence_length, "claim")
        union = [item for values in self.passage_positions for item in values]
        assert len(union) == len(set(union))
        assert not (set(union) & set(self.claim_positions))
        assert max(union + self.claim_positions) < min(self.query_positions)
        self.query_chunk = int(query_chunk)
        assert self.query_chunk > 0
        tokens = len(self.query_positions)
        self.ratio = np.empty((tokens, self.layers, self.heads), dtype=np.float32)
        self.passage_mass = np.empty(
            (tokens, self.layers, self.heads, 3), dtype=np.float32)
        self.claim_mass = np.empty(
            (tokens, self.layers, self.heads), dtype=np.float32)
        self.handles, self.pending, self.seen = [], {}, set()

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            attention = block.self_attn

            def before(module, args, kwargs, layer=layer):
                assert kwargs.get("past_key_value") is None
                assert kwargs.get("use_cache") in (None, False)
                assert "position_embeddings" in kwargs
                self.pending[layer] = {"rope": kwargs["position_embeddings"]}

            def query(module, args, output, layer=layer):
                assert "q" not in self.pending[layer]
                self.pending[layer]["q"] = output

            def key(module, args, output, layer=layer, attention=attention):
                self._capture(layer, attention, output)

            self.handles.extend((
                attention.register_forward_pre_hook(before, with_kwargs=True),
                attention.q_proj.register_forward_hook(query),
                attention.k_proj.register_forward_hook(key),
            ))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()

    def _capture(self, layer, module, keys):
        state = self.pending.pop(layer)
        heads = self.spec["heads"]
        kv_heads = self.spec["kv_heads"]
        dim = self.spec["head_dim"]
        assert keys.shape[0] == state["q"].shape[0] == 1
        assert keys.shape[1] == state["q"].shape[1] == self.sequence_length
        query = state["q"].view(1, -1, heads, dim).transpose(1, 2)
        key = keys.view(1, -1, kv_heads, dim).transpose(1, 2)
        cos, sin = state["rope"]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, heads // kv_heads)[0]
        query = query[0]
        absolute = torch.arange(self.sequence_length, device=self.device)
        quote_positions = torch.tensor(
            self.query_positions, dtype=torch.long, device=self.device)
        source_positions = torch.tensor(
            [item for values in self.passage_positions for item in values],
            dtype=torch.long, device=self.device)
        passage_positions = [torch.tensor(values, dtype=torch.long,
                                          device=self.device)
                             for values in self.passage_positions]
        claim_positions = torch.tensor(
            self.claim_positions, dtype=torch.long, device=self.device)
        tiny = torch.finfo(torch.float32).tiny
        for begin in range(0, len(self.query_positions), self.query_chunk):
            end = min(begin + self.query_chunk, len(self.query_positions))
            pos = quote_positions[begin:end]
            q = query.index_select(1, pos)
            logits = (q @ key.transpose(-2, -1)) * module.scaling
            logits.masked_fill_(absolute[None, None, :] > pos[None, :, None],
                                float("-inf"))
            weights = logits.float().softmax(-1)
            source_mean = weights.index_select(-1, source_positions).mean(-1)
            quote_attention = weights.index_select(-1, quote_positions)
            through = quote_positions[None, :] <= pos[:, None]
            quote_mean = (
                quote_attention * through[None]
            ).sum(-1) / through.sum(-1)[None]
            ratio = source_mean / (source_mean + quote_mean).clamp_min(tiny)
            self.ratio[begin:end, layer] = ratio.T.detach().cpu().numpy()
            for passage_index, passage in enumerate(passage_positions):
                mass = weights.index_select(-1, passage).sum(-1)
                self.passage_mass[begin:end, layer, :, passage_index] = (
                    mass.T.detach().cpu().numpy())
            mass = weights.index_select(-1, claim_positions).sum(-1)
            self.claim_mass[begin:end, layer] = mass.T.detach().cpu().numpy()
            del q, logits, weights, source_mean, quote_attention, quote_mean, ratio
        self.seen.add(layer)

    def finish(self):
        assert len(self.seen) == self.layers and not self.pending
        return summarize_attention(self.ratio, self.passage_mass,
                                   self.claim_mass)


def token_summary(logprob, entropy, margin):
    logprob = np.asarray(logprob, dtype=np.float32)
    entropy = np.asarray(entropy, dtype=np.float32)
    margin = np.asarray(margin, dtype=np.float32)
    assert logprob.shape == entropy.shape == margin.shape
    count = len(logprob)
    if not count:
        return np.zeros(TOKEN_WIDTH, dtype=np.float32)
    assert all(np.isfinite(value).all() for value in (logprob, entropy, margin))
    if count == 1:
        slope = 0.0
    else:
        x = np.arange(count, dtype=np.float64) / (count - 1)
        centered = x - x.mean()
        slope = float(np.dot(centered, logprob.astype(np.float64) -
                             float(logprob.mean(dtype=np.float64))) /
                      np.dot(centered, centered))
    values = np.asarray([
        logprob.mean(dtype=np.float64), logprob.std(ddof=0, dtype=np.float64),
        logprob.min(), np.percentile(logprob, 10, method="linear"),
        logprob[0], logprob[-1], slope,
        entropy.mean(dtype=np.float64), entropy.max(),
        margin.mean(dtype=np.float64), margin.min(),
    ], dtype=np.float32)
    assert values.shape == (TOKEN_WIDTH,) and np.isfinite(values).all()
    return values


def project_hidden(hidden: np.ndarray, pca: dict):
    hidden = np.asarray(hidden, dtype=np.float32)
    assert hidden.ndim == 2 and hidden.shape[1] == HIDDEN
    assert pca["mean"].shape == (HIDDEN,)
    assert pca["components"].shape == (PCA_WIDTH, HIDDEN)
    projected = ((hidden.astype(np.float64) - pca["mean"]) @
                 pca["components"].T).astype(np.float32)
    assert projected.shape == (len(hidden), PCA_WIDTH)
    assert np.isfinite(projected).all()
    stored = projected.astype(np.float16)
    assert np.isfinite(stored).all()
    return stored


def hidden_summary(hidden64: np.ndarray):
    hidden64 = np.asarray(hidden64)
    assert hidden64.ndim == 2 and hidden64.shape[1] == PCA_WIDTH
    if not len(hidden64):
        return np.zeros(HIDDEN_SUMMARY_WIDTH, dtype=np.float32)
    values = hidden64.astype(np.float32)
    result = np.concatenate((values.mean(axis=0), values.std(axis=0, ddof=0),
                             values[0], values[-1])).astype(np.float32)
    assert result.shape == (HIDDEN_SUMMARY_WIDTH,) and np.isfinite(result).all()
    return result


def combine_features(surface, tokens, relation, hidden, attention):
    result = np.concatenate((surface, tokens, relation, hidden, attention)).astype(np.float32)
    assert result.shape == (WHITEBOX_WIDTH,) and np.isfinite(result).all()
    return result


def _logit_statistics(logits: torch.Tensor, target_ids: torch.Tensor):
    """Float32 selected logprob, entropy, signed margin and argmax."""
    logits = logits.float()
    assert logits.ndim == 2 and target_ids.shape == (logits.shape[0],)
    logp = logits.log_softmax(-1)
    selected = logp.gather(1, target_ids[:, None]).squeeze(1)
    probability = logp.exp()
    entropy = -(probability * logp).sum(-1)
    chosen_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, target_ids[:, None], float("-inf"))
    margin = chosen_logits - other.max(-1).values
    argmax = logits.argmax(-1)
    arrays = tuple(value.detach().cpu().numpy() for value in
                   (selected, entropy, margin, argmax))
    assert all(np.isfinite(value).all() for value in arrays[:3])
    return arrays


@torch.inference_mode()
def cached_greedy_generate(model, tokenizer, prompt_ids, plan):
    """Generate IDs/stats only.  No Q/K hooks may be attached here."""
    assert not any(module._forward_hooks or module._forward_pre_hooks
                   for block in model.model.layers
                   for module in (block.self_attn, block.self_attn.q_proj,
                                  block.self_attn.k_proj))
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    output = model.model(input_ids=ids, attention_mask=mask, use_cache=True,
                         output_attentions=False, output_hidden_states=False,
                         return_dict=True)
    past = output.past_key_values
    next_hidden = output.last_hidden_state[:, -1]
    generated, logprob, entropy, margin = [], [], [], []
    terminal_eos_id = None
    stop_reason = None
    decoded = ""
    for step in range(plan["decoding"]["max_new_tokens_including_close"]):
        logits = model.lm_head(next_hidden).float()
        selected_id = int(logits.argmax(-1).item())
        target = torch.tensor([selected_id], dtype=torch.long, device=device)
        lp, ent, mar, argmax = _logit_statistics(logits, target)
        assert int(argmax[0]) == selected_id
        generated.append(selected_id)
        logprob.append(float(lp[0])); entropy.append(float(ent[0]))
        margin.append(float(mar[0]))
        decoded_with_special = tokenizer.decode(
            generated, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        if "</quote>" in decoded_with_special:
            decoded = decoded_with_special
            stop_reason = "close"
            break
        if selected_id == tokenizer.eos_token_id:
            terminal_eos_id = selected_id
            decoded = tokenizer.decode(
                generated[:-1], skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            stop_reason = "eos"
            break
        if step + 1 == plan["decoding"]["max_new_tokens_including_close"]:
            decoded = decoded_with_special
            stop_reason = "max"
            break
        next_ids = target[None]
        full_mask = torch.ones((1, len(prompt_ids) + len(generated)),
                               dtype=torch.long, device=device)
        output = model.model(
            input_ids=next_ids, attention_mask=full_mask,
            past_key_values=past, use_cache=True, output_attentions=False,
            output_hidden_states=False, return_dict=True)
        past = output.past_key_values
        next_hidden = output.last_hidden_state[:, -1]
    assert stop_reason in ("close", "eos", "max")
    del ids, mask, output, past, next_hidden
    arrays = {
        "generated_token_ids": np.asarray(generated, dtype=np.int32),
        "selected_logprob": np.asarray(logprob, dtype=np.float32),
        "vocab_entropy": np.asarray(entropy, dtype=np.float32),
        "signed_margin": np.asarray(margin, dtype=np.float32),
    }
    assert 0 < len(generated) <= MAX_NEW_TOKENS
    assert all(len(value) == len(generated) for value in arrays.values())
    assert all(np.isfinite(arrays[key]).all() for key in
               ("selected_logprob", "vocab_entropy", "signed_margin"))
    return arrays, {
        "decoded_generated_text": decoded,
        "stop_reason": stop_reason,
        "terminal_eos_id": terminal_eos_id,
    }


def generated_token_offsets(tokenizer, generated_ids, terminal_eos_id):
    """Decode actual IDs while preserving each token's final character span."""
    assert_frozen_tokenizer_decoder(tokenizer)
    ids = list(map(int, generated_ids))
    active_count = len(ids)
    if terminal_eos_id is not None:
        assert ids and ids[-1] == int(terminal_eos_id)
        active_count -= 1
    active_ids = ids[:active_count]
    pieces = tokenizer.convert_ids_to_tokens(
        active_ids, skip_special_tokens=False)
    assert len(pieces) == len(active_ids)
    assert all(isinstance(piece, str) for piece in pieces)
    pieces = [piece.replace("▁", " ") for piece in pieces]

    offsets = [None] * len(active_ids)
    chunks = []
    cursor = 0
    index = 0
    while index < len(pieces):
        match = BYTE_FALLBACK_TOKEN.fullmatch(pieces[index])
        if match is None:
            piece = pieces[index]
            chunks.append(piece)
            offsets[index] = (cursor, cursor + len(piece))
            cursor += len(piece)
            index += 1
            continue

        stop = index
        byte_values = []
        while stop < len(pieces):
            byte_match = BYTE_FALLBACK_TOKEN.fullmatch(pieces[stop])
            if byte_match is None:
                break
            byte_values.append(int(byte_match.group(1), 16))
            stop += 1
        raw_bytes = bytes(byte_values)
        try:
            decoded_run = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            decoded_run = "\N{REPLACEMENT CHARACTER}" * len(byte_values)
            for local in range(len(byte_values)):
                offsets[index + local] = (
                    cursor + local, cursor + local + 1)
        else:
            byte_cursor = 0
            for char_index, character in enumerate(decoded_run):
                width = len(character.encode("utf-8"))
                interval = (cursor + char_index, cursor + char_index + 1)
                for local in range(byte_cursor, byte_cursor + width):
                    offsets[index + local] = interval
                byte_cursor += width
            assert byte_cursor == len(byte_values)
        chunks.append(decoded_run)
        cursor += len(decoded_run)
        index = stop

    raw_text = "".join(chunks)
    stripped = 1 if raw_text.startswith(" ") else 0
    if stripped:
        offsets = [
            (0, 0) if right <= 1 else (max(0, left - 1), right - 1)
            for left, right in offsets
        ]
    decoded = raw_text[stripped:]
    oracle = tokenizer.decode(
        active_ids, skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    assert decoded == oracle, (
        "Provenance decoder disagrees with frozen tokenizer",
        digest(active_ids), digest(decoded), digest(oracle))
    if terminal_eos_id is not None:
        offsets.append((-1, -1))
    assert len(offsets) == len(ids)
    assert all(pair is not None for pair in offsets)
    assert all(pair == (-1, -1) or
               0 <= pair[0] <= pair[1] <= len(decoded)
               for pair in offsets)
    return decoded, np.asarray(offsets, dtype=np.int32)


@lru_cache(maxsize=4)
def assert_frozen_tokenizer_decoder(tokenizer) -> None:
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    assert backend.get("decoder") == EXPECTED_TOKENIZER_DECODER, (
        "Unsupported tokenizer decoder; provenance mapper needs review",
        backend.get("decoder"))


def generated_content_indices(parsed, offsets):
    text = parsed["raw_inner"]
    left = len(text) - len(text.lstrip(cpu.ASCII_EDGE))
    right = len(text.rstrip(cpu.ASCII_EDGE))
    close_start = len(parsed["raw_inner"])
    output = []
    for index, (start, end) in enumerate(offsets):
        start, end = int(start), int(end)
        if start < 0 or end <= start:
            continue
        overlaps_content = max(start, left) < min(end, right)
        overlaps_close = end > close_start
        if overlaps_content and not overlaps_close:
            output.append(index)
    return output


@torch.inference_mode()
def statistics_from_hidden(model, final_hidden, target_positions, target_ids):
    positions = np.asarray(target_positions, dtype=np.int64)
    targets = np.asarray(target_ids, dtype=np.int64)
    assert positions.shape == targets.shape and np.all(positions > 0)
    count = len(positions)
    selected = np.empty(count, dtype=np.float32)
    entropy = np.empty(count, dtype=np.float32)
    margin = np.empty(count, dtype=np.float32)
    argmax = np.empty(count, dtype=np.int32)
    device = final_hidden.device
    for begin in range(0, count, LOGIT_BATCH):
        end = min(begin + LOGIT_BATCH, count)
        pos = torch.tensor(positions[begin:end] - 1, dtype=torch.long,
                           device=device)
        target = torch.tensor(targets[begin:end], dtype=torch.long,
                              device=device)
        logits = model.lm_head(final_hidden.index_select(0, pos))
        lp, ent, mar, top = _logit_statistics(logits, target)
        selected[begin:end] = lp; entropy[begin:end] = ent
        margin[begin:end] = mar; argmax[begin:end] = top.astype(np.int32)
    assert all(np.isfinite(value).all() for value in
               (selected, entropy, margin))
    return {
        "selected_logprob": selected, "vocab_entropy": entropy,
        "signed_margin": margin, "argmax_token_ids": argmax,
        "target_token_ids": targets.astype(np.int32, copy=True),
    }


@torch.inference_mode()
def no_cache_quote_forward(model, input_ids, query_positions,
                           passage_positions, claim_positions, pca):
    """One authoritative full replay returning quote white-box components."""
    ids = list(map(int, input_ids))
    query_positions = list(map(int, query_positions))
    assert len(ids) <= MAX_CONTEXT
    device = model.model.embed_tokens.weight.device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tensor)
    if query_positions:
        with QuoteAttentionHooks(
                model, query_positions, passage_positions, claim_positions,
                len(ids)) as hooks:
            final = model.model(
                input_ids=tensor, attention_mask=mask, use_cache=False,
                past_key_values=None, output_attentions=False,
                output_hidden_states=False, return_dict=True).last_hidden_state[0]
        raw_attention, attention_view = hooks.finish()
        positions = torch.tensor(query_positions, dtype=torch.long,
                                 device=device)
        hidden = final.index_select(0, positions).float().cpu().numpy()
        hidden64 = project_hidden(hidden, pca)
    else:
        final = model.model(
            input_ids=tensor, attention_mask=mask, use_cache=False,
            past_key_values=None, output_attentions=False,
            output_hidden_states=False, return_dict=True).last_hidden_state[0]
        raw_attention, attention_view = zero_attention()
        hidden64 = np.zeros((0, PCA_WIDTH), dtype=np.float16)
    return final, hidden64, raw_attention, attention_view


def relation_context_quote(parsed: dict) -> str:
    """Exclude parser-recognized malformed quote tags from relation context."""
    quote = parsed["logical_quote"]
    for tag in ("<quote>", "</quote>"):
        quote = quote.replace(tag, "")
    quote = quote.strip(cpu.ASCII_EDGE)
    assert "<quote>" not in quote and "</quote>" not in quote
    return quote


@torch.inference_mode()
def relation_readout(model, tokenizer, plan, prompt, parsed):
    logical_quote = relation_context_quote(parsed)
    text = (prompt + logical_quote + "</quote>" +
            plan["relation_readout"]["suffix"])
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids and len(ids) <= MAX_CONTEXT
    device = model.model.embed_tokens.weight.device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tensor)
    final = model.model(
        input_ids=tensor, attention_mask=mask, use_cache=False,
        past_key_values=None, output_attentions=False,
        output_hidden_states=False, return_dict=True).last_hidden_state[0, -1]
    choice = torch.tensor(CHOICE_IDS, dtype=torch.long, device=device)
    logits = model.lm_head(final[None]).float()[0].index_select(0, choice)
    probability = logits.softmax(-1)
    entropy = -(probability * probability.log()).sum()
    top2 = logits.topk(2).values
    result = torch.cat((probability, entropy[None], (top2[0] - top2[1])[None]))
    output = result.detach().cpu().numpy().astype(np.float32)
    assert output.shape == (RELATION_WIDTH,) and np.isfinite(output).all()
    return output


def p2_geometry(tokenizer, prompt, prompt_char_regions, quote):
    full_text = prompt + quote + "</quote>"
    encoded = tokenizer(full_text, add_special_tokens=False,
                        return_offsets_mapping=True)
    quote_left, quote_right = len(prompt), len(prompt) + len(quote)
    regions = dict(prompt_char_regions)
    regions["quote"] = (quote_left, quote_right)
    positions = cpu.region_token_positions(
        full_text, encoded["offset_mapping"], regions)
    quote_positions = positions.pop("quote")
    assert quote_positions
    # Any token touching the close-tag bytes is excluded from content.
    close_right = quote_right + len("</quote>")
    filtered = []
    for position in quote_positions:
        left, right = map(int, encoded["offset_mapping"][position])
        assert max(left, quote_left) < min(right, quote_right)
        if right <= quote_right:
            filtered.append(position)
    assert filtered
    assert close_right == len(full_text)
    return full_text, list(map(int, encoded["input_ids"])), positions, filtered


def _prefix_arrays(prefix: str, raw: dict):
    return {
        f"{prefix}_ratio_token_mean": raw["ratio_token_mean"],
        f"{prefix}_ratio_token_min": raw["ratio_token_min"],
        f"{prefix}_ratio_token_max": raw["ratio_token_max"],
        f"{prefix}_passage_mass_mean": raw["passage_mass_mean"],
        f"{prefix}_claim_mass_mean": raw["claim_mass_mean"],
    }


def extract_p2(model, tokenizer, plan, row, pca, mechanical, candidates):
    prompt, char_regions = cpu.render_prompt_with_regions(
        plan["quote_prompt"]["utf8_template"], row)
    quote = mechanical["text"]
    passages = [row[f"passage_{index}"] for index in (1, 2, 3)]
    parsed = cpu.parse_generated_quote(quote + "</quote>", "close", passages)
    assert parsed["parse_valid"] and parsed["source_exact_substring"]
    surface = cpu.surface_features(
        parsed, row["claim_prompt_text"], passages, tokenizer, candidates)
    _full_text, full_ids, positions, quote_positions = p2_geometry(
        tokenizer, prompt, char_regions, quote)
    passage_positions = [positions[f"passage_{index}"] for index in (1, 2, 3)]
    final, hidden64, raw_attention, attention = no_cache_quote_forward(
        model, full_ids, quote_positions, passage_positions,
        positions["claim"], pca)
    target_ids = np.asarray(full_ids, dtype=np.int64)[quote_positions]
    statistics = statistics_from_hidden(model, final, quote_positions, target_ids)
    del final
    relation = relation_readout(model, tokenizer, plan, prompt, parsed)
    token_view = token_summary(
        statistics["selected_logprob"], statistics["vocab_entropy"],
        statistics["signed_margin"])
    hidden_view = hidden_summary(hidden64)
    combined = combine_features(surface, token_view, relation,
                                hidden_view, attention)
    arrays = {
        "p1_features": surface.astype(np.float32),
        "p2_surface21": surface.astype(np.float32),
        "p2_token_summary11": token_view,
        "p2_relation5": relation,
        "p2_hidden_summary256": hidden_view,
        "p2_attention_summary256": attention,
        "p2_features": combined,
        "p2_content_token_ids": target_ids.astype(np.int32),
        "p2_content_absolute_positions": np.asarray(
            quote_positions, dtype=np.int32),
        "p2_selected_logprob": statistics["selected_logprob"],
        "p2_vocab_entropy": statistics["vocab_entropy"],
        "p2_signed_margin": statistics["signed_margin"],
        "p2_hidden64": hidden64,
        **_prefix_arrays("p2", raw_attention),
    }
    metadata = {
        "mechanical_quote": quote,
        "mechanical_quote_sha256": mechanical["text_sha256"],
        "mechanical_passage_id": mechanical["passage_id"],
        "mechanical_sentence_id": mechanical["sentence_id"],
        "mechanical_bm25": mechanical["bm25"],
        "content_tokens": len(quote_positions),
        "input_tokens": len(full_ids),
        "parse": parsed,
    }
    return arrays, metadata


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and values.size > 0
    assert np.isfinite(values).all()
    return {
        "mean": float(values.mean(dtype=np.float64)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def cache_replay_qa(cached, replay, generated_ids, *, enforce_smoke_gate):
    """Describe both schedules; enforce V2 cross-path gates only for smoke."""
    generated_ids = np.asarray(generated_ids, dtype=np.int32)
    assert generated_ids.ndim == 1 and generated_ids.size > 0
    assert np.array_equal(replay["target_token_ids"], generated_ids), (
        "Replay teacher-forcing targets differ from cached generation IDs")
    for key in SMOKE_ABSOLUTE_TOLERANCES:
        assert cached[key].shape == generated_ids.shape
        assert replay[key].shape == generated_ids.shape
        assert np.isfinite(cached[key]).all(), ("cached", key)
        assert np.isfinite(replay[key]).all(), ("replay_QA", key)

    replay_argmax = np.asarray(replay["argmax_token_ids"], dtype=np.int32)
    assert replay_argmax.shape == generated_ids.shape
    mismatch_positions = np.flatnonzero(replay_argmax != generated_ids)
    mismatch_details = []
    high_confidence_conflicts = []
    for position_value in mismatch_positions:
        position = int(position_value)
        cached_margin = float(cached["signed_margin"][position])
        replay_margin = float(replay["signed_margin"][position])
        allowed_near_tie = (
            cached_margin <= NEAR_TIE_MARGIN and
            replay_margin >= -NEAR_TIE_MARGIN)
        detail = {
            "generated_position": position,
            "cached_generated_token_id": int(generated_ids[position]),
            "replay_argmax_token_id": int(replay_argmax[position]),
            "cached_generated_token_signed_margin": cached_margin,
            "replay_cached_token_signed_margin": replay_margin,
            "near_tie_gate_passed": bool(allowed_near_tie),
        }
        mismatch_details.append(detail)
        if not allowed_near_tie:
            high_confidence_conflicts.append(detail)

    drift = {}
    failed_drift_gates = []
    for key, tolerance in SMOKE_ABSOLUTE_TOLERANCES.items():
        cached64 = np.asarray(cached[key], dtype=np.float64)
        replay64 = np.asarray(replay[key], dtype=np.float64)
        absolute = np.abs(cached64 - replay64)
        maximum_position = int(np.argmax(absolute))
        distribution = _distribution(absolute)
        gate_passed = distribution["max"] <= tolerance
        drift[key] = {
            **distribution,
            "max_position": maximum_position,
            "cached_at_max": float(cached64[maximum_position]),
            "replay_at_max": float(replay64[maximum_position]),
            "rtol": 0.0,
            "smoke_atol": float(tolerance),
            "smoke_gate_enforced": bool(enforce_smoke_gate),
            "smoke_gate_passed": bool(gate_passed),
        }
        if not gate_passed:
            failed_drift_gates.append(key)

    summary = {
        "policy": ("fixed_smoke_hard_gate" if enforce_smoke_gate else
                   "full_extraction_record_only"),
        "generated_tokens": int(generated_ids.size),
        "teacher_forcing_target_ids_equal_cached": True,
        "argmax_mismatch_count": int(mismatch_positions.size),
        "argmax_mismatch_rate": float(
            mismatch_positions.size / generated_ids.size),
        "argmax_mismatch_positions": mismatch_positions.astype(int).tolist(),
        "argmax_mismatch_details": mismatch_details,
        "high_confidence_conflict_count": len(high_confidence_conflicts),
        "high_confidence_conflicts": high_confidence_conflicts,
        "near_tie_rule": {
            "cached_generated_token_signed_margin_lte": NEAR_TIE_MARGIN,
            "replay_cached_token_signed_margin_gte": -NEAR_TIE_MARGIN,
        },
        "drift_absolute_float64": drift,
        "failed_drift_gates": failed_drift_gates,
        "all_values_finite": True,
        "used_as_model_feature": False,
        "used_for_filtering_or_retry": False,
    }
    if enforce_smoke_gate:
        if high_confidence_conflicts:
            raise AssertionError((
                "V2 smoke high-confidence cache/replay argmax conflict",
                high_confidence_conflicts))
        if failed_drift_gates:
            raise AssertionError((
                "V2 smoke fixed absolute drift gate failed",
                failed_drift_gates, drift))
    return summary


def extract_p3(model, tokenizer, plan, row, pca, candidates, *, qa_mode):
    assert qa_mode in ("smoke", "formal_record_only")
    prompt, prompt_ids, positions = prompt_ids_and_regions(
        tokenizer, plan, row)
    cached, generation = cached_greedy_generate(
        model, tokenizer, prompt_ids, plan)
    decoded, offsets = generated_token_offsets(
        tokenizer, cached["generated_token_ids"], generation["terminal_eos_id"])
    assert decoded == generation["decoded_generated_text"]
    passages = [row[f"passage_{index}"] for index in (1, 2, 3)]
    parsed = cpu.parse_generated_quote(
        decoded, generation["stop_reason"], passages)
    content_indices = generated_content_indices(parsed, offsets)
    query_positions = [len(prompt_ids) + index for index in content_indices]
    full_ids = prompt_ids + cached["generated_token_ids"].astype(int).tolist()
    assert len(full_ids) <= MAX_CONTEXT
    passage_positions = [positions[f"passage_{index}"] for index in (1, 2, 3)]
    final, hidden64, raw_attention, attention = no_cache_quote_forward(
        model, full_ids, query_positions, passage_positions,
        positions["claim"], pca)
    generated_positions = np.arange(
        len(prompt_ids), len(full_ids), dtype=np.int64)
    replay = statistics_from_hidden(
        model, final, generated_positions,
        cached["generated_token_ids"].astype(np.int64))
    del final
    qa_summary = cache_replay_qa(
        cached, replay, cached["generated_token_ids"],
        enforce_smoke_gate=(qa_mode == "smoke"))
    selected_logprob = cached["selected_logprob"][content_indices]
    selected_entropy = cached["vocab_entropy"][content_indices]
    selected_margin = cached["signed_margin"][content_indices]
    surface = cpu.surface_features(
        parsed, row["claim_prompt_text"], passages, tokenizer, candidates)
    relation = relation_readout(model, tokenizer, plan, prompt, parsed)
    token_view = token_summary(
        selected_logprob, selected_entropy, selected_margin)
    hidden_view = hidden_summary(hidden64)
    combined = combine_features(surface, token_view, relation,
                                hidden_view, attention)
    content_ids = cached["generated_token_ids"][content_indices]
    arrays = {
        "p3_surface21": surface.astype(np.float32),
        "p3_token_summary11": token_view,
        "p3_relation5": relation,
        "p3_hidden_summary256": hidden_view,
        "p3_attention_summary256": attention,
        "p3_features": combined,
        "p3_generated_token_ids": cached["generated_token_ids"],
        "p3_generated_selected_logprob": cached["selected_logprob"],
        "p3_generated_vocab_entropy": cached["vocab_entropy"],
        "p3_generated_signed_margin": cached["signed_margin"],
        "p3_replay_selected_logprob": replay["selected_logprob"],
        "p3_replay_vocab_entropy": replay["vocab_entropy"],
        "p3_replay_signed_margin": replay["signed_margin"],
        "p3_replay_argmax_token_ids": replay["argmax_token_ids"],
        "p3_generated_char_offsets": offsets,
        "p3_content_generated_indices": np.asarray(
            content_indices, dtype=np.int32),
        "p3_content_token_ids": content_ids.astype(np.int32),
        "p3_content_absolute_positions": np.asarray(
            query_positions, dtype=np.int32),
        "p3_selected_logprob": selected_logprob.astype(np.float32),
        "p3_vocab_entropy": selected_entropy.astype(np.float32),
        "p3_signed_margin": selected_margin.astype(np.float32),
        "p3_hidden64": hidden64,
        **_prefix_arrays("p3", raw_attention),
    }
    metadata = {
        "decoded_generated_text": decoded,
        "generated_text_sha256": digest(decoded),
        "stop_reason": generation["stop_reason"],
        "terminal_eos_id": generation["terminal_eos_id"],
        "generated_tokens": len(cached["generated_token_ids"]),
        "content_tokens": len(content_indices),
        "prompt_tokens": len(prompt_ids),
        "input_tokens": len(full_ids),
        "prompt_token_ids_sha256": digest(prompt_ids),
        "generated_token_ids_sha256": digest(
            cached["generated_token_ids"]),
        "content_generated_indices_sha256": digest(
            np.asarray(content_indices, dtype=np.int32)),
        "content_absolute_positions_sha256": digest(
            np.asarray(query_positions, dtype=np.int32)),
        "generated_char_offsets_sha256": digest(offsets),
        "parse": parsed,
        "cache_replay_QA": qa_summary,
        "cache_replay_QA_policy": qa_summary["policy"],
    }
    return arrays, metadata, cached


def extract_claim(model, tokenizer, plan, row, pca, *, qa_mode):
    assert qa_mode in ("smoke", "formal_record_only")
    mechanical, candidates = sentence_candidates(row)
    p3_arrays, p3_meta, cached = extract_p3(
        model, tokenizer, plan, row, pca, candidates, qa_mode=qa_mode)
    # The cached generation state is destroyed inside cached_greedy_generate;
    # P3 replay above is already the sole hidden/attention source.
    p2_arrays, p2_meta = extract_p2(
        model, tokenizer, plan, row, pca, mechanical, candidates)
    arrays = {**p2_arrays, **p3_arrays}
    validate_arrays(arrays)
    metadata = {
        "P2": p2_meta, "P3": p3_meta,
        "feature_order": [
            "surface21", "token_summary11", "relation5",
            "hidden_summary256", "attention_summary256",
        ],
        "P3_cached_statistics_authoritative": True,
        "P3_replay_statistics_descriptive_QA_only": True,
        "P3_hidden_attention_source": (
            "post_generation_full_sequence_no_cache_replay"),
        "cache_replay_QA_policy": p3_meta["cache_replay_QA_policy"],
        "P2_trajectory": "whole_string_teacher_forced_no_cache",
        "generated_cache_discarded_before_replay": True,
    }
    return arrays, metadata


def validate_arrays(arrays: dict):
    assert set(arrays) == ARRAY_KEYS, (sorted(set(arrays) - ARRAY_KEYS),
                                      sorted(ARRAY_KEYS - set(arrays)))
    for key in FLOAT_ARRAY_KEYS:
        assert np.issubdtype(arrays[key].dtype, np.floating), (key, arrays[key].dtype)
        assert np.isfinite(arrays[key]).all(), key
    for key in INT_ARRAY_KEYS:
        assert np.issubdtype(arrays[key].dtype, np.integer), (key, arrays[key].dtype)

    for prefix in ("p2", "p3"):
        assert arrays[f"{prefix}_surface21"].shape == (SURFACE_WIDTH,)
        assert arrays[f"{prefix}_token_summary11"].shape == (TOKEN_WIDTH,)
        assert arrays[f"{prefix}_relation5"].shape == (RELATION_WIDTH,)
        assert arrays[f"{prefix}_hidden_summary256"].shape == (
            HIDDEN_SUMMARY_WIDTH,)
        assert arrays[f"{prefix}_attention_summary256"].shape == (
            ATTENTION_WIDTH,)
        assert arrays[f"{prefix}_features"].shape == (WHITEBOX_WIDTH,)
        reconstructed = combine_features(
            arrays[f"{prefix}_surface21"],
            arrays[f"{prefix}_token_summary11"],
            arrays[f"{prefix}_relation5"],
            arrays[f"{prefix}_hidden_summary256"],
            arrays[f"{prefix}_attention_summary256"])
        assert np.array_equal(reconstructed, arrays[f"{prefix}_features"])
        relation = arrays[f"{prefix}_relation5"]
        assert np.isclose(relation[:3].sum(), 1.0, rtol=1e-5, atol=1e-6)
        assert np.all((relation[:3] >= 0) & (relation[:3] <= 1))
        assert arrays[f"{prefix}_ratio_token_mean"].shape == (LAYERS, HEADS)
        assert arrays[f"{prefix}_ratio_token_min"].shape == (LAYERS, HEADS)
        assert arrays[f"{prefix}_ratio_token_max"].shape == (LAYERS, HEADS)
        assert arrays[f"{prefix}_passage_mass_mean"].shape == (3, LAYERS)
        assert arrays[f"{prefix}_claim_mass_mean"].shape == (LAYERS,)
        for name in ("ratio_token_mean", "ratio_token_min", "ratio_token_max",
                     "passage_mass_mean", "claim_mass_mean", "hidden64"):
            assert arrays[f"{prefix}_{name}"].dtype == np.float16

    assert arrays["p1_features"].shape == (SURFACE_WIDTH,)
    assert arrays["p1_features"].dtype == np.float32
    assert np.array_equal(arrays["p1_features"], arrays["p2_surface21"])
    t2 = len(arrays["p2_content_token_ids"])
    assert t2 > 0
    assert arrays["p2_content_absolute_positions"].shape == (t2,)
    assert arrays["p2_hidden64"].shape == (t2, PCA_WIDTH)
    for name in ("selected_logprob", "vocab_entropy", "signed_margin"):
        assert arrays[f"p2_{name}"].shape == (t2,)

    generated = arrays["p3_generated_token_ids"]
    g = len(generated)
    assert 0 < g <= MAX_NEW_TOKENS
    assert arrays["p3_generated_char_offsets"].shape == (g, 2)
    assert arrays["p3_replay_argmax_token_ids"].shape == (g,)
    for name in ("generated_selected_logprob", "generated_vocab_entropy",
                 "generated_signed_margin", "replay_selected_logprob",
                 "replay_vocab_entropy", "replay_signed_margin"):
        assert arrays[f"p3_{name}"].shape == (g,)
    indices = arrays["p3_content_generated_indices"]
    t3 = len(indices)
    assert np.all(indices[:-1] < indices[1:]) if t3 > 1 else True
    assert np.all((indices >= 0) & (indices < g))
    assert arrays["p3_content_token_ids"].shape == (t3,)
    assert np.array_equal(arrays["p3_content_token_ids"], generated[indices])
    assert arrays["p3_content_absolute_positions"].shape == (t3,)
    assert arrays["p3_hidden64"].shape == (t3, PCA_WIDTH)
    for name in ("selected_logprob", "vocab_entropy", "signed_margin"):
        assert arrays[f"p3_{name}"].shape == (t3,)
    assert np.array_equal(
        arrays["p3_selected_logprob"],
        arrays["p3_generated_selected_logprob"][indices])
    assert np.array_equal(
        arrays["p3_vocab_entropy"],
        arrays["p3_generated_vocab_entropy"][indices])
    assert np.array_equal(
        arrays["p3_signed_margin"],
        arrays["p3_generated_signed_margin"][indices])
    assert np.all(arrays["p3_generated_selected_logprob"] <= 1e-6)
    assert np.all(arrays["p3_generated_vocab_entropy"] >= -1e-6)
    assert np.all(arrays["p3_generated_signed_margin"] >= -1e-6)
    return True


def load_pca():
    plan = load_plan_and_check()
    expected = plan["input_locks"]["hidden_projection"]["sha256"]
    assert sha(PCA_PATH) == expected
    loaded = pickle.loads(PCA_PATH.read_bytes())
    assert loaded["mean"].shape == (HIDDEN,)
    assert loaded["components"].shape == (PCA_WIDTH, HIDDEN)
    assert loaded["whiten"] is False
    pca = {
        "mean": np.asarray(loaded["mean"], dtype=np.float64).copy(),
        "components": np.asarray(loaded["components"], dtype=np.float64).copy(),
    }
    del loaded
    assert np.isfinite(pca["mean"]).all()
    assert np.isfinite(pca["components"]).all()
    return pca


@lru_cache(maxsize=1)
def runtime_signature():
    plan = load_plan_and_check()
    manifest = read_json(MANIFEST_PATH)
    model_manifest = read_json(MODEL_MANIFEST)
    return {
        "version": VERSION,
        "runner_sha256": sha(Path(__file__)),
        "CPU_helper_sha256": sha(Path(cpu.__file__)),
        "feature_qa_sha256": sha(Path(feature_qa.__file__)),
        "BM25_source_sha256": sha(Path(bm25_source.__file__)),
        "audited_WDDM_gate_sha256": sha(Path(wddm_gate.__file__)),
        "protocol_sha256": sha(PROTOCOL_PATH),
        "plan_sha256": sha(PLAN_PATH),
        "numerical_protocol_sha256": sha(NUMERICAL_PROTOCOL_PATH),
        "sanitized_manifest_sha256": sha(MANIFEST_PATH),
        "sole_input_sha256": sha(INPUT_PATH),
        "sole_input_manifest_binding": manifest["files_sha256"][INPUT_PATH.name],
        "PCA_sha256": sha(PCA_PATH),
        "model_download_manifest_sha256": sha(MODEL_MANIFEST),
        "model_asset_declared_sha256": {
            item["filename"]: item["actual_sha256"]
            for item in model_manifest["files"]
        },
        "repo": REPO, "revision": REVISION,
        "model_load": MODEL_LOAD,
        "seed": SEED, "threads": THREADS,
        "query_chunk": QUERY_CHUNK, "logit_batch": LOGIT_BATCH,
        "token_offset_mapping": {
            "algorithm": "source-tracked Replace-ByteFallback-Fuse-Strip",
            "decoder_config_sha256": digest(EXPECTED_TOKENIZER_DECODER),
            "overlapping_byte_intervals": True,
            "zero_width_stripped_intervals": True,
        },
        "array_keys": sorted(ARRAY_KEYS),
        "feature_widths": {
            "P1": SURFACE_WIDTH, "P2": WHITEBOX_WIDTH,
            "P3": WHITEBOX_WIDTH,
            "components": [SURFACE_WIDTH, TOKEN_WIDTH, RELATION_WIDTH,
                           HIDDEN_SUMMARY_WIDTH, ATTENTION_WIDTH],
        },
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers",
                         "bitsandbytes", "accelerate", "numpy",
                         "scikit-learn")
        },
        "python": platform.python_version(),
        "torch_cuda_runtime": torch.version.cuda,
        "data_scope": "label_free_inputs.jsonl only; no gold/cal/test",
        "whitebox_execution": (
            "cached generation IDs/statistics; full no-cache replay for P3 "
            "hidden/attention; whole-string no-cache P2 teacher force"),
        "cache_replay_numerical_QA": {
            "cached_generation_IDs_and_statistics_authoritative": True,
            "replay_statistics_used_as_features": False,
            "smoke_only_hard_gate": True,
            "smoke_rtol": 0.0,
            "smoke_absolute_tolerances": SMOKE_ABSOLUTE_TOLERANCES,
            "near_tie_margin": NEAR_TIE_MARGIN,
            "formal_extraction": "record_only_no_filter_no_retry",
            "same_path_repeat_rtol": 0.0,
            "same_path_repeat_atol": REPEAT_ATOL,
        },
    }


@lru_cache(maxsize=1)
def runtime_signature_sha256():
    return digest(runtime_signature())


def record_signature(index: int, row: dict):
    return digest({
        "runtime_signature_sha256": runtime_signature_sha256(),
        "record_index": int(index), "row": row,
    })


def record_paths(index: int, directory=RECORD_DIR):
    stem = f"{index:07d}"
    directory = Path(directory)
    return (directory / f"{stem}.npz", directory / f"{stem}.json",
            directory / f"{stem}.commit.json")


def read_record(index: int, row: dict, *, load_arrays=True,
                directory=RECORD_DIR):
    npz_path, meta_path, commit_path = record_paths(index, directory)
    if not commit_path.exists():
        assert not npz_path.exists() and not meta_path.exists(), (
            "Uncommitted final artifact; inspect before resume", index)
        return None
    assert npz_path.exists() and meta_path.exists()
    commit = read_json(commit_path)
    metadata = read_json(meta_path)
    assert commit["record_index"] == index
    assert commit["response_id"] == row["response_id"]
    assert commit["microclaim_id"] == row["microclaim_id"]
    assert commit["row_sha256"] == digest(row)
    assert commit["record_signature"] == record_signature(index, row)
    assert commit["npz_sha256"] == sha(npz_path)
    assert commit["metadata_sha256"] == sha(meta_path)
    assert metadata["complete"] is True
    assert metadata["record_signature"] == commit["record_signature"]
    if not load_arrays:
        return commit
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_arrays(arrays)
    return arrays, metadata, commit


def save_record(index: int, row: dict, arrays: dict, extraction_meta: dict,
                *, directory=RECORD_DIR):
    validate_arrays(arrays)
    assert read_record(index, row, directory=directory) is None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path, commit_path = record_paths(index, directory)
    atomic_npz(npz_path, arrays)
    metadata = {
        "complete": True, "record_index": index,
        "response_id": row["response_id"],
        "source_id": row["source_id"], "group_id": row["group_id"],
        "microclaim_id": row["microclaim_id"],
        "microclaim_index": row["microclaim_index"],
        "row_sha256": digest(row),
        "record_signature": record_signature(index, row),
        "npz_sha256": sha(npz_path),
        "extraction": extraction_meta,
        "gold_fields_available": False,
        "calibration_or_test_opened": False,
        "scoring_run": False,
    }
    atomic_json(meta_path, metadata)
    commit = {
        key: metadata[key] for key in (
            "record_index", "response_id", "microclaim_id", "row_sha256",
            "record_signature", "npz_sha256")
    }
    commit["metadata_sha256"] = sha(meta_path)
    atomic_json(commit_path, commit)
    assert read_record(index, row, directory=directory) is not None


def verify_model_assets():
    manifest = read_json(MODEL_MANIFEST)
    assert manifest["status"] == "complete"
    assert manifest["repo_id"] == REPO and manifest["revision"] == REVISION
    for item in manifest["files"]:
        path = MODEL / item["filename"]
        assert item["status"] == "verified" and item["source_hash_match"]
        assert path.stat().st_size == item["actual_bytes"]
        assert sha(path) == item["actual_sha256"], item["filename"]


def require_runner_review(stage: str):
    assert stage in ("gpu-smoke", "extract")
    assert RUNNER_REVIEW_PATH.exists(), (
        "Independent GPU runner review has not been provided")
    review = read_json(RUNNER_REVIEW_PATH)
    assert review["status"] == "PASS"
    assert review["runner_sha256"] == sha(Path(__file__))
    assert review["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert review["plan_sha256"] == sha(PLAN_PATH)
    assert review["numerical_protocol_sha256"] == sha(NUMERICAL_PROTOCOL_PATH)
    assert review["CPU_selfcheck_sha256"] == sha(CPU_SELFCHECK_PATH)
    assert review["gpu_smoke_allowed"] is True
    if stage == "extract":
        assert review["full_extract_allowed"] is True
        assert SMOKE_REVIEW_PATH.exists(), (
            "Independent GPU smoke review required before full extraction")
        smoke_review = read_json(SMOKE_REVIEW_PATH)
        assert smoke_review["status"] == "PASS"
        assert smoke_review["runner_sha256"] == sha(Path(__file__))
        assert smoke_review["numerical_protocol_sha256"] == sha(
            NUMERICAL_PROTOCOL_PATH)
        assert smoke_review["GPU_smoke_sha256"] == sha(SMOKE_PATH)
        assert smoke_review["full_extract_allowed"] is True
    return review


@contextmanager
def exclusive_gpu(stage: str):
    """Delegate exactly to the already audited Windows-WDDM-safe gate."""
    assert GLOBAL_GPU_LOCK == wddm_gate.GLOBAL_GPU_LOCK
    assert MIN_FREE_GPU_BYTES == wddm_gate.MIN_FREE_GPU_BYTES
    with wddm_gate.exclusive_gpu(f"{VERSION}:{stage}") as token:
        yield token


def clean_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def load_nf4():
    assert torch.cuda.is_available()
    verify_model_assets()
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
        llm_int8_skip_modules=["lm_head"],
    )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = feature_qa.architecture(model)
    assert (spec["model_type"], spec["layers"], spec["heads"],
            spec["hidden_size"]) == ("llama", LAYERS, HEADS, HIDDEN)
    assert getattr(model, "is_loaded_in_4bit", False)
    assert model.model.embed_tokens.weight.device.type == "cuda"
    assert model.config._attn_implementation == "sdpa"
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    first = model.model.layers[0].self_attn.q_proj
    assert first.weight.quant_state.quant_type == "nf4"
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    return model, {
        "load_seconds": time.perf_counter() - started,
        "gpu_name": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "total_gpu_bytes": int(properties.total_memory),
        "allocated_after_load_bytes": int(torch.cuda.memory_allocated()),
        "reserved_after_load_bytes": int(torch.cuda.memory_reserved()),
        "attention_backend": model.config._attn_implementation,
        "architecture": spec,
        "quant_type": first.weight.quant_state.quant_type,
        "embedding_dtype": str(model.model.embed_tokens.weight.dtype),
        "lm_head_dtype": str(model.lm_head.weight.dtype),
    }


def dense_attention_oracle(attentions, query_positions, passage_positions,
                           claim_positions):
    tokens = len(query_positions)
    layers = len(attentions)
    heads = attentions[0].shape[1]
    ratio = np.empty((tokens, layers, heads), dtype=np.float32)
    passage_mass = np.empty((tokens, layers, heads, 3), dtype=np.float32)
    claim_mass = np.empty((tokens, layers, heads), dtype=np.float32)
    source = [item for values in passage_positions for item in values]
    for layer, attention in enumerate(attentions):
        value = attention[0].float().detach().cpu().numpy()
        for local, query in enumerate(query_positions):
            row = value[:, query]
            source_mean = row[:, source].mean(axis=1)
            through = [position for position in query_positions
                       if position <= query]
            quote_mean = row[:, through].mean(axis=1)
            ratio[local, layer] = source_mean / np.maximum(
                source_mean + quote_mean, np.finfo(np.float32).tiny)
            for passage_index, positions in enumerate(passage_positions):
                passage_mass[local, layer, :, passage_index] = row[:, positions].sum(axis=1)
            claim_mass[local, layer] = row[:, claim_positions].sum(axis=1)
    return summarize_attention(ratio, passage_mass, claim_mass)


def synthetic_arrays():
    relation = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
    surface = np.zeros(SURFACE_WIDTH, dtype=np.float32)
    token = np.zeros(TOKEN_WIDTH, dtype=np.float32)
    hidden_summary_value = np.zeros(HIDDEN_SUMMARY_WIDTH, dtype=np.float32)
    attention = np.zeros(ATTENTION_WIDTH, dtype=np.float32)
    p2 = combine_features(surface, token, relation,
                          hidden_summary_value, attention)
    p3 = p2.copy()
    raw, _view = zero_attention()
    arrays = {
        "p1_features": surface.copy(),
        "p2_surface21": surface.copy(), "p2_token_summary11": token.copy(),
        "p2_relation5": relation.copy(),
        "p2_hidden_summary256": hidden_summary_value.copy(),
        "p2_attention_summary256": attention.copy(), "p2_features": p2,
        "p2_content_token_ids": np.asarray([7], dtype=np.int32),
        "p2_content_absolute_positions": np.asarray([12], dtype=np.int32),
        "p2_selected_logprob": np.asarray([-1], dtype=np.float32),
        "p2_vocab_entropy": np.asarray([1], dtype=np.float32),
        "p2_signed_margin": np.asarray([-1], dtype=np.float32),
        "p2_hidden64": np.zeros((1, PCA_WIDTH), dtype=np.float16),
        **_prefix_arrays("p2", {key: value.copy() for key, value in raw.items()}),
        "p3_surface21": surface.copy(), "p3_token_summary11": token.copy(),
        "p3_relation5": relation.copy(),
        "p3_hidden_summary256": hidden_summary_value.copy(),
        "p3_attention_summary256": attention.copy(), "p3_features": p3,
        "p3_generated_token_ids": np.asarray([2], dtype=np.int32),
        "p3_generated_selected_logprob": np.asarray([0], dtype=np.float32),
        "p3_generated_vocab_entropy": np.asarray([0], dtype=np.float32),
        "p3_generated_signed_margin": np.asarray([0], dtype=np.float32),
        "p3_replay_selected_logprob": np.asarray([0], dtype=np.float32),
        "p3_replay_vocab_entropy": np.asarray([0], dtype=np.float32),
        "p3_replay_signed_margin": np.asarray([0], dtype=np.float32),
        "p3_replay_argmax_token_ids": np.asarray([2], dtype=np.int32),
        "p3_generated_char_offsets": np.asarray([[-1, -1]], dtype=np.int32),
        "p3_content_generated_indices": np.asarray([], dtype=np.int32),
        "p3_content_token_ids": np.asarray([], dtype=np.int32),
        "p3_content_absolute_positions": np.asarray([], dtype=np.int32),
        "p3_selected_logprob": np.asarray([], dtype=np.float32),
        "p3_vocab_entropy": np.asarray([], dtype=np.float32),
        "p3_signed_margin": np.asarray([], dtype=np.float32),
        "p3_hidden64": np.zeros((0, PCA_WIDTH), dtype=np.float16),
        **_prefix_arrays("p3", {key: value.copy() for key, value in raw.items()}),
    }
    validate_arrays(arrays)
    return arrays


def cpu_selfcheck():
    assert_cpu_only()
    torch.set_num_threads(THREADS)
    rows, manifest = load_inputs()
    plan = load_plan_and_check()
    tokenizer = tokenizer_load()
    pca = load_pca()
    chosen, lengths = fixed_smoke_indices(rows, tokenizer, plan)

    fingerprints = []
    for row in rows:
        mechanical, _candidates = sentence_candidates(row)
        quote_ids = tokenizer.encode(
            mechanical["text"] + "</quote>", add_special_tokens=False)
        fingerprints.append(
            f"{row['microclaim_id']}\t{mechanical['passage_id']}\t"
            f"{mechanical['sentence_id']}\t{mechanical['text_sha256']}\t"
            f"{mechanical['bm25']:.17g}\t{len(quote_ids)}\n")
    expected_mechanical = read_json(PREPARATION_PATH)["digests"][
        "mechanical_quote_rows"]
    assert digest("".join(fingerprints)) == expected_mechanical

    first = rows[chosen[0]]
    mechanical, candidates = sentence_candidates(first)
    prompt, char_regions = cpu.render_prompt_with_regions(
        plan["quote_prompt"]["utf8_template"], first)
    _text, p2_ids, p2_regions, p2_quote = p2_geometry(
        tokenizer, prompt, char_regions, mechanical["text"])
    assert p2_quote and all(p2_regions.values())
    parsed = cpu.parse_generated_quote(
        mechanical["text"] + "</quote>", "close",
        [first[f"passage_{index}"] for index in (1, 2, 3)])
    surface = cpu.surface_features(
        parsed, first["claim_prompt_text"],
        [first[f"passage_{index}"] for index in (1, 2, 3)],
        tokenizer, candidates)
    assert surface.shape == (SURFACE_WIDTH,)

    generated_ids = tokenizer.encode(
        "Alpha has 12 units.</quote>", add_special_tokens=False)
    decoded, offsets = generated_token_offsets(tokenizer, generated_ids, None)
    generated_parse = cpu.parse_generated_quote(
        decoded, "close", ["Alpha has 12 units.", "Beta.", "Gamma."])
    content_indices = generated_content_indices(generated_parse, offsets)
    assert content_indices and max(content_indices) < len(generated_ids)
    assert not any(offsets[index, 1] > len(generated_parse["raw_inner"])
                   for index in content_indices)
    unicode_ids = tokenizer.encode(
        "Café温度为12°C。</quote>", add_special_tokens=False)
    unicode_decoded, unicode_offsets = generated_token_offsets(
        tokenizer, unicode_ids, None)
    unicode_parse = cpu.parse_generated_quote(
        unicode_decoded, "close", ["Café温度为12°C。", "Beta.", "Gamma."])
    unicode_content = generated_content_indices(unicode_parse, unicode_offsets)
    assert unicode_content and unicode_parse["source_exact_substring"]

    # Generated SentencePiece IDs need not be the canonical re-encoding of
    # their decoded text (the first token can omit the dummy leading marker).
    noncanonical_ids = [8003, 292, 338]
    noncanonical_text, noncanonical_offsets = generated_token_offsets(
        tokenizer, noncanonical_ids, None)
    assert noncanonical_text == "Positioning is"
    assert tokenizer.encode(
        noncanonical_text, add_special_tokens=False) != noncanonical_ids
    assert noncanonical_offsets.tolist() == [[0, 8], [8, 11], [11, 14]]
    eos_text, eos_offsets = generated_token_offsets(
        tokenizer, noncanonical_ids + [tokenizer.eos_token_id],
        tokenizer.eos_token_id)
    assert eos_text == noncanonical_text
    assert eos_offsets[-1].tolist() == [-1, -1]

    byte_ids = tokenizer.encode("温", add_special_tokens=False)
    byte_text, byte_offsets = generated_token_offsets(
        tokenizer, byte_ids, None)
    byte_tokens = tokenizer.convert_ids_to_tokens(byte_ids)
    byte_indices = [index for index, token in enumerate(byte_tokens)
                    if BYTE_FALLBACK_TOKEN.fullmatch(token or "")]
    assert byte_text == "温" and len(byte_indices) == 3
    assert all(byte_offsets[index].tolist() == [0, 1]
               for index in byte_indices)

    leading_ids = [tokenizer.convert_tokens_to_ids("▁"),
                   tokenizer.convert_tokens_to_ids("hello")]
    leading_text, leading_offsets = generated_token_offsets(
        tokenizer, leading_ids, None)
    assert leading_text == "hello"
    assert leading_offsets.tolist() == [[0, 0], [0, 5]]
    invalid_byte_ids = [tokenizer.convert_tokens_to_ids(token)
                        for token in ("<0x61>", "<0xFF>", "<0x62>")]
    invalid_text, invalid_offsets = generated_token_offsets(
        tokenizer, invalid_byte_ids, None)
    assert invalid_text == "���"
    assert invalid_offsets.tolist() == [[0, 1], [1, 2], [2, 3]]

    nested_parse = cpu.parse_generated_quote(
        "<quote>Alpha has 12 units.</quote>", "close",
        ["Alpha has 12 units.", "Beta.", "Gamma."])
    assert not nested_parse["parse_valid"]
    assert "nested_or_second_quote_tag" in nested_parse["failure_reasons"]
    assert relation_context_quote(nested_parse) == "Alpha has 12 units."
    outside_parse = cpu.parse_generated_quote(
        "Alpha has 12 units.</quote>outside", "close",
        ["Alpha has 12 units.", "Beta.", "Gamma."])
    assert not outside_parse["parse_valid"]
    assert relation_context_quote(outside_parse) == "Alpha has 12 units."

    tied = torch.tensor([[3.0, 3.0, 1.0]])
    lp, ent, margin, argmax = _logit_statistics(tied, torch.tensor([0]))
    assert int(argmax[0]) == 0 and float(margin[0]) == 0
    assert np.isfinite(lp).all() and np.isfinite(ent).all()

    # Official eager attention is the independent oracle for the Q/K hook.
    from transformers import LlamaConfig, LlamaForCausalLM
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128,
            attention_dropout=0.0)
        config._attn_implementation = "eager"
        tiny = LlamaForCausalLM(config).cpu().eval()
        for parameter in tiny.parameters():
            parameter.requires_grad_(False)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]])
    passage_positions = [[1], [2], [3]]
    claim_positions = [4, 5]
    query_positions = [7, 8, 9]
    with torch.inference_mode():
        dense = tiny.model(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            use_cache=False, output_attentions=True, return_dict=True)
        expected_raw, expected_view = dense_attention_oracle(
            dense.attentions, query_positions, passage_positions,
            claim_positions)
        with QuoteAttentionHooks(
                tiny, query_positions, passage_positions, claim_positions,
                ids.shape[1], query_chunk=2) as hooks:
            hooked = tiny.model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, output_attentions=False,
                return_dict=True).last_hidden_state
        actual_raw, actual_view = hooks.finish()
    assert torch.equal(hooked, dense.last_hidden_state)
    assert np.allclose(actual_view, expected_view, rtol=1e-5, atol=1e-6)
    for key in expected_raw:
        assert np.array_equal(actual_raw[key], expected_raw[key]), key

    summary = token_summary(
        np.asarray([-3, -2, -1], dtype=np.float32),
        np.asarray([2, 1, 3], dtype=np.float32),
        np.asarray([-1, 0, 1], dtype=np.float32))
    assert summary.shape == (TOKEN_WIDTH,) and summary[6] > 0
    projected = project_hidden(np.zeros((3, HIDDEN), dtype=np.float32), pca)
    assert hidden_summary(projected).shape == (HIDDEN_SUMMARY_WIDTH,)
    assert np.array_equal(
        token_summary([], [], []), np.zeros(TOKEN_WIDTH, dtype=np.float32))
    zero_raw, zero_view = zero_attention()
    assert zero_view.shape == (ATTENTION_WIDTH,)
    assert all(np.count_nonzero(value) == 0 for value in zero_raw.values())

    # V2 numerical policy: exact binary boundaries pass, while full extraction
    # records the same events without turning them into filters or retries.
    synthetic_cached = {
        "selected_logprob": np.asarray([-1.0], dtype=np.float32),
        "vocab_entropy": np.asarray([2.0], dtype=np.float32),
        "signed_margin": np.asarray([0.5], dtype=np.float32),
    }
    synthetic_replay_boundary = {
        "selected_logprob": np.asarray([-0.9375], dtype=np.float32),
        "vocab_entropy": np.asarray([2.0625], dtype=np.float32),
        "signed_margin": np.asarray([-0.0], dtype=np.float32),
        "argmax_token_ids": np.asarray([12], dtype=np.int32),
        "target_token_ids": np.asarray([11], dtype=np.int32),
    }
    boundary_qa = cache_replay_qa(
        synthetic_cached, synthetic_replay_boundary,
        np.asarray([11], dtype=np.int32), enforce_smoke_gate=True)
    assert boundary_qa["argmax_mismatch_count"] == 1
    assert boundary_qa["failed_drift_gates"] == []
    assert boundary_qa["high_confidence_conflict_count"] == 0
    synthetic_replay_failure = {
        "selected_logprob": np.asarray([-0.5], dtype=np.float32),
        "vocab_entropy": np.asarray([3.0], dtype=np.float32),
        "signed_margin": np.asarray([-0.75], dtype=np.float32),
        "argmax_token_ids": np.asarray([12], dtype=np.int32),
        "target_token_ids": np.asarray([11], dtype=np.int32),
    }
    record_only_qa = cache_replay_qa(
        synthetic_cached, synthetic_replay_failure,
        np.asarray([11], dtype=np.int32), enforce_smoke_gate=False)
    assert record_only_qa["failed_drift_gates"]
    assert record_only_qa["high_confidence_conflict_count"] == 1
    try:
        cache_replay_qa(
            synthetic_cached, synthetic_replay_failure,
            np.asarray([11], dtype=np.int32), enforce_smoke_gate=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("Smoke-only hard gates did not fail closed")

    arrays = synthetic_arrays()
    synthetic_record_qa = cache_replay_qa(
        {
            "selected_logprob": arrays["p3_generated_selected_logprob"],
            "vocab_entropy": arrays["p3_generated_vocab_entropy"],
            "signed_margin": arrays["p3_generated_signed_margin"],
        },
        {
            "selected_logprob": arrays["p3_replay_selected_logprob"],
            "vocab_entropy": arrays["p3_replay_vocab_entropy"],
            "signed_margin": arrays["p3_replay_signed_margin"],
            "argmax_token_ids": arrays["p3_replay_argmax_token_ids"],
            "target_token_ids": arrays["p3_generated_token_ids"],
        },
        arrays["p3_generated_token_ids"], enforce_smoke_gate=False)
    with tempfile.TemporaryDirectory(prefix="forced_quote_gpu_runner_cpu_") as directory:
        save_record(0, first, arrays, {
            "synthetic_CPU_only": True,
            "P3": {"cache_replay_QA": synthetic_record_qa},
        },
                    directory=directory)
        loaded = read_record(0, first, directory=directory)
        assert loaded is not None
        loaded_arrays, _metadata, _commit = loaded
        assert all(np.array_equal(arrays[key], loaded_arrays[key])
                   for key in arrays)
        rebuilt_records, rebuilt_qa = _formal_batch_audit(
            [first], directory=directory, expected_rows=1)
        assert len(rebuilt_records) == 1
        assert rebuilt_qa["claims"] == 1
        assert rebuilt_qa["generated_tokens"] == 1

    wddm_selfcheck = wddm_gate.gpu_gate_selfcheck()
    assert wddm_selfcheck["status"] == "passed"
    assert wddm_selfcheck["nvidia_smi_called"] is False

    signature = runtime_signature()
    result = {
        "status": "passed_waiting_independent_GPU_runner_review",
        "runner_sha256": sha(Path(__file__)),
        "runtime_signature": signature,
        "runtime_signature_sha256": digest(signature),
        "protocol_sha256": sha(PROTOCOL_PATH),
        "plan_sha256": sha(PLAN_PATH),
        "numerical_protocol_sha256": sha(NUMERICAL_PROTOCOL_PATH),
        "sanitized_manifest_sha256": sha(MANIFEST_PATH),
        "sole_input_sha256": sha(INPUT_PATH),
        "rows": len(rows), "answers": EXPECTED_ANSWERS,
        "fixed_smoke_indices": chosen,
        "fixed_smoke_microclaim_ids": [rows[index]["microclaim_id"]
                                        for index in chosen],
        "fixed_smoke_prompt_lengths": [lengths[index] for index in chosen],
        "mechanical_quote_digest_reproduced": True,
        "whole_string_P2_geometry_checked": True,
        "actual_ID_provenance_offsets_and_close_exclusion_checked": True,
        "noncanonical_SentencePiece_ID_provenance_offsets_checked": True,
        "Unicode_byte_fallback_shared_character_offsets_checked": True,
        "invalid_byte_fallback_and_leading_strip_offsets_checked": True,
        "malformed_tags_and_outside_text_excluded_from_relation_context": True,
        "audited_WDDM_gate": {
            "source_sha256": sha(Path(wddm_gate.__file__)),
            "synthetic_selfcheck": wddm_selfcheck,
        },
        "official_apply_rotary_repeat_kv_hook_vs_dense_oracle": {
            "tiny_layers": 2, "tiny_heads": 4,
            "max_abs": float(np.max(np.abs(actual_view - expected_view))),
            "raw_float16_exact": True,
            "model_hidden_unchanged_exact": True,
        },
        "feature_widths_checked": {"P1": 21, "P2": 549, "P3": 549},
        "V2_numerical_policy_checked": {
            "fixed_smoke_absolute_boundaries_inclusive": True,
            "near_tie_argmax_flip_allowed_at_boundary": True,
            "high_confidence_argmax_conflict_fails_smoke": True,
            "full_extraction_drift_is_record_only": True,
            "cached_statistics_are_classifier_inputs": True,
            "replay_statistics_are_QA_only": True,
            "T0_token_hidden_attention_zero_rule": True,
            "all_records_without_complete_marker_CPU_rebuild_checked": True,
        },
        "atomic_claim_commit_and_resume_read_checked": True,
        "required_runner_review_path": str(RUNNER_REVIEW_PATH.resolve()),
        "required_runner_review_fields": {
            "status": "PASS", "runner_sha256": sha(Path(__file__)),
            "protocol_sha256": sha(PROTOCOL_PATH),
            "plan_sha256": sha(PLAN_PATH),
            "numerical_protocol_sha256": sha(NUMERICAL_PROTOCOL_PATH),
            "CPU_selfcheck_sha256": "SHA256(this file after materialization)",
            "gpu_smoke_allowed": True,
            "full_extract_allowed": "reviewer decision",
        },
        "GPU_used": False, "CUDA_initialized": False,
        "pretrained_model_loaded": False,
        "gold_bearing_source_opened": False,
        "calibration_or_test_opened": False,
        "scoring_run": False,
        "sole_data_input": INPUT_PATH.name,
    }
    atomic_json(CPU_SELFCHECK_PATH, result)
    assert_cpu_only()
    print("FORCED_QUOTE_GPU_RUNNER_CPU_SELFCHECK_PASSED", len(rows), flush=True)


def check_cpu_ready():
    check = read_json(CPU_SELFCHECK_PATH)
    assert check["status"] == "passed_waiting_independent_GPU_runner_review"
    assert check["runner_sha256"] == sha(Path(__file__))
    assert check["runtime_signature_sha256"] == runtime_signature_sha256()
    assert check["sole_input_sha256"] == sha(INPUT_PATH)
    assert check["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert check["plan_sha256"] == sha(PLAN_PATH)
    assert check["numerical_protocol_sha256"] == sha(NUMERICAL_PROTOCOL_PATH)
    assert check["GPU_used"] is False and check["scoring_run"] is False
    return check


def _repeat_p3_check(first_arrays, first_meta, second_arrays, second_meta):
    """Frozen shortest/longest same-path reproducibility oracle."""
    exact_array_keys = [
        "p3_generated_token_ids", "p3_generated_char_offsets",
        "p3_content_generated_indices", "p3_content_token_ids",
        "p3_content_absolute_positions",
    ]
    for key in exact_array_keys:
        assert np.array_equal(first_arrays[key], second_arrays[key]), key
    exact_meta_keys = [
        "decoded_generated_text", "generated_text_sha256", "stop_reason",
        "terminal_eos_id", "generated_tokens", "content_tokens",
        "prompt_tokens", "input_tokens", "prompt_token_ids_sha256",
        "generated_token_ids_sha256", "content_generated_indices_sha256",
        "content_absolute_positions_sha256", "generated_char_offsets_sha256",
        "parse",
    ]
    for key in exact_meta_keys:
        assert first_meta[key] == second_meta[key], key

    cached_stat_keys = [
        "p3_generated_selected_logprob", "p3_generated_vocab_entropy",
        "p3_generated_signed_margin",
    ]
    replay_hidden_attention_keys = [
        "p3_hidden64", "p3_ratio_token_mean", "p3_ratio_token_min",
        "p3_ratio_token_max", "p3_passage_mass_mean",
        "p3_claim_mass_mean", "p3_attention_summary256",
    ]
    errors = {"cached_statistics": {}, "replay_hidden_attention": {}}
    for category, keys in (
            ("cached_statistics", cached_stat_keys),
            ("replay_hidden_attention", replay_hidden_attention_keys)):
        for key in keys:
            left = first_arrays[key]
            right = second_arrays[key]
            assert left.shape == right.shape, key
            assert np.allclose(left, right, rtol=0.0, atol=REPEAT_ATOL), key
            errors[category][key] = float(np.max(np.abs(
                left.astype(np.float64) - right.astype(np.float64)))) \
                if left.size else 0.0
    errors.update({
        "rtol": 0.0,
        "atol": REPEAT_ATOL,
        "cached_generation_IDs_exact": True,
        "prompt_ID_hash_positions_offsets_stop_and_parse_exact": True,
    })
    return errors


def gpu_smoke():
    check_cpu_ready()
    require_runner_review("gpu-smoke")
    assert not SMOKE_PATH.exists(), "GPU smoke already exists; no silent rerun"
    rows, _manifest = load_inputs()
    plan = load_plan_and_check()
    tokenizer = tokenizer_load()
    pca = load_pca()
    chosen, lengths = fixed_smoke_indices(rows, tokenizer, plan)
    repeated_indices = {chosen[0], chosen[-1]}
    model = None
    started = time.perf_counter()
    reports = []
    with exclusive_gpu("gpu-smoke") as lease:
        try:
            model, model_meta = load_nf4()
            for index in chosen:
                row = rows[index]
                torch.cuda.reset_peak_memory_stats()
                item_started = time.perf_counter()
                arrays, metadata = extract_claim(
                    model, tokenizer, plan, row, pca, qa_mode="smoke")
                repeat_errors = None
                if index in repeated_indices:
                    mechanical, candidates = sentence_candidates(row)
                    del mechanical
                    repeated, repeat_meta, _cached = extract_p3(
                        model, tokenizer, plan, row, pca, candidates,
                        qa_mode="smoke")
                    repeat_errors = _repeat_p3_check(
                        arrays, metadata["P3"], repeated, repeat_meta)
                torch.cuda.synchronize()
                report = {
                    "record_index": index,
                    "smoke_role": ("tolerance_anchor" if index == chosen[0]
                                   else "validation_smoke"),
                    "microclaim_id": row["microclaim_id"],
                    "prompt_tokens": lengths[index],
                    "P2_content_tokens": len(arrays["p2_content_token_ids"]),
                    "P3_generated_tokens": len(arrays["p3_generated_token_ids"]),
                    "P3_content_tokens": len(arrays["p3_content_token_ids"]),
                    "P3_stop_reason": metadata["P3"]["stop_reason"],
                    "P3_parse_valid": metadata["P3"]["parse"]["parse_valid"],
                    "P3_source_exact": metadata["P3"]["parse"]["source_exact_substring"],
                    "P1_sha256": digest(arrays["p1_features"]),
                    "P2_sha256": digest(arrays["p2_features"]),
                    "P3_sha256": digest(arrays["p3_features"]),
                    "cache_replay_QA": metadata["P3"]["cache_replay_QA"],
                    "repeated_P3": repeat_errors,
                    "all_finite_and_shapes_valid": True,
                    "seconds": time.perf_counter() - item_started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
                reports.append(report)
                print("FORCED_QUOTE_GPU_SMOKE_CLAIM",
                      len(reports), len(chosen), json.dumps(report), flush=True)
                del arrays
                clean_gpu()
            smoke = {
                "status": "passed_not_full_extraction",
                "protocol_sha256": sha(PROTOCOL_PATH),
                "plan_sha256": sha(PLAN_PATH),
                "numerical_protocol_sha256": sha(NUMERICAL_PROTOCOL_PATH),
                "runtime_signature": runtime_signature(),
                "runtime_signature_sha256": runtime_signature_sha256(),
                "CPU_selfcheck_sha256": sha(CPU_SELFCHECK_PATH),
                "runner_review_sha256": sha(RUNNER_REVIEW_PATH),
                "lease": lease, "model": model_meta,
                "fixed_eight_claims": reports,
                "shortest_and_longest_generation_ID_repeat_exact": True,
                "shortest_and_longest_cached_statistics_repeat_within_tolerance": True,
                "shortest_and_longest_replay_hidden_attention_repeat_within_tolerance": True,
                "all_eight_fixed_cache_vs_replay_smoke_gates_passed": True,
                "tolerance_anchor_index": chosen[0],
                "independent_validation_smoke_indices": chosen[1:],
                "smoke_absolute_tolerances": SMOKE_ABSOLUTE_TOLERANCES,
                "smoke_rtol": 0.0,
                "near_tie_margin": NEAR_TIE_MARGIN,
                "formal_claim_records_written": 0,
                "seconds": time.perf_counter() - started,
                "GPU_used": True,
                "gold_bearing_source_opened": False,
                "calibration_or_test_opened": False,
                "scoring_run": False,
            }
            atomic_json(SMOKE_PATH, smoke)
        finally:
            if model is not None:
                del model
            clean_gpu()
    print("FORCED_QUOTE_GPU_SMOKE_PASSED", len(reports), flush=True)


def _formal_batch_audit(rows, *, directory=RECORD_DIR,
                        expected_rows=EXPECTED_ROWS):
    """Recompute record-only QA and aggregate across every generated token."""
    records = []
    differences = {key: [] for key in SMOKE_ABSOLUTE_TOLERANCES}
    generated_tokens = 0
    mismatches = 0
    high_confidence_conflicts = 0
    for index, row in enumerate(rows):
        loaded = read_record(
            index, row, load_arrays=True, directory=directory)
        assert loaded is not None
        arrays, metadata, commit = loaded
        generated = arrays["p3_generated_token_ids"]
        cached = {
            "selected_logprob": arrays["p3_generated_selected_logprob"],
            "vocab_entropy": arrays["p3_generated_vocab_entropy"],
            "signed_margin": arrays["p3_generated_signed_margin"],
        }
        replay = {
            "selected_logprob": arrays["p3_replay_selected_logprob"],
            "vocab_entropy": arrays["p3_replay_vocab_entropy"],
            "signed_margin": arrays["p3_replay_signed_margin"],
            "argmax_token_ids": arrays["p3_replay_argmax_token_ids"],
            "target_token_ids": generated,
        }
        recomputed = cache_replay_qa(
            cached, replay, generated, enforce_smoke_gate=False)
        stored = metadata["extraction"]["P3"]["cache_replay_QA"]
        assert digest(recomputed) == digest(stored), index
        assert stored["policy"] == "full_extraction_record_only"
        generated_tokens += len(generated)
        mismatches += stored["argmax_mismatch_count"]
        high_confidence_conflicts += stored["high_confidence_conflict_count"]
        for key in differences:
            differences[key].append(np.abs(
                cached[key].astype(np.float64) -
                replay[key].astype(np.float64)))
        _npz_path, _meta_path, commit_path = record_paths(index, directory)
        records.append({
            "record_index": index,
            "response_id": row["response_id"],
            "microclaim_id": row["microclaim_id"],
            "commit_sha256": sha(commit_path),
            "npz_sha256": commit["npz_sha256"],
            "metadata_sha256": commit["metadata_sha256"],
        })
    assert len(records) == expected_rows and generated_tokens > 0
    drift = {}
    for key, chunks in differences.items():
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        assert values.size == generated_tokens and np.isfinite(values).all()
        drift[key] = _distribution(values)
    batch_qa = {
        "policy": "full_extraction_record_only_no_filter_no_retry",
        "claims": len(records),
        "generated_tokens": generated_tokens,
        "argmax_mismatch_count": mismatches,
        "argmax_mismatch_rate": float(mismatches / generated_tokens),
        "descriptive_high_confidence_conflict_count": high_confidence_conflicts,
        "drift_absolute_float64": drift,
        "smoke_tolerances_not_enforced_on_full_extraction": True,
        "used_as_model_feature": False,
        "used_for_filtering_or_retry": False,
    }
    return records, batch_qa


def extract():
    check_cpu_ready()
    require_runner_review("extract")
    smoke = read_json(SMOKE_PATH)
    assert smoke["status"] == "passed_not_full_extraction"
    assert smoke["runtime_signature_sha256"] == runtime_signature_sha256()
    assert smoke["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert smoke["plan_sha256"] == sha(PLAN_PATH)
    assert smoke["numerical_protocol_sha256"] == sha(NUMERICAL_PROTOCOL_PATH)
    assert not EXTRACTION_COMPLETE.exists(), "Extraction already complete"
    rows, _manifest = load_inputs()
    plan = load_plan_and_check()
    pending = []
    for index, row in enumerate(rows):
        record = read_record(index, row, load_arrays=True)
        if record is None:
            pending.append(index)
        else:
            _arrays, metadata, _commit = record
            assert metadata["extraction"]["cache_replay_QA_policy"] == (
                "full_extraction_record_only")
    started = time.perf_counter()
    model = None
    lease = None
    model_meta = {
        "source": "existing_atomic_records_and_frozen_runtime_signature",
        "repo": REPO,
        "revision": REVISION,
    }
    if pending:
        tokenizer = tokenizer_load()
        pca = load_pca()
        with exclusive_gpu("extract") as active_lease:
            lease = active_lease
            try:
                model, model_meta = load_nf4()
                for progress, index in enumerate(pending, 1):
                    row = rows[index]
                    torch.cuda.reset_peak_memory_stats()
                    item_started = time.perf_counter()
                    arrays, metadata = extract_claim(
                        model, tokenizer, plan, row, pca,
                        qa_mode="formal_record_only")
                    metadata["seconds"] = time.perf_counter() - item_started
                    metadata["peak_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated())
                    metadata["peak_reserved_bytes"] = int(
                        torch.cuda.max_memory_reserved())
                    save_record(index, row, arrays, metadata)
                    del arrays
                    if progress % 25 == 0 or progress == len(pending):
                        print("FORCED_QUOTE_GPU_EXTRACTED", progress,
                              len(pending), "formal_total", index + 1,
                              EXPECTED_ROWS, flush=True)
                    if progress % 25 == 0:
                        gc.collect()
            finally:
                if model is not None:
                    del model
                clean_gpu()

    records, full_batch_qa = _formal_batch_audit(rows)
    complete = {
        "status": "complete_label_free_GPU_features_frozen_not_scored",
        "protocol_sha256": sha(PROTOCOL_PATH),
        "plan_sha256": sha(PLAN_PATH),
        "numerical_protocol_sha256": sha(NUMERICAL_PROTOCOL_PATH),
        "runtime_signature": runtime_signature(),
        "runtime_signature_sha256": runtime_signature_sha256(),
        "GPU_smoke_sha256": sha(SMOKE_PATH),
        "GPU_smoke_review_sha256": sha(SMOKE_REVIEW_PATH),
        "records": len(records), "answers": EXPECTED_ANSWERS,
        "record_order_digest": digest(records),
        "record_commits": records,
        "full_batch_cache_replay_QA": full_batch_qa,
        "resumed_existing_records": EXPECTED_ROWS - len(pending),
        "new_records": len(pending),
        "lease": lease, "model": model_meta,
        "seconds": time.perf_counter() - started,
        "GPU_used_this_invocation": bool(pending),
        "GPU_features_present": True,
        "completion_rebuilt_without_GPU": not bool(pending),
        "GPU_used": True,
        "gold_bearing_source_opened": False,
        "calibration_or_test_opened": False,
        "scoring_run": False,
    }
    atomic_json(EXTRACTION_COMPLETE, complete)
    print("FORCED_QUOTE_GPU_EXTRACTION_COMPLETE", len(records), flush=True)


def describe():
    assert_cpu_only()
    rows, _manifest = load_inputs()
    plan = load_plan_and_check()
    tokenizer = tokenizer_load()
    chosen, lengths = fixed_smoke_indices(rows, tokenizer, plan)
    print(json.dumps({
        "version": VERSION,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha(Path(__file__)),
        "stages": ["cpu-selfcheck", "gpu-smoke", "extract"],
        "automatic_stage_chaining": False,
        "sole_data_input": str(INPUT_PATH.resolve()),
        "claims": len(rows), "answers": EXPECTED_ANSWERS,
        "fixed_smoke_indices": chosen,
        "fixed_smoke_prompt_lengths": [lengths[index] for index in chosen],
        "runner_review_present": RUNNER_REVIEW_PATH.exists(),
        "GPU_used": False,
    }, ensure_ascii=False, indent=2))
    assert_cpu_only()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "cpu-selfcheck", "gpu-smoke", "extract", "describe"))
    stage = parser.parse_args().stage
    try:
        {
            "cpu-selfcheck": cpu_selfcheck,
            "gpu-smoke": gpu_smoke,
            "extract": extract,
            "describe": describe,
        }[stage]()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": stage, "exception": repr(error),
            "traceback": traceback.format_exc(),
            "runner_sha256": sha(Path(__file__)),
            "CUDA_initialized": torch.cuda.is_initialized(),
            "scoring_run": False,
        }
        atomic_json(OUT / f"GPU_RUNNER_FAILURE_{stage}_{time.time_ns()}.json",
                    failure)
        raise


if __name__ == "__main__":
    main()
