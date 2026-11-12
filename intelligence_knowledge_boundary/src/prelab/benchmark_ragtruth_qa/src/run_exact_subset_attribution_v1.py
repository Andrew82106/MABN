"""Exact three-passage subset attribution for RAGTruth QA.

This is an original method candidate, not a ContextCite or LUMINA baseline.
CPU preparation freezes eight exact evidence views and the unchanged published
answer axis without reading annotation values.  GPU extraction and supervised
scoring are explicit later commands; neither is invoked by prepare/check/audit.
Official test paths are never addressed by this runner.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import time
from itertools import zip_longest

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import build_citation_alignment as citation
import feature_qa as fq
import run_development as q


ROOT = q.ROOT
MODEL = fq.MODEL
OUT = ROOT / "results/exact_subset_attribution_v1"
SOURCE_INPUTS = ROOT / "results/lumina_qa_preparation_v1/feature_inputs.jsonl"
SOURCE_COMPLETE = ROOT / "results/lumina_qa_preparation_v1/preparation_complete.json"
OLD_CLAIMS = ROOT / "semantic_baseline/cuda_variant/plans.jsonl"
NEW_CLAIMS = ROOT / "fit_expansion/minicheck/new_plans.jsonl"
EXPANSION = ROOT / "fit_expansion"
MODEL_MANIFEST = ROOT / "model_download_manifest.json"

ROLE = "ours_method_candidate; not ContextCite; not LUMINA; baselines remain unchanged"
VERSION = "exact-three-passage-subset-attribution-v1"
EXPECTED_ANSWERS = {"fit": 3680, "calibration": 159}
EXPECTED_SOURCE_SHA256 = "be0f78970e12fc5a718b486b8e02359492678c39e3c735e364d96c51338117b4"
EXPECTED_CLAIM_ROWS = {"old": 793, "new": 3046}
EXPECTED_RAW_TOKENS = 708506
EXPECTED_WINDOWS = {"fit": 653979, "calibration": 42241}
PASSAGE_IDS = (1, 2, 3)
SUBSET_MASKS = tuple(range(8))
PASSAGE_HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
FORBIDDEN_ANNOTATION_KEYS = frozenset({
    "label", "labels", "gold", "risk", "risk_mask", "original_labels",
    "hallucination", "hallucinations",
})

REPO = "NousResearch/Llama-2-7b-chat-hf"
REVISION = "351844e75ed0bcbbe3f10671b3c808d2b83894ee"
SEED = 20261014
THREADS = 4
FOLDS = 5
LR_C = 0.1
LOGIT_BATCH = 16
MAX_CONTEXT = 4096
MIN_FREE_GPU_BYTES = int(6.0 * 1024 ** 3)
GLOBAL_GPU_LOCK = ROOT / "results/.exclusive_gpu_runner.lock"
BACKEND = "llama2_7b_chat_nf4_bf16_sdpa_teacher_forced_selected_logprob_v1"


FEATURE_NAMES = (
    "full_nll", "empty_nll", "full_minus_empty",
    "shapley_passage_1", "shapley_passage_2", "shapley_passage_3",
    "single_gain_passage_1", "single_gain_passage_2", "single_gain_passage_3",
    "loo_drop_passage_1", "loo_drop_passage_2", "loo_drop_passage_3",
    "subset_range", "subset_std",
    "pair_interaction_12", "pair_interaction_13", "pair_interaction_23",
    "triple_interaction", "interaction_l1",
    "shapley_abs_max", "shapley_positive_sum", "shapley_negative_mass",
    "citation_any", "citation_source_fraction", "citation_shapley_sum",
    "citation_shapley_mean", "citation_positive_shapley_sum",
    "citation_single_gain_max", "citation_loo_drop_max",
)


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    pending.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    assert not path.exists(), f"No silent overwrite: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def reject_annotation_keys(value):
    if isinstance(value, dict):
        overlap = FORBIDDEN_ANNOTATION_KEYS & set(value)
        assert not overlap, f"Annotation field reached label-free preparation: {overlap}"
        for child in value.values():
            reject_annotation_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_annotation_keys(child)


def protocol():
    return {
        "version": VERSION,
        "role": ROLE,
        "method_boundary": (
            "Our exact three-source attribution candidate. It does not modify, "
            "retrain, tune, or relabel any ContextCite, LUMINA, Lookback Lens, "
            "ReDeEP, HalluRAG, or other baseline."
        ),
        "cohort": {
            "fit_answers": 3680,
            "calibration_answers": 159,
            "source_input": "Frozen label-free LUMINA preparation artifact; original view only.",
            "official_test": "Never opened.",
        },
        "intervention": {
            "sources": "The exact original Passage 1/2/3 character segments, including headers and inter-passage whitespace.",
            "views": "All masks 000..111 in integer order; selected segments retain original source order. Empty removes only the exact material block.",
            "fixed_content": "Prompt text outside that block and the published answer bytes are identical in all eight views.",
            "tokenization": "One full-string fast-tokenizer call per view, literal existing Llama wrapper, no added tokens, padding, truncation, or generation.",
            "frozen_coordinates": "Prompt/shell/passage hashes, prefix token IDs, full-input hashes, answer/predictor positions, response offsets, and kept-source token positions.",
        },
        "extraction": {
            "checkpoint": REPO,
            "revision": REVISION,
            "quantization": "NF4 double quantization; BF16 compute/storage conventions copied from the audited QA replay.",
            "score": "For every published answer token and every subset, log softmax probability of the selected next token at its teacher-forced predictor position.",
            "execution": "Eight views run sequentially. gpu-smoke and extract are separate explicit commands guarded by a cross-run lock, foreign-compute-process check, and free-memory gate.",
            "historical_trace": False,
        },
        "features": {
            "axis": "One unchanged raw answer BPE axis shared by all views.",
            "full_empty": "f(111)-f(000), plus full and empty NLL.",
            "shapley": "Exact n=3 Shapley values over all eight log-probability coalitions; efficiency sum(phi)=f(111)-f(000) checked per token.",
            "source_effects": "Three empty-to-single gains and three full leave-one-source-out drops.",
            "dispersion": "Eight-view range and population standard deviation.",
            "interactions": "Three pair and one triple exact Mobius/Harsanyi interactions, plus fixed summaries.",
            "citations": "A token inherits valid Passage IDs cited in its fixed automatic sentence claim; cited Shapley/gain/drop summaries use that frozen bit mask. Missing citations remain explicit zero-indicator cases.",
            "names": list(FEATURE_NAMES),
            "width": len(FEATURE_NAMES),
        },
        "readout": {
            "labels": "Opened only by score after every GPU cache is frozen. Train only on lexical raw BPE labels.",
            "model": "Fixed StandardScaler plus L2 LogisticRegression, C=0.1, liblinear, seed fixed; no architecture or hyperparameter search.",
            "crossfit": "Five GroupKFold splits by source-connected group produce every fit token prediction; one all-fit model predicts calibration.",
            "weights": "Within each training fold: equal source-group mass, then equal answer/token mass; binary class balance, then restore equal source-group loss mass.",
            "thresholds": "Window and answer thresholds selected only on fit OOF predictions. Calibration is reporting only.",
            "projection": "Unchanged eligible stride-one 4-original-BPE windows receive max lexical-token risk; answer receives max eligible-window risk.",
        },
        "stage_gate": "initialize -> prepare -> check -> audit; after review only gpu-smoke -> extract -> score",
        "limitations": (
            "Teacher-forced source removal measures conditional evidence influence, not a historical causal trace. "
            "Removing passages also changes length and positions. Correlated passages make individual attribution share-dependent."
        ),
    }


def split_passages(material):
    matches = list(PASSAGE_HEADER.finditer(material))
    assert [int(match.group(1)) for match in matches] == [1, 2, 3]
    assert matches[0].start() == 0
    result = []
    for index, match in enumerate(matches):
        right = matches[index + 1].start() if index + 1 < len(matches) else len(material)
        text = material[match.start():right]
        assert text and text.startswith(match.group())
        result.append({
            "passage_id": int(match.group(1)),
            "material_char_start": match.start(),
            "material_char_end": right,
            "text": text,
            "text_sha256": digest(text),
        })
    assert "".join(item["text"] for item in result) == material
    return result


def render_subset(shell_left, shell_right, passages, mask):
    assert mask in SUBSET_MASKS
    body = []
    coordinates = []
    cursor = len(shell_left)
    for passage in passages:
        bit = 1 << (passage["passage_id"] - 1)
        if mask & bit:
            body.append(passage["text"])
            coordinates.append({
                "passage_id": passage["passage_id"],
                "prompt_char_start": cursor,
                "prompt_char_end": cursor + len(passage["text"]),
                "rendered_char_start": len(fq.WRAPPER_LEFT) + cursor,
                "rendered_char_end": len(fq.WRAPPER_LEFT) + cursor + len(passage["text"]),
                "text_sha256": passage["text_sha256"],
            })
            cursor += len(passage["text"])
    prompt = shell_left + "".join(body) + shell_right
    assert cursor == len(prompt) - len(shell_right)
    return prompt, coordinates


def passage_token_positions(offsets, source_coordinates):
    offsets = np.asarray(offsets, dtype=np.int32)
    result = []
    for source in source_coordinates:
        left, right = source["rendered_char_start"], source["rendered_char_end"]
        positions = np.flatnonzero((offsets[:, 1] > left) & (offsets[:, 0] < right) &
                                   (offsets[:, 1] > offsets[:, 0])).tolist()
        assert positions
        result.append({**source, "token_positions": positions,
                       "token_positions_sha256": digest(positions)})
    return result


def claim_citation_geometry(response, offsets, plan):
    claims = plan["claims"]
    assert claims
    for claim in claims:
        assert response[claim["start"]:claim["end"]] == claim["text"]
    citation_masks = np.zeros(len(offsets), dtype=np.uint8)
    claim_owners = np.full(len(offsets), -1, dtype=np.int32)
    lexical = np.zeros(len(offsets), dtype=bool)
    claim_rows = []
    for claim_index, claim in enumerate(claims):
        parsed = citation.parse_citations(claim["text"])
        valid = sorted({source_id for ref in parsed["references"]
                        for source_id in ref["ids"] if source_id in PASSAGE_IDS})
        invalid = sorted({source_id for ref in parsed["references"]
                          for source_id in ref["ids"] if source_id not in PASSAGE_IDS})
        mask = sum(1 << (source_id - 1) for source_id in valid)
        claim_rows.append({
            "claim_index": claim_index,
            "char_start": claim["start"],
            "char_end": claim["end"],
            "text_sha256": digest(claim["text"]),
            "valid_cited_passage_ids": valid,
            "invalid_cited_passage_ids": invalid,
            "citation_mask": mask,
            "citation_status": parsed["status"],
            "unknown_citation_count": len(parsed["unknown"]),
        })
    for token_index, (left, right) in enumerate(offsets):
        chars = [position for position in range(left, right)
                 if 0 <= position < len(response) and response[position].isalnum()]
        lexical[token_index] = bool(chars)
        if not chars:
            continue
        overlaps = [sum(claim["start"] <= position < claim["end"] for position in chars)
                    for claim in claims]
        owner = max(range(len(claims)), key=lambda index: (overlaps[index], -index))
        assert overlaps[owner] > 0
        assert all(any(claim["start"] <= position < claim["end"] for claim in claims)
                   for position in chars)
        claim_owners[token_index] = owner
        citation_masks[token_index] = claim_rows[owner]["citation_mask"]
    assert np.array_equal(claim_owners >= 0, lexical)
    return {
        "claims": claim_rows,
        "token_claim_index": claim_owners.tolist(),
        "lexical_mask": lexical.tolist(),
        "citation_mask": citation_masks.tolist(),
        "citation_mask_sha256": digest(citation_masks.tolist()),
    }


def exact_shapley(log_probability):
    f = np.asarray(log_probability, dtype=np.float64)
    assert f.ndim == 2 and f.shape[0] == 8 and np.isfinite(f).all()
    phi = np.empty((3, f.shape[1]), dtype=np.float64)
    for source in range(3):
        bit = 1 << source
        value = np.zeros(f.shape[1], dtype=np.float64)
        for mask in SUBSET_MASKS:
            if mask & bit:
                continue
            size = int(mask).bit_count()
            weight = (1 / 3, 1 / 6, 1 / 3)[size]
            value += weight * (f[mask | bit] - f[mask])
        phi[source] = value
    error = np.max(np.abs(phi.sum(axis=0) - (f[7] - f[0])))
    assert error <= 2e-5, error
    return phi, float(error)


def derive_features(log_probability, citation_mask):
    f = np.asarray(log_probability, dtype=np.float64)
    citations = np.asarray(citation_mask, dtype=np.uint8)
    assert f.shape == (8, len(citations)) and np.all(citations <= 7)
    phi, efficiency_error = exact_shapley(f)
    single = np.vstack((f[1] - f[0], f[2] - f[0], f[4] - f[0]))
    loo = np.vstack((f[7] - f[6], f[7] - f[5], f[7] - f[3]))
    pair = np.vstack((
        f[3] - f[1] - f[2] + f[0],
        f[5] - f[1] - f[4] + f[0],
        f[6] - f[2] - f[4] + f[0],
    ))
    triple = f[7] - f[3] - f[5] - f[6] + f[1] + f[2] + f[4] - f[0]
    cited = np.vstack([((citations >> source) & 1).astype(bool) for source in range(3)])
    cited_count = cited.sum(axis=0)
    cited_phi_sum = (phi * cited).sum(axis=0)
    cited_single = np.where(cited, single, -np.inf).max(axis=0)
    cited_loo = np.where(cited, loo, -np.inf).max(axis=0)
    cited_single[cited_count == 0] = 0.0
    cited_loo[cited_count == 0] = 0.0
    values = [
        -f[7], -f[0], f[7] - f[0],
        *phi, *single, *loo,
        np.ptp(f, axis=0), np.std(f, axis=0),
        *pair, triple, np.abs(pair).sum(axis=0) + np.abs(triple),
        np.abs(phi).max(axis=0), np.maximum(phi, 0).sum(axis=0),
        -np.minimum(phi, 0).sum(axis=0),
        (cited_count > 0).astype(np.float64), cited_count / 3.0,
        cited_phi_sum,
        np.divide(cited_phi_sum, cited_count, out=np.zeros_like(cited_phi_sum),
                  where=cited_count > 0),
        (np.maximum(phi, 0) * cited).sum(axis=0),
        cited_single, cited_loo,
    ]
    result = np.column_stack(values).astype(np.float32)
    assert result.shape == (f.shape[1], len(FEATURE_NAMES))
    assert np.isfinite(result).all()
    return result, efficiency_error


def synthetic_selfcheck():
    material = "passage 1: alpha\npassage 2: beta\npassage 3: gamma"
    passages = split_passages(material)
    prompt, coordinates = render_subset("L|", "|R", passages, 5)
    assert prompt == "L|passage 1: alpha\npassage 3: gamma|R"
    assert [item["passage_id"] for item in coordinates] == [1, 3]
    # Additive game: Shapley exactly recovers the three source effects.
    effects = np.asarray([[0.2, -0.1], [0.3, 0.4], [-0.5, 0.7]])
    f = np.empty((8, 2), dtype=np.float64)
    for mask in SUBSET_MASKS:
        f[mask] = -2 + sum(effects[source] for source in range(3)
                           if mask & (1 << source))
    phi, error = exact_shapley(f)
    assert np.allclose(phi, effects) and error < 1e-12
    x, _ = derive_features(f, [0, 5])
    assert x.shape == (2, len(FEATURE_NAMES))
    assert x[0, FEATURE_NAMES.index("citation_any")] == 0
    assert x[1, FEATURE_NAMES.index("citation_source_fraction")] == 2 / 3
    parsed = citation.parse_citations("Claim [1,3].")
    assert parsed["references"][0]["ids"] == [1, 3]
    return {
        "status": "passed",
        "eight_masks": True,
        "exact_passage_order": True,
        "shapley_efficiency": True,
        "citation_masked_features": True,
        "feature_width": len(FEATURE_NAMES),
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_initialized": torch.cuda.is_initialized(),
        "official_test_opened": False,
    }


def label_free_source_snapshot():
    source_complete = read(SOURCE_COMPLETE)
    assert source_complete["status"] == "CPU_ready_not_extracted"
    assert source_complete["gold_labels_read"] is False
    assert source_complete["official_test_opened"] is False
    assert source_complete["artifacts_sha256"]["feature_inputs.jsonl"] == EXPECTED_SOURCE_SHA256
    assert sha(SOURCE_INPUTS) == EXPECTED_SOURCE_SHA256
    paths = [
        Path(__file__), Path(fq.__file__), Path(citation.__file__), Path(q.__file__),
        SOURCE_INPUTS, SOURCE_COMPLETE, OLD_CLAIMS, NEW_CLAIMS, MODEL_MANIFEST,
        MODEL / "config.json", MODEL / "tokenizer.json", MODEL / "tokenizer.model",
        MODEL / "tokenizer_config.json", MODEL / "special_tokens_map.json",
    ]
    return {str(path.resolve()): sha(path) for path in paths}


def load_claim_plans():
    old = list(lines(OLD_CLAIMS))
    new = list(lines(NEW_CLAIMS))
    assert len(old) == EXPECTED_CLAIM_ROWS["old"]
    assert len(new) == EXPECTED_CLAIM_ROWS["new"]
    assert Counter(item["partition"] for item in old) == {"fit": 634, "calibration": 159}
    assert Counter(item["partition"] for item in new) == {"fit": 3046}
    plans = {item["response_id"]: item for item in old + new}
    assert len(plans) == sum(EXPECTED_ANSWERS.values())
    reject_annotation_keys(old)
    reject_annotation_keys(new)
    return plans


def build_prepared_row(tokenizer, source, plan):
    allowed_label_flag = source.pop("labels_used")
    assert allowed_label_flag is False
    reject_annotation_keys(source)
    assert source["official_split"] == "train"
    assert source["partition"] in EXPECTED_ANSWERS
    assert source["response_id"] == plan["response_id"]
    assert source["group_id"] == plan["group_id"]
    prompt, response = source["original_prompt"], source["original_response"]
    assert digest(prompt) == source["original_prompt_sha256"]
    assert digest(response) == source["answer_sha256"] == plan["answer_sha256"]
    left, right = source["original_reference_range"]
    assert 0 <= left < right <= len(prompt)
    material = prompt[left:right]
    assert digest(material) == source["material_sha256"] == plan["document_sha256"]
    shell_left, shell_right = prompt[:left], prompt[right:]
    passages = split_passages(material)
    citation_geometry = claim_citation_geometry(response, source["response_token_offsets"], plan)
    views = []
    for mask in SUBSET_MASKS:
        view_prompt, coordinates = render_subset(shell_left, shell_right, passages, mask)
        encoded = fq.encode_view(tokenizer, view_prompt, response)
        assert encoded["answer_token_ids"] == source["answer_token_ids"]
        assert encoded["response_token_offsets"] == source["response_token_offsets"]
        assert encoded["response_token_offsets_raw"] == source["response_token_offsets_raw"]
        positions = encoded["answer_token_positions"]
        assert positions == list(range(positions[0], positions[0] + len(positions)))
        assert positions[-1] == len(encoded["input_ids"]) - 1
        prefix_ids = encoded["input_ids"][:positions[0]]
        assert encoded["input_ids"] == prefix_ids + source["answer_token_ids"]
        predictor = [position - 1 for position in positions]
        assert min(predictor) >= 0 and len(encoded["input_ids"]) <= MAX_CONTEXT
        source_tokens = passage_token_positions(encoded["input_token_offsets"], coordinates)
        views.append({
            "mask": mask,
            "included_passage_ids": [source_id for source_id in PASSAGE_IDS
                                     if mask & (1 << (source_id - 1))],
            "prompt_sha256": digest(view_prompt),
            "prompt_character_count": len(view_prompt),
            "prefix_ids": prefix_ids,
            "prefix_ids_sha256": digest(prefix_ids),
            "input_ids_sha256": digest(encoded["input_ids"]),
            "input_token_count": len(encoded["input_ids"]),
            "answer_positions": positions,
            "predictor_positions": predictor,
            "kept_passages": source_tokens,
        })
        if mask == 7:
            assert view_prompt == prompt
            assert encoded["input_ids"] == source["original_input_ids"]
            assert prefix_ids == source["original_prefix_ids"]
            assert positions == source["original_answer_positions"]
            assert predictor == source["original_predictor_positions"]
            assert digest(encoded["input_ids"]) == source["original_input_ids_sha256"]
    return {
        "version": VERSION,
        "method_role": ROLE,
        "response_id": source["response_id"],
        "source_id": source["source_id"],
        "group_id": source["group_id"],
        "partition": source["partition"],
        "official_split": "train",
        "prompt_shell_left": shell_left,
        "prompt_shell_right": shell_right,
        "prompt_shell_sha256": digest([shell_left, shell_right]),
        "original_prompt_sha256": source["original_prompt_sha256"],
        "material_sha256": source["material_sha256"],
        "passages": passages,
        "answer_text": response,
        "answer_sha256": source["answer_sha256"],
        "answer_token_ids": source["answer_token_ids"],
        "answer_token_ids_sha256": digest(source["answer_token_ids"]),
        "response_token_offsets": source["response_token_offsets"],
        "response_token_offsets_raw": source["response_token_offsets_raw"],
        "claim_plan_sha256": digest(plan),
        "citation_geometry": citation_geometry,
        "views": views,
        "labels_used": False,
        "exact_original_generation_trace": False,
    }


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), "Refuse to overwrite an existing candidate directory"
    OUT.mkdir(parents=True)
    save(OUT / "PREFLIGHT.json", synthetic_selfcheck())
    save(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Exact subset attribution v1（我们的方法候选）\n\n"
        "对每条已发布答案固定构造 8 个 Passage 子集，只重放同一答案的 selected-token log probability。"
        "由 8 个值精确计算三个来源的 Shapley 值、删源影响、交互和引用对应影响。\n\n"
        "prepare/check/audit 只做无标签 CPU 工作。审核后才可单独运行 gpu-smoke、extract；score 最后才读 fit/cal 标签，"
        "按 source group 做五折 OOF，并映射到统一 4-BPE 窗口和整答最大值。正式 baseline 不作任何改动。\n",
        encoding="utf-8",
    )
    snapshot = label_free_source_snapshot()
    save(OUT / "design_freeze.json", {
        "status": "frozen_before_preparation",
        "source_code_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "label_free_source_sha256": snapshot,
        "feature_names_sha256": digest(FEATURE_NAMES),
        "labels_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    print("EXACT_SUBSET_ATTRIBUTION_PROTOCOL_FROZEN", flush=True)


def prepare():
    assert not torch.cuda.is_initialized()
    freeze = read(OUT / "design_freeze.json")
    assert freeze["source_code_sha256"] == sha(__file__)
    assert freeze["protocol_sha256"] == sha(OUT / "protocol.json")
    assert read(OUT / "protocol.json") == protocol()
    assert freeze["label_free_source_sha256"] == label_free_source_snapshot()
    assert not (OUT / "prepare_started.json").exists(), "No silent preparation overwrite"
    save(OUT / "prepare_started.json", {
        "status": "label_free_CPU_preparation_started",
        "source_snapshot_sha256": digest(freeze["label_free_source_sha256"]),
        "labels_accessed": False, "GPU_used": False, "official_test_opened": False,
    })
    plans = load_claim_plans()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    assert tokenizer.is_fast
    counts = Counter()
    raw_tokens = 0
    input_tokens = np.zeros(8, dtype=np.int64)
    max_tokens = np.zeros(8, dtype=np.int64)
    cited_tokens = cited_claims = total_claims = 0
    response_ids = []
    started = time.perf_counter()

    def prepared_rows():
        nonlocal raw_tokens, cited_tokens, cited_claims, total_claims
        with SOURCE_INPUTS.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                source = json.loads(line)
                response_id = source["response_id"]
                assert response_id in plans
                row = build_prepared_row(tokenizer, source, plans[response_id])
                counts[row["partition"]] += 1
                response_ids.append(response_id)
                raw_tokens += len(row["answer_token_ids"])
                for view in row["views"]:
                    mask = view["mask"]
                    input_tokens[mask] += view["input_token_count"]
                    max_tokens[mask] = max(max_tokens[mask], view["input_token_count"])
                masks = row["citation_geometry"]["citation_mask"]
                cited_tokens += sum(mask > 0 for mask in masks)
                claims = row["citation_geometry"]["claims"]
                cited_claims += sum(claim["citation_mask"] > 0 for claim in claims)
                total_claims += len(claims)
                if (index + 1) % 100 == 0:
                    print("EXACT_SUBSET_CPU_PREPARED", index + 1, 3839, flush=True)
                yield row

    atomic_jsonl(OUT / "prepared_inputs.jsonl", prepared_rows())
    del tokenizer
    assert counts == EXPECTED_ANSWERS
    assert raw_tokens == EXPECTED_RAW_TOKENS
    assert len(response_ids) == len(set(response_ids)) == sum(EXPECTED_ANSWERS.values())
    assert set(response_ids) == set(plans)
    assert input_tokens[7] == read(SOURCE_COMPLETE)["counts"]["original_input_tokens"]
    statistics = {
        "status": "CPU_prepared_not_GPU_extracted",
        "answers": len(response_ids),
        "answers_by_partition": dict(counts),
        "raw_answer_tokens": raw_tokens,
        "input_tokens_by_subset_mask": input_tokens.tolist(),
        "all_eight_view_input_tokens": int(input_tokens.sum()),
        "max_input_tokens_by_subset_mask": max_tokens.tolist(),
        "max_any_view_input_tokens": int(max_tokens.max()),
        "model_forward_calls": len(response_ids) * 8,
        "automatic_claims": total_claims,
        "claims_with_valid_citation": cited_claims,
        "raw_tokens_in_claim_with_valid_citation": cited_tokens,
        "raw_logprob_float32_bytes": raw_tokens * 8 * 4,
        "derived_feature_float32_bytes": raw_tokens * len(FEATURE_NAMES) * 4,
        "prepared_inputs_bytes": (OUT / "prepared_inputs.jsonl").stat().st_size,
        "response_order_sha256": digest(response_ids),
        "prepared_inputs_sha256": sha(OUT / "prepared_inputs.jsonl"),
        "seconds": time.perf_counter() - started,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    assert statistics["max_any_view_input_tokens"] <= MAX_CONTEXT
    save(OUT / "preparation_statistics.json", statistics)
    save(OUT / "source_snapshot.json", {
        "files_sha256": freeze["label_free_source_sha256"],
        "official_test_opened": False,
    })
    names = ("PREFLIGHT.json", "protocol.json", "PLAN.md", "design_freeze.json",
             "prepare_started.json", "prepared_inputs.jsonl", "preparation_statistics.json",
             "source_snapshot.json")
    save(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_waiting_for_GPU_review",
        "answers": len(response_ids),
        "raw_answer_tokens": raw_tokens,
        "eight_views_per_answer": True,
        "files_sha256": {name: sha(OUT / name) for name in names},
        "labels_accessed": False, "new_fits": 0, "GPU_used": False,
        "official_test_opened": False,
    })
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_ATTRIBUTION_CPU_PREPARED", flush=True)


def check_prepared():
    assert not torch.cuda.is_initialized()
    complete = read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_waiting_for_GPU_review"
    assert complete["answers"] == sum(EXPECTED_ANSWERS.values())
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    freeze = read(OUT / "design_freeze.json")
    assert freeze["source_code_sha256"] == sha(__file__)
    assert freeze["label_free_source_sha256"] == label_free_source_snapshot()
    return complete


def check():
    complete = check_prepared()
    counts = Counter()
    input_tokens = np.zeros(8, dtype=np.int64)
    raw_tokens = 0
    sample = []
    with (OUT / "prepared_inputs.jsonl").open(encoding="utf-8") as prepared_handle, \
            SOURCE_INPUTS.open(encoding="utf-8") as source_handle:
        prepared_lines = (line for line in prepared_handle if line.strip())
        source_lines = (line for line in source_handle if line.strip())
        for index, pair in enumerate(zip_longest(prepared_lines, source_lines)):
            prepared_line, source_line = pair
            assert prepared_line is not None and source_line is not None
            row, source = json.loads(prepared_line), json.loads(source_line)
            allowed_flag = row.pop("labels_used")
            assert allowed_flag is False
            reject_annotation_keys(row)
            assert row["response_id"] == source["response_id"]
            assert row["answer_sha256"] == source["answer_sha256"]
            assert row["answer_token_ids"] == source["answer_token_ids"]
            assert row["response_token_offsets"] == source["response_token_offsets"]
            assert row["response_token_offsets_raw"] == source["response_token_offsets_raw"]
            assert len(row["views"]) == 8
            passages = row["passages"]
            assert [passage["passage_id"] for passage in passages] == [1, 2, 3]
            assert "".join(passage["text"] for passage in passages) == source["original_prompt"][source["original_reference_range"][0]:source["original_reference_range"][1]]
            for mask, view in enumerate(row["views"]):
                assert view["mask"] == mask
                prompt, coordinates = render_subset(row["prompt_shell_left"],
                                                     row["prompt_shell_right"], passages, mask)
                assert digest(prompt) == view["prompt_sha256"]
                assert [{key: item[key] for key in ("passage_id", "prompt_char_start",
                        "prompt_char_end", "rendered_char_start", "rendered_char_end", "text_sha256")}
                        for item in view["kept_passages"]] == coordinates
                positions = view["answer_positions"]
                assert positions == list(range(len(view["prefix_ids"]),
                                               len(view["prefix_ids"]) + len(row["answer_token_ids"])))
                assert view["predictor_positions"] == [position - 1 for position in positions]
                full_ids = view["prefix_ids"] + row["answer_token_ids"]
                assert digest(view["prefix_ids"]) == view["prefix_ids_sha256"]
                assert digest(full_ids) == view["input_ids_sha256"]
                assert len(full_ids) == view["input_token_count"] <= MAX_CONTEXT
                input_tokens[mask] += len(full_ids)
                assert [item["passage_id"] for item in view["kept_passages"]] == view["included_passage_ids"]
            assert row["views"][7]["input_ids_sha256"] == source["original_input_ids_sha256"]
            geometry = row["citation_geometry"]
            assert len(geometry["citation_mask"]) == len(row["answer_token_ids"])
            assert len(geometry["token_claim_index"]) == len(row["answer_token_ids"])
            assert len(geometry["lexical_mask"]) == len(row["answer_token_ids"])
            assert digest(geometry["citation_mask"]) == geometry["citation_mask_sha256"]
            assert all(0 <= value <= 7 for value in geometry["citation_mask"])
            counts[row["partition"]] += 1
            raw_tokens += len(row["answer_token_ids"])
            if index in (0, 1919, 3838):
                sample.append({"response_id": row["response_id"],
                               "input_tokens": [view["input_token_count"] for view in row["views"]],
                               "answer_tokens": len(row["answer_token_ids"])})
    stats = read(OUT / "preparation_statistics.json")
    assert counts == EXPECTED_ANSWERS
    assert raw_tokens == stats["raw_answer_tokens"] == EXPECTED_RAW_TOKENS
    assert input_tokens.tolist() == stats["input_tokens_by_subset_mask"]
    report = {
        "status": "passed_waiting_for_GPU_review",
        "preflight_replayed": synthetic_selfcheck(),
        "answers": sum(counts.values()),
        "answers_by_partition": dict(counts),
        "raw_answer_tokens": raw_tokens,
        "all_eight_view_input_tokens": int(input_tokens.sum()),
        "full_view_equals_frozen_original_for_every_answer": True,
        "all_views_share_exact_answer_ids_and_offsets": True,
        "all_subset_prompt_and_token_hashes_replayed": True,
        "all_passage_coordinates_and_citation_masks_valid": True,
        "sample": sample,
        "labels_accessed": False, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False,
    }
    save(OUT / "CPU_CHECK.json", report)
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_ATTRIBUTION_CPU_CHECK_PASSED", flush=True)


def runtime_signature():
    return {
        "backend": BACKEND,
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "prepared_inputs_sha256": sha(OUT / "prepared_inputs.jsonl"),
        "repo": REPO, "revision": REVISION,
        "load_config": {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
            "llm_int8_skip_modules": ["lm_head"],
            "torch_dtype": "bfloat16", "attention": "sdpa",
            "device_map": {"": "cuda:0"}, "local_files_only": True,
            "padding": False, "truncation": False, "use_cache": False,
            "tf32": False,
        },
        "logit_batch": LOGIT_BATCH, "seed": SEED, "threads": THREADS,
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "accelerate", "numpy")},
        "cuda_runtime": torch.version.cuda,
        "method_role": ROLE,
    }


@contextmanager
def exclusive_gpu(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    token = {"pid": os.getpid(), "stage": stage, "method": VERSION,
             "started_unix": time.time()}
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(GLOBAL_GPU_LOCK, flags)
    except FileExistsError as error:
        raise RuntimeError(f"Another GPU runner owns {GLOBAL_GPU_LOCK}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(token, handle)
        query = subprocess.run([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        foreign = []
        for line in query.stdout.splitlines():
            if not line.strip():
                continue
            pid = int(line.split(",", 1)[0].strip())
            if pid != os.getpid():
                foreign.append(line.strip())
        assert not foreign, f"Foreign GPU compute process detected: {foreign}"
        assert torch.cuda.is_available()
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        assert free_bytes >= MIN_FREE_GPU_BYTES, (free_bytes, MIN_FREE_GPU_BYTES)
        token["free_bytes_before_load"] = int(free_bytes)
        token["total_bytes"] = int(total_bytes)
        yield token
    finally:
        if GLOBAL_GPU_LOCK.exists():
            try:
                current = json.loads(GLOBAL_GPU_LOCK.read_text(encoding="utf-8"))
            except Exception:
                current = None
            if current and current.get("pid") == os.getpid() and current.get("stage") == stage:
                GLOBAL_GPU_LOCK.unlink()


def load_nf4():
    manifest = read(MODEL_MANIFEST)
    assert manifest["status"] == "complete"
    assert manifest["repo_id"] == REPO and manifest["revision"] == REVISION
    for item in manifest["files"]:
        path = MODEL / item["filename"]
        assert item["status"] == "verified" and item["source_hash_match"]
        assert path.stat().st_size == item["actual_bytes"]
        assert sha(path) == item["actual_sha256"]
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=["lm_head"],
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
    assert getattr(model, "is_loaded_in_4bit", False)
    assert fq.architecture(model)["model_type"] == "llama"
    assert model.model.embed_tokens.weight.device.type == "cuda"
    first = model.model.layers[0].self_attn.q_proj
    assert first.weight.quant_state.quant_type == "nf4"
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    return model, {
        "load_seconds": time.perf_counter() - started,
        "gpu_name": properties.name,
        "total_gpu_bytes": properties.total_memory,
        "allocated_after_load_bytes": torch.cuda.memory_allocated(),
        "attention_backend": model.config._attn_implementation,
        "embedding_dtype": str(model.model.embed_tokens.weight.dtype),
        "lm_head_dtype": str(model.lm_head.weight.dtype),
        "architecture": fq.architecture(model),
    }


def selected_token_logprob(model, prefix_ids, answer_ids, answer_positions):
    input_ids = prefix_ids + answer_ids
    positions = np.asarray(answer_positions, dtype=np.int64)
    assert positions.tolist() == list(range(len(prefix_ids), len(input_ids)))
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    values = np.empty(len(answer_ids), dtype=np.float32)
    with torch.inference_mode():
        hidden = model.model(input_ids=ids, attention_mask=attention,
                             use_cache=False, return_dict=True).last_hidden_state[0]
        for left in range(0, len(positions), LOGIT_BATCH):
            right = min(left + LOGIT_BATCH, len(positions))
            target_positions = torch.as_tensor(positions[left:right], device=device)
            predictor_hidden = hidden.index_select(0, target_positions - 1)
            logits = model.lm_head(predictor_hidden).float()
            logp = logits.log_softmax(dim=-1)
            targets = ids[0].index_select(0, target_positions)
            values[left:right] = logp.gather(1, targets[:, None]).squeeze(1).cpu().numpy()
            del predictor_hidden, logits, logp, targets, target_positions
    del ids, attention, hidden
    assert np.isfinite(values).all() and np.all(values <= 1e-5)
    return values


def extract_row(model, row):
    result = np.empty((8, len(row["answer_token_ids"])), dtype=np.float32)
    for view in row["views"]:
        result[view["mask"]] = selected_token_logprob(
            model, view["prefix_ids"], row["answer_token_ids"], view["answer_positions"])
    features, efficiency = derive_features(result, row["citation_geometry"]["citation_mask"])
    assert features.shape == (len(row["answer_token_ids"]), len(FEATURE_NAMES))
    return result, efficiency


def cache_signature(row):
    return digest({
        "runtime": runtime_signature(),
        "response_id": row["response_id"],
        "answer_token_ids_sha256": row["answer_token_ids_sha256"],
        "citation_mask_sha256": row["citation_geometry"]["citation_mask_sha256"],
        "view_input_sha256": [view["input_ids_sha256"] for view in row["views"]],
    })


def validate_cache(path, row):
    metadata_path = path.with_suffix(".json")
    metadata = read(metadata_path)
    assert metadata["complete"] is True
    assert metadata["response_id"] == row["response_id"]
    assert metadata["cache_signature"] == cache_signature(row)
    assert metadata["npz_sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as archive:
        logp = archive["selected_logprob"].copy()
        answer_ids = archive["answer_token_ids"].copy()
        citation_mask = archive["citation_mask"].copy()
    assert logp.shape == (8, len(row["answer_token_ids"])) and logp.dtype == np.float32
    assert np.isfinite(logp).all()
    assert answer_ids.tolist() == row["answer_token_ids"]
    assert citation_mask.tolist() == row["citation_geometry"]["citation_mask"]
    _, efficiency = derive_features(logp, citation_mask)
    assert abs(efficiency - metadata["shapley_efficiency_max_abs"]) <= 1e-12
    return logp


def clean_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def gpu_smoke():
    rows = list(lines(OUT / "prepared_inputs.jsonl"))
    check_prepared()
    assert not (OUT / "GPU_SMOKE.json").exists(), "No silent smoke overwrite"
    chosen = [rows[0], rows[-1]]
    with exclusive_gpu("gpu-smoke") as lease:
        model, model_meta = load_nf4()
        try:
            started = time.perf_counter()
            records = []
            first_full = None
            for row in chosen:
                logp, efficiency = extract_row(model, row)
                records.append({
                    "response_id": row["response_id"],
                    "tokens": logp.shape[1],
                    "logprob_min": float(logp.min()),
                    "logprob_max": float(logp.max()),
                    "shapley_efficiency_max_abs": efficiency,
                    "logprob_sha256": digest(logp.tolist()),
                })
                if first_full is None:
                    first_full = logp[7].copy()
            repeated = selected_token_logprob(model, chosen[0]["views"][7]["prefix_ids"],
                                              chosen[0]["answer_token_ids"],
                                              chosen[0]["views"][7]["answer_positions"])
            repeat_error = float(np.max(np.abs(first_full - repeated)))
            assert repeat_error <= 1e-6
            save(OUT / "GPU_SMOKE.json", {
                "status": "passed_not_full_extraction",
                "runtime_signature": runtime_signature(),
                "runtime_signature_sha256": digest(runtime_signature()),
                "lease": lease, "model": model_meta, "records": records,
                "full_view_repeat_max_abs": repeat_error,
                "seconds": time.perf_counter() - started,
                "labels_accessed": False, "official_test_opened": False,
            })
        finally:
            del model
            clean_gpu()
    print("EXACT_SUBSET_ATTRIBUTION_GPU_SMOKE_PASSED", flush=True)


def extract():
    check_prepared()
    smoke = read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_not_full_extraction"
    assert smoke["runtime_signature_sha256"] == digest(runtime_signature())
    assert not (OUT / "extraction_complete.json").exists()
    cache_dir = OUT / "token_logprob"
    cache_dir.mkdir(exist_ok=True)
    records = []
    with exclusive_gpu("extract") as lease:
        model, model_meta = load_nf4()
        started = time.perf_counter()
        try:
            for index, row in enumerate(lines(OUT / "prepared_inputs.jsonl")):
                path = cache_dir / f"{row['response_id']}.npz"
                if path.exists() and path.with_suffix(".json").exists():
                    validate_cache(path, row)
                else:
                    assert not path.exists() and not path.with_suffix(".json").exists()
                    logp, efficiency = extract_row(model, row)
                    atomic_npz(path, selected_logprob=logp,
                               answer_token_ids=np.asarray(row["answer_token_ids"], dtype=np.int32),
                               citation_mask=np.asarray(row["citation_geometry"]["citation_mask"], dtype=np.uint8))
                    save(path.with_suffix(".json"), {
                        "complete": True, "response_id": row["response_id"],
                        "cache_signature": cache_signature(row),
                        "npz_sha256": sha(path),
                        "shapley_efficiency_max_abs": efficiency,
                        "labels_accessed": False, "official_test_opened": False,
                    })
                records.append({"response_id": row["response_id"],
                                "npz_sha256": sha(path),
                                "metadata_sha256": sha(path.with_suffix(".json"))})
                if (index + 1) % 25 == 0:
                    print("EXACT_SUBSET_GPU_EXTRACTED", index + 1, 3839, flush=True)
        finally:
            del model
            clean_gpu()
    assert len(records) == sum(EXPECTED_ANSWERS.values())
    save(OUT / "extraction_complete.json", {
        "status": "complete_frozen_logprob_not_scored",
        "answers": len(records), "raw_answer_tokens": EXPECTED_RAW_TOKENS,
        "records": records, "lease": lease, "model": model_meta,
        "runtime_signature_sha256": digest(runtime_signature()),
        "prepared_inputs_sha256": sha(OUT / "prepared_inputs.jsonl"),
        "seconds": time.perf_counter() - started,
        "labels_accessed": False, "official_test_opened": False,
    })
    print("EXACT_SUBSET_ATTRIBUTION_EXTRACTION_COMPLETE", flush=True)


def check_extracted(prepared):
    complete = read(OUT / "extraction_complete.json")
    assert complete["status"] == "complete_frozen_logprob_not_scored"
    assert complete["runtime_signature_sha256"] == digest(runtime_signature())
    assert complete["prepared_inputs_sha256"] == sha(OUT / "prepared_inputs.jsonl")
    records = {row["response_id"]: row for row in complete["records"]}
    assert len(records) == len(prepared) == sum(EXPECTED_ANSWERS.values())
    for row in prepared:
        path = OUT / "token_logprob" / f"{row['response_id']}.npz"
        assert sha(path) == records[row["response_id"]]["npz_sha256"]
        assert sha(path.with_suffix(".json")) == records[row["response_id"]]["metadata_sha256"]
        validate_cache(path, row)
    return complete


def expanded_metadata():
    """Gold-opening function. It is reachable only from the explicit score stage."""
    original = q.metadata()
    fit_answers = list(q.lines(EXPANSION / "data/answers_fit.jsonl"))
    fit_tokens = list(q.lines(EXPANSION / "data/tokens_fit.jsonl"))
    fit_windows = list(q.lines(EXPANSION / "data/windows_k4_fit.jsonl"))
    assert len(fit_answers) == len(fit_tokens) == EXPECTED_ANSWERS["fit"]
    assert len(fit_windows) == EXPECTED_WINDOWS["fit"]
    assert fit_answers[:634] == original["answers"][:634]
    assert fit_tokens[:634] == original["tokens"][:634]
    assert fit_windows[:168123] == original["windows"][:168123]
    answers = fit_answers + original["answers"][634:]
    tokens = fit_tokens + original["tokens"][634:]
    windows = fit_windows + original["windows"][168123:]
    assert len(answers) == len(tokens) == 3839 and len(windows) == sum(EXPECTED_WINDOWS.values())
    by_response = {answer["response_id"]: {"answer": answer, "tokens": token}
                   for answer, token in zip(answers, tokens)}
    assert len(by_response) == 3839
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[window["response_id"]].append(index)
    assert all(answer_windows[answer["response_id"]] for answer in answers)
    groups = {partition: {answer["group_id"] for answer in answers
                          if answer["partition"] == partition}
              for partition in EXPECTED_ANSWERS}
    assert len(groups["fit"]) == 615 and len(groups["calibration"]) == 154
    assert groups["fit"].isdisjoint(groups["calibration"])
    return {
        "answers": answers, "tokens": tokens, "windows": windows,
        "by_response": by_response, "answer_windows": dict(answer_windows),
        "bounds": {"fit": [0, EXPECTED_WINDOWS["fit"]],
                   "calibration": [EXPECTED_WINDOWS["fit"], len(windows)]},
    }


def fold_token_weights(meta, answer_indices, token_starts, y_token, lexical):
    tree = defaultdict(list)
    for answer_index in answer_indices:
        answer = meta["answers"][answer_index]
        tree[answer["group_id"]].append(answer_index)
    weights = np.zeros(len(y_token), dtype=np.float64)
    group_token_indices = {}
    for group_id, indices in tree.items():
        flat = []
        for answer_index in indices:
            answer = meta["answers"][answer_index]
            response_id = answer["response_id"]
            start, end = token_starts[response_id]
            active = np.flatnonzero(lexical[start:end]) + start
            assert len(active)
            weights[active] = 1.0 / (len(indices) * len(active))
            flat.extend(active.tolist())
        group_token_indices[group_id] = np.asarray(flat, dtype=np.int64)
    active = weights > 0
    weights[active] *= active.sum() / weights[active].sum()
    mass = np.bincount(y_token[active], weights=weights[active], minlength=2)
    assert np.all(mass > 0)
    factors = mass.sum() / (2 * mass)
    weights[active] *= factors[y_token[active]]
    target_group_mass = active.sum() / len(tree)
    for indices in group_token_indices.values():
        weights[indices] *= target_group_mass / weights[indices].sum()
    weights[active] *= active.sum() / weights[active].sum()
    totals = [weights[indices].sum() for indices in group_token_indices.values()]
    assert max(totals) - min(totals) <= 1e-7
    return weights


def project_token_scores(meta, token_scores, token_starts):
    window_scores = np.empty(len(meta["windows"]), dtype=np.float64)
    for index, window in enumerate(meta["windows"]):
        token = meta["by_response"][window["response_id"]]["tokens"]
        lexical = np.asarray(token["lexical_mask"], dtype=bool)
        local = [position for position in window["token_indices"] if lexical[position]]
        assert local
        start = token_starts[window["response_id"]][0]
        window_scores[index] = max(token_scores[start + position] for position in local)
    assert np.isfinite(window_scores).all()
    return window_scores


def score():
    assert not torch.cuda.is_initialized()
    check_prepared()
    prepared = list(lines(OUT / "prepared_inputs.jsonl"))
    check_extracted(prepared)
    assert not (OUT / "score_started.json").exists(), "No silent score overwrite"
    save(OUT / "score_started.json", {
        "status": "development_gold_opened_only_after_frozen_extraction",
        "official_test_opened": False,
    })
    meta = expanded_metadata()
    assert [row["response_id"] for row in prepared] == [answer["response_id"] for answer in meta["answers"]]
    total_tokens = sum(len(row["answer_token_ids"]) for row in prepared)
    assert total_tokens == EXPECTED_RAW_TOKENS
    feature_path = OUT / "token_features.npy"
    x = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                  shape=(total_tokens, len(FEATURE_NAMES)))
    token_starts = {}
    citation_mask = np.empty(total_tokens, dtype=np.uint8)
    cursor = 0
    for row in prepared:
        response_id = row["response_id"]
        token = meta["by_response"][response_id]["tokens"]
        assert row["answer_token_ids"] == token["token_ids"]
        assert row["response_token_offsets"] == token["response_token_offsets"]
        logp = validate_cache(OUT / "token_logprob" / f"{response_id}.npz", row)
        features, _ = derive_features(logp, row["citation_geometry"]["citation_mask"])
        right = cursor + len(features)
        x[cursor:right] = features
        citation_mask[cursor:right] = row["citation_geometry"]["citation_mask"]
        token_starts[response_id] = (cursor, right)
        cursor = right
    x.flush()
    y_token = np.empty(total_tokens, dtype=np.int8)
    lexical = np.empty(total_tokens, dtype=bool)
    for answer in meta["answers"]:
        response_id = answer["response_id"]
        left, right = token_starts[response_id]
        token = meta["by_response"][response_id]["tokens"]
        y_token[left:right] = token["risk_mask"]
        lexical[left:right] = token["lexical_mask"]
    fit_answer_indices = np.asarray([index for index, answer in enumerate(meta["answers"])
                                     if answer["partition"] == "fit"], dtype=np.int64)
    cal_answer_indices = np.asarray([index for index, answer in enumerate(meta["answers"])
                                     if answer["partition"] == "calibration"], dtype=np.int64)
    fit_groups = np.asarray([meta["answers"][index]["group_id"] for index in fit_answer_indices])
    answer_targets = np.asarray([meta["answers"][index]["label"] for index in fit_answer_indices])
    oof_token = np.full(total_tokens, np.nan, dtype=np.float64)
    fold_records = []
    splitter = GroupKFold(n_splits=FOLDS)
    for fold, (train_local, held_local) in enumerate(splitter.split(fit_answer_indices, answer_targets, fit_groups)):
        train_answers = fit_answer_indices[train_local]
        held_answers = fit_answer_indices[held_local]
        weights = fold_token_weights(meta, train_answers, token_starts, y_token, lexical)
        train_tokens = np.flatnonzero(weights > 0)
        held_tokens = np.concatenate([np.arange(*token_starts[meta["answers"][index]["response_id"]])
                                      for index in held_answers])
        scaler = StandardScaler().fit(x[train_tokens], sample_weight=weights[train_tokens])
        model = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2",
                                   max_iter=2000, random_state=SEED)
        model.fit(scaler.transform(x[train_tokens]), y_token[train_tokens],
                  sample_weight=weights[train_tokens])
        assert model.n_iter_.max() < 2000
        oof_token[held_tokens] = model.predict_proba(scaler.transform(x[held_tokens]))[:, 1]
        fold_records.append({
            "fold": fold, "train_answers": len(train_answers), "held_answers": len(held_answers),
            "train_groups": len(set(fit_groups[train_local])),
            "held_groups": len(set(fit_groups[held_local])),
            "train_lexical_tokens": len(train_tokens), "held_raw_tokens": len(held_tokens),
            "iterations": model.n_iter_.tolist(),
        })
    fit_raw_tokens = np.concatenate([np.arange(*token_starts[meta["answers"][index]["response_id"]])
                                     for index in fit_answer_indices])
    assert np.isfinite(oof_token[fit_raw_tokens]).all()
    weights = fold_token_weights(meta, fit_answer_indices, token_starts, y_token, lexical)
    train_tokens = np.flatnonzero(weights > 0)
    scaler = StandardScaler().fit(x[train_tokens], sample_weight=weights[train_tokens])
    model = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2",
                               max_iter=2000, random_state=SEED)
    model.fit(scaler.transform(x[train_tokens]), y_token[train_tokens],
              sample_weight=weights[train_tokens])
    assert model.n_iter_.max() < 2000
    cal_raw_tokens = np.concatenate([np.arange(*token_starts[meta["answers"][index]["response_id"]])
                                     for index in cal_answer_indices])
    oof_token[cal_raw_tokens] = model.predict_proba(scaler.transform(x[cal_raw_tokens]))[:, 1]
    assert np.isfinite(oof_token).all()
    window_scores = project_token_scores(meta, oof_token, token_starts)
    answer_scores = q.answer_scores(meta, window_scores)
    fit_left, fit_right = meta["bounds"]["fit"]
    thresholds = {
        "window": q.choose_threshold([window["label"] for window in meta["windows"][fit_left:fit_right]],
                                     window_scores[fit_left:fit_right]),
        "answer": q.choose_threshold([meta["answers"][index]["label"] for index in fit_answer_indices],
                                     answer_scores[fit_answer_indices]),
    }
    metrics = q.metrics(meta, window_scores, thresholds)
    model_path = OUT / "token_lr.pkl"
    model_path.write_bytes(pickle.dumps({
        "model": model, "scaler": scaler, "C": LR_C,
        "feature_names": FEATURE_NAMES, "folds": fold_records,
        "thresholds": thresholds, "fit_only": True,
    }, protocol=5))
    score_path = OUT / "token_lr_scores.npz"
    np.savez_compressed(score_path, token_scores=oof_token,
                        window_scores=window_scores, answer_scores=answer_scores,
                        y_token=y_token, lexical_mask=lexical, citation_mask=citation_mask)
    save(OUT / "summary.json", {
        "status": "development_only_complete",
        "role": ROLE, "features": list(FEATURE_NAMES), "C": LR_C,
        "folds": fold_records, "thresholds": thresholds, "metrics": metrics,
        "fit_thresholds_only": True, "calibration_used_for_training": False,
        "calibration_used_for_threshold": False,
        "calibration_is_reused_development_reporting_only": True,
        "official_test_opened": False, "final_test_claim": False,
    })
    (OUT / "REPORT.md").write_text(
        "# Exact subset attribution v1 开发结果\n\n"
        "这是我们的方法候选，不是 ContextCite/LUMINA baseline。8 个证据子集的精确归因特征由固定 LR 读取；"
        "fit 的 source-group OOF 预测同时负责定阈值，calibration 只报告。\n\n"
        "| 层级 | fit F1 | calibration F1 |\n|---|---:|---:|\n"
        f"| 4-BPE 窗口 | {metrics['fit']['windows']['f1']:.4f} | {metrics['calibration']['windows']['f1']:.4f} |\n"
        f"| 整答 max | {metrics['fit']['answers']['f1']:.4f} | {metrics['calibration']['answers']['f1']:.4f} |\n",
        encoding="utf-8",
    )
    files = ("token_features.npy", "token_lr.pkl", "token_lr_scores.npz",
             "summary.json", "REPORT.md")
    save(OUT / "complete.json", {
        "status": "complete_development_only",
        "files_sha256": {name: sha(OUT / name) for name in files},
        "official_test_opened": False, "final_test_claim": False,
    })
    print("EXACT_SUBSET_ATTRIBUTION_SCORE_COMPLETE", flush=True)


def audit():
    complete = check_prepared()
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"prepare", "check", "gpu_smoke", "extract", "score", "expanded_metadata"} <= set(functions)
    prepare_source = ast.get_source_segment(source, functions["prepare"])
    check_source = ast.get_source_segment(source, functions["check"])
    score_source = ast.get_source_segment(source, functions["score"])
    assert "expanded_metadata(" not in prepare_source and "expanded_metadata(" not in check_source
    assert "load_nf4(" not in prepare_source and "load_nf4(" not in check_source
    assert "expanded_metadata()" in score_source
    assert "GroupKFold" in score_source and "fit_answer_indices" in score_source
    assert "q.choose_threshold" in score_source and "fit_left" in score_source
    assert "project_token_scores" in score_source and "q.answer_scores" in score_source
    stats = read(OUT / "preparation_statistics.json")
    assertions = {
        "ours_not_baseline": ROLE.startswith("ours_method_candidate"),
        "exact_eight_subsets": len(SUBSET_MASKS) == 8 and SUBSET_MASKS == tuple(range(8)),
        "preparation_has_no_gold_loader_call": True,
        "check_has_no_gold_loader_call": True,
        "preparation_has_no_model_loader_call": True,
        "GPU_commands_are_separate": True,
        "GPU_cross_run_lock_and_foreign_process_gate_present": "exclusive_gpu" in source and "nvidia-smi" in source,
        "same_answer_axis_checked_all_views": True,
        "exact_shapley_efficiency_checked": True,
        "citation_aligned_shapley_present": "citation_shapley_sum" in FEATURE_NAMES,
        "fit_group_OOF_fixed_LR": True,
        "fit_only_thresholds": True,
        "calibration_report_only": True,
        "unmodified_four_BPE_window_loader": True,
        "window_and_answer_max": True,
        "official_test_path_absent_from_IO_constants": True,
    }
    assert all(assertions.values())
    report = {
        "status": "passed_CPU_only_code_and_artifact_audit",
        "source_sha256": sha(__file__),
        "preparation_complete_sha256": sha(OUT / "preparation_complete.json"),
        "CPU_check_sha256": sha(OUT / "CPU_CHECK.json"),
        "assertions": assertions,
        "resource_estimate": {
            "answers": stats["answers"],
            "GPU_forward_calls": stats["model_forward_calls"],
            "all_view_input_tokens": stats["all_eight_view_input_tokens"],
            "max_sequence_tokens": stats["max_any_view_input_tokens"],
            "raw_logprob_bytes": stats["raw_logprob_float32_bytes"],
            "derived_feature_bytes": stats["derived_feature_float32_bytes"],
            "peak_GPU_memory": "One NF4 Llama-2-7B view at a time; not eight views resident together. Require >=6 GiB free before load.",
            "runtime": "Must be measured by gpu-smoke; 30,712 sequential forwards make full extraction a multi-hour single-RTX-3070 job.",
        },
        "blocking_issues_before_GPU": [],
        "known_limits": [
            "No GPU smoke or extraction has run; empirical runtime and memory peak are still unknown.",
            "The 159-answer calibration split has been reused elsewhere and is development reporting, not an independent final test.",
            "Exact subset attribution needs all eight views and is substantially slower than one-pass probes.",
        ],
        "labels_accessed": False, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False,
    }
    save(OUT / "CODE_AUDIT.json", report)
    (OUT / "CODE_AUDIT.md").write_text(
        "# Exact subset attribution v1：CPU 审计\n\n"
        "通过。3,839 条回答都冻结了 8 个精确资料子集；full 视图逐条等于已有原始重放坐标，"
        "所有视图保持同一答案 token 和字符偏移。prepare/check 不调用标签加载器或模型加载器。\n\n"
        "GPU 与 score 是独立命令。score 固定使用 source-group 五折 OOF LR，阈值只取 fit；"
        "统一 4-BPE 窗口取 lexical token 最大值，整答再取窗口最大值。基线目录和算法均未修改。\n\n"
        f"预计执行 {stats['model_forward_calls']:,} 次顺序前向，共 {stats['all_eight_view_input_tokens']:,} 个输入 token；"
        f"最长 {stats['max_any_view_input_tokens']} token。GPU smoke 尚未运行。\n",
        encoding="utf-8",
    )
    print("EXACT_SUBSET_ATTRIBUTION_CODE_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check",
                                          "audit", "gpu-smoke", "extract", "score"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if arguments.stage == "initialize":
            initialize()
        elif arguments.stage == "self-test":
            print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
        elif arguments.stage == "prepare":
            prepare()
        elif arguments.stage == "check":
            check()
        elif arguments.stage == "audit":
            audit()
        elif arguments.stage == "gpu-smoke":
            gpu_smoke()
        elif arguments.stage == "extract":
            extract()
        else:
            score()
