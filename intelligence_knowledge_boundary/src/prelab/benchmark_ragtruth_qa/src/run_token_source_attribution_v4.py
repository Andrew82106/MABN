"""Token-level source-attribution replay for native RAGTruth QA.

The old semantic-source-attribution cache averages all lexical query BPEs in a
claim before writing a record.  This v4 runner keeps every lexical answer BPE
separate.  It is deliberately label blind: only frozen replay plans and exact
source/claim geometry are opened here.

CPU stages
----------
prepare
    Revalidate every source/claim coordinate and freeze the per-token index.
cpu-check
    Compare the bounded Q/K/V hook with an explicit dense-attention oracle on a
    tiny random CPU Llama, including a future-token counterfactual.
status
    Report committed extraction coverage without loading a pretrained model.

GPU stage (never invoked implicitly)
------------------------------------
extract
    Teacher-force the same frozen Llama-2-7b-chat NF4 replay and atomically save
    one resumable record per answer.

For query q and key k in head h, the frozen primitive is
``softmax(QK^T)(q,k,h) * ||V(k,h)||_2``.  Fragment aggregation remains a maximum
over keys, independently per head, matching semantic-source-attribution v1.
There is no query-token averaging.  The cache stores only head-averaged region
shares [token, layer, 3] and normalized source-sentence shares [token, layer, sentence]
as float16; no full attention or token x sentence x head tensor is stored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import feature_qa as fq  # noqa: E402
import run_feature_qa as llama_runner  # noqa: E402


VERSION = "token-source-attribution-v4"
LAYERS = 32
HEADS = 32
QUERY_BATCH = 8
THREADS = 4
SEED = 20260913
GLOBAL_GPU_LOCK = ROOT / "results/.exclusive_gpu_runner.lock"
FORBIDDEN_ANNOTATION_KEYS = {
    "label", "labels", "gold", "risk", "risk_mask", "original_labels",
    "hallucination", "hallucination_label", "token_labels",
    "risk_token_indices", "risk_character_spans",
}

SCOPES = {
    "native": {
        "out": ROOT / "results/token_source_attribution_v4",
        "plans": ROOT / "data/feature_preparation/plans.jsonl",
        "layouts": ROOT / "results/semantic_source_attribution_v1/layouts.jsonl",
        "answers": 793,
        "partitions": {"fit": 634, "calibration": 159},
        "claims": 11322,
        "lexical_bpe": 174518,
        "claim_sentence_edges": 166644,
        "token_sentence_pairs": 2535131,
    },
    # The formula and record schema are input-agnostic.  This optional scope is
    # ready for the already prepared 3,046 extra fit layouts, but native remains
    # the only scoring target in v4.
    "expanded-fit": {
        "out": ROOT / "results/token_source_attribution_v4_expanded_fit",
        "plans": ROOT / "fit_expansion/data/new_token_plans.jsonl",
        "layouts": ROOT / "results/semantic_source_attribution_expanded_fit_v1/layouts.jsonl",
        "answers": 3046,
        "partitions": {"fit": 3046},
        "claims": 25864,
        "lexical_bpe": 420782,
        "claim_sentence_edges": 390683,
        "token_sentence_pairs": 6298663,
    },
}


def cfg(scope: str) -> dict:
    result = dict(SCOPES[scope])
    result["scope"] = scope
    return result


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, ("Frozen JSON changed", str(path))
    else:
        atomic_json(path, value)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "CPU stage must not initialize CUDA"


def reject_annotation_keys(value) -> None:
    if isinstance(value, dict):
        found = FORBIDDEN_ANNOTATION_KEYS & set(value)
        assert not found, f"Annotation-bearing keys are forbidden: {sorted(found)}"
        for key, child in value.items():
            if key != "labels_used":
                reject_annotation_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_annotation_keys(child)


def software_signature() -> dict:
    versions = {}
    for package in ("torch", "transformers", "numpy", "bitsandbytes", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions,
            "torch_cuda_runtime": torch.version.cuda}


def model_asset_binding() -> dict:
    manifest_path = ROOT / "model_download_manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["status"] == "complete"
    assert manifest["repo_id"] == llama_runner.REPO
    assert manifest["revision"] == llama_runner.REVISION
    assets = {}
    for item in manifest["files"]:
        assert item["status"] == "verified" and item["source_hash_match"] is True
        path = fq.MODEL / item["filename"]
        assert path.is_file() and path.stat().st_size == item["actual_bytes"]
        assets[item["filename"]] = {
            "bytes": int(item["actual_bytes"]), "sha256": item["actual_sha256"]}
    config = read_json(fq.MODEL / "config.json")
    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == LAYERS
    assert config["num_attention_heads"] == HEADS
    assert config["hidden_size"] == 4096
    assert config["hidden_size"] // HEADS == 128
    return {"manifest_sha256": sha(manifest_path),
            "repo_id": manifest["repo_id"], "revision": manifest["revision"],
            "assets": assets}


def protocol(scope: str) -> dict:
    return {
        "version": VERSION, "scope": scope,
        "role": "ours token-level source-attribution extractor; formal baselines untouched",
        "data": {
            "native": "Frozen native QA fit634 plus calibration159 replay plans.",
            "expanded": "Optional same-schema extra fit replay layouts; scoring is not enabled in v4.",
            "official_test": "No official-test or withheld-data path occurs in this runner.",
        },
        "coordinates": {
            "source": "Every frozen rendered source-sentence span is re-intersected with original full-string tokenizer offsets; nonspace and alphanumeric coverage are asserted.",
            "answer": "Every lexical answer BPE is owned exactly once by a frozen claim and retains local plus absolute coordinates.",
            "query": "Post-token query at the original answer-token position; causal mask is rebuilt explicitly.",
            "previous_answer": "Only original answer positions strictly smaller than q; q itself is excluded.",
            "other": "All pre-answer positions outside the exact source-sentence union.",
        },
        "primitive": "float32 softmax attention(q,k) multiplied by the corresponding repeated V-head L2 norm",
        "aggregation": {
            "fragment": "Maximum over fragment key tokens independently per head, unchanged from semantic-source-attribution v1.",
            "query": "No claim/query averaging. Every lexical query BPE is an output row.",
            "heads": "Arithmetic mean across all heads after fragment aggregation.",
            "region_share": "Per layer normalize head-mean source, strict-previous-answer, and other-context relevance to sum one.",
            "sentence": "Per token/layer source-sentence relevance normalized across source sentences before float16 storage; an exact zero-source-mass row remains all zero.",
        },
        "raw_outputs": {
            "token_region_share": "float16 [lexical_BPE,32,3]",
            "token_sentence_share": "float16 [lexical_BPE,32,source_sentences_in_answer], each row sums one or is exactly zero",
            "coordinates": "query local/absolute position, token id, unique claim id, exact sentence identity",
            "forbidden": "No full attention matrix and no token x sentence x head cache.",
        },
        "downstream_contract": {
            "token_features": "Per layer: source/previous/other share, all-sentence attention-weighted frozen NLI E/N/C, top1 and top3 sentence concentration.",
            "window_supervision": "Unchanged eligible 4-raw-BPE stride-one windows are the training rows; no claim-score projection.",
            "optional_score_inputs": "Frozen generation NLL and independently fit-only upstream probabilities may be appended only in the scorer; scorer v4 excludes historically calibration-selected Lookback/large columns.",
            "split": "Source-group OOF on fit; freeze model and thresholds before one calibration evaluation; official test sealed.",
        },
        "model": {
            "checkpoint": llama_runner.REPO, "revision": llama_runner.REVISION,
            "load_config": llama_runner.LOAD_CONFIG,
            "teacher_forced": True, "trained_here": False,
        },
        "stage_gate": "prepare -> cpu-check -> explicit extract -> status; no training or labels in this runner",
        "labels_used": False, "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def mapped_span(rendered: str, offsets: np.ndarray, left: int, right: int) -> dict:
    assert 0 <= left < right <= len(rendered)
    positions = np.flatnonzero(
        (offsets[:, 1] > left) & (offsets[:, 0] < right) &
        (offsets[:, 1] > offsets[:, 0])).astype(np.int64)
    assert len(positions)
    coverage = np.zeros(right - left, dtype=bool)
    lexical_coverage = np.zeros(right - left, dtype=bool)
    intersections, lexical = [], []
    for position in positions:
        token_left, token_right = map(int, offsets[position])
        begin, end = max(token_left, left), min(token_right, right)
        assert begin < end
        intersections.append([begin, end])
        coverage[begin - left:end - left] = True
        if any(char.isalnum() for char in rendered[begin:end]):
            lexical.append(int(position))
            lexical_coverage[begin - left:end - left] = True
    nonspace = np.fromiter((not char.isspace() for char in rendered[left:right]), bool)
    alnum = np.fromiter((char.isalnum() for char in rendered[left:right]), bool)
    assert np.all(coverage | ~nonspace)
    assert lexical and np.all(lexical_coverage | ~alnum)
    return {"token_positions": positions.tolist(),
            "lexical_token_positions": lexical,
            "token_character_intersections": intersections,
            "boundary_crossing_token_positions": [
                int(position) for position in positions
                if int(offsets[position, 0]) < left or int(offsets[position, 1]) > right]}


def validate_layout(plan: dict, layout: dict) -> dict[str, np.ndarray | int]:
    """Rebuild all coordinates without opening annotation-bearing files."""
    reject_annotation_keys({key: value for key, value in plan.items()
                            if key != "labels_used"})
    reject_annotation_keys({key: value for key, value in layout.items()
                            if key != "labels_used"})
    assert plan["labels_used"] is False and layout["labels_used"] is False
    assert plan["official_split"] == "train"
    for key in ("response_id", "source_id", "group_id", "partition"):
        assert plan[key] == layout[key]
    assert layout["plan_sha256"] == digest(plan)
    assert digest(plan["original_response"]) == plan["answer_sha256"]
    view = plan["original"]
    rendered = fq.WRAPPER_LEFT + plan["released_prompt"] + fq.WRAPPER_RIGHT + plan["original_response"]
    assert digest(rendered) == view["rendered_text_sha256"]
    offsets = np.asarray(view["input_token_offsets"], dtype=np.int64)
    assert offsets.shape == (len(view["input_ids"]), 2)
    assert len(view["input_ids"]) == int(layout["sequence_length"])
    answer_positions = np.asarray(view["answer_token_positions"], dtype=np.int64)
    answer_ids = np.asarray(view["answer_token_ids"], dtype=np.int64)
    answer_offsets = np.asarray(view["response_token_offsets"], dtype=np.int64)
    assert np.array_equal(np.asarray(view["input_ids"])[answer_positions], answer_ids)
    assert np.array_equal(answer_positions, layout["answer_token_positions"])
    assert np.all(np.diff(answer_positions) == 1)
    assert answer_offsets.shape == (len(answer_positions), 2)

    sentence_union = set()
    passage_union = [set(), set(), set()]
    boundary = 0
    for expected_index, sentence in enumerate(layout["sentences"]):
        assert int(sentence["sentence_index"]) == expected_index
        left, right = map(int, sentence["rendered_character_range"])
        assert digest(rendered[left:right]) == sentence["text_sha256"]
        rebuilt = mapped_span(rendered, offsets, left, right)
        for key in ("token_positions", "lexical_token_positions",
                    "token_character_intersections", "boundary_crossing_token_positions"):
            assert rebuilt[key] == sentence[key], (plan["response_id"], expected_index, key)
        assert max(rebuilt["token_positions"]) < int(answer_positions[0])
        sentence_union.update(rebuilt["token_positions"])
        passage_id = int(sentence["passage_id"])
        assert 1 <= passage_id <= 3
        passage_union[passage_id - 1].update(rebuilt["token_positions"])
        boundary += bool(rebuilt["boundary_crossing_token_positions"])
    assert sorted(sentence_union) == layout["source_sentence_union_token_positions"]
    assert [sorted(values) for values in passage_union] == layout["passage_token_positions"]
    prefix = set(range(int(answer_positions[0])))
    assert sorted(prefix - sentence_union) == layout["other_context_token_positions"]
    assert not (sentence_union & set(layout["other_context_token_positions"]))

    owner = np.full(len(answer_positions), -1, dtype=np.int32)
    query_local, query_absolute, query_claim = [], [], []
    answer = plan["original_response"]
    for expected_claim, claim in enumerate(layout["claims"]):
        assert int(claim["claim_id"]) == expected_claim
        left, right = map(int, claim["claim_character_range"])
        assert digest(answer[left:right]) == claim["claim_text_sha256"]
        local = np.asarray(claim["lexical_answer_token_indices"], dtype=np.int64)
        absolute = np.asarray(claim["lexical_absolute_token_positions"], dtype=np.int64)
        assert len(local) and np.all(np.diff(local) > 0)
        assert np.array_equal(answer_positions[local], absolute)
        assert np.all(owner[local] == -1), "A lexical BPE must have exactly one claim owner"
        owner[local] = expected_claim
        rebuilt_intersections = []
        for token_index in local:
            token_left, token_right = map(int, answer_offsets[token_index])
            begin, end = max(token_left, left), min(token_right, right)
            assert begin < end and any(char.isalnum() for char in answer[begin:end])
            rebuilt_intersections.append([begin, end])
        assert rebuilt_intersections == claim["claim_character_intersections"]
        query_local.extend(local.tolist())
        query_absolute.extend(absolute.tolist())
        query_claim.extend([expected_claim] * len(local))
    order = np.argsort(query_local, kind="stable")
    query_local = np.asarray(query_local, dtype=np.int32)[order]
    query_absolute = np.asarray(query_absolute, dtype=np.int64)[order]
    query_claim = np.asarray(query_claim, dtype=np.int32)[order]
    assert np.all(np.diff(query_local) > 0)
    lexical = np.asarray([
        any(char.isalnum() for char in answer[max(0, int(left)):int(right)])
        for left, right in answer_offsets], dtype=bool)
    assert np.array_equal(owner >= 0, lexical)
    assert np.array_equal(owner[query_local], query_claim)
    assert np.array_equal(answer_positions[query_local], query_absolute)
    return {"query_local_answer_indices": query_local,
            "query_absolute_positions": query_absolute,
            "query_claim_ids": query_claim,
            "query_token_ids": answer_ids[query_local].astype(np.int64),
            "sentences": len(layout["sentences"]),
            "boundary_sentences": boundary}


def prepare(scope: str) -> dict:
    assert_cpu_only()
    config = cfg(scope)
    plans = json_lines(config["plans"])
    layouts = json_lines(config["layouts"])
    assert len(plans) == len(layouts) == config["answers"]
    assert [row["response_id"] for row in plans] == [row["response_id"] for row in layouts]
    assert Counter(row["partition"] for row in plans) == config["partitions"]
    response_offsets = [0]
    claim_offsets = [0]
    query_response_indices, query_local, query_absolute = [], [], []
    query_claim_local, query_claim_global, query_token_ids = [], [], []
    answer_token_counts, sentence_counts = [], []
    token_sentence_pairs = 0
    boundary_sentences = 0
    for response_index, (plan, layout) in enumerate(zip(plans, layouts)):
        checked = validate_layout(plan, layout)
        q_count = len(checked["query_local_answer_indices"])
        c_count = len(layout["claims"])
        s_count = int(checked["sentences"])
        query_response_indices.extend([response_index] * q_count)
        query_local.extend(checked["query_local_answer_indices"])
        query_absolute.extend(checked["query_absolute_positions"])
        query_claim_local.extend(checked["query_claim_ids"])
        query_claim_global.extend((claim_offsets[-1] + checked["query_claim_ids"]).tolist())
        query_token_ids.extend(checked["query_token_ids"])
        response_offsets.append(response_offsets[-1] + q_count)
        claim_offsets.append(claim_offsets[-1] + c_count)
        answer_token_counts.append(len(layout["answer_token_positions"]))
        sentence_counts.append(s_count)
        token_sentence_pairs += q_count * s_count
        boundary_sentences += int(checked["boundary_sentences"])
        if (response_index + 1) % 100 == 0:
            print("TOKEN_ATTRIBUTION_COORDINATES", response_index + 1,
                  config["answers"], flush=True)
    assert response_offsets[-1] == config["lexical_bpe"]
    assert claim_offsets[-1] == config["claims"]
    assert token_sentence_pairs == config["token_sentence_pairs"]
    assert sum(len(row["claims"]) * len(row["sentences"]) for row in layouts) == config["claim_sentence_edges"]

    output = config["out"]
    output.mkdir(parents=True, exist_ok=True)
    frozen_json(output / "protocol.json", protocol(scope))
    index = {
        "response_token_indptr": np.asarray(response_offsets, dtype=np.int64),
        "claim_indptr_by_response": np.asarray(claim_offsets, dtype=np.int64),
        "query_response_indices": np.asarray(query_response_indices, dtype=np.int32),
        "query_local_answer_indices": np.asarray(query_local, dtype=np.int32),
        "query_absolute_positions": np.asarray(query_absolute, dtype=np.int32),
        "query_claim_local_ids": np.asarray(query_claim_local, dtype=np.int32),
        "query_claim_global_ids": np.asarray(query_claim_global, dtype=np.int32),
        "query_token_ids": np.asarray(query_token_ids, dtype=np.int32),
        "answer_token_counts": np.asarray(answer_token_counts, dtype=np.int32),
        "sentence_counts": np.asarray(sentence_counts, dtype=np.int16),
    }
    index_path = output / "token_index.npz"
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(index)
            assert all(np.array_equal(loaded[key], value) for key, value in index.items())
    else:
        atomic_npz(index_path, index)
    raw_sentence_bytes = token_sentence_pairs * LAYERS * 2
    raw_region_bytes = config["lexical_bpe"] * LAYERS * 3 * 2
    raw_coordinate_bytes = sum(value.nbytes for value in index.values())
    estimate = {
        "scope": scope, "answers": config["answers"],
        "lexical_answer_BPE": config["lexical_bpe"],
        "claims": config["claims"],
        "claim_sentence_edges": config["claim_sentence_edges"],
        "token_sentence_pairs": token_sentence_pairs,
        "raw_array_bytes": {
            "token_sentence_share_float16": raw_sentence_bytes,
            "token_region_share_float16": raw_region_bytes,
            "global_coordinates": raw_coordinate_bytes,
            "total": raw_sentence_bytes + raw_region_bytes + raw_coordinate_bytes,
        },
        "compressed_npz_planning_range_bytes": [
            int((raw_sentence_bytes + raw_region_bytes) * .5),
            int((raw_sentence_bytes + raw_region_bytes) * .95)],
        "gpu_time_native_minutes": [20, 35] if scope == "native" else [45, 85],
        "basis": "Same 793-answer v1 replay measured 22.9 record-minutes total; v4 has the same forward pass and removes claim membership matmuls but writes more token rows.",
        "range_is_planning_not_statistical": True,
    }
    frozen_json(output / "RESOURCE_ESTIMATE.json", estimate)
    signature = {
        "version": VERSION, "scope": scope,
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(output / "protocol.json"),
        "plans_sha256": sha(config["plans"]),
        "layouts_sha256": sha(config["layouts"]),
        "token_index_sha256": sha(index_path),
        "model": model_asset_binding(), "load_config": llama_runner.LOAD_CONFIG,
        "software": software_signature(),
        "response_ids_in_order": [row["response_id"] for row in layouts],
        "labels_used": False, "official_test_opened": False,
    }
    frozen_json(output / "signature.json", signature)
    plan_text = (
        "# Token source attribution v4\n\n"
        "目标：删除 claim 内 query-token 平均。每个 lexical answer BPE 独立保留 32 层区域 share 与逐来源句归因；评分层再与冻结 claim-sentence NLI 的 E/N/C 合并。\n\n"
        "- 白盒公式：`attention(q,k) * ||V_head(k)||_2`，fragment 内逐 head 取 max；head 最后求均值。\n"
        "- 因果：query 是当前 token 的 post-token state；previous answer 严格 `< q`；未来 key 显式 mask。\n"
        "- 存储：不保存完整 attention，也不保存 token×sentence×head；native 原始数组约 189 MiB。\n"
        "- 评分：逐 token 每层 8 维；四个原始 BPE 直接汇总成窗口样本，source-group OOF；正式 v4 只追加 label-blind NLL，历史 cal-selected Lookback/large 禁入候选。\n"
        "- 封存：抽取不读标签；模型与阈值只由 fit OOF 冻结；cal 只评一次；official test 不打开；baseline 不修改。\n"
        "- 扩展：同一 runner 支持 `--scope expanded-fit` 的独立缓存；v4 首轮评分仅 native634+cal159。\n")
    plan_path = output / "PLAN.md"
    if plan_path.exists():
        assert plan_path.read_text(encoding="utf-8") == plan_text
    else:
        plan_path.write_text(plan_text, encoding="utf-8")
    report = {
        "status": "prepared_waiting_for_cpu_check",
        "scope": scope, "answers": config["answers"],
        "partitions": config["partitions"], "claims": config["claims"],
        "lexical_answer_BPE": config["lexical_bpe"],
        "token_sentence_pairs": token_sentence_pairs,
        "source_sentences_with_boundary_crossing_tokens": boundary_sentences,
        "all_source_coordinates_rebuilt": True,
        "all_lexical_answer_BPE_owned_exactly_once": True,
        "query_token_averaging": False,
        "strict_previous_answer_excludes_query": True,
        "raw_array_bytes": estimate["raw_array_bytes"],
        "signature_sha256": digest(signature),
        "gpu_started": False, "model_loaded": False,
        "labels_used": False, "official_test_opened": False,
        "formal_baselines_modified": False,
    }
    frozen_json(output / "PREPARATION.json", report)
    assert_cpu_only()
    print("TOKEN_ATTRIBUTION_V4_PREPARED", scope, config["lexical_bpe"], flush=True)
    return report


def load_prepared(scope: str):
    config = cfg(scope)
    output = config["out"]
    signature = read_json(output / "signature.json")
    preparation = read_json(output / "PREPARATION.json")
    assert signature["runner_sha256"] == sha(Path(__file__))
    assert signature["protocol_sha256"] == sha(output / "protocol.json")
    assert signature["plans_sha256"] == sha(config["plans"])
    assert signature["layouts_sha256"] == sha(config["layouts"])
    assert signature["token_index_sha256"] == sha(output / "token_index.npz")
    assert preparation["signature_sha256"] == digest(signature)
    plans, layouts = json_lines(config["plans"]), json_lines(config["layouts"])
    assert [row["response_id"] for row in plans] == signature["response_ids_in_order"]
    assert [row["response_id"] for row in layouts] == signature["response_ids_in_order"]
    return config, plans, layouts, signature, preparation


def unique_sorted(values, sequence_length: int, name: str) -> list[int]:
    result = sorted(set(map(int, values)))
    assert result and result[0] >= 0 and result[-1] < sequence_length, name
    return result


def query_geometry(layout: dict) -> tuple[list[int], list[int], list[int]]:
    triples = []
    for claim in layout["claims"]:
        for local, absolute in zip(claim["lexical_answer_token_indices"],
                                   claim["lexical_absolute_token_positions"]):
            triples.append((int(local), int(absolute), int(claim["claim_id"])))
    triples.sort()
    assert len(triples) == len({item[0] for item in triples})
    return ([item[0] for item in triples], [item[1] for item in triples],
            [item[2] for item in triples])


class TokenAttributionHooks:
    """Bounded per-token Q/K/V reconstruction with no claim averaging."""

    def __init__(self, model, layout: dict, query_batch: int = QUERY_BATCH):
        self.model = model
        self.spec = fq.architecture(model)
        self.device = model.model.embed_tokens.weight.device
        self.sequence_length = int(layout["sequence_length"])
        self.answer_positions = unique_sorted(
            layout["answer_token_positions"], self.sequence_length, "answer")
        self.source_positions = unique_sorted(
            layout["source_sentence_union_token_positions"], self.sequence_length, "source")
        self.other_positions = unique_sorted(
            layout["other_context_token_positions"], self.sequence_length, "other")
        self.sentence_positions = [
            unique_sorted(sentence["token_positions"], self.sequence_length,
                          f"sentence {sentence['sentence_index']}")
            for sentence in layout["sentences"]]
        local, absolute, claim = query_geometry(layout)
        self.query_local_answer_indices = local
        self.query_positions = unique_sorted(absolute, self.sequence_length, "query")
        self.query_claim_ids = claim
        assert len(self.query_positions) == len(absolute)
        assert self.query_positions == absolute
        assert set(self.query_positions) <= set(self.answer_positions)
        assert max(self.source_positions + self.other_positions) < min(self.answer_positions)
        assert not (set(self.source_positions) & set(self.other_positions))
        self.query_batch = int(query_batch)
        assert self.query_batch > 0
        q, l, s = len(self.query_positions), self.spec["layers"], len(self.sentence_positions)
        self.region_share = np.empty((q, l, 3), dtype=np.float32)
        self.sentence_share = np.empty((q, l, s), dtype=np.float32)
        self.zero_sentence_mass_rows = 0
        self.future_attention_max = 0.0
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
                assert layer in self.pending and "q" not in self.pending[layer]
                self.pending[layer]["q"] = output

            def key(module, args, output, layer=layer):
                assert layer in self.pending and "k" not in self.pending[layer]
                self.pending[layer]["k"] = output

            def value(module, args, output, layer=layer, attention=attention):
                self._capture(layer, attention, output)

            self.handles.extend((
                attention.register_forward_pre_hook(before, with_kwargs=True),
                attention.q_proj.register_forward_hook(query),
                attention.k_proj.register_forward_hook(key),
                attention.v_proj.register_forward_hook(value),
            ))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()

    @staticmethod
    def fragment(contribution: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return query x head after key-token max."""
        return contribution.index_select(-1, positions).amax(-1).transpose(0, 1)

    def _capture(self, layer: int, module, values: torch.Tensor) -> None:
        state = self.pending.pop(layer)
        assert set(state) == {"rope", "q", "k"}
        heads, kv_heads, dim = self.spec["heads"], self.spec["kv_heads"], self.spec["head_dim"]
        assert values.shape[:2] == state["q"].shape[:2] == state["k"].shape[:2] == (1, self.sequence_length)
        query = state["q"].view(1, -1, heads, dim).transpose(1, 2)
        key = state["k"].view(1, -1, kv_heads, dim).transpose(1, 2)
        value = values.view(1, -1, kv_heads, dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *state["rope"])
        key = repeat_kv(key, heads // kv_heads)[0]
        value = repeat_kv(value, heads // kv_heads)[0]
        query = query[0]
        value_norm = value.float().norm(p=2, dim=-1)
        absolute = torch.arange(self.sequence_length, device=self.device)
        queries = torch.tensor(self.query_positions, dtype=torch.long, device=self.device)
        answer = torch.tensor(self.answer_positions, dtype=torch.long, device=self.device)
        source = torch.tensor(self.source_positions, dtype=torch.long, device=self.device)
        other = torch.tensor(self.other_positions, dtype=torch.long, device=self.device)
        sentences = [torch.tensor(values, dtype=torch.long, device=self.device)
                     for values in self.sentence_positions]
        for begin in range(0, len(self.query_positions), self.query_batch):
            end = min(begin + self.query_batch, len(self.query_positions))
            positions = queries[begin:end]
            one_query = query.index_select(1, positions)
            logits = (one_query @ key.transpose(-2, -1)) * module.scaling
            future = absolute[None, None, :] > positions[None, :, None]
            logits.masked_fill_(future, float("-inf"))
            attention = logits.float().softmax(-1)
            if future.any():
                self.future_attention_max = max(
                    self.future_attention_max,
                    float(attention.masked_select(future.expand_as(attention)).max().item()))
            contribution = attention * value_norm[:, None, :]
            source_values = self.fragment(contribution, source)
            other_values = self.fragment(contribution, other)
            previous_mask = answer[None, :] < positions[:, None]
            selected = contribution.index_select(-1, answer)
            previous_values = selected.masked_fill(~previous_mask[None], float("-inf")).amax(-1).transpose(0, 1)
            previous_values = torch.where(torch.isfinite(previous_values), previous_values,
                                          torch.zeros_like(previous_values))
            region = torch.stack((source_values, previous_values, other_values), dim=1).mean(-1)
            region_total = region.sum(-1, keepdim=True)
            region = torch.where(
                region_total <= 0, torch.zeros_like(region),
                region / region_total.clamp_min(torch.finfo(torch.float32).tiny))
            self.region_share[begin:end, layer] = region.cpu().numpy()
            sentence = torch.stack(
                [self.fragment(contribution, token_positions) for token_positions in sentences],
                dim=1).mean(-1)
            sentence_total = sentence.sum(-1, keepdim=True)
            zero_mass = sentence_total <= 0
            sentence = torch.where(
                zero_mass, torch.zeros_like(sentence),
                sentence / sentence_total.clamp_min(torch.finfo(torch.float32).tiny))
            self.zero_sentence_mass_rows += int(zero_mass.sum().item())
            self.sentence_share[begin:end, layer] = sentence.cpu().numpy()
            del one_query, logits, attention, contribution, selected, previous_values
        assert self.future_attention_max == 0.0
        self.seen.add(layer)

    def finish(self) -> dict[str, np.ndarray]:
        assert len(self.seen) == self.spec["layers"] and not self.pending
        assert np.isfinite(self.region_share).all() and np.isfinite(self.sentence_share).all()
        assert np.all(self.region_share >= 0) and np.all(self.sentence_share >= 0)
        sentence_sum = self.sentence_share.sum(-1, dtype=np.float32)
        assert np.all((sentence_sum == 0) | np.isclose(sentence_sum, 1, atol=2e-6, rtol=0))
        return {
            "token_region_share": self.region_share.astype(np.float16),
            "token_sentence_share": self.sentence_share.astype(np.float16),
            "query_local_answer_indices": np.asarray(self.query_local_answer_indices, dtype=np.int32),
            "query_absolute_positions": np.asarray(self.query_positions, dtype=np.int32),
            "query_claim_ids": np.asarray(self.query_claim_ids, dtype=np.int32),
        }


@torch.inference_mode()
def extract_one(model, plan: dict, layout: dict) -> dict[str, np.ndarray]:
    assert not model.training and plan["response_id"] == layout["response_id"]
    ids, mask, _ = fq.tensor_input(model, plan["original"])
    with TokenAttributionHooks(model, layout) as hooks:
        model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                    output_attentions=False, output_hidden_states=False,
                    return_dict=True)
    arrays = hooks.finish()
    local = arrays["query_local_answer_indices"]
    arrays.update({
        "query_token_ids": np.asarray(plan["original"]["answer_token_ids"], dtype=np.int64)[local],
        "sentence_indices": np.arange(len(layout["sentences"]), dtype=np.int32),
        "sentence_passage_ids": np.asarray([row["passage_id"] for row in layout["sentences"]], dtype=np.int8),
        "sentence_ids": np.asarray([row["sentence_id"] for row in layout["sentences"]], dtype=np.int32),
        "sentence_identity_sha256": np.stack([
            np.frombuffer(bytes.fromhex(row["text_sha256"]), dtype=np.uint8)
            for row in layout["sentences"]]),
    })
    return arrays


def validate_arrays(arrays: dict[str, np.ndarray], plan: dict, layout: dict,
                    layers: int, heads: int) -> dict:
    required = {
        "token_region_share", "token_sentence_share",
        "query_local_answer_indices", "query_absolute_positions",
        "query_claim_ids", "query_token_ids", "sentence_indices",
        "sentence_passage_ids", "sentence_ids", "sentence_identity_sha256"}
    assert set(arrays) == required
    local, absolute, claims = query_geometry(layout)
    q, s = len(local), len(layout["sentences"])
    assert arrays["token_region_share"].shape == (q, layers, 3)
    assert arrays["token_sentence_share"].shape == (q, layers, s)
    assert arrays["token_region_share"].dtype == np.float16
    assert arrays["token_sentence_share"].dtype == np.float16
    assert np.array_equal(arrays["query_local_answer_indices"], local)
    assert np.array_equal(arrays["query_absolute_positions"], absolute)
    assert np.array_equal(arrays["query_claim_ids"], claims)
    expected_ids = np.asarray(plan["original"]["answer_token_ids"], dtype=np.int64)[local]
    assert np.array_equal(arrays["query_token_ids"], expected_ids)
    assert np.array_equal(arrays["sentence_indices"], np.arange(s, dtype=np.int32))
    assert arrays["sentence_identity_sha256"].shape == (s, 32)
    for key in ("token_region_share", "token_sentence_share"):
        assert np.isfinite(arrays[key]).all() and np.all(arrays[key] >= 0)
    region_sum = arrays["token_region_share"].astype(np.float32).sum(-1)
    region_active = region_sum > 0
    share_error = float(np.max(np.abs(region_sum[region_active] - 1))) if region_active.any() else 0.0
    assert share_error <= 0.001 and np.all((region_sum == 0) | region_active), share_error
    sentence_sum = arrays["token_sentence_share"].astype(np.float32).sum(-1)
    sentence_zero = sentence_sum == 0
    assert np.all(sentence_zero | np.isclose(sentence_sum, 1, atol=0.0011, rtol=0))
    raw = arrays["token_region_share"].nbytes + arrays["token_sentence_share"].nbytes
    return {"lexical_answer_BPE": q, "source_sentences": s,
            "token_sentence_pairs": q * s, "raw_float16_bytes": raw,
            "region_share_sum_max_abs_error_after_float16": share_error,
            "zero_source_sentence_mass_token_layers": int(sentence_zero.sum()),
            "heads_averaged_not_stored": heads}


class ValueCapture:
    def __init__(self, model):
        self.values, self.handles = {}, []
        for layer, block in enumerate(model.model.layers):
            self.handles.append(block.self_attn.v_proj.register_forward_hook(
                lambda module, args, output, layer=layer:
                self.values.__setitem__(layer, output.detach().clone())))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def dense_oracle(model, layout: dict, attentions, projected_values) -> dict[str, np.ndarray]:
    """Slow explicit oracle used only by cpu-check."""
    spec = fq.architecture(model)
    _, queries, _ = query_geometry(layout)
    source = layout["source_sentence_union_token_positions"]
    other = layout["other_context_token_positions"]
    answer = layout["answer_token_positions"]
    sentences = [row["token_positions"] for row in layout["sentences"]]
    q, l, h, s = len(queries), spec["layers"], spec["heads"], len(sentences)
    region_share = np.empty((q, l, 3), dtype=np.float32)
    sentence_relevance = np.empty((q, l, s), dtype=np.float32)

    def fragment(contribution, positions):
        return contribution[:, positions].max(axis=-1)

    for layer, native in enumerate(attentions):
        value = projected_values[layer].view(1, -1, spec["kv_heads"],
                                             spec["head_dim"]).transpose(1, 2)
        value = repeat_kv(value, spec["heads"] // spec["kv_heads"])[0]
        norm = value.float().norm(p=2, dim=-1).cpu().numpy()
        for query_index, position in enumerate(queries):
            row = native[0, :, position].float().cpu().numpy()
            assert np.max(row[:, position + 1:]) == 0 if position + 1 < row.shape[1] else True
            contribution = row * norm
            previous = [candidate for candidate in answer if candidate < position]
            values = np.asarray([
                fragment(contribution, source).mean(dtype=np.float32),
                fragment(contribution, previous).mean(dtype=np.float32) if previous else 0.0,
                fragment(contribution, other).mean(dtype=np.float32),
            ], dtype=np.float32)
            region_share[query_index, layer] = values / values.sum(dtype=np.float32)
            for sentence_index, token_positions in enumerate(sentences):
                sentence_relevance[query_index, layer, sentence_index] = fragment(
                    contribution, token_positions).mean(dtype=np.float32)
    total = sentence_relevance.sum(-1, keepdims=True, dtype=np.float32)
    sentence_share = np.divide(sentence_relevance, total,
                               out=np.zeros_like(sentence_relevance), where=total > 0)
    return {"token_region_share": region_share,
            "token_sentence_share": sentence_share}


def combine_token_features(region_share: np.ndarray, sentence_share: np.ndarray,
                           query_claim_ids: np.ndarray,
                           claim_sentence_nli: np.ndarray) -> np.ndarray:
    """Create the eight requested per-layer features without labels."""
    region = np.asarray(region_share, dtype=np.float32)
    sentence = np.asarray(sentence_share, dtype=np.float32)
    owners = np.asarray(query_claim_ids, dtype=np.int64)
    nli = np.asarray(claim_sentence_nli, dtype=np.float32)
    q, layers, source_sentences = sentence.shape
    assert region.shape == (q, layers, 3)
    assert nli.ndim == 3 and nli.shape[1:] == (source_sentences, 3)
    assert owners.shape == (q,) and owners.min() >= 0 and owners.max() < len(nli)
    assert np.isfinite(nli).all() and np.all(nli >= 0)
    assert np.allclose(nli.sum(-1), 1, atol=2e-6, rtol=0)
    total = sentence.sum(-1, keepdims=True, dtype=np.float32)
    assert np.all((total == 0) | np.isclose(total, 1, atol=0.0011, rtol=0))
    weight = np.divide(sentence, total, out=np.zeros_like(sentence), where=total > 0)
    weighted_nli = np.einsum("qls,qsn->qln", weight, nli[owners], optimize=True)
    ordered = np.sort(sentence, axis=-1)[:, :, ::-1]
    top1 = np.divide(ordered[:, :, :1], total,
                     out=np.zeros_like(ordered[:, :, :1]), where=total > 0)
    top3_raw = ordered[:, :, :min(3, source_sentences)].sum(-1, keepdims=True)
    top3 = np.divide(top3_raw, total, out=np.zeros_like(top3_raw), where=total > 0)
    result = np.concatenate((region, weighted_nli, top1, top3), axis=-1)
    assert result.shape == (q, layers, 8) and np.isfinite(result).all()
    region_sum = result[:, :, :3].sum(-1)
    assert np.all((region_sum == 0) | np.isclose(region_sum, 1, atol=0.001, rtol=0))
    nli_sum = result[:, :, 3:6].sum(-1)
    assert np.all((nli_sum == 0) | np.isclose(nli_sum, 1, atol=2e-5, rtol=0))
    return result.astype(np.float32, copy=False)


def cpu_check(scope: str) -> dict:
    assert_cpu_only()
    config, plans, layouts, signature, _ = load_prepared(scope)
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    config_tiny = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0)
    config_tiny._attn_implementation = "eager"
    model = LlamaForCausalLM(config_tiny).cpu().eval()
    ids = [1] + [2 + ((index * 7) % 120) for index in range(47)]
    answer_positions = list(range(18, 48))
    layout = {
        "sequence_length": 48, "answer_token_positions": answer_positions,
        "source_sentence_union_token_positions": [2, 3, 4, 6, 7],
        "other_context_token_positions": [0, 1, 5, 8, 9],
        "sentences": [
            {"sentence_index": 0, "token_positions": [2, 3]},
            {"sentence_index": 1, "token_positions": [4]},
            {"sentence_index": 2, "token_positions": [6, 7]}],
        "claims": [
            {"claim_id": 0, "lexical_answer_token_indices": [0, 2],
             "lexical_absolute_token_positions": [18, 20]},
            {"claim_id": 1, "lexical_answer_token_indices": [3, 5],
             "lexical_absolute_token_positions": [21, 23]}]}
    input_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        with TokenAttributionHooks(model, layout, query_batch=2) as hooks:
            hooked_hidden = model.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_attentions=False, output_hidden_states=False,
                return_dict=True).last_hidden_state.detach().clone()
        actual = hooks.finish()
        capture = ValueCapture(model)
        try:
            oracle_output = model.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_attentions=True, output_hidden_states=False,
                return_dict=True)
        finally:
            capture.close()
    oracle = dense_oracle(model, layout, oracle_output.attentions, capture.values)
    errors = {
        key: float(np.max(np.abs(actual[key].astype(np.float32) - oracle[key])))
        for key in ("token_region_share", "token_sentence_share")}
    assert max(errors.values()) <= 0.001, errors

    # Changing tokens strictly after the last queried position cannot affect a
    # causal query.  This catches an accidental unmasked future path.
    changed_ids = input_ids.clone()
    changed_ids[0, 24:] = (changed_ids[0, 24:] + 17) % 127
    with torch.inference_mode():
        with TokenAttributionHooks(model, layout, query_batch=3) as changed_hooks:
            model.model(input_ids=changed_ids, attention_mask=mask, use_cache=False,
                        output_attentions=False, output_hidden_states=False,
                        return_dict=True)
        changed = changed_hooks.finish()
    early = np.asarray([position < 24 for position in actual["query_absolute_positions"]])
    future_counterfactual_error = max(
        float(np.max(np.abs(actual["token_region_share"][early].astype(np.float32) -
                            changed["token_region_share"][early].astype(np.float32)))),
        float(np.max(np.abs(actual["token_sentence_share"][early].astype(np.float32) -
                            changed["token_sentence_share"][early].astype(np.float32)))))
    assert future_counterfactual_error == 0.0

    nli = np.asarray([
        [[.7, .2, .1], [.2, .7, .1], [.1, .2, .7]],
        [[.1, .3, .6], [.6, .3, .1], [.2, .6, .2]]], dtype=np.float32)
    token_features = combine_token_features(
        actual["token_region_share"], actual["token_sentence_share"],
        actual["query_claim_ids"], nli)
    assert token_features.shape == (4, 2, 8)
    tiny = np.full((2, 3, 7), np.float32(1e-40), dtype=np.float32)
    tiny /= tiny.sum(-1, keepdims=True, dtype=np.float32)
    tiny16 = tiny.astype(np.float16)
    assert np.all(tiny16.sum(-1, dtype=np.float32) > 0)
    zero_sentence = np.zeros((2, 3, 7), dtype=np.float16)
    zero_feature = combine_token_features(
        np.full((2, 3, 3), 1 / 3, dtype=np.float32), zero_sentence,
        np.asarray([0, 1]), np.full((2, 7, 3), 1 / 3, dtype=np.float32))
    assert np.all(zero_feature[:, :, 3:] == 0)
    real_queries = 0
    max_queries = 0
    for real_layout in layouts:
        local, absolute, owners = query_geometry(real_layout)
        assert len(local) == len(absolute) == len(owners)
        real_queries += len(local)
        max_queries = max(max_queries, len(local))
    assert real_queries == config["lexical_bpe"]
    result = {
        "status": "passed_ready_for_explicit_gpu_extract",
        "scope": scope, "runner_sha256": sha(Path(__file__)),
        "signature_sha256": digest(signature),
        "tiny_model": {"layers": 2, "heads": 4, "kv_heads": 2,
                       "sequence_tokens": 48, "query_tokens": 4},
        "oracle": "explicit output_attentions rows times independently captured/repeated V-head L2 norms",
        "max_abs_errors_after_float16": errors,
        "future_token_counterfactual_max_abs_error": future_counterfactual_error,
        "all_real_layout_token_check": {
            "answers": len(layouts), "lexical_answer_BPE": real_queries,
            "maximum_per_answer": max_queries},
        "post_token_query": True,
        "previous_answer_strictly_less_than_query": True,
        "query_self_excluded_from_previous_answer": True,
        "future_keys_explicitly_masked": True,
        "query_token_averaging": False,
        "token_sentence_head_tensor_stored": False,
        "sentence_share_normalized_before_float16": True,
        "zero_sentence_mass_fallback": "all-zero NLI and concentration features",
        "float16_underflow_stress_passed": True,
        "combined_token_feature_width_per_layer": token_features.shape[-1],
        "pretrained_model_loaded": False, "gpu_used": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "labels_used": False, "official_test_opened": False,
    }
    frozen_json(config["out"] / "CPU_DENSE_ORACLE.json", result)
    assert_cpu_only()
    print("TOKEN_ATTRIBUTION_V4_CPU_ORACLE_PASSED", scope, flush=True)
    return result


def record_paths(output: Path, response_id: str):
    folder = output / "features"
    return (folder / f"{response_id}.npz", folder / f"{response_id}.json",
            folder / f"{response_id}.commit.json")


def read_committed(config: dict, plan: dict, layout: dict, signature: dict,
                   load_arrays: bool = False):
    feature_path, metadata_path, commit_path = record_paths(config["out"], plan["response_id"])
    if not commit_path.exists():
        return None
    assert feature_path.is_file() and metadata_path.is_file()
    commit, metadata = read_json(commit_path), read_json(metadata_path)
    assert commit["response_id"] == metadata["response_id"] == plan["response_id"]
    assert commit["metadata_sha256"] == sha(metadata_path)
    assert commit["npz_sha256"] == metadata["npz_sha256"] == sha(feature_path)
    assert metadata["signature_sha256"] == digest(signature)
    assert metadata["plan_sha256"] == digest(plan)
    assert metadata["layout_sha256"] == digest(layout)
    if not load_arrays:
        return metadata
    with np.load(feature_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    validate_arrays(arrays, plan, layout, LAYERS, HEADS)
    return arrays, metadata


def quarantine_uncommitted(config: dict, response_id: str) -> list[str]:
    feature_path, metadata_path, commit_path = record_paths(config["out"], response_id)
    assert not commit_path.exists()
    candidates = [feature_path, metadata_path,
                  feature_path.with_suffix(feature_path.suffix + ".pending"),
                  metadata_path.with_suffix(metadata_path.suffix + ".pending"),
                  commit_path.with_suffix(commit_path.suffix + ".pending")]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return []
    quarantine = config["out"] / "uncommitted" / response_id
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in existing:
        destination = quarantine / f"{path.name}.{sha(path)}"
        if destination.exists():
            assert sha(destination) == sha(path)
            path.unlink()
        else:
            path.replace(destination)
        moved.append(str(destination.relative_to(config["out"])))
    return moved


def save_record(config: dict, plan: dict, layout: dict, signature: dict,
                arrays: dict[str, np.ndarray], seconds: float) -> dict:
    audit = validate_arrays(arrays, plan, layout, LAYERS, HEADS)
    feature_path, metadata_path, commit_path = record_paths(config["out"], plan["response_id"])
    assert not any(path.exists() for path in (feature_path, metadata_path, commit_path))
    atomic_npz(feature_path, arrays)
    metadata = {
        "status": "complete", "version": VERSION, "scope": config["scope"],
        "response_id": plan["response_id"], "source_id": plan["source_id"],
        "group_id": plan["group_id"], "partition": plan["partition"],
        "signature_sha256": digest(signature), "plan_sha256": digest(plan),
        "layout_sha256": digest(layout), "npz_sha256": sha(feature_path),
        "seconds": seconds, **audit,
        "labels_used": False, "official_test_opened": False}
    atomic_json(metadata_path, metadata)
    atomic_json(commit_path, {
        "status": "committed", "response_id": plan["response_id"],
        "npz_sha256": metadata["npz_sha256"],
        "metadata_sha256": sha(metadata_path)})
    assert read_committed(config, plan, layout, signature) == metadata
    return metadata


@contextmanager
def exclusive_file(path: Path, owner: str):
    descriptor, acquired = None, False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raw = path.read_text(encoding="utf-8").strip()
            try:
                payload = json.loads(raw)
                pid = int(payload["pid"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Compatibility with older ``version:pid`` owners of the shared GPU lock.
                pid = int(raw.rsplit(":", 1)[-1])
            if os.name == "nt":
                process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if process:
                    exit_code = ctypes.c_ulong()
                    queried = ctypes.windll.kernel32.GetExitCodeProcess(
                        process, ctypes.byref(exit_code))
                    ctypes.windll.kernel32.CloseHandle(process)
                    alive = bool(queried and exit_code.value == 259)  # STILL_ACTIVE
                else:
                    alive = False
            else:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
            if alive:
                raise RuntimeError(f"Active lock {path} is owned by PID {pid}")
            stale = path.parent / "stale_locks"
            stale.mkdir(parents=True, exist_ok=True)
            destination = stale / f"{path.name}.{time.time_ns()}.{sha(path)}"
            path.replace(destination)
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        acquired = True
        os.write(descriptor, json.dumps({"owner": owner, "version": VERSION,
                                         "pid": os.getpid(),
                                         "created_time_ns": time.time_ns()}).encode("utf-8"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired and path.exists():
            path.unlink()


@contextmanager
def exclusive_gpu():
    with exclusive_file(GLOBAL_GPU_LOCK, "gpu"):
        yield


def verify_model_assets(signature: dict) -> None:
    current = model_asset_binding()
    assert current == signature["model"]


def extract(scope: str, limit: int | None = None) -> dict:
    config, plans, layouts, signature, _ = load_prepared(scope)
    cpu = read_json(config["out"] / "CPU_DENSE_ORACLE.json")
    assert cpu["status"] == "passed_ready_for_explicit_gpu_extract"
    assert cpu["runner_sha256"] == sha(Path(__file__))
    assert cpu["signature_sha256"] == digest(signature)
    assert torch.cuda.is_available(), "CUDA GPU is required for NF4 extraction"
    assert shutil.disk_usage(config["out"]).free > 2 * 1024 ** 3
    verify_model_assets(signature)
    started = time.perf_counter()
    model = None
    with exclusive_gpu():
        try:
            # Acquire the process lock before recovery/commit inspection so two
            # extractors cannot quarantine or write the same answer shard.
            existing = [read_committed(config, plan, layout, signature)
                        for plan, layout in zip(plans, layouts)]
            missing = [index for index, item in enumerate(existing) if item is None]
            recovered = {plans[index]["response_id"]: quarantine_uncommitted(
                config, plans[index]["response_id"]) for index in missing}
            recovered = {key: value for key, value in recovered.items() if value}
            if limit is not None:
                assert limit > 0
                missing = missing[:limit]
            if missing:
                model, load_metadata = llama_runner.load_nf4()
                spec = fq.architecture(model)
                assert spec["layers"] == LAYERS and spec["heads"] == HEADS
                assert spec["head_dim"] == 128
                atomic_json(config["out"] / "extract_started.json", {
                    "status": "running", "version": VERSION, "scope": scope,
                    "signature_sha256": digest(signature),
                    "missing_at_start": len(missing), "limit": limit,
                    "quarantined_uncommitted_records": recovered,
                    "model_load": load_metadata,
                    "labels_used": False, "official_test_opened": False})
                for completed, index in enumerate(missing, 1):
                    one_started = time.perf_counter()
                    arrays = extract_one(model, plans[index], layouts[index])
                    save_record(config, plans[index], layouts[index], signature,
                                arrays, time.perf_counter() - one_started)
                    atomic_json(config["out"] / "progress.json", {
                        "status": "running", "completed_this_invocation": completed,
                        "scheduled_this_invocation": len(missing),
                        "records_committed_total": sum(record_paths(
                            config["out"], plan["response_id"])[2].exists() for plan in plans),
                        "last_response_id": plans[index]["response_id"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "labels_used": False, "official_test_opened": False})
                    print("TOKEN_ATTRIBUTION_V4_EXTRACTED", completed, len(missing),
                          plans[index]["response_id"], flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
    return status(scope, elapsed=time.perf_counter() - started)


def status(scope: str, elapsed: float | None = None) -> dict:
    config, plans, layouts, signature, preparation = load_prepared(scope)
    cpu_path = config["out"] / "CPU_DENSE_ORACLE.json"
    cpu_valid = False
    if cpu_path.exists():
        cpu = read_json(cpu_path)
        cpu_valid = (cpu.get("status") == "passed_ready_for_explicit_gpu_extract" and
                     cpu.get("runner_sha256") == sha(Path(__file__)) and
                     cpu.get("signature_sha256") == digest(signature))
    committed = [read_committed(config, plan, layout, signature)
                 for plan, layout in zip(plans, layouts)]
    complete = all(item is not None for item in committed)
    report = {
        "version": VERSION, "scope": scope,
        "prepared": preparation["status"] == "prepared_waiting_for_cpu_check",
        "cpu_dense_oracle_valid": cpu_valid,
        "records_complete": sum(item is not None for item in committed),
        "records_total": config["answers"],
        "lexical_answer_BPE_complete": sum(item["lexical_answer_BPE"] for item in committed if item),
        "token_sentence_pairs_complete": sum(item["token_sentence_pairs"] for item in committed if item),
        "gpu_started": (config["out"] / "extract_started.json").exists(),
        "feature_manifest_complete": (config["out"] / "feature_manifest.json").exists(),
        "elapsed_seconds_this_invocation": elapsed,
        "query_token_averaging": False,
        "labels_used": False, "official_test_opened": False,
        "formal_baselines_modified": False, "training_or_scoring_run": False}
    atomic_json(config["out"] / "status.json", report)
    if complete and not (config["out"] / "feature_manifest.json").exists():
        manifest = {
            "status": "complete", "version": VERSION, "scope": scope,
            "records_complete": config["answers"],
            "lexical_answer_BPE": config["lexical_bpe"],
            "token_sentence_pairs": config["token_sentence_pairs"],
            "signature_sha256": digest(signature),
            "records": committed,
            "files": {plan["response_id"]: {
                "npz_sha256": item["npz_sha256"],
                "metadata_sha256": sha(record_paths(config["out"], plan["response_id"])[1]),
                "commit_sha256": sha(record_paths(config["out"], plan["response_id"])[2])}
                for plan, item in zip(plans, committed)},
            "labels_used": False, "official_test_opened": False,
            "formal_baselines_modified": False, "training_or_scoring_run": False}
        frozen_json(config["out"] / "feature_manifest.json", manifest)
        report["feature_manifest_complete"] = True
        atomic_json(config["out"] / "status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "cpu-check", "extract", "status"))
    parser.add_argument("--scope", choices=tuple(SCOPES), default="native")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "status":
        status(args.scope)
        return
    config = cfg(args.scope)
    with exclusive_file(config["out"] / ".writer.lock", f"extractor-{args.command}"):
        if args.command == "prepare":
            prepare(args.scope)
        elif args.command == "cpu-check":
            cpu_check(args.scope)
        else:
            extract(args.scope, args.limit)


if __name__ == "__main__":
    main()
