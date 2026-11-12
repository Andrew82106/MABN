"""Label-blind semantic source-attribution extraction for native RAGTruth QA.

Stages
------
prepare
    Bind the frozen 793 replay plans to the frozen atomic-microclaim inputs and
    map exact source-sentence/claim character spans to original full-prompt
    token coordinates.  This stage is CPU-only and never opens annotations.
cpu-check
    Compare the bounded Q/K/V hook against an explicit ``output_attentions``
    and value-projection oracle on a tiny random CPU Llama.
extract
    Teacher-force the frozen Llama-2-7b-chat answers under the established NF4
    load recipe and save one atomic, resumable NPZ per answer.  No classifier is
    trained and no score or label is read here.
status
    Report committed cache coverage without loading a model.

For a query token q and source token s, the primitive relevance is
``attention(q, s) * ||V_s,head||_2``.  A fragment first takes a maximum over its
tokens independently inside each head.  Claim values then average over the
claim's lexical answer queries.  The compact views finally average heads.  The
CSR cache retains the pre-head-average claim x source-sentence x layer x head
values (float16); a token x sentence x head tensor is never materialized or
stored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
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


VERSION = "semantic-source-attribution-v1"
OUT = ROOT / "results/semantic_source_attribution_v1"
PLANS_PATH = ROOT / "data/feature_preparation/plans.jsonl"
PLANS_MANIFEST_PATH = ROOT / "data/feature_preparation/manifest.json"
ATOMIC_INPUT_PATH = ROOT / "results/atomic_microclaim_nli_v1/inputs.jsonl"
ATOMIC_COMPLETE_PATH = ROOT / "results/atomic_microclaim_nli_v1/preparation_complete.json"
ATOMIC_STATS_PATH = ROOT / "results/atomic_microclaim_nli_v1/preparation_statistics.json"
MODEL_MANIFEST_PATH = ROOT / "model_download_manifest.json"
MODEL = fq.MODEL
GLOBAL_GPU_LOCK = ROOT / "results/.exclusive_gpu_runner.lock"

EXPECTED_ANSWERS = 793
EXPECTED_PARTITIONS = {"fit": 634, "calibration": 159}
EXPECTED_CLAIMS = 11322
EXPECTED_CLAIM_SENTENCE_EDGES = 166644
EXPECTED_MODEL = {
    "model_type": "llama", "layers": 32, "heads": 32,
    "hidden_size": 4096, "head_dim": 128,
}
QUERY_BATCH = 8
TOP_K = 15
THREADS = 4
SEED = 20260913
FORBIDDEN_ANNOTATION_KEYS = {
    "label", "labels", "gold", "risk", "risk_mask", "original_labels",
    "hallucination", "hallucination_label", "token_labels",
}

PROTOCOL = {
    "version": VERSION,
    "role": "ours semantic source-attribution feature extractor; formal baselines untouched",
    "data": {
        "answers": "Frozen 793 native Llama-2-7b-chat RAGTruth QA development replay plans only.",
        "claims": "Frozen 11,322 scored atomic_microclaim_nli_v1 microclaims, without annotation values.",
        "test": "No official-test or withheld-data path occurs in this runner.",
    },
    "coordinates": {
        "source_sentences": "Atomic sentence document character spans are shifted by the frozen rendered-reference start and intersected with the original full-string tokenizer offsets. Every non-whitespace sentence character and every alphanumeric character is checked for coverage.",
        "claims": "Frozen lexical answer-token indices are mapped through each plan's original absolute answer_token_positions; claim text and alphanumeric overlap are rechecked.",
        "query_timing": "Post-token query at the original answer token position.",
        "previous_answer": "Strictly smaller absolute answer positions; the query token itself is excluded.",
        "other_context": "All pre-answer tokens outside the union of source-sentence tokens.",
    },
    "primitive": "float32 softmax attention(q,s) multiplied by the corresponding repeated V-head L2 norm",
    "aggregation": {
        "fragment": "Maximum over fragment tokens independently in every head.",
        "claim": "Arithmetic mean over that claim's unique lexical query tokens.",
        "compact": "Arithmetic mean over attention heads, retained per layer.",
        "csr": "For every claim and every source sentence retain layer x head values before the head mean as float16.",
        "top_k": TOP_K,
    },
    "outputs": {
        "compact_float32": [
            "source_total_relevance", "previous_answer_relevance",
            "other_context_relevance", "passage_relevance",
            "sentence_relevance", "top_sentence_indices",
            "top_sentence_relevance",
        ],
        "csr": [
            "Per-answer claim_sentence_mass shards whose response-order concatenation is float16 [166644,1024]",
            "global_csr_index.npz claim_indptr int64 [11323]",
            "claim/edge response indices and exact sentence identity",
        ],
        "forbidden": "No token x sentence x head cache and no full attention matrix is stored.",
    },
    "model": {
        "checkpoint": llama_runner.REPO,
        "revision": llama_runner.REVISION,
        "load_config": llama_runner.LOAD_CONFIG,
        "teacher_forced": True,
        "trained_here": False,
    },
    "stage_gate": "prepare -> cpu-check -> explicit extract -> status; no training/scoring command exists",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path = Path(path)
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


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    pending.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def _global_csr_index(layouts: list[dict]) -> dict[str, np.ndarray]:
    """Index the logical response-order concatenation of per-answer CSR shards."""
    indptr = [0]
    claim_response_indices, claim_local_ids, claim_microclaim_indices = [], [], []
    edge_response_indices, edge_sentence_indices = [], []
    edge_passage_ids, edge_sentence_ids, edge_sentence_hashes = [], [], []
    for response_index, layout in enumerate(layouts):
        sentences = layout["sentences"]
        for claim in layout["claims"]:
            claim_response_indices.append(response_index)
            claim_local_ids.append(claim["claim_id"])
            claim_microclaim_indices.append(claim["microclaim_index"])
            for sentence in sentences:
                edge_response_indices.append(response_index)
                edge_sentence_indices.append(sentence["sentence_index"])
                edge_passage_ids.append(sentence["passage_id"])
                edge_sentence_ids.append(sentence["sentence_id"])
                edge_sentence_hashes.append(np.frombuffer(
                    bytes.fromhex(sentence["text_sha256"]), dtype=np.uint8))
            indptr.append(indptr[-1] + len(sentences))
    assert len(indptr) == EXPECTED_CLAIMS + 1
    assert indptr[-1] == EXPECTED_CLAIM_SENTENCE_EDGES
    result = {
        "claim_indptr": np.asarray(indptr, dtype=np.int64),
        "claim_response_indices": np.asarray(claim_response_indices, dtype=np.int32),
        "claim_local_ids": np.asarray(claim_local_ids, dtype=np.int32),
        "claim_microclaim_indices": np.asarray(claim_microclaim_indices, dtype=np.int32),
        "edge_response_indices": np.asarray(edge_response_indices, dtype=np.int32),
        "edge_sentence_indices": np.asarray(edge_sentence_indices, dtype=np.int32),
        "edge_sentence_passage_ids": np.asarray(edge_passage_ids, dtype=np.int8),
        "edge_sentence_ids": np.asarray(edge_sentence_ids, dtype=np.int32),
        "edge_sentence_identity_sha256": np.stack(edge_sentence_hashes),
        "logical_claim_sentence_mass_shape": np.asarray(
            [EXPECTED_CLAIM_SENTENCE_EDGES,
             EXPECTED_MODEL["layers"] * EXPECTED_MODEL["heads"]], dtype=np.int64),
    }
    assert result["claim_indptr"].shape == (EXPECTED_CLAIMS + 1,)
    assert result["edge_sentence_identity_sha256"].shape == (
        EXPECTED_CLAIM_SENTENCE_EDGES, 32)
    return result


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "CPU stage must not initialize CUDA"


def reject_annotation_keys(value) -> None:
    if isinstance(value, dict):
        found = FORBIDDEN_ANNOTATION_KEYS & set(value)
        assert not found, f"Annotation-bearing keys are forbidden: {sorted(found)}"
        for child in value.values():
            reject_annotation_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_annotation_keys(child)


def model_asset_binding() -> dict:
    manifest = read_json(MODEL_MANIFEST_PATH)
    assert manifest["status"] == "complete"
    assert manifest["repo_id"] == llama_runner.REPO
    assert manifest["revision"] == llama_runner.REVISION
    assets = {}
    for item in manifest["files"]:
        assert item["status"] == "verified" and item["source_hash_match"] is True
        path = MODEL / item["filename"]
        assert path.is_file() and path.stat().st_size == item["actual_bytes"]
        assets[item["filename"]] = {
            "bytes": int(item["actual_bytes"]),
            "sha256": item["actual_sha256"],
        }
    config = read_json(MODEL / "config.json")
    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == EXPECTED_MODEL["layers"]
    assert config["num_attention_heads"] == EXPECTED_MODEL["heads"]
    assert config["hidden_size"] == EXPECTED_MODEL["hidden_size"]
    assert config["hidden_size"] // config["num_attention_heads"] == EXPECTED_MODEL["head_dim"]
    return {
        "manifest_sha256": sha(MODEL_MANIFEST_PATH),
        "repo_id": manifest["repo_id"], "revision": manifest["revision"],
        "assets": assets,
    }


def _mapped_span(rendered_text: str, offsets: np.ndarray,
                 left: int, right: int) -> dict:
    """Map one half-open character span without independently tokenizing it."""
    assert 0 <= left < right <= len(rendered_text)
    positions = np.flatnonzero(
        (offsets[:, 1] > left) & (offsets[:, 0] < right) &
        (offsets[:, 1] > offsets[:, 0])
    ).astype(np.int64)
    assert len(positions) > 0
    intersections = []
    coverage = np.zeros(right - left, dtype=bool)
    lexical_positions = []
    for pos in positions:
        token_left, token_right = map(int, offsets[pos])
        begin, end = max(token_left, left), min(token_right, right)
        assert begin < end
        intersections.append([begin, end])
        coverage[begin - left:end - left] = True
        if any(char.isalnum() for char in rendered_text[begin:end]):
            lexical_positions.append(int(pos))
    nonspace = np.fromiter((not char.isspace() for char in rendered_text[left:right]),
                           dtype=bool)
    alnum = np.fromiter((char.isalnum() for char in rendered_text[left:right]),
                        dtype=bool)
    assert np.all(coverage | ~nonspace), "Tokenizer offsets lost sentence content"
    lexical_coverage = np.zeros(right - left, dtype=bool)
    for pos in lexical_positions:
        begin = max(int(offsets[pos, 0]), left)
        end = min(int(offsets[pos, 1]), right)
        lexical_coverage[begin - left:end - left] = True
    assert lexical_positions and np.all(lexical_coverage | ~alnum)
    return {
        "token_positions": positions.tolist(),
        "lexical_token_positions": lexical_positions,
        "token_character_intersections": intersections,
        "boundary_crossing_token_positions": [
            int(pos) for pos in positions
            if int(offsets[pos, 0]) < left or int(offsets[pos, 1]) > right
        ],
    }


def _prepare_layout(plan: dict, atomic_row: dict) -> dict:
    reject_annotation_keys({key: value for key, value in atomic_row.items()
                            if key not in ("labels_used", "official_test_opened")})
    assert atomic_row["labels_used"] is False
    assert atomic_row["official_test_opened"] is False
    rid = plan["response_id"]
    assert rid == atomic_row["response_id"]
    assert plan["source_id"] == atomic_row["source_id"]
    assert plan["group_id"] == atomic_row["group_id"]
    assert plan["partition"] == atomic_row["partition"] in fq.PARTITIONS
    assert plan["official_split"] == "train"
    assert plan["labels_used"] is False
    assert digest(plan["original_response"]) == plan["answer_sha256"] == atomic_row["answer_sha256"]

    view = plan["original"]
    rendered = fq.WRAPPER_LEFT + plan["released_prompt"] + fq.WRAPPER_RIGHT + plan["original_response"]
    assert digest(rendered) == view["rendered_text_sha256"]
    offsets = np.asarray(view["input_token_offsets"], dtype=np.int64)
    assert offsets.shape == (len(view["input_ids"]), 2)
    assert np.all(offsets[:, 0] >= 0) and np.all(offsets[:, 1] <= len(rendered))
    answer_positions = np.asarray(view["answer_token_positions"], dtype=np.int64)
    answer_offsets = np.asarray(view["response_token_offsets"], dtype=np.int64)
    assert len(answer_positions) == len(answer_offsets) == len(atomic_row["lexical_token_microclaims"])
    assert digest(answer_offsets.tolist()) == atomic_row["response_token_offsets_sha256"]
    assert np.array_equal(np.asarray(view["input_ids"])[answer_positions],
                          np.asarray(view["answer_token_ids"]))
    assert np.all(np.diff(answer_positions) == 1)

    ref_left, ref_right = map(int, view["rendered_reference_character_range"])
    assert ref_left == len(fq.WRAPPER_LEFT) + (
        ref_left - len(fq.WRAPPER_LEFT))
    prompt_ref_left = ref_left - len(fq.WRAPPER_LEFT)
    prompt_ref_right = ref_right - len(fq.WRAPPER_LEFT)
    assert 0 <= prompt_ref_left < prompt_ref_right <= len(plan["released_prompt"])
    document = plan["released_prompt"][prompt_ref_left:prompt_ref_right]
    assert rendered[ref_left:ref_right] == document

    sentences = []
    passage_positions = []
    sentence_counter = 0
    for expected_pid, passage in enumerate(atomic_row["passages"], 1):
        assert passage["passage_id"] == expected_pid
        body_left, body_right = passage["body_char_start"], passage["body_char_end"]
        assert digest(document[body_left:body_right]) == passage["body_sha256"]
        local_passage_positions = set()
        for sentence in passage["sentences"]:
            doc_left = int(sentence["document_char_start"])
            doc_right = int(sentence["document_char_end"])
            assert document[doc_left:doc_right] == sentence["text"]
            assert digest(sentence["text"]) == sentence["text_sha256"]
            absolute_left, absolute_right = ref_left + doc_left, ref_left + doc_right
            mapping = _mapped_span(rendered, offsets, absolute_left, absolute_right)
            assert max(mapping["token_positions"]) < int(answer_positions[0])
            local_passage_positions.update(mapping["token_positions"])
            sentences.append({
                "sentence_index": sentence_counter,
                "passage_id": expected_pid,
                "sentence_id": int(sentence["sentence_id"]),
                "document_character_range": [doc_left, doc_right],
                "rendered_character_range": [absolute_left, absolute_right],
                "text_sha256": sentence["text_sha256"],
                **mapping,
            })
            sentence_counter += 1
        passage_positions.append(sorted(local_passage_positions))
        assert passage_positions[-1]
    sentence_union = sorted({pos for row in sentences for pos in row["token_positions"]})
    context_positions = set(map(int, view["context_token_positions"]))
    assert set(sentence_union) <= context_positions
    prefix_positions = set(range(int(answer_positions[0])))
    other_context = sorted(prefix_positions - set(sentence_union))
    assert other_context and not (set(other_context) & set(sentence_union))

    claims = []
    inverse = [[] for _ in answer_positions]
    answer = plan["original_response"]
    for claim_id, claim in enumerate(atomic_row["claims"]):
        assert claim["claim_id"] == claim_id
        assert claim["response_id"] == rid
        start, end = int(claim["start"]), int(claim["end"])
        assert answer[start:end] == claim["text"]
        local = list(map(int, claim["lexical_token_indices"]))
        assert local == sorted(set(local)) and local
        assert local[0] >= 0 and local[-1] < len(answer_positions)
        absolute = [int(answer_positions[index]) for index in local]
        intersections = []
        for local_index, absolute_position in zip(local, absolute):
            token_left, token_right = map(int, answer_offsets[local_index])
            begin, finish = max(token_left, start), min(token_right, end)
            assert begin < finish
            assert any(char.isalnum() for char in answer[begin:finish])
            intersections.append([begin, finish])
            inverse[local_index].append(claim_id)
            assert int(view["answer_token_positions"][local_index]) == absolute_position
        claims.append({
            "claim_id": claim_id,
            "microclaim_id": claim["microclaim_id"],
            "microclaim_index": int(claim["microclaim_index"]),
            "claim_character_range": [start, end],
            "claim_text_sha256": digest(claim["text"]),
            "lexical_answer_token_indices": local,
            "lexical_absolute_token_positions": absolute,
            "claim_character_intersections": intersections,
        })
    assert inverse == atomic_row["lexical_token_microclaims"]
    assert all(claim["lexical_absolute_token_positions"] for claim in claims)

    result = {
        "schema_version": VERSION,
        "response_id": rid, "source_id": plan["source_id"],
        "group_id": plan["group_id"], "partition": plan["partition"],
        "plan_sha256": digest(plan), "atomic_row_sha256": digest(atomic_row),
        "sequence_length": len(view["input_ids"]),
        "answer_token_positions": list(map(int, answer_positions)),
        "source_sentence_union_token_positions": sentence_union,
        "passage_token_positions": passage_positions,
        "other_context_token_positions": other_context,
        "sentences": sentences, "claims": claims,
        "claim_sentence_edges": len(claims) * len(sentences),
        "labels_used": False, "official_test_opened": False,
    }
    reject_annotation_keys({key: value for key, value in result.items()
                            if key not in ("labels_used", "official_test_opened")})
    return result


def _software_signature() -> dict:
    packages = ("torch", "transformers", "numpy", "bitsandbytes", "accelerate")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions,
            "torch_cuda_runtime": torch.version.cuda}


def prepare() -> dict:
    assert_cpu_only()
    plans_manifest = read_json(PLANS_MANIFEST_PATH)
    atomic_complete = read_json(ATOMIC_COMPLETE_PATH)
    atomic_stats = read_json(ATOMIC_STATS_PATH)
    assert plans_manifest["records"] == EXPECTED_ANSWERS
    assert plans_manifest["partitions"] == EXPECTED_PARTITIONS
    assert sha(PLANS_PATH) == plans_manifest["plans_jsonl_sha256"]
    assert atomic_complete["answers"] == EXPECTED_ANSWERS
    assert atomic_complete["microclaims"] == EXPECTED_CLAIMS
    assert atomic_complete["files_sha256"]["inputs.jsonl"] == sha(ATOMIC_INPUT_PATH)
    assert atomic_stats["labels_accessed"] is False
    assert atomic_stats["official_test_opened"] is False

    plans = json_lines(PLANS_PATH)
    atomic_rows = json_lines(ATOMIC_INPUT_PATH)
    assert len(plans) == len(atomic_rows) == EXPECTED_ANSWERS
    assert [row["response_id"] for row in plans] == [row["response_id"] for row in atomic_rows]
    assert Counter(row["partition"] for row in plans) == EXPECTED_PARTITIONS

    layouts = []
    for index, (plan, atomic_row) in enumerate(zip(plans, atomic_rows)):
        layouts.append(_prepare_layout(plan, atomic_row))
        if (index + 1) % 100 == 0:
            print("SEMANTIC_ATTRIBUTION_PREPARED", index + 1, EXPECTED_ANSWERS,
                  flush=True)
    claims = sum(len(row["claims"]) for row in layouts)
    sentences = sum(len(row["sentences"]) for row in layouts)
    edges = sum(row["claim_sentence_edges"] for row in layouts)
    assert claims == EXPECTED_CLAIMS
    assert edges == EXPECTED_CLAIM_SENTENCE_EDGES

    OUT.mkdir(parents=True, exist_ok=True)
    layouts_path = OUT / "layouts.jsonl"
    if layouts_path.exists():
        assert json_lines(layouts_path) == layouts, "Frozen layouts changed"
    else:
        atomic_jsonl(layouts_path, layouts)
    csr_index_path = OUT / "global_csr_index.npz"
    csr_index = _global_csr_index(layouts)
    if csr_index_path.exists():
        with np.load(csr_index_path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(csr_index)
            assert all(np.array_equal(loaded[key], value)
                       for key, value in csr_index.items())
    else:
        atomic_npz(csr_index_path, csr_index)
    frozen_json(OUT / "protocol.json", PROTOCOL)
    signature = {
        "version": VERSION,
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "plans_sha256": sha(PLANS_PATH),
        "plans_manifest_sha256": sha(PLANS_MANIFEST_PATH),
        "atomic_inputs_sha256": sha(ATOMIC_INPUT_PATH),
        "atomic_preparation_complete_sha256": sha(ATOMIC_COMPLETE_PATH),
        "atomic_preparation_statistics_sha256": sha(ATOMIC_STATS_PATH),
        "layouts_sha256": sha(layouts_path),
        "global_csr_index_sha256": sha(csr_index_path),
        "model": model_asset_binding(),
        "load_config": llama_runner.LOAD_CONFIG,
        "software": _software_signature(),
        "response_ids_in_order": [row["response_id"] for row in layouts],
        "labels_used": False, "official_test_opened": False,
    }
    frozen_json(OUT / "signature.json", signature)
    boundary_sentences = sum(
        bool(sentence["boundary_crossing_token_positions"])
        for row in layouts for sentence in row["sentences"]
    )
    report = {
        "status": "prepared_waiting_for_cpu_check",
        "answers": len(layouts), "partitions": EXPECTED_PARTITIONS,
        "microclaims": claims, "source_sentences": sentences,
        "claim_sentence_edges": edges,
        "expected_csr_shape": [edges, EXPECTED_MODEL["layers"] * EXPECTED_MODEL["heads"]],
        "global_claim_indptr_shape": [claims + 1],
        "expected_csr_float16_bytes": edges * EXPECTED_MODEL["layers"] * EXPECTED_MODEL["heads"] * 2,
        "source_sentences_with_boundary_crossing_tokens": boundary_sentences,
        "all_source_sentence_spans_mapped": True,
        "all_claim_lexical_tokens_mapped_to_absolute_positions": True,
        "strict_previous_answer_excludes_query": True,
        "other_context_separate": True,
        "layouts_sha256": signature["layouts_sha256"],
        "global_csr_index_sha256": signature["global_csr_index_sha256"],
        "signature_sha256": digest(signature),
        "gpu_started": False, "model_loaded": False,
        "labels_used": False, "official_test_opened": False,
    }
    frozen_json(OUT / "PREPARATION.json", report)
    assert_cpu_only()
    print("SEMANTIC_ATTRIBUTION_PREPARATION_COMPLETE", claims, edges, flush=True)
    return report


def load_prepared():
    signature = read_json(OUT / "signature.json")
    preparation = read_json(OUT / "PREPARATION.json")
    assert signature["runner_sha256"] == sha(Path(__file__))
    assert signature["protocol_sha256"] == sha(OUT / "protocol.json")
    assert signature["plans_sha256"] == sha(PLANS_PATH)
    assert signature["atomic_inputs_sha256"] == sha(ATOMIC_INPUT_PATH)
    assert signature["layouts_sha256"] == sha(OUT / "layouts.jsonl")
    assert signature["global_csr_index_sha256"] == sha(OUT / "global_csr_index.npz")
    assert preparation["signature_sha256"] == digest(signature)
    assert preparation["microclaims"] == EXPECTED_CLAIMS
    assert preparation["claim_sentence_edges"] == EXPECTED_CLAIM_SENTENCE_EDGES
    plans = json_lines(PLANS_PATH)
    layouts = json_lines(OUT / "layouts.jsonl")
    assert len(plans) == len(layouts) == EXPECTED_ANSWERS
    assert [p["response_id"] for p in plans] == signature["response_ids_in_order"]
    assert [x["response_id"] for x in layouts] == signature["response_ids_in_order"]
    with np.load(OUT / "global_csr_index.npz", allow_pickle=False) as index:
        assert index["claim_indptr"].shape == (EXPECTED_CLAIMS + 1,)
        assert int(index["claim_indptr"][-1]) == EXPECTED_CLAIM_SENTENCE_EDGES
        assert np.array_equal(index["logical_claim_sentence_mass_shape"],
                              [EXPECTED_CLAIM_SENTENCE_EDGES,
                               EXPECTED_MODEL["layers"] * EXPECTED_MODEL["heads"]])
    return plans, layouts, signature, preparation


def _unique_sorted(values, sequence_length: int, name: str) -> list[int]:
    result = sorted(set(map(int, values)))
    assert result and result[0] >= 0 and result[-1] < sequence_length, name
    return result


class SemanticAttributionHooks:
    """Bounded Q/K/V reconstruction; aggregation occurs before CPU transfer."""

    def __init__(self, model, layout: dict, query_batch: int = QUERY_BATCH):
        self.model = model
        self.spec = fq.architecture(model)
        self.device = model.model.embed_tokens.weight.device
        self.sequence_length = int(layout["sequence_length"])
        self.answer_positions = _unique_sorted(
            layout["answer_token_positions"], self.sequence_length, "answer")
        self.source_positions = _unique_sorted(
            layout["source_sentence_union_token_positions"], self.sequence_length,
            "source")
        self.other_context_positions = _unique_sorted(
            layout["other_context_token_positions"], self.sequence_length,
            "other context")
        self.passage_positions = [
            _unique_sorted(values, self.sequence_length, f"passage {index}")
            for index, values in enumerate(layout["passage_token_positions"], 1)
        ]
        self.sentence_positions = [
            _unique_sorted(row["token_positions"], self.sequence_length,
                           f"sentence {row['sentence_index']}")
            for row in layout["sentences"]
        ]
        self.claim_positions = [
            _unique_sorted(row["lexical_absolute_token_positions"],
                           self.sequence_length, f"claim {row['claim_id']}")
            for row in layout["claims"]
        ]
        query_values = sorted({pos for values in self.claim_positions for pos in values})
        self.query_positions = _unique_sorted(query_values, self.sequence_length, "query")
        assert set(self.query_positions) <= set(self.answer_positions)
        assert max(self.source_positions + self.other_context_positions) < min(self.answer_positions)
        assert not (set(self.source_positions) & set(self.other_context_positions))
        self.query_batch = int(query_batch)
        assert self.query_batch > 0

        q_lookup = {pos: index for index, pos in enumerate(self.query_positions)}
        membership = np.zeros((len(self.claim_positions), len(self.query_positions)), dtype=np.float32)
        for claim_id, positions in enumerate(self.claim_positions):
            weight = np.float32(1.0 / len(positions))
            for position in positions:
                membership[claim_id, q_lookup[position]] = weight
        # Re-normalize in the exact storage dtype.  For lengths such as 26,
        # repeated float32(1/n) can sum one ulp above the old 1e-7 gate.
        row_sums = membership.sum(axis=1, dtype=np.float32)
        assert np.all(row_sums > 0)
        membership /= row_sums[:, None]
        assert np.allclose(membership.sum(axis=1, dtype=np.float32), 1.0,
                           atol=5e-7, rtol=0)
        self.membership = torch.tensor(membership, device=self.device)

        c, l, h, s = (len(self.claim_positions), self.spec["layers"],
                      self.spec["heads"], len(self.sentence_positions))
        self.claim_sentence = np.empty((c, s, l, h), dtype=np.float32)
        self.compact_source = np.empty((c, l), dtype=np.float32)
        self.compact_previous = np.empty((c, l), dtype=np.float32)
        self.compact_other = np.empty((c, l), dtype=np.float32)
        self.compact_passage = np.empty((c, l, 3), dtype=np.float32)
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
    def _fragment(contribution: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return query x head after token-max; contribution is head x query x key."""
        return contribution.index_select(-1, positions).amax(-1).transpose(0, 1)

    def _capture(self, layer: int, module, values: torch.Tensor) -> None:
        state = self.pending.pop(layer)
        assert set(state) == {"rope", "q", "k"}
        heads, kv_heads, dim = (self.spec["heads"], self.spec["kv_heads"],
                                self.spec["head_dim"])
        assert values.shape[0] == state["q"].shape[0] == state["k"].shape[0] == 1
        assert values.shape[1] == state["q"].shape[1] == state["k"].shape[1] == self.sequence_length
        query = state["q"].view(1, -1, heads, dim).transpose(1, 2)
        key = state["k"].view(1, -1, kv_heads, dim).transpose(1, 2)
        value = values.view(1, -1, kv_heads, dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *state["rope"])
        key = repeat_kv(key, heads // kv_heads)[0]
        value = repeat_kv(value, heads // kv_heads)[0]
        query = query[0]
        value_norm = value.float().norm(p=2, dim=-1)
        absolute = torch.arange(self.sequence_length, device=self.device)
        queries = torch.tensor(self.query_positions, dtype=torch.long,
                               device=self.device)
        answer = torch.tensor(self.answer_positions, dtype=torch.long,
                              device=self.device)
        source = torch.tensor(self.source_positions, dtype=torch.long,
                              device=self.device)
        other = torch.tensor(self.other_context_positions, dtype=torch.long,
                             device=self.device)
        passages = [torch.tensor(x, dtype=torch.long, device=self.device)
                    for x in self.passage_positions]
        sentences = [torch.tensor(x, dtype=torch.long, device=self.device)
                     for x in self.sentence_positions]
        c, s, h = len(self.claim_positions), len(sentences), heads
        sentence_accumulator = torch.zeros((c, s, h), dtype=torch.float32,
                                           device=self.device)
        source_accumulator = torch.zeros((c, h), dtype=torch.float32,
                                         device=self.device)
        previous_accumulator = torch.zeros((c, h), dtype=torch.float32,
                                           device=self.device)
        other_accumulator = torch.zeros((c, h), dtype=torch.float32,
                                        device=self.device)
        passage_accumulator = torch.zeros((c, 3, h), dtype=torch.float32,
                                          device=self.device)

        for begin in range(0, len(self.query_positions), self.query_batch):
            end = min(begin + self.query_batch, len(self.query_positions))
            positions = queries[begin:end]
            q = query.index_select(1, positions)
            logits = (q @ key.transpose(-2, -1)) * module.scaling
            logits.masked_fill_(absolute[None, None, :] > positions[None, :, None],
                                float("-inf"))
            attention = logits.float().softmax(-1)
            contribution = attention * value_norm[:, None, :]
            membership = self.membership[:, begin:end]
            source_accumulator += membership @ self._fragment(contribution, source)
            other_accumulator += membership @ self._fragment(contribution, other)
            for passage_id, token_positions in enumerate(passages):
                passage_accumulator[:, passage_id] += (
                    membership @ self._fragment(contribution, token_positions))
            for sentence_id, token_positions in enumerate(sentences):
                sentence_accumulator[:, sentence_id] += (
                    membership @ self._fragment(contribution, token_positions))

            # Strict previous-answer pool: position < query, never <= query.
            previous = answer[None, :] < positions[:, None]
            selected = contribution.index_select(-1, answer)
            masked = selected.masked_fill(~previous[None], float("-inf"))
            previous_values = masked.amax(-1).transpose(0, 1)
            previous_values = torch.where(torch.isfinite(previous_values),
                                          previous_values,
                                          torch.zeros_like(previous_values))
            previous_accumulator += membership @ previous_values
            del q, logits, attention, contribution, membership, selected, masked

        self.claim_sentence[:, :, layer] = sentence_accumulator.cpu().numpy()
        self.compact_source[:, layer] = source_accumulator.mean(-1).cpu().numpy()
        self.compact_previous[:, layer] = previous_accumulator.mean(-1).cpu().numpy()
        self.compact_other[:, layer] = other_accumulator.mean(-1).cpu().numpy()
        self.compact_passage[:, layer] = passage_accumulator.mean(-1).cpu().numpy()
        self.seen.add(layer)

    def finish(self) -> dict[str, np.ndarray]:
        assert len(self.seen) == self.spec["layers"] and not self.pending
        assert np.isfinite(self.claim_sentence).all()
        sentence_relevance = self.claim_sentence.mean(axis=-1, dtype=np.float32).transpose(0, 2, 1)
        c, l, s = sentence_relevance.shape
        top_indices = np.full((c, l, TOP_K), -1, dtype=np.int32)
        top_values = np.zeros((c, l, TOP_K), dtype=np.float32)
        take = min(TOP_K, s)
        for claim in range(c):
            for layer in range(l):
                order = np.lexsort((np.arange(s), -sentence_relevance[claim, layer]))[:take]
                top_indices[claim, layer, :take] = order
                top_values[claim, layer, :take] = sentence_relevance[claim, layer, order]

        # CSR enumerates every sentence for every claim in deterministic order.
        claim_indptr = np.arange(c + 1, dtype=np.int64) * s
        claim_sentence_indices = np.tile(np.arange(s, dtype=np.int32), c)
        csr = self.claim_sentence.transpose(0, 1, 2, 3).reshape(c * s, l * self.spec["heads"])
        result = {
            "claim_sentence_mass": csr.astype(np.float16),
            "claim_indptr": claim_indptr,
            "claim_sentence_indices": claim_sentence_indices,
            "source_total_relevance": self.compact_source.astype(np.float32),
            "previous_answer_relevance": self.compact_previous.astype(np.float32),
            "other_context_relevance": self.compact_other.astype(np.float32),
            "passage_relevance": self.compact_passage.astype(np.float32),
            "sentence_relevance": sentence_relevance.astype(np.float32),
            "top_sentence_indices": top_indices,
            "top_sentence_relevance": top_values,
        }
        return result


@torch.inference_mode()
def extract_one(model, plan: dict, layout: dict) -> dict[str, np.ndarray]:
    assert not model.training
    assert plan["response_id"] == layout["response_id"]
    ids, mask, _ = fq.tensor_input(model, plan["original"])
    with SemanticAttributionHooks(model, layout) as hooks:
        model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                    output_attentions=False, output_hidden_states=False,
                    return_dict=True)
    arrays = hooks.finish()
    sentence_rows = layout["sentences"]
    claim_rows = layout["claims"]
    arrays.update({
        "claim_ids": np.asarray([x["claim_id"] for x in claim_rows], dtype=np.int32),
        "microclaim_indices": np.asarray([x["microclaim_index"] for x in claim_rows], dtype=np.int32),
        "sentence_indices": np.asarray([x["sentence_index"] for x in sentence_rows], dtype=np.int32),
        "sentence_passage_ids": np.asarray([x["passage_id"] for x in sentence_rows], dtype=np.int8),
        "sentence_ids": np.asarray([x["sentence_id"] for x in sentence_rows], dtype=np.int32),
        "sentence_identity_sha256": np.stack([
            np.frombuffer(bytes.fromhex(x["text_sha256"]), dtype=np.uint8)
            for x in sentence_rows
        ]),
        "answer_token_positions": np.asarray(layout["answer_token_positions"], dtype=np.int64),
    })
    return arrays


def validate_arrays(arrays: dict[str, np.ndarray], layout: dict,
                    layers: int, heads: int) -> dict:
    required = {
        "claim_sentence_mass", "claim_indptr", "claim_sentence_indices",
        "source_total_relevance", "previous_answer_relevance",
        "other_context_relevance", "passage_relevance", "sentence_relevance",
        "top_sentence_indices", "top_sentence_relevance", "claim_ids",
        "microclaim_indices", "sentence_indices", "sentence_passage_ids",
        "sentence_ids", "sentence_identity_sha256", "answer_token_positions",
    }
    assert set(arrays) == required
    c, s = len(layout["claims"]), len(layout["sentences"])
    assert arrays["claim_sentence_mass"].shape == (c * s, layers * heads)
    assert arrays["claim_sentence_mass"].dtype == np.float16
    assert arrays["claim_indptr"].shape == (c + 1,)
    assert arrays["claim_indptr"].dtype == np.int64
    assert np.array_equal(arrays["claim_indptr"], np.arange(c + 1, dtype=np.int64) * s)
    assert np.array_equal(arrays["claim_sentence_indices"], np.tile(np.arange(s, dtype=np.int32), c))
    for key in ("source_total_relevance", "previous_answer_relevance",
                "other_context_relevance"):
        assert arrays[key].shape == (c, layers) and arrays[key].dtype == np.float32
    assert arrays["passage_relevance"].shape == (c, layers, 3)
    assert arrays["sentence_relevance"].shape == (c, layers, s)
    assert arrays["top_sentence_indices"].shape == (c, layers, TOP_K)
    assert arrays["top_sentence_relevance"].shape == (c, layers, TOP_K)
    assert arrays["sentence_identity_sha256"].shape == (s, 32)
    float_keys = [key for key, value in arrays.items() if value.dtype.kind == "f"]
    assert all(np.isfinite(arrays[key]).all() for key in float_keys)
    assert all(np.all(arrays[key] >= 0) for key in float_keys)
    reconstructed = arrays["claim_sentence_mass"].astype(np.float32).reshape(c, s, layers, heads)
    quantized_mean = reconstructed.mean(-1, dtype=np.float32).transpose(0, 2, 1)
    # Compact float32 is computed before float16 storage; bounded quantization drift only.
    drift = float(np.max(np.abs(quantized_mean - arrays["sentence_relevance"])))
    assert drift <= 0.02, drift
    assert np.array_equal(arrays["claim_ids"], np.arange(c, dtype=np.int32))
    assert np.array_equal(arrays["sentence_indices"], np.arange(s, dtype=np.int32))
    assert np.array_equal(arrays["answer_token_positions"], layout["answer_token_positions"])
    return {"claims": c, "sentences": s, "edges": c * s,
            "csr_float16_head_mean_max_abs_drift": drift}


class _ValueCapture:
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


def _dense_oracle(model, layout, attentions, projected_values) -> dict[str, np.ndarray]:
    spec = fq.architecture(model)
    queries = sorted({pos for claim in layout["claims"]
                      for pos in claim["lexical_absolute_token_positions"]})
    q_lookup = {pos: index for index, pos in enumerate(queries)}
    memberships = []
    for claim in layout["claims"]:
        memberships.append([q_lookup[pos] for pos in claim["lexical_absolute_token_positions"]])
    source = layout["source_sentence_union_token_positions"]
    other = layout["other_context_token_positions"]
    passages = layout["passage_token_positions"]
    sentences = [x["token_positions"] for x in layout["sentences"]]
    answer = layout["answer_token_positions"]
    c, s, l, h = len(memberships), len(sentences), spec["layers"], spec["heads"]
    claim_sentence = np.empty((c, s, l, h), dtype=np.float32)
    compact_source = np.empty((c, l), dtype=np.float32)
    compact_previous = np.empty((c, l), dtype=np.float32)
    compact_other = np.empty((c, l), dtype=np.float32)
    compact_passage = np.empty((c, l, 3), dtype=np.float32)

    def fragment(contribution, positions):
        return contribution[:, positions].max(axis=-1)

    for layer, native in enumerate(attentions):
        value = projected_values[layer].view(1, -1, spec["kv_heads"],
                                             spec["head_dim"]).transpose(1, 2)
        value = repeat_kv(value, spec["heads"] // spec["kv_heads"])[0]
        norm = value.float().norm(p=2, dim=-1).cpu().numpy()
        for claim_id, query_indices in enumerate(memberships):
            per_source, per_previous, per_other = [], [], []
            per_passage = [[] for _ in range(3)]
            per_sentence = [[] for _ in range(s)]
            for query_index in query_indices:
                query = queries[query_index]
                row = native[0, :, query].float().cpu().numpy()
                contribution = row * norm
                per_source.append(fragment(contribution, source))
                previous = [position for position in answer if position < query]
                per_previous.append(fragment(contribution, previous) if previous
                                    else np.zeros(h, dtype=np.float32))
                per_other.append(fragment(contribution, other))
                for passage_id in range(3):
                    per_passage[passage_id].append(fragment(contribution, passages[passage_id]))
                for sentence_id in range(s):
                    per_sentence[sentence_id].append(fragment(contribution, sentences[sentence_id]))
            source_heads = np.mean(per_source, axis=0, dtype=np.float32)
            previous_heads = np.mean(per_previous, axis=0, dtype=np.float32)
            other_heads = np.mean(per_other, axis=0, dtype=np.float32)
            compact_source[claim_id, layer] = source_heads.mean(dtype=np.float32)
            compact_previous[claim_id, layer] = previous_heads.mean(dtype=np.float32)
            compact_other[claim_id, layer] = other_heads.mean(dtype=np.float32)
            for passage_id in range(3):
                heads_value = np.mean(per_passage[passage_id], axis=0, dtype=np.float32)
                compact_passage[claim_id, layer, passage_id] = heads_value.mean(dtype=np.float32)
            for sentence_id in range(s):
                claim_sentence[claim_id, sentence_id, layer] = np.mean(
                    per_sentence[sentence_id], axis=0, dtype=np.float32)
    return {
        "claim_sentence": claim_sentence,
        "source_total_relevance": compact_source,
        "previous_answer_relevance": compact_previous,
        "other_context_relevance": compact_other,
        "passage_relevance": compact_passage,
        "sentence_relevance": claim_sentence.mean(axis=-1, dtype=np.float32).transpose(0, 2, 1),
    }


def cpu_check() -> dict:
    global OUT
    assert_cpu_only()
    plans, layouts, signature, preparation = load_prepared()
    assert len(plans) == len(layouts) == EXPECTED_ANSWERS
    assert sum(len(x["claims"]) for x in layouts) == EXPECTED_CLAIMS
    assert sum(x["claim_sentence_edges"] for x in layouts) == EXPECTED_CLAIM_SENTENCE_EDGES

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).cpu().eval()
    real_membership_max_error = 0.0
    real_claim_lengths = Counter()
    for real_layout in layouts:
        geometry = SemanticAttributionHooks(model, real_layout, query_batch=2)
        sums = geometry.membership.cpu().numpy().sum(axis=1, dtype=np.float32)
        real_membership_max_error = max(
            real_membership_max_error, float(np.max(np.abs(sums - 1.0))))
        real_claim_lengths.update(
            len(claim["lexical_absolute_token_positions"])
            for claim in real_layout["claims"])
        del geometry
    assert real_claim_lengths[26] > 0
    assert real_membership_max_error <= 5e-7
    ids = [1] + [2 + ((index * 7) % 120) for index in range(47)]
    answer_positions = list(range(18, 48))
    layout = {
        "response_id": "tiny",
        "sequence_length": len(ids),
        "answer_token_positions": answer_positions,
        "source_sentence_union_token_positions": [2, 3, 4, 6, 7],
        "passage_token_positions": [[2, 3], [4], [6, 7]],
        "other_context_token_positions": [0, 1, 5, 8, 9],
        "sentences": [
            {"sentence_index": 0, "passage_id": 1, "sentence_id": 0,
             "token_positions": [2, 3], "text_sha256": "00" * 32},
            {"sentence_index": 1, "passage_id": 2, "sentence_id": 0,
             "token_positions": [4], "text_sha256": "11" * 32},
            {"sentence_index": 2, "passage_id": 3, "sentence_id": 0,
             "token_positions": [6, 7], "text_sha256": "22" * 32},
        ],
        "claims": [
            {"claim_id": 0, "microclaim_index": 0,
             "lexical_absolute_token_positions": [18, 20]},
            {"claim_id": 1, "microclaim_index": 1,
             "lexical_absolute_token_positions": [20, 21, 23]},
            {"claim_id": 2, "microclaim_index": 2,
             "lexical_absolute_token_positions": answer_positions[:26]},
        ],
    }
    input_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        with SemanticAttributionHooks(model, layout, query_batch=2) as hooks:
            hooked_hidden = model.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_attentions=False, output_hidden_states=False,
                return_dict=True).last_hidden_state.detach().clone()
        actual = hooks.finish()
        capture = _ValueCapture(model)
        try:
            oracle_output = model.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_attentions=True, output_hidden_states=False,
                return_dict=True)
        finally:
            capture.close()
    assert len(oracle_output.attentions) == 2 and len(capture.values) == 2
    oracle = _dense_oracle(model, layout, oracle_output.attentions, capture.values)
    errors = {}
    actual_csr = actual["claim_sentence_mass"].astype(np.float32).reshape(3, 3, 2, 4)
    errors["claim_sentence_mass_float16"] = float(np.max(np.abs(
        actual_csr - oracle["claim_sentence"])))
    for key in ("source_total_relevance", "previous_answer_relevance",
                "other_context_relevance", "passage_relevance",
                "sentence_relevance"):
        errors[key] = float(np.max(np.abs(actual[key] - oracle[key])))
    assert errors["claim_sentence_mass_float16"] <= 0.002
    assert max(value for key, value in errors.items()
               if key != "claim_sentence_mass_float16") <= 2e-7, errors
    # Query position 18 has no previous answer token.  Claim 0 also has q=20,
    # so its result must equal half the q=20 strict-previous contribution.
    assert np.all(actual["previous_answer_relevance"] >= 0)
    validate = validate_arrays({
        **actual,
        "claim_ids": np.arange(3, dtype=np.int32),
        "microclaim_indices": np.arange(3, dtype=np.int32),
        "sentence_indices": np.arange(3, dtype=np.int32),
        "sentence_passage_ids": np.asarray([1, 2, 3], dtype=np.int8),
        "sentence_ids": np.zeros(3, dtype=np.int32),
        "sentence_identity_sha256": np.stack([
            np.frombuffer(bytes.fromhex(x["text_sha256"]), dtype=np.uint8)
            for x in layout["sentences"]]),
        "answer_token_positions": np.asarray(answer_positions, dtype=np.int64),
    }, layout, 2, 4)
    real_out = OUT
    with tempfile.TemporaryDirectory(prefix="semantic_attr_resume_cpu_") as directory:
        OUT = Path(directory)
        try:
            orphan_feature, orphan_metadata, orphan_commit = record_paths("tiny")
            orphan_feature.parent.mkdir(parents=True)
            orphan_feature.write_bytes(b"uncommitted feature")
            orphan_metadata.write_text("{}\n", encoding="utf-8")
            assert not orphan_commit.exists()
            assert read_committed({"response_id": "tiny"}, {}, {}) is None
            quarantined = quarantine_uncommitted("tiny")
            assert len(quarantined) == 2
            assert not orphan_feature.exists() and not orphan_metadata.exists()
            assert all((OUT / path).exists() for path in quarantined)
        finally:
            OUT = real_out
    result = {
        "status": "passed_ready_for_explicit_gpu_extract",
        "runner_sha256": sha(Path(__file__)),
        "signature_sha256": digest(signature),
        "preparation_sha256": sha(OUT / "PREPARATION.json"),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "layouts_sha256": sha(OUT / "layouts.jsonl"),
        "global_csr_index_sha256": sha(OUT / "global_csr_index.npz"),
        "tiny_model": {"layers": 2, "heads": 4, "kv_heads": 2,
                       "sequence_tokens": len(ids),
                       "long_claim_lexical_tokens": 26},
        "oracle": "explicit output_attentions rows multiplied by independently captured/repeated V-head L2 norms",
        "max_abs_errors": errors,
        "all_real_layout_membership_check": {
            "answers": len(layouts), "claims": sum(real_claim_lengths.values()),
            "claims_with_26_lexical_tokens": real_claim_lengths[26],
            "maximum_lexical_tokens": max(real_claim_lengths),
            "max_float32_row_sum_abs_error": real_membership_max_error,
        },
        "validation": validate,
        "post_token_query": True,
        "previous_answer_strictly_less_than_query": True,
        "query_self_excluded_from_previous_answer": True,
        "claim_sentence_csr_before_head_mean": True,
        "token_sentence_head_tensor_materialized": False,
        "interrupted_uncommitted_record_quarantine_checked": True,
        "pretrained_model_loaded": False,
        "gpu_used": False, "cuda_initialized": torch.cuda.is_initialized(),
        "labels_used": False, "official_test_opened": False,
    }
    frozen_json(OUT / "CPU_SELFCHECK.json", result)
    assert torch.equal(hooked_hidden, oracle_output.last_hidden_state)
    assert_cpu_only()
    print("SEMANTIC_ATTRIBUTION_CPU_SELFCHECK_PASSED", json.dumps(errors), flush=True)
    return result


def verify_model_assets(signature: dict) -> None:
    assert signature["model"]["manifest_sha256"] == sha(MODEL_MANIFEST_PATH)
    for filename, expected in signature["model"]["assets"].items():
        path = MODEL / filename
        assert path.stat().st_size == expected["bytes"]
        assert sha(path) == expected["sha256"], filename


def record_paths(response_id) -> tuple[Path, Path, Path]:
    feature = OUT / "features" / f"{response_id}.npz"
    return feature, feature.with_suffix(".json"), feature.with_suffix(".commit.json")


def read_committed(plan: dict, layout: dict, signature: dict,
                   load_arrays: bool = False):
    feature_path, metadata_path, commit_path = record_paths(plan["response_id"])
    present = [path.exists() for path in (feature_path, metadata_path, commit_path)]
    if not any(present):
        return None
    if not all(present):
        # The commit marker is authoritative.  A crash before it may leave an
        # otherwise valid shard/metadata pair, but that pair was never committed.
        # Extraction quarantines those exact paths and recomputes the answer.
        assert not commit_path.exists(), (
            f"Commit marker exists for an incomplete record: {feature_path}")
        return None
    commit = read_json(commit_path)
    metadata = read_json(metadata_path)
    assert commit["response_id"] == plan["response_id"]
    assert commit["metadata_sha256"] == sha(metadata_path)
    assert metadata["response_id"] == plan["response_id"]
    assert metadata["signature_sha256"] == digest(signature)
    assert metadata["plan_sha256"] == digest(plan)
    assert metadata["layout_sha256"] == digest(layout)
    assert metadata["npz_sha256"] == commit["npz_sha256"] == sha(feature_path)
    if not load_arrays:
        return metadata
    with np.load(feature_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    validate_arrays(arrays, layout, EXPECTED_MODEL["layers"], EXPECTED_MODEL["heads"])
    return arrays, metadata


def quarantine_uncommitted(response_id) -> list[str]:
    """Move exact uncommitted files aside; never modify a committed record."""
    feature_path, metadata_path, commit_path = record_paths(response_id)
    assert not commit_path.exists()
    candidates = [
        feature_path, metadata_path,
        feature_path.with_suffix(feature_path.suffix + ".pending"),
        metadata_path.with_suffix(metadata_path.suffix + ".pending"),
        commit_path.with_suffix(commit_path.suffix + ".pending"),
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return []
    quarantine = OUT / "uncommitted" / str(response_id)
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in existing:
        destination = quarantine / f"{path.name}.{sha(path)}"
        if destination.exists():
            assert sha(destination) == sha(path)
            path.unlink()
        else:
            path.replace(destination)
        moved.append(str(destination.relative_to(OUT)))
    return moved


def save_record(plan: dict, layout: dict, signature: dict,
                arrays: dict[str, np.ndarray], seconds: float) -> dict:
    audit = validate_arrays(arrays, layout, EXPECTED_MODEL["layers"], EXPECTED_MODEL["heads"])
    feature_path, metadata_path, commit_path = record_paths(plan["response_id"])
    assert not any(path.exists() for path in (feature_path, metadata_path, commit_path))
    atomic_npz(feature_path, arrays)
    metadata = {
        "status": "complete", "version": VERSION,
        "response_id": plan["response_id"], "source_id": plan["source_id"],
        "group_id": plan["group_id"], "partition": plan["partition"],
        "signature_sha256": digest(signature),
        "plan_sha256": digest(plan), "layout_sha256": digest(layout),
        "npz_sha256": sha(feature_path), "seconds": seconds,
        **audit, "labels_used": False, "official_test_opened": False,
    }
    atomic_json(metadata_path, metadata)
    commit = {
        "status": "committed", "response_id": plan["response_id"],
        "npz_sha256": metadata["npz_sha256"],
        "metadata_sha256": sha(metadata_path),
    }
    atomic_json(commit_path, commit)
    assert read_committed(plan, layout, signature) == metadata
    return metadata


@contextmanager
def exclusive_gpu():
    descriptor = None
    acquired = False
    try:
        descriptor = os.open(str(GLOBAL_GPU_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        acquired = True
        os.write(descriptor, f"{VERSION}:{os.getpid()}".encode("utf-8"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired and GLOBAL_GPU_LOCK.exists():
            GLOBAL_GPU_LOCK.unlink()


def extract(limit: int | None = None) -> dict:
    plans, layouts, signature, _ = load_prepared()
    cpu = read_json(OUT / "CPU_SELFCHECK.json")
    assert cpu["status"] == "passed_ready_for_explicit_gpu_extract"
    assert cpu["runner_sha256"] == sha(Path(__file__))
    assert cpu["signature_sha256"] == digest(signature)
    assert torch.cuda.is_available(), "CUDA GPU is required for NF4 extraction"
    assert shutil.disk_usage(OUT).free > 2 * 1024 ** 3
    verify_model_assets(signature)
    existing = [read_committed(plan, layout, signature)
                for plan, layout in zip(plans, layouts)]
    missing = [index for index, item in enumerate(existing) if item is None]
    recovered = {
        str(plans[index]["response_id"]): quarantine_uncommitted(
            plans[index]["response_id"])
        for index in missing
    }
    recovered = {key: value for key, value in recovered.items() if value}
    if limit is not None:
        assert limit > 0
        missing = missing[:limit]
    model = None
    records = [item for item in existing if item is not None]
    started = time.perf_counter()
    with exclusive_gpu():
        try:
            if missing:
                model, load_metadata = llama_runner.load_nf4()
                spec = fq.architecture(model)
                assert spec["layers"] == EXPECTED_MODEL["layers"]
                assert spec["heads"] == EXPECTED_MODEL["heads"]
                assert spec["head_dim"] == EXPECTED_MODEL["head_dim"]
                atomic_json(OUT / "extract_started.json", {
                    "status": "running", "version": VERSION,
                    "signature_sha256": digest(signature),
                    "missing_at_start": len(missing), "limit": limit,
                    "quarantined_uncommitted_records": recovered,
                    "model_load": load_metadata,
                    "labels_used": False, "official_test_opened": False,
                })
                for completed, index in enumerate(missing, 1):
                    one_started = time.perf_counter()
                    arrays = extract_one(model, plans[index], layouts[index])
                    record = save_record(plans[index], layouts[index], signature,
                                         arrays, time.perf_counter() - one_started)
                    records.append(record)
                    atomic_json(OUT / "progress.json", {
                        "status": "running", "completed_this_invocation": completed,
                        "scheduled_this_invocation": len(missing),
                        "records_committed_total": sum(
                            record_paths(p["response_id"])[2].exists() for p in plans),
                        "last_response_id": plans[index]["response_id"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "labels_used": False, "official_test_opened": False,
                    })
                    print("SEMANTIC_ATTRIBUTION_EXTRACTED", completed, len(missing),
                          plans[index]["response_id"], flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    committed = [read_committed(plan, layout, signature)
                 for plan, layout in zip(plans, layouts)]
    complete = all(item is not None for item in committed)
    report = {
        "status": "complete" if complete else "partial_resumable",
        "records_complete": sum(item is not None for item in committed),
        "records_total": EXPECTED_ANSWERS,
        "claims_complete": sum(item["claims"] for item in committed if item),
        "claim_sentence_edges_complete": sum(item["edges"] for item in committed if item),
        "signature_sha256": digest(signature),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "labels_used": False, "official_test_opened": False,
        "training_or_scoring_run": False,
    }
    atomic_json(OUT / "status.json", report)
    if complete:
        manifest = {
            "status": "complete",
            "version": VERSION,
            "records_complete": EXPECTED_ANSWERS,
            "records_total": EXPECTED_ANSWERS,
            "claims_complete": EXPECTED_CLAIMS,
            "claim_sentence_edges_complete": EXPECTED_CLAIM_SENTENCE_EDGES,
            "logical_claim_sentence_mass_shape": [
                EXPECTED_CLAIM_SENTENCE_EDGES,
                EXPECTED_MODEL["layers"] * EXPECTED_MODEL["heads"],
            ],
            "global_claim_indptr_shape": [EXPECTED_CLAIMS + 1],
            "global_csr_index_sha256": sha(OUT / "global_csr_index.npz"),
            "signature_sha256": digest(signature),
            "labels_used": False, "official_test_opened": False,
            "training_or_scoring_run": False,
            "records": committed,
            "files": {plan["response_id"]: {
                "npz_sha256": item["npz_sha256"],
                "metadata_sha256": sha(record_paths(plan["response_id"])[1]),
                "commit_sha256": sha(record_paths(plan["response_id"])[2]),
            } for plan, item in zip(plans, committed)},
        }
        frozen_json(OUT / "feature_manifest.json", manifest)
    print("SEMANTIC_ATTRIBUTION_STATUS", json.dumps(report), flush=True)
    return report


def status() -> dict:
    plans, layouts, signature, preparation = load_prepared()
    cpu_path = OUT / "CPU_SELFCHECK.json"
    cpu_valid = False
    if cpu_path.exists():
        cpu = read_json(cpu_path)
        cpu_valid = (cpu.get("status") == "passed_ready_for_explicit_gpu_extract" and
                     cpu.get("runner_sha256") == sha(Path(__file__)) and
                     cpu.get("signature_sha256") == digest(signature))
    committed = []
    for plan, layout in zip(plans, layouts):
        committed.append(read_committed(plan, layout, signature))
    report = {
        "version": VERSION,
        "prepared": preparation["status"] == "prepared_waiting_for_cpu_check",
        "cpu_selfcheck_valid": cpu_valid,
        "records_complete": sum(item is not None for item in committed),
        "records_total": EXPECTED_ANSWERS,
        "claims_complete": sum(item["claims"] for item in committed if item),
        "claim_sentence_edges_complete": sum(item["edges"] for item in committed if item),
        "gpu_started": (OUT / "extract_started.json").exists(),
        "feature_manifest_complete": (OUT / "feature_manifest.json").exists(),
        "labels_used": False, "official_test_opened": False,
        "training_or_scoring_run": False,
    }
    atomic_json(OUT / "status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("cpu-check")
    extraction = sub.add_parser("extract")
    extraction.add_argument("--limit", type=int)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "cpu-check":
        cpu_check()
    elif args.command == "extract":
        extract(args.limit)
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
