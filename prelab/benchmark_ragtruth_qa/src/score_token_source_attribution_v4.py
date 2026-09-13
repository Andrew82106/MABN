"""Label-separated NLI materialization and direct 4-BPE scoring for v4.

Stages before ``fit`` are label blind.  ``fit`` opens only the native fit634
annotation files and freezes the model plus thresholds from source-group OOF.
``evaluate`` then opens calibration159 exactly once.  No official-test path is
present and no formal baseline artifact is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_atomic_microclaim_nli_v1 as atomic_nli  # noqa: E402
import run_token_source_attribution_v4 as extractor  # noqa: E402


VERSION = "token-source-attribution-v4-score"
OUT = ROOT / "results/token_source_attribution_v4_score"
ATTR_OUT = ROOT / "results/token_source_attribution_v4"
ATOMIC_ROWS = ROOT / "results/atomic_microclaim_nli_v1/inputs.jsonl"
PAIR_DIR = ROOT / "results/atomic_microclaim_nli_v1/pair_scores"
SEMANTIC_V1_SCORE = ROOT / "results/semantic_source_attribution_v1_score"
OOF_INPUT = ROOT / "results/group_crossfit_lb_large_v1/oof_input_scores.npy"
OOF_CONTROL = ROOT / "results/group_crossfit_lb_large_v1/group_oof.json"
OOF_PROTOCOL = ROOT / "results/group_crossfit_lb_large_v1/protocol.json"
LOOKBACK_PROTOCOL = ROOT / "results/lookback_regularization_v2/protocol.json"
LARGE_PROTOCOL = ROOT / "results/full_context_encoder_large_v1/protocol.json"
LARGE_COMPLETE = ROOT / "results/full_context_encoder_large_v1/full_finetune/complete.json"
GENERATION_BASE = ROOT / "results/development_v1/matrices/base.npy"
GENERATION_MANIFEST = ROOT / "results/development_v1/matrix_manifest.json"
GENERATION_FEATURE_MANIFEST = ROOT / "data/feature_manifest.json"
GENERATION_FEATURE_SIGNATURE = ROOT / "data/feature_signature.json"
GENERATION_FEATURE_DIR = ROOT / "data/features"
BASELINE_FILES = (
    ROOT / "FORMAL_BASELINE_RESULTS.md",
    ROOT / "BASELINE_PROTOCOL.md",
    ROOT / "results/group_crossfit_lb_large_v1/group_oof.json",
    ROOT / "results/group_crossfit_lb_large_v1/oof_input_scores.npy",
)
EXPECTED = {
    "answers": {"fit": 634, "calibration": 159},
    "groups": {"fit": 615, "calibration": 154},
    "claims": {"fit": 9055, "calibration": 2267},
    "windows": {"fit": 168123, "calibration": 42241},
    "lexical_bpe": {"fit": 139518, "calibration": 35000},
    "claim_sentence_edges": 166644,
}
LAYERS = 32
TOKEN_WIDTH = 8
LAYER_BANDS = 4
LAYERS_PER_BAND = LAYERS // LAYER_BANDS
FULL_ATTR_WINDOW_WIDTH = LAYERS * TOKEN_WIDTH * 2 + 1
BAND_ATTR_WINDOW_WIDTH = LAYER_BANDS * TOKEN_WIDTH * 2 + 1
FOLDS = 5
THREADS = 4
SEED = 20260913
C = 0.01
CANDIDATE_SPECS = {
    "token_attr_full_lr": {"layer_bands": False, "generation_nll": False,
                           "width": FULL_ATTR_WINDOW_WIDTH},
    "token_attr_full_nll_lr": {"layer_bands": False, "generation_nll": True,
                               "width": FULL_ATTR_WINDOW_WIDTH + 1},
    "token_attr_band4_lr": {"layer_bands": True, "generation_nll": False,
                            "width": BAND_ATTR_WINDOW_WIDTH},
    "token_attr_band4_nll_lr": {"layer_bands": True, "generation_nll": True,
                                "width": BAND_ATTR_WINDOW_WIDTH + 1},
}
CANDIDATES = tuple(CANDIDATE_SPECS)
assert LAYERS % LAYER_BANDS == 0


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


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "CPU stage initialized CUDA"


def protocol() -> dict:
    return {
        "version": VERSION,
        "scope": "Native RAGTruth QA fit634 plus calibration159 only.",
        "label_separation": {
            "initialize_link_materialize": "Only label-free plans, layouts, white-box caches, and frozen NLI probabilities are opened.",
            "fit": "Only answers/tokens/windows_fit are opened.",
            "evaluate": "Calibration annotations are opened once after fit_complete freezes candidate, model, and thresholds.",
            "official_test": "No official-test path occurs in this module.",
        },
        "nli": {
            "checkpoint": "tasksource/ModernBERT-base-nli",
            "revision": atomic_nli.MODEL_REVISION,
            "classes": ["support_entailment", "neutral", "conflict_contradiction"],
            "coverage": "Every exact frozen claim x source-sentence edge, not only claim-average top3.",
            "reuse": "Reuse only bit-identical probabilities for the exact checkpoint/revision/premise/hypothesis request; ambiguous duplicates are recomputed.",
            "combination": "For each token and layer, normalize all source-sentence attribution and take its weighted E/N/C average.",
        },
        "per_token_per_layer_features": [
            "source_share", "strict_previous_answer_share", "other_context_share",
            "attention_weighted_support", "attention_weighted_neutral",
            "attention_weighted_conflict", "top1_sentence_concentration",
            "top3_sentence_concentration"],
        "window_design": {
            "rows": "Every unchanged eligible 4-raw-BPE stride-one window; short answers retain their original shorter window.",
            "attribution": "Mean and max over the lexical BPE rows touched by that exact window, plus lexical-token fraction.",
            "registered_representations": "Full retains 32x8; band4 first averages each consecutive eight-layer band per token and retains 4x8. Both use token mean and max.",
            "fusion": "The formal optional column is frozen label-blind window mean NLL.",
            "excluded_diagnostics": "Existing Lookback/ModernBERT-large OOF columns are excluded from every formal candidate because their fixed C/epoch were historically selected on this same calibration159 split.",
            "no_claim_projection": True,
        },
        "training": {
            "candidates": list(CANDIDATES),
            "model": f"StandardScaler plus L2 liblinear logistic regression, fixed C={C}",
            "crossfit": f"{FOLDS}-fold GroupKFold by source-connected group.",
            "weights": "Equal source-group mass, answer mass within group, and window mass within answer; then one fit-only binary-class balance and global scale.",
            "selection": "Fit OOF only: maximize min(window F1, answer F1), then window F1, answer F1, precision, then earlier candidate.",
            "thresholds": "Window and answer thresholds are chosen on fit OOF and frozen.",
        },
        "calibration": "One strict evaluation with fit-frozen model and thresholds; no cal-F1Opt and no refit.",
        "formal_baselines_modified": False, "official_test_opened": False,
    }


def baseline_snapshot() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in BASELINE_FILES}


def generation_snapshot() -> dict[str, str]:
    manifest = read_json(GENERATION_MANIFEST)
    assert manifest["files_sha256"]["base"] == sha(GENERATION_BASE)
    features = read_json(GENERATION_FEATURE_MANIFEST)
    signature = read_json(GENERATION_FEATURE_SIGNATURE)
    assert features["complete"] and features["completed_records"] == 793
    assert features["signature_sha256"] == digest(signature)
    assert features["annotation_values_accessed"] is False
    return {"formal_nll_source": "Per-answer label-blind generation feature shards; recomputed by exact window token indices.",
            "feature_manifest_sha256": sha(GENERATION_FEATURE_MANIFEST),
            "feature_signature_sha256": sha(GENERATION_FEATURE_SIGNATURE),
            "reference_base_sha256": sha(GENERATION_BASE),
            "reference_matrix_manifest_sha256": sha(GENERATION_MANIFEST)}


def excluded_upstream_snapshot() -> dict:
    crossfit = read_json(OOF_PROTOCOL)
    lookback = read_json(LOOKBACK_PROTOCOL)
    large = read_json(LARGE_PROTOCOL)
    large_complete = read_json(LARGE_COMPLETE)
    assert "Historical choice of3 inherited from previous calibration-selected" in crossfit["large"]["selection"]
    assert "calibration" in lookback["selection"].lower()
    assert "calibration-only" in large["selection"]
    assert int(large_complete["selected"]["epoch"]) == 3
    return {
        "formal_candidate_columns": [],
        "excluded_columns": ["oof_lookback", "oof_modernbert_large"],
        "reason": "Both fixed upstream hyperparameters used this same calibration159 split: Lookback C and ModernBERT-large epoch3.",
        "oof_values_used_for_formal_fit_or_selection": False,
        "files_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            OOF_INPUT, OOF_CONTROL, OOF_PROTOCOL, LOOKBACK_PROTOCOL,
            LARGE_PROTOCOL, LARGE_COMPLETE)},
    }


def initialize() -> dict:
    assert_cpu_only()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / "protocol.json", protocol())
    snapshot = {
        "version": VERSION, "runner_sha256": sha(Path(__file__)),
        "extractor_runner_sha256": sha(Path(extractor.__file__)),
        "extractor_protocol_sha256": sha(ATTR_OUT / "protocol.json"),
        "extractor_signature_sha256": sha(ATTR_OUT / "signature.json"),
        "atomic_inputs_sha256": sha(ATOMIC_ROWS),
        "atomic_runner_sha256": sha(Path(atomic_nli.__file__)),
        "generation_nll_source": generation_snapshot(),
        "excluded_upstream": excluded_upstream_snapshot(),
        "baseline_files_sha256": baseline_snapshot(),
        "fit_label_files_opened": False,
        "calibration_label_files_opened": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / "source_snapshot.json", snapshot)
    frozen_json(OUT / "EXCLUDED_UPSTREAM.json", snapshot["excluded_upstream"])
    raw_estimate = read_json(ATTR_OUT / "RESOURCE_ESTIMATE.json")
    resource = {
        "native_raw_attribution_uncompressed_bytes": raw_estimate["raw_array_bytes"]["total"],
        "native_raw_attribution_compressed_planning_bytes": raw_estimate["compressed_npz_planning_range_bytes"],
        "materialized_token_features_uncompressed_bytes": sum(
            EXPECTED["lexical_bpe"].values()) * LAYERS * TOKEN_WIDTH * 2,
        "fit_shared_full_plus_band4_matrices_bytes": EXPECTED["windows"]["fit"] * (
            FULL_ATTR_WINDOW_WIDTH + 1 + BAND_ATTR_WINDOW_WIDTH + 1) * 4,
        "maximum_selected_calibration_matrix_bytes": EXPECTED["windows"]["calibration"] * (
            FULL_ATTR_WINDOW_WIDTH + 1) * 4,
        "gpu_minutes": {"all_sentence_nli_missing_requests": [7, 10],
                        "llama2_nf4_token_attribution": raw_estimate["gpu_time_native_minutes"]},
        "cpu_minutes_after_gpu": [8, 30],
        "minimum_free_disk_bytes_enforced_before_llama": 2 * 1024 ** 3,
        "planning_ranges_not_benchmarks": True,
    }
    frozen_json(OUT / "RESOURCE_ESTIMATE.json", resource)
    plan = (
        "# Token source attribution v4 scoring\n\n"
        "先在无标签阶段为全部 claim×source-sentence 冻结 NLI，并把白盒 shard 转成每 token、每层 8 维。随后直接以原 4-BPE 窗口为监督行。\n\n"
        "- fit：按 source group 五折 OOF；候选、模型和双阈值只看 fit OOF。\n"
        "- 四个固定候选：full 513/full+NLL 514、band4 65/band4+NLL 66；band4 对每个 token 的连续 8 层先求均值。\n"
        "- 全部候选固定 C=0.01。现有 Lookback/large OOF 的 C/epoch 曾由同一 cal159 选择，故只登记为污染诊断来源，禁止进入正式候选。\n"
        "- calibration：冻结后只执行一次严格评测，不计算 cal-F1Opt。\n"
        "- official test：无读取入口；正式 baseline 文件只校验哈希。\n")
    plan_path = OUT / "PLAN.md"
    if plan_path.exists():
        assert plan_path.read_text(encoding="utf-8") == plan
    else:
        plan_path.write_text(plan, encoding="utf-8")
    report = {
        "status": "initialized_label_free",
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "candidate_widths": {name: spec["width"]
                             for name, spec in CANDIDATE_SPECS.items()},
        "fit_label_files_opened": False,
        "calibration_label_files_opened": False,
        "gpu_used": False, "formal_baselines_modified": False,
        "official_test_opened": False}
    frozen_json(OUT / "INITIALIZED.json", report)
    print("TOKEN_ATTRIBUTION_V4_SCORE_INITIALIZED", flush=True)
    return report


def check_initialized() -> dict:
    initialized = read_json(OUT / "INITIALIZED.json")
    snapshot = read_json(OUT / "source_snapshot.json")
    assert snapshot["runner_sha256"] == sha(Path(__file__))
    assert snapshot["extractor_runner_sha256"] == sha(Path(extractor.__file__))
    assert snapshot["extractor_protocol_sha256"] == sha(ATTR_OUT / "protocol.json")
    assert snapshot["extractor_signature_sha256"] == sha(ATTR_OUT / "signature.json")
    assert snapshot["atomic_inputs_sha256"] == sha(ATOMIC_ROWS)
    assert snapshot["atomic_runner_sha256"] == sha(Path(atomic_nli.__file__))
    assert snapshot["generation_nll_source"] == generation_snapshot()
    assert snapshot["excluded_upstream"] == excluded_upstream_snapshot()
    assert snapshot["baseline_files_sha256"] == baseline_snapshot()
    assert initialized["source_snapshot_sha256"] == sha(OUT / "source_snapshot.json")
    return snapshot


def nli_request_id(premise: str, hypothesis: str) -> str:
    return digest({
        "checkpoint": "tasksource/ModernBERT-base-nli",
        "revision": atomic_nli.MODEL_REVISION,
        "premise": premise, "hypothesis": hypothesis})


def edge_id(row: dict, layout: dict, claim: dict, layout_claim: dict,
            sentence: dict) -> str:
    assert layout["atomic_row_sha256"] == digest(row)
    assert int(claim["claim_id"]) == int(layout_claim["claim_id"])
    assert claim["microclaim_id"] == layout_claim["microclaim_id"]
    assert digest(claim["hypothesis"]) == claim["hypothesis_sha256"]
    return digest({
        "response_id": row["response_id"],
        "claim_id": int(claim["claim_id"]),
        "microclaim_id": claim["microclaim_id"],
        "hypothesis_sha256": claim["hypothesis_sha256"],
        "sentence_index": int(sentence["sentence_index"]),
        "passage_id": int(sentence["passage_id"]),
        "sentence_id": int(sentence["sentence_id"]),
        "text_sha256": sentence["text_sha256"],
    })


def semantic_v1_probability_catalog() -> dict[str, list[np.ndarray]]:
    """Read the already frozen v1 top3 NLI without touching labels."""
    catalog: dict[str, list[np.ndarray]] = defaultdict(list)
    for partition in ("fit", "calibration"):
        existing_path = SEMANTIC_V1_SCORE / f"selected_nli_existing_{partition}.npz"
        missing_path = SEMANTIC_V1_SCORE / f"selected_nli_missing_scores_{partition}.npz"
        links_path = SEMANTIC_V1_SCORE / f"selected_nli_links_{partition}.jsonl"
        if not (existing_path.is_file() and missing_path.is_file() and links_path.is_file()):
            continue
        with np.load(existing_path, allow_pickle=False) as loaded:
            probabilities = loaded["probabilities"].copy()
            identities = loaded["request_identity_sha256"].copy()
        with np.load(missing_path, allow_pickle=False) as loaded:
            missing = {key.tobytes(): value.copy() for key, value in zip(
                loaded["request_identity_sha256"], loaded["probabilities"])}
        for claim, rank in zip(*np.where(np.isnan(probabilities).any(axis=2))):
            probabilities[claim, rank] = missing[identities[claim, rank].tobytes()]
        assert np.isfinite(probabilities).all()
        seen_claims = 0
        for answer in json_lines(links_path):
            for claim in answer["claims"]:
                global_claim = int(claim["global_claim_index"])
                assert global_claim == seen_claims
                for selected in claim["selected"]:
                    rank = int(selected["rank"]) - 1
                    request_id = selected["request_id"]
                    assert bytes.fromhex(request_id) == identities[global_claim, rank].tobytes()
                    catalog[request_id].append(probabilities[global_claim, rank].copy())
                seen_claims += 1
        assert seen_claims == EXPECTED["claims"][partition]
    return catalog


def link_nli() -> dict:
    """Enumerate all exact claim-sentence requests and reuse safe frozen values."""
    assert_cpu_only()
    check_initialized()
    config, plans, layouts, _, _ = extractor.load_prepared("native")
    rows = json_lines(ATOMIC_ROWS)
    assert len(rows) == len(layouts) == config["answers"]
    assert [row["response_id"] for row in rows] == [row["response_id"] for row in layouts]
    semantic_catalog = semantic_v1_probability_catalog()
    request_index: dict[str, int] = {}
    request_rows: list[dict] = []
    request_candidates: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, values in semantic_catalog.items():
        request_candidates[key].extend(values)
    edge_request_indices, edge_identities = [], []
    response_edge_indptr = [0]
    partition_edges = Counter()
    for answer_index, (row, layout) in enumerate(zip(rows, layouts)):
        extractor.reject_annotation_keys({key: value for key, value in row.items()
                                          if key != "labels_used"})
        assert row["labels_used"] is False and row["official_test_opened"] is False
        assert row["response_id"] == layout["response_id"]
        assert layout["atomic_row_sha256"] == digest(row)
        probability, _ = atomic_nli.validate_cache(
            PAIR_DIR / f"{row['response_id']}.npz", row)
        pairs, owners = atomic_nli.row_pairs(row)
        cached = {}
        for pair_index, owner in enumerate(owners):
            claim_id, passage_id, rank, sentence_id, second_sentence_id = map(int, owner)
            if rank <= 0:
                continue
            assert second_sentence_id == -1
            key = (claim_id, passage_id, sentence_id)
            assert key not in cached
            cached[key] = probability[pair_index].copy()
            request_candidates[nli_request_id(*pairs[pair_index])].append(
                probability[pair_index].copy())
        evidence = {(int(passage["passage_id"]), int(sentence["sentence_id"])): sentence
                    for passage in row["passages"] for sentence in passage["sentences"]}
        assert len(row["claims"]) == len(layout["claims"])
        for claim, layout_claim in zip(row["claims"], layout["claims"]):
            assert int(claim["claim_id"]) == int(layout_claim["claim_id"])
            assert claim["microclaim_id"] == layout_claim["microclaim_id"]
            for sentence in layout["sentences"]:
                sentence_key = (int(sentence["passage_id"]), int(sentence["sentence_id"]))
                source = evidence[sentence_key]
                assert source["text_sha256"] == sentence["text_sha256"]
                premise, hypothesis = source["text"], claim["hypothesis"]
                identity = nli_request_id(premise, hypothesis)
                if identity not in request_index:
                    request_index[identity] = len(request_rows)
                    request_rows.append({
                        "request_index": len(request_rows), "request_id": identity,
                        "premise": premise, "hypothesis": hypothesis})
                edge_request_indices.append(request_index[identity])
                edge_identities.append(np.frombuffer(bytes.fromhex(edge_id(
                    row, layout, claim, layout_claim, sentence)), dtype=np.uint8))
                cached_key = (int(claim["claim_id"]), *sentence_key)
                if cached_key in cached:
                    request_candidates[identity].append(cached[cached_key].copy())
                partition_edges[row["partition"]] += 1
        response_edge_indptr.append(len(edge_request_indices))
        if (answer_index + 1) % 100 == 0:
            print("TOKEN_ATTRIBUTION_NLI_LINK", answer_index + 1, len(rows), flush=True)
    assert len(edge_request_indices) == EXPECTED["claim_sentence_edges"]
    existing = np.full((len(request_rows), 3), np.nan, dtype=np.float32)
    ambiguous = 0
    for identity, index in request_index.items():
        candidates = request_candidates.get(identity, [])
        if not candidates:
            continue
        first = np.asarray(candidates[0], dtype=np.float32)
        if all(np.array_equal(first, np.asarray(value, dtype=np.float32))
               for value in candidates[1:]):
            existing[index] = first
        else:
            ambiguous += 1
    existing_mask = np.isfinite(existing).all(axis=1)
    assert np.allclose(existing[existing_mask].sum(1), 1, atol=2e-6, rtol=0)
    missing_rows = [row for row in request_rows if not existing_mask[row["request_index"]]]
    atomic_jsonl(OUT / "nli_requests_missing.jsonl", missing_rows)
    arrays = {
        "edge_request_indices": np.asarray(edge_request_indices, dtype=np.int32),
        "edge_identity_sha256": np.stack(edge_identities),
        "response_edge_indptr": np.asarray(response_edge_indptr, dtype=np.int64),
        "request_identity_sha256": np.stack([
            np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8)
            for row in request_rows]),
        "existing_probabilities": existing,
    }
    atomic_npz(OUT / "nli_link.npz", arrays)
    report = {
        "status": "linked_label_free_waiting_for_missing_nli" if missing_rows else "linked_all_reused",
        "answers": len(rows), "claim_sentence_edges": len(edge_request_indices),
        "edges_by_partition": dict(partition_edges),
        "unique_requests": len(request_rows),
        "reused_unique_requests": int(existing_mask.sum()),
        "missing_unique_requests": len(missing_rows),
        "ambiguous_existing_requests_recompute": ambiguous,
        "semantic_v1_catalog_requests": len(semantic_catalog),
        "nli_link_sha256": sha(OUT / "nli_link.npz"),
        "missing_requests_sha256": sha(OUT / "nli_requests_missing.jsonl"),
        "fit_label_files_opened": False,
        "calibration_label_files_opened": False,
        "gpu_used": False, "formal_baselines_modified": False,
        "official_test_opened": False}
    frozen_json(OUT / "NLI_LINK.json", report)
    print("TOKEN_ATTRIBUTION_NLI_LINKED", len(request_rows), len(missing_rows), flush=True)
    return report


def infer_nli() -> dict:
    """Explicit GPU stage; never called by initialize/link/materialize/status."""
    check_initialized()
    link = read_json(OUT / "NLI_LINK.json")
    requests = json_lines(OUT / "nli_requests_missing.jsonl")
    assert len(requests) == link["missing_unique_requests"]
    assert link["nli_link_sha256"] == sha(OUT / "nli_link.npz")
    assert link["missing_requests_sha256"] == sha(OUT / "nli_requests_missing.jsonl")
    with np.load(OUT / "nli_link.npz", allow_pickle=False) as loaded:
        all_identities = loaded["request_identity_sha256"].copy()
    for request in requests:
        assert request["request_id"] == nli_request_id(
            request["premise"], request["hypothesis"])
        assert bytes.fromhex(request["request_id"]) == all_identities[
            int(request["request_index"])].tobytes()
    assert torch.cuda.is_available()
    tokenizer = model = None
    started = time.perf_counter()
    with extractor.exclusive_gpu():
        try:
            tokenizer, model, load = atomic_nli.load_cuda()
            pairs = [(row["premise"], row["hypothesis"]) for row in requests]
            probability, shapes = atomic_nli.infer_pairs(
                tokenizer, model, pairs, torch.device("cuda:0")) if pairs else (
                    np.empty((0, 3), dtype=np.float32), [])
            memory = atomic_nli.memory_stats() if pairs else {}
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            atomic_nli.clean_gpu()
    identities = np.stack([
        np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8)
        for row in requests]) if requests else np.empty((0, 32), dtype=np.uint8)
    atomic_npz(OUT / "nli_missing_scores.npz", {
        "request_indices": np.asarray([row["request_index"] for row in requests], dtype=np.int32),
        "request_identity_sha256": identities,
        "probabilities": probability})
    report = {
        "status": "complete_frozen_probabilities_not_trained",
        "requests": len(requests), "checkpoint": "tasksource/ModernBERT-base-nli",
        "revision": atomic_nli.MODEL_REVISION, "model_sha256": atomic_nli.MODEL_SHA256,
        "load": load, "batch_shapes": shapes, "memory": memory,
        "seconds": time.perf_counter() - started,
        "npz_sha256": sha(OUT / "nli_missing_scores.npz"),
        "nli_link_sha256": link["nli_link_sha256"],
        "missing_requests_sha256": link["missing_requests_sha256"],
        "labels_used": False, "formal_baselines_modified": False,
        "official_test_opened": False}
    frozen_json(OUT / "NLI_INFERENCE.json", report)
    print("TOKEN_ATTRIBUTION_NLI_INFERRED", len(requests), flush=True)
    return report


def seal_nli() -> dict:
    assert_cpu_only()
    check_initialized()
    link_meta = read_json(OUT / "NLI_LINK.json")
    assert link_meta["nli_link_sha256"] == sha(OUT / "nli_link.npz")
    assert link_meta["missing_requests_sha256"] == sha(OUT / "nli_requests_missing.jsonl")
    with np.load(OUT / "nli_link.npz", allow_pickle=False) as loaded:
        edge = loaded["edge_request_indices"].copy()
        edge_identity = loaded["edge_identity_sha256"].copy()
        response_edge_indptr = loaded["response_edge_indptr"].copy()
        identities = loaded["request_identity_sha256"].copy()
        probability = loaded["existing_probabilities"].copy()
    missing_mask = ~np.isfinite(probability).all(axis=1)
    if missing_mask.any():
        inference = read_json(OUT / "NLI_INFERENCE.json")
        assert inference["npz_sha256"] == sha(OUT / "nli_missing_scores.npz")
        assert inference["nli_link_sha256"] == link_meta["nli_link_sha256"]
        assert inference["missing_requests_sha256"] == link_meta["missing_requests_sha256"]
        with np.load(OUT / "nli_missing_scores.npz", allow_pickle=False) as loaded:
            indices = loaded["request_indices"]
            missing_ids = loaded["request_identity_sha256"]
            values = loaded["probabilities"]
        assert np.array_equal(indices, np.flatnonzero(missing_mask))
        assert np.array_equal(identities[indices], missing_ids)
        probability[indices] = values
    assert np.isfinite(probability).all() and np.all(probability >= 0)
    assert np.allclose(probability.sum(1), 1, atol=2e-6, rtol=0)
    assert edge.shape == (EXPECTED["claim_sentence_edges"],)
    assert edge_identity.shape == (EXPECTED["claim_sentence_edges"], 32)
    assert response_edge_indptr.shape == (sum(EXPECTED["answers"].values()) + 1,)
    assert response_edge_indptr[0] == 0 and response_edge_indptr[-1] == len(edge)
    assert np.all(np.diff(response_edge_indptr) > 0)
    atomic_npz(OUT / "nli_bank.npz", {
        "edge_request_indices": edge,
        "edge_identity_sha256": edge_identity,
        "response_edge_indptr": response_edge_indptr,
        "request_identity_sha256": identities,
        "request_probabilities": probability})
    report = {
        "status": "sealed_all_claim_sentence_probabilities",
        "claim_sentence_edges": len(edge), "unique_requests": len(probability),
        "nli_bank_sha256": sha(OUT / "nli_bank.npz"),
        "nli_link_sha256": link_meta["nli_link_sha256"],
        "labels_used": False, "gpu_used": False,
        "formal_baselines_modified": False, "official_test_opened": False}
    frozen_json(OUT / "NLI_BANK.json", report)
    print("TOKEN_ATTRIBUTION_NLI_BANK_SEALED", len(edge), flush=True)
    return report


def materialized_paths(response_id: str):
    folder = OUT / "token_features"
    return (folder / f"{response_id}.npz", folder / f"{response_id}.json",
            folder / f"{response_id}.commit.json")


def validate_token_features(arrays: dict[str, np.ndarray], raw: dict[str, np.ndarray]) -> dict:
    required = {"token_features", "query_local_answer_indices",
                "query_absolute_positions", "query_claim_ids", "query_token_ids"}
    assert set(arrays) == required
    q = len(raw["query_local_answer_indices"])
    assert arrays["token_features"].shape == (q, LAYERS, TOKEN_WIDTH)
    assert arrays["token_features"].dtype == np.float16
    for key in required - {"token_features"}:
        assert np.array_equal(arrays[key], raw[key])
    values = arrays["token_features"].astype(np.float32)
    assert np.isfinite(values).all() and np.all(values >= 0) and np.all(values <= 1.001)
    region_sum = values[:, :, :3].sum(-1)
    region_active = region_sum > 0
    region_error = float(np.max(np.abs(region_sum[region_active] - 1))) if region_active.any() else 0.0
    nli_sum = values[:, :, 3:6].sum(-1)
    nli_active = nli_sum > 0
    nli_error = float(np.max(np.abs(nli_sum[nli_active] - 1))) if nli_active.any() else 0.0
    assert region_error <= 0.0011 and nli_error <= 0.0011
    return {"lexical_answer_BPE": q, "feature_width_per_layer": TOKEN_WIDTH,
            "raw_float16_bytes": arrays["token_features"].nbytes,
            "region_sum_max_abs_error": region_error,
            "nli_sum_max_abs_error": nli_error}


def read_materialized(response_id: str, raw: dict[str, np.ndarray] | None = None,
                      raw_npz_sha256: str | None = None,
                      nli_bank_sha256: str | None = None):
    feature_path, metadata_path, commit_path = materialized_paths(response_id)
    if not commit_path.exists():
        return None
    assert feature_path.is_file() and metadata_path.is_file()
    commit, metadata = read_json(commit_path), read_json(metadata_path)
    assert commit["response_id"] == metadata["response_id"] == response_id
    assert commit["npz_sha256"] == metadata["npz_sha256"] == sha(feature_path)
    assert commit["metadata_sha256"] == sha(metadata_path)
    if raw_npz_sha256 is not None:
        assert metadata["raw_npz_sha256"] == raw_npz_sha256
    if nli_bank_sha256 is not None:
        assert metadata["nli_bank_sha256"] == nli_bank_sha256
    if raw is None:
        return metadata
    with np.load(feature_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    validate_token_features(arrays, raw)
    return arrays, metadata


def materialize() -> dict:
    """Combine raw token attribution and all-sentence NLI, still label blind."""
    assert_cpu_only()
    check_initialized()
    config, plans, layouts, signature, _ = extractor.load_prepared("native")
    feature_manifest = read_json(ATTR_OUT / "feature_manifest.json")
    assert feature_manifest["status"] == "complete"
    bank_meta = read_json(OUT / "NLI_BANK.json")
    assert bank_meta["nli_bank_sha256"] == sha(OUT / "nli_bank.npz")
    with np.load(OUT / "nli_bank.npz", allow_pickle=False) as loaded:
        edge_requests = loaded["edge_request_indices"].copy()
        edge_identities = loaded["edge_identity_sha256"].copy()
        response_edge_indptr = loaded["response_edge_indptr"].copy()
        request_probability = loaded["request_probabilities"].copy()
    rows = json_lines(ATOMIC_ROWS)
    assert len(rows) == len(plans)
    completed = []
    for answer_index, (plan, layout, row) in enumerate(zip(plans, layouts, rows)):
        assert row["response_id"] == plan["response_id"] == layout["response_id"]
        assert layout["atomic_row_sha256"] == digest(row)
        raw, raw_meta = extractor.read_committed(
            config, plan, layout, signature, load_arrays=True)
        existing = read_materialized(
            plan["response_id"], raw, raw_meta["npz_sha256"],
            bank_meta["nli_bank_sha256"])
        c, s = len(layout["claims"]), len(layout["sentences"])
        assert len(row["claims"]) == c
        left, right = map(int, response_edge_indptr[answer_index:answer_index + 2])
        assert right - left == c * s
        local_edge = edge_requests[left:right]
        expected_edge_identities = np.stack([
            np.frombuffer(bytes.fromhex(edge_id(row, layout, claim, layout_claim, sentence)),
                          dtype=np.uint8)
            for claim, layout_claim in zip(row["claims"], layout["claims"])
            for sentence in layout["sentences"]])
        assert np.array_equal(edge_identities[left:right], expected_edge_identities)
        nli = request_probability[local_edge].reshape(c, s, 3)
        if existing is None:
            feature = extractor.combine_token_features(
                raw["token_region_share"], raw["token_sentence_share"],
                raw["query_claim_ids"], nli).astype(np.float16)
            arrays = {"token_features": feature,
                      **{key: raw[key] for key in (
                          "query_local_answer_indices", "query_absolute_positions",
                          "query_claim_ids", "query_token_ids")}}
            audit = validate_token_features(arrays, raw)
            feature_path, metadata_path, commit_path = materialized_paths(plan["response_id"])
            atomic_npz(feature_path, arrays)
            metadata = {
                "status": "complete", "version": VERSION,
                "response_id": plan["response_id"], "partition": plan["partition"],
                "raw_npz_sha256": raw_meta["npz_sha256"],
                "nli_bank_sha256": bank_meta["nli_bank_sha256"],
                "npz_sha256": sha(feature_path), **audit,
                "labels_used": False, "official_test_opened": False}
            atomic_json(metadata_path, metadata)
            atomic_json(commit_path, {
                "status": "committed", "response_id": plan["response_id"],
                "npz_sha256": metadata["npz_sha256"],
                "metadata_sha256": sha(metadata_path)})
            existing = arrays, metadata
        completed.append(existing[1])
        if (answer_index + 1) % 100 == 0:
            print("TOKEN_ATTRIBUTION_MATERIALIZED", answer_index + 1, len(plans), flush=True)
    assert int(response_edge_indptr[-1]) == EXPECTED["claim_sentence_edges"]
    assert sum(row["lexical_answer_BPE"] for row in completed) == sum(EXPECTED["lexical_bpe"].values())
    manifest = {
        "status": "complete_label_free_token_features",
        "answers": len(completed), "lexical_answer_BPE": sum(EXPECTED["lexical_bpe"].values()),
        "layers": LAYERS, "features_per_layer": TOKEN_WIDTH,
        "raw_float16_bytes": sum(row["raw_float16_bytes"] for row in completed),
        "attribution_manifest_sha256": sha(ATTR_OUT / "feature_manifest.json"),
        "nli_bank_sha256": bank_meta["nli_bank_sha256"],
        "records": completed,
        "labels_used": False, "formal_baselines_modified": False,
        "official_test_opened": False}
    frozen_json(OUT / "token_feature_manifest.json", manifest)
    print("TOKEN_ATTRIBUTION_FEATURES_COMPLETE", len(completed), flush=True)
    return manifest


def aggregate_window_features(token_features: np.ndarray,
                              query_local_indices: np.ndarray,
                              window_token_indices: list[int],
                              expected_lexical_indices: list[int] | None = None,
                              layer_bands: bool = False) -> np.ndarray:
    """Directly aggregate only the lexical BPEs touched by one frozen window."""
    query_local = np.asarray(query_local_indices, dtype=np.int32)
    positions = np.searchsorted(query_local, np.asarray(window_token_indices, dtype=np.int32))
    valid = (positions < len(query_local))
    matched = np.zeros(len(positions), dtype=bool)
    matched[valid] = query_local[positions[valid]] == np.asarray(window_token_indices, dtype=np.int32)[valid]
    matched_local = query_local[positions[matched]].tolist()
    if expected_lexical_indices is not None:
        assert matched_local == list(expected_lexical_indices)
    selected = token_features[positions[matched]].astype(np.float32)
    assert len(selected) > 0
    if layer_bands:
        selected = selected.reshape(
            len(selected), LAYER_BANDS, LAYERS_PER_BAND, TOKEN_WIDTH).mean(
                axis=2, dtype=np.float32)
    result = np.concatenate((selected.mean(0).reshape(-1),
                             selected.max(0).reshape(-1),
                             np.asarray([len(selected) / len(window_token_indices)], dtype=np.float32)))
    expected_width = BAND_ATTR_WINDOW_WIDTH if layer_bands else FULL_ATTR_WINDOW_WIDTH
    assert result.shape == (expected_width,) and np.isfinite(result).all()
    return result


def merge_intervals(values: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for left, right in sorted([list(map(int, pair)) for pair in values]):
        assert left <= right
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return merged


def load_generation_nll(response_id: str, token: dict, plan: dict,
                        record: dict, feature_signature_sha256: str) -> np.ndarray:
    path = GENERATION_FEATURE_DIR / f"{response_id}.npz"
    metadata_path = GENERATION_FEATURE_DIR / f"{response_id}.json"
    metadata = read_json(metadata_path)
    assert metadata["complete"] and metadata["response_id"] == response_id
    assert metadata["signature_sha256"] == feature_signature_sha256
    assert digest(plan) == record["plan_sha256"] == metadata["plan_sha256"]
    assert sha(path) == record["npz_sha256"] == metadata["npz_sha256"]
    assert sha(metadata_path) == record["metadata_sha256"]
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in (
            "nll", "token_ids", "answer_token_positions",
            "response_token_offsets", "response_token_offsets_raw")}
    assert arrays["nll"].shape == (token["token_count"],)
    assert arrays["nll"].dtype == np.float32
    assert np.isfinite(arrays["nll"]).all() and np.all(arrays["nll"] >= 0)
    for key in ("token_ids", "answer_token_positions", "response_token_offsets",
                "response_token_offsets_raw"):
        assert arrays[key].tolist() == token[key]
    return arrays["nll"]


def load_partition_meta(partition: str) -> dict:
    """Open exactly one annotation partition."""
    assert partition in ("fit", "calibration")
    answers = json_lines(ROOT / f"data/answers_{partition}.jsonl")
    tokens = json_lines(ROOT / f"data/tokens_{partition}.jsonl")
    windows = json_lines(ROOT / f"data/windows_k4_{partition}.jsonl")
    assert len(answers) == len(tokens) == EXPECTED["answers"][partition]
    assert len(windows) == EXPECTED["windows"][partition]
    assert [row["response_id"] for row in answers] == [row["response_id"] for row in tokens]
    _, plans, _, _, _ = extractor.load_prepared("native")
    plan_by_response = {row["response_id"]: row for row in plans if row["partition"] == partition}
    assert len(plan_by_response) == len(answers)
    feature_manifest = read_json(GENERATION_FEATURE_MANIFEST)
    feature_records = {row["response_id"]: row for row in feature_manifest["records"]}
    by_response = {answer["response_id"]: {"answer": answer, "tokens": token}
                   for answer, token in zip(answers, tokens)}
    nll_by_response = {}
    for answer, token in zip(answers, tokens):
        response_id = answer["response_id"]
        plan = plan_by_response[response_id]
        for key in ("response_id", "source_id", "group_id", "partition"):
            assert answer[key] == token[key] == plan[key]
        assert answer["answer_sha256"] == token["answer_sha256"] == plan["answer_sha256"]
        assert answer["original_response"] == token["original_response"] == plan["original_response"]
        original = plan["original"]
        for plan_key, token_key in (
                ("answer_token_ids", "token_ids"),
                ("answer_token_positions", "answer_token_positions"),
                ("response_token_offsets", "response_token_offsets"),
                ("response_token_offsets_raw", "response_token_offsets_raw")):
            assert original[plan_key] == token[token_key]
        nll_by_response[response_id] = load_generation_nll(
            response_id, token, plan, feature_records[response_id],
            feature_manifest["signature_sha256"])
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        assert window["partition"] == partition and window["eligible"]
        token = by_response[window["response_id"]]["tokens"]
        answer = by_response[window["response_id"]]["answer"]
        for key in ("response_id", "source_id", "group_id", "partition", "answer_id"):
            assert window[key] == token[key] == answer[key]
        ids = window["token_indices"]
        assert ids == list(range(window["token_start"], window["token_end"]))
        assert len(ids) == min(4, token["token_count"])
        assert any(token["lexical_mask"][item] for item in ids)
        lexical_ids = [item for item in ids if token["lexical_mask"][item]]
        assert window["lexical_token_indices"] == lexical_ids
        assert window["answer_token_positions"] == [token["answer_token_positions"][item]
                                                        for item in ids]
        assert window["token_ids"] == [token["token_ids"][item] for item in ids]
        intervals = merge_intervals([token["response_token_offsets"][item] for item in ids])
        assert window["character_intervals"] == intervals
        assert window["char_start"] == min(left for left, _ in intervals)
        assert window["char_end"] == max(right for _, right in intervals)
        assert window["bounding_text"] == token["original_response"][
            window["char_start"]:window["char_end"]]
        assert int(any(token["risk_mask"][item] for item in ids)) == int(window["label"])
        answer_windows[window["response_id"]].append(index)
    assert all(answer_windows[answer["response_id"]] for answer in answers)
    assert len({answer["group_id"] for answer in answers}) == EXPECTED["groups"][partition]
    return {"answers": answers, "tokens": tokens, "windows": windows,
            "by_response": by_response, "answer_windows": dict(answer_windows),
            "nll_by_response": nll_by_response}


def load_materialized_arrays(response_id: str, token: dict) -> dict[str, np.ndarray]:
    path, metadata_path, commit_path = materialized_paths(response_id)
    commit, metadata = read_json(commit_path), read_json(metadata_path)
    assert commit["metadata_sha256"] == sha(metadata_path)
    assert commit["npz_sha256"] == metadata["npz_sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    assert arrays["token_features"].shape[1:] == (LAYERS, TOKEN_WIDTH)
    local = arrays["query_local_answer_indices"].astype(np.int64)
    assert np.array_equal(local, np.flatnonzero(token["lexical_mask"]))
    assert arrays["query_token_ids"].tolist() == [token["token_ids"][index] for index in local]
    assert arrays["query_absolute_positions"].tolist() == [
        token["answer_token_positions"][index] for index in local]
    return arrays


def build_design(partition: str, candidate: str, meta: dict) -> np.memmap:
    assert candidate in CANDIDATES
    spec = CANDIDATE_SPECS[candidate]
    attr_width = BAND_ATTR_WINDOW_WIDTH if spec["layer_bands"] else FULL_ATTR_WINDOW_WIDTH
    width = int(spec["width"])
    stored_width = attr_width + 1
    representation = "band4" if spec["layer_bands"] else "full"
    compatible = [name for name, value in CANDIDATE_SPECS.items()
                  if value["layer_bands"] == spec["layer_bands"]]
    folder = OUT / "matrices"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{partition}_{representation}_with_nll.npy"
    manifest_path = folder / f"{partition}_{representation}_with_nll.json"
    pending = path.with_suffix(path.suffix + ".pending")
    if path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["representation"] == representation
        assert manifest["compatible_candidates"] == compatible
        assert manifest["sha256"] == sha(path)
        result = np.load(path, mmap_mode="r")
        assert result.shape == (EXPECTED["windows"][partition], stored_width)
        return result[:, :width]
    if path.exists() and not manifest_path.exists():
        # ``path`` is published only after a full flush below.  A crash between
        # that atomic rename and the manifest write is therefore recoverable.
        recovered = np.load(path, mmap_mode="r")
        assert recovered.shape == (EXPECTED["windows"][partition], stored_width)
        assert np.isfinite(recovered).all()
        manifest = {"partition": partition, "representation": representation,
                    "compatible_candidates": compatible,
                    "shape": list(recovered.shape), "dtype": str(recovered.dtype),
                    "bytes": int(recovered.nbytes), "sha256": sha(path)}
        frozen_json(manifest_path, manifest)
        return recovered[:, :width]
    assert not path.exists() and not manifest_path.exists()
    if pending.exists():
        pending.unlink()
    result = np.lib.format.open_memmap(
        pending, mode="w+", dtype=np.float32,
        shape=(EXPECTED["windows"][partition], stored_width))
    for answer_index, answer in enumerate(meta["answers"]):
        rid = answer["response_id"]
        token = meta["by_response"][rid]["tokens"]
        arrays = load_materialized_arrays(rid, token)
        for window_index in meta["answer_windows"][rid]:
            window = meta["windows"][window_index]
            result[window_index, :attr_width] = aggregate_window_features(
                arrays["token_features"], arrays["query_local_answer_indices"],
                window["token_indices"], window["lexical_token_indices"],
                layer_bands=spec["layer_bands"])
            result[window_index, attr_width] = meta["nll_by_response"][rid][
                window["token_indices"]].mean(dtype=np.float32)
        if (answer_index + 1) % 100 == 0:
            print("TOKEN_ATTRIBUTION_DESIGN", partition, candidate,
                  answer_index + 1, len(meta["answers"]), flush=True)
    result.flush()
    assert np.isfinite(result).all()
    del result
    pending.replace(path)
    result = np.load(path, mmap_mode="r")
    manifest = {"partition": partition, "representation": representation,
                "compatible_candidates": compatible,
                "shape": list(result.shape), "dtype": str(result.dtype),
                "bytes": int(result.nbytes), "sha256": sha(path)}
    frozen_json(manifest_path, manifest)
    return np.load(path, mmap_mode="r")[:, :width]


def window_weights(meta: dict, labels: np.ndarray, active: np.ndarray) -> np.ndarray:
    active_set = set(map(int, active))
    group_answers = defaultdict(lambda: defaultdict(list))
    for answer in meta["answers"]:
        ids = [index for index in meta["answer_windows"][answer["response_id"]]
               if index in active_set]
        if ids:
            group_answers[answer["group_id"]][answer["response_id"]] = ids
    weights = np.zeros(len(labels), dtype=np.float64)
    for answers in group_answers.values():
        for ids in answers.values():
            weights[ids] = 1 / (len(answers) * len(ids))
    mask = weights > 0
    mass = np.bincount(labels[mask], weights=weights[mask], minlength=2)
    assert np.all(mass > 0)
    weights[mask] *= (mass.sum() / (2 * mass))[labels[mask]]
    weights[mask] *= mask.sum() / weights[mask].sum()
    assert np.isfinite(weights).all() and np.all(weights[mask] > 0)
    return weights


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="liblinear", penalty="l2",
                           max_iter=2000, random_state=SEED))


def fit_model(model, x, y, weights):
    model.fit(x, y, standardscaler__sample_weight=weights,
              logisticregression__sample_weight=weights)
    return model


def choose_threshold(labels, scores) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    sorted_y, sorted_score = y[order], score[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    last = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    indices = np.flatnonzero(last)
    fn = int(y.sum()) - tp[indices]
    f1 = 2 * tp[indices] / np.maximum(1, 2 * tp[indices] + fp[indices] + fn)
    precision = tp[indices] / np.maximum(1, tp[indices] + fp[indices])
    best = max(range(len(indices)), key=lambda i: (f1[i], precision[i], sorted_score[indices[i]]))
    chosen = indices[best]
    return {"threshold": float(sorted_score[chosen]), "f1": float(f1[best]),
            "precision": float(precision[best]), "rows": len(y),
            "positive": int(y.sum())}


def metric(labels, scores, threshold) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    prediction = score >= threshold
    tp = int(np.count_nonzero(prediction & (y == 1)))
    fp = int(np.count_nonzero(prediction & (y == 0)))
    fn = int(np.count_nonzero(~prediction & (y == 1)))
    tn = int(np.count_nonzero(~prediction & (y == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-30, precision + recall),
            "auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score))}


def answer_scores(meta: dict, window_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    for answer in meta["answers"]:
        scores.append(max(float(window_scores[index])
                          for index in meta["answer_windows"][answer["response_id"]]))
        labels.append(int(answer["label"]))
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def thresholds_and_metrics(meta: dict, scores: np.ndarray) -> tuple[dict, dict]:
    window_y = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    answer_score, answer_y = answer_scores(meta, scores)
    thresholds = {"window": choose_threshold(window_y, scores),
                  "answer": choose_threshold(answer_y, answer_score)}
    metrics = {"windows": metric(window_y, scores, thresholds["window"]["threshold"]),
               "answers": metric(answer_y, answer_score, thresholds["answer"]["threshold"])}
    return thresholds, metrics


def cpu_check() -> dict:
    assert_cpu_only()
    check_initialized()
    token = np.arange(4 * LAYERS * TOKEN_WIDTH, dtype=np.float32).reshape(4, LAYERS, TOKEN_WIDTH) / 10000
    local = np.asarray([0, 2, 3, 5], dtype=np.int32)
    feature = aggregate_window_features(token, local, [1, 2, 3, 4])
    expected = np.concatenate((token[[1, 2]].mean(0).ravel(),
                               token[[1, 2]].max(0).ravel(), [.5])).astype(np.float32)
    assert np.array_equal(feature, expected)
    compact_token = token.reshape(4, LAYER_BANDS, LAYERS_PER_BAND,
                                  TOKEN_WIDTH).mean(axis=2, dtype=np.float32)
    compact = aggregate_window_features(token, local, [1, 2, 3, 4],
                                        layer_bands=True)
    expected_compact = np.concatenate((compact_token[[1, 2]].mean(0).ravel(),
                                       compact_token[[1, 2]].max(0).ravel(),
                                       [.5])).astype(np.float32)
    assert np.array_equal(compact, expected_compact)
    groups = np.repeat(np.arange(10), 2)
    for train, held in GroupKFold(FOLDS).split(np.arange(20), groups=groups):
        assert not set(groups[train]) & set(groups[held])
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([.1, .9, .2, .8])
    threshold = choose_threshold(y, scores)
    assert metric(y, scores, threshold["threshold"])["f1"] == 1
    result = {
        "status": "passed_label_free",
        "direct_window_aggregation_exact": True,
        "candidate_widths": {name: spec["width"]
                             for name, spec in CANDIDATE_SPECS.items()},
        "full_window_aggregation_exact": len(feature) == FULL_ATTR_WINDOW_WIDTH,
        "band4_window_aggregation_exact": len(compact) == BAND_ATTR_WINDOW_WIDTH,
        "source_group_folds_disjoint": True,
        "fit_threshold_selfcheck": True,
        "gpu_used": False, "fit_label_files_opened": False,
        "calibration_label_files_opened": False,
        "formal_baselines_modified": False, "official_test_opened": False}
    frozen_json(OUT / "CPU_SELFCHECK.json", result)
    print("TOKEN_ATTRIBUTION_V4_SCORE_CPU_CHECK_PASSED", flush=True)
    return result


def fit() -> dict:
    assert_cpu_only()
    snapshot = check_initialized()
    if (OUT / "fit_complete.json").exists():
        completed = read_json(OUT / "fit_complete.json")
        assert completed["model_sha256"] == sha(OUT / "model.pkl")
        assert completed["source_snapshot_sha256"] == sha(OUT / "source_snapshot.json")
        return completed
    assert read_json(OUT / "CPU_SELFCHECK.json")["status"] == "passed_label_free"
    token_manifest = read_json(OUT / "token_feature_manifest.json")
    assert token_manifest["status"] == "complete_label_free_token_features"
    fit_gate = {
        "status": "fit_annotations_opened_after_label_free_features_frozen",
        "token_feature_manifest_sha256": sha(OUT / "token_feature_manifest.json"),
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "calibration_label_files_opened": False,
        "official_test_opened": False}
    frozen_json(OUT / "fit_started.json", fit_gate)
    meta = load_partition_meta("fit")
    y = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    groups = np.asarray([row["group_id"] for row in meta["windows"]])
    indices = np.arange(len(y))
    folds = list(GroupKFold(FOLDS).split(indices, y, groups))
    history, stored = {}, {}
    final_models = {}
    with threadpool_limits(limits=THREADS):
        for candidate_index, candidate in enumerate(CANDIDATES):
            x = build_design("fit", candidate, meta)
            prediction = np.full(len(y), np.nan, dtype=np.float64)
            fold_log = []
            for fold, (train, held) in enumerate(folds):
                assert not set(groups[train]) & set(groups[held])
                weights = window_weights(meta, y, train)
                model = fit_model(make_model(), x[train], y[train], weights[train])
                prediction[held] = model.predict_proba(x[held])[:, 1]
                fold_log.append({"fold": fold, "train_windows": len(train),
                                 "held_windows": len(held),
                                 "train_groups": len(set(groups[train])),
                                 "held_groups": len(set(groups[held])),
                                 "group_overlap": 0})
            assert np.isfinite(prediction).all()
            thresholds, metrics = thresholds_and_metrics(meta, prediction)
            history[candidate] = {
                "fit_OOF_thresholds": thresholds, "fit_OOF_metrics": metrics,
                "folds": fold_log, "width": x.shape[1], "C": C}
            stored[candidate] = prediction
            weights = window_weights(meta, y, indices)
            final_models[candidate] = fit_model(make_model(), x, y, weights)
            print("TOKEN_ATTRIBUTION_V4_OOF", candidate,
                  round(metrics["windows"]["f1"], 6),
                  round(metrics["answers"]["f1"], 6), flush=True)
    selected = max(CANDIDATES, key=lambda name: (
        min(history[name]["fit_OOF_metrics"]["windows"]["f1"],
            history[name]["fit_OOF_metrics"]["answers"]["f1"]),
        history[name]["fit_OOF_metrics"]["windows"]["f1"],
        history[name]["fit_OOF_metrics"]["answers"]["f1"],
        history[name]["fit_OOF_metrics"]["windows"]["precision"],
        -CANDIDATES.index(name)))
    model_path = OUT / "model.pkl"
    pending_model = model_path.with_suffix(model_path.suffix + ".pending")
    pending_model.write_bytes(pickle.dumps({"selected": selected,
                                            "model": final_models[selected]}, protocol=5))
    pending_model.replace(model_path)
    atomic_npz(OUT / "fit_oof_scores.npz", {
        **{name: values for name, values in stored.items()},
        "selected_scores": stored[selected]})
    report = {
        "status": "selected_and_frozen_before_calibration",
        "selected_fit_OOF_only": selected, "candidates": history,
        "frozen_thresholds": history[selected]["fit_OOF_thresholds"],
        "fit_answers": len(meta["answers"]), "fit_groups": len(set(groups)),
        "fit_windows": len(y), "positive_windows": int(y.sum()),
        "model_sha256": sha(model_path),
        "fit_oof_scores_sha256": sha(OUT / "fit_oof_scores.npz"),
        "token_feature_manifest_sha256": sha(OUT / "token_feature_manifest.json"),
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "baseline_files_sha256": snapshot["baseline_files_sha256"],
        "calibration_labels_opened": False,
        "calibration_evaluations": 0,
        "gpu_used": False, "formal_baselines_modified": False,
        "official_test_opened": False}
    frozen_json(OUT / "fit_complete.json", report)
    print("TOKEN_ATTRIBUTION_V4_FIT_FROZEN", selected, flush=True)
    return report


def evaluate() -> dict:
    assert_cpu_only()
    snapshot = check_initialized()
    if (OUT / "complete.json").exists():
        completed = read_json(OUT / "complete.json")
        assert completed["scores_sha256"] == sha(OUT / "calibration_scores.npz")
        assert completed["fit_complete_sha256"] == sha(OUT / "fit_complete.json")
        return completed
    fit_done = read_json(OUT / "fit_complete.json")
    assert fit_done["status"] == "selected_and_frozen_before_calibration"
    assert fit_done["model_sha256"] == sha(OUT / "model.pkl")
    assert fit_done["baseline_files_sha256"] == baseline_snapshot() == snapshot["baseline_files_sha256"]
    calibration_gate = {
        "status": "calibration_opened_once_after_fit_freeze",
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "model_sha256": sha(OUT / "model.pkl"),
        "recovery_is_same_frozen_model_no_selection": True,
        "official_test_opened": False}
    frozen_json(OUT / "calibration_evaluation_started.json", calibration_gate)
    meta = load_partition_meta("calibration")
    selected = fit_done["selected_fit_OOF_only"]
    x = build_design("calibration", selected, meta)
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    assert payload["selected"] == selected
    scores = payload["model"].predict_proba(x)[:, 1]
    window_y = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    answer_score, answer_y = answer_scores(meta, scores)
    thresholds = fit_done["frozen_thresholds"]
    metrics = {
        "windows": metric(window_y, scores, thresholds["window"]["threshold"]),
        "answers": metric(answer_y, answer_score, thresholds["answer"]["threshold"])}
    atomic_npz(OUT / "calibration_scores.npz", {
        "window_scores": scores, "answer_scores": answer_score})
    result = {
        "status": "development_calibration_evaluated_once",
        "selected_fit_OOF_only": selected,
        "fit_OOF": fit_done["candidates"][selected]["fit_OOF_metrics"],
        "strict_calibration": metrics,
        "thresholds_from_fit_OOF": thresholds,
        "calibration_F1Opt_computed": False,
        "calibration_evaluations": 1,
        "scores_sha256": sha(OUT / "calibration_scores.npz"),
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "baseline_files_sha256": baseline_snapshot(),
        "calibration_labels_opened": True,
        "gpu_used": False, "formal_baselines_modified": False,
        "official_test_opened": False}
    frozen_json(OUT / "complete.json", result)
    report = (
        "# Token source attribution v4\n\n"
        f"选中 `{selected}`。fit source-group OOF 窗口/整答 F1："
        f"{result['fit_OOF']['windows']['f1']:.6f}/{result['fit_OOF']['answers']['f1']:.6f}；"
        f"严格 cal：{metrics['windows']['f1']:.6f}/{metrics['answers']['f1']:.6f}。\n\n"
        "cal 只评一次，未计算 cal-F1Opt；official test 未打开；baseline 未改。\n")
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("TOKEN_ATTRIBUTION_V4_CALIBRATION_COMPLETE",
          round(metrics["windows"]["f1"], 6),
          round(metrics["answers"]["f1"], 6), flush=True)
    return result


def status() -> dict:
    initialized = (OUT / "INITIALIZED.json").exists()
    report = {
        "version": VERSION, "initialized": initialized,
        "cpu_selfcheck": (OUT / "CPU_SELFCHECK.json").exists(),
        "nli_linked": (OUT / "NLI_LINK.json").exists(),
        "nli_inference_complete": (OUT / "NLI_INFERENCE.json").exists(),
        "nli_bank_sealed": (OUT / "NLI_BANK.json").exists(),
        "raw_attribution_complete": (ATTR_OUT / "feature_manifest.json").exists(),
        "token_features_complete": (OUT / "token_feature_manifest.json").exists(),
        "fit_complete": (OUT / "fit_complete.json").exists(),
        "calibration_evaluated": (OUT / "complete.json").exists(),
        "calibration_evaluation_started": (OUT / "calibration_evaluation_started.json").exists(),
        "official_test_opened": False,
        "formal_baselines_modified": False}
    if (OUT / "NLI_LINK.json").exists():
        link = read_json(OUT / "NLI_LINK.json")
        report["nli_unique_requests"] = link["unique_requests"]
        report["nli_missing_unique_requests"] = link["missing_unique_requests"]
    atomic_json(OUT / "status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "initialize", "cpu-check", "link-nli", "infer-nli", "seal-nli",
        "materialize", "fit", "evaluate", "status"))
    args = parser.parse_args()
    actions = {
        "initialize": initialize, "cpu-check": cpu_check,
        "link-nli": link_nli, "infer-nli": infer_nli,
        "seal-nli": seal_nli, "materialize": materialize,
        "fit": fit, "evaluate": evaluate, "status": status}
    if args.command == "status":
        actions[args.command]()
    else:
        with extractor.exclusive_file(
                ROOT / "results/.token_source_attribution_v4_score.lock",
                f"scorer-{args.command}"):
            actions[args.command]()


if __name__ == "__main__":
    main()
