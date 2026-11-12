"""Independent, read-only final audit for the common calibration benchmark.

This script deliberately opens only fit/calibration artifacts and frozen method
artifacts.  It never enumerates or opens the retired RAGTruth QA test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


AUDIT_DIR = Path(__file__).resolve().parent
BASE = AUDIT_DIR.parents[1]
WORKSPACE = BASE.parents[1]
DATA = BASE / "data"
RAG = BASE / "research" / "ragognizer_formal_baseline_v1"
REDEEP = BASE / "research" / "redeep_formal_baseline_v1"
ZERO_CHAIN = "0" * 64
TOL = 1e-12
PWSH = shutil.which("pwsh")
if PWSH is None:
    raise RuntimeError("pwsh is required for native hashing of large frozen artifacts")


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.hash_cache: dict[tuple[int, int, int, int], str] = {}

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if bool(condition) else "FAIL", "detail": detail}
        )

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.check(name, actual == expected, {"actual": actual, "expected": expected})

    def close(self, name: str, actual: float, expected: float, tol: float = TOL) -> None:
        delta = abs(float(actual) - float(expected))
        self.check(
            name,
            delta <= tol,
            {"actual": float(actual), "expected": float(expected), "abs_difference": delta, "tol": tol},
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.checks)

    def sha256_file(self, path: Path) -> str:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key in self.hash_cache:
            return self.hash_cache[key]
        # The CA Python build is unusually slow for long sequential hashlib
        # reads on this Windows volume.  PowerShell uses the native Windows
        # hashing path and gives the same SHA256 bytes much faster.
        if stat.st_size >= 32 * 1024 * 1024:
            quoted = str(path.resolve()).replace("'", "''")
            value = subprocess.check_output(
                [
                    PWSH,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"(Get-FileHash -Algorithm SHA256 -LiteralPath '{quoted}').Hash.ToLowerInvariant()",
                ],
                text=True,
            ).strip()
        else:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(16 * 1024 * 1024):
                    digest.update(chunk)
            value = digest.hexdigest()
        self.hash_cache[key] = value
        return value

    def remember_hash(self, path: Path, value: str) -> None:
        stat = path.stat()
        self.hash_cache[(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)] = value


A = Audit()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def f1_opt(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("invalid metric arrays")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite score")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    s = scores[order]
    total_positive = int(labels.sum())
    tp = 0
    fp = 0
    best: tuple[float, float, float] | None = None
    best_row: dict[str, Any] | None = None
    for index, (label, score) in enumerate(zip(y, s)):
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        if index + 1 < len(s) and s[index + 1] == score:
            continue
        fn = total_positive - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_positive if total_positive else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        key = (f1, precision, float(score))
        if best is None or key > best:
            best = key
            best_row = {
                "threshold": float(score),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
    assert best_row is not None
    predicted = scores >= best_row["threshold"]
    tn = int(((labels == 0) & (~predicted)).sum())
    return {
        "rows": int(len(labels)),
        "positives": total_positive,
        "f1_opt": best_row,
        "tn": tn,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def compare_metric(prefix: str, actual: dict[str, Any], reported: dict[str, Any]) -> None:
    A.equal(prefix + ".rows", actual["rows"], int(reported.get("rows", reported.get("n"))))
    A.equal(
        prefix + ".positives",
        actual["positives"],
        int(reported.get("positives", reported.get("positive"))),
    )
    report_f1 = reported.get("f1_opt", reported)
    for key in ("threshold", "f1", "precision"):
        if key in report_f1:
            A.close(prefix + "." + key, actual["f1_opt"][key], report_f1[key])
    if "recall" in report_f1:
        A.close(prefix + ".recall", actual["f1_opt"]["recall"], report_f1["recall"])
    for key in ("tp", "fp", "fn"):
        if key in report_f1:
            A.equal(prefix + "." + key, actual["f1_opt"][key], int(report_f1[key]))
    if "tn" in reported:
        A.equal(prefix + ".tn", actual["tn"], int(reported["tn"]))
    A.close(prefix + ".auroc", actual["auroc"], reported["auroc"])
    ap_key = "average_precision" if "average_precision" in reported else "ap"
    A.close(prefix + ".average_precision", actual["average_precision"], reported[ap_key])


def answer_max_from_ordered_windows(
    window_scores: np.ndarray, answers: list[dict[str, Any]], windows: list[dict[str, Any]]
) -> np.ndarray:
    by_answer: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(windows, window_scores):
        by_answer[str(row["answer_id"])].append(float(score))
    return np.asarray([max(by_answer[str(row["answer_id"])]) for row in answers], dtype=np.float64)


# ---------------------------------------------------------------------------
# Frozen common calibration labels and geometry (no test split).
# ---------------------------------------------------------------------------
cal_answers = load_jsonl(DATA / "answers_calibration.jsonl")
cal_windows = load_jsonl(DATA / "windows_k4_calibration.jsonl")
answer_ids = [str(row["answer_id"]) for row in cal_answers]
window_ids = [str(row["window_id"]) for row in cal_windows]
answer_labels = np.asarray([int(row["label"]) for row in cal_answers], dtype=np.int8)
window_labels = np.asarray([int(row["label"]) for row in cal_windows], dtype=np.int8)
answer_id_set = set(answer_ids)
window_id_set = set(window_ids)

A.equal("common.calibration.answers", len(cal_answers), 159)
A.equal("common.calibration.answer_groups", len({row["group_id"] for row in cal_answers}), 154)
A.equal("common.calibration.windows", len(cal_windows), 42241)
A.equal("common.calibration.positive_answers", int(answer_labels.sum()), 100)
A.equal("common.calibration.positive_windows", int(window_labels.sum()), 5984)
A.equal("common.calibration.unique_answer_ids", len(answer_id_set), 159)
A.equal("common.calibration.unique_window_ids", len(window_id_set), 42241)
A.check(
    "common.calibration.window_answer_membership",
    {str(row["answer_id"]) for row in cal_windows} == answer_id_set,
)


metrics: dict[str, dict[str, Any]] = {}
score_sources: dict[str, Any] = {}
print("stage: common geometry loaded", flush=True)


# Candidate: selected frozen semantic_claim + tail2 tree + large weight 0.4.
candidate_dir = BASE / "results" / "large_fixed_convex_v1"
candidate_json = load_json(candidate_dir / "semantic_claim__old_tree__large_weight0.4.json")
candidate_npz_path = candidate_dir / "semantic_claim__old_tree__large_weight0.4_scores.npz"
candidate_npz = np.load(candidate_npz_path, allow_pickle=False)
A.equal("candidate.score_sha256", A.sha256_file(candidate_npz_path), candidate_json["scores_sha256"])
A.equal("candidate.window_score_rows_all", len(candidate_npz["window_scores"]), 210364)
A.equal("candidate.answer_score_rows_all", len(candidate_npz["answer_scores"]), 793)
candidate_window = np.asarray(candidate_npz["window_scores"][-42241:], dtype=np.float64)
candidate_answer = np.asarray(candidate_npz["answer_scores"][-159:], dtype=np.float64)
candidate_answer_rebuilt = answer_max_from_ordered_windows(candidate_window, cal_answers, cal_windows)
A.close(
    "candidate.answer_equals_max_common_windows",
    float(np.max(np.abs(candidate_answer_rebuilt - candidate_answer))),
    0.0,
)
metrics["ours"] = {
    "window": f1_opt(window_labels, candidate_window),
    "answer": f1_opt(answer_labels, candidate_answer),
}
compare_metric(
    "candidate.window", metrics["ours"]["window"], candidate_json["metrics"]["calibration"]["windows"]
)
compare_metric(
    "candidate.answer", metrics["ours"]["answer"], candidate_json["metrics"]["calibration"]["answers"]
)
score_sources["ours"] = str(candidate_npz_path.relative_to(BASE))


# Lookback Lens frozen 8-BPE output mapped without parameters to common 4-BPE.
lookback_dir = BASE / "results" / "lookback_k4_evaluation_adapter_v1"
lookback_summary = load_json(lookback_dir / "summary.json")
lookback_path = lookback_dir / "adapted_scores.npz"
lookback = np.load(lookback_path, allow_pickle=False)
A.equal("lookback.score_sha256", A.sha256_file(lookback_path), lookback_summary["adapted_scores_sha256"])
A.check("lookback.answer_ids_exact_common_order", np.array_equal(lookback["answer_ids"], np.asarray(answer_ids)))
A.check("lookback.window_ids_exact_common_order", np.array_equal(lookback["window_ids"], np.asarray(window_ids)))
lookback_window = np.asarray(lookback["window_scores"], dtype=np.float64)
lookback_answer = np.asarray(lookback["answer_scores"], dtype=np.float64)
A.close(
    "lookback.answer_equals_max_common_windows",
    float(np.max(np.abs(answer_max_from_ordered_windows(lookback_window, cal_answers, cal_windows) - lookback_answer))),
    0.0,
)
metrics["Lookback Lens"] = {
    "window": f1_opt(window_labels, lookback_window),
    "answer": f1_opt(answer_labels, lookback_answer),
}
compare_metric(
    "lookback.window",
    metrics["Lookback Lens"]["window"],
    lookback_summary["adapted_project4_calibration"]["windows4"],
)
compare_metric(
    "lookback.answer",
    metrics["Lookback Lens"]["answer"],
    lookback_summary["adapted_project4_calibration"]["answers"],
)
score_sources["Lookback Lens"] = str(lookback_path.relative_to(BASE))


# LUMINA author formula, full 3,839 inference rows, common calibration subset.
lumina_dir = BASE / "results" / "lumina_k4_evaluation_adapter_v2"
lumina_summary = load_json(lumina_dir / "summary.json")
lumina_path = lumina_dir / "adapted_scores.npz"
lumina = np.load(lumina_path, allow_pickle=False)
A.equal("lumina.score_sha256", A.sha256_file(lumina_path), lumina_summary["adapted_scores_sha256"])
lumina_response_ids = np.asarray(lumina["response_ids"])
lumina_window_ids = np.asarray(lumina["window_ids"])
lumina_window_scores = np.asarray(lumina["window_scores"], dtype=np.float64)
lumina_answer_scores = np.asarray(lumina["answer_scores"], dtype=np.float64)
A.equal("lumina.full_answer_rows", len(lumina_response_ids), 3839)
A.equal("lumina.full_window_rows", len(lumina_window_ids), 696220)
lumina_methods = [str(x) for x in lumina["method_names"]]
A.check("lumina.primary_formula_present", "lumina" in lumina_methods, lumina_methods)
lumina_col = lumina_methods.index("lumina")
lumina_answer_indices = [i for i, rid in enumerate(lumina_response_ids) if str(rid) in answer_id_set]
lumina_window_indices = [i for i, wid in enumerate(lumina_window_ids) if str(wid) in window_id_set]
A.check(
    "lumina.answer_ids_exact_common_order",
    [str(lumina_response_ids[i]) for i in lumina_answer_indices] == answer_ids,
)
A.check(
    "lumina.window_ids_exact_common_order",
    [str(lumina_window_ids[i]) for i in lumina_window_indices] == window_ids,
)
lumina_window = np.asarray(lumina_window_scores[lumina_window_indices, lumina_col], dtype=np.float64)
lumina_answer = np.asarray(lumina_answer_scores[lumina_answer_indices, lumina_col], dtype=np.float64)
A.close(
    "lumina.answer_equals_max_common_windows",
    float(np.max(np.abs(answer_max_from_ordered_windows(lumina_window, cal_answers, cal_windows) - lumina_answer))),
    0.0,
)
metrics["LUMINA"] = {"window": f1_opt(window_labels, lumina_window), "answer": f1_opt(answer_labels, lumina_answer)}
lumina_reported = lumina_summary["all_fixed_readouts"]["lumina"]["calibration"]
compare_metric("lumina.window", metrics["LUMINA"]["window"], lumina_reported["windows"])
compare_metric("lumina.answer", metrics["LUMINA"]["answer"], lumina_reported["answers"])
score_sources["LUMINA"] = str(lumina_path.relative_to(BASE))


# GHOST frozen answer-only RF, uniformly broadcast by the common evaluator.
ghost_dir = BASE / "results" / "ghost_k4_evaluation_adapter_v1"
ghost_summary = load_json(ghost_dir / "summary.json")
ghost_path = ghost_dir / "adapted_scores.npz"
ghost = np.load(ghost_path, allow_pickle=False)
A.equal("ghost.score_sha256", A.sha256_file(ghost_path), ghost_summary["adapted_scores_sha256"])
ghost_response_ids = np.asarray(ghost["response_ids"])
ghost_window_ids = np.asarray(ghost["window_ids"])
ghost_window_scores = np.asarray(ghost["window_scores"], dtype=np.float64)
ghost_answer_scores = np.asarray(ghost["answer_scores"], dtype=np.float64)
ghost_answer_indices = [i for i, rid in enumerate(ghost_response_ids) if str(rid) in answer_id_set]
ghost_window_indices = [i for i, wid in enumerate(ghost_window_ids) if str(wid) in window_id_set]
A.check(
    "ghost.answer_ids_exact_common_order",
    [str(ghost_response_ids[i]) for i in ghost_answer_indices] == answer_ids,
)
A.check(
    "ghost.window_ids_exact_common_order",
    [str(ghost_window_ids[i]) for i in ghost_window_indices] == window_ids,
)
ghost_window = np.asarray(ghost_window_scores[ghost_window_indices], dtype=np.float64)
ghost_answer = np.asarray(ghost_answer_scores[ghost_answer_indices], dtype=np.float64)
ghost_answer_by_id = {str(ghost_response_ids[i]): float(ghost_answer_scores[i]) for i in ghost_answer_indices}
ghost_broadcast_delta = max(
    abs(float(ghost_window_scores[i]) - ghost_answer_by_id[str(ghost_window_ids[i]).split("__k4_")[0]])
    for i in ghost_window_indices
)
A.close("ghost.window_is_exact_answer_broadcast", ghost_broadcast_delta, 0.0)
metrics["GHOST"] = {"window": f1_opt(window_labels, ghost_window), "answer": f1_opt(answer_labels, ghost_answer)}
compare_metric("ghost.window", metrics["GHOST"]["window"], ghost_summary["common_metrics"]["calibration"]["windows"])
compare_metric("ghost.answer", metrics["GHOST"]["answer"], ghost_summary["common_metrics"]["calibration"]["answers"])
score_sources["GHOST"] = str(ghost_path.relative_to(BASE))
print("stage: candidate/lookback/lumina/ghost recomputed", flush=True)


# ReDeEP paper-explicit formula, full inference coverage and common cal subset.
redeep_score_path = REDEEP / "frozen_scores.npz"
redeep = np.load(redeep_score_path, allow_pickle=False)
redeep_results = load_json(REDEEP / "CAL_RESULTS.json")
redeep_verify = load_json(REDEEP / "SCORING_INDEPENDENT_VERIFICATION.json")
A.equal("redeep.score_sha256", A.sha256_file(redeep_score_path), redeep_verify["frozen_scores_sha256"])
redeep_answer_ids = np.asarray(redeep["answer_id"])
redeep_window_ids = np.asarray(redeep["window_id"])
redeep_answer_partitions = np.asarray(redeep["answer_partition"])
redeep_window_partitions = np.asarray(redeep["window_partition"])
redeep_paper_window_scores = np.asarray(redeep["paper_window_score"], dtype=np.float64)
redeep_paper_answer_scores = np.asarray(redeep["paper_answer_score"], dtype=np.float64)
A.equal("redeep.full_answer_rows", len(redeep_answer_ids), 3839)
A.equal("redeep.full_window_rows", len(redeep_window_ids), 696220)
redeep_answer_indices = [i for i, part in enumerate(redeep_answer_partitions) if str(part) == "calibration"]
redeep_window_indices = [i for i, part in enumerate(redeep_window_partitions) if str(part) == "calibration"]
A.check(
    "redeep.answer_ids_exact_common_order",
    [str(redeep_answer_ids[i]) for i in redeep_answer_indices] == answer_ids,
)
A.check(
    "redeep.window_ids_exact_common_order",
    [str(redeep_window_ids[i]) for i in redeep_window_indices] == window_ids,
)
redeep_window = np.asarray(redeep_paper_window_scores[redeep_window_indices], dtype=np.float64)
redeep_answer = np.asarray(redeep_paper_answer_scores[redeep_answer_indices], dtype=np.float64)
A.close(
    "redeep.answer_equals_max_common_windows",
    float(np.max(np.abs(answer_max_from_ordered_windows(redeep_window, cal_answers, cal_windows) - redeep_answer))),
    0.0,
)
metrics["ReDeEP"] = {"window": f1_opt(window_labels, redeep_window), "answer": f1_opt(answer_labels, redeep_answer)}
redeep_reported = redeep_results["identities"]["paper"]
compare_metric("redeep.window", metrics["ReDeEP"]["window"], redeep_reported["shared_k4_window"])
compare_metric("redeep.answer", metrics["ReDeEP"]["answer"], redeep_reported["shared_answer_max_window"])
score_sources["ReDeEP"] = str(redeep_score_path.relative_to(BASE))

# Verify ReDeEP's final manifest and all 3,839 raw feature payload hashes.
redeep_manifest_path = REDEEP / "FINAL_ARTIFACT_MANIFEST.json"
redeep_manifest = load_json(redeep_manifest_path)
redeep_manifest_failures: list[str] = []
for entry in redeep_manifest["entries"]:
    path = BASE / Path(entry["path"])
    if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or A.sha256_file(path) != entry["sha256"]:
        redeep_manifest_failures.append(entry["path"])
A.check("redeep.final_manifest_all_entries", not redeep_manifest_failures, redeep_manifest_failures)
redeep_raw_manifest = load_json(REDEEP / "RAW_FEATURE_MANIFEST.json")
A.equal("redeep.raw_manifest.files", redeep_raw_manifest["totals"]["files"], 3839)
A.equal("redeep.raw_manifest.fit", redeep_raw_manifest["totals"]["partition_files"]["fit"], 3680)
A.equal("redeep.raw_manifest.calibration", redeep_raw_manifest["totals"]["partition_files"]["calibration"], 159)
redeep_raw_failures: list[str] = []
for entry in redeep_raw_manifest["entries"]:
    path = BASE / Path(entry["file"])
    if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or A.sha256_file(path) != entry["sha256"]:
        redeep_raw_failures.append(entry["file"])
A.check("redeep.raw_payload_all_3839_hashes", not redeep_raw_failures, redeep_raw_failures[:10])
redeep_commit = subprocess.check_output(
    ["git", "-C", str(BASE / "third_party" / "ReDEeP-ICLR"), "rev-parse", "HEAD"], text=True
).strip()
A.equal("redeep.official_repo_commit", redeep_commit, "4d081915b8fb4430fda65c411da61540cc73cc57")
print("stage: redeep scores and 3839 raw hashes verified", flush=True)


# ---------------------------------------------------------------------------
# RAGognizer: full canonical stream/hash/mapping audit and common scoring.
# ---------------------------------------------------------------------------
rag_plan_path = RAG / "INFERENCE_PLAN.jsonl"
rag_raw_path = RAG / "NATIVE_PROBABILITIES.jsonl"
rag_shared_path = RAG / "SHARED_PROBABILITIES.jsonl"
rag_gpu = load_json(RAG / "GPU_RUN_AUDIT.json")
rag_adapter_audit = load_json(RAG / "ADAPTER_AUDIT.json")
rag_freeze = load_json(RAG / "SCORE_FREEZE.json")
rag_result = load_json(RAG / "CAL_RESULTS.json")
rag_status = load_json(RAG / "SCORING_STATUS.json")
rag_artifacts = load_json(RAG / "ARTIFACT_HASHES.json")

plan_digest = hashlib.sha256()
raw_digest = hashlib.sha256()
shared_digest = hashlib.sha256()
plan_chain = ZERO_CHAIN
raw_chain = ZERO_CHAIN
shared_chain = ZERO_CHAIN
rag_partition_counts: Counter[str] = Counter()
rag_partition_windows: Counter[str] = Counter()
rag_groups: dict[str, set[str]] = defaultdict(set)
rag_answer_ids: list[str] = []
rag_cal_raw: dict[str, dict[str, Any]] = {}
rag_cal_shared: dict[str, dict[str, Any]] = {}
rag_mapping_max_delta = 0.0
rag_mapping_index_failures = 0
rag_constant_failures = 0
rag_raw_hash_failures = 0
rag_shared_hash_failures = 0
rag_plan_hash_failures = 0
rag_raw_quantization_values: set[str] = set()
rag_coordinate_modes: Counter[str] = Counter()
rag_author_tokens = 0
rag_shared_slots = 0

with rag_plan_path.open("rb") as plan_handle, rag_raw_path.open("rb") as raw_handle, rag_shared_path.open("rb") as shared_handle:
    for index, triple in enumerate(zip_longest(plan_handle, raw_handle, shared_handle)):
        plan_line, raw_line, shared_line = triple
        if plan_line is None or raw_line is None or shared_line is None:
            A.check("ragognizer.stream_lengths_equal", False, {"first_mismatch_index": index})
            break
        plan_digest.update(plan_line)
        raw_digest.update(raw_line)
        shared_digest.update(shared_line)
        plan = json.loads(plan_line)
        raw = json.loads(raw_line)
        shared = json.loads(shared_line)

        plan_core = {key: value for key, value in plan.items() if key != "plan_row_sha256"}
        expected_plan_row_hash = sha256_json(plan_core)
        if plan.get("plan_row_sha256") != expected_plan_row_hash:
            rag_plan_hash_failures += 1
        plan_chain = sha256_text(plan_chain + "\n" + expected_plan_row_hash)

        raw_core = {
            key: value
            for key, value in raw.items()
            if key not in {"raw_row_sha256", "previous_chain_sha256", "chain_sha256"}
        }
        expected_raw_row_hash = sha256_json(raw_core)
        expected_raw_chain = sha256_text(raw_chain + "\n" + expected_raw_row_hash)
        if (
            raw.get("raw_row_sha256") != expected_raw_row_hash
            or raw.get("previous_chain_sha256") != raw_chain
            or raw.get("chain_sha256") != expected_raw_chain
        ):
            rag_raw_hash_failures += 1
        raw_chain = expected_raw_chain

        shared_core = {
            key: value
            for key, value in shared.items()
            if key not in {"adapted_row_sha256", "previous_chain_sha256", "chain_sha256"}
        }
        expected_shared_row_hash = sha256_json(shared_core)
        expected_shared_chain = sha256_text(shared_chain + "\n" + expected_shared_row_hash)
        if (
            shared.get("adapted_row_sha256") != expected_shared_row_hash
            or shared.get("previous_chain_sha256") != shared_chain
            or shared.get("chain_sha256") != expected_shared_chain
        ):
            rag_shared_hash_failures += 1
        shared_chain = expected_shared_chain

        answer_id = str(plan["answer_id"])
        partition = str(plan["partition"])
        rag_answer_ids.append(answer_id)
        rag_partition_counts[partition] += 1
        rag_groups[partition].add(str(plan["group_id"]))
        rag_partition_windows[partition] += len(plan["eligible_windows"])
        rag_author_tokens += len(raw["author_response_token_probabilities"])
        rag_shared_slots += len(shared["shared_bpe_token_probabilities"])
        rag_coordinate_modes[str(shared["author_coordinate_mode"])] += 1
        rag_raw_quantization_values.add(repr(raw.get("quantization")))

        constants_ok = (
            str(raw.get("answer_id")) == answer_id
            and str(shared.get("answer_id")) == answer_id
            and raw.get("partition") == partition
            and shared.get("partition") == partition
            and raw.get("answer_sha256") == plan.get("answer_sha256")
            and shared.get("answer_sha256") == plan.get("answer_sha256")
            and raw.get("plan_row_sha256") == plan.get("plan_row_sha256")
            and shared.get("plan_row_sha256") == plan.get("plan_row_sha256")
            and raw.get("route") == "model_card_default_transformer_heads"
            and raw.get("postprocessor") is False
            and raw.get("quantization") is None
            and raw.get("threshold_applied") is False
            and shared.get("labels_read") is False
            and shared.get("shared_thresholds_applied") is False
            and shared.get("raw_chain_sha256") == raw_chain
            and raw.get("author_response_token_ids") == plan["author_response_tokens"]["token_ids"]
            and raw.get("author_response_token_char_intervals") == plan["author_response_tokens"]["char_intervals"]
            and hashlib.sha256(plan["original_response"].encode("utf-8")).hexdigest() == plan["answer_sha256"]
            and hashlib.sha256(plan["released_prompt"].encode("utf-8")).hexdigest() == plan["prompt_sha256"]
        )
        if not constants_ok:
            rag_constant_failures += 1

        native_intervals = [tuple(map(int, pair)) for pair in plan["author_response_tokens"]["char_intervals"]]
        local_intervals = [tuple(map(int, pair)) for pair in plan["shared_bpe_tokens"]["char_intervals"]]
        expected_mapping: list[list[int]] = []
        left = 0
        for local_start, local_end in local_intervals:
            while left < len(native_intervals) and native_intervals[left][1] <= local_start:
                left += 1
            found: list[int] = []
            cursor = left
            while cursor < len(native_intervals) and native_intervals[cursor][0] < local_end:
                native_start, native_end = native_intervals[cursor]
                if max(local_start, native_start) < min(local_end, native_end):
                    found.append(cursor)
                cursor += 1
            expected_mapping.append(found)
        if expected_mapping != plan["mapping"]["local_to_author_token_indices"] or any(not x for x in expected_mapping):
            rag_mapping_index_failures += 1

        native_probabilities = [float(value) for value in raw["author_response_token_probabilities"]]
        expected_local = [math.fsum(native_probabilities[i] for i in indices) / len(indices) for indices in expected_mapping]
        expected_windows = []
        expected_window_ids = []
        for window in plan["eligible_windows"]:
            start = int(window["token_start"])
            count = int(window.get("slot_count", 4))
            expected_windows.append(math.fsum(expected_local[start : start + count]) / count)
            expected_window_ids.append(str(window["window_id"]))
        for expected, stored in zip(expected_local, shared["shared_bpe_token_probabilities"]):
            rag_mapping_max_delta = max(rag_mapping_max_delta, abs(expected - float(stored)))
        for expected, stored in zip(expected_windows, shared["shared_k4_window_probabilities"]):
            rag_mapping_max_delta = max(rag_mapping_max_delta, abs(expected - float(stored)))
        rag_mapping_max_delta = max(
            rag_mapping_max_delta,
            abs(max(expected_windows) - float(shared["shared_answer_probability"])),
            abs(max(native_probabilities) - float(shared["author_native_answer_probability"])),
        )
        if expected_window_ids != shared["window_ids"]:
            rag_mapping_index_failures += 1

        if partition == "calibration":
            rag_cal_raw[answer_id] = raw
            rag_cal_shared[answer_id] = shared

plan_file_hash = plan_digest.hexdigest()
raw_file_hash = raw_digest.hexdigest()
shared_file_hash = shared_digest.hexdigest()
A.remember_hash(rag_plan_path, plan_file_hash)
A.remember_hash(rag_raw_path, raw_file_hash)
A.remember_hash(rag_shared_path, shared_file_hash)
A.equal("ragognizer.formal.plan_sha256", plan_file_hash, rag_freeze["plan_sha256"])
A.equal("ragognizer.formal.raw_sha256", raw_file_hash, rag_freeze["raw_scores_sha256"])
A.equal("ragognizer.formal.shared_sha256", shared_file_hash, rag_freeze["adapted_scores_sha256"])
A.equal("ragognizer.formal.plan_rows", len(rag_answer_ids), 3839)
A.equal("ragognizer.formal.unique_answer_ids", len(set(rag_answer_ids)), 3839)
A.equal("ragognizer.formal.partition_answers", dict(rag_partition_counts), {"fit": 3680, "calibration": 159})
A.equal("ragognizer.formal.partition_windows", dict(rag_partition_windows), {"fit": 653979, "calibration": 42241})
A.equal("ragognizer.formal.total_windows", sum(rag_partition_windows.values()), 696220)
A.equal("ragognizer.formal.fit_groups", len(rag_groups["fit"]), 615)
A.equal("ragognizer.formal.calibration_groups", len(rag_groups["calibration"]), 154)
A.equal("ragognizer.formal.fit_cal_group_overlap", len(rag_groups["fit"] & rag_groups["calibration"]), 0)
A.check("ragognizer.formal.calibration_answer_set", set(rag_cal_shared) == answer_id_set)
A.equal("ragognizer.formal.plan_row_hash_failures", rag_plan_hash_failures, 0)
A.equal("ragognizer.formal.raw_hash_chain_failures", rag_raw_hash_failures, 0)
A.equal("ragognizer.formal.shared_hash_chain_failures", rag_shared_hash_failures, 0)
A.equal("ragognizer.formal.identity_constant_failures", rag_constant_failures, 0)
A.equal("ragognizer.formal.mapping_index_failures", rag_mapping_index_failures, 0)
A.close("ragognizer.formal.mapping_max_abs_difference", rag_mapping_max_delta, 0.0)
A.equal("ragognizer.formal.raw_final_chain", raw_chain, rag_freeze["raw_final_chain_sha256"])
A.equal("ragognizer.formal.shared_final_chain", shared_chain, rag_freeze["adapted_final_chain_sha256"])
A.equal("ragognizer.formal.quantization_values", sorted(rag_raw_quantization_values), ["None"])
A.equal("ragognizer.formal.coordinate_modes", dict(rag_coordinate_modes), {
    "author_pack_exact_fast_positions_verified": 3836,
    "fast_offsets_author_pack_byte_fallback": 3,
})
print("stage: ragognizer 3839 raw/adapted chains and mapping verified", flush=True)

rag_window_score_by_id: dict[str, float] = {}
for answer_id, row in rag_cal_shared.items():
    rag_window_score_by_id.update(
        {str(wid): float(score) for wid, score in zip(row["window_ids"], row["shared_k4_window_probabilities"])}
    )
A.check("ragognizer.calibration_window_set", set(rag_window_score_by_id) == window_id_set)
rag_window = np.asarray([rag_window_score_by_id[wid] for wid in window_ids], dtype=np.float64)
rag_answer = np.asarray([rag_cal_shared[rid]["shared_answer_probability"] for rid in answer_ids], dtype=np.float64)
A.close(
    "ragognizer.answer_equals_max_common_windows",
    float(np.max(np.abs(answer_max_from_ordered_windows(rag_window, cal_answers, cal_windows) - rag_answer))),
    0.0,
)
metrics["RAGognizer"] = {"window": f1_opt(window_labels, rag_window), "answer": f1_opt(answer_labels, rag_answer)}
rag_reported = rag_result["shared_main_table"]
compare_metric("ragognizer.window", metrics["RAGognizer"]["window"], rag_reported["four_bpe_window"])
compare_metric("ragognizer.answer", metrics["RAGognizer"]["answer"], rag_reported["answer_max_shared_window"])
score_sources["RAGognizer"] = str(rag_shared_path.relative_to(BASE))


# Formal vs earlier provisional calibration: exact raw and mapped values, then metrics.
provisional_raw_rows = load_jsonl(RAG / "PROVISIONAL_CAL_NATIVE.jsonl")
provisional_shared_rows = load_jsonl(RAG / "PROVISIONAL_CAL_SHARED.jsonl")
A.equal("ragognizer.provisional.raw_rows", len(provisional_raw_rows), 159)
A.equal("ragognizer.provisional.shared_rows", len(provisional_shared_rows), 159)
provisional_raw_by_id = {str(row["answer_id"]): row for row in provisional_raw_rows}
provisional_shared_by_id = {str(row["answer_id"]): row for row in provisional_shared_rows}
raw_numeric_equal = 0
shared_numeric_equal = 0
for answer_id in answer_ids:
    formal_raw = rag_cal_raw[answer_id]
    provisional_raw = provisional_raw_by_id[answer_id]
    if (
        formal_raw["author_response_token_ids"] == provisional_raw["author_response_token_ids"]
        and formal_raw["author_response_token_char_intervals"] == provisional_raw["author_response_token_char_intervals"]
        and formal_raw["author_response_token_probabilities"] == provisional_raw["author_response_token_probabilities"]
        and formal_raw["author_native_preds_0_6523_appendix_only"]
        == provisional_raw["author_native_preds_0_6523_appendix_only"]
        and formal_raw["raw_row_sha256"] == provisional_raw["raw_row_sha256"]
    ):
        raw_numeric_equal += 1
    formal_shared = rag_cal_shared[answer_id]
    provisional_shared = provisional_shared_by_id[answer_id]
    if (
        formal_shared["shared_bpe_token_probabilities"] == provisional_shared["shared_bpe_token_probabilities"]
        and formal_shared["window_ids"] == provisional_shared["window_ids"]
        and formal_shared["shared_k4_window_probabilities"] == provisional_shared["shared_k4_window_probabilities"]
        and formal_shared["shared_answer_probability"] == provisional_shared["shared_answer_probability"]
        and formal_shared["author_native_answer_probability"] == provisional_shared["author_native_answer_probability"]
    ):
        shared_numeric_equal += 1
A.equal("ragognizer.formal_equals_provisional_native_rows_exact", raw_numeric_equal, 159)
A.equal("ragognizer.formal_equals_provisional_shared_rows_exact", shared_numeric_equal, 159)
provisional_result = load_json(RAG / "PROVISIONAL_CAL_RESULTS.json")
A.check(
    "ragognizer.formal_equals_provisional_main_metrics_exact",
    rag_result["shared_main_table"]["four_bpe_window"]
    == provisional_result["shared_main_table"]["four_bpe_window"]
    and rag_result["shared_main_table"]["answer_max_shared_window"]
    == provisional_result["shared_main_table"]["answer_max_shared_window"],
)
A.check(
    "ragognizer.formal_equals_provisional_appendix_metrics_exact",
    rag_result["appendix_only"]["author_native_answer_max_token_at_0_6523"]
    == provisional_result["appendix_only"]["author_native_answer_max_token_at_0_6523"],
)
print("stage: ragognizer metrics and provisional equality verified", flush=True)


# RAGognizer frozen file/result chain.
rag_run_identity = rag_gpu["run_identity"]
A.equal("ragognizer.run_identity_payload_sha256", sha256_json(rag_run_identity), rag_gpu["run_identity_sha256"])
A.equal("ragognizer.gpu_audit.output_sha256", rag_gpu["output_sha256"], A.sha256_file(rag_raw_path))
A.equal("ragognizer.adapter_audit.raw_sha256", rag_adapter_audit["raw_sha256"], A.sha256_file(rag_raw_path))
A.equal("ragognizer.adapter_audit.output_sha256", rag_adapter_audit["output_sha256"], A.sha256_file(rag_shared_path))
A.equal("ragognizer.adapter_audit.gpu_audit_sha256", rag_adapter_audit["gpu_run_audit_sha256"], A.sha256_file(RAG / "GPU_RUN_AUDIT.json"))
freeze_payload = dict(rag_freeze)
stored_freeze_payload_hash = freeze_payload.pop("score_freeze_payload_sha256")
A.equal("ragognizer.score_freeze_payload_sha256", sha256_json(freeze_payload), stored_freeze_payload_hash)
A.equal("ragognizer.score_freeze.adapter_audit_sha256", rag_freeze["adapter_audit_sha256"], A.sha256_file(RAG / "ADAPTER_AUDIT.json"))
A.equal("ragognizer.score_freeze.gpu_audit_sha256", rag_freeze["gpu_run_audit_sha256"], A.sha256_file(RAG / "GPU_RUN_AUDIT.json"))
A.equal("ragognizer.score_freeze.evaluator_sha256", rag_freeze["evaluator_script_sha256"], A.sha256_file(RAG / "evaluate_shared.py"))
A.equal("ragognizer.result.score_freeze_file_sha256", rag_result["score_freeze_file_sha256"], A.sha256_file(RAG / "SCORE_FREEZE.json"))
A.equal("ragognizer.result.score_freeze_payload_sha256", rag_result["score_freeze_payload_sha256"], stored_freeze_payload_hash)
A.equal("ragognizer.status.result_sha256", rag_status["result_sha256"], A.sha256_file(RAG / "CAL_RESULTS.json"))
A.equal("ragognizer.status.score_freeze_sha256", rag_status["score_freeze_sha256"], A.sha256_file(RAG / "SCORE_FREEZE.json"))
A.equal("ragognizer.status.adapted_sha256", rag_status["adapted_scores_sha256"], A.sha256_file(rag_shared_path))
A.check("ragognizer.status.no_substitution", rag_status["substitution_used"] is False)
A.check("ragognizer.status.no_quantization", rag_status["quantization_used"] is False)
A.equal("ragognizer.gpu_audit.loaded_dtype", rag_gpu["loaded_parameter_dtype_tensor_counts"], {"torch.bfloat16": 746})
A.equal("ragognizer.gpu_audit.torch_dtype_argument", rag_gpu["torch_dtype_argument"], "torch.bfloat16")
A.check("ragognizer.gpu_audit.quantization_none", rag_gpu["quantization"] is None)
A.check("ragognizer.gpu_audit.postprocessor_false", rag_gpu["postprocessor"] is False)
A.equal("ragognizer.gpu_audit.route", rag_gpu["route"], "model_card_default_transformer_heads_with_accelerate_cpu_offload")

formal_chain_expected = {
    "plan_sha256": A.sha256_file(rag_plan_path),
    "runner_sha256": A.sha256_file(RAG / "run_inference.py"),
    "gpu_run_audit_sha256": A.sha256_file(RAG / "GPU_RUN_AUDIT.json"),
    "native_probabilities_sha256": A.sha256_file(rag_raw_path),
    "raw_final_chain_sha256": raw_chain,
    "adapter_sha256": A.sha256_file(RAG / "adapter.py"),
    "adapter_audit_sha256": A.sha256_file(RAG / "ADAPTER_AUDIT.json"),
    "shared_probabilities_sha256": A.sha256_file(rag_shared_path),
    "adapted_final_chain_sha256": shared_chain,
    "evaluator_sha256": A.sha256_file(RAG / "evaluate_shared.py"),
    "score_freeze_sha256": A.sha256_file(RAG / "SCORE_FREEZE.json"),
    "score_freeze_payload_sha256": stored_freeze_payload_hash,
    "cal_results_sha256": A.sha256_file(RAG / "CAL_RESULTS.json"),
    "scoring_status_sha256": A.sha256_file(RAG / "SCORING_STATUS.json"),
    "independent_auditor_sha256": A.sha256_file(RAG / "audit_scores_independent.py"),
    "independent_audit_sha256": A.sha256_file(RAG / "SCORE_INDEPENDENT_AUDIT.json"),
}
for key, expected in formal_chain_expected.items():
    A.equal("ragognizer.artifact_hashes.formal_chain." + key, rag_artifacts["formal_chain"][key], expected)

artifact_hash_failures: list[str] = []
for entry in rag_artifacts["files"]:
    path = RAG / Path(entry["path"])
    if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or A.sha256_file(path) != entry["sha256"]:
        artifact_hash_failures.append(entry["path"])
A.equal("ragognizer.artifact_hashes.file_count", len(rag_artifacts["files"]), rag_artifacts["file_count"])
A.check("ragognizer.artifact_hashes.all_files", not artifact_hash_failures, artifact_hash_failures)
print("stage: ragognizer formal artifact chain verified", flush=True)


# Official checkpoint/source identity, detection-head architecture, and runtime aliases.
model_manifest_path = RAG / "MODEL_MANIFEST.json"
model_manifest = load_json(model_manifest_path)
A.equal("ragognizer.model_manifest_sha256", A.sha256_file(model_manifest_path), rag_gpu["model_manifest_sha256"])
A.equal(
    "ragognizer.checkpoint_revision",
    model_manifest["official_model"]["revision"],
    "2b58ab9aa73f6d499ec72a38ced0976caf78b26f",
)
official_model_dir = WORKSPACE / Path(model_manifest["official_model"]["local_path"])
model_file_failures: list[str] = []
for entry in model_manifest["official_model"]["files"]:
    path = official_model_dir / Path(entry["path"])
    if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or A.sha256_file(path) != entry["sha256"]:
        model_file_failures.append(entry["path"])
A.equal("ragognizer.official_model_file_count", len(model_manifest["official_model"]["files"]), 10)
A.check("ragognizer.official_model_all_file_hashes", not model_file_failures, model_file_failures)

rag_repo = BASE / "third_party" / "RAGognizer"
heads_repo = BASE / "third_party" / "transformer-heads"
rag_commit = subprocess.check_output(["git", "-C", str(rag_repo), "rev-parse", "HEAD"], text=True).strip()
heads_commit = subprocess.check_output(["git", "-C", str(heads_repo), "rev-parse", "HEAD"], text=True).strip()
A.equal("ragognizer.official_repo_commit", rag_commit, model_manifest["official_source"]["commit"])
A.equal(
    "ragognizer.transformer_heads_commit",
    heads_commit,
    model_manifest["resolved_unpinned_transformer_heads_dependency"]["commit_used_for_this_audit"],
)
source_file_failures: list[str] = []
for entry in model_manifest["official_source"]["files"]:
    path = rag_repo / Path(entry["path"])
    if not path.is_file() or A.sha256_file(path) != entry["sha256"]:
        source_file_failures.append("RAGognizer/" + entry["path"])
for entry in model_manifest["resolved_unpinned_transformer_heads_dependency"]["files"]:
    path = heads_repo / Path(entry["path"])
    if not path.is_file() or A.sha256_file(path) != entry["sha256"]:
        source_file_failures.append("transformer-heads/" + entry["path"])
A.check("ragognizer.official_source_all_hashes", not source_file_failures, source_file_failures)

head_config = load_json(official_model_dir / "ft_llm" / "head_configs.json")["hallu_head_neg_16"]
A.equal("ragognizer.head.name", head_config["name"], "hallu_head_neg_16")
A.equal("ragognizer.head.layer_hook", head_config["layer_hook"], -16)
A.equal("ragognizer.head.architecture_config", [head_config["in_size"], head_config["hidden_size"], head_config["hidden_size"], head_config["num_outputs"]], [4096, 1024, 1024, 1])
A.equal("ragognizer.head.num_layers", head_config["num_layers"], 3)
A.equal("ragognizer.head.output_activation", head_config["output_activation"], "linear")
A.check("ragognizer.head.output_bias_false", head_config["output_bias"] is False)

def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_length))

head_header = safetensors_header(official_model_dir / "ft_llm" / "hallu_head_neg_16.safetensors")
head_shapes = {key: value["shape"] for key, value in head_header.items() if key != "__metadata__"}
expected_head_shapes = {
    "lins.0.bias": [1024],
    "lins.0.weight": [1024, 4096],
    "lins.1.bias": [1024],
    "lins.1.weight": [1024, 1024],
    "lins.2.weight": [1, 1024],
}
A.equal("ragognizer.head.checkpoint_shapes", head_shapes, expected_head_shapes)
A.equal(
    "ragognizer.head.checkpoint_dtypes",
    sorted({value["dtype"] for key, value in head_header.items() if key != "__metadata__"}),
    ["F32"],
)
head_parameter_count = sum(math.prod(shape) for shape in head_shapes.values())
A.equal("ragognizer.head.parameter_count", head_parameter_count, 5245952)

adapter_config = load_json(official_model_dir / "ft_llm" / "adapter_config.json")
A.equal("ragognizer.lora.rank", adapter_config["r"], 32)
A.equal("ragognizer.lora.alpha", adapter_config["lora_alpha"], 16)
A.close("ragognizer.lora.dropout", adapter_config["lora_dropout"], 0.0)
A.check("ragognizer.lora.inference_mode", adapter_config["inference_mode"] is True)

runtime_alias = load_json(RAG / "RUNTIME_ALIAS_AUDIT.json")
runtime_failures: list[str] = []
hardlink_failures: list[str] = []
for entry in runtime_alias["files"]:
    path = WORKSPACE / Path(entry["path"])
    source = WORKSPACE / Path(entry["source"])
    if (
        not path.is_file()
        or not source.is_file()
        or path.stat().st_size != int(entry["bytes"])
        or source.stat().st_size != int(entry["bytes"])
        or A.sha256_file(path) != entry["sha256"]
        or A.sha256_file(source) != entry["sha256"]
    ):
        runtime_failures.append(entry["path"])
    if entry["mode"] == "hardlink" and not os.path.samefile(path, source):
        hardlink_failures.append(entry["path"])
A.check("ragognizer.runtime_alias_all_hashes", not runtime_failures, runtime_failures)
A.check("ragognizer.runtime_alias_hardlinks", not hardlink_failures, hardlink_failures)
print("stage: ragognizer model/source/runtime identity verified", flush=True)

runner_text = (RAG / "run_inference.py").read_text(encoding="utf-8")
A.check("ragognizer.runner_loads_integrated_heads", "load_lora_with_heads(" in runner_text and "outputs.preds_by_head" in runner_text)
A.check("ragognizer.runner_selects_only_hallu_head_neg_16", 'matching != ["hallu_head_neg_16"]' in runner_text)
A.check("ragognizer.runner_sigmoid_native_logit", "torch.sigmoid(outputs.preds_by_head[matching[0]].flatten())" in runner_text)
A.check("ragognizer.runner_bfloat16_no_quantization", "quantization_config=None" in runner_text and "torch_dtype=torch.bfloat16" in runner_text)
A.check("ragognizer.runner_no_separate_mlp", "mlp_state.pt" not in runner_text)


# ---------------------------------------------------------------------------
# Public-document consistency and final mechanical comparison.
# ---------------------------------------------------------------------------
protocol_text = (BASE / "BASELINE_PROTOCOL.md").read_text(encoding="utf-8")
results_text = (BASE / "FORMAL_BASELINE_RESULTS.md").read_text(encoding="utf-8")
freeze_text = (BASE / "FORMAL_BASELINE_FREEZE.md").read_text(encoding="utf-8")
status_text = (BASE / "CURRENT_STATUS.md").read_text(encoding="utf-8")

expected_rounded = {
    "Lookback Lens": ("0.600882", "0.845455"),
    "GHOST": ("0.320361", "0.772201"),
    "LUMINA": ("0.331299", "0.785047"),
    "ReDeEP": ("0.329560", "0.772358"),
    "RAGognizer": ("0.507511", "0.806867"),
    "ours": ("0.690281", "0.891089"),
}
for method, pair in expected_rounded.items():
    actual_pair = (
        f"{metrics[method]['window']['f1_opt']['f1']:.6f}",
        f"{metrics[method]['answer']['f1_opt']['f1']:.6f}",
    )
    A.equal("rounded_metrics." + method, actual_pair, pair)
    A.check("documents.results_contains." + method, pair[0] in results_text and pair[1] in results_text)

A.check(
    "documents.protocol_redeep_completed",
    "ReDeEP论文公式逐token分数" in protocol_text
    and "完成并独立复算" in protocol_text
    and "0.329560/0.772358" in protocol_text,
)
A.check("documents.protocol_no_stale_redeep_pending", "ReDeEP已冻结论文公式、固定K/α/β与fit内头层选择规则，正在执行统一迁移" not in protocol_text)
A.check("documents.protocol_ragognizer_completed", "0.507511/0.806867" in protocol_text)
A.check("documents.status_candidate_and_baselines_complete", "0.690281/0.891089" in status_text and "0.507511/0.806867" in status_text and "0.329560/0.772358" in status_text)
A.check("documents.freeze_redeep_hash", A.sha256_file(redeep_manifest_path) in freeze_text)
A.check("documents.freeze_rag_artifact_hash", A.sha256_file(RAG / "ARTIFACT_HASHES.json") in freeze_text)
A.check("documents.freeze_rag_checkpoint_revision", model_manifest["official_model"]["revision"] in freeze_text)
A.check("documents.shared_protocol_counts", all(value in protocol_text for value in ["3,680", "159", "42,241", "5,984", "696,220"]))

baseline_names = ["Lookback Lens", "GHOST", "LUMINA", "ReDeEP", "RAGognizer"]
max_baseline_window_name = max(baseline_names, key=lambda name: metrics[name]["window"]["f1_opt"]["f1"])
max_baseline_answer_name = max(baseline_names, key=lambda name: metrics[name]["answer"]["f1_opt"]["f1"])
max_baseline_window = metrics[max_baseline_window_name]["window"]["f1_opt"]["f1"]
max_baseline_answer = metrics[max_baseline_answer_name]["answer"]["f1_opt"]["f1"]
ours_window = metrics["ours"]["window"]["f1_opt"]["f1"]
ours_answer = metrics["ours"]["answer"]["f1_opt"]["f1"]
mechanical = {
    "ours_window_f1": ours_window,
    "max_completed_baseline_window_f1": max_baseline_window,
    "max_completed_baseline_window_method": max_baseline_window_name,
    "window_margin": ours_window - max_baseline_window,
    "window_pass": ours_window >= max_baseline_window,
    "ours_answer_f1": ours_answer,
    "max_completed_baseline_answer_f1": max_baseline_answer,
    "max_completed_baseline_answer_method": max_baseline_answer_name,
    "answer_margin": ours_answer - max_baseline_answer,
    "answer_pass": ours_answer >= max_baseline_answer,
}
mechanical["both_pass"] = mechanical["window_pass"] and mechanical["answer_pass"]
A.check("mechanical.ours_ge_all_completed_baselines_both_metrics", mechanical["both_pass"], mechanical)


result = {
    "version": "final-baseline-audit-v4",
    "status": "PASS" if A.passed else "FAIL",
    "scope": "read-only fit/calibration audit; no retired test artifact opened; no GPU used",
    "common_evaluation": {
        "answers": len(cal_answers),
        "groups": len({row["group_id"] for row in cal_answers}),
        "windows": len(cal_windows),
        "positive_answers": int(answer_labels.sum()),
        "positive_windows": int(window_labels.sum()),
        "threshold_rule": "max F1, then precision, then higher threshold; score >= threshold",
    },
    "metrics": metrics,
    "score_sources": score_sources,
    "ragognizer": {
        "full_answers": len(rag_answer_ids),
        "partition_answers": dict(rag_partition_counts),
        "partition_windows": dict(rag_partition_windows),
        "author_tokens": rag_author_tokens,
        "shared_bpe_slots": rag_shared_slots,
        "mapping_max_abs_difference": rag_mapping_max_delta,
        "formal_equals_provisional_native_rows_exact": raw_numeric_equal,
        "formal_equals_provisional_shared_rows_exact": shared_numeric_equal,
        "formal_hash_chain": formal_chain_expected,
        "model_revision": model_manifest["official_model"]["revision"],
        "ragognizer_commit": rag_commit,
        "transformer_heads_commit": heads_commit,
        "loaded_parameter_dtype_tensor_counts": rag_gpu["loaded_parameter_dtype_tensor_counts"],
        "quantization": rag_gpu["quantization"],
        "postprocessor": rag_gpu["postprocessor"],
        "head_architecture": [4096, 1024, 1024, 1],
        "head_parameter_count": head_parameter_count,
    },
    "redeep": {
        "full_answers": len(redeep_answer_ids),
        "full_windows": len(redeep_window_ids),
        "raw_files_hashed": len(redeep_raw_manifest["entries"]),
        "official_repo_commit": redeep_commit,
    },
    "mechanical_comparison": mechanical,
    "public_document_consistency": {
        "files": ["BASELINE_PROTOCOL.md", "FORMAL_BASELINE_RESULTS.md", "FORMAL_BASELINE_FREEZE.md", "CURRENT_STATUS.md"],
        "consistent": all(
            row["status"] == "PASS" for row in A.checks if row["name"].startswith("documents.")
        ),
    },
    "test_dataset_artifacts_opened": [],
    "gpu_used": False,
    "checks": A.checks,
    "check_counts": dict(Counter(row["status"] for row in A.checks)),
}

audit_json_path = AUDIT_DIR / "AUDIT.json"
with audit_json_path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

table_rows = []
for method in ["ours", *baseline_names]:
    table_rows.append(
        f"| {method} | {metrics[method]['window']['f1_opt']['f1']:.6f} | "
        f"{metrics[method]['answer']['f1_opt']['f1']:.6f} | "
        f"{metrics[method]['window']['auroc']:.6f} | {metrics[method]['answer']['auroc']:.6f} |"
    )
failed = [row for row in A.checks if row["status"] == "FAIL"]
failure_lines = "\n".join(f"- `{row['name']}`: {row['detail']}" for row in failed) or "- 无。"
report = f"""# 最终正式基线独立验收 v4

**结论：{result['status']}。** 本次只读取冻结的 fit/calibration 产物，未打开退役 test 数据，也未使用 GPU。

统一计分分母独立核对为 **159 个回答、154 个材料组、42,241 个 4-BPE 窗口**；正类为 100 个回答、5,984 个窗口。所有方法都按窗口与整答分别执行同一条规则：`F1 → precision → 更高阈值`，且 `score >= threshold`。

| 方法 | 窗口 F1 | 整答 F1 | 窗口 AUROC | 整答 AUROC |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

机械比较结果：最强正式基线的窗口 F1 为 **{max_baseline_window:.6f}（{max_baseline_window_name}）**，整答 F1 为 **{max_baseline_answer:.6f}（{max_baseline_answer_name}）**。本文候选分别为 **{ours_window:.6f} / {ours_answer:.6f}**，领先 **{mechanical['window_margin']:.6f} / {mechanical['answer_margin']:.6f}**；因此“两项均不低于全部已完成正式基线”判定为 **{'PASS' if mechanical['both_pass'] else 'FAIL'}**。

RAGognizer 的正式链通过以下核验：全 3,839 答覆盖（fit 3,680、cal 159），共 696,220 个合格窗口；全部 raw/adapted 行的规范 JSON 哈希和前后链独立重算；raw→字符交叠→本地 BPE→4-BPE 窗口→整答 max 全量重算，最大绝对差为 `{rag_mapping_max_delta:.3g}`；formal 与 provisional 的 159 条原生概率和映射概率逐值完全一致；官方 checkpoint revision、两个源码 commit、LoRA、`hallu_head_neg_16` 的 4096→1024→1024→1 结构、BF16 运行记录和无量化状态均通过。完整 artifact、模型文件、运行别名及两份 Llama 权重 shard 的 SHA256 也已重新计算。

ReDeEP 的 3,839 个 raw 文件及 30 项最终清单哈希全部复核；其论文公式主结果与统一 cal 分数独立复算一致。Lookback、LUMINA、GHOST 和本文候选也均从各自冻结分数重新计算，并核对回答分数来自共同窗口 max（GHOST 为其冻结整答概率的统一广播）。

`BASELINE_PROTOCOL.md`、`FORMAL_BASELINE_RESULTS.md`、`FORMAL_BASELINE_FREEZE.md`、`CURRENT_STATUS.md` 的完成状态、分数、覆盖范围和冻结身份一致。这里仍是反复使用的 calibration 开发结果，不能写成独立测试成绩。

失败项：

{failure_lines}

机器可读结果：`AUDIT.json`。共 {len(A.checks)} 项检查：{result['check_counts']}。
"""
(AUDIT_DIR / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
print(canonical_json({"status": result["status"], "checks": result["check_counts"], "mechanical": mechanical}))
