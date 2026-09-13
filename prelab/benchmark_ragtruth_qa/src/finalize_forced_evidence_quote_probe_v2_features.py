"""Label-blind CPU finalizer for forced-evidence quote probe GPU records.

Formal execution is possible only after the 3,776 per-claim GPU commits and
their extraction receipt exist.  The finalizer verifies every hash and row in
the frozen clean-input order, materializes float32 P1/P2/P3 matrices, and emits
the exact two-file contract consumed by the evaluator plus a stronger lineage
receipt.  This module never opens gold, calibration, or official-test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import torch

import evaluate_forced_evidence_quote_probe_v2 as evaluator
import run_forced_evidence_quote_probe_v2_gpu as gpu_runner


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V1_OUT = ROOT / "results/forced_evidence_quote_probe_v1"
V1_RESEARCH = ROOT / "research/forced_evidence_quote_probe_v1"
OUT = ROOT / "results/forced_evidence_quote_probe_v2"
RESEARCH = ROOT / "research/forced_evidence_quote_probe_v2"
INPUT_PATH = V1_OUT / "label_free_inputs.jsonl"
CPU_MANIFEST_PATH = V1_OUT / "MANIFEST.json"
PREPARATION_PATH = V1_OUT / "PREPARATION.json"
PROTOCOL_PATH = V1_RESEARCH / "PROTOCOL.md"
PLAN_PATH = V1_RESEARCH / "PLAN.json"
NUMERICAL_PROTOCOL_PATH = RESEARCH / "NUMERICAL_PROTOCOL.md"
GPU_RUNNER_PATH = HERE / "run_forced_evidence_quote_probe_v2_gpu.py"
CPU_HELPER_PATH = HERE / "run_forced_evidence_quote_probe_v1.py"
INPUT_BUILDER_PATH = HERE / "prepare_forced_evidence_quote_probe_v1.py"
GPU_CPU_SELFCHECK_PATH = OUT / "GPU_RUNNER_CPU_SELFCHECK.json"
GPU_RUNNER_REVIEW_PATH = RESEARCH / "GPU_RUNNER_INDEPENDENT_REVIEW.json"
GPU_SMOKE_PATH = OUT / "GPU_SMOKE.json"
GPU_SMOKE_REVIEW_PATH = RESEARCH / "GPU_SMOKE_INDEPENDENT_REVIEW.json"
EVALUATOR_REVIEW_PATH = RESEARCH / "EVALUATOR_INDEPENDENT_REVIEW.json"
EVALUATOR_SELFTEST_PATH = OUT / "CPU_EVALUATOR_SELFTEST.json"
FINALIZER_REVIEW_PATH = RESEARCH / "FINALIZER_INDEPENDENT_REVIEW.json"
EXTRACTION_PATH = OUT / "GPU_EXTRACTION_COMPLETE.json"
RECORD_DIR = OUT / "gpu_claim_records"
FEATURE_DIR = OUT / "frozen_features"
FEATURE_ARRAY_NAME = "claim_features.npz"
FEATURE_MANIFEST_NAME = "FEATURE_MANIFEST.json"
FEATURE_FREEZE_NAME = "FEATURE_FREEZE.json"
LINEAGE_NAME = "FINALIZER_LINEAGE.json"
COMPLETE_NAME = "FINALIZER_COMPLETE.json"
SELFTEST_PATH = OUT / "CPU_FEATURE_FINALIZER_SELFTEST.json"

VERSION = "forced-evidence-quote-feature-finalizer-v2"
FEATURE_SCHEMA_VERSION = "forced-evidence-quote-features-v2"
EXPECTED_ROWS = 3_776
EXPECTED_ANSWERS = 256
EXPECTED_INPUT_SHA = "ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777"
EXPECTED_PROTOCOL_SHA = "78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa"
EXPECTED_PLAN_SHA = "6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c"
EXPECTED_NUMERICAL_PROTOCOL_SHA = "ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1"
EXPECTED_RUNNER_SHA = "eb4a51353ba4edaff2cff877f576ff206f1fca1d6bd820928fb0db4acebfcca2"
EXPECTED_GPU_CPU_SELFCHECK_SHA = "a6ef47c093b11f1665e340b703bbd3275c37bcac07025fe9ae74eed6381c86cf"
EXPECTED_CPU_MANIFEST_SHA = "b6808ae4f7a25b2bf2a126bdcb3bb3f2adbe99a12762724080367c181a49feb1"
EXPECTED_PREPARATION_SHA = "400f577f9e805ddfbfb385c13ed312e8394e8304a1a667ff69181a398a1cefc6"
EXPECTED_CPU_REVIEW_SHA = "fdfa7f4dcf53ed9be4fee81d604b910c44e2db8648631ff0aa04cbe489e4d0bb"
EXPECTED_CPU_HELPER_SHA = "3380729506c5dc23c4430313a1eb0cba6777b9ca24981d11170a2cc2256a0598"
EXPECTED_INPUT_BUILDER_SHA = "fb71081166421a01e9a73e6f3ca873ebd1d8e2b738b5dc5f1b028803a20c6c17"
EXPECTED_FEATURE_QA_SHA = "2b5c4011b9fea11c6888fc72e0930068ec42ce8a1ad79d2b7e4a8c6d27d78d73"
EXPECTED_BM25_SHA = "353129f81e194b87542ec60aa74a6db012cb437108384805ceaa1dd13d1bd64a"
EXPECTED_WDDM_SHA = "a55a7cb8a57e783eab39a09195880ed671c9f7f1b06ee77005f0f362b9fbc7fc"
# The source and synthetic receipt are frozen here; the independent review
# remains an explicit formal gate and is not fabricated by this implementation.
EXPECTED_EVALUATOR_SHA = "a80e1ac7a107d96e32dd55e47e4c8535c5e7a189e491335f519acb604883ac8f"
EXPECTED_EVALUATOR_SELFTEST_SHA = "c1379a9412a54d43d639af532fdcb95ce3a007324d50d06bbee1ba7309d97c78"
EXPECTED_EVALUATOR_REVIEW_SHA = "6e972db69d788e811edf268900bdaff30600a09dfbe76f13aec4f3797340aab0"
FEATURE_KEYS = ("P1", "P2", "P3")
FEATURE_DIMS = {"P1": 21, "P2": 549, "P3": 549}
OUTPUT_FIELDS = gpu_runner.cpu.OUTPUT_FIELDS
FORBIDDEN_KEYS = frozenset((
    "labels", "gold_label", "risk_mask", "risk_bpe_indices",
    "risk_bpe_fraction", "label_type", "answer_risk", "existing_scores",
    "original_response", "quality", "answer_generator",
))


class FinalizerFailure(RuntimeError):
    pass


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise FinalizerFailure(
                    f"Invalid JSONL at {path}:{line_number}") from error


def reject_taint(value) -> None:
    if isinstance(value, dict):
        overlap = set(value) & FORBIDDEN_KEYS
        if overlap:
            raise FinalizerFailure(f"Forbidden payload: {sorted(overlap)}")
        for child in value.values():
            reject_taint(child)
    elif isinstance(value, list):
        for child in value:
            reject_taint(child)


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refuse overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("xb") as handle:
        handle.write(json_bytes(value)); handle.flush(); os.fsync(handle.fileno())
    pending.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refuse overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    pending.replace(path)


def row_keys_digest(rows) -> str:
    serial = "".join(
        f"{row['response_id']}\t{row['microclaim_id']}\t"
        f"{int(row['microclaim_index'])}\n" for row in rows)
    return digest(serial)


def load_clean_rows():
    if sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA:
        raise FinalizerFailure("Protocol changed")
    if sha(PLAN_PATH) != EXPECTED_PLAN_SHA:
        raise FinalizerFailure("Plan changed")
    if sha(NUMERICAL_PROTOCOL_PATH) != EXPECTED_NUMERICAL_PROTOCOL_SHA:
        raise FinalizerFailure("V2 numerical protocol changed")
    manifest = read_json(CPU_MANIFEST_PATH)
    if manifest.get("sole_GPU_input") != INPUT_PATH.name:
        raise FinalizerFailure("CPU manifest sole input changed")
    if manifest["files_sha256"].get(INPUT_PATH.name) != EXPECTED_INPUT_SHA:
        raise FinalizerFailure("CPU manifest input hash changed")
    if sha(INPUT_PATH) != EXPECTED_INPUT_SHA:
        raise FinalizerFailure("Clean input bytes changed")
    rows = list(json_lines(INPUT_PATH))
    if len(rows) != EXPECTED_ROWS:
        raise FinalizerFailure("Clean input row count changed")
    for row in rows:
        if tuple(row) != OUTPUT_FIELDS:
            raise FinalizerFailure("Clean input schema/order changed")
        reject_taint(row)
        if row["model"] != "llama-2-7b-chat":
            raise FinalizerFailure("Non-native model row")
    if len({row["response_id"] for row in rows}) != EXPECTED_ANSWERS:
        raise FinalizerFailure("Answer count changed")
    if len({row["group_id"] for row in rows}) != EXPECTED_ANSWERS:
        raise FinalizerFailure("Group count changed")
    return rows


def require_keys(value: dict, keys, name: str, *, exact: bool = False) -> None:
    if not isinstance(value, dict):
        raise FinalizerFailure(f"{name} is not an object")
    keys = set(keys)
    missing = keys - set(value)
    extra = set(value) - keys if exact else set()
    if missing or extra:
        raise FinalizerFailure(
            f"{name} schema mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")


def require_false(value: dict, keys, name: str) -> None:
    for key in keys:
        if key not in value or value[key] is not False:
            raise FinalizerFailure(f"{name}.{key} must be exactly false")


def validate_static_locks() -> dict[str, str]:
    """Validate the reviewed CPU package and frozen executable sources."""
    cpu_review_path = V1_RESEARCH / "INDEPENDENT_REVIEW.json"
    for path in (GPU_CPU_SELFCHECK_PATH, GPU_RUNNER_REVIEW_PATH,
                 EVALUATOR_SELFTEST_PATH, EVALUATOR_REVIEW_PATH,
                 FINALIZER_REVIEW_PATH, cpu_review_path):
        if not path.is_file():
            raise FinalizerFailure(f"Required static review artifact absent: {path}")
    feature_qa_path = HERE / "feature_qa.py"
    bm25_path = HERE / "run_retrieved_evidence_nli_v1.py"
    wddm_path = HERE / "run_exact_subset_attribution_v2.py"
    locks = {
        "GPU_runner": sha(GPU_RUNNER_PATH),
        "GPU_runner_CPU_selfcheck": sha(GPU_CPU_SELFCHECK_PATH),
        "CPU_helper": sha(CPU_HELPER_PATH),
        "input_builder": sha(INPUT_BUILDER_PATH),
        "feature_qa": sha(feature_qa_path),
        "BM25_source": sha(bm25_path),
        "audited_WDDM_gate": sha(wddm_path),
        "CPU_manifest": sha(CPU_MANIFEST_PATH),
        "PREPARATION": sha(PREPARATION_PATH),
        "CPU_independent_review": sha(cpu_review_path),
        "evaluator": sha(Path(evaluator.__file__)),
        "evaluator_CPU_selftest": sha(EVALUATOR_SELFTEST_PATH),
        "evaluator_independent_review": sha(EVALUATOR_REVIEW_PATH),
        "protocol": sha(PROTOCOL_PATH),
        "plan": sha(PLAN_PATH),
        "numerical_protocol": sha(NUMERICAL_PROTOCOL_PATH),
        "label_free_inputs": sha(INPUT_PATH),
    }
    expected = {
        "GPU_runner": EXPECTED_RUNNER_SHA,
        "GPU_runner_CPU_selfcheck": EXPECTED_GPU_CPU_SELFCHECK_SHA,
        "CPU_helper": EXPECTED_CPU_HELPER_SHA,
        "input_builder": EXPECTED_INPUT_BUILDER_SHA,
        "feature_qa": EXPECTED_FEATURE_QA_SHA,
        "BM25_source": EXPECTED_BM25_SHA,
        "audited_WDDM_gate": EXPECTED_WDDM_SHA,
        "CPU_manifest": EXPECTED_CPU_MANIFEST_SHA,
        "PREPARATION": EXPECTED_PREPARATION_SHA,
        "CPU_independent_review": EXPECTED_CPU_REVIEW_SHA,
        "evaluator": EXPECTED_EVALUATOR_SHA,
        "evaluator_CPU_selftest": EXPECTED_EVALUATOR_SELFTEST_SHA,
        "evaluator_independent_review": EXPECTED_EVALUATOR_REVIEW_SHA,
        "protocol": EXPECTED_PROTOCOL_SHA,
        "plan": EXPECTED_PLAN_SHA,
        "numerical_protocol": EXPECTED_NUMERICAL_PROTOCOL_SHA,
        "label_free_inputs": EXPECTED_INPUT_SHA,
    }
    if locks != expected:
        changed = {key: {"expected": expected[key], "actual": locks[key]}
                   for key in locks if locks[key] != expected[key]}
        raise FinalizerFailure(f"Reviewed static lock changed: {changed}")

    check = read_json(GPU_CPU_SELFCHECK_PATH)
    if (check.get("status") != "passed_waiting_independent_GPU_runner_review"
            or check.get("runner_sha256") != EXPECTED_RUNNER_SHA
            or check.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA
            or check.get("plan_sha256") != EXPECTED_PLAN_SHA
            or check.get("numerical_protocol_sha256") !=
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            or check.get("sole_input_sha256") != EXPECTED_INPUT_SHA
            or int(check.get("rows", -1)) != EXPECTED_ROWS):
        raise FinalizerFailure("GPU runner CPU selfcheck binding changed")
    require_false(check, (
        "GPU_used", "CUDA_initialized", "pretrained_model_loaded",
        "gold_bearing_source_opened", "calibration_or_test_opened",
        "scoring_run"), "GPU runner CPU selfcheck")

    review = read_json(cpu_review_path)
    if review.get("review_status") != "PASS_CPU_PREPARATION_ONLY":
        raise FinalizerFailure("CPU independent preparation review is not PASS")
    reviewed = review.get("reviewed_final_hashes", {})
    for filename, expected_sha in {
            "PROTOCOL.md": EXPECTED_PROTOCOL_SHA,
            "PLAN.json": EXPECTED_PLAN_SHA,
            INPUT_BUILDER_PATH.name: EXPECTED_INPUT_BUILDER_SHA,
            CPU_HELPER_PATH.name: EXPECTED_CPU_HELPER_SHA,
            INPUT_PATH.name: EXPECTED_INPUT_SHA,
            CPU_MANIFEST_PATH.name: EXPECTED_CPU_MANIFEST_SHA,
            PREPARATION_PATH.name: EXPECTED_PREPARATION_SHA,
    }.items():
        if reviewed.get(filename) != expected_sha:
            raise FinalizerFailure(f"CPU review no longer binds {filename}")

    evaluator_check = read_json(EVALUATOR_SELFTEST_PATH)
    if not (evaluator_check.get("status") ==
            "passed_synthetic_CPU_only_no_real_scoring"
            and evaluator_check.get("script_sha256") == EXPECTED_EVALUATOR_SHA
            and evaluator_check.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA
            and evaluator_check.get("plan_sha256") == EXPECTED_PLAN_SHA
            and evaluator_check.get("numerical_protocol_sha256") ==
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            and evaluator_check.get("GPU_runner_sha256") == EXPECTED_RUNNER_SHA
            and evaluator_check.get("clean_input_sha256") == EXPECTED_INPUT_SHA
            and evaluator_check.get("sklearn_version") == "1.6.1"):
        raise FinalizerFailure("Evaluator CPU selftest binding changed")
    evaluator_review = read_json(EVALUATOR_REVIEW_PATH)
    if evaluator_review.get("decision", {}).get("status") != "PASS":
        raise FinalizerFailure("Evaluator independent V2 review is not PASS")
    reviewed_evaluator = evaluator_review.get("reviewed", {})
    if (reviewed_evaluator.get("evaluator", {}).get("sha256") !=
            EXPECTED_EVALUATOR_SHA
            or reviewed_evaluator.get("synthetic_selftest_receipt", {}).get(
                "sha256") != EXPECTED_EVALUATOR_SELFTEST_SHA
            or reviewed_evaluator.get("protocol", {}).get("sha256") !=
            EXPECTED_PROTOCOL_SHA
            or reviewed_evaluator.get("plan", {}).get("sha256") !=
            EXPECTED_PLAN_SHA
            or reviewed_evaluator.get("numerical_protocol", {}).get("sha256") !=
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            or reviewed_evaluator.get("GPU_runner", {}).get("sha256") !=
            EXPECTED_RUNNER_SHA
            or reviewed_evaluator.get("label_free_input_sha256") !=
            EXPECTED_INPUT_SHA):
        raise FinalizerFailure("Evaluator independent V2 review binding failed")

    validate_finalizer_review_gate(locks)
    locks["finalizer_CPU_selftest"] = sha(SELFTEST_PATH)
    locks["finalizer_independent_review"] = sha(FINALIZER_REVIEW_PATH)
    return locks


def validate_finalizer_review_gate(static_locks: dict[str, str]) -> None:
    if not SELFTEST_PATH.is_file():
        raise FinalizerFailure("Current finalizer synthetic selftest is absent")
    selfcheck = read_json(SELFTEST_PATH)
    if not (selfcheck.get("status") ==
            "passed_synthetic_CPU_only_formal_finalization_not_run"
            and selfcheck.get("finalizer_sha256") == sha(Path(__file__))
            and selfcheck.get("frozen_GPU_runner_sha256") == EXPECTED_RUNNER_SHA
            and selfcheck.get("formal_runner_lock_currently_matches") is True
            and selfcheck.get("formal_finalization_run") is False):
        raise FinalizerFailure("Finalizer selftest is stale or not clean")
    checks = selfcheck.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
            value is True for value in checks.values()):
        raise FinalizerFailure("Finalizer selftest checks are incomplete")
    require_false(selfcheck, (
        "GPU_used", "CUDA_initialized", "pretrained_model_loaded",
        "gold_read", "calibration_read", "official_test_read", "scoring_run",
    ), "finalizer selftest")
    if not FINALIZER_REVIEW_PATH.is_file():
        raise FinalizerFailure("Independent finalizer review is absent")
    review = read_json(FINALIZER_REVIEW_PATH)
    require_keys(review, (
        "status", "finalizer_sha256", "selftest_sha256", "runner_sha256",
        "evaluator_sha256", "protocol_sha256", "plan_sha256",
        "numerical_protocol_sha256",
        "formal_finalize_allowed"), "independent finalizer review")
    if not (review["status"] == "PASS"
            and review["finalizer_sha256"] == sha(Path(__file__))
            and review["selftest_sha256"] == sha(SELFTEST_PATH)
            and review["runner_sha256"] == EXPECTED_RUNNER_SHA
            and review["evaluator_sha256"] == EXPECTED_EVALUATOR_SHA
            and review["protocol_sha256"] == EXPECTED_PROTOCOL_SHA
            and review["plan_sha256"] == EXPECTED_PLAN_SHA
            and review["numerical_protocol_sha256"] ==
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            and review["formal_finalize_allowed"] is True):
        raise FinalizerFailure("Independent finalizer review binding failed")


def validate_execution_chain(extraction_path: Path = EXTRACTION_PATH) -> tuple[dict, dict]:
    """Validate GPU approvals and completion without opening feature labels."""
    for path in (GPU_RUNNER_REVIEW_PATH, GPU_SMOKE_PATH,
                 GPU_SMOKE_REVIEW_PATH, extraction_path):
        if not Path(path).is_file():
            raise FinalizerFailure(f"Required execution receipt absent: {path}")

    runner_review = read_json(GPU_RUNNER_REVIEW_PATH)
    require_keys(runner_review, (
        "status", "runner_sha256", "protocol_sha256", "plan_sha256",
        "numerical_protocol_sha256",
        "CPU_selfcheck_sha256", "gpu_smoke_allowed", "full_extract_allowed",
    ), "GPU runner review")
    if not (runner_review["status"] == "PASS"
            and runner_review["runner_sha256"] == EXPECTED_RUNNER_SHA
            and runner_review["protocol_sha256"] == EXPECTED_PROTOCOL_SHA
            and runner_review["plan_sha256"] == EXPECTED_PLAN_SHA
            and runner_review["numerical_protocol_sha256"] ==
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            and runner_review["CPU_selfcheck_sha256"] == EXPECTED_GPU_CPU_SELFCHECK_SHA
            and runner_review["gpu_smoke_allowed"] is True
            and runner_review["full_extract_allowed"] is True):
        raise FinalizerFailure("GPU runner review binding/authorization failed")

    smoke = read_json(GPU_SMOKE_PATH)
    require_keys(smoke, (
        "status", "protocol_sha256", "plan_sha256",
        "numerical_protocol_sha256", "runtime_signature",
        "runtime_signature_sha256",
        "CPU_selfcheck_sha256", "runner_review_sha256",
        "fixed_eight_claims", "all_eight_fixed_cache_vs_replay_smoke_gates_passed",
        "formal_claim_records_written", "GPU_used",
        "gold_bearing_source_opened", "calibration_or_test_opened",
        "scoring_run"), "GPU smoke")
    if (smoke["status"] != "passed_not_full_extraction"
            or smoke["protocol_sha256"] != EXPECTED_PROTOCOL_SHA
            or smoke["plan_sha256"] != EXPECTED_PLAN_SHA
            or smoke["numerical_protocol_sha256"] !=
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            or smoke["CPU_selfcheck_sha256"] != EXPECTED_GPU_CPU_SELFCHECK_SHA
            or smoke["runner_review_sha256"] != sha(GPU_RUNNER_REVIEW_PATH)
            or len(smoke["fixed_eight_claims"]) != 8
            or smoke["all_eight_fixed_cache_vs_replay_smoke_gates_passed"]
            is not True
            or int(smoke["formal_claim_records_written"]) != 0
            or smoke["GPU_used"] is not True):
        raise FinalizerFailure("GPU smoke receipt failed")
    require_false(smoke, ("gold_bearing_source_opened",
                          "calibration_or_test_opened", "scoring_run"),
                  "GPU smoke")

    smoke_review = read_json(GPU_SMOKE_REVIEW_PATH)
    require_keys(smoke_review, (
        "status", "runner_sha256", "numerical_protocol_sha256",
        "GPU_smoke_sha256",
        "full_extract_allowed"), "GPU smoke review")
    if not (smoke_review["status"] == "PASS"
            and smoke_review["runner_sha256"] == EXPECTED_RUNNER_SHA
            and smoke_review["numerical_protocol_sha256"] ==
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            and smoke_review["GPU_smoke_sha256"] == sha(GPU_SMOKE_PATH)
            and smoke_review["full_extract_allowed"] is True):
        raise FinalizerFailure("GPU smoke review binding/authorization failed")

    extraction = read_json(extraction_path)
    require_keys(extraction, (
        "status", "protocol_sha256", "plan_sha256",
        "numerical_protocol_sha256", "runtime_signature",
        "runtime_signature_sha256",
        "GPU_smoke_sha256", "GPU_smoke_review_sha256", "records",
        "answers", "record_order_digest", "record_commits",
        "full_batch_cache_replay_QA", "GPU_used",
        "gold_bearing_source_opened", "calibration_or_test_opened",
        "scoring_run"), "GPU extraction completion")
    reject_taint(extraction)
    if (extraction["status"] !=
            "complete_label_free_GPU_features_frozen_not_scored"
            or extraction["protocol_sha256"] != EXPECTED_PROTOCOL_SHA
            or extraction["plan_sha256"] != EXPECTED_PLAN_SHA
            or extraction["numerical_protocol_sha256"] !=
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            or int(extraction["records"]) != EXPECTED_ROWS
            or int(extraction["answers"]) != EXPECTED_ANSWERS
            or extraction["GPU_smoke_sha256"] != sha(GPU_SMOKE_PATH)
            or extraction["GPU_smoke_review_sha256"] != sha(GPU_SMOKE_REVIEW_PATH)
            or extraction["GPU_used"] is not True):
        raise FinalizerFailure("GPU extraction completion receipt failed")
    require_false(extraction, ("gold_bearing_source_opened",
                               "calibration_or_test_opened", "scoring_run"),
                  "GPU extraction completion")

    full_batch_qa = extraction["full_batch_cache_replay_QA"]
    require_keys(full_batch_qa, (
        "policy", "claims", "generated_tokens", "argmax_mismatch_count",
        "argmax_mismatch_rate", "descriptive_high_confidence_conflict_count",
        "drift_absolute_float64",
        "smoke_tolerances_not_enforced_on_full_extraction",
        "used_as_model_feature", "used_for_filtering_or_retry",
    ), "full-batch cache/replay QA", exact=True)
    if (full_batch_qa["policy"] !=
            "full_extraction_record_only_no_filter_no_retry"
            or int(full_batch_qa["claims"]) != EXPECTED_ROWS
            or int(full_batch_qa["generated_tokens"]) <= 0
            or full_batch_qa["smoke_tolerances_not_enforced_on_full_extraction"]
            is not True
            or full_batch_qa["used_as_model_feature"] is not False
            or full_batch_qa["used_for_filtering_or_retry"] is not False):
        raise FinalizerFailure("Full-batch V2 numerical QA policy changed")
    if set(full_batch_qa["drift_absolute_float64"]) != {
            "selected_logprob", "vocab_entropy", "signed_margin"}:
        raise FinalizerFailure("Full-batch V2 numerical QA signals changed")

    runtime = extraction["runtime_signature"]
    if digest(runtime) != extraction["runtime_signature_sha256"]:
        raise FinalizerFailure("Extraction runtime signature hash mismatch")
    if (runtime != smoke["runtime_signature"]
            or extraction["runtime_signature_sha256"] !=
            smoke["runtime_signature_sha256"]):
        raise FinalizerFailure("Smoke/extraction runtime signatures differ")
    runtime_locks = {
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "CPU_helper_sha256": EXPECTED_CPU_HELPER_SHA,
        "feature_qa_sha256": EXPECTED_FEATURE_QA_SHA,
        "BM25_source_sha256": EXPECTED_BM25_SHA,
        "audited_WDDM_gate_sha256": EXPECTED_WDDM_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "plan_sha256": EXPECTED_PLAN_SHA,
        "numerical_protocol_sha256": EXPECTED_NUMERICAL_PROTOCOL_SHA,
        "sanitized_manifest_sha256": EXPECTED_CPU_MANIFEST_SHA,
        "sole_input_sha256": EXPECTED_INPUT_SHA,
        "sole_input_manifest_binding": EXPECTED_INPUT_SHA,
    }
    for key, expected in runtime_locks.items():
        if runtime.get(key) != expected:
            raise FinalizerFailure(f"Runtime signature lock failed: {key}")
    if runtime.get("feature_widths") != {
            "P1": 21, "P2": 549, "P3": 549,
            "components": [21, 11, 5, 256, 256]}:
        raise FinalizerFailure("Runtime feature widths changed")
    if digest(extraction["record_commits"]) != extraction["record_order_digest"]:
        raise FinalizerFailure("Extraction record order digest mismatch")
    return extraction, {
        "GPU_runner_review": sha(GPU_RUNNER_REVIEW_PATH),
        "GPU_smoke": sha(GPU_SMOKE_PATH),
        "GPU_smoke_review": sha(GPU_SMOKE_REVIEW_PATH),
        "GPU_extraction_complete": sha(extraction_path),
    }


def record_paths(index: int, directory: Path):
    stem = f"{index:07d}"
    return (directory / f"{stem}.npz", directory / f"{stem}.json",
            directory / f"{stem}.commit.json")


def validate_feature_matrices(matrices: dict[str, np.ndarray], rows: int,
                              dims=FEATURE_DIMS) -> None:
    if tuple(matrices) != FEATURE_KEYS:
        raise FinalizerFailure("Feature matrix order must be P1/P2/P3")
    for name in FEATURE_KEYS:
        value = matrices[name]
        if value.shape != (rows, dims[name]) or value.dtype != np.float32:
            raise FinalizerFailure(
                f"{name} shape/dtype mismatch: {value.shape}/{value.dtype}")
        if not np.isfinite(value).all():
            raise FinalizerFailure(f"{name} contains non-finite values")
    if not np.array_equal(matrices["P1"], matrices["P2"][:, :21]):
        raise FinalizerFailure("P1 is not exactly P2's first 21 columns")


def recompute_v2_record_qa(arrays: dict[str, np.ndarray]) -> dict:
    """Rebuild the record-only V2 numerical summary from saved arrays."""
    return gpu_runner.cache_replay_qa(
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


def collect_records(rows: list[dict], extraction: dict, directory: Path,
                    runtime_sha256: str) -> dict[str, np.ndarray]:
    """Validate every committed record in clean-row order and stack features."""
    directory = Path(directory)
    declared = extraction.get("record_commits")
    if not isinstance(declared, list) or len(declared) != len(rows):
        raise FinalizerFailure("Extraction record list length mismatch")
    expected_files = set()
    collected = {name: [] for name in FEATURE_KEYS}
    commit_keys = {
        "record_index", "response_id", "microclaim_id", "row_sha256",
        "record_signature", "npz_sha256", "metadata_sha256",
    }
    summary_keys = {
        "record_index", "response_id", "microclaim_id", "commit_sha256",
        "npz_sha256", "metadata_sha256",
    }
    for index, row in enumerate(rows):
        npz_path, metadata_path, commit_path = record_paths(index, directory)
        expected_files.update((npz_path.name, metadata_path.name,
                               commit_path.name))
        for path in (npz_path, metadata_path, commit_path):
            if not path.is_file():
                raise FinalizerFailure(f"Missing committed record artifact: {path}")
        commit = read_json(commit_path)
        require_keys(commit, commit_keys, f"record {index} commit", exact=True)
        row_sha = digest(row)
        signature = digest({
            "runtime_signature_sha256": runtime_sha256,
            "record_index": index, "row": row,
        })
        expected_commit = {
            "record_index": index,
            "response_id": row["response_id"],
            "microclaim_id": row["microclaim_id"],
            "row_sha256": row_sha,
            "record_signature": signature,
            "npz_sha256": sha(npz_path),
            "metadata_sha256": sha(metadata_path),
        }
        if commit != expected_commit:
            raise FinalizerFailure(f"Record {index} commit binding mismatch")
        summary = declared[index]
        require_keys(summary, summary_keys, f"record {index} summary", exact=True)
        expected_summary = {
            "record_index": index,
            "response_id": row["response_id"],
            "microclaim_id": row["microclaim_id"],
            "commit_sha256": sha(commit_path),
            "npz_sha256": commit["npz_sha256"],
            "metadata_sha256": commit["metadata_sha256"],
        }
        if summary != expected_summary:
            raise FinalizerFailure(f"Record {index} extraction summary mismatch")

        metadata = read_json(metadata_path)
        reject_taint(metadata)
        require_keys(metadata, (
            "complete", "record_index", "response_id", "source_id",
            "group_id", "microclaim_id", "microclaim_index", "row_sha256",
            "record_signature", "npz_sha256", "gold_fields_available",
            "calibration_or_test_opened", "scoring_run", "extraction",
        ), f"record {index} metadata")
        if not (metadata["complete"] is True
                and metadata["record_index"] == index
                and metadata["response_id"] == row["response_id"]
                and metadata["source_id"] == row["source_id"]
                and metadata["group_id"] == row["group_id"]
                and metadata["microclaim_id"] == row["microclaim_id"]
                and metadata["microclaim_index"] == row["microclaim_index"]
                and metadata["row_sha256"] == row_sha
                and metadata["record_signature"] == signature
                and metadata["npz_sha256"] == commit["npz_sha256"]):
            raise FinalizerFailure(f"Record {index} metadata binding mismatch")
        require_false(metadata, ("gold_fields_available",
                                 "calibration_or_test_opened", "scoring_run"),
                      f"record {index} metadata")

        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        try:
            gpu_runner.validate_arrays(arrays)
        except (AssertionError, KeyError, ValueError) as error:
            raise FinalizerFailure(f"Record {index} runner array invalid") from error
        extraction_meta = metadata["extraction"]
        stored_qa = extraction_meta.get("P3", {}).get("cache_replay_QA")
        if (extraction_meta.get("cache_replay_QA_policy") !=
                "full_extraction_record_only"
                or not isinstance(stored_qa, dict)
                or stored_qa.get("policy") != "full_extraction_record_only"
                or digest(stored_qa) != digest(recompute_v2_record_qa(arrays))):
            raise FinalizerFailure(
                f"Record {index} V2 cache/replay QA binding mismatch")
        surfaces = {
            "P1": arrays["p1_features"],
            "P2": arrays["p2_features"],
            "P3": arrays["p3_features"],
        }
        for name, value in surfaces.items():
            if value.dtype != np.float32 or value.shape != (FEATURE_DIMS[name],):
                raise FinalizerFailure(f"Record {index} {name} invalid")
            collected[name].append(value.copy())

    actual_files = {path.name for path in directory.iterdir()}
    if actual_files != expected_files:
        raise FinalizerFailure(
            f"Record directory has missing/extra files: "
            f"missing={sorted(expected_files-actual_files)[:5]}, "
            f"extra={sorted(actual_files-expected_files)[:5]}")
    matrices = {name: np.stack(collected[name]).astype(np.float32, copy=False)
                for name in FEATURE_KEYS}
    validate_feature_matrices(matrices, len(rows))
    return matrices


def frozen_v2_runner_result_locks(lineage: dict) -> dict[str, str]:
    value = lineage.get("V2_runner_results")
    require_keys(value, (
        "GPU_runner_sha256", "numerical_protocol_sha256",
        "GPU_extraction_complete_sha256", "full_batch_cache_replay_QA",
        "full_batch_cache_replay_QA_sha256",
    ), "V2 runner-result lineage", exact=True)
    if (value["GPU_runner_sha256"] != EXPECTED_RUNNER_SHA
            or value["numerical_protocol_sha256"] !=
            EXPECTED_NUMERICAL_PROTOCOL_SHA
            or digest(value["full_batch_cache_replay_QA"]) !=
            value["full_batch_cache_replay_QA_sha256"]):
        raise FinalizerFailure("V2 runner-result lineage binding mismatch")
    for key in ("GPU_extraction_complete_sha256",
                "full_batch_cache_replay_QA_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64:
            raise FinalizerFailure(f"V2 runner-result {key} is not SHA-256")
    return {key: value[key] for key in (
        "GPU_runner_sha256", "numerical_protocol_sha256",
        "GPU_extraction_complete_sha256",
        "full_batch_cache_replay_QA_sha256")}


def feature_manifest(rows: list[dict], arrays_sha: str,
                     runner_result_locks: dict[str, str],
                     arrays_name: str = FEATURE_ARRAY_NAME) -> dict:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "status": "complete_label_free_features",
        "rows": len(rows),
        "row_keys_sha256": row_keys_digest(rows),
        "label_free_inputs_sha256": EXPECTED_INPUT_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "plan_sha256": EXPECTED_PLAN_SHA,
        **runner_result_locks,
        "arrays": {
            "path": arrays_name, "sha256": arrays_sha,
            "keys": list(FEATURE_KEYS),
            "dtypes": {name: "float32" for name in FEATURE_KEYS},
            "shapes": {name: [len(rows), FEATURE_DIMS[name]]
                       for name in FEATURE_KEYS},
        },
        "labels_used": False, "gold_read": False,
        "calibration_read": False, "official_test_read": False,
    }


def feature_freeze(rows: list[dict], manifest_sha: str,
                   runner_result_locks: dict[str, str]) -> dict:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "status": "complete_frozen_before_gold",
        "feature_manifest": FEATURE_MANIFEST_NAME,
        "feature_manifest_sha256": manifest_sha,
        "label_free_inputs_sha256": EXPECTED_INPUT_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "plan_sha256": EXPECTED_PLAN_SHA,
        **runner_result_locks,
        "rows": len(rows),
        "labels_used": False, "gold_read": False,
        "calibration_read": False, "official_test_read": False,
    }


def verify_bundle(directory: Path, rows: list[dict], *,
                  expected_lineage: dict | None = None) -> dict[str, str]:
    directory = Path(directory)
    paths = {
        "arrays": directory / FEATURE_ARRAY_NAME,
        "manifest": directory / FEATURE_MANIFEST_NAME,
        "freeze": directory / FEATURE_FREEZE_NAME,
        "lineage": directory / LINEAGE_NAME,
        "complete": directory / COMPLETE_NAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FinalizerFailure(f"Frozen bundle file absent: {path}")
    lineage = read_json(paths["lineage"])
    runner_result_locks = frozen_v2_runner_result_locks(lineage)
    manifest = read_json(paths["manifest"])
    freeze = read_json(paths["freeze"])
    expected_manifest_keys = {
        "schema_version", "status", "rows", "row_keys_sha256",
        "label_free_inputs_sha256", "protocol_sha256", "plan_sha256",
        "numerical_protocol_sha256", "GPU_runner_sha256",
        "GPU_extraction_complete_sha256", "full_batch_cache_replay_QA_sha256",
        "arrays", "labels_used", "gold_read", "calibration_read",
        "official_test_read",
    }
    expected_freeze_keys = {
        "schema_version", "status", "feature_manifest",
        "feature_manifest_sha256", "label_free_inputs_sha256",
        "protocol_sha256", "plan_sha256", "numerical_protocol_sha256",
        "GPU_runner_sha256", "GPU_extraction_complete_sha256",
        "full_batch_cache_replay_QA_sha256", "rows", "labels_used",
        "gold_read", "calibration_read", "official_test_read",
    }
    require_keys(manifest, expected_manifest_keys, "feature manifest", exact=True)
    require_keys(freeze, expected_freeze_keys, "feature freeze", exact=True)
    if manifest != feature_manifest(
            rows, sha(paths["arrays"]), runner_result_locks):
        raise FinalizerFailure("Feature manifest content mismatch")
    if freeze != feature_freeze(
            rows, sha(paths["manifest"]), runner_result_locks):
        raise FinalizerFailure("Feature freeze content mismatch")
    require_false(manifest, ("labels_used", "gold_read", "calibration_read",
                             "official_test_read"), "feature manifest")
    require_false(freeze, ("labels_used", "gold_read", "calibration_read",
                           "official_test_read"), "feature freeze")
    with np.load(paths["arrays"], allow_pickle=False) as archive:
        if archive.files != list(FEATURE_KEYS):
            raise FinalizerFailure("Frozen NPZ key order mismatch")
        matrices = {name: archive[name].copy() for name in archive.files}
    validate_feature_matrices(matrices, len(rows))
    complete = read_json(paths["complete"])
    reject_taint(lineage); reject_taint(complete)
    if expected_lineage is not None and lineage != expected_lineage:
        raise FinalizerFailure("Finalizer lineage content mismatch")
    require_keys(complete, (
        "schema_version", "status", "rows", "feature_array_sha256",
        "feature_manifest_sha256", "feature_freeze_sha256",
        "finalizer_lineage_sha256", "GPU_extraction_complete_sha256",
        "finalizer_source_sha256", "input_builder_sha256", "labels_used",
        "gold_read", "calibration_read", "official_test_read",
    ), "finalizer completion", exact=True)
    expected_complete = {
        "schema_version": VERSION,
        "status": "complete_atomic_label_free_feature_bundle",
        "rows": len(rows),
        "feature_array_sha256": sha(paths["arrays"]),
        "feature_manifest_sha256": sha(paths["manifest"]),
        "feature_freeze_sha256": sha(paths["freeze"]),
        "finalizer_lineage_sha256": sha(paths["lineage"]),
        "GPU_extraction_complete_sha256": lineage[
            "execution_receipts"]["GPU_extraction_complete"],
        "finalizer_source_sha256": lineage["code_locks"]["finalizer"],
        "input_builder_sha256": lineage["code_locks"]["input_builder"],
        "labels_used": False, "gold_read": False,
        "calibration_read": False, "official_test_read": False,
    }
    if complete != expected_complete:
        raise FinalizerFailure("Finalizer completion content mismatch")
    return {name: sha(path) for name, path in paths.items()}


def write_bundle(target: Path, rows: list[dict], matrices: dict[str, np.ndarray],
                 lineage: dict, *, evaluator_compat: bool = False
                 ) -> dict[str, str]:
    """Create the whole bundle in a sibling staging directory, then rename."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Refuse overwrite: {target}")
    validate_feature_matrices(matrices, len(rows))
    staging = target.with_name(
        f"{target.name}.pending.{os.getpid()}.{time.time_ns()}")
    if staging.exists():
        raise FileExistsError(f"Unexpected staging collision: {staging}")
    staging.mkdir(parents=True)
    try:
        array_path = staging / FEATURE_ARRAY_NAME
        manifest_path = staging / FEATURE_MANIFEST_NAME
        freeze_path = staging / FEATURE_FREEZE_NAME
        lineage_path = staging / LINEAGE_NAME
        complete_path = staging / COMPLETE_NAME
        runner_result_locks = frozen_v2_runner_result_locks(lineage)
        atomic_npz(array_path, matrices)
        atomic_json(manifest_path, feature_manifest(
            rows, sha(array_path), runner_result_locks))
        atomic_json(lineage_path, lineage)
        freeze_value = feature_freeze(
            rows, sha(manifest_path), runner_result_locks)
        complete_value = {
            "schema_version": VERSION,
            "status": "complete_atomic_label_free_feature_bundle",
            "rows": len(rows),
            "feature_array_sha256": sha(array_path),
            "feature_manifest_sha256": sha(manifest_path),
            "feature_freeze_sha256": bytes_sha(json_bytes(freeze_value)),
            "finalizer_lineage_sha256": sha(lineage_path),
            "GPU_extraction_complete_sha256": lineage[
                "execution_receipts"]["GPU_extraction_complete"],
            "finalizer_source_sha256": lineage["code_locks"]["finalizer"],
            "input_builder_sha256": lineage["code_locks"]["input_builder"],
            "labels_used": False, "gold_read": False,
            "calibration_read": False, "official_test_read": False,
        }
        atomic_json(complete_path, complete_value)
        atomic_json(freeze_path, freeze_value)  # evaluator gate is written last
        verified = verify_bundle(staging, rows, expected_lineage=lineage)
        if evaluator_compat:
            if evaluator.REAL_GOLD_OPEN_COUNT != 0:
                raise FinalizerFailure(
                    "Evaluator gold-open counter was nonzero before check")
            receipt, evaluator_rows = evaluator.load_frozen_features(staging)
            if evaluator.REAL_GOLD_OPEN_COUNT != 0:
                raise FinalizerFailure(
                    "Evaluator opened gold during feature validation")
            if len(evaluator_rows) != len(rows):
                raise FinalizerFailure(
                    "Evaluator feature contract returned wrong row count")
            if receipt.array_sha256 != verified["arrays"]:
                raise FinalizerFailure("Evaluator receipt array hash differs")
        staging.replace(target)
        return verified
    except BaseException:
        # Preserve failed staging for audit; never make a partial freeze visible.
        raise


def build_lineage(static: dict[str, str], execution: dict[str, str],
                  extraction: dict, rows: list[dict]) -> dict:
    return {
        "schema_version": VERSION,
        "status": "lineage_locked_before_gold",
        "rows": len(rows),
        "row_keys_sha256": row_keys_digest(rows),
        "runtime_signature_sha256": extraction["runtime_signature_sha256"],
        "record_order_digest": extraction["record_order_digest"],
        "V2_runner_results": {
            "GPU_runner_sha256": static["GPU_runner"],
            "numerical_protocol_sha256": static["numerical_protocol"],
            "GPU_extraction_complete_sha256": execution[
                "GPU_extraction_complete"],
            "full_batch_cache_replay_QA": extraction[
                "full_batch_cache_replay_QA"],
            "full_batch_cache_replay_QA_sha256": digest(
                extraction["full_batch_cache_replay_QA"]),
        },
        "code_locks": {
            "finalizer": sha(Path(__file__)),
            "GPU_runner": static["GPU_runner"],
            "CPU_helper": static["CPU_helper"],
            "input_builder": static["input_builder"],
            "feature_qa": static["feature_qa"],
            "BM25_source": static["BM25_source"],
            "audited_WDDM_gate": static["audited_WDDM_gate"],
            "evaluator_current_contract": sha(Path(evaluator.__file__)),
        },
        "CPU_package_locks": {
            "GPU_runner_CPU_selfcheck": static["GPU_runner_CPU_selfcheck"],
            "CPU_manifest": static["CPU_manifest"],
            "PREPARATION": static["PREPARATION"],
            "CPU_independent_review": static["CPU_independent_review"],
            "evaluator_CPU_selftest": static["evaluator_CPU_selftest"],
            "evaluator_independent_review": static[
                "evaluator_independent_review"],
            "finalizer_CPU_selftest": static["finalizer_CPU_selftest"],
            "finalizer_independent_review": static[
                "finalizer_independent_review"],
            "protocol": static["protocol"], "plan": static["plan"],
            "numerical_protocol": static["numerical_protocol"],
            "label_free_inputs": static["label_free_inputs"],
        },
        "execution_receipts": execution,
        "labels_used": False, "gold_read": False,
        "calibration_read": False, "official_test_read": False,
        "GPU_used_by_finalizer": False,
    }


def finalize() -> dict:
    if torch.cuda.is_initialized():
        raise FinalizerFailure("CUDA was already initialized; use a clean CPU process")
    if FEATURE_DIR.exists():
        raise FinalizerFailure("Frozen feature directory already exists")
    static = validate_static_locks()
    rows = load_clean_rows()
    extraction, execution = validate_execution_chain()
    matrices = collect_records(
        rows, extraction, RECORD_DIR, extraction["runtime_signature_sha256"])
    static_after = validate_static_locks()
    extraction_after, execution_after = validate_execution_chain()
    if (static_after != static or extraction_after != extraction
            or execution_after != execution):
        raise FinalizerFailure("Source or execution receipt changed during finalization")
    lineage = build_lineage(static, execution, extraction, rows)
    hashes = write_bundle(
        FEATURE_DIR, rows, matrices, lineage, evaluator_compat=True)
    if torch.cuda.is_initialized():
        raise FinalizerFailure("CPU finalizer initialized CUDA")
    return hashes


def expect_failure(callable_value, text: str) -> str:
    try:
        callable_value()
    except (FinalizerFailure, FileExistsError, AssertionError, ValueError,
            KeyError) as error:
        return f"{type(error).__name__}: {error}"
    raise AssertionError(f"Expected failure was accepted: {text}")


def synthetic_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        row = {
            "response_id": f"synthetic-response-{index:02d}",
            "source_id": f"synthetic-source-{index:02d}",
            "group_id": f"synthetic-group-{index:02d}",
            "model": "llama-2-7b-chat",
            "question": "Synthetic question?",
            "passage_1": "Evidence one.",
            "passage_2": "Evidence two.",
            "passage_3": "Evidence three.",
            "microclaim_id": f"synthetic-response-{index:02d}__atomic_000",
            "microclaim_index": 0,
            "claim_start": 0,
            "claim_end": 16,
            "claim_text_raw": "Synthetic claim.",
            "claim_prompt_text": "Synthetic claim.",
        }
        if tuple(row) != OUTPUT_FIELDS:
            raise AssertionError("Synthetic row field order drift")
        rows.append(row)
    return rows


def materialize_synthetic_records(directory: Path, rows: list[dict],
                                  runtime_sha256: str) -> dict:
    directory.mkdir(parents=True)
    summaries = []
    for index, row in enumerate(rows):
        arrays = gpu_runner.synthetic_arrays()
        record_qa = recompute_v2_record_qa(arrays)
        npz_path, metadata_path, commit_path = record_paths(index, directory)
        atomic_npz(npz_path, arrays)
        row_sha = digest(row)
        signature = digest({
            "runtime_signature_sha256": runtime_sha256,
            "record_index": index, "row": row,
        })
        metadata = {
            "complete": True, "record_index": index,
            "response_id": row["response_id"],
            "source_id": row["source_id"], "group_id": row["group_id"],
            "microclaim_id": row["microclaim_id"],
            "microclaim_index": row["microclaim_index"],
            "row_sha256": row_sha, "record_signature": signature,
            "npz_sha256": sha(npz_path),
            "extraction": {
                "synthetic_CPU_only": True,
                "cache_replay_QA_policy": "full_extraction_record_only",
                "P3": {"cache_replay_QA": record_qa},
            },
            "gold_fields_available": False,
            "calibration_or_test_opened": False, "scoring_run": False,
        }
        atomic_json(metadata_path, metadata)
        commit = {
            "record_index": index, "response_id": row["response_id"],
            "microclaim_id": row["microclaim_id"],
            "row_sha256": row_sha, "record_signature": signature,
            "npz_sha256": sha(npz_path),
            "metadata_sha256": sha(metadata_path),
        }
        atomic_json(commit_path, commit)
        summaries.append({
            "record_index": index, "response_id": row["response_id"],
            "microclaim_id": row["microclaim_id"],
            "commit_sha256": sha(commit_path),
            "npz_sha256": commit["npz_sha256"],
            "metadata_sha256": commit["metadata_sha256"],
        })
    return {
        "runtime_signature_sha256": runtime_sha256,
        "record_commits": summaries,
        "record_order_digest": digest(summaries),
    }


def archive_old_selftest() -> None:
    if not SELFTEST_PATH.exists():
        return
    archive = OUT / "superseded_CPU_feature_finalizer_selfchecks"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / (
        f"CPU_FEATURE_FINALIZER_SELFTEST_{time.time_ns()}_"
        f"{sha(SELFTEST_PATH)[:16]}.json")
    SELFTEST_PATH.replace(target)


def selftest() -> dict:
    if torch.cuda.is_initialized():
        raise FinalizerFailure("CUDA initialized before CPU synthetic selftest")
    runner_before = sha(GPU_RUNNER_PATH)
    checks = {}
    with tempfile.TemporaryDirectory(
            prefix="forced_quote_feature_finalizer_CPU_") as temp_name:
        temp = Path(temp_name)
        rows = synthetic_rows(5)
        runtime_sha256 = digest({"synthetic_runtime": VERSION})
        record_dir = temp / "records"
        extraction = materialize_synthetic_records(
            record_dir, rows, runtime_sha256)
        matrices = collect_records(
            rows, extraction, record_dir, runtime_sha256)
        checks["valid_commit_hash_row_order_collection"] = True
        checks["V2_record_only_cache_replay_QA_recomputed"] = True
        validate_feature_matrices(matrices, len(rows))
        checks["float32_shape_finite_P1_P2_prefix"] = True

        bad_summary = json.loads(json.dumps(extraction))
        bad_summary["record_commits"][0]["commit_sha256"] = "0" * 64
        checks["tampered_declared_commit_hash_rejected"] = bool(
            expect_failure(lambda: collect_records(
                rows, bad_summary, record_dir, runtime_sha256), "tampered hash"))
        swapped = rows.copy(); swapped[0], swapped[1] = swapped[1], swapped[0]
        checks["swapped_clean_row_order_rejected"] = bool(
            expect_failure(lambda: collect_records(
                swapped, extraction, record_dir, runtime_sha256), "row order"))

        orphan = record_dir / "orphan.pending"
        orphan.write_text("synthetic", encoding="utf-8")
        checks["orphan_record_artifact_rejected"] = bool(
            expect_failure(lambda: collect_records(
                rows, extraction, record_dir, runtime_sha256), "orphan"))
        orphan.unlink()

        nonfinite = {name: value.copy() for name, value in matrices.items()}
        nonfinite["P3"][0, 0] = np.nan
        checks["nonfinite_rejected"] = bool(expect_failure(
            lambda: validate_feature_matrices(nonfinite, len(rows)),
            "nonfinite"))
        mismatch = {name: value.copy() for name, value in matrices.items()}
        mismatch["P2"][0, 0] = 1.0
        checks["P1_P2_prefix_mismatch_rejected"] = bool(expect_failure(
            lambda: validate_feature_matrices(mismatch, len(rows)),
            "P1/P2 mismatch"))

        synthetic_batch_qa = {
            "policy": "synthetic_CPU_only_record_only_QA",
            "claims": len(rows),
        }
        lineage = {
            "schema_version": VERSION,
            "status": "synthetic_lineage_CPU_only",
            "V2_runner_results": {
                "GPU_runner_sha256": EXPECTED_RUNNER_SHA,
                "numerical_protocol_sha256":
                    EXPECTED_NUMERICAL_PROTOCOL_SHA,
                "GPU_extraction_complete_sha256": "1" * 64,
                "full_batch_cache_replay_QA": synthetic_batch_qa,
                "full_batch_cache_replay_QA_sha256": digest(
                    synthetic_batch_qa),
            },
            "code_locks": {
                "finalizer": sha(Path(__file__)),
                "input_builder": EXPECTED_INPUT_BUILDER_SHA,
            },
            "execution_receipts": {
                "GPU_extraction_complete": "1" * 64,
            },
            "labels_used": False, "gold_read": False,
            "calibration_read": False, "official_test_read": False,
            "GPU_used": False,
        }
        bundle = temp / "frozen_features"
        hashes = write_bundle(bundle, rows, matrices, lineage)
        verified = verify_bundle(bundle, rows, expected_lineage=lineage)
        if hashes != verified:
            raise AssertionError("Synthetic bundle verification hash drift")
        checks["finalizer_schema_bundle_verified"] = True
        manifest_value = read_json(bundle / FEATURE_MANIFEST_NAME)
        freeze_value = read_json(bundle / FEATURE_FREEZE_NAME)
        lineage_value = read_json(bundle / LINEAGE_NAME)
        for value in (manifest_value, freeze_value,
                      lineage_value["V2_runner_results"]):
            if (value["GPU_runner_sha256"] != EXPECTED_RUNNER_SHA
                    or value["numerical_protocol_sha256"] !=
                    EXPECTED_NUMERICAL_PROTOCOL_SHA):
                raise AssertionError("V2 runner/numerical freeze binding drift")
        checks["V2_runner_results_and_numerical_protocol_in_lineage_freeze"] = True
        checks["atomic_refuse_overwrite"] = bool(expect_failure(
            lambda: write_bundle(bundle, rows, matrices, lineage),
            "bundle overwrite"))
        checks["freeze_written_with_complete_hash_chain"] = True

        # A separate full-size label-free bundle must pass the evaluator's
        # actual frozen-feature loader.  The five-row unit fixture above cannot
        # exercise that contract because it is intentionally fixed at 3,776.
        clean_rows = load_clean_rows()
        contract_matrices = {
            name: np.zeros((len(clean_rows), FEATURE_DIMS[name]),
                           dtype=np.float32)
            for name in FEATURE_KEYS
        }
        contract_lineage = {
            **lineage,
            "status": "synthetic_evaluator_contract_CPU_only",
            "rows": len(clean_rows),
            "row_keys_sha256": row_keys_digest(clean_rows),
        }
        contract_bundle = temp / "evaluator_contract_frozen_features"
        write_bundle(contract_bundle, clean_rows, contract_matrices,
                     contract_lineage, evaluator_compat=True)
        if evaluator.REAL_GOLD_OPEN_COUNT != 0:
            raise AssertionError("Evaluator opened gold in compatibility selftest")
        checks["actual_evaluator_load_frozen_features_3776_rows"] = True

    if not all(checks.values()):
        raise AssertionError(f"Synthetic check failed: {checks}")
    if sha(GPU_RUNNER_PATH) != runner_before:
        raise AssertionError("GPU runner changed during finalizer selftest")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU synthetic selftest initialized CUDA")
    result = {
        "schema_version": VERSION,
        "status": "passed_synthetic_CPU_only_formal_finalization_not_run",
        "finalizer_sha256": sha(Path(__file__)),
        "frozen_GPU_runner_sha256": runner_before,
        "numerical_protocol_sha256": sha(NUMERICAL_PROTOCOL_PATH),
        "frozen_evaluator_sha256": sha(Path(evaluator.__file__)),
        "evaluator_selftest_sha256": sha(EVALUATOR_SELFTEST_PATH),
        "formal_runner_lock_currently_matches": (
            runner_before == EXPECTED_RUNNER_SHA),
        "synthetic_rows": 5,
        "checks": checks,
        "active_GPU_extraction_complete_present": EXTRACTION_PATH.exists(),
        "active_frozen_features_present": FEATURE_DIR.exists(),
        "formal_finalization_run": False,
        "formal_evaluator_independent_review_present":
            EVALUATOR_REVIEW_PATH.is_file(),
        "formal_finalizer_independent_review_present":
            FINALIZER_REVIEW_PATH.is_file(),
        "GPU_used": False, "CUDA_initialized": False,
        "pretrained_model_loaded": False,
        "gold_read": False, "calibration_read": False,
        "official_test_read": False, "scoring_run": False,
    }
    archive_old_selftest()
    atomic_json(SELFTEST_PATH, result)
    return result


def verify_formal() -> dict:
    if torch.cuda.is_initialized():
        raise FinalizerFailure("CUDA was already initialized")
    static = validate_static_locks()
    rows = load_clean_rows()
    extraction, execution = validate_execution_chain()
    lineage = build_lineage(static, execution, extraction, rows)
    hashes = verify_bundle(FEATURE_DIR, rows, expected_lineage=lineage)
    if evaluator.REAL_GOLD_OPEN_COUNT != 0:
        raise FinalizerFailure("Evaluator gold-open counter was nonzero")
    evaluator.load_frozen_features(FEATURE_DIR)
    if evaluator.REAL_GOLD_OPEN_COUNT != 0:
        raise FinalizerFailure("Evaluator opened gold during verification")
    if torch.cuda.is_initialized():
        raise FinalizerFailure("CPU verification initialized CUDA")
    return hashes


def describe() -> dict:
    return {
        "version": VERSION,
        "finalizer": str(Path(__file__).resolve()),
        "finalizer_sha256": sha(Path(__file__)),
        "frozen_GPU_runner_sha256": sha(GPU_RUNNER_PATH),
        "commands": ["selftest", "finalize", "verify", "describe"],
        "automatic_stage_chaining": False,
        "formal_gate_GPU_extraction_complete_present": EXTRACTION_PATH.exists(),
        "formal_output_present": FEATURE_DIR.exists(),
        "sole_feature_source": str(RECORD_DIR.resolve()),
        "sole_identity_source": str(INPUT_PATH.resolve()),
        "gold_or_calibration_or_test_read": False,
        "GPU_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "selftest", "finalize", "verify", "describe"))
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.command == "selftest":
            value = selftest()
        elif args.command == "finalize":
            value = finalize()
        elif args.command == "verify":
            value = verify_formal()
        else:
            value = describe()
        print(json.dumps({
            "command": args.command, "result": value,
            "seconds": time.perf_counter() - started,
        }, ensure_ascii=False, indent=2))
    except BaseException as error:
        print(json.dumps({
            "command": args.command, "status": "FAILED",
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "formal_output_present": FEATURE_DIR.exists(),
            "GPU_initialized": torch.cuda.is_initialized(),
        }, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
