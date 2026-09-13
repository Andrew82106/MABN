"""Independent stdlib audit of formal RAGognizer scores and cal metrics.

This script deliberately does not import adapter.py or evaluate_shared.py.  It
first authenticates the formal freeze/result chain, then streams the full plan,
native scores, and shared scores in lockstep.  Calibration gold is opened only
after every label-free coverage, hash, and arithmetic check has passed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = HERE / "INFERENCE_PLAN.jsonl"
RAW = HERE / "NATIVE_PROBABILITIES.jsonl"
GPU_AUDIT = HERE / "GPU_RUN_AUDIT.json"
ADAPTED = HERE / "SHARED_PROBABILITIES.jsonl"
ADAPTER_AUDIT = HERE / "ADAPTER_AUDIT.json"
FREEZE = HERE / "SCORE_FREEZE.json"
FORMAL_RESULT = HERE / "CAL_RESULTS.json"
FORMAL_STATUS = HERE / "SCORING_STATUS.json"
PROVISIONAL_RAW = HERE / "PROVISIONAL_CAL_NATIVE.jsonl"
CAL_WINDOWS = PROJECT / "data" / "windows_k4_calibration.jsonl"
CAL_ANSWERS = PROJECT / "data" / "answers_calibration.jsonl"
OUTPUT = HERE / "SCORE_INDEPENDENT_AUDIT.json"
RUNNER = HERE / "run_inference.py"
ADAPTER_SCRIPT = HERE / "adapter.py"
FORMAL_EVALUATOR = HERE / "evaluate_shared.py"
MODEL_MANIFEST = HERE / "MODEL_MANIFEST.json"
SOURCE_AUDIT = HERE / "SOURCE_AUDIT.json"
RUNTIME_ALIAS_AUDIT = HERE / "RUNTIME_ALIAS_AUDIT.json"

PLAN_SHA256 = "90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76"
ZERO_CHAIN = "0" * 64
AUTHOR_THRESHOLD = 0.6523
THRESHOLD_RULE = (
    "separate max F1, then higher precision, then higher threshold; "
    "score >= threshold"
)
CAL_WINDOWS_SHA256 = "dd7c17add9aed79f5b6637c87c6b918a17d9ac0ff73014318b29a6c24228b510"
CAL_ANSWERS_SHA256 = "2432052cd7d7d5185d95c9b57596f11979208a08a83579b83a48751510618645"
EXPECTED_ROWS = {"fit": 3_680, "calibration": 159}
EXPECTED_WINDOWS = {"fit": 653_979, "calibration": 42_241}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_ROWS.values())
EXPECTED_TOTAL_WINDOWS = sum(EXPECTED_WINDOWS.values())
METRIC_ABS_TOLERANCE = 1e-12
HEX64 = re.compile(r"[0-9a-f]{64}\Z")

RAW_KEYS = {
    "version",
    "route",
    "partition",
    "answer_id",
    "answer_sha256",
    "plan_sha256",
    "plan_row_sha256",
    "run_identity_sha256",
    "author_response_token_ids",
    "author_response_token_char_intervals",
    "author_response_token_probabilities",
    "author_native_preds_0_6523_appendix_only",
    "author_coordinate_mode",
    "author_public_packer_exact",
    "postprocessor",
    "quantization",
    "threshold_applied",
    "raw_row_sha256",
    "previous_chain_sha256",
    "chain_sha256",
}

ADAPTED_KEYS = {
    "version",
    "partition",
    "answer_id",
    "answer_sha256",
    "plan_row_sha256",
    "mapping_mode",
    "author_coordinate_mode",
    "author_public_packer_exact",
    "author_response_token_probabilities",
    "shared_bpe_token_probabilities",
    "window_ids",
    "shared_k4_window_probabilities",
    "shared_answer_probability",
    "author_native_answer_probability",
    "author_native_threshold_appendix_only",
    "shared_thresholds_applied",
    "labels_read",
    "raw_chain_sha256",
    "adapted_row_sha256",
    "previous_chain_sha256",
    "chain_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str, where: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except Exception as exc:
        raise ValueError(f"invalid strict JSON at {where}") from exc


def load_json(path: Path) -> dict:
    value = parse_json(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON is not an object: {path}")
    return value


def stream_jsonl(path: Path, digest: "hashlib._Hash") -> Iterable[tuple[int, dict]]:
    """Yield one object at a time while hashing the exact file bytes."""
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 at {path}:{line_no}") from exc
            value = parse_json(text, f"{path}:{line_no}")
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_no}")
            yield line_no, value


def require_hash(value: object, where: str) -> str:
    digest = str(value)
    if HEX64.fullmatch(digest) is None:
        raise ValueError(f"invalid SHA-256 at {where}")
    return digest


def validate_chained_row(
    row: dict, previous_chain: str, row_hash_key: str, where: str
) -> str:
    if set((row_hash_key, "previous_chain_sha256", "chain_sha256")) - set(row):
        raise ValueError(f"missing chain field at {where}")
    stored_row_hash = require_hash(row[row_hash_key], f"{where}.{row_hash_key}")
    stored_previous = require_hash(
        row["previous_chain_sha256"], f"{where}.previous_chain_sha256"
    )
    stored_chain = require_hash(row["chain_sha256"], f"{where}.chain_sha256")
    if stored_previous != previous_chain:
        raise ValueError(f"previous chain mismatch at {where}")
    core = {
        key: value
        for key, value in row.items()
        if key not in {row_hash_key, "previous_chain_sha256", "chain_sha256"}
    }
    recomputed_row_hash = sha256_json(core)
    if stored_row_hash != recomputed_row_hash:
        raise ValueError(f"canonical row hash mismatch at {where}")
    recomputed_chain = sha256_text(previous_chain + "\n" + recomputed_row_hash)
    if stored_chain != recomputed_chain:
        raise ValueError(f"canonical chain hash mismatch at {where}")
    return stored_chain


def validate_plan_row(row: dict, where: str) -> str:
    if row.get("version") != "ragognizer-full-inference-plan-v1":
        raise ValueError(f"plan version mismatch at {where}")
    stored_hash = require_hash(row.get("plan_row_sha256"), f"{where}.plan_row_sha256")
    core = {key: value for key, value in row.items() if key != "plan_row_sha256"}
    if stored_hash != sha256_json(core):
        raise ValueError(f"canonical plan row hash mismatch at {where}")
    return stored_hash


def finite_probabilities(value: object, where: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"empty or non-list probability vector at {where}")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"non-numeric probability at {where}[{index}]")
        score = float(item)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"probability outside [0,1] at {where}[{index}]")
        result.append(score)
    return result


def _intervals(value: object, where: str) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"empty or non-list interval vector at {where}")
    result = []
    previous_start = -1
    previous_end = -1
    for index, span in enumerate(value):
        if (
            not isinstance(span, list)
            or len(span) != 2
            or isinstance(span[0], bool)
            or isinstance(span[1], bool)
            or not isinstance(span[0], int)
            or not isinstance(span[1], int)
        ):
            raise ValueError(f"invalid interval at {where}[{index}]")
        start, end = span
        # UTF-8 byte-fallback tokens may share or nest the same character span.
        # Starts and ends must each be monotone, but native spans need not be
        # disjoint.  Local project-BPE spans are independently reconstructed in
        # the plan audit.
        if start < 0 or end <= start or start < previous_start or end < previous_end:
            raise ValueError(f"non-monotone interval at {where}[{index}]")
        result.append((start, end))
        previous_start = start
        previous_end = end
    return result


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Merge overlapping or touching character intervals deterministically."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def independently_derive_mapping(plan: dict, where: str) -> list[list[int]]:
    native = plan.get("author_response_tokens")
    local = plan.get("shared_bpe_tokens")
    mapping_record = plan.get("mapping")
    if not isinstance(native, dict) or not isinstance(local, dict):
        raise ValueError(f"missing token records at {where}")
    if not isinstance(mapping_record, dict):
        raise ValueError(f"missing mapping record at {where}")
    native_ids = native.get("token_ids")
    local_ids = local.get("token_ids")
    if not isinstance(native_ids, list) or not isinstance(local_ids, list) or not local_ids:
        raise ValueError(f"invalid token IDs at {where}")
    native_spans = _intervals(native.get("char_intervals"), f"{where}.native")
    local_spans = _intervals(local.get("char_intervals"), f"{where}.local")
    if len(native_ids) != len(native_spans) or len(local_ids) != len(local_spans):
        raise ValueError(f"token/span length mismatch at {where}")

    mode = mapping_record.get("mode")
    if mode == "exact_token":
        if native_ids != local_ids or native.get("char_intervals") != local.get(
            "char_intervals"
        ):
            raise ValueError(f"false exact-token mapping at {where}")
        derived = [[index] for index in range(len(local_ids))]
    elif mode == "character_overlap":
        derived = []
        native_cursor = 0
        for local_index, (local_start, local_end) in enumerate(local_spans):
            while (
                native_cursor < len(native_spans)
                and native_spans[native_cursor][1] <= local_start
            ):
                native_cursor += 1
            matches = []
            candidate = native_cursor
            while candidate < len(native_spans) and native_spans[candidate][0] < local_end:
                native_start, native_end = native_spans[candidate]
                if max(local_start, native_start) < min(local_end, native_end):
                    matches.append(candidate)
                candidate += 1
            if not matches:
                raise ValueError(f"unmapped local BPE slot at {where}[{local_index}]")
            derived.append(matches)
    else:
        raise ValueError(f"unknown mapping mode at {where}")

    stored = mapping_record.get("local_to_author_token_indices")
    if stored != derived:
        raise ValueError(f"stored mapping differs from independent overlap map at {where}")
    return derived


def recompute_shared_core(plan: dict, raw: dict, raw_chain: str, where: str) -> dict:
    probabilities = finite_probabilities(
        raw.get("author_response_token_probabilities"),
        f"{where}.author_response_token_probabilities",
    )
    author_tokens = plan["author_response_tokens"]
    if len(probabilities) != len(author_tokens["token_ids"]):
        raise ValueError(f"native token/probability length mismatch at {where}")

    mapping = independently_derive_mapping(plan, where)
    local_probabilities = []
    for local_index, source_indices in enumerate(mapping):
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(f"duplicate mapping index at {where}[{local_index}]")
        if any(index < 0 or index >= len(probabilities) for index in source_indices):
            raise ValueError(f"mapping index outside native vector at {where}[{local_index}]")
        values = [probabilities[index] for index in source_indices]
        local_probabilities.append(math.fsum(values) / len(values))

    local_count = len(local_probabilities)
    if local_count < 1:
        raise ValueError(f"answer has no local BPE slots at {where}")
    windows = plan.get("eligible_windows")
    if not isinstance(windows, list):
        raise ValueError(f"missing eligible windows at {where}")

    # The benchmark keeps only candidate windows containing at least one
    # lexical (isalnum-covered) local BPE.  Rebuild that label-free filter
    # independently instead of assuming that every stride-1 candidate is
    # eligible.  In particular, punctuation-only candidates create gaps in
    # the stored token_start sequence.
    response = plan.get("original_response")
    if not isinstance(response, str):
        raise ValueError(f"missing original response at {where}")
    local_spans = _intervals(
        plan["shared_bpe_tokens"].get("char_intervals"), f"{where}.local_windows"
    )
    if len(local_spans) != local_count or any(end > len(response) for _, end in local_spans):
        raise ValueError(f"local window spans exceed response at {where}")
    lexical = [
        any(response[index].isalnum() for index in range(start, end))
        for start, end in local_spans
    ]
    expected_slot_count = 4 if local_count >= 4 else local_count
    candidate_starts = range(local_count - 3) if local_count >= 4 else range(1)
    expected_starts = [
        start
        for start in candidate_starts
        if any(lexical[start : start + expected_slot_count])
    ]
    if len(windows) != len(expected_starts):
        raise ValueError(f"window count violates frozen short-window rule at {where}")

    window_ids = []
    window_probabilities = []
    for window_index, (window, expected_start) in enumerate(zip(windows, expected_starts)):
        if not isinstance(window, dict):
            raise ValueError(f"non-object window at {where}[{window_index}]")
        start = window.get("token_start")
        slot_count = window.get("slot_count")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(slot_count, bool)
            or not isinstance(slot_count, int)
            or start != expected_start
            or slot_count != expected_slot_count
        ):
            raise ValueError(f"window geometry mismatch at {where}[{window_index}]")
        end = start + slot_count
        if end > local_count:
            raise ValueError(f"window exceeds local BPE vector at {where}[{window_index}]")
        expected_window_id = f"{plan['answer_id']}__k4_{start:05d}"
        if window.get("window_id") != expected_window_id:
            raise ValueError(f"window ID mismatch at {where}[{window_index}]")
        expected_intervals = merge_intervals(local_spans[start:end])
        if window.get("character_intervals") != expected_intervals:
            raise ValueError(f"window character intervals differ at {where}[{window_index}]")
        window_ids.append(expected_window_id)
        window_probabilities.append(
            math.fsum(local_probabilities[start:end]) / slot_count
        )

    return {
        "version": "ragognizer-shared-k4-probabilities-v2",
        "partition": str(plan["partition"]),
        "answer_id": str(plan["answer_id"]),
        "answer_sha256": str(plan["answer_sha256"]),
        "plan_row_sha256": str(plan["plan_row_sha256"]),
        "mapping_mode": str(plan["mapping"]["mode"]),
        "author_coordinate_mode": str(author_tokens["coordinate_mode"]),
        "author_public_packer_exact": bool(author_tokens["official_pack_exact"]),
        "author_response_token_probabilities": probabilities,
        "shared_bpe_token_probabilities": local_probabilities,
        "window_ids": window_ids,
        "shared_k4_window_probabilities": window_probabilities,
        "shared_answer_probability": max(window_probabilities),
        "author_native_answer_probability": max(probabilities),
        "author_native_threshold_appendix_only": AUTHOR_THRESHOLD,
        "shared_thresholds_applied": False,
        "labels_read": False,
        "raw_chain_sha256": raw_chain,
    }


def validate_raw_against_plan(plan: dict, raw: dict, where: str) -> list[float]:
    if set(raw) != RAW_KEYS:
        raise ValueError(f"raw schema mismatch at {where}")
    answer_id = str(plan.get("answer_id"))
    required = {
        "version": "ragognizer-native-probabilities-v2",
        "route": "model_card_default_transformer_heads",
        "partition": plan.get("partition"),
        "answer_id": answer_id,
        "answer_sha256": plan.get("answer_sha256"),
        "plan_sha256": PLAN_SHA256,
        "plan_row_sha256": plan.get("plan_row_sha256"),
        "author_response_token_ids": plan["author_response_tokens"]["token_ids"],
        "author_response_token_char_intervals": plan["author_response_tokens"][
            "char_intervals"
        ],
        "author_coordinate_mode": plan["author_response_tokens"]["coordinate_mode"],
        "author_public_packer_exact": plan["author_response_tokens"][
            "official_pack_exact"
        ],
        "postprocessor": False,
        "quantization": None,
        "threshold_applied": False,
    }
    for key, expected in required.items():
        if raw.get(key) != expected or type(raw.get(key)) is not type(expected):
            raise ValueError(f"raw/plan field mismatch at {where}.{key}")
    require_hash(raw.get("run_identity_sha256"), f"{where}.run_identity_sha256")
    probabilities = finite_probabilities(
        raw.get("author_response_token_probabilities"),
        f"{where}.author_response_token_probabilities",
    )
    if len(probabilities) != len(required["author_response_token_ids"]):
        raise ValueError(f"raw probability length mismatch at {where}")
    expected_preds = [int(score >= AUTHOR_THRESHOLD) for score in probabilities]
    if raw.get("author_native_preds_0_6523_appendix_only") != expected_preds:
        raise ValueError(f"author 0.6523 token decisions mismatch at {where}")
    return probabilities


def validate_formal_hash_records() -> dict:
    """Authenticate freeze/result/status links before any score or gold read."""
    freeze = load_json(FREEZE)
    freeze_file_sha256 = sha256_file(FREEZE)
    frozen_payload_hash = require_hash(
        freeze.get("score_freeze_payload_sha256"),
        "SCORE_FREEZE.score_freeze_payload_sha256",
    )
    freeze_core = dict(freeze)
    del freeze_core["score_freeze_payload_sha256"]
    if frozen_payload_hash != sha256_json(freeze_core):
        raise ValueError("SCORE_FREEZE canonical payload hash mismatch")
    freeze_required = {
        "version": "ragognizer-formal-score-freeze-v2",
        "status": "full_scores_frozen_before_calibration_gold",
        "plan_sha256": PLAN_SHA256,
        "answers": EXPECTED_TOTAL_ROWS,
        "partition_answers": EXPECTED_ROWS,
        "eligible_windows": EXPECTED_TOTAL_WINDOWS,
        "partition_windows": EXPECTED_WINDOWS,
        "main_scoring_scope": "calibration_only_159_answers_after_full_3839_freeze",
        "fit_gold_opened": False,
        "calibration_gold_opened": False,
        "thresholds_applied": False,
        "test_opened": False,
    }
    for key, expected in freeze_required.items():
        if freeze.get(key) != expected:
            raise ValueError(f"SCORE_FREEZE field mismatch: {key}")
    for key in (
        "adapter_audit_sha256",
        "gpu_run_audit_sha256",
        "raw_scores_sha256",
        "raw_final_chain_sha256",
        "adapted_scores_sha256",
        "adapted_final_chain_sha256",
    ):
        require_hash(freeze.get(key), f"SCORE_FREEZE.{key}")

    adapter_audit = load_json(ADAPTER_AUDIT)
    gpu_audit = load_json(GPU_AUDIT)
    artifact_links = {
        "freeze -> adapter audit": (
            freeze.get("adapter_audit_sha256"),
            sha256_file(ADAPTER_AUDIT),
        ),
        "freeze -> GPU audit": (
            freeze.get("gpu_run_audit_sha256"),
            sha256_file(GPU_AUDIT),
        ),
        "adapter -> GPU audit": (
            adapter_audit.get("gpu_run_audit_sha256"),
            sha256_file(GPU_AUDIT),
        ),
        "adapter -> raw": (adapter_audit.get("raw_sha256"), sha256_file(RAW)),
        "adapter -> adapted": (
            adapter_audit.get("output_sha256"),
            sha256_file(ADAPTED),
        ),
        "freeze -> evaluator": (
            freeze.get("evaluator_script_sha256"),
            sha256_file(FORMAL_EVALUATOR),
        ),
        "adapter -> adapter script": (
            adapter_audit.get("adapter_script_sha256"),
            sha256_file(ADAPTER_SCRIPT),
        ),
        "GPU audit -> runner": (
            gpu_audit.get("runner_script_sha256"),
            sha256_file(RUNNER),
        ),
    }
    for name, (recorded, actual) in artifact_links.items():
        if recorded != actual:
            raise ValueError(f"artifact hash link mismatch: {name}")
    adapter_required = {
        "version": "ragognizer-adapter-run-audit-v2",
        "status": "complete",
        "adapter_scope": "formal_full",
        "partition_filter": None,
        "rows": EXPECTED_TOTAL_ROWS,
        "partition_rows": EXPECTED_ROWS,
        "windows": EXPECTED_TOTAL_WINDOWS,
        "partition_windows": EXPECTED_WINDOWS,
        "plan_sha256": PLAN_SHA256,
        "raw_final_chain_sha256": freeze.get("raw_final_chain_sha256"),
        "adapted_final_chain_sha256": freeze.get("adapted_final_chain_sha256"),
        "labels_read": False,
        "thresholds_applied": False,
        "subset_mode": False,
        "gpu_audit_verified": True,
    }
    for key, expected in adapter_required.items():
        if adapter_audit.get(key) != expected:
            raise ValueError(f"ADAPTER_AUDIT field mismatch: {key}")
    gpu_required = {
        "version": "ragognizer-route-a-gpu-run-audit-v2",
        "status": "complete",
        "run_scope": "formal_full",
        "rows": EXPECTED_TOTAL_ROWS,
        "partition_rows": EXPECTED_ROWS,
        "plan_sha256": PLAN_SHA256,
        "output_sha256": freeze.get("raw_scores_sha256"),
        "raw_final_chain_sha256": freeze.get("raw_final_chain_sha256"),
        "quantization": None,
        "postprocessor": False,
        "labels_read": False,
        "test_opened": False,
    }
    for key, expected in gpu_required.items():
        if gpu_audit.get(key) != expected:
            raise ValueError(f"GPU_RUN_AUDIT field mismatch: {key}")
    run_identity = gpu_audit.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("GPU_RUN_AUDIT lacks canonical run_identity")
    run_identity_sha256 = require_hash(
        gpu_audit.get("run_identity_sha256"), "GPU_RUN_AUDIT.run_identity_sha256"
    )
    if sha256_json(run_identity) != run_identity_sha256:
        raise ValueError("GPU_RUN_AUDIT run identity hash mismatch")
    identity_required = {
        "route": "model_card_default_transformer_heads_with_accelerate_cpu_offload",
        "device_placement_change_only": True,
        "plan_sha256": PLAN_SHA256,
        "torch_dtype_argument": "torch.bfloat16",
        "quantization": None,
        "postprocessor": False,
        "batch_size": 1,
        "author_probability": "sigmoid(integrated hallu_head_neg_16 logit)",
    }
    for key, expected in identity_required.items():
        if run_identity.get(key) != expected or gpu_audit.get(key) != expected:
            raise ValueError(f"GPU run identity field mismatch: {key}")
    if not isinstance(run_identity.get("versions"), dict) or not run_identity["versions"]:
        raise ValueError("GPU run identity dependency versions are absent")
    if gpu_audit.get("loaded_parameter_dtype_tensor_counts") != {"torch.bfloat16": 746}:
        raise ValueError("formal model parameter dtype identity changed")
    identity_artifacts = {
        "runner_script_sha256": RUNNER,
        "model_manifest_sha256": MODEL_MANIFEST,
        "source_audit_sha256": SOURCE_AUDIT,
        "runtime_alias_audit_sha256": RUNTIME_ALIAS_AUDIT,
    }
    for key, path in identity_artifacts.items():
        if run_identity.get(key) != sha256_file(path):
            raise ValueError(f"run identity artifact changed: {key}")

    formal_result = load_json(FORMAL_RESULT)
    formal_result_sha256 = sha256_file(FORMAL_RESULT)
    if formal_result.get("version") != "ragognizer-formal-calibration-results-v2":
        raise ValueError("CAL_RESULTS version mismatch")
    if formal_result.get("status") != "complete":
        raise ValueError("CAL_RESULTS is not complete")
    if formal_result.get("identity") != "official Llama-2 model-card transformer_heads checkpoint":
        raise ValueError("CAL_RESULTS method identity mismatch")
    if formal_result.get("method_status") != "2026_arxiv_preprint":
        raise ValueError("CAL_RESULTS publication identity mismatch")
    if formal_result.get("full_scores_frozen_before_calibration_gold") is not True:
        raise ValueError("CAL_RESULTS does not attest a pre-gold full freeze")
    if formal_result.get("score_freeze_file_sha256") != freeze_file_sha256:
        raise ValueError("CAL_RESULTS -> SCORE_FREEZE file hash link mismatch")
    if formal_result.get("score_freeze_payload_sha256") != frozen_payload_hash:
        raise ValueError("CAL_RESULTS -> SCORE_FREEZE payload hash link mismatch")
    if formal_result.get("adapted_scores_sha256") != freeze.get(
        "adapted_scores_sha256"
    ):
        raise ValueError("CAL_RESULTS -> adapted score hash link mismatch")
    if formal_result.get("test_opened") is not False:
        raise ValueError("CAL_RESULTS crossed the test boundary")
    if formal_result.get("full_coverage") != {
        "answers": EXPECTED_TOTAL_ROWS,
        "fit_answers": EXPECTED_ROWS["fit"],
        "calibration_answers": EXPECTED_ROWS["calibration"],
        "fit_gold_read": False,
    }:
        raise ValueError("CAL_RESULTS full-coverage record mismatch")
    if formal_result.get("calibration_scored") != {
        "answers": EXPECTED_ROWS["calibration"],
        "windows": EXPECTED_WINDOWS["calibration"],
    }:
        raise ValueError("CAL_RESULTS calibration coverage mismatch")

    status = load_json(FORMAL_STATUS)
    status_sha256 = sha256_file(FORMAL_STATUS)
    status_required = {
        "version": "ragognizer-formal-scoring-status-v2",
        "status": "complete",
        "result_sha256": formal_result_sha256,
        "score_freeze_sha256": freeze_file_sha256,
        "adapted_scores_sha256": freeze.get("adapted_scores_sha256"),
        "full_native_coverage_required": EXPECTED_TOTAL_ROWS,
        "full_native_coverage_complete": EXPECTED_TOTAL_ROWS,
        "fit_native_outputs": EXPECTED_ROWS["fit"],
        "calibration_native_outputs": EXPECTED_ROWS["calibration"],
        "main_scoring_scope": "calibration_only_159_answers",
        "substitution_used": False,
        "quantization_used": False,
        "fit_gold_opened": False,
        "test_opened": False,
    }
    for key, expected in status_required.items():
        if status.get(key) != expected:
            raise ValueError(f"SCORING_STATUS hash/status link mismatch: {key}")

    return {
        "freeze": freeze,
        "formal_result": formal_result,
        "freeze_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": frozen_payload_hash,
        "formal_result_sha256": formal_result_sha256,
        "formal_status_sha256": status_sha256,
        "gpu_audit": gpu_audit,
        "adapter_audit": adapter_audit,
    }


def audit_score_streams(freeze: dict) -> dict:
    plan_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    adapted_digest = hashlib.sha256()
    plan_rows = stream_jsonl(PLAN, plan_digest)
    raw_rows = stream_jsonl(RAW, raw_digest)
    adapted_rows = stream_jsonl(ADAPTED, adapted_digest)

    row_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    answer_ids = set()
    calibration_window_ids = []
    calibration_window_id_set = set()
    calibration_window_scores = []
    calibration_answer_ids = []
    calibration_answer_scores = []
    calibration_native_answer_scores = []
    ordered_answer_ids = []
    formal_calibration_native = {}
    plan_chain = ZERO_CHAIN
    raw_chain = ZERO_CHAIN
    adapted_chain = ZERO_CHAIN
    run_identity_sha256 = None
    mapped_local_slots = 0
    native_probability_values = 0

    sentinel = object()
    triples = zip_longest(plan_rows, raw_rows, adapted_rows, fillvalue=sentinel)
    for row_index, triple in enumerate(triples):
        if sentinel in triple:
            raise ValueError("plan/raw/adapted JSONL row counts differ")
        (plan_line, plan), (raw_line, raw), (adapted_line, adapted) = triple
        plan_where = f"plan line {plan_line}"
        raw_where = f"raw line {raw_line}"
        adapted_where = f"adapted line {adapted_line}"

        plan_row_hash = validate_plan_row(plan, plan_where)
        plan_chain = sha256_text(plan_chain + "\n" + plan_row_hash)
        partition = plan.get("partition")
        if partition not in EXPECTED_ROWS or plan.get("official_split") != "train":
            raise ValueError(f"partition boundary mismatch at {plan_where}")
        answer_id = str(plan.get("answer_id"))
        if answer_id in answer_ids:
            raise ValueError(f"duplicate answer_id: {answer_id}")
        answer_ids.add(answer_id)
        ordered_answer_ids.append(answer_id)

        probabilities = validate_raw_against_plan(plan, raw, raw_where)
        raw_chain = validate_chained_row(raw, raw_chain, "raw_row_sha256", raw_where)
        current_run_identity = str(raw["run_identity_sha256"])
        if run_identity_sha256 is None:
            run_identity_sha256 = current_run_identity
        elif current_run_identity != run_identity_sha256:
            raise ValueError("formal raw rows contain mixed run identities")

        expected_adapted_core = recompute_shared_core(
            plan, raw, raw_chain, f"answer {answer_id}"
        )
        if set(adapted) != ADAPTED_KEYS:
            raise ValueError(f"adapted schema mismatch at {adapted_where}")
        adapted_chain = validate_chained_row(
            adapted, adapted_chain, "adapted_row_sha256", adapted_where
        )
        actual_adapted_core = {
            key: value
            for key, value in adapted.items()
            if key
            not in {
                "adapted_row_sha256",
                "previous_chain_sha256",
                "chain_sha256",
            }
        }
        if actual_adapted_core != expected_adapted_core:
            differing = sorted(
                key
                for key in set(actual_adapted_core) | set(expected_adapted_core)
                if actual_adapted_core.get(key) != expected_adapted_core.get(key)
            )
            raise ValueError(
                f"independent adapted arithmetic mismatch at {adapted_where}: {differing}"
            )

        row_counts[partition] += 1
        row_window_ids = expected_adapted_core["window_ids"]
        row_window_scores = expected_adapted_core["shared_k4_window_probabilities"]
        window_counts[partition] += len(row_window_ids)
        mapped_local_slots += len(
            expected_adapted_core["shared_bpe_token_probabilities"]
        )
        native_probability_values += len(probabilities)
        if partition == "calibration":
            for window_id in row_window_ids:
                if window_id in calibration_window_id_set:
                    raise ValueError(f"duplicate calibration window_id: {window_id}")
                calibration_window_id_set.add(window_id)
            calibration_window_ids.extend(row_window_ids)
            calibration_window_scores.extend(row_window_scores)
            calibration_answer_ids.append(answer_id)
            calibration_answer_scores.append(
                expected_adapted_core["shared_answer_probability"]
            )
            calibration_native_answer_scores.append(
                expected_adapted_core["author_native_answer_probability"]
            )
            formal_calibration_native[answer_id] = {
                "answer_sha256": str(plan["answer_sha256"]),
                "plan_row_sha256": plan_row_hash,
                "probabilities": tuple(probabilities),
            }

    plan_sha256 = plan_digest.hexdigest()
    raw_sha256 = raw_digest.hexdigest()
    adapted_sha256 = adapted_digest.hexdigest()
    if row_index + 1 != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"formal answer coverage mismatch: {row_index + 1}")
    if dict(row_counts) != EXPECTED_ROWS:
        raise ValueError(f"formal partition answer coverage mismatch: {dict(row_counts)}")
    if dict(window_counts) != EXPECTED_WINDOWS:
        raise ValueError(
            f"formal partition window coverage mismatch: {dict(window_counts)}"
        )
    expected_hashes = {
        "plan": (plan_sha256, PLAN_SHA256),
        "freeze.plan": (plan_sha256, freeze.get("plan_sha256")),
        "raw": (raw_sha256, freeze.get("raw_scores_sha256")),
        "adapted": (adapted_sha256, freeze.get("adapted_scores_sha256")),
        "raw final chain": (raw_chain, freeze.get("raw_final_chain_sha256")),
        "adapted final chain": (
            adapted_chain,
            freeze.get("adapted_final_chain_sha256"),
        ),
    }
    for name, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    if len(calibration_window_ids) != EXPECTED_WINDOWS["calibration"]:
        raise ValueError("calibration window collection mismatch")
    if len(calibration_answer_ids) != EXPECTED_ROWS["calibration"]:
        raise ValueError("calibration answer collection mismatch")

    return {
        "plan_sha256": plan_sha256,
        "raw_sha256": raw_sha256,
        "adapted_sha256": adapted_sha256,
        "plan_independent_final_chain_sha256": plan_chain,
        "raw_final_chain_sha256": raw_chain,
        "adapted_final_chain_sha256": adapted_chain,
        "run_identity_sha256": run_identity_sha256,
        "ordered_answer_ids_sha256": sha256_json(ordered_answer_ids),
        "partition_answers": dict(row_counts),
        "partition_windows": dict(window_counts),
        "mapped_local_bpe_slots": mapped_local_slots,
        "native_probability_values": native_probability_values,
        "calibration_window_ids": calibration_window_ids,
        "calibration_window_scores": calibration_window_scores,
        "calibration_answer_ids": calibration_answer_ids,
        "calibration_answer_scores": calibration_answer_scores,
        "calibration_native_answer_scores": calibration_native_answer_scores,
        "formal_calibration_native": formal_calibration_native,
    }


def compare_provisional_calibration(formal_calibration_native: dict) -> dict:
    if not PROVISIONAL_RAW.exists():
        return {
            "status": "not_present",
            "path": PROVISIONAL_RAW.name,
            "required_if_present": True,
        }
    digest = hashlib.sha256()
    previous_chain = ZERO_CHAIN
    seen = set()
    matching_rows = 0
    compared_values = 0
    for line_no, row in stream_jsonl(PROVISIONAL_RAW, digest):
        where = f"{PROVISIONAL_RAW.name}:{line_no}"
        if set(row) != RAW_KEYS:
            raise ValueError(f"provisional raw schema mismatch at {where}")
        previous_chain = validate_chained_row(
            row, previous_chain, "raw_row_sha256", where
        )
        answer_id = str(row.get("answer_id"))
        if answer_id in seen:
            raise ValueError(f"duplicate provisional answer_id: {answer_id}")
        seen.add(answer_id)
        expected = formal_calibration_native.get(answer_id)
        if expected is None:
            raise ValueError(f"unknown provisional calibration answer_id: {answer_id}")
        if row.get("version") != "ragognizer-native-probabilities-v2":
            raise ValueError(f"provisional raw version mismatch at {where}")
        if row.get("partition") != "calibration":
            raise ValueError(f"provisional raw partition mismatch at {where}")
        if row.get("answer_sha256") != expected["answer_sha256"]:
            raise ValueError(f"provisional answer hash mismatch at {where}")
        if row.get("plan_row_sha256") != expected["plan_row_sha256"]:
            raise ValueError(f"provisional plan-row hash mismatch at {where}")
        probabilities = finite_probabilities(
            row.get("author_response_token_probabilities"),
            f"{where}.author_response_token_probabilities",
        )
        if tuple(probabilities) != expected["probabilities"]:
            raise ValueError(
                f"formal/provisional native probabilities differ for {answer_id}"
            )
        matching_rows += 1
        compared_values += len(probabilities)
    if seen != set(formal_calibration_native) or matching_rows != EXPECTED_ROWS[
        "calibration"
    ]:
        raise ValueError("formal/provisional calibration answer coverage differs")
    return {
        "status": "pass",
        "path": PROVISIONAL_RAW.name,
        "sha256": digest.hexdigest(),
        "rows_expected": EXPECTED_ROWS["calibration"],
        "rows_compared": matching_rows,
        "rows_exactly_equal": matching_rows,
        "probability_values_compared": compared_values,
        "all_author_response_token_probabilities_exactly_equal": True,
        "provisional_final_chain_sha256": previous_chain,
        "comparison_key": "answer_id",
    }


def load_calibration_gold(
    path: Path, id_key: str, expected_rows: int, expected_sha256: str
) -> tuple[dict[str, int], str]:
    digest = hashlib.sha256()
    result = {}
    for line_no, row in stream_jsonl(path, digest):
        where = f"{path.name}:{line_no}"
        if row.get("partition") != "calibration" or row.get("eligible") is not True:
            raise ValueError(f"non-calibration/ineligible gold row at {where}")
        identifier = str(row.get(id_key))
        if identifier in result:
            raise ValueError(f"duplicate gold ID at {where}: {identifier}")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"non-binary gold label at {where}")
        result[identifier] = label
    if len(result) != expected_rows:
        raise ValueError(f"gold row count mismatch for {path.name}: {len(result)}")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"frozen calibration gold hash changed: {path.name}")
    return result, actual_sha256


def validate_vectors(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores) or not labels:
        raise ValueError("invalid aligned label/score vectors")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("metric labels are not binary")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("metric scores are not finite")
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError("AUROC requires both calibration classes")


def grouped_score_counts(
    labels: Sequence[int], scores: Sequence[float], *, reverse: bool
) -> Iterable[tuple[float, int, int]]:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=reverse)
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        positives = 0
        negatives = 0
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                positives += 1
            else:
                negatives += 1
            index += 1
        yield float(score), positives, negatives


def f1_opt_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict:
    validate_vectors(labels, scores)
    total_positives = sum(labels)
    tp = 0
    fp = 0
    best = None
    for threshold, positive_group, negative_group in grouped_score_counts(
        labels, scores, reverse=True
    ):
        tp += positive_group
        fp += negative_group
        fn = total_positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_positives
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, precision, threshold, tp, fp, fn, recall)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise ValueError("no F1 threshold candidates")
    f1, precision, threshold, tp, fp, fn, recall = best
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    validate_vectors(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    lower_negatives = 0
    favorable_pairs = 0.0
    for _, positive_group, negative_group in grouped_score_counts(
        labels, scores, reverse=False
    ):
        favorable_pairs += positive_group * lower_negatives
        favorable_pairs += 0.5 * positive_group * negative_group
        lower_negatives += negative_group
    return favorable_pairs / (positives * negatives)


def sklearn_semantic_average_precision(
    labels: Sequence[int], scores: Sequence[float]
) -> float:
    """Non-interpolated AP: sum((recall gain) * precision) at tied scores."""
    validate_vectors(labels, scores)
    positives = sum(labels)
    tp = 0
    fp = 0
    average_precision = 0.0
    for _, positive_group, negative_group in grouped_score_counts(
        labels, scores, reverse=True
    ):
        tp += positive_group
        fp += negative_group
        precision = tp / (tp + fp)
        average_precision += (positive_group / positives) * precision
    return average_precision


def metrics(labels: Sequence[int], scores: Sequence[float]) -> dict:
    validate_vectors(labels, scores)
    return {
        "rows": len(labels),
        "positives": sum(labels),
        "f1_opt": f1_opt_threshold(labels, scores),
        "auroc": tie_aware_auroc(labels, scores),
        "average_precision": sklearn_semantic_average_precision(labels, scores),
    }


def metrics_at_fixed_threshold(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict:
    validate_vectors(labels, scores)
    tp = sum(label == 1 and score >= threshold for label, score in zip(labels, scores))
    fp = sum(label == 0 and score >= threshold for label, score in zip(labels, scores))
    fn = sum(label == 1 and score < threshold for label, score in zip(labels, scores))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "rows": len(labels),
        "positives": sum(labels),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "auroc": tie_aware_auroc(labels, scores),
        "average_precision": sklearn_semantic_average_precision(labels, scores),
    }


def compare_metric_block(independent: object, formal: object, where: str) -> dict:
    differences = []
    numeric_deltas = []

    def walk(left: object, right: object, path: str) -> None:
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                differences.append(path + ".keys")
                return
            for key in left:
                walk(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if left is not right:
                differences.append(path)
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(left, int) and isinstance(right, int):
                if left != right:
                    differences.append(path)
                return
            left_float = float(left)
            right_float = float(right)
            delta = abs(left_float - right_float)
            numeric_deltas.append(delta)
            if not math.isfinite(right_float) or delta > METRIC_ABS_TOLERANCE:
                differences.append(path)
            return
        if left != right:
            differences.append(path)

    walk(independent, formal, where)
    if differences:
        raise ValueError(f"formal metric mismatch: {differences[:8]}")
    return {
        "status": "exact_counts_and_within_float_tolerance",
        "float_absolute_tolerance": METRIC_ABS_TOLERANCE,
        "maximum_float_absolute_delta": max(numeric_deltas, default=0.0),
    }


def main() -> None:
    # This must complete before any calibration gold file is opened.
    records = validate_formal_hash_records()
    streams = audit_score_streams(records["freeze"])
    provisional_comparison = compare_provisional_calibration(
        streams["formal_calibration_native"]
    )

    # Gold access begins here, after the complete formal hash/arithmetic gates.
    if streams["run_identity_sha256"] != records["gpu_audit"].get(
        "run_identity_sha256"
    ):
        raise ValueError("raw row run identity differs from GPU_RUN_AUDIT")
    selection = records["gpu_audit"].get("selection")
    if selection != {
        "partition": None,
        "limit": None,
        "answer_ids_sha256": streams["ordered_answer_ids_sha256"],
    }:
        raise ValueError("GPU_RUN_AUDIT formal selection identity mismatch")

    window_gold, window_gold_sha256 = load_calibration_gold(
        CAL_WINDOWS,
        "window_id",
        EXPECTED_WINDOWS["calibration"],
        CAL_WINDOWS_SHA256,
    )
    answer_gold, answer_gold_sha256 = load_calibration_gold(
        CAL_ANSWERS,
        "answer_id",
        EXPECTED_ROWS["calibration"],
        CAL_ANSWERS_SHA256,
    )
    window_ids = streams["calibration_window_ids"]
    answer_ids = streams["calibration_answer_ids"]
    if set(window_ids) != set(window_gold) or len(window_ids) != len(window_gold):
        raise ValueError("calibration window score/gold join is not exact")
    if set(answer_ids) != set(answer_gold) or len(answer_ids) != len(answer_gold):
        raise ValueError("calibration answer score/gold join is not exact")
    window_labels = [window_gold[window_id] for window_id in window_ids]
    answer_labels = [answer_gold[answer_id] for answer_id in answer_ids]

    independent_window = metrics(window_labels, streams["calibration_window_scores"])
    independent_answer = metrics(answer_labels, streams["calibration_answer_scores"])
    independent_author = metrics_at_fixed_threshold(
        answer_labels,
        streams["calibration_native_answer_scores"],
        AUTHOR_THRESHOLD,
    )
    formal_result = records["formal_result"]
    main_table = formal_result.get("shared_main_table")
    appendix = formal_result.get("appendix_only")
    if not isinstance(main_table, dict) or not isinstance(appendix, dict):
        raise ValueError("CAL_RESULTS metric tables are missing")
    if main_table.get("threshold_rule") != THRESHOLD_RULE:
        raise ValueError("CAL_RESULTS threshold rule differs from the frozen rule")
    comparisons = {
        "four_bpe_window": compare_metric_block(
            independent_window,
            main_table.get("four_bpe_window"),
            "shared_main_table.four_bpe_window",
        ),
        "answer_max_shared_window": compare_metric_block(
            independent_answer,
            main_table.get("answer_max_shared_window"),
            "shared_main_table.answer_max_shared_window",
        ),
        "author_native_answer_max_token_at_0_6523": compare_metric_block(
            independent_author,
            appendix.get("author_native_answer_max_token_at_0_6523"),
            "appendix_only.author_native_answer_max_token_at_0_6523",
        ),
    }

    result = {
        "version": "ragognizer-formal-score-independent-audit-v1",
        "status": "pass",
        "auditor_script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_hash_chain": {
            "score_freeze_file_sha256": records["freeze_file_sha256"],
            "score_freeze_payload_sha256": records["freeze_payload_sha256"],
            "cal_results_sha256": records["formal_result_sha256"],
            "scoring_status_sha256": records["formal_status_sha256"],
            "validated_before_gold_access": True,
        },
        "streamed_score_audit": {
            "plan_sha256": streams["plan_sha256"],
            "raw_sha256": streams["raw_sha256"],
            "adapted_sha256": streams["adapted_sha256"],
            "plan_independent_final_chain_sha256": streams[
                "plan_independent_final_chain_sha256"
            ],
            "raw_final_chain_sha256": streams["raw_final_chain_sha256"],
            "adapted_final_chain_sha256": streams["adapted_final_chain_sha256"],
            "run_identity_sha256": streams["run_identity_sha256"],
            "answers": EXPECTED_TOTAL_ROWS,
            "partition_answers": streams["partition_answers"],
            "eligible_windows": EXPECTED_TOTAL_WINDOWS,
            "partition_windows": streams["partition_windows"],
            "mapped_local_bpe_slots": streams["mapped_local_bpe_slots"],
            "native_probability_values": streams["native_probability_values"],
            "all_plan_row_hashes_recomputed": True,
            "all_raw_row_hashes_and_chain_links_recomputed": True,
            "all_adapted_row_hashes_and_chain_links_recomputed": True,
            "all_token_to_local_bpe_means_recomputed": True,
            "all_four_or_actual_slot_window_means_recomputed": True,
            "all_answer_maxima_recomputed": True,
        },
        "formal_vs_provisional_calibration_native": provisional_comparison,
        "calibration_metrics_recomputed": {
            "four_bpe_window": independent_window,
            "answer_max_shared_window": independent_answer,
            "threshold_rule": THRESHOLD_RULE,
        },
        "appendix_only_recomputed": {
            "author_native_answer_max_token_at_0_6523": independent_author,
        },
        "formal_result_comparison": {
            "status": "pass",
            "blocks": comparisons,
        },
        "gold_access": {
            "began_after_formal_hash_and_score_gates": True,
            "calibration_answers_read": EXPECTED_ROWS["calibration"],
            "calibration_windows_read": EXPECTED_WINDOWS["calibration"],
            "calibration_answers_sha256": answer_gold_sha256,
            "calibration_windows_sha256": window_gold_sha256,
            "fit_gold_opened": False,
            "test_opened": False,
        },
        "implementation": {
            "adapter_module_imported": False,
            "evaluate_shared_module_imported": False,
            "third_party_packages_imported": False,
            "gpu_used": False,
        },
    }
    result["audit_payload_sha256"] = sha256_json(result)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
