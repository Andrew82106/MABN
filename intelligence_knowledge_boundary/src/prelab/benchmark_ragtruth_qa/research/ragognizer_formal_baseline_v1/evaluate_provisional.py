"""Freeze and score a calibration-only provisional RAGognizer run.

The freeze stage validates all 159 calibration outputs and never opens gold.
Only the later evaluate stage reads calibration labels.  This does not satisfy
the formal 3,839-answer coverage gate, so its result is always provisional.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from adapter import (
    FROZEN_VERSION,
    ZERO_CHAIN,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_chained_row,
    validate_plan_row,
)
from evaluate_shared import metrics, metrics_at_fixed_threshold


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = HERE / "INFERENCE_PLAN.jsonl"
RAW = HERE / "PROVISIONAL_CAL_NATIVE.jsonl"
GPU_AUDIT = HERE / "PROVISIONAL_CAL_GPU_AUDIT.json"
ADAPTED = HERE / "PROVISIONAL_CAL_SHARED.jsonl"
ADAPTER_AUDIT = HERE / "PROVISIONAL_CAL_ADAPTER_AUDIT.json"
FREEZE = HERE / "PROVISIONAL_CAL_SCORE_FREEZE.json"
RESULT = HERE / "PROVISIONAL_CAL_RESULTS.json"
STATUS = HERE / "PROVISIONAL_CAL_STATUS.json"
PROTOCOL = HERE / "PROVISIONAL_PROTOCOL.json"
CAL_WINDOWS = PROJECT / "data" / "windows_k4_calibration.jsonl"
CAL_ANSWERS = PROJECT / "data" / "answers_calibration.jsonl"
PLAN_SHA256 = "90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76"
EXPECTED_ANSWERS = 159
EXPECTED_WINDOWS = 42_241


def freeze_scores() -> dict:
    if sha256_file(PLAN) != PLAN_SHA256:
        raise RuntimeError("full plan hash changed")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_provisional_inference":
        raise RuntimeError("provisional protocol was not frozen before inference")
    if protocol.get("evaluator_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("provisional evaluator changed after protocol freeze")
    if protocol.get("shared_metric_library_sha256") != sha256_file(HERE / "evaluate_shared.py"):
        raise RuntimeError("shared metric library changed after protocol freeze")
    if protocol.get("plan_sha256") != PLAN_SHA256:
        raise RuntimeError("provisional protocol plan identity changed")

    plans_all = list(read_jsonl(PLAN))
    plans = [row for row in plans_all if row.get("partition") == "calibration"]
    rows = list(read_jsonl(ADAPTED))
    if len(plans_all) != 3_839 or len(plans) != EXPECTED_ANSWERS or len(rows) != len(plans):
        raise RuntimeError("provisional calibration coverage changed")
    if [row["answer_id"] for row in rows] != [row["answer_id"] for row in plans]:
        raise RuntimeError("provisional adapted order differs from calibration plan")

    adapted_sha256 = sha256_file(ADAPTED)
    audit = json.loads(ADAPTER_AUDIT.read_text(encoding="utf-8"))
    gpu_audit = json.loads(GPU_AUDIT.read_text(encoding="utf-8"))
    checks = {
        "status": "provisional_complete",
        "adapter_scope": "provisional_calibration",
        "partition_filter": "calibration",
        "rows": EXPECTED_ANSWERS,
        "partition_rows": {"calibration": EXPECTED_ANSWERS},
        "windows": EXPECTED_WINDOWS,
        "partition_windows": {"calibration": EXPECTED_WINDOWS},
        "plan_sha256": PLAN_SHA256,
        "raw_sha256": sha256_file(RAW),
        "gpu_run_audit_sha256": sha256_file(GPU_AUDIT),
        "output_sha256": adapted_sha256,
    }
    for key, expected in checks.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"provisional adapter audit {key} mismatch")
    if not audit.get("gpu_audit_verified"):
        raise RuntimeError("provisional adapter did not verify GPU audit")
    if audit.get("adapter_script_sha256") != protocol.get("adapter_script_sha256"):
        raise RuntimeError("adapter differs from the pre-inference provisional freeze")
    if gpu_audit.get("runner_script_sha256") != protocol.get("runner_script_sha256"):
        raise RuntimeError("runner differs from the pre-inference provisional freeze")
    if audit.get("labels_read") is not False or audit.get("thresholds_applied") is not False:
        raise RuntimeError("provisional adapter crossed the gold boundary")

    previous = ZERO_CHAIN
    window_count = 0
    for index, (plan, row) in enumerate(zip(plans, rows)):
        validate_plan_row(plan, f"plan[{index}]")
        answer_id = str(plan["answer_id"])
        if row.get("version") != FROZEN_VERSION:
            raise RuntimeError(f"adapted version changed for {answer_id}")
        if row.get("partition") != "calibration":
            raise RuntimeError(f"non-calibration row in provisional output: {answer_id}")
        if row.get("plan_row_sha256") != plan.get("plan_row_sha256"):
            raise RuntimeError(f"plan row identity changed for {answer_id}")
        if row.get("answer_sha256") != plan.get("answer_sha256"):
            raise RuntimeError(f"answer identity changed for {answer_id}")
        expected_window_ids = [window["window_id"] for window in plan["eligible_windows"]]
        if row.get("window_ids") != expected_window_ids:
            raise RuntimeError(f"window order changed for {answer_id}")
        window_scores = list(map(float, row["shared_k4_window_probabilities"]))
        token_scores = list(map(float, row["shared_bpe_token_probabilities"]))
        native_scores = list(map(float, row["author_response_token_probabilities"]))
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in window_scores + token_scores + native_scores
        ):
            raise RuntimeError(f"invalid probability for {answer_id}")
        if float(row["shared_answer_probability"]) != max(window_scores):
            raise RuntimeError(f"answer max(window) changed for {answer_id}")
        if float(row["author_native_answer_probability"]) != max(native_scores):
            raise RuntimeError(f"author max(token) changed for {answer_id}")
        previous = validate_chained_row(
            row, previous, "adapted_row_sha256", f"adapted[{index}]"
        )
        window_count += len(window_scores)
    if window_count != EXPECTED_WINDOWS or previous != audit.get("adapted_final_chain_sha256"):
        raise RuntimeError("provisional window coverage or final chain changed")

    payload = {
        "version": "ragognizer-provisional-cal-score-freeze-v1",
        "status": "provisional_calibration_scores_frozen_before_gold",
        "formal_status": "N/A_until_full_3839_coverage",
        "plan_sha256": PLAN_SHA256,
        "provisional_protocol_sha256": sha256_file(PROTOCOL),
        "runner_gpu_audit_sha256": sha256_file(GPU_AUDIT),
        "raw_scores_sha256": sha256_file(RAW),
        "adapter_audit_sha256": sha256_file(ADAPTER_AUDIT),
        "adapted_scores_sha256": adapted_sha256,
        "adapted_final_chain_sha256": previous,
        "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
        "shared_metric_library_sha256": sha256_file(HERE / "evaluate_shared.py"),
        "answers": EXPECTED_ANSWERS,
        "eligible_windows": EXPECTED_WINDOWS,
        "fit_gold_opened": False,
        "calibration_gold_opened": False,
        "test_opened": False,
    }
    payload["score_freeze_payload_sha256"] = sha256_json(payload)
    FREEZE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def evaluate() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    stored_hash = freeze.pop("score_freeze_payload_sha256", None)
    if freeze.get("status") != "provisional_calibration_scores_frozen_before_gold":
        raise RuntimeError("provisional score freeze is absent")
    if stored_hash != sha256_json(freeze):
        raise RuntimeError("provisional score freeze payload changed")
    freeze["score_freeze_payload_sha256"] = stored_hash
    file_checks = {
        ADAPTED: freeze["adapted_scores_sha256"],
        ADAPTER_AUDIT: freeze["adapter_audit_sha256"],
        GPU_AUDIT: freeze["runner_gpu_audit_sha256"],
        RAW: freeze["raw_scores_sha256"],
        PROTOCOL: freeze["provisional_protocol_sha256"],
        HERE / "evaluate_shared.py": freeze["shared_metric_library_sha256"],
    }
    for path, expected in file_checks.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen provisional artifact changed: {path.name}")

    rows = list(read_jsonl(ADAPTED))
    window_ids = [wid for row in rows for wid in row["window_ids"]]
    window_scores = [float(s) for row in rows for s in row["shared_k4_window_probabilities"]]
    answer_ids = [str(row["answer_id"]) for row in rows]
    answer_scores = [float(row["shared_answer_probability"]) for row in rows]
    native_scores = [float(row["author_native_answer_probability"]) for row in rows]

    # The only gold access in this script starts here.
    window_gold = {str(row["window_id"]): int(row["label"]) for row in read_jsonl(CAL_WINDOWS)}
    answer_gold = {str(row["answer_id"]): int(row["label"]) for row in read_jsonl(CAL_ANSWERS)}
    if set(window_ids) != set(window_gold) or len(window_ids) != len(window_gold):
        raise RuntimeError("provisional window score/gold join is not exact")
    if set(answer_ids) != set(answer_gold) or len(answer_ids) != len(answer_gold):
        raise RuntimeError("provisional answer score/gold join is not exact")
    window_labels = [window_gold[key] for key in window_ids]
    answer_labels = [answer_gold[key] for key in answer_ids]

    payload = {
        "version": "ragognizer-provisional-cal-results-v1",
        "status": "provisional_complete",
        "formal_status": "N/A_until_full_3839_coverage",
        "method_status": "2026_arxiv_preprint",
        "shared_main_table": {
            "four_bpe_window": metrics(window_labels, window_scores),
            "answer_max_shared_window": metrics(answer_labels, answer_scores),
            "threshold_rule": "max F1, then precision, then higher threshold; score >= threshold"
        },
        "appendix_only": {
            "author_native_answer_max_token_at_0_6523": metrics_at_fixed_threshold(
                answer_labels, native_scores, 0.6523
            )
        },
        "score_freeze_file_sha256": sha256_file(FREEZE),
        "score_freeze_payload_sha256": stored_hash,
        "fit_gold_opened": False,
        "test_opened": False,
    }
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = {
        "version": "ragognizer-provisional-cal-status-v1",
        "status": "provisional_complete_formal_N/A",
        "result_sha256": sha256_file(RESULT),
        "score_freeze_sha256": sha256_file(FREEZE),
        "formal_coverage_complete": False,
        "required_formal_answers": 3839,
        "provisional_calibration_answers": 159,
        "fit_gold_opened": False,
        "test_opened": False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    payload = freeze_scores() if args.stage == "freeze" else evaluate()
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
