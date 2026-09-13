"""Freeze full label-free RAGognizer scores, then score calibration only.

The freeze stage requires all 3,839 answers and never opens gold.  The evaluate
stage verifies that freeze, discards fit scores from metric computation, and
opens only the project's 159-answer calibration gold.  There is no test path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from adapter import (
    FROZEN_VERSION,
    ZERO_CHAIN,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_chained_row,
    validate_plan_row,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = HERE / "INFERENCE_PLAN.jsonl"
ADAPTED = HERE / "SHARED_PROBABILITIES.jsonl"
ADAPTER_AUDIT = HERE / "ADAPTER_AUDIT.json"
FREEZE = HERE / "SCORE_FREEZE.json"
RESULT = HERE / "CAL_RESULTS.json"
STATUS = HERE / "SCORING_STATUS.json"
CAL_WINDOWS = PROJECT / "data" / "windows_k4_calibration.jsonl"
CAL_ANSWERS = PROJECT / "data" / "answers_calibration.jsonl"
PLAN_SHA256 = "90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76"
EXPECTED_ROWS = {"fit": 3_680, "calibration": 159}
EXPECTED_WINDOWS = {"fit": 653_979, "calibration": 42_241}


def f1_opt_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if (
        y.shape != s.shape
        or y.ndim != 1
        or y.size == 0
        or not np.isfinite(s).all()
        or not set(map(int, np.unique(y))) <= {0, 1}
    ):
        raise ValueError("invalid aligned label/score vectors")
    best = None
    for threshold in np.unique(s):
        predicted = s >= threshold
        tp = int(np.sum((y == 1) & predicted))
        fp = int(np.sum((y == 0) & predicted))
        fn = int(np.sum((y == 1) & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, precision, float(threshold), tp, fp, fn, recall)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
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


def metrics(labels: Sequence[int], scores: Sequence[float]) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    return {
        "rows": int(y.size),
        "positives": int(y.sum()),
        "f1_opt": f1_opt_threshold(y, s),
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
    }


def metrics_at_fixed_threshold(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    predicted = s >= threshold
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "rows": int(y.size),
        "positives": int(y.sum()),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
    }


def _validate_adapter_audit(adapted_sha256: str) -> dict:
    audit = json.loads(ADAPTER_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("subset_mode") is not False:
        raise RuntimeError("formal score freeze requires a complete non-subset adapter audit")
    checks = {
        "rows": sum(EXPECTED_ROWS.values()),
        "partition_rows": EXPECTED_ROWS,
        "windows": sum(EXPECTED_WINDOWS.values()),
        "partition_windows": EXPECTED_WINDOWS,
        "plan_sha256": PLAN_SHA256,
        "output_sha256": adapted_sha256,
    }
    for key, expected in checks.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"adapter audit {key} mismatch")
    if not audit.get("gpu_audit_verified"):
        raise RuntimeError("adapter audit did not verify GPU_RUN_AUDIT")
    if audit.get("labels_read") is not False or audit.get("thresholds_applied") is not False:
        raise RuntimeError("adapter crossed the score/gold boundary")
    return audit


def freeze_scores() -> dict:
    if PLAN_SHA256 == "__FULL_PLAN_SHA256__":
        raise RuntimeError("evaluator is not frozen: full plan SHA256 placeholder remains")
    if sha256_file(PLAN) != PLAN_SHA256:
        raise RuntimeError("full plan hash changed")
    plans = list(read_jsonl(PLAN))
    rows = list(read_jsonl(ADAPTED))
    if len(plans) != sum(EXPECTED_ROWS.values()) or len(rows) != len(plans):
        raise RuntimeError("formal freeze requires all 3,839 fit+calibration answers")
    if [row["answer_id"] for row in rows] != [row["answer_id"] for row in plans]:
        raise RuntimeError("adapted answer order differs from full plan")
    adapted_sha256 = sha256_file(ADAPTED)
    adapter_audit = _validate_adapter_audit(adapted_sha256)

    previous_chain = ZERO_CHAIN
    partition_rows: Counter[str] = Counter()
    partition_windows: Counter[str] = Counter()
    for index, (plan, row) in enumerate(zip(plans, rows)):
        validate_plan_row(plan, f"plan[{index}]")
        answer_id = str(plan["answer_id"])
        if row.get("version") != FROZEN_VERSION:
            raise RuntimeError(f"unexpected adapted score version for {answer_id}")
        if row.get("partition") != plan.get("partition"):
            raise RuntimeError(f"adapted partition mismatch for {answer_id}")
        if row.get("shared_thresholds_applied") is not False or row.get("labels_read") is not False:
            raise RuntimeError("adapted score file crossed the label/threshold boundary")
        if row.get("answer_sha256") != plan.get("answer_sha256"):
            raise RuntimeError(f"answer hash mismatch for {answer_id}")
        if row.get("plan_row_sha256") != plan.get("plan_row_sha256"):
            raise RuntimeError(f"plan row hash mismatch for {answer_id}")
        expected_window_ids = [window["window_id"] for window in plan["eligible_windows"]]
        if row.get("window_ids") != expected_window_ids:
            raise RuntimeError(f"window order differs for {answer_id}")
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
        previous_chain = validate_chained_row(
            row, previous_chain, "adapted_row_sha256", f"adapted[{index}]"
        )
        partition = str(plan["partition"])
        partition_rows[partition] += 1
        partition_windows[partition] += len(window_scores)

    if dict(partition_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"adapted answer coverage changed: {dict(partition_rows)}")
    if dict(partition_windows) != EXPECTED_WINDOWS:
        raise RuntimeError(f"adapted window coverage changed: {dict(partition_windows)}")
    if previous_chain != adapter_audit.get("adapted_final_chain_sha256"):
        raise RuntimeError("adapter final chain hash mismatch")

    payload = {
        "version": "ragognizer-formal-score-freeze-v2",
        "status": "full_scores_frozen_before_calibration_gold",
        "plan_sha256": PLAN_SHA256,
        "adapter_audit_sha256": sha256_file(ADAPTER_AUDIT),
        "gpu_run_audit_sha256": adapter_audit["gpu_run_audit_sha256"],
        "raw_scores_sha256": adapter_audit["raw_sha256"],
        "raw_final_chain_sha256": adapter_audit["raw_final_chain_sha256"],
        "adapted_scores_sha256": adapted_sha256,
        "adapted_final_chain_sha256": previous_chain,
        "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
        "answers": sum(partition_rows.values()),
        "partition_answers": dict(partition_rows),
        "eligible_windows": sum(partition_windows.values()),
        "partition_windows": dict(partition_windows),
        "main_scoring_scope": "calibration_only_159_answers_after_full_3839_freeze",
        "fit_scores_purpose": "coverage_and_future_freeze_only",
        "fit_gold_opened": False,
        "calibration_gold_opened": False,
        "thresholds_applied": False,
        "test_opened": False,
    }
    payload["score_freeze_payload_sha256"] = sha256_json(payload)
    FREEZE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def evaluate() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "full_scores_frozen_before_calibration_gold":
        raise RuntimeError("full score freeze is absent or invalid")
    frozen_hash = freeze.pop("score_freeze_payload_sha256", None)
    if frozen_hash != sha256_json(freeze):
        raise RuntimeError("score freeze payload hash mismatch")
    freeze["score_freeze_payload_sha256"] = frozen_hash
    if sha256_file(ADAPTED) != freeze["adapted_scores_sha256"]:
        raise RuntimeError("adapted probabilities changed after score freeze")
    if sha256_file(ADAPTER_AUDIT) != freeze["adapter_audit_sha256"]:
        raise RuntimeError("adapter audit changed after score freeze")

    all_rows = list(read_jsonl(ADAPTED))
    rows = [row for row in all_rows if row.get("partition") == "calibration"]
    if len(all_rows) != 3_839 or len(rows) != 159:
        raise RuntimeError("full freeze or calibration filter coverage changed")
    window_ids = [window_id for row in rows for window_id in row["window_ids"]]
    window_scores = [
        float(score)
        for row in rows
        for score in row["shared_k4_window_probabilities"]
    ]
    answer_ids = [str(row["answer_id"]) for row in rows]
    answer_scores = [float(row["shared_answer_probability"]) for row in rows]
    native_answer_scores = [
        float(row["author_native_answer_probability"]) for row in rows
    ]
    if len(window_ids) != 42_241:
        raise RuntimeError("calibration window score coverage changed")

    # Gold access begins here and is limited to the two calibration-only files.
    window_gold = {
        str(row["window_id"]): int(row["label"]) for row in read_jsonl(CAL_WINDOWS)
    }
    answer_gold = {
        str(row["answer_id"]): int(row["label"]) for row in read_jsonl(CAL_ANSWERS)
    }
    if set(window_ids) != set(window_gold) or len(window_ids) != len(window_gold):
        raise RuntimeError("calibration window score/gold join is not exact")
    if set(answer_ids) != set(answer_gold) or len(answer_ids) != len(answer_gold):
        raise RuntimeError("calibration answer score/gold join is not exact")
    window_labels = [window_gold[window_id] for window_id in window_ids]
    answer_labels = [answer_gold[answer_id] for answer_id in answer_ids]

    payload = {
        "version": "ragognizer-formal-calibration-results-v2",
        "status": "complete",
        "identity": "official Llama-2 model-card transformer_heads checkpoint",
        "method_status": "2026_arxiv_preprint",
        "full_scores_frozen_before_calibration_gold": True,
        "full_coverage": {
            "answers": 3_839,
            "fit_answers": 3_680,
            "calibration_answers": 159,
            "fit_gold_read": False,
        },
        "shared_main_table": {
            "four_bpe_window": metrics(window_labels, window_scores),
            "answer_max_shared_window": metrics(answer_labels, answer_scores),
            "threshold_rule": (
                "separate max F1, then higher precision, then higher threshold; "
                "score >= threshold"
            ),
        },
        "appendix_only": {
            "author_native_answer_max_token_at_0_6523": metrics_at_fixed_threshold(
                answer_labels, native_answer_scores, 0.6523
            ),
            "warning": (
                "This uses project calibration answer gold with author-native "
                "tokenization/aggregation/threshold; it is not the paper table."
            ),
        },
        "calibration_scored": {"answers": len(answer_ids), "windows": len(window_ids)},
        "score_freeze_file_sha256": sha256_file(FREEZE),
        "score_freeze_payload_sha256": frozen_hash,
        "adapted_scores_sha256": freeze["adapted_scores_sha256"],
        "software": {
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "test_opened": False,
        "warning": (
            "F1Opt is selected and reported on the same repeatedly used "
            "calibration partition under the shared development protocol."
        ),
    }
    RESULT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "version": "ragognizer-formal-scoring-status-v2",
        "status": "complete",
        "full_native_coverage_required": 3_839,
        "full_native_coverage_complete": 3_839,
        "fit_native_outputs": 3_680,
        "calibration_native_outputs": 159,
        "main_scoring_scope": "calibration_only_159_answers",
        "result": RESULT.name,
        "result_sha256": sha256_file(RESULT),
        "score_freeze_sha256": sha256_file(FREEZE),
        "adapted_scores_sha256": freeze["adapted_scores_sha256"],
        "substitution_used": False,
        "quantization_used": False,
        "fit_gold_opened": False,
        "test_opened": False,
    }
    STATUS.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    payload = freeze_scores() if args.stage == "freeze" else evaluate()
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
