"""Read-only development acceptance check for the RAGTruth-QA benchmark.

This program only opens the explicit fit/calibration artifacts listed below and
in ``acceptance_inputs.current.json``.  It has no test/holdout path and writes
nothing.  PASS means a componentwise point-estimate win on the repeatedly used
calibration set; it is deliberately not an independent-test claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "acceptance_inputs.current.json"

ANSWERS_FIT = ROOT / "data" / "answers_fit.jsonl"
ANSWERS_CAL = ROOT / "data" / "answers_calibration.jsonl"
WINDOWS_FIT = ROOT / "data" / "windows_k4_fit.jsonl"
WINDOWS_CAL = ROOT / "data" / "windows_k4_calibration.jsonl"
EXPANDED_ANSWERS_FIT = ROOT / "fit_expansion" / "data" / "answers_fit.jsonl"

EXPECTED = {
    "fit_answers_native": 634,
    "fit_answers_expanded": 3680,
    "calibration_answers": 159,
    "fit_windows_native": 168123,
    "calibration_windows": 42241,
    "calibration_positive_answers": 100,
    "calibration_positive_windows": 5984,
    "fit_groups": 615,
    "calibration_groups": 154,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_root(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"artifact escapes benchmark root: {relative}") from exc
    # This audit is intentionally incapable of opening any test/holdout artifact.
    forbidden = {"test", "holdout", "sealed_test", "official_test"}
    lowered = {part.lower() for part in path.relative_to(ROOT).parts}
    if lowered & forbidden:
        raise RuntimeError(f"test/holdout artifact forbidden in this gate: {relative}")
    return path


def json_pointer(value: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer}")
    node = value
    for raw in pointer.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def metric_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def f1_opt(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    s = scores[order]
    positives = int(labels.sum())
    tp = fp = 0
    best: tuple[float, float, float] | None = None
    i = 0
    while i < len(s):
        threshold = float(s[i])
        j = i
        while j < len(s) and float(s[j]) == threshold:
            if int(y[j]) == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, precision, threshold)
        if best is None or candidate > best:
            best = candidate
        i = j
    assert best is not None
    return {"f1": best[0], "precision": best[1], "threshold": best[2]}


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def verify_artifacts(artifacts: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    checks = []
    for artifact in artifacts:
        path = inside_root(artifact["path"])
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        checks.append(
            {
                "path": artifact["path"],
                "exists": exists,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual,
                "passed": exists and actual == artifact["sha256"],
            }
        )
    return checks


def audit_candidate(spec: dict[str, Any]) -> dict[str, Any]:
    answers_fit = read_jsonl(ANSWERS_FIT)
    answers_cal = read_jsonl(ANSWERS_CAL)
    windows_fit = read_jsonl(WINDOWS_FIT)
    windows_cal = read_jsonl(WINDOWS_CAL)
    expanded_fit = read_jsonl(EXPANDED_ANSWERS_FIT)

    native_fit_ids = [str(row["answer_id"]) for row in answers_fit]
    cal_ids = [str(row["answer_id"]) for row in answers_cal]
    expanded_ids = [str(row["answer_id"]) for row in expanded_fit]
    fit_groups = {str(row["group_id"]) for row in expanded_fit}
    cal_groups = {str(row["group_id"]) for row in answers_cal}
    fit_sources = {str(row["source_id"]) for row in expanded_fit}
    cal_sources = {str(row["source_id"]) for row in answers_cal}

    split_checks = {
        "native_fit_answers": len(answers_fit),
        "expanded_fit_answers": len(expanded_fit),
        "calibration_answers": len(answers_cal),
        "native_fit_windows": len(windows_fit),
        "calibration_windows": len(windows_cal),
        "fit_groups": len(fit_groups),
        "calibration_groups": len(cal_groups),
        "answer_id_overlap_expanded_fit_cal": len(set(expanded_ids) & set(cal_ids)),
        "source_id_overlap_expanded_fit_cal": len(fit_sources & cal_sources),
        "group_id_overlap_expanded_fit_cal": len(fit_groups & cal_groups),
    }
    split_pass = split_checks == {
        "native_fit_answers": EXPECTED["fit_answers_native"],
        "expanded_fit_answers": EXPECTED["fit_answers_expanded"],
        "calibration_answers": EXPECTED["calibration_answers"],
        "native_fit_windows": EXPECTED["fit_windows_native"],
        "calibration_windows": EXPECTED["calibration_windows"],
        "fit_groups": EXPECTED["fit_groups"],
        "calibration_groups": EXPECTED["calibration_groups"],
        "answer_id_overlap_expanded_fit_cal": 0,
        "source_id_overlap_expanded_fit_cal": 0,
        "group_id_overlap_expanded_fit_cal": 0,
    }

    result_path = inside_root(spec["result_path"])
    scores_path = inside_root(spec["scores_path"])
    result = read_json(result_path)
    archive = np.load(scores_path, allow_pickle=False)
    window_scores = np.asarray(archive["window_scores"], dtype=np.float64)
    answer_scores = np.asarray(archive["answer_scores"], dtype=np.float64)
    all_windows = windows_fit + windows_cal
    all_answers = answers_fit + answers_cal
    lengths_pass = len(window_scores) == len(all_windows) and len(answer_scores) == len(all_answers)
    finite_pass = bool(np.isfinite(window_scores).all() and np.isfinite(answer_scores).all())

    maxima: dict[str, float] = defaultdict(lambda: -math.inf)
    for row, score in zip(all_windows, window_scores, strict=True):
        answer_id = str(row["answer_id"])
        maxima[answer_id] = max(maxima[answer_id], float(score))
    expected_answer_scores = np.asarray([maxima[str(row["answer_id"])] for row in all_answers])
    answer_max_abs_diff = float(np.max(np.abs(expected_answer_scores - answer_scores)))

    cal_window_scores = window_scores[len(windows_fit) :]
    cal_answer_scores = answer_scores[len(answers_fit) :]
    cal_window_labels = np.asarray([int(row["label"]) for row in windows_cal], dtype=np.int8)
    cal_answer_labels = np.asarray([int(row["label"]) for row in answers_cal], dtype=np.int8)
    label_counts_pass = (
        int(cal_window_labels.sum()) == EXPECTED["calibration_positive_windows"]
        and int(cal_answer_labels.sum()) == EXPECTED["calibration_positive_answers"]
    )

    window_threshold = float(result["thresholds"]["window"]["threshold"])
    answer_threshold = float(result["thresholds"]["answer"]["threshold"])
    window_metric = metric_at_threshold(cal_window_labels, cal_window_scores, window_threshold)
    answer_metric = metric_at_threshold(cal_answer_labels, cal_answer_scores, answer_threshold)
    window_opt = f1_opt(cal_window_labels, cal_window_scores)
    answer_opt = f1_opt(cal_answer_labels, cal_answer_scores)
    metric_pass = all(
        [
            close(window_metric["f1"], result["metrics"]["calibration"]["windows"]["f1"]),
            close(answer_metric["f1"], result["metrics"]["calibration"]["answers"]["f1"]),
            close(window_opt["f1"], result["thresholds"]["window"]["f1"]),
            close(window_opt["threshold"], window_threshold),
            close(answer_opt["f1"], result["thresholds"]["answer"]["f1"]),
            close(answer_opt["threshold"], answer_threshold),
        ]
    )
    artifact_checks = verify_artifacts(spec["artifacts"])
    passed = all(
        [
            split_pass,
            lengths_pass,
            finite_pass,
            answer_max_abs_diff <= 1e-12,
            label_counts_pass,
            metric_pass,
            all(check["passed"] for check in artifact_checks),
            result["candidate"] == spec["name"],
            int(result.get("new_fits", -1)) == 0,
        ]
    )
    return {
        "passed": passed,
        "split_checks": split_checks,
        "score_lengths_and_finiteness_passed": lengths_pass and finite_pass,
        "answer_equals_max_window_max_abs_difference": answer_max_abs_diff,
        "calibration_label_counts_passed": label_counts_pass,
        "stored_thresholds_metrics_and_F1Opt_replay_passed": metric_pass,
        "window_f1": window_metric["f1"],
        "answer_f1": answer_metric["f1"],
        "window_threshold": window_threshold,
        "answer_threshold": answer_threshold,
        "artifacts": artifact_checks,
        "data_interpretation": "No fit/cal answer, source, or material-group overlap was found. Calibration reuse for model/epoch/fusion/threshold selection is documented separately and prevents an independent-test claim.",
    }


def extract_baseline(spec: dict[str, Any]) -> dict[str, Any]:
    state = spec["state"]
    record: dict[str, Any] = {
        "name": spec["name"],
        "state": state,
        "formal_primary_identity": spec["formal_primary_identity"],
        "original_method": bool(spec.get("original_method", False)),
        "unified_common_evaluation": bool(spec.get("unified_common_evaluation", False)),
        "independent_verification": bool(spec.get("independent_verification", False)),
    }
    artifacts = verify_artifacts(spec.get("artifacts", []))
    record["artifacts"] = artifacts
    record["artifact_chain_passed"] = bool(artifacts) and all(item["passed"] for item in artifacts)
    if state == "pending":
        record["reason"] = spec["reason"]
        return record

    result_path = inside_root(spec["result_path"])
    result = read_json(result_path)
    window_f1 = float(json_pointer(result, spec["window_f1_pointer"]))
    answer_f1 = float(json_pointer(result, spec["answer_f1_pointer"]))
    rows_window = int(json_pointer(result, spec["window_rows_pointer"]))
    rows_answer = int(json_pointer(result, spec["answer_rows_pointer"]))
    metrics_valid = (
        math.isfinite(window_f1)
        and math.isfinite(answer_f1)
        and 0.0 <= window_f1 <= 1.0
        and 0.0 <= answer_f1 <= 1.0
        and rows_window == EXPECTED["calibration_windows"]
        and rows_answer == EXPECTED["calibration_answers"]
    )
    record.update(
        {
            "window_f1": window_f1,
            "answer_f1": answer_f1,
            "window_rows": rows_window,
            "answer_rows": rows_answer,
            "metrics_valid": metrics_valid,
        }
    )
    if state == "eligible":
        record["eligible_passed"] = all(
            [
                record["original_method"],
                record["unified_common_evaluation"],
                record["independent_verification"],
                record["artifact_chain_passed"],
                metrics_valid,
            ]
        )
    else:
        record["reason"] = spec["reason"]
        record["eligible_passed"] = False
    return record


def run() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    ours = audit_candidate(manifest["ours"])
    baselines = [extract_baseline(spec) for spec in manifest["baselines"]]
    names = [entry["name"] for entry in baselines]
    required = manifest["required_baselines"]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    missing = sorted(set(required) - set(names))
    unresolved = [
        entry["name"]
        for entry in baselines
        if entry["name"] in required and not entry.get("eligible_passed", False)
    ]
    eligible = [entry for entry in baselines if entry.get("eligible_passed", False)]

    window_winner = max(eligible, key=lambda entry: entry["window_f1"]) if eligible else None
    answer_winner = max(eligible, key=lambda entry: entry["answer_f1"]) if eligible else None
    epsilon = float(manifest["comparison_epsilon"])
    current_comparison = None
    if window_winner is not None and answer_winner is not None:
        current_comparison = {
            "window_envelope": {
                "name": window_winner["name"],
                "f1": window_winner["window_f1"],
                "ours_minus": ours["window_f1"] - window_winner["window_f1"],
            },
            "answer_envelope": {
                "name": answer_winner["name"],
                "f1": answer_winner["answer_f1"],
                "ours_minus": ours["answer_f1"] - answer_winner["answer_f1"],
            },
            "componentwise_not_lower": (
                ours["window_f1"] + epsilon >= window_winner["window_f1"]
                and ours["answer_f1"] + epsilon >= answer_winner["answer_f1"]
            ),
        }

    if not ours["passed"] or duplicates or missing:
        status = "ERROR"
    elif unresolved:
        status = "INCOMPLETE"
    elif current_comparison and current_comparison["componentwise_not_lower"]:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "version": "acceptance-gate-v1",
        "status": status,
        "claim_scope": "Repeatedly used calibration159 development point estimates only; not statistical non-inferiority, an independent test, or universal SOTA.",
        "candidate": ours,
        "baselines": baselines,
        "required_baselines": required,
        "missing_required_baselines": missing,
        "unresolved_required_baselines": unresolved,
        "duplicate_baseline_names": duplicates,
        "current_eligible_comparison": current_comparison,
        "mechanical_rule": "After every required baseline is eligible, PASS iff ours_window_F1 + 1e-12 >= max_i baseline_i_window_F1 AND ours_answer_F1 + 1e-12 >= max_i baseline_i_answer_F1. The two maxima may come from different baselines.",
        "diagnostic_identities_excluded": [
            "ReDeEP official-code formula diagnostic",
            "RAGognizer separate-MLP route B",
            "author-native metrics at non-shared granularity",
            "local modified/fused baseline variants",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return exit code 0 for an otherwise valid INCOMPLETE snapshot.",
    )
    args = parser.parse_args()
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] == "PASS":
        return 0
    if payload["status"] == "INCOMPLETE":
        return 0 if args.allow_incomplete else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
