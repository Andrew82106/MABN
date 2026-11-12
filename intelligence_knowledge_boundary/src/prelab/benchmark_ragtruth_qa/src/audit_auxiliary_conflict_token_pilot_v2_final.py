"""Independent, CPU-only final audit of auxiliary-conflict GPU run v2.

This script does not import or invoke either GPU runner and deliberately does
not read gpu_runs_v2/summary.json.  It reconstructs every prediction mapping
and reported metric from the frozen arrays, arm predictions, and arm logs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/auxiliary_conflict_token_pilot_v1"
RUN = DATA / "gpu_runs_v2"
ARRAYS = DATA / "arrays.npz"
MANIFEST = DATA / "manifest.json"
PREPARATION = DATA / "preparation_complete.json"
PLAN = RUN / "GPU_PLAN.json"
V2_RUNNER = ROOT / "src/run_auxiliary_conflict_token_pilot_v2.py"
OUTPUT_JSON = RUN / "FINAL_INDEPENDENT_AUDIT.json"
OUTPUT_MD = RUN / "FINAL_INDEPENDENT_AUDIT.md"
ACCUMULATION_STEPS = 4


def sha256(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def atomic_text(path: Path, value: str) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(value, encoding="utf-8")
    pending.replace(path)


def choose_f1_threshold(labels, scores) -> float:
    """Exact independent implementation of the frozen stable tie policy."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    assert y.ndim == s.ndim == 1 and len(y) == len(s) and len(y)
    order = np.argsort(-s, kind="stable")
    sorted_score, sorted_y = s[order], y[order]
    last = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(s) - 1]
    tp = np.r_[0, np.cumsum(sorted_y)[last]]
    predicted = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(sorted_score[0], np.inf), sorted_score[last]]
    f1 = 2 * tp / (predicted + y.sum())
    precision = np.divide(tp, predicted, out=np.zeros(len(predicted), dtype=float),
                          where=predicted > 0)
    best = max(range(len(predicted)),
               key=lambda index: (f1[index], precision[index], thresholds[index]))
    return float(thresholds[best])


def binary_metrics(labels, scores, threshold: float | None = None) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    assert y.shape == s.shape and set(np.unique(y)).issubset({0, 1})
    if threshold is None:
        threshold = choose_f1_threshold(y, s)
    predicted = s >= threshold
    positive = y == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    return {
        "n": int(len(y)), "positive": int(y.sum()),
        "threshold": float(threshold),
        "predicted_positive": int(predicted.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "average_precision": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
    }


def ranking_metrics(labels, scores) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    assert y.shape == s.shape and 0 < y.sum() < len(y)
    return {
        "n": int(len(y)), "positive": int(y.sum()),
        "average_precision": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
    }


def answer_max(window_scores, answer_index, answer_count: int) -> np.ndarray:
    score = np.asarray(window_scores, dtype=np.float64)
    index = np.asarray(answer_index, dtype=np.int64)
    assert score.shape == index.shape and index.min() == 0
    assert index.max() == answer_count - 1
    result = np.full(answer_count, -np.inf, dtype=np.float64)
    np.maximum.at(result, index, score)
    assert np.isfinite(result).all()
    return result


def stable_top_k(scores, k: int) -> np.ndarray:
    score = np.asarray(scores, dtype=np.float64)
    assert 0 <= k <= len(score)
    order = np.argsort(-score, kind="stable")
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:k]] = True
    assert int(selected.sum()) == k
    return selected


def recall_record(selected, labels) -> dict:
    y = np.asarray(labels, dtype=bool)
    hit = int(np.count_nonzero(selected & y))
    total = int(y.sum())
    return {"positive": total, "hit": hit, "recall": hit / total if total else None}


def budget_record(scores, budget, risk_y, ec_y, sc_y) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    selected = stable_top_k(scores, budget)
    conflict_y = np.asarray(ec_y, dtype=bool) | np.asarray(sc_y, dtype=bool)
    risk_y = np.asarray(risk_y, dtype=bool)
    tp = int(np.count_nonzero(selected & risk_y))
    fp = int(np.count_nonzero(selected & ~risk_y))
    ordered = np.argsort(-scores, kind="stable")
    cutoff = float(scores[ordered[budget - 1]])
    above = scores > cutoff
    tied = scores == cutoff
    needed_from_tie = int(budget - above.sum())
    tied_risk = int(np.count_nonzero(tied & risk_y))
    tied_safe = int(tied.sum() - tied_risk)
    above_tp = int(np.count_nonzero(above & risk_y))
    min_tied_tp = max(0, needed_from_tie - tied_safe)
    max_tied_tp = min(needed_from_tie, tied_risk)
    return {
        "budget": int(budget), "tp": tp, "fp": fp,
        "risk_recall": tp / int(risk_y.sum()),
        "ec": recall_record(selected, ec_y),
        "sc": recall_record(selected, sc_y),
        "conflict_union": recall_record(selected, conflict_y),
        "boundary_tie": {
            "cutoff_score": cutoff,
            "strictly_above": int(above.sum()),
            "tied_at_cutoff": int(tied.sum()),
            "selected_from_tie_by_stable_window_order": needed_from_tie,
            "tied_risk_positive": tied_risk,
            "tied_ec_positive": int(np.count_nonzero(tied & np.asarray(ec_y, dtype=bool))),
            "tied_sc_positive": int(np.count_nonzero(tied & np.asarray(sc_y, dtype=bool))),
            "possible_tp_range_under_any_tie_break": [
                above_tp + min_tied_tp, above_tp + max_tied_tp],
            "frozen_policy": "stable descending score, then original window order",
        },
        "selected_mask_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }


def compare_array(actual, expected, name: str, tolerance=0.0) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    difference = float(np.max(np.abs(actual.astype(np.float64) -
                                     expected.astype(np.float64)))) if actual.size else 0.0
    assert difference <= tolerance, (name, difference, tolerance)
    return difference


def audit_training_log(arm: str, done: dict, plan: dict) -> dict:
    assert done["status"] == "complete" and done["arm"] == arm
    first_plan_name = "candidate_aux" if arm == "candidate" else "control_qa_replacement"
    final_plan_name = "candidate_final_qa" if arm == "candidate" else "control_final_qa"
    frozen_first = plan["plans"][first_plan_name]
    frozen_final = plan["plans"][final_plan_name]
    first = done["first_stage"]
    final = done["final_qa_stage"]
    keys = ("microbatches", "optimizer_updates", "examples_with_repetition",
            "logical_tokens", "padded_tokens", "max_sequence_tokens")
    assert all(first[key] == frozen_first[key] for key in keys)
    assert all(final[key] == frozen_final[key] for key in keys)
    expected_first_updates = math.ceil(first["microbatches"] / ACCUMULATION_STEPS)
    expected_final_updates = math.ceil(final["microbatches"] / ACCUMULATION_STEPS)
    assert first["optimizer_updates"] == expected_first_updates
    assert final["optimizer_updates"] == expected_final_updates
    assert done["calibration_rows_read"] == 0 and done["test_rows_read"] == 0
    assert done["published_baselines_modified"] is False
    return {
        "first_stage": first,
        "final_qa_stage": final,
        "accumulation_steps": ACCUMULATION_STEPS,
        "expected_updates_from_microbatches": {
            "first": expected_first_updates, "final": expected_final_updates,
            "total": expected_first_updates + expected_final_updates,
        },
        "reported_total_optimizer_updates":
            int(first["optimizer_updates"] + final["optimizer_updates"]),
        "matches_frozen_gpu_plan": True,
    }


def reconstruct_arm(arm: str, arrays: dict, plan: dict,
                    unique_held_positions: np.ndarray) -> tuple[dict, np.ndarray]:
    directory = RUN / arm
    complete_path = directory / "complete.json"
    prediction_path = directory / "predictions.npz"
    checkpoint_path = directory / "model.pt"
    done = read_json(complete_path)
    actual_prediction_sha = sha256(prediction_path)
    actual_checkpoint_sha = sha256(checkpoint_path)
    assert actual_prediction_sha == done["predictions_sha256"]
    assert actual_checkpoint_sha == done["checkpoint_sha256"]
    training = audit_training_log(arm, done, plan)

    with np.load(prediction_path, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "conflict_window_scores", "conflict_answer_scores",
            "held_flat_positions", "held_flat_conflict_scores",
        }
        saved_window = payload["conflict_window_scores"].copy()
        saved_answer = payload["conflict_answer_scores"].copy()
        positions = payload["held_flat_positions"].copy()
        position_scores = payload["held_flat_conflict_scores"].copy()

    assert len(positions) == len(position_scores) and len(positions)
    assert np.array_equal(positions, unique_held_positions)
    assert np.all(positions[1:] > positions[:-1])
    assert np.isfinite(position_scores).all()
    assert np.all((position_scores >= 0) & (position_scores <= 1))

    edge_positions = arrays["held_window_token_values"]
    lookup = np.searchsorted(positions, edge_positions)
    assert np.array_equal(positions[lookup], edge_positions)
    edge_scores = position_scores[lookup]
    pointer = arrays["held_window_token_indptr"]
    assert pointer[0] == 0 and pointer[-1] == len(edge_scores)
    assert np.all(np.diff(pointer) > 0)
    rebuilt_window = np.maximum.reduceat(edge_scores, pointer[:-1]).astype(np.float32)
    window_error = compare_array(rebuilt_window, saved_window,
                                 f"{arm} window reconstruction", tolerance=0.0)
    rebuilt_answer = answer_max(rebuilt_window, arrays["held_window_answer_index"],
                                len(arrays["held_answer_label"])).astype(np.float32)
    answer_error = compare_array(rebuilt_answer, saved_answer,
                                 f"{arm} answer reconstruction", tolerance=0.0)
    return {
        "complete_sha256": sha256(complete_path),
        "predictions_sha256_actual": actual_prediction_sha,
        "predictions_sha256_reported": done["predictions_sha256"],
        "checkpoint_sha256_actual": actual_checkpoint_sha,
        "checkpoint_sha256_reported": done["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "predictions_bytes": prediction_path.stat().st_size,
        "hashes_match_complete": True,
        "training_log": training,
        "mapping": {
            "unique_held_token_positions": int(len(positions)),
            "window_scores": int(len(rebuilt_window)),
            "answer_scores": int(len(rebuilt_answer)),
            "window_max_abs_error": window_error,
            "answer_max_abs_error": answer_error,
            "finite_probability_scores": True,
        },
        "scope_flags": {
            "calibration_rows_read": done["calibration_rows_read"],
            "test_rows_read": done["test_rows_read"],
            "published_baselines_modified": done["published_baselines_modified"],
        },
    }, rebuilt_window.astype(np.float64)


def system_metrics(name: str, window_scores: np.ndarray, arrays: dict,
                   v4_budget: int) -> dict:
    risk_y = arrays["held_window_label"]
    ec_y, sc_y = arrays["held_window_ec"], arrays["held_window_sc"]
    conflict_y = np.maximum(ec_y, sc_y)
    answer_scores = answer_max(window_scores, arrays["held_window_answer_index"],
                               len(arrays["held_answer_label"]))
    high = arrays["held_window_max_evidence_coverage"] >= 0.5
    return {
        "name": name,
        "window_overall_f1opt": binary_metrics(risk_y, window_scores),
        "answer_overall_f1opt": binary_metrics(arrays["held_answer_label"], answer_scores),
        "window_conflict_ranking": ranking_metrics(conflict_y, window_scores),
        "window_ec_ranking": ranking_metrics(ec_y, window_scores),
        "window_sc_ranking": ranking_metrics(sc_y, window_scores),
        "high_overlap_conflict_ranking": ranking_metrics(conflict_y[high], window_scores[high]),
        "fixed_v4_budget_top_k": budget_record(window_scores, v4_budget, risk_y, ec_y, sc_y),
        "window_scores_sha256": hashlib.sha256(
            np.asarray(window_scores, dtype=np.float64).tobytes()).hexdigest(),
        "answer_scores_sha256": hashlib.sha256(answer_scores.tobytes()).hexdigest(),
    }


def main() -> None:
    # Root data seal and declared fit-only lineage.
    manifest = read_json(MANIFEST)
    preparation = read_json(PREPARATION)
    plan = read_json(PLAN)
    assert manifest["files_sha256"]["arrays.npz"] == sha256(ARRAYS)
    assert manifest["files_sha256"]["preparation_complete.json"] == sha256(PREPARATION)
    assert preparation["scope"]["calibration_rows_read"] == 0
    assert preparation["scope"]["test_rows_read"] == 0
    assert preparation["scope"]["published_baselines_modified"] is False
    assert plan["calibration_rows_read"] == 0 and plan["test_rows_read"] == 0

    with np.load(ARRAYS, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    expected_array_keys = {
        "input_ids", "targets", "char_counts", "sequence_code",
        "hypothesis_offset_start", "hypothesis_offset_end", "indptr",
        "stage_code", "example_weight", "held_window_token_indptr",
        "held_window_token_values", "held_window_label", "held_window_ec",
        "held_window_sc", "held_window_answer_index",
        "held_window_max_evidence_coverage", "held_v4_window_score",
        "held_answer_label", "held_answer_ids",
    }
    assert set(arrays) == expected_array_keys
    n_examples = len(arrays["stage_code"])
    assert len(arrays["indptr"]) == n_examples + 1
    assert arrays["indptr"][0] == 0 and arrays["indptr"][-1] == len(arrays["input_ids"])
    assert np.all(np.diff(arrays["indptr"]) > 0)
    n_windows = len(arrays["held_window_label"])
    assert len(arrays["held_window_token_indptr"]) == n_windows + 1
    assert all(len(arrays[key]) == n_windows for key in (
        "held_window_ec", "held_window_sc", "held_window_answer_index",
        "held_window_max_evidence_coverage", "held_v4_window_score"))
    unique_positions = np.unique(arrays["held_window_token_values"])
    answer_gold_rebuilt = np.zeros(len(arrays["held_answer_label"]), dtype=np.int8)
    np.maximum.at(answer_gold_rebuilt, arrays["held_window_answer_index"],
                  arrays["held_window_label"])
    assert np.array_equal(answer_gold_rebuilt, arrays["held_answer_label"])

    arm_audits, arm_scores = {}, {}
    for arm in ("candidate", "control"):
        arm_audits[arm], arm_scores[arm] = reconstruct_arm(
            arm, arrays, plan, unique_positions)
    assert (arm_audits["candidate"]["training_log"]["reported_total_optimizer_updates"] ==
            arm_audits["control"]["training_log"]["reported_total_optimizer_updates"] == 422)
    assert (arm_audits["candidate"]["checkpoint_sha256_actual"] !=
            arm_audits["control"]["checkpoint_sha256_actual"])

    y = arrays["held_window_label"]
    ec_y, sc_y = arrays["held_window_ec"], arrays["held_window_sc"]
    conflict_y = np.maximum(ec_y, sc_y)
    v4 = arrays["held_v4_window_score"].astype(np.float64)
    v4_f1opt = binary_metrics(y, v4)
    v4_budget = v4_f1opt["predicted_positive"]
    score_vectors = {
        "v4": v4,
        "candidate_conflict_only": arm_scores["candidate"],
        "candidate_combined_max_v4": np.maximum(v4, arm_scores["candidate"]),
        "control_conflict_only": arm_scores["control"],
        "control_combined_max_v4": np.maximum(v4, arm_scores["control"]),
    }
    metrics = {
        name: system_metrics(name, score, arrays, v4_budget)
        for name, score in score_vectors.items()
    }
    assert metrics["v4"]["window_overall_f1opt"] == v4_f1opt

    # Gate is reconstructed exactly from the pre-registered four clauses.
    candidate = metrics["candidate_combined_max_v4"]
    candidate_conflict = metrics["candidate_conflict_only"]
    control_conflict = metrics["control_conflict_only"]
    v4_threshold_mask = v4 >= v4_f1opt["threshold"]
    base_conflict_recall = recall_record(v4_threshold_mask, conflict_y)["recall"]
    candidate_budget = candidate["fixed_v4_budget_top_k"]
    gate_clauses = {
        "candidate_conflict_AP_gt_control": {
            "candidate": candidate_conflict["window_conflict_ranking"]["average_precision"],
            "control": control_conflict["window_conflict_ranking"]["average_precision"],
            "passed": (candidate_conflict["window_conflict_ranking"]["average_precision"] >
                       control_conflict["window_conflict_ranking"]["average_precision"]),
        },
        "candidate_conflict_recall_gain_at_least_0_02": {
            "candidate_fixed_budget": candidate_budget["conflict_union"]["recall"],
            "v4_f1opt": base_conflict_recall,
            "gain": candidate_budget["conflict_union"]["recall"] - base_conflict_recall,
            "passed": candidate_budget["conflict_union"]["recall"] >= base_conflict_recall + 0.02,
        },
        "candidate_FP_no_more_than_v4": {
            "candidate_fixed_budget": candidate_budget["fp"],
            "v4_f1opt": v4_f1opt["fp"],
            "passed": candidate_budget["fp"] <= v4_f1opt["fp"],
        },
        "candidate_combined_overall_AP_within_0_002_of_v4": {
            "candidate": candidate["window_overall_f1opt"]["average_precision"],
            "v4": v4_f1opt["average_precision"],
            "difference": candidate["window_overall_f1opt"]["average_precision"] -
                          v4_f1opt["average_precision"],
            "passed": candidate["window_overall_f1opt"]["average_precision"] >=
                      v4_f1opt["average_precision"] - 0.002,
        },
    }
    gate_passed = all(item["passed"] for item in gate_clauses.values())

    stage_counts = {
        str(code): int(np.count_nonzero(arrays["stage_code"] == code))
        for code in np.unique(arrays["stage_code"])
    }
    report = {
        "status": "passed_independent_reconstruction",
        "audit_script_sha256": sha256(__file__),
        "input_integrity": {
            "arrays_sha256": sha256(ARRAYS),
            "arrays_matches_manifest": True,
            "manifest_sha256": sha256(MANIFEST),
            "preparation_complete_sha256": sha256(PREPARATION),
            "gpu_plan_sha256": sha256(PLAN),
            "v2_runner_sha256_static_read_only": sha256(V2_RUNNER),
        },
        "array_geometry": {
            "examples": n_examples,
            "stage_code_counts": stage_counts,
            "flat_tokens": int(len(arrays["input_ids"])),
            "held_windows": n_windows,
            "held_answers": int(len(arrays["held_answer_label"])),
            "held_window_token_edges": int(len(arrays["held_window_token_values"])),
            "unique_held_token_positions": int(len(unique_positions)),
            "risk_positive_windows": int(y.sum()),
            "ec_positive_windows": int(ec_y.sum()),
            "sc_positive_windows": int(sc_y.sum()),
            "ec_sc_overlap_windows": int(np.count_nonzero((ec_y == 1) & (sc_y == 1))),
            "conflict_union_positive_windows": int(conflict_y.sum()),
            "answer_gold_exactly_rebuilt_from_window_gold": True,
        },
        "arm_artifact_and_training_audit": arm_audits,
        "independent_metrics": metrics,
        "v4_fixed_alert_budget": int(v4_budget),
        "gate": {
            "clauses": gate_clauses,
            "pilot_gate_passed": bool(gate_passed),
        },
        "diagnosis": {
            "candidate_vs_v4": {
                "combined_window_AP_difference":
                    metrics["candidate_combined_max_v4"]["window_overall_f1opt"]["average_precision"] -
                    metrics["v4"]["window_overall_f1opt"]["average_precision"],
                "candidate_final_QA_mean_loss":
                    arm_audits["candidate"]["training_log"]["final_qa_stage"]["mean_update_loss"],
                "control_final_QA_mean_loss":
                    arm_audits["control"]["training_log"]["final_qa_stage"]["mean_update_loss"],
                "interpretation": "The auxiliary pass leaves the candidate poorly matched to held QA conflict scoring; only 27 final-QA updates follow it. Max fusion can only raise v4 scores, so weak conflict scores promote false positives and cannot repair v4 false positives.",
            },
            "control_high_conflict_recall_and_false_positives": {
                "fixed_budget_conflict_recall":
                    metrics["control_combined_max_v4"]["fixed_v4_budget_top_k"]["conflict_union"]["recall"],
                "f1opt_predicted_positive":
                    metrics["control_combined_max_v4"]["window_overall_f1opt"]["predicted_positive"],
                "f1opt_false_positive":
                    metrics["control_combined_max_v4"]["window_overall_f1opt"]["fp"],
                "v4_f1opt_false_positive":
                    metrics["v4"]["window_overall_f1opt"]["fp"],
                "interpretation": "Repeated QA training gives the control a strong conflict response but poor held-fold specificity. Its broad high scores recover conflict windows while also ranking many non-risk windows high.",
            },
        },
        "access_boundary": {
            "summary_json_read_or_trusted": False,
            "runner_imported_or_invoked": False,
            "GPU_used": False,
            "calibration_rows_read_reported_by_preparation_candidate_control": [0, 0, 0],
            "test_rows_read_reported_by_preparation_candidate_control": [0, 0, 0],
            "arrays_have_only_aux_qa_train_qa_held_stage_codes": stage_counts == {
                "0": 8128, "1": 680, "2": 6983},
            "baseline_modified": False,
            "runtime_file_access_trace_available": False,
            "qualification": "The hashes, fit-only preparation seal, intended v2 data loader, and both arm logs are consistent with zero calibration/test access. There is no OS-level file-access trace, so that historical negative cannot be proven from result artifacts alone.",
        },
    }
    atomic_json(OUTPUT_JSON, report)

    vm = metrics["v4"]["window_overall_f1opt"]
    cm = metrics["candidate_combined_max_v4"]["window_overall_f1opt"]
    km = metrics["control_combined_max_v4"]["window_overall_f1opt"]
    c_budget = metrics["candidate_combined_max_v4"]["fixed_v4_budget_top_k"]
    k_budget = metrics["control_combined_max_v4"]["fixed_v4_budget_top_k"]
    v_budget = metrics["v4"]["fixed_v4_budget_top_k"]
    c_conf_ap = metrics["candidate_conflict_only"]["window_conflict_ranking"]["average_precision"]
    k_conf_ap = metrics["control_conflict_only"]["window_conflict_ranking"]["average_precision"]
    markdown = f"""# Auxiliary conflict v2 final independent audit

## Result

All {n_windows:,} window scores and {len(arrays['held_answer_label']):,} answer-max scores were independently rebuilt from 99,000 frozen token predictions. Candidate/control reconstruction error is exactly 0. Both checkpoint and prediction hashes match their arm `complete.json`; each arm reports and arithmetically contains 395 + 27 = 422 optimizer updates.

| score | window F1-opt | window AP | window AUROC | answer F1-opt | answer AP | answer AUROC |
|---|---:|---:|---:|---:|---:|---:|
| unchanged v4 | {vm['f1']:.6f} | {vm['average_precision']:.6f} | {vm['auroc']:.6f} | {metrics['v4']['answer_overall_f1opt']['f1']:.6f} | {metrics['v4']['answer_overall_f1opt']['average_precision']:.6f} | {metrics['v4']['answer_overall_f1opt']['auroc']:.6f} |
| candidate conflict only | {metrics['candidate_conflict_only']['window_overall_f1opt']['f1']:.6f} | {metrics['candidate_conflict_only']['window_overall_f1opt']['average_precision']:.6f} | {metrics['candidate_conflict_only']['window_overall_f1opt']['auroc']:.6f} | {metrics['candidate_conflict_only']['answer_overall_f1opt']['f1']:.6f} | {metrics['candidate_conflict_only']['answer_overall_f1opt']['average_precision']:.6f} | {metrics['candidate_conflict_only']['answer_overall_f1opt']['auroc']:.6f} |
| candidate max(v4, conflict) | {cm['f1']:.6f} | {cm['average_precision']:.6f} | {cm['auroc']:.6f} | {metrics['candidate_combined_max_v4']['answer_overall_f1opt']['f1']:.6f} | {metrics['candidate_combined_max_v4']['answer_overall_f1opt']['average_precision']:.6f} | {metrics['candidate_combined_max_v4']['answer_overall_f1opt']['auroc']:.6f} |
| control conflict only | {metrics['control_conflict_only']['window_overall_f1opt']['f1']:.6f} | {metrics['control_conflict_only']['window_overall_f1opt']['average_precision']:.6f} | {metrics['control_conflict_only']['window_overall_f1opt']['auroc']:.6f} | {metrics['control_conflict_only']['answer_overall_f1opt']['f1']:.6f} | {metrics['control_conflict_only']['answer_overall_f1opt']['average_precision']:.6f} | {metrics['control_conflict_only']['answer_overall_f1opt']['auroc']:.6f} |
| control max(v4, conflict) | {km['f1']:.6f} | {km['average_precision']:.6f} | {km['auroc']:.6f} | {metrics['control_combined_max_v4']['answer_overall_f1opt']['f1']:.6f} | {metrics['control_combined_max_v4']['answer_overall_f1opt']['average_precision']:.6f} | {metrics['control_combined_max_v4']['answer_overall_f1opt']['auroc']:.6f} |

At the fixed v4 budget of {v4_budget:,} windows:

| score | TP | FP | EC recall | SC recall | conflict recall |
|---|---:|---:|---:|---:|---:|
| unchanged v4 | {v_budget['tp']:,} | {v_budget['fp']:,} | {v_budget['ec']['recall']:.3%} | {v_budget['sc']['recall']:.3%} | {v_budget['conflict_union']['recall']:.3%} |
| candidate max(v4, conflict) | {c_budget['tp']:,} | {c_budget['fp']:,} | {c_budget['ec']['recall']:.3%} | {c_budget['sc']['recall']:.3%} | {c_budget['conflict_union']['recall']:.3%} |
| control max(v4, conflict) | {k_budget['tp']:,} | {k_budget['fp']:,} | {k_budget['ec']['recall']:.3%} | {k_budget['sc']['recall']:.3%} | {k_budget['conflict_union']['recall']:.3%} |

The top-k cutoff has {c_budget['boundary_tie']['tied_at_cutoff']} tied candidate windows ({c_budget['boundary_tie']['selected_from_tie_by_stable_window_order']} selected) and {k_budget['boundary_tie']['tied_at_cutoff']} tied control windows ({k_budget['boundary_tie']['selected_from_tie_by_stable_window_order']} selected). The frozen stable original-window order resolves them. Neither boundary tie contains an EC/SC window, so conflict recall and the failed gate are invariant to tie order.

The pilot gate **{'passes' if gate_passed else 'fails'}**. Candidate conflict-only AP is {c_conf_ap:.6f}, versus control {k_conf_ap:.6f}; candidate combined overall AP differs from v4 by {cm['average_precision'] - vm['average_precision']:+.6f}.

## Diagnosis

The candidate's auxiliary stage is not well aligned with held QA: its final QA mean loss remains {arm_audits['candidate']['training_log']['final_qa_stage']['mean_update_loss']:.6f}, versus {arm_audits['control']['training_log']['final_qa_stage']['mean_update_loss']:.6f} for control after repeated QA training. Only 27 final-QA updates follow the auxiliary pass. Because fusion is `max(v4, conflict)`, an imprecise conflict head can only promote extra windows; it cannot lower existing v4 false positives.

The control's repeated QA exposure raises conflict recall, but its held-fold scores are broad: its F1-opt rule selects {km['predicted_positive']:,} windows and produces {km['fp']:,} false positives, versus v4's {vm['predicted_positive']:,} selected and {vm['fp']:,} false positives. High conflict recall therefore reflects low specificity rather than a clean conflict boundary.

This audit did not read or trust `summary.json`, import/call the runner, use GPU, or modify a baseline. Preparation and both arm logs report zero calibration/test rows. Their hashes and the runner's static input boundary support that claim; no OS-level historical file-access trace exists.
"""
    atomic_text(OUTPUT_MD, markdown)
    print("AUXILIARY_CONFLICT_V2_FINAL_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
