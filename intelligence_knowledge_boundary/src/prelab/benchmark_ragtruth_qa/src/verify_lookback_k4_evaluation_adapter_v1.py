"""Independent recomputation of frozen Lookback 8-BPE -> project 4-BPE scores."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import run_development as q


ROOT = q.ROOT
DATA = ROOT / "data"
BASE = ROOT / "results/lookback_official_span_v1"
OUT = ROOT / "results/lookback_k4_evaluation_adapter_v1"
ROLE = "evaluation_adapter_for_frozen_lookback_baseline; no model change"
BASE_FIT_ANSWERS = 3680


def independent_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    last = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    true_positive = np.r_[0, np.cumsum(sorted_labels)[last]]
    predicted_positive = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[last]]
    f1 = 2 * true_positive / (predicted_positive + labels.sum())
    precision = np.divide(true_positive, predicted_positive, out=np.zeros(len(true_positive), float),
                          where=predicted_positive > 0)
    best = max(range(len(thresholds)), key=lambda j: (f1[j], precision[j], thresholds[j]))
    return {"threshold": float(thresholds[best]), "f1": float(f1[best]),
            "precision": float(precision[best]), "rows": len(labels),
            "positive": int(labels.sum())}


def independent_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    tp = int(np.count_nonzero((labels == 1) & predictions))
    fp = int(np.count_nonzero((labels == 0) & predictions))
    fn = int(np.count_nonzero((labels == 1) & ~predictions))
    tn = int(np.count_nonzero((labels == 0) & ~predictions))
    return {"n": len(labels), "positive": int(labels.sum()), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def independent_token_projection(token_count, starts, ends, span_scores):
    token_axis = np.arange(token_count)[None, :]
    starts = np.asarray(starts, dtype=np.int64)[:, None]
    ends = np.asarray(ends, dtype=np.int64)[:, None]
    span_scores = np.asarray(span_scores, dtype=np.float64)
    coverage = (token_axis >= starts) & (token_axis < ends)
    counts = coverage.sum(0)
    assert np.all((counts >= 1) & (counts <= 8))
    values = (coverage.astype(np.float64) * span_scores[:, None]).sum(0) / counts
    return values, counts


def independent_merge_intervals(intervals):
    ordered = sorted(tuple(map(int, item)) for item in intervals if int(item[1]) > int(item[0]))
    merged = []
    for left, right in ordered:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return [list(item) for item in merged]


def main():
    output = OUT / "INDEPENDENT_RECOMPUTE.json"
    assert not output.exists(), "No silent independent-verifier overwrite"
    protocol = q.read(OUT / "protocol.json")
    assert protocol["version"] == "lookback-frozen-score-to-project-k4-v1"
    assert protocol["role"] == ROLE
    snapshot = q.read(OUT / "source_snapshot.json")
    assert snapshot["role"] == ROLE
    for path, expected in snapshot["files_sha256"].items():
        assert q.sha(path) == expected, path
    base_complete = q.read(BASE / "complete.json")
    for name, expected in base_complete["files_sha256"].items():
        assert q.sha(BASE / name) == expected, name
    assert q.sha(BASE / "complete.json") == snapshot["frozen_baseline_complete_sha256"]
    assert q.sha(BASE / "scores.npz") == snapshot["frozen_baseline_scores_sha256"]
    assert q.sha(BASE / "lookback_lr.pkl") == snapshot["frozen_baseline_model_sha256"]
    answers = q.lines(DATA / "answers_calibration.jsonl")
    tokens = q.lines(DATA / "tokens_calibration.jsonl")
    windows = q.lines(DATA / "windows_k4_calibration.jsonl")
    assert len(answers) == len(tokens) == 159 and len(windows) == 42241
    base_ledger = q.read(BASE / "answer_order.json")
    with np.load(BASE / "span_geometry.npz", allow_pickle=False) as source:
        offsets = source["answer_window_offsets"].copy()
        answer_index = source["answer_index"].copy()
        starts = source["token_start"].copy()
        ends = source["token_end"].copy()
    with np.load(BASE / "scores.npz", allow_pickle=False) as source:
        native_scores = source["window_scores"].astype(np.float64)
    by_answer = defaultdict(list)
    for window_index, window in enumerate(windows):
        by_answer[window["response_id"]].append(window_index)
    recomputed_tokens = []
    recomputed_offsets = np.empty((159, 2), dtype=np.int64)
    recomputed_windows = np.empty(42241, dtype=np.float64)
    recomputed_answers = np.empty(159, dtype=np.float64)
    cursor = 0
    for local_answer, (answer, token) in enumerate(zip(answers, tokens)):
        response_id = answer["response_id"]
        assert response_id == token["response_id"] == base_ledger[BASE_FIT_ANSWERS + local_answer]["response_id"]
        lo, hi = map(int, offsets[BASE_FIT_ANSWERS + local_answer])
        assert np.all(answer_index[lo:hi] == BASE_FIT_ANSWERS + local_answer)
        value, coverage = independent_token_projection(token["token_count"], starts[lo:hi], ends[lo:hi],
                                                       native_scores[lo:hi])
        assert coverage.min() == 1 and coverage.max() <= 8
        recomputed_offsets[local_answer] = [cursor, cursor + len(value)]
        recomputed_tokens.append(value)
        cursor += len(value)
        positions = by_answer[response_id]
        assert positions
        for window_index in positions:
            window = windows[window_index]
            indices = window["token_indices"]
            assert len(indices) == 4 and indices == list(range(window["token_start"], window["token_end"]))
            assert window["token_ids"] == [token["token_ids"][j] for j in indices]
            assert window["answer_token_positions"] == [token["answer_token_positions"][j] for j in indices]
            assert window["character_intervals"] == independent_merge_intervals(
                [token["response_token_offsets"][j] for j in indices])
            recomputed_windows[window_index] = value[indices].sum(dtype=np.float64) / 4.0
        recomputed_answers[local_answer] = recomputed_windows[positions].max()
    recomputed_tokens = np.concatenate(recomputed_tokens)
    with np.load(OUT / "adapted_scores.npz", allow_pickle=False) as saved:
        assert saved["method_role"].item() == ROLE
        assert saved["window_ids"].tolist() == [row["window_id"] for row in windows]
        assert saved["answer_ids"].tolist() == [row["answer_id"] for row in answers]
        saved_tokens = saved["token_scores"].copy()
        saved_offsets = saved["answer_token_offsets"].copy()
        saved_windows = saved["window_scores"].copy()
        saved_answers = saved["answer_scores"].copy()
        assert saved["mapping_protocol_sha256"].item() == q.sha(OUT / "protocol.json")
        assert saved["frozen_baseline_scores_sha256"].item() == q.sha(BASE / "scores.npz")
        assert saved["mapping_ledger_sha256"].item() == q.digest(q.read(OUT / "mapping_ledger.json"))
    token_difference = float(np.max(np.abs(saved_tokens - recomputed_tokens)))
    window_difference = float(np.max(np.abs(saved_windows - recomputed_windows)))
    answer_difference = float(np.max(np.abs(saved_answers - recomputed_answers)))
    assert np.array_equal(saved_offsets, recomputed_offsets)
    assert token_difference <= 5e-16 and window_difference <= 5e-16 and answer_difference <= 5e-16
    window_labels = np.asarray([row["label"] for row in windows], dtype=np.int8)
    answer_labels = np.asarray([row["label"] for row in answers], dtype=np.int8)
    thresholds = {
        "window4": independent_threshold(window_labels, recomputed_windows),
        "answer": independent_threshold(answer_labels, recomputed_answers),
    }
    metrics = {
        "windows4": independent_metrics(window_labels, recomputed_windows, thresholds["window4"]["threshold"]),
        "answers": independent_metrics(answer_labels, recomputed_answers, thresholds["answer"]["threshold"]),
    }
    summary = q.read(OUT / "summary.json")
    assert thresholds == summary["thresholds_calibration_F1Opt"]
    assert metrics == summary["adapted_project4_calibration"]
    assert summary["native8_result_preserved_unchanged"] == q.read(BASE / "summary.json")["metrics"]["calibration"]
    complete = q.read(OUT / "complete.json")
    for name, expected in complete["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    report = {
        "status": "independent_recomputation_passed",
        "role": ROLE,
        "answers_recomputed": 159,
        "raw_tokens_recomputed": len(recomputed_tokens),
        "project4_windows_recomputed": len(recomputed_windows),
        "maximum_absolute_difference": {
            "token_scores": token_difference,
            "window_scores": window_difference,
            "answer_scores": answer_difference,
        },
        "fixed_float64_absolute_tolerance": 5e-16,
        "thresholds_exact": True,
        "metrics_exact": True,
        "frozen_baseline_files_unchanged": True,
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "verifier_sha256": q.sha(__file__),
        "runner_sha256": q.sha(ROOT / "src/run_lookback_k4_evaluation_adapter_v1.py"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "adapted_scores_sha256": q.sha(OUT / "adapted_scores.npz"),
        "official_test_opened": False,
        "final_test_claim": False,
    }
    q.save(output, report)
    print("LOOKBACK_K4_ADAPTER_INDEPENDENT_RECOMPUTATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
