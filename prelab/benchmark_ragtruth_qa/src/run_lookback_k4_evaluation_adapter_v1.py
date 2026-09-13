"""Map frozen Lookback Lens 8-BPE span scores to the project's 4-BPE interface.

This is an evaluation adapter. It never loads, trains, refits, or modifies the
Lookback classifier and never opens the sealed 150-answer test split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import run_development as q


ROOT = q.ROOT
DATA = ROOT / "data"
BASE = ROOT / "results/lookback_official_span_v1"
OUT = ROOT / "results/lookback_k4_evaluation_adapter_v1"
BASE_FIT_ANSWERS = 3680
CAL_ANSWERS = 159
CAL_WINDOWS = 42241
BASE_K = 8
PROJECT_K = 4
ROLE = "evaluation_adapter_for_frozen_lookback_baseline; no model change"


def protocol():
    return {
        "version": "lookback-frozen-score-to-project-k4-v1",
        "role": ROLE,
        "frozen_baseline": {
            "source": "lookback_official_span_v1",
            "model": "Existing lookback_lr.pkl is never loaded, refit, or rewritten.",
            "scores": "Read the immutable scores.npz window_scores and span_geometry.npz only.",
            "native_interface": "Original complete 8-raw-BPE stride-one span results remain unchanged and are copied into the new report only for traceability.",
        },
        "scope": "Exactly all 159 calibration answers and their 42241 existing eligible project 4-BPE windows. No project fit data and no sealed test150 data are opened by the adapter.",
        "mapping_fixed_before_labels": {
            "token": "For each original raw answer token, arithmetic mean of every frozen 8-BPE span score whose [token_start,token_end) covers that token.",
            "window4": "Arithmetic mean of the four raw-token scores at the existing window token_indices. Punctuation slots remain; only the project's already-frozen no-lexical window exclusion applies. Token offsets are independently merged before checking the stored character_intervals field.",
            "answer": "Maximum over every eligible project 4-BPE window in that answer.",
            "labels": "No answer/window label, risk mask, lexical mask, span coordinate, or metric enters mapping. Labels are opened only after adapted scores are frozen.",
            "coverage": "Every calibration answer has at least one native 8-BPE span; every raw token therefore has one to eight covering predictions and a unique arithmetic mean.",
        },
        "evaluation": {
            "window": "Calibration AUROC, average precision, and calibration F1Opt with its own calibration-selected threshold.",
            "answer": "Calibration AUROC, average precision, and calibration F1Opt with a separate calibration-selected threshold.",
            "warning": "F1Opt is optimized and evaluated on the same repeatedly-used development calibration set. It is the common project comparison rule, not held-out performance.",
        },
        "prohibitions": [
            "No baseline retraining, parameter edit, score transformation learned from labels, smoothing, fusion, or threshold reuse from another method.",
            "No change to lookback_official_span_v1 or any baseline artifact.",
            "No official test150 read or claim.",
        ],
        "integrity": "Protocol and source hashes precede mapping; existing complete output is hash-verified and never overwritten; a separate implementation must recompute mapping within fixed float64 absolute tolerance 5e-16 and reproduce thresholds/metrics exactly.",
    }


def visible_calibration():
    answer_keys = ("response_id", "source_id", "group_id", "partition", "answer_id",
                   "answer_sha256")
    token_keys = ("response_id", "source_id", "group_id", "partition", "answer_id",
                  "answer_sha256", "token_count", "token_ids", "answer_token_positions",
                  "response_token_offsets", "response_token_offsets_raw")
    window_keys = ("response_id", "source_id", "group_id", "partition", "answer_id",
                   "window_id", "token_start", "token_end", "token_indices", "token_ids",
                   "answer_token_positions", "character_intervals")
    raw_answers = q.lines(DATA / "answers_calibration.jsonl")
    raw_tokens = q.lines(DATA / "tokens_calibration.jsonl")
    raw_windows = q.lines(DATA / "windows_k4_calibration.jsonl")
    answers = [{key: row[key] for key in answer_keys} for row in raw_answers]
    tokens = [{key: row[key] for key in token_keys} for row in raw_tokens]
    windows = [{key: row[key] for key in window_keys} for row in raw_windows]
    assert len(answers) == len(tokens) == CAL_ANSWERS and len(windows) == CAL_WINDOWS
    assert [row["response_id"] for row in answers] == [row["response_id"] for row in tokens]
    assert all(row["partition"] == "calibration" for row in answers + tokens + windows)
    return answers, tokens, windows


def baseline_integrity():
    complete = q.read(BASE / "complete.json")
    assert complete["status"] == "complete" and complete["real_fits"] == 1
    assert complete["no_test"] and not complete["GPU_used"]
    for name, expected in complete["files_sha256"].items():
        assert q.sha(BASE / name) == expected, name
    prepared = q.read(BASE / "preparation_complete.json")
    assert prepared["status"] == "CPU_ready_not_fitted" and prepared["cal_scored_answers"] == 159
    for name in ("protocol.json", "span_geometry.npz", "answer_order.json"):
        assert q.sha(BASE / name) == prepared["files_sha256"][name], name
    summary = q.read(BASE / "summary.json")
    assert summary["scores_sha256"] == q.sha(BASE / "scores.npz")
    assert summary["model_sha256"] == q.sha(BASE / "lookback_lr.pkl")
    assert summary["real_fits"] == 1 and summary["no_test"]
    return complete, prepared, summary


def source_files():
    paths = [
        Path(__file__), ROOT / "src/verify_lookback_k4_evaluation_adapter_v1.py",
        OUT / "protocol.json", BASE / "complete.json", BASE / "preparation_complete.json",
        BASE / "protocol.json", BASE / "summary.json", BASE / "scores.npz",
        BASE / "span_geometry.npz", BASE / "answer_order.json", BASE / "lookback_lr.pkl",
        DATA / "gold_manifest.json", DATA / "answers_calibration.jsonl",
        DATA / "tokens_calibration.jsonl", DATA / "windows_k4_calibration.jsonl",
        OUT / "FAILURE_PREPARE_INTERFACE_ASSERTION.json",
        OUT / "FAILURE_INDEPENDENT_FLOAT_TOLERANCE.json",
    ]
    return {str(path.resolve()): q.sha(path) for path in paths}


def token_projection(token_count, starts, ends, scores):
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    assert len(starts) == len(ends) == len(scores) == token_count - BASE_K + 1
    assert np.array_equal(starts, np.arange(len(starts)))
    assert np.array_equal(ends, starts + BASE_K)
    owners = [[] for _ in range(token_count)]
    for score, left, right in zip(scores, starts, ends):
        for token_index in range(int(left), int(right)):
            owners[token_index].append(float(score))
    counts = np.asarray([len(values) for values in owners], dtype=np.int16)
    assert np.all((counts >= 1) & (counts <= BASE_K))
    projected = np.asarray([np.mean(values, dtype=np.float64) for values in owners], dtype=np.float64)
    assert np.isfinite(projected).all()
    return projected, counts


def merge_intervals(intervals):
    result = []
    for left, right in sorted([list(map(int, item)) for item in intervals]):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def tiny_check():
    scores = np.asarray([2.0, 4.0, 8.0])
    token, coverage = token_projection(10, [0, 1, 2], [8, 9, 10], scores)
    expected = np.asarray([2, 3, 14 / 3, 14 / 3, 14 / 3, 14 / 3, 14 / 3, 14 / 3, 6, 8])
    assert np.allclose(token, expected, rtol=0, atol=0)
    assert coverage.tolist() == [1, 2, 3, 3, 3, 3, 3, 3, 2, 1]
    four = np.asarray([token[index:index + PROJECT_K].mean() for index in range(7)])
    assert four.shape == (7,) and float(four.max()) == float(four[-1])
    return {
        "status": "passed",
        "role": ROLE,
        "token_mean_all_covering_native_spans": True,
        "four_token_arithmetic_mean": True,
        "answer_max": True,
        "labels_accessed": False,
        "baseline_model_loaded": False,
        "real_fits": 0,
        "official_test_opened": False,
    }


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "protocol.json").exists(), "Protocol already exists"
    q.save(OUT / "protocol.json", protocol())
    q.save(OUT / "CPU_SELFCHECK.json", tiny_check())
    print("LOOKBACK_K4_ADAPTER_PROTOCOL_FROZEN", flush=True)


def prepare():
    assert q.read(OUT / "protocol.json") == protocol()
    assert not (OUT / "prepare_started.json").exists(), "No silent preparation overwrite"
    baseline_complete, baseline_prepared, baseline_summary = baseline_integrity()
    answers, tokens, windows = visible_calibration()
    snapshot = source_files()
    q.save(OUT / "prepare_started.json", {
        "role": ROLE,
        "source_files_sha256": snapshot,
        "labels_accessed": False,
        "baseline_model_loaded": False,
        "official_test_opened": False,
    })
    base_ledger = q.read(BASE / "answer_order.json")
    with np.load(BASE / "span_geometry.npz", allow_pickle=False) as source:
        geometry = {name: source[name].copy() for name in ("answer_index", "token_start", "token_end", "answer_window_offsets")}
    with np.load(BASE / "scores.npz", allow_pickle=False) as source:
        native_scores = source["window_scores"].astype(np.float64)
    assert len(native_scores) == 681640 and geometry["answer_window_offsets"].shape == (3839, 2)
    assert [row["response_id"] for row in base_ledger[BASE_FIT_ANSWERS:]] == [row["response_id"] for row in answers]
    window_positions = defaultdict(list)
    for global_index, window in enumerate(windows):
        window_positions[window["response_id"]].append(global_index)
    token_values = []
    answer_token_offsets = np.empty((CAL_ANSWERS, 2), dtype=np.int64)
    adapted_windows = np.empty(CAL_WINDOWS, dtype=np.float64)
    adapted_answers = np.empty(CAL_ANSWERS, dtype=np.float64)
    mapping_ledger = []
    token_cursor = 0
    all_coverages = []
    for local_answer, (answer, token) in enumerate(zip(answers, tokens)):
        response_id = answer["response_id"]
        assert response_id == token["response_id"] and answer["answer_sha256"] == token["answer_sha256"]
        base_answer = BASE_FIT_ANSWERS + local_answer
        native_left, native_right = map(int, geometry["answer_window_offsets"][base_answer])
        assert native_right > native_left
        assert np.all(geometry["answer_index"][native_left:native_right] == base_answer)
        starts = geometry["token_start"][native_left:native_right]
        ends = geometry["token_end"][native_left:native_right]
        projected, coverage = token_projection(token["token_count"], starts, ends,
                                               native_scores[native_left:native_right])
        answer_token_offsets[local_answer] = [token_cursor, token_cursor + len(projected)]
        token_values.append(projected)
        token_cursor += len(projected)
        all_coverages.extend(coverage.tolist())
        positions = window_positions[response_id]
        assert positions
        for window_index in positions:
            window = windows[window_index]
            indices = window["token_indices"]
            assert len(indices) == PROJECT_K and indices == list(range(window["token_start"], window["token_end"]))
            assert window["token_ids"] == [token["token_ids"][j] for j in indices]
            assert window["answer_token_positions"] == [token["answer_token_positions"][j] for j in indices]
            assert window["character_intervals"] == merge_intervals(
                [token["response_token_offsets"][j] for j in indices])
            adapted_windows[window_index] = projected[indices].mean(dtype=np.float64)
        adapted_answers[local_answer] = adapted_windows[positions].max()
        mapping_ledger.append({
            "response_id": response_id,
            "answer_id": answer["answer_id"],
            "raw_tokens": token["token_count"],
            "native8_span_left": native_left,
            "native8_span_right": native_right,
            "native8_spans": native_right - native_left,
            "token_score_left": int(answer_token_offsets[local_answer, 0]),
            "token_score_right": int(answer_token_offsets[local_answer, 1]),
            "project4_window_indices": positions,
            "project4_windows": len(positions),
            "minimum_native_span_coverage_per_token": int(coverage.min()),
            "maximum_native_span_coverage_per_token": int(coverage.max()),
        })
    token_values = np.concatenate(token_values)
    assert len(token_values) == sum(row["token_count"] for row in tokens) == 42798
    assert np.isfinite(adapted_windows).all() and np.isfinite(adapted_answers).all()
    assert sorted(index for positions in window_positions.values() for index in positions) == list(range(CAL_WINDOWS))
    mapping_digest = q.digest(mapping_ledger)
    protocol_sha = q.sha(OUT / "protocol.json")
    baseline_score_sha = q.sha(BASE / "scores.npz")
    np.savez_compressed(
        OUT / "adapted_scores.npz",
        method_role=np.asarray(ROLE),
        token_scores=token_values,
        answer_token_offsets=answer_token_offsets,
        window_scores=adapted_windows,
        answer_scores=adapted_answers,
        window_ids=np.asarray([row["window_id"] for row in windows]),
        answer_ids=np.asarray([row["answer_id"] for row in answers]),
        mapping_ledger_sha256=np.asarray(mapping_digest),
        mapping_protocol_sha256=np.asarray(protocol_sha),
        frozen_baseline_scores_sha256=np.asarray(baseline_score_sha),
    )
    q.save(OUT / "mapping_ledger.json", mapping_ledger)
    q.save(OUT / "source_snapshot.json", {
        "role": ROLE,
        "files_sha256": snapshot,
        "frozen_baseline_complete_sha256": q.sha(BASE / "complete.json"),
        "frozen_baseline_scores_sha256": baseline_score_sha,
        "frozen_baseline_model_sha256": q.sha(BASE / "lookback_lr.pkl"),
        "official_test_opened": False,
    })
    assert snapshot == source_files()
    q.save(OUT / "preparation_complete.json", {
        "status": "complete_label_free_mapping_not_scored",
        "role": ROLE,
        "calibration_answers": len(answers),
        "calibration_raw_tokens": len(token_values),
        "calibration_project4_windows": len(windows),
        "all_answers_scorable": True,
        "native_span_coverage_minimum": min(all_coverages),
        "native_span_coverage_maximum": max(all_coverages),
        "labels_accessed": False,
        "baseline_model_loaded": False,
        "real_fits": 0,
        "adapted_scores_sha256": q.sha(OUT / "adapted_scores.npz"),
        "mapping_ledger_sha256": q.sha(OUT / "mapping_ledger.json"),
        "protocol_sha256": protocol_sha,
        "source_snapshot_sha256": q.sha(OUT / "source_snapshot.json"),
        "native8_results_preserved": baseline_summary["metrics"]["calibration"],
        "old_baseline_files_verified": sorted(baseline_complete["files_sha256"]),
        "old_baseline_geometry_verified": sorted(name for name in baseline_prepared["files_sha256"] if name in ("protocol.json", "span_geometry.npz", "answer_order.json")),
        "official_test_opened": False,
    })
    print("LOOKBACK_K4_ADAPTER_LABEL_FREE_MAPPING_COMPLETE", flush=True)


def check_prepared():
    assert q.read(OUT / "protocol.json") == protocol()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "complete_label_free_mapping_not_scored"
    assert complete["calibration_answers"] == CAL_ANSWERS
    assert complete["calibration_project4_windows"] == CAL_WINDOWS
    assert q.sha(OUT / "adapted_scores.npz") == complete["adapted_scores_sha256"]
    assert q.sha(OUT / "mapping_ledger.json") == complete["mapping_ledger_sha256"]
    assert q.sha(OUT / "protocol.json") == complete["protocol_sha256"]
    assert q.sha(OUT / "source_snapshot.json") == complete["source_snapshot_sha256"]
    snapshot = q.read(OUT / "source_snapshot.json")
    assert snapshot["files_sha256"] == source_files()
    baseline_integrity()
    return complete


def verify_existing_complete_and_refuse():
    complete = q.read(OUT / "complete.json")
    assert complete["status"] == "complete_calibration_F1Opt_development_only"
    for name, expected in complete["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    assert complete["adapted_scores_sha256"] == q.sha(OUT / "adapted_scores.npz")
    raise RuntimeError("Existing complete adapter result verified; refusing overwrite")


def score():
    if (OUT / "complete.json").exists():
        verify_existing_complete_and_refuse()
    outputs = ("calibration_f1opt_thresholds.json", "summary.json", "REPORT.md")
    assert not any((OUT / name).exists() for name in outputs), "Partial scoring output exists; refusing overwrite"
    check_prepared()
    raw_answers = q.lines(DATA / "answers_calibration.jsonl")
    raw_windows = q.lines(DATA / "windows_k4_calibration.jsonl")
    assert len(raw_answers) == CAL_ANSWERS and len(raw_windows) == CAL_WINDOWS
    with np.load(OUT / "adapted_scores.npz", allow_pickle=False) as saved:
        assert saved["method_role"].item() == ROLE
        assert saved["window_ids"].tolist() == [row["window_id"] for row in raw_windows]
        assert saved["answer_ids"].tolist() == [row["answer_id"] for row in raw_answers]
        window_scores = saved["window_scores"].copy()
        answer_scores = saved["answer_scores"].copy()
    window_labels = np.asarray([row["label"] for row in raw_windows], dtype=np.int8)
    answer_labels = np.asarray([row["label"] for row in raw_answers], dtype=np.int8)
    assert set(window_labels) == {0, 1} and set(answer_labels) == {0, 1}
    thresholds = {
        "window4": q.choose_threshold(window_labels, window_scores),
        "answer": q.choose_threshold(answer_labels, answer_scores),
    }
    q.save(OUT / "calibration_f1opt_thresholds.json", {
        "role": ROLE,
        "thresholds": thresholds,
        "selection_and_evaluation_split": "calibration",
        "development_optimistic": True,
        "real_fits": 0,
        "official_test_opened": False,
    })
    adapted_metrics = {
        "windows4": q.count(window_labels, window_scores, thresholds["window4"]["threshold"]),
        "answers": q.count(answer_labels, answer_scores, thresholds["answer"]["threshold"]),
    }
    native = q.read(BASE / "summary.json")["metrics"]["calibration"]
    summary = {
        "status": "complete_calibration_F1Opt_development_only",
        "role": ROLE,
        "mapping": protocol()["mapping_fixed_before_labels"],
        "adapted_project4_calibration": adapted_metrics,
        "thresholds_calibration_F1Opt": thresholds,
        "native8_result_preserved_unchanged": native,
        "frozen_baseline_scores_sha256": q.sha(BASE / "scores.npz"),
        "frozen_baseline_model_sha256": q.sha(BASE / "lookback_lr.pkl"),
        "adapted_scores_sha256": q.sha(OUT / "adapted_scores.npz"),
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "calibration_used_for_threshold": True,
        "official_test_opened": False,
        "final_test_claim": False,
    }
    q.save(OUT / "summary.json", summary)
    rows = [
        "# Lookback Lens：统一4-BPE评测适配",
        "",
        "只读取冻结的8-BPE预测及几何，没有加载、重训或改写基线模型。映射在读取标签前完成。",
        "",
        "| 接口 | 数量 | AUROC | AP | F1 | 阈值来源 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    native_window = native["spans8"]
    native_answer = native["answers_extra_adaptation"]
    adapted_window = adapted_metrics["windows4"]
    adapted_answer = adapted_metrics["answers"]
    rows += [
        f"| 作者结构8-BPE span | {native_window['n']} | {native_window['auroc']:.6f} | {native_window['average_precision']:.6f} | {native_window['f1']:.6f} | 原结果fit-only阈值 |",
        f"| 统一4-BPE窗口 | {adapted_window['n']} | {adapted_window['auroc']:.6f} | {adapted_window['average_precision']:.6f} | {adapted_window['f1']:.6f} | cal F1Opt={thresholds['window4']['threshold']:.9f} |",
        f"| 原8-BPE整答附加接口 | {native_answer['n']} | {native_answer['auroc']:.6f} | {native_answer['average_precision']:.6f} | {native_answer['f1']:.6f} | 原结果fit-only阈值 |",
        f"| 统一4-BPE整答 | {adapted_answer['n']} | {adapted_answer['auroc']:.6f} | {adapted_answer['average_precision']:.6f} | {adapted_answer['f1']:.6f} | cal F1Opt={thresholds['answer']['threshold']:.9f} |",
        "",
        "统一映射：raw token取所有覆盖它的原8-BPE分数均值；4-BPE窗取四个token均值；整答取全部4窗最大值。159答全部可评分。",
        "F1Opt的阈值与分数来自同一个calibration开发集，因此只用于共同口径比较；AUROC/AP不依赖阈值。原8-BPE结果仅原样保留，两个F1的阈值来源不同。",
        "sealed test150未读取；旧Lookback目录及模型、分数均未改动。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    names = ("calibration_f1opt_thresholds.json", "summary.json", "REPORT.md")
    q.save(OUT / "complete.json", {
        "status": "complete_calibration_F1Opt_development_only",
        "role": ROLE,
        "files_sha256": {name: q.sha(OUT / name) for name in names},
        "adapted_scores_sha256": q.sha(OUT / "adapted_scores.npz"),
        "preparation_complete_sha256": q.sha(OUT / "preparation_complete.json"),
        "frozen_baseline_complete_sha256": q.sha(BASE / "complete.json"),
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "official_test_opened": False,
        "final_test_claim": False,
    })
    print("LOOKBACK_K4_ADAPTER_SCORING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "score"))
    arguments = parser.parse_args()
    if arguments.stage == "initialize":
        initialize()
    elif arguments.stage == "self-test":
        print(json.dumps(tiny_check(), ensure_ascii=False), flush=True)
    elif arguments.stage == "prepare":
        prepare()
    elif arguments.stage == "check":
        check_prepared()
        print("LOOKBACK_K4_ADAPTER_PREPARATION_CHECK_PASSED", flush=True)
    else:
        score()
