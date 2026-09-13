"""Gold-only arithmetic audit for a hypothetical additive conflict head.

Reads the repeatedly viewed calibration-159 incumbent and expanded-v4 fit
held-fold-0 artifacts. It never reads official test data, trains a model, or
selects a score/threshold.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "RESULTS.json"

INC_SCORE = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
INC_META = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"
CAL_ANSWERS = ROOT / "data/answers_calibration.jsonl"
CAL_TOKENS = ROOT / "data/tokens_calibration.jsonl"
CAL_WINDOWS = ROOT / "data/windows_k4_calibration.jsonl"

V4_ARRAYS = ROOT / "results/auxiliary_conflict_token_pilot_v1/arrays.npz"
V4_SUMMARY = ROOT / "results/auxiliary_conflict_token_pilot_v1/gpu_runs_v2/summary.json"
TOKENS_FIT = ROOT / "fit_expansion/data/tokens_fit.jsonl"

TYPES = (
    "Evident Conflict",
    "Subtle Conflict",
    "Evident Baseless Info",
    "Subtle Baseless Info",
)
SHORT = {
    "Evident Conflict": "EC",
    "Subtle Conflict": "SC",
    "Evident Baseless Info": "EBI",
    "Subtle Baseless Info": "SBI",
}
TARGET_NUMERATOR = 3
TARGET_DENOMINATOR = 4


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    tn = int(np.count_nonzero(~y & ~pred))
    return {
        "n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp,
        "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def f1_after(base: dict, unique_tp: int, added_fp: int) -> float:
    tp = base["tp"] + unique_tp
    fn = base["fn"] - unique_tp
    fp = base["fp"] + added_fp
    return 2 * tp / (2 * tp + fp + fn)


def target_constant(base: dict) -> int:
    # F1 >= 3/4 iff 5*x - 3*y >= A for additive unique TP x and FP y.
    denominator = 2 * base["tp"] + base["fp"] + base["fn"]
    return 3 * denominator - 8 * base["tp"]


def max_fp_for_target(a: int, unique_tp: int):
    numerator = 5 * unique_tp - a
    return math.floor(numerator / 3) if numerator >= 0 else None


def subset_budget(base: dict, total_gold: int, existing_hits: int, available_misses: int) -> dict:
    a = target_constant(base)
    minimum = max(0, math.ceil(a / 5))
    feasible = available_misses >= minimum
    y_at_min = max_fp_for_target(a, minimum) if feasible else None
    y_at_perfect = max_fp_for_target(a, available_misses)
    out = {
        "total_gold_windows": total_gold,
        "incumbent_hits": existing_hits,
        "incumbent_misses_available_as_unique_TP": available_misses,
        "minimum_unique_TP_for_F1_0.75_at_zero_FP": minimum,
        "feasible_from_this_subset": feasible,
        "minimum_required_recall_of_incumbent_misses": minimum / available_misses if feasible and available_misses else None,
        "minimum_unique_recall_points_over_all_subset_gold": minimum / total_gold if feasible and total_gold else None,
        "added_FP_tolerated_at_minimum_unique_TP": y_at_min,
        "incremental_precision_required_at_minimum_point": (
            minimum / (minimum + y_at_min) if feasible and minimum + y_at_min else None
        ),
        "added_FP_tolerated_if_every_available_miss_is_repaired": y_at_perfect,
        "incremental_precision_required_at_perfect_repair_boundary": (
            available_misses / (available_misses + y_at_perfect)
            if y_at_perfect is not None and available_misses + y_at_perfect else None
        ),
        "perfect_repair_zero_FP_f1": f1_after(base, available_misses, 0),
    }
    if feasible:
        out["gross_head_recall_range_at_minimum_unique_TP_due_to_overlap"] = {
            "if_zero_overlap_with_incumbent_hits": minimum / total_gold,
            "if_all_incumbent_hits_are_repeated": (minimum + existing_hits) / total_gold,
            "note": "Only unique TP affects the union; repeated incumbent TP can inflate standalone head recall.",
        }
    return out


def frontier(base: dict, available_misses: int, requested_recalls: tuple[float, ...]) -> list[dict]:
    a = target_constant(base)
    minimum = max(0, math.ceil(a / 5))
    xs = {available_misses, min(available_misses, minimum)}
    xs.update(min(available_misses, math.ceil(available_misses * r)) for r in requested_recalls)
    rows = []
    for x in sorted(xs):
        y = max_fp_for_target(a, x)
        rows.append({
            "unique_tp": x,
            "recall_of_available_misses": x / available_misses if available_misses else None,
            "max_added_fp_for_F1_0.75": y,
            "minimum_incremental_precision_if_target_feasible": (
                x / (x + y) if y is not None and x + y else None
            ),
            "f1_at_zero_added_fp": f1_after(base, x, 0),
            "target_feasible_at_this_recall": y is not None,
        })
    return rows


def cal_type_masks(answers: list[dict], tokens: list[dict], windows: list[dict]) -> dict[str, np.ndarray]:
    answer_by_id = {str(row["response_id"]): row for row in answers}
    token_by_id = {str(row["response_id"]): row for row in tokens}
    assert set(answer_by_id) == set(token_by_id)
    token_types = {}
    for rid, token in token_by_id.items():
        mapping = [set() for _ in range(int(token["token_count"]))]
        labels = answer_by_id[rid]["original_labels"]
        for span in token["span_token_mapping"]:
            kind = labels[int(span["span_index"])]["label_type"]
            assert kind in TYPES
            for token_index in span["risk_token_indices"]:
                mapping[int(token_index)].add(kind)
        token_types[rid] = mapping
    masks = {SHORT[kind]: np.zeros(len(windows), dtype=bool) for kind in TYPES}
    for wi, window in enumerate(windows):
        rid = str(window["response_id"])
        kinds = set()
        for token_index in window["token_indices"]:
            kinds.update(token_types[rid][int(token_index)])
        for kind in kinds:
            masks[SHORT[kind]][wi] = True
    return masks


def fit0_type_masks(answer_ids: list[str], frozen_answer_index: np.ndarray) -> dict[str, np.ndarray]:
    wanted = set(answer_ids)
    by_id = {}
    with TOKENS_FIT.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["response_id"])
            if rid in wanted:
                assert row["partition"] == "fit" and rid not in by_id
                by_id[rid] = row
    assert set(by_id) == wanted
    masks = {SHORT[kind]: np.zeros(len(frozen_answer_index), dtype=bool) for kind in TYPES}
    cursor = 0
    for ai, rid in enumerate(answer_ids):
        row = by_id[rid]
        n = int(row["token_count"])
        lexical = np.asarray(row["lexical_mask"], dtype=bool)
        token_masks = {SHORT[kind]: np.zeros(n, dtype=bool) for kind in TYPES}
        for span in row["span_token_mapping"]:
            kind = row["original_labels"][int(span["span_index"])]["label_type"]
            token_masks[SHORT[kind]][np.asarray(span["risk_token_indices"], dtype=np.int64)] = True
        eligible = [start for start in range(max(0, n - 4 + 1)) if lexical[start:start + 4].any()]
        assert len(eligible) == int(np.count_nonzero(frozen_answer_index == ai))
        for start in eligible:
            for short, token_mask in token_masks.items():
                masks[short][cursor] = token_mask[start:start + 4].any()
            cursor += 1
    assert cursor == len(frozen_answer_index)
    return masks


def answer_oracle(
    y: np.ndarray,
    pred: np.ndarray,
    answer_has_conflict: np.ndarray,
) -> dict:
    current = counts(y, pred)
    oracle_pred = pred | answer_has_conflict
    oracle = counts(y, oracle_pred)
    return {
        "current": current,
        "answers_with_gold_conflict": int(answer_has_conflict.sum()),
        "gold_conflict_answers_already_positive": int(np.count_nonzero(answer_has_conflict & pred)),
        "unique_answer_TP_added": oracle["tp"] - current["tp"],
        "answer_FP_added": oracle["fp"] - current["fp"],
        "perfect_conflict_oracle": oracle,
    }


def dataset_result(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    type_masks: dict[str, np.ndarray],
    answer_y: np.ndarray,
    answer_pred: np.ndarray,
    answer_index: np.ndarray,
) -> dict:
    base = counts(y, pred)
    ec, sc = type_masks["EC"], type_masks["SC"]
    conflict = ec | sc
    baseless = type_masks.get("EBI", np.zeros(len(y), bool)) | type_masks.get("SBI", np.zeros(len(y), bool))
    assert np.all(~conflict | y)
    miss = y & ~pred

    subsets = {}
    for label, mask in (("EC", ec), ("SC", sc), ("EC_or_SC_union", conflict), ("all_FN", y)):
        total = int(mask.sum())
        hits = int(np.count_nonzero(mask & pred))
        available = int(np.count_nonzero(mask & miss))
        subsets[label] = subset_budget(base, total, hits, available)

    oracle_pred = pred | conflict
    oracle_window = counts(y, oracle_pred)
    n_answers = len(answer_y)
    answer_has_conflict = np.zeros(n_answers, dtype=bool)
    np.logical_or.at(answer_has_conflict, answer_index, conflict)

    return {
        "name": name,
        "current_window": base,
        "target_F1": 0.75,
        "exact_integer_condition": {
            "formula": "5*unique_TP - 3*added_FP >= A",
            "A": target_constant(base),
            "minimum_unique_TP_with_zero_added_FP": math.ceil(target_constant(base) / 5),
        },
        "gold_overlap_audit": {
            "EC_windows": int(ec.sum()),
            "SC_windows": int(sc.sum()),
            "EC_SC_overlap_windows": int(np.count_nonzero(ec & sc)),
            "conflict_union_windows": int(conflict.sum()),
            "conflict_union_incumbent_hits": int(np.count_nonzero(conflict & pred)),
            "conflict_union_unique_misses": int(np.count_nonzero(conflict & miss)),
            "baseless_union_windows_if_available": int(baseless.sum()),
            "conflict_baseless_overlap_windows_if_available": int(np.count_nonzero(conflict & baseless)),
        },
        "subset_target_budgets": subsets,
        "precision_recall_frontiers": {
            "EC_or_SC_union": frontier(base, subsets["EC_or_SC_union"]["incumbent_misses_available_as_unique_TP"],
                                       (0.5, 0.6, 0.7, 0.8, 0.9)),
            "all_FN": frontier(base, base["fn"], (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        },
        "gold_oracle_conflict_only_not_a_score": {
            "unique_window_TP_added": oracle_window["tp"] - base["tp"],
            "window_FP_added": oracle_window["fp"] - base["fp"],
            "remaining_FN": oracle_window["fn"],
            "window_metrics": oracle_window,
            "baseless_only_misses_are_unchanged": True,
            "answer_impact": answer_oracle(answer_y, answer_pred, answer_has_conflict),
        },
    }


def main() -> None:
    sources = (INC_SCORE, INC_META, CAL_ANSWERS, CAL_TOKENS, CAL_WINDOWS,
               V4_ARRAYS, V4_SUMMARY, TOKENS_FIT)
    assert all(path.is_file() for path in sources)

    # Current formal calibration-159 incumbent.
    inc_meta = json.loads(INC_META.read_text(encoding="utf-8"))
    answers = load_jsonl(CAL_ANSWERS)
    tokens = load_jsonl(CAL_TOKENS)
    windows = load_jsonl(CAL_WINDOWS)
    assert len(answers) == len(tokens) == 159 and len(windows) == 42241
    y_cal = np.asarray([row["label"] for row in windows], dtype=bool)
    type_cal = cal_type_masks(answers, tokens, windows)
    assert np.array_equal(np.logical_or.reduce(list(type_cal.values())), y_cal)
    with np.load(INC_SCORE, allow_pickle=False) as z:
        inc_window_all = z["window_scores"].astype(np.float64)
        inc_answer_all = z["answer_scores"].astype(np.float64)
    inc_window = inc_window_all[-len(windows):]
    inc_answer = inc_answer_all[-len(answers):]
    inc_pred = inc_window >= float(inc_meta["thresholds"]["window"]["threshold"])
    inc_answer_pred = inc_answer >= float(inc_meta["thresholds"]["answer"]["threshold"])
    inc_answer_y = np.asarray([row["label"] for row in answers], dtype=bool)
    cal_counts = counts(y_cal, inc_pred)
    cal_answer_counts = counts(inc_answer_y, inc_answer_pred)
    assert all(cal_counts[k] == inc_meta["metrics"]["calibration"]["windows"][k]
               for k in ("n", "positive", "tp", "fp", "fn", "tn"))
    assert all(cal_answer_counts[k] == inc_meta["metrics"]["calibration"]["answers"][k]
               for k in ("n", "positive", "tp", "fp", "fn", "tn"))
    answer_local = {str(row["response_id"]): i for i, row in enumerate(answers)}
    cal_answer_index = np.asarray([answer_local[str(row["response_id"])] for row in windows], dtype=np.int32)
    cal = dataset_result("formal_cal159_incumbent", y_cal, inc_pred, type_cal,
                         inc_answer_y, inc_answer_pred, cal_answer_index)

    # Expanded-v4 fit source-connected held fold 0.
    v4_summary = json.loads(V4_SUMMARY.read_text(encoding="utf-8"))
    with np.load(V4_ARRAYS, allow_pickle=False) as z:
        v4 = z["held_v4_window_score"].astype(np.float64)
        y_v4 = z["held_window_label"].astype(bool)
        ec_v4 = z["held_window_ec"].astype(bool)
        sc_v4 = z["held_window_sc"].astype(bool)
        v4_answer_y = z["held_answer_label"].astype(bool)
        v4_answer_index = z["held_window_answer_index"].astype(np.int32)
        v4_answer_ids = [str(x) for x in z["held_answer_ids"]]
    v4_threshold = float(v4_summary["v4"]["overall"]["threshold"])
    v4_pred = v4 >= v4_threshold
    v4_answer_score = np.full(len(v4_answer_y), -np.inf)
    np.maximum.at(v4_answer_score, v4_answer_index, v4)
    v4_answer_threshold = float(v4_summary["v4"]["answer_max_window"]["threshold"])
    v4_answer_pred = v4_answer_score >= v4_answer_threshold
    assert all(counts(y_v4, v4_pred)[k] == v4_summary["v4"]["overall"][k]
               for k in ("n", "positive", "tp", "fp", "fn", "tn"))
    assert all(counts(v4_answer_y, v4_answer_pred)[k] == v4_summary["v4"]["answer_max_window"][k]
               for k in ("n", "positive", "tp", "fp", "fn", "tn"))
    type_v4 = fit0_type_masks(v4_answer_ids, v4_answer_index)
    assert np.array_equal(type_v4["EC"], ec_v4)
    assert np.array_equal(type_v4["SC"], sc_v4)
    assert np.array_equal(np.logical_or.reduce(list(type_v4.values())), y_v4)
    fit0 = dataset_result("expanded_v4_fit_held_fold0", y_v4, v4_pred, type_v4,
                          v4_answer_y, v4_answer_pred, v4_answer_index)

    output = {
        "status": "complete_gold_arithmetic_diagnosis_not_a_model_result",
        "scope": {
            "formal_calibration_answers_read": 159,
            "fit_fold0_answers_read": len(v4_answer_y),
            "official_test_opened": False,
            "models_developed_or_selected": 0,
            "thresholds_selected": 0,
            "GPU_used": False,
            "baseline_modified": False,
            "gold_oracles_are_scores": False,
        },
        "derivation": {
            "additive_union": "New head may only add alerts. x is gold-positive windows missed by the incumbent and uniquely added; y is incumbent-negative gold-negative windows added.",
            "target": "2(TP+x)/(2(TP+x)+FP+y+FN-x) >= 3/4",
            "integer_reduction": "5x-3y >= 3(2TP+FP+FN)-8TP",
            "overlap": "Head TP already hit by the incumbent does not change union F1 and is excluded from x.",
        },
        "datasets": {"cal159": cal, "fit_fold0_v4": fit0},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
