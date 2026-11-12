"""Read-only, fit-fold-0 diagnosis of auxiliary conflict v2.

This script reads only frozen fit artifacts. It does not train, tune, load model
weights, use a GPU, or touch calibration/test data.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / "results/auxiliary_conflict_token_pilot_v1"
RUN = PILOT / "gpu_runs_v2"
ARRAYS = PILOT / "arrays.npz"
CANDIDATE = RUN / "candidate/predictions.npz"
CONTROL = RUN / "control/predictions.npz"
SUMMARY = RUN / "summary.json"
TOKENS_FIT = ROOT / "fit_expansion/data/tokens_fit.jsonl"
OUTPUT = HERE / "RESULTS.json"

K = 4
TYPE_NAMES = (
    "Evident Conflict",
    "Subtle Conflict",
    "Evident Baseless Info",
    "Subtle Baseless Info",
)
TYPE_SHORT = {
    "Evident Conflict": "EC",
    "Subtle Conflict": "SC",
    "Evident Baseless Info": "EBI",
    "Subtle Baseless Info": "SBI",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ap(y: np.ndarray, score: np.ndarray):
    y = np.asarray(y, dtype=np.int8)
    if not len(y) or y.min() == y.max():
        return None
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    ranked_y, ranked_s = y[order], score[order]
    ends = np.r_[np.flatnonzero(ranked_s[1:] != ranked_s[:-1]), len(y) - 1]
    tp = np.cumsum(ranked_y)[ends].astype(np.float64)
    previous_tp = np.r_[0.0, tp[:-1]]
    precision = tp / (ends + 1)
    return float(np.sum((tp - previous_tp) * precision) / y.sum())


def auc(y: np.ndarray, score: np.ndarray):
    y = np.asarray(y, dtype=np.int8)
    if not len(y) or y.min() == y.max():
        return None
    ranks = rankdata(np.asarray(score, dtype=np.float64))
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def rankdata(values: np.ndarray, method: str = "average") -> np.ndarray:
    """Small NumPy-only equivalent of scipy.stats.rankdata(method='average')."""
    assert method == "average"
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ends = np.r_[np.flatnonzero(sorted_values[1:] != sorted_values[:-1]), len(values) - 1]
    starts = np.r_[0, ends[:-1] + 1]
    ranked = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranked[order[start:end + 1]] = (start + end) / 2 + 1
    return ranked


def metric(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(score) >= threshold
    y = np.asarray(y, dtype=bool)
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    tn = int(np.count_nonzero(~pred & ~y))
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "predicted_positive": int(pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "average_precision": ap(y, score),
        "auroc": auc(y, score),
    }


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-np.asarray(score), kind="stable")
    out = np.zeros(len(score), dtype=bool)
    out[order[:k]] = True
    return out


def quantiles(score: np.ndarray) -> dict:
    ps = (0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    qs = np.quantile(score, ps)
    return {
        "n": int(len(score)),
        "mean": float(np.mean(score)),
        "std": float(np.std(score)),
        **{f"q{int(p * 100):02d}": float(q) for p, q in zip(ps, qs)},
    }


def corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        x, y = x[mask], y[mask]
    # Average ranks give a deterministic Spearman value in the presence of ties.
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    return {
        "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rx, ry)[0, 1]),
    }


def load_held_token_rows(answer_ids: list[str]) -> list[dict]:
    wanted = set(answer_ids)
    rows = {}
    with TOKENS_FIT.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["response_id"])
            if rid in wanted:
                assert row["partition"] == "fit"
                assert rid not in rows
                rows[rid] = row
    assert set(rows) == wanted
    return [rows[rid] for rid in answer_ids]


def load_and_audit_prediction(path: Path, frozen: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        window = z["conflict_window_scores"].copy()
        answer = z["conflict_answer_scores"].copy()
        positions = z["held_flat_positions"].astype(np.int64)
        flat_scores = z["held_flat_conflict_scores"].copy()
    assert len(np.unique(positions)) == len(positions)
    dense = np.full(len(frozen["input_ids"]), -np.inf, dtype=flat_scores.dtype)
    dense[positions] = flat_scores
    values = frozen["held_window_token_values"]
    indptr = frozen["held_window_token_indptr"]
    projected = np.maximum.reduceat(dense[values], indptr[:-1])
    assert np.array_equal(projected, window)
    answer_projected = np.full(len(frozen["held_answer_label"]), -np.inf, dtype=window.dtype)
    np.maximum.at(answer_projected, frozen["held_window_answer_index"], window)
    assert np.array_equal(answer_projected, answer)
    return window.astype(np.float64), {
        "flat_positions_unique": True,
        "stored_window_scores_bitwise_match_flat_projection": True,
        "stored_answer_scores_bitwise_match_window_max": True,
        "flat_positions": int(len(positions)),
        "windows": int(len(window)),
        "answers": int(len(answer)),
    }


def reconstruct_windows(rows: list[dict], frozen: dict[str, np.ndarray]) -> dict:
    n_window = len(frozen["held_window_label"])
    masks = {TYPE_SHORT[name]: np.zeros(n_window, dtype=bool) for name in TYPE_NAMES}
    answer_has = {TYPE_SHORT[name]: np.zeros(n_window, dtype=bool) for name in TYPE_NAMES}
    starts = np.empty(n_window, dtype=np.int32)
    lengths = np.empty(n_window, dtype=np.int32)
    relpos = np.empty(n_window, dtype=np.float64)
    rebuilt_y = np.zeros(n_window, dtype=np.int8)
    rebuilt_answer = np.empty(n_window, dtype=np.int32)
    cursor = 0
    answer_lengths = []

    for ai, row in enumerate(rows):
        lexical = np.asarray(row["lexical_mask"], dtype=bool)
        risk = np.asarray(row["risk_mask"], dtype=bool)
        n = int(row["token_count"])
        assert len(lexical) == len(risk) == n
        answer_lengths.append(n)
        token_masks = {TYPE_SHORT[name]: np.zeros(n, dtype=bool) for name in TYPE_NAMES}
        for mapping in row["span_token_mapping"]:
            span_index = int(mapping["span_index"])
            label_type = row["original_labels"][span_index]["label_type"]
            assert label_type in TYPE_SHORT
            indices = np.asarray(mapping["risk_token_indices"], dtype=np.int64)
            token_masks[TYPE_SHORT[label_type]][indices] = True
        any_type = np.logical_or.reduce(list(token_masks.values())) if token_masks else np.zeros(n, bool)
        assert np.array_equal(any_type, risk)

        eligible = [start for start in range(max(0, n - K + 1))
                    if lexical[start:start + K].any()]
        stop = cursor + len(eligible)
        frozen_count = int(np.count_nonzero(frozen["held_window_answer_index"] == ai))
        assert len(eligible) == frozen_count
        starts[cursor:stop] = eligible
        lengths[cursor:stop] = n
        rebuilt_answer[cursor:stop] = ai
        for local, start in enumerate(eligible):
            at = cursor + local
            rebuilt_y[at] = int(risk[start:start + K].any())
            relpos[at] = (start + (K - 1) / 2) / max(1, n - 1)
            for short, tm in token_masks.items():
                masks[short][at] = tm[start:start + K].any()
                answer_has[short][at] = tm.any()
        cursor = stop

    assert cursor == n_window
    assert np.array_equal(rebuilt_y, frozen["held_window_label"])
    assert np.array_equal(rebuilt_answer, frozen["held_window_answer_index"])
    assert np.array_equal(masks["EC"].astype(np.int8), frozen["held_window_ec"])
    assert np.array_equal(masks["SC"].astype(np.int8), frozen["held_window_sc"])
    assert np.array_equal(np.logical_or.reduce(list(masks.values())).astype(np.int8), rebuilt_y)
    return {
        "type_masks": masks,
        "answer_has_type": answer_has,
        "starts": starts,
        "answer_length": lengths,
        "relative_position": relpos,
        "answer_lengths": np.asarray(answer_lengths, dtype=np.int32),
        "audit": {
            "window_count": n_window,
            "answer_count": len(rows),
            "held_label_exact_match": True,
            "held_answer_index_exact_match": True,
            "held_ec_exact_match": True,
            "held_sc_exact_match": True,
            "union_four_types_exact_match_to_held_label": True,
            "type_window_counts": {short: int(mask.sum()) for short, mask in masks.items()},
        },
    }


def added_breakdown(
    name: str,
    aux: np.ndarray,
    v4: np.ndarray,
    y: np.ndarray,
    threshold: float,
    recon: dict,
    coverage: np.ndarray,
) -> dict:
    base = v4 >= threshold
    added = ~base & (aux >= threshold)
    tp = added & (y == 1)
    fp = added & (y == 0)
    types = recon["type_masks"]
    answer_types = recon["answer_has_type"]

    local_type = {short: int(np.count_nonzero(tp & mask)) for short, mask in types.items()}
    # FPs are clean by window definition. Answer-level incidence diagnoses spillover
    # elsewhere in an error-bearing answer; it is intentionally overlap-counted.
    fp_answer_type = {short: int(np.count_nonzero(fp & mask))
                      for short, mask in answer_types.items()}
    any_answer_error = np.logical_or.reduce(list(answer_types.values()))

    def cell(mask: np.ndarray) -> dict:
        a = added & mask
        t = int(np.count_nonzero(tp & mask))
        f = int(np.count_nonzero(fp & mask))
        return {
            "windows": int(np.count_nonzero(mask)),
            "added_alerts": int(a.sum()),
            "added_tp": t,
            "added_fp": f,
            "added_precision": t / (t + f) if t + f else None,
        }

    high = coverage >= 0.5
    answer_lengths = recon["answer_lengths"]
    q1, q2, q3 = [int(x) for x in np.quantile(answer_lengths, [0.25, 0.5, 0.75], method="nearest")]
    wl = recon["answer_length"]
    length_cells = {
        f"len_le_{q1}": wl <= q1,
        f"len_{q1 + 1}_{q2}": (wl > q1) & (wl <= q2),
        f"len_{q2 + 1}_{q3}": (wl > q2) & (wl <= q3),
        f"len_ge_{q3 + 1}": wl > q3,
    }
    rp = recon["relative_position"]
    position_cells = {
        "early_[0,1/3)": rp < 1 / 3,
        "middle_[1/3,2/3)": (rp >= 1 / 3) & (rp < 2 / 3),
        "late_[2/3,1]": rp >= 2 / 3,
    }
    return {
        "arm": name,
        "added_alerts": int(added.sum()),
        "added_tp": int(tp.sum()),
        "added_fp": int(fp.sum()),
        "added_precision": float(tp.sum() / added.sum()) if added.any() else None,
        "tp_by_local_gold_type_overlap_counted": local_type,
        "tp_by_local_gold_type_and_coverage": {
            short: {
                "high_ge_0.5": int(np.count_nonzero(tp & mask & (coverage >= 0.5))),
                "low_lt_0.5": int(np.count_nonzero(tp & mask & (coverage < 0.5))),
            }
            for short, mask in types.items()
        },
        "tp_without_any_type": int(np.count_nonzero(tp & ~np.logical_or.reduce(list(types.values())))),
        "fp_local_gold_type_note": "all added FP are locally clean by definition",
        "fp_in_answer_containing_type_overlap_counted": fp_answer_type,
        "fp_in_answer_with_any_error": int(np.count_nonzero(fp & any_answer_error)),
        "fp_in_fully_safe_answer": int(np.count_nonzero(fp & ~any_answer_error)),
        "coverage": {"high_ge_0.5": cell(high), "low_lt_0.5": cell(~high)},
        "answer_length_answer_level_quartile_edges": [q1, q2, q3],
        "answer_length": {key: cell(mask) for key, mask in length_cells.items()},
        "relative_window_position": {key: cell(mask) for key, mask in position_cells.items()},
    }


def diagnostic_scores(v4: np.ndarray, aux: np.ndarray) -> dict[str, np.ndarray]:
    # These are prespecified algebraic diagnostics only: no fitted parameter,
    # selected weight, or selected cutoff enters them.
    n = len(v4)
    v4_rank = (rankdata(v4, method="average") - 0.5) / n
    aux_rank = (rankdata(aux, method="average") - 0.5) / n
    return {
        "aux_raw": aux,
        "v4_raw": v4,
        "max_OR": np.maximum(v4, aux),
        "raw_residual_aux_minus_v4": aux - v4,
        "equal_rank_sum": v4_rank + aux_rank,
        "min_AND": np.minimum(v4, aux),
    }


def main() -> None:
    for path in (ARRAYS, CANDIDATE, CONTROL, SUMMARY, TOKENS_FIT):
        assert path.is_file(), path
    with np.load(ARRAYS, allow_pickle=False) as z:
        frozen = {key: z[key].copy() for key in z.files}
    candidate, candidate_projection_audit = load_and_audit_prediction(CANDIDATE, frozen)
    control, control_projection_audit = load_and_audit_prediction(CONTROL, frozen)
    frozen_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    threshold = float(frozen_summary["v4"]["overall"]["threshold"])

    y = frozen["held_window_label"].astype(np.int8)
    ec = frozen["held_window_ec"].astype(bool)
    sc = frozen["held_window_sc"].astype(bool)
    conflict_y = (ec | sc).astype(np.int8)
    v4 = frozen["held_v4_window_score"].astype(np.float64)
    coverage = frozen["held_window_max_evidence_coverage"].astype(np.float64)
    assert len(y) == len(v4) == len(candidate) == len(control)

    answer_ids = [str(x) for x in frozen["held_answer_ids"]]
    token_rows = load_held_token_rows(answer_ids)
    recon = reconstruct_windows(token_rows, frozen)
    miss = v4 < threshold
    high = coverage >= 0.5

    score_map = {"v4": v4, "candidate": candidate, "control": control}
    distribution = {}
    for name, score in score_map.items():
        distribution[name] = {
            "all": quantiles(score),
            "binary_positive": quantiles(score[y == 1]),
            "binary_negative": quantiles(score[y == 0]),
            "conflict_positive": quantiles(score[conflict_y == 1]),
            "above_frozen_v4_threshold": int(np.count_nonzero(score >= threshold)),
        }

    correlations = {}
    for a, b in (("v4", "candidate"), ("v4", "control"), ("candidate", "control")):
        correlations[f"{a}__{b}"] = {
            "all": corr(score_map[a], score_map[b]),
            "v4_miss": corr(score_map[a], score_map[b], miss),
            "binary_positive": corr(score_map[a], score_map[b], y == 1),
            "binary_negative": corr(score_map[a], score_map[b], y == 0),
            "high_coverage": corr(score_map[a], score_map[b], high),
            "low_coverage": corr(score_map[a], score_map[b], ~high),
        }

    miss_conflict = {
        "region": {
            "windows": int(miss.sum()),
            "binary_positive": int(y[miss].sum()),
            "conflict_positive": int(conflict_y[miss].sum()),
            "conflict_prevalence": float(conflict_y[miss].mean()),
        },
        "average_precision": {name: ap(conflict_y[miss], score[miss])
                              for name, score in score_map.items()},
        "high_coverage": {
            "windows": int(np.count_nonzero(miss & high)),
            "conflict_positive": int(conflict_y[miss & high].sum()),
            "average_precision": {name: ap(conflict_y[miss & high], score[miss & high])
                                  for name, score in score_map.items()},
        },
        "low_coverage": {
            "windows": int(np.count_nonzero(miss & ~high)),
            "conflict_positive": int(conflict_y[miss & ~high].sum()),
            "average_precision": {name: ap(conflict_y[miss & ~high], score[miss & ~high])
                                  for name, score in score_map.items()},
        },
    }

    fixed_metrics = {"v4": metric(y, v4, threshold)}
    fixed_type_recall = {}
    matched_budget = {}
    budget = fixed_metrics["v4"]["predicted_positive"]
    for name, aux in (("candidate", candidate), ("control", control)):
        combined = np.maximum(v4, aux)
        fixed_metrics[f"{name}_max"] = metric(y, combined, threshold)
        selected = top_k(combined, budget)
        matched_budget[name] = {
            "budget": budget,
            "tp": int(np.count_nonzero(selected & (y == 1))),
            "fp": int(np.count_nonzero(selected & (y == 0))),
            "conflict_recall": float(np.count_nonzero(selected & (conflict_y == 1)) /
                                     conflict_y.sum()),
            "ec_recall": float(np.count_nonzero(selected & ec) / ec.sum()),
            "sc_recall": float(np.count_nonzero(selected & sc) / sc.sum()),
        }
        fixed_pred = combined >= threshold
        fixed_type_recall[f"{name}_max"] = {
            "conflict": float(np.count_nonzero(fixed_pred & (conflict_y == 1)) /
                              conflict_y.sum()),
            "EC": float(np.count_nonzero(fixed_pred & ec) / ec.sum()),
            "SC": float(np.count_nonzero(fixed_pred & sc) / sc.sum()),
        }
    base_selected = v4 >= threshold
    matched_budget["v4"] = {
        "budget": budget,
        "tp": int(np.count_nonzero(base_selected & (y == 1))),
        "fp": int(np.count_nonzero(base_selected & (y == 0))),
        "conflict_recall": float(np.count_nonzero(base_selected & (conflict_y == 1)) /
                                 conflict_y.sum()),
        "ec_recall": float(np.count_nonzero(base_selected & ec) / ec.sum()),
        "sc_recall": float(np.count_nonzero(base_selected & sc) / sc.sum()),
    }
    fixed_type_recall["v4"] = {
        "conflict": matched_budget["v4"]["conflict_recall"],
        "EC": matched_budget["v4"]["ec_recall"],
        "SC": matched_budget["v4"]["sc_recall"],
    }

    diagnostic = {}
    for arm, aux in (("candidate", candidate), ("control", control)):
        diagnostic[arm] = {}
        for name, score in diagnostic_scores(v4, aux).items():
            diagnostic[arm][name] = {
                "overall_binary_ap": ap(y, score),
                "overall_conflict_ap": ap(conflict_y, score),
                "v4_miss_binary_ap": ap(y[miss], score[miss]),
                "v4_miss_conflict_ap": ap(conflict_y[miss], score[miss]),
            }
        and_pred = (v4 >= threshold) & (aux >= threshold)
        at = int(np.count_nonzero(and_pred & (y == 1)))
        af = int(np.count_nonzero(and_pred & (y == 0)))
        afn = int(np.count_nonzero(~and_pred & (y == 1)))
        diagnostic[arm]["fixed_threshold_AND_gate"] = {
            "threshold_for_both_scores": threshold,
            "predicted_positive": int(and_pred.sum()),
            "tp": at,
            "fp": af,
            "precision": at / (at + af) if at + af else 0.0,
            "recall": at / y.sum(),
            "f1": 2 * at / (2 * at + af + afn),
            "conflict_recall": float(np.count_nonzero(and_pred & (conflict_y == 1)) /
                                     conflict_y.sum()),
            "cannot_recover_v4_misses_by_construction": True,
        }

    output = {
        "scope": {
            "data": "QA fit only; source-connected held fold 0, 747 answers",
            "calibration_rows_read": 0,
            "test_rows_read": 0,
            "gpu_used": False,
            "model_weights_loaded": False,
            "training_or_tuning": False,
            "threshold_search": False,
            "baseline_modified": False,
            "exploratory_only": True,
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in (ARRAYS, CANDIDATE, CONTROL, SUMMARY, TOKENS_FIT)},
        "frozen_v4_threshold": threshold,
        "population": {
            "answers": len(answer_ids),
            "windows": len(y),
            "binary_positive": int(y.sum()),
            "conflict_positive": int(conflict_y.sum()),
            "ec": int(ec.sum()),
            "sc": int(sc.sum()),
            "high_coverage": int(high.sum()),
            "low_coverage": int((~high).sum()),
        },
        "mapping_audit": recon["audit"],
        "prediction_projection_audit": {
            "candidate": candidate_projection_audit,
            "control": control_projection_audit,
        },
        "score_distribution": distribution,
        "score_correlations": correlations,
        "v4_miss_conflict": miss_conflict,
        "fixed_v4_threshold_metrics": fixed_metrics,
        "fixed_v4_threshold_type_recall": fixed_type_recall,
        "matched_v4_alert_budget": matched_budget,
        "max_added_alert_breakdown": {
            "candidate": added_breakdown("candidate", candidate, v4, y, threshold,
                                         recon, coverage),
            "control": added_breakdown("control", control, v4, y, threshold,
                                       recon, coverage),
        },
        "parameter_free_diagnostics_no_selection": diagnostic,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "mapping": output["mapping_audit"],
        "fixed": fixed_metrics,
        "miss_conflict": miss_conflict,
        "added": output["max_added_alert_breakdown"],
        "diagnostic": diagnostic,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
