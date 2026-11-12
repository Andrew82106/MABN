"""Frozen held-fit fold-0 audit of the type-aware contradiction probability.

The candidate list is fixed in the accompanying PROTOCOL.md.  This script reads
only fit-prefix rows and does not load the v4 arrays file, because that file also
contains calibration labels.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4 = ROOT / "results/microclaim_crossencoder_expanded_v4/fold_0"
TYPE = ROOT / "results/microclaim_crossencoder_typeaware_v5/fold_0_pilot"
OUT = ROOT / "results/typeaware_contradiction_fold0_pilot_v1"

N_FIT_CLAIMS = 34_919
N_FIT_ANSWERS = 3_680
FOLD = 0
K = 4
TYPES = (
    "Evident Conflict",
    "Subtle Conflict",
    "Evident Baseless Info",
    "Subtle Baseless Info",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_prefix(path: Path, count: int):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index in range(count):
            line = handle.readline()
            assert line, (path, index)
            rows.append(json.loads(line))
    return rows


def rows_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def choose_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    assert set(labels.tolist()) == {0, 1} and np.isfinite(scores).all()
    order = np.argsort(-scores, kind="stable")
    ss, yy = scores[order], labels[order]
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.r_[0, np.cumsum(yy)[last]]
    n = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(ss[0], np.inf), ss[last]]
    f1 = 2 * tp / (n + labels.sum())
    precision = np.divide(tp, n, out=np.zeros(len(n), dtype=float), where=n > 0)
    best = max(range(len(n)), key=lambda j: (f1[j], precision[j], thresholds[j]))
    return float(thresholds[best])


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.count_nonzero((labels == 1) & predicted))
    fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted))
    tn = int(np.count_nonzero((labels == 0) & ~predicted))
    return {
        "n": int(len(labels)), "positive": int(labels.sum()),
        "threshold": float(threshold), "predicted_positive": int(predicted.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }, predicted


def top_k_mask(scores, count):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:count]] = True
    assert int(mask.sum()) == count
    return mask


def recall(mask, labels):
    labels = np.asarray(labels, dtype=bool)
    denominator = int(labels.sum())
    numerator = int(np.count_nonzero(mask & labels))
    return {"tp": numerator, "positive": denominator,
            "recall": numerator / denominator if denominator else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_hash_before = sha256(OUT / "PROTOCOL.md")

    v4_complete = read_json(V4 / "complete.json")
    type_complete = read_json(TYPE / "complete.json")
    assert sha256(V4 / "predictions.npz") == v4_complete["predictions_sha256"]
    assert sha256(TYPE / "predictions.npz") == type_complete["predictions_sha256"]
    with np.load(V4 / "predictions.npz", allow_pickle=False) as z:
        v4_indices = z["held_indices"].copy()
        v4_claim = z["held_scores"].astype(np.float64)
    with np.load(TYPE / "predictions.npz", allow_pickle=False) as z:
        type_indices = z["held_indices"].copy()
        type_claim_saved = z["held_risk_scores"].astype(np.float64)
        type_prob = z["held_threeway_probabilities"].astype(np.float64)
    assert np.array_equal(v4_indices, type_indices)
    assert type_prob.shape == (len(type_indices), 3)
    type_claim = type_prob[:, 1] + type_prob[:, 2]
    assert np.allclose(type_claim, type_claim_saved, atol=2e-7, rtol=2e-6)

    # These functions stop at the frozen fit-prefix boundary.  They never ask
    # for one additional line, so no calibration record is parsed.
    examples = read_prefix(DATA / "examples.jsonl", N_FIT_CLAIMS)
    answers = read_prefix(DATA / "answers.jsonl", N_FIT_ANSWERS)
    assert all(row["example_index"] == i and row["partition"] == "fit"
               for i, row in enumerate(examples))
    assert all(row["response_index"] == i and row["partition"] == "fit"
               for i, row in enumerate(answers))

    held_set = set(map(int, type_indices.tolist()))
    assert held_set == {i for i, row in enumerate(examples) if row["held_fold"] == FOLD}
    position = {int(index): pos for pos, index in enumerate(type_indices)}
    examples_by_response = defaultdict(list)
    for row in examples:
        if row["held_fold"] == FOLD:
            examples_by_response[str(row["response_id"])].append(row)

    gold, type_gold = [], {name: [] for name in TYPES}
    owner_indices = []
    window_meta = []
    held_answer_count = 0
    for answer in answers:
        rid = str(answer["response_id"])
        local = examples_by_response.get(rid)
        if not local:
            continue
        held_answer_count += 1
        expected = list(range(int(answer["example_start"]), int(answer["example_end"])))
        assert [int(row["example_index"]) for row in local] == expected
        assert all(row["held_fold"] == FOLD for row in local)

        lexical = np.asarray(answer["lexical_mask"], dtype=bool)
        risk = np.asarray(answer["risk_mask"], dtype=bool)
        offsets = np.asarray(answer["response_token_offsets"], dtype=np.int64)
        text = answer["original_response"]
        assert len(lexical) == len(risk) == len(offsets) == int(answer["token_count"])
        eligible = [start for start in range(max(0, len(lexical) - K + 1))
                    if lexical[start:start + K].any()]
        assert len(eligible) == int(answer["eligible_window_count"])

        # Deduplicate official spans repeated across overlapping microclaims.
        spans = {}
        for row in local:
            for span in row["overlapping_gold_spans"]:
                key = (span["label_type"], int(span["start"]), int(span["end"]),
                       bool(span["implicit_true"]), bool(span["due_to_null"]))
                assert span["label_type"] in TYPES
                spans[key] = span
        masks = {name: np.zeros(len(lexical), dtype=bool) for name in TYPES}
        for (name, left, right, _, _), _span in spans.items():
            assert 0 <= left <= right <= len(text)
            for ti, (a, b) in enumerate(offsets):
                # Match the frozen gold: only alphanumeric characters inside
                # the exact official character span make a token risky.
                lo, hi = max(int(a), left), min(int(b), right)
                if lo < hi and any(text[c].isalnum() for c in range(lo, hi)):
                    masks[name][ti] = True
        union = np.logical_or.reduce([masks[name] for name in TYPES])
        assert np.array_equal(union, risk), rid

        owners = {start: [] for start in eligible}
        for row in local:
            claim_index = int(row["example_index"])
            for start in row["mapped_eligible_window_starts"]:
                if int(start) in owners:
                    owners[int(start)].append(position[claim_index])
        for start in eligible:
            assert owners[start], (rid, start)
            owner_indices.append(np.asarray(sorted(set(owners[start])), dtype=np.int64))
            gold.append(int(risk[start:start + K].any()))
            for name in TYPES:
                type_gold[name].append(int(masks[name][start:start + K].any()))
            window_meta.append({"response_id": rid, "token_start": int(start)})
        assert sum(gold[-len(eligible):]) == int(answer["positive_window_count"])

    gold = np.asarray(gold, dtype=np.int8)
    type_gold = {name: np.asarray(values, dtype=np.int8) for name, values in type_gold.items()}
    assert np.array_equal(np.logical_or.reduce([type_gold[name].astype(bool) for name in TYPES]),
                          gold.astype(bool))
    assert len(gold) == len(owner_indices) and held_answer_count > 0

    def project(claim_scores):
        return np.asarray([np.max(claim_scores[owners]) for owners in owner_indices], dtype=np.float64)

    v4_window = project(v4_claim)
    type_window = project(type_claim)
    contradiction_window = project(type_prob[:, 2])
    candidates = {
        "v4_risk": v4_window,
        "type_risk": type_window,
        "type_contradiction": contradiction_window,
        "max_v4_contradiction": np.maximum(v4_window, contradiction_window),
        "convex_75v4_25c": 0.75 * v4_window + 0.25 * contradiction_window,
        "convex_50v4_50c": 0.50 * v4_window + 0.50 * contradiction_window,
    }
    assert list(candidates) == [
        "v4_risk", "type_risk", "type_contradiction", "max_v4_contradiction",
        "convex_75v4_25c", "convex_50v4_50c",
    ]

    optimized, optimized_masks = {}, {}
    for name, scores in candidates.items():
        threshold = choose_threshold(gold, scores)
        result, predicted = metrics(gold, scores, threshold)
        result["type_recall"] = {kind: recall(predicted, labels) for kind, labels in type_gold.items()}
        result["combined_conflict_recall"] = recall(
            predicted, type_gold["Evident Conflict"].astype(bool) |
            type_gold["Subtle Conflict"].astype(bool))
        optimized[name] = result
        optimized_masks[name] = predicted

    budget = optimized["v4_risk"]["predicted_positive"]
    matched = {}
    for name, scores in candidates.items():
        predicted = top_k_mask(scores, budget)
        tp = int(np.count_nonzero(predicted & (gold == 1)))
        fp = int(np.count_nonzero(predicted & (gold == 0)))
        matched[name] = {
            "alert_budget": int(budget), "tp": tp, "fp": fp,
            "precision": tp / budget if budget else 0.0,
            "overall_recall": tp / int(gold.sum()),
            "type_recall": {kind: recall(predicted, labels) for kind, labels in type_gold.items()},
            "combined_conflict_recall": recall(
                predicted, type_gold["Evident Conflict"].astype(bool) |
                type_gold["Subtle Conflict"].astype(bool)),
        }

    conflict_gold = (type_gold["Evident Conflict"].astype(bool) |
                     type_gold["Subtle Conflict"].astype(bool)).astype(np.int8)
    conflict_ranking = {}
    v4_rejected = ~optimized_masks["v4_risk"]
    for name in ("v4_risk", "type_risk", "type_contradiction"):
        scores = candidates[name]
        conflict_ranking[name] = {
            "all_windows_ap": float(average_precision_score(conflict_gold, scores)),
            "inside_v4_rejected_ap": float(average_precision_score(
                conflict_gold[v4_rejected], scores[v4_rejected])),
            "inside_v4_rejected_rows": int(v4_rejected.sum()),
            "inside_v4_rejected_conflict_positive": int(conflict_gold[v4_rejected].sum()),
        }

    base = optimized["v4_risk"]
    base_match = matched["v4_risk"]
    decisions = {}
    for name in ("max_v4_contradiction", "convex_75v4_25c", "convex_50v4_50c"):
        overall_route = (
            optimized[name]["average_precision"] >= base["average_precision"] + 0.005 and
            optimized[name]["f1"] >= base["f1"] + 0.005
        )
        targeted_route = (
            matched[name]["combined_conflict_recall"]["recall"] >=
            base_match["combined_conflict_recall"]["recall"] + 0.02 and
            matched[name]["fp"] <= base_match["fp"] and
            optimized[name]["average_precision"] >= base["average_precision"] - 0.002
        )
        decisions[name] = {"overall_route": bool(overall_route),
                           "targeted_route": bool(targeted_route),
                           "continue": bool(overall_route or targeted_route)}
    worth = any(value["continue"] for value in decisions.values())

    result = {
        "status": "complete",
        "scope": {
            "partition": "fit only", "held_fold": FOLD,
            "calibration_rows_read": 0, "test_rows_read": 0,
            "new_model_trained": False, "published_baseline_involved": False,
            "protocol_sha256_before_metrics": protocol_hash_before,
        },
        "population": {
            "held_answers": held_answer_count, "held_microclaims": len(type_indices),
            "held_windows": len(gold), "positive_windows": int(gold.sum()),
            "type_positive_windows": {name: int(values.sum()) for name, values in type_gold.items()},
            "combined_conflict_positive_windows": int(conflict_gold.sum()),
        },
        "held_f1_opt_diagnostic": optimized,
        "matched_v4_alert_budget": matched,
        "conflict_ranking_diagnostic": conflict_ranking,
        "continuation_rule": decisions,
        "worth_running_remaining_folds": bool(worth),
        "artifact_hashes": {
            "v4_predictions": sha256(V4 / "predictions.npz"),
            "type_predictions": sha256(TYPE / "predictions.npz"),
            "parsed_fit_examples_canonical": rows_sha256(examples),
            "parsed_fit_answers_canonical": rows_sha256(answers),
        },
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TYPEAWARE_CONTRADICTION_FOLD0_AUDIT_COMPLETE", worth)


if __name__ == "__main__":
    main()
