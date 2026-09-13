#!/usr/bin/env python3
"""Independent verification of OSR v2 saved fit-OOF predictions."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np


HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
OUT = HERE / "fit_oof_v2"
ANSWERS = BENCH / "data" / "answers_fit.jsonl"
TOKENS = BENCH / "data" / "tokens_fit.jsonl"
CANDIDATES = ("anchored_filter", "emission_identity", "transition_only")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fit(path: Path) -> list[dict]:
    if path.name not in {"answers_fit.jsonl", "tokens_fit.jsonl"}:
        raise AssertionError("non-fit source")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("partition") != "fit" for row in rows):
        raise AssertionError("non-fit row")
    return rows


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold_for(group_id: str) -> int:
    raw = hashlib.sha256(f"ragtruth_fit_human\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % 5


def raw_windows(count: int):
    return [(left, min(left + 4, count)) for left in range(max(1, count - 3))]


def logit(value: float) -> float:
    value = min(1.0 - 1e-7, max(1e-7, float(value)))
    return math.log(value) - math.log1p(-value)


def logistic(value: float) -> float:
    value = min(40.0, max(-40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    positives = int(labels.sum())
    tp = fp = 0
    best = (0.0, 0.0, 0.0, float(np.nextafter(1.0, 2.0)))
    threshold = best[-1]
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        found = int(y[index:end].sum())
        tp += found
        fp += end - index - found
        recall = tp / positives
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = (f1, precision, recall, float(s[index]))
        if key > best:
            best = key
            threshold = float(s[index])
        index = end
    return threshold


def ranking(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    order = np.argsort(scores, kind="stable")
    s, y = scores[order], labels[order]
    negative_before = 0
    concordance = 0.0
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        positive_group = int(y[index:end].sum())
        negative_group = end - index - positive_group
        concordance += positive_group * (negative_before + 0.5 * negative_group)
        negative_before += negative_group
        index = end
    auroc = concordance / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = fp = 0
    previous_recall = average_precision = 0.0
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        found = int(y[index:end].sum())
        tp += found
        fp += end - index - found
        recall = tp / positives
        average_precision += (recall - previous_recall) * tp / (tp + fp)
        previous_recall = recall
        index = end
    return average_precision, auroc


def metric(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    selected = scores >= threshold
    tp = int(np.sum(selected & (labels == 1)))
    fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum(~selected & (labels == 1)))
    tn = int(np.sum(~selected & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap, auc = ranking(labels, scores)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "count": len(labels), "positive": int(labels.sum()),
            "prevalence": float(labels.mean()), "average_precision": ap, "auroc": auc}


def assert_close(actual, expected, name: str, tolerance: float = 2e-12) -> None:
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance):
            raise AssertionError((name, actual, expected))
    elif actual != expected:
        raise AssertionError((name, actual, expected))


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUT / "METRICS.json").read_text(encoding="utf-8"))
    models = json.loads((OUT / "FOLD_MODELS.json").read_text(encoding="utf-8"))
    for name, record in manifest["outputs"].items():
        path = OUT / name
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise AssertionError(f"output hash mismatch: {name}")
    if manifest["calibration_or_test_labels_read"] or manifest["gpu_started"] or manifest["baseline_mutated"]:
        raise AssertionError("scope flag failure")

    answers = {str(row["response_id"]): row for row in read_fit(ANSWERS)}
    gold_tokens = {str(row["response_id"]): row for row in read_fit(TOKENS)}
    token_rows = read_gzip(OUT / "TOKENS.jsonl.gz")
    window_rows = read_gzip(OUT / "WINDOWS.jsonl.gz")
    answer_rows = read_gzip(OUT / "ANSWERS.jsonl.gz")
    span_rows = read_gzip(OUT / "SPANS.jsonl.gz")
    if not (len(answers) == len(gold_tokens) == len(answer_rows) == 634):
        raise AssertionError("answer count")
    if len(token_rows) != 139518 or len(window_rows) != 168123 or len(span_rows) != 646:
        raise AssertionError("prediction count")

    priors = {int(row["fold"]): row["natural_priors"] for row in models["folds"]}
    token_by_answer: dict[str, list[dict]] = {}
    groups = {}
    for row in token_rows:
        rid = row["response_id"]
        gold = gold_tokens[rid]
        raw = int(row["raw_bpe_index"])
        expected_fold = fold_for(gold["group_id"])
        if row["fold"] != expected_fold or row["group_id"] != gold["group_id"]:
            raise AssertionError("token fold/group")
        if not gold["lexical_mask"][raw] or row["gold_risk"] != gold["risk_mask"][raw]:
            raise AssertionError("token gold")
        fold_prior = priors[expected_fold]
        for family, balanced_name, corrected_name in (
            ("emission", "p_emit_balanced", "p_emit"),
            ("onset", "p_on_balanced", "p_on"),
            ("continuation", "p_cont_balanced", "p_cont"),
        ):
            expected = logistic(logit(row[balanced_name]) + logit(fold_prior[family]))
            assert_close(row[corrected_name], expected, f"{family} prior correction", 3e-6)
        assert_close(row["emission_identity"], row["p_emit"], "identity", 0.0)
        token_by_answer.setdefault(rid, []).append(row)
        groups.setdefault(row["group_id"], set()).add(row["fold"])
    if len(groups) != 615 or any(len(value) != 1 for value in groups.values()):
        raise AssertionError("source-group split")

    for rid, rows in token_by_answer.items():
        rows.sort(key=lambda row: int(row["lexical_index"]))
        previous_transition = previous_anchor = 0.0
        expected_raw = [i for i, value in enumerate(gold_tokens[rid]["lexical_mask"]) if value]
        if [int(row["raw_bpe_index"]) for row in rows] != expected_raw:
            raise AssertionError("lexical token coverage/order")
        for row in rows:
            transition = (1.0 - previous_transition) * row["p_on"] + previous_transition * row["p_cont"]
            assert_close(row["transition_only"], transition, "transition recurrence", 3e-6)
            previous_transition = row["transition_only"]
            predicted = (1.0 - previous_anchor) * row["p_on"] + previous_anchor * row["p_cont"]
            anchored = logistic(logit(predicted) + logit(row["p_emit_balanced"]))
            assert_close(row["anchored_filter"], anchored, "anchored recurrence", 3e-6)
            previous_anchor = row["anchored_filter"]
            forced = logistic(logit(row["p_cont"]) + logit(row["p_emit_balanced"]))
            assert_close(row["forced_previous_risk_anchored"], forced, "forced exit", 3e-6)

    saved_windows = {(row["response_id"], int(row["raw_bpe_start"])): row for row in window_rows}
    if len(saved_windows) != len(window_rows):
        raise AssertionError("duplicate window")
    category_counts = {"released_onset": 0, "internal_continuation": 0, "clean": 0}
    for rid, gold in gold_tokens.items():
        token_lookup = {int(row["raw_bpe_index"]): row for row in token_by_answer[rid]}
        onset = {int(mapping["risk_token_indices"][0]) for mapping in gold["span_token_mapping"]
                 if mapping["risk_token_indices"]}
        for left, right in raw_windows(len(gold["lexical_mask"])):
            lexical = [index for index in range(left, right) if gold["lexical_mask"][index]]
            if not lexical:
                continue
            row = saved_windows.pop((rid, left))
            risk = int(any(gold["risk_mask"][index] for index in lexical))
            category = "released_onset" if risk and any(index in onset for index in lexical) else \
                       "internal_continuation" if risk else "clean"
            if row["gold_risk"] != risk or row["gold_category"] != category:
                raise AssertionError("window gold geometry")
            category_counts[category] += 1
            for name in CANDIDATES:
                expected = max(token_lookup[index][name] for index in lexical)
                assert_close(row[name], expected, f"window max {name}", 0.0)
    if saved_windows:
        raise AssertionError("unexpected windows")

    answer_by_id = {row["response_id"]: row for row in answer_rows}
    windows_by_id: dict[str, list[dict]] = {}
    for row in window_rows:
        windows_by_id.setdefault(row["response_id"], []).append(row)
    for rid, row in answer_by_id.items():
        if row["gold_risk"] != answers[rid]["label"] or row["fold"] != fold_for(answers[rid]["group_id"]):
            raise AssertionError("answer gold/fold")
        for name in CANDIDATES:
            assert_close(row[name], max(item[name] for item in windows_by_id[rid]),
                         f"answer max {name}", 0.0)

    window_y = np.asarray([row["gold_risk"] for row in window_rows], dtype=np.uint8)
    answer_y = np.asarray([row["gold_risk"] for row in answer_rows], dtype=np.uint8)
    for name in CANDIDATES:
        window_s = np.asarray([row[name] for row in window_rows], dtype=np.float64)
        answer_s = np.asarray([row[name] for row in answer_rows], dtype=np.float64)
        for scope, labels, scores, saved in (
            ("window", window_y, window_s, metrics["candidates"][name]["window"]),
            ("answer", answer_y, answer_s, metrics["candidates"][name]["answer_max"]),
        ):
            threshold = choose_threshold(labels, scores)
            assert_close(threshold, saved["threshold"], f"{name} {scope} threshold", 0.0)
            reproduced = metric(labels, scores, threshold)
            for key, expected in saved.items():
                assert_close(reproduced[key], expected, f"{name} {scope} {key}")
        selected = window_s >= metrics["candidates"][name]["window"]["threshold"]
        for category, expected_count in category_counts.items():
            mask = np.asarray([row["gold_category"] == category for row in window_rows], dtype=bool)
            saved = metrics["candidates"][name]["window_categories_at_window_threshold"][category]
            if saved["count"] != expected_count or saved["selected"] != int(np.sum(selected & mask)):
                raise AssertionError("category diagnostic")
            assert_close(saved["selection_rate"], float(np.mean(selected[mask])), "category rate")

    if len(models["folds"]) != 5 or {row["fold"] for row in models["folds"]} != set(range(5)):
        raise AssertionError("fold models")
    for fold in models["folds"]:
        if len(fold["scaler_mean"]) != 68 or len(fold["scaler_std"]) != 68:
            raise AssertionError("scaler shape")
        if set(fold["parameters"]) != {"emission", "onset", "continuation"}:
            raise AssertionError("head set")
        if any(len(values) != 69 for values in fold["parameters"].values()):
            raise AssertionError("head shape")
        if any(fold["final_weighted_losses"][name] >= fold["initial_weighted_losses"][name]
               for name in fold["parameters"]):
            raise AssertionError("loss descent")

    result = {
        "schema_version": "osr-emission-anchor-fit-oof-independent-verification-v2",
        "status": "PASS",
        "outputs_hash_verified": len(manifest["outputs"]),
        "answers": len(answer_rows), "groups": len(groups),
        "lexical_tokens": len(token_rows), "eligible_windows": len(window_rows),
        "released_spans": len(span_rows),
        "group_folds_recomputed": True,
        "token_labels_reconstructed_from_fit": True,
        "prior_correction_recomputed": True,
        "all_three_decoders_recomputed": True,
        "raw_4bpe_geometry_reconstructed": True,
        "answer_max_exact": True,
        "thresholds_and_primary_metrics_reproduced": True,
        "fold_model_shapes_and_loss_descent_checked": True,
        "calibration_or_test_labels_read": False,
        "gpu_started": False,
        "baseline_mutated": False,
    }
    pending = OUT / "VERIFY.json.pending"
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    pending.replace(OUT / "VERIFY.json")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
