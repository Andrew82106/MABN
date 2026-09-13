#!/usr/bin/env python3
"""CPU-only, fit-only, five-fold source-group OOF for OSR v2.

The only label-bearing inputs accepted by the imported loader are
answers_fit.jsonl and tokens_fit.jsonl. Cached fit hidden_last/lb/nll arrays are
read after their metadata partition and SHA-256 are checked.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np


HERE = Path(__file__).resolve().parent
V1_DIR = HERE.parent / "onset_segment_probe_v1_design"
V1_RUNNER = V1_DIR / "run_human_only_oof.py"
PROTOCOL = HERE / "PROTOCOL.json"
OUT = HERE / "fit_oof_v2"

FOLDS = 5
DIM = 68
WIDTH = 4
EPOCHS = 25
LEARNING_RATE = 0.0003
BATCH_ANSWERS = 16
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
TRAIN_SEED = 20260913
CANDIDATES = ("anchored_filter", "emission_identity", "transition_only")
PRIMARY = "anchored_filter"


def load_v1_module():
    spec = importlib.util.spec_from_file_location("osr_v1_fit_loader", V1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 fit loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = load_v1_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(value, encoding="utf-8")
    pending.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def logit_scalar(value: float) -> float:
    clipped = min(1.0 - 1e-7, max(1e-7, float(value)))
    return math.log(clipped) - math.log1p(-clipped)


def family_examples(record: dict, family: str) -> tuple[np.ndarray, np.ndarray]:
    start = int(record["lex_start"])
    states = record["states"].astype(np.float32)
    if family == "emission":
        local = np.arange(len(states), dtype=np.int64)
    elif family == "onset":
        previous = np.r_[np.uint8(0), record["states"][:-1]]
        local = np.flatnonzero(previous == 0).astype(np.int64)
    elif family == "continuation":
        previous = np.r_[np.uint8(0), record["states"][:-1]]
        local = np.flatnonzero(previous == 1).astype(np.int64)
    else:
        raise KeyError(family)
    return start + local, states[local]


def head_weight_map(records: list[dict], train_ids: list[int], family: str) -> tuple[dict[int, np.ndarray], float, dict]:
    eligible_by_group: dict[str, list[int]] = defaultdict(list)
    cache = {}
    for answer_index in train_ids:
        item = family_examples(records[answer_index], family)
        cache[answer_index] = item
        if len(item[0]):
            eligible_by_group[records[answer_index]["group_id"]].append(answer_index)
    if not eligible_by_group:
        raise AssertionError(f"no {family} examples")
    base = {}
    group_mass = 1.0 / len(eligible_by_group)
    for answer_ids in eligible_by_group.values():
        answer_mass = group_mass / len(answer_ids)
        for answer_index in answer_ids:
            count = len(cache[answer_index][0])
            base[answer_index] = np.full(count, answer_mass / count, dtype=np.float64)
    class_mass = np.zeros(2, dtype=np.float64)
    raw_counts = np.zeros(2, dtype=np.int64)
    for answer_index, weights in base.items():
        labels = cache[answer_index][1].astype(np.int64)
        class_mass[0] += weights[labels == 0].sum()
        class_mass[1] += weights[labels == 1].sum()
        raw_counts[0] += int(np.sum(labels == 0))
        raw_counts[1] += int(np.sum(labels == 1))
    if np.any(class_mass <= 0) or not math.isclose(float(class_mass.sum()), 1.0, abs_tol=2e-6):
        raise AssertionError((family, class_mass))
    factors = 0.5 / class_mass
    balanced = {}
    for answer_index, weights in base.items():
        labels = cache[answer_index][1].astype(np.int64)
        balanced[answer_index] = (weights * factors[labels]).astype(np.float32)
    total = sum(float(values.sum()) for values in balanced.values())
    if not math.isclose(total, 1.0, abs_tol=2e-6):
        raise AssertionError((family, total))
    audit = {
        "raw_negative": int(raw_counts[0]),
        "raw_positive": int(raw_counts[1]),
        "raw_positive_rate": float(raw_counts[1] / raw_counts.sum()),
        "hierarchical_natural_negative_mass": float(class_mass[0]),
        "hierarchical_natural_positive_mass": float(class_mass[1]),
        "class_balance_factor_negative": float(factors[0]),
        "class_balance_factor_positive": float(factors[1]),
    }
    return balanced, float(class_mass[1]), audit


def batch_gradient(matrix: np.ndarray, records: list[dict], batch: np.ndarray, family: str,
                   weight_map: dict[int, np.ndarray], params: np.ndarray) -> tuple[np.ndarray, float, int]:
    all_indices, all_labels, all_weights = [], [], []
    for answer_index in map(int, batch):
        if answer_index not in weight_map:
            continue
        indices, labels = family_examples(records[answer_index], family)
        all_indices.append(indices)
        all_labels.append(labels)
        all_weights.append(weight_map[answer_index])
    if not all_indices:
        return np.zeros_like(params), 0.0, 0
    indices = np.concatenate(all_indices)
    labels = np.concatenate(all_labels)
    weights = np.concatenate(all_weights).astype(np.float32)
    weights /= weights.sum()
    x = matrix[indices]
    scores = sigmoid(x @ params[:-1] + params[-1])
    error = (scores - labels) * weights
    gradient = np.r_[x.T @ error, error.sum()].astype(np.float32)
    scores = np.clip(scores, 1e-7, 1.0 - 1e-7)
    loss = -float(np.sum(weights * (labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores))))
    return gradient, loss, len(indices)


def full_loss(matrix: np.ndarray, records: list[dict], train_ids: list[int], family: str,
              weight_map: dict[int, np.ndarray], params: np.ndarray) -> float:
    total = 0.0
    for answer_index in train_ids:
        if answer_index not in weight_map:
            continue
        indices, labels = family_examples(records[answer_index], family)
        scores = sigmoid(matrix[indices] @ params[:-1] + params[-1])
        scores = np.clip(scores, 1e-7, 1.0 - 1e-7)
        weights = weight_map[answer_index]
        total += -float(np.sum(weights * (labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores))))
    return total


def decode_heads(x: np.ndarray, params: dict[str, np.ndarray], priors: dict[str, float]) -> dict[str, np.ndarray]:
    balanced_logits = {name: x @ params[name][:-1] + params[name][-1]
                       for name in ("emission", "onset", "continuation")}
    balanced = {name: sigmoid(value) for name, value in balanced_logits.items()}
    corrected = {name: sigmoid(balanced_logits[name] + np.float32(logit_scalar(priors[name])))
                 for name in balanced_logits}
    length = len(x)
    transition_only = np.empty(length, dtype=np.float32)
    anchored = np.empty(length, dtype=np.float32)
    forced_previous_risk = np.empty(length, dtype=np.float32)
    previous_transition = np.float32(0.0)
    previous_anchored = np.float32(0.0)
    for index in range(length):
        trans = ((np.float32(1.0) - previous_transition) * corrected["onset"][index] +
                 previous_transition * corrected["continuation"][index])
        transition_only[index] = trans
        previous_transition = trans
        predicted = ((np.float32(1.0) - previous_anchored) * corrected["onset"][index] +
                     previous_anchored * corrected["continuation"][index])
        anchored[index] = sigmoid(np.asarray([logit_scalar(float(predicted)) +
                                              float(balanced_logits["emission"][index])],
                                             dtype=np.float32))[0]
        previous_anchored = anchored[index]
        forced_previous_risk[index] = sigmoid(np.asarray([
            logit_scalar(float(corrected["continuation"][index])) +
            float(balanced_logits["emission"][index])], dtype=np.float32))[0]
    return {
        "p_emit_balanced": balanced["emission"],
        "p_on_balanced": balanced["onset"],
        "p_cont_balanced": balanced["continuation"],
        "p_emit": corrected["emission"],
        "p_on": corrected["onset"],
        "p_cont": corrected["continuation"],
        "anchored_filter": anchored,
        "emission_identity": corrected["emission"],
        "transition_only": transition_only,
        "forced_previous_risk_anchored": forced_previous_risk,
    }


def train_fold(matrix: np.ndarray, records: list[dict], fold: int) -> tuple[dict[int, dict], dict]:
    train_ids = [i for i, row in enumerate(records) if row["fold"] != fold]
    held_ids = [i for i, row in enumerate(records) if row["fold"] == fold]
    train_groups = {records[i]["group_id"] for i in train_ids}
    held_groups = {records[i]["group_id"] for i in held_ids}
    if train_groups & held_groups:
        raise AssertionError("source-group leakage")
    train_lex = np.concatenate([np.arange(records[i]["lex_start"], records[i]["lex_end"])
                                for i in train_ids])
    mean = matrix[train_lex].mean(axis=0, dtype=np.float64)
    std = matrix[train_lex].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    x = ((matrix - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)

    weights, priors, weight_audits = {}, {}, {}
    for family in ("emission", "onset", "continuation"):
        weights[family], priors[family], weight_audits[family] = head_weight_map(
            records, train_ids, family)
    params = {name: np.zeros(DIM + 1, dtype=np.float32)
              for name in ("emission", "onset", "continuation")}
    moments = {name: np.zeros(DIM + 1, dtype=np.float32) for name in params}
    variances = {name: np.zeros(DIM + 1, dtype=np.float32) for name in params}
    steps = {name: 0 for name in params}
    initial = {name: full_loss(x, records, train_ids, name, weights[name], params[name])
               for name in params}
    history = []
    rng = np.random.Generator(np.random.PCG64(TRAIN_SEED + fold))
    for epoch in range(EPOCHS):
        order = rng.permutation(np.asarray(train_ids, dtype=np.int32))
        epoch_loss = defaultdict(list)
        for begin in range(0, len(order), BATCH_ANSWERS):
            batch = order[begin:begin + BATCH_ANSWERS]
            for name in params:
                gradient, loss, count = batch_gradient(x, records, batch, name, weights[name], params[name])
                if not count:
                    continue
                norm = float(np.linalg.norm(gradient))
                if norm > GRAD_CLIP:
                    gradient *= np.float32(GRAD_CLIP / norm)
                steps[name] += 1
                step = steps[name]
                moments[name] *= ADAM_BETA1
                moments[name] += (1.0 - ADAM_BETA1) * gradient
                variances[name] *= ADAM_BETA2
                variances[name] += (1.0 - ADAM_BETA2) * gradient * gradient
                corrected_m = moments[name] / (1.0 - ADAM_BETA1 ** step)
                corrected_v = variances[name] / (1.0 - ADAM_BETA2 ** step)
                params[name][:-1] *= np.float32(1.0 - LEARNING_RATE * WEIGHT_DECAY)
                params[name] -= (np.float32(LEARNING_RATE) * corrected_m /
                                 (np.sqrt(corrected_v) + ADAM_EPS))
                epoch_loss[name].append(loss)
        if epoch in {0, EPOCHS - 1}:
            history.append({"epoch": epoch + 1,
                            **{f"mean_batch_{name}_loss": float(np.mean(epoch_loss[name]))
                               for name in params}})
    final = {name: full_loss(x, records, train_ids, name, weights[name], params[name])
             for name in params}
    predictions = {}
    for answer_index in held_ids:
        row = records[answer_index]
        predictions[answer_index] = decode_heads(
            x[row["lex_start"]:row["lex_end"]], params, priors)
    model = {
        "fold": fold,
        "train_answers": len(train_ids),
        "held_answers": len(held_ids),
        "train_groups": len(train_groups),
        "held_groups": len(held_groups),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "parameters": {name: value.tolist() for name, value in params.items()},
        "natural_priors": priors,
        "weight_audits": weight_audits,
        "initial_weighted_losses": initial,
        "final_weighted_losses": final,
        "loss_history": history,
        "optimizer_steps": steps,
    }
    return predictions, model


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    positives = int(labels.sum())
    tp = fp = 0
    best_key = (0.0, 0.0, 0.0, float(np.nextafter(1.0, 2.0)))
    best = best_key[-1]
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        found = int(sorted_labels[index:end].sum())
        tp += found
        fp += end - index - found
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = (f1, precision, recall, float(sorted_scores[index]))
        if key > best_key:
            best_key = key
            best = float(sorted_scores[index])
        index = end
    return best


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {"count": len(labels), "positive": positives,
                "prevalence": positives / len(labels) if len(labels) else None,
                "average_precision": None, "auroc": None}
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
    previous_recall = 0.0
    average_precision = 0.0
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
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return {"count": len(labels), "positive": positives, "prevalence": positives / len(labels),
            "average_precision": average_precision, "auroc": auroc}


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.uint8)
    selected = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int(np.sum(selected & (labels == 1)))
    fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum(~selected & (labels == 1)))
    tn = int(np.sum(~selected & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            **ranking_metrics(labels, scores)}


def category_metrics(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    selected = scores >= threshold
    output = {}
    for name in ("released_onset", "internal_continuation", "clean"):
        mask = np.asarray([row["gold_category"] == name for row in rows], dtype=bool)
        rate = float(np.mean(selected[mask])) if mask.any() else None
        output[name] = {"count": int(mask.sum()), "selected": int(np.sum(selected & mask)),
                        "selection_rate": rate,
                        "metric_name": "false_positive_rate" if name == "clean" else "recall"}
    return output


def summary(values: list[float | None]) -> dict:
    clean = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not len(clean):
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": float(clean.mean()), "std": float(clean.std(ddof=0)),
            "min": float(clean.min()), "max": float(clean.max())}


def evaluate(records: list[dict], predictions: dict[int, dict], fold_models: list[dict]):
    token_rows, window_rows, answer_rows, span_rows = [], [], [], []
    head_y = {name: [] for name in ("emission", "onset", "continuation")}
    head_s = {name: [] for name in head_y}
    for answer_index, row in enumerate(records):
        pred = predictions[answer_index]
        raw_scores = {name: np.full(row["raw_count"], np.nan, dtype=np.float32)
                      for name in CANDIDATES}
        for name in CANDIDATES:
            raw_scores[name][row["lexical_raw"]] = pred[name]
        previous = np.r_[np.uint8(0), row["states"][:-1]]
        for local, raw in enumerate(row["lexical_raw"]):
            token_rows.append({
                "response_id": row["response_id"], "group_id": row["group_id"], "fold": row["fold"],
                "raw_bpe_index": int(raw), "lexical_index": local,
                "gold_risk": int(row["states"][local]), "released_onset": int(row["y_first"][local]),
                "binary_onset": int(row["states"][local] == 1 and previous[local] == 0),
                **{name: float(pred[name][local]) for name in (
                    "p_emit_balanced", "p_on_balanced", "p_cont_balanced",
                    "p_emit", "p_on", "p_cont", "anchored_filter", "emission_identity",
                    "transition_only", "forced_previous_risk_anchored")},
            })
        for family, mask, score_name in (
            ("emission", np.ones(len(previous), dtype=bool), "p_emit"),
            ("onset", previous == 0, "p_on"),
            ("continuation", previous == 1, "p_cont"),
        ):
            head_y[family].extend(row["states"][mask].tolist())
            head_s[family].extend(pred[score_name][mask].tolist())
        answer_windows = []
        onset_set = set(row["released_onset_raw"])
        for left, right in V1.windows_for_count(row["raw_count"]):
            lexical_raw = [index for index in range(left, right) if row["lexical_mask"][index]]
            if not lexical_raw:
                continue
            gold_risk = int(any(row["risk_mask"][index] for index in lexical_raw))
            is_onset = bool(gold_risk and any(index in onset_set for index in lexical_raw))
            category = "released_onset" if is_onset else "internal_continuation" if gold_risk else "clean"
            item = {"response_id": row["response_id"], "group_id": row["group_id"],
                    "fold": row["fold"], "raw_bpe_start": left, "raw_bpe_end": right,
                    "gold_risk": gold_risk, "gold_category": category,
                    **{name: float(np.nanmax(raw_scores[name][lexical_raw])) for name in CANDIDATES}}
            window_rows.append(item)
            answer_windows.append(item)
        answer_rows.append({"response_id": row["response_id"], "group_id": row["group_id"],
                            "fold": row["fold"], "gold_risk": row["answer_label"],
                            "eligible_windows": len(answer_windows),
                            **{name: max(item[name] for item in answer_windows) for name in CANDIDATES}})
        raw_to_local = {int(raw): local for local, raw in enumerate(row["lexical_raw"])}
        for span_index, mapping in enumerate(row["span_mappings"]):
            local = [raw_to_local[int(raw)] for raw in mapping if int(raw) in raw_to_local]
            span_rows.append({"response_id": row["response_id"], "group_id": row["group_id"],
                              "fold": row["fold"], "span_index": span_index,
                              "error_type": row["error_types"][span_index],
                              "lexical_bpe_count": len(local),
                              **{f"{name}_max": max(float(pred[name][i]) for i in local)
                                 for name in CANDIDATES},
                              **{f"{name}_min": min(float(pred[name][i]) for i in local)
                                 for name in CANDIDATES}})

    window_y = np.asarray([row["gold_risk"] for row in window_rows], dtype=np.uint8)
    answer_y = np.asarray([row["gold_risk"] for row in answer_rows], dtype=np.uint8)
    candidate_metrics = {}
    for name in CANDIDATES:
        window_s = np.asarray([row[name] for row in window_rows], dtype=np.float64)
        answer_s = np.asarray([row[name] for row in answer_rows], dtype=np.float64)
        window_threshold = choose_threshold(window_y, window_s)
        answer_threshold = choose_threshold(answer_y, answer_s)
        categories = category_metrics(window_rows, window_s, window_threshold)
        spans = []
        for row in span_rows:
            spans.append({"hit": row[f"{name}_max"] >= window_threshold,
                          "full": row[f"{name}_min"] >= window_threshold})
        folds = []
        for fold in range(FOLDS):
            window_mask = np.asarray([row["fold"] == fold for row in window_rows], dtype=bool)
            answer_mask = np.asarray([row["fold"] == fold for row in answer_rows], dtype=bool)
            fold_window_rows = [row for row in window_rows if row["fold"] == fold]
            folds.append({
                "fold": fold,
                "window": binary_metrics(window_y[window_mask], window_s[window_mask], window_threshold),
                "answer_max": binary_metrics(answer_y[answer_mask], answer_s[answer_mask], answer_threshold),
                "window_categories_at_pooled_threshold": category_metrics(
                    fold_window_rows, window_s[window_mask], window_threshold),
            })
        stability = {
            "window_f1": summary([item["window"]["f1"] for item in folds]),
            "window_precision": summary([item["window"]["precision"] for item in folds]),
            "window_recall": summary([item["window"]["recall"] for item in folds]),
            "window_auroc": summary([item["window"]["auroc"] for item in folds]),
            "window_average_precision": summary([item["window"]["average_precision"] for item in folds]),
            "answer_f1": summary([item["answer_max"]["f1"] for item in folds]),
            "answer_auroc": summary([item["answer_max"]["auroc"] for item in folds]),
            "clean_window_fpr": summary([
                item["window_categories_at_pooled_threshold"]["clean"]["selection_rate"] for item in folds]),
        }
        candidate_metrics[name] = {
            "role": "primary" if name == PRIMARY else
                    "identity/no-transition control" if name == "emission_identity" else "no-emission ablation",
            "window": binary_metrics(window_y, window_s, window_threshold),
            "answer_max": binary_metrics(answer_y, answer_s, answer_threshold),
            "window_categories_at_window_threshold": categories,
            "spans_at_window_threshold": {
                "count": len(spans), "any_hit": sum(item["hit"] for item in spans),
                "any_hit_rate": float(np.mean([item["hit"] for item in spans])),
                "full_coverage": sum(item["full"] for item in spans),
                "full_coverage_rate": float(np.mean([item["full"] for item in spans])),
            },
            "fold_metrics_at_pooled_oof_thresholds": folds,
            "five_fold_stability": stability,
        }

    heads = {}
    for name in ("emission", "onset", "continuation"):
        labels = np.asarray(head_y[name], dtype=np.uint8)
        scores = np.asarray(head_s[name], dtype=np.float64)
        threshold = choose_threshold(labels, scores)
        heads[name] = binary_metrics(labels, scores, threshold)
    stop_y = 1 - np.asarray(head_y["continuation"], dtype=np.uint8)
    stop_s = 1.0 - np.asarray(head_s["continuation"], dtype=np.float64)
    heads["stop"] = binary_metrics(stop_y, stop_s, choose_threshold(stop_y, stop_s))

    primary = candidate_metrics[PRIMARY]
    identity = candidate_metrics["emission_identity"]
    category = primary["window_categories_at_window_threshold"]
    folds_above = sum(item["window"]["auroc"] is not None and item["window"]["auroc"] > 0.55
                      for item in primary["fold_metrics_at_pooled_oof_thresholds"])
    checks = {
        "primary_window_f1_at_least_0_6902813989031736": primary["window"]["f1"] >= 0.6902813989031736,
        "primary_answer_f1_at_least_0_8910891089108911": primary["answer_max"]["f1"] >= 0.8910891089108911,
        "primary_answer_auroc_at_least_0_60": primary["answer_max"]["auroc"] >= 0.60,
        "released_onset_window_recall_at_least_0_50": category["released_onset"]["selection_rate"] >= 0.50,
        "internal_continuation_window_recall_at_least_0_50": category["internal_continuation"]["selection_rate"] >= 0.50,
        "clean_window_fpr_at_most_0_10": category["clean"]["selection_rate"] <= 0.10,
        "at_least_4_folds_window_auroc_above_0_55": folds_above >= 4,
        "anchored_window_f1_at_least_identity": primary["window"]["f1"] >= identity["window"]["f1"],
        "anchored_answer_auroc_at_least_identity": primary["answer_max"]["auroc"] >= identity["answer_max"]["auroc"],
    }
    clean_forced = np.asarray([row["forced_previous_risk_anchored"] for row in token_rows
                               if row["gold_risk"] == 0], dtype=np.float64)
    exit_audit = {
        "definition": "At every gold-clean lexical token, force previous risk probability to 1, then apply the continuation transition and current-token emission likelihood ratio.",
        "clean_tokens": len(clean_forced),
        "mean_forced_previous_risk_score": float(clean_forced.mean()),
        "p90_forced_previous_risk_score": float(np.quantile(clean_forced, 0.90)),
        "below_primary_window_threshold_rate": float(np.mean(clean_forced < primary["window"]["threshold"])),
        "analytic_zero_emission_limit": 0.0,
    }
    metrics = {
        "schema_version": "osr-emission-anchor-human-fit-group-oof-v2",
        "status": "COMPLETE",
        "scope": "RAGTruth human fit634 only; strict five-fold source-group OOF; CPU; no calibration/test",
        "primary_candidate": PRIMARY,
        "candidates": candidate_metrics,
        "heads_after_training_fold_prior_correction": heads,
        "fold_natural_priors": [{"fold": model["fold"], **model["natural_priors"]}
                                for model in fold_models],
        "false_state_exit_audit": exit_audit,
        "calibration_investment_gate": {
            "status": "PROCEED_TO_FROZEN_CALIBRATION_CONSIDERATION" if all(checks.values()) else
                      "STOP_BEFORE_CALIBRATION",
            "all_required": True,
            "checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
            "calibration_read": False,
        },
        "counts": {"answers": len(answer_rows),
                   "groups": len({row["group_id"] for row in records}),
                   "lexical_tokens": len(token_rows), "eligible_windows": len(window_rows),
                   "released_spans": len(span_rows)},
    }
    return metrics, token_rows, window_rows, answer_rows, span_rows


def selfcheck() -> dict:
    x = np.zeros((3, DIM), dtype=np.float32)
    params = {name: np.zeros(DIM + 1, dtype=np.float32)
              for name in ("emission", "onset", "continuation")}
    priors = {"emission": 0.1, "onset": 0.01, "continuation": 0.9}
    decoded = decode_heads(x, params, priors)
    if not all(decoded[name].shape == (3,) for name in CANDIDATES):
        raise AssertionError("decoder shape")
    strong_clean = params.copy()
    strong_clean = {name: value.copy() for name, value in params.items()}
    strong_clean["emission"][-1] = -40.0
    exited = decode_heads(x[:1], strong_clean, priors)["forced_previous_risk_anchored"][0]
    if exited > 1e-5:
        raise AssertionError("direct emission failed to exit forced false-risk state")
    labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    scores = np.asarray([0.1, 0.8, 0.8, 0.2], dtype=np.float64)
    metric = binary_metrics(labels, scores, choose_threshold(labels, scores))
    if not (0 <= metric["auroc"] <= 1 and 0 <= metric["average_precision"] <= 1):
        raise AssertionError("metric bounds")
    return {"status": "PASS", "device": "CPU", "gpu_framework_imported": False,
            "checks": ["three frozen decoder outputs", "direct emission false-state exit",
                       "threshold tie rule", "ranking metric bounds"]}


def run(verify_npz_hashes: bool) -> None:
    started = time.perf_counter()
    if json.loads(PROTOCOL.read_text(encoding="utf-8"))["status"] != "FROZEN_BEFORE_EXECUTION":
        raise AssertionError("protocol was not frozen before execution")
    matrix, records, source_provenance = V1.load_dataset(verify_npz_hashes)
    if matrix.shape != (139518, DIM) or len(records) != 634:
        raise AssertionError("unexpected fit cache geometry")
    predictions = {}
    models = []
    for fold in range(FOLDS):
        fold_predictions, model = train_fold(matrix, records, fold)
        if set(predictions) & set(fold_predictions):
            raise AssertionError("duplicate OOF prediction")
        predictions.update(fold_predictions)
        models.append(model)
        print("OSR_V2_FIT_OOF_FOLD", fold, "COMPLETE", flush=True)
    if set(predictions) != set(range(len(records))):
        raise AssertionError("incomplete OOF coverage")
    metrics, tokens, windows, answers, spans = evaluate(records, predictions, models)
    provenance = {
        **source_provenance,
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "reused_fit_loader_sha256": sha256_file(V1_RUNNER),
        "v1_protocol_sha256": sha256_file(V1_DIR / "PROTOCOL.json"),
        "npz_hashes_recomputed": verify_npz_hashes,
    }
    model_output = {
        "schema_version": "osr-emission-anchor-fold-models-v2",
        "protocol": {
            "model": "OSR-Emission-Anchor-68-v2",
            "primary": PRIMARY,
            "candidate_order": list(CANDIDATES),
            "folds": FOLDS, "input_dimension": DIM,
            "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
            "answer_batch_size": BATCH_ANSWERS, "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm_per_head": GRAD_CLIP,
            "terminal_rule": "no duplicated EOS stop representation",
            "calibration_or_test_labels_read": False,
            "gpu_started": False, "baseline_mutated": False,
        },
        "provenance": provenance,
        "folds": models,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "METRICS.json", metrics)
    atomic_json(OUT / "FOLD_MODELS.json", model_output)
    write_jsonl_gz(OUT / "TOKENS.jsonl.gz", tokens)
    write_jsonl_gz(OUT / "WINDOWS.jsonl.gz", windows)
    write_jsonl_gz(OUT / "ANSWERS.jsonl.gz", answers)
    write_jsonl_gz(OUT / "SPANS.jsonl.gz", spans)
    output_names = ["METRICS.json", "FOLD_MODELS.json", "TOKENS.jsonl.gz",
                    "WINDOWS.jsonl.gz", "ANSWERS.jsonl.gz", "SPANS.jsonl.gz"]
    manifest = {
        "schema_version": "osr-emission-anchor-fit-oof-manifest-v2",
        "status": "COMPLETE", "wall_seconds": time.perf_counter() - started,
        "outputs": {name: {"sha256": sha256_file(OUT / name),
                           "bytes": (OUT / name).stat().st_size} for name in output_names},
        "provenance": provenance, "selfcheck": selfcheck(),
        "calibration_or_test_labels_read": False,
        "gpu_started": False, "baseline_mutated": False,
    }
    atomic_json(OUT / "MANIFEST.json", manifest)
    print(json.dumps({"status": "COMPLETE", "primary": metrics["candidates"][PRIMARY],
                      "gate": metrics["calibration_investment_gate"],
                      "wall_seconds": manifest["wall_seconds"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selfcheck", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--skip-npz-hash-verification", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), indent=2))
    else:
        run(not args.skip_npz_hash_verification)


if __name__ == "__main__":
    main()
