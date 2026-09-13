#!/usr/bin/env python3
"""Independent CPU verification of saved OSR human-only OOF geometry/results."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
OUT = HERE / "human_only_oof_v1"
ANSWERS = BENCH / "data" / "answers_fit.jsonl"
TOKENS = BENCH / "data" / "tokens_fit.jsonl"
NAMESPACE = "ragtruth_fit_human"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if path.name not in {"answers_fit.jsonl", "tokens_fit.jsonl"}:
        raise AssertionError("non-fit source")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("partition") != "fit" for row in rows):
        raise AssertionError("non-fit row")
    return rows


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold_for(group: str) -> int:
    value = hashlib.sha256(f"{NAMESPACE}\0{group}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") % 5


def windows(count: int):
    return [(left, min(left + 4, count)) for left in range(max(1, count - 3))]


def choose(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    total = int(labels.sum()); tp = fp = 0
    best = (0.0, 0.0, 0.0, float(np.nextafter(1.0, 2.0)))
    at = best[-1]; index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        found = int(y[index:end].sum()); tp += found; fp += end - index - found
        recall = tp / total; precision = tp / (tp + fp)
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        candidate = (f1, precision, recall, float(s[index]))
        if candidate > best:
            best = candidate; at = float(s[index])
        index = end
    return at


def ranking(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum()); negatives = len(labels) - positives
    order = np.argsort(scores, kind="stable"); s, y = scores[order], labels[order]
    neg_before = 0; concordance = 0.0; index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        pg = int(y[index:end].sum()); ng = end - index - pg
        concordance += pg * (neg_before + 0.5 * ng); neg_before += ng; index = end
    auc = concordance / (positives * negatives)
    order = np.argsort(-scores, kind="stable"); s, y = scores[order], labels[order]
    tp = fp = 0; previous = ap = 0.0; index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        found = int(y[index:end].sum()); tp += found; fp += end - index - found
        recall = tp / positives; ap += (recall - previous) * tp / (tp + fp)
        previous = recall; index = end
    return ap, auc


def metric(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    selected = scores >= threshold
    tp = int(np.sum(selected & (labels == 1))); fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum(~selected & (labels == 1))); tn = int(np.sum(~selected & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap, auc = ranking(labels, scores)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "count": len(labels), "positive": int(labels.sum()),
            "prevalence": float(labels.mean()), "average_precision": ap, "auroc": auc}


def close(actual, expected, name: str, tolerance: float = 2e-12) -> None:
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
        if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise AssertionError(f"output hash/size mismatch: {name}")
    answers = {str(row["response_id"]): row for row in read_jsonl(ANSWERS)}
    token_gold = {str(row["response_id"]): row for row in read_jsonl(TOKENS)}
    token_pred = read_gzip(OUT / "TOKENS.jsonl.gz")
    window_pred = read_gzip(OUT / "WINDOWS.jsonl.gz")
    answer_pred = read_gzip(OUT / "ANSWERS.jsonl.gz")
    span_pred = read_gzip(OUT / "SPANS.jsonl.gz")
    if not (len(answers) == len(token_gold) == len(answer_pred) == 634):
        raise AssertionError("answer completeness failure")
    if len(token_pred) != 139518 or len(window_pred) != 168123 or len(span_pred) != 646:
        raise AssertionError("prediction completeness failure")

    token_by_answer = {}
    for row in token_pred:
        rid = row["response_id"]; gold = token_gold[rid]
        if row["group_id"] != gold["group_id"] or row["fold"] != fold_for(gold["group_id"]):
            raise AssertionError("token group/fold mismatch")
        raw = int(row["raw_bpe_index"])
        if not gold["lexical_mask"][raw] or row["gold_risk"] != gold["risk_mask"][raw]:
            raise AssertionError("token label mismatch")
        if not all(0 <= float(row[key]) <= 1 for key in ("p_on", "p_cont", "risk_score")):
            raise AssertionError("probability out of range")
        token_by_answer.setdefault(rid, {})[raw] = row
    for rid, gold in token_gold.items():
        expected = {i for i, value in enumerate(gold["lexical_mask"]) if value}
        if set(token_by_answer.get(rid, {})) != expected:
            raise AssertionError("lexical token coverage mismatch")

    saved_windows = {(row["response_id"], int(row["raw_bpe_start"])): row for row in window_pred}
    if len(saved_windows) != len(window_pred):
        raise AssertionError("duplicate saved window")
    category_counts = {"released_onset": 0, "internal_continuation": 0, "clean": 0}
    for rid, gold in token_gold.items():
        onset = {int(item["risk_token_indices"][0]) for item in gold["span_token_mapping"]
                 if item["risk_token_indices"]}
        for left, right in windows(len(gold["lexical_mask"])):
            lexical = [i for i in range(left, right) if gold["lexical_mask"][i]]
            if not lexical:
                continue
            row = saved_windows.pop((rid, left))
            expected_score = max(token_by_answer[rid][i]["risk_score"] for i in lexical)
            close(row["score"], expected_score, "window score", 0.0)
            risk = int(any(gold["risk_mask"][i] for i in lexical))
            category = "released_onset" if risk and any(i in onset for i in lexical) else \
                       "internal_continuation" if risk else "clean"
            if row["gold_risk"] != risk or row["gold_category"] != category:
                raise AssertionError("window gold geometry mismatch")
            category_counts[category] += 1
    if saved_windows:
        raise AssertionError("unexpected saved windows")

    answer_by_id = {row["response_id"]: row for row in answer_pred}
    window_scores_by_id = {}
    for row in window_pred:
        window_scores_by_id.setdefault(row["response_id"], []).append(row["score"])
    for rid, row in answer_by_id.items():
        close(row["score"], max(window_scores_by_id[rid]), "answer max", 0.0)
        if row["gold_risk"] != answers[rid]["label"] or row["fold"] != fold_for(answers[rid]["group_id"]):
            raise AssertionError("answer gold/fold mismatch")
    groups = {}
    for row in answer_pred:
        groups.setdefault(row["group_id"], set()).add(row["fold"])
    if any(len(value) != 1 for value in groups.values()) or len(groups) != 615:
        raise AssertionError("group split failure")

    wy = np.asarray([row["gold_risk"] for row in window_pred], dtype=np.uint8)
    ws = np.asarray([row["score"] for row in window_pred], dtype=np.float64)
    ay = np.asarray([row["gold_risk"] for row in answer_pred], dtype=np.uint8)
    ass = np.asarray([row["score"] for row in answer_pred], dtype=np.float64)
    wt, at = choose(wy, ws), choose(ay, ass)
    close(wt, metrics["window"]["threshold"], "window threshold", 0.0)
    close(at, metrics["answer_max"]["threshold"], "answer threshold", 0.0)
    for scope, reproduced, saved in (("window", metric(wy, ws, wt), metrics["window"]),
                                     ("answer", metric(ay, ass, at), metrics["answer_max"])):
        for key, expected in saved.items():
            close(reproduced[key], expected, f"{scope}.{key}")
    if category_counts != {name: item["count"] for name, item in
                           metrics["window_categories_at_window_threshold"].items()}:
        raise AssertionError("window category counts mismatch")
    if len(models["folds"]) != 5 or {row["fold"] for row in models["folds"]} != set(range(5)):
        raise AssertionError("fold model completeness failure")
    for fold in models["folds"]:
        if not (len(fold["scaler_mean"]) == len(fold["scaler_std"]) == 68 and
                len(fold["onset_parameters"]) == len(fold["continuation_parameters"]) == 69):
            raise AssertionError("fold model shape mismatch")
        for family in fold["initial_weighted_losses"]:
            if fold["final_weighted_losses"][family] >= fold["initial_weighted_losses"][family]:
                raise AssertionError("training loss did not improve")

    result = {
        "schema_version": "osr-human-only-oof-independent-verification-v1",
        "status": "PASS",
        "outputs_hash_verified": len(manifest["outputs"]),
        "answers": len(answer_pred), "groups": len(groups), "lexical_tokens": len(token_pred),
        "eligible_windows": len(window_pred), "released_spans": len(span_pred),
        "group_folds_recomputed": True, "raw_4bpe_geometry_reconstructed": True,
        "token_labels_reconstructed_from_fit": True, "answer_max_exact": True,
        "thresholds_and_primary_metrics_reproduced": True,
        "fold_model_shapes_and_loss_descent_checked": True,
        "calibration_or_test_labels_read": False, "gpu_started": False,
    }
    pending = OUT / "VERIFY.json.pending"
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(OUT / "VERIFY.json")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
