"""Summarize the held-out API MAS evaluation with uncertainty intervals.

The prediction file is produced by ``evaluate_a2a_fusion_v1.py``.  This report
adds AUROC, AUPRC, ECE, and episode-bootstrap 95% intervals without touching
the underlying traces or labels.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def auc(scores: list[float], ys: list[int]) -> float:
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if a > b else .5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))


def auprc(scores: list[float], ys: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(ys)
    if positives == 0:
        return float("nan")
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    for i in order:
        if ys[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def stats(scores: list[float], ys: list[int], threshold: float = .5) -> dict[str, float]:
    pred = [int(s >= threshold) for s in scores]
    tp = sum(a == b == 1 for a, b in zip(pred, ys))
    tn = sum(a == b == 0 for a, b in zip(pred, ys))
    fp = sum(a == 1 and b == 0 for a, b in zip(pred, ys))
    fn = sum(a == 0 and b == 1 for a, b in zip(pred, ys))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = sum((s - y) ** 2 for s, y in zip(scores, ys)) / len(ys)
    bins = [[] for _ in range(10)]
    for s, y in zip(scores, ys):
        bins[min(9, int(s * 10))].append((s, y))
    ece = sum((len(bucket) / len(ys)) * abs(sum(s for s, _ in bucket) / len(bucket) - sum(y for _, y in bucket) / len(bucket)) for bucket in bins if bucket)
    return {"f1": f1, "precision": precision, "recall": recall, "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "brier": brier, "auroc": auc(scores, ys), "auprc": auprc(scores, ys), "ece": ece}


def percentile(values: list[float], q: float) -> float:
    values = sorted(v for v in values if v == v)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(len(values) - 1, lo + 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    ys = [int(row["y"]) for row in rows]
    model_names = ["layer1_bn", "layer2_rules", "flat_logistic", "strongest_feature", "fused"]
    rng = random.Random(20261002)
    result: dict[str, dict] = {"episodes": len(rows), "positive": sum(ys), "negative": len(ys) - sum(ys), "models": {}}
    for name in model_names:
        scores = [float(row[name]) for row in rows]
        point = stats(scores, ys)
        samples = {key: [] for key in point}
        for _ in range(max(0, args.bootstrap)):
            indices = [rng.randrange(len(rows)) for _ in rows]
            sample_scores = [scores[i] for i in indices]
            sample_y = [ys[i] for i in indices]
            item = stats(sample_scores, sample_y)
            for key, value in item.items():
                samples[key].append(value)
        intervals = {key: [percentile(values, .025), percentile(values, .975)] for key, values in samples.items()}
        result["models"][name] = {"point": point, "bootstrap_95": intervals}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
