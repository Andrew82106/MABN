"""Counterfactual explanation audit for the frozen two-layer runtime model.

For each held-out episode, remove the highest-scoring observed path edge and
measure the risk-score decrease. This is a faithfulness diagnostic, not a
claim that the model discovered the true causal path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]


def load_journal():
    path = Path(__file__).with_name("evaluate_independent_mas_journal_v1.py")
    spec = importlib.util.spec_from_file_location("journal_explain_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_two():
    path = Path(__file__).with_name("evaluate_two_layer_v2.py")
    spec = importlib.util.spec_from_file_location("two_layer_explain", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remove_edge(ep: dict, source: str, target: str, sequence) -> dict:
    clone = json.loads(json.dumps(ep))
    kept = []
    for event in clone.get("events", []):
        same = event.get("event_type") in {"message_sent", "delegation"} and event.get("actor") == source and event.get("target") == target
        if sequence is not None:
            same = same and event.get("sequence") == sequence
        if not same:
            kept.append(event)
    clone["events"] = kept
    return clone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / 'experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl')
    ap.add_argument("--labels", type=Path, default=ROOT / 'experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl')
    ap.add_argument("--out", type=Path, default=ROOT / 'experiments/submission_audit/runs/final_eval/explanation_counterfactual_v1')
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    journal = load_journal(); two = load_two()
    traces = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {r["episode_id"]: r for r in (json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip())}
    y = np.asarray([int(labels[e["episode_id"]]["label"]) for e in traces], dtype=int)
    groups = np.asarray([labels[e["episode_id"]]["scenario_family"] for e in traces])
    records = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
        train_eps = [{**traces[i], "label": int(y[i])} for i in train]
        bn = two.fit_bn(train_eps, y[train])
        theta, _ = two.fit_model(train_eps, y[train], bn, args.seed + 100 + fold)
        for i in test:
            ep = traces[i]
            point, explanation = two._propagate(theta, two.event_features(ep), bn, explain=True)
            paths = explanation.get("top_paths") or []
            if not paths or not paths[0].get("edges"):
                records.append({"episode_id": ep["episode_id"], "label": int(y[i]), "fold": fold, "score": float(point), "counterfactual_score": float(point), "delta": 0.0, "has_path": False, "removed_edge": None})
                continue
            edge = paths[0]["edges"][0]
            cf = remove_edge(ep, edge.get("source"), edge.get("target"), edge.get("sequence"))
            cf_score = two._propagate(theta, two.event_features(cf), bn)
            records.append({"episode_id": ep["episode_id"], "label": int(y[i]), "fold": fold, "score": float(point), "counterfactual_score": float(cf_score), "delta": float(point - cf_score), "has_path": True, "removed_edge": {"source": edge.get("source"), "target": edge.get("target"), "sequence": edge.get("sequence")}})
    arr = np.asarray([r["delta"] for r in records], dtype=float)
    pos = np.asarray([r["label"] for r in records], dtype=bool)
    result = {"schema": "independent-mas-explanation-counterfactual-v1", "n": len(records), "path_found_rate": float(np.mean([r["has_path"] for r in records])), "mean_score_drop": float(arr.mean()), "median_score_drop": float(np.median(arr)), "positive_mean_score_drop": float(arr[pos].mean()), "negative_mean_score_drop": float(arr[~pos].mean()), "positive_drop_rate": float(np.mean(arr[pos] > 0)), "negative_drop_rate": float(np.mean(arr[~pos] > 0)), "interpretation": "Removing the highest-scoring observed path edge changes the monitor score; this is a faithfulness diagnostic, not causal ground truth."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "REPORT.json").write_text(json.dumps({**result, "records": records}, indent=2), encoding="utf-8")
    (args.out / "predictions.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8")
    (args.out / "REPORT.md").write_text("\n".join(["# Independent MAS explanation counterfactual audit", "", f"- n={result['n']}", f"- top observed path found: {result['path_found_rate']:.3f}", f"- mean risk-score drop after deleting top edge: {result['mean_score_drop']:.3f}", f"- positive episodes mean drop: {result['positive_mean_score_drop']:.3f}", f"- negative episodes mean drop: {result['negative_mean_score_drop']:.3f}", f"- positive drop rate: {result['positive_drop_rate']:.3f}", "", "This measures score faithfulness to the monitor's own path explanation; it is not causal discovery or ground-truth attribution."]) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
