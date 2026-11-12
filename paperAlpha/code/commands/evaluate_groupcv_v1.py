"""Repeated group holdout for an honest runtime-only MAS comparison.

The previous fixed sorted-family split over-represented n=8 topologies.  This
script uses repeated GroupShuffleSplit over scenario families and evaluates the
same public runtime trace under two feature contracts:

* observed: all public runtime fields, including permission_mismatch;
* parity: permission_mismatch masked for every method.

The second contract is the fair comparison when permission mismatch is a
strong proxy for the hidden authorization mechanism.  No workflow_dag feature
is used here; that normative setting is evaluated separately by v3.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


FORBIDDEN_EVENT = {"label", "external_effect", "tool_side_effect", "hidden_fields", "label_source"}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(trace: dict) -> None:
    if any(k in trace for k in {"label", "external_effect", "hidden_fields", "label_source", "scenario_family"}):
        raise ValueError("public trace contains evaluation-only fields")
    for event in trace.get("events", []):
        leaked = FORBIDDEN_EVENT.intersection(event)
        if leaked:
            raise ValueError(f"public event contains evaluation-only fields: {sorted(leaked)}")


def mask_permission(trace: dict) -> dict:
    result = copy.deepcopy(trace)
    for event in result.get("events", []):
        event.pop("permission_mismatch", None)
        event.pop("permission_denied", None)
    return result


def summarize(rows: list[dict]) -> dict:
    methods = sorted({m for row in rows for m in row["metrics"]})
    out = {}
    for method in methods:
        vals = {key: np.asarray([r["metrics"][method][key] for r in rows], dtype=float) for key in ("f1", "auroc", "auprc", "brier", "ece_10bin")}
        out[method] = {key: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0} for key, v in vals.items()}
    return out


def one_split(fair, two, eps: list[dict], y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int) -> dict:
    results = {}
    graph_x = np.asarray([fair.graph_features(e) for e in eps])
    graph_model = fair.fit_logistic(graph_x[train], y[train], seed + 1)
    p_train = fair.predict_logistic(graph_model, graph_x[train])
    p_test = fair.predict_logistic(graph_model, graph_x[test])
    th = fair.best_threshold(y[train], p_train)
    results["graph_features_learned"] = fair.metrics(y[test], p_test, th)

    train_eps = [eps[i] for i in train]
    test_eps = [eps[i] for i in test]
    bn = two.fit_bn(train_eps, y[train])
    theta, opt = two.fit_model(train_eps, y[train], bn, seed + 2)
    p_train = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps])
    p_test = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in test_eps])
    th = fair.best_threshold(y[train], p_train)
    results["two_layer_bn_graph_v2"] = fair.metrics(y[test], p_test, th)

    return {"metrics": results, "optimizer": {"success": bool(opt["success"]), "iterations": int(opt["iterations"])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v2/traces_public.jsonl'))
    ap.add_argument("--labels", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v2/labels.jsonl'))
    ap.add_argument("--out", type=Path, default=Path('experiments/synthetic_mas/runs/groupcv_v1'))
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=.20)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    fair = load("fair_groupcv", Path(__file__).with_name("evaluate_independent_fair_baselines_v1.py"))
    two = load("two_groupcv", Path(__file__).with_name("evaluate_two_layer_v2.py"))
    traces = read_jsonl(args.input)
    labels = {r["episode_id"]: {"label": int(r["label"]), "scenario_family": r.get("scenario_family", "unknown")} for r in read_jsonl(args.labels)}
    for trace in traces:
        validate(trace)

    all_runs = []
    for mode in ("observed", "parity"):
        mode_rows = []
        mode_traces = traces if mode == "observed" else [mask_permission(t) for t in traces]
        eps = [fair.public_episode(t, labels) for t in mode_traces]
        y = np.asarray([e["label"] for e in eps], dtype=int)
        groups = np.asarray([e["scenario_family"] for e in eps])
        splitter = GroupShuffleSplit(n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed)
        for split_id, (train, test) in enumerate(splitter.split(np.zeros(len(eps)), y, groups), start=1):
            if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
                raise ValueError("a group split has only one class")
            result = one_split(fair, two, eps, y, train, test, args.seed + split_id * 100)
            mode_rows.append({"split": split_id, "train_count": int(len(train)), "test_count": int(len(test)), "train_groups": int(len(set(groups[train]))), "test_groups": int(len(set(groups[test]))), **result})
        all_runs.append({"mode": mode, "runs": mode_rows, "summary": summarize(mode_rows)})

    report = {
        "protocol": {
            "name": "repeated_scenario_family_group_holdout",
            "n_splits": args.n_splits,
            "test_size": args.test_size,
            "runtime_only": True,
            "workflow_dag_used": False,
            "threshold": "best training threshold per method and split",
        },
        "dataset": {"input": str(args.input), "labels": str(args.labels), "n": len(traces), "positive": int(sum(labels[t["episode_id"]]["label"] for t in traces))},
        "modes": all_runs,
        "interpretation": [
            "observed mode reflects the deployed public trace contract but permission_mismatch is a mechanism-specific proxy in this synthetic benchmark.",
            "parity mode masks that field for every method and is the fairer comparison of structural runtime evidence.",
            "This report does not evaluate the normative workflow_dag; use hierarchical_monitor_v3 for that separate setting.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Repeated group-CV runtime-only MAS evaluation", "", "随机按 scenario family 分组，避免同一拓扑族泄漏到 train/test；不使用 workflow_dag。", ""]
    for block in all_runs:
        lines += [f"## {block['mode']}", "", "| method | F1 mean±sd | AUROC mean±sd | AUPRC mean±sd | Brier mean±sd |", "|---|---:|---:|---:|---:|"]
        for method, vals in block["summary"].items():
            lines.append(f"| {method} | {vals['f1']['mean']:.3f}±{vals['f1']['sd']:.3f} | {vals['auroc']['mean']:.3f}±{vals['auroc']['sd']:.3f} | {vals['auprc']['mean']:.3f}±{vals['auprc']['sd']:.3f} | {vals['brier']['mean']:.3f}±{vals['brier']['sd']:.3f} |")
        lines.append("")
    lines += ["## Interpretation", "- `parity` is the primary fair ablation when permission mismatch is treated as a mechanism-specific proxy.", "- Results are runtime-only; normative workflow context is reported separately."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
