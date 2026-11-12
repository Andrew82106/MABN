"""Group-OOF fusion evaluation for the MAS monitor.

The prior Fusion v3 trained its meta head on base predictions produced on the
same samples.  This script keeps an outer group holdout and trains the fusion
head only on inner group-OOF predictions, then refits the base components on
the full outer training partition.  It is slower but avoids meta-level
in-sample optimism.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def coverage(two, ep):
    parsed = two.event_features(ep)
    return float(sum(f[4] == 0 for _, _, f, _ in parsed["edges"]) / max(1, len(parsed["edges"])))


def fit_components(fair, two, eps, y, fit_idx, score_idx, seed):
    graph_x = np.asarray([fair.graph_features(e) for e in eps])
    graph_model = fair.fit_logistic(graph_x[fit_idx], y[fit_idx], seed + 1)
    p_global = fair.predict_logistic(graph_model, graph_x[score_idx])
    fit_eps = [eps[i] for i in fit_idx]
    score_eps = [eps[i] for i in score_idx]
    bn = two.fit_bn(fit_eps, y[fit_idx])
    theta, opt = two.fit_model(fit_eps, y[fit_idx], bn, seed + 2)
    p_local = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in score_eps])
    cov = np.asarray([coverage(two, e) for e in score_eps])
    return p_local, p_global, cov, opt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v2/traces_public.jsonl'))
    ap.add_argument("--labels", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v2/labels.jsonl'))
    ap.add_argument("--out", type=Path, default=Path('experiments/synthetic_mas/runs/fusion_oof_v1'))
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--inner-folds", type=int, default=3)
    args = ap.parse_args()

    fair = load("fair_fusion_oof", Path(__file__).with_name("evaluate_independent_fair_baselines_v1.py"))
    two = load("two_fusion_oof", Path(__file__).with_name("evaluate_two_layer_v2.py"))
    traces = read(args.input)
    labels = {r["episode_id"]: {"label": int(r["label"]), "scenario_family": r.get("scenario_family", "unknown")} for r in read(args.labels)}
    eps = [fair.public_episode(t, labels) for t in traces]
    y = np.asarray([e["label"] for e in eps], dtype=int)
    groups = np.asarray([e["scenario_family"] for e in eps])
    outer_splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=args.seed)
    outer_train, outer_test = next(outer_splitter.split(np.zeros(len(eps)), y, groups))

    # Inner group OOF predictions for the meta head.
    oof_local = np.full(len(outer_train), np.nan)
    oof_global = np.full(len(outer_train), np.nan)
    oof_cov = np.full(len(outer_train), np.nan)
    inner_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=args.inner_folds)
    optimizers = []
    for fold, (fit_pos, val_pos) in enumerate(splitter.split(np.zeros(len(outer_train)), y[outer_train], inner_groups), start=1):
        fit_idx = outer_train[fit_pos]
        val_idx = outer_train[val_pos]
        pl, pg, cv, opt = fit_components(fair, two, eps, y, fit_idx, val_idx, args.seed + fold * 100)
        oof_local[val_pos] = pl
        oof_global[val_pos] = pg
        oof_cov[val_pos] = cv
        optimizers.append(opt)
    if np.isnan(oof_local).any():
        raise AssertionError("inner OOF did not cover every outer-train episode")
    z_oof = np.c_[logit(oof_local), logit(oof_global), oof_cov]
    fusion_model = fair.fit_logistic(z_oof, y[outer_train], args.seed + 500)
    p_oof = fair.predict_logistic(fusion_model, z_oof)
    threshold = fair.best_threshold(y[outer_train], p_oof)

    # Refit both base components on all outer-train groups before scoring the
    # untouched outer test groups.
    p_local_test, p_global_test, cov_test, final_opt = fit_components(fair, two, eps, y, outer_train, outer_test, args.seed + 900)
    p_test = fair.predict_logistic(fusion_model, np.c_[logit(p_local_test), logit(p_global_test), cov_test])
    results = {
        "global_graph": fair.metrics(y[outer_test], p_global_test, fair.best_threshold(y[outer_train], fair.predict_logistic(fair.fit_logistic(np.asarray([fair.graph_features(e) for e in eps])[outer_train], y[outer_train], args.seed + 901), np.asarray([fair.graph_features(e) for e in eps])[outer_train]))),
        "local_message": fair.metrics(y[outer_test], p_local_test, fair.best_threshold(y[outer_train], oof_local)),
        "fusion_oof": fair.metrics(y[outer_test], p_test, threshold),
    }
    report = {
        "protocol": {"outer": "GroupShuffleSplit 80/20", "inner": f"{args.inner_folds}-fold GroupKFold OOF", "scenario_family_as_feature": False},
        "dataset": {"n": len(eps), "positive": int(y.sum()), "outer_train": int(len(outer_train)), "outer_test": int(len(outer_test))},
        "metrics": results,
        "fusion_parameters": [float(v) for v in fusion_model[0]],
        "optimizers": optimizers + [final_opt],
        "limitations": ["Synthetic hidden-effect benchmark.", "OOF removes meta-head in-sample fitting but does not establish external validity."],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Group-OOF Fusion v1", "", "融合头只在 inner group-OOF 预测上训练，外层测试组完全隔离。", "", "| method | F1 | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|"]
    for name, value in results.items():
        lines.append(f"| {name} | {value['f1']:.3f} | {value['auroc']:.3f} | {value['auprc']:.3f} | {value['brier']:.3f} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
