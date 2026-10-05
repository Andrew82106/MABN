"""Final grouped evaluation for the hierarchical MAS risk monitor.

The public trace contains the normative workflow DAG and the observed runtime
events.  The hidden labels are joined only for fitting/evaluation.  All model
features are extracted before the label join, and topology families are split
as groups so the same family cannot occur in train and test for a seed.

This script writes one compact result directory intended for the submission
package.  It is an offline evaluation; no API call is made here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grouped_split(eps: list[dict], y: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray, dict]:
    groups = np.asarray(sorted({str(e["scenario_family"]) for e in eps}), dtype=object)
    n_hold = max(1, int(round(len(groups) * fraction)))
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        order = rng.permutation(groups)
        held = set(str(x) for x in order[:n_hold])
        test = np.asarray([i for i, e in enumerate(eps) if str(e["scenario_family"]) in held], dtype=int)
        train = np.asarray([i for i, e in enumerate(eps) if str(e["scenario_family"]) not in held], dtype=int)
        if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2:
            return train, test, {
                "type": "random_scenario_family_group_holdout",
                "seed": seed,
                "attempt": attempt,
                "holdout_fraction": fraction,
                "groups_total": int(len(groups)),
                "groups_train": int(len(groups) - len(held)),
                "groups_test": int(len(held)),
                "held_out_groups": sorted(held),
                "train_count": int(len(train)),
                "test_count": int(len(test)),
                "train_positive": int(y[train].sum()),
                "test_positive": int(y[test].sum()),
            }
    raise RuntimeError(f"could not produce a two-class grouped split for seed {seed}")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def metric(fair, y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    out = fair.metrics(y, p, threshold)
    out["ece_10bin"] = ece(y, p)
    out["n"] = int(len(y))
    out["positive"] = int(y.sum())
    return out


def taint_path_score(fair, ep: dict) -> float:
    _, edges, sources, sinks = fair.graph_parts(ep)
    seen, _ = fair.reachable(sources, sinks, edges, taint_only=True)
    return float(bool(set(sinks).intersection(seen)))


def knowledge_template_features(ep: dict) -> np.ndarray:
    """Transparent, pre-effect knowledge-template observables.

    These are not hidden mechanism labels.  They encode two literature-backed
    monitor questions: does an observed edge cross the declared workflow, and
    does a privileged action have an explicit authorization mismatch?  The
    quantities are normalized only for numerical conditioning; no fitted
    parameter or label is used here.
    """
    expected = {(str(u), str(v)) for u, v in (ep.get("workflow_dag") or {}).get("edges", []) if u and v and u != v}
    events = ep.get("events", [])
    runtime = [e for e in events if e.get("event_type") in {"message_sent", "delegation"} and e.get("actor") and e.get("target")]
    runtime_pairs = {(str(e["actor"]), str(e["target"])) for e in runtime}
    unexpected = [e for e in runtime if (str(e["actor"]), str(e["target"])) not in expected]
    sources = {str(e["actor"]) for e in events if e.get("event_type") == "input_received" and e.get("data_class") == "untrusted" and e.get("actor")}
    privileged_tools = {"payments.write", "filesystem.write", "shell.exec"}
    sinks = {str(e["actor"]) for e in events if e.get("event_type") == "tool_call" and e.get("tool") in privileged_tools}
    mismatches = [e for e in events if e.get("event_type") == "tool_call" and bool(e.get("permission_mismatch") or e.get("permission_denied"))]
    unexpected_to_sink = [e for e in unexpected if str(e["target"]) in sinks]
    unexpected_from_source = [e for e in unexpected if str(e["actor"]) in sources]
    untrusted_unexpected = [e for e in unexpected if e.get("data_class") == "untrusted"]
    uncertain_unexpected = [e for e in unexpected if e.get("data_class") in {None, "unknown", "redacted"} or float(e.get("edge_confidence", 1.0) or 0.0) <= 0.0]
    missing = expected - runtime_pairs
    n_edges = max(1, len(runtime))
    n_expected = max(1, len(expected))
    return np.asarray([
        min(len(mismatches) / 2.0, 1.0),
        float(bool(mismatches)),
        min(len(unexpected) / n_edges, 1.0),
        min(len(unexpected_to_sink) / n_edges, 1.0),
        min(len(unexpected_from_source) / n_edges, 1.0),
        min(len(untrusted_unexpected) / n_edges, 1.0),
        min(len(uncertain_unexpected) / n_edges, 1.0),
        min(len(missing) / n_expected, 1.0),
        float(bool(sources.intersection(sinks))),
        min(len(sinks) / 3.0, 1.0),
    ], dtype=float)


def grouped_kfold(eps: list[dict], indices: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return non-overlapping group folds for out-of-fold component scores."""
    groups = sorted({str(eps[int(i)]["scenario_family"]) for i in indices})
    rng = np.random.default_rng(seed)
    groups = list(np.asarray(groups, dtype=object)[rng.permutation(len(groups))])
    buckets = {g: j % folds for j, g in enumerate(groups)}
    out = []
    for fold in range(folds):
        valid = np.asarray([int(i) for i in indices if buckets[str(eps[int(i)]["scenario_family"])] == fold], dtype=int)
        train = np.asarray([int(i) for i in indices if buckets[str(eps[int(i)]["scenario_family"])] != fold], dtype=int)
        if len(valid) and len(train):
            out.append((train, valid))
    return out


def component_scores(eps: list[dict], y: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray,
                     fair, two, hier, seed: int, wf_all: np.ndarray, kt_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit the two learned component monitors on fit_idx and score eval_idx."""
    fit_eps = [eps[int(i)] for i in fit_idx]
    eval_eps = [eps[int(i)] for i in eval_idx]
    yfit = y[fit_idx]
    bn = two.fit_bn(fit_eps, yfit)
    theta, opt = two.fit_model(fit_eps, yfit, bn, seed)
    p_two = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in eval_eps], dtype=float)
    wf_model = fair.fit_logistic(wf_all[fit_idx], yfit, seed + 1)
    p_wf = fair.predict_logistic(wf_model, wf_all[eval_idx])
    return p_two, p_wf, eval_idx, {"two_layer": opt, "theta": theta, "bn": bn}


def fusion_features(p_two: np.ndarray, p_wf: np.ndarray, wf: np.ndarray, kt: np.ndarray) -> np.ndarray:
    logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1 - 1e-5))
    return np.c_[logit(p_two), logit(p_wf), wf[:, 3], wf[:, 4], wf[:, 5], wf[:, 7], kt]


def run_seed_strict(eps: list[dict], y: np.ndarray, seed: int, fraction: float, fair, two, hier) -> dict:
    """Nested group evaluation with fold-out fusion and a separate calibration group."""
    outer_train, outer_test, split = grouped_split(eps, y, seed, fraction)
    wf_all = np.asarray([hier.workflow_features(e) for e in eps])
    kt_all = np.asarray([knowledge_template_features(e) for e in eps])

    # The ordinary baselines are fitted only on the outer training groups.
    scores: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    feature_sets = {
        "flat_event_logistic": np.asarray([fair.aggregate_features(e) for e in eps]),
        "single_agent_local": np.asarray([fair.local_features(e) for e in eps]),
        "runtime_graph_logistic": np.asarray([fair.graph_features(e) for e in eps]),
        "knowledge_template_only": kt_all,
    }
    ytr, yte = y[outer_train], y[outer_test]
    for offset, (name, features) in enumerate(feature_sets.items()):
        p_tr, p_te, th = fit_logistic(fair, features[outer_train], ytr, features[outer_test], seed + offset)
        scores[name] = (p_tr, p_te, th)

    # Keep the external message-fusion baseline as a direct model, not an
    # in-sample score fusion, so it has the same outer information boundary.
    unified = load_module("strict_unified", SCRIPTS / "evaluate_unified_monitor_v2.py")
    from scipy.optimize import minimize
    train_eps = [eps[int(i)] for i in outer_train]
    test_eps = [eps[int(i)] for i in outer_test]
    theta0 = np.zeros(14)
    theta0[:7] = np.random.default_rng(seed).normal(0, 0.1, 7)
    theta0[7:11] = np.array([-1.0, 2.0, 0.5, 0.5])
    theta0[11:] = [-2.0, 4.0, 0.5]
    fit = minimize(unified.loss, theta0, args=(train_eps, ytr), method="L-BFGS-B",
                   bounds=[(-8, 8)] * 14, options={"maxiter": 500, "ftol": 1e-10})
    p_tr = np.asarray([unified.forward(fit.x, e) for e in train_eps])
    p_te = np.asarray([unified.forward(fit.x, e) for e in test_eps])
    scores["unified_message_baseline"] = (p_tr, p_te, fair.best_threshold(ytr, p_tr))

    # Transparent path rule is not learned and therefore has no calibration fit.
    rule_tr = np.asarray([taint_path_score(fair, e) for e in train_eps])
    rule_te = np.asarray([taint_path_score(fair, e) for e in test_eps])
    scores["taint_path_rule"] = (rule_tr, rule_te, 0.5)

    # Reserve whole topology groups for calibration.  The fusion model only
    # sees component predictions produced out-of-fold on the remaining groups.
    sub_eps = [eps[int(i)] for i in outer_train]
    _, calib_rel, _ = grouped_split(sub_eps, ytr, seed + 700, 0.25)
    calib_idx = outer_train[calib_rel]
    calib_set = set(int(i) for i in calib_idx)
    fusion_fit_idx = np.asarray([int(i) for i in outer_train if int(i) not in calib_set], dtype=int)
    oof_x, oof_y = [], []
    oof_meta = []
    for fold, (inner_fit, inner_valid) in enumerate(grouped_kfold(eps, fusion_fit_idx, 3, seed + 710)):
        p2, pw, valid_idx, opt = component_scores(eps, y, inner_fit, inner_valid, fair, two, hier,
                                                   seed + 720 + fold * 10, wf_all, kt_all)
        oof_x.append(fusion_features(p2, pw, wf_all[valid_idx], kt_all[valid_idx]))
        oof_y.append(y[valid_idx])
        oof_meta.append(opt)
    oof_x = np.vstack(oof_x)
    oof_y = np.concatenate(oof_y)
    fusion_model = fair.fit_logistic(oof_x, oof_y, seed + 760)
    p_cal_2, p_cal_w, cal_idx, cal_opt = component_scores(eps, y, fusion_fit_idx, calib_idx, fair, two, hier,
                                                           seed + 770, wf_all, kt_all)
    cal_x = fusion_features(p_cal_2, p_cal_w, wf_all[cal_idx], kt_all[cal_idx])
    raw_cal = fair.predict_logistic(fusion_model, cal_x)
    raw_test_2, raw_test_w, test_idx, test_opt = component_scores(eps, y, outer_train, outer_test, fair, two, hier,
                                                                    seed + 780, wf_all, kt_all)
    p_two_train = np.asarray([two._propagate(test_opt["theta"], two.event_features(e), test_opt["bn"])
                              for e in train_eps], dtype=float)
    scores["two_layer_bn_runtime"] = (p_two_train, raw_test_2, fair.best_threshold(ytr, p_two_train))
    test_x = fusion_features(raw_test_2, raw_test_w, wf_all[test_idx], kt_all[test_idx])
    raw_test = fair.predict_logistic(fusion_model, test_x)
    if len(np.unique(y[calib_idx])) == 2:
        # Fit a one-dimensional Platt calibrator only on untouched calibration
        # groups; the outer test is never used to choose this mapping.
        from sklearn.linear_model import LogisticRegression
        cal_model = LogisticRegression(C=1e6, max_iter=1000, random_state=seed + 790).fit(
            np.log(np.clip(raw_cal, 1e-5, 1 - 1e-5)).reshape(-1, 1), y[calib_idx])
        p_cal = cal_model.predict_proba(np.log(np.clip(raw_cal, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
        p_test = cal_model.predict_proba(np.log(np.clip(raw_test, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
    else:
        cal_model = None
        p_cal, p_test = raw_cal, raw_test
    threshold = fair.best_threshold(y[calib_idx], p_cal)
    scores["ours_hierarchical_bn_runtime"] = (p_cal, p_test, threshold)
    metrics = {name: metric(fair, yte, pte, threshold)
               for name, (_, pte, threshold) in scores.items()}
    predictions = {
        "episode_id": [eps[int(i)]["episode_id"] for i in outer_test],
        "label": yte.astype(int).tolist(),
        "mechanism": [eps[int(i)].get("mechanism", "unknown") for i in outer_test],
        "scores": {name: pte.astype(float).tolist() for name, (_, pte, _) in scores.items()},
    }
    return {"seed": seed, "split": split, "metrics": metrics, "predictions": predictions,
            "optimizer": {"unified_success": bool(fit.success), "unified_iterations": int(fit.nit),
                          "strict_fusion": {"outer_train": int(len(outer_train)), "fusion_fit": int(len(fusion_fit_idx)),
                                             "calibration": int(len(calib_idx)), "oof_rows": int(len(oof_y)),
                                             "inner_folds": len(oof_meta), "calibrator": "platt_on_group_holdout" if cal_model else "identity"},
                          "component_fit_diagnostics": {"calibration": cal_opt["two_layer"], "test": test_opt["two_layer"]}}}


def fit_logistic(fair, x_train, y_train, x_test, seed):
    model = fair.fit_logistic(x_train, y_train, seed)
    p_train = fair.predict_logistic(model, x_train)
    p_test = fair.predict_logistic(model, x_test)
    return p_train, p_test, fair.best_threshold(y_train, p_train)


def run_seed(eps: list[dict], y: np.ndarray, seed: int, fraction: float, fair, two, hier) -> dict:
    train, test, split = grouped_split(eps, y, seed, fraction)
    tr = [eps[i] for i in train]
    te = [eps[i] for i in test]
    ytr, yte = y[train], y[test]
    scores: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    feature_sets = {
        "flat_event_logistic": np.asarray([fair.aggregate_features(e) for e in eps]),
        "single_agent_local": np.asarray([fair.local_features(e) for e in eps]),
        "runtime_graph_logistic": np.asarray([fair.graph_features(e) for e in eps]),
        "knowledge_template_only": np.asarray([knowledge_template_features(e) for e in eps]),
    }
    for offset, (name, features) in enumerate(feature_sets.items()):
        scores[name] = fit_logistic(fair, features[train], ytr, features[test], seed + offset)

    # Existing unconstrained fusion baseline.
    unified = load_module("final_unified", SCRIPTS / "evaluate_unified_monitor_v2.py")
    theta0 = np.zeros(14)
    theta0[:7] = np.random.default_rng(seed).normal(0, 0.1, 7)
    theta0[7:11] = np.array([-1.0, 2.0, 0.5, 0.5])
    theta0[11:] = [-2.0, 4.0, 0.5]
    from scipy.optimize import minimize
    fit = minimize(unified.loss, theta0, args=(tr, ytr), method="L-BFGS-B",
                   bounds=[(-8, 8)] * 14, options={"maxiter": 500, "ftol": 1e-10})
    p_train = np.asarray([unified.forward(fit.x, e) for e in tr])
    p_test = np.asarray([unified.forward(fit.x, e) for e in te])
    scores["unified_message_baseline"] = (p_train, p_test, fair.best_threshold(ytr, p_train))

    # The published two-layer BN + runtime-message model.
    bn = two.fit_bn(tr, ytr)
    theta, opt = two.fit_model(tr, ytr, bn, seed + 100)
    p_train = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in tr])
    p_test = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in te])
    scores["two_layer_bn_runtime"] = (p_train, p_test, fair.best_threshold(ytr, p_train))

    # Normative workflow context and the proposed hierarchical fusion.
    wf_all = np.asarray([hier.workflow_features(e) for e in eps])
    wf_train, wf_test = wf_all[train], wf_all[test]
    wf_tr, wf_te, wf_th = fit_logistic(fair, wf_train, ytr, wf_test, seed + 120)
    scores["workflow_only"] = (wf_tr, wf_te, wf_th)
    logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1 - 1e-5))
    kt_all = np.asarray([knowledge_template_features(e) for e in eps])
    fusion_train = np.c_[logit(p_train), logit(wf_tr), wf_train[:, 3], wf_train[:, 4], wf_train[:, 5], wf_train[:, 7], kt_all[train]]
    fusion_test = np.c_[logit(p_test), logit(wf_te), wf_test[:, 3], wf_test[:, 4], wf_test[:, 5], wf_test[:, 7], kt_all[test]]
    fu_tr, fu_te, fu_th = fit_logistic(fair, fusion_train, ytr, fusion_test, seed + 121)
    scores["ours_hierarchical_bn_runtime"] = (fu_tr, fu_te, fu_th)

    # Transparent non-learning reference used by graph-based MAS monitors.
    rule_train = np.asarray([taint_path_score(fair, e) for e in tr])
    rule_test = np.asarray([taint_path_score(fair, e) for e in te])
    scores["taint_path_rule"] = (rule_train, rule_test, 0.5)

    metrics = {name: metric(fair, yte, pte, threshold)
               for name, (_, pte, threshold) in scores.items()}
    predictions = {
        "episode_id": [e["episode_id"] for e in te],
        "label": yte.astype(int).tolist(),
        "mechanism": [e.get("mechanism", "unknown") for e in te],
        "scores": {name: pte.astype(float).tolist() for name, (_, pte, _) in scores.items()},
    }
    return {"seed": seed, "split": split, "metrics": metrics, "predictions": predictions,
            "optimizer": {"two_layer": opt, "unified_success": bool(fit.success), "unified_iterations": int(fit.nit)}}


def mechanism_report(runs: list[dict], method: str) -> dict:
    rows = []
    for run in runs:
        pred = run["predictions"]
        scores = pred["scores"][method]
        rows.extend({"label": int(y), "score": float(p), "mechanism": m}
                     for y, p, m in zip(pred["label"], scores, pred["mechanism"]))
    out = {}
    for mechanism in sorted({r["mechanism"] for r in rows}):
        group = [r for r in rows if r["mechanism"] == mechanism]
        y = np.asarray([r["label"] for r in group], dtype=int)
        p = np.asarray([r["score"] for r in group], dtype=float)
        # Use a compact mechanism report; benign has no positive class.
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
        out[mechanism] = {
            "n": int(len(group)), "positive": int(y.sum()),
            "positive_rate": float(y.mean()) if len(y) else None,
            "f1_at_0_5": float(f1_score(y, p >= .5, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "auprc": float(average_precision_score(y, p)) if len(y) and y.sum() else None,
        }
    return out


def aggregate(runs: list[dict]) -> dict:
    methods = list(runs[0]["metrics"])
    summary = {}
    for method in methods:
        rows = [run["metrics"][method] for run in runs]
        summary[method] = {key: {
            "mean": float(np.mean([r[key] for r in rows if r.get(key) is not None])),
            "sd": float(np.std([r[key] for r in rows if r.get(key) is not None], ddof=1)) if len([r for r in rows if r.get(key) is not None]) > 1 else 0.0,
            "per_seed": [r.get(key) for r in rows],
        } for key in ("f1", "precision", "recall", "auroc", "auprc", "brier", "ece_10bin")}
    ours = "ours_hierarchical_bn_runtime"
    deltas = {}
    for method in methods:
        if method == ours:
            continue
        f1_delta = np.asarray([r["metrics"][ours]["f1"] - r["metrics"][method]["f1"] for r in runs])
        auc_delta = np.asarray([r["metrics"][ours]["auroc"] - r["metrics"][method]["auroc"] for r in runs])
        brier_delta = np.asarray([r["metrics"][ours]["brier"] - r["metrics"][method]["brier"] for r in runs])
        deltas[method] = {
            "f1_mean_delta": float(f1_delta.mean()), "f1_per_seed": f1_delta.tolist(),
            "auroc_mean_delta": float(auc_delta.mean()), "auroc_per_seed": auc_delta.tolist(),
            "brier_mean_delta": float(brier_delta.mean()), "brier_per_seed": brier_delta.tolist(),
            "paired_wilcoxon_f1_p": float(wilcoxon(f1_delta).pvalue) if np.any(f1_delta) else 1.0,
        }
    return {"mean_sd": summary, "paired_deltas_vs_ours": deltas}


def write_report(out: Path, report: dict) -> None:
    methods = report["aggregate"]["mean_sd"]
    lines = [
        "# Final grouped evaluation: hierarchical MAS risk monitor", "",
        "数据为 independent_mas_v3 的公开轨迹；规范 workflow DAG 与运行时事件图分开输入。按 topology family 做分组留出；主模型的融合头采用组级折外训练，并在独立校准组选择阈值。", "",
        "| 方法 | F1 | AUROC | AUPRC | Brier | ECE |", "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["single_agent_local", "flat_event_logistic", "runtime_graph_logistic", "knowledge_template_only", "taint_path_rule",
             "unified_message_baseline", "two_layer_bn_runtime", "workflow_only", "ours_hierarchical_bn_runtime"]
    for name in order:
        if name not in methods:
            continue
        v = methods[name]
        lines.append(f"| {name} | {v['f1']['mean']:.3f}±{v['f1']['sd']:.3f} | {v['auroc']['mean']:.3f}±{v['auroc']['sd']:.3f} | {v['auprc']['mean']:.3f}±{v['auprc']['sd']:.3f} | {v['brier']['mean']:.3f}±{v['brier']['sd']:.3f} | {v['ece_10bin']['mean']:.3f}±{v['ece_10bin']['sd']:.3f} |")
    lines += ["", "## 机制分解（我们的模型）", "", "| 机制 | n | 正例率 | F1@0.5 | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in report["mechanisms"].items():
        auc = "n/a" if value["auroc"] is None else f"{value['auroc']:.3f}"
        ap = "n/a" if value["auprc"] is None else f"{value['auprc']:.3f}"
        lines.append(f"| {name} | {value['n']} | {value['positive_rate']:.3f} | {value['f1_at_0_5']:.3f} | {auc} | {ap} |")
    n_runs = len(report.get("runs", []))
    lines += ["", f"## 相对主模型的 {n_runs} 次配对增量", "", f"正值表示主模型更好；Wilcoxon p 值只基于 {n_runs} 个分组种子的配对差异，不把 4,000 条轨迹当作独立检验。", "", "| 对照 | ΔF1 | ΔAUROC | ΔBrier | p(F1) |", "|---|---:|---:|---:|---:|"]
    for name, value in report["aggregate"]["paired_deltas_vs_ours"].items():
        lines.append(f"| {name} | {value['f1_mean_delta']:.3f} | {value['auroc_mean_delta']:.3f} | {value['brier_mean_delta']:.3f} | {value['paired_wilcoxon_f1_p']:.4f} |")
    lines += ["", "## 解释边界", "", "- Layer 1 是可读的局部风险 BN；workflow DAG 是规范先验，不是因果发现结果。", "- Layer 2 使用显式运行时消息/委派边，并通过共享融合层完成系统级风险评估。", "- hidden effect、标签、机制名和 scenario family 均不进入模型特征；scenario family 只用于分组切分。", "- `taint_path_rule` 是透明规则基线；其他方法均在相同训练组上拟合。", "- 该数据集是独立的合成 MAS 基准，不能替代真实 API MAS 的外部效度验证。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results/independent_mas_v3/traces_public.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/independent_mas_v3/labels.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/submission/final_eval")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20261001, 20261002, 20261003, 20261004, 20261005])
    ap.add_argument("--holdout-fraction", type=float, default=.20)
    args = ap.parse_args()
    fair = load_module("final_fair", SCRIPTS / "evaluate_independent_fair_baselines_v1.py")
    two = load_module("final_two", SCRIPTS / "evaluate_two_layer_v2.py")
    hier = load_module("final_hier", SCRIPTS / "evaluate_hierarchical_monitor_v3.py")
    traces = read_jsonl(args.input)
    label_rows = read_jsonl(args.labels)
    labels = {str(row["episode_id"]): row for row in label_rows}
    eps = []
    for trace in traces:
        eid = str(trace["episode_id"])
        row = labels[eid]
        public = fair.public_episode(trace, labels)
        public["mechanism"] = row.get("mechanism", "unknown")
        eps.append(public)
    y = np.asarray([int(e["label"]) for e in eps], dtype=int)
    # The active submission protocol uses nested grouped fitting.  The earlier
    # in-sample score-fusion routine remains above only for historical replay;
    # it is not used to produce the active table.
    runs = [run_seed_strict(eps, y, seed, args.holdout_fraction, fair, two, hier) for seed in args.seeds]
    report = {
        "protocol": {"dataset": "independent_mas_v3", "n": len(eps), "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "seeds": args.seeds, "holdout_fraction": args.holdout_fraction, "public_only": True},
        "model": {"name": "HierarchicalNormativeRuntimeMonitor-v4-strict", "layer1": "episode-balanced Laplace local BN", "layer2": "explicit runtime-message propagation", "fusion": "group-fold-out workflow-context and BN/runtime fusion with separate group calibration", "causal_claim": False},
        "aggregate": aggregate(runs),
        "mechanisms": mechanism_report(runs, "ours_hierarchical_bn_runtime"),
        "runs": runs,
        "limitations": ["independent_mas_v3 uses hidden-effect labels from a synthetic generator", "workflow DAG is normative scaffolding, not a discovered causal graph", "all baselines use the same public fields and group splits", "fusion scores are fit out-of-fold within each outer training set and calibrated on a separate group holdout", "ALTEDA is represented by a transparent runtime-graph proxy, not claimed as a reproduction of its private implementation"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out, report)
    print(json.dumps({"out": str(args.out), "n": len(eps), "seeds": args.seeds, "methods": list(report["aggregate"]["mean_sd"]), "ours": report["aggregate"]["mean_sd"]["ours_hierarchical_bn_runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
