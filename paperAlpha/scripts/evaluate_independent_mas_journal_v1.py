"""Journal-grade grouped evaluation for the two-line MAS monitor.

This script keeps the normative workflow DAG and the runtime event graph
separate.  The proposed monitor is deliberately small and inspectable:

* workflow BN: Laplace-smoothed episode CPTs over binary workflow evidence,
  combined by a noisy-OR;
* runtime BN: the existing observable message/delegation propagation model;
* fusion: a train-only logistic calibrator over the two scores and explicit
  missingness/uncertainty features.

The benchmark labels are joined only after features are constructed.  Group
folds are scenario-family holdouts, so an exact workflow family never appears
in both train and test.  This is still a synthetic independent benchmark and
must not be described as a real-world human-labelled result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback
    StratifiedGroupKFold = None

PRIV_TOOLS = {"payments.write", "filesystem.write", "shell.exec"}
FORBIDDEN = {"label", "external_effect", "hidden_fields", "label_source", "scenario_family", "mechanism"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested_keys(value) -> set[str]:
    """Collect keys recursively for a defensive public-trace schema audit."""
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def load_two_layer():
    path = Path(__file__).with_name("evaluate_two_layer_v2.py")
    spec = importlib.util.spec_from_file_location("two_layer_monitor_v2_journal", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reachable(sources: set[str], edges: Iterable[tuple[str, str]], sink: set[str]):
    adj: dict[str, set[str]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    seen = set(sources)
    queue = deque((s, 0) for s in sources)
    distance = None
    while queue:
        u, depth = queue.popleft()
        if u in sink:
            distance = depth if distance is None else min(distance, depth)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append((v, depth + 1))
    return bool(sink.intersection(seen)), distance


def observable_parts(ep: dict):
    events = ep.get("events", [])
    workflow = [tuple(edge) for edge in ep.get("workflow_dag", {}).get("edges", [])]
    observed = [
        e for e in events
        if e.get("event_type") in {"message_sent", "delegation"} and e.get("target")
    ]
    observed_edges = {(e.get("actor"), e.get("target")) for e in observed}
    source = {
        e.get("actor")
        for e in events
        if e.get("event_type") == "input_received" and e.get("data_class") == "untrusted"
    }
    sink = {
        e.get("actor")
        for e in events
        if e.get("event_type") == "tool_call" and e.get("tool") in PRIV_TOOLS
    }
    return workflow, observed, observed_edges, source, sink


def workflow_vector(ep: dict) -> np.ndarray:
    """Binary/continuous workflow evidence; no event outcome fields."""
    workflow, observed, observed_edges, source, sink = observable_parts(ep)
    wf = set(workflow)
    path, distance = _reachable(source, wf, sink)
    nodes = set(ep.get("agents", []))
    indeg = {a: 0 for a in nodes}
    outdeg = {a: 0 for a in nodes}
    for u, v in workflow:
        outdeg[u] = outdeg.get(u, 0) + 1
        indeg[v] = indeg.get(v, 0) + 1
    expected_missing = len(wf - observed_edges) / max(1, len(wf))
    return np.asarray(
        [
            float(len(nodes)),
            float(len(workflow)),
            float(len(source)),
            float(len(sink)),
            float(path),
            1.0 / (1.0 + distance) if distance is not None else 0.0,
            float(max(outdeg.values(), default=0)),
            float(max(indeg.values(), default=0)),
            float(np.mean(list(outdeg.values()))) if outdeg else 0.0,
            float(expected_missing),
        ],
        dtype=float,
    )


def runtime_vector(ep: dict) -> np.ndarray:
    """Compact runtime-only evidence used for an ablation and calibration."""
    _, observed, observed_edges, source, sink = observable_parts(ep)
    untrusted_edges = {
        (e.get("actor"), e.get("target"))
        for e in observed
        if e.get("data_class") == "untrusted"
    }
    path, distance = _reachable(source, untrusted_edges, sink)
    unexpected = sum(e.get("edge_kind") == "unexpected" for e in observed)
    unknown = sum(e.get("data_class") in {"unknown", "redacted"} for e in observed)
    mismatch = sum(bool(e.get("permission_mismatch")) for e in ep.get("events", []))
    return np.asarray(
        [
            float(len(observed)),
            float(len(observed_edges)),
            float(sum(e.get("data_class") == "untrusted" for e in observed)),
            float(unknown),
            float(unexpected),
            float(mismatch),
            float(path),
            1.0 / (1.0 + distance) if distance is not None else 0.0,
        ],
        dtype=float,
    )


def episode_noisy_or_fit(x: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0) -> dict:
    """Fit a weakly supervised noisy-OR CPD on episode labels.

    ``y`` is an episode outcome, not a node outcome.  The old implementation
    estimated ``P(y|f=1)`` for every factor and then multiplied those values as
    if they were independent latent causes; with ten mostly-always-positive
    structural factors this saturated near one.  Here each factor has an
    explicit latent hazard ``H_f`` with ``P(H_f=1|f=1)=q_f`` and a fixed
    Laplace-smoothed leak.  The q values are fitted jointly by episode BCE,
    which is the identifiable noisy-OR likelihood under this weak supervision.
    """
    binary = (np.asarray(x, dtype=float) > 0).astype(float)
    y = np.asarray(y, dtype=float)
    prior = float((y.sum() + 1.0) / (len(y) + 2.0))
    rng = np.random.default_rng(seed)
    # Small initial hazard probabilities avoid the all-factors-active
    # saturation that occurs when each q is estimated as P(y|f=1).
    initial_q = np.full(binary.shape[1], 0.05, dtype=float)
    initial_q += rng.normal(0.0, 0.005, binary.shape[1])
    initial_q = np.clip(initial_q, 0.005, 0.2)
    x0 = np.log(initial_q / (1.0 - initial_q))

    def unpack(theta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(theta, -12.0, 12.0)))

    def objective(theta: np.ndarray) -> float:
        q = unpack(theta)
        p = 1.0 - (1.0 - prior) * np.prod(1.0 - binary * q[None, :], axis=1)
        bce = -np.mean(y * np.log(np.clip(p, 1e-7, 1 - 1e-7)) + (1.0 - y) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)))
        return float(bce + 1e-3 * np.sum(theta * theta))

    result = minimize(objective, x0, method="L-BFGS-B", bounds=[(-12.0, 12.0)] * len(x0), options={"maxiter": 500, "ftol": 1e-10})
    q = unpack(result.x)
    return {"prior": prior, "names": names, "cpt": {name: float(q[j]) for j, name in enumerate(names)}, "fit": {"success": bool(result.success), "loss": float(result.fun), "iterations": int(result.nit), "method": "joint_noisy_or_bce"}}


def workflow_bn_fit(x: np.ndarray, y: np.ndarray) -> dict:
    names = [
        "agent_count", "workflow_edge_count", "source_count", "privileged_sink_count",
        "normative_source_sink_path", "normative_short_path", "max_out_degree",
        "max_in_degree", "mean_out_degree", "expected_edge_missingness",
    ]
    # Fit latent factor hazards jointly against the episode outcome.  This is
    # a weakly supervised conditional BN/noisy-OR CPD, not a node-label CPT.
    return episode_noisy_or_fit(x, y, names)


def workflow_bn_score(x: np.ndarray, bn: dict) -> np.ndarray:
    out = []
    for row in x:
        score = 1.0
        for j, name in enumerate(bn["names"]):
            if row[j] > 0:
                score *= 1.0 - min(.995, max(.001, bn["cpt"][name]))
        out.append(1.0 - score if np.any(row > 0) else bn["prior"])
    return np.asarray(out, dtype=float)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    values = np.unique(np.r_[0.0, p, 1.0])
    return float(max(values, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def ece(y: np.ndarray, p: np.ndarray) -> float:
    result = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            result += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return result


def score_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": float(ece(y, p)),
        "threshold": float(threshold),
        "positive_predictions": int(pred.sum()),
    }


def score_metrics_with_predictions(y: np.ndarray, p: np.ndarray, pred: np.ndarray, threshold_note: str) -> dict:
    """Ranking/calibration from pooled OOF scores; classification from fold-local thresholds."""
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": float(ece(y, p)),
        "threshold": threshold_note,
        "positive_predictions": int(pred.sum()),
    }


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(train_x, train_y)
    return model.predict_proba(train_x)[:, 1], model.predict_proba(test_x)[:, 1], model


def fit_runtime_bn_inner(two, train_eps: list[dict], y: np.ndarray, bn: dict, seed: int, maxiter: int = 80):
    """Fit the same runtime BN with a bounded inner-CV budget."""
    rng = np.random.default_rng(seed)
    x0 = np.array([-2., 3., .2, .8, -.4, .1, .1, -2., 4., .2, 1., 1., .3], dtype=float)
    x0 += rng.normal(0, .03, len(x0))

    def objective(theta):
        p = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps])
        bce = -np.mean(y * np.log(np.clip(p, 1e-7, 1 - 1e-7)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)))
        return float(bce + 1e-3 * np.sum(theta * theta))

    result = minimize(objective, x0, method="L-BFGS-B", bounds=[(-8.0, 8.0)] * len(x0), options={"maxiter": maxiter, "ftol": 1e-7})
    return result.x


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target: float = 0.05) -> float:
    negatives = p[y == 0]
    if not len(negatives):
        return 1.0
    return float(np.quantile(negatives, 1.0 - target, method="higher"))


def fixed_fpr_summary(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "fpr": float(fp / max(1, fp + tn)),
        "recall": float(tp / max(1, tp + fn)),
        "predicted_positive": int(pred.sum()),
    }


def bootstrap_summary(y: np.ndarray, p: np.ndarray, pred: np.ndarray, groups: np.ndarray, seed: int, n: int = 1000) -> dict:
    """Cluster bootstrap over scenario families, not IID episodes."""
    rng = np.random.default_rng(seed)
    unique_groups = np.asarray(sorted(set(groups)))
    values = {"auroc": [], "auprc": [], "brier": [], "f1": []}
    for _ in range(n):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        values["auroc"].append(float(roc_auc_score(y[idx], p[idx])))
        values["auprc"].append(float(average_precision_score(y[idx], p[idx])))
        values["brier"].append(float(brier_score_loss(y[idx], p[idx])))
        values["f1"].append(float(f1_score(y[idx], pred[idx], zero_division=0)))
    return {
        metric: {
            "mean": float(np.mean(vals)),
            "low": float(np.quantile(vals, 0.025)),
            "high": float(np.quantile(vals, 0.975)),
            "n": len(vals),
        }
        for metric, vals in values.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("paperAlpha/results/independent_mas_v3/traces_public.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("paperAlpha/results/independent_mas_v3/labels.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("paperAlpha/results/independent_mas_journal_v1"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    traces = read_jsonl(args.input)
    labels = {row["episode_id"]: row for row in read_jsonl(args.labels)}
    forbidden_hits = []
    for trace in traces:
        hit = FORBIDDEN.intersection(nested_keys(trace))
        if hit:
            forbidden_hits.append(sorted(hit))
    if forbidden_hits:
        raise ValueError(f"public trace contains evaluation-only fields: {forbidden_hits[0]}")
    y = np.asarray([int(labels[t["episode_id"]]["label"]) for t in traces], dtype=int)
    groups = np.asarray([labels[t["episode_id"]].get("scenario_family", "unknown") for t in traces])
    mechanisms = np.asarray([labels[t["episode_id"]].get("mechanism", "unknown") for t in traces])
    wf = np.asarray([workflow_vector(t) for t in traces], dtype=float)
    rt = np.asarray([runtime_vector(t) for t in traces], dtype=float)
    two = load_two_layer()

    names = ["workflow_bn", "runtime_bn", "workflow_logistic", "runtime_logistic", "hierarchical_fusion"]
    oof = {name: np.zeros(len(traces), dtype=float) for name in names}
    oof_pred = {name: np.zeros(len(traces), dtype=bool) for name in names}
    oof_fixed5 = {name: np.zeros(len(traces), dtype=bool) for name in names}
    fold_thresholds = {name: [] for name in names}
    fold_records = []
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_name = "StratifiedGroupKFold by scenario_family"
    else:
        splitter = GroupKFold(n_splits=args.folds)
        split_name = "GroupKFold by scenario_family"
    for fold, (train, test) in enumerate(splitter.split(wf, y, groups), start=1):
        wf_bn = workflow_bn_fit(wf[train], y[train])
        p_wf_tr = workflow_bn_score(wf[train], wf_bn); p_wf_te = workflow_bn_score(wf[test], wf_bn)
        oof["workflow_bn"][test] = p_wf_te
        threshold = best_threshold(y[train], p_wf_tr); oof_pred["workflow_bn"][test] = p_wf_te >= threshold; fold_thresholds["workflow_bn"].append(float(threshold))
        oof_fixed5["workflow_bn"][test] = p_wf_te >= threshold_at_fpr(y[train], p_wf_tr)
        rt_tr, rt_te, _ = fit_logistic(rt[train], y[train], rt[test], args.seed + fold)
        oof["runtime_logistic"][test] = rt_te
        threshold = best_threshold(y[train], rt_tr); oof_pred["runtime_logistic"][test] = rt_te >= threshold; fold_thresholds["runtime_logistic"].append(float(threshold))
        oof_fixed5["runtime_logistic"][test] = rt_te >= threshold_at_fpr(y[train], rt_tr)

        train_eps = [{**traces[i], "label": int(y[i])} for i in train]
        test_eps = [{**traces[i], "label": int(y[i])} for i in test]
        bn = two.fit_bn(train_eps, y[train])
        theta, opt = two.fit_model(train_eps, y[train], bn, args.seed + 100 + fold)
        p_rt_bn_tr = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps])
        p_rt_bn_te = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in test_eps])
        oof["runtime_bn"][test] = p_rt_bn_te
        threshold = best_threshold(y[train], p_rt_bn_tr); oof_pred["runtime_bn"][test] = p_rt_bn_te >= threshold; fold_thresholds["runtime_bn"].append(float(threshold))
        oof_fixed5["runtime_bn"][test] = p_rt_bn_te >= threshold_at_fpr(y[train], p_rt_bn_tr)

        wf_tr, wf_te, _ = fit_logistic(wf[train], y[train], wf[test], args.seed + 200 + fold)
        oof["workflow_logistic"][test] = wf_te
        threshold = best_threshold(y[train], wf_tr); oof_pred["workflow_logistic"][test] = wf_te >= threshold; fold_thresholds["workflow_logistic"].append(float(threshold))
        oof_fixed5["workflow_logistic"][test] = wf_te >= threshold_at_fpr(y[train], wf_tr)

        # Fusion base scores are generated by an inner grouped OOF split.  This
        # prevents a base model from seeing its own training label before its
        # score is consumed by the fusion head.
        inner_oof_wf = np.zeros(len(train), dtype=float)
        inner_oof_rt = np.zeros(len(train), dtype=float)
        if StratifiedGroupKFold is not None:
            inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed + fold)
        else:
            inner_splitter = GroupKFold(n_splits=3)
        train_groups = groups[train]
        for inner, (itr, iva) in enumerate(inner_splitter.split(wf[train], y[train], train_groups), start=1):
            inner_bn = workflow_bn_fit(wf[train][itr], y[train][itr])
            inner_oof_wf[iva] = workflow_bn_score(wf[train][iva], inner_bn)
            inner_eps_train = [{**traces[train[i]], "label": int(y[train[i]])} for i in itr]
            inner_eps_val = [{**traces[train[i]], "label": int(y[train[i]])} for i in iva]
            inner_bn_rt = two.fit_bn(inner_eps_train, y[train][itr])
            inner_theta_rt = fit_runtime_bn_inner(two, inner_eps_train, y[train][itr], inner_bn_rt, args.seed + 400 + 10 * fold + inner)
            inner_oof_rt[iva] = np.asarray([two._propagate(inner_theta_rt, two.event_features(e), inner_bn_rt) for e in inner_eps_val])
        fusion_tr = np.c_[wf[train], rt[train], inner_oof_wf, inner_oof_rt]
        fusion_te = np.c_[wf[test], rt[test], p_wf_te, p_rt_bn_te]
        f_tr, f_te, fusion_model = fit_logistic(fusion_tr, y[train], fusion_te, args.seed + 300 + fold)
        oof["hierarchical_fusion"][test] = f_te
        # The fusion threshold is also selected from an inner OOF prediction,
        # not from the fusion head's in-sample scores.
        fusion_inner_pred = np.zeros(len(train), dtype=float)
        for inner, (itr, iva) in enumerate(inner_splitter.split(fusion_tr, y[train], train_groups), start=1):
            inner_head = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed + 500 + 10 * fold + inner).fit(fusion_tr[itr], y[train][itr])
            fusion_inner_pred[iva] = inner_head.predict_proba(fusion_tr[iva])[:, 1]
        threshold = best_threshold(y[train], fusion_inner_pred)
        oof_pred["hierarchical_fusion"][test] = f_te >= threshold; fold_thresholds["hierarchical_fusion"].append(float(threshold))
        oof_fixed5["hierarchical_fusion"][test] = f_te >= threshold_at_fpr(y[train], fusion_inner_pred)
        fold_records.append({"fold": fold, "train_count": int(len(train)), "test_count": int(len(test)), "train_groups": int(len(set(groups[train]))), "test_groups": int(len(set(groups[test]))), "positive_test": int(y[test].sum()), "negative_test": int((1-y[test]).sum()), "runtime_optimizer_success": bool(opt.get("success", False)), "fusion_inner_oof": True, "fusion_base_scores": ["workflow_bn_inner_oof", "runtime_bn_inner_oof"], "fusion_coefficients": [float(v) for v in fusion_model.coef_[0]], "fusion_intercept": float(fusion_model.intercept_[0])})

    results = {name: score_metrics_with_predictions(y, p, oof_pred[name], "fold-local train-only") for name, p in oof.items()}
    fixed_fpr = {name: fixed_fpr_summary(y, oof_fixed5[name]) for name in names}
    fused_bootstrap = bootstrap_summary(y, oof["hierarchical_fusion"], oof_pred["hierarchical_fusion"], groups, args.seed)
    mechanism_breakdown = {}
    for mechanism in sorted(set(mechanisms)):
        mask = mechanisms == mechanism
        p = oof["hierarchical_fusion"][mask]
        mechanism_breakdown[mechanism] = {
            "n": int(mask.sum()),
            "positive": int(y[mask].sum()),
            "auroc": float(roc_auc_score(y[mask], p)) if len(np.unique(y[mask])) > 1 else None,
            "auprc": float(average_precision_score(y[mask], p)) if y[mask].sum() and (1-y[mask]).sum() else None,
            "f1": float(f1_score(y[mask], oof_pred["hierarchical_fusion"][mask], zero_division=0)),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"dataset": {"input": str(args.input), "labels": str(args.labels), "n": len(y), "positive": int(y.sum()), "negative": int(len(y)-y.sum()), "groups": int(len(set(groups))), "public_fields_only": True}, "protocol": {"split": split_name, "thresholds": "fold-local thresholds; fusion threshold from nested inner-OOF scores; ranking/calibration are pooled outer OOF", "fixed_fpr": "fold-local 5% train-negative FPR thresholds", "forbidden_features": sorted(FORBIDDEN)}, "methods": results, "fixed_fpr_5pct": fixed_fpr, "hierarchical_fusion_bootstrap_95": fused_bootstrap, "hierarchical_fusion_by_mechanism": mechanism_breakdown, "fold_thresholds": fold_thresholds, "folds": fold_records}, indent=2), encoding="utf-8")
    with (args.out / "predictions_oof.jsonl").open("w", encoding="utf-8") as handle:
        for i, trace in enumerate(traces):
            handle.write(json.dumps({"episode_id": trace["episode_id"], "group": groups[i], "label": int(y[i]), **{k: float(v[i]) for k, v in oof.items()}}, sort_keys=True) + "\n")
    lines = ["# Independent MAS journal evaluation v1", "", f"Five-fold {split_name}; no family is shared between train and test in a fold.", "The public trace contains the normative workflow DAG and runtime events; labels are used only after feature construction.", "", f"- n={len(y)}, positives={int(y.sum())}, groups={len(set(groups))}", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m = results[name]; lines.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['brier']:.3f} | {m['ece_10bin']:.3f} |")
    lines += ["", "## Fixed 5% FPR (fold-local)", "", "| method | empirical FPR | recall |", "|---|---:|---:|"]
    for name in names:
        m = fixed_fpr[name]; lines.append(f"| {name} | {m['fpr']:.3f} | {m['recall']:.3f} |")
    lines += ["", "## Interpretation", "- `workflow_bn` is the normative DAG line; `runtime_bn` is the observable message/delegation line.", "- `hierarchical_fusion` is train-fitted only and exposes its two line scores plus missingness/unknown-evidence terms.", "- Confidence intervals and mechanism breakdown are in `metrics.json`; the benchmark is synthetic and requires confirmation on an independently generated/API cohort before a journal claim.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
