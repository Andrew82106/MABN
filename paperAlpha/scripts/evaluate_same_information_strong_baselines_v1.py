"""Strict same-information Logistic/HistGB baselines for the canonical MAS fold.

The feature union is exactly the public 78-dimensional contract used by the
existing matched-information audit: aggregate event counts (15), pooled local
agent evidence (12), runtime graph summaries (25), normative/runtime workflow
relations (16), and transparent knowledge-template evidence (10).  No fitted
BN score, label, hidden mechanism, or family identifier is an input feature.

Each outer fold is the canonical StratifiedGroupKFold used by
``evaluate_independent_mas_journal_v1.py`` (five folds, seed 20261002).  A
four-fold grouped OOF run inside the outer training groups supplies calibration
and the F1/5%-FPR thresholds.  The final estimator is refit on the full outer
training groups and is then scored once on the untouched outer test groups.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FAIR = _load("paperalpha_fair_baselines_v1", "evaluate_independent_fair_baselines_v1.py")
HIER = _load("paperalpha_hierarchical_features_v3", "evaluate_hierarchical_monitor_v3.py")
FINAL = _load("paperalpha_final_features", "evaluate_submission_final.py")


BLOCKS: tuple[tuple[str, int, str], ...] = (
    ("aggregate_event_counts", 15, "evaluate_independent_fair_baselines_v1.aggregate_features"),
    ("local_agent_pool", 12, "evaluate_independent_fair_baselines_v1.local_features"),
    ("runtime_graph_summary", 25, "evaluate_independent_fair_baselines_v1.graph_features"),
    ("normative_runtime_relation", 16, "evaluate_hierarchical_monitor_v3.workflow_features"),
    ("knowledge_templates", 10, "evaluate_submission_final.knowledge_template_features"),
)
FEATURE_DIM = sum(dim for _, dim, _ in BLOCKS)
FORBIDDEN_KEYS = {"label", "external_effect", "hidden_fields", "label_source", "scenario_family", "mechanism"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        return {str(k) for k in value} | set().union(*(_nested_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_nested_keys(v) for v in value)) if value else set()
    return set()


def _validate_public(trace: dict) -> None:
    leaked = FORBIDDEN_KEYS.intersection(_nested_keys(trace))
    if leaked:
        raise ValueError(f"forbidden label-like fields: {sorted(leaked)}")
    HIER.validate_public_trace(trace)


def feature_matrix(traces: list[dict]) -> tuple[np.ndarray, dict]:
    rows = []
    for trace in traces:
        _validate_public(trace)
        # The feature functions are intentionally called before labels are
        # loaded/joined, so an accidental label dependency fails loudly.
        blocks = [
            np.asarray(FAIR.aggregate_features(trace), dtype=float),
            np.asarray(FAIR.local_features(trace), dtype=float),
            np.asarray(FAIR.graph_features(trace), dtype=float),
            np.asarray(HIER.workflow_features(trace), dtype=float),
            np.asarray(FINAL.knowledge_template_features(trace), dtype=float),
        ]
        for (name, expected, _), value in zip(BLOCKS, blocks):
            if value.ndim != 1 or len(value) != expected or not np.all(np.isfinite(value)):
                raise ValueError(f"feature block {name} has invalid shape/value: {value.shape}")
        rows.append(np.concatenate(blocks))
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(traces), FEATURE_DIM):
        raise ValueError(f"feature matrix shape {matrix.shape}, expected {(len(traces), FEATURE_DIM)}")
    contract = {
        "feature_dim": FEATURE_DIM,
        "blocks": [{"name": n, "dim": d, "source": s} for n, d, s in BLOCKS],
        "forbidden": sorted(FORBIDDEN_KEYS),
        "boundary": "public trace only; no learned BN scores, labels, hidden mechanisms or family IDs",
    }
    return matrix, contract


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & ((p < high) if high < 1.0 else (p <= high))
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": float(_ece(y, p)),
        "threshold": float(threshold),
        "predicted_positive": int(pred.sum()),
    }


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    values = np.unique(np.r_[0.0, p, 1.0])
    return float(max(values, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def _fpr_threshold(y: np.ndarray, p: np.ndarray, target: float = 0.05) -> float:
    negatives = p[y == 0]
    if not len(negatives):
        return 1.0
    return float(np.quantile(negatives, 1.0 - target, method="higher"))


def _calibrate(raw: np.ndarray, y: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Fit a Platt map on grouped-OOF scores, never on outer test scores."""
    if len(np.unique(y)) < 2:
        raise ValueError("calibration fold must contain both classes")
    z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(z.reshape(-1, 1), y)

    def apply(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        zv = np.log(np.clip(v, 1e-6, 1 - 1e-6) / np.clip(1 - v, 1e-6, 1 - v))
        return model.predict_proba(zv.reshape(-1, 1))[:, 1]

    return apply, {"intercept": float(model.intercept_[0]), "coefficient": float(model.coef_[0, 0])}


def _make_model(kind: str, seed: int):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed))
    if kind == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, max_depth=3, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
    raise ValueError(kind)


def _inner_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Return training-only OOF scores and inner-fold lineage."""
    out = np.zeros(len(y), dtype=float)
    lineage = []
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for inner, (fit, valid) in enumerate(splitter.split(x, y, groups), start=1):
        model = _make_model(kind, seed + inner)
        model.fit(x[fit], y[fit])
        out[valid] = model.predict_proba(x[valid])[:, 1]
        lineage.append({
            "inner_fold": inner,
            "fit_indices": fit.tolist(),
            "valid_indices": valid.tolist(),
            "fit_groups": sorted(set(str(g) for g in groups[fit])),
            "valid_groups": sorted(set(str(g) for g in groups[valid])),
            "fit_count": int(len(fit)), "valid_count": int(len(valid)),
        })
    if not np.all(np.isfinite(out)) or np.all(out == 0):
        raise RuntimeError(f"invalid inner OOF scores for {kind}")
    return out, lineage


def _group_metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    values_b, values_e, sizes = [], [], []
    for group in sorted(set(groups)):
        ix = groups == group
        values_b.append(float(brier_score_loss(y[ix], p[ix])))
        values_e.append(float(_ece(y[ix], p[ix])))
        sizes.append(int(ix.sum()))
    return {
        "macro_brier": float(np.mean(values_b)),
        "macro_ece_10bin": float(np.mean(values_e)),
        "weighted_brier": float(np.average(values_b, weights=sizes)),
        "weighted_ece_10bin": float(np.average(values_e, weights=sizes)),
        "groups": int(len(values_b)),
        "min_group_size": int(min(sizes)),
        "max_group_size": int(max(sizes)),
    }


def _bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray, seed: int, reps: int) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(groups)), dtype=object)
    rows = {g: np.flatnonzero(groups == g) for g in unique}
    draws = {"micro_brier": [], "macro_brier": [], "micro_ece": [], "macro_ece": []}
    for _ in range(reps):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([rows[g] for g in chosen])
        b = _group_metrics(y[ix], p[ix], groups[ix])
        draws["micro_brier"].append(float(brier_score_loss(y[ix], p[ix])))
        draws["micro_ece"].append(float(_ece(y[ix], p[ix])))
        draws["macro_brier"].append(b["macro_brier"])
        draws["macro_ece"].append(b["macro_ece_10bin"])
    return {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "n": len(v)} for k, v in draws.items()}


def run(args: argparse.Namespace) -> dict:
    traces = read_jsonl(args.input)
    labels_rows = read_jsonl(args.labels)
    labels = {row["episode_id"]: row for row in labels_rows}
    if len(labels) != len(labels_rows):
        raise ValueError("duplicate labels")
    # Construct the complete feature union before joining labels.
    x, contract = feature_matrix(traces)
    missing = [t["episode_id"] for t in traces if t.get("episode_id") not in labels]
    if missing:
        raise ValueError(f"missing labels for {missing[:3]}")
    y = np.asarray([int(labels[t["episode_id"]]["label"]) for t in traces], dtype=int)
    groups = np.asarray([str(labels[t["episode_id"]].get("scenario_family", "unknown")) for t in traces], dtype=object)
    if len(np.unique(y)) < 2:
        raise ValueError("labels need both classes")
    args.out.mkdir(parents=True, exist_ok=True)
    oof = {f"{kind}_raw": np.zeros(len(y), float) for kind in ("logistic", "histgb")}
    oof.update({f"{kind}_calibrated": np.zeros(len(y), float) for kind in ("logistic", "histgb")})
    oof_pred = {kind: np.zeros(len(y), bool) for kind in ("logistic", "histgb")}
    oof_fpr = {kind: np.zeros(len(y), bool) for kind in ("logistic", "histgb")}
    lineage = []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(outer.split(x, y, groups), start=1):
        fold_record = {"fold": fold, "outer_train_indices": train.tolist(), "outer_test_indices": test.tolist(),
                       "outer_train_groups": sorted(set(str(g) for g in groups[train])),
                       "outer_test_groups": sorted(set(str(g) for g in groups[test])),
                       "outer_train_count": int(len(train)), "outer_test_count": int(len(test)),
                       "outer_test_positive": int(y[test].sum()), "outer_test_negative": int((1-y[test]).sum()),
                       "models": {}}
        for kind in ("logistic", "histgb"):
            start = time.perf_counter()
            inner_raw, inner_lineage = _inner_oof(x[train], y[train], groups[train], kind, args.seed + fold * 100)
            apply_cal, cal_params = _calibrate(inner_raw, y[train])
            calibrated_inner = apply_cal(inner_raw)
            threshold = _best_threshold(y[train], calibrated_inner)
            fpr_threshold = _fpr_threshold(y[train], calibrated_inner)
            model = _make_model(kind, args.seed + fold)
            model.fit(x[train], y[train])
            raw_test = model.predict_proba(x[test])[:, 1]
            calibrated_test = apply_cal(raw_test)
            oof[f"{kind}_raw"][test] = raw_test
            oof[f"{kind}_calibrated"][test] = calibrated_test
            oof_pred[kind][test] = calibrated_test >= threshold
            oof_fpr[kind][test] = calibrated_test >= fpr_threshold
            elapsed = time.perf_counter() - start
            fold_record["models"][kind] = {
                "threshold_f1": float(threshold), "threshold_fpr_005": float(fpr_threshold),
                "calibration": cal_params, "inner_lineage": inner_lineage,
                "fit_seconds_including_inner_oof": float(elapsed),
            }
        lineage.append(fold_record)

    metrics = {}
    for kind in ("logistic", "histgb"):
        p = oof[f"{kind}_calibrated"]
        metrics[kind] = {
            "pooled_oof": _metrics(y, p, 0.5),
            "fold_local_threshold_metrics": {
                "f1": float(f1_score(y, oof_pred[kind], zero_division=0)),
                "precision": float(precision_score(y, oof_pred[kind], zero_division=0)),
                "recall": float(recall_score(y, oof_pred[kind], zero_division=0)),
            },
            "fixed_fpr_005": {
                "fpr": float(np.mean(oof_fpr[kind][y == 0])),
                "recall": float(np.mean(oof_fpr[kind][y == 1])),
            },
            "group_calibration": _group_metrics(y, p, groups),
            "cluster_bootstrap_95": _bootstrap(y, p, groups, args.seed + (11 if kind == "logistic" else 22), args.bootstrap),
        }
    output_rows = []
    for i, trace in enumerate(traces):
        fold = next(rec["fold"] for rec in lineage if i in rec["outer_test_indices"])
        output_rows.append({"episode_id": trace["episode_id"], "group": str(groups[i]), "label": int(y[i]), "outer_fold": fold,
                            **{k: float(v[i]) for k, v in oof.items()},
                            "logistic_fold_pred": bool(oof_pred["logistic"][i]), "histgb_fold_pred": bool(oof_pred["histgb"][i]),
                            "logistic_fixed_fpr_pred": bool(oof_fpr["logistic"][i]), "histgb_fixed_fpr_pred": bool(oof_fpr["histgb"][i])})
    (args.out / "predictions_oof.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in output_rows), encoding="utf-8")
    (args.out / "feature_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (args.out / "fold_lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    result = {
        "schema": "same-information-strong-baselines-v1",
        "dataset": {"n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y)-y.sum()), "groups": int(len(set(groups))), "input_sha256": sha256(args.input), "labels_sha256": sha256(args.labels)},
        "protocol": {"outer": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261002)", "inner": "4-fold StratifiedGroupKFold OOF inside each outer training group", "calibration": "Platt calibration fitted only on inner OOF training-group predictions", "thresholds": "F1 and 5%-negative-FPR thresholds fitted only on calibrated inner OOF scores", "feature_contract": contract},
        "models": {"logistic": {"C": 1.0, "standardized": True}, "histgb": {"max_iter": 150, "max_leaf_nodes": 15, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False}},
        "metrics": metrics, "folds": lineage,
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Same-information strong baselines", "", f"n={len(y)}, positives={int(y.sum())}, groups={len(set(groups))}, features={FEATURE_DIM}", "", "| method | AUROC | AUPRC | Brier | ECE | macro Brier | macro ECE | fixed-FPR recall |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("logistic", "histgb"):
        m = metrics[kind]; p = m["pooled_oof"]; g = m["group_calibration"]
        lines.append(f"| {kind} | {p['auroc']:.3f} | {p['auprc']:.3f} | {p['brier']:.3f} | {p['ece_10bin']:.3f} | {g['macro_brier']:.3f} | {g['macro_ece_10bin']:.3f} | {m['fixed_fpr_005']['recall']:.3f} |")
    lines += ["", "Scores are pooled outer OOF predictions. Thresholds and calibration use only inner grouped OOF scores within each outer training fold.", "The feature union is public and label-blind; this is a strong matched-information baseline, not a reproduction of another paper's native target.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "results/independent_mas_v3/traces_public.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/independent_mas_v3/labels.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/submission/development/strong_baselines_canonical_20261006")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    result = run(args)
    print(json.dumps({k: {"auroc": v["pooled_oof"]["auroc"], "auprc": v["pooled_oof"]["auprc"], "brier": v["pooled_oof"]["brier"], "macro_brier": v["group_calibration"]["macro_brier"]} for k, v in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
