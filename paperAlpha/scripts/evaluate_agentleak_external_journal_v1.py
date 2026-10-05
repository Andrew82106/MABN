"""External MAS validation on the public AgentLeak trace release.

The target is the final coordinator-to-user literal-policy violation.  The
monitor sees only the static functional DAG and messages available before the
final output.  ``results``, ``attack_family``, model identity and final output
are label/evaluation channels and never enter features.

This is a separate external validation: it does not refit or alter the frozen
independent_mas_journal_v1 result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import inspect_agentleak_release as release


def load_journal_module():
    path = ROOT / "scripts" / "evaluate_independent_mas_journal_v1.py"
    spec = importlib.util.spec_from_file_location("journal_eval_for_agentleak", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


J = load_journal_module()
TWO = J.load_two_layer()
TWO.PRIV_TOOLS.add("external_release")
J.PRIV_TOOLS.add("external_release")

UPSTREAM_REVISION = release.REVISION
PHASES = ("early", "late")
FORBIDDEN_FEATURE_KEYS = {"results", "attack_family", "model", "label", "leaked_fields", "has_leak"}

# Surface-only, prospective indicators.  No vault value or final label is
# used.  The API monitor may inspect the message text but need not know the
# hidden vault contents.
SURFACE_RE = re.compile(
    r"(?:canary[_-]|social\s+security|\bssn\b|diagnosis|medications?|allergies?|"
    r"insurance(?:[_ -]?id)?|confidential|secret|salary|routing|account\s+number|"
    r"date\s+of\s+birth|\bdob\b|phone\s+number|private\s+notes?)",
    re.IGNORECASE,
)


def fit_model_fast(train_eps, y, bn, seed):
    """Same regularized objective as the frozen monitor, with a bounded
    external-validation budget.  The independent journal result keeps its
    original 500-iteration budget; this transfer check is not that result."""
    rng = np.random.default_rng(seed)
    x0 = np.array([-2., 3., .2, .8, -.4, .1, .1, -2., 4., .2, 1., 1., .3], dtype=float) + rng.normal(0, .03, 13)

    def objective(t):
        p = np.asarray([TWO._propagate(t, TWO.event_features(e), bn) for e in train_eps])
        bce = -np.mean(y * np.log(np.clip(p, 1e-7, 1 - 1e-7)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)))
        return float(bce + 1e-3 * np.sum(t * t))

    from scipy.optimize import minimize
    result = minimize(objective, x0, method="L-BFGS-B", bounds=[(-8., 8.)] * 13,
                      options={"maxiter": 120, "ftol": 1e-9})
    return result.x, {"success": bool(result.success), "iterations": int(result.nit), "train_loss": float(result.fun), "message": str(result.message)}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface_signal(text: str) -> bool:
    return bool(SURFACE_RE.search(text or ""))


def make_episode(trace: dict, message_count: int) -> dict:
    """Project one pre-final prefix into the generic two-layer event schema."""
    messages = trace["channel_messages"][:message_count]
    agents = ["single_agent", "coordinator", "worker", "memory", "external-user"]
    # Functional workflow remains acyclic.  The observed worker->coordinator
    # return message is runtime evidence, not a causal-discovery claim.
    workflow_edges = [
        ["worker", "coordinator"],
        ["coordinator", "external-user"],
        ["worker", "memory"],
        ["single_agent", "external-user"],
    ]
    events = [
        {"event_type": "input_received", "actor": "worker", "data_class": "untrusted", "sequence": 1},
        {"event_type": "input_received", "actor": "single_agent", "data_class": "untrusted", "sequence": 1},
    ]
    for i, message in enumerate(messages, start=2):
        source, target = message.get("source"), message.get("target")
        if source not in agents or target not in agents:
            continue
        event_type = "delegation" if message.get("channel") == "C2" else "message_sent"
        edge_kind = "workflow" if [source, target] in workflow_edges else "unexpected"
        events.append({
            "event_type": event_type,
            "actor": source,
            "target": target,
            "data_class": "untrusted" if surface_signal(message.get("content", "")) else "trusted",
            "edge_kind": edge_kind,
            "edge_confidence": 1.0,
            "sequence": i,
        })
    # The monitor knows that the coordinator is the eventual external release
    # sink from the declared workflow, while the final message itself is not
    # present in either prefix.
    events.append({"event_type": "tool_call", "actor": "coordinator", "tool": "external_release", "sequence": 99})
    # Public prefix text is an allowed API observation.  The final release
    # message and vault values are still absent, so this cannot expose the
    # AgentLeak target while enabling the runtime semantic evidence subgraph.
    prefix_text = "\n".join(str(message.get("content", "")) for message in messages)
    return {"agents": agents, "workflow_dag": {"edges": workflow_edges}, "events": events,
            "text": prefix_text}


def load_rows(upstream: Path):
    rows = []
    paths = sorted((upstream / "benchmarks/ieee_repro/results/traces").glob("trace_*.json"))
    for path in paths:
        trace = read_json(path)
        checked = release.inspect_trace(trace)
        if not checked["eligible"]:
            continue
        # Recompute the label from the reviewed public trace semantics rather
        # than trusting the stored results object.
        final_fields = release.matched_fields(
            trace["channel_messages"][4]["content"],
            trace["input"]["vault"],
            trace["input"]["request"],
            trace["input"]["allowed_set"].get("fields", []),
        )
        y = int(bool(final_fields))
        group = checked["request_identity"]
        rows.append({
            "id": trace["trace_id"],
            "group": group,
            "vertical": trace.get("vertical"),
            "y": y,
            "early": make_episode(trace, 2),
            "late": make_episode(trace, 4),
        })
    if not rows:
        raise ValueError("no eligible AgentLeak traces")
    return rows


def metric(y, p, pred=None):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= .5 if pred is None else np.asarray(pred, dtype=bool)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_predictions": int(pred.sum()),
    }


def bootstrap(y, p, pred, groups, seed=20261003, n=500):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.asarray(sorted(set(groups)))
    values = {k: [] for k in ("auroc", "auprc", "brier", "f1")}
    for _ in range(n):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        values["auroc"].append(float(roc_auc_score(y[idx], p[idx])))
        values["auprc"].append(float(average_precision_score(y[idx], p[idx])))
        values["brier"].append(float(brier_score_loss(y[idx], p[idx])))
        values["f1"].append(float(f1_score(y[idx], pred[idx], zero_division=0)))
    return {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "n": len(v)} for k, v in values.items()}


def run(rows, out: Path, folds: int, seed: int):
    y = np.asarray([r["y"] for r in rows], dtype=int)
    groups = np.asarray([r["group"] for r in rows])
    traces = [{"agents": r["early"]["agents"], "workflow_dag": r["early"]["workflow_dag"], "events": []} for r in rows]
    # Build the same observable vectors used by the frozen two-line monitor.
    wf = np.asarray([J.workflow_vector(r["late"]) for r in rows])
    rt = np.asarray([J.runtime_vector(r["late"]) for r in rows])
    names = ("workflow_bn", "runtime_bn", "workflow_logistic", "runtime_logistic", "hierarchical_fusion")
    results = {phase: {name: np.zeros(len(rows)) for name in names} for phase in PHASES}
    preds = {phase: {name: np.zeros(len(rows), dtype=bool) for name in names} for phase in PHASES}
    fixed = {phase: {name: np.zeros(len(rows), dtype=bool) for name in names} for phase in PHASES}
    split = J.StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_records = []
    for fold, (train, test) in enumerate(split.split(wf, y, groups), start=1):
        wf_bn = J.workflow_bn_fit(wf[train], y[train])
        p_wf_train = J.workflow_bn_score(wf[train], wf_bn)
        p_wf_test = J.workflow_bn_score(wf[test], wf_bn)
        rt_train, rt_test, _ = J.fit_logistic(rt[train], y[train], rt[test], seed + fold)
        phase_eps = {phase: ([rows[i][phase] for i in train], [rows[i][phase] for i in test]) for phase in PHASES}
        phase_fit_records = {}
        for phase in PHASES:
            xwf = np.asarray([J.workflow_vector(e) for e in phase_eps[phase][0]])
            xwf_te = np.asarray([J.workflow_vector(e) for e in phase_eps[phase][1]])
            xrt = np.asarray([J.runtime_vector(e) for e in phase_eps[phase][0]])
            xrt_te = np.asarray([J.runtime_vector(e) for e in phase_eps[phase][1]])
            # The BN parameters are fit separately per prefix, preserving the
            # temporal claim and avoiding a future-prefix feature.
            p0 = J.workflow_bn_fit(xwf, y[train]); a_tr = J.workflow_bn_score(xwf, p0); a_te = J.workflow_bn_score(xwf_te, p0)
            b0 = TWO.fit_bn(phase_eps[phase][0], y[train]); bt, opt = fit_model_fast(phase_eps[phase][0], y[train], b0, seed + 600 + fold)
            b_tr = np.asarray([TWO._propagate(bt, TWO.event_features(e), b0) for e in phase_eps[phase][0]])
            b_te = np.asarray([TWO._propagate(bt, TWO.event_features(e), b0) for e in phase_eps[phase][1]])
            c_tr, c_te, _ = J.fit_logistic(xwf, y[train], xwf_te, seed + 700 + fold)
            d_tr, d_te, _ = J.fit_logistic(xrt, y[train], xrt_te, seed + 800 + fold)
            # Prefix fusion uses only prefix-specific base scores.  A small
            # train-only logistic head is sufficient for the external check.
            ftr = np.c_[a_tr, b_tr, c_tr, d_tr]
            fte = np.c_[a_te, b_te, c_te, d_te]
            fm = LogisticRegression(C=1.0, max_iter=1000, random_state=seed + 900 + fold).fit(ftr, y[train])
            fp_tr, fp_te = fm.predict_proba(ftr)[:, 1], fm.predict_proba(fte)[:, 1]
            phase_fit_records[phase] = {"optimizer_success": bool(opt.get("success", False)), "fusion_coefficients": [float(v) for v in fm.coef_[0]]}
            scores = {"workflow_bn": (a_tr, a_te), "runtime_bn": (b_tr, b_te), "workflow_logistic": (c_tr, c_te), "runtime_logistic": (d_tr, d_te), "hierarchical_fusion": (fp_tr, fp_te)}
            for name, (trp, tep) in scores.items():
                results[phase][name][test] = tep
                preds[phase][name][test] = tep >= J.best_threshold(y[train], trp)
                fixed[phase][name][test] = tep >= J.threshold_at_fpr(y[train], trp)
        fold_records.append({"fold": fold, "train": len(train), "test": len(test), "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])), "positive_test": int(y[test].sum()), "phase_fit": phase_fit_records})
        print(json.dumps({"fold": fold, "completed": True, "test": len(test)}), flush=True)
    summaries = {}
    fixed_summaries = {}
    for phase in PHASES:
        summaries[phase] = {name: metric(y, results[phase][name], preds[phase][name]) for name in names}
        fixed_summaries[phase] = {name: {"fpr": float(((fixed[phase][name]) & (y == 0)).sum() / max(1, (y == 0).sum())), "recall": float(((fixed[phase][name]) & (y == 1)).sum() / max(1, (y == 1).sum())), "predicted_positive": int(fixed[phase][name].sum())} for name in names}
    out.mkdir(parents=True, exist_ok=False)
    pred_path = out / "predictions_oof.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"id": row["id"], "group": row["group"], "label": row["y"], **{phase: {name: float(results[phase][name][i]) for name in names} for phase in PHASES}}, sort_keys=True) + "\n")
    metrics = {"dataset": {"n": len(rows), "positive": int(y.sum()), "negative": int((1-y).sum()), "request_groups": int(len(set(groups))), "upstream_revision": UPSTREAM_REVISION, "source": str((ROOT / "data/external/AgentLeak").resolve())}, "protocol": {"split": "five-fold StratifiedGroupKFold by request identity", "features": "static acyclic functional DAG + pre-final observable message markers", "forbidden_features": sorted(FORBIDDEN_FEATURE_KEYS), "label": "recomputed final coordinator-to-user literal disclosure", "fusion": "train-only prefix scores; no future output; external transfer diagnostic", "new_api_calls": 0}, "methods": summaries, "fixed_fpr_5pct": fixed_summaries, "bootstrap_hierarchical_fusion": {phase: bootstrap(y, results[phase]["hierarchical_fusion"], preds[phase]["hierarchical_fusion"], groups) for phase in PHASES}, "folds": fold_records}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# External AgentLeak MAS journal validation v1", "", "Five-fold grouped validation on the public AgentLeak release; request identities are held out between folds.", "The final output and stored result fields are never features. Prefixes stop before the final coordinator-to-user message.", "", f"- n={len(rows)}, positives={int(y.sum())}, request groups={len(set(groups))}", "", "## Late prefix (before final release)", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m = summaries["late"][name]; report.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['brier']:.3f} |")
    report += ["", "## Early prefix", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m = summaries["early"][name]; report.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['brier']:.3f} |")
    report += ["", "## Scope", "This is an external privacy-leak transfer validation, not a claim that literal-leak labels cover all MAS hazards. Surface markers are observable text cues; they do not use vault values. The report is independent of the frozen synthetic journal cohort but still requires semantic labels, missingness stress and online cost evaluation before submission.", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (out / "PLAN.json").write_text(json.dumps({"schema": "agentleak-external-journal-v1", "upstream_revision": UPSTREAM_REVISION, "rows": len(rows), "positive": int(y.sum()), "request_groups": len(set(groups)), "source_sha256": {"evaluate_agentleak_external_journal_v1.py": sha(Path(__file__))}, "new_api_calls": 0, "frozen_main_result_untouched": True}, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=ROOT / "data/external/AgentLeak")
    parser.add_argument("--out", type=Path, default=ROOT / "results/submission/development/agentleak_external_journal_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    run(load_rows(args.upstream.resolve()), args.out.resolve(), args.folds, args.seed)
