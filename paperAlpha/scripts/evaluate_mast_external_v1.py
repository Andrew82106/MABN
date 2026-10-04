"""External MAST/MAD audit using trajectory-only public text features.

The released MAST labels are not used during feature construction. The full
MAD score is a secondary external benchmark because its labels come from the
released MAST annotation pipeline; the 19-trace human subset is reported as a
small, independent sanity check and never pooled into the main result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/external/MAST/MAD_full_dataset.json"
HUMAN = ROOT / "data/external/MAST/MAD_human_labelled_dataset.json"
DEFAULT_OUT = ROOT / "results/submission/final_eval/mast_external_v1"


def trace_text(row: dict) -> str:
    trace = row.get("trace", "")
    if isinstance(trace, dict):
        return str(trace.get("trajectory", ""))
    return str(trace)


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def public_features(row: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract two inspectable evidence lines from trajectory text only."""
    text = trace_text(row)
    low = text.lower()
    role_hits = set()
    for pat in (
        r"response from\s+([^\n:]{2,80})",
        r"message to\s+([^\n:]{2,80})",
        r"sending message to\s+([^\n:]{2,80})",
        r"(?:agent|role)[ _-]*([a-z][a-z0-9_-]{1,30})",
    ):
        role_hits.update(x.strip().lower() for x in re.findall(pat, text, flags=re.I))
    agent_count = min(max(1, len(role_hits)), 12)
    messages = count(r"(?:message to|response from|send_message|sending message)", text)
    tools = count(r"(?:code execution output|callex?tension|tool call|api call|execute\(|tool_name)", text)
    delegations = count(r"(?:delegate|handoff|send_message|message to)", text)
    errors = count(r"(?:error|exception|traceback|failed|failure|timeout|unable to)", text)
    verify = count(r"(?:verify|verification|validate|validation|assert|test result|review)", text)
    stop = count(r"(?:termination|terminate|stop condition|finished|completed|final answer|done)", text)
    repetition = count(r"(?:repeat|again|retry|re-run|backtrack|reset|same step)", text)
    conflict = count(r"(?:contradict|disagree|inconsistent|ignore(?:d)? (?:the )?(?:suggestion|message)|withhold|did not share)", low)
    injection = count(r"(?:ignore previous|system prompt|prompt injection|untrusted|malicious instruction)", low)
    # These features represent missing normative evidence, not hidden labels.
    missing_verify = int(verify == 0)
    missing_stop = int(stop == 0)
    repeated = int(repetition > 0)
    runtime_failure = int(errors > 0 or conflict > 0 or repeated)
    normative = np.asarray([
        int(agent_count > 1), int(messages > 0), int(delegations > 0),
        missing_verify, missing_stop, int(tools > 0), int(injection > 0),
    ], dtype=float)
    runtime = np.asarray([
        min(messages / 20.0, 1.0), min(tools / 20.0, 1.0),
        min(errors / 10.0, 1.0), min(conflict / 5.0, 1.0),
        float(repeated), float(missing_verify), float(missing_stop),
        float(runtime_failure),
    ], dtype=float)
    audit = {
        "trajectory_chars": len(text), "agent_role_hits": len(role_hits),
        "messages": messages, "tools": tools, "delegations": delegations,
        "errors": errors, "verification_mentions": verify,
        "termination_mentions": stop, "repetition_mentions": repetition,
        "conflict_mentions": conflict, "injection_mentions": injection,
    }
    return normative, runtime, audit


def label_full(row: dict) -> int:
    return int(any(bool(v) for v in (row.get("mast_annotation") or {}).values()))


def label_human(row: dict) -> int:
    active = []
    for ann in row.get("annotations", []):
        votes = sum(bool(ann.get(k)) for k in ("annotator_1", "annotator_2", "annotator_3"))
        if votes >= 2:
            active.append(ann)
    return int(bool(active))


def fit_bn(x: np.ndarray, y: np.ndarray) -> dict:
    prior = float((y.sum() + 1.0) / (len(y) + 2.0))
    cpt = []
    for j in range(x.shape[1]):
        m = x[:, j] > 0
        cpt.append(float((y[m].sum() + 1.0) / (m.sum() + 2.0)) if m.any() else prior)
    return {"prior": prior, "cpt": cpt}


def score_bn(x: np.ndarray, bn: dict) -> np.ndarray:
    out = []
    for row in x:
        active = np.flatnonzero(row > 0)
        if len(active) == 0:
            out.append(bn["prior"])
        else:
            prod = 1.0
            for j in active:
                prod *= 1.0 - min(.995, max(.001, bn["cpt"][int(j)]))
            out.append(1.0 - prod)
    return np.asarray(out, dtype=float)


def folds(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    n_groups = len(np.unique(groups))
    n = min(5, n_groups)
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=23).split(x, y, groups)
    return GroupKFold(n_splits=n).split(x, y, groups)


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": safe_auc(y, p),
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    vals = np.unique(np.r_[0.0, p, 1.0])
    return float(max(vals, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def fit_fold(train_idx, test_idx, xn, xr, y, groups, seed=23):
    bn_n = fit_bn(xn[train_idx], y[train_idx]); bn_r = fit_bn(xr[train_idx], y[train_idx])
    pn_train = score_bn(xn[train_idx], bn_n); pr_train = score_bn(xr[train_idx], bn_r)
    pn_test = score_bn(xn[test_idx], bn_n); pr_test = score_bn(xr[test_idx], bn_r)
    # Inner OOF scores prevent fitting the fusion head on in-sample BN scores.
    inner_groups = groups[train_idx]
    inner_y = y[train_idx]
    inner = np.zeros((len(train_idx), 2), dtype=float)
    inner_split = list(folds(xn[train_idx], inner_y, inner_groups))
    for ii, (a, b) in enumerate(inner_split):
        ta = train_idx[a]; tb = train_idx[b]
        ib_n = fit_bn(xn[ta], y[ta]); ib_r = fit_bn(xr[ta], y[ta])
        inner[b, 0] = score_bn(xn[tb], ib_n); inner[b, 1] = score_bn(xr[tb], ib_r)
    fusion = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(inner, inner_y)
    p_train = fusion.predict_proba(np.c_[pn_train, pr_train])[:, 1]
    p_test = fusion.predict_proba(np.c_[pn_test, pr_test])[:, 1]
    threshold = best_threshold(inner_y, fusion.predict_proba(inner)[:, 1])
    flat = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
        np.c_[xn[train_idx], xr[train_idx]], y[train_idx])
    flat_test = flat.predict_proba(np.c_[xn[test_idx], xr[test_idx]])[:, 1]
    rule_test = np.maximum(pn_test, pr_test)
    return p_test, flat_test, rule_test, threshold


def evaluate(rows: list[dict], out: Path) -> dict:
    xn, xr, audits, y, groups = [], [], [], [], []
    for row in rows:
        a, b, audit = public_features(row); xn.append(a); xr.append(b); audits.append(audit)
        y.append(label_full(row)); groups.append(f"{row.get('mas_name','unknown')}::{row.get('benchmark_name','unknown')}")
    xn = np.asarray(xn); xr = np.asarray(xr); y = np.asarray(y, dtype=int); groups = np.asarray(groups)
    p_fusion = np.zeros(len(y)); p_flat = np.zeros(len(y)); p_rule = np.zeros(len(y)); thresholds = []
    for train_idx, test_idx in folds(xn, y, groups):
        a, b, c, threshold = fit_fold(train_idx, test_idx, xn, xr, y, groups)
        p_fusion[test_idx] = a; p_flat[test_idx] = b; p_rule[test_idx] = c; thresholds.append(threshold)
    # Use the mean fold threshold only for a descriptive classification number.
    threshold = float(statistics.mean(thresholds))
    report = {
        "n": int(len(y)), "positives": int(y.sum()), "groups": int(len(np.unique(groups))),
        "label_source": "released MAST annotation (secondary external benchmark; not human gold)",
        "feature_boundary": "trajectory text only; excludes mast_annotation, MAS/benchmark identifiers and human annotations",
        "methods": {
            "two_line_bn_fusion": metrics(y, p_fusion, threshold),
            "flat_logistic_public_text": metrics(y, p_flat, threshold),
            "max_line_rule": metrics(y, p_rule, threshold),
        },
        "threshold_mean": threshold,
        "group_definition": "mas_name::benchmark_name",
        "predictions": [], "feature_audit": audits,
    }
    for row, pf, pl, pr in zip(rows, p_fusion, p_flat, p_rule):
        report["predictions"].append({"trace_id": row.get("trace_id"), "mas_name": row.get("mas_name"), "benchmark_name": row.get("benchmark_name"), "y": label_full(row), "fusion": float(pf), "flat": float(pl), "rule": float(pr)})
    for field in ("mas_name", "benchmark_name"):
        by_group = {}
        for value in sorted({str(x[field]) for x in report["predictions"]}):
            subset = [x for x in report["predictions"] if str(x[field]) == value]
            gy = np.asarray([x["y"] for x in subset], dtype=int)
            gp = np.asarray([x["fusion"] for x in subset], dtype=float)
            by_group[value] = {
                "n": int(len(gy)), "positives": int(gy.sum()),
                "auroc": safe_auc(gy, gp),
                "auprc": float(average_precision_score(gy, gp)) if gy.sum() else None,
            }
        report[f"by_{field}"] = by_group
    return report


def human_audit(rows: list[dict], full_rows: list[dict]) -> dict:
    xn, xr, y = [], [], []
    for row in full_rows:
        a, b, _ = public_features(row); xn.append(a); xr.append(b); y.append(label_full(row))
    xn = np.asarray(xn); xr = np.asarray(xr); y = np.asarray(y, dtype=int)
    bn_n = fit_bn(xn, y); bn_r = fit_bn(xr, y)
    # The full MAST annotation is training material for this secondary audit;
    # human rows are not included in fitting unless their trajectory duplicates a full row.
    p = []
    yh = []
    for row in rows:
        a, b, _ = public_features(row); p.append(float(max(score_bn(a[None, :], bn_n)[0], score_bn(b[None, :], bn_r)[0]))); yh.append(label_human(row))
    yh = np.asarray(yh, dtype=int); p = np.asarray(p)
    return {"n": int(len(yh)), "positives": int(yh.sum()), "label_source": "majority vote of three human annotators", "note": "19-trace pilot; 18 positives, so this is descriptive only", "max_line_score": metrics(yh, p, best_threshold(yh, p) if len(np.unique(yh)) > 1 else .5)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--full", type=Path, default=FULL); ap.add_argument("--human", type=Path, default=HUMAN); ap.add_argument("--out", type=Path, default=DEFAULT_OUT); args = ap.parse_args()
    full_rows = json.loads(args.full.read_text(encoding="utf-8")); human_rows = json.loads(args.human.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    report = evaluate(full_rows, args.out); report["human_pilot"] = human_audit(human_rows, full_rows)
    report["data_sha256"] = {"full": hashlib.sha256(args.full.read_bytes()).hexdigest(), "human": hashlib.sha256(args.human.read_bytes()).hexdigest()}
    (args.out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pred = "\n".join(json.dumps(x, ensure_ascii=False) for x in report["predictions"]) + "\n"
    (args.out / "predictions.jsonl").write_text(pred, encoding="utf-8")
    (args.out / "FEATURE_AUDIT.json").write_text(json.dumps(report["feature_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    f = report["methods"]["two_line_bn_fusion"]
    h = report["human_pilot"]
    mas_aucs = [v["auroc"] for v in report["by_mas_name"].values() if v["auroc"] is not None]
    md = ["# External MAST/MAD audit", "", f"- Full release: {report['n']} traces, {report['positives']} positive MAST annotations, {report['groups']} MAS×benchmark groups.", "- Features: trajectory text only; labels, identifiers and annotations excluded.", f"- Two-line BN fusion: AUROC {f['auroc']:.3f}, AUPRC {f['auprc']:.3f}, F1 {f['f1']:.3f}, Brier {f['brier']:.3f}.", f"- MAS-level AUROC range: {min(mas_aucs):.3f}--{max(mas_aucs):.3f}; transfer is heterogeneous and the pooled score must not hide system-specific failures.", f"- Human pilot: {h['n']} traces, {h['positives']} positive; descriptive only.", "", "This is external transfer evidence, not an independent human-gold superiority result: the full release uses the published MAST annotation pipeline, and the human subset is too small and imbalanced for a powered estimate.", ""]
    (args.out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"n": report["n"], "positives": report["positives"], "groups": report["groups"], "fusion": f, "human_pilot": h}, ensure_ascii=False))


if __name__ == "__main__":
    main()
