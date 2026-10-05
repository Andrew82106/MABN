"""External ATBench transfer audit for the frozen two-layer monitor.

ATBench labels and metadata are read only after feature construction.  The
monitor is fitted on independent_mas_v3 and is never tuned on ATBench.  This
is a single/general-agent transfer audit, not a MAS headline benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_two_layer_v2 as monitor


READ_HINTS = re.compile(
    r"(?:^|_)(?:get|fetch|search|lookup|list|read|extract|scrape|parse|"
    r"summarize|check|validate|verify|find|download|ocr|convert|capture|"
    r"take|predict|format|generate)(?:_|$)|(?:_)(?:info|data|details|"
    r"results?|report|text|metadata|reviews?|profile)(?:_|$)", re.I
)


def _action_name(raw):
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return obj.get("name") if isinstance(obj, dict) and isinstance(obj.get("name"), str) else None


def _tool_class(name: str) -> str:
    """Map arbitrary ATBench tool names to the monitor's fixed sink vocabulary.

    This is a pre-registered name-only policy prior, independent of ATBench
    labels/reasons.  Read-like tools are non-privileged; other tools are
    represented as a potentially side-effecting shell operation.
    """
    return "search.read" if READ_HINTS.search(name or "") else "shell.exec"


def adapt_record(row: dict, index: int) -> tuple[dict, dict]:
    """Create an episode from public ``contents`` only and an audit row."""
    messages = row.get("contents")
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        messages = messages[0]
    if not isinstance(messages, list):
        messages = []
    agents = ["agent"]
    events = []
    user_text, public_text = [], []
    seq = 0
    tool_names = []
    parse_failures = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            public_text.append(content)
            if role == "user":
                user_text.append(content)
        if role == "user" and isinstance(content, str):
            events.append({"actor": "agent", "event_type": "input_received",
                           "data_class": "untrusted", "sequence": seq,
                           "content": content})
            seq += 1
        if role == "agent":
            if isinstance(message.get("thought"), str):
                public_text.append(message["thought"])
            action = message.get("action")
            name = _action_name(action)
            if isinstance(action, str) and action.startswith("{") and name is None:
                parse_failures += 1
            if name:
                tool_names.append(name)
                events.append({"actor": "agent", "event_type": "tool_call",
                               "tool": _tool_class(name), "sequence": seq,
                               "content": " ".join(x for x in
                                                   (message.get("thought"), action)
                                                   if isinstance(x, str))})
                seq += 1
    episode = {"episode_id": f"atbench-{index:04d}", "agents": agents,
               "events": events, "task_text": " ".join(user_text),
               "text": " ".join(public_text)}
    audit = {"episode_id": episode["episode_id"], "tool_names": tool_names,
             "parse_failures": parse_failures, "event_count": len(events),
             "user_message_count": len(user_text),
             "public_text_chars": len(episode["text"])}
    return episode, audit


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(y, p, threshold=.5):
    pred = p >= threshold
    return {"f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "threshold": threshold, "predicted_positive": int(pred.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("paperAlpha/data/external/ATBench/test.json"))
    ap.add_argument("--train-traces", type=Path, default=Path("paperAlpha/results/independent_mas_v3/traces_public.jsonl"))
    ap.add_argument("--train-labels", type=Path, default=Path("paperAlpha/results/independent_mas_v3/labels.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("paperAlpha/results/submission/development/atbench_external_20261005"))
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Fit only on the independent MAS benchmark.  No ATBench metadata is
    # accessed until after all adapted features have been constructed.
    train_eps = monitor.read_jsonl(args.train_traces)
    labels = {r["episode_id"]: int(r["label"]) for r in read_jsonl(args.train_labels)}
    y_train = np.asarray([labels[e["episode_id"]] for e in train_eps], dtype=int)
    bn = monitor.fit_bn(train_eps, y_train)
    theta, optimizer = monitor.fit_model(train_eps, y_train, bn, args.seed)

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    episodes, audit_rows = [], []
    for i, row in enumerate(raw):
        ep, audit = adapt_record(row, i)
        episodes.append(ep); audit_rows.append(audit)
    # Labels are joined only now, after adapter construction.
    y = np.asarray([int(str(row.get("label")) == "1") for row in raw], dtype=int)
    p = np.asarray([monitor._propagate(theta, monitor.event_features(ep), bn)
                    for ep in episodes])
    # Feature baselines are fit on independent_mas_v3 and transferred frozen.
    trparsed = [monitor.event_features(e) for e in train_eps]
    teparsed = [monitor.event_features(e) for e in episodes]
    agg_names = ["event_count", "agent_count", "message_count", "privileged_calls", "untrusted_inputs"]
    xtr = np.asarray([[q["aggregate"][k] for k in agg_names] for q in trparsed])
    xte = np.asarray([[q["aggregate"][k] for k in agg_names] for q in teparsed])
    bs = monitor.baseline_scores(episodes, bn, (xtr, y_train))
    baseline_metrics = {k: metric(y, v) for k, v in bs.items() if v is not None}
    predictions = []
    for i, (ep, pp) in enumerate(zip(episodes, p)):
        point, explanation = monitor._propagate(theta, monitor.event_features(ep), bn, explain=True)
        # Keep labels in the evaluation output only; they were not in features.
        predictions.append({"episode_id": ep["episode_id"], "label": int(y[i]),
                            "risk_probability": float(point), "explanation": explanation})
    metrics = {"ours": metric(y, p), "baselines": baseline_metrics}
    strata = {}
    for key in ("risk_source", "failure_mode", "real_world_harm"):
        groups = {}
        for i, row in enumerate(raw):
            value = row.get(key, "unknown")
            groups.setdefault(str(value), []).append(i)
        strata[key] = {name: {"n": len(ix), "positive": int(y[ix].sum()),
                              "metrics": metric(y[ix], p[ix]) if len(set(y[ix])) > 1 else None}
                       for name, ix in sorted(groups.items())}
    audit = {
        "n": len(raw), "positive": int(y.sum()), "negative": int((1-y).sum()),
        "label_balance": float(y.mean()), "unique_risk_sources": len({r.get("risk_source") for r in raw}),
        "unique_failure_modes": len({r.get("failure_mode") for r in raw}),
        "unique_harms": len({r.get("real_world_harm") for r in raw}),
        "parse_failures": int(sum(r["parse_failures"] for r in audit_rows)),
        "episodes_with_tools": int(sum(bool(r["tool_names"]) for r in audit_rows)),
        "mean_event_count": float(np.mean([r["event_count"] for r in audit_rows])),
        "mean_public_text_chars": float(np.mean([r["public_text_chars"] for r in audit_rows])),
        "tool_name_count": len({t for r in audit_rows for t in r["tool_names"]}),
        "feature_label_exclusion": ["label", "risk_source", "failure_mode", "reason", "real_world_harm"],
    }
    report = {
        "dataset": {"name": "ATBench test", "path": str(args.input), "role": "external single/general-agent transfer"},
        "training": {"dataset": "independent_mas_v3", "n": len(train_eps),
                     "positive": int(y_train.sum()), "seed": args.seed,
                     "optimizer": optimizer, "atbench_labels_used_for_fit": False},
        "model": {"name": "TwoLayerBNGraphMonitor-v2", "layer1": "frozen local BN CPT",
                  "layer2": "frozen edge-gated noisy-OR propagation + fixed semantic policy evidence",
                  "threshold": .5, "causal_claim": False},
        "metrics": metrics, "strata": strata, "audit": audit,
        "limitations": [
            "ATBench is not a multi-agent benchmark and is reported only as external transfer.",
            "Labels are benchmark annotations; no ATBench label, reason, failure mode, or harm field enters features or fitting.",
            "Tool-name normalization is a fixed name-only policy prior; it is not learned or tuned on ATBench.",
            "Results do not replace independent human/labelling validation for MAS deployment.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "predictions.jsonl").write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in predictions) + "\n", encoding="utf-8")
    (args.out / "adapter_audit.jsonl").write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in audit_rows) + "\n", encoding="utf-8")
    lines = ["# ATBench external transfer audit", "", "冻结模型在 independent_mas_v3 上拟合；ATBench 仅在特征构造完成后用于计算指标。ATBench 是单代理/通用 Agent 迁移集，不是 MAS 主结果。", "", "## Overall", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in [("two-layer (ours)", metrics["ours"])] + list(metrics["baselines"].items()):
        lines.append(f"| {name} | {value['f1']:.3f} | {value['precision']:.3f} | {value['recall']:.3f} | {value['auroc']:.3f} | {value['auprc']:.3f} | {value['brier']:.3f} |")
    lines += ["", "## Audit", "", f"- n={audit['n']}; positive={audit['positive']}; negative={audit['negative']}; parse failures={audit['parse_failures']}.", f"- {audit['tool_name_count']} unique tool names; mean public text length={audit['mean_public_text_chars']:.1f} characters.", "- Excluded from features: label, risk_source, failure_mode, reason, real_world_harm.", "", "## Boundaries", "", "- This is external transfer evidence only; it must not be merged into the MAS headline table.", "- Tool-name normalization is a fixed prior, and benchmark labels are not independent human deployment labels."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
