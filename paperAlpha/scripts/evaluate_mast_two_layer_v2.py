"""Public MAST/MAD transfer audit for the two-layer monitor v2.

This adapter is deliberately conservative: only the released trajectory text is
used to construct observable agents, message edges, and tool-call events.  The
released MAST annotations are joined *after* feature construction and are used
only as an external evaluation target.  ``--no-semantic`` removes all message
text from the fixed policy-evidence subgraph, giving an apples-to-apples
structural-line ablation of ``evaluate_two_layer_v2.py``.

The output belongs under ``results/submission/development`` and is not a main
submission result.  MAST labels are released annotation labels, not a fresh
independent security gold standard.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/external/MAST/MAD_full_dataset.json"
DEFAULT_OUT = ROOT / "results/submission/development/mast_two_layer_v2_20261005"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_two_layer_v2 as _v2  # noqa: E402

# v2 deliberately keeps its public functional API simple and reparses each
# episode inside the optimizer.  MAST trajectories are long, so memoize the
# immutable projection for this transfer audit; this changes no features.
_EVENT_CACHE: dict[tuple[int, str], dict[str, Any]] = {}
_ORIGINAL_EVENT_FEATURES = _v2.event_features


def _cached_event_features(ep: dict[str, Any]) -> dict[str, Any]:
    # Include task_text so the structural ablation (empty text) can never
    # reuse a cached semantic parse if Python recycles an object id.
    key = (id(ep), str(ep.get("task_text", "")))
    parsed = _EVENT_CACHE.get(key)
    if parsed is None:
        parsed = _ORIGINAL_EVENT_FEATURES(ep)
        _EVENT_CACHE[key] = parsed
    return parsed


_v2.event_features = _cached_event_features
evaluate_split = _v2.evaluate_split


def _name(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw.strip())
    raw = re.sub(r"\s+Agent$", "", raw, flags=re.I)
    return raw.lower().replace(" ", "_") or "unknown"


def _label(row: dict[str, Any]) -> int:
    return int(any(bool(v) for v in (row.get("mast_annotation") or {}).values()))


def _event(event_type: str, actor: str, target: str, content: str = "", *, tool: str | None = None) -> dict[str, Any]:
    low = content.lower()
    data_class = "untrusted" if any(x in low for x in ("prompt injection", "injected", "malicious instruction", "untrusted")) else "trusted"
    kind = "workflow" if event_type in {"message_sent", "delegation"} else "tool"
    return {
        "event_type": event_type,
        "actor": actor,
        "target": target,
        "content": content,
        "data_class": data_class,
        "edge_kind": kind,
        "edge_confidence": 1.0,
        "tool": tool,
    }


def row_to_episode(row: dict[str, Any], *, include_text: bool) -> dict[str, Any]:
    """Project one MAST trajectory into the v2 public event contract."""
    trace = row.get("trace", "")
    if isinstance(trace, dict):
        trace = trace.get("trajectory", "")
    trace = str(trace)
    supervisor = "supervisor"
    agents = {supervisor}
    events: list[dict[str, Any]] = []
    # Message headers are stable in the MAST release.  We retain only the
    # short header as event content; the full trace is optional task text.
    for line in trace.splitlines():
        text = line.strip()
        if not text:
            continue
        m = re.match(r"Message to (.+?)(?:\s+Agent)?$", text, re.I)
        if m:
            target = _name(m.group(1)); agents.add(target)
            events.append(_event("message_sent", supervisor, target, text))
            continue
        m = re.match(r"Response from (.+?)(?:\s+Agent)?$", text, re.I)
        if m:
            actor = _name(m.group(1)); agents.add(actor)
            events.append(_event("message_sent", actor, supervisor, text))
            continue
        m = re.match(r"Reply from (.+?) to (?:Supervisor)(?:\s+Agent)?$", text, re.I)
        if m:
            actor = _name(m.group(1)); agents.add(actor)
            events.append(_event("message_sent", actor, supervisor, text))
            continue
        m = re.search(r"send_message\(\s*app_name\s*=\s*['\"]([^'\"]+)", text, re.I)
        if m:
            target = _name(m.group(1)); agents.add(target)
            events.append(_event("delegation", supervisor, target, text))
            continue
        if re.search(r"(?:apis\.|execute\(|tool call|call_extension|calle?xtension)", text, re.I):
            tool = "shell.exec"
            if re.search(r"(?:payment|purchase|transfer|venmo)", text, re.I):
                tool = "payments.write"
            events.append(_event("tool_call", supervisor, supervisor, text, tool=tool))
    return {
        "episode_id": f"mast::{row.get('mas_name','unknown')}::{row.get('benchmark_name','unknown')}::{row.get('trace_id')}",
        "agents": sorted(agents),
        "events": events,
        "task_text": trace if include_text else "",
        "label": _label(row),
        "mas_name": row.get("mas_name"),
        "benchmark_name": row.get("benchmark_name"),
    }


def _folds(eps: list[dict[str, Any]], y: np.ndarray, groups: np.ndarray):
    n = min(5, len(np.unique(groups)))
    x = np.zeros((len(eps), 1), dtype=float)
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=23).split(x, y, groups)
    return GroupKFold(n_splits=n).split(x, y, groups)


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = p >= 0.5
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)),
        "threshold": 0.5,
    }


def evaluate(rows: list[dict[str, Any]], *, include_text: bool, seed: int = 20261005) -> dict[str, Any]:
    eps = [row_to_episode(row, include_text=include_text) for row in rows]
    y = np.asarray([e["label"] for e in eps], dtype=int)
    groups = np.asarray([f"{e['mas_name']}::{e['benchmark_name']}" for e in eps])
    pred = np.zeros(len(eps), dtype=float)
    fold_sizes = []
    fold_reports = []
    for fold, (tr, te) in enumerate(_folds(eps, y, groups)):
        result = evaluate_split([eps[i] for i in tr], [eps[i] for i in te], y[tr], y[te], seed + fold)
        pred[te] = np.asarray([x["risk_probability"] for x in result["predictions"]])
        fold_sizes.append({"fold": fold, "train": int(len(tr)), "test": int(len(te))})
        fold_reports.append({"fold": fold, "metrics": result["metrics"]["ours"], "optimizer": result["optimizer"]})
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "groups": int(len(np.unique(groups))),
        "include_public_text_in_semantic_subgraph": bool(include_text),
        "label_source": "released MAST mast_annotation; external transfer only",
        "feature_boundary": "trajectory text and message headers only; released annotations excluded from events",
        "metrics": _metric(y, pred),
        "folds": fold_sizes,
        "fold_reports": fold_reports,
        "predictions": [{"episode_id": e["episode_id"], "y": int(y[i]), "p": float(pred[i])} for i, e in enumerate(eps)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    structural = evaluate(rows, include_text=False)
    semantic = evaluate(rows, include_text=True)
    report = {
        "dataset": {"path": str(args.input), "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()},
        "structural_only": structural,
        "semantic_runtime_extension": semantic,
        "comparison": {
            "delta_auroc": None if structural["metrics"]["auroc"] is None else semantic["metrics"]["auroc"] - structural["metrics"]["auroc"],
            "delta_auprc": None if structural["metrics"]["auprc"] is None else semantic["metrics"]["auprc"] - structural["metrics"]["auprc"],
            "delta_brier": semantic["metrics"]["brier"] - structural["metrics"]["brier"],
        },
        "limitations": [
            "MAST/MAD released annotations are a secondary external target, not newly collected independent security gold labels.",
            "The MAST trajectory is text-centric; many rows have no explicit machine-readable tool arguments or data-flow provenance, so the v2 graph is a conservative projection.",
            "The semantic policy subgraph is fixed and inspectable; it is not an LLM judge and was not tuned on MAST labels.",
        ],
    }
    (args.out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "REPORT.md").write_text(
        "\n".join([
            "# MAST external transfer: TwoLayerBNGraphMonitor-v2",
            "",
            "This is a development-only external transfer audit. Labels are the released MAST annotations.",
            "",
            "| condition | AUROC | AUPRC | F1 | Brier |",
            "|---|---:|---:|---:|---:|",
            f"| structural-only (text disabled) | {structural['metrics']['auroc']:.3f} | {structural['metrics']['auprc']:.3f} | {structural['metrics']['f1']:.3f} | {structural['metrics']['brier']:.3f} |",
            f"| semantic runtime extension | {semantic['metrics']['auroc']:.3f} | {semantic['metrics']['auprc']:.3f} | {semantic['metrics']['f1']:.3f} | {semantic['metrics']['brier']:.3f} |",
            "",
            f"Delta (semantic - structural): AUROC {report['comparison']['delta_auroc']:.3f}; AUPRC {report['comparison']['delta_auprc']:.3f}; Brier {report['comparison']['delta_brier']:.3f}.",
            "",
            "The projection uses public trajectory text and message headers only; released annotations are joined after feature construction. This audit is not an independent human-gold superiority claim.",
            "",
        ]) + "\n", encoding="utf-8")
    print(json.dumps({"n": report["semantic_runtime_extension"]["n"], "structural": structural["metrics"], "semantic": semantic["metrics"], "comparison": report["comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
