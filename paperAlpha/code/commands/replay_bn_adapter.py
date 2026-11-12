"""Replay normalized MAS records through the pilot hierarchical BN.

This module deliberately separates *scoring* from *evaluation*.  A normalized
record may be a task template or an execution event and may not carry a
security label.  In that case the replay emits an exploratory score and
evidence only; it never invents a label and never calls ``evaluate``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_model_test_v1 import FEATURES, predict


RISK_TERMS = {
    "untrusted_input": ("external", "injected", "untrusted", "malicious", "poisoned"),
    "permission_mismatch": ("unauthorized", "forbidden", "permission", "access_denied", "wrong_recipient"),
    "memory_poison": ("memory", "stale", "persisted", "shared_state", "shared state"),
    "tool_side_effect": ("tool_", "publish", "payment", "delete", "send", "export", "submit", "execute"),
    "anomaly_path": ("retry", "failure", "failed", "error", "parse", "bypass", "unexpected", "unstarted"),
}


def _blob(value: Any) -> str:
    """Make a bounded, lowercase text view without changing source data."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:20000].lower()


def _event_text(event: dict[str, Any]) -> str:
    # Do not persist this full text in the output; it is only a feature signal.
    return _blob({
        "event_type": event.get("event_type"),
        "role": event.get("role", event.get("role_id")),
        "agent_id": event.get("agent_id"),
        "source_agent": event.get("source_agent"),
        "target_agent": event.get("target_agent"),
        "tool": event.get("tool"),
        "payload": event.get("payload"),
        "details": event.get("details"),
        "message": event.get("message"),
    })


def events_to_features(events: Iterable[dict[str, Any]]) -> tuple[dict[str, int], str, dict[str, Any]]:
    """Convert visible event fields to the pilot BN feature contract.

    This is intentionally conservative: cross-agent propagation requires an
    explicit source/target transition (or a delegation/handoff marker), while
    risk accumulation is a derived structural flag, not a security label.
    """
    event_list = [e for e in events if isinstance(e, dict)]
    texts = [_event_text(e) for e in event_list]
    joined = " ".join(texts)
    flags = {name: int(any(term in joined for term in terms)) for name, terms in RISK_TERMS.items()}

    explicit_handoff = any(any(token in text for token in ("delegate", "handoff", "propagate")) for text in texts)
    transitions = 0
    for event in event_list:
        source = event.get("source_agent")
        target = event.get("target_agent")
        if source and target and str(source) != str(target):
            transitions += 1
    flags["cross_agent_propagation"] = int(bool(explicit_handoff or transitions))
    active_local = sum(flags[name] for name in RISK_TERMS)
    flags["risk_accumulation"] = int(active_local >= 2 or len(event_list) >= 12 and active_local >= 1)
    # Keep the exact six-feature contract and avoid accidental extra columns.
    features = {name: int(bool(flags.get(name, 0))) for name in FEATURES}
    evidence = {
        "event_count": len(event_list),
        "source_target_transitions": transitions,
        "active_features": [name for name in FEATURES if features[name]],
        "feature_method": "deterministic visible-field keyword/transition mapping",
    }
    return features, " ".join(evidence["active_features"]), evidence


def _binary_label(record: dict[str, Any]) -> int | None:
    """Read an explicit binary security label, never infer one from failures."""
    for key in ("risk_label", "security_label", "label"):
        value = record.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int) and value in (0, 1):
            return value
        if isinstance(value, str) and value.strip().lower() in {"0", "1", "safe", "risk", "positive", "negative"}:
            return int(value.strip().lower() in {"1", "risk", "positive"})
    return None


def _events_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept either an event record, an ``events`` array, or a task DAG."""
    events = record.get("events")
    if isinstance(events, list):
        return [item for item in events if isinstance(item, dict)]
    dag = record.get("workflow_dag")
    if isinstance(dag, dict) and isinstance(dag.get("nodes"), list):
        # Normative task templates are structural scaffolding, not executions.
        return [
            {"event_type": "tool_template", "agent_id": node.get("role"), "tool": node.get("tool"),
             "source_agent": node.get("role"), "target_agent": node.get("role")}
            for node in dag["nodes"] if isinstance(node, dict)
        ]
    return [record]


def replay(records: Iterable[dict[str, Any]], method: str = "hierarchical_bn") -> dict[str, Any]:
    """Score records and return predictions plus a label-gated metric status."""
    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        # The pre_exp1 adapter already emits the exact BN contract.  Reuse it
        # verbatim when present; otherwise derive features from visible events.
        supplied = record.get("features")
        if isinstance(supplied, dict) and all(name in supplied for name in FEATURES):
            features = {name: int(bool(supplied.get(name))) for name in FEATURES}
            text = str(record.get("text") or "")
            evidence = {
                "event_count": record.get("events", 1),
                "active_features": [name for name in FEATURES if features[name]],
                "feature_method": "precomputed_adapter_features",
            }
        else:
            features, text, evidence = events_to_features(_events_from_record(record))
        row = {"id": str(record.get("episode_id", record.get("task_id", record.get("run_id", index)))),
               "features": features, "text": text}
        score, node, active = predict(row, method)
        predictions.append({
            "id": row["id"],
            "source": record.get("source"),
            "risk_label": _binary_label(record),
            "risk_label_source": "explicit_record_field" if _binary_label(record) is not None else None,
            "score": round(float(score), 8),
            "risk_node": node,
            "evidence": active,
            "features": features,
            "feature_evidence": evidence,
        })
    labels = [p["risk_label"] for p in predictions]
    has_complete_labels = bool(labels) and all(label in (0, 1) for label in labels)
    if has_complete_labels:
        metric_status = "labels_present_but_evaluation_delegated"
        metric_note = "Use run_model_test_v1.evaluate only when labels are explicit binary security outcomes."
    else:
        metric_status = "not_evaluated_no_complete_risk_labels"
        metric_note = "Scores are exploratory structural BN outputs; no security metric is computed."
    return {
        "method": method,
        "n_records": len(predictions),
        "metric_status": metric_status,
        "metric_note": metric_note,
        "predictions": predictions,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay normalized MAS records through hierarchical_bn")
    parser.add_argument("--input", required=True, type=Path, help="normalized JSONL adapter output")
    parser.add_argument("--output", required=True, type=Path, help="replay JSON output")
    args = parser.parse_args()
    result = replay(_read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("method", "n_records", "metric_status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
