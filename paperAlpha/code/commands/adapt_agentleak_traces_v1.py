"""Read-only AgentLeak -> MAS observable-event projection.

This adapter is intentionally a validation-data adapter, not a trainer.  It
projects the public channel messages in AgentLeak release traces into the
shared ``mas-adapter-v1`` event contract and writes labels to a separate file.
The following source fields are never copied into event features: ``results``,
per-message ``has_leak``/``leaked_fields``, ``attack_family``, private vault
values, LLM prompts, and model/vertical identifiers.  Labels are joined only
after feature extraction by a downstream evaluator.

No source files are modified and no network access is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'code/external/AgentLeak/benchmarks/ieee_repro/results/traces'
DEFAULT_OUTPUT = ROOT / 'experiments/privacy_flow/runs/submission__agentleak_external_projection_v1'
SCHEMA_VERSION = "mas-adapter-agentleak-v1"
FEATURES = (
    "untrusted_input",
    "permission_mismatch",
    "memory_poison",
    "tool_side_effect",
    "cross_agent_propagation",
    "anomaly_path",
    "risk_accumulation",
)

_TEXT_TERMS = {
    "untrusted_input": ("injected", "injection", "poisoned", "malicious", "untrusted", "attack"),
    "permission_mismatch": ("unauthorized", "forbidden", "permission", "privacy", "confidential", "ssn", "diagnosis"),
    "memory_poison": ("memory", "cache", "persist", "stored", "shared state"),
    "tool_side_effect": ("send", "delete", "payment", "execute", "publish", "export", "write"),
    "anomaly_path": ("retry", "loop", "unexpected", "bypass", "override", "misunderstanding"),
}
def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _safe_text(value: Any, limit: int = 12000) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else _json(value)
    return text[:limit]


def _contains(text: str, terms: tuple[str, ...]) -> int:
    lower = text.lower()
    return int(any(term in lower for term in terms))


def _files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []
    return sorted(p for p in input_path.glob("*.json") if p.is_file())


def _feature_row(message: dict[str, Any], *, sequence: int, trace_messages: list[dict[str, Any]]) -> dict[str, int]:
    text = _safe_text(message.get("content"))
    channel = str(message.get("channel") or "")
    source = str(message.get("source") or "")
    target = str(message.get("target") or "")
    features = {name: _contains(text, terms) for name, terms in _TEXT_TERMS.items()}
    # These are structural observables, not labels: an inter-agent hop and a
    # memory write are visible in the channel envelope itself.
    features["cross_agent_propagation"] = int(channel == "C2" and source != target and bool(source) and bool(target))
    features["memory_poison"] = int(features["memory_poison"] or channel == "C5")
    prior_cross_agent = sum(
        int(str(row.get("channel") or "") == "C2" and row.get("source") != row.get("target"))
        for row in trace_messages[:sequence]
    )
    features["risk_accumulation"] = int(prior_cross_agent >= 1 or sequence >= 3)
    return {name: int(bool(features.get(name, 0))) for name in FEATURES}


def _event(trace: dict[str, Any], message: dict[str, Any], sequence: int, path: Path) -> dict[str, Any]:
    trace_id = str(trace.get("trace_id"))
    source = message.get("source")
    target = message.get("target")
    event_type = {
        "C1": "final_output",
        "C2": "inter_agent_message",
        "C5": "memory_write",
    }.get(str(message.get("channel") or ""), "observable_message")
    event_id = f"{trace_id}:m{sequence:02d}"
    text = _safe_text(message.get("content"))
    features = _feature_row(message, sequence=sequence, trace_messages=trace.get("channel_messages") or [])
    source_record = {
        "path": path.name,
        "message_index": sequence,
        "artifact_type": "agentleak_channel_message",
        "authenticity": "observed_public_channel_message",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "run_id": trace_id,
        "episode_id": str(trace.get("scenario_id")) if trace.get("scenario_id") is not None else None,
        "record_kind": "runtime_trace",
        "event_type": event_type,
        "stage": None,
        "agent_id": str(source) if source is not None else None,
        "role": str(source) if source is not None else None,
        "message": text,
        "message_text": text,
        "message_id": str(message.get("llm_call_id")) if message.get("llm_call_id") else None,
        "message_source": str(source) if source is not None else None,
        "message_target": str(target) if target is not None else None,
        "tool": None,
        "tool_arguments": None,
        "tool_result": None,
        # Channel messages have no event-level timestamp in the release
        # schema; do not manufacture one from trace order.
        "timestamp": None,
        "parent": [],
        "chain_parent": None,
        "permission": {
            "policy_present": isinstance((trace.get("input") or {}).get("allowed_set"), dict),
            "allowed_field_count": len(((trace.get("input") or {}).get("allowed_set") or {}).get("fields") or []),
            "forbidden_field_count": len(((trace.get("input") or {}).get("allowed_set") or {}).get("forbidden_fields") or []),
        },
        "external_effect": None,
        "external_effect_basis": None,
        "external_effect_scope": None,
        "risk_label": None,
        "risk_label_source": None,
        "outcome_flags": {},
        "features": features,
        "feature_reasons": {name: "channel_envelope_or_public_message_text" for name, value in features.items() if value},
        "source": source_record,
        "context": {"sequence_num": sequence, "channel": message.get("channel")},
        "eligibility": {
            "structural_graph": bool(source or target),
            "temporal_graph": False,
            "bn_feature_row": True,
            "risk_evaluation": False,
            "replay_evidence": True,
        },
        "source_record_hash": _sha({"trace_id": trace_id, "sequence": sequence, "channel": message}),
    }


def _label_row(trace: dict[str, Any], path: Path) -> dict[str, Any]:
    results = trace.get("results") if isinstance(trace.get("results"), dict) else {}
    return {
        "episode_id": str(trace.get("scenario_id")) if trace.get("scenario_id") is not None else None,
        "run_id": str(trace.get("trace_id")) if trace.get("trace_id") is not None else None,
        "label": int(bool(results.get("multi_leaked"))),
        "channel_labels": {
            "C1": int(bool(results.get("c1_leaked"))),
            "C2": int(bool(results.get("c2_leaked"))),
            "C5": int(bool(results.get("c5_leaked"))),
        },
        "label_source": "AgentLeak_release_results.multi_leaked; joined after observable-feature extraction",
        "source": {"path": path.name, "artifact_type": "reference_label"},
    }


def adapt(input_path: Path, output_dir: Path, *, limit_files: int | None = None) -> dict[str, Any]:
    files = _files(input_path)
    if limit_files is not None:
        files = files[: max(0, limit_files)]
    output_dir = output_dir.resolve()
    if output_dir == input_path.resolve() or output_dir.is_relative_to(input_path.resolve()):
        raise ValueError("output must be outside the read-only AgentLeak input root")
    events: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    invalid = 0
    for path in files:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(trace, dict):
                raise ValueError("trace object required")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            invalid += 1
            continue
        messages = trace.get("channel_messages")
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            if isinstance(message, dict):
                events.append(_event(trace, message, index, path))
        labels.append(_label_row(trace, path))
    # A trace-level observable view is convenient for the existing episode
    # evaluators.  It is built only from the event stream above; no reference
    # result or per-message leak flag participates in this aggregation.
    episode_rows: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = str(event["run_id"])
        row = episode_rows.setdefault(run_id, {
            "episode_id": event.get("episode_id"),
            "run_id": run_id,
            "features": {name: 0 for name in FEATURES},
            "text": [],
            "event_count": 0,
            "channels": [],
            "agents": [],
        })
        row["event_count"] += 1
        channel = event.get("context", {}).get("channel")
        if channel not in row["channels"]:
            row["channels"].append(channel)
        for agent in (event.get("message_source"), event.get("message_target")):
            if agent and agent not in row["agents"]:
                row["agents"].append(agent)
        row["text"].append(event.get("message_text") or "")
        for name in FEATURES:
            row["features"][name] = max(row["features"][name], int(event["features"].get(name, 0)))
    episode_rows_list = []
    for row in episode_rows.values():
        row["text"] = " ".join(row["text"])
        row["channel_count"] = len(row["channels"])
        row["agent_count"] = len(row["agents"])
        # The source topology is fixed in AgentLeak; retain the counts as
        # observables, but do not invent a richer DAG than the release shows.
        episode_rows_list.append(row)
    episode_rows_list.sort(key=lambda row: str(row["run_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "events.jsonl").write_text("".join(_json(row) + "\n" for row in events), encoding="utf-8")
    (output_dir / "labels_reference.jsonl").write_text("".join(_json(row) + "\n" for row in labels), encoding="utf-8")
    (output_dir / "episodes_observable.jsonl").write_text("".join(_json(row) + "\n" for row in episode_rows_list), encoding="utf-8")
    (output_dir / "schema.json").write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "mas-adapter-agentleak-v1.schema.json",
        "title": "AgentLeak observable MAS event projection",
        "type": "object",
        "required": ["schema_version", "event_id", "run_id", "event_type", "features", "source"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "event_id": {"type": "string"},
            "run_id": {"type": "string"},
            "event_type": {"type": "string"},
            "features": {"type": "object", "required": list(FEATURES)},
            "risk_label": {"type": ["null", "integer"]},
            "source": {"type": "object"},
        },
        "additionalProperties": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_channel = Counter(str(e.get("context", {}).get("channel")) for e in events)
    positives = sum(int(row["label"]) for row in labels)
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(input_path.resolve()),
        "files_read": len(files),
        "invalid_files": invalid,
        "trace_count": len(labels),
        "event_count": len(events),
        "episode_view_count": len(episode_rows_list),
        "positive_reference_labels": positives,
        "event_types": dict(Counter(str(e.get("event_type")) for e in events)),
        "channel_counts": dict(by_channel),
        "feature_fields": list(FEATURES),
        "label_join": "labels_reference.jsonl is separate and must be joined only after feature extraction",
        "excluded_from_features": ["results", "channel_messages.has_leak", "channel_messages.leaked_fields", "attack_family", "input.vault", "llm_calls.system_prompt", "llm_calls.user_prompt", "model", "vertical"],
        "source_files": [p.name for p in files],
        "status": "ready_for_external_validation_projection" if labels else "no_valid_traces",
    }
    (output_dir / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "ADAPTER_REPORT.md").write_text(
        "\n".join([
            "# AgentLeak external MAS projection v1", "",
            "This is a read-only projection of the AgentLeak release traces.", "",
            f"- Traces: **{len(labels)}**; observable channel events: **{len(events)}**.",
            f"- Reference MAS leakage positives (`multi_leaked`): **{positives}**.",
            f"- Invalid input files: **{invalid}**.", "",
            "## Leakage-safe boundary", "",
            "The event stream and `episodes_observable.jsonl` contain only public channel envelope fields and message text. They do not contain AgentLeak `results`, per-message leak flags, attack family, private vault values, prompts, or model/domain identifiers. `labels_reference.jsonl` is a reference-only join file and must not enter feature construction.", "",
            "## Limitations", "",
            "The release benchmark uses a fixed two-agent coordinator/worker topology and channel-level privacy-leakage labels. It is an external MAS validation cohort, not a direct test of all operational safety hazards or of arbitrary topologies.", "",
        ]) + "\n", encoding="utf-8")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()
    result = adapt(args.input.resolve(), args.output.resolve(), limit_files=args.limit_files or None)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
