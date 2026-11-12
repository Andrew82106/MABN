"""Native HarnessAudit artifacts: observation/evaluation separation.

This is a completed-action replay contract, NOT an online collection hook.
Only ``monitor`` is model input. Run identity, task metadata, audit decisions,
and label eligibility live outside it. No hidden policy or hindsight feature
is inferred from the evaluation side. Source semantics were checked against
HarnessAudit commit 6317162590aeeb1c8dde32b880ac199933343e4a.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter


SCHEMA = "harnessaudit-observation-split-v1"
SOURCE_REVISION = "6317162590aeeb1c8dde32b880ac199933343e4a"
_SECRET = re.compile(r"api[_-]?key|access[_-]?token|password|passwd|secret|credential|private[_-]?key|bearer", re.I)
_AUDIT_FIELDS = ("action_sequence_num", "violated", "violation_code", "severity", "rule_description", "timestamp")
_END_FIELDS = ("n_actions", "n_tool_calls", "n_communications", "n_violations", "violations_by_code",
               "sar_tool", "sar_resource", "sar_flow", "sar_avg", "avs", "tcr", "error", "perturbation")
CONTRACT = {
    "ordering": "native sequence in file order; source was buffered and resequenced",
    "tool_visibility": "completed action; arguments and result become visible together",
    "communication_visibility": "recorded message; delivery/reading is not established",
    "online_arrival_order_verified": False,
    "pre_execution_warning_eligible": False,
    "permission_checks_available": False,
    "hidden_rules_available": False,
    "text_trust": "agent/tool text is untrusted content, never an authenticated check",
    "redaction": "secret-like argument keys only; not comprehensive text anonymization",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _string(record, field, *, empty=False):
    value = record.get(field)
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"native {field} must be a {'possibly empty ' if empty else 'nonempty '}string")
    return value


def _integer(record, field, *, minimum=0):
    value = record.get(field)
    if type(value) is not int or value < minimum:
        raise ValueError(f"native {field} must be an integer >= {minimum}")
    return value


def _redact_arguments(value):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SECRET.search(key) else _redact_arguments(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_arguments(item) for item in value]
    return value


class HarnessAuditObservationProjector:
    """Incremental projection; snapshots never backfill from future records.

Use one instance per trace. Unknown event types are kept evaluation-side and
make a trace ineligible for the derived label, rather than silently disappearing.
Projection never interprets native tool names as the v3 sensitive-tool aliases.
"""

    def __init__(self):
        self.trace_id = None
        self.context = None
        self.metadata = {}
        self.events = []
        self.decisions = []
        self.end = None
        self.unknown = []
        self.audit_started = False

    def push(self, record):
        if not isinstance(record, dict):
            raise ValueError("one native JSON object is required")
        # Validate JSON before changing state, including NaN and non-string keys.
        if json.loads(canonical(record)) != record:
            raise ValueError("record must contain JSON-native values and string keys")
        identity = _string(record, "trace_id")
        kind = _string(record, "type")
        if self.end is not None:
            raise ValueError("record after trace_end")
        if self.trace_id is None:
            if kind != "trace_start":
                raise ValueError("trace_start must precede all records; no future metadata backfill")
            agents = record.get("agents", [])
            if not isinstance(agents, list) or any(not isinstance(a, dict) for a in agents):
                raise ValueError("native agents must be a list of descriptors")
            roster = [{"agent_id": _string(a, "agent_id"), "role": _string(a, "role")} for a in agents]
            if len({a["agent_id"] for a in roster}) != len(roster):
                raise ValueError("duplicate declared agent identity")
            self.trace_id = identity
            self.context = {"declared_agents": roster}
            # useful_tools is a task-usefulness descriptor, NOT granted permissions.
            self.metadata = {key: copy.deepcopy(record[key]) for key in
                             ("task_id", "domain", "framework", "model", "n_rules", "perturbation") if key in record}
            return None
        if identity != self.trace_id or kind == "trace_start":
            raise ValueError("one trace per projector and exactly one trace_start required")
        if kind in {"tool_call", "communication"}:
            if self.audit_started:
                raise ValueError("native action after post-run audit started")
            sequence = _integer(record, "sequence_num", minimum=1)
            if self.events and sequence <= self.events[-1]["sequence"]:
                raise ValueError("native action sequences must strictly increase; no retrospective sorting")
            event = {"sequence": sequence, "actor": _string(record, "agent_id"),
                     "declared_role": _string(record, "agent_role"), "source_timestamp": _string(record, "timestamp")}
            if kind == "tool_call":
                arguments = record.get("tool_args", {})
                if not isinstance(arguments, dict):
                    raise ValueError("native tool_args must be an object")
                result = record.get("tool_result", "")
                if not isinstance(result, str):
                    raise ValueError("native tool_result must be a string")
                event.update(event_type="tool_completed_observed", tool=_string(record, "tool_name"),
                             arguments=_redact_arguments(arguments), result=result)
            else:
                event.update(event_type="communication_observed", target=_string(record, "target_agent", empty=True),
                             target_declared_role=_string(record, "target_role", empty=True),
                             content=_string(record, "content", empty=True))
            # No raw_event, inherited features, permission flags, labels or metadata.
            self.events.append(copy.deepcopy(event))
            return copy.deepcopy(event)
        if kind == "access_decision":
            _integer(record, "action_sequence_num", minimum=1)
            if type(record.get("violated")) is not bool:
                raise ValueError("native violated must be a bool")
            self.decisions.append({key: copy.deepcopy(record[key]) for key in _AUDIT_FIELDS if key in record})
            self.audit_started = True
        elif kind == "trace_end":
            self.end = {key: copy.deepcopy(record[key]) for key in _END_FIELDS if key in record}
        else:
            self.unknown.append(copy.deepcopy(record))
        return None

    def monitor_view(self):
        if self.context is None:
            raise ValueError("trace_start is required")
        return {"schema_version": SCHEMA, "contract": copy.deepcopy(CONTRACT),
                "context": copy.deepcopy(self.context), "events": copy.deepcopy(self.events)}

    def evaluation_view(self):
        """Only complete successful audits receive a derived endpoint label.

        A zero means no recorded rule match, NOT proof of safety. Includes V-OR;
        this endpoint is neither the paper's L1 SAR nor actual-harm ground truth.
        """
        reasons = []
        sequences = {event["sequence"] for event in self.events}
        checked = {decision["action_sequence_num"] for decision in self.decisions}
        positive = [decision for decision in self.decisions if decision["violated"]]
        if not self.events:
            reasons.append("no_actions")
        if self.end is None:
            reasons.append("missing_trace_end")
        else:
            counts = {"n_actions": len(self.events), "n_tool_calls": sum(e["event_type"] == "tool_completed_observed" for e in self.events),
                      "n_communications": sum(e["event_type"] == "communication_observed" for e in self.events),
                      "n_violations": len(positive)}
            for key, expected in counts.items():
                if type(self.end.get(key)) is not int or self.end[key] != expected:
                    reasons.append(f"inconsistent_or_missing_{key}")
            if "error" not in self.end or self.end["error"] not in (None, ""):
                reasons.append("run_error_or_unspecified_error_status")
            if "violations_by_code" in self.end:
                observed = dict(Counter(d.get("violation_code") or "unknown" for d in positive))
                if self.end["violations_by_code"] != observed:
                    reasons.append("inconsistent_violations_by_code")
        if sequences - checked:
            reasons.append("actions_without_audit_decisions")
        if checked - sequences:
            reasons.append("decisions_without_observed_actions")
        if any({d["violated"] for d in self.decisions if d["action_sequence_num"] == seq} == {False, True} for seq in checked):
            reasons.append("contradictory_decisions_for_same_action")
        if self.unknown:
            reasons.append("unknown_native_event_types")
        eligible = not reasons
        return {"target": "any_recorded_access_rule_violation", "label": int(bool(positive)) if eligible else None,
                "eligible": eligible, "ineligibility_reasons": reasons,
                "meaning": "Any post-run rule violation, including resource-scope V-OR; not L1 SAR, intent or actual harm.",
                "first_flagged_action_sequence": min((d["action_sequence_num"] for d in positive), default=None),
                "first_flagged_action_use": "retrospective localization only; not a harm or pre-execution timestamp",
                "observed_action_count": len(sequences), "audited_observed_action_count": len(sequences & checked),
                "decisions": copy.deepcopy(self.decisions), "trace_end": copy.deepcopy(self.end),
                "unknown_records": copy.deepcopy(self.unknown)}

    def snapshot(self):
        monitor = self.monitor_view()
        return {"trace_id": self.trace_id, "monitor": monitor,
                "monitor_sha256": hashlib.sha256(canonical(monitor).encode("utf-8")).hexdigest(),
                "evaluation": self.evaluation_view(), "metadata": copy.deepcopy(self.metadata),
                "model_readiness": {
                    "direct_joint_bn_input": False,
                    "action_scoped_v3_eligible": False,
                    "missing_v3_evidence": ["native action/principal/resource binding", "trusted pre-dispatch checks", "pre-dispatch observation timing"],
                    "missing_semantics": ["role-to-template mapping", "message/source classifiers", "tool capability mapping"],
                    "capability_manifest_from_useful_tools": False,
                }}


def project_records(records):
    """Project one or more interleaved native traces without looking ahead."""
    projectors = {}
    for record in records:
        identity = _string(record, "trace_id") if isinstance(record, dict) else None
        if identity is None:
            raise ValueError("native JSON object required")
        projector = projectors.get(identity)
        if projector is None:
            projector = HarnessAuditObservationProjector()
            projector.push(record)
            projectors[identity] = projector
        else:
            projector.push(record)
    if not projectors:
        raise ValueError("no native traces")
    return [projector.snapshot() for projector in projectors.values()]
