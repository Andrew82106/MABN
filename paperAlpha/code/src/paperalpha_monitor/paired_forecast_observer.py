"""Condition-blind observer boundary for paired, prompt-intervened acquisition.

The raw stage/SDK claim stays PRIVATE. Verify its exact request, feedback and
sequence boundary with the original projector *before* exporting a public view.
Only configured system prompts are removed; actual reports, arguments and tool
results remain evidence even when they quote an attack or the word 'condition'.

This is a deterministic request-time reconstruction, not a claim that a monitor
ran online. No response, target, private Bank state or condition map is an input.
Absolute and monotonic clocks stay in the private acquisition only. Latency
evaluation is separate, and neither clocks nor association IDs are predictors.
"""
from __future__ import annotations

import copy
import math
from uuid import UUID

from .contract_forecast_projection import FEATURE_NAMES, HORIZON, project_forecast_input
from paperalpha_runtime.journal import RuntimeEvidenceProjector


_PREFIX_SCHEMA = "harnessaudit-sandbox-monitor-prefix-v1"
_DECLARATIONS = ("workflow", "stage_purposes", "output_delivery", "role_bindings", "tools", "normative")
_OBSERVABILITY = ("regime", "state_visibility", "timing", "workflow")
_ROW_KEYS = {"schema_version", "sequence", "kind", "payload"}
_CLOCK_KEYS = {"observed_at", "monotonic_ns"}
_BINDING = {"invocation_id", "actor", "bank_role", "role_binding_source", "tool", "provider_call_id"}
_PAYLOAD_KEYS = {
    "run_started": {"agents", "external_endpoints", "domain", "role_bindings", "source"},
    "communication": {"actor", "target", "content", "actor_role", "target_role",
                      "actor_role_source", "target_role_source", "source"},
    "tool_handler_started": _BINDING | {"arguments"},
    "tool_native_result": _BINDING | {"arguments", "result", "backend", "semantic_success"},
    "tool_handler_returned": _BINDING | {"result", "semantic_success"},
    "tool_handler_failed": _BINDING | {"error_type"},
    "run_finished": {"status", "error_type"},
}


class PairedObserverError(ValueError):
    """No permitted observer view can be certified; do not substitute safe."""


def _need(condition, message):
    if not condition:
        raise PairedObserverError(message)


def _configuration(value):
    """Strip prompt fields ONLY in static declarations, never runtime payloads."""
    if isinstance(value, dict):
        return {key: _configuration(child) for key, child in value.items() if key != "system_prompt"}
    if isinstance(value, list):
        return [_configuration(child) for child in value]
    return copy.deepcopy(value)


def _task(prefix):
    task = prefix.get("public_task")
    _need(isinstance(task, dict) and isinstance(task.get("goal"), str), "legal task goal required")
    agents = task.get("agents")
    _need(isinstance(agents, list) and bool(agents), "legal task role/description inventory required")
    result = []
    for agent in agents:
        _need(isinstance(agent, dict) and isinstance(agent.get("role"), str) and bool(agent["role"])
              and isinstance(agent.get("description"), str), "legal task role/description required")
        result.append({"role": agent["role"], "description": agent["description"]})
    _need(len({a["role"] for a in result}) == len(result), "duplicate legal task role")
    return {"goal": task["goal"], "agents": result}


def public_prefix(raw_prefix, legitimate_initial_prefix):
    """Export a sequence-preserving view using an audited, unmodified setup.

    The caller must obtain ``legitimate_initial_prefix`` from the original
    task, not a condition-specific prompt or result. It must be an empty,
    trace-only setup. This helper cannot itself prove the template's provenance.
    Configured goal/roles/descriptions, graph, tools and policy must agree;
    prompts are omitted uniformly. Unknown top-level acquisition fields are
    discarded; unexpected journal envelope/payload fields are rejected. Tool
    argument/result objects are actual observations and are never key-redacted.
    Derived events/coverage are rebuilt, not accepted from the supplied prefix.
    """
    raw, legal = raw_prefix, legitimate_initial_prefix
    _need(isinstance(raw, dict) and isinstance(legal, dict), "prefix mappings required")
    _need(raw.get("schema") == legal.get("schema") == _PREFIX_SCHEMA, "supported monitor prefix required")
    _need(type(legal.get("sequence")) is int and legal["sequence"] == 0 and legal.get("journal") == [],
          "legitimate template must be an empty initial prefix")
    for prefix in (raw, legal):
        obs = prefix.get("observability")
        _need(isinstance(obs, dict) and obs.get("regime") == "trace_only"
              and not obs.get("state_visibility"), "trace-only setup without private state required")
    task = _task(legal)
    _need(_task(raw) == task, "goal or role/description changed from legitimate setup")
    declarations = {}
    for key in _DECLARATIONS:
        _need(key in raw and key in legal, "missing declared configuration: " + key)
        declarations[key] = _configuration(legal[key])
        _need(_configuration(raw[key]) == declarations[key], "declared configuration changed: " + key)
    _need(isinstance(declarations["workflow"], dict) and isinstance(declarations["workflow"].get("nodes"), list)
          and isinstance(declarations["role_bindings"], dict), "workflow and role bindings required")
    for key in _OBSERVABILITY:
        _need(raw["observability"].get(key) == legal["observability"].get(key),
              "observation regime changed: " + key)
    sequence, rows = raw.get("sequence"), raw.get("journal")
    _need(type(sequence) is int and sequence >= 0 and isinstance(rows, list) and len(rows) == sequence,
          "complete ordered journal prefix required")
    projector, public_rows, events = RuntimeEvidenceProjector(), [], []
    for row in rows:
        _need(isinstance(row, dict) and _ROW_KEYS <= set(row) <= _ROW_KEYS | _CLOCK_KEYS,
              "unsupported or private journal envelope field")
        payload, kind = row["payload"], row["kind"]
        _need(isinstance(kind, str) and kind in _PAYLOAD_KEYS and isinstance(payload, dict)
              and set(payload) <= _PAYLOAD_KEYS[kind], "unsupported or private journal payload field")
        if kind == "run_started":
            _need(payload.get("agents") == declarations["workflow"]["nodes"]
                  and payload.get("role_bindings") == declarations["role_bindings"],
                  "run participants differ from legitimate setup")
        public_row = {key: copy.deepcopy(row[key]) for key in ("schema_version", "sequence", "kind", "payload")}
        event = projector.push(public_row)
        public_rows.append(public_row)
        if event is not None:
            events.append(event)
    observability = {key: copy.deepcopy(legal["observability"][key])
                     for key in _OBSERVABILITY if key in legal["observability"]}
    observability.update(
        agent_configuration="All configured agent system prompts are omitted; actual observed outputs are unchanged.",
        clocks="Absolute and monotonic clocks omitted; recorded sequence is unchanged. Latency is evaluated privately.",
        projection="Deterministic recorded-prefix reconstruction; not evidence of an online detector run.",
    )
    return {"schema": _PREFIX_SCHEMA, "sequence": sequence, "public_task": task, **declarations,
            "observability": observability, "journal": public_rows, "events": events, "coverage": projector.report()}


def project_paired_forecast_input(stage_record, request_claim, journal, contract,
                                  legitimate_initial_prefix, *, episode_id=None):
    """Privately verify the raw request, then export condition-blind features.

    Do not sanitize or rehash raw inputs before this call. The original
    projector validates exact instructions, packet, request SHA and tool feedback
    to fix the request's horizon. Only afterwards is its view sanitized. This
    wrapper preserves the same 21 features and current-A evidence, but never
    returns the raw request SHA or caller-supplied metadata. Episode IDs, when
    present, must be canonical UUID4 association keys from the private runner;
    the caller is responsible for random generation, not condition encoding.
    """
    # In particular, sanitization cannot hide a corrupt raw claim/feedback link.
    verified = project_forecast_input(stage_record, request_claim, journal, contract)
    view = public_prefix(verified["visible_prefix"], legitimate_initial_prefix)
    _need(verified["covered"] is True and verified["horizon"] == HORIZON
          and verified["prefix_horizon"] == view["sequence"], "verified request horizon mismatch")
    _need(tuple(verified["feature_names"]) == FEATURE_NAMES and tuple(verified["features"]) == FEATURE_NAMES
          and all(type(v) is float and math.isfinite(v) for v in verified["features"].values()),
          "verified numeric feature schema mismatch")
    meta = {key: copy.deepcopy(verified["metadata"][key])
            for key in ("contract_id", "task_id", "stage_id", "claim_number")}
    if episode_id is not None:
        try:
            identity = UUID(episode_id) if isinstance(episode_id, str) else None
        except ValueError:
            identity = None
        _need(identity is not None and identity.version == 4 and str(identity) == episode_id,
              "canonical opaque UUID4 episode id required")
        meta["episode_id"] = episode_id
    return {"schema": "paired-contract-forecast-observer-v1", "covered": True, "horizon": HORIZON,
            "prefix_horizon": view["sequence"], "feature_names": list(FEATURE_NAMES),
            "features": copy.deepcopy(verified["features"]), "current_a": verified["current_a"],
            "current_state": verified["current_state"], "metadata": meta,
            "evidence_path": copy.deepcopy(verified["evidence_path"]), "visible_prefix": view,
            "boundaries": copy.deepcopy(verified["boundaries"]) + [
                "Raw private request verified before observer sanitization; no condition, private prompt, hash or clock metadata exported.",
                "Association metadata and evidence identities are for grouping/audit only, never numeric predictors.",
                "Future target extraction is a separate private operation; this view never consumes a response or oracle."]}
