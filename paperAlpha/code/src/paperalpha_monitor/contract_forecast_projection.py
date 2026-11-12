"""Pure API-request-time projection for the two declared procedural contracts.

Horizon: the NEXT API reply's FIRST actual tool invocation. A reply with no
tools is STOP only when independently verified as an explicit final reply.
This module never receives that reply, outcome labels or private Bank state.
It is not an any-tool-in-batch or whole-episode forecast.

The SDK claim includes full in-node history. function_call_output.call_id binds
only already-returned results to journal provider_call_id. We never infer time
from file numbers, timestamps, response text or the eventual episode result.
Protocol: https://developers.openai.com/api/docs/guides/function-calling
"""
from __future__ import annotations

import copy
from fnmatch import fnmatch
import hashlib
import json

from .harnessaudit_relational_evidence import interpret_relational_evidence
from paperalpha_runtime.journal import RuntimeEvidenceProjector


FEATURE_NAMES = (
    "bias", "required_update_role_known", "required_update_role_match",
    "trigger_tool_scope_known", "trigger_tool_scope_allowed", "trigger_resource_restricted",
    "trigger_tool_available", "state_unknown", "past_related_update_count",
    "past_related_update_cross_role_count", "past_trigger_count", "past_trigger_rejection_count",
    "evidence_role_count", "evidence_cross_role_count", "request_feedback_count",
    "predecessor_count", "predecessor_role_count", "workflow_depth_fraction",
    "workflow_indegree", "workflow_outdegree", "workflow_sink",
)
HORIZON = "next_api_reply_first_actual_tool_call_or_explicit_final_stop"
PRIVATE_KEYS = {"ORACLE", "oracle", "private_labels", "evaluation_config", "initial_state",
                "final_state", "state_before", "state_after", "tool_transitions",
                "observable_state_before", "observable_state_after"}


class ForecastProjectionUncovered(ValueError):
    """The request's visible boundary cannot be certified; never impute safe."""

    def __init__(self, code):
        self.code = code
        super().__init__("forecast projection uncovered: " + code)


def _need(condition, code):
    if not condition:
        raise ForecastProjectionUncovered(code)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _public(value):
    if isinstance(value, dict):
        _need(not set(value).intersection(PRIVATE_KEYS), "private_state_in_public_input")
        for child in value.values():
            _public(child)
    elif isinstance(value, list):
        for child in value:
            _public(child)


def _workflow(prefix, stage_id, packet):
    workflow, bindings = prefix.get("workflow", {}), prefix.get("role_bindings", {})
    nodes, edges = workflow.get("nodes"), workflow.get("edges")
    _need(isinstance(nodes, list) and nodes and len(nodes) == len(set(nodes)) and stage_id in nodes,
          "explicit_workflow_nodes_required")
    _need(isinstance(edges, list) and set(bindings) == set(nodes), "workflow_bindings_incomplete")
    incoming, outgoing = {n: [] for n in nodes}, {n: [] for n in nodes}
    seen = set()
    for edge in edges:
        _need(isinstance(edge, (list, tuple)) and len(edge) == 2, "invalid_workflow_edge")
        u, v = edge
        _need(u in incoming and v in incoming and u != v and (u, v) not in seen, "invalid_workflow_edge")
        seen.add((u, v)); incoming[v].append(u); outgoing[u].append(v)
    depth, remaining = {}, set(nodes)
    while remaining:
        ready = [n for n in nodes if n in remaining and all(p in depth for p in incoming[n])]
        _need(bool(ready), "workflow_cycle")
        for n in ready:
            depth[n] = max((depth[p] + 1 for p in incoming[n]), default=0)
            remaining.remove(n)
    role = bindings[stage_id]
    _need(packet.get("bank_role") == role.get("role") and
          packet.get("role_binding_source") == role.get("source"), "packet_role_binding_conflict")
    reports = packet.get("input", {}).get("predecessor_reports")
    _need(isinstance(reports, list) and len(reports) == len(incoming[stage_id]), "predecessor_inventory_mismatch")
    senders = set()
    for report in reports:
        _need(isinstance(report, dict), "invalid_predecessor_report")
        sender = report.get("sender")
        _need(sender in incoming[stage_id] and sender not in senders, "predecessor_inventory_mismatch")
        senders.add(sender)
        _need(report.get("sender_role") == bindings[sender].get("role") and
              report.get("sender_role_source") == bindings[sender].get("source"), "predecessor_role_conflict")
        _need(any(r["kind"] == "communication" and r["payload"].get("actor") == sender and
                  r["payload"].get("target") == stage_id and r["payload"].get("content") == report.get("content")
                  for r in prefix["journal"]), "predecessor_report_not_yet_visible")
    return {"predecessor_count": len(reports), "predecessor_role_count": len({r["sender_role"] for r in reports}),
            "workflow_depth_fraction": depth[stage_id] / max(1, max(depth.values())),
            "workflow_indegree": len(incoming[stage_id]), "workflow_outdegree": len(outgoing[stage_id]),
            "workflow_sink": int(not outgoing[stage_id])}


def _history(request, packet):
    _need(request.get("previous_response_id") is None and request.get("conversation_id") is None,
          "remote_context_not_observable")
    _need(request.get("system_instructions") == packet.get("instructions"), "request_instructions_differ_from_stage")
    items = request.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    _need(isinstance(items, list) and items, "explicit_sdk_input_required")
    first = items[0]
    _need(isinstance(first, dict) and first.get("role") == "user" and
          first.get("content") == _canonical(packet.get("input")), "request_initial_context_differs_from_stage")
    calls, outputs = {}, {}
    for item in items[1:]:
        _need(isinstance(item, dict), "invalid_sdk_input_item")
        kind, identity = item.get("type"), item.get("call_id")
        if kind == "function_call":
            _need(isinstance(identity, str) and identity and identity not in calls,
                  "duplicate_or_missing_input_call_identity")
            try:
                args = json.loads(item.get("arguments", ""))
            except (TypeError, ValueError):
                raise ForecastProjectionUncovered("unparsed_prior_call_arguments") from None
            _need(isinstance(args, dict) and isinstance(item.get("name"), str), "unparsed_prior_call_arguments")
            calls[identity] = {"name": item["name"], "arguments": args}
        elif kind == "function_call_output":
            _need(identity in calls and identity not in outputs and isinstance(item.get("output"), str),
                  "unpaired_or_duplicate_function_output")
            outputs[identity] = item["output"]
        else:
            _need(kind == "reasoning" or item.get("role") == "assistant", "unsupported_in_node_context_item")
    _need(set(calls) == set(outputs), "prior_tool_calls_without_returned_outputs")
    return calls, outputs


def _request_prefix(stage, claim, journal):
    _need(isinstance(stage, dict) and isinstance(claim, dict) and isinstance(journal, list), "structured_inputs_required")
    _need(claim.get("schema") == "sdk-model-call-claim", "sdk_request_claim_required")
    _need(type(claim.get("call")) is int and claim["call"] > 0, "invalid_claim_number")
    request, base, packet = claim.get("request"), stage.get("prefix_before"), stage.get("packet")
    _need(isinstance(request, dict) and isinstance(base, dict) and isinstance(packet, dict), "stage_request_context_required")
    _need(claim.get("request_sha256") == _digest(request), "request_claim_hash_mismatch")
    _need(stage.get("prefix_sha256") == _digest(base), "stage_prefix_hash_mismatch")
    stage_id, start = stage.get("stage_id"), base.get("sequence")
    _need(isinstance(stage_id, str) and stage_id == packet.get("node_id") == claim.get("node_id"), "claim_stage_mismatch")
    _need(type(start) is int and start >= 1 and stage.get("sequence") == start, "stage_start_boundary_missing")
    _public(base); _public(packet)
    _need(base.get("observability", {}).get("regime") == "trace_only", "api_visible_observations_only")
    _need(isinstance(base.get("journal"), list) and base["journal"] == journal[:start], "stage_not_a_journal_prefix")
    calls, outputs = _history(request, packet)
    links, horizon = [], start
    for provider_id, text in outputs.items():
        linked = [r for r in journal if r.get("payload", {}).get("provider_call_id") == provider_id]
        entries = [r for r in linked if r.get("kind") == "tool_handler_started"]
        terminals = [r for r in linked if r.get("kind") in {"tool_handler_returned", "tool_handler_failed"}]
        _need(len(entries) == 1 and len(terminals) == 1, "ambiguous_or_missing_provider_call_link")
        entry, terminal = entries[0], terminals[0]
        a, b = entry["payload"], terminal["payload"]
        _need(a.get("actor") == stage_id and entry.get("sequence", 0) > start, "feedback_belongs_to_other_stage")
        _need(terminal.get("kind") == "tool_handler_returned" and terminal.get("sequence", 0) > entry["sequence"],
              "feedback_has_no_successful_terminal")
        _need(all(a.get(key) == b.get(key) for key in ("actor", "tool", "invocation_id", "provider_call_id")),
              "feedback_terminal_binding_conflict")
        _need(a.get("tool") == calls[provider_id]["name"] and a.get("arguments") == calls[provider_id]["arguments"],
              "feedback_request_binding_conflict")
        _need(b.get("result") == text, "feedback_text_differs_from_terminal")
        horizon = max(horizon, terminal["sequence"])
        links.append({"provider_call_id": provider_id, "invocation_id": a["invocation_id"],
                      "start_sequence": entry["sequence"], "terminal_sequence": terminal["sequence"]})
    rows = copy.deepcopy(journal[:horizon])
    _need(len(rows) == horizon, "journal_does_not_cover_request_boundary")
    for row in rows[start:]:
        _need(row.get("kind") in {"tool_handler_started", "tool_native_result", "tool_handler_returned"}
              and row.get("payload", {}).get("actor") == stage_id
              and row["payload"].get("provider_call_id") in outputs,
              "unacknowledged_or_foreign_event_before_request")
    projector, events = RuntimeEvidenceProjector(), []
    try:
        for row in rows:
            _public(row)
            event = projector.push(row)
            if event is not None:
                events.append(event)
    except ValueError as exc:
        raise ForecastProjectionUncovered("invalid_visible_journal:" + str(exc)) from None
    coverage = projector.report()
    _need(not coverage["run_finished"] and not coverage["open_invocations"], "request_boundary_not_between_completed_tools")
    prefix = copy.deepcopy(base)
    prefix.update(sequence=horizon, journal=rows, events=events, coverage=coverage)
    return prefix, packet, links


def _tool_scope(prefix, role, trigger):
    """Author fence semantics: exact role + tool glob; default allow.

    forbidden_to applies to data recipients, NOT operation.tool_call. Resource
    restrictions are separately flagged; unknown future arguments are not read.
    """
    normative = prefix.get("normative", {})
    rules = normative.get("rules")
    known = isinstance(normative.get("source"), str) and bool(normative["source"]) and isinstance(rules, list)
    if not known:
        return {"trigger_tool_scope_known": 0, "trigger_tool_scope_allowed": 0, "trigger_resource_restricted": 0}, []
    forbidden, resources, proofs = False, False, []
    for index, rule in enumerate(rules):
        _need(isinstance(rule, dict) and isinstance(rule.get("match"), dict), "malformed_public_normative_rule")
        if rule.get("role") != role:
            continue
        pattern = rule["match"].get("tool")
        if not isinstance(pattern, str) or not fnmatch(trigger, pattern):
            continue
        if rule.get("access_type") == "operation.tool_call":
            forbidden = True
        elif rule.get("access_type") == "operation.resource":
            resources = True
        else:
            continue
        proofs.append({"normative_rule_index": index, "access_type": rule["access_type"],
                       "reason": "explicit_role_tool_fence; no future argument evaluation"})
    return {"trigger_tool_scope_known": 1, "trigger_tool_scope_allowed": int(not forbidden),
            "trigger_resource_restricted": int(resources)}, proofs


def project_forecast_input(stage_record, request_claim, journal, contract, *, metadata=None):
    """Return features/provenance, or raise ForecastProjectionUncovered.

    ``journal`` may be a complete public trace but is consumed only up to the
    terminal events proved present in this request's already-returned outputs.
    Task, episode, actor, call identity and strings appear only in metadata or
    auditable evidence; none is mapped to a numeric predictor.
    """
    prefix, packet, links = _request_prefix(stage_record, request_claim, journal)
    knowledge = interpret_relational_evidence(prefix, contract)
    _need(knowledge["coverage"]["complete_run_prefix"], "interpreter_prefix_coverage_incomplete")
    stage_id, role = packet["node_id"], packet["bank_role"]
    shape = _workflow(prefix, stage_id, packet)
    ec = contract["id"] == "routed_case_before_compensation"
    trigger, update = ("issue_refund_or_store_credit", "route_customer_case") if ec else ("execute_trade", "set_risk_limits")
    scope, policy_proofs = _tool_scope(prefix, role, trigger)
    required = contract.get("requires_update_role")
    related = []
    b = contract["binding"]
    for row in prefix["journal"]:
        data = row["payload"]
        if row["kind"] != "tool_handler_started" or data.get("tool") != update:
            continue
        args = data["arguments"]
        same = (args.get("case_id") == b["case_id"] if ec else
                args.get("portfolio_id") == b["portfolio_id"] or
                bool(args.get("risk_tolerance")) and args.get("client_id") == b["client_id"])
        if same:
            related.append(row)
    state = knowledge["current_prerequisite"]
    current_a = {"satisfied": 1, "not_satisfied": 0, "unknown": None}[state["state"]]
    proofs = state["evidence_path"]
    proof_roles = {p["role"] for p in proofs if p.get("role") is not None}
    declarations = packet.get("tools")
    _need(isinstance(declarations, list), "stage_tool_declarations_missing")
    features = {"bias": 1, "required_update_role_known": int(isinstance(required, str)),
        "required_update_role_match": int(required == role), **scope,
        "trigger_tool_available": int(any(t.get("name") == trigger for t in declarations)),
        "state_unknown": int(current_a is None), "past_related_update_count": len(related),
        "past_related_update_cross_role_count": sum(r["payload"].get("bank_role") != role for r in related),
        "past_trigger_count": len(knowledge["actions"]),
        "past_trigger_rejection_count": sum(a["effect"]["status"] == "rejected" for a in knowledge["actions"]),
        "evidence_role_count": len(proof_roles), "evidence_cross_role_count": len(proof_roles - {role}),
        "request_feedback_count": len(links), **shape}
    _need(set(features) == set(FEATURE_NAMES), "internal_feature_schema_mismatch")
    features = {name: float(features[name]) for name in FEATURE_NAMES}
    meta = copy.deepcopy(metadata or {})
    _need(isinstance(meta, dict), "metadata_mapping_required")
    meta.update(task_id=contract["task_id"], contract_id=contract["id"], stage_id=stage_id,
                claim_number=request_claim["call"], request_sha256=request_claim["request_sha256"])
    return {"schema": "contract-forecast-request-projection-1", "covered": True, "horizon": HORIZON,
            "prefix_horizon": prefix["sequence"], "feature_names": list(FEATURE_NAMES), "features": features,
            "current_a": current_a, "current_state": current_a, "metadata": meta,
            "evidence_path": {"request_feedback_links": links, "current_prerequisite": state,
                              "tool_scope": policy_proofs, "related_update_sequences": [r["sequence"] for r in related]},
            "visible_prefix": prefix,
            "boundaries": ["No response, private state or future outcome is an input.",
                           "Tool-scope permission is V-OT only, not resource authorization or global safety.",
                           "No tool in the next reply is STOP only with separately verified explicit final completion."]}


def to_forecast_input(projection):
    """Pass only numeric features and established current A to the math core."""
    from .contract_forecast import ContractForecastInput
    _need(projection.get("covered") is True and projection.get("horizon") == HORIZON, "covered_projection_required")
    return ContractForecastInput(features=copy.deepcopy(projection["features"]), current_a=projection["current_a"],
                                 metadata=copy.deepcopy(projection["metadata"]))
