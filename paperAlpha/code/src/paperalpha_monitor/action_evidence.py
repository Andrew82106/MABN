"""Action-scoped, phase-aware compilation of externally observed MAS events.

The adapter must authenticate logger/checker origin outside this module. Event
text, bare permission flags and agent self-reports cannot create an accepted
check. Strict binding prevents a check being reused for changed arguments.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .dynamic_joint_bn import JointTrace, Operation, Step, KNOWLEDGE
from .joint_stream import JointEventCompiler
from .streaming_bn import PRIVILEGED_TOOLS, reject_outcomes

from ._resources import static_resource

SPEC_PATH = static_resource('knowledge/templates/action_scoped_bn_v3.json')
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
SPEC_SHA256 = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
if SPEC["base_registry_id"] != KNOWLEDGE.registry_id or SPEC["permission_states"] != list(KNOWLEDGE.states("permission")):
    raise ValueError("action specification uses an unsupported base registry")
if SPEC["action_binding_fields"] != ["action_id", "actor", "principal", "tool", "resource", "arguments"]:
    raise ValueError("unsupported action binding specification")
if len(SPEC["information_head_features"]) != 4 or len(SPEC["head_features"]) != 4 or len(SPEC["head_initial_coefficients"]) != 4:
    raise ValueError("unsupported action readout specification")
BINDING_FIELDS = tuple(SPEC["action_binding_fields"])


@dataclass(frozen=True)
class ActionOperation(Operation):
    rule: str


@dataclass(frozen=True)
class ActionStep(Step):
    pending_pairs: tuple[tuple[int, int], ...]
    dispatched_pairs: tuple[tuple[int, int], ...]
    actions: tuple[dict, ...]
    ignored_evidence: tuple[dict, ...]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def binding(event):
    for field in BINDING_FIELDS:
        if field not in event:
            raise ValueError(f"action requires explicit {field}; missing scope is not inferred")
        if field != "arguments" and (not isinstance(event[field], str) or not event[field]):
            raise ValueError(f"action {field} must be a nonempty string")
    if not isinstance(event["arguments"], dict):
        raise ValueError("action arguments must be a structured dictionary")
    value = {field: copy.deepcopy(event[field]) for field in BINDING_FIELDS}
    # Validate serializability and reject NaN before committing any event.
    return value, hashlib.sha256(canonical(value).encode()).hexdigest()


class ActionEventCompiler:
    step_type = ActionStep

    def __init__(self, *, trusted_checkers=(), max_variables=12, **context):
        if not isinstance(trusted_checkers, (list, tuple, set, frozenset)) or any(not isinstance(x, str) or not x for x in trusted_checkers):
            raise ValueError("trusted_checkers must contain nonempty configured identities")
        self.trusted_checkers = frozenset(trusted_checkers)
        self._base = JointEventCompiler(max_variables=max_variables, **context)
        self.max_variables = max_variables
        self.permission_mode = self._base.validator.permission_mode
        self.variables, self._base_indices, self._actions, self._seen_events = [], {}, {}, {}
        self.last_step, self.last_was_duplicate = None, False
        self._sync_variables()

    @property
    def workflow(self):
        return copy.deepcopy(self._base.workflow)

    @property
    def event_count(self):
        return self._base.event_count

    def _sync_variables(self):
        for old in range(len(self._base_indices), len(self._base.variables)):
            value = copy.deepcopy(self._base.variables[old])
            if value["kind"] == "workflow_node":
                value.update(kind="information_state", risk_channel="information")
            self._base_indices[old] = len(self.variables)
            self.variables.append(value)

    @staticmethod
    def _sanitized(event):
        clean = copy.deepcopy(event)
        # A raw flag (possibly produced by the monitored agent) is not an
        # authenticated action-scoped check in this opt-in input protocol.
        for key in ("permission_mismatch", "permission_denied"):
            clean.pop(key, None)
        if clean.get("event_type") in {"authorization_checked", "action_cancelled"}:
            clean["event_type"] = "action_observation"
        return clean

    def _is_action(self, event):
        return event["event_type"] in {"authorization_checked", "action_cancelled"} or (
            event["event_type"] in {"tool_proposed", "tool_call"} and event.get("tool") in PRIVILEGED_TOOLS)

    def _validate_event(self, event):
        reject_outcomes(event, "event")

    def _binding(self, event):
        return binding(event)

    def _action_metadata(self, event, bound):
        return {}

    def _validate_check_scope(self, bound):
        pass  # v3 binding() has already required the complete scope.

    def _operation_parameter(self, operation):
        return operation.parameter

    def _authorization_parameter(self, action, status):
        return KNOWLEDGE.cell_index("authorization", {"role": action["role"], "permission": status})

    def _validate_capacity(self, keys, new_action, dispatch):
        extra = int(new_action) + int(dispatch)
        if len(self.variables) + len(keys - self._base.indices.keys()) + extra > self.max_variables:
            raise ValueError("action exact state exceeds variable limit; event was not accepted")

    def push(self, event):
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")
        self._validate_event(event)
        fingerprint = canonical(event)
        event_id = event.get("event_id")
        if event_id is not None and (not isinstance(event_id, str) or not event_id):
            raise ValueError("event_id must be a nonempty string")
        if event_id is not None and event_id in self._seen_events:
            if self._seen_events[event_id] != fingerprint:
                raise ValueError("conflicting repeated event_id")
            self.last_was_duplicate = True
            self.last_step = replace(self.last_step, operations=(), ignored_evidence=())
            return copy.deepcopy(self.last_step)
        clean = self._sanitized(event)
        keys = self._base._preflight(clean)
        sequence = event.get("sequence", self._base.validator.last_sequence + 1)
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= self._base.validator.last_sequence:
            raise ValueError("sequence must strictly increase in arrival order")
        kind = event["event_type"]
        is_action = self._is_action(event)
        actions = copy.deepcopy(self._actions)
        new_action, dispatch, accepted_check, ignored = False, False, False, []
        if any(k in event for k in ("permission_mismatch", "permission_denied")):
            ignored.append({"sequence": sequence, "reason": "bare_permission_flags_are_not_bound_checker_evidence"})
        if is_action:
            bound, digest = self._binding(event)
            action_id = bound["action_id"]
            old = actions.get(action_id)
            if old is not None and old["binding_sha256"] != digest:
                raise ValueError("action binding changed: actor/principal/tool/resource/arguments require a new action_id")
            if old is None:
                if kind not in {"tool_proposed", "tool_call"}:
                    raise ValueError("check/cancellation requires an already observed action")
                new_action = True
                node = self._base.aliases.get(bound["actor"], bound["actor"])
                actions[action_id] = {"action_id": action_id, "binding": bound, "binding_sha256": digest,
                    "node": node, "role": self._base.roles.get(node, "unknown"), "phase": "pending",
                    "created_sequence": sequence, "dispatch_sequence": None, "cancelled_sequence": None,
                    "permission_status": "unknown", "accepted_check": None, "check_revision": -1,
                    "authorization_variable": None, "information_at_dispatch_variable": None,
                    **self._action_metadata(event, bound)}
            action = actions[action_id]
            if action["phase"] != "pending":
                raise ValueError("completed/cancelled action cannot be changed; post-hoc reassessment is separate")
            if kind == "authorization_checked":
                self._validate_check_scope(bound)
                checker = event.get("checker_id")
                status, revision = event.get("permission_status"), event.get("check_revision")
                expiry = event.get("valid_until_sequence")
                if not isinstance(checker, str) or not checker or status not in SPEC["permission_states"]:
                    raise ValueError("check requires a checker_id and declared permission_status")
                if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                    raise ValueError("check_revision must be a nonnegative integer")
                if not isinstance(expiry, int) or isinstance(expiry, bool) or expiry < sequence:
                    raise ValueError("check requires a non-expired valid_until_sequence")
                if checker not in self.trusted_checkers or self.permission_mode == "masked":
                    ignored.append({"sequence": sequence, "reason": "unconfigured_checker" if checker not in self.trusted_checkers else "permission_masked"})
                else:
                    previous_check = action["accepted_check"]
                    if previous_check is not None and previous_check["checker_id"] != checker:
                        raise ValueError("conflicting checker streams require an explicit aggregation model")
                    if revision <= action["check_revision"]:
                        raise ValueError("check revisions must strictly increase; retry via event_id instead")
                    accepted_check = True
            dispatch = kind == "tool_call"
        self._validate_capacity(keys, new_action, dispatch)

        # The shared validator performs all remaining validation before it
        # commits. Above, no live actions or probability variables were changed.
        base_step = self._base.push(clean)
        self._sync_variables()
        operations = []
        for operation in base_step.operations:
            if operation.evidence["template"] == "authorization":
                continue
            receiver = self._base_indices[operation.receiver]
            sender = None if operation.sender is None else self._base_indices[operation.sender]
            rule = "seed_or" if sender is None else "carrier_or"
            operations.append(ActionOperation(receiver, sender, self._operation_parameter(operation), operation.confidence_feature,
                operation.is_message, copy.deepcopy(operation.evidence), rule))

        def revise(action, status, reason):
            parameter = self._authorization_parameter(action, status)
            operations.append(ActionOperation(action["authorization_variable"], None, parameter, 0., False,
                {"sequence": sequence, "template": "action_authorization", "registry_id": SPEC["registry_id"],
                 "action_id": action["action_id"], "binding_sha256": action["binding_sha256"],
                 "role": action["role"], "permission_status": status, "reason": reason}, "replace"))
            action["permission_status"] = status

        # Expiry also occurs on unrelated events, not only when a tool dispatches.
        for action in actions.values():
            check = action["accepted_check"]
            if action["phase"] == "pending" and check is not None and check["valid_until_sequence"] < sequence and action["permission_status"] != "unknown":
                revise(action, "unknown", "check_expired")
        if new_action:
            action = actions[action_id]
            action["authorization_variable"] = len(self.variables)
            self.variables.append({"kind": "action_authorization_state", "id": action_id, "node": action["node"],
                                   "role": action["role"], "binding_sha256": action["binding_sha256"]})
            revise(action, "unknown", "first_observed_action")
        if is_action:
            action = actions[action_id]
            if accepted_check:
                action["check_revision"] = event["check_revision"]
                action["accepted_check"] = {k: event[k] for k in ("checker_id", "check_revision", "permission_status", "valid_until_sequence")}
                action["accepted_check"]["observed_sequence"] = sequence
                revise(action, event["permission_status"], "latest_complete_check")
            if dispatch:
                action["phase"], action["dispatch_sequence"] = "dispatched", sequence
                frozen = action["information_at_dispatch_variable"] = len(self.variables)
                self.variables.append({"kind": "dispatched_information_state", "id": action_id, "node": action["node"],
                                       "binding_sha256": action["binding_sha256"]})
                information = self._base_indices[self._base.indices[("node", action["node"])]]
                operations.append(ActionOperation(frozen, information, -1, 0., False,
                    {"sequence": sequence, "template": "dispatch_information_snapshot", "action_id": action_id,
                     "binding_sha256": action["binding_sha256"]}, "copy"))
            elif kind == "action_cancelled":
                action["phase"], action["cancelled_sequence"] = "cancelled", sequence
        self._actions = actions
        if event_id is not None:
            self._seen_events[event_id] = fingerprint
        remap = lambda xs: tuple(self._base_indices[i] for i in xs)
        active = [a for a in actions.values() if a["phase"] != "cancelled"]
        observables = base_step.observables.copy()
        # Replace the legacy unscoped permission summaries with accepted checks.
        observables[8:11] = [bool(active), any(a["permission_status"] != "unknown" for a in active),
                             any(a["permission_status"] == "mismatch" for a in active)]
        pending = tuple((a["authorization_variable"], self._base_indices[self._base.indices[("node", a["node"])]])
                        for a in actions.values() if a["phase"] == "pending")
        dispatched = tuple((a["authorization_variable"], a["information_at_dispatch_variable"])
                           for a in actions.values() if a["phase"] == "dispatched")
        self.last_step = ActionStep(sequence, tuple(operations), observables, remap(base_step.observed_nodes),
            remap(base_step.capable_nodes), remap(base_step.terminal_nodes), pending, dispatched,
            tuple(copy.deepcopy(list(actions.values()))), tuple(ignored))
        self.last_was_duplicate = False
        return copy.deepcopy(self.last_step)


def compile_action_trace(trace, **context):
    reject_outcomes(trace)
    compiler = ActionEventCompiler(**context)
    steps = tuple(compiler.push(event) for event in trace.get("events", []))
    if not steps:
        raise ValueError("a nonempty episode is required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), steps, compiler.workflow)
