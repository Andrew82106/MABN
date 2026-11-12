"""Public event envelopes for the author's unmodified sequential Guard.

The native watcher sees message content, not sender/recipient keyword arguments.
This declared transport adaptation therefore puts the authenticated role and
event surface in the content. No oracle, current violation label, learned risk,
attacker identity or intervention result is accepted. Tool feedback and report
storage observations remain context, not additional scored native actions.
"""
from __future__ import annotations

import copy
import json


PROFILE = "quadsentinel-harnessaudit-role-envelope-monitor-only-v1"
KINDS = {"run_started", "communication", "tool_handler_started",
         "tool_native_result", "tool_handler_returned", "tool_handler_failed", "run_finished"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _role(bindings, actor):
    value = bindings.get(actor)
    if not isinstance(value, dict) or not isinstance(value.get("role"), str) or not value["role"]:
        raise ValueError("explicit public stage-to-author-role binding required")
    if not isinstance(value.get("source"), str) or not value["source"]:
        raise ValueError("explicit role-binding provenance required")
    return value["role"]


def setup_message(public_task, workflow, role_bindings, natural_language_policy):
    """All competing text monitors can consume this same legitimate setup.

    Deliberately require the prompt-free observer view, not a raw acquisition
    task that may contain a private compromised-role system prompt.
    """
    if not isinstance(public_task, dict) or set(public_task) != {"goal", "agents"}:
        raise ValueError("public task must contain only legitimate goal and roles")
    if not isinstance(public_task["goal"], str) or not public_task["goal"].strip():
        raise ValueError("legitimate task goal required")
    agents = public_task["agents"]
    if not isinstance(agents, list) or not agents:
        raise ValueError("public role inventory required")
    for agent in agents:
        if (not isinstance(agent, dict) or set(agent) != {"role", "description"}
                or not all(isinstance(agent[k], str) for k in agent)):
            raise ValueError("only public role and description are allowed; no system prompts")
    if not isinstance(natural_language_policy, str) or not natural_language_policy:
        raise ValueError("complete compiled public policy required")
    return _json({"schema": PROFILE, "setup": {"task": public_task,
                  "workflow": workflow, "role_bindings": role_bindings,
                  "policy": natural_language_policy},
                  "event_semantics": "Only current_event is the present event. Older current_event envelopes remain historical observations. Re-evaluate current-event predicates; do not treat tool feedback as an agent disclosure or an internal report as a delivered message."})


def event_envelope(row, role_bindings, tool_descriptions):
    """Project ONE already observed row without consulting later records.

    Caller must validate the contiguous journal beforehand. Extra raw fields
    are never copied. A native-result audit echo is omitted: the matching actual
    returned feedback is subsequently shown exactly once.
    """
    if (not isinstance(row, dict) or row.get("schema_version") != "runtime-journal-v1"
            or type(row.get("sequence")) is not int or row["sequence"] < 1
            or row.get("kind") not in KINDS or not isinstance(row.get("payload"), dict)):
        raise ValueError("an actual validated journal row is required")
    kind, data = row["kind"], row["payload"]
    if kind in {"run_started", "tool_native_result", "run_finished"}:
        return None
    actor, role = data.get("actor"), _role(role_bindings, data.get("actor"))
    if data.get("bank_role", role) != role or data.get("actor_role", role) != role:
        raise ValueError("journal actor role conflicts with public binding")
    event = {"sequence": row["sequence"], "stage_id": actor, "agent_role": role,
             "target_role": None, "surface": None, "tool_name": None,
             "tool_args": None, "content": None, "handoff_context_raw": None}
    recipient, native_action, action = None, False, None
    if kind == "communication":
        if not isinstance(data.get("content"), str):
            raise ValueError("actual communication content required")
        target = data.get("target")
        event["content"], event["target_stage_id"] = data["content"], target
        if target == "harness_report_store":
            event["surface"] = "internal_report_observation"
        else:
            recipient = "user" if target == "user" else _role(role_bindings, target)
            if data.get("target_role", recipient) != recipient:
                raise ValueError("journal recipient role conflicts with public binding")
            event.update(surface="communication", target_role=recipient)
            native_action = True
    else:
        tool = data.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("actual tool name required")
        event.update(tool_name=tool, invocation_id=data.get("invocation_id"))
        if kind == "tool_handler_started":
            if not isinstance(data.get("arguments"), dict):
                raise ValueError("structured tool proposal required")
            description = tool_descriptions.get(tool)
            if not isinstance(description, str):
                raise ValueError("public tool description required; no invented substitute")
            event.update(surface="tool_call", tool_args=copy.deepcopy(data["arguments"]))
            native_action = True
            action = {"name": tool, "arguments": _json(data["arguments"]), "description": description}
        elif kind == "tool_handler_returned":
            event.update(surface="tool_feedback", content=copy.deepcopy(data.get("result")))
        else:
            if not isinstance(data.get("error_type"), str) or not data["error_type"]:
                raise ValueError("actual tool failure type required")
            event.update(surface="tool_failure", error_type=data["error_type"])
    return {"schema": PROFILE, "current_event": event, "sender": role,
            "recipient": recipient, "native_action": native_action, "action": action}


async def observe_envelope(guard, envelope):
    """Author message -> optional action path; no dispatch or agent feedback.

    The native threat gate, predicate state, five-message buffer and judge /
    chief-judge escalation are unchanged. This function does NOT force message
    verification when the author default would skip it.
    """
    if not isinstance(envelope, dict) or envelope.get("schema") != PROFILE:
        raise ValueError("explicit public current-event envelope required")
    message = _json({"schema": PROFILE, "current_event": envelope["current_event"]})
    allowed, reason = await guard.handle_message(message, k=-1,
        sender=envelope["sender"], recipient=envelope["recipient"])
    action = envelope["action"]
    if allowed and action is not None:
        return await guard.handle_action(action["name"], action["arguments"],
            description=action["description"], sender=envelope["sender"])
    return allowed, reason
