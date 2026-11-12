"""Fail-closed deterministic ledger replay."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .canonical import content_hash
from .ledger import LedgerIntegrityError, iter_events, verify_ledger
from .models import Action
from .state import StateTransitionError, apply_action, apply_effects, state_hash


class ReplayError(RuntimeError):
    pass


def replay(
    ledger_path: Path,
    initial_state: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_head: str | None = None,
    expected_file_hash: str | None = None,
    expected_final_state_hash: str | None = None,
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    if audit is not None:
        audit["model_calls"] = 0
        audit["tool_executions"] = 0
    try:
        verify_ledger(
            ledger_path, expected_run_id=expected_run_id, expected_head=expected_head,
            expected_file_hash=expected_file_hash,
        )
    except LedgerIntegrityError as exc:
        raise ReplayError(str(exc)) from exc
    state = copy.deepcopy(dict(initial_state))
    initial_hash = state_hash(state)
    finished_hash: str | None = None
    for event in iter_events(ledger_path):
        payload = event["payload"]
        if event["event_type"] == "run_started" and payload["initial_state_hash"] != initial_hash:
            raise ReplayError("Initial state does not match ledger")
        if payload.get("state_before_hash") != state_hash(state):
            raise ReplayError(f"State-before hash mismatch at sequence {event['sequence']}")
        if event["event_type"] == "action_applied":
            state = apply_action(state, Action.from_dict(payload["action"]))
        if event["event_type"] == "tool_applied":
            effects = payload.get("effects")
            request = payload.get("request")
            summary = payload.get("result_summary")
            if (
                not isinstance(effects, list)
                or not effects
                or not isinstance(request, Mapping)
                or not isinstance(summary, Mapping)
            ):
                raise ReplayError("Malformed recorded tool effect")
            final_effect = effects[-1]
            try:
                entry = final_effect["params"]["value"]
                path = final_effect["params"]["path"]
                if (
                    final_effect["kind"] != "append_value"
                    or path != ["tool_results", request["actor"]]
                    or entry["tool_name"] != request["tool_name"]
                    or content_hash(entry["result"]) != summary["result_hash"]
                ):
                    raise ReplayError("Recorded tool result summary or visibility mismatch")
                state = apply_effects(state, effects)
            except (KeyError, TypeError, StateTransitionError) as exc:
                raise ReplayError("Malformed recorded tool effects") from exc
        if payload.get("state_after_hash") != state_hash(state):
            raise ReplayError(f"State-after hash mismatch at sequence {event['sequence']}")
        if event["event_type"] == "run_finished":
            finished_hash = payload["final_state_hash"]
    actual = state_hash(state)
    if finished_hash is None or finished_hash != actual:
        raise ReplayError("Missing or inconsistent run_finished event")
    if expected_final_state_hash is not None and actual != expected_final_state_hash:
        raise ReplayError("Final state does not match trusted receipt")
    return state
