"""Fail-closed deterministic ledger replay."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .ledger import LedgerIntegrityError, iter_events, verify_ledger
from .models import Action
from .state import apply_action, state_hash


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
) -> dict[str, Any]:
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
