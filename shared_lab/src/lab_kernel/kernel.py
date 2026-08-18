"""Single authorization gateway and deterministic execution engine."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ledger import AppendOnlyLedger
from .models import Action, Hook
from .permissions import PermissionDenied, PermissionEnforcer
from .provenance import build_provenance, write_receipt
from .scenario import Scenario
from .state import StateTransitionError, apply_action, state_hash


@dataclass(frozen=True)
class KernelResult:
    accepted: bool
    reason: str | None
    action: Action | None


class Kernel:
    def __init__(
        self,
        scenario: Scenario,
        ledger_path: Path,
        run_id: str,
        *,
        hooks: Iterable[Hook] = (),
        receipt_path: Path | None = None,
        code_roots: Iterable[Path] | None = None,
    ) -> None:
        self.scenario = scenario
        self._state = scenario.initial_state()
        self.initial_state = copy.deepcopy(self._state)
        self.ledger = AppendOnlyLedger(ledger_path, run_id)
        self.run_id = run_id
        self.permissions = PermissionEnforcer()
        self.hooks = tuple(hooks)
        discovered_roots = {Path(__file__).resolve().parent}
        for hook in self.hooks:
            discovered_roots.add(Path(inspect.getfile(hook.__class__)).resolve().parent)
        self.code_roots = tuple(Path(item).resolve() for item in (code_roots or sorted(discovered_roots)))
        self.provenance = build_provenance(scenario.root, self.code_roots)
        self.receipt_path = (receipt_path or ledger_path.with_suffix(".receipt.json")).resolve()
        self.evaluation: Mapping[str, Any] | None = None
        self.ledger.append("run_started", {
            "scenario_tree_hash": scenario.tree_sha256,
            "provenance_hash": self.provenance["provenance_hash"],
            "initial_state_hash": state_hash(self._state),
            "state_before_hash": state_hash(self._state),
            "state_after_hash": state_hash(self._state),
        })

    @property
    def state(self) -> dict[str, Any]:
        """Return an isolated snapshot so callers cannot bypass dispatch()."""

        return copy.deepcopy(self._state)

    def _context(self) -> dict[str, Any]:
        return {"state": copy.deepcopy(self._state), "scenario_id": self.scenario.config["scenario_id"]}

    def _edge_allowed(self, action: Action) -> bool:
        if action.kind != "send_message":
            return True
        message = action.params.get("message", {})
        return any(
            edge.get("id") == message.get("edge_id")
            and edge.get("source") == action.actor
            and edge.get("target") == message.get("recipient")
            for edge in self._state.get("edges", [])
        )

    def dispatch(self, action: Action) -> KernelResult:
        before = state_hash(self._state)
        try:
            self.permissions.require(self._state, action.actor, action.capability)
        except PermissionDenied as exc:
            self.ledger.append("action_denied", {
                "action": action.to_dict(), "reason": str(exc),
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(False, str(exc), None)
        if not self._edge_allowed(action):
            reason = "Message edge is absent, cut, or does not match sender and recipient"
            self.ledger.append("action_denied", {
                "action": action.to_dict(), "reason": reason,
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(False, reason, None)
        transformed: Action | None = action
        for hook in self.hooks:
            method = getattr(hook, "before_action", None)
            if method is not None and transformed is not None:
                transformed = method(transformed, self._context())
        if transformed is None:
            self.ledger.append("action_suppressed", {
                "original_action": action.to_dict(),
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(True, "suppressed_by_hook", None)
        if (
            transformed.actor != action.actor
            or transformed.capability != action.capability
            or transformed.kind != action.kind
        ):
            reason = "Hook may change action parameters only, not authority or kind"
            self.ledger.append("action_denied", {
                "action": transformed.to_dict(), "reason": reason,
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(False, reason, None)
        try:
            self.permissions.require(self._state, transformed.actor, transformed.capability)
        except PermissionDenied as exc:
            self.ledger.append("action_denied", {
                "action": transformed.to_dict(), "reason": str(exc),
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(False, str(exc), None)
        if not self._edge_allowed(transformed):
            reason = "Hook-produced message edge is absent, cut, or mismatched"
            self.ledger.append("action_denied", {
                "action": transformed.to_dict(), "reason": reason,
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(False, reason, None)
        try:
            next_state = apply_action(self._state, transformed)
        except (KeyError, TypeError, StateTransitionError) as exc:
            reason = f"Invalid state transition: {exc}"
            self.ledger.append("action_denied", {
                "action": transformed.to_dict(), "reason": reason,
                "state_before_hash": before, "state_after_hash": before,
            })
            return KernelResult(False, reason, None)
        after = state_hash(next_state)
        self.ledger.append("action_applied", {
            "action": transformed.to_dict(), "original_action": action.to_dict(),
            "state_before_hash": before, "state_after_hash": after,
        })
        self._state = next_state
        return KernelResult(True, None, transformed)

    def observation_for(self, agent_id: str) -> Mapping[str, Any]:
        observation: Mapping[str, Any] = {
            "agent_id": agent_id,
            "state": copy.deepcopy(self._state),
            "inbox": [
                copy.deepcopy(item) for item in self._state.get("messages", [])
                if item.get("recipient") == agent_id
            ],
        }
        for hook in self.hooks:
            method = getattr(hook, "transform_observation", None)
            if method is not None:
                observation = method(agent_id, observation, self._context())
        return observation

    def finish(self) -> None:
        current = state_hash(self._state)
        self.evaluation = dict(self.scenario.evaluator()(self.state, self.scenario.config))
        self.ledger.append("run_evaluated", {
            "evaluation": dict(self.evaluation),
            "state_before_hash": current,
            "state_after_hash": current,
        })
        self.ledger.append("run_finished", {
            "final_state_hash": current,
            "state_before_hash": current,
            "state_after_hash": current,
        })
        self.ledger.close()
        write_receipt(
            self.receipt_path,
            self.provenance,
            run_id=self.run_id,
            ledger_path=self.ledger.path,
            ledger_head=self.ledger.head_hash,
            event_count=self.ledger.count,
            final_state=self._state,
        )
