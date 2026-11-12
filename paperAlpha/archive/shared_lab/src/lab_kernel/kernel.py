"""Single authorization gateway and deterministic execution engine."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ledger import AppendOnlyLedger
from .models import Action, AgentRequest, Hook
from .observation import build_observation, finalize_hook_observation
from .permissions import PermissionDenied, PermissionEnforcer
from .provenance import build_provenance, write_receipt
from .scenario import Scenario
from .state import StateTransitionError, apply_action, state_hash
from .tools import ToolError, ToolRegistry


@dataclass(frozen=True)
class KernelResult:
    accepted: bool
    reason: str | None
    action: Action | None
    tool_result: Mapping[str, Any] | None = None


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
        self.tools = ToolRegistry(scenario, self.permissions)
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

    def deny_agent_request(
        self, actor: str, category: str, reason: str
    ) -> KernelResult:
        before = state_hash(self._state)
        self.ledger.append("action_denied", {
            "request": {"actor": actor, "category": category},
            "reason": reason,
            "state_before_hash": before,
            "state_after_hash": before,
        })
        return KernelResult(False, reason, None)

    def dispatch_agent_request(self, request: AgentRequest) -> KernelResult:
        """Only entry point for untrusted/model-controlled agent intent."""

        if not isinstance(request.params, Mapping):
            return self.deny_agent_request(
                request.actor, "malformed_action", "Agent request params must be an object"
            )
        if request.kind == "send_message":
            expected = {"edge_id", "recipient", "content"}
            if (
                set(request.params) != expected
                or not isinstance(request.params["edge_id"], str)
                or not request.params["edge_id"]
                or not isinstance(request.params["recipient"], str)
                or not request.params["recipient"]
            ):
                return self.deny_agent_request(
                    request.actor, "malformed_action", "Message request fields are not exact"
                )
            try:
                json.dumps(request.params["content"], ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError):
                return self.deny_agent_request(
                    request.actor, "malformed_action", "Message content must be finite JSON"
                )
            message = {
                "edge_id": request.params["edge_id"],
                "recipient": request.params["recipient"],
                "content": copy.deepcopy(request.params["content"]),
                "sender": request.actor,
            }
            return self.dispatch(Action(
                request.actor, "message.send", "send_message", {"message": message}
            ))
        if request.kind == "call_tool":
            if (
                set(request.params) != {"tool_name", "arguments"}
                or not isinstance(request.params["tool_name"], str)
                or not request.params["tool_name"]
                or not isinstance(request.params["arguments"], Mapping)
            ):
                return self.deny_agent_request(
                    request.actor, "malformed_action", "Tool request fields are not exact"
                )
            return self._dispatch_tool(
                request.actor,
                request.params["tool_name"],
                request.params["arguments"],
            )
        return self.deny_agent_request(
            request.actor, "unknown_action", "Untrusted agents may only send messages or call tools"
        )

    def _dispatch_tool(
        self, actor: str, tool_name: str, arguments: Any
    ) -> KernelResult:
        before = state_hash(self._state)
        request = {
            "actor": actor,
            "kind": "call_tool",
            "tool_name": tool_name,
            "arguments": copy.deepcopy(arguments),
        }
        try:
            execution = self.tools.execute(self._state, actor, tool_name, arguments)
        except ToolError as exc:
            self.ledger.append("action_denied", {
                "request": {**request, "category": exc.category},
                "reason": str(exc),
                "state_before_hash": before,
                "state_after_hash": before,
            })
            return KernelResult(False, str(exc), None)
        next_state = copy.deepcopy(dict(execution.next_state))
        after = state_hash(next_state)
        self.ledger.append("tool_applied", {
            "request": request,
            "required_capability": execution.required_capability,
            "result_summary": dict(execution.result_summary),
            "effects": [copy.deepcopy(dict(item)) for item in execution.effects],
            "state_before_hash": before,
            "state_after_hash": after,
        })
        self._state = next_state
        return KernelResult(True, None, None, copy.deepcopy(dict(execution.result)))

    def dispatch(self, action: Action) -> KernelResult:
        before = state_hash(self._state)
        try:
            self.permissions.require_action(
                self._state, action.actor, action.kind, action.capability
            )
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
            self.permissions.require_action(
                self._state,
                transformed.actor,
                transformed.kind,
                transformed.capability,
            )
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
        base = build_observation(self.scenario, self._state, agent_id)
        observation: Mapping[str, Any] = copy.deepcopy(base)
        for hook in self.hooks:
            method = getattr(hook, "transform_observation", None)
            if method is not None:
                observation = method(agent_id, observation, self._context())
        policy = self.scenario.observation_policy(agent_id)
        return finalize_hook_observation(base, observation, policy["hook_fields"])

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
