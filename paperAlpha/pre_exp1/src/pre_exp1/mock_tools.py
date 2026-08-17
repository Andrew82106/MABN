"""Deterministic local-only mock database and publication sink."""

from __future__ import annotations

import copy
from typing import Any

from .event_log import AppendOnlyEventLog
from .permissions import PermissionEnforcer
from .state import apply_replay_action, reset_tool_state, stable_hash


class IdempotencyConflict(RuntimeError):
    pass


class LocalMockTools:
    """Episode-scoped local tools with code permissions and idempotency."""

    def __init__(
        self,
        *,
        state: dict[str, Any],
        permissions: PermissionEnforcer,
        event_log: AppendOnlyEventLog,
        run_id: str,
        episode_id: str,
        treatment: str,
    ) -> None:
        self.state = state
        self.permissions = permissions
        self.event_log = event_log
        self.run_id = run_id
        self.episode_id = episode_id
        self.treatment = treatment

    def get_internal_vendor_record(
        self,
        vendor_id: str,
        *,
        agent_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {
            "tool_name": "get_internal_vendor_record",
            "vendor_id": vendor_id,
            "idempotency_key": idempotency_key,
        }
        request_fingerprint = stable_hash(request)
        before = stable_hash(self.state)
        self.permissions.require(
            agent_id,
            "get_internal_vendor_record",
            event_log=self.event_log,
            episode_id=self.episode_id,
            treatment=self.treatment,
            payload=request,
            state_hash=before,
        )
        existing = self.state["tool_state"]["database_reads"].get(idempotency_key)
        if existing is not None:
            if existing.get("request_fingerprint") != request_fingerprint:
                self._raise_idempotency_conflict(
                    agent_id=agent_id,
                    tool_name="get_internal_vendor_record",
                    idempotency_key=idempotency_key,
                    request=request,
                    state_hash=before,
                )
            self.event_log.append(
                episode_id=self.episode_id,
                event_type="tool_idempotent_replay",
                agent_id=agent_id,
                edge_id=None,
                treatment=self.treatment,
                payload=request,
                state_before_hash=before,
                state_after_hash=before,
                details={
                    "tool_name": "get_internal_vendor_record",
                    "idempotency_key": idempotency_key,
                },
            )
            return copy.deepcopy(existing["result"])
        if vendor_id not in self.state["vendor_records"]:
            raise KeyError(f"Unknown fictional vendor: {vendor_id}")
        result = copy.deepcopy(self.state["vendor_records"][vendor_id])
        action = {
            "type": "record_database_read",
            "idempotency_key": idempotency_key,
            "agent_id": agent_id,
            "vendor_id": vendor_id,
            "result": result,
            "request_fingerprint": request_fingerprint,
        }
        apply_replay_action(self.state, action)
        after = stable_hash(self.state)
        self.event_log.append(
            episode_id=self.episode_id,
            event_type="tool_read_internal_record",
            agent_id=agent_id,
            edge_id=None,
            treatment=self.treatment,
            payload={"request": request, "result": result},
            state_before_hash=before,
            state_after_hash=after,
            details={
                "tool_name": "get_internal_vendor_record",
                "idempotency_key": idempotency_key,
                "replay_action": action,
            },
        )
        return result

    def publish_public_report(
        self,
        report: dict[str, Any],
        *,
        agent_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {
            "tool_name": "publish_public_report",
            "report": copy.deepcopy(report),
            "idempotency_key": idempotency_key,
        }
        request_fingerprint = stable_hash(request)
        before = stable_hash(self.state)
        self.permissions.require(
            agent_id,
            "publish_public_report",
            event_log=self.event_log,
            episode_id=self.episode_id,
            treatment=self.treatment,
            payload=request,
            state_hash=before,
        )
        existing = self.state["tool_state"]["publications"].get(idempotency_key)
        if existing is not None:
            if existing.get("request_fingerprint") != request_fingerprint:
                self._raise_idempotency_conflict(
                    agent_id=agent_id,
                    tool_name="publish_public_report",
                    idempotency_key=idempotency_key,
                    request=request,
                    state_hash=before,
                )
            self.event_log.append(
                episode_id=self.episode_id,
                event_type="tool_idempotent_replay",
                agent_id=agent_id,
                edge_id=None,
                treatment=self.treatment,
                payload=request,
                state_before_hash=before,
                state_after_hash=before,
                details={
                    "tool_name": "publish_public_report",
                    "idempotency_key": idempotency_key,
                },
            )
            return {
                "status": "already_published",
                "idempotency_key": idempotency_key,
            }
        action = {
            "type": "record_publication",
            "idempotency_key": idempotency_key,
            "agent_id": agent_id,
            "report": copy.deepcopy(report),
            "request_fingerprint": request_fingerprint,
        }
        apply_replay_action(self.state, action)
        after = stable_hash(self.state)
        self.event_log.append(
            episode_id=self.episode_id,
            event_type="tool_publish_local_sink",
            agent_id=agent_id,
            edge_id=None,
            treatment=self.treatment,
            payload=request,
            state_before_hash=before,
            state_after_hash=after,
            details={
                "tool_name": "publish_public_report",
                "idempotency_key": idempotency_key,
                "local_only": True,
                "replay_action": action,
            },
        )
        return {"status": "published_locally", "idempotency_key": idempotency_key}

    def reset(self) -> None:
        reset_tool_state(self.state)

    def state_hash(self) -> str:
        return stable_hash(self.state["tool_state"])

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state["tool_state"])

    def _raise_idempotency_conflict(
        self,
        *,
        agent_id: str,
        tool_name: str,
        idempotency_key: str,
        request: dict[str, Any],
        state_hash: str,
    ) -> None:
        reason = (
            f"Idempotency key {idempotency_key!r} was reused with a different "
            f"{tool_name} request"
        )
        self.event_log.append(
            episode_id=self.episode_id,
            event_type="tool_idempotency_conflict",
            agent_id=agent_id,
            edge_id=None,
            treatment=self.treatment,
            payload=request,
            state_before_hash=state_hash,
            state_after_hash=state_hash,
            details={
                "tool_name": tool_name,
                "idempotency_key": idempotency_key,
                "reason": reason,
            },
        )
        raise IdempotencyConflict(reason)
