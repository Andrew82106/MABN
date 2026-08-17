"""Builders for auditable isolation, edge cutting, revocation, and rollback actions."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from lab_kernel.canonical import content_hash
from lab_kernel.models import Action


def isolate(actor: str, target: str, capability: str = "defense.isolate") -> Action:
    return Action(actor, capability, "isolate_agent", {"agent_id": target})


def cut_edge(actor: str, edge_id: str, capability: str = "defense.cut_edge") -> Action:
    return Action(actor, capability, "cut_edge", {"edge_id": edge_id})


def revoke(actor: str, target: str, revoked: str, capability: str = "defense.revoke") -> Action:
    return Action(actor, capability, "revoke_permission", {"agent_id": target, "capability": revoked})


def rollback(actor: str, snapshot: Mapping[str, Any], capability: str = "defense.rollback") -> Action:
    copied = copy.deepcopy(dict(snapshot))
    return Action(actor, capability, "rollback_state", {"snapshot": copied, "snapshot_hash": content_hash(copied)})
