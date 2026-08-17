from __future__ import annotations

import pytest

from pre_exp1.mock_tools import LocalMockTools
from pre_exp1.permissions import PermissionDenied


def _tools(state, permissions, log):
    return LocalMockTools(
        state=state,
        permissions=permissions,
        event_log=log,
        run_id="TEST-RUN",
        episode_id=state["episode_id"],
        treatment="original",
    )


def test_unauthorized_agent_cannot_read_internal_record(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory()
    tools = _tools(episode_state, permissions, log)
    with pytest.raises(PermissionDenied):
        tools.get_internal_vendor_record(
            "VENDOR-001",
            agent_id="coordinator",
            idempotency_key="denied-read",
        )
    assert episode_state["tool_state"]["database_reads"] == {}


def test_unauthorized_agent_cannot_publish(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory()
    tools = _tools(episode_state, permissions, log)
    with pytest.raises(PermissionDenied):
        tools.publish_public_report(
            {"vendor_id": "VENDOR-001"},
            agent_id="coordinator",
            idempotency_key="denied-publish",
        )
    assert episode_state["tool_state"]["publications"] == {}


def test_permission_denials_are_logged(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory()
    tools = _tools(episode_state, permissions, log)
    with pytest.raises(PermissionDenied):
        tools.get_internal_vendor_record(
            "VENDOR-001",
            agent_id="report_publisher",
            idempotency_key="denied",
        )
    events = list(log.read())
    assert len(events) == 1
    assert events[0]["event_type"] == "permission_denied"
    assert events[0]["details"]["tool_name"] == "get_internal_vendor_record"
    assert events[0]["state_before_hash"] == events[0]["state_after_hash"]


def test_authorized_roles_can_call_only_their_tools(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory()
    tools = _tools(episode_state, permissions, log)
    record = tools.get_internal_vendor_record(
        "VENDOR-001",
        agent_id="internal_data_query",
        idempotency_key="allowed-read",
    )
    result = tools.publish_public_report(
        {"vendor_id": record["vendor_id"]},
        agent_id="report_publisher",
        idempotency_key="allowed-publish",
    )
    assert result["status"] == "published_locally"
    assert permissions.is_allowed(
        "internal_data_query", "get_internal_vendor_record"
    )
    assert not permissions.is_allowed(
        "internal_data_query", "publish_public_report"
    )

