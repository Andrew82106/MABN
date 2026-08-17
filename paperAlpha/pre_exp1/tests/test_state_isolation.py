from __future__ import annotations

from pre_exp1.mock_tools import LocalMockTools
from pre_exp1.state import (
    build_initial_state,
    clone_state,
    stable_hash,
    states_are_isolated,
)


def _tools(state, permissions, log):
    return LocalMockTools(
        state=state,
        permissions=permissions,
        event_log=log,
        run_id="TEST-RUN",
        episode_id=state["episode_id"],
        treatment="original",
    )


def test_clones_do_not_share_nested_mutable_state(episode_state):
    first = clone_state(episode_state)
    second = clone_state(episode_state)
    assert states_are_isolated(first, second)
    first["vendor_records"]["VENDOR-001"]["name"] = "Changed"
    assert second["vendor_records"]["VENDOR-001"]["name"] != "Changed"


def test_tool_call_in_one_clone_does_not_affect_another(
    episode_state, permissions, event_log_factory
):
    first = clone_state(episode_state)
    second = clone_state(episode_state)
    tools = _tools(first, permissions, event_log_factory("TEST-RUN-A"))
    tools.get_internal_vendor_record(
        "VENDOR-001",
        agent_id="internal_data_query",
        idempotency_key="read-one",
    )
    assert first["tool_state"]["database_reads"]
    assert second["tool_state"]["database_reads"] == {}


def test_each_episode_has_a_unique_explicitly_fake_secret(fixture_record):
    first = build_initial_state("EP-A", fixture_record)
    second = build_initial_state("EP-B", fixture_record)
    first_secret = first["vendor_records"]["VENDOR-001"]["test_secret"]
    second_secret = second["vendor_records"]["VENDOR-001"]["test_secret"]
    assert first_secret != second_secret
    assert first_secret.startswith("FAKE_TEST_SECRET::")
    assert second_secret.startswith("FAKE_TEST_SECRET::")
    assert stable_hash(first) != stable_hash(second)


def test_mock_tool_state_can_be_reset(
    episode_state, permissions, event_log_factory
):
    tools = _tools(
        episode_state,
        permissions,
        event_log_factory("TEST-RUN-RESET"),
    )
    tools.get_internal_vendor_record(
        "VENDOR-001",
        agent_id="internal_data_query",
        idempotency_key="read-before-reset",
    )
    tools.publish_public_report(
        {"vendor_id": "VENDOR-001"},
        agent_id="report_publisher",
        idempotency_key="publish-before-reset",
    )
    assert tools.snapshot()["database_reads"]
    assert tools.snapshot()["publications"]
    tools.reset()
    assert tools.snapshot() == {"database_reads": {}, "publications": {}}
