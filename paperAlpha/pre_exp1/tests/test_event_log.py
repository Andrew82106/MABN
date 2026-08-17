from __future__ import annotations

import json

from pre_exp1.mock_tools import LocalMockTools
from pre_exp1.state import stable_hash


REQUIRED_FIELDS = {
    "run_id",
    "episode_id",
    "event_id",
    "event_type",
    "timestamp",
    "agent_id",
    "edge_id",
    "treatment",
    "payload_hash",
    "state_before_hash",
    "state_after_hash",
    "details",
}


def _append_sample(log, index):
    state_hash = stable_hash({"index": index})
    return log.append(
        episode_id="TEST-EP",
        event_type="sample",
        agent_id="coordinator",
        edge_id="edge_coordinator_to_internal",
        treatment="original",
        payload={"index": index},
        state_before_hash=state_hash,
        state_after_hash=state_hash,
        details={},
    )


def test_jsonl_lines_parse_independently_and_have_required_fields(
    event_log_factory,
):
    log = event_log_factory()
    _append_sample(log, 1)
    _append_sample(log, 2)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 2
    assert all(REQUIRED_FIELDS == set(event) for event in parsed)


def test_event_ids_are_unique_and_append_survives_reopen(event_log_factory):
    log = event_log_factory("APPEND-RUN")
    first = _append_sample(log, 1)
    initial_bytes = log.path.read_bytes()
    reopened = type(log)(log.path, "APPEND-RUN")
    second = _append_sample(reopened, 2)
    assert first.event_id != second.event_id
    assert reopened.path.read_bytes().startswith(initial_bytes)


def test_two_open_log_instances_generate_distinct_event_ids(
    event_log_factory,
):
    first_log = event_log_factory("CONCURRENT-ID-RUN")
    second_log = type(first_log)(
        first_log.path,
        "CONCURRENT-ID-RUN",
    )
    first = _append_sample(first_log, 1)
    second = _append_sample(second_log, 2)
    assert first.event_id != second.event_id
    assert len({event["event_id"] for event in first_log.read()}) == 2


def test_run_episode_edge_and_treatment_are_traceable(event_log_factory):
    log = event_log_factory()
    event = _append_sample(log, 1).to_dict()
    assert event["run_id"] == "TEST-RUN"
    assert event["episode_id"] == "TEST-EP"
    assert event["edge_id"] == "edge_coordinator_to_internal"
    assert event["treatment"] == "original"


def test_tool_state_changes_have_hashes_and_no_hidden_reasoning(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory()
    tools = LocalMockTools(
        state=episode_state,
        permissions=permissions,
        event_log=log,
        run_id="TEST-RUN",
        episode_id=episode_state["episode_id"],
        treatment="original",
    )
    tools.get_internal_vendor_record(
        "VENDOR-001",
        agent_id="internal_data_query",
        idempotency_key="read",
    )
    event = list(log.read())[0]
    assert event["state_before_hash"] != event["state_after_hash"]
    encoded = json.dumps(event, sort_keys=True).lower()
    assert "chain_of_thought" not in encoded
    assert "hidden_reasoning" not in encoded
