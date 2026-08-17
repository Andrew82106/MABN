from __future__ import annotations

import pytest

from pre_exp1.mock_tools import (
    IdempotencyConflict,
    LocalMockTools,
)
from pre_exp1.replay import replay_episode
from pre_exp1.sandbox import run_scripted_episode
from pre_exp1.state import stable_hash


def _dangerous_episode(event_log_factory):
    log = event_log_factory("REPLAY-RUN")
    state, outcome = run_scripted_episode(
        run_id="REPLAY-RUN",
        episode_id="REPLAY-EP",
        scenario="dangerous_original",
        treatment="original",
        event_log=log,
    )
    return log, state, outcome


def test_transcript_replay_matches_original_final_state(event_log_factory):
    log, state, outcome = _dangerous_episode(event_log_factory)
    result = replay_episode(list(log.read()), "REPLAY-EP")
    assert result["matched"] is True
    assert result["actual_final_state_hash"] == outcome.final_state_hash
    assert stable_hash(state) == outcome.final_state_hash


def test_replay_does_not_repeat_publication_side_effect(event_log_factory):
    log, state, _ = _dangerous_episode(event_log_factory)
    original_count = len(state["tool_state"]["publications"])
    result = replay_episode(list(log.read()), "REPLAY-EP")
    assert original_count == 1
    assert len(state["tool_state"]["publications"]) == 1
    assert len(result["replayed_state"]["tool_state"]["publications"]) == 1


def test_replay_never_calls_live_mock_tools(event_log_factory, monkeypatch):
    log, _, _ = _dangerous_episode(event_log_factory)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Replay called a live mock tool")

    monkeypatch.setattr(
        LocalMockTools,
        "get_internal_vendor_record",
        fail_if_called,
    )
    monkeypatch.setattr(
        LocalMockTools,
        "publish_public_report",
        fail_if_called,
    )
    assert replay_episode(list(log.read()), "REPLAY-EP")["matched"] is True


def test_tool_idempotency_keys_prevent_duplicate_state_changes(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory("IDEMPOTENCY-RUN")
    tools = LocalMockTools(
        state=episode_state,
        permissions=permissions,
        event_log=log,
        run_id="IDEMPOTENCY-RUN",
        episode_id=episode_state["episode_id"],
        treatment="original",
    )
    report = {"vendor_id": "VENDOR-001"}
    first = tools.publish_public_report(
        report,
        agent_id="report_publisher",
        idempotency_key="same-key",
    )
    hash_after_first = stable_hash(episode_state)
    second = tools.publish_public_report(
        report,
        agent_id="report_publisher",
        idempotency_key="same-key",
    )
    assert first["status"] == "published_locally"
    assert second["status"] == "already_published"
    assert stable_hash(episode_state) == hash_after_first
    assert len(episode_state["tool_state"]["publications"]) == 1


def test_idempotency_key_reuse_with_different_payload_is_rejected(
    episode_state, permissions, event_log_factory
):
    log = event_log_factory("IDEMPOTENCY-CONFLICT")
    tools = LocalMockTools(
        state=episode_state,
        permissions=permissions,
        event_log=log,
        run_id="IDEMPOTENCY-CONFLICT",
        episode_id=episode_state["episode_id"],
        treatment="original",
    )
    tools.publish_public_report(
        {"vendor_id": "VENDOR-001"},
        agent_id="report_publisher",
        idempotency_key="conflict-key",
    )
    with pytest.raises(IdempotencyConflict):
        tools.publish_public_report(
            {"vendor_id": "DIFFERENT"},
            agent_id="report_publisher",
            idempotency_key="conflict-key",
        )
    assert len(episode_state["tool_state"]["publications"]) == 1
    assert list(log.read())[-1]["event_type"] == "tool_idempotency_conflict"
