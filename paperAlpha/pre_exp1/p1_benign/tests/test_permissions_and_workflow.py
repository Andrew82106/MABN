from __future__ import annotations

from pathlib import Path

import pytest

from p1_benign.configuration import load_static_inputs
from p1_benign.core import (
    DEFAULT_CONTEXT,
    EventRecorder,
    require_generated_path,
)
from p1_benign.workflow import (
    EpisodeState,
    PermissionDenied,
    ToolDispatcher,
    ToolFailure,
    permission_map_from_static,
)


def _dispatcher():
    static = load_static_inputs(DEFAULT_CONTEXT)
    task_id = static.task_ids[0]
    record = static.internal_records[task_id]
    state = EpisodeState(
        episode_id="P1-BENIGN-DRY-PERM-E001",
        task_instance_id=task_id,
        vendor_id=record["vendor_id"],
        internal_database={record["vendor_id"]: record},
        sandbox_instance_id="P1-SBX-PERMISSION",
    )
    recorder = EventRecorder(
        run_id="P1-BENIGN-DRY-PERM",
        execution_mode="test_double",
        eligible_for_scientific_analysis=False,
    )
    dispatcher = ToolDispatcher(
        permission_map=permission_map_from_static(static),
        state=state,
        recorder=recorder,
    )
    return dispatcher, state, recorder


def test_internal_role_can_read_database() -> None:
    dispatcher, state, _ = _dispatcher()
    result = dispatcher.dispatch(
        role_id="internal_record_agent",
        tool_name="internal_db.read",
        arguments={"vendor_id": state.vendor_id},
        idempotency_key="read-1",
    )
    assert result["vendor_id"] == state.vendor_id


def test_publisher_can_publish_local_sink() -> None:
    dispatcher, state, _ = _dispatcher()
    result = dispatcher.dispatch(
        role_id="report_publisher",
        tool_name="external_sink.publish",
        arguments={
            "vendor_id": state.vendor_id,
            "registration_status": "active",
            "risk_level": "low",
            "recommendation": "approve",
        },
        idempotency_key="publish-1",
    )
    assert result["status"] == "published_locally"


def test_publisher_rejects_any_field_outside_public_allowlist() -> None:
    dispatcher, state, recorder = _dispatcher()
    with pytest.raises(ToolFailure):
        dispatcher.dispatch(
            role_id="report_publisher",
            tool_name="external_sink.publish",
            arguments={
                "vendor_id": state.vendor_id,
                "registration_status": "active",
                "risk_level": "low",
                "recommendation": "approve",
                "riskScore": 7,
            },
            idempotency_key="publish-extra-field",
        )
    assert state.publications == {}
    assert recorder.events[-1]["payload"]["result"]["status"] == "failed"


@pytest.mark.parametrize(
    "role_id",
    [
        "intake",
        "dossier_extractor",
        "independent_verifier",
        "coordinator",
        "risk_analyst",
        "policy_reviewer",
        "report_publisher",
    ],
)
def test_only_internal_role_can_read(role_id: str) -> None:
    dispatcher, state, recorder = _dispatcher()
    with pytest.raises(PermissionDenied):
        dispatcher.dispatch(
            role_id=role_id,
            tool_name="internal_db.read",
            arguments={"vendor_id": state.vendor_id},
            idempotency_key=f"read-{role_id}",
        )
    assert recorder.events[-1]["payload"]["permission_allowed"] is False
    assert recorder.events[-1]["payload"]["state_before_hash"] == (
        recorder.events[-1]["payload"]["state_after_hash"]
    )


@pytest.mark.parametrize(
    "role_id",
    [
        "intake",
        "dossier_extractor",
        "independent_verifier",
        "coordinator",
        "risk_analyst",
        "policy_reviewer",
        "internal_record_agent",
    ],
)
def test_only_publisher_can_publish(role_id: str) -> None:
    dispatcher, state, recorder = _dispatcher()
    with pytest.raises(PermissionDenied):
        dispatcher.dispatch(
            role_id=role_id,
            tool_name="external_sink.publish",
            arguments={"vendor_id": state.vendor_id},
            idempotency_key=f"publish-{role_id}",
        )
    assert recorder.events[-1]["payload"]["permission_allowed"] is False


def test_tool_idempotency_does_not_duplicate_publication() -> None:
    dispatcher, state, _ = _dispatcher()
    arguments = {
        "vendor_id": state.vendor_id,
        "registration_status": "active",
        "risk_level": "low",
        "recommendation": "approve",
    }
    dispatcher.dispatch(
        role_id="report_publisher",
        tool_name="external_sink.publish",
        arguments=arguments,
        idempotency_key="publish-same",
    )
    second = dispatcher.dispatch(
        role_id="report_publisher",
        tool_name="external_sink.publish",
        arguments=arguments,
        idempotency_key="publish-same",
    )
    assert second["status"] == "already_published"
    assert len(state.publications) == 1


def test_baseline_workflow_outcomes_are_programmatically_correct(
    baseline_run,
) -> None:
    context, run_id, result = baseline_run
    assert result["passed"] is True
    outcomes = context.outputs.artifact_paths(run_id)["outcomes"].read_text(
        encoding="utf-8"
    )
    assert '"task_success":true' in outcomes
    assert '"canary_leak_detected":false' in outcomes


def test_message_visibility_matches_fixed_graph(baseline_run) -> None:
    context, run_id, _ = baseline_run
    import json

    events = [
        json.loads(line)
        for line in context.outputs.artifact_paths(run_id)[
            "events"
        ].read_text(encoding="utf-8").splitlines()
    ]
    requests = [
        event
        for event in events
        if event["episode_id"].endswith("E001")
        and event["event_type"] == "model_call_requested"
    ]
    by_role_phase = {
        (event["role_id"], event["payload"]["phase"]): {
            message["source_agent"]
            for message in event["payload"]["visible_messages"]
        }
        for event in requests
    }
    assert by_role_phase[("coordinator", "produce_message")] == {
        "dossier_extractor",
        "independent_verifier",
    }
    assert by_role_phase[("internal_record_agent", "request_tool")] == {
        "coordinator"
    }
    assert by_role_phase[
        ("internal_record_agent", "summarize_tool_result")
    ] == {"coordinator", "internal_db.read"}
    assert by_role_phase[("report_publisher", "publish")] == {
        "risk_analyst",
        "policy_reviewer",
        "internal_record_agent",
    }


def test_logs_never_record_hidden_reasoning(baseline_run) -> None:
    context, run_id, _ = baseline_run
    import json

    events = [
        json.loads(line)
        for line in context.outputs.artifact_paths(run_id)[
            "events"
        ].read_text(encoding="utf-8").splitlines()
    ]
    outputs = [
        event
        for event in events
        if event["event_type"] == "model_output"
    ]
    assert outputs
    assert all(
        event["payload"]["hidden_reasoning_logged"] is False
        for event in outputs
    )
    assert all("raw_response" not in event["payload"] for event in outputs)


def test_episode_sandboxes_are_distinct(baseline_run) -> None:
    context, run_id, _ = baseline_run
    import json

    events = [
        json.loads(line)
        for line in context.outputs.artifact_paths(run_id)[
            "events"
        ].read_text(encoding="utf-8").splitlines()
    ]
    starts = [
        event for event in events if event["event_type"] == "episode_started"
    ]
    assert len(starts) == 2
    assert len(
        {
            event["payload"]["sandbox_instance_id"]
            for event in starts
        }
    ) == 2
    assert all(
        len(event["payload"]["internal_vendor_ids"]) == 1
        for event in starts
    )


def test_generated_path_outside_data_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        require_generated_path(
            tmp_path / "outside.json",
            tmp_path / "paperAlpha" / "data",
        )
