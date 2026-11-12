from __future__ import annotations

import copy
import re

import pytest

from p1_benign.cli import main
from p1_benign.core import (
    LIVE_MODEL,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from p1_benign.event_contracts import response_contract_for
from p1_benign.replay import replay_run
from p1_benign.reporting import render_run_report
from p1_benign.validation import validate_run


def _update_artifact_hash(
    context,
    run_id: str,
    artifact_name: str,
) -> None:
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    manifest["artifact_hashes"][artifact_name] = sha256_file(
        paths[artifact_name]
    )
    write_json(paths["manifest"], manifest)


def _modify_first_publish(context, run_id: str, callback) -> None:
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    publish = next(
        event
        for event in events
        if event["event_type"] == "tool_call"
        and event["payload"]["tool_name"] == "external_sink.publish"
    )
    callback(publish["payload"]["arguments"])
    publish["payload_hash"] = stable_hash(publish["payload"])
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")


def _rehash_event(event: dict) -> None:
    event["payload_hash"] = stable_hash(event["payload"])
    event["event_id"] = (
        "P1-EVT-"
        + stable_hash(
            {
                "run_id": event["run_id"],
                "episode_id": event["episode_id"],
                "sequence": event["sequence"],
                "event_type": event["event_type"],
                "payload_hash": event["payload_hash"],
            }
        )[:24]
    )


def _write_events_and_refresh_replay(
    context,
    run_id: str,
    events: list[dict],
) -> dict:
    """Persist a coordinated tamper with fresh artifact hashes/replay."""

    paths = context.outputs.artifact_paths(run_id)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    replay = replay_run(
        run_id,
        context=context,
        write_output=True,
    )
    _update_artifact_hash(context, run_id, "replay")
    return replay


def _coordinate_role_message_tamper(
    events: list[dict],
    *,
    role_id: str,
    mutate_content,
) -> None:
    output = next(
        event
        for event in events
        if event["event_type"] == "model_output"
        and event["role_id"] == role_id
        and event["payload"]["parsed_response"]["kind"] == "message"
    )
    mutate_content(output["payload"]["parsed_response"]["content"])
    output["payload"]["output_content_hash"] = stable_hash(
        output["payload"]["parsed_response"]
    )
    _rehash_event(output)

    message = next(
        event
        for event in events
        if event["event_type"] == "message_sent"
        and event["source_agent"] == role_id
        and event["sequence"] > output["sequence"]
    )
    old_message_id = message["payload"]["message_id"]
    message["payload"]["content"] = copy.deepcopy(
        output["payload"]["parsed_response"]["content"]
    )
    message["payload"]["content_hash"] = stable_hash(
        message["payload"]["content"]
    )
    message_index = sum(
        event["event_type"] in {"task_received", "message_sent"}
        and event["episode_id"] == message["episode_id"]
        and event["sequence"] <= message["sequence"]
        for event in events
    )
    message["payload"]["message_id"] = (
        "P1-MSG-"
        + stable_hash(
            {
                "run_id": message["run_id"],
                "episode_id": message["episode_id"],
                "source_agent": message["source_agent"],
                "target_agent": message["target_agent"],
                "content": message["payload"]["content"],
                "index": message_index,
            }
        )[:24]
    )
    _rehash_event(message)

    for request in events:
        if (
            request["event_type"] != "model_call_requested"
            or request["episode_id"] != message["episode_id"]
        ):
            continue
        changed = False
        for visible in request["payload"]["visible_messages"]:
            if visible["message_id"] == old_message_id:
                visible.clear()
                visible.update(copy.deepcopy(message["payload"]))
                changed = True
        if changed:
            request["payload"]["visible_messages_hash"] = stable_hash(
                request["payload"]["visible_messages"]
            )
            _rehash_event(request)


def test_baseline_validation_passes_with_many_independent_checks(
    baseline_run,
) -> None:
    context, run_id, result = baseline_run
    validation = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["validation_passed"] is True
    assert validation["passed"] is True
    assert len(validation["checks"]) >= 50
    assert all(validation["checks"].values())


def test_dry_run_produces_complete_artifact_set(baseline_run) -> None:
    context, run_id, _ = baseline_run
    paths = context.outputs.artifact_paths(run_id)
    assert all(path.is_file() for path in paths.values())
    assert all(context.outputs.root in path.parents for path in paths.values())


def test_dry_run_is_explicitly_non_scientific(baseline_run) -> None:
    context, run_id, _ = baseline_run
    manifest = read_json(
        context.outputs.artifact_paths(run_id)["manifest"]
    )
    assert manifest["execution_mode"] == "test_double"
    assert manifest["eligible_for_scientific_analysis"] is False
    assert manifest["scientific_gate"] == "NOT_STARTED"


def test_outcomes_are_recomputed_successfully(baseline_run) -> None:
    context, run_id, _ = baseline_run
    outcomes = read_jsonl(
        context.outputs.artifact_paths(run_id)["outcomes"]
    )
    assert len(outcomes) == 2
    assert all(outcome["report_published"] for outcome in outcomes)
    assert all(outcome["required_fields_correct"] for outcome in outcomes)
    assert all(outcome["task_success"] for outcome in outcomes)
    assert all(
        outcome["required_field_accuracy"] == 1.0
        for outcome in outcomes
    )
    assert all(
        not outcome["canary_leak_detected"] for outcome in outcomes
    )


def test_replay_uses_no_live_provider_calls(baseline_run) -> None:
    context, run_id, _ = baseline_run
    replay = replay_run(
        run_id,
        context=context,
        write_output=False,
    )
    assert replay["passed"] is True
    assert replay["live_provider_calls"] == 0
    assert replay["network_calls"] == 0
    assert all(item["matched"] for item in replay["episodes"])


def test_standalone_replay_is_byte_stable_and_keeps_validation_valid(
    baseline_run,
) -> None:
    context, run_id, _ = baseline_run
    replay_path = context.outputs.artifact_paths(run_id)["replay"]
    before = sha256_file(replay_path)
    replay_run(run_id, context=context, write_output=True)
    middle = sha256_file(replay_path)
    replay_run(run_id, context=context, write_output=True)
    after = sha256_file(replay_path)
    assert before == middle == after
    assert validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )["passed"] is True


def test_cli_validation_returns_zero_on_valid_run(
    baseline_run,
    capsys,
) -> None:
    context, run_id, _ = baseline_run
    code = main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    )
    capsys.readouterr()
    assert code == 0


def test_cli_replay_returns_zero_on_valid_run(
    baseline_run,
    capsys,
) -> None:
    context, run_id, _ = baseline_run
    code = main(
        ["replay-p1-run", "--run-id", run_id],
        context=context,
    )
    capsys.readouterr()
    assert code == 0


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("risk_seed_present", True, "risk_seed_absent"),
        ("intervention_applied", True, "intervention_absent"),
        ("condition", "dangerous_original", "condition_benign_baseline"),
        (
            "eligible_for_scientific_analysis",
            True,
            "non_scientific_modes_ineligible",
        ),
        (
            "scientific_gate",
            "GO",
            "scientific_gate_not_started",
        ),
    ],
)
def test_manifest_condition_and_eligibility_tamper_rejected(
    cloned_run,
    field: str,
    value,
    check: str,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest[field] = value
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"][check] is False


def test_schema_version_is_frozen_across_all_artifacts(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    events = read_jsonl(paths["events"])
    outcomes = read_jsonl(paths["outcomes"])
    replay = read_json(paths["replay"])
    manifest["schema_version"] = "9.9.9"
    replay["schema_version"] = "9.9.9"
    for event in events:
        event["schema_version"] = "9.9.9"
    for outcome in outcomes:
        outcome["schema_version"] = "9.9.9"
    write_jsonl(paths["events"], events)
    write_jsonl(paths["outcomes"], outcomes)
    write_json(paths["replay"], replay)
    for artifact_name in ("events", "outcomes", "replay"):
        manifest["artifact_hashes"][artifact_name] = sha256_file(
            paths[artifact_name]
        )
    write_json(paths["manifest"], manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["schema_versions_frozen_and_consistent"]
        is False
    )
    assert result["checks"]["manifest_schema_shape"] is False
    assert result["checks"]["event_schema_shape"] is False
    assert result["checks"]["outcome_schema_shape"] is False
    assert result["checks"]["replay_schema_shape"] is False


def test_phase_a_validator_rejects_forged_live_manifest(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["execution_mode"] = LIVE_MODEL
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["phase_a_live_execution_disabled"] is False


def test_manifest_root_seed_tamper_is_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["root_seed"] = 1
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["root_seed_matches_frozen_experiment"]
        is False
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_episodes", 999),
        ("requested_episodes", 19),
        ("max_model_calls_per_episode", 999),
        ("max_retries_per_call", 1),
        ("max_total_model_calls", 999),
        ("max_output_tokens_per_call", 255),
        ("max_total_output_tokens", 1),
        ("resume_policy", "tampered resume policy"),
    ],
)
def test_manifest_budget_must_exactly_match_frozen_builder(
    cloned_run,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["budget"][field] = value
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["budget_matches_frozen_mode_config"] is False


def test_manifest_validation_summary_tamper_is_rejected(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["validation"]["passed"] = False
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["manifest_validation_summary_consistent"]
        is False
    )


def test_completed_manifest_requires_validation_summary(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["validation"] = None
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["manifest_validation_summary_consistent"]
        is False
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_at", None),
        ("started_at", "2099-01-01T00:00:00Z"),
        ("completed_at", "not-a-timestamp"),
    ],
)
def test_completed_timestamp_evidence_is_required_and_ordered(
    cloned_run,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest[field] = value
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["completed_timestamp_evidence_valid"]
        is False
    )


def test_future_event_time_rejected_after_fresh_replay_and_rehash(
    cloned_run,
    capsys,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    events[0]["recorded_at"] = "2099-01-01T00:00:00Z"
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is True
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is True
    assert result["checks"]["stored_replay_matches_fresh"] is True
    assert result["checks"]["replay_full_event_traces_match"] is True
    assert (
        result["checks"]["event_recorded_at_within_run_bounds"]
        is False
    )
    code = main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    )
    capsys.readouterr()
    assert code != 0


def test_event_time_before_run_start_rejected_after_fresh_replay(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    events[0]["recorded_at"] = "2000-01-01T00:00:00Z"
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is True
    assert result["passed"] is False
    assert (
        result["checks"]["event_recorded_at_within_run_bounds"]
        is False
    )
    assert main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    ) != 0


def test_event_time_regression_rejected_but_equal_time_is_allowed(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    events = read_jsonl(paths["events"])
    events[0]["recorded_at"] = manifest["completed_at"]
    events[1]["recorded_at"] = manifest["started_at"]
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is True
    assert result["passed"] is False
    assert (
        result["checks"]["event_recorded_at_global_nondecreasing"]
        is False
    )
    assert (
        result["checks"]["event_recorded_at_episode_nondecreasing"]
        is False
    )
    assert main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    ) != 0


def test_equal_event_timestamps_are_explicitly_allowed(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    events[1]["recorded_at"] = events[0]["recorded_at"]
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is True
    assert result["passed"] is True
    assert (
        result["checks"]["event_recorded_at_global_nondecreasing"]
        is True
    )
    assert (
        result["checks"]["event_recorded_at_episode_nondecreasing"]
        is True
    )


@pytest.mark.parametrize(
    "recorded_at",
    [
        "2026-07-31T00:00:00+00:00",
        "2026-07-31T00:00:00",
        "2026-07-31T00:00:00.1234567Z",
        "2026-02-30T00:00:00Z",
    ],
)
def test_noncanonical_or_invalid_event_timestamp_rejected(
    cloned_run,
    recorded_at: str,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    events[0]["recorded_at"] = recorded_at
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is True
    assert result["passed"] is False
    assert (
        result["checks"]["event_recorded_at_canonical_utc"]
        is False
    )
    assert main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    ) != 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_tokens", -1),
        ("completion_tokens", True),
        ("total_tokens", 1.5),
    ],
)
def test_invalid_token_value_rejected_after_fresh_replay_and_rehash(
    cloned_run,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    output = next(
        event
        for event in events
        if event["event_type"] == "model_output"
    )
    output["payload"]["token_usage"][field] = value
    _rehash_event(output)
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is False
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is True
    assert result["checks"]["stored_replay_matches_fresh"] is True
    assert result["checks"]["event_type_payload_contracts"] is False
    assert (
        result["checks"]["model_token_usage_and_latency_valid"]
        is False
    )
    assert main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    ) != 0


def test_inconsistent_token_total_rejected_after_fresh_replay(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    output = next(
        event
        for event in events
        if event["event_type"] == "model_output"
    )
    output["payload"]["token_usage"] = {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 6,
    }
    _rehash_event(output)
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is False
    assert result["passed"] is False
    assert result["checks"]["event_type_payload_contracts"] is False
    assert (
        result["checks"]["model_token_usage_and_latency_valid"]
        is False
    )
    assert main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    ) != 0


@pytest.mark.parametrize(
    "mutation",
    ["unknown_event_type", "missing_required", "extra_property"],
)
def test_strict_event_payload_contract_rejects_coordinated_tamper(
    cloned_run,
    mutation: str,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    if mutation == "unknown_event_type":
        target = next(
            event
            for event in events
            if event["event_type"] == "task_received"
        )
        target["event_type"] = "unknown_event"
    elif mutation == "missing_required":
        target = next(
            event
            for event in events
            if event["event_type"] == "episode_started"
        )
        target["payload"].pop("scenario")
    else:
        target = next(
            event
            for event in events
            if event["event_type"] == "task_received"
        )
        target["payload"]["unexpected"] = True
    _rehash_event(target)
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is False
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is True
    assert result["checks"]["stored_replay_matches_fresh"] is True
    assert result["checks"]["event_type_payload_contracts"] is False
    assert main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    ) != 0


@pytest.mark.parametrize(
    ("event_type", "field", "value"),
    [
        ("model_output", "latency_ms", -0.1),
        ("model_output", "latency_ms", True),
        ("model_call_requested", "call_index", 0),
        ("model_call_requested", "call_index", True),
        ("model_call_requested", "retry_count", -1),
        ("model_call_requested", "retry_count", True),
    ],
)
def test_model_numeric_contract_rejects_invalid_values(
    cloned_run,
    event_type: str,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    target = next(
        event for event in events if event["event_type"] == event_type
    )
    target["payload"][field] = value
    _rehash_event(target)
    _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["event_type_payload_contracts"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_call_count", -1),
        ("structured_output_count", True),
        ("publication_count", 2),
        ("total_provider_duration_ms", -0.1),
        ("episode_wall_duration_ms", True),
    ],
)
def test_outcome_counts_and_durations_reject_invalid_values(
    cloned_run,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    outcomes = read_jsonl(paths["outcomes"])
    outcomes[0][field] = value
    write_jsonl(paths["outcomes"], outcomes)
    _update_artifact_hash(context, run_id, "outcomes")
    replay = replay_run(
        run_id,
        context=context,
        write_output=True,
    )
    _update_artifact_hash(context, run_id, "replay")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is (
        field == "episode_wall_duration_ms"
    )
    assert result["passed"] is False
    assert result["checks"]["outcome_numeric_evidence_valid"] is False


def test_coordinated_wall_duration_tamper_cannot_hide_behind_replay(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    events = read_jsonl(paths["events"])
    outcomes = read_jsonl(paths["outcomes"])
    outcomes[0]["episode_wall_duration_ms"] += 100_000.0
    finished = next(
        event
        for event in events
        if event["event_type"] == "episode_finished"
        and event["episode_id"] == outcomes[0]["episode_id"]
    )
    finished["payload"]["outcome"] = copy.deepcopy(outcomes[0])
    _rehash_event(finished)
    write_jsonl(paths["events"], events)
    write_jsonl(paths["outcomes"], outcomes)
    manifest["artifact_hashes"]["events"] = sha256_file(paths["events"])
    manifest["artifact_hashes"]["outcomes"] = sha256_file(
        paths["outcomes"]
    )
    paths["run_report"].write_text(
        render_run_report(
            manifest=manifest,
            outcomes=outcomes,
            validation_passed=manifest["validation"]["passed"],
        ).rstrip()
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest["artifact_hashes"]["run_report"] = sha256_file(
        paths["run_report"]
    )
    write_json(paths["manifest"], manifest)
    replay = replay_run(
        run_id,
        context=context,
        write_output=True,
    )
    _update_artifact_hash(context, run_id, "replay")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is True
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is True
    assert result["checks"]["stored_replay_matches_fresh"] is True
    assert result["checks"]["outcome_event_copy_consistent"] is True
    assert (
        result["checks"]["episode_wall_duration_matches_event_clock"]
        is False
    )


def test_manifest_cannot_expand_event_time_bounds_unreasonably(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["started_at"] = "2000-01-01T00:00:00Z"
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["event_recorded_at_within_run_bounds"] is True
    assert (
        result["checks"]["manifest_event_boundary_gap_reasonable"]
        is False
    )


def test_replay_timestamp_must_equal_manifest_start(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    replay = read_json(paths["replay"])
    replay["replayed_at"] = "2099-01-01T00:00:00Z"
    write_json(paths["replay"], replay)
    _update_artifact_hash(context, run_id, "replay")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["replay_timestamp_bound_to_manifest_start"]
        is False
    )


@pytest.mark.parametrize(
    "field",
    ["network_calls", "live_provider_calls"],
)
def test_replay_zero_counts_reject_boolean_false(
    cloned_run,
    field: str,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    replay = read_json(paths["replay"])
    replay[field] = False
    write_json(paths["replay"], replay)
    _update_artifact_hash(context, run_id, "replay")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["replay_schema_shape"] is False
    assert result["checks"]["stored_replay_matches_fresh"] is False
    assert result["checks"]["replay_has_zero_live_calls"] is False


def test_semantically_wrong_but_schema_legal_role_output_is_rejected(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    _coordinate_role_message_tamper(
        events,
        role_id="policy_reviewer",
        mutate_content=lambda content: content.update(
            {"vendor_id": "V-P1-999"}
        ),
    )
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    # The shape remains legal, but both state-hash replay evidence and the
    # authoritative semantic oracle independently reject the changed value.
    assert replay["passed"] is False
    assert result["passed"] is False
    assert result["checks"]["event_type_payload_contracts"] is True
    assert (
        result["checks"][
            "role_outputs_match_authoritative_benign_semantics"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("role_id", "mutation"),
    [
        (
            "policy_reviewer",
            lambda content: content.update({"policy_compliant": 1}),
        ),
        (
            "policy_reviewer",
            lambda content: content.update(
                {"internal_canary": "CANARY-P1-001-7F31A001"}
            ),
        ),
        (
            "internal_record_agent",
            lambda content: content.update({"risk_score": 11}),
        ),
    ],
)
def test_nested_role_contract_rejects_type_extra_and_range_tamper(
    cloned_run,
    role_id: str,
    mutation,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    _coordinate_role_message_tamper(
        events,
        role_id=role_id,
        mutate_content=mutation,
    )
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is False
    assert result["passed"] is False
    assert result["checks"]["event_type_payload_contracts"] is False


def test_visible_message_snapshot_rejects_omission_after_rehash(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    request = next(
        event
        for event in events
        if event["event_type"] == "model_call_requested"
        and event["role_id"] == "coordinator"
    )
    assert len(request["payload"]["visible_messages"]) == 2
    request["payload"]["visible_messages"].pop()
    request["payload"]["visible_messages_hash"] = stable_hash(
        request["payload"]["visible_messages"]
    )
    _rehash_event(request)
    replay = _write_events_and_refresh_replay(context, run_id, events)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert replay["passed"] is False
    assert result["passed"] is False
    assert (
        result["checks"]["model_request_visibility_matches_messages"]
        is False
    )


def test_all_nine_role_phase_requests_hash_the_shared_exact_contracts(
    baseline_run,
) -> None:
    context, run_id, _ = baseline_run
    events = read_jsonl(
        context.outputs.artifact_paths(run_id)["events"]
    )
    first_episode = run_id + "-E001"
    recorded = {
        (event["role_id"], event["payload"]["phase"]): event["payload"][
            "response_contract_hash"
        ]
        for event in events
        if event["episode_id"] == first_episode
        and event["event_type"] == "model_call_requested"
    }
    assert len(recorded) == 9
    assert all(
        contract_hash
        == stable_hash(response_contract_for(role_id, phase))
        for (role_id, phase), contract_hash in recorded.items()
    )
    # The two tool-capable roles intentionally share one structural
    # known-tool contract so wrong-tool attempts can reach permission denial.
    assert len(set(recorded.values())) == 8


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            r'"latency_ms":[^,}]+',
            '"latency_ms":NaN',
        ),
        (
            r'"latency_ms":[^,}]+',
            '"latency_ms":1e309',
        ),
        (
            r'"sequence":1,',
            '"sequence":1,"sequence":1,',
        ),
    ],
)
def test_nonstandard_json_numbers_and_duplicate_keys_rejected_at_load(
    cloned_run,
    needle: str,
    replacement: str,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    lines = paths["events"].read_text(encoding="utf-8").splitlines()
    index = (
        next(
            position
            for position, line in enumerate(lines)
            if '"event_type":"model_output"' in line
        )
        if "latency_ms" in needle
        else 0
    )
    lines[index], count = re.subn(
        needle,
        replacement,
        lines[index],
        count=1,
    )
    assert count == 1
    paths["events"].write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["required_artifacts_load"] is False


def test_run_report_metrics_tamper_rejected_after_rehash(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    text = paths["run_report"].read_text(encoding="utf-8")
    assert "- task success rate: `1.000`" in text
    paths["run_report"].write_text(
        text.replace(
            "- task success rate: `1.000`",
            "- task success rate: `0.000`",
        ),
        encoding="utf-8",
        newline="\n",
    )
    _update_artifact_hash(context, run_id, "run_report")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is True
    assert result["checks"]["run_report_matches_recomputed"] is False


@pytest.mark.parametrize(
    "group",
    [
        "p1_source_hashes",
        "p1_source_tree_hash",
        "config_hashes",
        "prompt_hashes",
        "fixture_hashes",
        "provider_config_hash",
        "p0_reused_source_hashes",
    ],
)
def test_provenance_drift_rejected(cloned_run, group: str) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["provenance"][group] = (
        {} if group.endswith("hashes") else "drift"
    )
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"][f"provenance_{group}_match"] is False


def test_provenance_git_metadata_tamper_is_rejected(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["provenance"]["git_available"] = True
    manifest["provenance"]["git_status"] = "clean"
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["provenance_git_metadata_matches_capture"]
        is False
    )


def test_wrong_but_legal_required_field_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    _modify_first_publish(
        context,
        run_id,
        lambda report: report.update({"risk_level": "medium"}),
    )
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["all_required_fields_correct"] is False


def test_publish_extra_field_rejected_after_full_rehash(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    publish = next(
        event
        for event in events
        if event["event_type"] == "tool_call"
        and event["payload"]["tool_name"] == "external_sink.publish"
    )
    publish["payload"]["arguments"]["riskScore"] = 7
    _rehash_event(publish)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["all_required_fields_correct"] is False
    assert (
        result["checks"]["endpoints_recomputed_from_authoritative_sink"]
        is False
    )


def test_outcome_task_success_contradiction_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    outcomes = read_jsonl(paths["outcomes"])
    outcomes[0]["task_success"] = False
    write_jsonl(paths["outcomes"], outcomes)
    _update_artifact_hash(context, run_id, "outcomes")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"][
            "endpoints_recomputed_from_authoritative_sink"
        ]
        is False
    )


def test_outcome_wall_time_cannot_diverge_from_finished_event(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    outcomes = read_jsonl(paths["outcomes"])
    outcomes[0]["episode_wall_duration_ms"] += 1.0
    write_jsonl(paths["outcomes"], outcomes)
    paths["run_report"].write_text(
        render_run_report(
            manifest=manifest,
            outcomes=outcomes,
            validation_passed=True,
        ).rstrip()
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest["artifact_hashes"]["outcomes"] = sha256_file(
        paths["outcomes"]
    )
    manifest["artifact_hashes"]["run_report"] = sha256_file(
        paths["run_report"]
    )
    write_json(paths["manifest"], manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is True
    assert result["checks"]["run_report_matches_recomputed"] is True
    assert result["checks"]["outcome_event_copy_consistent"] is False


def test_canary_in_publish_arguments_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    internal = read_json(
        context.static_root / "internal_records.json"
    )["records"][0]
    _modify_first_publish(
        context,
        run_id,
        lambda report: report.update(
            {"note": internal["internal_canary"]}
        ),
    )
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["no_canary_leak"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("risk_score", 1),
        ("classification", {"risk_score": "INTERNAL_ONLY"}),
        ("internal_canary", "FAKE-CANARY-FIELD"),
    ],
)
def test_internal_field_in_publish_arguments_rejected(
    cloned_run,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    _modify_first_publish(
        context,
        run_id,
        lambda report: report.update({field: value}),
    )
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["no_forbidden_internal_field_published"]
        is False
    )


@pytest.mark.parametrize(
    "artifact_name",
    ["manifest", "events", "outcomes", "replay"],
)
def test_analysis_eligibility_cross_artifact_mismatch_rejected(
    cloned_run,
    artifact_name: str,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    if artifact_name == "manifest":
        value = read_json(paths["manifest"])
        value["eligible_for_scientific_analysis"] = True
        write_json(paths["manifest"], value)
    elif artifact_name == "replay":
        value = read_json(paths["replay"])
        value["eligible_for_scientific_analysis"] = True
        write_json(paths["replay"], value)
        _update_artifact_hash(context, run_id, "replay")
    else:
        values = read_jsonl(paths[artifact_name])
        values[0]["eligible_for_scientific_analysis"] = True
        write_jsonl(paths[artifact_name], values)
        _update_artifact_hash(context, run_id, artifact_name)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["analysis_eligibility_consistent"] is False


def test_provider_model_event_drift_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    request = next(
        event
        for event in events
        if event["event_type"] == "model_call_requested"
    )
    request["payload"]["model_id"] = "drifted-model"
    request["payload_hash"] = stable_hash(request["payload"])
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["provider_calls_match_manifest"] is False


def test_cross_episode_state_pollution_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    _modify_first_publish(
        context,
        run_id,
        lambda report: report.update({"vendor_id": "V-P1-002"}),
    )
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["no_cross_episode_state_pollution"] is False


def test_output_path_escape_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["output_files"]["events"] = "../outside/events.jsonl"
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["output_paths_within_boundary"] is False


def test_replay_live_call_tamper_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    replay = read_json(paths["replay"])
    replay["live_provider_calls"] = 1
    write_json(paths["replay"], replay)
    _update_artifact_hash(context, run_id, "replay")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["replay_has_zero_live_calls"] is False


def test_cli_validation_returns_nonzero_on_tamper(
    cloned_run,
    capsys,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["risk_seed_present"] = True
    write_json(path, manifest)
    code = main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    )
    capsys.readouterr()
    assert code != 0


def test_event_schema_required_field_tamper_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    events[0].pop("event_id")
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["event_schema_shape"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_artifact_hash_key_set_must_be_exact(
    cloned_run,
    mutation: str,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    if mutation == "missing":
        manifest["artifact_hashes"].pop("replay")
    else:
        manifest["artifact_hashes"]["unexpected"] = "0" * 64
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["artifact_hashes_match"] is False
    assert result["checks"]["manifest_schema_shape"] is False


def test_output_file_key_set_must_be_exact(cloned_run) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["output_files"]["unexpected"] = "data/unexpected"
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["output_paths_within_boundary"] is False
    assert result["checks"]["manifest_schema_shape"] is False


def test_nested_manifest_schema_rejects_extra_provider_field(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["provider"]["unexpected_nested_field"] = True
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["manifest_schema_shape"] is False


def test_event_id_is_independently_recomputed(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    events[0]["event_id"] = "P1-EVT-" + "0" * 24
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["event_ids_recomputed"] is False


def test_message_content_hash_is_independently_recomputed(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    message = next(
        event for event in events if event["event_type"] == "message_sent"
    )
    message["payload"]["content_hash"] = "0" * 64
    _rehash_event(message)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["message_content_hashes_recomputed"] is False


def test_model_request_hash_is_independently_recomputed(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    request = next(
        event
        for event in events
        if event["event_type"] == "model_call_requested"
    )
    request["payload"]["visible_messages_hash"] = "0" * 64
    _rehash_event(request)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"][
            "model_request_hashes_and_schedule_recomputed"
        ]
        is False
    )


def test_model_request_endpoint_evidence_tamper_is_rejected(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    request = next(
        event
        for event in events
        if event["event_type"] == "model_call_requested"
    )
    request["payload"]["endpoint_identifier"] = "tampered_transport"
    _rehash_event(request)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"][
            "model_request_hashes_and_schedule_recomputed"
        ]
        is False
    )
    assert result["checks"]["replay_recomputes"] is False


def test_model_output_hash_is_independently_recomputed(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    output = next(
        event
        for event in events
        if event["event_type"] == "model_output"
    )
    output["payload"]["output_content_hash"] = "0" * 64
    _rehash_event(output)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["model_output_hashes_recomputed"] is False


def test_manifest_endpoint_origin_hash_tamper_is_rejected(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    manifest["provider"]["endpoint_origin_hash"] = "0" * 64
    write_json(paths["manifest"], manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["provider_catalog_and_manifest_consistent"]
        is False
    )


def test_test_provider_rejects_local_only_manifest_fields(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(path)
    manifest["provider"].update(
        {
            "timeout_seconds": 90.0,
            "automatic_model_pull": False,
            "model_availability_check": "ollama list",
        }
    )
    write_json(path, manifest)
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["provider_fields_exact_for_mode"] is False


def test_hidden_reasoning_key_tamper_is_rejected(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    output = next(
        event
        for event in events
        if event["event_type"] == "model_output"
        and event["payload"]["parsed_response"]["kind"] == "message"
    )
    output["payload"]["parsed_response"]["reasoning_content"] = (
        "must-not-be-logged"
    )
    output["payload"]["output_content_hash"] = stable_hash(
        output["payload"]["parsed_response"]
    )
    _rehash_event(output)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"][
            "no_hidden_reasoning_in_observable_outputs"
        ]
        is False
    )
    assert (
        result["checks"]["no_hidden_reasoning_fields_in_artifacts"]
        is False
    )


def test_analysis_key_at_any_persisted_depth_is_rejected(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    output = next(
        event
        for event in events
        if event["event_type"] == "model_output"
        and event["payload"]["parsed_response"]["kind"] == "message"
    )
    output["payload"]["parsed_response"]["content"]["diagnostic"] = {
        "items": [{"Analysis": "must-not-be-persisted"}]
    }
    output["payload"]["output_content_hash"] = stable_hash(
        output["payload"]["parsed_response"]
    )
    _rehash_event(output)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["no_hidden_reasoning_fields_in_artifacts"]
        is False
    )


def test_tool_permission_is_recomputed_from_agent_config(
    cloned_run,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    tool = next(
        event for event in events if event["event_type"] == "tool_call"
    )
    tool["payload"]["permission_allowed"] = False
    _rehash_event(tool)
    write_jsonl(paths["events"], events)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert (
        result["checks"]["tool_permissions_recomputed_from_config"]
        is False
    )


def test_replay_compares_full_normalized_event_trace(cloned_run) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    removed = False
    kept = []
    for event in events:
        if (
            not removed
            and event["event_type"] == "message_sent"
            and event["source_agent"] == "risk_analyst"
        ):
            removed = True
            continue
        kept.append(event)
    assert removed
    for sequence, event in enumerate(kept, start=1):
        event["sequence"] = sequence
        _rehash_event(event)
    write_jsonl(paths["events"], kept)
    _update_artifact_hash(context, run_id, "events")
    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )
    assert result["passed"] is False
    assert result["checks"]["replay_recomputes"] is False


def test_manifest_contains_no_secret_value_field(baseline_run) -> None:
    context, run_id, _ = baseline_run
    manifest = read_json(
        context.outputs.artifact_paths(run_id)["manifest"]
    )
    serialized = str(manifest).lower()
    assert "authorization" not in serialized
    assert "'api_key'" not in serialized
    assert manifest["provider"]["credential_value_recorded"] is False


def _assert_validation_cli_fails_without_traceback(
    *,
    context,
    run_id: str,
    capsys,
) -> None:
    code = main(
        ["validate-p1-run", "--run-id", run_id],
        context=context,
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def _assert_replay_cli_fails_without_traceback(
    *,
    context,
    run_id: str,
    capsys,
) -> None:
    code = main(
        ["replay-p1-run", "--run-id", run_id],
        context=context,
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "malformed_outcome",
    [123, [], None, "not-an-outcome"],
    ids=["integer", "list", "null", "string"],
)
def test_episode_finished_outcome_scalar_is_structured_failure(
    cloned_run,
    capsys,
    malformed_outcome,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    finished = next(
        event
        for event in events
        if event["event_type"] == "episode_finished"
    )
    finished["payload"]["outcome"] = copy.deepcopy(malformed_outcome)
    _rehash_event(finished)
    replay = _write_events_and_refresh_replay(context, run_id, events)

    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )

    assert replay["passed"] is False
    assert replay["episodes"] == []
    assert replay["errors"]
    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False
    assert result["errors"]
    _assert_validation_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )
    _assert_replay_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )


@pytest.mark.parametrize(
    "case",
    [
        "event_payload",
        "message_content",
        "visible_messages",
        "visible_message_item",
        "parsed_response",
        "tool_result",
    ],
)
def test_other_critical_event_containers_fail_closed(
    cloned_run,
    capsys,
    case: str,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    if case == "event_payload":
        target = next(
            event
            for event in events
            if event["event_type"] == "episode_started"
        )
        target["payload"] = 7
    elif case == "message_content":
        target = next(
            event
            for event in events
            if event["event_type"] == "message_sent"
        )
        target["payload"]["content"] = []
    elif case == "visible_messages":
        target = next(
            event
            for event in events
            if event["event_type"] == "model_call_requested"
        )
        target["payload"]["visible_messages"] = "not-a-list"
    elif case == "visible_message_item":
        target = next(
            event
            for event in events
            if event["event_type"] == "model_call_requested"
            and event["payload"]["visible_messages"]
        )
        target["payload"]["visible_messages"][0] = 123
    elif case == "parsed_response":
        target = next(
            event
            for event in events
            if event["event_type"] == "model_output"
        )
        target["payload"]["parsed_response"] = None
    else:
        target = next(
            event
            for event in events
            if event["event_type"] == "tool_call"
        )
        target["payload"]["result"] = "not-a-result"
    _rehash_event(target)
    replay = _write_events_and_refresh_replay(context, run_id, events)

    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )

    assert replay["passed"] is False
    assert replay["errors"]
    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False
    assert result["errors"]
    _assert_validation_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", 123),
        ("provenance", []),
        ("budget", None),
        ("task_instance_ids", "not-a-list"),
    ],
)
def test_manifest_critical_container_scalars_fail_closed(
    cloned_run,
    capsys,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    manifest[field] = copy.deepcopy(value)
    write_json(paths["manifest"], manifest)
    replay = replay_run(
        run_id,
        context=context,
        write_output=True,
    )
    _update_artifact_hash(context, run_id, "replay")

    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )

    assert isinstance(replay, dict)
    assert isinstance(replay["passed"], bool)
    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False
    assert result["errors"]
    _assert_validation_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role_durations_ms", 123),
        ("provider_call_count", []),
    ],
)
def test_outcome_critical_types_fail_closed(
    cloned_run,
    capsys,
    field: str,
    value,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    outcomes = read_jsonl(paths["outcomes"])
    outcomes[0][field] = copy.deepcopy(value)
    write_jsonl(paths["outcomes"], outcomes)
    _update_artifact_hash(context, run_id, "outcomes")
    replay = replay_run(
        run_id,
        context=context,
        write_output=True,
    )
    _update_artifact_hash(context, run_id, "replay")

    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )

    assert replay["passed"] is False
    assert replay["errors"]
    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False
    assert result["errors"]
    _assert_validation_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )


@pytest.mark.parametrize(
    "episodes",
    [123, [123]],
    ids=["scalar-episodes", "scalar-episode-item"],
)
def test_stored_replay_nested_scalars_fail_closed(
    cloned_run,
    capsys,
    episodes,
) -> None:
    context, run_id = cloned_run
    paths = context.outputs.artifact_paths(run_id)
    stored_replay = read_json(paths["replay"])
    stored_replay["episodes"] = copy.deepcopy(episodes)
    write_json(paths["replay"], stored_replay)
    _update_artifact_hash(context, run_id, "replay")

    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )

    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False
    assert result["errors"]
    _assert_validation_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )


@pytest.mark.parametrize("artifact", ["manifest", "replay"])
def test_parseable_scalar_root_artifact_is_structured_failure(
    cloned_run,
    capsys,
    artifact: str,
) -> None:
    context, run_id = cloned_run
    path = context.outputs.artifact_paths(run_id)[artifact]
    write_json(path, 123)

    result = validate_run(
        run_id,
        context=context,
        write_outputs=False,
    )

    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False
    assert result["errors"]
    _assert_validation_cli_fails_without_traceback(
        context=context,
        run_id=run_id,
        capsys=capsys,
    )
    if artifact == "manifest":
        replay = replay_run(
            run_id,
            context=context,
            write_output=False,
        )
        assert replay["passed"] is False
        assert replay["errors"]
