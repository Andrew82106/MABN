"""The Q0-only readiness runner; it has no live-provider capability."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .errors import ContractError, FakeProviderError, FakeProviderTimeout, QualificationError
from .fake_provider import DeterministicFakeProvider
from .fixtures import load_fixture_sets
from .paths import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    append_jsonl,
    assert_execution_roots,
    artifact_paths,
    canonical_json,
    assert_workspace_root,
    safe_write_text,
    sha256_bytes,
    sha256_file,
)
from .protocol import (
    can_write_public_sink,
    load_permission_rule,
    load_protocol_config,
    load_schema,
    render_prompt,
    validate_response,
)
from .provenance import (
    capture_protected_tree_hashes,
    collect_code_provenance,
    collect_static_input_hashes,
    load_reference_identity,
    runtime_identity,
)


TERMINAL_STATES = frozenset({"model_output", "model_output_rejected", "model_call_failed"})


def _identity_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_kind": config["artifact_kind"],
        "phase": config["phase"],
        "data_role": config["data_role"],
        "execution_mode": config["execution_mode"],
        **config["identity"],
    }


def _assert_output_is_new(paths: dict[str, Path], data_root: Path) -> None:
    for name in ("events", "transcripts", "outcomes", "manifest", "public_sink", "validation", "replay", "report"):
        path = paths[name]
        try:
            path.relative_to((data_root / "readiness_dry").resolve(strict=False))
        except ValueError as exc:
            raise QualificationError("output_path_invalid", "derived output path is outside Q0 data root") from exc
        if path.exists():
            raise QualificationError("run_already_exists", "the requested Q0 run ID already has artifacts")


def _write_new_json(path: Path, root: Path, value: dict[str, Any]) -> None:
    safe_write_text(path, root, canonical_json(value) + "\n")


def _event_base(identity: dict[str, Any], run_id: str, request_id: str, task: dict[str, Any], case_id: str, event_index: int) -> dict[str, Any]:
    return {
        **identity,
        "run_id": run_id,
        "request_id": request_id,
        "event_index": event_index,
        "task_id": task["task_id"],
        "stage": task["stage"],
        "role": task["role"],
        "case_id": case_id,
        "attempt": 1,
        "retry_count": 0,
        "provider_kind": "deterministic_fake_provider",
    }


def _process_request(
    provider: DeterministicFakeProvider,
    task: dict[str, Any],
    case_id: str,
    case_kind: str,
    request_id: str,
    event_index: int,
    identity: dict[str, Any],
    run_id: str,
    schema: dict[str, Any],
    permission_rule: dict[str, Any],
    code_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str] | None]:
    rendered_prompt = render_prompt(code_root, task)
    prompt_sha256 = sha256_bytes(rendered_prompt.encode("utf-8"))
    event = _event_base(identity, run_id, request_id, task, case_id, event_index)
    transcript: dict[str, Any] = {
        **identity,
        "run_id": run_id,
        "request_id": request_id,
        "task_id": task["task_id"],
        "case_id": case_id,
        "prompt_sha256": prompt_sha256,
        "provider_kind": "deterministic_fake_provider",
    }
    try:
        reply = provider.invoke(task, rendered_prompt, case_kind)
        transcript["raw_output"] = reply.raw_output
        transcript["raw_output_sha256"] = sha256_bytes(reply.raw_output.encode("utf-8"))
        try:
            report = validate_response(reply.raw_output, task, schema)
        except QualificationError as exc:
            event.update({"terminal_state": "model_output_rejected", "failure_code": exc.code})
            transcript.update({"terminal_state": "model_output_rejected", "failure_code": exc.code})
            return event, transcript, None
        event["terminal_state"] = "model_output"
        transcript["terminal_state"] = "model_output"
        return event, transcript, report if can_write_public_sink(task, report, permission_rule) else None
    except (FakeProviderError, FakeProviderTimeout) as exc:
        event.update({"terminal_state": "model_call_failed", "failure_code": exc.code})
        transcript.update({"terminal_state": "model_call_failed", "failure_code": exc.code})
        return event, transcript, None


def _build_outcomes(identity: dict[str, Any], run_id: str, events: list[dict[str, Any]], fake_calls: int, sink_records: int) -> dict[str, Any]:
    state_counts = Counter(event["terminal_state"] for event in events)
    task_counts = Counter(
        f"{event['stage']}:{event['role']}"
        for event in events
        if event["case_id"] == "valid" and event["terminal_state"] == "model_output"
    )
    negative_counts = Counter(
        event["terminal_state"] for event in events if event["case_id"] != "valid"
    )
    negative_case_counts: dict[str, Counter[str]] = {}
    for event in events:
        if event["case_id"] == "valid":
            continue
        negative_case_counts.setdefault(event["case_id"], Counter())[event["terminal_state"]] += 1
    return {
        **identity,
        "run_id": run_id,
        "request_count": len(events),
        "terminal_state_counts": dict(sorted(state_counts.items())),
        "valid_task_counts": dict(sorted(task_counts.items())),
        "negative_terminal_state_counts": dict(sorted(negative_counts.items())),
        "negative_case_terminal_counts": {
            case_id: dict(sorted(counts.items()))
            for case_id, counts in sorted(negative_case_counts.items())
        },
        "fake_provider_calls": fake_calls,
        "public_sink_records": sink_records,
        "q1_live_model_calls_authorized": False,
        "p1_p2_status": "locked",
    }


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    root = paths["root"]
    return {
        name: sha256_file(paths[name], root)
        for name in ("events", "transcripts", "outcomes", "public_sink")
    }


def run_readiness_dry(
    *,
    code_root: Path = PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    workspace_root: Path | None = None,
    run_id: str | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Create one immutable Q0 dry-run artifact without any model or network call."""
    code_root = code_root.resolve(strict=False)
    data_root = data_root.resolve(strict=False)
    workspace_root = (workspace_root or code_root.parents[2]).resolve(strict=False)
    assert_execution_roots(code_root, data_root, test_mode=test_mode)
    assert_workspace_root(workspace_root)
    config = load_protocol_config(code_root)
    selected_run_id = run_id or config["fixed_q0_run_id"]
    if selected_run_id != config["fixed_q0_run_id"]:
        raise QualificationError("run_id_mismatch", "Q0 only permits the frozen run ID")
    paths = artifact_paths(data_root, selected_run_id)
    _assert_output_is_new(paths, data_root)
    schema = load_schema(code_root)
    permission_rule = load_permission_rule(code_root)
    screen, confirmation, negative_cases = load_fixture_sets(code_root)
    tasks = screen + confirmation
    expected_request_count = len(tasks) + len(negative_cases)
    # The runner consumes only its own frozen source-identity copy.  Validation
    # independently compares those identities with the four authorized source
    # files; no historical source file is used as a runtime task input here.
    source_identity = load_reference_identity(code_root)
    protected_before = capture_protected_tree_hashes(workspace_root)
    identity = _identity_fields(config)
    provider = DeterministicFakeProvider()
    events: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    sink_records: list[dict[str, str]] = []

    request_index = 0
    for task in tasks:
        request_index += 1
        event, transcript, sink_record = _process_request(
            provider,
            task,
            "valid",
            "valid",
            f"Q0-REQ-{request_index:03d}",
            request_index,
            identity,
            selected_run_id,
            schema,
            permission_rule,
            code_root,
        )
        events.append(event)
        transcripts.append(transcript)
        if sink_record is not None:
            sink_records.append(sink_record)

    for negative_index, negative_case in enumerate(negative_cases):
        request_index += 1
        task = tasks[negative_index % len(tasks)]
        event, transcript, sink_record = _process_request(
            provider,
            task,
            negative_case["case_id"],
            negative_case["kind"],
            f"Q0-REQ-{request_index:03d}",
            request_index,
            identity,
            selected_run_id,
            schema,
            permission_rule,
            code_root,
        )
        if event["terminal_state"] != negative_case["expected_terminal_state"]:
            raise QualificationError("fake_case_mismatch", "fake negative case did not produce its frozen terminal state")
        if sink_record is not None:
            raise QualificationError("sink_policy_failed", "negative output reached the public sink")
        events.append(event)
        transcripts.append(transcript)

    if len(events) != expected_request_count or provider.calls != expected_request_count:
        raise QualificationError("readiness_count_mismatch", "Q0 request accounting does not close")
    if any(event["terminal_state"] not in TERMINAL_STATES for event in events):
        raise QualificationError("terminal_state_invalid", "Q0 contains an invalid terminal state")

    for event in events:
        append_jsonl(paths["events"], paths["root"], event)
    for transcript in transcripts:
        append_jsonl(paths["transcripts"], paths["root"], transcript)
    for sink_record in sink_records:
        append_jsonl(paths["public_sink"], paths["root"], sink_record)
    outcomes = _build_outcomes(identity, selected_run_id, events, provider.calls, len(sink_records))
    _write_new_json(paths["outcomes"], paths["root"], outcomes)

    protected_after = capture_protected_tree_hashes(workspace_root)
    if protected_before != protected_after:
        raise QualificationError("protected_tree_changed", "a protected historical tree changed during Q0")
    manifest = {
        **identity,
        "run_id": selected_run_id,
        "source_identity": source_identity,
        "frozen_protocol": config,
        "static_input_hashes": collect_static_input_hashes(code_root),
        "code_provenance": collect_code_provenance(code_root),
        "runtime": runtime_identity(),
        "artifact_layout": {
            name: paths[name].relative_to(paths["root"]).as_posix()
            for name in ("events", "transcripts", "outcomes", "public_sink", "validation", "replay", "report")
        },
        "artifact_hashes": _artifact_hashes(paths),
        "protected_tree_hashes_before": protected_before,
        "protected_tree_hashes_after": protected_after,
        "fake_provider_calls": provider.calls,
        "request_count": len(events),
        "Q1_live_model_calls": "not_authorized",
        "P1_P2_status": "locked",
    }
    _write_new_json(paths["manifest"], paths["root"], manifest)
    return {"ok": True, "run_id": selected_run_id, "manifest_path": str(paths["manifest"]), "fake_provider_calls": provider.calls}
