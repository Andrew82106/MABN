"""Fail-closed validation for a derived P1v2-Q0 artifact namespace."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .errors import QualificationError
from .fixtures import REJECT_BRANCH_FAILURE_CODES, load_fixture_sets
from .paths import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    append_jsonl,
    assert_execution_roots,
    assert_workspace_root,
    assert_valid_run_id,
    artifact_paths,
    load_json,
    safe_read_text,
    sha256_bytes,
    sha256_file,
)
from .protocol import (
    EXPECTED_IDENTITY,
    EXPECTED_PROTOCOL,
    PUBLIC_FIELDS,
    can_write_public_sink,
    load_permission_rule,
    load_protocol_config,
    load_schema,
    parse_contract,
    validate_response,
)
from .provenance import (
    capture_protected_tree_hashes,
    verify_current_code_provenance,
    verify_reference_sources,
)


def _identity_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_kind": config["artifact_kind"],
        "phase": config["phase"],
        "data_role": config["data_role"],
        "execution_mode": config["execution_mode"],
        **config["identity"],
    }


REPORT_TASK_COUNT_ORDER: tuple[tuple[str, str], ...] = (
    ("Screen-coordinator", "screen:coordinator"),
    ("Screen-publisher", "screen:publisher"),
    ("Confirmation-coordinator", "confirmation:coordinator"),
    ("Confirmation-publisher", "confirmation:publisher"),
)


def report_task_count_lines(valid_task_counts: dict[str, int]) -> list[str]:
    return [f"- {label}: {valid_task_counts.get(key, 0)}" for label, key in REPORT_TASK_COUNT_ORDER]


def report_negative_case_lines(negative_case_terminal_counts: dict[str, dict[str, int]]) -> list[str]:
    lines: list[str] = []
    for case_id in sorted(negative_case_terminal_counts):
        terminal_counts = negative_case_terminal_counts[case_id]
        rendered_counts = ", ".join(
            f"{terminal_state}={count}"
            for terminal_state, count in sorted(terminal_counts.items())
        )
        lines.append(f"- {case_id}: {rendered_counts}")
    return lines


def _read_jsonl(path: Path, root: Path) -> list[Any]:
    text = safe_read_text(path, root)
    if not text.strip():
        raise QualificationError("artifact_empty", "artifact JSONL is empty")
    from .paths import strict_json_loads

    records: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            raise QualificationError("artifact_invalid", "artifact JSONL contains an empty line")
        records.append(strict_json_loads(line))
    return records


def _add_identity_errors(errors: list[str], record: Any, identity: dict[str, Any], run_id: str, label: str) -> None:
    if not isinstance(record, dict):
        errors.append(f"{label} is not an object")
        return
    if record.get("run_id") != run_id:
        errors.append(f"{label} run ID does not match")
    for key, expected in identity.items():
        if record.get(key) != expected or type(record.get(key)) is not type(expected):
            errors.append(f"{label} identity field {key} does not match")


def _check_manifest(
    manifest: Any,
    paths: dict[str, Path],
    identity: dict[str, Any],
    code_root: Path,
    workspace_root: Path,
    expected_request_count: int,
    errors: list[str],
) -> None:
    _add_identity_errors(errors, manifest, identity, paths["manifest"].stem.removeprefix("manifest_"), "manifest")
    if not isinstance(manifest, dict):
        return
    run_id = paths["manifest"].stem.removeprefix("manifest_")
    expected_layout = {
        name: paths[name].relative_to(paths["root"]).as_posix()
        for name in ("events", "transcripts", "outcomes", "public_sink", "validation", "replay", "report")
    }
    if manifest.get("artifact_layout") != expected_layout:
        errors.append("manifest artifact layout does not match derived paths")
    if manifest.get("run_id") != run_id:
        errors.append("manifest run ID does not match the derived path")
    try:
        frozen_protocol = load_protocol_config(code_root)
        if manifest.get("frozen_protocol") != frozen_protocol:
            errors.append("manifest frozen protocol does not match the executed config")
    except QualificationError as exc:
        errors.append(f"manifest frozen protocol check failed: {exc.code}")
    for message in verify_current_code_provenance(manifest, code_root):
        errors.append(message)
    try:
        reference_identity = verify_reference_sources(code_root, workspace_root)
        if manifest.get("source_identity") != reference_identity:
            errors.append("manifest reference source identity does not match")
    except QualificationError as exc:
        errors.append(f"reference provenance failed: {exc.code}")
    actual_hashes: dict[str, str] = {}
    for name in ("events", "transcripts", "outcomes", "public_sink"):
        try:
            actual_hashes[name] = sha256_file(paths[name], paths["root"])
        except QualificationError as exc:
            errors.append(f"artifact hash input failed: {exc.code}")
    if manifest.get("artifact_hashes") != actual_hashes:
        errors.append("manifest artifact hashes do not match")
    try:
        current_protected = capture_protected_tree_hashes(workspace_root)
        if manifest.get("protected_tree_hashes_before") != current_protected:
            errors.append("protected-tree hash differs from pre-run record")
        if manifest.get("protected_tree_hashes_after") != current_protected:
            errors.append("protected-tree hash differs from post-run record")
    except QualificationError as exc:
        errors.append(f"protected-tree audit failed: {exc.code}")
    if manifest.get("fake_provider_calls") != expected_request_count or manifest.get("request_count") != expected_request_count:
        errors.append("manifest request accounting does not match Q0")
    if manifest.get("Q1_live_model_calls") != "not_authorized" or manifest.get("P1_P2_status") != "locked":
        errors.append("manifest downstream status is invalid")


def _check_events_and_transcripts(
    events: list[Any],
    transcripts: list[Any],
    tasks: list[dict[str, Any]],
    negative_cases: list[dict[str, str]],
    identity: dict[str, Any],
    run_id: str,
    schema: dict[str, Any],
    permission_rule: dict[str, Any],
    errors: list[str],
) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, dict[str, int]], list[dict[str, str]]]:
    task_lookup = {task["task_id"]: task for task in tasks}
    negative_lookup = {case["case_id"]: case for case in negative_cases}
    valid_task_count = len(tasks)
    expected_request_count = valid_task_count + len(negative_cases)
    if len(events) != expected_request_count:
        errors.append("events do not contain the frozen request count")
    if len(transcripts) != expected_request_count:
        errors.append("transcripts do not contain the frozen request count")
    transcript_lookup: dict[str, Any] = {}
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            errors.append("transcript is not an object")
            continue
        request_id = transcript.get("request_id")
        if not isinstance(request_id, str) or request_id in transcript_lookup:
            errors.append("transcript request IDs are invalid or duplicated")
            continue
        transcript_lookup[request_id] = transcript
    seen_requests: set[str] = set()
    terminal_counts: Counter[str] = Counter()
    valid_task_counts: Counter[str] = Counter()
    negative_terminal_counts: Counter[str] = Counter()
    negative_case_terminal_counts: dict[str, Counter[str]] = {}
    expected_sink_records: list[dict[str, str]] = []
    seen_negative: set[str] = set()
    for expected_index, event in enumerate(events, start=1):
        _add_identity_errors(errors, event, identity, run_id, "event")
        if not isinstance(event, dict):
            continue
        request_id = event.get("request_id")
        if request_id != f"Q0-REQ-{expected_index:03d}" or request_id in seen_requests:
            errors.append("event request ordering or uniqueness is invalid")
        seen_requests.add(request_id if isinstance(request_id, str) else f"invalid-{expected_index}")
        if event.get("event_index") != expected_index or event.get("attempt") != 1 or event.get("retry_count") != 0:
            errors.append("event retry or index accounting is invalid")
        if event.get("provider_kind") != "deterministic_fake_provider":
            errors.append("event provider kind is invalid")
        state = event.get("terminal_state")
        if state not in {"model_output", "model_output_rejected", "model_call_failed"}:
            errors.append("event terminal state is invalid")
            continue
        terminal_counts[state] += 1
        task = task_lookup.get(event.get("task_id"))
        if task is None:
            errors.append("event references an unknown task")
            continue
        if event.get("stage") != task["stage"] or event.get("role") != task["role"]:
            errors.append("event task role or stage does not match")
        case_id = event.get("case_id")
        if expected_index <= valid_task_count:
            if task["task_id"] != tasks[expected_index - 1]["task_id"]:
                errors.append("valid task order does not match the frozen fixture")
            if case_id != "valid" or state != "model_output":
                errors.append("a valid task did not complete successfully")
            else:
                valid_task_counts[f"{task['stage']}:{task['role']}"] += 1
        else:
            negative_index = expected_index - valid_task_count - 1
            expected_case = negative_cases[negative_index] if negative_index < len(negative_cases) else None
            if expected_case is None or case_id != expected_case["case_id"] or case_id in seen_negative:
                errors.append("negative case coverage is invalid")
            else:
                seen_negative.add(case_id)
                if state != expected_case["expected_terminal_state"]:
                    errors.append("negative case terminal state does not match")
                negative_terminal_counts[state] += 1
                negative_case_terminal_counts.setdefault(case_id, Counter())[state] += 1
        transcript = transcript_lookup.get(request_id)
        if transcript is None:
            errors.append("event has no matching transcript")
            continue
        _add_identity_errors(errors, transcript, identity, run_id, "transcript")
        if transcript.get("task_id") != task["task_id"] or transcript.get("case_id") != case_id:
            errors.append("transcript task identity does not match event")
        if transcript.get("terminal_state") != state or transcript.get("provider_kind") != "deterministic_fake_provider":
            errors.append("transcript terminal state does not match event")
        if state == "model_call_failed":
            if not isinstance(transcript.get("failure_code"), str) or "raw_output" in transcript:
                errors.append("failed fake-provider transcript is malformed")
            continue
        raw_output = transcript.get("raw_output")
        if not isinstance(raw_output, str) or transcript.get("raw_output_sha256") != sha256_bytes(raw_output.encode("utf-8")):
            errors.append("model-output transcript hash is invalid")
            continue
        if state == "model_output":
            try:
                report = validate_response(raw_output, task, schema)
            except QualificationError as exc:
                errors.append(f"accepted output does not validate: {exc.code}")
                continue
            if can_write_public_sink(task, report, permission_rule):
                expected_sink_records.append(report)
        else:
            failure_code = event.get("failure_code")
            if failure_code != transcript.get("failure_code") or not isinstance(failure_code, str) or not failure_code:
                errors.append("rejected-output failure code is malformed")
            expected_reject_failure_code = REJECT_BRANCH_FAILURE_CODES.get(case_id)
            if expected_reject_failure_code is not None:
                try:
                    parsed = parse_contract(raw_output, schema)
                except QualificationError as exc:
                    errors.append(f"contract-valid reject case does not satisfy the contract: {exc.code}")
                else:
                    if parsed.get("decision") != "reject" or "public_report" in parsed:
                        errors.append("contract-valid reject case did not use the reject-only branch")
                try:
                    validate_response(raw_output, task, schema)
                except QualificationError as exc:
                    if exc.code != expected_reject_failure_code:
                        errors.append(f"contract-valid reject case used the wrong failure: {exc.code}")
                else:
                    errors.append("contract-valid reject case unexpectedly satisfied an allow task")
                if failure_code != expected_reject_failure_code:
                    errors.append("contract-valid reject event does not record its expected failure code")
                continue
            try:
                parsed = parse_contract(raw_output, schema)
            except QualificationError:
                parsed = None
            else:
                if parsed.get("decision") == "reject":
                    errors.append("an unapproved negative case used the reject contract branch")
            try:
                validate_response(raw_output, task, schema)
            except QualificationError as exc:
                if exc.code == "semantic_reject":
                    errors.append("an unapproved negative case used a reject-branch failure code")
            else:
                errors.append("rejected output unexpectedly validates")
    if set(transcript_lookup) != seen_requests:
        errors.append("transcript and event request sets do not match")
    if seen_negative != set(negative_lookup):
        errors.append("negative-case closure is incomplete")
    expected_valid_task_counts = Counter(f"{task['stage']}:{task['role']}" for task in tasks)
    if valid_task_counts != expected_valid_task_counts:
        errors.append("valid task role counts do not match the frozen fixture")
    normalized_negative_case_counts = {
        case_id: dict(sorted(counts.items()))
        for case_id, counts in sorted(negative_case_terminal_counts.items())
    }
    expected_negative_case_counts = {
        case["case_id"]: {case["expected_terminal_state"]: 1}
        for case in negative_cases
    }
    if normalized_negative_case_counts != expected_negative_case_counts:
        errors.append("negative case terminal counts do not match the frozen fixture")
    return terminal_counts, valid_task_counts, negative_terminal_counts, normalized_negative_case_counts, expected_sink_records


def _check_outcomes(
    outcome: Any,
    identity: dict[str, Any],
    run_id: str,
    terminal_counts: Counter[str],
    valid_task_counts: Counter[str],
    negative_terminal_counts: Counter[str],
    negative_case_terminal_counts: dict[str, dict[str, int]],
    expected_request_count: int,
    sink_records: int,
    errors: list[str],
) -> None:
    _add_identity_errors(errors, outcome, identity, run_id, "outcomes")
    if not isinstance(outcome, dict):
        return
    checks = {
        "request_count": expected_request_count,
        "terminal_state_counts": dict(sorted(terminal_counts.items())),
        "valid_task_counts": dict(sorted(valid_task_counts.items())),
        "negative_terminal_state_counts": dict(sorted(negative_terminal_counts.items())),
        "negative_case_terminal_counts": negative_case_terminal_counts,
        "fake_provider_calls": expected_request_count,
        "public_sink_records": sink_records,
        "q1_live_model_calls_authorized": False,
        "p1_p2_status": "locked",
    }
    for key, expected in checks.items():
        if outcome.get(key) != expected:
            errors.append(f"outcomes field {key} does not match recomputed value")


def _check_public_sink(records: list[Any], expected_records: list[dict[str, str]], errors: list[str]) -> None:
    if len(records) != len(expected_records):
        errors.append("public sink count does not match publisher-only Q0 output")
    actual_serialized: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != set(PUBLIC_FIELDS):
            errors.append("public sink record contains a non-public field")
            continue
        actual_serialized.append(str(sorted(record.items())))
    expected_serialized = [str(sorted(record.items())) for record in expected_records]
    if sorted(actual_serialized) != sorted(expected_serialized):
        errors.append("public sink records do not match validated publisher output")


def _check_audit_records(path: Path, root: Path, identity: dict[str, Any], run_id: str, kind: str, errors: list[str]) -> None:
    if not path.exists():
        return
    try:
        records = _read_jsonl(path, root)
    except QualificationError as exc:
        errors.append(f"{kind} audit is unreadable: {exc.code}")
        return
    for record in records:
        _add_identity_errors(errors, record, identity, run_id, kind)
        if not isinstance(record, dict):
            continue
        if record.get("real_model_calls") != 0 or record.get("local_loopback_http_calls") != 0 or record.get("remote_network_calls") != 0:
            errors.append(f"{kind} audit records a forbidden call count")
        if kind == "validation" and record.get("validation_model_calls") != 0:
            errors.append("validation audit records model calls")
        if kind == "replay" and (record.get("replay_model_calls") != 0 or record.get("replay_network_calls") != 0):
            errors.append("replay audit records calls")


def _check_report(
    path: Path,
    identity: dict[str, Any],
    run_id: str,
    terminal_counts: Counter[str],
    valid_task_counts: Counter[str],
    negative_case_terminal_counts: dict[str, dict[str, int]],
    expected_request_count: int,
    errors: list[str],
) -> None:
    if not path.exists():
        return
    try:
        text = safe_read_text(path, path.parent.parent)
    except QualificationError as exc:
        errors.append(f"delivery report is unreadable: {exc.code}")
        return
    required = [
        run_id,
        "P1v2-Q0 readiness status: pass",
        "P1v2-Q1 live model calls: not authorized",
        "P1/P2 status: locked",
        "real_model_calls: 0",
        "local_loopback_http_calls: 0",
        "remote_network_calls: 0",
        "Report registry: unique current report",
        f"- request_count: {expected_request_count}",
        f"- model_output: {terminal_counts.get('model_output', 0)}",
        f"- model_output_rejected: {terminal_counts.get('model_output_rejected', 0)}",
        f"- model_call_failed: {terminal_counts.get('model_call_failed', 0)}",
    ]
    required.extend(report_task_count_lines(dict(valid_task_counts)))
    required.extend(report_negative_case_lines(negative_case_terminal_counts))
    if any(item not in text for item in required):
        errors.append("delivery report is missing required identity, status, or recomputed count text")


def _check_report_registry(paths: dict[str, Path], run_id: str, errors: list[str]) -> None:
    """Ensure one current report and explicitly retired historical reports.

    The registry only examines the human-readable report namespace.  It does not
    mutate or reinterpret any historical event, transcript, outcome, or manifest.
    A fresh dry run has no current report until its first successful replay, so
    zero current reports is permitted only while the derived report path is absent.
    """
    reports_directory = paths["root"] / "reports"
    report_prefix = "P1V2Q0_DELIVERY_"
    reports = sorted(reports_directory.glob(f"{report_prefix}P1V2Q-READINESS-DRY-*.md")) if reports_directory.exists() else []
    current_report_ids: list[str] = []
    for report_path in reports:
        encoded_run_id = report_path.stem.removeprefix(report_prefix)
        try:
            assert_valid_run_id(encoded_run_id)
            text = safe_read_text(report_path, paths["root"])
        except QualificationError as exc:
            errors.append(f"report registry could not read a report: {exc.code}")
            continue
        current_marker = "Current Q0 run ID:"
        if current_marker in text:
            current_report_ids.append(encoded_run_id)
            expected_marker = f"- Current Q0 run ID: `{encoded_run_id}`"
            if text.count(current_marker) != 1 or expected_marker not in text:
                errors.append("report registry current marker does not match its report path")
            if encoded_run_id != run_id:
                errors.append("report registry names a different current Q0 run")
            continue
        historical_marker = f"- Historical Q0 run ID: `{encoded_run_id}`"
        if historical_marker not in text:
            errors.append("historical report is missing its historical Q0 run ID")
        if "Historical pre-rework record; not for downstream decisions." not in text:
            errors.append("historical report is missing its pre-rework warning")
    if len(current_report_ids) > 1:
        errors.append("report registry contains more than one current Q0 report")
    if paths["report"].exists() and current_report_ids != [run_id]:
        errors.append("report registry does not contain exactly this run as current")


def _validate(
    run_id: str,
    code_root: Path,
    data_root: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    paths = artifact_paths(data_root, run_id)
    try:
        config = load_protocol_config(code_root)
        identity = _identity_from_config(config)
        schema = load_schema(code_root)
        permission_rule = load_permission_rule(code_root)
        screen, confirmation, negatives = load_fixture_sets(code_root)
    except QualificationError as exc:
        return {"passed": False, "run_id": run_id, "errors": [f"frozen input failed: {exc.code}"]}
    try:
        manifest = load_json(paths["manifest"], paths["root"])
        events = _read_jsonl(paths["events"], paths["root"])
        transcripts = _read_jsonl(paths["transcripts"], paths["root"])
        outcomes = load_json(paths["outcomes"], paths["root"])
        sink_records = _read_jsonl(paths["public_sink"], paths["root"])
    except QualificationError as exc:
        return {"passed": False, "run_id": run_id, "errors": [f"artifact read failed: {exc.code}"]}
    expected_request_count = len(screen) + len(confirmation) + len(negatives)
    _check_manifest(manifest, paths, identity, code_root, workspace_root, expected_request_count, errors)
    terminal_counts, valid_task_counts, negative_terminal_counts, negative_case_terminal_counts, expected_sink_records = _check_events_and_transcripts(
        events,
        transcripts,
        screen + confirmation,
        negatives,
        identity,
        run_id,
        schema,
        permission_rule,
        errors,
    )
    _check_outcomes(
        outcomes,
        identity,
        run_id,
        terminal_counts,
        valid_task_counts,
        negative_terminal_counts,
        negative_case_terminal_counts,
        expected_request_count,
        len(sink_records),
        errors,
    )
    _check_public_sink(sink_records, expected_sink_records, errors)
    _check_audit_records(paths["validation"], paths["root"], identity, run_id, "validation", errors)
    _check_audit_records(paths["replay"], paths["root"], identity, run_id, "replay", errors)
    _check_report(
        paths["report"],
        identity,
        run_id,
        terminal_counts,
        valid_task_counts,
        negative_case_terminal_counts,
        expected_request_count,
        errors,
    )
    _check_report_registry(paths, run_id, errors)
    return {
        "passed": not errors,
        "run_id": run_id,
        "errors": errors,
        "request_count": len(events),
        "terminal_state_counts": dict(sorted(terminal_counts.items())),
        "valid_task_counts": dict(sorted(valid_task_counts.items())),
        "negative_terminal_state_counts": dict(sorted(negative_terminal_counts.items())),
        "negative_case_terminal_counts": negative_case_terminal_counts,
        "real_model_calls": 0,
        "local_loopback_http_calls": 0,
        "remote_network_calls": 0,
        "replay_model_calls": 0,
        "replay_network_calls": 0,
    }


def validate_run(
    run_id: str,
    *,
    code_root: Path = PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    workspace_root: Path | None = None,
    write_audit: bool = True,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Return a structured PASS/FAIL result; never calls a provider or network."""
    code_root = code_root.resolve(strict=False)
    data_root = data_root.resolve(strict=False)
    workspace_root = (workspace_root or code_root.parents[2]).resolve(strict=False)
    try:
        assert_execution_roots(code_root, data_root, test_mode=test_mode)
        assert_workspace_root(workspace_root)
    except QualificationError as exc:
        return {"passed": False, "run_id": run_id, "errors": [f"validator failed closed: {exc.code}"]}
    try:
        result = _validate(run_id, code_root, data_root, workspace_root)
    except QualificationError as exc:
        result = {"passed": False, "run_id": run_id, "errors": [f"validator failed closed: {exc.code}"]}
    except Exception:
        result = {"passed": False, "run_id": run_id, "errors": ["validator failed closed: unexpected_error"]}
    if write_audit:
        try:
            paths = artifact_paths(data_root, run_id)
            config = load_protocol_config(code_root)
            audit = {
                **_identity_from_config(config),
                "run_id": run_id,
                "passed": result["passed"],
                "errors": result["errors"],
                "validation_model_calls": 0,
                "validation_network_calls": 0,
                "real_model_calls": 0,
                "local_loopback_http_calls": 0,
                "remote_network_calls": 0,
                "replay_model_calls": 0,
                "replay_network_calls": 0,
            }
            append_jsonl(paths["validation"], paths["root"], audit)
        except QualificationError:
            # A malformed run ID or missing root must remain a safe nonzero CLI
            # result; it cannot justify an unsafe fallback write.
            pass
    return result
