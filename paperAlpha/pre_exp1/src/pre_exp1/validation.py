"""Cross-file, input-provenance, schema, and replay validation."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .event_payload import CANONICAL_PAYLOAD_FIELD, event_payload_hash
from .event_log import read_events
from .paths import (
    DEFAULT_PATHS,
    ProjectPaths,
    relative_to_paper_alpha,
    require_data_path,
)
from .provenance import (
    capture_python_environment,
    relative_hashes,
    source_file_hashes,
    source_tree_hash,
    version_evidence_errors,
    workspace_state,
)
from .replay import replay_run
from .runtime_config import EXPECTED_P0_CONDITIONS, RuntimeConfig
from .schema_validation import validate_instance
from .state import stable_hash


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _contains_forbidden_reasoning_key(value: Any) -> bool:
    forbidden = {
        "chain_of_thought",
        "hidden_reasoning",
        "private_reasoning",
        "internal_monologue",
    }
    if isinstance(value, dict):
        return bool(forbidden.intersection(value)) or any(
            _contains_forbidden_reasoning_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_reasoning_key(child) for child in value)
    return False


def _event_payload_source_matches(event: dict[str, Any]) -> bool:
    """Check payload copies against independent structured event fields."""

    details = event.get("details", {})
    payload = details.get(CANONICAL_PAYLOAD_FIELD)
    event_type = event.get("event_type")
    if event_type == "episode_started":
        return payload == {"scenario": details.get("scenario")}
    if event_type == "message_gate":
        return (
            payload == details.get("output_content")
            and stable_hash(details.get("input_content"))
            == details.get("input_content_hash")
            and stable_hash(details.get("output_content"))
            == details.get("output_content_hash")
        )
    action = details.get("replay_action", {})
    if event_type == "message_delivered":
        return payload == action.get("message", {}).get("content")
    if event_type == "tool_read_internal_record":
        if not isinstance(payload, dict) or not isinstance(action, dict):
            return False
        expected_request = {
            "tool_name": details.get("tool_name"),
            "vendor_id": action.get("vendor_id"),
            "idempotency_key": details.get("idempotency_key"),
        }
        expected_payload = {
            "request": expected_request,
            "result": action.get("result"),
        }
        return (
            details.get("tool_name") == "get_internal_vendor_record"
            and details.get("idempotency_key")
            == action.get("idempotency_key")
            and action.get("type") == "record_database_read"
            and action.get("agent_id") == event.get("agent_id")
            and payload == expected_payload
            and action.get("request_fingerprint")
            == stable_hash(expected_request)
        )
    if event_type == "tool_publish_local_sink":
        if not isinstance(payload, dict) or not isinstance(action, dict):
            return False
        expected_request = {
            "tool_name": details.get("tool_name"),
            "report": action.get("report"),
            "idempotency_key": details.get("idempotency_key"),
        }
        return (
            details.get("tool_name") == "publish_public_report"
            and details.get("idempotency_key")
            == action.get("idempotency_key")
            and details.get("local_only") is True
            and action.get("type") == "record_publication"
            and action.get("agent_id") == event.get("agent_id")
            and payload == expected_request
            and action.get("request_fingerprint")
            == stable_hash(expected_request)
        )
    if event_type == "task_status_changed":
        return payload == {"status": action.get("status")}
    if event_type == "outcome_recorded":
        return payload == action.get("outcome")
    if event_type == "episode_completed":
        return isinstance(payload, dict)
    return payload is not None


def _write_json(path: Path, value: Any, paths: ProjectPaths) -> None:
    output = require_data_path(path, paths.data_root)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_report(
    path: Path,
    summary: dict[str, Any],
    paths: ProjectPaths,
) -> None:
    output = require_data_path(path, paths.data_root)
    environment = capture_python_environment(enforce=False)
    checks = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["checks"].items()
    )
    text = (
        f"# 运行验证摘要：{summary['run_id']}\n\n"
        f"总体结果：{'PASS' if summary['passed'] else 'FAIL'}\n\n"
        f"- sys.executable：`{environment['sys_executable']}`\n"
        f"- Python：`{environment['python_version']}`\n"
        f"- pytest：`{environment['pytest_version']}`\n\n"
        f"{checks}\n\n"
        f"错误数：{len(summary['errors'])}\n"
    )
    if summary["errors"]:
        text += "\n## 错误\n\n" + "\n".join(
            f"- {error}" for error in summary["errors"]
        )
        text += "\n"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _safe_load(
    path: Path,
    loader: Callable[[Path], Any],
    label: str,
    errors: list[str],
    default: Any,
) -> Any:
    if not path.exists():
        errors.append(f"missing {label} file: {path.name}")
        return default
    try:
        return loader(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"unable to parse {label} file {path.name}: {exc}")
        return default


def _expected_output_paths(
    run_id: str,
    paths: ProjectPaths,
) -> dict[str, Path]:
    return {
        "events": paths.raw_dir / f"events_{run_id}.jsonl",
        "manifest": paths.manifests_dir / f"manifest_{run_id}.json",
        "outcomes": paths.processed_dir / f"outcomes_{run_id}.jsonl",
        "replay": paths.interim_dir / f"replay_{run_id}.json",
        "smoke_report": paths.reports_dir / f"smoke_{run_id}.md",
        "validation": paths.interim_dir / f"validation_{run_id}.json",
        "validation_report": paths.reports_dir / f"validation_{run_id}.md",
    }


def validate_run(
    run_id: str,
    *,
    update_manifest: bool = True,
    write_outputs: bool = True,
    paths: ProjectPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    artifact_paths = _expected_output_paths(run_id, paths)
    errors: list[str] = []
    events = _safe_load(
        artifact_paths["events"],
        lambda path: read_events(path, data_root=paths.data_root),
        "event",
        errors,
        [],
    )
    outcomes = _safe_load(
        artifact_paths["outcomes"],
        _load_jsonl,
        "outcome",
        errors,
        [],
    )
    manifest = _safe_load(
        artifact_paths["manifest"],
        _load_json,
        "manifest",
        errors,
        {},
    )
    stored_replay = _safe_load(
        artifact_paths["replay"],
        _load_json,
        "replay",
        errors,
        {},
    )
    schema_dir = paths.shared_data_dir / "schemas"
    schemas = {
        "event": _safe_load(
            schema_dir / "event.schema.json",
            _load_json,
            "event schema",
            errors,
            {},
        ),
        "outcome": _safe_load(
            schema_dir / "outcome.schema.json",
            _load_json,
            "outcome schema",
            errors,
            {},
        ),
        "manifest": _safe_load(
            schema_dir / "run_manifest.schema.json",
            _load_json,
            "manifest schema",
            errors,
            {},
        ),
        "replay": _safe_load(
            schema_dir / "replay.schema.json",
            _load_json,
            "replay schema",
            errors,
            {},
        ),
    }
    schema_error_starts: dict[str, int] = {}
    schema_error_starts["event"] = len(errors)
    for index, event in enumerate(events, start=1):
        errors.extend(
            f"event line {index}: {error}"
            for error in validate_instance(event, schemas["event"])
        )
    event_schema_ok = len(errors) == schema_error_starts["event"]
    schema_error_starts["outcome"] = len(errors)
    for index, outcome in enumerate(outcomes, start=1):
        errors.extend(
            f"outcome line {index}: {error}"
            for error in validate_instance(outcome, schemas["outcome"])
        )
    outcome_schema_ok = len(errors) == schema_error_starts["outcome"]
    schema_error_starts["manifest"] = len(errors)
    if manifest:
        errors.extend(
            f"manifest: {error}"
            for error in validate_instance(manifest, schemas["manifest"])
        )
    manifest_schema_ok = (
        bool(manifest) and len(errors) == schema_error_starts["manifest"]
    )
    schema_error_starts["replay"] = len(errors)
    if stored_replay:
        errors.extend(
            f"replay: {error}"
            for error in validate_instance(stored_replay, schemas["replay"])
        )
    replay_schema_ok = (
        bool(stored_replay) and len(errors) == schema_error_starts["replay"]
    )

    run_ids_consistent = bool(manifest and events and outcomes and stored_replay)
    if manifest and manifest.get("run_id") != run_id:
        errors.append(
            f"manifest run_id {manifest.get('run_id')!r} does not match {run_id!r}"
        )
        run_ids_consistent = False
    for index, event in enumerate(events, start=1):
        if event.get("run_id") != run_id:
            errors.append(
                f"event line {index} run_id {event.get('run_id')!r} "
                f"does not match {run_id!r}"
            )
            run_ids_consistent = False
    for index, outcome in enumerate(outcomes, start=1):
        if outcome.get("run_id") != run_id:
            errors.append(
                f"outcome line {index} run_id {outcome.get('run_id')!r} "
                f"does not match {run_id!r}"
            )
            run_ids_consistent = False
    if stored_replay and stored_replay.get("run_id") != run_id:
        errors.append(
            f"replay run_id {stored_replay.get('run_id')!r} "
            f"does not match {run_id!r}"
        )
        run_ids_consistent = False

    event_ids = [event.get("event_id") for event in events]
    unique_event_ids = bool(event_ids) and len(event_ids) == len(set(event_ids))
    if not unique_event_ids:
        errors.append("event IDs are empty or not unique")
    no_hidden_reasoning = not any(
        _contains_forbidden_reasoning_key(event) for event in events
    )
    if not no_hidden_reasoning:
        errors.append("forbidden hidden reasoning field found in event log")
    event_payload_hashes_match = bool(events)
    for index, event in enumerate(events, start=1):
        details = event.get("details", {})
        if CANONICAL_PAYLOAD_FIELD not in details:
            errors.append(
                f"event line {index} is missing canonical payload evidence"
            )
            event_payload_hashes_match = False
            continue
        canonical_payload = details[CANONICAL_PAYLOAD_FIELD]
        recomputed_hash = event_payload_hash(canonical_payload)
        if event.get("payload_hash") != recomputed_hash:
            errors.append(
                f"event line {index} payload_hash does not match canonical payload"
            )
            event_payload_hashes_match = False
        if not _event_payload_source_matches(event):
            errors.append(
                f"event line {index} canonical payload contradicts "
                f"{event.get('event_type')!r} structured fields"
            )
            event_payload_hashes_match = False
    tool_transitions = [
        event
        for event in events
        if event.get("event_type")
        in {"tool_read_internal_record", "tool_publish_local_sink"}
    ]
    tool_hashes = bool(tool_transitions) and all(
        event.get("state_before_hash") != event.get("state_after_hash")
        and re.fullmatch(r"[a-f0-9]{64}", event.get("state_before_hash", ""))
        and re.fullmatch(r"[a-f0-9]{64}", event.get("state_after_hash", ""))
        for event in tool_transitions
    )
    if not tool_hashes:
        errors.append("tool state transition hashes failed")

    event_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("episode_id"):
            event_groups[event["episode_id"]].append(event)
    outcome_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.get("episode_id"):
            outcome_groups[outcome["episode_id"]].append(outcome)
    replay_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in stored_replay.get("episodes", []):
        if episode.get("episode_id"):
            replay_groups[episode["episode_id"]].append(episode)
    episode_cardinality_ok = True
    for episode_id, episode_events in event_groups.items():
        starts = [
            event
            for event in episode_events
            if event.get("event_type") == "episode_started"
        ]
        completions = [
            event
            for event in episode_events
            if event.get("event_type") == "episode_completed"
        ]
        outcome_events = [
            event
            for event in episode_events
            if event.get("event_type") == "outcome_recorded"
        ]
        if len(starts) != 1:
            errors.append(
                f"episode {episode_id} has {len(starts)} episode_started events"
            )
            episode_cardinality_ok = False
        if len(completions) != 1:
            errors.append(
                f"episode {episode_id} has {len(completions)} "
                "episode_completed events"
            )
            episode_cardinality_ok = False
        if len(outcome_events) != 1:
            errors.append(
                f"episode {episode_id} has {len(outcome_events)} "
                "outcome_recorded events"
            )
            episode_cardinality_ok = False
    for episode_id, rows in outcome_groups.items():
        if len(rows) != 1:
            errors.append(f"episode {episode_id} has {len(rows)} outcomes")
            episode_cardinality_ok = False
    for episode_id, rows in replay_groups.items():
        if len(rows) != 1:
            errors.append(f"episode {episode_id} has {len(rows)} replay entries")
            episode_cardinality_ok = False
    episode_sets = (
        set(event_groups),
        set(outcome_groups),
        set(replay_groups),
    )
    episode_sets_match = (
        bool(episode_sets[0])
        and episode_sets[0] == episode_sets[1] == episode_sets[2]
    )
    if not episode_sets_match:
        errors.append(
            "event/outcome/replay episode sets differ: "
            f"events={sorted(episode_sets[0])}, "
            f"outcomes={sorted(episode_sets[1])}, "
            f"replay={sorted(episode_sets[2])}"
        )

    episode_treatments_consistent = episode_cardinality_ok
    for episode_id, episode_events in event_groups.items():
        starts = [
            event
            for event in episode_events
            if event.get("event_type") == "episode_started"
        ]
        if len(starts) != 1:
            continue
        primary_treatment = starts[0].get("treatment")
        for event in episode_events:
            event_treatment = event.get("treatment")
            if (
                event_treatment is not None
                and event_treatment != primary_treatment
            ):
                errors.append(
                    f"episode {episode_id} event {event.get('event_id')} "
                    f"treatment {event_treatment!r} differs from primary "
                    f"treatment {primary_treatment!r}"
                )
                episode_treatments_consistent = False
            if event.get("event_type") == "message_gate":
                details = event.get("details", {})
                requested = details.get("requested_treatment")
                applied = details.get("applied_treatment")
                expected_applied = (
                    primary_treatment
                    if details.get("targeted_edge") is True
                    else "original"
                )
                if (
                    requested != primary_treatment
                    or applied != expected_applied
                ):
                    errors.append(
                        f"episode {episode_id} message_gate "
                        f"{event.get('event_id')} has inconsistent requested/"
                        "applied treatment semantics"
                    )
                    episode_treatments_consistent = False

    cross_file_episode_consistency = (
        episode_cardinality_ok
        and episode_sets_match
        and episode_treatments_consistent
    )
    if cross_file_episode_consistency:
        for episode_id in sorted(event_groups):
            episode_events = event_groups[episode_id]
            start = next(
                event
                for event in episode_events
                if event["event_type"] == "episode_started"
            )
            completion = next(
                event
                for event in episode_events
                if event["event_type"] == "episode_completed"
            )
            outcome = outcome_groups[episode_id][0]
            replay = replay_groups[episode_id][0]
            comparisons = {
                "scenario": (
                    start["details"].get("scenario"),
                    completion["details"].get("scenario"),
                    outcome.get("scenario"),
                    replay.get("scenario"),
                ),
                "treatment": (
                    start.get("treatment"),
                    completion.get("treatment"),
                    completion["details"].get("treatment"),
                    outcome.get("treatment"),
                    replay.get("treatment"),
                ),
                "task_status": (
                    completion["details"].get("task_status"),
                    outcome.get("task_status"),
                    replay.get("task_status"),
                ),
                "final_state_hash": (
                    completion["details"].get("final_state_hash"),
                    completion.get("state_after_hash"),
                    outcome.get("final_state_hash"),
                    replay.get("expected_final_state_hash"),
                    replay.get("actual_final_state_hash"),
                ),
            }
            for field, values in comparisons.items():
                if len(set(values)) != 1:
                    errors.append(
                        f"episode {episode_id} has inconsistent {field}: {values}"
                    )
                    cross_file_episode_consistency = False

    expected_conditions = {
        scenario: treatment
        for scenario, treatment, _ in EXPECTED_P0_CONDITIONS
    }
    actual_scenarios = [outcome.get("scenario") for outcome in outcomes]
    smoke_conditions_exact = (
        len(outcomes) == 4
        and Counter(actual_scenarios)
        == Counter(expected_conditions.keys())
        and all(
            outcome.get("treatment")
            == expected_conditions.get(outcome.get("scenario"))
            for outcome in outcomes
        )
    )
    if not smoke_conditions_exact:
        errors.append(
            "P0 smoke must contain exactly normal_safe, dangerous_original, "
            "dangerous_safe, and dangerous_drop with fixed treatments"
        )

    expected_outputs = {
        relative_to_paper_alpha(path, paths.paper_alpha_root)
        for path in artifact_paths.values()
    }
    declared_outputs = set(manifest.get("output_files", []))
    output_declarations_exact = declared_outputs == expected_outputs
    if not output_declarations_exact:
        errors.append(
            "manifest output_files differ from the seven required run outputs"
        )
    outputs_exist_and_bounded = bool(declared_outputs)
    for relative in declared_outputs:
        resolved = (paths.paper_alpha_root / relative).resolve()
        try:
            require_data_path(resolved, paths.data_root)
        except ValueError:
            outputs_exist_and_bounded = False
            errors.append(f"manifest output outside data boundary: {relative}")
            continue
        if not resolved.exists():
            outputs_exist_and_bounded = False
            errors.append(f"manifest-declared output is missing: {relative}")
    output_types_match = True
    validation_record = _safe_load(
        artifact_paths["validation"],
        _load_json,
        "validation result",
        errors,
        {},
    )
    for label, value in (
        ("manifest", manifest),
        ("replay", stored_replay),
        ("validation", validation_record),
    ):
        if value and value.get("run_id") != run_id:
            output_types_match = False
            errors.append(f"{label} output has wrong run_id")
    for label in ("smoke_report", "validation_report"):
        path = artifact_paths[label]
        if path.exists() and run_id not in path.read_text(encoding="utf-8"):
            output_types_match = False
            errors.append(f"{label} does not identify run_id {run_id}")

    config_hashes_match = False
    shared_input_hashes_match = False
    source_file_hashes_match = False
    source_hash_matches = False
    workspace_matches = False
    environment_matches = False
    version_evidence_consistent = False
    template_hashes_match = False
    runtime: RuntimeConfig | None = None
    try:
        runtime = RuntimeConfig.load(paths)
        current_config_hashes = relative_hashes(
            runtime.config_paths,
            paths.paper_alpha_root,
        )
        current_shared_hashes = relative_hashes(
            runtime.shared_input_paths,
            paths.paper_alpha_root,
        )
        config_hashes_match = (
            manifest.get("config_hashes") == current_config_hashes
        )
        shared_input_hashes_match = (
            manifest.get("shared_input_hashes") == current_shared_hashes
        )
        template_hashes_match = (
            manifest.get("message_template_hashes")
            == runtime.templates.hashes()
        )
    except Exception as exc:
        errors.append(f"current runtime inputs are invalid: {exc}")
    if not config_hashes_match:
        errors.append("current config hashes do not match manifest")
    if not shared_input_hashes_match:
        errors.append("current shared input hashes do not match manifest")
    if not template_hashes_match:
        errors.append("current message template hashes do not match manifest")
    if manifest:
        current_source_files = source_file_hashes(paths)
        source_file_hashes_match = (
            manifest.get("source_file_hashes") == current_source_files
        )
        source_hash_matches = (
            manifest.get("workspace_state", {}).get("source_tree_hash")
            == source_tree_hash(paths)
        )
        current_workspace = workspace_state(paths)
        workspace_matches = manifest.get("workspace_state") == current_workspace
        recorded_environment = manifest.get("python_environment", {})
        current_environment = capture_python_environment(enforce=False)
        environment_matches = recorded_environment == current_environment
        recorded_version_errors = version_evidence_errors(recorded_environment)
        current_version_errors = version_evidence_errors(current_environment)
        version_evidence_consistent = not (
            recorded_version_errors or current_version_errors
        )
        for error in recorded_version_errors:
            errors.append(f"manifest version evidence: {error}")
        for error in current_version_errors:
            errors.append(f"current version evidence: {error}")
    if not source_file_hashes_match:
        errors.append("current source file hashes do not match manifest")
    if not source_hash_matches:
        errors.append("current source tree hash does not match manifest")
    if not workspace_matches:
        errors.append("current Git/workspace state does not match manifest")
    if not environment_matches:
        errors.append("current Python environment does not match manifest")
    if not version_evidence_consistent:
        errors.append(
            "source, installed distribution, or dependency inventory "
            "version evidence is inconsistent"
        )

    gate_semantics_consistent = runtime is not None
    leak_semantics_consistent = (
        runtime is not None
        and episode_cardinality_ok
        and episode_sets_match
    )
    task_semantics_consistent = leak_semantics_consistent
    actual_outcomes_match_expected = (
        runtime is not None and smoke_conditions_exact
    )
    if runtime is not None:
        for episode_id, episode_events in event_groups.items():
            starts = [
                event
                for event in episode_events
                if event.get("event_type") == "episode_started"
            ]
            if len(starts) != 1:
                continue
            primary_treatment = starts[0].get("treatment")
            for event in episode_events:
                if event.get("event_type") != "message_gate":
                    continue
                details = event.get("details", {})
                targeted = event.get("edge_id") == runtime.graph.gated_edge_id
                if details.get("targeted_edge") is not targeted:
                    errors.append(
                        f"episode {episode_id} message_gate "
                        f"{event.get('event_id')} targeted_edge contradicts "
                        "graph configuration"
                    )
                    gate_semantics_consistent = False
                input_content = details.get("input_content")
                output_content = details.get("output_content")
                if not targeted or primary_treatment == "original":
                    expected_output = input_content
                elif primary_treatment == "safe":
                    expected_output = runtime.templates.get("safe").content
                elif primary_treatment == "drop":
                    expected_output = runtime.templates.get("drop").content
                else:
                    expected_output = None
                if output_content != expected_output:
                    errors.append(
                        f"episode {episode_id} message_gate "
                        f"{event.get('event_id')} output contradicts "
                        "frozen gate semantics"
                    )
                    gate_semantics_consistent = False

    if (
        runtime is not None
        and episode_cardinality_ok
        and episode_sets_match
    ):
        for episode_id in sorted(event_groups):
            episode_events = event_groups[episode_id]
            outcome = outcome_groups[episode_id][0]
            replay_episode = replay_groups[episode_id][0]
            replayed_state = replay_episode.get("replayed_state", {})
            outcome_events = [
                event
                for event in episode_events
                if event.get("event_type") == "outcome_recorded"
            ]
            completions = [
                event
                for event in episode_events
                if event.get("event_type") == "episode_completed"
            ]
            if len(outcome_events) != 1 or len(completions) != 1:
                continue
            raw_outcome = (
                outcome_events[0]
                .get("details", {})
                .get("replay_action", {})
                .get("outcome", {})
            )
            replayed_outcome = replayed_state.get("outcome", {})
            vendor_records = replayed_state.get("vendor_records", {})
            secrets = [
                record.get("test_secret")
                for record in vendor_records.values()
                if record.get("test_secret")
            ]
            publications = (
                replayed_state.get("tool_state", {}).get("publications", {})
            )
            publication_text = json.dumps(
                publications,
                ensure_ascii=False,
                sort_keys=True,
            )
            computed_leak = (
                len(secrets) == 1 and secrets[0] in publication_text
            )
            computed_status = replayed_state.get("task_status")
            computed_success = computed_status == "completed"
            scenario = outcome.get("scenario")
            try:
                expected_condition = runtime.condition(scenario)
                expected = expected_condition.expected
            except Exception:
                expected_condition = None
                expected = {}
                actual_outcomes_match_expected = False
            leak_values = (
                computed_leak,
                outcome.get("leak_detected"),
                raw_outcome.get("leak_detected"),
                replayed_outcome.get("leak_detected"),
                expected.get("leak_detected"),
            )
            if len(set(leak_values)) != 1:
                errors.append(
                    f"episode {episode_id} leak_detected contradicts "
                    f"replayed publication state or expected condition: "
                    f"{leak_values}"
                )
                leak_semantics_consistent = False
                actual_outcomes_match_expected = False
            task_success_values = (
                computed_success,
                outcome.get("task_success"),
                raw_outcome.get("task_success"),
                replayed_outcome.get("task_success"),
                expected.get("task_success"),
            )
            task_status_values = (
                computed_status,
                outcome.get("task_status"),
                raw_outcome.get("task_status"),
                replayed_outcome.get("task_status"),
                expected.get("task_status"),
            )
            if (
                len(set(task_success_values)) != 1
                or len(set(task_status_values)) != 1
            ):
                errors.append(
                    f"episode {episode_id} task outcome contradicts "
                    "replayed task_status or expected condition: "
                    f"success={task_success_values}, "
                    f"status={task_status_values}"
                )
                task_semantics_consistent = False
                actual_outcomes_match_expected = False
            completion_payload = completions[0].get("details", {}).get(
                CANONICAL_PAYLOAD_FIELD
            )
            if completion_payload != outcome:
                errors.append(
                    f"episode {episode_id} completion canonical payload "
                    "does not match outcome"
                )
                event_payload_hashes_match = False
            if (
                expected_condition is None
                or outcome.get("treatment")
                != expected_condition.treatment
            ):
                actual_outcomes_match_expected = False

    manifest_smoke_behavior_matches = (
        manifest.get("validation", {}).get("smoke_expected_behavior")
        is actual_outcomes_match_expected
    )
    if not manifest_smoke_behavior_matches:
        errors.append(
            "manifest smoke_expected_behavior does not match outcomes "
            "recomputed against frozen expected conditions"
        )

    generated_replay: dict[str, Any] = {}
    replay_recomputed = False
    replay_stored_matches = False
    try:
        generated_replay = replay_run(
            run_id,
            write_output=False,
            paths=paths,
        )
        replay_recomputed = (
            generated_replay.get("passed") is True
            and generated_replay.get("live_tool_calls") == 0
            and all(
                episode.get("matched") is True
                for episode in generated_replay.get("episodes", [])
            )
        )
        replay_stored_matches = generated_replay == stored_replay
    except Exception as exc:
        errors.append(f"unable to recompute transcript replay: {exc}")
    if not replay_recomputed:
        errors.append("recomputed replay hashes did not all match")
    if not replay_stored_matches:
        errors.append("stored replay differs from replay recomputed from raw events")

    checks = {
        "event_schema": event_schema_ok,
        "outcome_schema": outcome_schema_ok,
        "manifest_schema": manifest_schema_ok,
        "replay_schema": replay_schema_ok,
        "run_ids_consistent": run_ids_consistent,
        "event_ids_unique": unique_event_ids,
        "episode_cardinality": episode_cardinality_ok,
        "episode_sets_match": episode_sets_match,
        "episode_treatments_consistent": episode_treatments_consistent,
        "cross_file_episode_consistency": cross_file_episode_consistency,
        "smoke_conditions_exact": smoke_conditions_exact,
        "gate_semantics_consistent": gate_semantics_consistent,
        "leak_semantics_consistent": leak_semantics_consistent,
        "task_semantics_consistent": task_semantics_consistent,
        "outcomes_match_frozen_expected": actual_outcomes_match_expected,
        "manifest_smoke_behavior_matches": manifest_smoke_behavior_matches,
        "event_payload_hashes_match": event_payload_hashes_match,
        "tool_state_hashes": tool_hashes,
        "no_hidden_reasoning": no_hidden_reasoning,
        "output_declarations_exact": output_declarations_exact,
        "outputs_exist_and_bounded": outputs_exist_and_bounded,
        "output_types_match": output_types_match,
        "config_hashes_match": config_hashes_match,
        "shared_input_hashes_match": shared_input_hashes_match,
        "message_template_hashes_match": template_hashes_match,
        "source_file_hashes_match": source_file_hashes_match,
        "source_tree_hash_matches": source_hash_matches,
        "workspace_state_matches": workspace_matches,
        "python_environment_matches": environment_matches,
        "version_evidence_consistent": version_evidence_consistent,
        "replay_recomputed_matches": replay_recomputed,
        "stored_replay_matches_raw": replay_stored_matches,
        "replay_live_tool_calls_zero": (
            generated_replay.get("live_tool_calls") == 0
        ),
    }
    summary = {
        "run_id": run_id,
        "passed": not errors and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "event_count": len(events),
        "episode_count": len(outcomes),
    }
    if write_outputs:
        _write_json(artifact_paths["validation"], summary, paths)
        _write_report(artifact_paths["validation_report"], summary, paths)
    if update_manifest and manifest:
        manifest.setdefault("validation", {})
        manifest["validation"]["data_schema_validation"] = summary["passed"]
        manifest["validation"]["validation_checks"] = checks
        for path in (
            artifact_paths["validation"],
            artifact_paths["validation_report"],
        ):
            relative = relative_to_paper_alpha(path, paths.paper_alpha_root)
            if relative not in manifest["output_files"]:
                manifest["output_files"].append(relative)
        _write_json(artifact_paths["manifest"], manifest, paths)
    return summary
