"""Fail-closed validation for untrusted P1v2-A offline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import ContractPolicy
from .core import (
    DEFAULT_DATA_ROOT,
    artifact_paths,
    capture_execution_inputs,
    capture_provenance,
    load_config,
    read_jsonl,
    read_strict_json,
    require_p1v2_data_root,
    resolve_generated_path,
    sha256_file,
)
from .replay import (
    _manifest_is_well_formed,
    _parse_events,
    _parse_outcomes,
    replay_run,
)


def _error(errors: list[dict[str, str]], code: str) -> None:
    errors.append({"code": code})


def _safe_run_id(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _result(run_id: Any, errors: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "artifact_kind": "p1v2_readiness_dry_validation",
        "deterministic_fixture_resolutions": 0,
        "errors": errors,
        "external_provider_calls": 0,
        "network_calls": 0,
        "passed": not errors,
        "real_model_calls": 0,
        "run_id": _safe_run_id(run_id),
    }


def _closure_is_valid(
    *,
    manifest: dict[str, Any],
    expected_cases: list[dict[str, str]],
    events: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> bool:
    resolution_count = manifest["deterministic_fixture_resolutions"]
    if resolution_count != len(expected_cases) or resolution_count != len(events) or resolution_count != len(outcomes):
        _error(errors, "fixture_resolution_closure_mismatch")
        return False
    expected_case_ids = [item["case_id"] for item in expected_cases]
    event_case_ids = [item["case_id"] for item in events]
    outcome_case_ids = [item["case_id"] for item in outcomes]
    if expected_case_ids != event_case_ids or expected_case_ids != outcome_case_ids:
        _error(errors, "case_sequence_closure_mismatch")
        return False
    for expected, event, outcome in zip(expected_cases, events, outcomes, strict=True):
        if (
            event["expected_role"] != expected["expected_role"]
            or event["expected_episode_id"] != expected["expected_episode_id"]
            or event["data_role"] != "p1v2_readiness_dry"
            or outcome["data_role"] != "p1v2_readiness_dry"
            or outcome["source_event_id"] != event["event_id"]
            or outcome["sequence"] != event["sequence"]
        ):
            _error(errors, "event_outcome_context_mismatch")
            return False
    return True


def _validate_impl(
    *,
    data_root: Path | str,
    run_id: str,
    allow_provisional_artifact_hashes: bool,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    root = require_p1v2_data_root(data_root)
    expected_paths = artifact_paths(run_id)
    manifest = read_strict_json(
        resolve_generated_path(root, expected_paths["manifest"]),
        allowed_root=root,
    )
    parsed_manifest = _manifest_is_well_formed(
        manifest,
        run_id=run_id,
        expected_paths=expected_paths,
        allow_provisional_artifact_hashes=allow_provisional_artifact_hashes,
        errors=errors,
    )
    if parsed_manifest is None:
        return _result(run_id, errors)
    policy, expected_cases = parsed_manifest
    for name, expected_hash in manifest["artifact_hashes"].items():
        path = resolve_generated_path(root, expected_paths[name])
        if sha256_file(path, allowed_root=root) != expected_hash:
            _error(errors, "artifact_hash_mismatch")
            return _result(run_id, errors)
    events = _parse_events(
        read_jsonl(resolve_generated_path(root, expected_paths["events"]), allowed_root=root),
        errors,
    )
    outcomes = _parse_outcomes(
        read_jsonl(resolve_generated_path(root, expected_paths["outcomes"]), allowed_root=root),
        errors,
    )
    if events is None or outcomes is None:
        return _result(run_id, errors)
    if not _closure_is_valid(
        manifest=manifest,
        expected_cases=expected_cases,
        events=events,
        outcomes=outcomes,
        errors=errors,
    ):
        return _result(run_id, errors)
    try:
        current_config = load_config()
        current_policy = ContractPolicy.from_mapping(
            {
                "allow_continue_value": current_config["allow_continue_value"],
                "allowed_roles": current_config["allowed_roles"],
                "reject_value": current_config["reject_value"],
                "required_public_fields": current_config["required_public_fields"],
            }
        ).as_mapping()
        if manifest["contract_policy"] != current_policy:
            _error(errors, "recorded_config_policy_mismatch")
        if [item["case_id"] for item in expected_cases] != current_config["expected_case_ids"]:
            _error(errors, "expected_case_config_mismatch")
    except Exception:
        _error(errors, "active_config_invalid")
    try:
        if manifest["execution_inputs"] != capture_execution_inputs():
            _error(errors, "execution_input_provenance_mismatch")
    except Exception:
        _error(errors, "execution_input_provenance_unavailable")
    try:
        if manifest["provenance"] != capture_provenance():
            _error(errors, "provenance_mismatch")
    except Exception:
        _error(errors, "provenance_unavailable")
    replay = replay_run(
        data_root=root,
        run_id=run_id,
        allow_provisional_artifact_hashes=allow_provisional_artifact_hashes,
    )
    if replay["passed"] is not True:
        _error(errors, "fresh_replay_failed")
    return _result(run_id, errors)


def validate_run(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    run_id: str,
    allow_provisional_artifact_hashes: bool = False,
) -> dict[str, Any]:
    """Validate a recorded artifact without fixture resolution or any writer I/O."""

    try:
        return _validate_impl(
            data_root=data_root,
            run_id=run_id,
            allow_provisional_artifact_hashes=allow_provisional_artifact_hashes,
        )
    except Exception:
        return _result(run_id, [{"code": "manifest_input_invalid"}])
