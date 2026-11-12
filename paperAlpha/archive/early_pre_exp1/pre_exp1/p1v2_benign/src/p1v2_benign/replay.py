"""Pure, fail-closed transcript replay of recorded offline fixture responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import ContractPolicy, load_schema_contract
from .core import (
    DEFAULT_DATA_ROOT,
    artifact_paths,
    canonical_json_bytes,
    is_sha256,
    read_jsonl,
    read_strict_json,
    require_p1v2_data_root,
    resolve_generated_path,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from .workflow import MockPublicSink, evaluate_response


EVENT_KEYS = {
    "case_id",
    "data_role",
    "event_id",
    "expected_episode_id",
    "expected_role",
    "raw_response",
    "raw_response_sha256",
    "sequence",
}
OUTCOME_KEYS = {
    "case_id",
    "contract_valid",
    "data_role",
    "decision",
    "public_report_sha256",
    "public_sink_published",
    "rejection_code",
    "sequence",
    "source_event_id",
}
MANIFEST_KEYS = {
    "artifact_hashes",
    "artifact_kind",
    "artifacts",
    "contract_policy",
    "data_role",
    "deterministic_fixture_resolutions",
    "eligible_for_causal_effect_analysis",
    "eligible_for_confirmatory_analysis",
    "eligible_for_p1_gate_analysis",
    "eligible_for_p2",
    "execution_inputs",
    "execution_source",
    "expected_cases",
    "expected_public_sink_case_ids",
    "external_provider_calls",
    "is_new_p1_run",
    "network_calls",
    "p1_go",
    "p2_allowed",
    "provenance",
    "real_model_calls",
    "run_id",
    "scope",
}
IDENTITY_KEYS = {"relative_path", "sha256", "size_bytes"}


def _error(errors: list[dict[str, str]], code: str) -> None:
    errors.append({"code": code})


def _safe_run_id(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _result(run_id: Any, errors: list[dict[str, str]], replayed_case_count: int = 0) -> dict[str, Any]:
    return {
        "artifact_kind": "p1v2_readiness_dry_replay",
        "deterministic_fixture_resolutions": 0,
        "errors": errors,
        "external_provider_calls": 0,
        "network_calls": 0,
        "passed": not errors,
        "real_model_calls": 0,
        "replayed_case_count": replayed_case_count,
        "run_id": _safe_run_id(run_id),
    }


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _identity_is_well_formed(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == IDENTITY_KEYS
        and isinstance(value["relative_path"], str)
        and value["relative_path"].startswith("pre_exp1/p1v2_benign/")
        and is_sha256(value["sha256"])
        and _is_exact_int(value["size_bytes"])
        and value["size_bytes"] >= 0
    )


def _execution_inputs_are_well_formed(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"config", "entry_scripts", "fixture", "prompt", "schema", "source_modules"}:
        return False
    if not all(_identity_is_well_formed(value[name]) for name in ("config", "fixture", "prompt", "schema")):
        return False
    scripts = value["entry_scripts"]
    modules = value["source_modules"]
    return (
        isinstance(scripts, list)
        and len(scripts) == 3
        and all(_identity_is_well_formed(item) for item in scripts)
        and isinstance(modules, list)
        and bool(modules)
        and all(_identity_is_well_formed(item) for item in modules)
    )


def _provenance_is_well_formed(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"algorithm", "files", "tree_sha256"}:
        return False
    files = value["files"]
    return (
        value["algorithm"] == "sha256"
        and isinstance(files, list)
        and bool(files)
        and is_sha256(value["tree_sha256"])
        and all(
            isinstance(item, dict)
            and set(item) == {"path", "sha256", "size_bytes"}
            and isinstance(item["path"], str)
            and item["path"].startswith("pre_exp1/p1v2_benign/")
            and is_sha256(item["sha256"])
            and _is_exact_int(item["size_bytes"])
            and item["size_bytes"] >= 0
            for item in files
        )
    )


def _parse_expected_cases(value: Any, errors: list[dict[str, str]]) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        _error(errors, "expected_cases_invalid")
        return None
    parsed: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"case_id", "expected_episode_id", "expected_role"}:
            _error(errors, "expected_case_structure_invalid")
            return None
        if not all(_nonempty_string(item[key]) for key in ("case_id", "expected_episode_id", "expected_role")):
            _error(errors, "expected_case_value_invalid")
            return None
        parsed.append(
            {
                "case_id": item["case_id"],
                "expected_episode_id": item["expected_episode_id"],
                "expected_role": item["expected_role"],
            }
        )
    identifiers = [item["case_id"] for item in parsed]
    if len(identifiers) != len(set(identifiers)):
        _error(errors, "duplicate_expected_case_id")
        return None
    return parsed


def _parse_events(value: Any, errors: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        _error(errors, "events_invalid")
        return None
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != EVENT_KEYS:
            _error(errors, "event_structure_invalid")
            return None
        string_keys = (
            "case_id",
            "data_role",
            "event_id",
            "expected_episode_id",
            "expected_role",
            "raw_response",
            "raw_response_sha256",
        )
        if not all(_nonempty_string(item[key]) for key in string_keys) or not _is_exact_int(item["sequence"]):
            _error(errors, "event_value_invalid")
            return None
        if item["sequence"] != index:
            _error(errors, "event_sequence_invalid")
            return None
        if not is_sha256(item["event_id"]) or not is_sha256(item["raw_response_sha256"]):
            _error(errors, "event_hash_format_invalid")
            return None
        body = {key: item[key] for key in item if key != "event_id"}
        if item["event_id"] != sha256_bytes(canonical_json_bytes(body)):
            _error(errors, "event_id_mismatch")
            return None
        if item["raw_response_sha256"] != sha256_text(item["raw_response"]):
            _error(errors, "event_raw_response_hash_mismatch")
            return None
        parsed.append(dict(item))
    identifiers = [item["case_id"] for item in parsed]
    if len(identifiers) != len(set(identifiers)):
        _error(errors, "duplicate_event_case_id")
        return None
    return parsed


def _parse_outcomes(value: Any, errors: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        _error(errors, "outcomes_invalid")
        return None
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != OUTCOME_KEYS:
            _error(errors, "outcome_structure_invalid")
            return None
        string_keys = ("case_id", "data_role", "decision", "source_event_id")
        if not all(_nonempty_string(item[key]) for key in string_keys) or not _is_exact_int(item["sequence"]):
            _error(errors, "outcome_value_invalid")
            return None
        if item["sequence"] != index:
            _error(errors, "outcome_sequence_invalid")
            return None
        if type(item["contract_valid"]) is not bool or type(item["public_sink_published"]) is not bool:
            _error(errors, "outcome_boolean_invalid")
            return None
        if not is_sha256(item["source_event_id"]):
            _error(errors, "outcome_source_event_id_invalid")
            return None
        report_hash = item["public_report_sha256"]
        if report_hash is not None and not is_sha256(report_hash):
            _error(errors, "outcome_report_hash_invalid")
            return None
        if item["rejection_code"] is not None and not _nonempty_string(item["rejection_code"]):
            _error(errors, "outcome_rejection_code_invalid")
            return None
        parsed.append(dict(item))
    identifiers = [item["case_id"] for item in parsed]
    if len(identifiers) != len(set(identifiers)):
        _error(errors, "duplicate_outcome_case_id")
        return None
    return parsed


def _manifest_is_well_formed(
    manifest: Any,
    *,
    run_id: str,
    expected_paths: dict[str, str],
    allow_provisional_artifact_hashes: bool,
    errors: list[dict[str, str]],
) -> tuple[ContractPolicy, list[dict[str, str]]] | None:
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        _error(errors, "manifest_structure_invalid")
        return None
    identity_expectations = {
        "artifact_kind": "p1v2_readiness_dry",
        "data_role": "p1v2_readiness_dry",
        "eligible_for_causal_effect_analysis": False,
        "eligible_for_confirmatory_analysis": False,
        "eligible_for_p1_gate_analysis": False,
        "eligible_for_p2": False,
        "execution_source": "deterministic_fixture",
        "external_provider_calls": 0,
        "is_new_p1_run": False,
        "network_calls": 0,
        "p1_go": False,
        "p2_allowed": False,
        "real_model_calls": 0,
        "run_id": run_id,
        "scope": "offline_output_contract_admission_only",
    }
    for key, expected in identity_expectations.items():
        if manifest[key] != expected:
            _error(errors, f"identity_mismatch_{key}")
    if not _is_exact_int(manifest["deterministic_fixture_resolutions"]) or manifest["deterministic_fixture_resolutions"] < 1:
        _error(errors, "fixture_resolution_count_invalid")
    if manifest["artifacts"] != expected_paths:
        _error(errors, "artifact_routing_mismatch")
        return None
    expected_hash_names = {"events", "outcomes"} if allow_provisional_artifact_hashes else {"events", "outcomes", "validation", "replay", "report"}
    hashes = manifest["artifact_hashes"]
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_names or not all(is_sha256(item) for item in hashes.values()):
        _error(errors, "artifact_hash_structure_invalid")
        return None
    if not _execution_inputs_are_well_formed(manifest["execution_inputs"]):
        _error(errors, "execution_inputs_invalid")
        return None
    if not _provenance_is_well_formed(manifest["provenance"]):
        _error(errors, "provenance_structure_invalid")
        return None
    expected_cases = _parse_expected_cases(manifest["expected_cases"], errors)
    if expected_cases is None:
        return None
    published = manifest["expected_public_sink_case_ids"]
    if (
        not isinstance(published, list)
        or any(not _nonempty_string(item) for item in published)
        or len(published) != len(set(published))
    ):
        _error(errors, "expected_public_sink_case_ids_invalid")
        return None
    try:
        policy = ContractPolicy.from_mapping(manifest["contract_policy"])
    except Exception:
        _error(errors, "contract_policy_invalid")
        return None
    return policy, expected_cases


def _replay_impl(*, data_root: Path | str, run_id: str, allow_provisional_artifact_hashes: bool) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    root = require_p1v2_data_root(data_root)
    expected_paths = artifact_paths(run_id)
    manifest_path = resolve_generated_path(root, expected_paths["manifest"])
    manifest = read_strict_json(manifest_path, allowed_root=root)
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
    hashes = manifest["artifact_hashes"]
    for name, expected_hash in hashes.items():
        path = resolve_generated_path(root, expected_paths[name])
        if sha256_file(path, allowed_root=root) != expected_hash:
            _error(errors, "artifact_hash_mismatch")
            return _result(run_id, errors)
    events = _parse_events(read_jsonl(resolve_generated_path(root, expected_paths["events"]), allowed_root=root), errors)
    outcomes = _parse_outcomes(read_jsonl(resolve_generated_path(root, expected_paths["outcomes"]), allowed_root=root), errors)
    if events is None or outcomes is None:
        return _result(run_id, errors)
    resolution_count = manifest["deterministic_fixture_resolutions"]
    if resolution_count != len(expected_cases) or resolution_count != len(events) or resolution_count != len(outcomes):
        _error(errors, "fixture_resolution_closure_mismatch")
        return _result(run_id, errors)
    expected_case_ids = [item["case_id"] for item in expected_cases]
    event_case_ids = [item["case_id"] for item in events]
    outcome_case_ids = [item["case_id"] for item in outcomes]
    if expected_case_ids != event_case_ids or expected_case_ids != outcome_case_ids:
        _error(errors, "case_sequence_closure_mismatch")
        return _result(run_id, errors)
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
            return _result(run_id, errors)
    try:
        schema = load_schema_contract()
    except Exception:
        _error(errors, "schema_contract_unavailable")
        return _result(run_id, errors)
    sink = MockPublicSink()
    replayed_case_ids: list[str] = []
    for expected, event, stored in zip(expected_cases, events, outcomes, strict=True):
        replayed = evaluate_response(
            raw_response=event["raw_response"],
            expected_role=expected["expected_role"],
            expected_episode_id=expected["expected_episode_id"],
            policy=policy,
            schema=schema,
            sink=sink,
        )
        expected_outcome = {
            "contract_valid": replayed["contract_valid"],
            "decision": replayed["decision"],
            "public_report_sha256": replayed["public_report_sha256"],
            "public_sink_published": replayed["public_sink_published"],
            "rejection_code": replayed["rejection_code"],
        }
        if any(stored[key] != expected_value for key, expected_value in expected_outcome.items()):
            _error(errors, "outcome_semantic_mismatch")
            return _result(run_id, errors)
        replayed_case_ids.append(event["case_id"])
    actual_published = [
        outcome["case_id"]
        for outcome in outcomes
        if outcome["public_sink_published"]
    ]
    if manifest["expected_public_sink_case_ids"] != actual_published or len(sink.published_reports) != len(actual_published):
        _error(errors, "public_sink_summary_mismatch")
    return _result(run_id, errors, len(replayed_case_ids))


def replay_run(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    run_id: str,
    allow_provisional_artifact_hashes: bool = False,
) -> dict[str, Any]:
    """Replay stored records only; it never loads a fixture source or writes files."""

    try:
        return _replay_impl(
            data_root=data_root,
            run_id=run_id,
            allow_provisional_artifact_hashes=allow_provisional_artifact_hashes,
        )
    except Exception:
        return _result(run_id, [{"code": "replay_input_invalid"}])
