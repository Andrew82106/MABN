"""The sole writer for a deterministic P1v2-A offline readiness artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import ContractPolicy, SchemaContract, policy_matches_schema, schema_contract_from_mapping
from .core import (
    CODE_NAMESPACE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_FIXTURE_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_RUN_ID,
    DEFAULT_SCHEMA_PATH,
    artifact_paths,
    canonical_json_bytes,
    capture_execution_inputs,
    capture_provenance,
    ensure_new_output_paths,
    load_config_with_identity,
    read_canonical_json_with_identity,
    read_canonical_text_with_identity,
    require_p1v2_data_root,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
    write_text,
)
from .workflow import MockPublicSink, evaluate_response


FIXTURE_CASE_KEYS = {"case_id", "expected_episode_id", "expected_role", "raw_response"}


def _validate_fixture_cases(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid_fixture_cases")
    cases: list[dict[str, str]] = []
    seen_case_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != FIXTURE_CASE_KEYS:
            raise ValueError("invalid_fixture_case_keys")
        if any(not isinstance(item[key], str) or not item[key] for key in FIXTURE_CASE_KEYS):
            raise ValueError("invalid_fixture_case_value")
        case_id = item["case_id"]
        if case_id in seen_case_ids:
            raise ValueError("duplicate_fixture_case_id")
        seen_case_ids.add(case_id)
        cases.append({key: item[key] for key in FIXTURE_CASE_KEYS})
    return cases


def load_fixture_cases_with_identity() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Resolve only the frozen fixture under the P1v2 code namespace."""

    value, identity = read_canonical_json_with_identity(DEFAULT_FIXTURE_PATH, DEFAULT_FIXTURE_PATH)
    return _validate_fixture_cases(value), identity


def load_fixture_cases() -> list[dict[str, str]]:
    return load_fixture_cases_with_identity()[0]


def _policy_from_config(config: dict[str, Any]) -> ContractPolicy:
    return ContractPolicy.from_mapping(
        {
            "allow_continue_value": config["allow_continue_value"],
            "allowed_roles": config["allowed_roles"],
            "reject_value": config["reject_value"],
            "required_public_fields": config["required_public_fields"],
        }
    )


def _build_dry_bundle() -> tuple[
    dict[str, Any],
    ContractPolicy,
    list[dict[str, Any]],
    list[dict[str, Any]],
    MockPublicSink,
    dict[str, Any],
]:
    """Freeze canonical inputs once, then produce only deterministic records."""

    config, config_identity = load_config_with_identity()
    cases, fixture_identity = load_fixture_cases_with_identity()
    schema_value, schema_identity = read_canonical_json_with_identity(DEFAULT_SCHEMA_PATH, DEFAULT_SCHEMA_PATH)
    schema: SchemaContract = schema_contract_from_mapping(schema_value)
    _, prompt_identity = read_canonical_text_with_identity(DEFAULT_PROMPT_PATH, DEFAULT_PROMPT_PATH)
    case_ids = [case["case_id"] for case in cases]
    if case_ids != config["expected_case_ids"]:
        raise ValueError("fixture_cases_do_not_match_active_config")
    policy = _policy_from_config(config)
    if not policy_matches_schema(policy, schema):
        raise ValueError("config_schema_mismatch")
    events: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    sink = MockPublicSink()
    for sequence, case in enumerate(cases, start=1):
        event_body = {
            "case_id": case["case_id"],
            "data_role": config["data_role"],
            "expected_episode_id": case["expected_episode_id"],
            "expected_role": case["expected_role"],
            "raw_response": case["raw_response"],
            "raw_response_sha256": sha256_text(case["raw_response"]),
            "sequence": sequence,
        }
        event = {"event_id": sha256_bytes(canonical_json_bytes(event_body)), **event_body}
        events.append(event)
        decision = evaluate_response(
            raw_response=case["raw_response"],
            expected_role=case["expected_role"],
            expected_episode_id=case["expected_episode_id"],
            policy=policy,
            schema=schema,
            sink=sink,
        )
        outcomes.append(
            {
                "case_id": case["case_id"],
                "contract_valid": decision["contract_valid"],
                "data_role": config["data_role"],
                "decision": decision["decision"],
                "public_report_sha256": decision["public_report_sha256"],
                "public_sink_published": decision["public_sink_published"],
                "rejection_code": decision["rejection_code"],
                "sequence": sequence,
                "source_event_id": event["event_id"],
            }
        )
    inputs = capture_execution_inputs(
        config_identity=config_identity,
        fixture_identity=fixture_identity,
        prompt_identity=prompt_identity,
        schema_identity=schema_identity,
    )
    return config, policy, events, outcomes, sink, inputs


def build_dry_records() -> tuple[dict[str, Any], ContractPolicy, list[dict[str, Any]], list[dict[str, Any]], MockPublicSink]:
    """Public read-only record builder using only frozen canonical inputs."""

    config, policy, events, outcomes, sink, _ = _build_dry_bundle()
    return config, policy, events, outcomes, sink


def _manifest(
    *,
    run_id: str,
    config: dict[str, Any],
    policy: ContractPolicy,
    events: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    execution_inputs: dict[str, Any],
    output_paths: dict[str, Path],
    data_root: Path,
) -> dict[str, Any]:
    published_case_ids = [row["case_id"] for row in outcomes if row["public_sink_published"]]
    return {
        "artifact_hashes": {
            "events": sha256_file(output_paths["events"], allowed_root=data_root),
            "outcomes": sha256_file(output_paths["outcomes"], allowed_root=data_root),
        },
        "artifact_kind": "p1v2_readiness_dry",
        "artifacts": artifact_paths(run_id),
        "contract_policy": policy.as_mapping(),
        "data_role": config["data_role"],
        "deterministic_fixture_resolutions": len(events),
        "eligible_for_causal_effect_analysis": False,
        "eligible_for_confirmatory_analysis": False,
        "eligible_for_p1_gate_analysis": False,
        "eligible_for_p2": False,
        "execution_inputs": execution_inputs,
        "execution_source": config["execution_source"],
        "expected_cases": [
            {
                "case_id": event["case_id"],
                "expected_episode_id": event["expected_episode_id"],
                "expected_role": event["expected_role"],
            }
            for event in events
        ],
        "expected_public_sink_case_ids": published_case_ids,
        "external_provider_calls": 0,
        "is_new_p1_run": False,
        "network_calls": 0,
        "p1_go": False,
        "p2_allowed": False,
        "provenance": capture_provenance(),
        "real_model_calls": 0,
        "run_id": run_id,
        "scope": "offline_output_contract_admission_only",
    }


def _run_report(run_id: str, outcomes: list[dict[str, Any]]) -> str:
    published_count = sum(1 for row in outcomes if row["public_sink_published"])
    rejected_count = len(outcomes) - published_count
    return "\n".join(
        [
            "# P1v2-A offline readiness dry-run",
            "",
            f"- Dry-run ID: `{run_id}`",
            "- Data role: `p1v2_readiness_dry`.",
            "- Scope: offline output-contract admission only; not a new P1 run.",
            "- Execution source: frozen deterministic fixture text only.",
            "- Real model calls: 0; external provider calls: 0; network calls: 0.",
            "- P1/P2/confirmatory/causal eligibility: all false; `p1_go=false`; `p2_allowed=false`.",
            f"- Contract cases: {len(outcomes)}; fixture resolutions: {len(outcomes)}; allowed public sink records: {published_count}; rejected records: {rejected_count}.",
            "- Rejected fixture responses were retained only as raw offline test evidence and were not published.",
            "",
        ]
    )


def create_dry_run(*, data_root: Path = DEFAULT_DATA_ROOT, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    """Create one new non-overwritable artifact using only canonical frozen inputs."""

    root = require_p1v2_data_root(data_root)
    output_paths = ensure_new_output_paths(root, run_id)
    config, policy, events, outcomes, _, execution_inputs = _build_dry_bundle()
    write_jsonl(output_paths["events"], events, allowed_root=root)
    write_jsonl(output_paths["outcomes"], outcomes, allowed_root=root)
    manifest = _manifest(
        run_id=run_id,
        config=config,
        policy=policy,
        events=events,
        outcomes=outcomes,
        execution_inputs=execution_inputs,
        output_paths=output_paths,
        data_root=root,
    )
    write_json(output_paths["manifest"], manifest, allowed_root=root)

    from .replay import replay_run
    from .validation import validate_run

    replay = replay_run(data_root=root, run_id=run_id, allow_provisional_artifact_hashes=True)
    if not replay["passed"]:
        raise RuntimeError("new_dry_run_failed_replay")
    write_json(output_paths["replay"], replay, allowed_root=root)
    write_text(output_paths["report"], _run_report(run_id, outcomes), allowed_root=root)
    validation = validate_run(data_root=root, run_id=run_id, allow_provisional_artifact_hashes=True)
    if not validation["passed"]:
        raise RuntimeError("new_dry_run_failed_self_check")
    write_json(output_paths["validation"], validation, allowed_root=root)
    manifest["artifact_hashes"] = {
        name: sha256_file(output_paths[name], allowed_root=root)
        for name in ("events", "outcomes", "validation", "replay", "report")
    }
    write_json(output_paths["manifest"], manifest, allowed_root=root)
    final_validation = validate_run(data_root=root, run_id=run_id)
    if not final_validation["passed"]:
        raise RuntimeError("new_dry_run_failed_final_validation")
    return {
        "artifact_kind": "p1v2_readiness_dry",
        "data_root": str(root),
        "passed": True,
        "replay_passed": replay["passed"],
        "run_id": run_id,
        "validation_passed": final_validation["passed"],
    }
