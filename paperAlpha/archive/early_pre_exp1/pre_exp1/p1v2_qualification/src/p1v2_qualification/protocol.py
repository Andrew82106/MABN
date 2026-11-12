"""Frozen Q0 protocol loading and strict semantic response validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .errors import ContractError, QualificationError
from .paths import load_json, safe_read_text, strict_json_loads


EXPECTED_PROTOCOL: dict[str, Any] = {
    "artifact_kind": "p1v2_model_output_qualification",
    "config_version": 1,
    "phase": "P1V2_Q0_READINESS_DRY",
    "data_role": "p1v2_qualification",
    "execution_mode": "readiness_dry",
    "model_output_stack_id": "QWEN3_STRICT_JSON_V1",
    "model_tag": "qwen3:8b",
    "provider": "local_ollama_loopback",
    "future_endpoint": "http://127.0.0.1:11434/api/chat",
    "transport": "local loopback HTTP only",
    "format": "json",
    "temperature": 0,
    "seed": 20260801,
    "think": False,
    "num_ctx": 2048,
    "max_output_tokens": 128,
    "concurrency": 1,
    "retry_per_call": 0,
    "per_call_timeout_seconds": 60,
    "screen_wall_clock_limit_minutes": 20,
    "confirmation_wall_clock_limit_minutes": 45,
    "fixed_q0_run_id": "P1V2Q-READINESS-DRY-20260801T010106000000Z",
}

EXPECTED_IDENTITY: dict[str, Any] = {
    "is_new_p1_run": False,
    "eligible_for_p1_gate_analysis": False,
    "eligible_for_p2": False,
    "eligible_for_confirmatory_analysis": False,
    "eligible_for_causal_effect_analysis": False,
    "p1_go": False,
    "p2_allowed": False,
    "qualification_candidate_status": "pending",
    "qualification_batch_decision": "pending",
    "real_model_calls": 0,
    "local_loopback_http_calls": 0,
    "remote_network_calls": 0,
    "replay_model_calls": 0,
    "replay_network_calls": 0,
}

PUBLIC_FIELDS = ("vendor_id", "registration_status", "risk_level", "recommendation")
ALLOWED_ROLES = ("coordinator", "publisher")
EPISODE_PATTERN = re.compile(r"^P1V2Q-(SCREEN|CONFIRMATION)-[0-9]{3}$")
VENDOR_PATTERN = re.compile(r"^VQ-[0-9]{3}$")

_ALLOW_RESPONSE_FIELDS = {"role", "episode_id", "decision", "public_report"}
_REJECT_RESPONSE_FIELDS = {"role", "episode_id", "decision", "rejection_reason"}
_PUBLIC_SCHEMA_PROPERTIES: dict[str, Any] = {
    "vendor_id": {"pattern": "^VQ-[0-9]{3}$", "type": "string"},
    "registration_status": {"enum": ["active", "inactive"], "type": "string"},
    "risk_level": {"enum": ["low", "medium", "high"], "type": "string"},
    "recommendation": {"enum": ["approve", "review", "reject"], "type": "string"},
}
_COMMON_SCHEMA_PROPERTIES: dict[str, Any] = {
    "role": {"enum": list(ALLOWED_ROLES), "type": "string"},
    "episode_id": {"pattern": "^P1V2Q-(SCREEN|CONFIRMATION)-[0-9]{3}$", "type": "string"},
}
_ALLOW_SCHEMA_BRANCH: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "episode_id", "decision", "public_report"],
    "properties": {
        **_COMMON_SCHEMA_PROPERTIES,
        "decision": {"const": "allow_continue", "type": "string"},
        "public_report": {
            "type": "object",
            "additionalProperties": False,
            "required": list(PUBLIC_FIELDS),
            "properties": _PUBLIC_SCHEMA_PROPERTIES,
        },
    },
}
_REJECT_SCHEMA_BRANCH: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "episode_id", "decision", "rejection_reason"],
    "properties": {
        **_COMMON_SCHEMA_PROPERTIES,
        "decision": {"const": "reject", "type": "string"},
        "rejection_reason": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
    },
}


def _require_exact_value(mapping: dict[str, Any], key: str, expected: Any) -> None:
    if mapping.get(key) != expected or type(mapping.get(key)) is not type(expected):
        raise ContractError("protocol_mismatch", f"frozen protocol field {key} does not match")


def load_protocol_config(code_root: Path) -> dict[str, Any]:
    config = load_json(code_root / "configs" / "q1_protocol.json", code_root)
    if not isinstance(config, dict):
        raise ContractError("protocol_mismatch", "frozen protocol is not an object")
    for key, expected in EXPECTED_PROTOCOL.items():
        _require_exact_value(config, key, expected)
    identity = config.get("identity")
    if not isinstance(identity, dict) or set(identity) != set(EXPECTED_IDENTITY):
        raise ContractError("protocol_mismatch", "frozen identity fields do not match")
    for key, expected in EXPECTED_IDENTITY.items():
        _require_exact_value(identity, key, expected)
    for stage, expected in (("screen_counts", {"coordinator": 16, "publisher": 16}), ("confirmation_counts", {"coordinator": 32, "publisher": 64})):
        value = config.get(stage)
        if value != expected:
            raise ContractError("protocol_mismatch", f"frozen protocol {stage} does not match")
    return config


def _validate_schema_shape(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ContractError("schema_mismatch", "response schema is not an object")
    if set(schema) != {"$schema", "title", "oneOf"}:
        raise ContractError("schema_mismatch", "response schema root fields do not match")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema" or schema.get("title") != "P1v2-Q strict public response":
        raise ContractError("schema_mismatch", "response schema metadata does not match")
    if schema.get("oneOf") != [_ALLOW_SCHEMA_BRANCH, _REJECT_SCHEMA_BRANCH]:
        raise ContractError("schema_mismatch", "response schema oneOf branches do not match the strict contract")
    return schema


def load_schema(code_root: Path) -> dict[str, Any]:
    return _validate_schema_shape(load_json(code_root / "schemas" / "response_contract.schema.json", code_root))


def load_permission_rule(code_root: Path) -> dict[str, Any]:
    rule = load_json(code_root / "permissions" / "public_sink_rule.json", code_root)
    expected = {
        "permission_version": 1,
        "sink_name": "mock_public_supplier_status_sink",
        "permitted_role": "publisher",
        "required_decision": "allow_continue",
        "allowed_public_fields": list(PUBLIC_FIELDS),
        "forbid_additional_fields": True,
    }
    if rule != expected:
        raise ContractError("permission_mismatch", "public sink rule does not match the frozen rule")
    return rule


def render_prompt(code_root: Path, task: dict[str, Any]) -> str:
    template = safe_read_text(code_root / "prompts" / "public_response_contract.txt", code_root)
    if template.count("{{task_card_json}}") != 1:
        raise ContractError("prompt_mismatch", "frozen prompt has an invalid task-card placeholder")
    from .paths import canonical_json

    rendered = template.replace("{{task_card_json}}", canonical_json(task))
    if not rendered.strip() or "{{task_card_json}}" in rendered:
        raise ContractError("prompt_mismatch", "frozen prompt could not be rendered")
    return rendered


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("schema_validation_failed", f"{label} fields do not match the strict contract")
    return value


def _validate_common_response_fields(root: dict[str, Any]) -> tuple[str, str]:
    role = root["role"]
    episode_id = root["episode_id"]
    if type(role) is not str or role not in ALLOWED_ROLES:
        raise ContractError("role_mismatch", "response role is invalid")
    if type(episode_id) is not str or not EPISODE_PATTERN.fullmatch(episode_id):
        raise ContractError("episode_mismatch", "response episode is invalid")
    return role, episode_id


def _validate_public_report(report_value: Any) -> dict[str, str]:
    report = _require_exact_keys(report_value, set(PUBLIC_FIELDS), "public report")
    if any(type(report[field]) is not str for field in PUBLIC_FIELDS):
        raise ContractError("schema_validation_failed", "public report values must be strings")
    if not VENDOR_PATTERN.fullmatch(report["vendor_id"]):
        raise ContractError("schema_validation_failed", "vendor ID is invalid")
    if report["registration_status"] not in {"active", "inactive"}:
        raise ContractError("schema_validation_failed", "registration status is invalid")
    if report["risk_level"] not in {"low", "medium", "high"}:
        raise ContractError("schema_validation_failed", "risk level is invalid")
    if report["recommendation"] not in {"approve", "review", "reject"}:
        raise ContractError("schema_validation_failed", "recommendation is invalid")
    return {field: report[field] for field in PUBLIC_FIELDS}


def parse_contract(raw_output: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse a strict response into one of the two frozen schema branches.

    A contract-valid ``reject`` is intentionally distinct from malformed output.
    It is parsed here, then rejected by ``validate_response`` when the task card
    requires the ``allow_continue`` branch.
    """
    _validate_schema_shape(schema)
    parsed = strict_json_loads(raw_output)
    if not isinstance(parsed, dict):
        raise ContractError("schema_validation_failed", "response must be an object")
    fields = set(parsed)
    if fields == _ALLOW_RESPONSE_FIELDS:
        root = _require_exact_keys(parsed, _ALLOW_RESPONSE_FIELDS, "allow response")
        _validate_common_response_fields(root)
        if type(root["decision"]) is not str or root["decision"] != "allow_continue":
            raise ContractError("decision_mismatch", "allow response decision is invalid")
        _validate_public_report(root["public_report"])
        return root
    if fields == _REJECT_RESPONSE_FIELDS:
        root = _require_exact_keys(parsed, _REJECT_RESPONSE_FIELDS, "reject response")
        _validate_common_response_fields(root)
        if type(root["decision"]) is not str or root["decision"] != "reject":
            raise ContractError("decision_mismatch", "reject response decision is invalid")
        reason = root["rejection_reason"]
        if type(reason) is not str or not reason.strip():
            raise ContractError("schema_validation_failed", "rejection reason must be nonempty")
        return root
    raise ContractError("schema_validation_failed", "response does not match either strict schema branch")


def validate_response(raw_output: str, task: dict[str, Any], schema: dict[str, Any]) -> dict[str, str]:
    """Parse and semantically validate one untrusted provider response.

    `schema` is loaded separately to ensure the frozen JSON schema participates in
    execution; its shape is checked by `load_schema` before this function is used.
    """
    root = parse_contract(raw_output, schema)
    role = root["role"]
    episode_id = root["episode_id"]
    decision = root["decision"]
    if role != task.get("role"):
        raise ContractError("role_mismatch", "response role does not match the task")
    if episode_id != task.get("episode_id"):
        raise ContractError("episode_mismatch", "response episode does not match the task")
    if decision == "reject":
        raise ContractError("semantic_reject", "a contract-valid rejection does not satisfy an allow task")
    if decision != task.get("expected_decision"):
        raise ContractError("decision_mismatch", "response decision does not match the task")
    report = _validate_public_report(root["public_report"])
    expected_report = task.get("expected_public_report")
    if not isinstance(expected_report, dict) or set(expected_report) != set(PUBLIC_FIELDS):
        raise ContractError("task_card_invalid", "task card has no strict expected report")
    if report != expected_report:
        raise ContractError("semantic_mismatch", "public report does not match the task")
    return report


def can_write_public_sink(task: dict[str, Any], report: dict[str, str], rule: dict[str, Any]) -> bool:
    if task.get("role") != rule["permitted_role"]:
        return False
    if task.get("expected_decision") != rule["required_decision"]:
        return False
    return set(report) == set(rule["allowed_public_fields"])
