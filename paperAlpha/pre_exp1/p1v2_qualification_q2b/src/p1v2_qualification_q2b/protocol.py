"""Frozen request construction, model-list normalization, and strict response checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT, canonical_json, load_json, sha256_file, sha256_text
from .errors import ContractViolation


RESPONSE_FIELDS = ("agent_role", "episode_id", "decision", "task_value")
ROLE_VALUES = {"coordinator", "publisher"}
DECISION_VALUES = {"screen_safe", "confirm_safe", "publish_safe"}
EPISODE_RE = re.compile(r"^P1V2Q2B-(SCREEN|CONFIRMATION)-[0-9]{3}$")
FROZEN_ASSET_PATHS = (
    "configs/q2b_remote_protocol.json",
    "configs/q2b_provider_profile.json",
    "prompts/response_contract.txt",
    "schemas/response_contract.schema.json",
    "fixtures/screen_tasks.json",
    "fixtures/confirmation_tasks.json",
    "permissions/public_sink_rule.json",
    "provenance/source_contracts.json",
)


@dataclass(frozen=True)
class FrozenAssets:
    root: Path
    protocol: dict[str, Any]
    provider_profile: dict[str, Any]
    prompt: str
    schema: dict[str, Any]
    permissions: dict[str, Any]
    screen_tasks: list[dict[str, Any]]
    confirmation_tasks: list[dict[str, Any]]
    asset_hashes: dict[str, str]


@dataclass(frozen=True)
class ModelListIdentity:
    normalized_sha256: str
    target_present: bool
    model_count: int


@dataclass(frozen=True)
class AdaptedCompletion:
    parsed: dict[str, str]
    content_sha256: str
    safe_identity: dict[str, str]
    public_sink_record: dict[str, str] | None


def _path(root: Path, relative: str) -> Path:
    return root / relative


def load_frozen_assets(root: Path = PACKAGE_ROOT) -> FrozenAssets:
    protocol = load_json(_path(root, "configs/q2b_remote_protocol.json"))
    profile = load_json(_path(root, "configs/q2b_provider_profile.json"))
    prompt = _path(root, "prompts/response_contract.txt").read_text(encoding="utf-8")
    schema = load_json(_path(root, "schemas/response_contract.schema.json"))
    permissions = load_json(_path(root, "permissions/public_sink_rule.json"))
    screen = load_json(_path(root, "fixtures/screen_tasks.json"))
    confirmation = load_json(_path(root, "fixtures/confirmation_tasks.json"))
    hashes = {relative: sha256_file(_path(root, relative)) for relative in FROZEN_ASSET_PATHS}
    return FrozenAssets(root, protocol, profile, prompt, schema, permissions, screen["tasks"], confirmation["tasks"], hashes)


def validate_task_card(task: dict[str, Any], phase: str | None = None) -> None:
    required = {"episode_id", "phase", "agent_role", "expected_decision", "task_value", "public_sink"}
    if set(task) != required:
        raise ContractViolation("task_card_fields", "infrastructure_failure")
    if task["phase"] not in {"screen", "confirmation"} or (phase is not None and task["phase"] != phase):
        raise ContractViolation("task_card_phase", "infrastructure_failure")
    if task["agent_role"] not in ROLE_VALUES or task["expected_decision"] not in DECISION_VALUES:
        raise ContractViolation("task_card_role_or_decision", "infrastructure_failure")
    if not isinstance(task["episode_id"], str) or not EPISODE_RE.fullmatch(task["episode_id"]):
        raise ContractViolation("task_card_episode", "infrastructure_failure")
    if not isinstance(task["task_value"], str) or not task["task_value"]:
        raise ContractViolation("task_card_value", "infrastructure_failure")
    if task["public_sink"] is not (task["agent_role"] == "publisher"):
        raise ContractViolation("task_card_sink", "infrastructure_failure")


def validate_frozen_assets(assets: FrozenAssets) -> None:
    expected = {
        "provider_kind": "openai_compatible_gateway",
        "base_url_identity": "http://127.0.0.1:58661/v1",
        "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions",
        "model": "gpt-5.3-codex-spark",
        "concurrency": 1,
        "retry_count": 0,
        "stream": False,
        "timeout_seconds": 60,
        "batch_wall_clock_seconds": 1200,
    }
    for key, value in expected.items():
        if assets.protocol.get(key) != value:
            raise ContractViolation("frozen_protocol_drift", "infrastructure_failure")
    if assets.protocol.get("stage_counts") != {"screen_gate": 8, "screen_total": 32, "confirmation_total": 96}:
        raise ContractViolation("stage_counts_contract", "infrastructure_failure")
    if assets.protocol.get("budget") != {"max_models_calls": 2, "max_completion_calls": 128}:
        raise ContractViolation("budget_contract", "infrastructure_failure")
    expected_optional = {"temperature": "not_sent", "top_p": "not_sent", "max_tokens": "not_sent", "seed": "not_sent"}
    if assets.protocol.get("request_contract", {}).get("optional_decoding_fields") != expected_optional:
        raise ContractViolation("optional_fields_contract", "infrastructure_failure")
    if len(assets.screen_tasks) != 32 or len(assets.confirmation_tasks) != 96:
        raise ContractViolation("fixture_count", "infrastructure_failure")
    for task in assets.screen_tasks:
        validate_task_card(task, "screen")
    for task in assets.confirmation_tasks:
        validate_task_card(task, "confirmation")
    if assets.permissions != {
        "rule_version": "1.0.0",
        "public_sink_role": "publisher",
        "allowed_fields": list(RESPONSE_FIELDS),
        "coordinator_may_write_public_sink": False,
        "unstarted_cards_may_write_public_sink": False,
    }:
        raise ContractViolation("permissions_contract", "infrastructure_failure")


def build_request(task: dict[str, Any], assets: FrozenAssets) -> dict[str, Any]:
    validate_task_card(task)
    return {
        "model": assets.protocol["model"],
        "stream": False,
        "messages": [{"role": "user", "content": assets.prompt.rstrip() + "\n\nTASK_CARD=" + canonical_json(task)}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "p1v2_q2b_response", "strict": True, "schema": assets.schema},
        },
    }


def parse_model_list(raw_body: bytes, target_model: str) -> ModelListIdentity:
    try:
        value = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("models_response_json", "infrastructure_failure") from exc
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise ContractViolation("models_response_shape", "infrastructure_failure")
    ids: list[str] = []
    for item in value["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ContractViolation("models_response_shape", "infrastructure_failure")
        ids.append(item["id"])
    normalized = {"model_ids": sorted(set(ids))}
    return ModelListIdentity(sha256_text(canonical_json(normalized)), target_model in ids, len(ids))


def _safe_string(value: Any) -> str:
    return value if isinstance(value, str) and value else "not_provided"


def _validate_parsed(parsed: Any, task: dict[str, Any]) -> dict[str, str]:
    if not isinstance(parsed, dict) or set(parsed) != set(RESPONSE_FIELDS):
        raise ContractViolation("response_schema", "model_failure")
    if any(not isinstance(parsed[field], str) for field in RESPONSE_FIELDS):
        raise ContractViolation("response_schema", "model_failure")
    if parsed["agent_role"] not in ROLE_VALUES or parsed["decision"] not in DECISION_VALUES:
        raise ContractViolation("response_schema", "model_failure")
    if not EPISODE_RE.fullmatch(parsed["episode_id"]):
        raise ContractViolation("response_schema", "model_failure")
    if not (1 <= len(parsed["task_value"]) <= 160):
        raise ContractViolation("response_schema", "model_failure")
    expected = {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }
    for key, expected_value in expected.items():
        if parsed[key] != expected_value:
            raise ContractViolation("response_" + key, "model_failure")
    return {field: parsed[field] for field in RESPONSE_FIELDS}


def adapt_completion(raw_body: bytes, headers: dict[str, str], task: dict[str, Any], assets: FrozenAssets) -> AdaptedCompletion:
    """Strictly adapt an HTTP-200 body; rejected raw content never leaves this function."""
    try:
        envelope = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("response_envelope_json", "infrastructure_failure") from exc
    if not isinstance(envelope, dict):
        raise ContractViolation("unknown_envelope", "infrastructure_failure")
    if "error" in envelope:
        raise ContractViolation("provider_error", "infrastructure_failure")
    declared_model = envelope.get("model")
    if not isinstance(declared_model, str) or declared_model != assets.protocol["model"]:
        raise ContractViolation("model_mismatch", "infrastructure_failure")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ContractViolation("missing_choices", "model_failure")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ContractViolation("unknown_envelope", "model_failure")
    if "tool_calls" in message:
        raise ContractViolation("tool_calls", "model_failure")
    if "function_call" in message:
        raise ContractViolation("function_calls", "model_failure")
    if "refusal" in message:
        raise ContractViolation("refusal", "model_failure")
    if "reasoning_content" in message:
        raise ContractViolation("reasoning_detected", "model_failure")
    if "reasoning" in message and message["reasoning"] not in (None, ""):
        raise ContractViolation("reasoning_detected", "model_failure")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ContractViolation("empty_content", "model_failure")
    if "<think>" in content.lower():
        raise ContractViolation("reasoning_detected", "model_failure")
    try:
        parsed = _validate_parsed(json.loads(content), task)
    except json.JSONDecodeError as exc:
        raise ContractViolation("response_content_json", "model_failure") from exc
    request_id = _safe_string(headers.get("x-request-id") or headers.get("request-id"))
    safe_identity = {
        "provider_declared_model": declared_model,
        "request_id": request_id,
        "system_fingerprint": _safe_string(envelope.get("system_fingerprint")),
        "provider_version": _safe_string(envelope.get("provider_version")),
    }
    return AdaptedCompletion(
        parsed=parsed,
        content_sha256=sha256_text(content),
        safe_identity=safe_identity,
        public_sink_record=dict(parsed) if task["agent_role"] == "publisher" else None,
    )
