"""Frozen request construction and strict, offline response adaptation.

There is deliberately no network client in this module.  The adapter accepts only
serialized mock envelopes supplied by an injected in-process transport.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT, canonical_json, load_json, sha256_file, sha256_text
from .errors import ProtocolViolation


RESPONSE_FIELDS = ("agent_role", "episode_id", "decision", "task_value")
ROLE_VALUES = {"coordinator", "publisher"}
DECISION_VALUES = {"screen_safe", "confirm_safe", "publish_safe"}
EPISODE_RE = re.compile(r"^P1V2Q2A-(SCREEN|CONFIRMATION)-[0-9]{3}$")


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
class AdaptedResponse:
    parsed: dict[str, str]
    content_sha256: str
    safe_identity: dict[str, str]
    public_sink_record: dict[str, str] | None


def _asset_path(root: Path, relative: str) -> Path:
    return root / relative


FROZEN_ASSET_PATHS = (
    "configs/q2a_remote_protocol.json",
    "configs/q2a_provider_profile.json",
    "prompts/response_contract.txt",
    "schemas/response_contract.schema.json",
    "fixtures/screen_tasks.json",
    "fixtures/confirmation_tasks.json",
    "permissions/public_sink_rule.json",
    "provenance/source_contracts.json",
)


def load_frozen_assets(root: Path = PACKAGE_ROOT) -> FrozenAssets:
    protocol = load_json(_asset_path(root, "configs/q2a_remote_protocol.json"))
    provider_profile = load_json(_asset_path(root, "configs/q2a_provider_profile.json"))
    prompt = _asset_path(root, "prompts/response_contract.txt").read_text(encoding="utf-8")
    schema = load_json(_asset_path(root, "schemas/response_contract.schema.json"))
    permissions = load_json(_asset_path(root, "permissions/public_sink_rule.json"))
    screen_fixture = load_json(_asset_path(root, "fixtures/screen_tasks.json"))
    confirmation_fixture = load_json(_asset_path(root, "fixtures/confirmation_tasks.json"))
    hashes = {relative: sha256_file(_asset_path(root, relative)) for relative in FROZEN_ASSET_PATHS}
    return FrozenAssets(
        root=root,
        protocol=protocol,
        provider_profile=provider_profile,
        prompt=prompt,
        schema=schema,
        permissions=permissions,
        screen_tasks=screen_fixture["tasks"],
        confirmation_tasks=confirmation_fixture["tasks"],
        asset_hashes=hashes,
    )


def validate_task_card(task: dict[str, Any], expected_phase: str | None = None) -> None:
    required = {"episode_id", "phase", "agent_role", "expected_decision", "task_value", "public_sink"}
    if set(task) != required:
        raise ProtocolViolation("task_card_fields")
    if task["phase"] not in {"screen", "confirmation"}:
        raise ProtocolViolation("task_card_phase")
    if expected_phase is not None and task["phase"] != expected_phase:
        raise ProtocolViolation("task_card_phase")
    if task["agent_role"] not in ROLE_VALUES:
        raise ProtocolViolation("task_card_role")
    if task["expected_decision"] not in DECISION_VALUES:
        raise ProtocolViolation("task_card_decision")
    if not isinstance(task["episode_id"], str) or not EPISODE_RE.fullmatch(task["episode_id"]):
        raise ProtocolViolation("task_card_episode")
    if not isinstance(task["task_value"], str) or not task["task_value"]:
        raise ProtocolViolation("task_card_value")
    if task["public_sink"] is not (task["agent_role"] == "publisher"):
        raise ProtocolViolation("task_card_sink")


def validate_frozen_assets(assets: FrozenAssets) -> None:
    expected_protocol = {
        "provider_kind": "openai_compatible_gateway",
        "base_url_identity": "http://127.0.0.1:58661/v1",
        "models_path": "/models",
        "chat_path": "/chat/completions",
        "model": "gpt-5.3-codex-spark",
        "concurrency": 1,
        "retry_count": 0,
        "stream": False,
    }
    for key, value in expected_protocol.items():
        if assets.protocol.get(key) != value:
            raise ProtocolViolation("frozen_protocol_drift")
    optional_fields = assets.protocol.get("request_contract", {}).get("optional_decoding_fields")
    if optional_fields != {"temperature": "not_sent", "top_p": "not_sent", "max_tokens": "not_sent", "seed": "not_sent"}:
        raise ProtocolViolation("optional_fields_contract")
    if len(assets.screen_tasks) != 32 or len(assets.confirmation_tasks) != 96:
        raise ProtocolViolation("fixture_count")
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
        raise ProtocolViolation("permissions_contract")


def build_request(task: dict[str, Any], assets: FrozenAssets) -> dict[str, Any]:
    validate_task_card(task)
    body = assets.prompt.rstrip() + "\n\nTASK_CARD=" + canonical_json(task)
    return {
        "model": assets.protocol["model"],
        "stream": False,
        "messages": [{"role": "user", "content": body}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "p1v2_q2a_response",
                "strict": True,
                "schema": assets.schema,
            },
        },
    }


def _safe_identity(envelope: dict[str, Any], assets: FrozenAssets) -> dict[str, str]:
    model = envelope.get("model")
    if not isinstance(model, str) or model != assets.protocol["model"]:
        raise ProtocolViolation("model_mismatch")
    request_id = envelope.get("x_request_id", "not_provided")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolViolation("request_id_invalid")
    version = envelope.get("provider_version", "not_provided")
    if not isinstance(version, str) or not version:
        raise ProtocolViolation("provider_version_invalid")
    return {
        "normalized_base_url_identity": assets.provider_profile["normalized_base_url_identity"],
        "model_id": assets.protocol["model"],
        "provider_declared_model": model,
        "request_id": request_id,
        "provider_version": version,
    }


def verify_identity_profile(identity: dict[str, Any], assets: FrozenAssets) -> dict[str, str]:
    allowed = {
        "normalized_base_url_identity",
        "model_id",
        "provider_declared_model",
        "request_id",
        "provider_version",
    }
    if set(identity) != allowed:
        raise ProtocolViolation("identity_profile_drift")
    expected_base = assets.provider_profile["normalized_base_url_identity"]
    expected_model = assets.protocol["model"]
    if identity["normalized_base_url_identity"] != expected_base:
        raise ProtocolViolation("identity_profile_drift")
    if identity["model_id"] != expected_model or identity["provider_declared_model"] != expected_model:
        raise ProtocolViolation("identity_profile_drift")
    for key in ("request_id", "provider_version"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ProtocolViolation("identity_profile_drift")
    return {key: str(identity[key]) for key in sorted(allowed)}


def _validate_parsed_response(parsed: Any, task: dict[str, Any]) -> dict[str, str]:
    if not isinstance(parsed, dict) or set(parsed) != set(RESPONSE_FIELDS):
        raise ProtocolViolation("response_schema")
    if any(not isinstance(parsed[field], str) for field in RESPONSE_FIELDS):
        raise ProtocolViolation("response_schema")
    if parsed["agent_role"] not in ROLE_VALUES or parsed["decision"] not in DECISION_VALUES:
        raise ProtocolViolation("response_schema")
    if not EPISODE_RE.fullmatch(parsed["episode_id"]):
        raise ProtocolViolation("response_schema")
    if not (1 <= len(parsed["task_value"]) <= 160):
        raise ProtocolViolation("response_schema")
    expected = {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }
    for key, expected_value in expected.items():
        if parsed[key] != expected_value:
            raise ProtocolViolation("response_" + key)
    return {field: parsed[field] for field in RESPONSE_FIELDS}


def adapt_openai_response(raw_envelope: str, task: dict[str, Any], assets: FrozenAssets) -> AdaptedResponse:
    """Accept one strict successful envelope or raise a fixed protocol code.

    The caller must never persist raw_envelope after a failure.  This function also
    deliberately returns only parsed public fields, not an envelope copy.
    """
    if not isinstance(raw_envelope, str):
        raise ProtocolViolation("response_envelope_type")
    try:
        envelope = json.loads(raw_envelope)
    except json.JSONDecodeError as exc:
        raise ProtocolViolation("response_envelope_json") from exc
    if not isinstance(envelope, dict):
        raise ProtocolViolation("unknown_envelope")
    if "error" in envelope:
        raise ProtocolViolation("provider_error")
    safe_identity = _safe_identity(envelope, assets)
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProtocolViolation("missing_choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProtocolViolation("unknown_envelope")
    if "tool_calls" in message or "function_call" in message:
        raise ProtocolViolation("tool_calls")
    if "refusal" in message:
        raise ProtocolViolation("refusal")
    if "reasoning_content" in message:
        raise ProtocolViolation("reasoning_detected")
    if message.get("reasoning"):
        raise ProtocolViolation("reasoning_detected")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProtocolViolation("empty_content")
    if "<think>" in content.lower():
        raise ProtocolViolation("reasoning_detected")
    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProtocolViolation("response_content_json") from exc
    parsed = _validate_parsed_response(parsed_json, task)
    public_sink_record = dict(parsed) if task["agent_role"] == "publisher" else None
    return AdaptedResponse(
        parsed=parsed,
        content_sha256=sha256_text(content),
        safe_identity=safe_identity,
        public_sink_record=public_sink_record,
    )
