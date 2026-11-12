"""Repair-native frozen request, safe identity extraction, and content assessment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT, canonical_json, load_json, sha256_file, sha256_text


FROZEN_MODEL = "gpt-5.3-codex-spark"
RESPONSE_FIELDS = ("agent_role", "episode_id", "decision", "task_value")
FROZEN_ASSETS = (
    "configs/repair_provider_profile.json",
    "configs/repair_protocol.json",
    "fixtures/repair_task.json",
    "prompts/repair_prompt.txt",
    "schemas/repair_response_schema.json",
    "provenance/source_contracts.json",
)


@dataclass(frozen=True)
class RepairAssets:
    root: Path
    provider_profile: dict[str, Any]
    protocol: dict[str, Any]
    task: dict[str, Any]
    prompt: str
    schema: dict[str, Any]
    hashes: dict[str, str]


@dataclass(frozen=True)
class Assessment:
    status: str
    failure_code: str | None
    failure_class: str | None
    parsed: dict[str, str] | None
    safe_identity: dict[str, str]
    content_sha256: str | None


def load_assets(root: Path = PACKAGE_ROOT) -> RepairAssets:
    hashes = {relative: sha256_file(root / relative) for relative in FROZEN_ASSETS}
    return RepairAssets(
        root=root,
        provider_profile=load_json(root / "configs" / "repair_provider_profile.json"),
        protocol=load_json(root / "configs" / "repair_protocol.json"),
        task=load_json(root / "fixtures" / "repair_task.json"),
        prompt=(root / "prompts" / "repair_prompt.txt").read_text(encoding="utf-8"),
        schema=load_json(root / "schemas" / "repair_response_schema.json"),
        hashes=hashes,
    )


def frozen_metadata_profile(assets: RepairAssets) -> tuple[str, tuple[str, ...]]:
    """Return the sole frozen model-list authority for metadata identity."""
    profile = assets.provider_profile
    expected_keys = {"target_model_id", "expected_model_ids", "model_list_canonicalization"}
    if not isinstance(profile, dict) or set(profile) != expected_keys:
        raise ValueError("frozen_metadata_profile_shape")
    target = profile.get("target_model_id")
    model_ids = profile.get("expected_model_ids")
    canonicalization = profile.get("model_list_canonicalization")
    if target != FROZEN_MODEL or canonicalization != "canonical_json_sorted_model_ids_v1":
        raise ValueError("frozen_metadata_profile_target")
    if not isinstance(model_ids, list) or not model_ids or any(not isinstance(model_id, str) or not model_id for model_id in model_ids):
        raise ValueError("frozen_metadata_profile_models")
    if model_ids != sorted(model_ids) or len(set(model_ids)) != len(model_ids) or target not in model_ids:
        raise ValueError("frozen_metadata_profile_models")
    return target, tuple(model_ids)


def build_request(task: dict[str, Any], assets: RepairAssets) -> dict[str, Any]:
    return {
        "model": FROZEN_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": assets.prompt.rstrip() + "\nTASK_CARD=" + canonical_json(task)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "q2b_repair_response", "strict": True, "schema": assets.schema}},
    }


def request_sha256(task: dict[str, Any], assets: RepairAssets) -> str:
    return sha256_text(canonical_json(build_request(task, assets)))


def _safe_string(value: Any) -> str:
    return value if isinstance(value, str) and value else "not_provided"


def unavailable_identity(reason: str) -> dict[str, str]:
    return {"availability": "unavailable", "identity_reason": reason}


def _available_identity(envelope: dict[str, Any], headers: dict[str, str]) -> dict[str, str]:
    return {
        "availability": "available",
        "provider_declared_model": _safe_string(envelope.get("model")),
        "request_id": _safe_string(headers.get("x-request-id") or headers.get("request-id")),
        "system_fingerprint": _safe_string(envelope.get("system_fingerprint")),
        "provider_version": _safe_string(envelope.get("provider_version")),
    }


def _failure(code: str, failure_class: str, identity: dict[str, str]) -> Assessment:
    return Assessment("failed", code, failure_class, None, identity, None)


def assess_transport_failure(code: str) -> Assessment:
    """Represent an HTTP/timeout outcome without retaining service error text."""
    return _failure(code, "infrastructure", unavailable_identity("transport_failure"))


def assess_success_envelope(raw_body: bytes, headers: dict[str, str], task: dict[str, Any]) -> Assessment:
    """Extract safe identity first; raw content is not returned or persisted."""
    try:
        envelope = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _failure("malformed_envelope", "infrastructure", unavailable_identity("malformed_envelope"))
    if not isinstance(envelope, dict):
        return _failure("unknown_envelope", "infrastructure", unavailable_identity("unknown_envelope"))
    identity = _available_identity(envelope, headers)
    if identity["provider_declared_model"] != FROZEN_MODEL:
        return _failure("model_mismatch", "infrastructure", identity)
    if "error" in envelope:
        return _failure("provider_error", "infrastructure", identity)
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return _failure("missing_choices", "model_output", identity)
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return _failure("unknown_envelope", "model_output", identity)
    if "tool_calls" in message:
        return _failure("tool_calls", "model_output", identity)
    if "function_call" in message:
        return _failure("function_calls", "model_output", identity)
    if "refusal" in message:
        return _failure("refusal", "model_output", identity)
    if "reasoning_content" in message or ("reasoning" in message and message["reasoning"] not in (None, "")):
        return _failure("reasoning_detected", "model_output", identity)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return _failure("empty_content", "model_output", identity)
    if "<think>" in content.lower():
        return _failure("reasoning_detected", "model_output", identity)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _failure("content_json", "model_output", identity)
    expected = {field: task[field] for field in RESPONSE_FIELDS}
    if not isinstance(parsed, dict) or set(parsed) != set(RESPONSE_FIELDS):
        return _failure("response_schema", "model_output", identity)
    if any(not isinstance(parsed[field], str) for field in RESPONSE_FIELDS):
        return _failure("response_schema", "model_output", identity)
    for field, value in expected.items():
        if parsed.get(field) != value:
            return _failure("response_" + field, "model_output", identity)
    return Assessment("completed", None, None, expected, identity, sha256_text(content))
