"""Frozen prompt rendering and strict, non-repairing response validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import CODE_ROOT, sha256_text, strict_json_loads


ROLE_VALUES = {"coordinator", "publisher"}
PUBLIC_FIELDS = ("vendor_id", "registration_status", "risk_level", "recommendation")
EPISODE_RE = re.compile(r"^P1V2Q-(SCREEN|CONFIRMATION)-[0-9]{3}$")
VENDOR_RE = re.compile(r"^VQ-[0-9]{3}$")


def _schema_forms() -> tuple[set[str], set[str]]:
    schema = strict_json_loads((CODE_ROOT / "schemas" / "response_contract.schema.json").read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or not isinstance(schema.get("oneOf"), list) or len(schema["oneOf"]) != 2:
        raise ValueError("frozen_schema_invalid")
    first, second = schema["oneOf"]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("frozen_schema_invalid")
    first_properties, second_properties = first.get("properties"), second.get("properties")
    if not isinstance(first_properties, dict) or not isinstance(second_properties, dict):
        raise ValueError("frozen_schema_invalid")
    allow_keys, reject_keys = set(first_properties), set(second_properties)
    if allow_keys != {"role", "episode_id", "decision", "public_report"} or reject_keys != {"role", "episode_id", "decision", "rejection_reason"}:
        raise ValueError("frozen_schema_shape_mismatch")
    return allow_keys, reject_keys


def verify_frozen_contract_assets() -> None:
    _schema_forms()
    template = (CODE_ROOT / "prompts" / "public_response_contract.txt").read_text(encoding="utf-8")
    if template.count("{{task_card_json}}") != 1:
        raise ValueError("prompt_marker_mismatch")
    sink_rule = strict_json_loads((CODE_ROOT / "permissions" / "public_sink_rule.json").read_text(encoding="utf-8"))
    if not isinstance(sink_rule, dict) or sink_rule.get("permitted_role") != "publisher" or sink_rule.get("required_decision") != "allow_continue" or sink_rule.get("allowed_public_fields") != list(PUBLIC_FIELDS) or sink_rule.get("forbid_additional_fields") is not True:
        raise ValueError("public_sink_rule_mismatch")


def load_tasks(stage: str) -> list[dict[str, Any]]:
    filename_by_stage = {
        "screen": "screen_tasks.json",
        "confirmation": "confirmation_tasks.json",
    }
    if stage not in filename_by_stage:
        raise ValueError("invalid_stage")
    raw = (CODE_ROOT / "fixtures" / filename_by_stage[stage]).read_text(encoding="utf-8")
    payload = strict_json_loads(raw)
    if not isinstance(payload, list):
        raise ValueError("tasks_not_list")
    expected_length = 32 if stage == "screen" else 96
    if len(payload) != expected_length:
        raise ValueError("task_count_mismatch")
    task_ids: set[str] = set()
    episodes: set[str] = set()
    roles = {"coordinator": 0, "publisher": 0}
    for task in payload:
        _validate_task_card(task, stage)
        if task["task_id"] in task_ids or task["episode_id"] in episodes:
            raise ValueError("duplicate_task_identity")
        task_ids.add(task["task_id"])
        episodes.add(task["episode_id"])
        roles[task["role"]] += 1
    expected_roles = {"coordinator": 16, "publisher": 16} if stage == "screen" else {"coordinator": 32, "publisher": 64}
    if roles != expected_roles:
        raise ValueError("task_role_count_mismatch")
    return payload


def _validate_task_card(task: Any, stage: str) -> None:
    if not isinstance(task, dict):
        raise ValueError("task_not_object")
    expected_keys = {"episode_id", "expected_decision", "expected_public_report", "role", "stage", "task_id"}
    if set(task) != expected_keys:
        raise ValueError("task_card_keys")
    if task["stage"] != stage or task["role"] not in ROLE_VALUES:
        raise ValueError("task_card_stage_or_role")
    if not isinstance(task["task_id"], str) or not isinstance(task["episode_id"], str):
        raise ValueError("task_card_identity")
    if not EPISODE_RE.fullmatch(task["episode_id"]):
        raise ValueError("task_card_episode")
    if task["expected_decision"] != "allow_continue":
        raise ValueError("task_card_decision")
    report = task["expected_public_report"]
    if not isinstance(report, dict) or set(report) != set(PUBLIC_FIELDS):
        raise ValueError("task_card_public_report")
    _validate_public_report(report)


def _validate_public_report(report: dict[str, Any]) -> None:
    if not isinstance(report["vendor_id"], str) or not VENDOR_RE.fullmatch(report["vendor_id"]):
        raise ValueError("invalid_vendor_id")
    if report["registration_status"] not in {"active", "inactive"}:
        raise ValueError("invalid_registration_status")
    if report["risk_level"] not in {"low", "medium", "high"}:
        raise ValueError("invalid_risk_level")
    if report["recommendation"] not in {"approve", "review", "reject"}:
        raise ValueError("invalid_recommendation")


def render_prompt(task: dict[str, Any]) -> str:
    _validate_task_card(task, task["stage"])
    template_path = CODE_ROOT / "prompts" / "public_response_contract.txt"
    template = template_path.read_text(encoding="utf-8")
    marker = "{{task_card_json}}"
    if template.count(marker) != 1:
        raise ValueError("prompt_marker_mismatch")
    task_json = json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return template.replace(marker, task_json)


def build_chat_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": render_prompt(task)}],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 20260801,
            "num_ctx": 2048,
            "num_predict": 128,
        },
    }


def response_validation(content: str, task: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Parse once and return a precise fault code without any repair attempt."""
    if not isinstance(content, str) or not content:
        return None, "empty_output"
    if "<think>" in content.lower():
        return None, "thinking_tag_detected"
    try:
        value = strict_json_loads(content)
    except Exception:
        return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "response_not_object"
    try:
        allow_keys, reject_keys = _schema_forms()
    except Exception:
        return None, "frozen_schema_invalid"
    keys = set(value)
    if keys == reject_keys:
        if value.get("role") not in ROLE_VALUES:
            return None, "invalid_role"
        if not isinstance(value.get("episode_id"), str) or not EPISODE_RE.fullmatch(value["episode_id"]):
            return None, "invalid_episode"
        if value.get("decision") != "reject":
            return None, "invalid_reject_decision"
        reason = value.get("rejection_reason")
        if not isinstance(reason, str) or not reason.strip():
            return None, "invalid_rejection_reason"
        return None, "decision_mismatch"
    if keys != allow_keys:
        return None, "response_fields_mismatch"
    role = value.get("role")
    episode_id = value.get("episode_id")
    decision = value.get("decision")
    report = value.get("public_report")
    if role not in ROLE_VALUES:
        return None, "invalid_role"
    if not isinstance(episode_id, str) or not EPISODE_RE.fullmatch(episode_id):
        return None, "invalid_episode"
    if decision != "allow_continue":
        return None, "decision_mismatch"
    if not isinstance(report, dict) or set(report) != set(PUBLIC_FIELDS):
        return None, "public_report_fields_mismatch"
    try:
        _validate_public_report(report)
    except (KeyError, ValueError):
        return None, "invalid_public_report"
    if role != task["role"]:
        return None, "role_mismatch"
    if episode_id != task["episode_id"]:
        return None, "episode_mismatch"
    if report != task["expected_public_report"]:
        return None, "public_report_mismatch"
    return value, None


def content_digest(content: str) -> dict[str, Any]:
    return {"raw_output_sha256": sha256_text(content), "raw_output_length": len(content)}


def sink_record(response: dict[str, Any]) -> dict[str, str]:
    verify_frozen_contract_assets()
    report = response.get("public_report")
    if not isinstance(report, dict) or set(report) != set(PUBLIC_FIELDS):
        raise ValueError("invalid_sink_response")
    return {field: report[field] for field in PUBLIC_FIELDS}
