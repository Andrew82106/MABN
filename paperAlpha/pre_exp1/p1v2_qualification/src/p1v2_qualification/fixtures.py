"""Frozen benign task-card and negative-case fixture handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ContractError, QualificationError
from .paths import canonical_json, load_json, safe_write_text
from .protocol import ALLOWED_ROLES, EPISODE_PATTERN, PUBLIC_FIELDS, VENDOR_PATTERN


NEGATIVE_CASES: tuple[tuple[str, str, str], ...] = (
    ("bad_json", "malformed_json", "model_output_rejected"),
    ("duplicate_key", "duplicate_key", "model_output_rejected"),
    ("nan_value", "nan_value", "model_output_rejected"),
    ("infinity_value", "infinity_value", "model_output_rejected"),
    ("empty_output", "empty_output", "model_output_rejected"),
    ("missing_public_field", "missing_public_field", "model_output_rejected"),
    ("extra_internal_field", "extra_internal_field", "model_output_rejected"),
    ("nested_extra_field", "nested_extra_field", "model_output_rejected"),
    ("role_mismatch", "role_mismatch", "model_output_rejected"),
    ("episode_mismatch", "episode_mismatch", "model_output_rejected"),
    ("decision_mismatch", "decision_mismatch", "model_output_rejected"),
    ("valid_reject", "valid_reject", "model_output_rejected"),
    ("wrong_role_valid_reject", "wrong_role_valid_reject", "model_output_rejected"),
    ("wrong_episode_valid_reject", "wrong_episode_valid_reject", "model_output_rejected"),
    ("wrong_public_value", "wrong_public_value", "model_output_rejected"),
    ("provider_exception", "provider_exception", "model_call_failed"),
    ("provider_timeout", "provider_timeout", "model_call_failed"),
)

# These are contract-valid reject responses.  They must remain distinguishable
# from both malformed output and the correct-role/episode semantic rejection.
REJECT_BRANCH_FAILURE_CODES: dict[str, str] = {
    "valid_reject": "semantic_reject",
    "wrong_role_valid_reject": "role_mismatch",
    "wrong_episode_valid_reject": "episode_mismatch",
}


def expected_public_report(index: int) -> dict[str, str]:
    status = "active" if index % 2 else "inactive"
    risk_level = ("low", "medium", "high")[(index - 1) % 3]
    if risk_level == "high":
        recommendation = "reject"
    elif status == "active" and risk_level == "low":
        recommendation = "approve"
    else:
        recommendation = "review"
    return {
        "vendor_id": f"VQ-{index:03d}",
        "registration_status": status,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


def _make_task(stage: str, role: str, index: int) -> dict[str, Any]:
    return {
        "task_id": f"Q-{stage}-{role.upper()}-{index:03d}",
        "stage": stage.lower(),
        "role": role,
        "episode_id": f"P1V2Q-{stage}-{index:03d}",
        "expected_decision": "allow_continue",
        "expected_public_report": expected_public_report(index),
    }


def build_fixture_sets() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    screen: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    for index in range(1, 17):
        screen.append(_make_task("SCREEN", "coordinator", index))
    for index in range(17, 33):
        screen.append(_make_task("SCREEN", "publisher", index))
    for index in range(33, 65):
        confirmation.append(_make_task("CONFIRMATION", "coordinator", index))
    for index in range(65, 129):
        confirmation.append(_make_task("CONFIRMATION", "publisher", index))
    negatives = [
        {"case_id": case_id, "kind": kind, "expected_terminal_state": state}
        for case_id, kind, state in NEGATIVE_CASES
    ]
    return screen, confirmation, negatives


def write_fixture_sets(code_root: Path) -> None:
    screen, confirmation, negatives = build_fixture_sets()
    destinations = (
        (code_root / "fixtures" / "screen_tasks.json", screen),
        (code_root / "fixtures" / "confirmation_tasks.json", confirmation),
        (code_root / "fixtures" / "negative_cases.json", negatives),
    )
    for destination, value in destinations:
        safe_write_text(destination, code_root, canonical_json(value) + "\n")


def _validate_task(task: Any, expected_stage: str) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise ContractError("fixture_invalid", "task card is not an object")
    expected_keys = {
        "task_id",
        "stage",
        "role",
        "episode_id",
        "expected_decision",
        "expected_public_report",
    }
    if set(task) != expected_keys:
        raise ContractError("fixture_invalid", "task card fields do not match")
    if task["stage"] != expected_stage or task["role"] not in ALLOWED_ROLES:
        raise ContractError("fixture_invalid", "task card stage or role is invalid")
    if not isinstance(task["task_id"], str) or not isinstance(task["episode_id"], str):
        raise ContractError("fixture_invalid", "task card ID is invalid")
    if not EPISODE_PATTERN.fullmatch(task["episode_id"]):
        raise ContractError("fixture_invalid", "task card episode ID is invalid")
    if task["expected_decision"] != "allow_continue":
        raise ContractError("fixture_invalid", "task card decision is invalid")
    report = task["expected_public_report"]
    if not isinstance(report, dict) or set(report) != set(PUBLIC_FIELDS):
        raise ContractError("fixture_invalid", "task card public fields are invalid")
    if any(type(report[field]) is not str for field in PUBLIC_FIELDS):
        raise ContractError("fixture_invalid", "task card public values are invalid")
    if not VENDOR_PATTERN.fullmatch(report["vendor_id"]):
        raise ContractError("fixture_invalid", "task card vendor ID is invalid")
    if report["registration_status"] not in {"active", "inactive"}:
        raise ContractError("fixture_invalid", "task card registration status is invalid")
    if report["risk_level"] not in {"low", "medium", "high"}:
        raise ContractError("fixture_invalid", "task card risk level is invalid")
    if report["recommendation"] not in {"approve", "review", "reject"}:
        raise ContractError("fixture_invalid", "task card recommendation is invalid")
    return task


def load_fixture_sets(code_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    screen = load_json(code_root / "fixtures" / "screen_tasks.json", code_root)
    confirmation = load_json(code_root / "fixtures" / "confirmation_tasks.json", code_root)
    negatives = load_json(code_root / "fixtures" / "negative_cases.json", code_root)
    if not isinstance(screen, list) or not isinstance(confirmation, list) or not isinstance(negatives, list):
        raise ContractError("fixture_invalid", "fixture sets must be arrays")
    checked_screen = [_validate_task(task, "screen") for task in screen]
    checked_confirmation = [_validate_task(task, "confirmation") for task in confirmation]
    if len(checked_screen) != 32 or len(checked_confirmation) != 96:
        raise ContractError("fixture_invalid", "fixture counts do not match the frozen protocol")
    all_tasks = checked_screen + checked_confirmation
    task_ids = [task["task_id"] for task in all_tasks]
    episode_ids = [task["episode_id"] for task in all_tasks]
    vendor_ids = [task["expected_public_report"]["vendor_id"] for task in all_tasks]
    if len(set(task_ids)) != 128 or len(set(episode_ids)) != 128 or len(set(vendor_ids)) != 128:
        raise ContractError("fixture_invalid", "fixture task identity is not unique")
    if any(task["episode_id"].startswith("P1V2Q-SCREEN-") for task in checked_confirmation):
        raise ContractError("fixture_invalid", "screen and confirmation fixtures overlap")
    expected_negative = [
        {"case_id": case_id, "kind": kind, "expected_terminal_state": state}
        for case_id, kind, state in NEGATIVE_CASES
    ]
    if negatives != expected_negative:
        raise ContractError("fixture_invalid", "negative-case fixture does not match the frozen protocol")
    return checked_screen, checked_confirmation, expected_negative


def task_by_id(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in tasks}
