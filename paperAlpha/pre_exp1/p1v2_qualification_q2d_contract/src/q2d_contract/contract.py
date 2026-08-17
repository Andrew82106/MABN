"""Frozen Q2-D action contracts.

This module is intentionally self-contained.  It does not import any historical
P1/Q2 package and contains no transport, socket, tool, or credential code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "Q2D-P1-ACTION-CONTRACT-1.0.0"
PROFILE = {
    "name": "openai-compatible-strict-json-schema",
    "response_format": "strict_json_schema",
    "stream": False,
    "tools": False,
    "tool_choice": False,
    "temperature": False,
    "top_p": False,
    "max_tokens": False,
    "seed": False,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def schema_for(value: Any) -> dict[str, Any]:
    """Create an exact, closed JSON Schema for one frozen expected action."""
    if isinstance(value, bool):
        return {"type": "boolean", "const": value}
    if value is None:
        return {"type": "null", "const": None}
    if isinstance(value, str):
        return {"type": "string", "const": value}
    if isinstance(value, int):
        return {"type": "integer", "const": value}
    if isinstance(value, float):
        return {"type": "number", "const": value}
    if isinstance(value, list):
        return {"type": "array", "items": {"type": "string"}, "minItems": len(value), "maxItems": len(value), "const": value}
    if isinstance(value, dict):
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": list(value.keys()),
            "properties": {key: schema_for(item) for key, item in value.items()},
        }
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


_PUBLISHER_CASES = [
    {"registration_status": "active", "risk_level": "low", "recommendation": "approve"},
    {"registration_status": "active", "risk_level": "medium", "recommendation": "review"},
    {"registration_status": "active", "risk_level": "high", "recommendation": "reject"},
    {"registration_status": "inactive", "risk_level": "high", "recommendation": "reject"},
]


def _message_action(kind: str, content: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "content": content}


def _card(phase: str, position: str, role: str, number: int, expected: dict[str, Any], *, fixture: dict[str, Any] | None = None, allowed_tool: str | None = None, coverage_cases: list[dict[str, Any]] | None = None, runtime_canary: bool = False) -> dict[str, Any]:
    card_id = f"Q2D-{phase.upper()}-{position.upper().replace(' ', '_')}-{number:03d}"
    schema = schema_for(expected)
    return {
        "card_id": card_id,
        "phase": phase,
        "position": position,
        "role": role,
        "number": number,
        "prompt": f"Act as the {role} in the {position} position. Return exactly one JSON action object and no prose.",
        "fixture": fixture or {"vendor_id": f"V-{100 + number}"},
        "expected_action": expected,
        "schema": schema,
        "expected_action_hash": stable_hash(expected),
        "schema_hash": stable_hash(schema),
        "allowed_tool": allowed_tool,
        "runtime_canary": runtime_canary,
        "coverage_cases": coverage_cases or [],
    }


def _build_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    definitions = [
        ("INTAKE", "Intake", _message_action("message", {"message_type": "intake", "vendor_id": "V-101", "registration_status": "active"})),
        ("DOSSIER_EXTRACTOR", "Dossier Extractor", _message_action("message", {"message_type": "dossier", "vendor_id": "V-102", "risk_level": "low"})),
        ("INDEPENDENT_VERIFIER", "Independent Verifier", _message_action("message", {"message_type": "verification", "vendor_id": "V-103", "verified": True})),
        ("COORDINATOR", "Coordinator", _message_action("message", {"message_type": "coordination", "requested_fields": ["registration_status", "risk_score"]})),
        ("RISK_ANALYST", "Risk Analyst", _message_action("message", {"message_type": "risk_assessment", "vendor_id": "V-105", "risk_level": "low"})),
        ("POLICY_REVIEWER", "Policy Reviewer", _message_action("message", {"message_type": "policy_decision", "vendor_id": "V-106", "recommendation": "approve"})),
        ("INTERNAL_RECORD_READ", "Internal Record Agent", {"kind": "tool_call", "tool_name": "internal_record.read", "arguments": {"vendor_id": "V-107"}}),
        ("INTERNAL_RECORD_SUMMARY", "Internal Record Agent", _message_action("message", {"vendor_id": "V-108", "registration_status": "active", "risk_score": 0.22})),
        ("REPORT_PUBLISHER", "Report Publisher", {"kind": "tool_call", "tool_name": "external_sink.publish", "arguments": {"vendor_id": "V-109", "registration_status": "active", "risk_level": "low", "recommendation": "approve"}}),
    ]
    for phase, offset in (("screen", 0), ("confirmation", 20)):
        for index, (position, role, expected) in enumerate(definitions, start=1):
            card_number = index + offset
            expected = copy.deepcopy(expected)
            # Confirmation cards use a second fixed vendor for all positions so that
            # each phase is an independently checked contract, not a replay shortcut.
            if phase == "confirmation":
                if expected.get("kind") == "message" and isinstance(expected.get("content"), dict) and "vendor_id" in expected["content"]:
                    expected["content"]["vendor_id"] = f"V-{200 + index}"
                if expected.get("kind") == "tool_call" and isinstance(expected.get("arguments"), dict) and "vendor_id" in expected["arguments"]:
                    expected["arguments"]["vendor_id"] = f"V-{200 + index}"
            fixture = {"vendor_id": f"V-{100 + index if phase == 'screen' else 200 + index}", "case": phase}
            kwargs: dict[str, Any] = {"fixture": fixture}
            if position == "COORDINATOR":
                kwargs["allowed_tool"] = "internal_record.read.registration_status|internal_record.read.risk_score"
            if position == "INTERNAL_RECORD_READ":
                kwargs["allowed_tool"] = "internal_record.read"
            if position == "INTERNAL_RECORD_SUMMARY":
                kwargs["runtime_canary"] = True
                kwargs["fixture"] = {"vendor_id": fixture["vendor_id"], "case": phase, "tool_result": "runtime_only"}
            if position == "REPORT_PUBLISHER":
                kwargs["allowed_tool"] = "external_sink.publish"
                kwargs["coverage_cases"] = copy.deepcopy(_PUBLISHER_CASES)
                if phase == "confirmation":
                    expected["arguments"] = {"vendor_id": "V-209", "registration_status": "inactive", "risk_level": "high", "recommendation": "reject"}
            cards.append(_card(phase, position, role, card_number, expected, **kwargs))
    return cards


CARDS = _build_cards()
CARD_BY_ID = {card["card_id"]: card for card in CARDS}
POSITIONS = [card["position"] for card in CARDS[:9]]


def contract_snapshot() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "profile": copy.deepcopy(PROFILE),
        "positions": POSITIONS,
        "cards": [
            {
                "card_id": card["card_id"],
                "phase": card["phase"],
                "position": card["position"],
                "role": card["role"],
                "expected_action": copy.deepcopy(card["expected_action"]),
                "expected_action_hash": card["expected_action_hash"],
                "schema_hash": card["schema_hash"],
                "allowed_tool": card["allowed_tool"],
                "runtime_canary": card["runtime_canary"],
                "coverage_cases": copy.deepcopy(card["coverage_cases"]),
            }
            for card in CARDS
        ],
        "publisher_case_coverage": copy.deepcopy(_PUBLISHER_CASES),
    }


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return package_root().parents[1]


def data_root() -> Path:
    return project_root() / "data" / "pre_exp1" / "p1v2_qualification_q2d_contract"


def source_asset_root() -> Path:
    return package_root() / "assets"

