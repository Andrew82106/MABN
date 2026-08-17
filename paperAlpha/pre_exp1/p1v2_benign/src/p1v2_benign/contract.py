"""Schema-backed, fail-closed public-response contract for offline fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .core import (
    CODE_NAMESPACE_ROOT,
    DEFAULT_SCHEMA_PATH,
    StrictJsonError,
    canonical_json_bytes,
    read_strict_json,
    sha256_bytes,
    strict_json_loads,
)


@dataclass(frozen=True)
class ContractPolicy:
    allowed_roles: tuple[str, ...]
    required_public_fields: tuple[str, ...]
    allow_continue_value: str
    reject_value: str

    @classmethod
    def from_mapping(cls, value: Any) -> "ContractPolicy":
        expected = {"allow_continue_value", "allowed_roles", "reject_value", "required_public_fields"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid_contract_policy")
        roles = value["allowed_roles"]
        fields = value["required_public_fields"]
        allow_value = value["allow_continue_value"]
        reject_value = value["reject_value"]
        if not isinstance(roles, list) or not isinstance(fields, list):
            raise ValueError("invalid_contract_policy_lists")
        if not roles or not fields:
            raise ValueError("empty_contract_policy_list")
        if any(not isinstance(item, str) or not item for item in roles):
            raise ValueError("invalid_contract_policy_roles")
        if any(not isinstance(item, str) or not item for item in fields):
            raise ValueError("invalid_contract_policy_fields")
        if len(roles) != len(set(roles)) or len(fields) != len(set(fields)):
            raise ValueError("duplicate_contract_policy_value")
        if (
            not isinstance(allow_value, str)
            or not isinstance(reject_value, str)
            or not allow_value
            or not reject_value
            or allow_value == reject_value
        ):
            raise ValueError("invalid_contract_policy_decisions")
        return cls(tuple(roles), tuple(fields), allow_value, reject_value)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "allow_continue_value": self.allow_continue_value,
            "allowed_roles": list(self.allowed_roles),
            "reject_value": self.reject_value,
            "required_public_fields": list(self.required_public_fields),
        }


@dataclass(frozen=True)
class SchemaContract:
    allowed_roles: tuple[str, ...]
    allow_continue_value: str
    reject_value: str
    required_public_fields: tuple[str, ...]
    episode_pattern: str
    rejection_reason_pattern: str
    vendor_pattern: str
    registration_status_values: tuple[str, ...]
    risk_level_values: tuple[str, ...]
    recommendation_values: tuple[str, ...]


def _nonempty_string_list(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("invalid_schema_string_list")
    return tuple(value)


def _branch_for_decision(branches: Any, decision: str) -> dict[str, Any]:
    if not isinstance(branches, list) or len(branches) != 2:
        raise ValueError("invalid_schema_branches")
    matches: list[dict[str, Any]] = []
    for branch in branches:
        if not isinstance(branch, dict) or branch.get("type") != "object":
            raise ValueError("invalid_schema_branch")
        properties = branch.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("invalid_schema_properties")
        decision_schema = properties.get("decision")
        if isinstance(decision_schema, dict) and decision_schema.get("const") == decision:
            matches.append(branch)
    if len(matches) != 1:
        raise ValueError("ambiguous_schema_decision_branch")
    return matches[0]


def _string_schema(value: Any, *, pattern: bool = False, enum: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "string":
        raise ValueError("invalid_schema_string")
    if pattern and not isinstance(value.get("pattern"), str):
        raise ValueError("invalid_schema_string_pattern")
    if enum:
        _nonempty_string_list(value.get("enum"))
    return value


def schema_contract_from_mapping(schema: Any) -> SchemaContract:
    """Turn the strict production schema shape into executable constraints."""

    if (
        not isinstance(schema, dict)
        or set(schema) != {"$schema", "oneOf"}
        or schema["$schema"] != "https://json-schema.org/draft/2020-12/schema"
    ):
        raise ValueError("invalid_schema_root")
    allow = _branch_for_decision(schema["oneOf"], "allow_continue")
    reject = _branch_for_decision(schema["oneOf"], "reject")
    allow_props = allow.get("properties")
    reject_props = reject.get("properties")
    if not isinstance(allow_props, dict) or not isinstance(reject_props, dict):
        raise ValueError("invalid_schema_properties")
    if set(allow_props) != {"role", "episode_id", "decision", "public_report"}:
        raise ValueError("invalid_allow_schema_properties")
    if set(reject_props) != {"role", "episode_id", "decision", "rejection_reason"}:
        raise ValueError("invalid_reject_schema_properties")
    if _nonempty_string_list(allow.get("required")) != ("role", "episode_id", "decision", "public_report"):
        raise ValueError("invalid_allow_schema_required")
    if _nonempty_string_list(reject.get("required")) != ("role", "episode_id", "decision", "rejection_reason"):
        raise ValueError("invalid_reject_schema_required")
    if allow.get("additionalProperties") is not False or reject.get("additionalProperties") is not False:
        raise ValueError("schema_allows_extra_response_properties")
    if allow_props["decision"] != {"const": "allow_continue"} or reject_props["decision"] != {"const": "reject"}:
        raise ValueError("invalid_schema_decision")
    role_schema = _string_schema(allow_props["role"], enum=True)
    reject_role_schema = _string_schema(reject_props["role"], enum=True)
    episode_schema = _string_schema(allow_props["episode_id"], pattern=True)
    reject_episode_schema = _string_schema(reject_props["episode_id"], pattern=True)
    if role_schema != reject_role_schema or episode_schema != reject_episode_schema:
        raise ValueError("inconsistent_common_schema_fields")
    roles = _nonempty_string_list(role_schema["enum"])
    rejection_schema = _string_schema(reject_props["rejection_reason"], pattern=True)
    if rejection_schema.get("minLength") != 1 or rejection_schema["pattern"] != ".*\\S.*":
        raise ValueError("invalid_schema_rejection_reason")
    report_schema = allow_props["public_report"]
    if (
        not isinstance(report_schema, dict)
        or report_schema.get("type") != "object"
        or report_schema.get("additionalProperties") is not False
        or not isinstance(report_schema.get("properties"), dict)
    ):
        raise ValueError("invalid_schema_public_report")
    report_props = report_schema["properties"]
    required_fields = _nonempty_string_list(report_schema.get("required"))
    if set(report_props) != set(required_fields):
        raise ValueError("schema_public_report_field_mismatch")
    if required_fields != ("vendor_id", "registration_status", "risk_level", "recommendation"):
        raise ValueError("unexpected_schema_public_field_order")
    vendor_schema = _string_schema(report_props.get("vendor_id"), pattern=True)
    registration_schema = _string_schema(report_props.get("registration_status"), enum=True)
    risk_schema = _string_schema(report_props.get("risk_level"), enum=True)
    recommendation_schema = _string_schema(report_props.get("recommendation"), enum=True)
    return SchemaContract(
        allowed_roles=roles,
        allow_continue_value="allow_continue",
        reject_value="reject",
        required_public_fields=required_fields,
        episode_pattern=episode_schema["pattern"],
        rejection_reason_pattern=rejection_schema["pattern"],
        vendor_pattern=vendor_schema["pattern"],
        registration_status_values=_nonempty_string_list(registration_schema["enum"]),
        risk_level_values=_nonempty_string_list(risk_schema["enum"]),
        recommendation_values=_nonempty_string_list(recommendation_schema["enum"]),
    )


def load_schema_contract() -> SchemaContract:
    """Parse the production schema and expose the exact executable constraints."""

    return schema_contract_from_mapping(read_strict_json(DEFAULT_SCHEMA_PATH, allowed_root=CODE_NAMESPACE_ROOT))


def policy_matches_schema(policy: ContractPolicy, schema: SchemaContract) -> bool:
    return (
        policy.allowed_roles == schema.allowed_roles
        and policy.required_public_fields == schema.required_public_fields
        and policy.allow_continue_value == schema.allow_continue_value
        and policy.reject_value == schema.reject_value
    )


def _fallback_rejected(code: str) -> dict[str, Any]:
    return {
        "contract_valid": False,
        "decision": "reject",
        "public_report": None,
        "rejection_code": code,
    }


def _rejected(policy: ContractPolicy, code: str) -> dict[str, Any]:
    return {
        "contract_valid": False,
        "decision": policy.reject_value,
        "public_report": None,
        "rejection_code": code,
    }


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_response(
    raw_response: Any,
    expected_role: Any,
    expected_episode_id: Any,
    policy: Any,
    schema: SchemaContract | None = None,
) -> dict[str, Any]:
    """Validate the real schema contract with no recovery or side effects."""

    try:
        safe_policy = ContractPolicy.from_mapping(policy.as_mapping() if isinstance(policy, ContractPolicy) else policy)
    except Exception:
        return _fallback_rejected("invalid_contract_policy")
    try:
        safe_schema = load_schema_contract() if schema is None else schema
        if not isinstance(safe_schema, SchemaContract):
            raise ValueError("invalid_schema_contract")
    except Exception:
        return _rejected(safe_policy, "schema_contract_invalid")
    if not policy_matches_schema(safe_policy, safe_schema):
        return _rejected(safe_policy, "schema_policy_mismatch")
    if not isinstance(raw_response, str):
        return _rejected(safe_policy, "invalid_raw_response")
    if not isinstance(expected_role, str) or not isinstance(expected_episode_id, str):
        return _rejected(safe_policy, "invalid_expected_context")
    try:
        response = strict_json_loads(raw_response)
    except StrictJsonError as exc:
        return _rejected(safe_policy, exc.code)
    except Exception:
        return _rejected(safe_policy, "invalid_json")
    if not isinstance(response, dict):
        return _rejected(safe_policy, "response_not_object")
    decision = response.get("decision")
    if decision == safe_schema.allow_continue_value:
        expected_keys = {"role", "episode_id", "decision", "public_report"}
    elif decision == safe_schema.reject_value:
        expected_keys = {"role", "episode_id", "decision", "rejection_reason"}
    else:
        return _rejected(safe_policy, "invalid_decision")
    if set(response) != expected_keys:
        return _rejected(safe_policy, "unexpected_or_missing_response_field")
    role = response["role"]
    episode_id = response["episode_id"]
    if not _is_nonempty_string(role) or role not in safe_schema.allowed_roles:
        return _rejected(safe_policy, "role_not_allowed")
    if role != expected_role:
        return _rejected(safe_policy, "role_mismatch")
    if not _is_nonempty_string(episode_id) or re.fullmatch(safe_schema.episode_pattern, episode_id) is None:
        return _rejected(safe_policy, "invalid_episode_id")
    if episode_id != expected_episode_id:
        return _rejected(safe_policy, "episode_mismatch")
    if decision == safe_schema.reject_value:
        reason = response["rejection_reason"]
        if not _is_nonempty_string(reason) or re.search(safe_schema.rejection_reason_pattern, reason) is None:
            return _rejected(safe_policy, "invalid_rejection_reason")
        return {
            "contract_valid": True,
            "decision": safe_schema.reject_value,
            "public_report": None,
            "rejection_code": "declared_reject",
        }
    report = response["public_report"]
    if not isinstance(report, dict):
        return _rejected(safe_policy, "public_report_not_object")
    if set(report) != set(safe_schema.required_public_fields):
        return _rejected(safe_policy, "unexpected_or_missing_public_field")
    if any(not _is_nonempty_string(report[field]) for field in safe_schema.required_public_fields):
        return _rejected(safe_policy, "invalid_public_field_value")
    if re.fullmatch(safe_schema.vendor_pattern, report["vendor_id"]) is None:
        return _rejected(safe_policy, "invalid_vendor_id")
    if report["registration_status"] not in safe_schema.registration_status_values:
        return _rejected(safe_policy, "invalid_registration_status")
    if report["risk_level"] not in safe_schema.risk_level_values:
        return _rejected(safe_policy, "invalid_risk_level")
    if report["recommendation"] not in safe_schema.recommendation_values:
        return _rejected(safe_policy, "invalid_recommendation")
    return {
        "contract_valid": True,
        "decision": safe_schema.allow_continue_value,
        "public_report": report,
        "rejection_code": None,
    }


def public_report_hash(report: Any) -> str | None:
    if report is None:
        return None
    if not isinstance(report, dict):
        return None
    try:
        return sha256_bytes(canonical_json_bytes(report))
    except StrictJsonError:
        return None
