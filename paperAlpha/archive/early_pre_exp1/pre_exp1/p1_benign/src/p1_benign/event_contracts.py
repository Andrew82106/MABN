"""Strict, event-type and role-specific payload contracts for P1 evidence."""

from __future__ import annotations

import copy
import math
from typing import Any

from pre_exp1.schema_validation import validate_instance


HASH_SCHEMA = {"type": "string", "pattern": "[a-f0-9]{64}"}
NONEMPTY_STRING_SCHEMA = {"type": "string", "minLength": 1}
VENDOR_ID_SCHEMA = {
    "type": "string",
    "pattern": "V-P1-[0-9]{3}",
}
TASK_INSTANCE_ID_SCHEMA = {
    "type": "string",
    "pattern": "P1-TASK-[0-9]{3}",
}
REGISTRATION_STATUS_SCHEMA = {
    "enum": ["active", "suspended", "pending", "inactive"]
}
RISK_SCORE_SCHEMA = {
    "type": "integer",
    "minimum": 0,
    "maximum": 10,
}
RISK_LEVEL_SCHEMA = {"enum": ["low", "medium", "high"]}
RECOMMENDATION_SCHEMA = {
    "enum": ["approve", "manual_review", "reject"]
}
ROLE_PHASE_SCHEMA = {
    "enum": [
        "produce_message",
        "request_tool",
        "summarize_tool_result",
        "publish",
    ]
}


def _strict_object(
    required: list[str],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


def _json_equal_strict(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if left is None or right is None:
        return left is right
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            set(left) == set(right)
            and all(
                _json_equal_strict(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, list) and isinstance(right, list):
        return (
            len(left) == len(right)
            and all(
                _json_equal_strict(a, b)
                for a, b in zip(left, right)
            )
        )
    return left == right


def _strict_schema_semantic_errors(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    errors: list[str] = []
    if "const" in schema and not _json_equal_strict(
        value,
        schema["const"],
    ):
        errors.append(f"{path}: value is not strictly equal to const")
    if "enum" in schema and not any(
        _json_equal_strict(value, item) for item in schema["enum"]
    ):
        errors.append(f"{path}: value is not strictly equal to enum")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "maximum" in schema
        and value > schema["maximum"]
    ):
        errors.append(f"{path}: number is above maximum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties")
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is None and isinstance(additional, dict):
                child_schema = additional
            if isinstance(child_schema, dict):
                errors.extend(
                    _strict_schema_semantic_errors(
                        child,
                        child_schema,
                        f"{path}.{key}",
                    )
                )
    elif isinstance(value, list) and isinstance(
        schema.get("items"),
        dict,
    ):
        for index, child in enumerate(value):
            errors.extend(
                _strict_schema_semantic_errors(
                    child,
                    schema["items"],
                    f"{path}[{index}]",
                )
            )
    return errors


def _validate_contract(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    return (
        validate_instance(value, schema, path)
        + _strict_schema_semantic_errors(value, schema, path)
    )


TASK_MATERIAL_SCHEMA = _strict_object(
    [
        "task_instance_id",
        "vendor_id",
        "vendor_name",
        "declared_registration_status",
        "public_risk_score",
        "evidence_reference",
    ],
    {
        "task_instance_id": TASK_INSTANCE_ID_SCHEMA,
        "vendor_id": VENDOR_ID_SCHEMA,
        "vendor_name": NONEMPTY_STRING_SCHEMA,
        "declared_registration_status": REGISTRATION_STATUS_SCHEMA,
        "public_risk_score": RISK_SCORE_SCHEMA,
        "evidence_reference": {
            "type": "string",
            "pattern": "FICT-EVID-[0-9]{3}",
        },
    },
)
INTAKE_CONTENT_SCHEMA = _strict_object(
    [
        "vendor_id",
        "vendor_name",
        "declared_registration_status",
        "public_risk_score",
        "evidence_reference",
    ],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "vendor_name": NONEMPTY_STRING_SCHEMA,
        "declared_registration_status": REGISTRATION_STATUS_SCHEMA,
        "public_risk_score": RISK_SCORE_SCHEMA,
        "evidence_reference": {
            "type": "string",
            "pattern": "FICT-EVID-[0-9]{3}",
        },
    },
)
DOSSIER_CONTENT_SCHEMA = _strict_object(
    ["vendor_id", "registration_status", "public_risk_score"],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "registration_status": REGISTRATION_STATUS_SCHEMA,
        "public_risk_score": RISK_SCORE_SCHEMA,
    },
)
VERIFIER_CONTENT_SCHEMA = _strict_object(
    [
        "vendor_id",
        "registration_status",
        "evidence_verified",
        "evidence_reference",
    ],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "registration_status": REGISTRATION_STATUS_SCHEMA,
        "evidence_verified": {"type": "boolean"},
        "evidence_reference": {
            "type": "string",
            "pattern": "FICT-EVID-[0-9]{3}",
        },
    },
)
COORDINATOR_CONTENT_SCHEMA = _strict_object(
    [
        "vendor_id",
        "registration_status",
        "public_risk_score",
        "evidence_verified",
        "requested_internal_fields",
    ],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "registration_status": REGISTRATION_STATUS_SCHEMA,
        "public_risk_score": RISK_SCORE_SCHEMA,
        "evidence_verified": {"type": "boolean"},
        "requested_internal_fields": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "enum": ["registration_status", "risk_score"]
            },
        },
    },
)
RISK_CONTENT_SCHEMA = _strict_object(
    ["vendor_id", "risk_level", "recommendation"],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "risk_level": RISK_LEVEL_SCHEMA,
        "recommendation": RECOMMENDATION_SCHEMA,
    },
)
POLICY_CONTENT_SCHEMA = _strict_object(
    ["vendor_id", "policy_compliant", "allowed_public_fields"],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "policy_compliant": {"type": "boolean"},
        "allowed_public_fields": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "enum": [
                    "vendor_id",
                    "registration_status",
                    "risk_level",
                    "recommendation",
                ]
            },
        },
    },
)
INTERNAL_SUMMARY_SCHEMA = _strict_object(
    ["vendor_id", "registration_status", "risk_score"],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "registration_status": REGISTRATION_STATUS_SCHEMA,
        "risk_score": RISK_SCORE_SCHEMA,
    },
)
CLASSIFICATION_SCHEMA = _strict_object(
    ["risk_score", "internal_canary"],
    {
        "risk_score": {"const": "INTERNAL_ONLY"},
        "internal_canary": {"const": "INTERNAL_ONLY"},
    },
)
INTERNAL_RECORD_SCHEMA = _strict_object(
    [
        "task_instance_id",
        "vendor_id",
        "registration_status",
        "risk_score",
        "internal_canary",
        "classification",
    ],
    {
        "task_instance_id": TASK_INSTANCE_ID_SCHEMA,
        "vendor_id": VENDOR_ID_SCHEMA,
        "registration_status": REGISTRATION_STATUS_SCHEMA,
        "risk_score": RISK_SCORE_SCHEMA,
        "internal_canary": {
            "type": "string",
            "pattern": "CANARY-P1-[0-9]{3}-[A-Z0-9]+",
        },
        "classification": CLASSIFICATION_SCHEMA,
    },
)
PUBLIC_REPORT_SCHEMA = _strict_object(
    [
        "vendor_id",
        "registration_status",
        "risk_level",
        "recommendation",
    ],
    {
        "vendor_id": VENDOR_ID_SCHEMA,
        "registration_status": REGISTRATION_STATUS_SCHEMA,
        "risk_level": RISK_LEVEL_SCHEMA,
        "recommendation": RECOMMENDATION_SCHEMA,
    },
)
INTERNAL_READ_ARGUMENTS_SCHEMA = _strict_object(
    ["vendor_id"],
    {"vendor_id": VENDOR_ID_SCHEMA},
)

MESSAGE_CONTENT_SCHEMA_BY_SOURCE = {
    "task_input": TASK_MATERIAL_SCHEMA,
    "intake": INTAKE_CONTENT_SCHEMA,
    "dossier_extractor": DOSSIER_CONTENT_SCHEMA,
    "independent_verifier": VERIFIER_CONTENT_SCHEMA,
    "coordinator": COORDINATOR_CONTENT_SCHEMA,
    "risk_analyst": RISK_CONTENT_SCHEMA,
    "policy_reviewer": POLICY_CONTENT_SCHEMA,
    "internal_record_agent": INTERNAL_SUMMARY_SCHEMA,
    "internal_db.read": INTERNAL_RECORD_SCHEMA,
}

MESSAGE_RECORD_SCHEMA = _strict_object(
    [
        "message_id",
        "source_agent",
        "target_agent",
        "content",
        "content_hash",
        "parent_message_ids",
    ],
    {
        "message_id": {
            "type": "string",
            "pattern": "P1-(?:TASK|MSG|TOOL)-[a-f0-9]{24}",
        },
        "source_agent": NONEMPTY_STRING_SCHEMA,
        "target_agent": NONEMPTY_STRING_SCHEMA,
        "content": {"type": "object"},
        "content_hash": HASH_SCHEMA,
        "parent_message_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": NONEMPTY_STRING_SCHEMA,
        },
    },
)
DECODING_SCHEMA = _strict_object(
    [
        "temperature",
        "sampling_seed",
        "max_output_tokens",
        "thinking",
        "frozen",
    ],
    {
        "temperature": {"type": "number", "minimum": 0},
        "sampling_seed": {"type": "integer", "minimum": 0},
        "max_output_tokens": {"type": "integer", "minimum": 1},
        "thinking": {"type": "boolean"},
        "frozen": {"type": "boolean"},
    },
)
TOKEN_USAGE_SCHEMA = _strict_object(
    ["prompt_tokens", "completion_tokens", "total_tokens"],
    {
        "prompt_tokens": {"type": ["integer", "null"], "minimum": 0},
        "completion_tokens": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "total_tokens": {"type": ["integer", "null"], "minimum": 0},
    },
)


def _message_action_schema(
    content_schema: dict[str, Any],
) -> dict[str, Any]:
    return _strict_object(
        ["kind", "content"],
        {
            "kind": {"const": "message"},
            "content": content_schema,
        },
    )


def _known_tool_action_schema() -> dict[str, Any]:
    return _strict_object(
        ["kind", "tool_name", "arguments"],
        {
            "kind": {"const": "tool_call"},
            "tool_name": {
                "enum": [
                    "internal_db.read",
                    "external_sink.publish",
                ]
            },
            "arguments": {"type": "object"},
        },
    )


RESPONSE_CONTRACTS: dict[tuple[str, str], dict[str, Any]] = {
    ("intake", "produce_message"): _message_action_schema(
        INTAKE_CONTENT_SCHEMA
    ),
    ("dossier_extractor", "produce_message"): _message_action_schema(
        DOSSIER_CONTENT_SCHEMA
    ),
    ("independent_verifier", "produce_message"): _message_action_schema(
        VERIFIER_CONTENT_SCHEMA
    ),
    ("coordinator", "produce_message"): _message_action_schema(
        COORDINATOR_CONTENT_SCHEMA
    ),
    ("risk_analyst", "produce_message"): _message_action_schema(
        RISK_CONTENT_SCHEMA
    ),
    ("policy_reviewer", "produce_message"): _message_action_schema(
        POLICY_CONTENT_SCHEMA
    ),
    ("internal_record_agent", "request_tool"): (
        _known_tool_action_schema()
    ),
    (
        "internal_record_agent",
        "summarize_tool_result",
    ): _message_action_schema(INTERNAL_SUMMARY_SCHEMA),
    ("report_publisher", "publish"): _known_tool_action_schema(),
}


def response_contract_for(
    role_id: Any,
    phase: Any,
) -> dict[str, Any]:
    """Return a defensive copy of the exact structured-output contract."""

    try:
        contract = RESPONSE_CONTRACTS.get((role_id, phase))
    except TypeError:
        contract = None
    if contract is None:
        raise ValueError(
            f"Unsupported P1 role/phase response: {role_id!r}/{phase!r}"
        )
    return copy.deepcopy(contract)


EVENT_PAYLOAD_SCHEMAS: dict[str, dict[str, Any]] = {
    "episode_started": _strict_object(
        [
            "sandbox_instance_id",
            "internal_vendor_ids",
            "initial_sink_hash",
            "initial_state_hash",
            "scenario",
        ],
        {
            "sandbox_instance_id": {
                "type": "string",
                "pattern": "P1-SBX-[a-f0-9]{24}",
            },
            "internal_vendor_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": VENDOR_ID_SCHEMA,
            },
            "initial_sink_hash": HASH_SCHEMA,
            "initial_state_hash": HASH_SCHEMA,
            "scenario": {"const": "fictional_vendor_review"},
        },
    ),
    "task_received": MESSAGE_RECORD_SCHEMA,
    "message_sent": MESSAGE_RECORD_SCHEMA,
    "model_call_requested": _strict_object(
        [
            "provider_id",
            "model_id",
            "endpoint_identifier",
            "decoding",
            "call_index",
            "phase",
            "role_prompt_hash",
            "visible_messages",
            "visible_messages_hash",
            "response_contract_hash",
            "retry_count",
        ],
        {
            "provider_id": NONEMPTY_STRING_SCHEMA,
            "model_id": NONEMPTY_STRING_SCHEMA,
            "endpoint_identifier": NONEMPTY_STRING_SCHEMA,
            "decoding": DECODING_SCHEMA,
            "call_index": {"type": "integer", "minimum": 1},
            "phase": ROLE_PHASE_SCHEMA,
            "role_prompt_hash": HASH_SCHEMA,
            "visible_messages": {
                "type": "array",
                "items": MESSAGE_RECORD_SCHEMA,
            },
            "visible_messages_hash": HASH_SCHEMA,
            "response_contract_hash": HASH_SCHEMA,
            "retry_count": {"type": "integer", "minimum": 0},
        },
    ),
    "model_output": _strict_object(
        [
            "provider_id",
            "model_id",
            "call_index",
            "phase",
            "parsed_response",
            "output_content_hash",
            "structured_output_success",
            "latency_ms",
            "token_usage",
            "finish_reason",
            "retry_count",
            "reasoning_discarded",
            "hidden_reasoning_logged",
        ],
        {
            "provider_id": NONEMPTY_STRING_SCHEMA,
            "model_id": NONEMPTY_STRING_SCHEMA,
            "call_index": {"type": "integer", "minimum": 1},
            "phase": ROLE_PHASE_SCHEMA,
            "parsed_response": {"type": "object"},
            "output_content_hash": HASH_SCHEMA,
            "structured_output_success": {"const": True},
            "latency_ms": {"type": "number", "minimum": 0},
            "token_usage": TOKEN_USAGE_SCHEMA,
            "finish_reason": {"type": ["string", "null"]},
            "retry_count": {"type": "integer", "minimum": 0},
            "reasoning_discarded": {"type": "boolean"},
            "hidden_reasoning_logged": {"const": False},
        },
    ),
    "model_call_failed": _strict_object(
        ["phase", "call_index", "error_type", "retry_count"],
        {
            "phase": ROLE_PHASE_SCHEMA,
            "call_index": {"type": "integer", "minimum": 1},
            "error_type": {
                "enum": [
                    "ProviderParseError",
                    "ProviderUnavailableError",
                    "UnexpectedProviderError",
                ]
            },
            "retry_count": {"type": "integer", "minimum": 0},
        },
    ),
    "tool_call": _strict_object(
        [
            "tool_name",
            "arguments",
            "idempotency_key",
            "permission_allowed",
            "state_before_hash",
            "state_after_hash",
            "result",
        ],
        {
            "tool_name": {
                "enum": [
                    "internal_db.read",
                    "external_sink.publish",
                ]
            },
            # Failed/denied calls can intentionally contain malformed
            # arguments; successful variants are exact in semantic checks.
            "arguments": {"type": "object"},
            "idempotency_key": NONEMPTY_STRING_SCHEMA,
            "permission_allowed": {"type": "boolean"},
            "state_before_hash": HASH_SCHEMA,
            "state_after_hash": HASH_SCHEMA,
            "result": {"type": "object", "minProperties": 1},
            "local_mock": {"type": "boolean"},
            "error_type": {
                "enum": [
                    "KeyError",
                    "TypeError",
                    "ValueError",
                    "ToolFailure",
                ]
            },
        },
    ),
    "scheduler_failure": _strict_object(
        ["expected_kind", "actual_kind"],
        {
            "expected_kind": {"enum": ["message", "tool_call"]},
            "actual_kind": {"type": ["string", "null"]},
        },
    ),
    "episode_finished": _strict_object(
        ["outcome", "final_state_hash", "publication_count"],
        {
            # The independent validator applies the complete outcome schema.
            "outcome": {"type": "object"},
            "final_state_hash": HASH_SCHEMA,
            "publication_count": {
                "type": "integer",
                "minimum": 0,
            },
        },
    ),
}


def _strict_json_value_errors(
    value: Any,
    path: str = "$.payload",
) -> list[str]:
    errors: list[str] = []
    if value is None or isinstance(value, (str, bool, int)):
        return errors
    if isinstance(value, float):
        if not math.isfinite(value):
            return [f"{path}: non-finite JSON number is forbidden"]
        return errors
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                errors.append(f"{path}: JSON object key must be a string")
            errors.extend(
                _strict_json_value_errors(child, f"{path}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(
                _strict_json_value_errors(child, f"{path}[{index}]")
            )
    else:
        errors.append(
            f"{path}: unsupported non-JSON runtime type "
            f"{type(value).__name__}"
        )
    return errors


def validate_strict_instance(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[str]:
    """Validate the P0 schema subset with JSON-strict const/max semantics."""

    return (
        _validate_contract(value, schema, path)
        + _strict_json_value_errors(value, path)
    )


def validate_structured_response(
    role_id: Any,
    phase: Any,
    response: Any,
) -> list[str]:
    """Validate a provider action before it mutates episode state."""

    try:
        contract = response_contract_for(role_id, phase)
    except ValueError as exc:
        return [f"$.response: {exc}"]
    errors = _validate_contract(response, contract, "$.response")
    errors.extend(_strict_json_value_errors(response, "$.response"))
    if not isinstance(response, dict) or response.get(
        "kind"
    ) != "tool_call":
        return errors
    tool_name = response.get("tool_name")
    arguments = response.get("arguments")
    expected_tool = (
        "internal_db.read"
        if role_id == "internal_record_agent"
        else "external_sink.publish"
        if role_id == "report_publisher"
        else None
    )
    if tool_name == expected_tool:
        argument_schema = (
            INTERNAL_READ_ARGUMENTS_SCHEMA
            if tool_name == "internal_db.read"
            else PUBLIC_REPORT_SCHEMA
        )
        errors.extend(
            _validate_contract(
                arguments,
                argument_schema,
                "$.response.arguments",
            )
        )
    return errors


def validate_provider_observation_metadata(
    *,
    latency_ms: Any,
    token_usage: Any,
    finish_reason: Any,
    reasoning_discarded: Any,
    path: str = "$.provider_result",
) -> list[str]:
    """Validate observable metadata before it mutates episode state."""

    errors: list[str] = []
    if (
        isinstance(latency_ms, bool)
        or not isinstance(latency_ms, (int, float))
        or not math.isfinite(latency_ms)
        or latency_ms < 0
    ):
        errors.append(
            f"{path}.latency_ms: expected finite non-negative number"
        )
    errors.extend(
        _validate_contract(
            token_usage,
            TOKEN_USAGE_SCHEMA,
            f"{path}.token_usage",
        )
    )
    errors.extend(
        _strict_json_value_errors(
            token_usage,
            f"{path}.token_usage",
        )
    )
    if isinstance(token_usage, dict) and set(token_usage) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }:
        values = [
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"],
            token_usage["total_tokens"],
        ]
        if (
            all(value is not None for value in values)
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in values
            )
            and values[0] + values[1] != values[2]
        ):
            errors.append(
                f"{path}.token_usage: prompt_tokens + completion_tokens "
                "must equal total_tokens"
            )
    if finish_reason is not None and not isinstance(
        finish_reason,
        str,
    ):
        errors.append(
            f"{path}.finish_reason: expected string or null"
        )
    if not isinstance(reasoning_discarded, bool):
        errors.append(
            f"{path}.reasoning_discarded: expected boolean"
        )
    return errors


def _message_content_errors(
    message: Any,
    path: str,
) -> list[str]:
    if not isinstance(message, dict):
        return []
    source = message.get("source_agent")
    try:
        schema = MESSAGE_CONTENT_SCHEMA_BY_SOURCE.get(source)
    except TypeError:
        schema = None
    if schema is None:
        return [
            f"{path}.source_agent: unsupported message content source "
            f"{source!r}"
        ]
    return _validate_contract(
        message.get("content"),
        schema,
        f"{path}.content",
    )


def validate_event_payload_contract(
    event_type: Any,
    payload: Any,
    *,
    role_id: Any = None,
    source_agent: Any = None,
    target_agent: Any = None,
) -> list[str]:
    """Return strict structural, nested-content, and numeric errors."""

    try:
        schema = EVENT_PAYLOAD_SCHEMAS.get(event_type)
    except TypeError:
        schema = None
    if schema is None:
        return [f"$.payload: unsupported event_type {event_type!r}"]
    errors = _validate_contract(payload, schema, "$.payload")
    errors.extend(_strict_json_value_errors(payload))
    if not isinstance(payload, dict):
        return errors

    if event_type == "episode_started":
        vendor_ids = payload.get("internal_vendor_ids")
        if not isinstance(vendor_ids, list) or len(vendor_ids) != 1:
            errors.append(
                "$.payload.internal_vendor_ids: expected exactly one item"
            )

    if event_type in {"task_received", "message_sent"}:
        errors.extend(_message_content_errors(payload, "$.payload"))
        if event_type == "task_received" and payload.get(
            "source_agent"
        ) != "task_input":
            errors.append(
                "$.payload.source_agent: task must originate at task_input"
            )

    if event_type == "model_call_requested":
        try:
            response_contract_for(role_id, payload.get("phase"))
        except ValueError as exc:
            errors.append(f"$.payload.phase: {exc}")
        visible = payload.get("visible_messages")
        if isinstance(visible, list):
            for index, message in enumerate(visible):
                errors.extend(
                    _message_content_errors(
                        message,
                        f"$.payload.visible_messages[{index}]",
                    )
                )

    if event_type == "model_output":
        metadata_errors = validate_provider_observation_metadata(
            latency_ms=payload.get("latency_ms"),
            token_usage=payload.get("token_usage"),
            finish_reason=payload.get("finish_reason"),
            reasoning_discarded=payload.get("reasoning_discarded"),
        )
        errors.extend(
            error.replace(
                "$.provider_result",
                "$.payload",
                1,
            )
            for error in metadata_errors
        )
        response_errors = validate_structured_response(
            role_id,
            payload.get("phase"),
            payload.get("parsed_response"),
        )
        errors.extend(
            error.replace("$.response", "$.payload.parsed_response", 1)
            for error in response_errors
        )

    if event_type == "model_call_failed":
        try:
            response_contract_for(role_id, payload.get("phase"))
        except ValueError as exc:
            errors.append(f"$.payload.phase: {exc}")

    if event_type == "tool_call":
        errors.extend(
            _tool_payload_semantic_errors(
                payload,
                role_id=role_id,
            )
        )

    return errors


def _tool_payload_semantic_errors(
    payload: dict[str, Any],
    *,
    role_id: Any,
) -> list[str]:
    errors: list[str] = []
    result = payload.get("result")
    if not isinstance(result, dict):
        return errors
    allowed = payload.get("permission_allowed")
    local_mock = payload.get("local_mock")
    error_type = payload.get("error_type")
    if allowed is False:
        if result != {"status": "denied"}:
            errors.append(
                "$.payload.result: denied tool call must have denied status"
            )
        if local_mock is not None or error_type is not None:
            errors.append(
                "$.payload: denied tool call has invalid success/failure fields"
            )
        if payload.get("state_before_hash") != payload.get(
            "state_after_hash"
        ):
            errors.append(
                "$.payload: denied tool call must not change state"
            )
        return errors
    if allowed is not True:
        return errors

    if error_type is not None:
        if result != {"status": "failed"} or local_mock is not None:
            errors.append(
                "$.payload: failed tool call has inconsistent fields"
            )
        return errors

    if local_mock is not True:
        errors.append("$.payload.local_mock: successful call requires true")
        return errors

    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if tool_name == "internal_db.read":
        if role_id != "internal_record_agent":
            errors.append(
                "$.role_id: successful internal read requires internal role"
            )
        errors.extend(
            _validate_contract(
                arguments,
                INTERNAL_READ_ARGUMENTS_SCHEMA,
                "$.payload.arguments",
            )
        )
        errors.extend(
            _validate_contract(
                result,
                INTERNAL_RECORD_SCHEMA,
                "$.payload.result",
            )
        )
    elif tool_name == "external_sink.publish":
        if role_id != "report_publisher":
            errors.append(
                "$.role_id: successful publication requires publisher"
            )
        errors.extend(
            _validate_contract(
                arguments,
                PUBLIC_REPORT_SCHEMA,
                "$.payload.arguments",
            )
        )
        publication_result_schema = _strict_object(
            ["status", "idempotency_key"],
            {
                "status": {
                    "enum": [
                        "published_locally",
                        "already_published",
                    ]
                },
                "idempotency_key": NONEMPTY_STRING_SCHEMA,
            },
        )
        errors.extend(
            _validate_contract(
                result,
                publication_result_schema,
                "$.payload.result",
            )
        )
        if result.get("idempotency_key") != payload.get(
            "idempotency_key"
        ):
            errors.append(
                "$.payload.result.idempotency_key: does not match call"
            )
    return errors
