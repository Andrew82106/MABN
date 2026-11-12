"""Strict, deterministic adapter for model-shaped JSON text.

The adapter never executes a tool and never performs I/O.  It returns only a
safe category and hash of an accepted action; raw response text is not retained.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .contract import canonical_json, stable_hash


class DuplicateKeyError(ValueError):
    pass


class NonFiniteError(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError("duplicate object key")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise NonFiniteError("non-finite JSON number")


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "null":
        return value is None
    return False


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, expected_type):
        return [f"{path}:wrong_type"]
    if "const" in schema and value != schema["const"]:
        return [f"{path}:const_mismatch"]
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            errors.extend(f"{path}:missing:{key}" for key in missing)
        properties = schema.get("properties", {})
        extra = sorted(set(value) - set(properties))
        if extra and schema.get("additionalProperties") is False:
            errors.extend(f"{path}:extra:{key}" for key in extra)
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(_schema_errors(value[key], child_schema, f"{path}.{key}"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) != schema["minItems"]:
            errors.append(f"{path}:array_length")
        if "maxItems" in schema and len(value) != schema["maxItems"]:
            errors.append(f"{path}:array_length")
        if "items" in schema:
            for index, child in enumerate(value):
                errors.extend(_schema_errors(child, schema["items"], f"{path}[{index}]"))
    return errors


def _category(errors: list[str], parsed: dict[str, Any] | None, expected: dict[str, Any]) -> str:
    if parsed is not None:
        if "tool_calls" in parsed:
            return "native_tool_calls"
        if "function_call" in parsed:
            return "native_function_call"
        if "refusal" in parsed:
            return "refusal"
        if any(key in parsed for key in ("reasoning", "thought", "analysis")):
            return "reasoning_field"
    if any(":missing:" in error for error in errors):
        return "missing_field"
    if any(":extra:" in error for error in errors):
        return "extra_field"
    if any(":wrong_type" in error for error in errors):
        return "wrong_type"
    if parsed is not None:
        if parsed.get("kind") != expected.get("kind"):
            return "wrong_kind"
        if expected.get("kind") == "tool_call":
            if parsed.get("tool_name") != expected.get("tool_name"):
                return "wrong_tool"
            if parsed.get("arguments") != expected.get("arguments"):
                return "wrong_arguments"
        elif parsed.get("content") != expected.get("content"):
            return "semantic_error"
    return "schema_error"


def adapt(raw: Any, card: dict[str, Any]) -> dict[str, Any]:
    """Parse and strictly validate one fake/model response."""
    if not isinstance(raw, (str, bytes, bytearray)):
        return {"passed": False, "category": "non_json", "action_hash": None}
    try:
        parsed = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except DuplicateKeyError:
        return {"passed": False, "category": "duplicate_key", "action_hash": None}
    except NonFiniteError:
        return {"passed": False, "category": "nonfinite", "action_hash": None}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"passed": False, "category": "non_json", "action_hash": None}
    if not isinstance(parsed, dict):
        return {"passed": False, "category": "non_json", "action_hash": None}
    errors = _schema_errors(parsed, card["schema"])
    if errors:
        return {"passed": False, "category": _category(errors, parsed, card["expected_action"]), "action_hash": None}
    return {"passed": True, "category": "accepted", "action_hash": _hash(parsed)}


__all__ = ["adapt", "DuplicateKeyError", "NonFiniteError"]
