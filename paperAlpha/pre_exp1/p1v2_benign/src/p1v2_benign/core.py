"""Strict local primitives and path boundaries for P1v2-A offline admission."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any


CODE_NAMESPACE_ROOT = Path(__file__).resolve().parents[2]
PAPER_ALPHA_ROOT = CODE_NAMESPACE_ROOT.parents[1]
DEFAULT_DATA_ROOT = PAPER_ALPHA_ROOT / "data" / "pre_exp1" / "p1v2_benign"
DEFAULT_CONFIG_PATH = CODE_NAMESPACE_ROOT / "configs" / "readiness.json"
DEFAULT_FIXTURE_PATH = CODE_NAMESPACE_ROOT / "fixtures" / "response_cases.json"
DEFAULT_SCHEMA_PATH = CODE_NAMESPACE_ROOT / "schemas" / "response_contract.schema.json"
DEFAULT_PROMPT_PATH = CODE_NAMESPACE_ROOT / "prompts" / "public_response_contract.txt"
DEFAULT_RUN_ID = "P1V2-READINESS-DRY-20260801T000002000000Z"
RUN_ID_PREFIX = "P1V2-READINESS-DRY-"

CONFIG_KEYS = {
    "allow_continue_value",
    "allowed_roles",
    "config_version",
    "data_role",
    "expected_case_ids",
    "execution_source",
    "reject_value",
    "required_public_fields",
    "run_id_prefix",
}
TRACEABLE_DIRECTORIES = ("configs", "fixtures", "prompts", "schemas", "scripts", "src")
TRACEABLE_SUFFIXES = {".json", ".py", ".txt"}
ENTRY_SCRIPT_RELATIVE_PATHS = (
    "scripts/run_readiness_dry.py",
    "scripts/validate_readiness_dry.py",
    "scripts/replay_readiness_dry.py",
)
OLD_READONLY_RELATIVE_ROOTS = (
    "pre_exp1/p1_benign",
    "pre_exp1/p1_model_qualification",
    "pre_exp1/p1_scientific_benign",
    "data/shared/p1_benign",
    "data/pre_exp1/p1_benign",
    "data/pre_exp1/p1_model_qualification",
    "data/pre_exp1/p1_scientific_benign",
)
OLD_READONLY_ROOTS = tuple(PAPER_ALPHA_ROOT / relative for relative in OLD_READONLY_RELATIVE_ROOTS)


class StrictJsonError(ValueError):
    """A JSON decoding failure exposed only through a stable machine code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class PathBoundaryError(ValueError):
    """A path was rejected before it could be opened, created, or traversed."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise StrictJsonError("duplicate_json_key")
        output[key] = value
    return output


def _reject_nonstandard_number(_: str) -> None:
    raise StrictJsonError("nonstandard_json_number")


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictJsonError("invalid_json_number") from exc
    if not math.isfinite(parsed):
        raise StrictJsonError("nonfinite_json_number")
    return parsed


def strict_json_loads(text: str) -> Any:
    """Parse only finite, duplicate-free JSON with no blank payloads."""

    if not isinstance(text, str) or not text.strip():
        raise StrictJsonError("empty_json")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonstandard_number,
            parse_float=_finite_float,
        )
    except StrictJsonError:
        raise
    except (TypeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrictJsonError("invalid_json") from exc


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictJsonError("noncanonical_json_value") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise StrictJsonError("invalid_hash_text")
    return sha256_bytes(value.encode("utf-8"))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _raw_path(value: Path | str) -> str:
    """Reject lexical hazards before any resolve/open/create operation."""

    if not isinstance(value, (Path, str)):
        raise PathBoundaryError("invalid_path_type")
    raw = str(value)
    if not raw:
        raise PathBoundaryError("empty_path")
    normalized = raw.replace("/", "\\")
    lowered = normalized.lower()
    if lowered.startswith("\\\\"):
        raise PathBoundaryError("forbidden_network_or_device_path")
    parts = [part.lower() for part in PureWindowsPath(normalized).parts]
    if ".env" in parts:
        raise PathBoundaryError("forbidden_environment_path")
    if ".." in parts:
        raise PathBoundaryError("forbidden_parent_traversal")
    enclosed = "\\" + lowered.strip("\\") + "\\"
    for relative_root in OLD_READONLY_RELATIVE_ROOTS:
        marker = relative_root.replace("/", "\\").lower().strip("\\")
        if f"\\{marker}\\" in enclosed:
            raise PathBoundaryError("forbidden_legacy_path")
    return raw


def _resolve_without_opening(value: Path | str) -> Path:
    raw = _raw_path(value)
    candidate = Path(raw).resolve(strict=False)
    for old_root in OLD_READONLY_ROOTS:
        try:
            candidate.relative_to(old_root.resolve(strict=False))
        except ValueError:
            continue
        raise PathBoundaryError("forbidden_legacy_path")
    return candidate


def _normalized_windows_path(value: Path | str) -> str:
    return _raw_path(value).replace("/", "\\").lower().rstrip("\\")


def _is_lexically_under(value: Path | str, root: Path | str) -> bool:
    candidate = _normalized_windows_path(value)
    allowed = _normalized_windows_path(root)
    return candidate == allowed or candidate.startswith(allowed + "\\")


def _require_under(value: Path | str, allowed_root: Path | str, *, code: str) -> Path:
    candidate = _resolve_without_opening(value)
    root = _resolve_without_opening(allowed_root)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathBoundaryError(code) from exc
    return candidate


def require_p1v2_data_root(value: Path | str) -> Path:
    if not _is_lexically_under(value, DEFAULT_DATA_ROOT):
        raise PathBoundaryError("data_root_outside_p1v2_namespace")
    return _require_under(value, DEFAULT_DATA_ROOT, code="data_root_outside_p1v2_namespace")


def require_canonical_input(value: Path | str, expected: Path) -> Path:
    if _normalized_windows_path(value) != _normalized_windows_path(expected):
        raise PathBoundaryError("noncanonical_input_path")
    candidate = _require_under(value, CODE_NAMESPACE_ROOT, code="input_outside_p1v2_code_namespace")
    expected_resolved = _require_under(expected, CODE_NAMESPACE_ROOT, code="canonical_input_invalid")
    if candidate != expected_resolved:
        raise PathBoundaryError("noncanonical_input_path")
    return candidate


def resolve_generated_path(data_root: Path | str, relative_path: str) -> Path:
    root = require_p1v2_data_root(data_root)
    if not isinstance(relative_path, str) or not relative_path:
        raise PathBoundaryError("invalid_artifact_path")
    normalized = relative_path.replace("/", "\\")
    pure = PureWindowsPath(normalized)
    if (
        pure.is_absolute()
        or normalized.startswith("\\")
        or ".." in pure.parts
        or ".env" in [part.lower() for part in pure.parts]
    ):
        raise PathBoundaryError("artifact_path_escapes_data_root")
    return _require_under(root / relative_path, root, code="artifact_path_escapes_data_root")


def _read_bytes(path: Path | str, *, allowed_root: Path | str, code: str) -> tuple[Path, bytes]:
    safe_path = _require_under(path, allowed_root, code=code)
    try:
        return safe_path, safe_path.read_bytes()
    except OSError as exc:
        raise StrictJsonError("unreadable_input") from exc


def _decode_utf8(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJsonError("invalid_utf8") from exc


def read_strict_json(path: Path | str, *, allowed_root: Path | str) -> Any:
    _, value = _read_bytes(path, allowed_root=allowed_root, code="read_path_outside_allowed_root")
    return strict_json_loads(_decode_utf8(value))


def read_strict_text(path: Path | str, *, allowed_root: Path | str) -> str:
    _, value = _read_bytes(path, allowed_root=allowed_root, code="read_path_outside_allowed_root")
    return _decode_utf8(value)


def read_jsonl(path: Path | str, *, allowed_root: Path | str) -> list[dict[str, Any]]:
    _, content = _read_bytes(path, allowed_root=allowed_root, code="read_path_outside_allowed_root")
    rows: list[dict[str, Any]] = []
    for line in _decode_utf8(content).splitlines():
        if not line.strip():
            raise StrictJsonError("empty_jsonl_line")
        value = strict_json_loads(line)
        if not isinstance(value, dict):
            raise StrictJsonError("jsonl_row_not_object")
        rows.append(value)
    return rows


def sha256_file(path: Path | str, *, allowed_root: Path | str) -> str:
    _, value = _read_bytes(path, allowed_root=allowed_root, code="hash_path_outside_allowed_root")
    return sha256_bytes(value)


def write_json(path: Path | str, value: Any, *, allowed_root: Path | str) -> None:
    safe_path = _require_under(path, allowed_root, code="write_path_outside_allowed_root")
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path | str, rows: list[dict[str, Any]], *, allowed_root: Path | str) -> None:
    safe_path = _require_under(path, allowed_root, code="write_path_outside_allowed_root")
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def write_text(path: Path | str, value: str, *, allowed_root: Path | str) -> None:
    if not isinstance(value, str):
        raise StrictJsonError("invalid_text_output")
    safe_path = _require_under(path, allowed_root, code="write_path_outside_allowed_root")
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(value, encoding="utf-8")


def require_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not re.fullmatch(r"P1V2-READINESS-DRY-[A-Z0-9T-]+Z", run_id):
        raise ValueError("invalid_run_id")


def artifact_paths(run_id: str) -> dict[str, str]:
    require_run_id(run_id)
    return {
        "events": f"readiness_dry/raw/events_{run_id}.jsonl",
        "outcomes": f"readiness_dry/processed/outcomes_{run_id}.jsonl",
        "validation": f"readiness_dry/interim/validation_{run_id}.json",
        "replay": f"readiness_dry/interim/replay_{run_id}.json",
        "manifest": f"readiness_dry/manifests/manifest_{run_id}.json",
        "report": f"readiness_dry/reports/run_{run_id}.md",
    }


def ensure_new_output_paths(data_root: Path | str, run_id: str) -> dict[str, Path]:
    root = require_p1v2_data_root(data_root)
    paths = {name: resolve_generated_path(root, relative) for name, relative in artifact_paths(run_id).items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("dry_run_id_already_exists")
    return paths


def _unique_nonempty_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )


def _canonical_identity(path: Path, content: bytes) -> dict[str, Any]:
    safe_path = _require_under(path, CODE_NAMESPACE_ROOT, code="provenance_path_outside_code_namespace")
    return {
        "relative_path": f"pre_exp1/p1v2_benign/{safe_path.relative_to(CODE_NAMESPACE_ROOT).as_posix()}",
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def read_canonical_json_with_identity(path: Path, expected: Path) -> tuple[Any, dict[str, Any]]:
    canonical = require_canonical_input(path, expected)
    safe_path, content = _read_bytes(canonical, allowed_root=CODE_NAMESPACE_ROOT, code="read_path_outside_allowed_root")
    return strict_json_loads(_decode_utf8(content)), _canonical_identity(safe_path, content)


def read_canonical_text_with_identity(path: Path, expected: Path) -> tuple[str, dict[str, Any]]:
    canonical = require_canonical_input(path, expected)
    safe_path, content = _read_bytes(canonical, allowed_root=CODE_NAMESPACE_ROOT, code="read_path_outside_allowed_root")
    return _decode_utf8(content), _canonical_identity(safe_path, content)


def _validate_config_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CONFIG_KEYS:
        raise ValueError("invalid_readiness_config_keys")
    if type(value["config_version"]) is not int or value["config_version"] != 1:
        raise ValueError("unsupported_readiness_config_version")
    if value["data_role"] != "p1v2_readiness_dry" or value["execution_source"] != "deterministic_fixture":
        raise ValueError("invalid_readiness_identity_config")
    if value["run_id_prefix"] != RUN_ID_PREFIX:
        raise ValueError("invalid_config_run_id_prefix")
    if not isinstance(value["allow_continue_value"], str) or not isinstance(value["reject_value"], str):
        raise ValueError("invalid_decision_config")
    if not value["allow_continue_value"] or value["allow_continue_value"] == value["reject_value"]:
        raise ValueError("ambiguous_decision_values")
    for list_key in ("allowed_roles", "required_public_fields", "expected_case_ids"):
        if not _unique_nonempty_strings(value[list_key]):
            raise ValueError(f"invalid_config_{list_key}")
    return value


def _validate_config_schema_alignment(config: dict[str, Any]) -> None:
    """Make the executable config unusable when it drifts from the real schema."""

    from .contract import ContractPolicy, load_schema_contract, policy_matches_schema

    policy = ContractPolicy.from_mapping(
        {
            "allow_continue_value": config["allow_continue_value"],
            "allowed_roles": config["allowed_roles"],
            "reject_value": config["reject_value"],
            "required_public_fields": config["required_public_fields"],
        }
    )
    if not policy_matches_schema(policy, load_schema_contract()):
        raise ValueError("config_schema_mismatch")


def load_config_with_identity(path: Path | str = DEFAULT_CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    value, identity = read_canonical_json_with_identity(Path(path), DEFAULT_CONFIG_PATH)
    config = _validate_config_value(value)
    _validate_config_schema_alignment(config)
    return config, identity


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return load_config_with_identity(path)[0]


def _identity(path: Path) -> dict[str, Any]:
    safe_path, content = _read_bytes(path, allowed_root=CODE_NAMESPACE_ROOT, code="provenance_path_outside_code_namespace")
    return _canonical_identity(safe_path, content)


def capture_execution_inputs(
    *,
    config_identity: dict[str, Any] | None = None,
    fixture_identity: dict[str, Any] | None = None,
    schema_identity: dict[str, Any] | None = None,
    prompt_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_modules = [
        _identity(path)
        for path in sorted((CODE_NAMESPACE_ROOT / "src").rglob("*.py"))
        if path.is_file()
    ]
    return {
        "config": config_identity if config_identity is not None else _identity(DEFAULT_CONFIG_PATH),
        "entry_scripts": [_identity(CODE_NAMESPACE_ROOT / relative) for relative in ENTRY_SCRIPT_RELATIVE_PATHS],
        "fixture": fixture_identity if fixture_identity is not None else _identity(DEFAULT_FIXTURE_PATH),
        "prompt": prompt_identity if prompt_identity is not None else _identity(DEFAULT_PROMPT_PATH),
        "schema": schema_identity if schema_identity is not None else _identity(DEFAULT_SCHEMA_PATH),
        "source_modules": source_modules,
    }


def capture_provenance() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for directory_name in TRACEABLE_DIRECTORIES:
        directory = CODE_NAMESPACE_ROOT / directory_name
        if not directory.is_dir():
            raise ValueError("missing_traceable_directory")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in TRACEABLE_SUFFIXES:
                identity = _identity(path)
                records.append(
                    {
                        "path": identity["relative_path"],
                        "sha256": identity["sha256"],
                        "size_bytes": identity["size_bytes"],
                    }
                )
    if not records:
        raise ValueError("empty_provenance")
    return {
        "algorithm": "sha256",
        "files": records,
        "tree_sha256": sha256_bytes(canonical_json_bytes(records)),
    }
