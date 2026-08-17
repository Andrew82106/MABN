"""Path and serialization boundaries for the offline qualification framework."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .errors import PathPolicyError, QualificationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[2]
DEFAULT_DATA_ROOT = (
    WORKSPACE_ROOT / "paperAlpha" / "data" / "pre_exp1" / "p1v2_qualification"
)

RUN_ID_PATTERN = re.compile(r"^P1V2Q-READINESS-DRY-[0-9]{8}T[0-9]{12}Z$")
SAFE_INPUT_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".py", ".md"})
SAFE_OUTPUT_SUFFIXES = frozenset({".json", ".jsonl", ".md"})


def _raw_path_is_forbidden(path: str | Path) -> bool:
    raw = str(path).replace("/", "\\")
    lowered = raw.lower()
    if lowered.startswith("\\\\") or lowered.startswith("\\\\?\\") or lowered.startswith("\\\\.\\"):
        return True
    parts = [part for part in raw.split("\\") if part]
    return any(part == ".." or part.lower() == ".env" for part in parts)


def _ensure_safe_path(
    path: str | Path,
    root: Path,
    allowed_suffixes: Iterable[str],
    *,
    purpose: str,
) -> Path:
    if _raw_path_is_forbidden(path):
        raise PathPolicyError("unsafe_path", f"{purpose} path is forbidden")
    candidate = Path(path)
    if candidate.suffix.lower() not in set(allowed_suffixes):
        raise PathPolicyError("unsafe_suffix", f"{purpose} path has a forbidden suffix")
    try:
        resolved_root = root.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise PathPolicyError("outside_allowed_root", f"{purpose} path is outside its allowed root") from exc
    return resolved


def safe_read_path(path: str | Path, root: Path, allowed_suffixes: Iterable[str] = SAFE_INPUT_SUFFIXES) -> Path:
    return _ensure_safe_path(path, root, allowed_suffixes, purpose="input")


def safe_write_path(path: str | Path, root: Path, allowed_suffixes: Iterable[str] = SAFE_OUTPUT_SUFFIXES) -> Path:
    return _ensure_safe_path(path, root, allowed_suffixes, purpose="output")


def safe_read_text(path: str | Path, root: Path) -> str:
    checked = safe_read_path(path, root)
    try:
        return checked.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualificationError("input_unreadable", "allowed input could not be read") from exc


def safe_read_bytes(path: str | Path, root: Path) -> bytes:
    checked = safe_read_path(path, root)
    try:
        return checked.read_bytes()
    except OSError as exc:
        raise QualificationError("input_unreadable", "allowed input could not be read") from exc


def safe_write_text(path: str | Path, root: Path, content: str, *, append: bool = False) -> None:
    checked = safe_write_path(path, root)
    try:
        checked.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "x"
        with checked.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise QualificationError("output_already_exists", "append-only output already exists") from exc
    except OSError as exc:
        raise QualificationError("output_unwritable", "allowed output could not be written") from exc


def append_jsonl(path: str | Path, root: Path, record: dict[str, Any]) -> None:
    safe_write_text(path, root, canonical_json(record) + "\n", append=True)


def strict_json_loads(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise QualificationError("invalid_json", "JSON input is empty")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QualificationError("duplicate_key", "JSON contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise QualificationError("non_finite_number", "JSON contains a non-finite number")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except QualificationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QualificationError("invalid_json", "JSON input is invalid") from exc


def load_json(path: str | Path, root: Path) -> Any:
    return strict_json_loads(safe_read_text(path, root))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, root: Path) -> str:
    return sha256_bytes(safe_read_bytes(path, root))


def relative_posix(path: Path, root: Path) -> str:
    checked = safe_read_path(path, root)
    return checked.relative_to(root.resolve(strict=False)).as_posix()


def assert_valid_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise QualificationError("invalid_run_id", "run ID is not in the P1v2-Q0 namespace")
    return run_id


def artifact_paths(data_root: Path, run_id: str) -> dict[str, Path]:
    assert_valid_run_id(run_id)
    root = data_root.resolve(strict=False)
    readiness = root / "readiness_dry"
    return {
        "root": readiness,
        "events": readiness / "events" / f"events_{run_id}.jsonl",
        "transcripts": readiness / "transcripts" / f"transcripts_{run_id}.jsonl",
        "outcomes": readiness / "outcomes" / f"outcomes_{run_id}.json",
        "manifest": readiness / "manifests" / f"manifest_{run_id}.json",
        "public_sink": readiness / "public_sink" / f"public_sink_{run_id}.jsonl",
        "validation": readiness / "validation" / f"validation_{run_id}.jsonl",
        "replay": readiness / "replay" / f"replay_{run_id}.jsonl",
        "report": readiness / "reports" / f"P1V2Q0_DELIVERY_{run_id}.md",
    }


def tree_hash(root: Path, relative_directory: str) -> tuple[str, dict[str, str]]:
    directory = safe_read_path(root / relative_directory / "placeholder.py", root).parent
    if not directory.exists() or not directory.is_dir():
        raise QualificationError("source_tree_missing", "tracked source tree is missing")
    entries: dict[str, str] = {}
    for candidate in sorted(directory.rglob("*.py")):
        relative = relative_posix(candidate, root)
        entries[relative] = sha256_file(candidate, root)
    if not entries:
        raise QualificationError("source_tree_empty", "tracked source tree is empty")
    return sha256_bytes(canonical_json(entries).encode("utf-8")), entries


def normalize_executable(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def assert_execution_roots(code_root: Path, data_root: Path, *, test_mode: bool = False) -> None:
    """Reject arbitrary code/data roots before any framework I/O occurs.

    Public Q0 entry points are fixed to the two task-book-approved roots.  Tests
    may use an isolated copy underneath this package's tests directory only.
    """
    resolved_code = code_root.resolve(strict=False)
    resolved_data = data_root.resolve(strict=False)
    if not test_mode:
        if resolved_code != PROJECT_ROOT.resolve(strict=False) or resolved_data != DEFAULT_DATA_ROOT.resolve(strict=False):
            raise PathPolicyError("execution_root_invalid", "Q0 public execution roots are fixed")
        return
    test_root = (PROJECT_ROOT / "tests").resolve(strict=False)
    try:
        code_relative = resolved_code.relative_to(test_root)
        data_relative = resolved_data.relative_to(test_root)
    except ValueError as exc:
        raise PathPolicyError("execution_root_invalid", "test execution roots are outside the isolated test root") from exc
    if not code_relative.parts or not data_relative.parts:
        raise PathPolicyError("execution_root_invalid", "test execution roots are invalid")
    if not code_relative.parts[0].startswith(".tmp_q0_") or data_relative.parts[0] != code_relative.parts[0]:
        raise PathPolicyError("execution_root_invalid", "test execution roots are not in one isolated copy")


def assert_workspace_root(workspace_root: Path) -> None:
    if workspace_root.resolve(strict=False) != WORKSPACE_ROOT.resolve(strict=False):
        raise PathPolicyError("workspace_root_invalid", "Q0 workspace root is fixed")
