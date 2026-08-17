"""Small deterministic helpers.  This module never reads process environment data."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_ROOT.parents[1] / "data" / "pre_exp1" / PACKAGE_ROOT.name
RUN_ID_RE = re.compile(r"^P1V2Q2A-READINESS-DRY-\d{8}T\d{12}Z$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_run_id() -> str:
    return "P1V2Q2A-READINESS-DRY-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid_run_id")


def relative_file_hashes(root: Path, suffixes: tuple[str, ...] = (".py",)) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def safe_error_text(exc: BaseException) -> str:
    """Return only a fixed error category; never surface arbitrary transport text."""
    category = getattr(exc, "category", None)
    if isinstance(category, str) and category:
        return category
    return "offline_framework_error"
