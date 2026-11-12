"""Filesystem and canonicalization helpers for the derived offline audit."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_ROOT.parents[1] / "data" / "pre_exp1" / PACKAGE_ROOT.name
ORIGINAL_CODE_ROOT = PACKAGE_ROOT.parent / "p1v2_qualification_q2b"
ORIGINAL_DATA_ROOT = PACKAGE_ROOT.parents[1] / "data" / "pre_exp1" / "p1v2_qualification_q2b"
ORIGINAL_RUN_ID = "P1V2Q2B-REMOTE-20260802T033140463464Z"
HISTORICAL_PRE_REWORK_AUDIT_ID = "P1V2Q2B-REPAIR-AUDIT-20260802T035603081713Z"
AUDIT_ID_RE = re.compile(r"^P1V2Q2B-REPAIR-AUDIT-\d{8}T\d{12}Z$")


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


def make_audit_id() -> str:
    return "P1V2Q2B-REPAIR-AUDIT-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def validate_audit_id(audit_id: str) -> None:
    if not AUDIT_ID_RE.fullmatch(audit_id):
        raise ValueError("invalid_audit_id")


def tree_hashes(root: Path, namespace: str) -> dict[str, str]:
    """Read bytes only; the caller never writes into the inspected root."""
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[namespace + "/" + path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def original_tree_hashes() -> dict[str, str]:
    result = tree_hashes(ORIGINAL_CODE_ROOT, "original_code")
    result.update(tree_hashes(ORIGINAL_DATA_ROOT, "original_data"))
    return result


def source_file_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for folder in ("src", "scripts"):
        root = PACKAGE_ROOT / folder
        for path in sorted(root.rglob("*.py")):
            result[folder + "/" + path.relative_to(root).as_posix()] = sha256_file(path)
    return result
