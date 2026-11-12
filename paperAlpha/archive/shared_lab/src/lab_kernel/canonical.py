"""Canonical serialization and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path, files: Iterable[Path] | None = None) -> tuple[str, dict[str, str]]:
    root = root.resolve()
    selected = files if files is not None else (p for p in root.rglob("*") if p.is_file())
    hashes = {
        path.resolve().relative_to(root).as_posix(): file_hash(path.resolve())
        for path in sorted(selected, key=lambda item: item.as_posix())
    }
    return content_hash(hashes), hashes
