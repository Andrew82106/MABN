"""Safe artifact paths, hashing, and provenance for Q2-D."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contract import canonical_json, data_root, package_root


ARTIFACT_NAMES = ("manifest.json", "ledger.json", "validation.json", "replay.json", "run_report.md")
FORBIDDEN_PERSISTED_TOKENS = (
    "prompt", "raw_response", "original_answer", "canary-value", "authorization", "api_key",
    "reasoning", "server_error", "tool_parameters", "tool_call_arguments", "internal_classification",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_tree_hash(mapping: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(mapping).encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def source_provenance() -> dict[str, Any]:
    root = package_root()
    source_files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    asset_files = sorted(path for path in (root / "assets").rglob("*") if path.is_file()) if (root / "assets").exists() else []
    source_map = {_relative(path, root): sha256_file(path) for path in source_files}
    asset_map = {_relative(path, root): sha256_file(path) for path in asset_files}
    return {
        "contract_version": "Q2D-P1-ACTION-CONTRACT-1.0.0",
        "source_files": source_map,
        "source_tree_hash": stable_tree_hash(source_map),
        "asset_files": asset_map,
        "asset_tree_hash": stable_tree_hash(asset_map),
    }


def run_root(run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in run_id):
        raise ValueError("invalid run id")
    root = data_root() / "runs" / run_id
    require_data_path(root)
    return root


def require_data_path(path: Path) -> Path:
    resolved, root = path.resolve(), data_root().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("path escapes Q2-D data root")
    return resolved


def artifact_paths(run_id: str) -> dict[str, Path]:
    root = run_root(run_id)
    return {
        "manifest": root / "manifest.json",
        "ledger": root / "ledger.json",
        "validation": root / "validation.json",
        "replay": root / "replay.json",
        "run_report": root / "run_report.md",
    }


def _assert_safe_persisted(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).lower()
    for token in FORBIDDEN_PERSISTED_TOKENS:
        if token in text:
            raise ValueError(f"forbidden persisted token: {token}")


def safe_write_json(path: Path, value: Any) -> None:
    require_data_path(path)
    _assert_safe_persisted(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def safe_write_text(path: Path, text: str) -> None:
    require_data_path(path)
    lowered = text.lower()
    for token in FORBIDDEN_PERSISTED_TOKENS:
        if token in lowered:
            raise ValueError(f"forbidden persisted token: {token}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    require_data_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["ARTIFACT_NAMES", "artifact_paths", "data_root", "read_json", "run_root", "safe_write_json", "safe_write_text", "sha256_file", "source_provenance", "stable_tree_hash"]
