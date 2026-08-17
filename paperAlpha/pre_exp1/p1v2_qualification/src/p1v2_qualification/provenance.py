"""Deterministic provenance and protected-tree auditing for Q0 only."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .errors import QualificationError
from .paths import (
    SAFE_INPUT_SUFFIXES,
    canonical_json,
    load_json,
    relative_posix,
    safe_read_bytes,
    safe_read_path,
    sha256_bytes,
    sha256_file,
    tree_hash,
)


REFERENCE_SOURCE_SPECS: tuple[tuple[str, str], ...] = (
    (
        "paperAlpha/pre_exp1/p1v2_benign/schemas/response_contract.schema.json",
        "0d8ff62da193ededb93016fe33dd3c48dda39854b7e766285736e2b68aa43fec",
    ),
    (
        "paperAlpha/pre_exp1/p1v2_benign/prompts/public_response_contract.txt",
        "90920fd5368a9723b04986ced1c5df7cfe508fdd3e113e2894d3810d3287e64c",
    ),
    (
        "paperAlpha/pre_exp1/p1v2_benign/configs/readiness.json",
        "cbdd66f51cf9e9a56ddf1ae4b6c301187a32d109567509381fc85c25501a0965",
    ),
    (
        "paperAlpha/data/pre_exp1/p1v2_benign/readiness_dry/manifests/manifest_P1V2-READINESS-DRY-20260801T000002000000Z.json",
        "f239ba3f5dbf85f250a3d5868922914c8ca1f902b45e8ae6386ef61314a4f726",
    ),
)

STATIC_INPUTS = (
    "configs/q1_protocol.json",
    "schemas/response_contract.schema.json",
    "prompts/public_response_contract.txt",
    "fixtures/screen_tasks.json",
    "fixtures/confirmation_tasks.json",
    "fixtures/negative_cases.json",
    "permissions/public_sink_rule.json",
    "provenance/reference_contract_sources.json",
)
ENTRYPOINTS = (
    "scripts/run_readiness_dry.py",
    "scripts/validate_readiness_dry.py",
    "scripts/replay_readiness_dry.py",
    "scripts/build_fixtures.py",
)
PROTECTED_ROOTS = (
    "paperAlpha/pre_exp1/p1_benign",
    "paperAlpha/pre_exp1/p1_model_qualification",
    "paperAlpha/pre_exp1/p1v2_benign",
    "paperAlpha/data/pre_exp1/p1_benign",
    "paperAlpha/data/pre_exp1/p1_model_qualification",
    "paperAlpha/data/pre_exp1/p1v2_benign",
    "paperAlpha/data/pre_exp1/p1_scientific_benign",
)


def collect_static_input_hashes(code_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(code_root / relative, code_root)
        for relative in STATIC_INPUTS
    }


def collect_entrypoint_hashes(code_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(code_root / relative, code_root)
        for relative in ENTRYPOINTS
    }


def collect_code_provenance(code_root: Path) -> dict[str, Any]:
    source_tree_hash, source_files = tree_hash(code_root, "src/p1v2_qualification")
    return {
        "source_tree_hash": source_tree_hash,
        "source_files": source_files,
        "entrypoint_hashes": collect_entrypoint_hashes(code_root),
    }


def _expected_reference_metadata() -> list[dict[str, str]]:
    return [
        {"relative_path": relative_path, "sha256": digest}
        for relative_path, digest in REFERENCE_SOURCE_SPECS
    ]


def load_reference_identity(code_root: Path) -> list[dict[str, str]]:
    record = load_json(code_root / "provenance" / "reference_contract_sources.json", code_root)
    if not isinstance(record, dict) or record.get("reference_set") != "P1v2-A contract sources":
        raise QualificationError("reference_identity_invalid", "reference source identity is invalid")
    sources = record.get("sources")
    if sources != _expected_reference_metadata():
        raise QualificationError("reference_identity_invalid", "reference source identity does not match the allowed source set")
    return _expected_reference_metadata()


def verify_reference_sources(code_root: Path, workspace_root: Path) -> list[dict[str, str]]:
    """Verify only the four task-book-authorized P1v2-A source identities."""
    expected = load_reference_identity(code_root)
    verified: list[dict[str, str]] = []
    for relative_path, expected_hash in REFERENCE_SOURCE_SPECS:
        source_path = workspace_root / Path(relative_path)
        try:
            # The source path is constructed from a literal allowlist, never a
            # manifest-controlled path.  It does not include credentials.
            actual_hash = sha256_bytes(source_path.read_bytes())
        except OSError as exc:
            raise QualificationError("reference_source_unreadable", "authorized reference source could not be read") from exc
        if actual_hash != expected_hash:
            raise QualificationError("reference_source_drift", "authorized reference source hash has drifted")
        verified.append({"relative_path": relative_path, "sha256": actual_hash})
    if verified != expected:
        raise QualificationError("reference_source_drift", "authorized reference source identity has drifted")
    return verified


def _hash_protected_directory(root: Path, workspace_root: Path) -> str:
    resolved_root = root.resolve(strict=False)
    allowed_workspace = workspace_root.resolve(strict=False)
    try:
        resolved_root.relative_to(allowed_workspace)
    except ValueError as exc:
        raise QualificationError("protected_tree_invalid", "protected tree is outside workspace") from exc
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise QualificationError("protected_tree_missing", "a protected tree is missing")
    files: dict[str, str] = {}
    for directory, _, names in os.walk(resolved_root, followlinks=False):
        for name in sorted(names):
            if name.lower() == ".env":
                raise QualificationError("protected_tree_forbidden", "protected-tree audit encountered a forbidden credential path")
            candidate = Path(directory) / name
            if candidate.is_symlink():
                raise QualificationError("protected_tree_symlink", "protected-tree audit encountered a symlink")
            try:
                candidate.relative_to(resolved_root)
                data = candidate.read_bytes()
            except (OSError, ValueError) as exc:
                raise QualificationError("protected_tree_unreadable", "protected-tree audit could not read a file") from exc
            files[candidate.relative_to(resolved_root).as_posix()] = sha256_bytes(data)
    return sha256_bytes(canonical_json(files).encode("utf-8"))


def capture_protected_tree_hashes(workspace_root: Path) -> dict[str, str]:
    return {
        relative_root: _hash_protected_directory(workspace_root / Path(relative_root), workspace_root)
        for relative_root in PROTECTED_ROOTS
    }


def runtime_identity() -> dict[str, str]:
    return {"python_executable": sys.executable, "python_version": sys.version.split()[0]}


def verify_current_code_provenance(manifest: dict[str, Any], code_root: Path) -> list[str]:
    errors: list[str] = []
    actual_inputs = collect_static_input_hashes(code_root)
    if manifest.get("static_input_hashes") != actual_inputs:
        errors.append("static input provenance does not match")
    actual_code = collect_code_provenance(code_root)
    if manifest.get("code_provenance") != actual_code:
        errors.append("code provenance does not match")
    actual_runtime = runtime_identity()
    if manifest.get("runtime") != actual_runtime:
        errors.append("runtime identity does not match")
    return errors
