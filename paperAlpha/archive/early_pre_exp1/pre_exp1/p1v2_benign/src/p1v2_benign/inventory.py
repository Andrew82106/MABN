"""Read-only source-tree and dependency inventories required for delivery."""

from __future__ import annotations

import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .core import (
    DEFAULT_DATA_ROOT,
    OLD_READONLY_RELATIVE_ROOTS,
    PAPER_ALPHA_ROOT,
    canonical_json_bytes,
    read_strict_json,
    require_p1v2_data_root,
    resolve_generated_path,
    sha256_bytes,
    write_json,
    write_text,
)


READONLY_RELATIVE_ROOTS = OLD_READONLY_RELATIVE_ROOTS


def _trusted_readonly_hash(path: Path) -> str:
    """Hash only a fixed task-book read-only root, never a caller-supplied path."""

    return sha256_bytes(path.read_bytes())


def readonly_tree_inventory() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative_root in READONLY_RELATIVE_ROOTS:
        absolute_root = PAPER_ALPHA_ROOT / relative_root
        if not absolute_root.is_dir():
            raise ValueError("readonly_root_missing")
        for path in sorted(absolute_root.rglob("*")):
            if path.is_file():
                content = path.read_bytes()
                files.append(
                    {
                        "path": path.relative_to(PAPER_ALPHA_ROOT).as_posix(),
                        "sha256": sha256_bytes(content),
                        "size_bytes": len(content),
                    }
                )
    return {
        "artifact_kind": "p1v2_readonly_tree_inventory",
        "files": files,
        "roots": list(READONLY_RELATIVE_ROOTS),
        "tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def dependency_inventory() -> dict[str, Any]:
    distributions: list[dict[str, str]] = []
    for item in metadata.distributions():
        name = item.metadata.get("Name")
        if name:
            distributions.append({"name": name.lower(), "version": item.version})
    distributions.sort(key=lambda row: (row["name"], row["version"]))
    return {
        "artifact_kind": "p1v2_dependency_inventory",
        "distributions": distributions,
        "python_version": platform.python_version(),
        "sys_executable": sys.executable,
    }


def _baseline_label(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9_]{1,32}", value) is None:
        raise ValueError("invalid_baseline_label")
    return value


def capture_baseline(
    *,
    phase: str,
    label: str = "rework1",
    data_root: Path | str = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    if phase not in {"before", "after"}:
        raise ValueError("invalid_baseline_phase")
    safe_label = _baseline_label(label)
    root = require_p1v2_data_root(data_root)
    tree_path = resolve_generated_path(root, f"environment/readonly_tree_{safe_label}_{phase}.json")
    dependency_path = resolve_generated_path(root, f"environment/dependency_inventory_{safe_label}_{phase}.json")
    if tree_path.exists() or dependency_path.exists():
        raise FileExistsError("baseline_phase_already_exists")
    tree = readonly_tree_inventory()
    dependencies = dependency_inventory()
    write_json(tree_path, tree, allowed_root=root)
    write_json(dependency_path, dependencies, allowed_root=root)
    return {
        "dependency_inventory_path": str(dependency_path),
        "label": safe_label,
        "phase": phase,
        "readonly_tree_path": str(tree_path),
        "tree_sha256": tree["tree_sha256"],
    }


def compare_baselines(
    *,
    label: str = "rework1",
    data_root: Path | str = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    safe_label = _baseline_label(label)
    root = require_p1v2_data_root(data_root)
    before_tree = read_strict_json(
        resolve_generated_path(root, f"environment/readonly_tree_{safe_label}_before.json"),
        allowed_root=root,
    )
    after_tree = read_strict_json(
        resolve_generated_path(root, f"environment/readonly_tree_{safe_label}_after.json"),
        allowed_root=root,
    )
    before_deps = read_strict_json(
        resolve_generated_path(root, f"environment/dependency_inventory_{safe_label}_before.json"),
        allowed_root=root,
    )
    after_deps = read_strict_json(
        resolve_generated_path(root, f"environment/dependency_inventory_{safe_label}_after.json"),
        allowed_root=root,
    )
    tree_unchanged = before_tree == after_tree
    dependencies_unchanged = before_deps == after_deps
    comparison = {
        "artifact_kind": "p1v2_protection_and_dependency_comparison",
        "dependencies_unchanged": dependencies_unchanged,
        "label": safe_label,
        "passed": tree_unchanged and dependencies_unchanged,
        "readonly_tree_unchanged": tree_unchanged,
    }
    output = resolve_generated_path(root, f"environment/protection_and_dependency_comparison_{safe_label}.json")
    report_path = resolve_generated_path(root, f"readiness_dry/reports/protection_and_dependency_comparison_{safe_label}.md")
    if output.exists() or report_path.exists():
        raise FileExistsError("baseline_comparison_already_exists")
    write_json(output, comparison, allowed_root=root)
    write_text(
        report_path,
        "# Protection and dependency comparison\n\n"
        f"- Label: `{safe_label}`.\n"
        f"- Read-only tree unchanged: `{str(tree_unchanged).lower()}`.\n"
        f"- Dependency inventory unchanged: `{str(dependencies_unchanged).lower()}`.\n"
        f"- Overall comparison passed: `{str(comparison['passed']).lower()}`.\n",
        allowed_root=root,
    )
    return comparison
