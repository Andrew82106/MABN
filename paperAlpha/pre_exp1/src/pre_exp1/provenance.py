"""Configuration, input, source, workspace, and Python environment provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .paths import ProjectPaths
from .state import stable_hash


EXPECTED_EXECUTABLE = Path(
    r"D:\anaconda\envs\multi_agent_graph\python.exe"
).resolve()
EXPECTED_PYTHON_VERSION = "3.11.15"
DISTRIBUTION_NAME = "paperalpha-pre-exp1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_hashes(
    paths: Iterable[Path],
    project_root: Path,
) -> dict[str, str]:
    return {
        path.resolve().relative_to(project_root.resolve()).as_posix(): sha256_file(
            path
        )
        for path in sorted((item.resolve() for item in paths), key=str)
    }


def source_file_hashes(paths: ProjectPaths) -> dict[str, str]:
    tracked = [
        *paths.pre_exp1_root.joinpath("src").rglob("*.py"),
        *paths.pre_exp1_root.joinpath("scripts").rglob("*.py"),
    ]
    return relative_hashes(tracked, paths.paper_alpha_root)


def source_tree_hash(paths: ProjectPaths) -> str:
    return stable_hash(source_file_hashes(paths))


def workspace_state(paths: ProjectPaths) -> dict[str, Any]:
    repository_root = paths.paper_alpha_root.parent
    git_dir = repository_root / ".git"
    if not git_dir.exists():
        return {
            "git_available": False,
            "git_status": "not_a_git_repository",
            "source_tree_hash": source_tree_hash(paths),
        }
    result = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "git_available": True,
        "git_status": result.stdout.splitlines(),
        "source_tree_hash": source_tree_hash(paths),
    }


def _distribution_inventory() -> list[str]:
    inventory = {
        f"{distribution.metadata['Name'].lower()}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(inventory)


def capture_python_environment(*, enforce: bool = False) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    version = platform.python_version()
    if enforce and executable != EXPECTED_EXECUTABLE:
        raise RuntimeError(
            f"P0 must run with {EXPECTED_EXECUTABLE}, got {executable}"
        )
    if enforce and version != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"P0 must run with Python {EXPECTED_PYTHON_VERSION}, got {version}"
        )
    try:
        pytest_version: str | None = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = None
    inventory = _distribution_inventory()
    installed_distribution_version = importlib.metadata.version(
        DISTRIBUTION_NAME
    )
    return {
        "sys_executable": str(executable),
        "python_version": version,
        "python_full_version": sys.version,
        "platform": platform.platform(),
        "pytest_version": pytest_version,
        "source_version": __version__,
        "installed_distribution_version": installed_distribution_version,
        "dependency_inventory": inventory,
        "dependency_inventory_hash": stable_hash(inventory),
        "dependency_count": len(inventory),
    }


def environment_matches(recorded: dict[str, Any]) -> bool:
    return recorded == capture_python_environment(enforce=False)


def version_evidence_errors(environment: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_version = environment.get("source_version")
    installed_version = environment.get("installed_distribution_version")
    inventory = environment.get("dependency_inventory", [])
    expected_inventory_entry = (
        f"{DISTRIBUTION_NAME}=={installed_version}".lower()
    )
    if source_version != installed_version:
        errors.append(
            "source version and installed distribution version differ: "
            f"{source_version!r} != {installed_version!r}"
        )
    if expected_inventory_entry not in inventory:
        errors.append(
            "dependency inventory does not contain the installed project "
            f"distribution {expected_inventory_entry!r}"
        )
    if environment.get("dependency_inventory_hash") != stable_hash(inventory):
        errors.append("dependency inventory hash is not reproducible")
    if environment.get("dependency_count") != len(inventory):
        errors.append("dependency inventory count is inconsistent")
    return errors
