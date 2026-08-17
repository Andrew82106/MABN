"""Secret-free P1 provenance and environment evidence."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

import pre_exp1

from . import __version__
from .core import ProjectContext, file_hashes, stable_hash


DISTRIBUTION_NAME = "paperalpha-pre-exp1"
EXPECTED_EXECUTABLE = Path(
    r"D:\anaconda\envs\multi_agent_graph\python.exe"
).resolve()
EXPECTED_PYTHON = "3.11.15"


def source_paths(context: ProjectContext) -> tuple[Path, ...]:
    return tuple(
        sorted(
            [
                *context.source_root.rglob("*.py"),
                *(context.p1_root / "scripts").rglob("*.py"),
            ],
            key=lambda path: path.as_posix(),
        )
    )


def config_paths(context: ProjectContext) -> tuple[Path, ...]:
    return tuple(sorted(context.config_root.rglob("*.json"), key=str))


def prompt_paths(context: ProjectContext) -> tuple[Path, ...]:
    return tuple(sorted(context.prompt_root.rglob("*.txt"), key=str))


def fixture_paths(context: ProjectContext) -> tuple[Path, ...]:
    return tuple(sorted(context.static_root.rglob("*.*"), key=str))


def p0_reused_source_paths(context: ProjectContext) -> tuple[Path, ...]:
    root = context.paper_alpha_root / "pre_exp1" / "src" / "pre_exp1"
    return (
        root / "__init__.py",
        root / "schema_validation.py",
        root / "state.py",
    )


def capture_environment(*, enforce: bool = False) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    python_version = platform.python_version()
    if enforce and executable != EXPECTED_EXECUTABLE:
        raise RuntimeError(
            f"P1 must run with {EXPECTED_EXECUTABLE}, got {executable}"
        )
    if enforce and python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            f"P1 must run with Python {EXPECTED_PYTHON}, "
            f"got {python_version}"
        )
    inventory = sorted(
        {
            (
                f"{distribution.metadata['Name'].lower()}"
                f"=={distribution.version}"
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    return {
        "sys_executable": str(executable),
        "python_version": python_version,
        "python_full_version": sys.version,
        "platform": platform.platform(),
        "pytest_version": importlib.metadata.version("pytest"),
        "p1_source_version": __version__,
        "p0_source_version": pre_exp1.__version__,
        "p0_installed_distribution": DISTRIBUTION_NAME,
        "p0_installed_distribution_version": (
            importlib.metadata.version(DISTRIBUTION_NAME)
        ),
        "dependency_inventory": inventory,
        "dependency_inventory_hash": stable_hash(inventory),
        "dependency_count": len(inventory),
    }


def capture_provenance(context: ProjectContext) -> dict[str, Any]:
    p1_source_hashes = file_hashes(
        source_paths(context),
        context.paper_alpha_root,
    )
    configs = file_hashes(
        config_paths(context),
        context.paper_alpha_root,
    )
    prompts = file_hashes(
        prompt_paths(context),
        context.paper_alpha_root,
    )
    fixtures = file_hashes(
        fixture_paths(context),
        context.paper_alpha_root,
    )
    p0_hashes = file_hashes(
        p0_reused_source_paths(context),
        context.paper_alpha_root,
    )
    environment = capture_environment(enforce=True)
    return {
        "git_available": False,
        "git_status": "not_used_for_provenance",
        "p1_source_hashes": p1_source_hashes,
        "p1_source_tree_hash": stable_hash(p1_source_hashes),
        "config_hashes": configs,
        "config_tree_hash": stable_hash(configs),
        "prompt_hashes": prompts,
        "prompt_tree_hash": stable_hash(prompts),
        "fixture_hashes": fixtures,
        "fixture_tree_hash": stable_hash(fixtures),
        "provider_config_hash": configs[
            "pre_exp1/p1_benign/configs/providers.json"
        ],
        "p0_reused_source_hashes": p0_hashes,
        "p0_reused_source_tree_hash": stable_hash(p0_hashes),
        "python_environment": environment,
    }


def current_provenance_hash_groups(
    context: ProjectContext,
) -> dict[str, Any]:
    """Recompute stable, non-secret groups for validation."""

    source = file_hashes(source_paths(context), context.paper_alpha_root)
    configs = file_hashes(config_paths(context), context.paper_alpha_root)
    prompts = file_hashes(prompt_paths(context), context.paper_alpha_root)
    fixtures = file_hashes(fixture_paths(context), context.paper_alpha_root)
    p0_hashes = file_hashes(
        p0_reused_source_paths(context),
        context.paper_alpha_root,
    )
    return {
        "p1_source_hashes": source,
        "p1_source_tree_hash": stable_hash(source),
        "config_hashes": configs,
        "config_tree_hash": stable_hash(configs),
        "prompt_hashes": prompts,
        "prompt_tree_hash": stable_hash(prompts),
        "fixture_hashes": fixtures,
        "fixture_tree_hash": stable_hash(fixtures),
        "provider_config_hash": configs[
            "pre_exp1/p1_benign/configs/providers.json"
        ],
        "p0_reused_source_hashes": p0_hashes,
        "p0_reused_source_tree_hash": stable_hash(p0_hashes),
    }
