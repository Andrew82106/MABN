from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

import pytest

from pre_exp1 import __version__
from pre_exp1.event_log import AppendOnlyEventLog
from pre_exp1.paths import (
    CONFIG_DIR,
    INTERIM_DIR,
    PAPER_ALPHA_ROOT,
    REPORTS_DIR,
    SHARED_DATA_DIR,
    ProjectPaths,
    ensure_data_dirs,
)
from pre_exp1.permissions import PermissionEnforcer
from pre_exp1.runtime_config import RuntimeConfig
from pre_exp1.sandbox import run_p0_smoke
from pre_exp1.state import build_initial_state


def pytest_configure(config) -> None:
    config._pre_exp1_started_at = time.perf_counter()


def pytest_sessionfinish(session, exitstatus) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats if reporter is not None else {}
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", [])) + len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    duration = time.perf_counter() - session.config._pre_exp1_started_at
    ensure_data_dirs()
    report = REPORTS_DIR / "test_results_rework_2026-07-31.md"
    report.write_text(
        "\n".join(
            [
                "# P0 最终验收 payload 补丁完整测试摘要",
                "",
                f"- sys.executable：`{Path(sys.executable).resolve()}`",
                f"- Python：`{platform.python_version()}`",
                f"- pytest：`{importlib.metadata.version('pytest')}`",
                f"- source version：`{__version__}`",
                "- installed distribution version："
                f"`{importlib.metadata.version('paperalpha-pre-exp1')}`",
                "- 命令：`conda run --no-capture-output "
                "-n multi_agent_graph python -B -m pytest "
                "-p no:cacheprovider`",
                f"- 收集：{session.testscollected}",
                f"- 通过：{passed}",
                f"- 失败：{failed}",
                f"- 跳过：{skipped}",
                f"- 用时：{duration:.2f} 秒",
                "",
                "本报告只验证 P0 工具，不代表实验性研究结论。",
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture
def event_log_factory() -> Iterator:
    ensure_data_dirs()
    temporary_directories: list[tempfile.TemporaryDirectory[str]] = []

    def factory(run_id: str = "TEST-RUN") -> AppendOnlyEventLog:
        temporary = tempfile.TemporaryDirectory(dir=INTERIM_DIR)
        temporary_directories.append(temporary)
        return AppendOnlyEventLog(Path(temporary.name) / "events.jsonl", run_id)

    yield factory
    for temporary in temporary_directories:
        temporary.cleanup()


@pytest.fixture
def fixture_record() -> dict:
    path = SHARED_DATA_DIR / "fixtures" / "vendor_records.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["records"][0]


@pytest.fixture
def episode_state(fixture_record: dict) -> dict:
    return build_initial_state("TEST-EPISODE", fixture_record)


@pytest.fixture
def permissions() -> PermissionEnforcer:
    return PermissionEnforcer.from_config(CONFIG_DIR / "agents.json")


@pytest.fixture
def runtime() -> RuntimeConfig:
    return RuntimeConfig.load()


@pytest.fixture(scope="session")
def baseline_tamper_project() -> Iterator[tuple[Path, str]]:
    """Build one valid isolated run below official interim data."""

    ensure_data_dirs()
    temporary = tempfile.TemporaryDirectory(dir=INTERIM_DIR)
    project_root = Path(temporary.name) / "paperAlpha"
    shutil.copytree(
        PAPER_ALPHA_ROOT / "pre_exp1" / "configs",
        project_root / "pre_exp1" / "configs",
    )
    shutil.copytree(
        PAPER_ALPHA_ROOT / "pre_exp1" / "src",
        project_root / "pre_exp1" / "src",
    )
    shutil.copytree(
        PAPER_ALPHA_ROOT / "pre_exp1" / "scripts",
        project_root / "pre_exp1" / "scripts",
    )
    shutil.copytree(
        PAPER_ALPHA_ROOT / "data" / "shared",
        project_root / "data" / "shared",
    )
    run_id = "P0-SMOKE-TAMPER-BASE"
    run_p0_smoke(run_id, project_root=project_root)
    yield project_root, run_id
    temporary.cleanup()


@pytest.fixture
def tamper_project(
    baseline_tamper_project: tuple[Path, str],
) -> Iterator[tuple[ProjectPaths, str]]:
    baseline_root, run_id = baseline_tamper_project
    temporary = tempfile.TemporaryDirectory(dir=INTERIM_DIR)
    project_root = Path(temporary.name) / "paperAlpha"
    shutil.copytree(baseline_root, project_root)
    yield ProjectPaths.from_root(project_root), run_id
    temporary.cleanup()
