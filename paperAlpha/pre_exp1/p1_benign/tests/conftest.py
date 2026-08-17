from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterator

import pytest


P1_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = P1_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from p1_benign.core import DEFAULT_CONTEXT, TEST_DOUBLE  # noqa: E402
from p1_benign.providers import DeterministicTestProvider  # noqa: E402
from p1_benign.runner import run_p1  # noqa: E402


@pytest.fixture
def test_context(tmp_path: Path):
    return DEFAULT_CONTEXT.with_test_output(tmp_path / "paperAlpha")


@pytest.fixture(scope="session")
def baseline_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[object, str, dict]]:
    root = tmp_path_factory.mktemp("p1_baseline") / "paperAlpha"
    context = DEFAULT_CONTEXT.with_test_output(root)
    run_id = "P1-BENIGN-DRY-PYTEST-BASE"
    result = run_p1(
        context=context,
        provider=DeterministicTestProvider(),
        execution_mode=TEST_DOUBLE,
        episode_count=2,
        run_id=run_id,
        eligible_for_scientific_analysis=False,
    )
    yield context, run_id, result


@pytest.fixture
def cloned_run(
    tmp_path: Path,
    baseline_run: tuple[object, str, dict],
):
    baseline_context, run_id, _ = baseline_run
    target_root = tmp_path / "paperAlpha"
    shutil.copytree(
        baseline_context.outputs.data_root.parent,
        target_root,
    )
    return DEFAULT_CONTEXT.with_test_output(target_root), run_id

