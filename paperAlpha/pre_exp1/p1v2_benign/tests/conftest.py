from __future__ import annotations

import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = CODE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture
def permitted_data_root() -> Path:
    """A disposable child of the only P1v2 namespace writers may use."""

    from p1v2_benign.core import DEFAULT_DATA_ROOT

    root = DEFAULT_DATA_ROOT / "_test_scratch" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
