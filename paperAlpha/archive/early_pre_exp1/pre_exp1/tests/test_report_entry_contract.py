"""Archived report-hook checks, moved out of the active MAS suite."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _legacy_report_hook():
    path = Path(__file__).with_name('conftest.py')
    spec = importlib.util.spec_from_file_location("legacy_report_hook_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_report_is_disabled_by_default(tmp_path, monkeypatch):
    hook = _legacy_report_hook()
    monkeypatch.setattr(hook, "REPORTS_DIR", tmp_path)
    def no_write():
        raise AssertionError("disabled report must not create output directories")
    monkeypatch.setattr(hook, "ensure_data_dirs", no_write)
    hook.pytest_sessionfinish(SimpleNamespace(config=SimpleNamespace(
        getoption=lambda *args, **kwargs: False)), 0)
    assert not list(tmp_path.iterdir())


def test_explicit_legacy_report_refuses_mixed_test_statistics(tmp_path, monkeypatch):
    hook = _legacy_report_hook()
    monkeypatch.setattr(hook, "REPORTS_DIR", tmp_path)
    session = SimpleNamespace(config=SimpleNamespace(getoption=lambda *args, **kwargs: True),
                              items=[SimpleNamespace(path=Path(__file__).parents[2] / "outside_test.py")])
    with pytest.warns(pytest.PytestWarning, match="non-P0"):
        hook.pytest_sessionfinish(session, 0)
    assert not list(tmp_path.iterdir())


def test_explicit_legacy_report_still_supports_legacy_only_runs(tmp_path, monkeypatch):
    hook = _legacy_report_hook()
    monkeypatch.setattr(hook, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(hook, "ensure_data_dirs", lambda: None)
    reporter = SimpleNamespace(stats={"passed": [object()]})
    config = SimpleNamespace(getoption=lambda *args, **kwargs: True,
                             pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter))
    hook.pytest_configure(config)
    session = SimpleNamespace(config=config, testscollected=1,
        items=[SimpleNamespace(path=Path(__file__).with_name('test_example.py'))])
    hook.pytest_sessionfinish(session, 0)
    report = tmp_path / "test_results_rework_2026-07-31.md"
    assert report.exists() and "通过：1" in report.read_text(encoding="utf-8")
