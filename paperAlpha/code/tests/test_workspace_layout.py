"""Layout contracts; no API calls, scoring, or historical artifact writes."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paperalpha_runtime.workspace import project_root, resolve_historical_path


ROOT = Path(__file__).resolve().parents[2]


def test_checkout_root_and_canonical_roots():
    assert project_root() == ROOT
    manifest = json.loads((ROOT / "code/workspace_paths.json").read_text(encoding="utf-8"))
    expected = {"docs", "knowledge", "code", "data", "experiments", "archive"}
    assert expected.issubset({p.name for p in ROOT.iterdir() if p.is_dir()})
    assert not any((ROOT / name).exists() for name in (
        "src", "scripts", "tests", "research", "configs", "results", "submission", "pre_exp1", "literature"))
    assert len({entry["source"] for entry in manifest["moves"]}) == len(manifest["moves"])


@pytest.mark.parametrize("old,new", [
    ("src/paperalpha_monitor/__init__.py", "code/src/paperalpha_monitor/__init__.py"),
    ("research/risk_catalog.json", "knowledge/risks/risk_catalog.json"),
    ("submission/protocol.json", "experiments/protocol.json"),
    ("results/independent_mas_journal_v1/REPORT.md",
     "experiments/synthetic_mas/runs/independent_mas_journal_v1/REPORT.md"),
])
def test_historical_paths_resolve_without_aliases(old, new):
    assert resolve_historical_path(old) == ROOT / new
    assert resolve_historical_path("paperAlpha/" + old) == ROOT / new
    assert resolve_historical_path(ROOT / old) == ROOT / new
    assert not (ROOT / old).exists()


@pytest.mark.parametrize("value", ["../elsewhere", "src/../../elsewhere", "C:/outside/file"])
def test_historical_lookup_rejects_outside_paths(value):
    with pytest.raises(ValueError):
        resolve_historical_path(value)


def test_current_path_and_unknown_old_path():
    assert resolve_historical_path("knowledge/risks/risk_catalog.json") == ROOT / "knowledge/risks/risk_catalog.json"
    with pytest.raises(FileNotFoundError):
        resolve_historical_path("unknown_old_folder/file")


def test_experiment_index_preserves_original_identity():
    index = json.loads((ROOT / "experiments/index.json").read_text(encoding="utf-8"))
    seen = set()
    for family in index["families"]:
        for run in family["runs"]:
            assert run["path"].startswith("experiments/" + family["name"] + "/runs/")
            assert run["path"] not in seen
            seen.add(run["path"])
            assert run["previous_path"].startswith("results/")
    assert index["primary_development_run"] in seen


@pytest.mark.parametrize("command", ["independent", "binding", "costs"])
def test_current_cli_help_from_another_working_directory(tmp_path, command):
    script = ROOT / "code/src/paperalpha_runtime/cli.py"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(script), command, "--help"],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
