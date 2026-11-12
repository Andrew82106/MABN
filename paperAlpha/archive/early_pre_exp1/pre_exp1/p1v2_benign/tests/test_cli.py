from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from p1v2_benign import cli
from p1v2_benign.core import artifact_paths, read_strict_json, write_json
from p1v2_benign.runner import create_dry_run


RUN_ID = "P1V2-READINESS-DRY-CLI-001Z"
CODE_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = CODE_ROOT.parents[1]


def _snapshot(root: Path) -> list[tuple[str, str]]:
    return sorted(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*")
        if path.is_file()
    )


def _malform_manifest(root: Path, value: object) -> None:
    path = root / artifact_paths(RUN_ID)["manifest"]
    manifest = read_strict_json(path, allowed_root=root)
    manifest["expected_cases"][0]["case_id"] = value
    write_json(path, manifest, allowed_root=root)


def test_public_validation_and_replay_are_read_only(permitted_data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    create_dry_run(data_root=permitted_data_root, run_id=RUN_ID)
    before = _snapshot(permitted_data_root)
    assert cli.main(["validate", "--run-id", RUN_ID, "--data-root", str(permitted_data_root)]) == 0
    assert cli.main(["replay", "--run-id", RUN_ID, "--data-root", str(permitted_data_root)]) == 0
    capsys.readouterr()
    assert _snapshot(permitted_data_root) == before


@pytest.mark.parametrize("bad_value", [[], {}, True, None])
def test_malformed_artifact_cli_is_nonzero_structured_and_hides_traceback(
    permitted_data_root: Path,
    bad_value: object,
) -> None:
    create_dry_run(data_root=permitted_data_root, run_id=RUN_ID)
    _malform_manifest(permitted_data_root, bad_value)
    command = [
        sys.executable,
        "-B",
        str(CODE_ROOT / "scripts" / "validate_readiness_dry.py"),
        "--run-id",
        RUN_ID,
        "--data-root",
        str(permitted_data_root),
    ]
    completed = subprocess.run(command, cwd=PAPER_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["passed"] is False
    assert payload["errors"]


def test_cli_rejects_unc_without_traceback() -> None:
    command = [
        sys.executable,
        "-B",
        str(CODE_ROOT / "scripts" / "replay_readiness_dry.py"),
        "--run-id",
        RUN_ID,
        "--data-root",
        r"\\server\share\p1v2",
    ]
    completed = subprocess.run(command, cwd=PAPER_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["passed"] is False


def test_source_does_not_import_prior_p1_or_network_clients() -> None:
    forbidden = (
        "import p1_benign",
        "from p1_benign",
        "import p1_scientific_benign",
        "from p1_scientific_benign",
        "import p1_model_qualification",
        "from p1_model_qualification",
        "import socket",
        "import urllib",
        "import requests",
        "import subprocess",
        "dotenv",
        "ollama",
    )
    for path in (CODE_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert all(fragment not in text for fragment in forbidden), path
