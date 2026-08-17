from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from p1v2_benign import runner as runner_module
from p1v2_benign.core import (
    PathBoundaryError,
    artifact_paths,
    canonical_json_bytes,
    load_config,
    read_jsonl,
    read_strict_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from p1v2_benign.inventory import capture_baseline, compare_baselines, dependency_inventory, readonly_tree_inventory
from p1v2_benign.replay import replay_run
from p1v2_benign.runner import create_dry_run
from p1v2_benign.validation import validate_run


CODE_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = CODE_ROOT.parents[1]
RUN_ID = "P1V2-READINESS-DRY-TEST-001Z"


@pytest.fixture
def artifact_root(permitted_data_root: Path) -> Path:
    result = create_dry_run(data_root=permitted_data_root, run_id=RUN_ID)
    assert result["artifact_kind"] == "p1v2_readiness_dry"
    return permitted_data_root


def _manifest(root: Path) -> dict:
    return read_strict_json(root / artifact_paths(RUN_ID)["manifest"], allowed_root=root)


def _write_manifest(root: Path, manifest: dict) -> None:
    write_json(root / artifact_paths(RUN_ID)["manifest"], manifest, allowed_root=root)


def _rehash(root: Path, manifest: dict, name: str) -> None:
    path = root / artifact_paths(RUN_ID)[name]
    manifest["artifact_hashes"][name] = sha256_file(path, allowed_root=root)
    _write_manifest(root, manifest)


def test_dry_manifest_has_literal_offline_identity_closure_and_replays(artifact_root: Path) -> None:
    manifest = _manifest(artifact_root)
    assert manifest["artifact_kind"] == "p1v2_readiness_dry"
    assert manifest["data_role"] == "p1v2_readiness_dry"
    assert manifest["is_new_p1_run"] is False
    assert manifest["eligible_for_p1_gate_analysis"] is False
    assert manifest["eligible_for_p2"] is False
    assert manifest["eligible_for_confirmatory_analysis"] is False
    assert manifest["eligible_for_causal_effect_analysis"] is False
    assert manifest["p1_go"] is False
    assert manifest["p2_allowed"] is False
    assert manifest["real_model_calls"] == 0
    assert set(manifest["artifact_hashes"]) == {"events", "outcomes", "validation", "replay", "report"}
    events = read_jsonl(artifact_root / artifact_paths(RUN_ID)["events"], allowed_root=artifact_root)
    outcomes = read_jsonl(artifact_root / artifact_paths(RUN_ID)["outcomes"], allowed_root=artifact_root)
    assert manifest["deterministic_fixture_resolutions"] == len(manifest["expected_cases"]) == len(events) == len(outcomes)
    assert [row["case_id"] for row in manifest["expected_cases"]] == [row["case_id"] for row in events] == [row["case_id"] for row in outcomes]
    assert validate_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is True
    assert replay_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is True


def test_independent_literal_provenance_oracle_covers_actual_inputs_scripts_and_sources(artifact_root: Path) -> None:
    manifest = _manifest(artifact_root)
    expected_inputs = {
        "config": "pre_exp1/p1v2_benign/configs/readiness.json",
        "fixture": "pre_exp1/p1v2_benign/fixtures/response_cases.json",
        "prompt": "pre_exp1/p1v2_benign/prompts/public_response_contract.txt",
        "schema": "pre_exp1/p1v2_benign/schemas/response_contract.schema.json",
    }
    for name, expected_path in expected_inputs.items():
        identity = manifest["execution_inputs"][name]
        assert identity["relative_path"] == expected_path
        assert len(identity["sha256"]) == 64
        assert identity["size_bytes"] > 0
    assert [entry["relative_path"] for entry in manifest["execution_inputs"]["entry_scripts"]] == [
        "pre_exp1/p1v2_benign/scripts/run_readiness_dry.py",
        "pre_exp1/p1v2_benign/scripts/validate_readiness_dry.py",
        "pre_exp1/p1v2_benign/scripts/replay_readiness_dry.py",
    ]
    assert [entry["relative_path"] for entry in manifest["execution_inputs"]["source_modules"]] == [
        "pre_exp1/p1v2_benign/src/p1v2_benign/__init__.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/cli.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/contract.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/core.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/inventory.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/replay.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/runner.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/validation.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/workflow.py",
    ]
    expected_provenance = set(expected_inputs.values()) | {
        "pre_exp1/p1v2_benign/scripts/capture_baseline.py",
        "pre_exp1/p1v2_benign/scripts/compare_baselines.py",
        "pre_exp1/p1v2_benign/scripts/replay_readiness_dry.py",
        "pre_exp1/p1v2_benign/scripts/run_readiness_dry.py",
        "pre_exp1/p1v2_benign/scripts/validate_readiness_dry.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/__init__.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/cli.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/contract.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/core.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/inventory.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/replay.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/runner.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/validation.py",
        "pre_exp1/p1v2_benign/src/p1v2_benign/workflow.py",
    }
    assert {entry["path"] for entry in manifest["provenance"]["files"]} == expected_provenance


@pytest.mark.parametrize("bad_value", [[], {}, True, None])
@pytest.mark.parametrize("target", ["manifest", "event", "outcome"])
def test_malformed_nested_artifacts_fail_closed_in_both_public_apis(
    artifact_root: Path,
    target: str,
    bad_value: object,
) -> None:
    manifest = _manifest(artifact_root)
    if target == "manifest":
        manifest["expected_cases"][0]["case_id"] = bad_value
        _write_manifest(artifact_root, manifest)
    elif target == "event":
        path = artifact_root / artifact_paths(RUN_ID)["events"]
        events = read_jsonl(path, allowed_root=artifact_root)
        events[0]["case_id"] = bad_value
        write_jsonl(path, events, allowed_root=artifact_root)
        _rehash(artifact_root, manifest, "events")
    else:
        path = artifact_root / artifact_paths(RUN_ID)["outcomes"]
        outcomes = read_jsonl(path, allowed_root=artifact_root)
        outcomes[0]["case_id"] = bad_value
        write_jsonl(path, outcomes, allowed_root=artifact_root)
        _rehash(artifact_root, manifest, "outcomes")
    validation = validate_run(data_root=artifact_root, run_id=RUN_ID)
    replay = replay_run(data_root=artifact_root, run_id=RUN_ID)
    assert validation["passed"] is False and validation["errors"]
    assert replay["passed"] is False and replay["errors"]


def test_resolution_count_and_duplicate_event_closure_are_not_set_only_checks(artifact_root: Path) -> None:
    manifest = _manifest(artifact_root)
    manifest["deterministic_fixture_resolutions"] += 1
    _write_manifest(artifact_root, manifest)
    assert validate_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is False
    assert replay_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is False


def test_recomputed_duplicate_event_still_fails_exact_closure(artifact_root: Path) -> None:
    events_path = artifact_root / artifact_paths(RUN_ID)["events"]
    events = read_jsonl(events_path, allowed_root=artifact_root)
    duplicate = copy.deepcopy(events[0])
    duplicate["sequence"] = len(events) + 1
    event_body = {key: value for key, value in duplicate.items() if key != "event_id"}
    duplicate["event_id"] = sha256_bytes(canonical_json_bytes(event_body))
    events.append(duplicate)
    write_jsonl(events_path, events, allowed_root=artifact_root)
    manifest = _manifest(artifact_root)
    manifest["deterministic_fixture_resolutions"] = len(events)
    _rehash(artifact_root, manifest, "events")
    assert replay_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is False


def test_semantic_outcome_tamper_fails_even_after_rehash(artifact_root: Path) -> None:
    paths = artifact_paths(RUN_ID)
    outcome_path = artifact_root / paths["outcomes"]
    outcomes = read_jsonl(outcome_path, allowed_root=artifact_root)
    outcomes[0]["contract_valid"] = False
    outcomes[0]["decision"] = "reject"
    outcomes[0]["public_sink_published"] = False
    outcomes[0]["public_report_sha256"] = None
    outcomes[0]["rejection_code"] = "declared_reject"
    write_jsonl(outcome_path, outcomes, allowed_root=artifact_root)
    manifest = _manifest(artifact_root)
    _rehash(artifact_root, manifest, "outcomes")
    result = validate_run(data_root=artifact_root, run_id=RUN_ID)
    assert result["passed"] is False
    assert {error["code"] for error in result["errors"]} >= {"fresh_replay_failed"}


def test_legal_but_wrong_provenance_hash_fails(artifact_root: Path) -> None:
    manifest = _manifest(artifact_root)
    files = manifest["provenance"]["files"]
    manifest["provenance"]["files"][0]["sha256"] = files[1]["sha256"]
    _write_manifest(artifact_root, manifest)
    result = validate_run(data_root=artifact_root, run_id=RUN_ID)
    assert result["passed"] is False
    assert {error["code"] for error in result["errors"]} >= {"provenance_mismatch"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/readiness.json",
        "fixtures/response_cases.json",
        "schemas/response_contract.schema.json",
        "prompts/public_response_contract.txt",
        "scripts/run_readiness_dry.py",
        "scripts/validate_readiness_dry.py",
        "scripts/replay_readiness_dry.py",
        "src/p1v2_benign/workflow.py",
    ],
)
def test_canonical_input_and_source_drift_breaks_execution_provenance(artifact_root: Path, relative_path: str) -> None:
    target = CODE_ROOT / relative_path
    original = target.read_bytes()
    target.write_bytes(original + b"\n")
    try:
        result = validate_run(data_root=artifact_root, run_id=RUN_ID)
        assert result["passed"] is False
        assert {error["code"] for error in result["errors"]} & {
            "execution_input_provenance_mismatch",
            "provenance_mismatch",
            "active_config_invalid",
        }
    finally:
        target.write_bytes(original)


def test_writer_exposes_no_alternate_config_or_fixture_path_and_rejects_external_config_before_read(
    permitted_data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "config_path" not in inspect.signature(create_dry_run).parameters
    assert "fixture_path" not in inspect.signature(create_dry_run).parameters
    assert "config_path" not in inspect.signature(runner_module.build_dry_records).parameters
    calls: list[str] = []

    def blocked(*args, **kwargs):
        calls.append("io")
        raise AssertionError("path must be rejected before I/O")

    monkeypatch.setattr(Path, "read_bytes", blocked)
    with pytest.raises(PathBoundaryError):
        load_config(permitted_data_root / "external_config.json")
    assert calls == []


def test_env_unc_legacy_and_nonallowed_roots_are_rejected_before_any_path_io(
    permitted_data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def blocked(*args, **kwargs):
        calls.append("io")
        raise AssertionError("rejected path must not be touched")

    monkeypatch.setattr(Path, "read_bytes", blocked)
    monkeypatch.setattr(Path, "write_bytes", blocked)
    monkeypatch.setattr(Path, "mkdir", blocked)
    monkeypatch.setattr(Path, "exists", blocked)
    monkeypatch.setattr(Path, "resolve", blocked)
    with pytest.raises(PathBoundaryError):
        load_config(permitted_data_root / ".env")
    old_root = PAPER_ROOT / "pre_exp1" / "p1_benign"
    with pytest.raises(PathBoundaryError):
        create_dry_run(data_root=old_root, run_id="P1V2-READINESS-DRY-BOUNDARY-001Z")
    for rejected_root in (r"\\server\share\p1v2", r"\\?\C:\device\p1v2", r"\\.\C:\device\p1v2"):
        assert validate_run(data_root=rejected_root, run_id=RUN_ID)["passed"] is False
        assert replay_run(data_root=rejected_root, run_id=RUN_ID)["passed"] is False
    nonallowed_root = r"D:\outside-p1v2-namespace"
    assert validate_run(data_root=nonallowed_root, run_id=RUN_ID)["passed"] is False
    assert replay_run(data_root=nonallowed_root, run_id=RUN_ID)["passed"] is False
    with pytest.raises(PathBoundaryError):
        capture_baseline(phase="before", data_root=old_root)
    assert calls == []


def test_read_only_validate_and_replay_do_not_resolve_fixture(monkeypatch: pytest.MonkeyPatch, artifact_root: Path) -> None:
    def forbidden_fixture_resolution(*args, **kwargs):
        raise AssertionError("fixture resolution must not occur")

    monkeypatch.setattr(runner_module, "load_fixture_cases", forbidden_fixture_resolution)
    assert validate_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is True
    assert replay_run(data_root=artifact_root, run_id=RUN_ID)["passed"] is True


def test_readonly_tree_and_dependency_snapshots_compare_without_changes(permitted_data_root: Path) -> None:
    assert readonly_tree_inventory()["roots"] == [
        "pre_exp1/p1_benign",
        "pre_exp1/p1_model_qualification",
        "pre_exp1/p1_scientific_benign",
        "data/shared/p1_benign",
        "data/pre_exp1/p1_benign",
        "data/pre_exp1/p1_model_qualification",
        "data/pre_exp1/p1_scientific_benign",
    ]
    assert dependency_inventory()["distributions"]
    capture_baseline(phase="before", label="pytest", data_root=permitted_data_root)
    capture_baseline(phase="after", label="pytest", data_root=permitted_data_root)
    assert compare_baselines(label="pytest", data_root=permitted_data_root)["passed"] is True
