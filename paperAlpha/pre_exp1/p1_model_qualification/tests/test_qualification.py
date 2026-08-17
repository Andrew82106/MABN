from __future__ import annotations

import copy
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from p1_benign.providers import DeterministicTestProvider
from p1_model_qualification import cli
from p1_model_qualification.core import (
    CANDIDATE_MODELS,
    DEFAULT_CONTEXT,
    EXECUTION_MODE,
    PHASE,
    TASK_ORDER,
    capture_provenance,
    load_static_inputs,
    read_json,
    read_jsonl,
    stable_hash,
    validate_frozen_config,
    write_json,
    write_jsonl,
)
from p1_model_qualification.providers import (
    QualificationOllamaProvider,
    coding_plan_readiness,
    inspect_local_models,
    validate_loopback_endpoint,
)
from p1_model_qualification.replay import replay_run
from p1_model_qualification.runner import run_qualification
from p1_model_qualification.validation import validate_run


def _context(tmp_path: Path):
    return DEFAULT_CONTEXT.with_test_output(tmp_path / "paperAlpha")


def _run(context, suffix: str = "BASE") -> tuple[str, dict]:
    run_id = f"P1-BENIGN-QUAL-DRY-PYTEST-{suffix}"
    result = run_qualification(
        context=context,
        provider=DeterministicTestProvider(),
        execution_mode="test_double",
        run_id=run_id,
    )
    return run_id, result


def _rehash_event(event: dict) -> None:
    event["payload_hash"] = stable_hash(event["payload"])
    identity = {
        "run_id": event["run_id"],
        "episode_id": event["episode_id"],
        "sequence": event["sequence"],
        "event_type": event["event_type"],
        "payload_hash": event["payload_hash"],
    }
    event["event_id"] = f"P1Q-EVT-{stable_hash(identity)[:24]}"


def _refresh_hashes(context, run_id: str, *names: str) -> None:
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    import hashlib

    for name in names:
        manifest["artifact_hashes"][name] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
    write_json(paths["manifest"], manifest, context.outputs.data_root)


def test_three_frozen_tasks_are_unique_and_outside_scientific_candidates() -> None:
    static = load_static_inputs()
    assert static.task_ids == TASK_ORDER
    assert len({row["vendor_id"] for row in static.materials.values()}) == 3
    assert {row["report"]["risk_level"] for row in static.expected_reports.values()} == {"low", "medium", "high"}
    prompt_text = "\n".join(static.workflow.prompts.values())
    for record in static.internal_records.values():
        assert record["internal_canary"] not in prompt_text
        assert record["internal_canary"] not in str(static.materials)
        assert record["internal_canary"] not in str(static.expected_reports)


def test_candidate_models_and_full_budget_are_exactly_frozen() -> None:
    static = load_static_inputs()
    assert static.qualification["candidate_models"] == list(CANDIDATE_MODELS)
    assert static.qualification["budget"]["maximum_calls_total"] == 54
    assert static.qualification["budget"]["concurrency"] == 1
    assert static.qualification["budget"]["retry_per_call"] == 0
    assert static.qualification["eligible_for_scientific_analysis"] is False


def test_deterministic_three_episode_run_recomputes_endpoints_and_replays(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, result = _run(context)
    assert result["validation"]["passed"] is True
    assert result["replay"]["passed"] is True
    outcomes = result["outcomes"]
    assert len(outcomes) == 3
    assert all(row["phase"] == PHASE for row in outcomes)
    assert all(row["task_success"] and row["required_fields_correct"] for row in outcomes)
    assert all(row["provider_call_count"] == 9 == row["structured_output_count"] for row in outcomes)
    assert validate_run(run_id, context=context, write_outputs=False)["passed"] is True
    assert replay_run(run_id, context=context, write_output=False)["passed"] is True


@pytest.mark.parametrize("model", ["qwen3:latest", "deepseek-r1:8b", ""])
def test_unapproved_model_tags_are_rejected(model: str) -> None:
    with pytest.raises(ValueError):
        QualificationOllamaProvider(model_id=model)


@pytest.mark.parametrize("endpoint", ["https://127.0.0.1:11434", "http://10.0.0.8:11434", "http://example.com:11434", "http://127.0.0.1:11434/api/chat"])
def test_non_loopback_or_non_origin_endpoints_are_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_loopback_endpoint(endpoint)


@pytest.mark.parametrize("episodes", [0, 1, 2, 4])
def test_qualification_requires_exactly_three_episodes(tmp_path: Path, episodes: int) -> None:
    with pytest.raises(ValueError):
        run_qualification(
            context=_context(tmp_path),
            provider=DeterministicTestProvider(),
            execution_mode="test_double",
            episode_count=episodes,
        )


@pytest.mark.parametrize("field,value", [("eligible_for_scientific_analysis", True), ("risk_seed_present", True), ("intervention_applied", True)])
def test_scientific_and_risk_labels_are_rejected_before_execution(field: str, value: bool) -> None:
    config = copy.deepcopy(load_static_inputs().qualification)
    config[field] = value
    with pytest.raises(ValueError):
        validate_frozen_config(config)


@pytest.mark.parametrize("field,value", [("concurrency", 2), ("retry_per_call", 1), ("maximum_calls_total", 55), ("episodes_per_model", 4)])
def test_expanded_budget_or_parallelism_is_rejected(field: str, value: int) -> None:
    config = copy.deepcopy(load_static_inputs().qualification)
    config["budget"][field] = value
    with pytest.raises(ValueError):
        validate_frozen_config(config)


def test_reversed_task_order_is_rejected_before_model_calls() -> None:
    config = copy.deepcopy(load_static_inputs().qualification)
    config["task_order"] = list(reversed(config["task_order"]))
    with pytest.raises(ValueError):
        validate_frozen_config(config)


def test_scientific_task_overlap_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    static_root = tmp_path / "static"
    shutil.copytree(DEFAULT_CONTEXT.static_root, static_root)
    materials_path = static_root / "vendor_materials.json"
    materials = read_json(materials_path)
    materials["records"][0]["vendor_id"] = "V-P1-001"
    materials_path.write_text(__import__("json").dumps(materials), encoding="utf-8")
    isolated = replace(context, static_root=static_root)
    with pytest.raises(ValueError):
        load_static_inputs(isolated)


def test_inventory_reports_missing_models_without_pull_or_mutation(monkeypatch) -> None:
    seen: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(command, **kwargs):
        seen.append(command)
        return Completed("NAME ID SIZE MODIFIED\n")

    monkeypatch.setattr("p1_model_qualification.providers.subprocess.run", fake_run)
    result = inspect_local_models()
    assert result["missing_models"] == list(CANDIDATE_MODELS)
    assert all(not ({"pull", "create", "copy", "rm"} & set(command)) for command in seen)
    assert seen == [["ollama", "--version"], ["ollama", "list"]]


def test_coding_plan_policy_gate_never_checks_credentials(monkeypatch) -> None:
    monkeypatch.setenv("ASTRON_API_KEY", "must-not-be-read")
    result = coding_plan_readiness()
    assert result["ready"] is False
    assert result["credential_checked"] is False
    assert result["network_request_attempted"] is False


def test_output_path_escape_is_a_validation_failure(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _ = _run(context)
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    manifest["output_files"]["events"] = "../p1_benign/raw/escaped.jsonl"
    write_json(paths["manifest"], manifest, context.outputs.data_root)
    result = validate_run(run_id, context=context, write_outputs=False)
    assert result["passed"] is False
    assert any("output paths" in error for error in result["errors"])


def test_malformed_nested_artifact_returns_structured_cli_failure(tmp_path: Path, capsys, monkeypatch) -> None:
    context = _context(tmp_path)
    run_id, _ = _run(context)
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    target = next(row for row in events if row["event_type"] == "model_output")
    target["payload"]["parsed_response"] = None
    _rehash_event(target)
    write_jsonl(paths["events"], events, context.outputs.data_root)
    _refresh_hashes(context, run_id, "events")
    monkeypatch.setattr(cli, "DEFAULT_CONTEXT", context)
    code = cli.main(["validate", "--run-id", run_id])
    output = capsys.readouterr().out
    assert code == 1
    assert "Traceback" not in output
    assert '"passed": false' in output


def test_coordinated_semantic_tamper_is_rejected_after_rehash_and_fresh_replay(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _ = _run(context)
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    target = next(
        row
        for row in events
        if row["event_type"] == "tool_call" and row["payload"]["tool_name"] == "external_sink.publish"
    )
    target["payload"]["arguments"]["vendor_id"] = "V-P1-902"
    _rehash_event(target)
    write_jsonl(paths["events"], events, context.outputs.data_root)
    _refresh_hashes(context, run_id, "events")
    replay_run(run_id, context=context, write_output=True)
    _refresh_hashes(context, run_id, "replay")
    result = validate_run(run_id, context=context, write_outputs=False)
    assert result["passed"] is False
    assert any("authoritative" in error or "replay" in error for error in result["errors"])


def test_provenance_drift_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _ = _run(context)
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    manifest["provenance"]["reused_phase_a_tree_hash"] = "0" * 64
    write_json(paths["manifest"], manifest, context.outputs.data_root)
    result = validate_run(run_id, context=context, write_outputs=False)
    assert result["passed"] is False
    assert any("provenance drift" in error for error in result["errors"])


def test_scalar_manifest_fails_closed_without_exception(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _ = _run(context)
    path = context.outputs.artifact_paths(run_id)["manifest"]
    write_json(path, 123, context.outputs.data_root)
    result = validate_run(run_id, context=context, write_outputs=False)
    assert result["passed"] is False
    assert result["checks"]["untrusted_artifact_structure_safe"] is False


def test_real_mode_requires_explicit_authorization_before_inventory(tmp_path: Path, monkeypatch) -> None:
    provider = QualificationOllamaProvider(model_id="qwen3:8b")
    monkeypatch.setattr("p1_model_qualification.runner.inspect_local_models", lambda: pytest.fail("inventory should not be called"))
    with pytest.raises(PermissionError):
        run_qualification(
            context=_context(tmp_path),
            provider=provider,
            execution_mode=EXECUTION_MODE,
            allow_local_qualification=False,
        )


def test_manifest_records_all_actual_reuse_and_execution_hashes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_id, _ = _run(context)
    manifest = read_json(context.outputs.artifact_paths(run_id)["manifest"])
    provenance = manifest["provenance"]
    assert provenance == capture_provenance(context)
    assert provenance["qualification_execution_files"]
    assert provenance["reused_phase_a_files"]
    assert provenance["qualification_static_files"]
