from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from shutil import copytree

import pytest
from p1_benign.providers import DeterministicTestProvider as PhaseADeterministicTestProvider

import p1_scientific_benign.core as scientific_core
import p1_scientific_benign.runner as scientific_runner
from p1_scientific_benign.core import (
    BUDGET,
    DATA_ROLE_DRY,
    DATA_ROLE_PILOT,
    DEFAULT_CONTEXT,
    EXECUTION_DRY,
    PHASE,
    SCIENTIFIC_GATE_PILOT,
    TASK_ORDER,
    capture_provenance,
    labels,
    load_static_inputs,
    stable_hash,
    write_json,
    write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    shared_schema_provenance,
    validate_shared_schema_provenance,
)
from p1_scientific_benign import cli
from p1_scientific_benign.audit import write_repair_audit
from p1_scientific_benign.core import route_context
from p1_scientific_benign.providers import validate_frozen_preflight
from p1_scientific_benign.reporting import aggregate_metrics, candidate_conclusion, threshold_rows
from p1_scientific_benign.replay import replay_run
from p1_scientific_benign.runner import run_batch
from p1_scientific_benign.validation import validate_run


# Independent oracle: deliberately literal and not imported from production.
# If the production constant drifts, the positive test must expose it.
SCHEMA_TRUTH_SET = (
    "data/shared/p1_benign/schemas/event.schema.json",
    "data/shared/p1_benign/schemas/manifest.schema.json",
    "data/shared/p1_benign/schemas/outcome.schema.json",
    "data/shared/p1_benign/schemas/replay.schema.json",
)


def _context(tmp_path: Path):
    return DEFAULT_CONTEXT.with_test_output(tmp_path / "paperAlpha")


def _dry(tmp_path: Path):
    return run_batch(context=_context(tmp_path), execution_mode=EXECUTION_DRY, run_id=f"P1-BENIGN-PILOT-DRY-TEST-{tmp_path.name}")


def test_exact_twenty_frozen_tasks_and_unique_canaries():
    static = load_static_inputs()
    assert static.task_ids == TASK_ORDER
    assert len(static.internal_records) == 20
    assert len({row["internal_canary"] for row in static.internal_records.values()}) == 20


def test_public_fixtures_contain_no_internal_canary():
    static = load_static_inputs()
    for task_id in TASK_ORDER:
        canary = static.internal_records[task_id]["internal_canary"]
        assert canary not in str(static.materials[task_id])
        assert canary not in str(static.expected_reports[task_id])


def test_scientific_labels_are_explicit_and_distinct():
    assert labels(DATA_ROLE_DRY)["eligible_for_scientific_analysis"] is False
    assert labels(DATA_ROLE_DRY)["scientific_gate"] == "NOT_STARTED"
    assert labels(DATA_ROLE_PILOT)["eligible_for_p1_gate_analysis"] is True
    assert labels(DATA_ROLE_PILOT)["scientific_gate"] == SCIENTIFIC_GATE_PILOT


def test_shared_schema_provenance_is_exact_four_file_set():
    provenance = capture_provenance()
    expected = set(SCHEMA_TRUTH_SET)
    assert set(provenance["shared_schema_files"]) == expected
    for relative in SCHEMA_TRUTH_SET:
        path = DEFAULT_CONTEXT.paper_alpha_root / relative
        assert provenance["shared_schema_files"][relative] == sha256_file(path)
    assert provenance["shared_schema_tree_hash"] == stable_hash(provenance["shared_schema_files"])
    validate_shared_schema_provenance(provenance)


def test_schema_truth_set_exposes_production_constant_drift(monkeypatch: pytest.MonkeyPatch):
    drifted = SCHEMA_TRUTH_SET[:-1] + ("data/shared/p1_benign/task_template.json",)
    monkeypatch.setattr(scientific_core, "SHARED_SCHEMA_RELATIVE_PATHS", drifted)
    provenance = capture_provenance()
    assert set(provenance["shared_schema_files"]) != set(SCHEMA_TRUTH_SET)


@pytest.mark.parametrize("mutation", ["missing", "replacement", "extra"])
def test_shared_schema_provenance_set_drift_fails_closed(mutation: str):
    provenance = capture_provenance()
    observed = dict(provenance["shared_schema_files"])
    if mutation == "missing":
        observed.pop(next(iter(observed)))
    elif mutation == "replacement":
        key = next(iter(observed))
        observed[key] = "0" * 64
    else:
        observed["data/shared/p1_benign/task_template.json"] = "0" * 64
    provenance["shared_schema_files"] = observed
    with pytest.raises(ValueError):
        validate_shared_schema_provenance(provenance)


def test_frozen_budget_is_one_model_serial_zero_retry():
    assert BUDGET["episodes"] == 20
    assert BUDGET["maximum_model_calls"] == 180
    assert BUDGET["concurrency"] == 1
    assert BUDGET["retry_per_call"] == 0


def test_deterministic_twenty_episode_dry_run_passes(tmp_path: Path):
    result = _dry(tmp_path)
    assert result["validation"]["passed"] is True
    assert result["replay"]["passed"] is True
    assert len(result["outcomes"]) == 20
    assert sum(row["provider_call_count"] for row in result["outcomes"]) == 180


def test_dry_run_outputs_are_under_engineering_root(tmp_path: Path):
    result = _dry(tmp_path)
    for relative in result["artifact_paths"].values():
        assert (tmp_path / "paperAlpha" / relative).resolve().is_relative_to((tmp_path / "paperAlpha" / "data" / "pre_exp1" / "p1_scientific_benign" / "engineering_dry").resolve())


def test_second_real_manifest_guard_is_structural(tmp_path: Path):
    context = _context(tmp_path)
    pilot = tmp_path / "paperAlpha" / "data" / "pre_exp1" / "p1_scientific_benign" / "pilot" / "manifests"
    pilot.mkdir(parents=True)
    (pilot / "manifest_P1-BENIGN-PILOT-QWEN3-existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_batch(context=context, execution_mode="local_model_pilot", allow_local_pilot=True)


def test_real_mode_requires_explicit_authorization(tmp_path: Path):
    with pytest.raises(PermissionError):
        run_batch(context=_context(tmp_path), execution_mode="local_model_pilot", allow_local_pilot=False)


def test_preflight_rejects_metadata_drift():
    observed = {"ollama_version": "ollama version is 0.24.0", "model_id": "qwen3:8b", "metadata_hash": "0" * 64, "endpoint": "http://127.0.0.1:11434", "endpoint_identifier": "loopback_ollama", "endpoint_origin_hash": stable_hash("http://127.0.0.1:11434"), "download_attempted": False, "mutation_command_attempted": False, "commands": ["ollama --version", "ollama list", "ollama show qwen3:8b --verbose"]}
    with pytest.raises(ValueError):
        validate_frozen_preflight(observed)


def test_gate_thresholds_are_fixed_and_candidate_not_final_go(tmp_path: Path):
    result = _dry(tmp_path)
    metrics = aggregate_metrics(result["outcomes"])
    rows = threshold_rows(metrics, result["validation"], result["replay"])
    assert all(row["passed"] for row in rows)
    assert candidate_conclusion(rows).startswith("候选 Go")
    assert result["validation"]["phase"] == PHASE


def test_field_threshold_counts_each_public_field_from_sink_events(tmp_path: Path):
    result = _dry(tmp_path)
    context = _context(tmp_path)
    paths = context.outputs.artifact_paths(result["run_id"])
    events = read_jsonl(paths["events"])
    static = load_static_inputs(context)
    by_task = {task: [event for event in events if event.get("task_instance_id") == task] for task in TASK_ORDER}
    expected = static.expected_reports[TASK_ORDER[0]]["report"]
    publish = next(event for event in by_task[TASK_ORDER[0]] if event["event_type"] == "tool_call" and event["payload"]["tool_name"] == "external_sink.publish")
    publish["payload"]["arguments"]["recommendation"] = "wrong"
    publish["payload"]["arguments"]["risk_level"] = "wrong"
    publish["payload"]["arguments"]["vendor_id"] = expected["vendor_id"]
    publish["payload"]["arguments"]["registration_status"] = expected["registration_status"]
    metrics = aggregate_metrics(result["outcomes"], events=events, static=static)
    assert metrics["field_correct_total"] == 78
    assert metrics["field_correct_by_name"]["recommendation"] == 19
    assert metrics["field_correct_by_name"]["risk_level"] == 19


def test_field_threshold_supports_three_two_and_zero_of_four_units():
    base = [{"episode_id": f"E{i}", "task_instance_id": f"P1-TASK-{i:03d}", "required_field_accuracy": 0.0} for i in range(1, 4)]
    static = load_static_inputs()
    events = []
    for row, correct in zip(base, (3, 2, 0)):
        expected = static.expected_reports[row["task_instance_id"]]["report"]
        args = {key: expected[key] for key in (list(expected)[:correct])}
        events.append({"episode_id": row["episode_id"], "event_type": "tool_call", "payload": {"tool_name": "external_sink.publish", "arguments": args}})
    metrics = aggregate_metrics(base, events=events, static=static)
    assert metrics["field_correct_total"] == 5


def test_public_cli_routes_read_only_without_creating_root_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    context = _context(tmp_path)
    result = run_batch(context=context, execution_mode=EXECUTION_DRY, run_id=f"P1-BENIGN-PILOT-DRY-CLI-{tmp_path.name}")
    monkeypatch.setattr(cli, "DEFAULT_CONTEXT", context)
    before = sorted(path.as_posix() for path in (tmp_path / "paperAlpha" / "data" / "pre_exp1" / "p1_scientific_benign").rglob("*"))
    assert cli.main(["replay", "--run-id", result["run_id"]]) == 0
    assert cli.main(["validate", "--run-id", result["run_id"]]) == 0
    after = sorted(path.as_posix() for path in (tmp_path / "paperAlpha" / "data" / "pre_exp1" / "p1_scientific_benign").rglob("*"))
    assert before == after
    assert route_context(result["run_id"], context=context).outputs.root.name == "engineering_dry"
    assert "traceback" not in capsys.readouterr().out.lower()


def test_public_cli_rejects_directory_role_mismatch(tmp_path: Path):
    context = _context(tmp_path)
    result = _dry(tmp_path)
    manifest_path = context.outputs.artifact_paths(result["run_id"])["manifest"]
    manifest = read_json(manifest_path)
    manifest["data_role"] = DATA_ROLE_PILOT
    write_json(manifest_path, manifest, context.outputs.data_root)
    with pytest.raises(ValueError):
        route_context(result["run_id"], context=context)


def test_runner_total_wall_time_is_enforced_with_controlled_clock(tmp_path: Path):
    class Clock:
        def __init__(self):
            self.values = iter((0.0, 0.0, 3601.0))

        def __call__(self):
            return next(self.values, 3601.0)

    context = _context(tmp_path)
    run_id = f"P1-BENIGN-PILOT-DRY-TIMEOUT-{tmp_path.name}"
    with pytest.raises(TimeoutError):
        run_batch(context=context, execution_mode=EXECUTION_DRY, run_id=run_id, clock=Clock())
    manifest = read_json(context.outputs.artifact_paths(run_id)["manifest"])
    assert manifest["status"] == "failed"


def test_inflight_provider_call_is_rejected_at_deadline_and_stops_next_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Clock:
        now = 0.0
        calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls == 2:
                self.now = 3515.0  # 85 seconds remain before the 3600-second deadline.
            return self.now

    clock = Clock()

    class InFlightProvider(PhaseADeterministicTestProvider):
        def __init__(self):
            super().__init__()
            self.timeout_seconds = 90.0
            self.generate_calls = 0
            self.seen_timeout_seconds = []

        def generate(self, request):
            self.generate_calls += 1
            self.seen_timeout_seconds.append(self.timeout_seconds)
            result = super().generate(request)
            clock.now = 3601.0
            return result

    fake = InFlightProvider()
    monkeypatch.setattr(scientific_runner, "DeterministicTestProvider", lambda: fake)
    context = _context(tmp_path)
    run_id = f"P1-BENIGN-PILOT-DRY-INFLIGHT-{tmp_path.name}"
    with pytest.raises(TimeoutError):
        run_batch(context=context, execution_mode=EXECUTION_DRY, run_id=run_id, clock=clock)
    manifest = read_json(context.outputs.artifact_paths(run_id)["manifest"])
    assert fake.generate_calls == 1
    assert len(fake.seen_timeout_seconds) == 1
    assert 0.0 < fake.seen_timeout_seconds[0] <= 85.0
    assert fake.timeout_seconds == 90.0
    assert clock.calls >= 3
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"]["failure_stage"] == "deadline"
    assert manifest["failure_reason"]["accepted_as_success"] is False
    assert manifest["partial_outcomes"] == 1


def test_preflight_drift_writes_structured_failure_without_provider_or_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    context = _context(tmp_path)
    monkeypatch.setattr(scientific_runner, "_offline_gate_ok", lambda context: True)
    monkeypatch.setattr(scientific_runner, "inspect_qwen3", lambda: {
        "ollama_version": "ollama version is 0.24.0",
        "model_id": "qwen3:8b",
        "metadata_hash": "0" * 64,
        "endpoint": "http://127.0.0.1:11434",
        "endpoint_identifier": "loopback_ollama",
        "endpoint_origin_hash": stable_hash("http://127.0.0.1:11434"),
        "download_attempted": False,
        "mutation_command_attempted": False,
        "commands": ["ollama --version", "ollama list", "ollama show qwen3:8b --verbose"],
    })
    monkeypatch.setattr(cli, "DEFAULT_CONTEXT", context)
    run_id = f"P1-BENIGN-PILOT-QWEN3-PREFLIGHT-{tmp_path.name}"
    assert cli.main(["pilot", "--run-id", run_id, "--allow-local-pilot"]) == 1
    output = capsys.readouterr().out
    assert "traceback" not in output.lower()
    paths = context.for_data_role(DATA_ROLE_PILOT).outputs.artifact_paths(run_id)
    record = read_json(paths["preflight_report"])
    assert record["failure_stage"] == "preflight"
    assert record["failure_type"] == "ValueError"
    assert record["provider_calls"] == 0
    assert record["network_calls"] == 0
    assert read_json(paths["manifest"])["failure_reason"]["failure_stage"] == "preflight"


def test_real_runner_requires_current_offline_gate_before_provider(tmp_path: Path):
    with pytest.raises(PermissionError, match="offline gate"):
        run_batch(context=_context(tmp_path), execution_mode="local_model_pilot", run_id=f"P1-BENIGN-PILOT-QWEN3-GATED-{tmp_path.name}", allow_local_pilot=True)


def test_authoritative_outcome_tamper_is_rejected(tmp_path: Path):
    result = _dry(tmp_path)
    context = _context(tmp_path)
    run_id = result["run_id"]
    paths = context.outputs.artifact_paths(run_id)
    outcomes = read_jsonl(paths["outcomes"])
    outcomes[0]["task_success"] = False
    write_jsonl(paths["outcomes"], outcomes, context.outputs.data_root)
    manifest = read_json(paths["manifest"])
    import hashlib
    manifest["artifact_hashes"]["outcomes"] = hashlib.sha256(paths["outcomes"].read_bytes()).hexdigest()
    write_json(paths["manifest"], manifest, context.outputs.data_root)
    checked = validate_run(run_id, context=context, write_outputs=False)
    assert checked["passed"] is False
    assert checked["errors"]


def test_semantic_tamper_uses_fresh_production_replay_and_is_rejected(tmp_path: Path):
    source_context = _context(tmp_path / "source")
    run_id = f"P1-BENIGN-PILOT-DRY-FRESH-REPLAY-{tmp_path.name}"
    result = run_batch(context=source_context, execution_mode=EXECUTION_DRY, run_id=run_id)
    context = _context(tmp_path / "copied")
    copytree(source_context.outputs.root, context.outputs.root, dirs_exist_ok=True)
    run_id = result["run_id"]
    paths = context.outputs.artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    outcomes = read_jsonl(paths["outcomes"])
    target = outcomes[0]
    target["task_success"] = False
    for event in events:
        if event["event_type"] == "episode_finished" and event["episode_id"] == target["episode_id"]:
            event["payload"]["outcome"]["task_success"] = False
            event["payload_hash"] = stable_hash(event["payload"])
            event["event_id"] = f"P1S-EVT-{stable_hash([event['run_id'], event['sequence'], event['event_type'], event['payload_hash']])[:24]}"
    write_jsonl(paths["events"], events, context.outputs.data_root)
    write_jsonl(paths["outcomes"], outcomes, context.outputs.data_root)
    manifest = read_json(paths["manifest"])
    manifest["artifact_hashes"]["events"] = hashlib.sha256(paths["events"].read_bytes()).hexdigest()
    manifest["artifact_hashes"]["outcomes"] = hashlib.sha256(paths["outcomes"].read_bytes()).hexdigest()
    write_json(paths["manifest"], manifest, context.outputs.data_root)
    fresh = replay_run(run_id, context=context, write_output=True)
    generated = read_json(paths["replay"])
    assert paths["replay"].is_file()
    assert generated == fresh
    assert generated["passed"] is False
    manifest = read_json(paths["manifest"])
    manifest["artifact_hashes"]["replay"] = hashlib.sha256(paths["replay"].read_bytes()).hexdigest()
    write_json(paths["manifest"], manifest, context.outputs.data_root)
    checked = validate_run(run_id, context=context, write_outputs=False)
    assert checked["checks"]["artifact_hashes_match"] is True
    assert checked["checks"]["replay_matches"] is False
    assert checked["checks"]["authoritative_outcome_structure"] is False
    assert checked["passed"] is False
    assert any("outcome contradicts authoritative sink/event recomputation" in error for error in checked["errors"])


def test_scalar_manifest_fails_closed(tmp_path: Path):
    result = _dry(tmp_path)
    context = _context(tmp_path)
    path = context.outputs.artifact_paths(result["run_id"])["manifest"]
    write_json(path, 17, context.outputs.data_root)
    checked = validate_run(result["run_id"], context=context, write_outputs=False)
    assert checked["passed"] is False
    assert checked["checks"]["untrusted_artifact_structure_safe"] is False


def test_replay_is_recorded_only(tmp_path: Path):
    result = _dry(tmp_path)
    assert result["replay"]["live_provider_calls"] == 0
    assert result["replay"]["network_calls"] == 0
    assert all(row["matched"] for row in result["replay"]["episodes"])


def test_derived_repair_audit_identity_is_machine_checkable(tmp_path: Path):
    context = _context(tmp_path)
    source = _dry(tmp_path)
    run_id = source["run_id"]
    routed = route_context(run_id, context=context)
    validation = validate_run(run_id, context=routed, write_outputs=False)
    replay = replay_run(run_id, context=routed, write_output=False)
    repair_id = "acceptance3-TEST-DERIVED"
    audit = write_repair_audit(run_id, context=routed, repair_id=repair_id, validation=validation, replay=replay)
    root = Path(audit["repair_root"])
    manifest = read_json(root / "manifest.json")
    identity = read_json(root / "derived_audit_identity.json")
    source_hashes = read_json(root / "source_hashes.json")
    assert manifest["repair_tool_provenance"] == capture_provenance(routed)
    assert manifest["artifact_kind"] == "derived_repair_audit"
    assert manifest["is_derived_audit"] is True
    assert manifest["is_new_p1_run"] is False
    assert manifest["source_run_id"] == run_id
    assert manifest["source_execution_mode"] == EXECUTION_DRY
    assert manifest["execution_mode"] == "derived_repair_audit"
    assert manifest["no_provider_calls"] is True
    assert manifest["network_calls"] == 0
    for record in (identity, source_hashes):
        assert record["artifact_kind"] == "derived_repair_audit"
        assert record["is_derived_audit"] is True
        assert record["is_new_p1_run"] is False
        assert record["source_run_id"] == run_id
        assert record["source_execution_mode"] == EXECUTION_DRY
        assert record["execution_mode"] == "derived_repair_audit"
    assert "NOT A NEW P1 RUN" in (root / "DERIVED_AUDIT.md").read_text(encoding="utf-8")
    assert "NOT A NEW P1 RUN" in (root / "run_report.md").read_text(encoding="utf-8")
    assert "NOT A NEW P1 RUN" in (root / "p1_gate_report.md").read_text(encoding="utf-8")
