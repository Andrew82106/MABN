from __future__ import annotations

import json

import pytest

from pre_exp1 import cli
from pre_exp1.event_payload import event_payload_hash
from pre_exp1.validation import validate_run


def _outcome_path(paths, run_id):
    return paths.processed_dir / f"outcomes_{run_id}.jsonl"


def _event_path(paths, run_id):
    return paths.raw_dir / f"events_{run_id}.jsonl"


def _manifest_path(paths, run_id):
    return paths.manifests_dir / f"manifest_{run_id}.json"


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _validate(paths, run_id):
    return validate_run(
        run_id,
        update_manifest=False,
        write_outputs=False,
        paths=paths,
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _event_of_type(events, event_type):
    return next(
        event for event in events if event["event_type"] == event_type
    )


def _rewrite_event_payload_hash(event):
    event["payload_hash"] = event_payload_hash(
        event["details"]["canonical_payload"]
    )


def _assert_payload_validation_failure(
    paths,
    run_id,
    monkeypatch,
    capsys,
):
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["event_payload_hashes_match"] is False
    assert any(
        "canonical payload contradicts" in error
        for error in summary["errors"]
    )
    monkeypatch.setattr(
        cli,
        "validate_run",
        lambda candidate_run_id: _validate(paths, candidate_run_id),
    )
    assert cli.main(["validate-run", "--run-id", run_id]) == 1
    assert '"passed": false' in capsys.readouterr().out


def test_isolated_baseline_copy_validates(tamper_project):
    paths, run_id = tamper_project
    summary = _validate(paths, run_id)
    assert summary["passed"] is True, summary["errors"]


def test_deleted_outcome_is_detected(tamper_project):
    paths, run_id = tamper_project
    outcomes = _read_jsonl(_outcome_path(paths, run_id))
    _write_jsonl(_outcome_path(paths, run_id), outcomes[:-1])
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert any("episode sets differ" in error for error in summary["errors"])


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("run_id", "WRONG-RUN", "outcome line 1 run_id"),
        ("treatment", "drop", "inconsistent treatment"),
        ("final_state_hash", "0" * 64, "inconsistent final_state_hash"),
    ],
)
def test_outcome_cross_file_tampering_is_detected(
    tamper_project,
    field,
    value,
    expected_error,
):
    paths, run_id = tamper_project
    outcomes = _read_jsonl(_outcome_path(paths, run_id))
    outcomes[0][field] = value
    _write_jsonl(_outcome_path(paths, run_id), outcomes)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert any(expected_error in error for error in summary["errors"])


def test_missing_manifest_declared_output_is_detected(tamper_project):
    paths, run_id = tamper_project
    (paths.reports_dir / f"smoke_{run_id}.md").unlink()
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert any(
        "manifest-declared output is missing" in error
        for error in summary["errors"]
    )


def test_config_drift_is_detected(tamper_project):
    paths, run_id = tamper_project
    config_path = paths.config_dir / "agents.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["tamper_marker"] = True
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert "current config hashes do not match manifest" in summary["errors"]


def test_shared_fixture_drift_is_detected(tamper_project):
    paths, run_id = tamper_project
    fixture_path = paths.shared_data_dir / "fixtures" / "vendor_records.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload["records"][0]["name"] = "Tampered Fictional Vendor"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert "current shared input hashes do not match manifest" in summary["errors"]


def test_duplicate_episode_outcome_is_detected(tamper_project):
    paths, run_id = tamper_project
    outcomes = _read_jsonl(_outcome_path(paths, run_id))
    outcomes.append(dict(outcomes[0]))
    _write_jsonl(_outcome_path(paths, run_id), outcomes)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert any("has 2 outcomes" in error for error in summary["errors"])


def test_missing_episode_completed_is_detected(tamper_project):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    target_episode = events[0]["episode_id"]
    events = [
        event
        for event in events
        if not (
            event["episode_id"] == target_episode
            and event["event_type"] == "episode_completed"
        )
    ]
    _write_jsonl(_event_path(paths, run_id), events)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert any(
        "0 episode_completed" in error for error in summary["errors"]
    )


def test_schema_valid_leak_outcome_contradiction_is_detected(tamper_project):
    paths, run_id = tamper_project
    outcomes = _read_jsonl(_outcome_path(paths, run_id))
    outcomes[0]["leak_detected"] = not outcomes[0]["leak_detected"]
    _write_jsonl(_outcome_path(paths, run_id), outcomes)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["leak_semantics_consistent"] is False
    assert any(
        "leak_detected contradicts" in error for error in summary["errors"]
    )


def test_schema_valid_task_outcome_contradiction_is_detected(tamper_project):
    paths, run_id = tamper_project
    outcomes = _read_jsonl(_outcome_path(paths, run_id))
    outcomes[0]["task_success"] = not outcomes[0]["task_success"]
    _write_jsonl(_outcome_path(paths, run_id), outcomes)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["task_semantics_consistent"] is False
    assert any(
        "task outcome contradicts" in error for error in summary["errors"]
    )


def test_schema_valid_intermediate_treatment_contradiction_is_detected(
    tamper_project,
):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    target = next(
        event for event in events if event["event_type"] == "message_gate"
    )
    target["treatment"] = (
        "safe" if target["treatment"] != "safe" else "drop"
    )
    _write_jsonl(_event_path(paths, run_id), events)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["episode_treatments_consistent"] is False
    assert any(
        "differs from primary treatment" in error
        for error in summary["errors"]
    )


def test_schema_valid_event_payload_hash_tampering_is_detected(
    tamper_project,
):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    events[0]["payload_hash"] = "f" * 64
    _write_jsonl(_event_path(paths, run_id), events)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["event_payload_hashes_match"] is False
    assert any(
        "payload_hash does not match canonical payload" in error
        for error in summary["errors"]
    )


def test_schema_valid_manifest_smoke_boolean_tampering_is_detected(
    tamper_project,
):
    paths, run_id = tamper_project
    manifest_path = _manifest_path(paths, run_id)
    manifest = _read_json(manifest_path)
    current = manifest["validation"]["smoke_expected_behavior"]
    manifest["validation"]["smoke_expected_behavior"] = not current
    _write_json(manifest_path, manifest)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["manifest_smoke_behavior_matches"] is False
    assert any(
        "manifest smoke_expected_behavior does not match" in error
        for error in summary["errors"]
    )


def test_schema_valid_installed_distribution_version_tampering_is_detected(
    tamper_project,
):
    paths, run_id = tamper_project
    manifest_path = _manifest_path(paths, run_id)
    manifest = _read_json(manifest_path)
    manifest["python_environment"]["installed_distribution_version"] = "0.1.0"
    _write_json(manifest_path, manifest)
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["version_evidence_consistent"] is False
    assert any(
        "manifest version evidence" in error for error in summary["errors"]
    )


def test_execution_script_source_drift_is_detected(tamper_project):
    paths, run_id = tamper_project
    script_path = paths.pre_exp1_root / "scripts" / "run_p0_smoke.py"
    script_path.write_text(
        script_path.read_text(encoding="utf-8")
        + "\n# isolated source-drift test\n",
        encoding="utf-8",
    )
    summary = _validate(paths, run_id)
    assert summary["passed"] is False
    assert summary["checks"]["source_file_hashes_match"] is False
    assert "current source file hashes do not match manifest" in summary["errors"]


def test_read_vendor_id_coordinated_payload_tamper_is_detected(
    tamper_project,
    monkeypatch,
    capsys,
):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    event = _event_of_type(events, "tool_read_internal_record")
    event["details"]["canonical_payload"]["request"]["vendor_id"] = (
        "VENDOR-TAMPERED"
    )
    _rewrite_event_payload_hash(event)
    _write_jsonl(_event_path(paths, run_id), events)
    _assert_payload_validation_failure(
        paths,
        run_id,
        monkeypatch,
        capsys,
    )


def test_read_idempotency_key_coordinated_payload_tamper_is_detected(
    tamper_project,
    monkeypatch,
    capsys,
):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    event = _event_of_type(events, "tool_read_internal_record")
    event["details"]["canonical_payload"]["request"]["idempotency_key"] = (
        "TAMPERED:read:vendor"
    )
    _rewrite_event_payload_hash(event)
    _write_jsonl(_event_path(paths, run_id), events)
    _assert_payload_validation_failure(
        paths,
        run_id,
        monkeypatch,
        capsys,
    )


def test_publish_idempotency_key_coordinated_payload_tamper_is_detected(
    tamper_project,
    monkeypatch,
    capsys,
):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    event = _event_of_type(events, "tool_publish_local_sink")
    event["details"]["canonical_payload"]["idempotency_key"] = (
        "TAMPERED:publish:report"
    )
    _rewrite_event_payload_hash(event)
    _write_jsonl(_event_path(paths, run_id), events)
    _assert_payload_validation_failure(
        paths,
        run_id,
        monkeypatch,
        capsys,
    )


@pytest.mark.parametrize(
    "event_type",
    ["tool_read_internal_record", "tool_publish_local_sink"],
)
def test_tool_request_fingerprint_mismatch_is_detected(
    tamper_project,
    monkeypatch,
    capsys,
    event_type,
):
    paths, run_id = tamper_project
    events = _read_jsonl(_event_path(paths, run_id))
    event = _event_of_type(events, event_type)
    event["details"]["replay_action"]["request_fingerprint"] = "f" * 64
    _write_jsonl(_event_path(paths, run_id), events)
    _assert_payload_validation_failure(
        paths,
        run_id,
        monkeypatch,
        capsys,
    )


def test_cli_returns_nonzero_when_validation_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "validate_run",
        lambda run_id: {
            "run_id": run_id,
            "passed": False,
            "checks": {"tamper_detected": False},
            "errors": ["isolated tamper"],
        },
    )
    assert cli.main(["validate-run", "--run-id", "TAMPERED-RUN"]) == 1
    assert '"passed": false' in capsys.readouterr().out
