import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
from monitor_joint_events import load_model
from demonstrate_action_scoped_bn import fixture
from paperalpha_monitor import ActionScopedJointBN, ActionRiskStream


def test_json_cli_uses_strict_action_protocol_and_preserves_rejection_recovery(tmp_path):
    events, context = fixture()
    parameters, configured = tmp_path / "parameters.json", tmp_path / "context.json"
    parameters.write_text(json.dumps(ActionScopedJointBN().parameters()), encoding="utf-8")
    configured.write_text(json.dumps(context), encoding="utf-8")
    assert isinstance(load_model(parameters), ActionScopedJointBN)
    command = [sys.executable, "-B", str(ROOT / 'code/commands/monitor_joint_events.py'), "--model-parameters", str(parameters),
               "--context", str(configured), "--explain", "--threshold", "0"]
    payload = "\n".join([json.dumps(events[0]), "not json", *map(json.dumps, events[1:])])
    result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=20)
    assert result.returncode == 1, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[1]["status"] == "rejected" and rows[1]["risk_probability"] is None and rows[1]["alarm"] is None
    assert rows[-1]["model_name"] == "ActionScopedJointBN-v3"
    assert rows[-1]["first_alarm_sequence"] == 0 and rows[-1]["alarm"]
    assert rows[-1]["accepted_event_count"] == len(events)
    assert rows[-1]["training_optimizer_success"] is None
    artifact = json.loads(parameters.read_text())
    artifact["action_registry"]["specification_sha256"] = "wrong"
    parameters.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="specification"):
        load_model(parameters)


def test_stream_is_frozen_at_construction_and_no_event_is_not_safe():
    model = ActionScopedJointBN()
    events, context = fixture()
    stream = ActionRiskStream(model, **context)
    with pytest.raises(ValueError, match="no event"):
        stream.snapshot()
    model.theta[:] = 8
    context["trusted_checkers"].clear()
    stream.push(events[2])
    stream.push(events[3])
    assert stream.explain()["actions"][0]["permission_status"] == "mismatch"
    assert stream.snapshot()["alarm"] is None
    with pytest.raises(ValueError, match="threshold"):
        ActionRiskStream(model, threshold=float("nan"))


def test_demonstration_freezes_sources_and_marks_parameters_untrained(tmp_path):
    output = tmp_path / "demonstration"
    command = [sys.executable, "-B", str(ROOT / 'code/commands/demonstrate_action_scoped_bn.py'), "--out", str(output)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "REPORT.json").read_text())
    assert not report["parameters_trained"] and report["status"] == "passed"
    assert report["max_historical_marginal_error"] < 1e-13
    assert (output / "source_snapshot/knowledge/templates/action_scoped_bn_v3.json").exists()
    again = subprocess.run(command, text=True, capture_output=True, timeout=20)
    assert again.returncode != 0 and "Refusing to overwrite" in again.stderr
