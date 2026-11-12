import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
from monitor_joint_events import load_model, monitor_lines
from paperalpha_monitor import DynamicJointBN, JointRiskStream


def model_file(tmp_path):
    model = DynamicJointBN()
    model.optimizer = {"success": True, "message": "unit fixture only"}
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"joint_bn": {"bn": model.parameters()}}), encoding="utf-8")
    return path


def test_json_parameter_loading_does_not_require_pickle(tmp_path):
    model = load_model(model_file(tmp_path))
    assert model.optimizer["success"]
    with pytest.raises(ValueError, match="selected method"):
        load_model(model_file(tmp_path), "histgb_flat_prefix")


def test_line_adapter_reports_rejection_and_continues_from_last_valid_state():
    stream = JointRiskStream(DynamicJointBN())
    first = {"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted"}
    second = {"sequence": 1, "actor": "a", "target": "b", "event_type": "message_sent"}
    rows = list(monitor_lines(stream, [json.dumps(first), "{invalid", json.dumps(second)], explain=True))
    assert [r["status"] for r in rows] == ["accepted", "rejected", "accepted"]
    assert rows[1]["risk_probability"] is None and rows[1]["alarm"] is None
    assert rows[-1]["accepted_event_count"] == 2
    assert "variables" in rows[-1]


def test_cli_produces_one_json_response_per_line_and_nonzero_on_rejection(tmp_path):
    command = [sys.executable, "-B", str(ROOT / 'code/commands/monitor_joint_events.py'),
               "--model-parameters", str(model_file(tmp_path)), "--threshold", "0", "--explain"]
    event = {"sequence": 0, "actor": "a", "event_type": "input_received"}
    valid = subprocess.run(command, input=json.dumps(event) + "\n", capture_output=True, text=True, timeout=20)
    assert valid.returncode == 0, valid.stderr
    result = json.loads(valid.stdout)
    assert result["status"] == "accepted" and result["alarm"] is True
    assert result["first_alarm_sequence"] == 0
    assert result["deployment_validation"].startswith("not established")
    invalid = subprocess.run(command, input=json.dumps(event) + '\n{"label":1}\n', capture_output=True, text=True, timeout=20)
    assert invalid.returncode == 1
    rows = [json.loads(line) for line in invalid.stdout.splitlines()]
    assert len(rows) == 2 and rows[-1]["status"] == "rejected"


def test_parameter_layout_mismatch_rejected(tmp_path):
    path = model_file(tmp_path)
    value = json.loads(path.read_text())
    value["joint_bn"]["bn"]["parameter_names"] = ["wrong"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="layout"):
        load_model(path)
