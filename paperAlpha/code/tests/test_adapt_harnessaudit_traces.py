from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/adapt_harnessaudit_traces.py'
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "harnessaudit_runtime_minimal.jsonl"


def _module():
    spec = importlib.util.spec_from_file_location("adapt_harnessaudit_traces", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_trace_projection_is_read_only_and_conservative(tmp_path: Path) -> None:
    module = _module()
    before = FIXTURE.read_bytes()
    summary = module.adapt(FIXTURE, tmp_path / "out")
    assert FIXTURE.read_bytes() == before
    assert summary["files"] == 1
    assert summary["events"] == 5
    assert summary["risk_label_events"] == 0

    rows = [json.loads(line) for line in (tmp_path / "out" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["schema_version"] == "mas-adapter-v1" for row in rows)
    tool = next(row for row in rows if row["event_type"] == "tool_call")
    assert tool["agent_id"] == "analyst-1"
    assert tool["role"] == "analyst"
    assert tool["tool"] == "read_record"
    assert tool["tool_arguments"]["api_key"] == "[REDACTED]"
    assert tool["external_effect"] is False

    communication = next(row for row in rows if row["event_type"] == "communication")
    assert communication["message_source"] == "analyst-1"
    assert communication["message_target"] == "coordinator-1"
    assert communication["features"]["cross_agent_propagation"] == 1

    decision = next(row for row in rows if row["event_type"] == "access_decision")
    assert decision["risk_label"] is None
    assert decision["outcome_flags"]["violated"] is True
    assert decision["parent"] == [tool["event_id"]]

