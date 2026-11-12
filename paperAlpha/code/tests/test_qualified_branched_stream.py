import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/tests')]
from test_branched_authority_pilot import setup, execute, models
from paperalpha_runtime.branched_authority_pilot import monitor_configuration
from paperalpha_monitor.branched_authority_monitor import BranchedAuthorityMonitor
from paperalpha_monitor.qualified_branched_stream import QualifiedBranchedStream


@pytest.mark.parametrize("completed", [True, False])
def test_live_prefix_never_opens_endpoint_heads_and_completion_is_explicit(completed):
    env, _, saved = setup(treatment="revoke_a_stale")
    execute(env)
    stream = QualifiedBranchedStream(BranchedAuthorityMonitor(monitor_configuration(env.config), *models()))
    for record in saved["monitor_events"]:
        result = stream.push(record)
        for value in result["views"].values():
            for row in value["dispatches"]:
                assert row["endpoint_heads"]["scores"] is None
                if row["reference_two_failures"]["target"]["action_count"] == 1:
                    assert row["reference_two_failures"]["model_probability"] == 0
    final = stream.finish(completed=completed)
    assert all(v["endpoint_heads"]["applicable"] is completed for v in final["views"].values())
    with pytest.raises(ValueError, match="completed"):
        stream.push(saved["monitor_events"][-1])
    with pytest.raises(ValueError, match="single"):
        stream.finish(completed=True)


def test_no_action_completion_does_not_invent_safe_scores():
    env, monitor, _ = setup()
    result = QualifiedBranchedStream(monitor).finish(completed=True)
    assert all(v["endpoint_heads"] is None for v in result["views"].values())
    assert not result["proves_system_safety"]


def test_new_scoped_entry_point_receives_real_event_protocol(tmp_path):
    env, _, saved = setup()
    execute(env)
    (tmp_path / "config.json").write_text(json.dumps(monitor_configuration(env.config)), encoding="utf-8")
    for name, data in zip(("reference_fit.json", "system_fit.json", "logistic.json"), models()):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    records = [*saved["monitor_events"], {"control": "finish", "completed": True}]
    run = subprocess.run([sys.executable, "-B", str(ROOT / 'code/commands/monitor_scoped_branched_authority.py'),
        "--config", str(tmp_path / "config.json"), "--models", str(tmp_path)],
        input="".join(json.dumps(r) + "\n" for r in records), capture_output=True, text=True, timeout=40)
    assert run.returncode == 0 and not run.stderr
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert rows[0]["protocol"] == "scoped-branched-authority-v1"
    assert rows[-1]["result"]["views"]["complete"]["endpoint_heads"]["applicable"]
    assert all(d["endpoint_heads"]["scores"] is None for r in rows[1:-1]
        for v in r["result"]["views"].values() for d in v["dispatches"])
