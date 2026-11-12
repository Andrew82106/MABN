"""The new child protocol must preserve scores and expose missing observations."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from monitor_delegation_pilot import make_compiler
from paperalpha_monitor.authority_qualification import QualifiedDelegationStream
from paperalpha_monitor.delegation_conditioned_bn import DelegationConditionedBN
from paperalpha_runtime.delegation_pilot import PilotEnvironment, configuration


def observed(variant):
    config, saved = configuration("qualified-cli", variant), defaultdict(list)
    env = PilotEnvironment(config, observe=lambda _: None, save=lambda name, row: saved[name].append(row))
    resource = config["requested_resource"]
    grant = env.invoke("planner", "issue_delegation", {"parent_grant_id": "root", "resource": resource}, "grant")
    env.after_planner("Delegation recorded, verify the current authority.")
    env.invoke("worker", "simulate_payment", {"resource": resource, "amount": "50",
        "grant_ids": [grant["record"]["payload"]["grant_id"]]}, "payment")
    return config, saved["monitor_events"]


def invoke(tmp_path, config, records, fitted=True):
    model = DelegationConditionedBN()
    if fitted:
        model.fitted_targets = ["episode_outcome"]  # Protocol fixture, not accuracy evidence.
    for name, value in (("config", config), ("model", model.parameters())):
        (tmp_path / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
    process = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / 'code/commands/monitor_qualified_delegation.py'),
        "--config", str(tmp_path / "config.json"), "--model", str(tmp_path / "model.json")],
        input="".join(json.dumps(r) + "\n" for r in records), text=True, encoding="utf-8",
        capture_output=True, timeout=30)
    return process, [json.loads(line) for line in process.stdout.splitlines()], model


@pytest.mark.parametrize("variant", ["normal", "revoked", "narrow_scope", "missing_root"])
def test_child_is_exactly_the_qualified_stream(tmp_path, variant):
    config, records = observed(variant)
    process, lines, model = invoke(tmp_path, config, records)
    assert process.returncode == 0 and process.stderr == ""
    assert lines[0]["ready"] and not lines[0]["automatic_blocking"] and not lines[0]["calibrated"]
    stream = QualifiedDelegationStream(model, make_compiler(config))
    assert [r["result"] for r in lines[1:]] == [stream.push(r) for r in records]
    assert len(lines) == len(records) + 1


def test_missing_revoke_keeps_probability_but_denies_safety_interpretation(tmp_path):
    config, records = observed("revoked")
    config["coverage_complete"] = False
    visible = [r for r in records if r.get("kind") != "revoke"]
    process, lines, model = invoke(tmp_path, config, visible)
    result = lines[-1]["result"]["dispatches"][0]
    assert process.returncode == 0 and result["reference_status"] == "supported_reference"
    assert result["evidence_insufficient"] and result["current_authority_assessment"] == "unresolved"
    assert result["qualification_reasons"] == ["authority_updates_may_be_missing"]
    original, engine = make_compiler(config), model.engine()
    for record in visible:
        step = original.push(record)
        if step is not None:
            engine.apply(step, original.compact.variables)
    assert result["model_point_probability"] == engine.probability and not result["proves_safety"]


def test_bad_event_ends_protocol_without_scoring_following_events(tmp_path):
    config, records = observed("normal")
    process, lines, _ = invoke(tmp_path, config, [records[0], {"sequence": 0}, *records[1:]])
    assert process.returncode == 1
    assert lines[-1] == {"ok": False, "error_type": "ValueError"} and len(lines) == 3


def test_untrained_outcome_head_cannot_start_qualified_reporting(tmp_path):
    config, records = observed("normal")
    process, lines, _ = invoke(tmp_path, config, records, fitted=False)
    assert process.returncode != 0 and not lines
    assert "separately supervised endpoint head" in process.stderr
