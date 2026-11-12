"""Save a label-free lifecycle mechanism demonstration, not a benchmark.

No real tools or API calls are executed. Parameters are explicitly untrained;
the small trace checks temporal and binding semantics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor import ActionScopedJointBN, ActionRiskStream
from paperalpha_monitor.action_evidence import compile_action_trace
import evaluate_prefix_forecast_v1 as protocol


def fixture():
    scope = {"action_id": "inspect-workspace", "actor": "executor", "principal": "user-1", "tool": "shell.exec",
             "resource": "workspace-1", "arguments": {"command": "inspect"}}
    def action(sequence, event_type="tool_proposed", **updates):
        return {**scope, "sequence": sequence, "event_id": f"event-{sequence}", "event_type": event_type, **updates}
    def check(sequence, revision, status, until):
        return action(sequence, "authorization_checked", checker_id="policy-service", check_revision=revision,
                      permission_status=status, valid_until_sequence=until)
    events = [
        {"sequence": 0, "actor": "reader", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"},
        {"sequence": 1, "actor": "reader", "target": "executor", "event_type": "message_sent", "data_class": "untrusted"},
        action(2), check(3, 0, "mismatch", 10), check(4, 1, "no_mismatch", 4),
        {"sequence": 5, "actor": "executor", "event_type": "heartbeat"},
        check(6, 2, "no_mismatch", 7), action(7, "tool_call"),
        {"sequence": 8, "actor": "executor", "event_type": "input_received", "source_id": "later-doc", "data_class": "untrusted"},
        action(9, action_id="second-action", arguments={"command": "inspect-other"}),
        action(10, "action_cancelled", action_id="second-action", arguments={"command": "inspect-other"}),
    ]
    context = {"trusted_checkers": ["policy-service"], "roles": {"reader": "reader", "executor": "executor"}}
    return events, context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/action_scoped_mechanism_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite mechanism demonstration")
    events, context = fixture()
    model, rows, max_score_error, max_mass_error = ActionScopedJointBN(), [], 0., 0.
    stream = ActionRiskStream(model, **context)
    for index, event in enumerate(events):
        stream.push(event)
        prefix = compile_action_trace({"events": events[:index + 1]}, **context)
        scores, _, mass = model._forward(model.theta, prefix)
        score_error = abs(scores[-1] - stream.probability)
        mass_error = float(np.max(np.abs(mass - stream.distribution)))
        np.testing.assert_allclose(scores[-1], stream.probability, atol=1e-13)
        np.testing.assert_allclose(mass, stream.distribution, atol=1e-13)
        max_score_error, max_mass_error = max(max_score_error, score_error), max(max_mass_error, mass_error)
        rows.append({"event": event, **stream.explain()})
    first = lambda i: next(a for a in rows[i]["actions"] if a["action_id"] == "inspect-workspace")
    before, mismatch, clean, expired = (first(i)["authorization_risk_hypothesis"] for i in (2, 3, 4, 5))
    if not clean < before < mismatch or abs(expired - before) > 1e-13:
        raise AssertionError("initial-parameter revision/expiry mechanism differs")
    numeric = {"authorization_risk_hypothesis", "information_at_dispatch_hypothesis"}
    if {k: v for k, v in first(7).items() if k not in numeric} != {k: v for k, v in first(10).items() if k not in numeric}:
        raise AssertionError("later input/cancellation changed dispatched historical record")
    history_error = max(abs(first(7)[k] - first(10)[k]) for k in numeric)
    if history_error > 1e-13:
        raise AssertionError("later input/cancellation changed dispatched state marginals")
    args.out.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(args.out, ROOT / 'knowledge/templates/action_scoped_bn_v3.json')
    for script in (Path(__file__), ROOT / 'code/commands/monitor_joint_events.py'):
        relative = script.resolve().relative_to(ROOT.resolve())
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, target)
        snapshots[relative.as_posix()] = {"original": str(script), "sha256": protocol.digest(target)}
    protocol.save_json(args.out / "model_parameters.json", model.parameters())
    protocol.save_json(args.out / "context.json", context)
    with (args.out / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    protocol.save_json(args.out / "manifest.json", {"source_snapshot": snapshots, "parameters_trained": False,
        "scope": "Hand-constructed lifecycle mechanism fixture, not a labeled test set or external validation.",
        "input_sha256": {name: protocol.digest(args.out / name) for name in ("events.jsonl", "context.json", "model_parameters.json")}})
    result = {"status": "passed", "parameters_trained": False, "events": len(events), "parameters": len(model.theta),
        "max_batch_stream_score_error": max_score_error, "max_joint_mass_error": max_mass_error,
        "authorization_revision_example": {"unknown": before, "mismatch": mismatch, "no_mismatch": clean, "expired": expired},
        "history_unchanged_after_later_input_and_other_cancellation": True,
        "max_historical_marginal_error": history_error,
        "scope": "No API/model/tool calls, labels, baseline comparison, fitted accuracy or safety guarantees.", "prefixes": rows}
    protocol.save_json(args.out / "REPORT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "prefixes"}, indent=2))


if __name__ == "__main__":
    main()
