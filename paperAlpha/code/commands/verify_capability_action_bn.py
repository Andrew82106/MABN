"""Freeze capability coverage and bounded mechanism checks, NOT accuracy.

No model API calls, training labels or benchmark policy enter this run.
Real lifecycle replay stops explicitly at the exact-inference capacity.
It never resets/chunks an episode to pretend the full sequence was handled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.capability_action_bn import CapabilityActionBN, CapabilityRiskStream, compile_capability_trace
from paperalpha_monitor.tool_capabilities import load_tool_registry, DEFAULT_PATH
from paperalpha_monitor.action_joint_bn import ActionScopedJointBN, ActionRiskStream
from paperalpha_monitor.action_evidence import compile_action_trace
from paperalpha_monitor.streaming_bn import PRIVILEGED_TOOLS
import evaluate_prefix_forecast_v1 as protocol
from demonstrate_action_scoped_bn import fixture


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def public_profile_check(registry, catalog):
    # Field projection happens before matching/annotation verification.
    public = {t["name"]: t["description"] for t in yaml.safe_load(catalog.read_text(encoding="utf-8"))["tools"]}
    profiles = registry.specification()["tools"]
    if {p["name"] for p in profiles} != set(public):
        raise AssertionError("catalog/profile tool sets differ")
    for p in profiles:
        if p["description"] != public[p["name"]]:
            raise AssertionError("profile public description differs from upstream")
    return {"tools": len(profiles), "all_descriptions_match": True,
            "annotation_method": "manual_review", "permissions_and_runtime_effects_verified": False}


def tool_coverage(registry, runs):
    samples = []
    for run in runs:
        for e in records(run / "arrival_events.jsonl"):
            if e["surface"] != "tool_call":
                continue
            name = e["data"]["tool_name"]
            samples.append({"run": run.name, "arrival_index": e["arrival_index"], "tool": name,
                            "profile": registry.resolve(name), "legacy_name_match": name in PRIVILEGED_TOOLS})
    return {"observed_tool_calls": len(samples), "legacy_name_matches": sum(s["legacy_name_match"] for s in samples),
            "resolved_primary_capabilities": dict(Counter(s["profile"]["capability"] for s in samples)),
            "unknown_calls": sum(s["profile"]["capability"] == "unknown" for s in samples), "calls": samples}


def lifecycle_events(path):
    """Tools-only projection in recorded handler-entry order, not MAS replay.

    No communication timestamps are merged heuristically. Missing scope stays
    null. Results and audit decisions are never fed into the event compiler.
    """
    result, seen = [], set()
    for e in records(path):
        if e["event_type"] != "tool_handler_started":
            continue
        identity = e["invocation_id"]
        if identity in seen:
            raise ValueError("duplicate actual handler invocation")
        seen.add(identity)
        result.append({"sequence": len(result), "event_id": identity, "event_type": "tool_call",
            "action_id": identity, "actor": e["actor"], "tool": e["tool"], "arguments": copy.deepcopy(e["arguments"]),
            "principal": None, "resource": None})
    return result


def bounded_replay(model, registry, events):
    stream = CapabilityRiskStream(model, tool_registry=registry)
    rows, stop = [], None
    for i, event in enumerate(events):
        before = None if not rows else stream.explain()
        try:
            stream.push(event)
        except ValueError as error:
            if "variable limit" not in str(error):
                raise
            if rows and stream.explain() != before:
                raise AssertionError("capacity error changed state")
            stop = {"reason": str(error), "first_unprocessed_sequence": event["sequence"]}
            break
        prefix = compile_capability_trace({"events": events[:i + 1]}, tool_registry=registry)
        scores, _, mass = model._forward(model.theta, prefix)
        np.testing.assert_allclose(scores[-1], stream.probability, atol=1e-14, rtol=0)
        np.testing.assert_allclose(mass, stream.distribution, atol=1e-14, rtol=0)
        rows.append(stream.explain())
    return {"kind": "tools_only_prefix_interface_check", "source_handler_entries": len(events),
        "accepted_prefix_events": len(rows), "unprocessed_events": len(events) - len(rows),
        "whole_episode_covered": stop is None, "stop": stop, "parameters_trained": False,
        "risk_scores_are_performance_evidence": False, "prefixes": rows}


def frozen_v3_regression(directory):
    package = directory / "source_snapshot/src/paperalpha_monitor"
    name = "_paperalpha_frozen_action_v3"
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    old_action = importlib.import_module(name + ".action_joint_bn")
    old_compile = importlib.import_module(name + ".action_evidence").compile_action_trace
    events = records(directory / "events.jsonl")
    context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
    rng, max_score, max_mass = np.random.default_rng(1909), 0., 0.
    current, old = ActionScopedJointBN(), old_action.ActionScopedJointBN()
    for i in range(13):
        theta = current.initial_parameters() + (0 if i == 0 else rng.normal(0, .5, len(current.theta)))
        current.theta, old.theta = theta.copy(), theta.copy()
        now, before = ActionRiskStream(current, **context), old_action.ActionRiskStream(old, **context)
        for e in events:
            now.push(e)
            before.push(e)
            max_score = max(max_score, abs(now.probability - before.probability))
            max_mass = max(max_mass, float(np.max(np.abs(now.distribution - before.distribution))))
        a = current.predict([compile_action_trace({"events": events}, **context)])[0]
        b = old.predict([old_compile({"events": events}, **context)])[0]
        max_score = max(max_score, float(np.max(np.abs(a - b))))
    if max(max_score, max_mass) > 1e-14:
        raise AssertionError("v3 mechanism numerics changed")
    return {"parameter_vectors": 13, "prefixes_per_vector": len(events), "max_score_error": max_score,
            "max_joint_mass_error": max_mass, "scope": "saved lifecycle fixture, not every possible v3 input"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/capability_action_verified_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite capability verification")
    registry, model = load_tool_registry(), CapabilityActionBN()
    runs = [ROOT / 'experiments/harnessaudit/runs/harnessaudit_api_pilot_dl14_20260919', ROOT / 'experiments/harnessaudit/runs/harnessaudit_api_pilot_dl11_lifecycle_20260919']
    catalog = ROOT / 'code/external/HarnessAudit/multi_agent/tools/daily_life.yaml'
    legacy = ROOT / 'experiments/model_mechanisms/runs/action_scoped_mechanism_verified_20260919'
    life = runs[1] / "tool_lifecycle.jsonl"
    events = lifecycle_events(life)
    public = public_profile_check(registry, catalog)
    covered = tool_coverage(registry, runs)
    bounded = bounded_replay(model, registry, events)
    regression = frozen_v3_regression(legacy)
    # A separate hand-constructed MAS lifecycle tests communication, checks,
    # expiry and cancellation. It is not derived from either real API task.
    manual, context = fixture()
    for e in manual:
        if "tool" in e:
            e["tool"] = "authorize_payment"
    context["tool_registry"] = registry
    stream, example, max_error = CapabilityRiskStream(model, **context), [], 0.
    for i, event in enumerate(manual):
        stream.push(event)
        trace = compile_capability_trace({"events": manual[:i + 1]}, **context)
        score, _, mass = model._forward(model.theta, trace)
        max_error = max(max_error, abs(score[-1] - stream.probability), float(np.max(np.abs(mass - stream.distribution))))
        example.append(stream.explain())
    if max_error > 1e-14:
        raise AssertionError("manual capability mechanism mismatch")
    args.out.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(args.out, DEFAULT_PATH)
    for path in (Path(__file__), ROOT / 'code/commands/demonstrate_action_scoped_bn.py'):
        relative = path.resolve().relative_to(ROOT.resolve())
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        snapshots[relative.as_posix()] = {"original": str(path), "sha256": protocol.digest(target)}
    files = [catalog, life, DEFAULT_PATH, *(r / "arrival_events.jsonl" for r in runs),
             legacy / "events.jsonl", legacy / "context.json",
             *sorted((legacy / "source_snapshot").rglob("*.py")), *sorted((legacy / "source_snapshot/configs").glob("*.json"))]
    protocol.save_json(args.out / "manifest.json", {"source_snapshot": snapshots,
        "inputs": {str(p): protocol.digest(p) for p in files}, "parameters_trained": False,
        "input_contract": "public descriptions; tool names for coverage; handler-entry fields for tools-only prefix check; no labels/policies/results in model inputs"})
    protocol.save_json(args.out / "model_parameters.json", model.parameters())
    protocol.save_json(args.out / "lifecycle_tools_only_projection.json", events)
    protocol.save_json(args.out / "manual_events.json", manual)
    protocol.save_json(args.out / "COVERAGE.json", covered)
    protocol.save_json(args.out / "BOUNDED_REPLAY.json", bounded)
    protocol.save_json(args.out / "MANUAL_MECHANISM.json", {"events": len(manual), "max_batch_stream_error": max_error, "prefixes": example})
    summary = {"status": "passed_for_stated_checks", "parameters": len(model.theta), "parameters_trained": False,
        "public_profile_check": public, "tool_coverage": {k: v for k, v in covered.items() if k != "calls"},
        "real_tools_only_prefix": {k: v for k, v in bounded.items() if k != "prefixes"},
        "v3_regression": regression, "manual_mechanism_events": len(manual), "max_batch_stream_error": max_error,
        "limitations": ["No fitted accuracy, calibration or baseline comparison.", "Primary capabilities are manual hypotheses, not runtime effect verification.",
            "Real tools-only prefix omits messages and stops at capacity; not a whole-MAS monitor evaluation.",
            "Missing action scope is not fabricated. No permission check is obtained from a tool name."]}
    protocol.save_json(args.out / "REPORT.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
