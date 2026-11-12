"""Freeze exact-elimination comparisons and long tools-only replay.

This has no fitted accuracy, API calls, benchmark labels or SOTA assertions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.capability_action_bn import CapabilityActionBN, compile_capability_trace
from paperalpha_monitor.compact_action_bn import CompactCapabilityBN, CompactCapabilityStream, compile_compact_trace
from paperalpha_monitor.tool_capabilities import load_tool_registry, DEFAULT_PATH
import evaluate_prefix_forecast_v1 as protocol
from demonstrate_action_scoped_bn import fixture
from verify_capability_action_bn import lifecycle_events


def project_full_mass(trace, mass, compact):
    mapping = [(next(i for i, v in enumerate(trace.variables) if v["kind"] == c["kind"] and v["id"] == c["id"]), j)
               for j, c in enumerate(compact.variables)]
    step, result = trace.steps[-1], np.zeros((1 << compact.n_variables, 3))
    for state, probability in enumerate(mass):
        index = sum(((state >> old) & 1) << new for old, new in mapping)
        any_a = any((state >> a) & 1 for a, _ in step.dispatched_pairs)
        overlap = any(((state >> a) & 1) and ((state >> d) & 1) for a, d in step.dispatched_pairs)
        result[index, 2 if overlap else 1 if any_a else 0] += probability
    return result


def small_equivalence(registry):
    events, ctx = fixture()
    for e in events:
        if "tool" in e:
            e["tool"] = "authorize_payment"
    ctx["tool_registry"] = registry
    a = compile_capability_trace({"events": events}, **ctx)
    b = compile_compact_trace({"events": events}, **ctx)
    full, compact = CapabilityActionBN(), CompactCapabilityBN()
    errors = {"score": 0., "joint_marginal": 0., "loss": 0., "gradient": 0.}
    for seed in range(13):
        theta = full.initial_parameters() + (0 if seed == 0 else np.random.default_rng(seed).normal(0, .8, 150))
        full.theta, compact.theta = theta.copy(), theta.copy()
        for i in range(len(events)):
            left = compile_capability_trace({"events": events[:i + 1]}, **ctx)
            right = compile_compact_trace({"events": events[:i + 1]}, **ctx)
            p, _, m = full._forward(theta, left)
            q, _, n = compact._forward(theta, right)
            errors["score"] = max(errors["score"], float(np.max(np.abs(p - q))))
            errors["joint_marginal"] = max(errors["joint_marginal"], float(np.max(np.abs(project_full_mass(left, m, right) - n))))
        for label in (0, 1):
            for prefix in (False, True):
                loss, grad = full.episode_loss_gradient(theta, a, label, prefix)
                new_loss, new_grad = compact.episode_loss_gradient(theta, b, label, prefix)
                errors["loss"] = max(errors["loss"], abs(loss - new_loss))
                errors["gradient"] = max(errors["gradient"], float(np.max(np.abs(grad - new_grad))))
    if max(errors.values()) > 1e-12:
        raise AssertionError(f"exact elimination mismatch: {errors}")
    return {"parameter_vectors": 13, "prefixes_per_vector": len(events), "gradient_comparisons": 52,
            "max_absolute_errors": errors}, events


def real_tools_replay(model, registry, events, previous):
    stream = CompactCapabilityStream(model, tool_registry=registry)
    rows = [stream.push(e) for e in events]
    trace = compile_compact_trace({"events": events}, tool_registry=registry)
    scores = model.predict([trace])[0]
    max_error = float(np.max(np.abs(scores - [r["risk_probability"] for r in rows])))
    old = json.loads(previous.read_text(encoding="utf-8"))["prefixes"]
    old_error = max(abs(p["risk_probability"] - rows[i]["risk_probability"]) for i, p in enumerate(old))
    if max(max_error, old_error) > 1e-12:
        raise AssertionError("real tool prefix numerical mismatch")
    loss, gradient = model.episode_loss_gradient(model.theta, trace, 0)
    # 0 here is a derivative probe only, explicitly NOT a label assigned to
    # this budget-truncated task. No fit or evaluation uses this probe.
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise AssertionError("long real tool trace has nonfinite computational derivative")
    return {"scope": "all recorded tool entries only; not all MAS communications or a completed task",
        "source_tool_entries": len(events), "accepted_tool_entries": len(rows), "unprocessed_tool_entries": 0,
        "information_variables": trace.n_variables, "retained_joint_states": stream.engine.distribution.size,
        "full_enumeration_logical_bits": trace.n_variables + 2 * len(events),
        "max_batch_stream_score_error": max_error, "max_saved_full_prefix_error": old_error,
        "computational_gradient_probe_finite": True, "gradient_probe_is_task_label": False,
        "parameters_trained": False, "prefixes": rows, "last_explanation": stream.explain()}


def stress_events(actions=1000):
    events = [{"sequence": 0, "event_type": "input_received", "actor": "reader", "source_id": "shared", "data_class": "untrusted"}]
    actors = ("reader", "planner", "executor")
    tools = ("get_customer_context", "message_provider", "authorize_payment")
    for i in range(actions):
        actor, target = actors[i % 3], actors[(i + 1) % 3]
        if i % 5 == 0:
            events.append({"sequence": len(events), "event_type": "message_sent", "actor": actor,
                           "target": target, "data_class": "untrusted" if i % 2 else "unknown"})
        events.append({"sequence": len(events), "event_id": f"call-{i}", "event_type": "tool_call", "actor": actor,
                       "action_id": f"action-{i}", "tool": tools[i % 3], "arguments": {"item": i}})
    return events


def long_stress(model, registry, events):
    stream = CompactCapabilityStream(model, tool_registry=registry)
    start, peak, max_mass_error = time.perf_counter(), 0, 0.
    for i, e in enumerate(events):
        stream.push(e)
        if not np.isfinite(stream.engine.probability) or not 0 <= stream.engine.probability <= 1:
            raise AssertionError("invalid stress score")
        peak = max(peak, stream.engine.distribution.size)
        max_mass_error = max(max_mass_error, abs(float(stream.engine.distribution.sum()) - 1))
        if (i + 1) % 250 == 0:
            print(f"compact stress accepted={i + 1} states={peak}", flush=True)
    if max_mass_error > 1e-12:
        raise AssertionError("stress probability mass drift")
    return {"scope": "hand-constructed sequential-state stress test, no labels or real tools",
        "events": len(events), "tool_dispatches": sum(e["event_type"] == "tool_call" for e in events),
        "accepted_events": stream.compiler.event_count, "information_variables": len(stream.compiler.variables),
        "peak_retained_joint_states": peak, "max_probability_mass_error": max_mass_error,
        "elapsed_seconds": time.perf_counter() - start, "history_reset_or_event_dropped": False,
        "parameters_trained": False, "final_score": stream.engine.probability}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/compact_action_verified_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite compact verification")
    registry, model = load_tool_registry(), CompactCapabilityBN()
    equivalent, manual = small_equivalence(registry)
    lifecycle = ROOT / 'experiments/harnessaudit/runs/harnessaudit_api_pilot_dl11_lifecycle_20260919/tool_lifecycle.jsonl'
    previous = ROOT / 'experiments/model_mechanisms/runs/capability_action_verified_20260919/BOUNDED_REPLAY.json'
    events = lifecycle_events(lifecycle)
    real = real_tools_replay(model, registry, events, previous)
    stress = stress_events()
    long = long_stress(model, registry, stress)
    args.out.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(args.out, DEFAULT_PATH)
    for path in (Path(__file__), ROOT / 'code/commands/demonstrate_action_scoped_bn.py', ROOT / 'code/commands/verify_capability_action_bn.py'):
        relative = path.resolve().relative_to(ROOT.resolve())
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        snapshots[relative.as_posix()] = {"original": str(path), "sha256": protocol.digest(target)}
    protocol.save_json(args.out / "manifest.json", {"source_snapshot": snapshots,
        "inputs": {str(p): protocol.digest(p) for p in (lifecycle, previous, DEFAULT_PATH)},
        "parameters_trained": False, "target_unchanged": "capability-v4 terminal risk given observed prefix"})
    protocol.save_json(args.out / "model_parameters.json", model.parameters())
    protocol.save_json(args.out / "real_tools_only_events.json", events)
    protocol.save_json(args.out / "REAL_TOOLS_REPLAY.json", real)
    protocol.save_json(args.out / "manual_events.json", manual)
    protocol.save_json(args.out / "stress_events.json", stress)
    result = {"status": "passed_for_stated_checks", "small_exact_equivalence": equivalent,
        "real_tools_replay": {k: v for k, v in real.items() if k not in {"prefixes", "last_explanation"}},
        "synthetic_long_sequence": long, "parameters": len(model.theta), "parameters_trained": False,
        "limitations": ["Exact only for current v4 independence/factor/readout assumptions.",
            "Information/source state remains exponential; metadata and prefix compilation are not constant-space.",
            "No complete real MAS trace, learned accuracy, calibration or baseline advantage verified."]}
    protocol.save_json(args.out / "REPORT.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
