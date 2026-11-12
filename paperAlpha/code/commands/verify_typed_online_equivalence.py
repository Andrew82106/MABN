"""Full-corpus replay: frozen typed batch vs incremental and executable graph.

State bits are compared by semantic identity, not their allocation order.
Labels are not parsed, no model is fitted, and no accuracy claim is made.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor import TypedRiskStream
from paperalpha_monitor.typed_joint_bn import compile_typed_trace
from paperalpha_monitor.typed_graph import compile_typed_graph
from monitor_joint_events import load_model
import evaluate_prefix_forecast_v1 as protocol


def project_mass(distribution, variables, target_variables):
    """Marginalize/reorder by (kind,id), rejecting nonzero future-state mass."""
    identity = lambda v: (v["kind"], v["id"])
    indices = {identity(v): i for i, v in enumerate(variables)}
    if len(indices) != len(variables) or len({identity(v) for v in target_variables}) != len(target_variables):
        raise ValueError("duplicate state identity")
    positions = [indices[identity(v)] for v in target_variables]
    if any(variables[i] != v for i, v in zip(positions, target_variables)):
        raise ValueError("state metadata differs")
    states = np.arange(1 << len(variables))
    future = set(range(len(variables))) - set(positions)
    for i in future:
        np.testing.assert_equal(distribution[(states & (1 << i)) != 0], 0.)
    buckets = np.zeros(len(states), dtype=np.int64)
    for target, old in enumerate(positions):
        buckets |= ((states >> old) & 1) << target
    return np.bincount(buckets, weights=distribution, minlength=1 << len(positions))


def load_reference(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    snapshot = run / "source_snapshot"
    for relative, record in manifest["source_snapshot"].items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot.resolve()) or protocol.digest(path) != record["sha256"]:
            raise ValueError("frozen reference source changed: " + relative)
    name = "_typed_online_reference"
    if name in sys.modules:
        raise ValueError("fresh process required")
    package = types.ModuleType(name)
    package.__path__ = [str(snapshot / 'code/src/paperalpha_monitor')]
    sys.modules[name] = package
    return importlib.import_module(name + ".typed_joint_bn"), manifest


def verify(run, output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite equivalence results")
    reference, manifest = load_reference(run)
    audit = json.loads((run / "AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["run_manifest_sha256"] != protocol.digest(run / "manifest.json"):
        raise ValueError("full reference audit required")
    for relative, expected in manifest["input_sha256"].items():
        if protocol.digest(ROOT / relative) != expected:
            raise ValueError("reference input changed")
    parameters = run / "runtime_only__observed/20260919/model_parameters.json"
    if protocol.digest(parameters) != audit["output_file_sha256"][parameters.relative_to(run).as_posix()]:
        raise ValueError("audited parameter file changed")
    current = load_model(parameters, "typed_joint_bn")
    saved = json.loads(parameters.read_text(encoding="utf-8"))["typed_joint_bn"]
    artifact = (parameters.parent / saved["artifact"]).resolve()
    if not artifact.is_relative_to(parameters.parent.resolve()) or protocol.digest(artifact) != saved["sha256"]:
        raise ValueError("model artifact changed")
    if saved["bn"]["parameter_names"] != reference.PARAMETER_NAMES:
        raise ValueError("reference parameter layout differs")
    old = reference.TypedJointBN(typed_head=current.typed_head, use_messages=current.use_messages, use_latent_head=current.use_latent_head)
    old.theta = current.theta.copy()
    data_paths = [p for p in manifest["input_sha256"] if Path(p).name == "traces_public.jsonl"]
    if len(data_paths) != 1:
        raise ValueError("expected exactly one public trace input")
    traces = protocol.read_jsonl(ROOT / data_paths[0])
    output.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(output, ROOT / 'experiments/model_mechanisms/configs/typed_joint_evaluation_v2.json')
    for script in (Path(__file__), ROOT / 'code/commands/monitor_joint_events.py'):
        relative = script.resolve().relative_to(ROOT.resolve())
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, target)
        snapshots[relative.as_posix()] = {"original": str(script), "sha256": protocol.digest(target)}
    protocol.save_json(output / "manifest.json", {"reference_run": str(run),
        "reference_manifest_sha256": protocol.digest(run / "manifest.json"),
        "reference_audit_sha256": protocol.digest(run / "AUDIT.json"),
        "reference_parameter_file_sha256": protocol.digest(parameters), "theta": current.theta.tolist(),
        "input_sha256": manifest["input_sha256"], "source_snapshot": snapshots,
        "scope": "One fixed learned vector; all public traces/settings, full typed joint state, incremental scores and executable graph. No labels or fitting."})
    results, started = [], time.perf_counter()
    for setting in manifest["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        score_error, mass_error, graph_error, count = 0., 0., 0., 0
        for episode, trace in enumerate(traces, 1):
            context = {"workflow": trace["workflow_dag"] if setting["workflow"] == "workflow_known" else None,
                       "permission_mode": setting["permission"], "max_variables": manifest["config"]["max_joint_variables"]}
            frozen = reference.compile_typed_trace(trace, **context)
            expected, tape, _ = old._forward(old.theta, frozen, retain=True)
            current_trace = compile_typed_trace(trace, **context)
            graph = compile_typed_graph(current_trace, typed_head=current.typed_head)
            graph_scores = graph.predict(current.theta, use_messages=current.use_messages, use_latent_head=current.use_latent_head)
            np.testing.assert_allclose(graph_scores, expected, atol=1e-12, rtol=1e-12)
            graph_error = max(graph_error, float(np.max(np.abs(graph_scores - expected))))
            stream = TypedRiskStream(current, **context)
            if not len(trace["events"]) == len(expected) == len(tape):
                raise ValueError("prefix count differs")
            for event, probability, cache in zip(trace["events"], expected, tape):
                actual = stream.push(event)["risk_probability"]
                projected = project_mass(cache[1], frozen.variables, stream._compiler.variables)
                np.testing.assert_allclose(actual, probability, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(stream._distribution, projected, atol=1e-12, rtol=1e-12)
                score_error = max(score_error, abs(actual - probability))
                mass_error = max(mass_error, float(np.max(np.abs(stream._distribution - projected))))
                count += 1
            if len(frozen.variables) != len(stream._compiler.variables):
                raise ValueError("final variable inventory differs")
            if episode % 1000 == 0:
                print(f"verified typed online/graph {key}: {episode}/{len(traces)}", flush=True)
        results.append({"setting": key, "episodes": len(traces), "prefixes": count,
                        "max_probability_error": score_error, "max_joint_mass_error": mass_error, "max_graph_error": graph_error})
    result = {"status": "passed", "settings": results, "elapsed_seconds": time.perf_counter() - started,
        "episode_count": sum(r["episodes"] for r in results), "prefix_count": sum(r["prefixes"] for r in results),
        **{k: max(r[k] for r in results) for k in ("max_probability_error", "max_joint_mass_error", "max_graph_error")},
        "limitations": ["Implementation equivalence, not evidence of real risk semantics or external generalization.",
                        "One learned vector used in every setting only to verify numerical implementation.",
                        "No live agent-framework integration, text extraction, authorization recovery or causal validation.",
                        "The synthetic corpus lacks post-dispatch messages, explicit source IDs and role manifests; dedicated unit cases supplement it."]}
    protocol.save_json(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/typed_joint_bn_v2_20260919')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/typed_online_graph_equivalence_20260919')
    args = parser.parse_args()
    print(json.dumps(verify(args.run.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
