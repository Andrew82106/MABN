"""Compare the incremental monitor with the pre-refactor frozen batch model.

All four visibility settings and every existing public development trace are
checked. This is numerical/temporal equivalence, not a new performance result.
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
from paperalpha_monitor import DynamicJointBN, JointRiskStream
import evaluate_prefix_forecast_v1 as protocol


def load_reference(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    snapshot = run / "source_snapshot"
    for relative, record in manifest["source_snapshot"].items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot.resolve()) or protocol.digest(path) != record["sha256"]:
            raise ValueError("frozen reference source changed: " + relative)
    package = types.ModuleType("_online_reference_monitor")
    package.__path__ = [str(snapshot / 'code/src/paperalpha_monitor')]
    if package.__name__ in sys.modules:
        raise ValueError("reference package already loaded; use a fresh process")
    sys.modules[package.__name__] = package
    module = importlib.import_module(package.__name__ + ".dynamic_joint_bn")
    return module, manifest


def verify(reference_run, output, include_graph=False):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite equivalence check: {output}")
    reference, manifest = load_reference(reference_run)
    for relative, expected in manifest["input_sha256"].items():
        if protocol.digest(ROOT / relative) != expected:
            raise ValueError("input no longer matches reference")
    parameter_file = reference_run / "runtime_only__observed/20260919/model_parameters.json"
    parameter_record = json.loads(parameter_file.read_text(encoding="utf-8"))["joint_bn"]
    artifact = parameter_file.parent / parameter_record["artifact"]
    if protocol.digest(artifact) != parameter_record["sha256"]:
        raise ValueError("reference model artifact changed")
    if parameter_record["bn"]["parameter_names"] != reference.PARAMETER_NAMES:
        raise ValueError("reference parameter layout differs")
    theta = np.asarray(parameter_record["bn"]["theta"])
    # No labels are loaded or used to select episodes or tune parameters.
    traces = protocol.read_jsonl(ROOT / manifest["config"]["data"] / "traces_public.jsonl")
    old, current = reference.DynamicJointBN(), DynamicJointBN()
    old.theta, current.theta = theta.copy(), theta.copy()
    output.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(output, ROOT / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json')
    source = Path(__file__).resolve()
    relative = source.relative_to(ROOT.resolve())
    target = output / "source_snapshot" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    snapshots[relative.as_posix()] = {"original": str(source), "sha256": protocol.digest(target)}
    protocol.save_json(output / "manifest.json", {"reference_run": str(reference_run),
        "reference_manifest_sha256": protocol.digest(reference_run / "manifest.json"),
        "reference_parameter_file_sha256": protocol.digest(parameter_file),
        "reference_model_sha256": parameter_record["sha256"], "theta": theta.tolist(),
        "input_sha256": manifest["input_sha256"], "source_snapshot": snapshots,
        "include_knowledge_graph": include_graph,
        "scope": "same learned parameters, all public traces, all visibility settings; numerical equivalence only"})
    rows, started = [], time.perf_counter()
    for setting in manifest["config"]["settings"]:
        score_error, mass_error, graph_error, prefixes = 0., 0., 0., 0
        key = setting["workflow"] + "__" + setting["permission"]
        for count, trace in enumerate(traces, 1):
            context = {"workflow": trace["workflow_dag"] if setting["workflow"] == "workflow_known" else None,
                       "permission_mode": setting["permission"], "max_variables": manifest["config"]["max_joint_variables"]}
            compiled = reference.compile_trace(trace, **context)
            expected, tape, _ = old._forward(theta, compiled, retain=True)
            if include_graph:
                from paperalpha_monitor.dynamic_joint_bn import compile_trace as compile_current
                from paperalpha_monitor.joint_graph import compile_joint_graph
                graph = compile_joint_graph(compile_current(trace, **context))
                graph_scores = graph.predict(theta)
                np.testing.assert_allclose(graph_scores, expected, atol=1e-12, rtol=1e-12)
                graph_error = max(graph_error, float(np.max(np.abs(graph_scores - expected))))
            stream = JointRiskStream(current, **context)
            for event, probability, cache in zip(trace["events"], expected, tape):
                actual = stream.push(event)["risk_probability"]
                distribution = stream._distribution
                old_distribution = cache[1]
                np.testing.assert_allclose(actual, probability, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(distribution, old_distribution[:len(distribution)], atol=1e-12, rtol=1e-12)
                np.testing.assert_equal(old_distribution[len(distribution):], 0.)
                score_error = max(score_error, abs(actual - probability))
                mass_error = max(mass_error, float(np.max(np.abs(distribution - old_distribution[:len(distribution)]))))
                prefixes += 1
            if stream._compiler.variables != list(compiled.variables):
                raise AssertionError("variable identity/order differs from frozen batch")
            if count % 1000 == 0:
                print(f"verified online {key}: {count}/{len(traces)} episodes", flush=True)
        rows.append({"setting": key, "episodes": len(traces), "prefixes": prefixes,
                     "max_probability_error": score_error, "max_joint_mass_error": mass_error,
                     "max_knowledge_graph_error": graph_error if include_graph else None})
    result = {"status": "passed", "settings": rows, "elapsed_seconds": time.perf_counter() - started,
              "episode_count": sum(r["episodes"] for r in rows), "prefix_count": sum(r["prefixes"] for r in rows),
              "max_probability_error": max(r["max_probability_error"] for r in rows),
              "max_joint_mass_error": max(r["max_joint_mass_error"] for r in rows),
              "max_knowledge_graph_error": max(r["max_knowledge_graph_error"] for r in rows) if include_graph else None,
              "limitations": ["Equivalence to a frozen implementation does not establish that its assumptions are true.",
                              "One learned parameter vector is used across settings solely for implementation comparison.",
                              "No labels, retraining, new accuracy result or independent real-world validation.",
                              "Explicit-role, shared-source-ID and recovery edge cases require separate tests; the benchmark lacks these annotations."]}
    protocol.save_json(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_v1_20260919')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/joint_online_equivalence_20260919')
    parser.add_argument("--include-graph", action="store_true")
    args = parser.parse_args()
    result = verify(args.reference_run.resolve(), args.out.resolve(), args.include_graph)
    print(json.dumps({k: result[k] for k in ("status", "episode_count", "prefix_count", "max_probability_error", "max_joint_mass_error")}, indent=2))


if __name__ == "__main__":
    main()
