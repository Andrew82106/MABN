"""Export label-blind prefix examples with an executable knowledge-backed DAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.dynamic_joint_bn import compile_trace
from paperalpha_monitor.streaming_bn import PRIVILEGED_TOOLS
from monitor_joint_events import load_model
import evaluate_prefix_forecast_v1 as protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_v1_20260919')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/joint_knowledge_graph_examples_20260919')
    parser.add_argument("--method", default="joint_bn", choices=("joint_bn", "typed_joint_bn"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite graph examples")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    selected_id = sorted(manifest["splits"]["20260919"]["test"]["episode_ids"])[0]
    trace_paths = [p for p in manifest["input_sha256"] if Path(p).name == "traces_public.jsonl"]
    if len(trace_paths) != 1:
        raise ValueError("exactly one public trace source is required")
    path = ROOT / trace_paths[0]
    if protocol.digest(path) != manifest["input_sha256"][path.relative_to(ROOT).as_posix()]:
        raise ValueError("trace input changed")
    trace = next(t for t in protocol.read_jsonl(path) if t["episode_id"] == selected_id)
    parameters = args.run / "runtime_only__observed/20260919/model_parameters.json"
    model = load_model(parameters, args.method)
    if model.parameters()["name"] == "TypedJointBN-v2":
        from paperalpha_monitor.typed_joint_bn import compile_typed_trace
        compiler, config = compile_typed_trace, "typed_joint_evaluation_v2.json"
    else:
        compiler, config = compile_trace, "dynamic_joint_bn_v1.json"
    dispatch = next((i for i, e in enumerate(trace["events"]) if e["event_type"] == "tool_call" and e.get("tool") in PRIVILEGED_TOOLS), None)
    lengths = sorted({1, len(trace["events"]), *([] if dispatch is None or dispatch == 0 else [dispatch])})
    examples = []
    for length in lengths:
        prefix = compiler({"episode_id": selected_id, "events": trace["events"][:length]})
        explanation = model.explain(prefix, include_graph=True)
        if abs(explanation["graph_risk_probability"] - explanation["risk_probability"]) > 1e-12:
            raise AssertionError("graph and model differ")
        examples.append({"prefix_length": length, "last_sequence": prefix.steps[-1].sequence, **explanation})
    args.out.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(args.out, ROOT / 'knowledge/templates' / config)
    for script in (Path(__file__), ROOT / 'code/commands/monitor_joint_events.py'):
        relative = script.resolve().relative_to(ROOT.resolve())
        destination = args.out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, destination)
        snapshots[relative.as_posix()] = {"original": str(script), "sha256": protocol.digest(destination)}
    protocol.save_json(args.out / "manifest.json", {"source_run": str(args.run.resolve()),
        "run_manifest_sha256": protocol.digest(args.run / "manifest.json"), "method": args.method,
        "source_snapshot": snapshots, "public_trace_sha256": protocol.digest(path)})
    protocol.save_json(args.out / "graphs.json", {"selection": "first lexicographic test ID; no labels loaded",
        "episode_id": selected_id, "workflow_mode": "runtime_only", "parameter_file_sha256": protocol.digest(parameters),
        "examples": examples, "scope": "exact model explanations only, not new accuracy or causal evidence"})
    print(json.dumps({"episode_id": selected_id, "prefixes": lengths,
                      "graph_nodes": [len(e["knowledge_graph"]["nodes"]) for e in examples]}, indent=2))


if __name__ == "__main__":
    main()
