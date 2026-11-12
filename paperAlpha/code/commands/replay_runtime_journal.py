"""Sequential replay of shared runtime observations, not online efficacy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.journal import RuntimeEvidenceProjector
from paperalpha_monitor.compact_action_bn import CompactCapabilityBN, CompactCapabilityStream
from paperalpha_monitor.tool_capabilities import load_tool_registry, DEFAULT_PATH
from paperalpha_monitor.capability_action_bn import PARAMETER_NAMES, MODEL_SHA256
import evaluate_prefix_forecast_v1 as protocol


def replay(input_path, output, registry_path=DEFAULT_PATH, parameters=None):
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite runtime replay")
    registry, model = load_tool_registry(registry_path), CompactCapabilityBN()
    if parameters is not None:
        params = json.loads(Path(parameters).read_text(encoding="utf-8"))
        if params.get("parameter_names") != list(PARAMETER_NAMES) or params.get("capability_schema", {}).get("sha256") != MODEL_SHA256:
            raise ValueError("matching capability-v4 parameter layout and specification required")
        model.theta = np.asarray(params["theta"], dtype=float)
        for name in ("use_messages", "use_latent_head"):
            if type(params.get(name)) is not bool:
                raise ValueError("explicit boolean model switches required")
            setattr(model, name, params[name])
    monitor = CompactCapabilityStream(model, tool_registry=registry)
    projector = RuntimeEvidenceProjector()
    output.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(output, registry_path)
    for path in [Path(__file__), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py")]:
        relative = path.resolve().relative_to(ROOT.resolve())
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        snapshots[relative.as_posix()] = {"original": str(path), "sha256": protocol.digest(target)}
    shutil.copy2(input_path, output / "runtime_events.jsonl")
    accepted, failure, last = 0, None, None
    with (output / "monitor_events.jsonl").open("x", encoding="utf-8") as evidence, (output / "predictions.jsonl").open("x", encoding="utf-8") as scores:
        try:
            # Read in capture order. No complete-episode feature pass or sort.
            with (output / "runtime_events.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    event = projector.push(json.loads(line))
                    if event is not None:
                        last = monitor.push(event)
                        evidence.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                        scores.write(json.dumps(last, ensure_ascii=False, allow_nan=False) + "\n")
                        accepted += 1
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error)}
    report = {"status": "failed" if failure else "replayed", "failure": failure, "accepted_model_events": accepted,
        "journal": projector.report(), "last_snapshot": last, "model_name": model.model_name,
        "parameter_source": "untrained_default" if parameters is None else str(parameters),
        "parameter_training_or_calibration_independently_verified": False,
        "semantics": {"messages": "unknown information class; content not analyzed", "returns": "observed only, not converted into new input-risk sources",
                      "permissions": "unknown unless separately instrumented; no post-run audit input"},
        "scope": "offline sequential replay, not live warning latency, full risk coverage or predictive performance"}
    protocol.save_json(output / "REPORT.json", report)
    protocol.save_json(output / "model_parameters.json", model.parameters())
    inputs = [input_path, Path(registry_path), *([] if parameters is None else [Path(parameters)])]
    protocol.save_json(output / "manifest.json", {"source_snapshot": snapshots,
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}})
    if failure:
        raise ValueError("runtime replay failed; partial report and observations preserved")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tool-registry", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--parameters", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.input, args.out, args.tool_registry, args.parameters), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
