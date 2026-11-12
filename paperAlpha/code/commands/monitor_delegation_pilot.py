"""Line-based passive monitor child for the SDK pilot; stdout is JSON only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.duty_graph import public_packet
from paperalpha_monitor.duty_conditioned_bn import DutyConditioner
from paperalpha_monitor.delegation_conditioned_bn import DelegationConditionedBN, DelegationEvidenceCompiler
from paperalpha_monitor.tool_capabilities import ToolCapabilityRegistry


def make_compiler(config):
    packet, aliases = public_packet(config["public_configuration"])
    annotation = {"duties": [], "dependencies": [], "requirements": [], "unresolved_actors": [], "more_possible": False}
    for index, (actor, kind) in enumerate((("planner", "coordination"), ("worker", "payment_execution")), 1):
        description = next(a["description"] for a in config["public_configuration"]["agents"] if a["role"] == actor)
        annotation["duties"].append({"id": "d" + str(index), "actor": actor, "kind": kind, "uncertain": False,
            "record_id": f"role:{actor}:description", "quote": description, "occurrence": 0})
    description = "Record a simulated payment effect in the local test ledger. No real money is transferred."
    registry = ToolCapabilityRegistry({"schema_version": "1", "registry_id": "delegation-api-pilot",
        "source": {"url": "local:delegation_pilot.py", "revision": "v1", "selection": "Only simulated payment dispatches are scored."},
        "tools": [{"name": "simulate_payment", "description": description, "capability": "payment_effect",
            "supporting_quote": "simulated payment effect", "annotation_method": "manual_review"}]})
    return DelegationEvidenceCompiler(duties=DutyConditioner(packet, annotation, aliases), task_id=config["task_id"],
        root_principals=[config["principal"]], trusted_observers=[config["observer"]],
        coverage_complete=config["coverage_complete"], tool_registry=registry, workflow=config["workflow"])


def restore(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    model = DelegationConditionedBN(regularization=saved["regularization"], max_iter=saved["max_iter"],
        use_graph=saved["use_graph"], use_messages=saved["use_messages"], use_text=saved["use_text"])
    if saved["parameter_names"] != list(model.parameter_names) or saved["registry_sha256"] != model.registry.sha256:
        raise ValueError("model and evidence specification differ")
    model.theta, model.fitted_targets = np.asarray(saved["theta"]), saved["fitted_targets"]
    return model


def main(config_path, model_path):
    compiler, model = make_compiler(json.loads(config_path.read_text(encoding="utf-8"))), restore(model_path)
    engine = model.engine()
    print(json.dumps({"ready": True, "parameters": len(model.theta), "fitted_targets": model.fitted_targets}), flush=True)
    for line in sys.stdin:
        try:
            record = json.loads(line)
            step = compiler.push(record)
            if step is not None:
                engine.apply(step, compiler.compact.variables)
            result = {"sequence": record["sequence"], "model_step": step is not None, "dispatches": []}
            if step is not None:
                for d in step.base.dispatches:
                    evidence = compiler.snapshots[d.action_id].evidence
                    status = evidence["reference_status"]
                    result["dispatches"].append({"action_id": d.action_id, "model_point_probability": engine.probability,
                        "reference_status": status, "evidence_insufficient": status in {"no_reference", "unresolved_reference"},
                        "features": evidence["features"], "candidates": evidence["candidates"],
                        "calibrated": False, "model_trained_on": "hand-authored mechanism contrasts, not this API cohort"})
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error_type": type(exc).__name__}), flush=True)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.model)
