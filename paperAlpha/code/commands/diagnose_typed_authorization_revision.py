"""Demonstrate v2's action-evidence revision gap without training or labels.

Same proposal, same action ID, two later permission observations. This is a
mechanism diagnostic, not a benchmark, attack, or evidence of actual grants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor import TypedRiskStream
from monitor_joint_events import load_model
import evaluate_prefix_forecast_v1 as protocol


def diagnose(model):
    proposal = {"sequence": 0, "event_type": "tool_proposed", "actor": "executor", "tool": "shell.exec", "action_id": "action-1"}
    cases = []
    for permission in (False, True):
        stream = TypedRiskStream(model)
        stream.push(proposal)
        before = stream.explain()
        dispatch = {**proposal, "sequence": 1, "event_type": "tool_call", "permission_mismatch": permission}
        stream.push(dispatch)
        after = stream.explain()
        auth = lambda record: next(v["marginal"] for v in record["variables"] if v["kind"] == "authorization_state")
        cases.append({"events": [proposal, dispatch], "permission_mismatch": permission,
                      "authorization_before": auth(before), "authorization_after": auth(after),
                      "risk_before": before["risk_probability"], "risk_after": after["risk_probability"],
                      "new_latent_transitions": after["latest_event_transitions"],
                      "global_permission_check_seen": after["observed_features"]["permission_check_seen"],
                      "global_permission_mismatch_seen": after["observed_features"]["permission_mismatch_seen"]})
    return {"cases": cases,
        "same_authorization_state_under_opposite_later_checks": cases[0]["authorization_after"] == cases[1]["authorization_after"],
        "later_check_updates_latent": any(c["authorization_before"] != c["authorization_after"] for c in cases),
        "scope": "Known v2 limitation: later same-action evidence updates global features, not its already-seeded latent state. No claim that a clean flag proves authorization."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/typed_joint_bn_v2_20260919')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/typed_authorization_revision_diagnostic_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite diagnostic")
    audit = json.loads((args.run / "AUDIT.json").read_text(encoding="utf-8"))
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["run_manifest_sha256"] != protocol.digest(manifest_path):
        raise ValueError("complete audited reference required")
    results = []
    for setting in manifest["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        for seed in manifest["config"]["expected_seeds"]:
            path = args.run / key / str(seed) / "model_parameters.json"
            digest = protocol.digest(path)
            if digest != audit["output_file_sha256"][path.relative_to(args.run).as_posix()]:
                raise ValueError("audited parameter record changed")
            model = load_model(path, "typed_joint_bn")
            results.append({"training_setting": key, "seed": seed, "parameter_file_sha256": digest, **diagnose(model)})
    args.out.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(args.out, ROOT / 'experiments/model_mechanisms/configs/typed_joint_evaluation_v2.json')
    for source in (Path(__file__), ROOT / 'code/commands/monitor_joint_events.py'):
        relative = source.resolve().relative_to(ROOT.resolve())
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relative.as_posix()] = {"original": str(source), "sha256": protocol.digest(target)}
    report = {"reference_run": str(args.run.resolve()), "reference_manifest_sha256": protocol.digest(manifest_path),
        "reference_audit_sha256": protocol.digest(args.run / "AUDIT.json"), "source_snapshot": snapshots,
        "diagnostic_context": "All vectors evaluated with permission observations visible and no workflow; training_setting identifies provenance, not the diagnostic context.",
        "models": results, "fitting_or_labels_used": False}
    protocol.save_json(args.out / "REPORT.json", report)
    print(json.dumps({"models": len(results), "unchanged_under_opposite_checks": sum(r["same_authorization_state_under_opposite_later_checks"] for r in results),
                      "latent_revision_detected": sum(r["later_check_updates_latent"] for r in results)}, indent=2))


if __name__ == "__main__":
    main()
