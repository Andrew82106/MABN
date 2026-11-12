"""Compare one frozen, shared phase policy against whole-trajectory thresholds.

No model fitting. Per-method policies use validation negatives only. This is
an alarm decision experiment on existing development probabilities, not a new
architecture result. Calibration artifacts are all frozen before test scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
import evaluate_prefix_forecast_v1 as protocol
from paperalpha_monitor.alarm_policy import calibrate_policy, decide_policy, dispatch_phases
from paperalpha_monitor.streaming_bn import PRIVILEGED_TOOLS


def alarm_metrics(labels, events, decisions):
    labels = np.asarray(labels)
    if not len(labels) or len(labels) != len(events) or len(labels) != len(decisions) or not np.isin(labels, [0, 1]).all():
        raise ValueError("binary labels and episode decisions must align")
    if any(not es or len(es) != len(ds) for es, ds in zip(events, decisions)):
        raise ValueError("one decision per event required")
    first = [next((i for i, d in enumerate(ds) if d["alarm"]), None) for ds in decisions]
    alarms = np.asarray([i is not None for i in first])
    negative, positive = labels == 0, labels == 1
    proxies = {}
    for name, privileged in (("first_tool_call", False), ("first_privileged_dispatch", True)):
        eligible, early = 0, 0
        for y, es, alarm in zip(labels, events, first):
            dispatch = next((i for i, e in enumerate(es) if e.get("event_type") == "tool_call"
                             and (not privileged or e.get("tool") in PRIVILEGED_TOOLS)), None)
            if y == 1 and dispatch is not None:
                eligible += 1
                early += int(alarm is not None and alarm < dispatch)
        proxies[name] = {"eligible_positive_n": eligible, "strictly_before_alarm_n": early,
                         "recall": early / eligible if eligible else None}
    return {"negative_episode_n": int(negative.sum()), "positive_episode_n": int(positive.sum()),
            "any_alarm_false_positive_rate": float(alarms[negative].mean()) if negative.any() else None,
            "any_alarm_recall": float(alarms[positive].mean()) if positive.any() else None,
            "warning_proxies": proxies, "actual_violation_lead_time": None}


def read_run(source):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    if not results["complete"]:
        raise ValueError("source experiment must be complete")
    audit = json.loads((source / "AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["run_manifest_sha256"] != protocol.digest(source / "manifest.json"):
        raise ValueError("source replay audit missing or stale")
    for relative, expected in manifest["input_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or protocol.digest(path) != expected:
            raise ValueError("source data changed")
    traces = protocol.read_jsonl(ROOT / manifest["config"]["data"] / "traces_public.jsonl")
    labels = {r["episode_id"]: r["label"] for r in protocol.read_jsonl(ROOT / manifest["config"]["data"] / "labels.jsonl")}
    splits, _ = protocol.resolve_manifest_splits(manifest, traces)
    by_id = {t["episode_id"]: t for t in traces}
    runs, hashes = [], {}
    for run in results["runs"]:
        relative = f"{run['setting']}/{run['seed']}/predictions.jsonl"
        path = source / relative
        digest = protocol.digest(path)
        if digest != audit["output_file_sha256"][relative]:
            raise ValueError("source predictions differ from replay audit")
        hashes[relative] = digest
        rows = protocol.read_jsonl(path)
        selected = splits[run["seed"]]
        val = [i for i in selected["validation"] if labels[traces[i]["episode_id"]] == 0]
        expected = [(part, traces[i]["episode_id"]) for part, indices in (("validation_negative", val), ("test", selected["test"])) for i in indices]
        if [(r["split"], r["episode_id"]) for r in rows] != expected:
            raise ValueError("source prediction inventory differs")
        for row in rows:
            es = by_id[row["episode_id"]]["events"]
            if row["label"] != labels[row["episode_id"]] or row["sequence"] != [e["sequence"] for e in es] or set(row["prefix_scores"]) != set(manifest["methods"]):
                raise ValueError("prediction row metadata differs")
        runs.append((run, rows))
    expected_keys = {(s["workflow"] + "__" + s["permission"], seed) for s in manifest["config"]["settings"] for seed in splits}
    if len(runs) != len(expected_keys) or {(r["setting"], r["seed"]) for r, _ in runs} != expected_keys:
        raise ValueError("source run inventory differs")
    return manifest, by_id, runs, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / 'experiments/runtime_alarm/configs/phase_alarm_policy_v1.json')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/runtime_alarm/runs/phase_alarm_policy_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite policy experiment")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = (ROOT / config["source_run"]).resolve()
    manifest, traces, runs, hashes = read_run(source)
    if config["total_empirical_false_alarm_budget"] != manifest["config"]["threshold"]["validation_negative_any_alarm_fpr"]:
        raise ValueError("total budget differs from baseline protocol")
    args.out.mkdir(parents=True)
    snapshots = protocol.snapshot_sources(args.out, args.config)
    script = Path(__file__)
    target = args.out / "source_snapshot/code/commands" / script.name
    shutil.copy2(script, target)
    snapshots['code/commands/' + script.name] = {"original": str(script), "sha256": protocol.digest(target)}
    protocol.save_json(args.out / "manifest.json", {"config": config, "source_run": str(source),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": protocol.digest(source / "manifest.json"),
        "source_metrics_sha256": protocol.digest(source / "metrics.json"),
        "source_audit_sha256": protocol.digest(source / "AUDIT.json"),
        "prediction_sha256": hashes, "source_snapshot": snapshots})
    calibrations = []
    for run, rows in runs:
        negative = [r for r in rows if r["split"] == "validation_negative"]
        for method in manifest["methods"]:
            pairs = [(traces[r["episode_id"]]["events"], r["prefix_scores"][method]) for r in negative]
            calibration = calibrate_policy(pairs, config["phase_budgets"], config["total_empirical_false_alarm_budget"])
            global_threshold = protocol.choose_alarm_threshold([max(scores) for _, scores in pairs], config["total_empirical_false_alarm_budget"])
            if global_threshold != run["metrics"][method]["threshold_selection"]:
                raise ValueError("original baseline threshold cannot be reproduced")
            calibrations.append({"setting": run["setting"], "seed": run["seed"], "method": method,
                "validation_negative_ids": [r["episode_id"] for r in negative],
                "global_threshold": global_threshold, **calibration})
    protocol.save_json(args.out / "calibration.json", {"completed_before_test_decisions": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "records": calibrations})
    frozen_calibration_sha = protocol.digest(args.out / "calibration.json")
    lookup = {(r["setting"], r["seed"], r["method"]): r for r in calibrations}
    results, decisions_out = [], args.out / "test_decisions.jsonl"
    with decisions_out.open("w", encoding="utf-8") as handle:
        for run, rows in runs:
            test = [r for r in rows if r["split"] == "test"]
            events, labels = [traces[r["episode_id"]]["events"] for r in test], [r["label"] for r in test]
            for method in manifest["methods"]:
                calibration = lookup[(run["setting"], run["seed"], method)]
                scores = [r["prefix_scores"][method] for r in test]
                global_threshold = calibration["global_threshold"]["threshold"]
                global_decisions = [[{"alarm": p >= global_threshold} for p in ps] for ps in scores]
                phase_decisions = [decide_policy(es, ps, calibration["policy"]) for es, ps in zip(events, scores)]
                actual_global = alarm_metrics(labels, events, global_decisions)
                for key, value in actual_global.items():
                    if value != run["metrics"][method][key]:
                        raise ValueError("global policy metrics differ from original evaluation: " + key)
                results.append({"setting": run["setting"], "seed": run["seed"], "method": method,
                    "global": actual_global, "phase": alarm_metrics(labels, events, phase_decisions),
                    "probability_scores_changed": False})
                for row, es, gd, pd in zip(test, events, global_decisions, phase_decisions):
                    handle.write(json.dumps({"setting": run["setting"], "seed": run["seed"], "method": method,
                        "episode_id": row["episode_id"], "label": row["label"], "sequence": row["sequence"],
                        "phase": dispatch_phases(es), "global_alarm": [d["alarm"] for d in gd],
                        "phase_alarm": [d["alarm"] for d in pd]}, allow_nan=False) + "\n")
    if protocol.digest(args.out / "calibration.json") != frozen_calibration_sha:
        raise ValueError("calibration changed during evaluation")
    protocol.save_json(args.out / "metrics.json", {"complete": True, "records": results,
        "calibration_sha256": frozen_calibration_sha, "test_decisions_sha256": protocol.digest(decisions_out),
        "scope": config["scope"], "probability_scores_changed": False})
    lines = ["# Fixed phase-aware alarm policy: development comparison", "",
        "Same fitted probabilities and 0.10 total empirical validation budget for all methods; fixed pre/post allocation 0.08/0.02.",
        "First privileged dispatch is only a timing proxy. Different test false-alarm rates are not equal-FPR comparisons.", "",
        "| setting | method | global FPR | phase FPR | global early recall | phase early recall | global any-alarm recall | phase any-alarm recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for setting in sorted({r["setting"] for r in results}):
        for method in manifest["methods"]:
            selected = [r for r in results if r["setting"] == setting and r["method"] == method]
            values = [np.mean([r[p][metric] if metric != "early" else r[p]["warning_proxies"]["first_privileged_dispatch"]["recall"] for r in selected])
                      for metric in ("any_alarm_false_positive_rate", "early", "any_alarm_recall") for p in ("global", "phase")]
            lines.append(f"| {setting} | {method} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "All rows are equal-seed averages over overlapping reused development partitions, not independent replications.",
              "No default deployment policy is changed by this experiment; no new architecture or probability-calibration gain is claimed."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"method_comparisons": len(results), "output": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
