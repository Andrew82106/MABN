"""Independent arithmetic checks for fit continuation and phase decisions.

No refitting, threshold search on test data or mutations of original results.
Optimization uses the frozen BN loss; phase decisions are recomputed here
without importing the experimental alarm-policy implementation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import joblib
import numpy as np

from audit_dynamic_joint_bn_run import load_frozen
from audit_prefix_forecast_run import check, compare, contained, load_json, sha

ROOT = Path(__file__).resolve().parents[2]


def audit_optimization(run, root):
    manifest, plan, decisions = [load_json(run / name) for name in
                                ("manifest.json", "continuation_plan.json", "training_decisions.json")]
    check(decisions["complete"], "training decisions incomplete")
    check(sha(run / "continuation_plan.json") == decisions["plan_sha256"] == manifest["continuation"]["plan_sha256"], "plan hash differs")
    check(sha(run / "training_decisions.json") == manifest["continuation"]["training_decisions_sha256"], "training decision hash differs")
    for relative, digest in plan["source_sha256"].items():
        check(sha(contained(run, relative)) == digest, "continuation source changed")
    source = Path(plan["source_run"])
    check(sha(source / "manifest.json") == plan["source_manifest_sha256"] and
          sha(source / "metrics.json") == plan["source_metrics_sha256"], "original run changed")
    original = load_json(source / "manifest.json")
    check(manifest["config"] == original["config"] and manifest["splits"] == original["splits"] and
          manifest["input_sha256"] == original["input_sha256"], "model/data/split protocol changed")
    for relative, digest in manifest["input_sha256"].items():
        check(sha(contained(root, relative)) == digest, "input changed")
    evaluator = load_frozen(run, manifest)
    protocol = evaluator.protocol
    traces = protocol.read_jsonl(contained(root, manifest["config"]["data"]) / "traces_public.jsonl")
    labels = {r["episode_id"]: r["label"] for r in protocol.read_jsonl(contained(root, manifest["config"]["data"]) / "labels.jsonl")}
    splits, _ = protocol.resolve_manifest_splits(manifest, traces)
    config = plan["config"]
    config_files = [run / p for p in plan["source_sha256"] if p.endswith(".json")]
    check(len(config_files) == 1 and load_json(config_files[0]) == config, "frozen continuation config differs")
    by_key = {(r["setting"], r["seed"], r["method"]): r for r in decisions["records"]}
    check(len(by_key) == len(decisions["records"]), "duplicate training records")
    checked, worst = set(), 0.

    def diagnostic(model, fit, ys):
        loss, gradient = model.loss_gradient(model.theta, fit, ys, True)
        # Independently zero only the coordinates whose descent is out of bounds.
        feasible = np.asarray([g if (x > config["bounds"][0] or g <= 0) and
                    (x < config["bounds"][1] or g >= 0) else 0. for x, g in zip(model.theta, gradient)])
        pg = float(np.max(np.abs(feasible)))
        return {"fit_objective": float(loss), "gradient_inf": float(np.max(np.abs(gradient))),
                "projected_gradient_inf": pg, "stationarity_pass": pg <= config["stationarity_tolerance"],
                "optimizer": model.optimizer}

    for setting in manifest["config"]["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        for seed, parts in splits.items():
            fit = [evaluator.prepare_trace(traces[i], setting, manifest["config"]["max_joint_variables"]).joint for i in parts["fit"]]
            ys = np.asarray([labels[traces[i]["episode_id"]] for i in parts["fit"]])
            old_directory, new_directory = source / key / str(seed), run / key / str(seed)
            old_parameters, new_parameters = [load_json(d / "model_parameters.json") for d in (old_directory, new_directory)]
            for method in manifest["methods"]:
                old, new = old_parameters[method], new_parameters[method]
                old_path, new_path = [contained(d, p["artifact"]) for d, p in ((old_directory, old), (new_directory, new))]
                check(sha(old_path) == old["sha256"] and sha(new_path) == new["sha256"], "model hash differs")
                if method not in config["methods"]:
                    check(old == new, "non-joint baseline changed")
                    continue
                record = by_key[(key, seed, method)]
                check(record["fit_episode_ids"] == [traces[i]["episode_id"] for i in parts["fit"]], "fit IDs differ")
                check(record["original_model_sha256"] == old["sha256"] and record["result_model_sha256"] == new["sha256"], "model lineage differs")
                check(record["original_parameters_sha256"] == sha(old_directory / "model_parameters.json"), "original parameter file changed")
                before_model, after_model = joblib.load(old_path), joblib.load(new_path)
                check(before_model.regularization == after_model.regularization and
                      before_model.use_messages == after_model.use_messages and
                      before_model.use_latent_head == after_model.use_latent_head, "objective/architecture changed")
                before, after = diagnostic(before_model, fit, ys), diagnostic(after_model, fit, ys)
                compare(before, record["before"], "before fit diagnostic")
                compare(after, record["after"], "after fit diagnostic")
                selected = not before_model.optimizer["success"] or not before["stationarity_pass"]
                check(record["selected"] == selected, "fit-only continuation selection differs")
                if selected:
                    candidate = copy.deepcopy(before_model)
                    candidate.theta = np.asarray(record["candidate"]["theta"])
                    candidate.optimizer = record["candidate"]["optimizer"]
                    candidate_diagnostic = diagnostic(candidate, fit, ys)
                    compare(candidate_diagnostic, {k: record["candidate"][k] for k in candidate_diagnostic}, "candidate fit diagnostic")
                    accepted = candidate_diagnostic["fit_objective"] <= before["fit_objective"]
                    check(record["accepted"] == accepted, "training-objective acceptance differs")
                    if accepted:
                        np.testing.assert_array_equal(after_model.theta, candidate.theta)
                if not record["accepted"]:
                    check(old == new, "unselected/rejected model changed")
                worst = max(worst, after["projected_gradient_inf"])
                checked.add((key, seed, method))
            print(f"fit audit {key} {seed}", flush=True)
    check(checked == set(by_key), "training record inventory differs")
    check(len(checked) == decisions["checked_n"], "training count differs")
    check(sum(r["selected"] for r in by_key.values()) == decisions["selected_n"] and
          sum(r["accepted"] for r in by_key.values()) == decisions["accepted_n"] and
          sum(not r["after"]["stationarity_pass"] for r in by_key.values()) == decisions["stationarity_failed_n"], "decision totals differ")
    return {"status": "passed", "checked_joint_estimators": len(checked), "max_projected_gradient_inf": worst,
            "training_decisions_sha256": sha(run / "training_decisions.json"),
            "scope": "Frozen fit objectives, stationarity, model lineage and selection/acceptance rules; not global optimality or independent proof of original fitting history."}


def audit_phase(run, root):
    manifest, result = load_json(run / "manifest.json"), load_json(run / "metrics.json")
    check(result["complete"] and not result["probability_scores_changed"], "incomplete/non-decision experiment")
    check(sha(run / "calibration.json") == result["calibration_sha256"] and
          sha(run / "test_decisions.jsonl") == result["test_decisions_sha256"], "output hashes differ")
    for relative, record in manifest["source_snapshot"].items():
        check(sha(contained(run / "source_snapshot", relative)) == record["sha256"], "frozen policy source changed")
    source = Path(manifest["source_run"])
    for filename, key in (("manifest.json", "source_manifest_sha256"), ("metrics.json", "source_metrics_sha256"), ("AUDIT.json", "source_audit_sha256")):
        check(sha(source / filename) == manifest[key], "source experiment changed")
    base, source_audit = load_json(source / "manifest.json"), load_json(source / "AUDIT.json")
    check(source_audit["status"] == "passed" and source_audit["run_manifest_sha256"] == manifest["source_manifest_sha256"], "source replay audit differs")
    for relative, digest in base["input_sha256"].items():
        check(sha(contained(root, relative)) == digest, "input changed")
    def rows(path):
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    traces = {r["episode_id"]: r for r in rows(contained(root, base["config"]["data"]) / "traces_public.jsonl")}
    calibration = load_json(run / "calibration.json")
    calibration_index = {(r["setting"], r["seed"], r["method"]): r for r in calibration["records"]}
    metrics = {(r["setting"], r["seed"], r["method"]): r for r in result["records"]}
    check(len(calibration_index) == len(calibration["records"]) and len(metrics) == len(result["records"]), "duplicate policy records")
    config = manifest["config"]
    # Read the actual tool inventory from the frozen score implementation.
    evaluator = load_frozen(source, base)
    privileged = evaluator.protocol.PRIVILEGED_TOOLS
    phase_names = ("before_dispatch", "after_dispatch")
    check(sum(config["phase_budgets"].values()) <= config["total_empirical_false_alarm_budget"] + 1e-15, "invalid total budget")
    checked, count = set(), 0
    decision_handle = (run / "test_decisions.jsonl").open(encoding="utf-8")
    try:
        for relative, digest in manifest["prediction_sha256"].items():
            file = contained(source, relative)
            check(sha(file) == digest == source_audit["output_file_sha256"][relative], "source scores changed")
            setting, seed = file.parent.parent.name, int(file.parent.name)
            predictions = rows(file)
            negatives = [r for r in predictions if r["split"] == "validation_negative"]
            test = [r for r in predictions if r["split"] == "test"]
            for method in base["methods"]:
                key = setting, seed, method
                record = calibration_index[key]
                check([r["episode_id"] for r in negatives] == record["validation_negative_ids"] and
                      all(r["label"] == 0 for r in negatives), "calibration IDs/classes differ")
                maxima = []
                for row in negatives:
                    values = [[], []]
                    phase = 0
                    for e, score in zip(traces[row["episode_id"]]["events"], row["prefix_scores"][method]):
                        if e.get("event_type") == "tool_call" and e.get("tool") in privileged:
                            phase = 1
                        values[phase].append(score)
                    maxima.append([max(v) if v else None for v in values])
                thresholds = []
                phase_alarm_n = []
                for i, name in enumerate(phase_names):
                    observed = sorted(m[i] for m in maxima if m[i] is not None)
                    allowed = math.floor(config["phase_budgets"][name] * len(negatives))
                    threshold = float(np.nextafter(1., np.inf)) if not observed else (
                        0. if len(observed) <= allowed else float(np.nextafter(observed[-allowed - 1], np.inf)))
                    check(threshold == record["policy"]["thresholds"][name], "independent phase threshold differs")
                    alarms = sum(v >= threshold for v in observed)
                    phase_alarm_n.append(alarms)
                    compare({"threshold": threshold, "negative_episode_n": len(negatives),
                        "phase_observed_episode_n": len(observed), "allowed_alarm_n": allowed,
                        "actual_alarm_n": alarms, "empirical_fpr_all_negative_episodes": alarms / len(negatives),
                        "budget": config["phase_budgets"][name], "phase_disabled_no_calibration_coverage": not len(observed)},
                        record["phase_calibration"][name], "phase calibration")
                    thresholds.append(threshold)
                union_n = sum(any(v is not None and v >= t for v, t in zip(m, thresholds)) for m in maxima)
                check(union_n == record["any_alarm_n"] and sum(phase_alarm_n) == record["union_bound_alarm_n"] and
                      union_n / len(negatives) <= config["total_empirical_false_alarm_budget"] + 1e-15, "full policy budget differs")
                decisions = {"global": [], "phase": []}
                for row in test:
                    es, ps, phase = traces[row["episode_id"]]["events"], row["prefix_scores"][method], 0
                    phases, pd = [], []
                    for e, score in zip(es, ps):
                        if e.get("event_type") == "tool_call" and e.get("tool") in privileged:
                            phase = 1
                        phases.append(phase_names[phase])
                        pd.append(score >= thresholds[phase])
                    gd = [p >= record["global_threshold"]["threshold"] for p in ps]
                    actual = {"setting": setting, "seed": seed, "method": method, "episode_id": row["episode_id"],
                              "label": row["label"], "sequence": row["sequence"], "phase": phases,
                              "global_alarm": gd, "phase_alarm": pd}
                    line = decision_handle.readline()
                    check(bool(line), "missing test decision row")
                    compare(actual, json.loads(line), "test decisions")
                    count += len(ps)
                    decisions["global"].append(gd)
                    decisions["phase"].append(pd)
                for name, alarm_rows in decisions.items():
                    ys = [r["label"] for r in test]
                    first = [next((i for i, a in enumerate(alarms) if a), None) for alarms in alarm_rows]
                    n0, n1 = ys.count(0), ys.count(1)
                    proxies = {}
                    for proxy, restricted in (("first_tool_call", False), ("first_privileged_dispatch", True)):
                        eligible, early = 0, 0
                        for r, alarm in zip(test, first):
                            dispatch = next((i for i, e in enumerate(traces[r["episode_id"]]["events"])
                                if e.get("event_type") == "tool_call" and (not restricted or e.get("tool") in privileged)), None)
                            if r["label"] == 1 and dispatch is not None:
                                eligible += 1
                                early += int(alarm is not None and alarm < dispatch)
                        proxies[proxy] = {"eligible_positive_n": eligible, "strictly_before_alarm_n": early,
                                          "recall": early / eligible if eligible else None}
                    actual = {"negative_episode_n": n0, "positive_episode_n": n1,
                        "any_alarm_false_positive_rate": sum(a is not None and y == 0 for y, a in zip(ys, first)) / n0 if n0 else None,
                        "any_alarm_recall": sum(a is not None and y == 1 for y, a in zip(ys, first)) / n1 if n1 else None,
                        "warning_proxies": proxies, "actual_violation_lead_time": None}
                    compare(actual, metrics[key][name], "policy metrics")
                checked.add(key)
        check(not decision_handle.read().strip(), "extra test decision rows")
    finally:
        decision_handle.close()
    check(checked == set(metrics) == set(calibration_index), "policy result inventory differs")
    return {"status": "passed", "checked_policy_comparisons": len(checked), "checked_test_prefix_decisions_per_policy": count,
            "calibration_sha256": sha(run / "calibration.json"), "test_decisions_sha256": sha(run / "test_decisions.jsonl"),
            "scope": "Independent order-statistic thresholds, empirical union budgets, chronological decisions and metrics; no population or real-harm guarantee."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("optimization", "phase"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.run / ("OPTIMIZATION_AUDIT.json" if args.mode == "optimization" else "AUDIT.json")
    if output.exists():
        raise FileExistsError("Refusing to overwrite follow-up audit")
    result = (audit_optimization if args.mode == "optimization" else audit_phase)(args.run.resolve(), args.root.resolve())
    result.update(audited_at_utc=datetime.now(timezone.utc).isoformat(), audit_script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
