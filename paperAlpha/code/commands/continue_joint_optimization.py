"""Fit-only continuation of a frozen joint-model experiment, into a NEW run.

The original artifacts are never modified. Every joint-family estimator is
checked under the same criterion. All training decisions are committed before
new validation/test prediction. The existing frozen replay auditor also works
on the output; the separate continuation record preserves the extra fit code.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from scipy.optimize import minimize

from audit_dynamic_joint_bn_run import load_frozen
from audit_prefix_forecast_run import check, contained, load_json, sha

ROOT = Path(__file__).resolve().parents[2]


def projected_gradient(theta, gradient, bounds=(-8., 8.)):
    theta, gradient = np.asarray(theta, float), np.asarray(gradient, float)
    if theta.shape != gradient.shape or not np.isfinite(theta).all() or not np.isfinite(gradient).all():
        raise ValueError("finite, aligned parameters and gradients required")
    low, high = bounds
    if not low < high or np.any(theta < low) or np.any(theta > high):
        raise ValueError("parameters outside optimization bounds")
    blocked = ((theta <= low) & (gradient > 0)) | ((theta >= high) & (gradient < 0))
    return np.where(blocked, 0., gradient)


def fit_diagnostic(model, traces, labels, config):
    loss, gradient = model.loss_gradient(model.theta, traces, labels, True)
    projected = projected_gradient(model.theta, gradient, config["bounds"])
    if not np.isfinite(loss):
        raise ValueError("nonfinite fit objective")
    norm = float(np.max(np.abs(projected)))
    return {"fit_objective": float(loss), "gradient_inf": float(np.max(np.abs(gradient))),
            "projected_gradient_inf": norm,
            "stationarity_pass": norm <= config["stationarity_tolerance"],
            "optimizer": copy.deepcopy(model.optimizer)}


def continue_fit(model, fit_traces, fit_labels, config):
    """This function deliberately has NO validation/test inputs."""
    before = fit_diagnostic(model, fit_traces, fit_labels, config)
    selected = not before["optimizer"].get("success", False) or not before["stationarity_pass"]
    record = {"before": before, "selected": bool(selected), "accepted": False,
              "additional_fit_seconds": 0., "candidate": None}
    if selected:
        original = model.theta.copy()
        old_optimizer, old_max_iter = copy.deepcopy(model.optimizer), model.max_iter
        started = time.perf_counter()
        fitted = minimize(model.loss_gradient, original, args=(fit_traces, fit_labels, True), jac=True,
                          method="L-BFGS-B", bounds=[tuple(config["bounds"])] * len(original),
                          options={"maxiter": config["max_additional_iterations"],
                                   "ftol": config["ftol"], "gtol": config["gtol"]})
        record["additional_fit_seconds"] = time.perf_counter() - started
        model.theta = fitted.x
        model.max_iter = config["max_additional_iterations"]
        model.optimizer = {"success": bool(fitted.success), "iterations": int(fitted.nit),
                           "loss": float(fitted.fun), "message": str(fitted.message)}
        candidate = fit_diagnostic(model, fit_traces, fit_labels, config)
        record["candidate"] = {**candidate, "theta": model.theta.tolist()}
        record["accepted"] = candidate["fit_objective"] <= before["fit_objective"]
        if not record["accepted"]:
            model.theta, model.optimizer, model.max_iter = original, old_optimizer, old_max_iter
        record["after"] = candidate if record["accepted"] else before
    else:
        record["after"] = before
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_v1_20260919')
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / 'experiments/model_mechanisms/configs/joint_optimizer_continuation_v1.json')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_converged_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite continuation run")
    run, root, output = args.run.resolve(), args.root.resolve(), args.out.resolve()
    manifest, old_results = load_json(run / "manifest.json"), load_json(run / "metrics.json")
    config = load_json(args.config)
    check(old_results["complete"], "reference experiment must be complete")
    check(config["methods"] == ["joint_bn", "no_messages", "no_latent_head"], "joint-family inventory changed")
    check(config["bounds"] == [-8., 8.] and 0 < config["stationarity_tolerance"] < 1,
          "invalid fit diagnostic protocol")
    for relative, expected in manifest["input_sha256"].items():
        check(sha(contained(root, relative)) == expected, "input changed: " + relative)
    check(sha(contained(root, manifest["config"]["reference_manifest"])) == manifest["reference_manifest_sha256"],
          "reference split manifest changed")
    evaluator = load_frozen(run, manifest)  # verifies/imports the ORIGINAL model and feature code
    protocol = evaluator.protocol
    traces = protocol.read_jsonl(contained(root, manifest["config"]["data"]) / "traces_public.jsonl")
    label_rows = protocol.read_jsonl(contained(root, manifest["config"]["data"]) / "labels.jsonl")
    lookup = {r["episode_id"]: r["label"] for r in label_rows}
    check(len(lookup) == len(label_rows) and set(lookup) == {t["episode_id"] for t in traces}, "label IDs differ")
    labels = np.asarray([lookup[t["episode_id"]] for t in traces])
    check(np.isin(labels, [0, 1]).all(), "binary labels required")
    parts, groups = protocol.resolve_manifest_splits(manifest, traces)
    old_by_key = {(r["setting"], r["seed"]): r for r in old_results["runs"]}
    expected_keys = {(s["workflow"] + "__" + s["permission"], seed)
                     for s in manifest["config"]["settings"] for seed in parts}
    check(len(old_by_key) == len(old_results["runs"]) and set(old_by_key) == expected_keys, "run inventory mismatch")
    output.mkdir(parents=True)
    shutil.copytree(run / "source_snapshot", output / "source_snapshot")
    extra_sources = {}
    for source in [Path(__file__), args.config, Path(__file__).with_name("audit_dynamic_joint_bn_run.py"),
                   Path(__file__).with_name("audit_prefix_forecast_run.py")]:
        target = output / "continuation_source" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        extra_sources[target.relative_to(output).as_posix()] = sha(target)
    plan = {"config": config, "source_run": str(run), "source_manifest_sha256": sha(run / "manifest.json"),
            "source_metrics_sha256": sha(run / "metrics.json"), "source_sha256": extra_sources,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "All fit-only decisions precede recomputed held-out predictions. Reused development data."}
    protocol.save_json(output / "continuation_plan.json", plan)
    records, trained = [], {}
    # Stage 1: only fit features are constructed. No validation/test scoring here.
    for setting in manifest["config"]["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        for seed, selected in parts.items():
            directory, previous = output / key / str(seed), run / key / str(seed)
            directory.mkdir(parents=True)
            parameter_path = previous / "model_parameters.json"
            parameters = load_json(parameter_path)
            check(set(parameters) == set(manifest["methods"]), "method inventory differs")
            episodes = [evaluator.prepare_trace(traces[i], setting, manifest["config"]["max_joint_variables"])
                        for i in selected["fit"]]
            fit_traces = [e.joint for e in episodes]
            for name in manifest["methods"]:
                record = parameters[name]
                artifact = contained(previous, record["artifact"])
                check(sha(artifact) == record["sha256"], "source model changed")
                target = directory / record["artifact"]
                if name not in config["methods"]:
                    shutil.copy2(artifact, target)
                    continue
                model = joblib.load(artifact)  # own hash-verified artifacts only
                check(isinstance(model, evaluator.DynamicJointBN), "joint estimator class differs")
                check(model.use_messages == (name != "no_messages") and
                      model.use_latent_head == (name != "no_latent_head"), "ablation flags differ")
                before_parameters = model.parameters()
                check(before_parameters == record["bn"], "stored model parameters differ")
                report = continue_fit(model, fit_traces, labels[selected["fit"]], config)
                report.update(setting=key, seed=seed, method=name, original_model_sha256=record["sha256"],
                              original_parameters_sha256=sha(parameter_path),
                              fit_episode_ids=[traces[i]["episode_id"] for i in selected["fit"]])
                if report["accepted"]:
                    joblib.dump(model, target)
                    record.update(sha256=sha(target), bn=model.parameters())
                else:
                    shutil.copy2(artifact, target)
                report["result_model_sha256"] = sha(target)
                trained[(key, seed, name)] = report
                records.append(report)
                protocol.save_json(output / "training_progress.json", {"complete": False, "records": records})
                print(f"fit-only {key} {seed} {name}: selected={report['selected']} "
                      f"accepted={report['accepted']} pg={report['after']['projected_gradient_inf']:.3g}", flush=True)
            protocol.save_json(directory / "model_parameters.json", parameters)
    decisions = {"complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                 "plan_sha256": sha(output / "continuation_plan.json"), "records": records,
                 "checked_n": len(records), "selected_n": sum(r["selected"] for r in records),
                 "accepted_n": sum(r["accepted"] for r in records),
                 "stationarity_failed_n": sum(not r["after"]["stationarity_pass"] for r in records)}
    protocol.save_json(output / "training_decisions.json", decisions)
    decision_sha = sha(output / "training_decisions.json")
    new_manifest = copy.deepcopy(manifest)
    new_manifest["continuation"] = {"plan_sha256": sha(output / "continuation_plan.json"),
                                    "training_decisions_sha256": decision_sha,
                                    "first_stage_config_unchanged": True}
    protocol.save_json(output / "manifest.json", new_manifest)
    limitations = old_results["limitations"] + [
        "Joint-family continuation selected by fit-only stationarity/optimizer status; all decisions frozen before held-out scores.",
        "Base config records the first-stage budget; continuation_plan.json records the additional optimization protocol.",
    ]
    results = []
    evaluator.write_report(output, results, len(expected_keys), limitations)
    # Stage 2: models are fixed. Recalibrate thresholds and report ALL results.
    for setting in manifest["config"]["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        episodes = [evaluator.prepare_trace(t, setting, manifest["config"]["max_joint_variables"]) for t in traces]
        for seed, selected in parts.items():
            check(sha(output / "training_decisions.json") == decision_sha, "training decisions changed")
            directory, previous = output / key / str(seed), run / key / str(seed)
            result = copy.deepcopy(old_by_key[(key, seed)])
            parameters = load_json(directory / "model_parameters.json")
            rows = protocol.read_jsonl(previous / "predictions.jsonl")
            val = selected["validation"][labels[selected["validation"]] == 0]
            test = selected["test"]
            by_split = {part: [r for r in rows if r["split"] == part] for part in ("validation_negative", "test")}
            expected_ids = [(part, traces[i]["episode_id"]) for part, indices in (("validation_negative", val), ("test", test)) for i in indices]
            check([(r["split"], r["episode_id"]) for r in rows] == expected_ids, "source prediction IDs differ")
            for name in config["methods"]:
                report = trained[(key, seed, name)]
                if not report["accepted"]:
                    continue
                model = joblib.load(directory / parameters[name]["artifact"])
                predictions = {}
                for part, indices in (("validation_negative", val), ("test", test)):
                    predictions[part] = evaluator.predict(model, [episodes[i] for i in indices])
                    for row, scores in zip(by_split[part], predictions[part]):
                        row["prefix_scores"][name] = scores.tolist()
                threshold = protocol.choose_alarm_threshold([float(p.max()) for p in predictions["validation_negative"]],
                    manifest["config"]["threshold"]["validation_negative_any_alarm_fpr"])
                result["metrics"][name] = {"threshold_selection": threshold,
                    "fit_seconds": result["metrics"][name]["fit_seconds"] + report["additional_fit_seconds"],
                    "optimizer_success": model.optimizer["success"],
                    **protocol.evaluate_scores(labels[test], predictions["test"], [traces[i] for i in test], threshold["threshold"])}
            with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            protocol.save_json(directory / "metrics.json", result)
            results.append(result)
            evaluator.write_report(output, results, len(expected_keys), limitations)
    protocol.save_json(output / "training_progress.json", decisions)
    print(json.dumps({k: decisions[k] for k in ("checked_n", "selected_n", "accepted_n", "stationarity_failed_n")}), flush=True)


if __name__ == "__main__":
    main()
