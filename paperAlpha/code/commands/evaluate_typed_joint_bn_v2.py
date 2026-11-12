"""Frozen-split experiment of separate information/authorization latent states.

All six previous controls are preserved. Their saved predictions must have a
passing frozen replay audit. No phase-aware policy, changed labels or new data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
import evaluate_dynamic_joint_bn_v1 as common
from paperalpha_monitor.typed_joint_bn import TypedJointBN, type_trace, PARAMETER_NAMES, SPEC_SHA256
from paperalpha_monitor.streaming_bn import PRIVILEGED_TOOLS

protocol = common.protocol
METHOD = "typed_joint_bn"


def observability_report(traces):
    late_messages, early_permission, proposals, recovery = 0, 0, 0, 0
    for trace in traces:
        es = trace["events"]
        dispatch = next((i for i, e in enumerate(es) if e.get("event_type") == "tool_call" and e.get("tool") in PRIVILEGED_TOOLS), len(es))
        late_messages += int(any(e["event_type"] in {"message_sent", "delegation"} for e in es[dispatch + 1:]))
        early_permission += int(any("permission_mismatch" in e for e in es[:dispatch]))
        proposals += sum(e["event_type"] == "tool_proposed" for e in es)
        recovery += sum(e["event_type"] in {"review_completed", "sanitization_completed", "permission_revoked"} for e in es)
    return {"episodes": len(traces), "event_types": dict(Counter(e["event_type"] for t in traces for e in t["events"])),
        "episodes_with_message_after_privileged_dispatch": late_messages,
        "episodes_with_permission_flag_before_privileged_dispatch": early_permission,
        "proposal_events": proposals, "explicit_review_or_recovery_events": recovery,
        "interpretation": "Visible evidence coverage, not an information-theoretic prediction bound. Missing early permission flags do not exclude all statistical early signals."}


def write_report(output, runs, expected):
    protocol.save_json(output / "metrics.json", {"complete": len(runs) == expected, "expected_runs": expected, "runs": runs,
        "scope": "Reused synthetic development data; semantic repair is not validated by scenarios absent from this data."})
    lines = ["# Typed joint BN development experiment", "", f"Completed {len(runs)}/{expected} settings/seeds.", "",
        "Same prefixes/splits/threshold budget; all original controls retained. No phase-policy thresholds.", "",
        "| setting | seed | method | prefix Brier | terminal AUROC | any-alarm FPR | early recall | any-alarm recall |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        for name in (METHOD, "joint_bn", "no_messages", "histgb_flat_prefix"):
            m = run["metrics"][name]
            values = [m["prefix_brier_episode_equal"], m["terminal"]["auroc"], m["any_alarm_false_positive_rate"],
                      m["warning_proxies"]["first_privileged_dispatch"]["recall"], m["any_alarm_recall"]]
            lines.append(f"| {run['setting']} | {run['seed']} | {name} | " + " | ".join("NA" if x is None else f"{x:.4f}" for x in values) + " |")
    lines += ["", "All seven methods, all thresholds and optimization diagnostics are in metrics.json.",
        "Latent types are hypotheses, not supervised risk-type truths. Early recall uses dispatch, not independently labeled harm.",
        "A corrected non-transfer rule cannot be validated by this dataset's absence of communication after sensitive calls."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / 'experiments/model_mechanisms/configs/typed_joint_evaluation_v2.json')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/typed_joint_bn_v2_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite typed experiment")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = (ROOT / config["reference_run"]).resolve()
    reference, old_results, audit = [json.loads((source / name).read_text(encoding="utf-8")) for name in ("manifest.json", "metrics.json", "AUDIT.json")]
    if not old_results["complete"] or audit["status"] != "passed" or audit["run_manifest_sha256"] != protocol.digest(source / "manifest.json"):
        raise ValueError("source experiment needs a current, passing replay audit")
    if sorted(config["expected_seeds"]) != sorted(map(int, reference["splits"])):
        raise ValueError("seed inventory differs from source experiment")
    if reference["config"]["threshold"]["validation_negative_any_alarm_fpr"] != .10:
        raise ValueError("typed experiment requires the same 0.10 alarm budget as its reference controls")
    for relative, digest in reference["input_sha256"].items():
        if protocol.digest(ROOT / relative) != digest:
            raise ValueError("input changed")
    traces = protocol.read_jsonl(ROOT / reference["config"]["data"] / "traces_public.jsonl")
    labels = {r["episode_id"]: r["label"] for r in protocol.read_jsonl(ROOT / reference["config"]["data"] / "labels.jsonl")}
    splits, groups = protocol.resolve_manifest_splits(reference, traces)
    old_by_key = {(r["setting"], r["seed"]): r for r in old_results["runs"]}
    args.out.mkdir(parents=True)
    snapshots = common.snapshot_sources(args.out, args.config)
    target = args.out / "source_snapshot/code/commands" / Path(__file__).name
    shutil.copy2(Path(__file__), target)
    snapshots['code/commands/' + Path(__file__).name] = {"original": str(Path(__file__)), "sha256": protocol.digest(target)}
    manifest = {"config": config, "reference_run": str(source), "reference_manifest_sha256": protocol.digest(source / "manifest.json"),
        "reference_audit_sha256": protocol.digest(source / "AUDIT.json"), "reference_metrics_sha256": protocol.digest(source / "metrics.json"),
        "input_sha256": reference["input_sha256"], "splits": reference["splits"], "settings": reference["config"]["settings"],
        "source_snapshot": snapshots, "typed_parameter_names": PARAMETER_NAMES, "typed_specification_sha256": SPEC_SHA256,
        "methods": [METHOD, *reference["methods"]], "baseline_method_names": reference["methods"],
        "observability": observability_report(traces), "scope": config["scope"]}
    protocol.save_json(args.out / "manifest.json", manifest)
    runs, expected = [], len(manifest["settings"]) * len(splits)
    write_report(args.out, runs, expected)
    for setting in manifest["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        print("compiling typed " + key, flush=True)
        # Both models consume exactly the same event compiler and visible metadata.
        base = [common.prepare_trace(t, setting, reference["config"]["max_joint_variables"]).joint for t in traces]
        typed = [type_trace(t, config["max_joint_variables"]) for t in base]
        for seed, parts in splits.items():
            previous, directory = source / key / str(seed), args.out / key / str(seed)
            directory.mkdir(parents=True)
            parameters_path = previous / "model_parameters.json"
            if protocol.digest(parameters_path) != audit["output_file_sha256"][parameters_path.relative_to(source).as_posix()]:
                raise ValueError("source parameter file changed")
            old_parameters = json.loads(parameters_path.read_text(encoding="utf-8"))["joint_bn"]["bn"]
            if old_parameters["parameter_names"] != common.PARAMETER_NAMES or old_parameters["regularization"] != config["training"]["regularization"]:
                raise ValueError("base parameter layout/objective differs")
            warm = np.r_[old_parameters["theta"], 0., 0.]
            model = TypedJointBN(**config["training"])
            model.theta = warm.copy()
            fit = [typed[i] for i in parts["fit"]]
            ys = np.asarray([labels[traces[i]["episode_id"]] for i in parts["fit"]])
            initial_loss, _ = model.loss_gradient(model.theta, fit, ys)
            if not np.isclose(initial_loss, old_parameters["optimizer"]["loss"], atol=1e-12, rtol=1e-12):
                raise ValueError("warm-start fit objective differs from the reference on this temporal-pattern benchmark")
            print(f"fitting typed {key} seed={seed}", flush=True)
            started = time.perf_counter()
            model.fit(fit, ys)
            elapsed = time.perf_counter() - started
            loss, gradient = model.loss_gradient(model.theta, fit, ys)
            feasible = np.where(((model.theta <= -8.) & (gradient > 0)) | ((model.theta >= 8.) & (gradient < 0)), 0., gradient)
            training = {"fit_episode_ids": [traces[i]["episode_id"] for i in parts["fit"]],
                "warm_start_parameter_file_sha256": protocol.digest(parameters_path), "warm_start_theta": warm.tolist(),
                "initial_fit_objective": initial_loss, "final_fit_objective": loss,
                "projected_gradient_inf": float(np.max(np.abs(feasible))), "optimizer": model.optimizer,
                "joint_variable_histogram": dict(Counter(t.n_variables for t in fit)), "fit_seconds": elapsed}
            # Freeze model and fit diagnostics before any held-out probability evaluation.
            artifact = directory / (METHOD + ".joblib")
            joblib.dump(model, artifact)
            protocol.save_json(directory / "model_parameters.json", {METHOD: {"artifact": artifact.name,
                "sha256": protocol.digest(artifact), "bn": model.parameters()}})
            protocol.save_json(directory / "training.json", training)
            rows_path = previous / "predictions.jsonl"
            source_prediction_hash = protocol.digest(rows_path)
            if source_prediction_hash != audit["output_file_sha256"][rows_path.relative_to(source).as_posix()]:
                raise ValueError("source predictions differ from replay audit")
            rows = protocol.read_jsonl(rows_path)
            val = [i for i in parts["validation"] if labels[traces[i]["episode_id"]] == 0]
            test = parts["test"]
            expected_ids = [(p, traces[i]["episode_id"]) for p, indices in (("validation_negative", val), ("test", test)) for i in indices]
            if [(r["split"], r["episode_id"]) for r in rows] != expected_ids:
                raise ValueError("source prediction inventory differs")
            predictions, equivalence_error = {}, 0.
            collapsed = TypedJointBN(typed_head=False)
            collapsed.theta = warm.copy()
            for part, indices in (("validation_negative", val), ("test", test)):
                selected = [typed[i] for i in indices]
                predictions[part] = model.predict(selected)
                collapsed_scores = collapsed.predict(selected)
                target_rows = [r for r in rows if r["split"] == part]
                for row, scores, equal in zip(target_rows, predictions[part], collapsed_scores):
                    before = np.asarray(row["prefix_scores"]["joint_bn"])
                    equivalence_error = max(equivalence_error, float(np.max(np.abs(equal - before))))
                    row["prefix_scores"][METHOD] = scores.tolist()
            # This equality is expected only for this dataset's temporal pattern.
            if manifest["observability"]["episodes_with_message_after_privileged_dispatch"] == 0 and equivalence_error > 1e-12:
                raise ValueError("collapsed typed readout no longer matches the source model on this temporal pattern")
            threshold = protocol.choose_alarm_threshold([float(p.max()) for p in predictions["validation_negative"]], .10)
            result = copy.deepcopy(old_by_key[(key, seed)])
            result["metrics"][METHOD] = {"threshold_selection": threshold, "fit_seconds": elapsed,
                "optimizer_success": model.optimizer["success"],
                **protocol.evaluate_scores([labels[traces[i]["episode_id"]] for i in test], predictions["test"],
                                           [traces[i] for i in test], threshold["threshold"])}
            result["typed_training"] = training
            result["source_prediction_sha256"] = source_prediction_hash
            result["collapsed_head_vs_source_max_probability_error"] = equivalence_error
            with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            protocol.save_json(directory / "metrics.json", result)
            runs.append(result)
            write_report(args.out, runs, expected)
            print(f"finished typed {key} {seed}: brier={result['metrics'][METHOD]['prefix_brier_episode_equal']:.4f} "
                  f"converged={model.optimizer['success']} fit_s={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
