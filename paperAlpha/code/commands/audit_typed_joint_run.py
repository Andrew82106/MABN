"""Frozen replay and fit-objective audit of the typed BN experiment.

Original control columns are checked byte-for-value against their previously
audited predictions; only the new typed estimator is re-executed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np

from audit_prefix_forecast_run import check, compare, contained, independent_threshold, load_json, sha

ROOT = Path(__file__).resolve().parents[2]


def load_evaluator(run, manifest):
    check(not any(n == "paperalpha_monitor" or n.startswith("paperalpha_monitor.") for n in sys.modules), "fresh audit process required")
    snapshot = run / "source_snapshot"
    for relative, record in manifest["source_snapshot"].items():
        check(sha(contained(snapshot, relative)) == record["sha256"], "frozen source changed")
    check(load_json(snapshot / 'experiments/model_mechanisms/configs/typed_joint_evaluation_v2.json') == manifest["config"], "frozen config differs")
    script = snapshot / 'code/commands/evaluate_typed_joint_bn_v2.py'
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("_frozen_typed_evaluation", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, imported in tuple(sys.modules.items()):
        if name.startswith("paperalpha_monitor") or name in {"evaluate_dynamic_joint_bn_v1", "evaluate_prefix_forecast_v1"}:
            path = Path(imported.__file__).resolve()
            check(path.is_relative_to(snapshot.resolve()) and path.relative_to(snapshot.resolve()).as_posix() in manifest["source_snapshot"], "unfrozen dependency")
    check(module.PARAMETER_NAMES == manifest["typed_parameter_names"] and module.SPEC_SHA256 == manifest["typed_specification_sha256"], "typed schema differs")
    return module


def audit(run, root, allow_partial=False):
    manifest, results = load_json(run / "manifest.json"), load_json(run / "metrics.json")
    evaluator = load_evaluator(run, manifest)
    protocol = evaluator.protocol
    source = Path(manifest["reference_run"])
    for filename, field in (("manifest.json", "reference_manifest_sha256"), ("metrics.json", "reference_metrics_sha256"), ("AUDIT.json", "reference_audit_sha256")):
        check(sha(source / filename) == manifest[field], "reference changed")
    reference, source_audit = load_json(source / "manifest.json"), load_json(source / "AUDIT.json")
    check(source_audit["status"] == "passed" and source_audit["run_manifest_sha256"] == manifest["reference_manifest_sha256"], "reference audit stale")
    check(manifest["splits"] == reference["splits"] and manifest["settings"] == reference["config"]["settings"] and
          manifest["input_sha256"] == reference["input_sha256"], "source protocol changed")
    check(reference["config"]["threshold"]["validation_negative_any_alarm_fpr"] == .10, "unequal source/new alarm budgets")
    for relative, digest in manifest["input_sha256"].items():
        check(sha(contained(root, relative)) == digest, "input changed")
    traces = protocol.read_jsonl(contained(root, reference["config"]["data"]) / "traces_public.jsonl")
    labels = {r["episode_id"]: r["label"] for r in protocol.read_jsonl(contained(root, reference["config"]["data"]) / "labels.jsonl")}
    splits, groups = protocol.resolve_manifest_splits(manifest, traces)
    compare(evaluator.observability_report(traces), manifest["observability"], "visible evidence inventory")
    by_key = {(r["setting"], r["seed"]): r for r in results["runs"]}
    expected = {(s["workflow"] + "__" + s["permission"], seed) for s in manifest["settings"] for seed in splits}
    check(len(by_key) == len(results["runs"]) and bool(by_key) and set(by_key) <= expected, "completed run inventory differs")
    check(results["expected_runs"] == len(expected) and results["complete"] == (set(by_key) == expected), "completion state differs")
    check(allow_partial or results["complete"], "experiment incomplete; use --allow-partial for scoped audit")
    baseline_results = {(r["setting"], r["seed"]): r for r in load_json(source / "metrics.json")["runs"]}
    checked_prefixes, inherited_rows, max_error, files = 0, 0, 0., {}
    for setting in manifest["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        if not any(k == key for k, _ in by_key):
            continue
        base = [evaluator.common.prepare_trace(t, setting, reference["config"]["max_joint_variables"]).joint for t in traces]
        typed = [evaluator.type_trace(t, manifest["config"]["max_joint_variables"]) for t in base]
        for seed, parts in splits.items():
            if (key, seed) not in by_key:
                continue
            directory, previous = run / key / str(seed), source / key / str(seed)
            result = load_json(directory / "metrics.json")
            compare(result, by_key[(key, seed)], "per-run metrics")
            rows, old_rows = [protocol.read_jsonl(p / "predictions.jsonl") for p in (directory, previous)]
            old_hash = sha(previous / "predictions.jsonl")
            check(old_hash == result["source_prediction_sha256"] == source_audit["output_file_sha256"][(previous / "predictions.jsonl").relative_to(source).as_posix()], "source predictions changed")
            check(len(rows) == len(old_rows), "prediction row count differs")
            for current, old in zip(rows, old_rows):
                check(set(current["prefix_scores"]) == set(manifest["methods"]), "method inventory differs")
                copied = {**current, "prefix_scores": {name: current["prefix_scores"][name] for name in manifest["baseline_method_names"]}}
                compare(copied, old, "inherited baseline row")
                inherited_rows += len(manifest["baseline_method_names"])
            for name in manifest["baseline_method_names"]:
                compare(result["metrics"][name], baseline_results[(key, seed)]["metrics"][name], "inherited baseline metrics")
            parameters = load_json(directory / "model_parameters.json")[evaluator.METHOD]
            artifact = contained(directory, parameters["artifact"])
            check(sha(artifact) == parameters["sha256"], "typed model changed")
            model = joblib.load(artifact)
            check(isinstance(model, evaluator.TypedJointBN) and model.typed_head and model.use_messages and model.use_latent_head, "typed architecture differs")
            compare(model.parameters(), parameters["bn"], "typed parameters")
            training = load_json(directory / "training.json")
            compare(training, result["typed_training"], "typed training record")
            old_parameter_path = previous / "model_parameters.json"
            check(sha(old_parameter_path) == training["warm_start_parameter_file_sha256"] == source_audit["output_file_sha256"][old_parameter_path.relative_to(source).as_posix()], "warm-start lineage differs")
            old_parameters = load_json(old_parameter_path)["joint_bn"]["bn"]
            warm = np.r_[old_parameters["theta"], 0., 0.]
            np.testing.assert_array_equal(warm, training["warm_start_theta"])
            check(training["fit_episode_ids"] == [traces[i]["episode_id"] for i in parts["fit"]], "fit IDs differ")
            fit, ys = [typed[i] for i in parts["fit"]], np.asarray([labels[traces[i]["episode_id"]] for i in parts["fit"]])
            initial_loss, _ = model.loss_gradient(warm, fit, ys)
            final_loss, gradient = model.loss_gradient(model.theta, fit, ys)
            feasible = np.where(((model.theta <= -8.) & (gradient > 0)) | ((model.theta >= 8.) & (gradient < 0)), 0., gradient)
            compare(initial_loss, training["initial_fit_objective"], "initial loss")
            compare(initial_loss, old_parameters["optimizer"]["loss"], "warm-start vs original fit objective")
            compare(final_loss, training["final_fit_objective"], "final loss")
            compare(float(np.max(np.abs(feasible))), training["projected_gradient_inf"], "final gradient")
            validation = [i for i in parts["validation"] if labels[traces[i]["episode_id"]] == 0]
            scores, equivalence_error = {}, 0.
            collapsed = evaluator.TypedJointBN(typed_head=False)
            collapsed.theta = warm.copy()
            for part, indices in (("validation_negative", validation), ("test", parts["test"])):
                selected = [r for r in rows if r["split"] == part]
                check([r["episode_id"] for r in selected] == [traces[i]["episode_id"] for i in indices], "prediction split IDs differ")
                scores[part] = model.predict([typed[i] for i in indices])
                initial_scores = collapsed.predict([typed[i] for i in indices])
                for row, prediction, same in zip(selected, scores[part], initial_scores):
                    saved = np.asarray(row["prefix_scores"][evaluator.METHOD])
                    np.testing.assert_allclose(prediction, saved, atol=1e-12, rtol=1e-12)
                    max_error = max(max_error, float(np.max(np.abs(prediction - saved))))
                    equivalence_error = max(equivalence_error, float(np.max(np.abs(same - row["prefix_scores"]["joint_bn"]))))
                    checked_prefixes += len(prediction)
            compare(equivalence_error, result["collapsed_head_vs_source_max_probability_error"], "collapsed-head equivalence")
            maxima = [float(ps.max()) for ps in scores["validation_negative"]]
            threshold = protocol.choose_alarm_threshold(maxima, .1)
            check(threshold["threshold"] == independent_threshold(maxima, .1), "independent threshold differs")
            metric = {"threshold_selection": threshold, "fit_seconds": training["fit_seconds"],
                "optimizer_success": model.optimizer["success"], **protocol.evaluate_scores(
                    [labels[traces[i]["episode_id"]] for i in parts["test"]], scores["test"], [traces[i] for i in parts["test"]], threshold["threshold"])}
            compare(metric, result["metrics"][evaluator.METHOD], "typed metrics")
            for filename in ("metrics.json", "training.json", "model_parameters.json", "predictions.jsonl", parameters["artifact"]):
                path = directory / filename
                files[path.relative_to(run).as_posix()] = sha(path)
            print(f"typed audit {key} {seed}", flush=True)
    return {"status": "passed" if results["complete"] else "partial_passed", "completed_runs": len(by_key),
        "expected_runs": len(expected), "checked_typed_prefix_scores": checked_prefixes,
        "checked_inherited_method_episode_rows": inherited_rows, "max_probability_difference": max_error,
        "run_manifest_sha256": sha(run / "manifest.json"), "output_file_sha256": files,
        "scope": "Frozen typed-model predictions/fit objectives and unchanged previously audited baseline columns; not independent external or risk-type validation."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/typed_joint_bn_v2_20260919')
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = args.out or args.run / "AUDIT.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite typed audit")
    result = audit(args.run.resolve(), args.root.resolve(), args.allow_partial)
    result.update(audited_at_utc=datetime.now(timezone.utc).isoformat(), audit_script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "output_file_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
