"""Recompute a prefix experiment using its frozen package and saved estimators.

No model is refitted. Successful replay verifies artifact consistency, not
independent scientific validity, calibrated probabilities, or safety.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[2]
METHODS = {"terminal_bn", "prefix_bn", "logistic_prefix", "histgb_prefix"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    check(path.is_relative_to(Path(root).resolve()), f"path escapes intended directory: {relative}")
    return path


def compare(actual, expected, location):
    """Compare all metric fields, preserving exact keys, None and counts."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), "keys differ: " + location)
        for name in expected:
            compare(actual[name], expected[name], location + "/" + name)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), "list differs: " + location)
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, location + "/" + str(i))
    elif isinstance(expected, float):
        check(np.isclose(actual, expected, atol=1e-12, rtol=1e-12), "numeric mismatch: " + location)
    else:
        check(actual == expected, "value mismatch: " + location)


def load_frozen_evaluator(run, manifest):
    snapshot = run / "source_snapshot"
    check(not any(name == "paperalpha_monitor" or name.startswith("paperalpha_monitor.") for name in sys.modules),
          "a monitor package is already imported; run this auditor in a fresh Python process")
    expected_code = {}
    for relative, record in manifest["source_snapshot"].items():
        path = contained(snapshot, relative)
        check(sha(path) == record["sha256"], "frozen source hash mismatch: " + relative)
        expected_code[record["original"]] = record["sha256"]
    check(expected_code == manifest["code_sha256"], "source snapshot and code hash inventories differ")
    config_path = snapshot / 'experiments/runtime_alarm/configs/prefix_forecast_v1.json'
    check(load_json(config_path) == manifest["config"], "frozen config differs from manifest")
    script = snapshot / 'code/commands/evaluate_prefix_forecast_v1.py'
    spec = importlib.util.spec_from_file_location("_frozen_prefix_forecast_v1", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, imported in tuple(sys.modules.items()):
        if name == "paperalpha_monitor" or name.startswith("paperalpha_monitor."):
            path = Path(imported.__file__).resolve()
            check(path.is_relative_to(snapshot.resolve()), "loaded non-frozen monitor module: " + str(path))
            relative = path.relative_to(snapshot.resolve()).as_posix()
            check(relative in manifest["source_snapshot"], "loaded unrecorded source: " + relative)
    check(module.CELL_NAMES == manifest["feature_names"], "frozen feature layout differs")
    return module


def independent_threshold(maxima, alpha):
    """Order-statistic check independent of the evaluator's candidate loop."""
    ordered = np.sort(np.asarray(maxima))
    check(ordered.size > 0, "validation-negative scores missing")
    allowed = int(np.floor(alpha * len(ordered)))
    if allowed >= len(ordered):
        return 0.0
    return float(np.nextafter(ordered[len(ordered) - allowed - 1], np.inf))


def audit(run, root):
    manifest_file = run / "manifest.json"
    manifest = load_json(manifest_file)
    config = manifest["config"]
    for relative, expected in manifest["input_sha256"].items():
        check(sha(contained(root, relative)) == expected, "input hash mismatch: " + relative)
    reference_path = contained(root, config["reference_manifest"])
    check(sha(reference_path) == manifest["reference_manifest_sha256"], "source manifest hash mismatch")
    reference = load_json(reference_path)
    check(manifest["splits"] == reference["splits"], "split IDs/metadata differ from source manifest")
    check(config["settings"] == reference["config"]["settings"], "settings differ from source manifest")
    check(config["data"] == reference["config"]["data"], "data path differs from source manifest")
    check(sorted(map(int, manifest["splits"])) == sorted(config["expected_seeds"]), "seed set differs")
    check({k.replace("\\", "/"): v for k, v in manifest["input_sha256"].items()} ==
          {k.replace("\\", "/"): v for k, v in reference["input_sha256"].items()}, "source data hashes differ")
    evaluator = load_frozen_evaluator(run, manifest)
    data = contained(root, config["data"])
    traces = evaluator.read_jsonl(data / "traces_public.jsonl")
    label_rows = evaluator.read_jsonl(data / "labels.jsonl")
    labels_by_id = {row["episode_id"]: row["label"] for row in label_rows}
    check(len(labels_by_id) == len(label_rows) and set(labels_by_id) == {t["episode_id"] for t in traces}, "duplicate or missing label IDs")
    labels = np.asarray([labels_by_id[t["episode_id"]] for t in traces])
    check(np.isin(labels, [0, 1]).all(), "nonbinary labels")
    for trace in traces:
        evaluator.reject_outcomes(trace)
    splits, groups = evaluator.resolve_manifest_splits(manifest, traces)
    root_results = load_json(run / "metrics.json")["runs"]
    run_index = {(row["setting"], row["seed"]): row for row in root_results}
    expected_runs = {(s["workflow"] + "__" + s["permission"], seed) for s in config["settings"] for seed in splits}
    check(len(run_index) == len(root_results) and set(run_index) == expected_runs, "missing, duplicate or unexpected runs")
    checked_episodes, checked_prefixes, max_error = 0, 0, 0.0
    checked_models, summaries, file_hashes = 0, {}, {}
    for setting in config["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        trajectories = [evaluator.replay(t, workflow=t["workflow_dag"] if setting["workflow"] == "workflow_known" else None,
                                        permission_mode=setting["permission"]) for t in traces]
        check(all(trajectories), "empty trajectories")
        setting_results = []
        for seed, parts in splits.items():
            directory = contained(run, Path(key) / str(seed))
            result = load_json(directory / "metrics.json")
            check(result == run_index[(key, seed)], "per-run and combined metrics differ")
            params = load_json(directory / "model_parameters.json")
            check(set(params) == set(result["metrics"]) == METHODS, "method inventory differs")
            rows = evaluator.read_jsonl(directory / "predictions.jsonl")
            test = parts["test"]
            val = parts["validation"][labels[parts["validation"]] == 0]
            expected_ids = [(part, traces[i]["episode_id"]) for part, indices in (("validation_negative", val), ("test", test)) for i in indices]
            check([(row["split"], row["episode_id"]) for row in rows] == expected_ids, "saved prediction IDs/order differ")
            rows_by_part = {part: [row for row in rows if row["split"] == part] for part in ("validation_negative", "test")}
            for part, indices in (("validation_negative", val), ("test", test)):
                for row, i in zip(rows_by_part[part], indices):
                    check(row["label"] == labels[i] and row["group"] == groups[i], "saved label/group differs")
                    check(row["sequence"] == [snapshot.sequence for snapshot in trajectories[i]], "saved prefix sequences differ")
                    check(set(row["prefix_scores"]) == METHODS, "prediction method inventory differs")
            training = result["training"]
            fit_lengths = [len(trajectories[i]) for i in parts["fit"]]
            compare(training, {"fit_episode_n": len(fit_lengths), "fit_prefix_n": sum(fit_lengths),
                               "prefix_weight_sum": float(len(fit_lengths)), "per_episode_weight": 1.0}, "training metadata")
            for method, parameter in params.items():
                artifact = contained(directory, parameter["artifact"])
                check(sha(artifact) == parameter["sha256"], "estimator hash mismatch: " + str(artifact))
                file_hashes[artifact.relative_to(run).as_posix()] = parameter["sha256"]
                model = joblib.load(artifact)  # Own saved artifact, hash checked; frozen package already selected.
                is_bn = isinstance(model, evaluator.TemplateBN)
                check(is_bn == (method in {"terminal_bn", "prefix_bn"}), "estimator class mismatch")
                if is_bn:
                    compare(model.parameters(), parameter["bn"], key + "/" + method + "/parameters")
                else:
                    estimator = model.named_steps["logisticregression"] if method == "logistic_prefix" else model
                    for name, value in config["baselines"][method].items():
                        check(estimator.get_params()[name] == value, "baseline hyperparameter differs: " + method + "/" + name)
                actual = {}
                for part, indices in (("validation_negative", val), ("test", test)):
                    probabilities = evaluator.predict_trajectories(model, [trajectories[i] for i in indices])
                    for recomputed, row in zip(probabilities, rows_by_part[part]):
                        stored = np.asarray(row["prefix_scores"][method])
                        check(recomputed.shape == stored.shape, "prefix array length differs")
                        np.testing.assert_allclose(recomputed, stored, atol=1e-12, rtol=1e-12)
                        max_error = max(max_error, float(np.max(np.abs(recomputed - stored))))
                        checked_episodes += 1
                        checked_prefixes += len(recomputed)
                    actual[part] = probabilities
                maxima = [float(p.max()) for p in actual["validation_negative"]]
                alpha = config["alarm_threshold"]["validation_negative_any_alarm_fpr"]
                selected = evaluator.choose_alarm_threshold(maxima, alpha)
                check(selected["threshold"] == independent_threshold(maxima, alpha), "independent quantile threshold differs")
                check(selected["threshold"] == result["metrics"][method]["threshold"], "stored threshold differs from validation recalculation")
                recalculated = {"threshold_selection": selected,
                                **evaluator.evaluate_scores(labels[test], actual["test"], [traces[i] for i in test], selected["threshold"])}
                if is_bn:
                    recalculated["optimizer_success"] = model.optimizer["success"]
                compare(recalculated, result["metrics"][method], f"{key}/{seed}/{method}/metrics")
                checked_models += 1
            for filename in ("predictions.jsonl", "metrics.json", "model_parameters.json"):
                path = directory / filename
                file_hashes[path.relative_to(run).as_posix()] = sha(path)
            setting_results.append(result)
            print(f"verified {key} seed={seed}", flush=True)
        summaries[key] = {method: {
            "prefix_brier_episode_equal_mean": float(np.mean([r["metrics"][method]["prefix_brier_episode_equal"] for r in setting_results])),
            "any_alarm_fpr_mean": float(np.mean([r["metrics"][method]["any_alarm_false_positive_rate"] for r in setting_results])),
            "pre_privileged_dispatch_recall_mean": float(np.mean([r["metrics"][method]["warning_proxies"]["first_privileged_dispatch"]["recall"] for r in setting_results])),
        } for method in sorted(METHODS)}
    runtime_versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    return {
        "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen-estimator score, threshold and metric recomputation; no refitting or independent external validity test",
        "run_manifest_sha256": sha(manifest_file), "audit_script_sha256": sha(Path(__file__)),
        "frozen_evaluator": str(Path(evaluator.__file__).resolve()),
        "frozen_core": str(Path(sys.modules["paperalpha_monitor.streaming_bn"].__file__).resolve()),
        "runtime_versions": runtime_versions, "original_versions": manifest["versions"],
        "versions_match": runtime_versions == manifest["versions"],
        "runs": len(expected_runs), "checked_estimators": checked_models,
        "checked_method_episode_predictions": checked_episodes, "checked_method_prefix_scores": checked_prefixes,
        "max_absolute_score_error": max_error, "output_file_sha256": file_hashes,
        "summaries_equal_seed_mean": summaries,
        "limitations": ["Existing synthetic data and correlated holdouts remain development diagnostics.",
                        "Estimator artifacts were replayed, not retrained; fitting-data lineage is not independently proven.",
                        "Metrics use the frozen evaluator; exact replay cannot establish correctness of every modeling assumption.",
                        "The threshold bound is empirical validation FPR only; no population safety guarantee.",
                        "No independent harm timestamp; pre-dispatch recall is not measured harm prevention."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/runtime_alarm/runs/prefix_forecast_v1_20260919')
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.out or run / "AUDIT.json"
    if output.exists():
        raise FileExistsError(f"Audit already exists; supply a different --out: {output}")
    result = audit(run, args.root.resolve())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "runs", "checked_estimators", "checked_method_prefix_scores", "max_absolute_score_error", "summaries_equal_seed_mean")}, indent=2))


if __name__ == "__main__":
    main()
