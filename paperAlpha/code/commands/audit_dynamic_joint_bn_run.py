"""Replay saved joint-model/control predictions from the frozen source tree.

No refitting. A partial audit covers completed runs only and cannot certify
the still-running experiment or scientific external validity.
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


def load_frozen(run, manifest):
    snapshot = run / "source_snapshot"
    check(not any(name == "paperalpha_monitor" or name.startswith("paperalpha_monitor.") for name in sys.modules),
          "auditor requires a fresh Python process")
    expected = {}
    for relative, record in manifest["source_snapshot"].items():
        check(sha(contained(snapshot, relative)) == record["sha256"], "frozen source changed: " + relative)
        expected[record["original"]] = record["sha256"]
    check(expected == manifest["code_sha256"], "code inventory differs")
    check(load_json(snapshot / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json') == manifest["config"], "config differs")
    script = snapshot / 'code/commands/evaluate_dynamic_joint_bn_v1.py'
    spec = importlib.util.spec_from_file_location("_frozen_joint_eval", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, imported in tuple(sys.modules.items()):
        if name == "paperalpha_monitor" or name.startswith("paperalpha_monitor.") or name == "evaluate_prefix_forecast_v1":
            path = Path(imported.__file__).resolve()
            check(path.is_relative_to(snapshot.resolve()), "non-frozen import: " + str(path))
            check(path.relative_to(snapshot.resolve()).as_posix() in manifest["source_snapshot"], "unrecorded dependency")
    check(module.FLAT_FEATURE_NAMES == manifest["flat_feature_names"], "flat features differ")
    check(module.PARAMETER_NAMES == manifest["joint_parameter_names"], "joint parameters differ")
    check(list(module.METHODS) == manifest["methods"], "method inventory differs")
    return module


def audit(run, root, allow_partial=False):
    manifest = load_json(run / "manifest.json")
    config = manifest["config"]
    for relative, expected in manifest["input_sha256"].items():
        check(sha(contained(root, relative)) == expected, "input changed: " + relative)
    reference_path = contained(root, config["reference_manifest"])
    check(sha(reference_path) == manifest["reference_manifest_sha256"], "reference manifest changed")
    reference = load_json(reference_path)
    check(manifest["splits"] == reference["splits"], "splits differ")
    check(config["settings"] == reference["config"]["settings"] and config["data"] == reference["config"]["data"], "setting/data mismatch")
    check(manifest["input_sha256"] == {k.replace("\\", "/"): v for k, v in reference["input_sha256"].items()}, "reference input hashes differ")
    evaluator = load_frozen(run, manifest)
    protocol = evaluator.protocol
    traces = protocol.read_jsonl(contained(root, config["data"]) / "traces_public.jsonl")
    label_rows = protocol.read_jsonl(contained(root, config["data"]) / "labels.jsonl")
    lookup = {row["episode_id"]: row["label"] for row in label_rows}
    check(len(lookup) == len(label_rows) and set(lookup) == {t["episode_id"] for t in traces}, "invalid label IDs")
    labels = np.asarray([lookup[t["episode_id"]] for t in traces])
    check(np.isin(labels, [0, 1]).all(), "nonbinary outcomes")
    for trace in traces:
        evaluator.reject_outcomes(trace)
    splits, groups = protocol.resolve_manifest_splits(manifest, traces)
    check(sorted(splits) == sorted(config["expected_seeds"]), "seed mismatch")
    results = load_json(run / "metrics.json")
    index = {(row["setting"], row["seed"]): row for row in results["runs"]}
    expected_keys = {(s["workflow"] + "__" + s["permission"], seed) for s in config["settings"] for seed in splits}
    check(len(index) == len(results["runs"]) and set(index) <= expected_keys, "invalid completed run inventory")
    check(results["expected_runs"] == len(expected_keys), "expected run count differs")
    check(results["complete"] == (set(index) == expected_keys), "completion field inconsistent")
    check(bool(index), "no completed runs to audit")
    check(allow_partial or results["complete"], "experiment incomplete; use --allow-partial only for explicitly scoped progress audits")
    method_n, prefix_n, episode_n, error = 0, 0, 0, 0.
    checked_files, summaries = {}, {}
    for setting in config["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        if not any((key, seed) in index for seed in splits):
            continue
        episodes = [evaluator.prepare_trace(t, setting, config["max_joint_variables"]) for t in traces]
        checked = []
        for seed, parts in splits.items():
            if (key, seed) not in index:
                continue
            directory = contained(run, Path(key) / str(seed))
            result = load_json(directory / "metrics.json")
            check(result == index[(key, seed)], "per-run metrics differ")
            training = json.loads(json.dumps(evaluator.training_diagnostics([episodes[i] for i in parts["fit"]])))
            compare(training, result["training"], "training metadata")
            parameters = load_json(directory / "model_parameters.json")
            check(set(parameters) == set(evaluator.METHODS) == set(result["metrics"]), "missing or unexpected methods")
            rows = protocol.read_jsonl(directory / "predictions.jsonl")
            test = parts["test"]
            val = parts["validation"][labels[parts["validation"]] == 0]
            expected_ids = [(part, traces[i]["episode_id"]) for part, selected in (("validation_negative", val), ("test", test)) for i in selected]
            check([(r["split"], r["episode_id"]) for r in rows] == expected_ids, "prediction IDs/order differ")
            by_part = {part: [r for r in rows if r["split"] == part] for part in ("validation_negative", "test")}
            for part, selected in (("validation_negative", val), ("test", test)):
                for row, i in zip(by_part[part], selected):
                    check(row["label"] == labels[i] and row["group"] == groups[i], "prediction label/group differs")
                    check(row["sequence"] == [s.sequence for s in episodes[i].joint.steps], "event sequence differs")
                    check(set(row["prefix_scores"]) == set(evaluator.METHODS), "prefix method inventory differs")
            for name, record in parameters.items():
                artifact = contained(directory, record["artifact"])
                check(sha(artifact) == record["sha256"], "model artifact changed")
                checked_files[artifact.relative_to(run).as_posix()] = record["sha256"]
                model = joblib.load(artifact)  # Own hash-verified output, never third-party pickle.
                is_joint = isinstance(model, evaluator.DynamicJointBN)
                is_template = isinstance(model, evaluator.TemplateBN)
                check(is_joint == (name in {"joint_bn", "no_messages", "no_latent_head"}), "joint estimator class mismatch")
                check(is_template == (name == "template_bn_prefix"), "template estimator class mismatch")
                if is_joint or is_template:
                    compare(model.parameters(), record["bn"], "saved BN parameters")
                else:
                    classifier = model.named_steps["logisticregression"] if name == "logistic_flat_prefix" else model
                    for k, v in config["baselines"][name].items():
                        check(classifier.get_params()[k] == v, "baseline hyperparameter differs")
                if is_joint:
                    check(model.use_messages == (name != "no_messages"), "message ablation mismatch")
                    check(model.use_latent_head == (name != "no_latent_head"), "latent ablation mismatch")
                actual = {}
                for part, selected in (("validation_negative", val), ("test", test)):
                    actual[part] = evaluator.predict(model, [episodes[i] for i in selected])
                    for recomputed, row in zip(actual[part], by_part[part]):
                        stored = np.asarray(row["prefix_scores"][name])
                        check(recomputed.shape == stored.shape, "prefix score shape differs")
                        np.testing.assert_allclose(recomputed, stored, atol=1e-12, rtol=1e-12)
                        error = max(error, float(np.max(np.abs(recomputed - stored))))
                        prefix_n += len(recomputed)
                        episode_n += 1
                maxima = [float(p.max()) for p in actual["validation_negative"]]
                alpha = config["threshold"]["validation_negative_any_alarm_fpr"]
                threshold = protocol.choose_alarm_threshold(maxima, alpha)
                check(threshold["threshold"] == independent_threshold(maxima, alpha), "independent threshold calculation differs")
                recalculated = {"threshold_selection": threshold, "fit_seconds": result["metrics"][name]["fit_seconds"],
                                **protocol.evaluate_scores(labels[test], actual["test"], [traces[i] for i in test], threshold["threshold"])}
                if is_joint or is_template:
                    recalculated["optimizer_success"] = model.optimizer["success"]
                compare(recalculated, result["metrics"][name], key + "/" + str(seed) + "/" + name)
                method_n += 1
            for filename in ("metrics.json", "predictions.jsonl", "model_parameters.json"):
                path = directory / filename
                checked_files[path.relative_to(run).as_posix()] = sha(path)
            checked.append(result)
            print(f"verified {key} seed={seed}", flush=True)
        summaries[key] = {name: {
            "completed_seeds": len(checked),
            "prefix_brier_mean": float(np.mean([r["metrics"][name]["prefix_brier_episode_equal"] for r in checked])),
            "any_alarm_fpr_mean": float(np.mean([r["metrics"][name]["any_alarm_false_positive_rate"] for r in checked])),
            "pre_privileged_recall_mean": float(np.mean([r["metrics"][name]["warning_proxies"]["first_privileged_dispatch"]["recall"] for r in checked])),
        } for name in evaluator.METHODS}
    return {"status": "passed" if results["complete"] else "partial_passed",
            "scope": "frozen model replay for the listed completed runs only; no retraining or external validity proof",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_manifest_sha256": sha(run / "manifest.json"), "audit_script_sha256": sha(Path(__file__)),
            "completed_run_n": len(index), "expected_run_n": len(expected_keys),
            "checked_estimators": method_n, "checked_method_prefix_scores": prefix_n,
            "checked_method_episode_predictions": episode_n, "max_absolute_score_error": error,
            "output_file_sha256": checked_files, "completed_seed_summaries": summaries,
            "limitations": ["Replaying frozen outputs does not independently establish fitting-data lineage.",
                            "Fit duration is recorded metadata, not independently replayed timing.",
                            "Pre-dispatch recall is a proxy; there is no independently labeled harm timestamp.",
                            "Thresholds constrain empirical validation false alarms, not population safety."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_v1_20260919')
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = args.out or args.run / "AUDIT.json"
    if output.exists():
        raise FileExistsError(f"Audit already exists: {output}; supply another --out")
    result = audit(args.run.resolve(), args.root.resolve(), args.allow_partial)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "completed_run_n", "checked_estimators", "checked_method_prefix_scores", "max_absolute_score_error")}, indent=2))


if __name__ == "__main__":
    main()
