"""Frozen-split comparison of the actual joint two-layer model and controls.

All evidence is prefix-only. These reused synthetic traces are development
material, not independent confirmation or published MAS baseline reproduction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
sys.path.insert(0, str(ROOT / 'code/src'))
import evaluate_prefix_forecast_v1 as protocol  # noqa: E402
from paperalpha_monitor.dynamic_joint_bn import (  # noqa: E402
    DynamicJointBN, JointTrace, OBSERVABLES, PARAMETER_NAMES, compile_trace,
)
from paperalpha_monitor.streaming_bn import CELL_NAMES, TemplateBN, reject_outcomes, replay  # noqa: E402

METHODS = ("joint_bn", "no_messages", "no_latent_head", "template_bn_prefix",
           "logistic_flat_prefix", "histgb_flat_prefix")
FLAT_FEATURE_NAMES = ["observable:" + name for name in OBSERVABLES] + ["template:" + name for name in CELL_NAMES]


@dataclass(frozen=True)
class EpisodeFeatures:
    joint: JointTrace
    templates: tuple
    flat: np.ndarray


def prepare_trace(trace, setting, max_variables=10):
    if setting["workflow"] not in {"runtime_only", "workflow_known"}:
        raise ValueError("unsupported workflow setting")
    workflow = trace["workflow_dag"] if setting["workflow"] == "workflow_known" else None
    # The flat controls below assume the current role-free benchmark. A future
    # role/capability experiment needs an explicit matched feature contract.
    if workflow is not None and any(workflow.get(k) for k in ("roles", "agent_to_node")):
        raise ValueError("role/group metadata requires a separately specified matched baseline")
    options = {"workflow": workflow, "permission_mode": setting["permission"]}
    joint = compile_trace(trace, max_variables=max_variables, **options)
    templates = tuple(replay(trace, **options))
    if len(templates) != len(joint.steps) or any(a.sequence != b.sequence for a, b in zip(templates, joint.steps)):
        raise ValueError("joint and baseline prefix sequences differ")
    flat = np.asarray([np.r_[step.observables, template.vector()]
                       for step, template in zip(joint.steps, templates)])
    if flat.shape != (len(joint.steps), len(FLAT_FEATURE_NAMES)) or not np.isfinite(flat).all():
        raise ValueError("invalid flat feature layout")
    return EpisodeFeatures(joint, templates, flat)


def fit_estimators(episodes, labels, config, seed):
    """Generator accepts ONLY fit episodes/labels; save each result promptly."""
    snapshots, targets, weights = protocol.flatten_training([e.templates for e in episodes], labels)
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2:
        raise ValueError("fit episodes require both outcome classes")
    joint_traces = [e.joint for e in episodes]
    for name, kwargs in (("joint_bn", {}), ("no_messages", {"use_messages": False}),
                         ("no_latent_head", {"use_latent_head": False})):
        print("fitting " + name, flush=True)
        started = time.perf_counter()
        model = DynamicJointBN(**config["training"], **kwargs).fit(joint_traces, labels)
        yield name, model, time.perf_counter() - started
    started = time.perf_counter()
    model = TemplateBN(**config["baselines"]["template_bn_prefix"]).fit(snapshots, targets, sample_weight=weights)
    yield "template_bn_prefix", model, time.perf_counter() - started
    x = np.concatenate([e.flat for e in episodes])
    started = time.perf_counter()
    model = make_pipeline(StandardScaler(), LogisticRegression(**config["baselines"]["logistic_flat_prefix"], random_state=seed))
    model.fit(x, targets, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    yield "logistic_flat_prefix", model, time.perf_counter() - started
    started = time.perf_counter()
    model = HistGradientBoostingClassifier(**config["baselines"]["histgb_flat_prefix"], random_state=seed)
    model.fit(x, targets, sample_weight=weights)
    yield "histgb_flat_prefix", model, time.perf_counter() - started


def predict(model, episodes):
    if not episodes:
        raise ValueError("nonempty prediction episode list required")
    if isinstance(model, DynamicJointBN):
        result = model.predict([e.joint for e in episodes])
    elif isinstance(model, TemplateBN):
        result = protocol.predict_trajectories(model, [e.templates for e in episodes])
    else:
        probabilities = model.predict_proba(np.concatenate([e.flat for e in episodes]))[:, 1]
        result = list(np.split(probabilities, np.cumsum([len(e.flat) for e in episodes])[:-1]))
    for scores, episode in zip(result, episodes):
        if len(scores) != len(episode.joint.steps) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("invalid per-prefix probability output")
    return result


def training_diagnostics(episodes):
    return {
        "fit_episode_n": len(episodes), "fit_prefix_n": sum(len(e.joint.steps) for e in episodes),
        "prefix_weight_sum": float(len(episodes)), "per_episode_weight": 1.0,
        "joint_variable_histogram": dict(sorted(Counter(e.joint.n_variables for e in episodes).items())),
        "declared_node_roles": dict(Counter(v["role"] for e in episodes for v in e.joint.variables if v["kind"] == "workflow_node")),
        "operation_parameter_occurrences": dict(Counter(PARAMETER_NAMES[op.parameter] for e in episodes for s in e.joint.steps for op in s.operations)),
    }


def snapshot_sources(output, config_path):
    snapshots = protocol.snapshot_sources(output, config_path)
    path = Path(__file__).resolve()
    relative = path.relative_to(ROOT.resolve())
    destination = output / "source_snapshot" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    snapshots[relative.as_posix()] = {"original": str(path), "sha256": protocol.digest(destination)}
    return snapshots


def write_report(output, runs, expected_runs, limitations):
    protocol.save_json(output / "metrics.json", {"complete": len(runs) == expected_runs,
                                                 "expected_runs": expected_runs, "runs": runs, "limitations": limitations})
    lines = ["# Joint two-layer BN: frozen-split development comparison", "",
             f"Completed settings/seeds: {len(runs)}/{expected_runs}. No published MAS method reproduction or real-world superiority claim.", "",
             "All methods train on the same fit episodes with equal total weight per episode. Alarm thresholds use validation negatives only.", "",
             "| setting | seed | method | prefix Brier ↓ | terminal AUROC ↑ | terminal F1 | any-alarm FPR | pre-privileged recall | fit seconds | converged |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    def show(value):
        return "NA" if value is None else f"{value:.4f}"
    for run in runs:
        for name, metric in run["metrics"].items():
            values = [metric["prefix_brier_episode_equal"], metric["terminal"]["auroc"], metric["terminal"]["f1"],
                      metric["any_alarm_false_positive_rate"], metric["warning_proxies"]["first_privileged_dispatch"]["recall"],
                      metric["fit_seconds"]]
            lines.append(f"| {run['setting']} | {run['seed']} | {name} | " + " | ".join(map(show, values)) + f" | {metric.get('optimizer_success', 'NA')} |")
    lines += ["", "## Boundaries", "", *("- " + item for item in limitations)]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_v1_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {args.out}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference_path = ROOT / config["reference_manifest"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if config["settings"] != reference["config"]["settings"] or config["data"] != reference["config"]["data"]:
        raise ValueError("settings/data differ from frozen manifest")
    if sorted(config["expected_seeds"]) != sorted(map(int, reference["splits"])):
        raise ValueError("seed inventory differs from frozen manifest")
    trace_file, label_file = [ROOT / config["data"] / name for name in ("traces_public.jsonl", "labels.jsonl")]
    input_hashes = {p.relative_to(ROOT).as_posix(): protocol.digest(p) for p in (trace_file, label_file)}
    if input_hashes != {k.replace("\\", "/"): v for k, v in reference["input_sha256"].items()}:
        raise ValueError("input hashes differ from frozen manifest")
    traces, label_rows = protocol.read_jsonl(trace_file), protocol.read_jsonl(label_file)
    lookup = {r["episode_id"]: r["label"] for r in label_rows}
    if len(lookup) != len(label_rows) or set(lookup) != {t["episode_id"] for t in traces}:
        raise ValueError("label IDs must uniquely match trace IDs")
    for trace in traces:
        reject_outcomes(trace)
    labels = np.asarray([lookup[t["episode_id"]] for t in traces])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("binary labels required")
    splits, groups = protocol.resolve_manifest_splits(reference, traces)
    for parts in splits.values():
        if len(np.unique(labels[parts["fit"]])) != 2 or not (labels[parts["validation"]] == 0).any():
            raise ValueError("fit requires both classes and validation requires negatives")
    args.out.mkdir(parents=True)
    snapshots = snapshot_sources(args.out, args.config)
    manifest = {"scope": "reused synthetic development diagnostic; not independent confirmation",
                "config": config, "reference_manifest_sha256": protocol.digest(reference_path),
                "input_sha256": input_hashes, "splits": reference["splits"], "source_snapshot": snapshots,
                "code_sha256": {v["original"]: v["sha256"] for v in snapshots.values()},
                "methods": list(METHODS), "flat_feature_names": FLAT_FEATURE_NAMES,
                "joint_parameter_names": PARAMETER_NAMES,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    protocol.save_json(args.out / "manifest.json", manifest)
    limitations = config["limitations"] + [
        "All six methods have matched raw information, not identical representations: flat controls use 13 summaries plus 49 chronological template counts.",
        "no_messages removes only learned state transport; observed message summaries remain. no_latent_head retains only the 13 observable summaries in prediction.",
        "Different holdouts overlap; dispersion is split sensitivity, not independent-replication uncertainty.",
        "Optimizer nonconvergence is retained in the report, never discarded or rerun based on test scores.",
    ]
    runs, expected_runs = [], len(config["settings"]) * len(splits)
    write_report(args.out, runs, expected_runs, limitations)
    for setting in config["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        print("compiling " + key, flush=True)
        episodes = [prepare_trace(t, setting, config["max_joint_variables"]) for t in traces]
        for seed, parts in splits.items():
            print(f"running {key} seed={seed}", flush=True)
            directory = args.out / key / str(seed)
            directory.mkdir(parents=True)
            fit, test = parts["fit"], parts["test"]
            val = parts["validation"][labels[parts["validation"]] == 0]
            training = training_diagnostics([episodes[i] for i in fit])
            result, parameters, predictions = {}, {}, {}
            for name, model, elapsed in fit_estimators([episodes[i] for i in fit], labels[fit], config, seed):
                validation_scores = predict(model, [episodes[i] for i in val])
                threshold = protocol.choose_alarm_threshold([float(p.max()) for p in validation_scores], config["threshold"]["validation_negative_any_alarm_fpr"])
                scores = predict(model, [episodes[i] for i in test])
                metric = {"threshold_selection": threshold, "fit_seconds": elapsed,
                          **protocol.evaluate_scores(labels[test], scores, [traces[i] for i in test], threshold["threshold"])}
                if isinstance(model, (DynamicJointBN, TemplateBN)):
                    metric["optimizer_success"] = model.optimizer["success"]
                result[name] = metric
                artifact = directory / (name + ".joblib")
                joblib.dump(model, artifact)
                parameters[name] = {"artifact": artifact.name, "sha256": protocol.digest(artifact)}
                if isinstance(model, (DynamicJointBN, TemplateBN)):
                    parameters[name]["bn"] = model.parameters()
                else:
                    parameters[name]["hyperparameters"] = config["baselines"][name]
                predictions[name] = {"validation_negative": validation_scores, "test": scores}
                protocol.save_json(directory / "partial_metrics.json", {"setting": key, "seed": seed, "methods_finished": list(result), "metrics": result})
                protocol.save_json(directory / "model_parameters.json", parameters)
                print(f"finished {name}: fit_s={elapsed:.1f} prefix_brier={metric['prefix_brier_episode_equal']:.4f} converged={metric.get('optimizer_success', 'NA')}", flush=True)
            with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for part, indices in (("validation_negative", val), ("test", test)):
                    for position, i in enumerate(indices):
                        handle.write(json.dumps({"split": part, "episode_id": traces[i]["episode_id"], "label": int(labels[i]),
                            "group": groups[i], "sequence": [s.sequence for s in episodes[i].joint.steps],
                            "prefix_scores": {name: predictions[name][part][position].tolist() for name in METHODS}}, allow_nan=False) + "\n")
            run = {"setting": key, "seed": seed, "training": training, "metrics": result}
            protocol.save_json(directory / "metrics.json", run)
            runs.append(run)
            write_report(args.out, runs, expected_runs, limitations)
    print("completed " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
