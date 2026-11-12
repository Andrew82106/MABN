"""Episode-balanced forecasting of terminal risk from observed prefixes.

The previously viewed synthetic data remains development evidence. Prefixes
share an episode outcome, not node labels, and are not independent trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.streaming_bn import (  # noqa: E402
    CELL_NAMES, PRIVILEGED_TOOLS, TemplateBN, reject_outcomes, replay, topology_group,
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def snapshot_sources(output, config_path):
    """Preserve executed source dependencies before later model upgrades."""
    paths = [Path(__file__), Path(config_path), *(ROOT / 'code/src/paperalpha_monitor').glob("*.py")]
    registry = ROOT / 'knowledge/templates/risk_templates_v1.json'
    if registry.exists():
        paths.append(registry)
        paths.append(ROOT / 'knowledge/risks/risk_catalog.json')
    joint_registry = ROOT / 'knowledge/templates/joint_risk_templates_v1.json'
    if joint_registry.exists():
        paths.append(joint_registry)
    typed_registry = ROOT / 'knowledge/templates/typed_joint_bn_v2.json'
    if typed_registry.exists():
        paths.append(typed_registry)
    action_registry = ROOT / 'knowledge/templates/action_scoped_bn_v3.json'
    if action_registry.exists():
        paths.append(action_registry)
    snapshots = {}
    for path in dict.fromkeys(p.resolve() for p in paths):
        try:
            relative = path.relative_to(ROOT.resolve())
        except ValueError:
            relative = Path("external") / path.name
        destination = Path(output) / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshots[str(relative).replace("\\", "/")] = {"original": str(path), "sha256": digest(destination)}
    return snapshots


def flatten_training(trajectories, labels):
    """One unit of fit weight per episode, distributed over all its prefixes."""
    labels = np.asarray(labels)
    if len(labels) != len(trajectories) or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("one binary terminal label per nonempty episode is required")
    lengths = np.asarray([len(rows) for rows in trajectories])
    if np.any(lengths == 0):
        raise ValueError("empty episodes have no supported prefix policy")
    snapshots = [snapshot for rows in trajectories for snapshot in rows]
    return snapshots, np.repeat(labels, lengths), np.concatenate([np.full(n, 1.0 / n) for n in lengths])


def choose_alarm_threshold(negative_episode_maxima, max_fpr=0.10):
    """Only validation-negative maxima enter this function; no test scores."""
    maxima = np.asarray(negative_episode_maxima, dtype=float)
    if maxima.ndim != 1 or not maxima.size or not np.isfinite(maxima).all() or np.any((maxima < 0) | (maxima > 1)):
        raise ValueError("finite validation-negative maxima in [0,1] are required")
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0,1]")
    # Immediately above a tied score is the smallest float that excludes it
    # under >=. Above 1 is a valid all-negative decision, not a probability.
    candidates = np.unique(np.r_[0.0, np.nextafter(maxima, np.inf), np.nextafter(1.0, np.inf)])
    threshold = next(float(t) for t in candidates if np.mean(maxima >= t) <= max_fpr)
    return {
        "threshold": threshold, "validation_negative_n": len(maxima),
        "validation_negative_alarm_n": int(np.sum(maxima >= threshold)),
        "validation_negative_any_alarm_fpr": float(np.mean(maxima >= threshold)),
        "requested_empirical_max_fpr": float(max_fpr), "decision": "score >= threshold",
        "selection": "validation-negative episode maxima only; no population guarantee",
    }


def resolve_manifest_splits(manifest, traces):
    """Reuse exact IDs; never resample a split after inspecting outcomes."""
    ids = [trace["episode_id"] for trace in traces]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trace IDs")
    index = {episode_id: i for i, episode_id in enumerate(ids)}
    groups = np.asarray([topology_group(trace["workflow_dag"]) for trace in traces])
    splits = {}
    for seed, parts in manifest["splits"].items():
        selected, seen, used_groups = {}, set(), set()
        for name in ("fit", "validation", "test"):
            part_ids = parts[name]["episode_ids"]
            if not part_ids or len(part_ids) != len(set(part_ids)) or not set(part_ids) <= set(index) or seen.intersection(part_ids):
                raise ValueError("manifest IDs must be nonempty, known, unique and disjoint")
            rows = np.asarray([index[episode_id] for episode_id in part_ids])
            part_groups = set(groups[rows])
            if part_groups != set(parts[name]["groups"]) or part_groups.intersection(used_groups):
                raise ValueError("manifest topology groups differ or overlap")
            selected[name], seen = rows, seen.union(part_ids)
            used_groups.update(part_groups)
        if seen != set(ids):
            raise ValueError("manifest must partition every trace exactly once")
        splits[int(seed)] = selected
    return splits, groups


def fit_models(fit_trajectories, fit_labels, config, seed):
    """Deliberately accepts only fit data, not validation or test trajectories."""
    snapshots, prefix_labels, weights = flatten_training(fit_trajectories, fit_labels)
    if len(np.unique(fit_labels)) != 2:
        raise ValueError("fit episodes require both classes")
    models = {
        "terminal_bn": TemplateBN(**config["training"]).fit([rows[-1] for rows in fit_trajectories], np.asarray(fit_labels)),
        "prefix_bn": TemplateBN(**config["training"]).fit(snapshots, prefix_labels, sample_weight=weights),
    }
    x = np.asarray([snapshot.vector() for snapshot in snapshots])
    logistic = make_pipeline(StandardScaler(), LogisticRegression(**config["baselines"]["logistic_prefix"], random_state=seed))
    logistic.fit(x, prefix_labels, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    histgb = HistGradientBoostingClassifier(**config["baselines"]["histgb_prefix"], random_state=seed)
    histgb.fit(x, prefix_labels, sample_weight=weights)
    models.update(logistic_prefix=logistic, histgb_prefix=histgb)
    return models, {"fit_episode_n": len(fit_labels), "fit_prefix_n": len(snapshots),
                    "prefix_weight_sum": float(weights.sum()), "per_episode_weight": 1.0}


def predict_trajectories(model, trajectories):
    snapshots = [snapshot for rows in trajectories for snapshot in rows]
    if not snapshots or any(not rows for rows in trajectories):
        raise ValueError("nonempty trajectories required")
    if isinstance(model, TemplateBN):
        scores = model.predict(snapshots)
    else:
        scores = model.predict_proba(np.asarray([snapshot.vector() for snapshot in snapshots]))[:, 1]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("model produced invalid probability scores")
    return list(np.split(scores, np.cumsum([len(rows) for rows in trajectories])[:-1]))


def evaluate_scores(labels, probabilities, traces, threshold):
    labels = np.asarray(labels)
    if len(labels) != len(probabilities) or len(labels) != len(traces) or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("scores, binary labels and traces must align")
    if any(len(p) != len(t["events"]) or len(p) == 0 for p, t in zip(probabilities, traces)):
        raise ValueError("exactly one score per nonempty event prefix is required")
    terminal = np.asarray([p[-1] for p in probabilities])
    alarms = [next((k for k, p in enumerate(rows) if p >= threshold), None) for rows in probabilities]
    negatives, positives = labels == 0, labels == 1
    any_alarm = np.asarray([alarm is not None for alarm in alarms])
    proxies = {}
    for name, privileged in (("first_tool_call", False), ("first_privileged_dispatch", True)):
        eligible, early = 0, 0
        for y, alarm, trace in zip(labels, alarms, traces):
            dispatch = next((k for k, event in enumerate(trace["events"])
                             if event.get("event_type") == "tool_call"
                             and (not privileged or event.get("tool") in PRIVILEGED_TOOLS)), None)
            if y == 1 and dispatch is not None:
                eligible += 1
                early += int(alarm is not None and alarm < dispatch)
        proxies[name] = {"eligible_positive_n": eligible, "strictly_before_alarm_n": early,
                         "recall": early / eligible if eligible else None}
    return {
        "terminal": {
            "f1": float(f1_score(labels, terminal >= threshold, zero_division=0)),
            "auroc": float(roc_auc_score(labels, terminal)) if len(np.unique(labels)) == 2 else None,
            "auprc": float(average_precision_score(labels, terminal)) if positives.any() else None,
            "brier": float(np.mean((terminal - labels) ** 2)),
        },
        "prefix_brier_episode_equal": float(np.mean([np.mean((np.asarray(p) - y) ** 2) for y, p in zip(labels, probabilities)])),
        "any_alarm_false_positive_rate": float(any_alarm[negatives].mean()) if negatives.any() else None,
        "any_alarm_recall": float(any_alarm[positives].mean()) if positives.any() else None,
        "negative_episode_n": int(negatives.sum()), "positive_episode_n": int(positives.sum()),
        "warning_proxies": proxies, "actual_violation_lead_time": None,
        "threshold": float(threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / 'experiments/runtime_alarm/configs/prefix_forecast_v1.json')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/runtime_alarm/runs/prefix_forecast_v1_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {args.out}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference_path = ROOT / config["reference_manifest"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if config["settings"] != reference["config"]["settings"] or config["data"] != reference["config"]["data"]:
        raise ValueError("settings and data must match the frozen source manifest")
    if sorted(map(int, reference["splits"])) != sorted(config["expected_seeds"]):
        raise ValueError("source manifest seeds do not match protocol")
    trace_file, label_file = [ROOT / config["data"] / name for name in ("traces_public.jsonl", "labels.jsonl")]
    input_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in (trace_file, label_file)}
    source_hashes = {key.replace("\\", "/"): value for key, value in reference["input_sha256"].items()}
    if input_hashes != source_hashes:
        raise ValueError("data hashes differ from the frozen source manifest")
    traces, label_rows = read_jsonl(trace_file), read_jsonl(label_file)
    lookup = {row["episode_id"]: row["label"] for row in label_rows}
    if len(lookup) != len(label_rows) or set(lookup) != {t["episode_id"] for t in traces}:
        raise ValueError("label IDs must uniquely match trace IDs")
    for trace in traces:
        reject_outcomes(trace)
    labels = np.asarray([lookup[t["episode_id"]] for t in traces])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("binary episode labels required")
    splits, groups = resolve_manifest_splits(reference, traces)
    for parts in splits.values():
        if len(np.unique(labels[parts["fit"]])) != 2 or not (labels[parts["validation"]] == 0).any():
            raise ValueError("fit needs both classes and validation needs negatives; do not resample")
    args.out.mkdir(parents=True)
    source_snapshot = snapshot_sources(args.out, args.config)
    manifest = {
        "scope": "reused synthetic development diagnostic, not confirmation",
        "config": config, "reference_manifest": str(reference_path),
        "reference_manifest_sha256": digest(reference_path), "input_sha256": input_hashes,
        "source_snapshot": source_snapshot,
        "code_sha256": {record["original"]: record["sha256"] for record in source_snapshot.values()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "splits": reference["splits"], "feature_names": CELL_NAMES,
        "prefix_distribution": "select an episode uniformly, then a nonempty observed-event prefix uniformly",
        "threshold_scope": "empirical validation-negative any-alarm constraint; no test tuning or safety guarantee",
    }
    save_json(args.out / "manifest.json", manifest)
    runs = []
    for setting in config["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        trajectories = [replay(t, workflow=t["workflow_dag"] if setting["workflow"] == "workflow_known" else None,
                               permission_mode=setting["permission"]) for t in traces]
        if any(not rows for rows in trajectories):
            raise ValueError("empty episodes are outside this protocol")
        for seed, parts in splits.items():
            fit, val, test = (parts[name] for name in ("fit", "validation", "test"))
            val_negative = val[labels[val] == 0]
            print(f"running {key} seed={seed}", flush=True)
            run_dir = args.out / key / str(seed)
            run_dir.mkdir(parents=True)
            models, training = fit_models([trajectories[i] for i in fit], labels[fit], config, seed)
            result, parameters, predictions = {}, {}, {}
            for name, model in models.items():
                validation_scores = predict_trajectories(model, [trajectories[i] for i in val_negative])
                threshold = choose_alarm_threshold([float(p.max()) for p in validation_scores],
                                                   config["alarm_threshold"]["validation_negative_any_alarm_fpr"])
                test_scores = predict_trajectories(model, [trajectories[i] for i in test])
                result[name] = {"threshold_selection": threshold,
                                **evaluate_scores(labels[test], test_scores, [traces[i] for i in test], threshold["threshold"])}
                predictions[name] = {"validation_negative": validation_scores, "test": test_scores}
                artifact = run_dir / (name + ".joblib")
                joblib.dump(model, artifact)
                parameters[name] = {"artifact": artifact.name, "sha256": digest(artifact),
                                    "training_target": config["terminal_bn_target"] if name == "terminal_bn" else config["forecast_target"]}
                if isinstance(model, TemplateBN):
                    parameters[name]["bn"] = model.parameters()
                    result[name]["optimizer_success"] = model.optimizer["success"]
                else:
                    parameters[name]["hyperparameters"] = config["baselines"][name]
            run = {"setting": key, "seed": seed, "training": training, "metrics": result}
            runs.append(run)
            save_json(run_dir / "metrics.json", run)
            save_json(run_dir / "model_parameters.json", parameters)
            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for part_name, indices in (("validation_negative", val_negative), ("test", test)):
                    for position, i in enumerate(indices):
                        handle.write(json.dumps({"split": part_name, "episode_id": traces[i]["episode_id"],
                            "label": int(labels[i]), "group": groups[i],
                            "sequence": [s.sequence for s in trajectories[i]],
                            "prefix_scores": {name: p[part_name][position].tolist() for name, p in predictions.items()}}, allow_nan=False) + "\n")
    limitations = [
        "Reused synthetic development splits; no original published MAS baseline reproduction.",
        "Prefix labels denote terminal episode outcomes, not observed prefix violations or node labels.",
        "Forecast probability estimation is the training objective, not proof of calibration.",
        "Before a tool proposal/call, only global_context is active; the complete two-layer model is not implemented.",
        "No independent harm timestamp; strict pre-dispatch recalls are proxies, not harm prevention.",
        "Validation FPR constraint is empirical only; test FPR may exceed it and safety is not guaranteed.",
        "Within-episode prefixes and overlapping held-out splits are correlated; no independent-trial confidence interval.",
    ]
    save_json(args.out / "metrics.json", {"runs": runs, "limitations": limitations})
    lines = ["# Prefix terminal-risk forecasting: development diagnostic", "",
             "目标：根据当前前缀预测任务结束时的风险标签。每条训练轨迹总权重相同。", "",
             "报警阈值只由验证集负任务的整条轨迹最大分数决定，经验 any-alarm FPR ≤ 0.10；这不是安全保证。", "",
             "| setting | seed | method | terminal F1 | terminal AUROC | terminal AUPRC | terminal Brier | episode-equal prefix Brier | any-alarm FPR | pre-first-tool recall |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    def display(value):
        return "NA" if value is None else f"{value:.3f}"
    for run in runs:
        for name, m in run["metrics"].items():
            values = [m["terminal"][k] for k in ("f1", "auroc", "auprc", "brier")]
            values += [m["prefix_brier_episode_equal"], m["any_alarm_false_positive_rate"], m["warning_proxies"]["first_tool_call"]["recall"]]
            lines.append(f"| {run['setting']} | {run['seed']} | {name} | " + " | ".join(map(display, values)) + " |")
    lines += ["", "## Boundaries", "", *("- " + item for item in limitations)]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("completed " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
