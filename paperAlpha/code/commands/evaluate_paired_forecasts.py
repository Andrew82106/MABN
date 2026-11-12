"""Two-family DEVELOPMENT cross-fit, never an untouched submission test.

All repetitions of a source task stay together. Hyperparameters below are
fixed before reading paired outcomes; no per-test tuning or best-run selection.
The two-family sample cannot support a population confidence interval or a
low-false-alarm claim. Full-prefix API forecasts are evaluated separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
import numpy as np
from collect_submission_harnessaudit import read, sha, write_new
from paperalpha_monitor.contract_forecast import ContractForecastBN, ContractForecastInput, VARIABLES
from paperalpha_monitor.contract_forecast_learning import fit_contract_traces
from paperalpha_monitor.contract_forecast_projection import FEATURE_NAMES
from paperalpha_monitor.contract_forecast_baselines import ContractFactorHistGB, DEFAULT_TREE_CONFIG

CONFIG = {"linear_regularization": .1, "linear_max_iter": 500,
          "tree_config": DEFAULT_TREE_CONFIG, "tree_max_rounds": 5,
          "log_score_floor": 1e-15, "split": "leave_source_task_family_out",
          "alarm_threshold": None, "selection_on_test": False}
SOURCES = ('code/commands/evaluate_paired_forecasts.py', 'code/tests/test_paired_forecast_evaluation.py',
           'code/src/paperalpha_monitor/contract_forecast.py',
           'code/src/paperalpha_monitor/contract_forecast_learning.py',
           'code/src/paperalpha_monitor/contract_forecast_baselines.py')
STATES = ContractForecastBN().states


def score_distribution(probabilities, labels, risk_label, *, current_a=None):
    p = np.asarray(probabilities, float)
    if p.shape != (32,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > 1e-8:
        raise ValueError("proper normalized next-call distribution required")
    # Use the core validator only for support/label semantics, never fitted values.
    selected = ContractForecastBN()._label_mask(labels)
    known = sum(labels.get(k) is not None for k in VARIABLES)
    informative = any(labels.get(k) is not None for k in ("C", "B", "S")) or (
        current_a is None and labels.get("A") is not None)
    if current_a is not None and labels.get("A") is not None and current_a != labels["A"]:
        raise ValueError("target contradicts hard current evidence")
    mass = float(p[selected].sum())
    fully_known = known == 4
    risk = float(sum(v for s, v in zip(STATES, p) if s[1] == "TRIGGER" and not(s[0] and s[2]) and s[3]))
    if risk_label is not None and risk_label not in (0, 1):
        raise ValueError("risk target must be binary or missing")
    truth = selected.astype(float) if fully_known else None
    return {"known_label_fields": known, "joint_fully_known": fully_known,
            "marginal_log_loss": -math.log(max(mass, CONFIG["log_score_floor"])) if informative else None,
            "zero_target_mass": bool(informative and mass == 0),
            "joint_log_loss": -math.log(max(mass, CONFIG["log_score_floor"])) if fully_known else None,
            "joint_brier": float(np.sum((p-truth)**2)) if fully_known else None,
            "risk_probability": risk, "risk_label": risk_label,
            "risk_brier": (risk-risk_label)**2 if risk_label is not None else None}


def folds_for(episodes):
    ids = [e["episode_id"] for e in episodes]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("unique nonempty episode inventory required")
    tasks = sorted({e["task_id"] for e in episodes})
    if len(tasks) != 2:
        raise ValueError("this development protocol requires exactly its two source task families")
    return [{"test_task": t, "train_ids": [e["episode_id"] for e in episodes if e["task_id"] != t],
             "test_ids": [e["episode_id"] for e in episodes if e["task_id"] == t]} for t in tasks]


def group_scores(rows):
    """Window mean within each episode, then episode mean within each family."""
    keys = ("joint_log_loss", "joint_brier", "risk_brier", "marginal_log_loss")
    output = {}
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        families = []
        for task in sorted({r["task_id"] for r in subset}):
            task_rows = [r for r in subset if r["task_id"] == task]
            episode_values = []
            for identity in sorted({r["episode_id"] for r in task_rows}):
                scores = [r["scores"] for r in task_rows if r["episode_id"] == identity]
                episode_values.append({key: float(np.mean([s[key] for s in scores if s[key] is not None]))
                    if any(s[key] is not None for s in scores) else None for key in keys})
            families.append({"task_id": task, "episodes": len(episode_values),
                **{key: float(np.mean([s[key] for s in episode_values if s[key] is not None]))
                   if any(s[key] is not None for s in episode_values) else None for key in keys}})
        output[method] = {"families": families,
            "task_macro": {key: float(np.mean([s[key] for s in families if s[key] is not None]))
                           if any(s[key] is not None for s in families) else None for key in keys},
            "predictions": len(subset), "zero_target_mass_count": sum(r["scores"]["zero_target_mass"] for r in subset),
            "score_denominators": {key: {
                "windows": sum(r["scores"][key] is not None for r in subset),
                "episodes": len({r["episode_id"] for r in subset if r["scores"][key] is not None}),
                "task_families": len({r["task_id"] for r in subset if r["scores"][key] is not None})} for key in keys}}
    return output


def run(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("evaluation already exists; never overwrite")
    report = read(dataset / "private/REPORT.json")
    if report.get("status") != "projected":
        raise ValueError("complete independently projected dataset required")
    for relative, expected in report["artifact_sha256"].items():
        path = (dataset / relative).resolve()
        if not path.is_relative_to(dataset) or sha(path) != expected:
            raise ValueError("frozen predictor/target artifact changed")
    episodes = [read(p) for p in sorted((dataset / "public").glob("*.json"))]
    expected_ids = read(dataset / "PLAN.json")["episode_ids"]
    if set(e["episode_id"] for e in episodes) != set(expected_ids) or len(episodes) != len(expected_ids):
        raise ValueError("full frozen cohort denominator required")
    folds = folds_for(episodes)
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    output.mkdir(parents=True)
    for name in SOURCES:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_new(output / "PLAN.json", {"schema": "paired-forecast-development-evaluation-v1",
        "dataset_report_sha256": sha(dataset / "private/REPORT.json"), "config": CONFIG,
        "folds": folds, "source_sha256": hashes, "future_reply_scope": "first actual invocation only",
        "methods": ["inspectable_conditional_bn", "same_graph_histgb", "same_marginals_conditional_independence_ablation"],
        "no_final_submission_claim": True,
        "grouping": "All repeats and conditions of a source task remain in one fold. Two families do not establish arbitrary MAS generalization.",
        "scores": "Average windows within episode, then episodes within family, then families equally. Unknown targets omitted only from corresponding score and reported in coverage.",
        "linear_matched_control": "The same conditional GLMs are the BN itself; not counted as an independent algorithm expected to lose.",
        "thresholds_or_intervals": "Not fitted: two families are insufficient for population-level false-alarm bounds or a credible cluster confidence interval."})
    table, coverage = {}, []
    for episode in episodes:
        identity = episode["episode_id"]
        private = read(dataset / "private" / (identity + ".json"))
        label_map = {r["claim_number"]: r for r in private["labels"]}
        if len(label_map) != len(private["labels"]) or set(label_map) != {r["claim_number"] for r in episode["windows"]}:
            raise ValueError("input/target inventories differ")
        items = []
        for row in episode["windows"]:
            if not row["covered"]:
                continue
            p = row["projection"]
            sample = ContractForecastInput(features=p["features"], current_a=p["current_a"], metadata={})
            items.append({"sample": sample, "target": label_map[row["claim_number"]], "claim_number": row["claim_number"]})
        table[identity] = items
        coverage.append({"episode_id": identity, "task_id": episode["task_id"], "claims": len(episode["windows"]),
            "covered_predictor_windows": len(items), "known_risk_labels": sum(i["target"]["risk_label"] is not None for i in items)})
    predictions, fit_summaries = [], []
    for fold in folds:
        folder = output / "models" / fold["test_task"]
        folder.mkdir(parents=True)
        train = [table[k] for k in fold["train_ids"] if table[k]]
        traces = [[r["sample"] for r in trace] for trace in train]
        targets = [[r["target"]["labels"] for r in trace] for trace in train]
        started = time.perf_counter()
        bn = fit_contract_traces(traces, targets, feature_names=FEATURE_NAMES,
            regularization=CONFIG["linear_regularization"], max_iter=CONFIG["linear_max_iter"])
        linear_seconds = time.perf_counter()-started
        bn.save(folder / "bn.json")
        if not bn.optimizer["success"]:
            raise ValueError("optimizer did not converge; retain diagnostics, do not report trained success")
        started = time.perf_counter()
        tree = ContractFactorHistGB(FEATURE_NAMES, tree_config=CONFIG["tree_config"], max_rounds=CONFIG["tree_max_rounds"]).fit(traces, targets)
        tree_seconds = time.perf_counter()-started
        tree.save(folder / "histgb.joblib")
        restored_bn, restored_tree = ContractForecastBN.load(folder / "bn.json"), ContractFactorHistGB.load(folder / "histgb.joblib")
        fit_summaries.append({"test_task": fold["test_task"], "train_episode_ids": fold["train_ids"],
            "linear_seconds": linear_seconds, "tree_seconds": tree_seconds,
            "linear_fit": bn.optimizer, "tree_fit": tree.fit_report})
        for identity in fold["test_ids"]:
            for row in table[identity]:
                sample, target = row["sample"], row["target"]
                values = {"inspectable_conditional_bn": bn.joint(sample)["probabilities"],
                          "same_graph_histgb": tree.joint(sample)["probabilities"],
                          "same_marginals_conditional_independence_ablation": bn.independent_control(sample)["probabilities"]}
                if (not np.array_equal(values["inspectable_conditional_bn"], restored_bn.joint(sample)["probabilities"])
                        or not np.array_equal(values["same_graph_histgb"], restored_tree.joint(sample)["probabilities"])):
                    raise ValueError("saved model cannot exactly reproduce held-out predictions")
                for method, value in values.items():
                    predictions.append({"method": method, "episode_id": identity, "task_id": fold["test_task"],
                        "claim_number": row["claim_number"], "probabilities": np.asarray(value).tolist(),
                        "scores": score_distribution(value, target["labels"], target["risk_label"], current_a=sample.current_a)})
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("evaluation source changed during run")
    write_new(output / "PREDICTIONS.json", predictions)
    artifact_hashes = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
                       if p.is_file() and p != output / "REPORT.json"}
    write_new(output / "REPORT.json", {"schema": "paired-forecast-development-results-v1", "status": "evaluated",
        "plan_sha256": sha(output / "PLAN.json"), "coverage": coverage, "fit_summaries": fit_summaries,
        "scores": group_scores(predictions), "publication_ready": False,
        "independent_task_families": 2, "no_model_superiority_guarantee": True,
        "native_MAS_guard_comparison": "Separate current-event task, not these future distributions.",
        "alarm_evaluated": False, "confidence_interval": None,
        "artifact_sha256": artifact_hashes,
        "integrity_boundary": "Complete file inventory at write time, not a signature or independent model-reproduction audit."})
    return {"status": "evaluated", "folds": len(folds), "predictions": len(predictions), "publication_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired/forecast')
    parser.add_argument("--output", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired/evaluation')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output)))
