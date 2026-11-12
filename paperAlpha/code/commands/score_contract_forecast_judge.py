"""Rescore receipt-bound forecasts on identical public windows, without API or fit.

Write-time manifests bind files, not external authenticity. Reparse judge replies
and remove all duplicate identities before joining private labels. Numeric cached
scores are ignored. This is not an independent replay of fitted model predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
import numpy as np
from collect_submission_harnessaudit import read, sha, write_new, digest
from collect_contract_forecast_judge import load_plan
import evaluate_paired_forecasts as numeric
from evaluate_paired_forecasts import STATES, score_distribution, group_scores
from paperalpha_monitor.contract_forecast_llm import STATES as FEASIBLE, parse_forecast
from paperalpha_monitor.contract_forecast import VARIABLES
from paperalpha_runtime.privacy_transfer import delivered_text

METHODS = ("inspectable_conditional_bn", "same_graph_histgb",
           "same_marginals_conditional_independence_ablation")
SCORING_SOURCES = ('code/commands/score_contract_forecast_judge.py', 'code/tests/test_contract_forecast_judge_scoring.py',
    'code/commands/collect_contract_forecast_judge.py', 'code/commands/collect_submission_harnessaudit.py',
    'code/commands/evaluate_paired_forecasts.py', 'code/src/paperalpha_monitor/contract_forecast.py',
    'code/src/paperalpha_monitor/contract_forecast_llm.py', 'code/src/paperalpha_runtime/privacy_transfer.py')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify_manifest(root, entries, *, excluded):
    """Reject additions as well as deletions/mutations; never regenerate hashes."""
    root = Path(root).resolve()
    require(isinstance(entries, dict) and bool(entries), "complete artifact manifest required")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.relative_to(root).as_posix() not in excluded}
    require(actual == set(entries), "artifact inventory differs")
    for relative, expected in entries.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and sha(path) == expected, "artifact content changed: " + relative)


def validate_distribution(probabilities, current_a=None):
    p = np.asarray(probabilities, float)
    require(p.shape == (32,) and np.isfinite(p).all() and (p >= 0).all()
            and abs(float(p.sum()) - 1) <= 1e-8, "proper normalized next-call distribution required")
    require(all(value == 0 for state, value in zip(STATES, p) if state not in FEASIBLE),
            "forecast assigns mass outside feasible support")
    require(current_a is None or (type(current_a) is int and current_a in (0, 1)), "hard A must be 0/1 or unknown")
    require(current_a is None or all(value == 0 for state, value in zip(STATES, p) if state[0] != current_a),
            "forecast contradicts hard current evidence")
    return p


def expand_distribution(prediction):
    states, values = prediction.get("states"), prediction.get("probabilities")
    require(isinstance(states, list) and isinstance(values, list) and len(states) == len(values),
            "explicit aligned forecast states required")
    tuples = []
    for state in states:
        require(isinstance(state, dict) and set(state) == set(VARIABLES), "exact state variable schema required")
        require(all(type(state[k]) is int and state[k] in (0, 1) for k in ("A", "B", "S"))
                and isinstance(state["C"], str), "typed state values required")
        tuples.append(tuple(state[k] for k in VARIABLES))
    require(len(tuples) == len(set(tuples)) and set(tuples) == set(FEASIBLE),
            "each feasible state must occur exactly once")
    require(all(type(v) in (int, float) for v in values), "numeric probabilities required")
    result = np.zeros(32)
    for state, value in zip(tuples, values):
        result[STATES.index(state)] = value
    return validate_distribution(result)


def reconstruct_judge(judge, plan, collected):
    """No labels enter: acceptance follows the actual stored replies."""
    expected_keys = [(c["episode_id"], c["claim_number"]) for c in plan["cases"]]
    require(len(expected_keys) == len(set(expected_keys)), "duplicate scheduled request")
    records = []
    for index, case in enumerate(plan["cases"], 1):
        folder = judge / "runs" / f"{index:04d}"
        request, preliminary = read(folder / "REQUEST.json"), read(folder / "RESULT.json")
        require({k: v for k, v in request.items() if k != "monotonic_ns"}
                == {"plan_sha256": plan["sha256"], **case}, "request receipt differs from frozen schedule")
        require(request["request_sha256"] == digest(request["request"]), "request receipt hash differs")
        require(type(request.get("monotonic_ns")) is int and request["monotonic_ns"] >= 0,
                "request observation clock required")
        record = {"episode_id": case["episode_id"], "task_id": case["task_id"],
                  "claim_number": case["claim_number"], "index": index, "status": "failed", "error_type": None}
        path = folder / "RESPONSE.json"
        if path.exists():
            response = read(path)
            record["sdk_response_id"] = response.get("id")
            try:
                if (not isinstance(response.get("id"), str) or not response["id"].strip()
                        or response.get("error") is not None):
                    raise ValueError("invalid response identity or explicit API error")
                text = delivered_text(response)
                if text["native_refusal"]:
                    raise ValueError("refusal is not a forecast")
                record["prediction"] = parse_forecast(text["text"], case["current_a"])
                record["status"] = "predicted"
            except Exception as exc:
                record["error_type"] = type(exc).__name__
        else:
            require(preliminary.get("status") == "failed" and isinstance(preliminary.get("error_type"), str)
                    and bool(preliminary["error_type"]), "missing response needs a retained failure receipt")
            record["error_type"] = preliminary["error_type"]
        require(type(preliminary.get("elapsed_ns")) is int and preliminary["elapsed_ns"] >= 0,
                "nonnegative recorded duration required")
        record["elapsed_ns"] = preliminary["elapsed_ns"]
        require(record == preliminary, "per-call parse does not reproduce from its response receipt")
        records.append(record)
    ids = Counter(r["sdk_response_id"] for r in records
                  if isinstance(r.get("sdk_response_id"), str) and r["sdk_response_id"].strip())
    duplicates = {identity for identity, count in ids.items() if count > 1}
    affected = []
    for record in records:
        if isinstance(record.get("sdk_response_id"), str) and record["sdk_response_id"] in duplicates:
            affected.append(record["index"])
            record.update(status="failed", error_type="DuplicateResponseIdentity")
            record.pop("prediction", None)
    identity_audit = read(judge / "IDENTITY_AUDIT.json")
    require(identity_audit.get("duplicate_response_ids") == sorted(duplicates)
            and identity_audit.get("affected_indices") == affected, "response identity audit does not reproduce")
    require(collected.get("records") == records, "judge REPORT does not reproduce from receipts")
    outcomes = dict(Counter(r["status"] for r in records))
    require(collected.get("api_attempts") == len(records) and collected.get("outcomes") == outcomes
            and collected.get("status") == ("completed_with_failures" if outcomes.get("failed") else "completed"),
            "judge summary denominator differs from actual requests")
    return records


def load_dataset(dataset, projection):
    verify_manifest(dataset, projection["artifact_sha256"], excluded={"private/REPORT.json"})
    require(projection.get("status") == "projected", "completed independent projection required")
    source_plan = read(dataset / "PLAN.json")
    identities = source_plan["episode_ids"]
    require(bool(identities) and len(identities) == len(set(identities)), "unique full episode inventory required")
    episodes, targets, hard, tasks = [], {}, {}, {}
    for episode in identities:
        public, private = read(dataset / "public" / (episode + ".json")), read(dataset / "private" / (episode + ".json"))
        require(public.get("episode_id") == private.get("episode_id") == episode
                and public.get("task_id") == private.get("task_id"), "public/private episode identity differs")
        episodes.append(public)
        labels = {r["claim_number"]: r for r in private["labels"]}
        windows = {r["claim_number"]: r for r in public["windows"]}
        require(len(labels) == len(private["labels"]) and len(windows) == len(public["windows"])
                and set(labels) == set(windows), "complete unique input/target windows required")
        for number, window in windows.items():
            key = episode, number
            targets[key], tasks[key] = labels[number], public["task_id"]
            if window["covered"]:
                hard[key] = window["projection"]["current_a"]
    return episodes, targets, hard, tasks


def numeric_predictions(numerical, dataset, episodes, targets, hard, tasks):
    plan, report = read(numerical / "PLAN.json"), read(numerical / "REPORT.json")
    require(report.get("status") == "evaluated" and report.get("plan_sha256") == sha(numerical / "PLAN.json"),
            "completed matching numerical report required")
    require(plan.get("dataset_report_sha256") == sha(dataset / "private/REPORT.json")
            and plan.get("methods") == list(METHODS) and plan.get("folds") == numeric.folds_for(episodes)
            and plan.get("config") == numeric.CONFIG, "numerical dataset/method/split profile differs")
    verify_manifest(numerical, report["artifact_sha256"], excluded={"REPORT.json"})
    required = {"PLAN.json", "PREDICTIONS.json", *(
        f"models/{fold['test_task']}/{name}" for fold in plan["folds"] for name in ("bn.json", "histgb.joblib"))}
    require(required <= set(report["artifact_sha256"]), "numerical checkpoint inventory incomplete")
    require(set(plan["source_sha256"]) == set(numeric.SOURCES), "complete numerical source freeze required")
    for name, expected in plan["source_sha256"].items():
        require(sha(numerical / "source" / name) == expected and sha(ROOT / name) == expected,
                "numerical source snapshot differs")
    rows, seen = [], {method: set() for method in METHODS}
    for record in read(numerical / "PREDICTIONS.json"):
        method, key = record["method"], (record["episode_id"], record["claim_number"])
        require(method in seen and key in hard and key not in seen[method],
                "unexpected or duplicate numerical prediction")
        require(record["task_id"] == tasks[key], "numerical task grouping differs")
        seen[method].add(key)
        p = validate_distribution(record["probabilities"], hard[key])
        target = targets[key]
        rows.append({"method": method, "episode_id": key[0], "claim_number": key[1], "task_id": tasks[key],
            "probabilities": p.tolist(),
            "scores": score_distribution(p, target["labels"], target["risk_label"], current_a=hard[key])})
    require(all(keys == set(hard) for keys in seen.values()), "each fixed numerical method needs every covered window")
    return rows, report


def run(judge, dataset, numerical):
    judge, dataset, numerical = (Path(p).resolve() for p in (judge, dataset, numerical))
    destination = judge / "SCORES.json"
    require(not destination.exists(), "judge already scored; never overwrite")
    sources = {name: sha(ROOT / name) for name in SCORING_SOURCES}
    input_reports = {path: sha(path) for path in (
        judge / "REPORT.json", dataset / "private/REPORT.json", numerical / "REPORT.json")}
    plan, collected = load_plan(judge), read(judge / "REPORT.json")
    require(Path(plan["dataset"]).resolve() == dataset and collected.get("plan_sha256") == plan["sha256"],
            "judge/dataset plan differs")
    verify_manifest(judge, collected["artifact_sha256"], excluded={"REPORT.json"})
    records = reconstruct_judge(judge, plan, collected)  # Before reading any labels.
    projection = read(dataset / "private/REPORT.json")
    episodes, labels, hard, tasks = load_dataset(dataset, projection)
    require({(c["episode_id"], c["claim_number"]) for c in plan["cases"]} == set(hard),
            "judge schedule is not the full covered public window set")
    controls, numerical_report = numeric_predictions(numerical, dataset, episodes, labels, hard, tasks)
    scored, failures = [], []
    for record in records:
        key = record["episode_id"], record["claim_number"]
        require(record["task_id"] == tasks[key], "judge task grouping differs")
        target = labels[key]
        if record["status"] != "predicted":
            failures.append({"episode_id": key[0], "claim_number": key[1],
                "error_type": record["error_type"], "known_risk_target": target["risk_label"] is not None})
            continue
        p = validate_distribution(expand_distribution(record["prediction"]), hard[key])
        scored.append({"method": "full_public_prefix_llm", "episode_id": key[0],
            "claim_number": key[1], "task_id": tasks[key], "probabilities": p.tolist(),
            "scores": score_distribution(p, target["labels"], target["risk_label"], current_a=hard[key])})
    eligible = {(r["episode_id"], r["claim_number"]) for r in scored}
    common = [r for r in controls if (r["episode_id"], r["claim_number"]) in eligible] + scored
    common_keys = {method: [(r["episode_id"], r["claim_number"]) for r in common if r["method"] == method]
                   for method in (*METHODS, "full_public_prefix_llm")}
    require(all(len(keys) == len(eligible) and set(keys) == eligible for keys in common_keys.values()),
            "common-window comparison is not exactly paired")
    verify_manifest(judge, collected["artifact_sha256"], excluded={"REPORT.json"})
    verify_manifest(dataset, projection["artifact_sha256"], excluded={"private/REPORT.json"})
    verify_manifest(numerical, numerical_report["artifact_sha256"], excluded={"REPORT.json"})
    require(all(sha(ROOT / name) == value for name, value in sources.items()), "scoring source changed during execution")
    require(all(sha(path) == value for path, value in input_reports.items()), "input report changed during scoring")
    report = {"schema": "paired-full-prefix-forecast-development-comparison-v1",
        "judge_collection_sha256": sha(judge / "REPORT.json"), "source_sha256": sources,
        "dataset_report_sha256": sha(dataset / "private/REPORT.json"),
        "numerical_report_sha256": sha(numerical / "REPORT.json"),
        "numerical_predictions_sha256": sha(numerical / "PREDICTIONS.json"),
        "judge_requests": len(records), "valid_judge_forecasts": len(scored), "failures": failures,
        "common_window_keys": [list(key) for key in sorted(eligible)],
        "common_windows_per_method": {method: len(keys) for method, keys in common_keys.items()},
        "judge_available_case_scores": group_scores(scored), "common_valid_window_scores": group_scores(common),
        "judge_predictions": scored, "numeric_cached_scores_used": False,
        "selection_warning": "Common-window scores are conditional on valid judge responses, not full-cohort superiority. Failures are retained, never imputed as zero risk.",
        "integrity_boundary": "Write-time inventories and rechecked receipt/label bindings, not signatures or an independent replay of trained models.",
        "independent_task_families": len({e["task_id"] for e in episodes}), "publication_ready": False,
        "population_interval": None, "alarm_or_calibration_fit": False,
        "scope": "Development first-next-reply forecast only, not whole-MAS safety or a current-event Guard score."}
    write_new(destination, report)
    return {"scored": len(scored), "failed": len(failures), "publication_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired'
    parser.add_argument("--judge", type=Path, default=base / "llm_judge")
    parser.add_argument("--dataset", type=Path, default=base / "forecast")
    parser.add_argument("--numerical", type=Path, default=base / "evaluation")
    args = parser.parse_args()
    print(json.dumps(run(args.judge, args.dataset, args.numerical)))
