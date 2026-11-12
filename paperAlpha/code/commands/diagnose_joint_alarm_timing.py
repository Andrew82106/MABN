"""Validation-negative-only diagnosis of trajectory-max alarm thresholds.

No model refit, test-score tuning, or changes to stored experiment results.
The pre-dispatch-only threshold is a diagnostic counterfactual, not a deployed
policy and not evidence of improved positive recall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_prefix_forecast_v1 as protocol
from paperalpha_monitor.streaming_bn import PRIVILEGED_TOOLS

ROOT = Path(__file__).resolve().parents[2]


def summarize_negative_scores(episodes, threshold, alpha):
    pre, whole, late_peak, has_dispatch, suppressed = [], [], 0, 0, 0
    for trace, scores in episodes:
        scores = np.asarray(scores, dtype=float)
        if len(scores) != len(trace["events"]) or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("invalid per-prefix scores")
        cut = next((i for i, e in enumerate(trace["events"]) if e.get("event_type") == "tool_call" and e.get("tool") in PRIVILEGED_TOOLS), len(scores))
        whole.append(float(scores.max()))
        if cut < len(scores):
            has_dispatch += 1
            # Earliest peak: ties before dispatch are not classified as late.
            late_peak += int(np.argmax(scores) >= cut)
        if cut > 0:
            maximum = float(scores[:cut].max())
            pre.append(maximum)
            suppressed += int(maximum < threshold)
    counterfactual = protocol.choose_alarm_threshold(pre, alpha) if pre else None
    return {"negative_episode_n": len(episodes), "with_privileged_dispatch_n": has_dispatch,
            "first_peak_at_or_after_dispatch_n": late_peak,
            "late_first_peak_fraction_given_dispatch": late_peak / has_dispatch if has_dispatch else None,
            "with_pre_dispatch_prefix_n": len(pre), "pre_dispatch_max_median": float(np.median(pre)) if pre else None,
            "whole_trajectory_max_median": float(np.median(whole)) if whole else None,
            "actual_full_trajectory_threshold": threshold,
            "pre_dispatch_negative_alarm_fraction_at_actual_threshold": 1 - suppressed / len(pre) if pre else None,
            "counterfactual_pre_dispatch_only_threshold": counterfactual,
            "interpretation": "Negative calibration trajectories only; lower hypothetical threshold does not prove useful early recall or a full-trajectory false-alarm guarantee."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dynamic_joint_bn_v1_20260919')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/runtime_alarm/runs/joint_alarm_timing_diagnostic_20260919')
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite timing diagnosis")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    if not metrics["complete"] and not args.allow_partial:
        raise ValueError("experiment incomplete; use --allow-partial for explicitly limited diagnosis")
    path = ROOT / manifest["config"]["data"] / "traces_public.jsonl"
    if protocol.digest(path) != manifest["input_sha256"][path.relative_to(ROOT).as_posix()]:
        raise ValueError("trace data differs from frozen experiment")
    traces = {t["episode_id"]: t for t in protocol.read_jsonl(path)}
    results, hashes = [], {}
    for run in metrics["runs"]:
        file = args.run / run["setting"] / str(run["seed"]) / "predictions.jsonl"
        hashes[file.relative_to(args.run).as_posix()] = protocol.digest(file)
        rows = [r for r in protocol.read_jsonl(file) if r["split"] == "validation_negative"]
        if not rows or any(r["label"] != 0 for r in rows):
            raise ValueError("validation negative inventory is invalid")
        permitted = set(manifest["splits"][str(run["seed"])]["validation"]["episode_ids"])
        if len({r["episode_id"] for r in rows}) != len(rows) or not {r["episode_id"] for r in rows} <= permitted:
            raise ValueError("diagnostic rows are not unique validation IDs")
        for method in manifest["methods"]:
            summary = summarize_negative_scores([(traces[r["episode_id"]], r["prefix_scores"][method]) for r in rows],
                run["metrics"][method]["threshold"], manifest["config"]["threshold"]["validation_negative_any_alarm_fpr"])
            results.append({"setting": run["setting"], "seed": run["seed"], "method": method, **summary})
    args.out.mkdir(parents=True)
    protocol.save_json(args.out / "diagnostic.json", {"scope": "validation-negative threshold diagnostic only",
        "experiment_complete_at_read": metrics["complete"], "completed_runs": len(metrics["runs"]),
        "run_manifest_sha256": protocol.digest(args.run / "manifest.json"), "prediction_sha256": hashes,
        "script_sha256": protocol.digest(Path(__file__)), "results": results})
    print(json.dumps({"completed_runs": len(metrics["runs"]), "method_diagnostics": len(results), "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
