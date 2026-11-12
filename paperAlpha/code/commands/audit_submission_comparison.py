"""Matched-information audit of the exploratory whole-trace result table.

No API calls.  This is a development comparison, not an untouched test.  It
adds strong controls with the *union* of the public normative/runtime/template
features.  The historical models selected thresholds on fit data; the audit
uses that same rule only to diagnose the existing table, not endorse it as a
submission protocol.  Publication evaluation needs separate calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recall_at_fpr(y, p, budget=0.05):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr[fpr <= budget])) if np.any(fpr <= budget) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / 'experiments/submission_audit/runs/final_eval')
    args = ap.parse_args()
    table = args.out / "metrics.json"
    original = json.loads(table.read_text(encoding="utf-8"))
    evaluator = load("comparison_final", ROOT / 'code/commands/evaluate_submission_final.py')
    fair = load("comparison_fair", ROOT / 'code/commands/evaluate_independent_fair_baselines_v1.py')
    hier = load("comparison_hier", ROOT / 'code/commands/evaluate_hierarchical_monitor_v3.py')
    trace_path = ROOT / 'experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl'
    labels_path = ROOT / 'experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl'
    traces = read_lines(trace_path)
    for trace in traces:
        hier.validate_public_trace(trace)
        if "mechanism" in trace:
            raise ValueError("private mechanism in public feature input")
    # The design matrix exists before any label/family join.
    x = np.asarray([np.r_[fair.aggregate_features(e), fair.local_features(e),
        fair.graph_features(e), hier.workflow_features(e),
        evaluator.knowledge_template_features(e)] for e in traces])
    labels = {r["episode_id"]: r for r in read_lines(labels_path)}
    y = np.asarray([labels[e["episode_id"]]["label"] for e in traces], dtype=int)
    groups = np.asarray([labels[e["episode_id"]]["scenario_family"] for e in traces])
    results = []
    with threadpool_limits(limits=1):
        for run in original["runs"]:
            held = set(run["split"]["held_out_groups"])
            te = np.flatnonzero([g in held for g in groups])
            tr = np.flatnonzero([g not in held for g in groups])
            sub_eps = [dict(traces[int(i)], scenario_family=str(groups[int(i)]), label=int(y[int(i)])) for i in tr]
            _, calib_rel, _ = evaluator.grouped_split(sub_eps, y[tr], run["seed"] + 700, 0.25)
            calib = tr[calib_rel]
            calib_set = set(int(i) for i in calib)
            fit = np.asarray([int(i) for i in tr if int(i) not in calib_set], dtype=int)
            if [traces[i]["episode_id"] for i in te] != run["predictions"]["episode_id"]:
                raise ValueError("audit split/prediction order differs")
            methods = {
                "matched_public_logistic": make_pipeline(StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=1000, random_state=run["seed"])),
                "matched_public_histgb": HistGradientBoostingClassifier(max_iter=150,
                    max_leaf_nodes=15, max_depth=3, min_samples_leaf=20,
                    l2_regularization=1.0, early_stopping=False, random_state=run["seed"]),
            }
            scores, metrics, costs = {}, {}, {}
            for name, model in methods.items():
                start = time.perf_counter()
                model.fit(x[fit], y[fit])
                costs[name] = {"training_seconds": time.perf_counter() - start}
                pfit = model.predict_proba(x[fit])[:, 1]
                pcal = model.predict_proba(x[calib])[:, 1]
                ptest = model.predict_proba(x[te])[:, 1]
                # Match the proposed monitor's independent calibration
                # boundary: threshold and one-dimensional probability
                # calibration use groups never seen during fitting.
                cal = LogisticRegression(C=1e6, max_iter=1000, random_state=run["seed"])
                cal.fit(np.log(np.clip(pcal, 1e-5, 1 - 1e-5)).reshape(-1, 1), y[calib])
                pcal = cal.predict_proba(np.log(np.clip(pcal, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
                ptest = cal.predict_proba(np.log(np.clip(ptest, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
                threshold = fair.best_threshold(y[calib], pcal)
                metrics[name] = evaluator.metric(fair, y[te], ptest, threshold)
                scores[name] = ptest.tolist()
            ours = "ours_hierarchical_bn_runtime"
            metrics[ours] = run["metrics"][ours]
            scores[ours] = run["predictions"]["scores"][ours]
            results.append({"seed": run["seed"], "split": run["split"], "metrics": metrics,
                "costs": costs, "predictions": {"episode_id": run["predictions"]["episode_id"],
                    "label": y[te].tolist(), "scores": scores}})
    safety = {}
    for name in ("matched_public_logistic", "matched_public_histgb", "ours_hierarchical_bn_runtime"):
        values = [recall_at_fpr(np.asarray(r["predictions"]["label"]), np.asarray(r["predictions"]["scores"][name]))
                  for r in results]
        safety[name] = {"metric": "recall_at_fpr_0.05", "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)), "per_seed": values}
    audit = {"schema": "matched-information-whole-trace-development-audit",
        "submission_ready": False, "api_calls": 0,
        "source_table_sha256": sha(table), "public_traces_sha256": sha(trace_path),
        "labels_sha256": sha(labels_path), "design_dimensions": list(x.shape),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / 'code/commands/evaluate_submission_final.py',
            ROOT / 'code/commands/evaluate_independent_fair_baselines_v1.py',
            ROOT / 'code/commands/evaluate_hierarchical_monitor_v3.py']},
        "feature_blocks": ["flat events", "local events", "runtime graph",
            "normative workflow comparison", "authorization and deviation templates"],
        "aggregate": evaluator.aggregate(results), "safety_metrics": safety, "runs": results,
        "cautions": ["The same exposed synthetic development data were used to modify templates.",
            "Whole-trajectory inputs do not prove pre-action warning.",
            "Historical baselines omitted explicit permission_mismatch despite its availability to ours.",
            "The active table uses grouped out-of-fold fusion and group-held-out calibration; this audit applies the same fit/calibration boundary to matched controls.",
            "Five overlapping repeated holdouts are not five independent populations.",
            "This comparison cannot prove native ALTEDA or other MAS guard superiority."]}
    path = args.out / "comparison_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(path), "design_dimensions": list(x.shape),
        "mean_sd": audit["aggregate"]["mean_sd"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
