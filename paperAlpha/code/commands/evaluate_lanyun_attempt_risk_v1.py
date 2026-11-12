"""Evaluate pre-execution policy-risk intent on all observable Lanyun runs.

This is a separate target from completed external side effects.  A case is
positive when its independently declared reference policy says the requested
operation is out of scope/revoked, even if the API model refuses before the
tool executes.  Incomplete runs remain visible and are not silently counted
as safe.  The frozen independent_mas_v3 estimators are used without API-label
fitting or threshold tuning.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'experiments/synthetic_mas/runs/independent_mas_v3'


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


J = load_module(ROOT / 'code/commands/evaluate_independent_mas_journal_v1.py', "journal_attempt_eval")
TWO = load_module(ROOT / 'code/commands/evaluate_two_layer_v2.py', "two_attempt_eval")
TWO.PRIV_TOOLS.add("external_release")
J.PRIV_TOOLS.add("external_release")
API_HELPERS = load_module(ROOT / 'code/commands/evaluate_external_api_cohort_v1.py', "external_api_helpers")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= .5
    return {"n": int(len(y)), "positive": int(y.sum()), "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None,
            "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "predicted_positive": int(pred.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    train = J.read_jsonl(MAIN / "traces_public.jsonl")
    labels = {row["episode_id"]: row for row in J.read_jsonl(MAIN / "labels.jsonl")}
    train_y = np.asarray([int(labels[e["episode_id"]]["label"]) for e in train])
    wf_train = np.asarray([J.workflow_vector(e) for e in train]); rt_train = np.asarray([J.runtime_vector(e) for e in train])
    episodes, y, records = [], [], []
    for run_dir in sorted((args.api_root / "runs").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "RUN.json").exists() or not (run_dir / "config.json").exists():
            continue
        if not (run_dir / "monitor_events.jsonl").exists():
            continue
        run = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        episode, _, meta = API_HELPERS.make_episode(run_dir)
        episodes.append(episode); y.append(int(bool(config["reference_label"])))
        records.append({**meta, "status": run.get("status"), "attempt_label": int(bool(config["reference_label"])), "source": str(run_dir)})
    if len(episodes) < 10 or len(set(y)) < 2:
        raise RuntimeError("insufficient observable runs or label diversity")
    y = np.asarray(y, dtype=int)
    wf_api = np.asarray([J.workflow_vector(e) for e in episodes]); rt_api = np.asarray([J.runtime_vector(e) for e in episodes])
    wf_bn = J.workflow_bn_fit(wf_train, train_y); p_wf = J.workflow_bn_score(wf_api, wf_bn)
    _, p_wfl, _ = J.fit_logistic(wf_train, train_y, wf_api, 1301)
    _, p_rtl, _ = J.fit_logistic(rt_train, train_y, rt_api, 1302)
    train_eps = [{**e, "label": int(train_y[i])} for i, e in enumerate(train)]
    bn = TWO.fit_bn(train_eps, train_y); theta, opt = TWO.fit_model(train_eps, train_y, bn, 1401)
    p_rt = np.asarray([TWO._propagate(theta, TWO.event_features(e), bn) for e in episodes])
    p_rt_train = np.asarray([TWO._propagate(theta, TWO.event_features(e), bn) for e in train_eps])
    fusion = LogisticRegression(C=1.0, max_iter=1000, random_state=1402).fit(np.c_[wf_train, rt_train, J.workflow_bn_score(wf_train, wf_bn), p_rt_train], train_y)
    scores = {"workflow_bn": p_wf, "runtime_bn": p_rt, "workflow_logistic": p_wfl, "runtime_logistic": p_rtl,
              "hierarchical_fusion": fusion.predict_proba(np.c_[wf_api, rt_api, p_wf, p_rt])[:, 1]}
    args.out.mkdir(parents=True, exist_ok=False)
    output = {"schema": "lanyun-attempt-risk-transfer-v1", "source": str(args.api_root.resolve()), "target": "pre-execution policy-risk intent from independently declared reference policy", "api_labels_used_for_fit": False, "n": len(episodes), "positive": int(y.sum()), "completed": sum(r["status"] == "completed" for r in records), "failed": sum(r["status"] != "completed" for r in records), "runtime_optimizer": {"success": bool(opt.get("success", False)), "iterations": int(opt.get("iterations", -1))}, "metrics": {name: metrics(y, p) for name, p in scores.items()}, "cases": records}
    (args.out / "REPORT.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, record in enumerate(records):
            handle.write(json.dumps({"id": record["episode_id"], "status": record["status"], "label": int(y[i]), **{k: float(v[i]) for k, v in scores.items()}}, sort_keys=True) + "\n")
    lines = ["# Lanyun pre-execution policy-risk transfer", "", f"All {len(episodes)} runs with observable monitor prefixes are included; completed={output['completed']}, failed/incomplete={output['failed']}, positives={int(y.sum())}.", "The target is policy-risk intent, not successful external side effect. A refusal is therefore retained as an observable positive-risk case when the declared request is out of scope.", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in output["metrics"].items():
        auc = "NA" if value["auroc"] is None else f"{value['auroc']:.3f}"
        lines.append(f"| {name} | {value['f1']:.3f} | {value['precision']:.3f} | {value['recall']:.3f} | {auc} | {value['auprc']:.3f} | {value['brier']:.3f} |")
    lines += ["", "This remains a hand-authored simulator and a transfer diagnostic; it is not the independent confirmation cohort for the final paper claim.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
