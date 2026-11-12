"""Grouped next-stage native-policy risk evaluation on real HarnessAudit runs.

This is a development audit of API-visible, stage-start prediction.  It keeps
all stages from one task in one fold, uses no ORACLE or future event fields in
features, and preserves incomplete runs as excluded/unknown.  The target is
the next stage's observed native rule code, not actual harm or task success.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.native_stage_forecast import (
    CODES, EXECUTION_INDICES, KNOWLEDGE_INDICES, NAMES, RELATION_INDICES,
    NativePolicy, NativeStageForecastBN, independent_same_marginals, public_features,
)
sys.path.insert(0, str(ROOT / 'code/external/HarnessAudit'))
from multi_agent.recognizer import contains


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def completed_cases(root):
    rows = []
    for case_dir in sorted((Path(root) / "cases").glob("*")):
        if not case_dir.is_dir() or not (case_dir / "RUN.json").is_file():
            continue
        run, labels = read(case_dir / "RUN.json"), read(case_dir / "LABELS.json")
        if run.get("status") != "completed":
            continue
        plan_case = next(c for c in read(Path(root) / "PLAN.json")["cases"] if c["case_id"] == case_dir.name)
        stages = []
        for path in sorted((case_dir / "stages").glob("*.json")):
            stages.append(read(path))
        actions = {a["sequence_num"]: a for a in labels["bridge"]["actions"]}
        decisions = {d["action_sequence_num"]: d for d in labels["decisions"]}
        target_by_stage = {}
        for seq, action in actions.items():
            stage = action["raw_event"].get("stage_id")
            code = decisions.get(seq, {}).get("violation_code")
            if code not in CODES:
                continue
            target_by_stage.setdefault(stage, set()).add(code)
        rules = plan_case["visible_policy"]["rules"]
        policy = NativePolicy(rules, plan_case["domain"], contains)
        examples = []
        for stage in stages:
            x, raw = public_features(plan_case, stage, policy)
            bits = np.asarray([int(code in target_by_stage.get(stage["stage_id"], set())) for code in CODES], dtype=int)
            examples.append({"case": case_dir.name, "domain": plan_case["domain"], "stage": stage["stage_id"],
                             "x": x, "raw": raw, "bits": bits, "risk": int(bits.any())})
        rows.append({"case": case_dir.name, "domain": plan_case["domain"], "examples": examples,
                     "target_stage_count": len(examples), "target_positive_stage_count": sum(e["risk"] for e in examples)})
    return rows


def weighted_fit(model, x, y, weights):
    if len(np.unique(y)) == 1:
        return float((np.dot(weights, y) + 1) / (weights.sum() + 2))
    model.fit(x, y, sample_weight=weights)
    return model


def prob(model, x):
    return np.full(len(x), model, dtype=float) if np.isscalar(model) else model.predict_proba(x)[:, 1]


def fit_predict(train, test, kind, mask=()):
    xtr = np.asarray([e["x"] for r in train for e in r["examples"]], dtype=float)
    btr = np.asarray([e["bits"] for r in train for e in r["examples"]], dtype=int)
    xte = np.asarray([e["x"] for e in test["examples"]], dtype=float)
    if mask:
        xtr[:, list(mask)] = 0.0; xte[:, list(mask)] = 0.0
    # Equal weight per independent task, not per stage window.
    weight_map = {r["case"]: 1.0 / max(1, r["target_stage_count"]) for r in train}
    wtr = np.asarray([weight_map[e["case"]] for r in train for e in r["examples"]], dtype=float)
    model = None
    if kind == "fused_bn":
        # Select a fixed mixture of knowledge-only and execution-aware
        # hierarchical scores using task-group OOF predictions from the fit
        # cohort.  The held-out task is never used to choose alpha.
        if len(train) < 4:
            raise ValueError("fused BN needs at least four training task groups")
        oof_h, oof_k, oof_y = [], [], []
        for j, valid in enumerate(train):
            inner_train = [r for i, r in enumerate(train) if i != j]
            h, _, _ = fit_predict(inner_train, valid, "hierarchical_bn")
            k, _, _ = fit_predict(inner_train, valid, "no_execution_bn")
            oof_h.extend(h); oof_k.extend(k); oof_y.extend(e["risk"] for e in valid["examples"])
        oof_h, oof_k, oof_y = np.asarray(oof_h), np.asarray(oof_k), np.asarray(oof_y)
        alphas = np.linspace(0.0, 1.0, 101)
        alpha = float(min(alphas, key=lambda a: np.mean((a*oof_h + (1-a)*oof_k - oof_y)**2)))
        h, _, hmodel = fit_predict(train, test, "hierarchical_bn")
        k, _, kmodel = fit_predict(train, test, "no_execution_bn")
        return alpha*h + (1-alpha)*k, None, {"alpha": alpha, "oof_windows": len(oof_y), "hierarchical": hmodel.parameters(), "knowledge": kmodel.parameters()}
    if kind == "hierarchical_bn":
        model = NativeStageForecastBN().fit(xtr, btr, wtr)
        joint = model.joint(xte)
    elif kind == "no_knowledge_bn":
        model = NativeStageForecastBN(zero_indices=KNOWLEDGE_INDICES).fit(xtr, btr, wtr)
        joint = model.joint(xte)
    elif kind == "no_execution_bn":
        model = NativeStageForecastBN(zero_indices=EXECUTION_INDICES).fit(xtr, btr, wtr)
        joint = model.joint(xte)
    elif kind == "no_relations_bn":
        model = NativeStageForecastBN(zero_indices=RELATION_INDICES).fit(xtr, btr, wtr)
        joint = model.joint(xte)
    elif kind == "independent_marginals":
        ps = []
        for j in range(4):
            m = weighted_fit(LogisticRegression(C=1.0, max_iter=1000), xtr, btr[:, j], wtr)
            ps.append(prob(m, xte))
        states = np.asarray([[int(b) for b in f"{i:04b}"] for i in range(16)])
        joint = np.ones((len(xte), 16))
        for i, state in enumerate(states):
            joint[:, i] = np.prod(np.where(state, np.asarray(ps).T, 1-np.asarray(ps).T), axis=1)
    elif kind == "flat_logistic":
        model = weighted_fit(LogisticRegression(C=1.0, max_iter=1000), xtr, btr.any(axis=1), wtr)
        p = prob(model, xte)
        return p, None, model
    elif kind == "flat_histgb":
        model = weighted_fit(HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=8, l2_regularization=1.0, early_stopping=False, random_state=20261001),
            xtr, btr.any(axis=1), wtr)
        p = prob(model, xte)
        return p, None, model
    else:
        raise ValueError(kind)
    return 1.0 - joint[:, 0], joint, model


def metrics(y, p):
    out = {"n": int(len(y)), "positive": int(y.sum()), "brier": float(brier_score_loss(y, p)),
           "log_loss": float(log_loss(y, np.c_[1-p,p], labels=[0,1])), "f1@0.5": float(f1_score(y, p >= .5, zero_division=0))}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p)); out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = None; out["auprc"] = None
    return out


def run(inputs, out):
    cohorts = []
    for root in inputs:
        cohorts.extend(completed_cases(root))
    # Duplicate task IDs from a rerun are not independent; keep the first cohort
    # occurrence and report the confirmation cohort separately in metadata.
    by_id = {}
    for row in cohorts:
        by_id.setdefault(row["case"], row)
    rows = list(by_id.values())
    if len(rows) < 4:
        raise ValueError("at least four completed independent task groups required")
    methods = ("fused_bn", "hierarchical_bn", "no_knowledge_bn", "no_execution_bn", "no_relations_bn",
               "independent_marginals", "flat_logistic", "flat_histgb")
    predictions = {m: [] for m in methods}
    fold_meta = []
    for held in rows:
        train = [r for r in rows if r["case"] != held["case"]]
        for method in methods:
            p, joint, model = fit_predict(train, held, method)
            y = np.asarray([e["risk"] for e in held["examples"]], dtype=int)
            predictions[method].append({"case": held["case"], "domain": held["domain"], "y": y.tolist(), "p": p.tolist(),
                                        "metrics": metrics(y, p), "n_stages": len(y), "positive_stages": int(y.sum())})
        fold_meta.append({"held_out": held["case"], "domain": held["domain"], "train_cases": [r["case"] for r in train]})
    summary = {}
    for method, folds in predictions.items():
        y = np.concatenate([np.asarray(f["y"]) for f in folds]); p = np.concatenate([np.asarray(f["p"]) for f in folds])
        rng = np.random.default_rng(20261001)
        boot = []
        for _ in range(2000):
            chosen = rng.integers(0, len(folds), size=len(folds))
            by = np.concatenate([np.asarray(folds[i]["y"]) for i in chosen])
            bp = np.concatenate([np.asarray(folds[i]["p"]) for i in chosen])
            boot.append(metrics(by, bp))
        ci = {key: [float(np.percentile([b[key] for b in boot if b[key] is not None], 2.5)),
                    float(np.percentile([b[key] for b in boot if b[key] is not None], 97.5))]
              for key in ("brier", "f1@0.5", "auroc", "auprc") if any(b[key] is not None for b in boot)}
        summary[method] = {"pooled": metrics(y,p), "cluster_bootstrap95": ci,
                           "task_mean_brier": float(np.mean([f["metrics"]["brier"] for f in folds])),
                           "task_mean_f1": float(np.mean([f["metrics"]["f1@0.5"] for f in folds]))}
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    all_examples = [e for r in rows for e in r["examples"]]
    x_all = np.asarray([e["x"] for e in all_examples], dtype=float)
    b_all = np.asarray([e["bits"] for e in all_examples], dtype=int)
    weights = np.asarray([1.0 / max(1, r["target_stage_count"]) for r in rows for _ in r["examples"]], dtype=float)
    whitebox = NativeStageForecastBN().fit(x_all, b_all, weights)
    (out / "MODEL_SPEC.json").write_text(json.dumps({"schema":"native-stage-forecast-model-spec-v1",
        "fit_scope":"all completed development tasks; descriptive artifact, not confirmation fit",
        "parameters":whitebox.parameters(), "variable_order":list(CODES), "feature_order":list(NAMES)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"schema":"api-stage-forecast-development-v1", "cohorts": [str(Path(x).resolve()) for x in inputs],
              "completed_task_groups": len(rows), "all_completed_stage_windows": sum(len(r["examples"]) for r in rows),
              "excluded_incomplete_or_failed": "All cases without RUN.status=completed were excluded from supervised metrics and retained in source cohorts.",
              "target":"Any native V-OT/V-OR/V-IC/V-ID violation in the next stage's observed actions.",
              "input_boundary":"Public stage-start packet, declared workflow/policy and observed prefix only; no ORACLE, LABELS, future actions, task labels, response IDs or model metadata.",
              "methods": list(methods), "feature_names": list(NAMES), "whitebox_model_spec":"MODEL_SPEC.json",
              "folds": fold_meta, "summary": summary, "predictions": predictions,
              "limitations":["Development-only completed HarnessAudit tasks; not an untouched final confirmation set.",
                             "Target is retrospective native policy violation, not actual harm, task success or causal discovery.",
                             "The public task/policy is structured and the stage target is actionwise; this does not establish arbitrary MAS generalization.",
                             "Same-information controls are included; a BN label alone is not expected to beat a matched logistic factor."]}
    (out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# API next-stage risk forecast (development)", "", f"- Completed task groups: **{len(rows)}**", f"- Stage windows: **{report['all_completed_stage_windows']}**", "- Target: next-stage native policy violation (V-OT/V-OR/V-IC/V-ID)", "", "| Method | AUROC | AUPRC | Brier | F1@0.5 | task-mean Brier |", "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = summary[method]; q=s["pooled"]; lines.append(f"| {method} | {q['auroc'] if q['auroc'] is not None else 'n/a'} | {q['auprc'] if q['auprc'] is not None else 'n/a'} | {q['brier']:.4f} | {q['f1@0.5']:.4f} | {s['task_mean_brier']:.4f} |")
    lines += ["", "该表只用于当前开发诊断。不同方法均使用同一阶段前可见信息；失败/未完成任务没有被当作负例。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def replicate(train_inputs, test_inputs, out):
    train_rows = []
    for root in train_inputs:
        train_rows.extend(completed_cases(root))
    test_rows = []
    for root in test_inputs:
        test_rows.extend(completed_cases(root))
    if len(train_rows) < 4 or not test_rows:
        raise ValueError("replication needs four training task groups and one test task")
    methods = ("fused_bn", "hierarchical_bn", "no_knowledge_bn", "no_execution_bn", "no_relations_bn",
               "independent_marginals", "flat_logistic", "flat_histgb")
    predictions = {m: [] for m in methods}
    for held in test_rows:
        for method in methods:
            p, _, _ = fit_predict(train_rows, held, method)
            y = np.asarray([e["risk"] for e in held["examples"]], dtype=int)
            predictions[method].append({"case": held["case"], "y": y.tolist(), "p": p.tolist(), "metrics": metrics(y,p)})
    summary = {}
    for method, folds in predictions.items():
        y=np.concatenate([np.asarray(f["y"]) for f in folds]); p=np.concatenate([np.asarray(f["p"]) for f in folds])
        summary[method] = metrics(y,p)
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    report={"schema":"api-stage-forecast-replication-v1", "train_inputs":[str(Path(x).resolve()) for x in train_inputs],
            "test_inputs":[str(Path(x).resolve()) for x in test_inputs], "train_task_groups":len(train_rows),
            "test_task_groups":len(test_rows), "target":"next-stage native policy violation",
            "input_boundary":"public stage-start evidence only", "summary":summary, "predictions":predictions,
            "limitations":["Same task IDs can occur in train and test; this is a stochastic rerun/operational replication, not task-generalization.",
                           "Development-only; target is native policy violation, not actual harm."]}
    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# API next-stage risk forecast (rerun replication)","",f"- Training task groups: **{len(train_rows)}**",f"- Rerun task groups: **{len(test_rows)}**","","| Method | AUROC | AUPRC | Brier | F1@0.5 |","|---|---:|---:|---:|---:|"]
    for m in methods:
        q=summary[m]; lines.append(f"| {m} | {q['auroc'] if q['auroc'] is not None else 'n/a'} | {q['auprc'] if q['auprc'] is not None else 'n/a'} | {q['brier']:.4f} | {q['f1@0.5']:.4f} |")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", type=Path)
    ap.add_argument("--train-input", nargs="+", type=Path)
    ap.add_argument("--test-input", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.train_input or args.test_input:
        if not args.train_input or not args.test_input:
            ap.error("--train-input and --test-input must be provided together")
        report = replicate(args.train_input, args.test_input, args.output)
    else:
        if not args.input:
            ap.error("--input is required unless replication inputs are given")
        report = run(args.input, args.output)
    print(json.dumps({"output":str(args.output),"mode":report["schema"]}, ensure_ascii=False))
