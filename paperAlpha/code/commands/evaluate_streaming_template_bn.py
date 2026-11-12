"""Matched-evidence, three-way group evaluation of the reusable streaming BN.

Existing synthetic v3 data is a DEVELOPMENT set, already inspected previously.
No novelty, real-world superiority, or independent safety guarantee follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.streaming_bn import (  # noqa: E402
    CELL_NAMES, TemplateBN, replay, topology_group, reject_outcomes,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_groups(groups: np.ndarray, seed: int, test_fraction=.2, val_fraction=.25):
    indices = np.arange(len(groups))
    remaining, test = next(GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed,
    ).split(indices, groups=groups))
    fit_pos, val_pos = next(GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed + 1,
    ).split(remaining, groups=groups[remaining]))
    fit, val = remaining[fit_pos], remaining[val_pos]
    for left, right in ((fit, val), (fit, test), (val, test)):
        assert not set(groups[left]).intersection(groups[right])
    return fit, val, test


def choose_threshold(y, p):
    # Include an all-negative choice; deterministic high-threshold tie break.
    candidates = np.unique(np.r_[p, 0.0, np.nextafter(1.0, 2.0)])
    return float(max(candidates, key=lambda t: (f1_score(y, p >= t, zero_division=0), t)))


def metrics(y, p, threshold):
    pred = p >= threshold
    negatives = y == 0
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "brier": float(brier_score_loss(y, p)), "ece": float(ece),
        "false_positive_rate": float(pred[negatives].mean()) if negatives.any() else None,
        "threshold": float(threshold), "n": len(y), "positive": int(y.sum()),
    }


def warning_proxy(labels, scores_by_episode, traces, threshold):
    """Only dispatch-relative diagnostics, never a fabricated harm timestamp."""
    early, eligible, negative_alarms, negative_count, leads = 0, 0, 0, 0, []
    for y, probabilities, trace in zip(labels, scores_by_episode, traces):
        alarm = next((k for k, p in enumerate(probabilities) if p >= threshold), None)
        dispatch = next((k for k, e in enumerate(trace["events"])
                         if e.get("event_type") == "tool_call"
                         and e.get("tool") in {"shell.exec", "payments.write", "filesystem.write"}), None)
        if y == 0:
            negative_count += 1
            negative_alarms += int(alarm is not None)
        elif dispatch is not None:
            eligible += 1
            if alarm is not None and alarm < dispatch:  # exclude the dispatch itself
                early += 1
                leads.append(dispatch - alarm)
    return {
        "actual_violation_lead_time": None,
        "missing_reference": "no independent violation timestamp in dataset",
        "proxy": "before first observed privileged dispatch, NOT before proven harm",
        "positive_with_dispatch": eligible, "positive_alarm_before_dispatch": early,
        "pre_dispatch_recall": early / eligible if eligible else None,
        "negative_episode_any_alarm_rate": negative_alarms / negative_count if negative_count else None,
        "lead_event_count_conditional_on_early_alarm": float(np.mean(leads)) if leads else None,
        "threshold_scope": "chosen for terminal validation F1, not calibrated for sequential false alarms",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / 'experiments/model_mechanisms/configs/streaming_template_bn_v1.json')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/streaming_template_bn_20260919')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = ROOT / config["data"]
    trace_file, label_file = data / "traces_public.jsonl", data / "labels.jsonl"
    traces, label_rows = read_jsonl(trace_file), read_jsonl(label_file)
    lookup = {row["episode_id"]: row for row in label_rows}
    if len(lookup) != len(label_rows) or len({e["episode_id"] for e in traces}) != len(traces):
        raise ValueError("duplicate episode IDs")
    if set(lookup) != {e["episode_id"] for e in traces}:
        raise ValueError("trace/label IDs must match exactly")
    for trace in traces:
        reject_outcomes(trace)
    y = np.asarray([lookup[e["episode_id"]]["label"] for e in traces], dtype=int)
    groups = np.asarray([topology_group(e["workflow_dag"]) for e in traces])
    mechanisms = np.asarray([lookup[e["episode_id"]].get("mechanism", "unknown") for e in traces])
    # Freeze inputs/code/config and splits BEFORE fitting or reporting test metrics.
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {args.out}")
    args.out.mkdir(parents=True)
    sources = [Path(__file__), args.config, *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"),
               ROOT / 'knowledge/templates/risk_templates_v1.json', ROOT / 'knowledge/risks/risk_catalog.json']
    for path in sources:
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    manifest = {
        "scope": "reused synthetic development holdouts, not confirmatory results",
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (trace_file, label_file)},
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "config": config, "n": len(traces), "group_count": len(set(groups)),
        "model_feature_contract": "identical named evidence vectors for BN/logistic/HistGB; masked permissions removed for ALL",
        "split_metadata_is_feature": False, "splits": {},
    }
    splits = {}
    for seed in config["seeds"]:
        fit, val, test = split_groups(groups, seed, config["outer_test_fraction"],
                                      config["validation_fraction_of_remaining"])
        for part in (fit, val, test):
            if len(np.unique(y[part])) != 2:
                raise ValueError("a preregistered partition has only one class; do not select another seed silently")
        splits[seed] = fit, val, test
        manifest["splits"][str(seed)] = {
            name: {"episode_ids": [traces[i]["episode_id"] for i in part],
                   "groups": sorted(set(groups[part])), "positive": int(y[part].sum()),
                   "agent_counts": dict(__import__("collections").Counter(str(len(traces[i]["agents"])) for i in part))}
            for name, part in (("fit", fit), ("validation", val), ("test", test))
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    runs = []
    for setting in config["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        trajectories = [
            replay(trace, workflow=trace["workflow_dag"] if setting["workflow"] == "workflow_known" else None,
                   permission_mode=setting["permission"]) for trace in traces
        ]
        if any(not trajectory for trajectory in trajectories):
            raise ValueError("empty trace requires an explicit initial-state evaluation policy")
        final = [trajectory[-1] for trajectory in trajectories]
        x = np.asarray([s.vector() for s in final])
        for seed, (fit, val, test) in splits.items():
            print(f"running {key} seed={seed} fit/val/test={len(fit)}/{len(val)}/{len(test)}", flush=True)
            run_dir = args.out / key / str(seed)
            run_dir.mkdir(parents=True)
            predictions, fitted, result = {}, {}, {}
            parameters = {}
            for exclusion in (None, *config["ablations"]):
                name = "template_bn" if exclusion is None else "without_" + exclusion
                model = TemplateBN(exclude=frozenset({exclusion}) if exclusion else frozenset(),
                                   regularization=config["training"]["regularization"])
                model.fit([final[i] for i in fit], y[fit])
                threshold = choose_threshold(y[val], model.predict([final[i] for i in val]))
                p_test = model.predict([final[i] for i in test])
                result[name] = metrics(y[test], p_test, threshold)
                result[name]["optimizer_success"] = model.optimizer["success"]
                predictions[name], fitted[name] = p_test, model
                parameters[name] = model.parameters()
            for name, estimator in (
                ("logistic_same_evidence", make_pipeline(StandardScaler(), LogisticRegression(
                    **config["baselines"]["logistic_same_evidence"], random_state=seed))),
                ("histgb_same_evidence", HistGradientBoostingClassifier(
                    **config["baselines"]["histgb_same_evidence"], random_state=seed)),
            ):
                estimator.fit(x[fit], y[fit])
                threshold = choose_threshold(y[val], estimator.predict_proba(x[val])[:, 1])
                p_test = estimator.predict_proba(x[test])[:, 1]
                result[name] = metrics(y[test], p_test, threshold)
                predictions[name] = p_test
            full = fitted["template_bn"]
            prefix_probabilities = [full.predict(trajectories[i]) for i in test]
            warning = warning_proxy(y[test], prefix_probabilities, [traces[i] for i in test],
                                    result["template_bn"]["threshold"])
            by_mechanism = {
                str(mechanism): {
                    name: metrics(y[test][mechanisms[test] == mechanism], p[mechanisms[test] == mechanism], result[name]["threshold"])
                    for name, p in predictions.items()
                } for mechanism in sorted(set(mechanisms[test]))
            }
            run = {"setting": key, "seed": seed, "metrics": result,
                   "warning_proxy": warning, "mechanism_metrics": by_mechanism}
            runs.append(run)
            (run_dir / "metrics.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
            (run_dir / "model_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for pos, i in enumerate(test):
                    handle.write(json.dumps({
                        "episode_id": traces[i]["episode_id"], "label": int(y[i]), "group": groups[i],
                        "scores": {name: float(values[pos]) for name, values in predictions.items()},
                        "prefix_scores": prefix_probabilities[pos].tolist(),
                    }) + "\n")
            examples = [{"episode_id": traces[i]["episode_id"],
                         "prediction": full.explain(final[i])} for i in test[:10]]
            (run_dir / "explanations.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
            print({name: round(m["auroc"], 4) for name, m in result.items()}, flush=True)
    summaries = {}
    for setting in sorted({r["setting"] for r in runs}):
        subset = [r for r in runs if r["setting"] == setting]
        summaries[setting] = {}
        for method in subset[0]["metrics"]:
            summaries[setting][method] = {
                metric: {
                    "mean": float(np.mean([r["metrics"][method][metric] for r in subset])),
                    "sd": float(np.std([r["metrics"][method][metric] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                } for metric in ("f1", "auroc", "auprc", "brier", "ece")
            }
    payload = {
        "summaries": summaries, "runs": runs,
        "limitations": [
            "Synthetic v3 was previously inspected; every result is development evidence.",
            "Episode supervision does not identify true per-node or per-mechanism risk labels.",
            "Graph witnesses are chronological and explicit, not learned causal edges.",
            "Template-OR assumes conditional independence; shared evidence can violate it.",
            "No independent violation timestamp: dispatch-relative proxy is not harm lead time.",
            "F1 threshold comes from validation; sequential false alarms are not calibrated.",
            "No raw-text semantic extractor or original published MAS baseline reproduction here.",
            "Multiple random holdouts overlap; SD is split sensitivity, not an independent-trial CI.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Streaming conditional template BN: development evaluation", "",
             "真实逐事件输入；CPT 用任务标签联合学习；fit / validation / test 按公开工作流拓扑隔离。所有基线获得同一份证据。", "",
             "这是已反复检查过的合成开发集，不能作为独立论文验收集。", ""]
    for setting, methods in summaries.items():
        lines += [f"## {setting}", "", "| method | F1 mean±sd | AUROC mean±sd | AUPRC | Brier |",
                  "|---|---:|---:|---:|---:|"]
        for method, m in methods.items():
            lines.append(f"| {method} | {m['f1']['mean']:.3f}±{m['f1']['sd']:.3f} | "
                         f"{m['auroc']['mean']:.3f}±{m['auroc']['sd']:.3f} | "
                         f"{m['auprc']['mean']:.3f} | {m['brier']['mean']:.3f} |")
        lines.append("")
    lines += ["## 预警与解释边界", "",
              "- 没有独立违规时间戳，真实提前量为不可评估；仅报告首个高权限调用之前的诊断。",
              "- 前缀分数不能查看后续输入、Agent、工具或消息；旧 temporal v1 的 sink 激活时间不符合这一要求。",
              "- 解释是模型分数的精确分解，不是真实因果归因或节点风险真值。",
              "- 权限观测、规范 DAG 均分 setting；未知权限不等于授权，未知 DAG 不等于所有通信都违规。",
              "- 参数、阈值、样本标识、输入和代码哈希随结果保存；此次未调用大模型或重跑真实 Agent。"]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("completed " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
