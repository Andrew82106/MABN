"""Post-hoc calibration diagnostic of risk-run onset vs continuation.

This script is descriptive only.  It must not select a model, feature, threshold,
or official-test decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCORES = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
WINDOWS = ROOT / "data/windows_k4_calibration.jsonl"
FIT_WINDOW_COUNT = 168_123
CAL_WINDOW_COUNT = 42_241
FROZEN_THRESHOLD = 0.657469850500832


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    z = np.load(SCORES)
    scores = np.asarray(z["window_scores"], dtype=np.float64)
    assert scores.shape == (FIT_WINDOW_COUNT + CAL_WINDOW_COUNT,)
    scores = scores[FIT_WINDOW_COUNT:]
    rows = [json.loads(line) for line in WINDOWS.open(encoding="utf-8")]
    assert len(rows) == CAL_WINDOW_COUNT == len(scores)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    assert int(labels.sum()) == 5_984

    kind = np.full(len(rows), "clean", dtype=object)
    by_answer: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_answer.setdefault(str(row["response_id"]), []).append(i)

    runs: list[list[int]] = []
    for indices in by_answer.values():
        p = 0
        while p < len(indices):
            if labels[indices[p]] == 0:
                p += 1
                continue
            q = p + 1
            while (
                q < len(indices)
                and labels[indices[q]] == 1
                and rows[indices[q]]["token_start"] == rows[indices[q - 1]]["token_start"] + 1
            ):
                q += 1
            run = indices[p:q]
            runs.append(run)
            kind[run[0]] = "onset1"
            kind[run[1:4]] = "early2_4"
            kind[run[4:]] = "continuation5plus"
            p = q

    strata = {}
    for name in ("onset1", "early2_4", "continuation5plus"):
        pos = kind == name
        neg = kind == "clean"
        eval_mask = pos | neg
        strata[name] = {
            "n": int(pos.sum()),
            "recall_at_frozen_incumbent_threshold": float((scores[pos] >= FROZEN_THRESHOLD).mean()),
            "score_mean": float(scores[pos].mean()),
            "score_median": float(np.median(scores[pos])),
            "one_vs_clean_AUROC": float(roc_auc_score(pos[eval_mask].astype(np.int8), scores[eval_mask])),
            "one_vs_clean_AP": float(average_precision_score(pos[eval_mask].astype(np.int8), scores[eval_mask])),
        }

    clean = kind == "clean"
    lengths = np.asarray([len(run) for run in runs], dtype=np.int64)
    first_detected_offsets = []
    fully_missed = 0
    fully_missed_windows = 0
    for run in runs:
        hits = np.flatnonzero(scores[run] >= FROZEN_THRESHOLD)
        if len(hits) == 0:
            fully_missed += 1
            fully_missed_windows += len(run)
        else:
            first_detected_offsets.append(int(hits[0]))

    result = {
        "status": "complete_posthoc_diagnostic_only",
        "scope": "repeatedly-used calibration only; no fit selection and no test access",
        "forbidden_interpretation": "Not an unbiased estimate and not permission to select an onset model on calibration.",
        "incumbent": "semantic_claim__old_tree__large_weight0.4",
        "frozen_threshold": FROZEN_THRESHOLD,
        "counts": {
            "answers": len(by_answer),
            "eligible_windows": len(rows),
            "risk_windows": int(labels.sum()),
            "risk_runs": len(runs),
            "fully_missed_runs": fully_missed,
            "fully_missed_run_windows": fully_missed_windows,
            "run_length_min": int(lengths.min()),
            "run_length_median": float(np.median(lengths)),
            "run_length_mean": float(lengths.mean()),
            "run_length_max": int(lengths.max()),
        },
        "strata": strata,
        "clean": {
            "n": int(clean.sum()),
            "false_positive_rate_at_frozen_incumbent_threshold": float((scores[clean] >= FROZEN_THRESHOLD).mean()),
            "score_mean": float(scores[clean].mean()),
        },
        "run_detection": {
            "any_window_recall": float((len(runs) - fully_missed) / len(runs)),
            "first_detected_offset_median_among_detected": float(np.median(first_detected_offsets)),
            "first_detected_offset_mean_among_detected": float(np.mean(first_detected_offsets)),
        },
        "f1_075_increment_budget_from_current_counts": {
            "current_tp_fp_fn": [3_839, 1_300, 2_145],
            "minimum_added_tp_if_zero_added_fp": 532,
            "if_added_tp_900_max_added_fp": 614.3333333333334,
            "interpretation": "Arithmetic budget only; it does not assume an achievable onset detector.",
        },
        "input_sha256": {str(SCORES.relative_to(ROOT)): sha256(SCORES), str(WINDOWS.relative_to(ROOT)): sha256(WINDOWS)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# 当前候选的错误起点诊断

这是反复使用过的 calibration 上的**事后原因诊断**，没有读取新测试集，不能据此选择模型或声称泛化提升。

- 170 个风险窗口连续段中，{fully_missed} 个整段完全漏报，共含 {fully_missed_windows} 个风险窗口。
- 第一个风险窗口召回：{strata['onset1']['recall_at_frozen_incumbent_threshold']:.4f}。
- 第 2--4 个风险窗口召回：{strata['early2_4']['recall_at_frozen_incumbent_threshold']:.4f}。
- 第 5 个及以后风险窗口召回：{strata['continuation5plus']['recall_at_frozen_incumbent_threshold']:.4f}。
- 干净窗口误报率：{result['clean']['false_positive_rate_at_frozen_incumbent_threshold']:.4f}。
- 从当前 TP/FP/FN 出发，若新增 900 个正确风险窗，最多可同时新增约 614 个误报窗仍达到 F1=0.75；这是算术预算，不是模型成绩。

当前候选明显更容易在错误已经延续后报警。后续可在 fit/train-only 数据上独立设计“起点头＋延续头”，但任何结构、阈值和训练方案必须先于 calibration 结果冻结。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(json.dumps(result["strata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
