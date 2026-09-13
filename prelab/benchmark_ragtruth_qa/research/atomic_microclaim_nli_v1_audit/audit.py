"""Post-hoc error and geometry audit for atomic_microclaim_nli_v1."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_atomic_microclaim_nli_v1 as method  # noqa: E402
import run_development as q  # noqa: E402

OUT = HERE


def save(path, value):
    path = Path(path); assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def type_recall(meta, prediction, partition):
    left, right = meta["bounds"][partition]
    answers = {answer["response_id"]: answer for answer in meta["answers"]}
    total, hit = defaultdict(int), defaultdict(int)
    for wi in range(left, right):
        window = meta["windows"][wi]
        if not window["label"]: continue
        labels = answers[window["response_id"]]["original_labels"]
        kinds = set()
        for label in labels:
            if any(max(span["start"], label["start"]) < min(span["end"], label["end"])
                   for span in window["risk_character_spans"]):
                kinds.add(label["label_type"])
        for kind in kinds:
            total[kind] += 1; hit[kind] += int(prediction[wi])
    return {kind: {"positive_windows": total[kind], "hits": hit[kind],
                   "recall": hit[kind] / total[kind]} for kind in sorted(total)}


def main():
    assert q.read(method.OUT / "INDEPENDENT_FINAL_AUDIT.json")["status"] == "passed_independent_final_audit"
    meta = q.metadata(); rows = method.json_lines(method.OUT / "inputs.jsonl")
    summary = q.read(method.OUT / "summary.json")
    with np.load(method.OUT / "scores.npz", allow_pickle=False) as data:
        score = data["window_scores"].astype(np.float64)
    incumbent_result = q.read(method.INCUMBENT_RESULT)
    with np.load(method.INCUMBENT_SCORES, allow_pickle=False) as data:
        incumbent_score = data["window_scores"].astype(np.float64)
    threshold = summary["thresholds_from_fit_OOF"]["window"]["threshold"]
    incumbent_threshold = incumbent_result["thresholds"]["window"]["threshold"]
    pred = score >= threshold; incumbent_pred = incumbent_score >= incumbent_threshold
    cal_left, cal_right = meta["bounds"]["calibration"]
    cal = slice(cal_left, cal_right)
    y = np.asarray([window["label"] for window in meta["windows"]], dtype=bool)

    claim_y = np.load(method.OUT / "microclaim_labels.npy").astype(np.float64)
    offsets = {}; cursor = 0
    for row in rows: offsets[row["response_id"]] = cursor; cursor += len(row["claims"])
    oracle_score = method.project(meta, rows, offsets, claim_y)
    oracle_thresholds = {"window": {"threshold": .5}, "answer": {"threshold": .5}}
    oracle_metrics = q.metrics(meta, oracle_score, oracle_thresholds)

    relation_masks = {name: np.zeros(len(meta["windows"]), dtype=bool) for name in method.atomic.FEATURE_NAMES}
    lookup = {row["response_id"]: row for row in rows}
    for answer in meta["answers"]:
        row = lookup[answer["response_id"]]; owners = row["lexical_token_microclaims"]
        for wi in meta["answer_windows"][answer["response_id"]]:
            cids = {cid for ti in meta["windows"][wi]["token_indices"] for cid in owners[ti]}
            for name in relation_masks:
                relation_masks[name][wi] = any(row["claims"][cid]["feature_vector"][name] for cid in cids)

    subgroup = {}
    for name, mask in relation_masks.items():
        active = mask[cal]
        cy, cp, cip = y[cal][active], pred[cal][active], incumbent_pred[cal][active]
        subgroup[name] = {
            "windows": int(active.sum()), "positive": int(cy.sum()),
            "new_hits": int(np.count_nonzero(cy & cp)), "new_false_positives": int(np.count_nonzero(~cy & cp)),
            "incumbent_hits": int(np.count_nonzero(cy & cip)), "incumbent_false_positives": int(np.count_nonzero(~cy & cip)),
        }
    cy, cp, cip = y[cal], pred[cal], incumbent_pred[cal]
    comparison = {
        "new_true_positive_not_incumbent": int(np.count_nonzero(cy & cp & ~cip)),
        "incumbent_true_positive_not_new": int(np.count_nonzero(cy & cip & ~cp)),
        "new_false_positive_not_incumbent": int(np.count_nonzero(~cy & cp & ~cip)),
        "incumbent_false_positive_not_new": int(np.count_nonzero(~cy & cip & ~cp)),
        "both_true_positive": int(np.count_nonzero(cy & cp & cip)),
        "both_false_positive": int(np.count_nonzero(~cy & cp & cip)),
    }
    union_score = (pred | incumbent_pred).astype(np.float64)
    intersection_score = (pred & incumbent_pred).astype(np.float64)
    comparison["posthoc_union_cal"] = q.count(cy, union_score[cal], .5)
    comparison["posthoc_intersection_cal"] = q.count(cy, intersection_score[cal], .5)
    result = {
        "status": "complete_posthoc_development_error_audit",
        "new_strict_cal": summary["strict_fit_threshold_to_calibration"]["calibration"],
        "incumbent_cal": incumbent_result["metrics"]["calibration"],
        "comparison": comparison,
        "label_type_recall": {
            "new": type_recall(meta, pred, "calibration"),
            "incumbent": type_recall(meta, incumbent_pred, "calibration"),
        },
        "relation_subgroups_calibration": subgroup,
        "atomic_geometry_gold_microclaim_oracle": oracle_metrics,
        "interpretation_limits": [
            "Oracle uses gold-derived microclaim labels and measures geometry only, not deployable prediction.",
            "Union/intersection and subgroup comparisons are post-hoc calibration diagnostics, not fit-selected candidates.",
            "The incumbent threshold has repeatedly used calibration and is only a fixed development reference.",
        ],
        "formal_baselines_modified": False, "official_test_opened": False,
        "source_sha256": {
            str((method.OUT / "summary.json").relative_to(ROOT)): q.sha(method.OUT / "summary.json"),
            str((method.OUT / "scores.npz").relative_to(ROOT)): q.sha(method.OUT / "scores.npz"),
            str(method.INCUMBENT_SCORES.relative_to(ROOT)): q.sha(method.INCUMBENT_SCORES),
            str(method.INCUMBENT_RESULT.relative_to(ROOT)): q.sha(method.INCUMBENT_RESULT),
        },
    }
    save(OUT / "AUDIT.json", result)
    nr, ir = result["label_type_recall"]["new"], result["label_type_recall"]["incumbent"]
    lines = [
        "# Atomic microclaim NLI v1 error audit", "",
        f"新方法严格cal窗口F1 {result['new_strict_cal']['windows']['f1']:.6f}，现有模型 {result['incumbent_cal']['windows']['f1']:.6f}。",
        f"新方法独有真阳性 {comparison['new_true_positive_not_incumbent']}，但独有误报 {comparison['new_false_positive_not_incumbent']}；现有模型独有真阳性 {comparison['incumbent_true_positive_not_new']}。", "",
        "| 标签类型 | 新方法召回 | 现有模型召回 |", "|---|---:|---:|",
    ]
    for kind in nr:
        lines.append(f"| {kind} | {nr[kind]['recall']:.4f} ({nr[kind]['hits']}/{nr[kind]['positive_windows']}) | {ir[kind]['recall']:.4f} ({ir[kind]['hits']}/{ir[kind]['positive_windows']}) |")
    oracle = oracle_metrics["calibration"]
    lines += ["", f"若直接知道每个微主张的gold标签，按同一窗口投影的几何上限为窗口F1 {oracle['windows']['f1']:.6f}、整答F1 {oracle['answers']['f1']:.6f}。",
              "", "结论：原子切分的定位几何足够；失败来自风险判别信号的精度，尤其新增召回伴随过多误报。union/intersection只作事后诊断，不能作为新结果。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparison": comparison,
                      "oracle_cal": oracle, "type_recall": nr}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
