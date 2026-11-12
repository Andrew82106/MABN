"""Post-hoc development audit for microclaim_crossencoder_v1."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q
import run_microclaim_crossencoder_v1 as method


def save(path, value):
    path = Path(path); assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def type_recall(meta, prediction):
    left, right = meta["bounds"]["calibration"]
    answers = {answer["response_id"]: answer for answer in meta["answers"]}
    total, hit = defaultdict(int), defaultdict(int)
    for wi in range(left, right):
        window = meta["windows"][wi]
        if not window["label"]: continue
        kinds = set()
        for label in answers[window["response_id"]]["original_labels"]:
            if any(max(span["start"], label["start"]) < min(span["end"], label["end"])
                   for span in window["risk_character_spans"]):
                kinds.add(label["label_type"])
        for kind in kinds:
            total[kind] += 1; hit[kind] += int(prediction[wi])
    return {kind: {"positive_windows": total[kind], "hits": hit[kind],
                   "recall": hit[kind] / total[kind]} for kind in sorted(total)}


def compare(y, candidate, reference):
    return {
        "candidate_true_positive_not_reference": int(np.count_nonzero(y & candidate & ~reference)),
        "reference_true_positive_not_candidate": int(np.count_nonzero(y & reference & ~candidate)),
        "candidate_false_positive_not_reference": int(np.count_nonzero(~y & candidate & ~reference)),
        "reference_false_positive_not_candidate": int(np.count_nonzero(~y & reference & ~candidate)),
        "both_true_positive": int(np.count_nonzero(y & candidate & reference)),
        "both_false_positive": int(np.count_nonzero(~y & candidate & reference)),
    }


def main():
    complete = q.read(method.OUT / "complete.json")
    assert complete["status"] == "complete_development_only"
    summary = q.read(method.OUT / "summary.json"); meta = q.metadata()
    with np.load(method.OUT / "scores.npz", allow_pickle=False) as z:
        main_score = z["main_window_scores"].astype(np.float64)
        main_claim = z["main_claim_scores"].astype(np.float64)
        control_claim = z["frozen_claim_scores"].astype(np.float64)
    old_summary = q.read(method.ATOMIC / "summary.json")
    with np.load(method.ATOMIC / "scores.npz", allow_pickle=False) as z:
        old_score = z["window_scores"].astype(np.float64)
    incumbent_result = q.read(method.INCUMBENT)
    incumbent_scores_path = method.ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
    with np.load(incumbent_scores_path, allow_pickle=False) as z:
        incumbent_score = z["window_scores"].astype(np.float64)

    main_t = summary["trained_candidate"]["thresholds_from_fit_OOF"]["window"]["threshold"]
    old_t = old_summary["thresholds_from_fit_OOF"]["window"]["threshold"]
    incumbent_t = incumbent_result["thresholds"]["window"]["threshold"]
    main_pred = main_score >= main_t; old_pred = old_score >= old_t; incumbent_pred = incumbent_score >= incumbent_t
    left, right = meta["bounds"]["calibration"]; cal = slice(left, right)
    y_window = np.asarray([row["label"] for row in meta["windows"]], dtype=bool)
    cy = y_window[cal]

    arrays = method.load_arrays(); fit_n = method.EXPECTED_CLAIMS["fit"]
    y_claim = arrays["labels"]; fit = slice(0, fit_n); cal_claim = slice(fit_n, len(y_claim))
    claim_ranking = {}
    for name, values in (("trained", main_claim), ("frozen_control", control_claim)):
        claim_ranking[name] = {
            "fit": {"auroc": float(roc_auc_score(y_claim[fit], values[fit])),
                    "average_precision": float(average_precision_score(y_claim[fit], values[fit]))},
            "calibration": {"auroc": float(roc_auc_score(y_claim[cal_claim], values[cal_claim])),
                            "average_precision": float(average_precision_score(y_claim[cal_claim], values[cal_claim]))},
        }

    old_vs_inc = compare(cy, old_pred[cal], incumbent_pred[cal])
    main_vs_inc = compare(cy, main_pred[cal], incumbent_pred[cal])
    main_vs_old = compare(cy, main_pred[cal], old_pred[cal])
    old_extra_fp = old_vs_inc["candidate_false_positive_not_reference"]
    main_extra_fp = main_vs_inc["candidate_false_positive_not_reference"]
    result = {
        "status": "complete_posthoc_development_audit",
        "strict_calibration": {
            "trained": summary["trained_candidate"]["strict_fit_threshold_to_calibration"]["calibration"],
            "frozen_control": summary["minimal_frozen_control"]["strict_fit_threshold_to_calibration"]["calibration"],
            "old_atomic_NLI": old_summary["strict_fit_threshold_to_calibration"]["calibration"],
            "incumbent": incumbent_result["metrics"]["calibration"],
        },
        "window_comparison": {"old_atomic_vs_incumbent": old_vs_inc,
                              "trained_vs_incumbent": main_vs_inc,
                              "trained_vs_old_atomic": main_vs_old,
                              "old_extra_false_positives": old_extra_fp,
                              "trained_extra_false_positives": main_extra_fp,
                              "extra_false_positive_reduction": old_extra_fp - main_extra_fp,
                              "extra_false_positive_reduction_fraction": (old_extra_fp - main_extra_fp) / old_extra_fp if old_extra_fp else None},
        "label_type_recall": {"trained": type_recall(meta, main_pred),
                              "old_atomic_NLI": type_recall(meta, old_pred),
                              "incumbent": type_recall(meta, incumbent_pred)},
        "microclaim_ranking": claim_ranking,
        "limits": ["All calibration comparisons are development diagnostics, not official test claims.",
                   "Unique TP/FP comparisons use each method's frozen fit-derived threshold.",
                   "The full-evidence verifier is a candidate component; this audit does not alter or weaken any formal baseline."],
        "formal_baselines_modified": False, "official_test_opened": False,
        "source_sha256": {"candidate_summary": q.sha(method.OUT / "summary.json"),
                          "candidate_scores": q.sha(method.OUT / "scores.npz"),
                          "old_atomic_summary": q.sha(method.ATOMIC / "summary.json"),
                          "old_atomic_scores": q.sha(method.ATOMIC / "scores.npz"),
                          "incumbent_result": q.sha(method.INCUMBENT),
                          "incumbent_scores": q.sha(incumbent_scores_path)},
    }
    save(HERE / "AUDIT.json", result)
    trained = result["strict_calibration"]["trained"]
    lines = ["# Microclaim full-evidence cross-encoder v1 audit", "",
             f"严格 calibration：窗口 F1 {trained['windows']['f1']:.6f}，整答 F1 {trained['answers']['f1']:.6f}。",
             f"相对 incumbent 的独有误报：旧 atomic NLI {old_extra_fp}，新模型 {main_extra_fp}，减少 {old_extra_fp-main_extra_fp}。", "",
             "| 标签类型 | 新模型召回 | 旧 atomic NLI | incumbent |", "|---|---:|---:|---:|"]
    for kind in result["label_type_recall"]["trained"]:
        a = result["label_type_recall"]["trained"][kind]; b = result["label_type_recall"]["old_atomic_NLI"][kind]; d = result["label_type_recall"]["incumbent"][kind]
        lines.append(f"| {kind} | {a['recall']:.4f} | {b['recall']:.4f} | {d['recall']:.4f} |")
    lines += ["", "这些都是反复使用 calibration 后的开发诊断；official test 未打开，正式 baseline 未修改。", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("MICROCLAIM_CROSSENCODER_POSTHOC_AUDIT_COMPLETE", json.dumps({
        "window_f1": trained["windows"]["f1"], "answer_f1": trained["answers"]["f1"],
        "old_extra_fp": old_extra_fp, "trained_extra_fp": main_extra_fp}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
