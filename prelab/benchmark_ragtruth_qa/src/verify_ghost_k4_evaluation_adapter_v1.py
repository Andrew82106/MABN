"""Independent arithmetic replay of the frozen GHOST common-metric adapter."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q
import run_ghost_k4_evaluation_adapter_v1 as target


def counts(y, score, threshold):
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(score) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def choose(y, score):
    options = [np.nextafter(np.max(score), np.inf), *np.unique(score)]
    values = []
    for threshold in options:
        result = counts(y, score, threshold)
        values.append((result[-1], result[-3], float(threshold)))
    return max(values)[2]


def main():
    complete = q.read(target.OUT / "complete.json")
    for name, expected in complete["files_sha256"].items():
        assert q.sha(target.OUT / name) == expected
    meta, answers, windows = target.check_prepared()
    summary = q.read(target.OUT / "summary.json")
    lo, hi = meta["bounds"]["calibration"]
    yw = np.asarray([row["label"] for row in meta["windows"][lo:hi]], np.int8)
    ya = np.asarray([row["label"] for row in meta["answers"][634:]], np.int8)
    sw, sa = windows[lo:hi], answers[634:]
    tw, ta = choose(yw, sw), choose(ya, sa)
    assert tw == summary["thresholds_calibration_F1Opt"]["window"]["threshold"]
    assert ta == summary["thresholds_calibration_F1Opt"]["answer"]["threshold"]
    wm = summary["common_metrics"]["calibration"]["windows"]
    am = summary["common_metrics"]["calibration"]["answers"]
    wc, ac = counts(yw, sw, tw), counts(ya, sa, ta)
    assert wc[:4] == (wm["tp"], wm["fp"], wm["fn"], wm["tn"])
    assert ac[:4] == (am["tp"], am["fp"], am["fn"], am["tn"])
    assert abs(wc[-1] - wm["f1"]) < 1e-15 and abs(ac[-1] - am["f1"]) < 1e-15
    assert abs(roc_auc_score(yw, sw) - wm["auroc"]) < 1e-15
    assert abs(average_precision_score(yw, sw) - wm["average_precision"]) < 1e-15
    assert abs(roc_auc_score(ya, sa) - am["auroc"]) < 1e-15
    assert abs(average_precision_score(ya, sa) - am["average_precision"]) < 1e-15
    report = {
        "status": "independent_mapping_and_metric_replay_passed",
        "answers_exact": len(answers),
        "windows_exact": len(windows),
        "all_windows_equal_their_frozen_answer_score": True,
        "thresholds_and_confusions_exact": True,
        "AUROC_and_AP_exact": True,
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "official_test_opened": False,
    }
    q.save(target.OUT / "INDEPENDENT_VERIFY.json", report)
    print("GHOST_COMMON_EVALUATION_INDEPENDENT_VERIFY_PASSED", flush=True)


if __name__ == "__main__":
    main()
