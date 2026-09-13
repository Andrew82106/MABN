"""Independent metric, split, and artifact audit for aligned_evidence_head_v1."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/aligned_evidence_head_v1"
AUDIT_DIR = ROOT / "research/aligned_evidence_head_v1_audit"
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count(y, score, threshold):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pred = score >= threshold
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    return {
        "n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def answer_scores(answers, windows, score):
    by_answer = defaultdict(list)
    for index, window in enumerate(windows):
        by_answer[window["response_id"]].append(index)
    return np.asarray([score[by_answer[answer["response_id"]]].max() for answer in answers])


def exact_metrics(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        if isinstance(expected[key], float):
            assert abs(expected[key] - actual[key]) < 1e-15, (key, expected[key], actual[key])
        else:
            assert expected[key] == actual[key], (key, expected[key], actual[key])


def run():
    done = read(RESULT / "complete.json")
    names = {"summary": "summary.json", "report": "REPORT.md", "scores": "scores.npz",
             "fit_complete": "fit_complete.json", "model": "model.pkl"}
    for key, name in names.items():
        assert done[f"{key}_sha256"] == sha(RESULT / name)
    summary = read(RESULT / "summary.json")
    fit = read(RESULT / "fit_complete.json")
    protocol = read(RESULT / "protocol.json")
    initialized = read(RESULT / "initialized.json")
    assert initialized["protocol_sha256"] == sha(RESULT / "protocol.json")
    assert len(protocol["readout"]["candidates"]) == 4
    assert fit["selected"] == max(fit["candidates"], key=lambda name: fit["candidates"][name]["selection_key"])
    assert summary["selected_fit_only"] == fit["selected"]
    assert not summary["calibration_used_for_selection"] and not summary["calibration_F1Opt_computed"]
    assert summary["calibration_evaluations"] == 1
    assert not done["GPU_used"] and not done["formal_baselines_modified"] and not done["official_test_opened"]

    fit_answers = lines(ROOT / "data/answers_fit.jsonl")
    fit_windows = lines(ROOT / "data/windows_k4_fit.jsonl")
    cal_answers = lines(ROOT / "data/answers_calibration.jsonl")
    cal_windows = lines(ROOT / "data/windows_k4_calibration.jsonl")
    with np.load(RESULT / "fit_scores.npz", allow_pickle=False) as z:
        fit_score = z["selected_gated_window_oof"].astype(np.float64)
        fit_base = z["incumbent_window"].astype(np.float64)
        any_y = z["any_claim_labels"].astype(np.int8)
        conflict_y = z["conflict_claim_labels"].astype(np.int8)
    with np.load(RESULT / "scores.npz", allow_pickle=False) as z:
        cal_score = z["gated_window_scores"].astype(np.float64)
        cal_base = z["incumbent_window_scores"].astype(np.float64)
        additions = z["additions"].astype(bool)
        stored_answers = z["answer_scores"].astype(np.float64)
    thresholds = fit["base_fit_thresholds"]

    fy = np.asarray([row["label"] for row in fit_windows], dtype=np.int8)
    fay = np.asarray([row["label"] for row in fit_answers], dtype=np.int8)
    cy = np.asarray([row["label"] for row in cal_windows], dtype=np.int8)
    cay = np.asarray([row["label"] for row in cal_answers], dtype=np.int8)
    fa = answer_scores(fit_answers, fit_windows, fit_score)
    ca = answer_scores(cal_answers, cal_windows, cal_score)
    assert np.array_equal(ca, stored_answers)
    fit_metric = {"windows": count(fy, fit_score, thresholds["window"]["threshold"]),
                  "answers": count(fay, fa, thresholds["answer"]["threshold"])}
    cal_metric = {"windows": count(cy, cal_score, thresholds["window"]["threshold"]),
                  "answers": count(cay, ca, thresholds["answer"]["threshold"])}
    for level in ("windows", "answers"):
        exact_metrics(summary["fit_OOF"][level], fit_metric[level])
        exact_metrics(summary["strict_calibration"][level], cal_metric[level])

    base_prediction = cal_base >= thresholds["window"]["threshold"]
    selected_prediction = cal_score >= thresholds["window"]["threshold"]
    assert np.array_equal(additions, ~base_prediction & selected_prediction)
    assert np.all(~base_prediction | selected_prediction)
    recomputed_add = {
        "added": int(additions.sum()),
        "added_tp": int(np.count_nonzero(additions & (cy == 1))),
        "added_fp": int(np.count_nonzero(additions & (cy == 0))),
        "incremental_precision": float(cy[additions].mean()) if additions.any() else 0.0,
        "base_tp_retained": int(np.count_nonzero(base_prediction & (cy == 1) & selected_prediction)),
        "base_tp_total": int(np.count_nonzero(base_prediction & (cy == 1))),
        "base_fp_retained": int(np.count_nonzero(base_prediction & (cy == 0) & selected_prediction)),
        "base_fp_total": int(np.count_nonzero(base_prediction & (cy == 0))),
        "base_fp_removed": int(np.count_nonzero(base_prediction & (cy == 0) & ~selected_prediction)),
    }
    assert recomputed_add == summary["strict_calibration_additions"]

    atomic_rows = lines(ROOT / "results/atomic_microclaim_nli_v1/inputs.jsonl")
    fit_rows = [row for row in atomic_rows if row["partition"] == "fit"]
    response_ids, groups = [], []
    for row in fit_rows:
        response_ids.extend([row["response_id"]] * len(row["claims"]))
        groups.extend([row["group_id"]] * len(row["claims"]))
    groups = np.asarray(groups)
    assert len(groups) == len(any_y) == 9055 and np.all(conflict_y <= any_y)
    folds = list(GroupKFold(5).split(np.arange(len(groups)), any_y, groups))
    for candidate in fit["candidates"].values():
        target = any_y if candidate["target"] == "any_error" else conflict_y
        seen = np.zeros(len(groups), dtype=np.int8)
        for logged, (train, held) in zip(candidate["folds"], folds):
            assert not set(groups[train]) & set(groups[held])
            seen[held] += 1
            assert logged["train_claims"] == len(train) and logged["held_claims"] == len(held)
            assert logged["train_groups"] == len(set(groups[train]))
            assert logged["held_groups"] == len(set(groups[held]))
            assert logged["train_positive"] == int(target[train].sum())
            assert logged["held_positive"] == int(target[held].sum())
        assert np.all(seen == 1)

    audit = {
        "passed": True,
        "selected": fit["selected"],
        "candidate_count": len(fit["candidates"]),
        "feature_width": fit["feature_width"],
        "pair_slot_width": fit["slot_width"],
        "pair_alignment_protocol": protocol["pair_alignment"],
        "fit_group_OOF_partition_exact": True,
        "fit_metrics_independently_recomputed": fit_metric,
        "strict_calibration_metrics_independently_recomputed": cal_metric,
        "strict_calibration_additions_independently_recomputed": recomputed_add,
        "calibration_used_for_selection": False,
        "calibration_F1Opt_computed": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "audited_files_sha256": {str((RESULT / name).relative_to(ROOT)): sha(RESULT / name)
                                  for name in ("protocol.json", "initialized.json", "fit_complete.json",
                                               "fit_scores.npz", "model.pkl", "scores.npz",
                                               "summary.json", "REPORT.md", "complete.json")},
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    target = AUDIT_DIR / "AUDIT.json"
    assert not target.exists(), f"Refuse overwrite: {target}"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    complete = {"passed": True, "audit_sha256": sha(target), "audit_code_sha256": sha(Path(__file__))}
    (AUDIT_DIR / "complete.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ALIGNED_EVIDENCE_HEAD_V1_INDEPENDENT_AUDIT_PASSED")


if __name__ == "__main__":
    run()
