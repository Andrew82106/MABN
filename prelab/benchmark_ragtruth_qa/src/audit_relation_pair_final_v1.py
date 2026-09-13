"""CPU-only final audit for the completed relation-pair transfer run."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

import run_development as q
from diagnose_completed_tail import KINDS


ROOT = q.ROOT
RUN = ROOT / "results/relation_pair_fast_v1"
MODEL = RUN / "relation_rank025"
INC = ROOT / "results/large_fixed_convex_v1"
INC_DIAG = ROOT / "results/selected_convex_error_diagnosis_v1/summary.json"
INC_NAME = "semantic_claim__old_tree__large_weight0.4"


def metrics(labels, scores, threshold):
    return q.count(np.asarray(labels), np.asarray(scores), float(threshold))


def main():
    assert not torch.cuda.is_initialized()
    prepared = q.read(RUN / "preparation_complete.json")
    complete = q.read(MODEL / "complete.json")
    auxiliary = q.read(MODEL / "auxiliary_final.json")
    assert prepared["status"] == "prepared_not_trained"
    assert complete["status"] == "complete_development_only"
    assert complete["variant"] == "relation_rank025"
    assert complete["official_test_opened"] is False
    assert complete["preparation_sha256"] == q.sha(RUN / "preparation_complete.json")
    assert complete["auxiliary"] == auxiliary
    assert auxiliary["checkpoint_sha256"] == q.sha(MODEL / "auxiliary_final.pt")
    assert (auxiliary["pair_presentations"], auxiliary["forward_calls"], auxiliary["optimizer_updates"],
            auxiliary["logical_input_tokens"]) == (4704, 9408, 588, 7457373)
    assert complete["optimizer_updates"] == 1968

    history = complete["all_QA_epochs"]
    assert [entry["epoch"] for entry in history] == [1, 2, 3]
    assert complete["selected"] == max(history, key=lambda entry: entry["selection_key"])
    assert complete["selected"]["epoch"] == 2
    epoch_audit = []
    selected_archive = None
    for entry in history:
        epoch = entry["epoch"]
        stem = f"qa_epoch_{epoch:02d}"
        assert q.read(MODEL / f"{stem}.json") == entry
        for suffix, digest in entry["artifacts_sha256"].items():
            assert q.sha(MODEL / f"{stem}{suffix}") == digest
        with np.load(MODEL / f"{stem}_scores.npz", allow_pickle=False) as z:
            arrays = {name: z[name].copy() for name in z.files}
        assert (len(arrays["fit_window_scores"]), len(arrays["cal_window_scores"]),
                len(arrays["fit_answer_scores"]), len(arrays["cal_answer_scores"])) == (653979, 42241, 3680, 159)
        recomputed = {
            "windows": metrics(arrays["cal_window_labels"], arrays["cal_window_scores"],
                               entry["thresholds"]["window"]["threshold"]),
            "answers": metrics(arrays["cal_answer_labels"], arrays["cal_answer_scores"],
                               entry["thresholds"]["answer"]["threshold"]),
        }
        assert recomputed == entry["calibration"]
        assert q.choose_threshold(arrays["cal_window_labels"], arrays["cal_window_scores"]) == entry["thresholds"]["window"]
        assert q.choose_threshold(arrays["cal_answer_labels"], arrays["cal_answer_scores"]) == entry["thresholds"]["answer"]
        epoch_audit.append({"epoch": epoch, "selection_key": entry["selection_key"],
                            "window_f1": recomputed["windows"]["f1"],
                            "answer_f1": recomputed["answers"]["f1"],
                            "scores_sha256": q.sha(MODEL / f"{stem}_scores.npz")})
        if epoch == complete["selected"]["epoch"]:
            selected_archive = arrays
    assert selected_archive is not None

    meta = q.metadata()
    left, right = meta["bounds"]["calibration"]
    windows = meta["windows"][left:right]
    y = np.asarray([window["label"] for window in windows], bool)
    assert np.array_equal(y, selected_archive["cal_window_labels"].astype(bool))
    selected_threshold = complete["selected"]["thresholds"]["window"]["threshold"]
    pred = selected_archive["cal_window_scores"] >= selected_threshold

    typed = {kind: defaultdict(set) for kind in KINDS}
    for token in meta["tokens"]:
        if token["partition"] != "calibration":
            continue
        for span in token["span_token_mapping"]:
            kind = token["original_labels"][span["span_index"]]["label_type"]
            typed[kind][token["response_id"]].update(span["risk_token_indices"])
    masks = {kind: np.asarray([bool(set(window["token_indices"]) & typed[kind][window["response_id"]])
                              for window in windows]) for kind in KINDS}
    assert [int(masks[kind].sum()) for kind in KINDS] == [4086, 814, 997, 109]
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), y)
    by_type = {kind: {"risk_windows": int(mask.sum()), "detected": int(pred[mask].sum()),
                      "missed": int((~pred[mask]).sum()), "recall": float(pred[mask].mean())}
               for kind, mask in masks.items()}

    inc_diag = q.read(INC_DIAG)
    assert inc_diag["candidate"] == INC_NAME
    inc_entry = q.read(INC / f"{INC_NAME}.json")
    inc_scores_path = INC / f"{INC_NAME}_scores.npz"
    assert q.sha(inc_scores_path) == inc_entry["scores_sha256"] == inc_diag["scores_sha256"]
    with np.load(inc_scores_path, allow_pickle=False) as z:
        inc_all_scores = z["window_scores"].copy()
    inc_scores = inc_all_scores[left:right]
    inc_pred = inc_scores >= inc_entry["thresholds"]["window"]["threshold"]
    assert q.metrics(meta, inc_all_scores, inc_entry["thresholds"])["calibration"] == inc_diag["metrics"]

    type_comparison = {}
    for kind, mask in masks.items():
        old = inc_diag["type_recall_only"][kind]
        new = by_type[kind]
        assert old["risk_windows"] == new["risk_windows"]
        type_comparison[kind] = {
            "incumbent_detected": old["detected"], "relation_detected": new["detected"],
            "detected_difference": new["detected"] - old["detected"],
            "incumbent_recall": old["recall"], "relation_recall": new["recall"],
            "recall_difference": new["recall"] - old["recall"],
            "decision_overlap": {
                "both_detect": int((inc_pred & pred & mask).sum()),
                "incumbent_only": int((inc_pred & ~pred & mask).sum()),
                "relation_only": int((~inc_pred & pred & mask).sum()),
                "neither": int((~inc_pred & ~pred & mask).sum()),
            },
        }

    new_metrics = complete["selected"]["calibration"]
    old_metrics = inc_diag["metrics"]
    deltas = {}
    for level in ("windows", "answers"):
        deltas[level] = {key: new_metrics[level][key] - old_metrics[level][key]
                         for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "auroc", "average_precision")}
    overall_overlap = {
        "positive_windows": {
            "both_detect": int((inc_pred & pred & y).sum()),
            "incumbent_only": int((inc_pred & ~pred & y).sum()),
            "relation_only": int((~inc_pred & pred & y).sum()),
            "neither": int((~inc_pred & ~pred & y).sum()),
        },
        "negative_windows": {
            "both_alert": int((inc_pred & pred & ~y).sum()),
            "incumbent_only": int((inc_pred & ~pred & ~y).sum()),
            "relation_only": int((~inc_pred & pred & ~y).sum()),
            "neither": int((~inc_pred & ~pred & ~y).sum()),
        },
    }

    answers = [answer for answer in meta["answers"] if answer["partition"] == "calibration"]
    clean_ids = {answer["response_id"] for answer in answers if answer["label"] == 0}
    clean = np.asarray([window["response_id"] in clean_ids for window in windows])
    fp_location = {
        "relation_clean_answer_windows": int((pred & ~y & clean).sum()),
        "relation_risky_answer_windows": int((pred & ~y & ~clean).sum()),
        "incumbent_clean_answer_windows": inc_diag["false_positive_windows_clean_answers"],
        "incumbent_risky_answer_windows": inc_diag["false_positive_windows_risky_answers"],
    }
    assert fp_location["relation_clean_answer_windows"] + fp_location["relation_risky_answer_windows"] == new_metrics["windows"]["fp"]

    report = {
        "status": "passed_complete_development_only",
        "selected_epoch_recomputed": 2,
        "epochs": epoch_audit,
        "selected_metrics": new_metrics,
        "incumbent_metrics": old_metrics,
        "relation_minus_incumbent": deltas,
        "type_recall_fixed_selected_threshold": by_type,
        "type_comparison": type_comparison,
        "decision_overlap": overall_overlap,
        "false_positive_location": fp_location,
        "runtime": {"seconds": complete["seconds"], "auxiliary_seconds": auxiliary["seconds"],
                    "peak_cuda_allocated_bytes": auxiliary["peak_cuda_allocated_bytes"]},
        "hashes": {"preparation": q.sha(RUN / "preparation_complete.json"),
                   "complete": q.sha(MODEL / "complete.json"),
                   "auxiliary_checkpoint": q.sha(MODEL / "auxiliary_final.pt"),
                   "selected_scores": q.sha(MODEL / "qa_epoch_02_scores.npz"),
                   "incumbent_scores": q.sha(inc_scores_path)},
        "limits": "Post-selection descriptive development audit. Type masks overlap. Each model keeps its own already-selected thresholds; no new fit, fusion or threshold is introduced.",
        "training_or_inference_in_this_audit": False,
        "GPU_used_in_this_audit": False,
        "official_test_opened": False,
        "baseline_modified": False,
    }
    q.save(RUN / "FINAL_INDEPENDENT_AUDIT.json", report)
    assert not torch.cuda.is_initialized()
    print("RELATION_PAIR_FINAL_INDEPENDENT_AUDIT_PASSED")


if __name__ == "__main__":
    main()
