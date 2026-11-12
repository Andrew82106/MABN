"""Read-only stage auditor for the running microclaim cross-encoder experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_development as q
import run_microclaim_crossencoder_v1 as c


def fold_audit(fold, records, arrays):
    directory = c.OUT / f"fold_{fold}"
    complete = q.read(directory / "complete.json")
    assert complete["status"] == "fold_complete" and complete["fold"] == fold
    assert complete["checkpoint_sha256"] == q.sha(directory / "model.pt")
    assert complete["predictions_sha256"] == q.sha(directory / "predictions.npz")
    expected = np.flatnonzero(arrays["fold_assignment"] == fold)
    train = np.flatnonzero((arrays["fold_assignment"] >= 0) & (arrays["fold_assignment"] != fold))
    with np.load(directory / "predictions.npz", allow_pickle=False) as z:
        assert set(z.files) == {"held_indices", "held_scores"}
        indices, scores = z["held_indices"], z["held_scores"]
    assert np.array_equal(indices, expected) and scores.shape == expected.shape
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    for field in ("group_id", "source_id", "response_id", "answer_sha256"):
        values = np.asarray([row[field] for row in records])
        assert not (set(values[train]) & set(values[expected]))
    plan = q.read(c.OUT / "preparation.json")["fold_batch_plan"][fold]
    training = complete["training"]
    assert training["examples"] == plan["train_claims"] == len(train)
    assert training["microbatches"] == plan["microbatches"]
    assert training["optimizer_updates"] == plan["optimizer_updates"]
    assert training["logical_tokens"] == plan["logical_tokens"]
    assert training["padded_tokens"] == plan["padded_tokens"]
    assert not complete["calibration_labels_used"] and not complete["threshold_selected"]
    assert not complete["official_test_opened"]
    return {"stage": f"fold_{fold}", "held": len(expected), "score_min": float(scores.min()),
            "score_max": float(scores.max()), "seconds": complete["seconds"],
            "checkpoint_sha256": complete["checkpoint_sha256"]}


def full_audit(records, arrays):
    directory = c.OUT / "full_fit"; complete = q.read(directory / "complete.json")
    assert complete["status"] == "full_fit_complete"
    assert complete["checkpoint_sha256"] == q.sha(directory / "model.pt")
    assert complete["predictions_sha256"] == q.sha(directory / "predictions.npz")
    expected = np.flatnonzero(arrays["fold_assignment"] == -1)
    with np.load(directory / "predictions.npz", allow_pickle=False) as z:
        assert set(z.files) == {"calibration_indices", "calibration_scores"}
        indices, scores = z["calibration_indices"], z["calibration_scores"]
    assert np.array_equal(indices, expected) and np.isfinite(scores).all()
    plan = q.read(c.OUT / "preparation.json")["fold_batch_plan"][c.FOLDS]
    training = complete["training"]
    for key in ("examples", "microbatches", "optimizer_updates", "logical_tokens", "padded_tokens"):
        planned_key = "train_claims" if key == "examples" else key
        assert training[key] == plan[planned_key]
    assert not complete["calibration_labels_used"] and not complete["threshold_selected"]
    assert not complete["official_test_opened"]
    return {"stage": "full_fit", "calibration": len(expected), "score_min": float(scores.min()),
            "score_max": float(scores.max()), "seconds": complete["seconds"],
            "checkpoint_sha256": complete["checkpoint_sha256"]}


def control_audit(arrays):
    directory = c.OUT / "frozen_control"; complete = q.read(directory / "complete.json")
    assert complete["status"] == "frozen_control_complete" and not complete["trained"]
    assert complete["scores_sha256"] == q.sha(directory / "scores.npz")
    with np.load(directory / "scores.npz", allow_pickle=False) as z:
        assert set(z.files) == {"claim_scores"}; scores = z["claim_scores"]
    assert scores.shape == arrays["labels"].shape and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all() and not complete["official_test_opened"]
    return {"stage": "frozen_control", "claims": len(scores), "score_min": float(scores.min()),
            "score_max": float(scores.max()), "trained": False}


def final_audit():
    complete = q.read(c.OUT / "complete.json"); summary = q.read(c.OUT / "summary.json")
    assert complete["status"] == "complete_development_only"
    assert complete["summary_sha256"] == q.sha(c.OUT / "summary.json")
    assert complete["scores_sha256"] == q.sha(c.OUT / "scores.npz")
    assert complete["report_sha256"] == q.sha(c.OUT / "REPORT.md")
    assert summary["five_source_group_OOF_models"] and summary["separate_full_fit_calibration_model"]
    assert not summary["formal_baselines_modified"] and not summary["official_test_opened"]
    return {"stage": "final", "window_f1": summary["trained_candidate"]["strict_fit_threshold_to_calibration"]["calibration"]["windows"]["f1"],
            "answer_f1": summary["trained_candidate"]["strict_fit_threshold_to_calibration"]["calibration"]["answers"]["f1"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage")
    args = parser.parse_args(); records, arrays = c.check_prepared()
    if args.stage.startswith("fold_"):
        result = fold_audit(int(args.stage.split("_")[1]), records, arrays)
    elif args.stage == "full_fit": result = full_audit(records, arrays)
    elif args.stage == "control": result = control_audit(arrays)
    elif args.stage == "final": result = final_audit()
    else: raise ValueError(args.stage)
    print("MICROCLAIM_CROSSENCODER_STAGE_AUDIT_PASSED", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__": main()
