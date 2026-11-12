"""Independent smoke/final audits for within-answer contrastive-v5 GPU run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / "results/microclaim_within_answer_contrastive_v5"
BASE = ROOT / "results/microclaim_crossencoder_expanded_v4"
DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
OUT = ROOT / "results/microclaim_within_answer_contrastive_v5_gpu_v2"
FIT = 34_919
CAL = 2_267
FOLDS = 5


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def write_once(name, result):
    path = OUT / name; assert not path.exists()
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def common():
    prep = read(OUT / "preparation_complete.json")
    for name, expected in prep["files_sha256"].items(): assert sha(OUT / name) == expected
    for path, expected in prep["source_sha256"].items(): assert sha(path) == expected, path
    assert prep["calibration_labels_used"] is False and prep["formal_baselines_modified"] is False
    pair = load_npz(DESIGN / "pair_arrays.npz")
    base_arrays = load_npz(BASE / "encoded_inputs.npz")
    return prep, pair, base_arrays


def smoke_audit():
    prep, pair, base_arrays = common(); smoke = read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_disposable_warm_start_no_formal_training"
    indices = np.asarray(smoke["smoke_pair_indices"], dtype=np.int64)
    assert len(indices) == smoke["training"]["pairs"] and len(set(indices)) == len(indices)
    assert np.all(pair["held_fold"][indices] != 0)
    assert smoke["warm_start_checkpoint_sha256"] == read(BASE / "fold_0/complete.json")["checkpoint_sha256"]
    assert smoke["training"]["lambda"] == .25 and smoke["repeat_max_abs_difference"] <= 2e-6
    assert smoke["training"]["forward_logits_dtypes"] == ["torch.bfloat16"]
    assert smoke["formal_checkpoint_saved"] is False
    assert smoke["calibration_labels_used"] is False and smoke["formal_baselines_modified"] is False
    assert not any((OUT / name).exists() for name in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "full_fit"])
    result = {
        "status": "independent_GPU_smoke_audit_passed", "smoke_pairs": len(indices),
        "all_smoke_pairs_exclude_held_fold_0": True,
        "exact_warm_start_checkpoint": True, "combined_loss_lambda": .25,
        "BF16_forward": True, "formal_training_started": False,
        "peak_cuda_reserved_bytes": smoke["peak_cuda_reserved_bytes"],
        "calibration_labels_used": False, "official_test_opened": False,
        "formal_baselines_modified": False, "GPU_smoke_sha256": sha(OUT / "GPU_SMOKE.json"),
    }
    write_once("GPU_SMOKE_INDEPENDENT_AUDIT.json", result)
    print("WITHIN_ANSWER_V5_GPU_SMOKE_INDEPENDENT_AUDIT_PASSED", len(indices))


def project_windows(v4, claim_scores):
    indptr, owners = v4["window_claim_indptr"], v4["window_claim_example_index"]
    output = np.empty(len(v4["window_label"]), dtype=np.float32)
    for index in range(len(output)):
        output[index] = np.max(claim_scores[owners[indptr[index]:indptr[index + 1]]])
    return output


def choose_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable"); ss, yy = scores[order], labels[order]
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.r_[0, np.cumsum(yy)[last]]; n = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(ss[0], np.inf), ss[last]]
    f1 = 2 * tp / (n + labels.sum())
    precision = np.divide(tp, n, out=np.zeros(len(n)), where=n > 0)
    best = max(range(len(n)), key=lambda j: (f1[j], precision[j], thresholds[j]))
    return float(thresholds[best])


def counts(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8); predicted = np.asarray(scores) >= threshold
    tp = int(np.count_nonzero((labels == 1) & predicted)); fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted)); tn = int(np.count_nonzero((labels == 0) & ~predicted))
    return {"n": len(labels), "positive": int(labels.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def final_audit():
    prep, pair, base_arrays = common()
    claim_scores = np.full(FIT + CAL, np.nan, dtype=np.float32)
    fold_details = []
    for fold in range(FOLDS):
        directory = OUT / f"fold_{fold}"; complete = read(directory / "complete.json")
        assert complete["status"] == "fold_complete" and complete["fold"] == fold
        assert complete["warm_start_checkpoint_sha256"] == read(BASE / f"fold_{fold}/complete.json")["checkpoint_sha256"]
        assert complete["checkpoint_sha256"] == sha(directory / "model.pt")
        assert complete["predictions_sha256"] == sha(directory / "predictions.npz")
        assert complete["calibration_labels_used_for_training_or_selection"] is False
        prediction = load_npz(directory / "predictions.npz")
        indices, scores = prediction["held_indices"], prediction["held_scores"]
        assert np.all(base_arrays["fold_assignment"][indices] == fold) and np.isnan(claim_scores[indices]).all()
        assert complete["training"]["pairs"] == int(np.count_nonzero(pair["held_fold"] != fold))
        claim_scores[indices] = scores
        fold_details.append({"fold": fold, "claims": len(indices), "pairs": complete["training"]["pairs"]})
    directory = OUT / "full_fit"; complete = read(directory / "complete.json")
    assert complete["status"] == "full_fit_complete"
    assert complete["warm_start_checkpoint_sha256"] == read(BASE / "full_fit/complete.json")["checkpoint_sha256"]
    assert complete["checkpoint_sha256"] == sha(directory / "model.pt")
    assert complete["training"]["pairs"] == len(pair["held_fold"])
    prediction = load_npz(directory / "predictions.npz")
    indices, cal_scores = prediction["calibration_indices"], prediction["calibration_scores"]
    assert np.array_equal(indices, np.arange(FIT, FIT + CAL)); claim_scores[indices] = cal_scores
    assert np.isfinite(claim_scores).all()
    saved = load_npz(OUT / "scores.npz")
    assert np.array_equal(saved["claim_scores"], claim_scores)
    v4 = load_npz(DATA / "arrays.npz"); windows = project_windows(v4, claim_scores)
    assert np.array_equal(saved["window_scores"], windows)
    answers = [json.loads(line) for line in (DATA / "answers.jsonl").open(encoding="utf-8")]
    answer_scores = np.asarray([np.max(windows[row["window_array_start"]:row["window_array_end"]])
                                for row in answers], dtype=np.float32)
    assert np.array_equal(saved["answer_scores"], answer_scores)
    fit_window = v4["window_response_index"] < 3680
    fit_answer = np.arange(3680); cal_answer = np.arange(3680, len(answers))
    window_threshold = choose_threshold(v4["window_label"][fit_window], windows[fit_window])
    answer_labels = np.asarray([row["answer_risk"] for row in answers], dtype=np.int8)
    answer_threshold = choose_threshold(answer_labels[fit_answer], answer_scores[fit_answer])
    replay = {
        "fit": {"windows": counts(v4["window_label"][fit_window], windows[fit_window], window_threshold),
                "answers": counts(answer_labels[fit_answer], answer_scores[fit_answer], answer_threshold)},
        "calibration": {"windows": counts(v4["window_label"][~fit_window], windows[~fit_window], window_threshold),
                        "answers": counts(answer_labels[cal_answer], answer_scores[cal_answer], answer_threshold)},
    }
    summary = read(OUT / "summary.json")
    assert summary["trained_candidate"]["strict_fit_threshold_to_calibration"] == replay
    assert summary["calibration_used_for_training_or_selection"] is False
    result = {
        "status": "independent_final_audit_passed", "folds": fold_details,
        "full_fit_pairs": complete["training"]["pairs"],
        "claim_score_assembly_exact": True, "window_projection_exact": True,
        "answer_projection_exact": True, "fit_threshold_and_metrics_exact": True,
        "strict_calibration": replay["calibration"],
        "calibration_labels_used_for_training_or_selection": False,
        "official_test_opened": False, "formal_baselines_modified": False,
        "summary_sha256": sha(OUT / "summary.json"), "scores_sha256": sha(OUT / "scores.npz"),
    }
    write_once("FINAL_INDEPENDENT_AUDIT.json", result)
    print("WITHIN_ANSWER_V5_FINAL_INDEPENDENT_AUDIT_PASSED",
          replay["calibration"]["windows"]["f1"], replay["calibration"]["answers"]["f1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "final"))
    stage = parser.parse_args().stage
    if stage == "smoke": smoke_audit()
    else: final_audit()


if __name__ == "__main__":
    main()
