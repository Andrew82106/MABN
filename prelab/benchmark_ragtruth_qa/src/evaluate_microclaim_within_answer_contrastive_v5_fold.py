"""Fit-only pilot evaluation for one completed v5 OOF fold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as metrics  # noqa: E402


RUN = ROOT / "results/microclaim_within_answer_contrastive_v5_gpu_v2"
BASE = ROOT / "results/microclaim_crossencoder_expanded_v4"
DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
DESIGN = ROOT / "results/microclaim_within_answer_contrastive_v5"
FIT_CLAIMS = 34_919


def sha(path):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def project(v4, claim_scores, mask):
    indptr, owners = v4["window_claim_indptr"], v4["window_claim_example_index"]
    output = np.empty(int(mask.sum()), dtype=np.float32); cursor = 0
    for index in np.flatnonzero(mask):
        output[cursor] = np.max(claim_scores[owners[indptr[index]:indptr[index + 1]]]); cursor += 1
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    fold = parser.parse_args().fold
    directory = RUN / f"fold_{fold}"; complete = read(directory / "complete.json")
    assert complete["status"] == "fold_complete" and complete["fold"] == fold
    assert complete["predictions_sha256"] == sha(directory / "predictions.npz")
    prediction = load(directory / "predictions.npz")
    base_input = load(BASE / "encoded_inputs.npz"); assignment = base_input["fold_assignment"]
    indices, candidate_values = prediction["held_indices"], prediction["held_scores"]
    assert np.all(assignment[indices] == fold)
    with np.load(BASE / "scores.npz", allow_pickle=False) as values:
        base_claim = values["main_claim_scores"].copy()
    candidate_claim = base_claim.copy(); candidate_claim[indices] = candidate_values
    v4 = load(DATA / "arrays.npz")
    first_owner = v4["window_claim_example_index"][v4["window_claim_indptr"][:-1]]
    window_mask = (first_owner < FIT_CLAIMS) & (assignment[first_owner] == fold)
    window_label = v4["window_label"][window_mask]
    candidate_window = project(v4, candidate_claim, window_mask)
    base_window = project(v4, base_claim, window_mask)
    answers = [json.loads(line) for line in (DATA / "answers.jsonl").open(encoding="utf-8")]
    response_indices = sorted(set(int(value) for value in v4["window_response_index"][window_mask]))
    assert max(response_indices) < 3680
    candidate_answer = []; base_answer = []; answer_label = []
    # Project all fit windows once for answer maxima.
    fit_window_mask = first_owner < FIT_CLAIMS
    candidate_fit_window = project(v4, candidate_claim, fit_window_mask)
    base_fit_window = project(v4, base_claim, fit_window_mask)
    for response_index in response_indices:
        row = answers[response_index]
        candidate_answer.append(float(np.max(candidate_fit_window[row["window_array_start"]:row["window_array_end"]])))
        base_answer.append(float(np.max(base_fit_window[row["window_array_start"]:row["window_array_end"]])))
        answer_label.append(int(row["answer_risk"]))
    labels = base_input["labels"][indices]
    with np.load(DESIGN / "pair_arrays.npz", allow_pickle=False) as values:
        pairs = {name: values[name].copy() for name in values.files}
    pair_mask = pairs["held_fold"] == fold
    pos, neg = pairs["positive_index"][pair_mask], pairs["negative_index"][pair_mask]
    def one(y, score):
        threshold = metrics.choose_threshold(y, score)
        return {"F1Opt_threshold": threshold,
                "metrics": metrics.count(y, score, threshold["threshold"])}
    result = {
        "status": "fit_held_fold_pilot_only", "fold": fold,
        "held_claims": len(indices), "held_windows": int(window_mask.sum()),
        "held_answers": len(response_indices), "held_pairs": int(pair_mask.sum()),
        "candidate": {"microclaims": one(labels, candidate_values),
                      "windows": one(window_label, candidate_window),
                      "answers": one(answer_label, candidate_answer),
                      "pair_order_accuracy": float(np.mean(candidate_claim[pos] > candidate_claim[neg]))},
        "v4_read_only_reference": {"microclaims": one(labels, base_claim[indices]),
                                   "windows": one(window_label, base_window),
                                   "answers": one(answer_label, base_answer),
                                   "pair_order_accuracy": float(np.mean(base_claim[pos] > base_claim[neg]))},
        "interpretation": "Fold-local F1Opt/AUROC/AP and held-pair ordering are pilot diagnostics; no calibration/test row is read and no candidate is selected here.",
        "calibration_rows_read": 0, "official_test_opened": False,
        "formal_baselines_modified": False,
        "predictions_sha256": sha(directory / "predictions.npz"),
    }
    output = directory / "PILOT_EVALUATION.json"; assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [f"# Within-answer v5 fold-{fold} pilot", "",
              "| 层级 | v5 F1Opt | v4 F1Opt | v5 AUROC | v4 AUROC |",
              "|---|---:|---:|---:|---:|",
              *[f"| {level} | {result['candidate'][level]['metrics']['f1']:.6f} | "
                f"{result['v4_read_only_reference'][level]['metrics']['f1']:.6f} | "
                f"{result['candidate'][level]['metrics']['auroc']:.6f} | "
                f"{result['v4_read_only_reference'][level]['metrics']['auroc']:.6f} |"
                for level in ("microclaims", "windows", "answers")], "",
              f"held 同答 pair 排序准确率：v5 {result['candidate']['pair_order_accuracy']:.6f}；v4 {result['v4_read_only_reference']['pair_order_accuracy']:.6f}。", "",
              "仅为 fit held-fold 诊断，不读取 calibration/test，不用于改超参数。", ""]
    (directory / "PILOT_EVALUATION.md").write_text("\n".join(report), encoding="utf-8")
    print("WITHIN_ANSWER_V5_FOLD_PILOT_EVALUATED", fold,
          result["candidate"]["windows"]["metrics"]["f1"],
          result["candidate"]["answers"]["metrics"]["f1"])


if __name__ == "__main__":
    main()
