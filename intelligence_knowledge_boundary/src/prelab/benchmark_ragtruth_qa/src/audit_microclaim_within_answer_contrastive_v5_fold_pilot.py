"""Independent fit-only audit and stop report for the v5 fold-0 pilot."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/microclaim_within_answer_contrastive_v5_gpu_v2"
BASE = ROOT / "results/microclaim_crossencoder_expanded_v4"
DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
DESIGN = ROOT / "results/microclaim_within_answer_contrastive_v5"
FOLD = 0
FIT_CLAIMS = 34_919
FIT_ANSWERS = 3_680


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path, names=None):
    with np.load(path, allow_pickle=False) as values:
        use = values.files if names is None else names
        return {name: values[name].copy() for name in use}


def threshold(y, score):
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable"); ss, yy = score[order], y[order]
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.r_[0, np.cumsum(yy)[last]]; n = np.r_[0, last + 1]
    values = np.r_[np.nextafter(ss[0], np.inf), ss[last]]
    f1 = 2 * tp / (n + y.sum()); precision = np.divide(tp, n, out=np.zeros(len(n)), where=n > 0)
    best = max(range(len(n)), key=lambda j: (f1[j], precision[j], values[j]))
    return float(values[best])


def metric(y, score):
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=np.float64); cut = threshold(y, score)
    p = score >= cut; tp = int(np.count_nonzero((y == 1) & p)); fp = int(np.count_nonzero((y == 0) & p))
    fn = int(np.count_nonzero((y == 1) & ~p)); tn = int(np.count_nonzero((y == 0) & ~p))
    return {"threshold": cut, "n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score))}


def q(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "p99": float(np.quantile(values, .99)), "above_0.95": float(np.mean(values >= .95)),
            "above_0.99": float(np.mean(values >= .99))}


def project_selected(v4, scores, mask):
    indptr, owners = v4["window_claim_indptr"], v4["window_claim_example_index"]
    output = np.empty(int(mask.sum()), dtype=np.float32)
    for out_index, window_index in enumerate(np.flatnonzero(mask)):
        output[out_index] = np.max(scores[owners[indptr[window_index]:indptr[window_index + 1]]])
    return output


def main():
    directory = RUN / "fold_0"; audit_path = directory / "INDEPENDENT_PILOT_AUDIT.json"
    assert not audit_path.exists()
    complete = read(directory / "complete.json")
    assert complete["status"] == "fold_complete" and complete["fold"] == FOLD
    assert sha(directory / "model.pt") == complete["checkpoint_sha256"]
    assert sha(directory / "predictions.npz") == complete["predictions_sha256"]
    assert complete["warm_start_checkpoint_sha256"] == read(BASE / "fold_0/complete.json")["checkpoint_sha256"]
    assert complete["calibration_labels_used_for_training_or_selection"] is False
    prediction = load(directory / "predictions.npz")
    base_input = load(BASE / "encoded_inputs.npz", ("labels", "fold_assignment"))
    indices, candidate = prediction["held_indices"], prediction["held_scores"]
    assert np.array_equal(indices, np.flatnonzero(base_input["fold_assignment"] == FOLD))
    assert max(indices) < FIT_CLAIMS
    y_claim = base_input["labels"][indices]
    with np.load(BASE / "scores.npz", allow_pickle=False) as values:
        base_claim_all = values["main_claim_scores"].copy()
    base_claim = base_claim_all[indices]
    candidate_all = base_claim_all.copy(); candidate_all[indices] = candidate
    v4 = load(DATA / "arrays.npz")
    first_owner = v4["window_claim_example_index"][v4["window_claim_indptr"][:-1]]
    held_window = (first_owner < FIT_CLAIMS) & (base_input["fold_assignment"][first_owner] == FOLD)
    candidate_window = project_selected(v4, candidate_all, held_window)
    base_window = project_selected(v4, base_claim_all, held_window)
    y_window = v4["window_label"][held_window]
    # Parse only the fit prefix; calibration answer labels are never parsed.
    answers = []
    with (DATA / "answers.jsonl").open(encoding="utf-8") as handle:
        for _ in range(FIT_ANSWERS): answers.append(json.loads(handle.readline()))
    held_responses = sorted(set(int(value) for value in v4["window_response_index"][held_window]))
    full_fit_window = first_owner < FIT_CLAIMS
    candidate_windows_fit = project_selected(v4, candidate_all, full_fit_window)
    base_windows_fit = project_selected(v4, base_claim_all, full_fit_window)
    candidate_answer, base_answer, y_answer = [], [], []
    for response_index in held_responses:
        row = answers[response_index]; left, right = row["window_array_start"], row["window_array_end"]
        candidate_answer.append(float(np.max(candidate_windows_fit[left:right])))
        base_answer.append(float(np.max(base_windows_fit[left:right])))
        y_answer.append(int(row["answer_risk"]))
    pair = load(DESIGN / "pair_arrays.npz")
    held_pair = pair["held_fold"] == FOLD; train_pair = ~held_pair
    assert int(train_pair.sum()) == complete["training"]["pairs"] == 5348
    pos, neg = pair["positive_index"][held_pair], pair["negative_index"][held_pair]
    candidate_margin = candidate_all[pos] - candidate_all[neg]
    base_margin = base_claim_all[pos] - base_claim_all[neg]
    result_metrics = {
        "candidate": {"microclaims": metric(y_claim, candidate), "windows": metric(y_window, candidate_window),
                      "answers": metric(y_answer, candidate_answer)},
        "v4": {"microclaims": metric(y_claim, base_claim), "windows": metric(y_window, base_window),
               "answers": metric(y_answer, base_answer)},
    }
    deltas = {level: {key: result_metrics["candidate"][level][key] - result_metrics["v4"][level][key]
                      for key in ("f1", "auroc", "average_precision")}
              for level in ("microclaims", "windows", "answers")}
    existing = read(directory / "PILOT_EVALUATION.json")
    for method, old_name in (("candidate", "candidate"), ("v4", "v4_read_only_reference")):
        for level in ("microclaims", "windows", "answers"):
            prior = existing[old_name][level]["metrics"]
            for key in ("n", "positive", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "auroc", "average_precision"):
                assert result_metrics[method][level][key] == prior[key], (method, level, key)
    diagnostics = {
        "candidate_score_by_label": {"safe": q(candidate[y_claim == 0]), "risk": q(candidate[y_claim == 1])},
        "v4_score_by_label": {"safe": q(base_claim[y_claim == 0]), "risk": q(base_claim[y_claim == 1])},
        "held_pair_order": {
            "pairs": len(pos), "candidate_accuracy": float(np.mean(candidate_margin > 0)),
            "v4_accuracy": float(np.mean(base_margin > 0)),
            "candidate_margin": q(candidate_margin), "v4_margin": q(base_margin),
        },
    }
    assert diagnostics["held_pair_order"]["candidate_accuracy"] == existing["candidate"]["pair_order_accuracy"]
    assert diagnostics["held_pair_order"]["v4_accuracy"] == existing["v4_read_only_reference"]["pair_order_accuracy"]
    unused = [name for name in ("fold_1", "fold_2", "fold_3", "fold_4", "full_fit") if (RUN / name).exists()]
    assert not unused
    audit = {
        "status": "independent_fold0_pilot_audit_passed_stop_candidate",
        "training_integrity": {"warm_start_exact": True, "train_pairs": int(train_pair.sum()),
                               "held_pairs_excluded": int(held_pair.sum()), "held_claims": len(indices),
                               "BF16_forward": complete["training"]["forward_logits_dtypes"] == ["torch.bfloat16"],
                               "calibration_labels_used_for_training_or_selection": False},
        "metrics": result_metrics, "delta_candidate_minus_v4": deltas, "diagnostics": diagnostics,
        "pilot_gate": {"pass": False,
                       "reason": "Candidate loses window F1, AUROC and AP and also loses held-pair ordering versus its v4 warm start."},
        "remaining_formal_models_started": [], "calibration_answer_rows_parsed": 0,
        "official_test_opened": False, "formal_baselines_modified": False,
        "fold_complete_sha256": sha(directory / "complete.json"),
        "predictions_sha256": sha(directory / "predictions.npz"),
    }
    pending = audit_path.with_suffix(".json.pending")
    pending.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8"); pending.replace(audit_path)
    report = ["# Within-answer contrastive v5：fold-0 独立审计与停止结论", "",
              "fold-0 的 checkpoint、held 索引、pair 排除和全部指标已独立复算；与原 pilot 文件逐项一致。", "",
              "| 层级 | v5 F1 | v4 F1 | ΔF1 | v5 AP | v4 AP | ΔAP |",
              "|---|---:|---:|---:|---:|---:|---:|",
              *[f"| {level} | {result_metrics['candidate'][level]['f1']:.6f} | {result_metrics['v4'][level]['f1']:.6f} | {deltas[level]['f1']:+.6f} | "
                f"{result_metrics['candidate'][level]['average_precision']:.6f} | {result_metrics['v4'][level]['average_precision']:.6f} | {deltas[level]['average_precision']:+.6f} |"
                for level in ("microclaims", "windows", "answers")], "",
              f"held pair 排序准确率从 {diagnostics['held_pair_order']['v4_accuracy']:.6f} 降到 {diagnostics['held_pair_order']['candidate_accuracy']:.6f}。", "",
              "失败原因：pair pass 在训练 pair 上形成很强排序，但没有迁移到未见 source；同时整体分数明显饱和，安全微主张也被推高。结果是窗口 TP 减少、FP 增加，整答召回上升但误报大幅增加。静态表面相似负例不是可靠的模型困难负例。", "",
              "pilot gate 失败，停止该候选；fold 1-4 和 full-fit 均未启动。calibration/test 未参与，baseline 未修改。", ""]
    (RUN / "PILOT_STOP_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("WITHIN_ANSWER_V5_FOLD0_INDEPENDENT_AUDIT_PASSED_STOP",
          result_metrics["candidate"]["windows"]["f1"], result_metrics["v4"]["windows"]["f1"])


if __name__ == "__main__":
    main()
