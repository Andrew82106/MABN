"""Independent final audit for the frozen Atomic Evidence Router v1 run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/atomic_evidence_router_v1"
UPSTREAM = ROOT / "results/atomic_microclaim_nli_v1"


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


def same(left, right, path="root"):
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, float):
        assert np.isclose(left, right, rtol=0, atol=1e-12), (path, left, right)
    else:
        assert left == right, (path, left, right)


def choose(labels, scores):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable"); ss = scores[order]; yy = labels[order]
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.r_[0, np.cumsum(yy)[last]]; predicted = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(ss[0], np.inf), ss[last]]
    f1 = 2 * tp / (predicted + labels.sum())
    precision = np.divide(tp, predicted, out=np.zeros(len(predicted), float), where=predicted > 0)
    best = max(range(len(thresholds)), key=lambda i: (f1[i], precision[i], thresholds[i]))
    return {"threshold": float(thresholds[best]), "fit_f1": float(f1[best]),
            "fit_precision": float(precision[best])}


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.count_nonzero((labels == 1) & predicted)); fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted)); tn = int(np.count_nonzero((labels == 0) & ~predicted))
    return {"n": len(labels), "positive": int(labels.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def atomic_json(path, value):
    assert not path.exists(), str(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    assert not torch.cuda.is_initialized()
    upstream_audit = read(UPSTREAM / "INDEPENDENT_FINAL_AUDIT.json")
    assert upstream_audit["status"] == "passed_independent_final_audit"
    assert upstream_audit["frozen_file_hashes_match"] and upstream_audit["bound_source_hashes_match"]

    complete = read(OUT / "complete.json")
    assert complete["status"] == "fit_oof_complete_gate_failed_calibration_unopened"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    summary = read(OUT / "summary.json")
    assert summary["status"] == complete["status"]
    assert summary["gate"]["status"] == "FAILED_CALIBRATION_UNOPENED"
    assert not summary["calibration_labels_accessed"]
    assert not summary["calibration_used_for_structure_or_threshold"]
    assert "calibration" not in summary

    window_labels = [row["label"] for row in lines(ROOT / "data/windows_k4_fit.jsonl")]
    answer_labels = [row["label"] for row in lines(ROOT / "data/answers_fit.jsonl")]
    with np.load(OUT / "scores.npz", allow_pickle=False) as saved:
        expected_keys = {"fit_claim_indices"}
        for routes in (1, 4):
            expected_keys |= {f"claim_scores_K{routes}_fit_oof", f"window_scores_K{routes}_fit_oof",
                              f"answer_scores_K{routes}_fit_oof"}
        assert set(saved.files) == expected_keys
        assert saved["fit_claim_indices"].shape == (9055,)
        assert all(np.isfinite(saved[name]).all() for name in saved.files)
        recomputed = {}
        for routes in (1, 4):
            ws = saved[f"window_scores_K{routes}_fit_oof"]
            answers = saved[f"answer_scores_K{routes}_fit_oof"]
            thresholds = {"window": choose(window_labels, ws), "answer": choose(answer_labels, answers)}
            result = {"windows": metrics(window_labels, ws, thresholds["window"]["threshold"]),
                      "answers": metrics(answer_labels, answers, thresholds["answer"]["threshold"])}
            same(thresholds, summary["fit_OOF"][f"K{routes}"]["thresholds_from_fit_OOF"], f"K{routes}.thresholds")
            same(result, summary["fit_OOF"][f"K{routes}"]["fit_OOF_metrics"], f"K{routes}.metrics")
            recomputed[routes] = result

    models = torch.load(OUT / "models.pt", map_location="cpu", weights_only=False)
    assert set(models["crossfit"]) == {1, 4}
    assert all(len(models["crossfit"][routes]) == 5 for routes in (1, 4))
    assert models["full_fit"] is None
    relation = summary["gate"]["relation_control"]["fit_OOF_metrics"]
    k1, k4 = recomputed[1], recomputed[4]
    gate = (k4["windows"]["f1"] > k1["windows"]["f1"] and
            k4["answers"]["f1"] > k1["answers"]["f1"] and
            k4["windows"]["f1"] > relation["windows"]["f1"] and
            k4["answers"]["f1"] > relation["answers"]["f1"])
    assert not gate
    result = {
        "status": "passed_fit_OOF_gate_failure_recomputed",
        "upstream_independent_audit_passed": True,
        "K1_fit_OOF": k1, "K4_fit_OOF": k4,
        "relation_control_fit_OOF": relation,
        "gate_recomputed": False,
        "calibration_arrays_absent": True,
        "calibration_labels_accessed": False,
        "full_fit_model_absent": True,
        "frozen_hashes_match": True,
        "formal_baselines_modified": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    atomic_json(OUT / "INDEPENDENT_FINAL_AUDIT.json", result)
    assert not torch.cuda.is_initialized()
    print("ATOMIC_EVIDENCE_ROUTER_INDEPENDENT_FINAL_AUDIT_PASSED_GATE_FAILED_AS_EXPECTED")


if __name__ == "__main__":
    main()
