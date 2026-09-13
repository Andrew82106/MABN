"""Independent arithmetic and immutability audit for specificity_gate_v1."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402

OUT = ROOT / "results/specificity_gate_v1"


def main():
    initialized = q.read(OUT / "initialized.json")
    frozen = q.read(OUT / "fit_complete.json")
    summary = q.read(OUT / "summary.json")
    complete = q.read(OUT / "complete.json")
    assert q.sha(OUT / "protocol.json") == initialized["protocol_sha256"] == frozen["protocol_sha256"]
    for relative, digest in frozen["source_sha256"].items():
        assert q.sha(ROOT / relative) == digest
    assert q.sha(OUT / "fit_complete.json") == summary["fit_complete_sha256"]
    assert q.sha(OUT / "summary.json") == complete["summary_sha256"]
    assert q.sha(OUT / "REPORT.md") == complete["report_sha256"]
    assert q.sha(OUT / "scores.npz") == summary["scores_sha256"] == complete["scores_sha256"]
    meta = q.metadata()
    with np.load(OUT / "scores.npz", allow_pickle=False) as data:
        window = data["window_scores"]
        answer = data["answer_scores"]
    assert window.shape == (210364,) and answer.shape == (793,)
    assert np.array_equal(answer, q.answer_scores(meta, window))
    strict = q.metrics(meta, window, summary["thresholds_frozen_from_fit"])["calibration"]
    assert strict == summary["strict_calibration"]
    lo, hi = meta["bounds"]["calibration"]
    cal_answer_ids = [i for i, a in enumerate(meta["answers"]) if a["partition"] == "calibration"]
    diagnostic_thresholds = {
        "window": q.choose_threshold([w["label"] for w in meta["windows"][lo:hi]], window[lo:hi]),
        "answer": q.choose_threshold([meta["answers"][i]["label"] for i in cal_answer_ids], answer[cal_answer_ids]),
    }
    assert diagnostic_thresholds == summary["calibration_F1Opt_diagnostic_not_strict"]["thresholds"]
    diagnostic = q.metrics(meta, window, diagnostic_thresholds)["calibration"]
    assert diagnostic == summary["calibration_F1Opt_diagnostic_not_strict"]["metrics"]
    reference = q.read(ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json")["metrics"]["calibration"]
    accepted = strict["windows"]["f1"] > reference["windows"]["f1"] and strict["answers"]["f1"] >= reference["answers"]["f1"]
    assert accepted == summary["accepted_as_replacement"] == complete["accepted_as_replacement"]
    result = {
        "passed": True, "selected": summary["selected_fit_only"],
        "strict_calibration": strict,
        "calibration_F1Opt_diagnostic_not_strict": diagnostic,
        "accepted_as_replacement": accepted,
        "calibration_used_for_selection": False, "formal_baselines_modified": False,
        "official_test_opened": False,
    }
    path = OUT / "INDEPENDENT_AUDIT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SPECIFICITY_GATE_V1_INDEPENDENT_AUDIT_PASSED", strict["windows"]["f1"], strict["answers"]["f1"])


if __name__ == "__main__":
    main()
