"""Independent final audit for atomic_microclaim_nli_v1."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import run_atomic_microclaim_nli_v1 as experiment  # noqa: E402
import run_development as q  # noqa: E402


def same(left, right, path="root"):
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left: same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)): same(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        assert np.isclose(left, right, rtol=0, atol=1e-12), (path, left, right)
    else:
        assert left == right, (path, left, right)


def main():
    out = experiment.OUT
    destination = out / "INDEPENDENT_FINAL_AUDIT.json"
    assert not destination.exists(), f"Refuse overwrite: {destination}"
    rows, prepared = experiment.check_prepared()
    extracted = experiment.check_extracted(rows)
    completion = q.read(out / "complete.json")
    assert completion["status"] == "complete_development_only"
    for name, expected in completion["files_sha256"].items():
        assert q.sha(out / name) == expected, name
    summary = q.read(out / "summary.json")
    assert summary["formal_baselines_modified"] is False
    assert summary["official_test_opened"] is False
    assert summary["fit_only_model_and_threshold_selection"] is True
    with np.load(out / "scores.npz", allow_pickle=False) as data:
        scores = data["window_scores"].astype(np.float64)
        raw_scores = data["raw_window_scores"].astype(np.float64)
        assert data["claim_scores"].shape == (11322,)
        assert scores.shape == raw_scores.shape == (210364,)
    meta = q.metadata()
    strict = q.metrics(meta, scores, summary["thresholds_from_fit_OOF"])
    common = experiment.cal_diagnostic(meta, scores)
    raw_strict = q.metrics(meta, raw_scores, summary["raw_NLI"]["thresholds_from_fit"])
    raw_common = experiment.cal_diagnostic(meta, raw_scores)
    same(strict, summary["strict_fit_threshold_to_calibration"], "strict")
    same(common, summary["common_calibration_F1Opt_diagnostic"], "common")
    same(raw_strict, summary["raw_NLI"]["strict"], "raw_strict")
    same(raw_common, summary["raw_NLI"]["common_calibration_F1Opt_diagnostic"], "raw_common")
    source_hashes = summary["source_sha256"]
    for relative, expected in source_hashes.items():
        assert q.sha(ROOT / relative) == expected, relative
    result = {
        "status": "passed_independent_final_audit",
        "prepared_answers": prepared["answers"], "microclaims": prepared["microclaims"],
        "pairs": extracted["pairs"], "reused_old_pairs": extracted["reused_old_pairs"],
        "new_model_pairs": extracted["new_model_pairs"],
        "selected": summary["selected_fit_OOF_only"],
        "strict_metrics_exactly_recomputed": True,
        "common_calibration_diagnostic_exactly_recomputed": True,
        "frozen_file_hashes_match": True, "bound_source_hashes_match": True,
        "formal_baselines_modified": False, "official_test_opened": False,
        "final_test_claim": False,
    }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
