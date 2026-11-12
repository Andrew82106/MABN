"""Post-run audit without a second calibration-label evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402
import run_semantic_source_attribution_combined_fit_v1 as combined  # noqa: E402
import run_semantic_window_v2_expanded_fit_v1 as run  # noqa: E402


def main() -> None:
    out = run.OUT
    prep = run.read(out / "preparation_complete.json")
    fit = run.read(out / "fit_complete.json")
    summary = run.read(out / "summary.json")
    provenance = run.read(out / "PROVENANCE_AUDIT.json")
    source = run.read(out / "source_snapshot.json")
    assert provenance["in_sample_supervised_columns"] == 0
    assert provenance["answer_coverage_exactly_once"] is True
    assert prep["native_feature_replay"]["native_whitebox_geometry_exact"] is True
    assert fit["selected"]["candidate"] == "whitebox_geometry__C0.1"
    assert summary["calibration_thresholds_fitted"] == 0
    assert summary["calibration_evaluations"] == 1
    assert summary["fit_OOF_thresholds"] == fit["selected"]["fit_OOF_thresholds"]
    for name, expected in fit["artifacts_sha256"].items():
        path = Path(run.__file__) if name == "runner" else out / name
        assert run.sha(path) == expected
    assert run.sha(out / "calibration_scores.npz") == summary["artifacts_sha256"]["calibration_scores.npz"]
    assert source["formal_baseline_files_sha256"] == run.baseline_snapshot()

    _, projection = combined.load_projection()
    labels = projection["window_labels"].astype(np.int8)
    groups = combined.group_indices("window")
    coverage = np.zeros(run.FIT_WINDOWS, dtype=np.int8)
    splits = list(GroupKFold(run.DOWNSTREAM_FOLDS).split(np.arange(run.FIT_WINDOWS), labels, groups))
    for train, held in splits:
        combined.assert_source_connected_disjoint(train, held, "window")
        coverage[held] += 1
    assert np.all(coverage == 1)

    with np.load(out / "fit_scores.npz", allow_pickle=False) as scores:
        window = scores["window_scores"]
        answer = scores["answer_scores"]
        assert np.array_equal(scores["window_labels"], labels)
        assert np.array_equal(scores["answer_labels"], projection["response_labels"])
    thresholds = fit["selected"]["fit_OOF_thresholds"]
    replay = {
        "windows": q.count(labels, window, thresholds["window"]["threshold"]),
        "answers": q.count(projection["response_labels"], answer, thresholds["answer"]["threshold"]),
    }
    assert replay == fit["selected"]["fit_OOF"]
    payload = pickle.loads((out / "model.pkl").read_bytes())
    assert payload["version"] == run.VERSION
    assert payload["variant"] == run.VARIANT and payload["C"] == run.C_VALUE
    with np.load(out / "calibration_scores.npz", allow_pickle=False) as cal:
        assert cal["window_scores"].shape == (run.CAL_WINDOWS,)
        assert cal["answer_scores"].shape == (run.CAL_ANSWERS,)
        assert np.isfinite(cal["window_scores"]).all() and np.isfinite(cal["answer_scores"]).all()

    diagnostic = run.read(out / "FIT_DOMAIN_DIAGNOSTIC.json")
    fit_rows = [json.loads(line) for line in
                (run.ROOT / "fit_expansion/data/fit.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    assert len(fit_rows) == run.FIT_ANSWERS
    generators = np.asarray([row["model"] for row in fit_rows], dtype=str)
    response_window = projection["window_response_index"]
    for generator, saved in diagnostic["by_generator"].items():
        answer_indices = np.flatnonzero(generators == generator)
        answer_mask = np.zeros(run.FIT_ANSWERS, dtype=bool)
        answer_mask[answer_indices] = True
        window_mask = answer_mask[response_window]
        expected_answer = q.count(
            projection["response_labels"][answer_indices], answer[answer_indices],
            thresholds["answer"]["threshold"],
        )
        expected_window = q.count(
            labels[window_mask], window[window_mask], thresholds["window"]["threshold"],
        )
        assert expected_answer == saved["answers"]["at_frozen_threshold"]
        assert expected_window == saved["windows"]["at_frozen_threshold"]
        assert saved["answers"]["positive"] == int(projection["response_labels"][answer_indices].sum())
        assert saved["windows"]["positive"] == int(labels[window_mask].sum())

    audit = {
        "status": "passed_without_reopening_calibration_labels",
        "upstream_supervised_columns_source_group_OOF": True,
        "in_sample_supervised_columns": 0,
        "native_feature_rows_exact": run.NATIVE_WINDOWS,
        "downstream_five_fold_group_coverage_exactly_once": True,
        "saved_fit_metrics_exact": True,
        "per_generator_fit_OOF_counts_and_metrics_exact": True,
        "full_model_loadable": True,
        "calibration_score_shapes_and_finiteness_valid": True,
        "calibration_labels_reopened_by_audit": False,
        "calibration_thresholds_fitted": 0,
        "formal_baseline_files_unchanged": True,
        "official_test_opened": False,
        "GPU_used": False,
    }
    run.atomic_json(out / "INDEPENDENT_AUDIT.json", audit)
    print("EXPANDED_WINDOW_V2_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
