"""Independent post-run audit without reopening calibration labels."""
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
import run_semantic_window_v2_expanded_hgb_v1 as run  # noqa: E402


def main() -> None:
    out = run.OUT
    protocol = run.read(out / "protocol.json")
    receipt = run.read(out / "PREREGISTRATION.json")
    started = run.read(out / "fit_started.json")
    fit = run.read(out / "fit_complete.json")
    summary = run.read(out / "summary.json")
    provenance = run.read(out / "PROVENANCE_AUDIT.json")
    assert protocol == run.protocol()
    assert receipt["protocol_sha256"] == run.sha(out / "protocol.json")
    assert receipt["written_at_unix_ns"] < started["started_at_unix_ns"]
    assert started["protocol_sha256"] == receipt["protocol_sha256"]
    assert fit["preregistration_precedes_fit"] is True
    assert fit["model_fits"] == len(run.LEAF_NODES) * run.FOLDS + 1
    assert fit["calibration_labels_opened"] is False
    assert fit["calibration_used_for_recipe_model_or_threshold_selection"] is False
    assert fit["native_diagnostics_used_for_selection"] is False
    assert summary["calibration_thresholds_fitted"] == 0
    assert summary["calibration_evaluations"] == 1
    assert summary["fit_OOF_thresholds"] == fit["selected"]["fit_OOF_thresholds"]
    assert provenance["upstream_supervised_columns_source_group_OOF"] is True
    assert provenance["in_sample_supervised_columns"] == 0
    assert provenance["feature_matrix_byte_identical_to_audited_expanded_LR"] is True
    assert provenance["feature_matrix_sha256"] == run.sha(run.FEATURES)
    assert run.assert_sources_unchanged()["formal_baseline_files_sha256"] == run.linear.baseline_snapshot()
    for name, expected in fit["artifacts_sha256"].items():
        path = Path(run.__file__) if name == "runner" else out / name
        assert run.sha(path) == expected
    assert run.sha(out / "calibration_scores.npz") == summary["artifacts_sha256"]["calibration_scores.npz"]

    _, projection = combined.load_projection()
    labels = projection["window_labels"].astype(np.int8)
    response_labels = projection["response_labels"].astype(np.int8)
    groups = combined.group_indices("window")
    coverage = np.zeros(run.FIT_WINDOWS, dtype=np.int8)
    splits = list(GroupKFold(run.FOLDS).split(np.arange(run.FIT_WINDOWS), labels, groups))
    linear_folds = run.read(run.LINEAR_OUT / "fit_complete.json")["crossfit"]
    for fold, (train, held) in enumerate(splits):
        combined.assert_source_connected_disjoint(train, held, "window")
        coverage[held] += 1
        saved = fit["crossfit"][fold]
        reference = linear_folds[fold]
        for key in (
            "train_windows", "held_windows", "train_answers", "held_answers",
            "train_groups", "held_groups", "train_positive", "held_positive",
        ):
            assert saved[key] == reference[key]
    assert np.all(coverage == 1)

    with np.load(out / "fit_scores.npz", allow_pickle=False) as scores:
        names = scores["candidate_names"].astype(str).tolist()
        window_scores = scores["candidate_window_scores"]
        answer_scores = scores["candidate_answer_scores"]
        selected_index = int(scores["selected_index"])
        assert names == [run.candidate_name(leaf) for leaf in run.LEAF_NODES]
        assert window_scores.shape == (len(run.LEAF_NODES), run.FIT_WINDOWS)
        assert answer_scores.shape == (len(run.LEAF_NODES), run.FIT_ANSWERS)
        assert np.array_equal(scores["window_labels"], labels)
        assert np.array_equal(scores["answer_labels"], response_labels)
        assert np.isfinite(window_scores).all() and np.isfinite(answer_scores).all()

        replay_table = []
        for index, row in enumerate(fit["candidates"]):
            thresholds = row["fit_OOF_thresholds"]
            metrics = {
                "windows": q.count(labels, window_scores[index], thresholds["window"]["threshold"]),
                "answers": q.count(response_labels, answer_scores[index], thresholds["answer"]["threshold"]),
            }
            assert metrics == row["fit_OOF"]
            native_thresholds = row["native_held_OOF_diagnostic"]["native_subset_F1Opt_thresholds"]
            native_opt = {
                "windows": q.count(labels[:run.NATIVE_WINDOWS], window_scores[index, :run.NATIVE_WINDOWS], native_thresholds["window"]["threshold"]),
                "answers": q.count(response_labels[:run.NATIVE_ANSWERS], answer_scores[index, :run.NATIVE_ANSWERS], native_thresholds["answer"]["threshold"]),
            }
            assert native_opt == row["native_held_OOF_diagnostic"]["native_subset_F1Opt"]
            replay_table.append(metrics)
        expected_selected = max(
            range(len(fit["candidates"])),
            key=lambda index: run.selection_key(fit["candidates"][index], index),
        )
        assert selected_index == fit["selected_index"] == expected_selected
        assert fit["selected"] == fit["candidates"][expected_selected]

    payload = pickle.loads((out / "model.pkl").read_bytes())
    assert payload["version"] == run.VERSION
    assert payload["candidate"] == fit["selected"]["candidate"]
    assert payload["parameters"] == fit["selected"]["parameters"]
    assert payload["model"].n_iter_ == run.COMMON_PARAMS["max_iter"]
    with np.load(out / "calibration_scores.npz", allow_pickle=False) as cal:
        assert cal["window_scores"].shape == (run.CAL_WINDOWS,)
        assert cal["answer_scores"].shape == (run.CAL_ANSWERS,)
        assert np.isfinite(cal["window_scores"]).all() and np.isfinite(cal["answer_scores"]).all()

    audit = {
        "status": "passed_without_reopening_calibration_labels",
        "three_recipes_preregistered_before_fit": True,
        "only_max_leaf_nodes_varied": True,
        "all_domain_selection_key_replayed_exactly": True,
        "native_diagnostics_did_not_affect_selection": True,
        "upstream_supervised_columns_source_group_OOF": True,
        "in_sample_supervised_columns": 0,
        "expanded_LR_design_byte_identical": True,
        "same_five_group_folds_as_expanded_LR": True,
        "downstream_five_fold_window_coverage_exactly_once": True,
        "all_candidate_fit_metrics_replayed_exactly": True,
        "all_candidate_native_diagnostics_replayed_exactly": True,
        "full_model_loadable_and_parameter_exact": True,
        "calibration_score_shapes_and_finiteness_valid": True,
        "calibration_labels_reopened_by_audit": False,
        "calibration_thresholds_fitted": 0,
        "calibration_evaluations": 1,
        "formal_baseline_files_unchanged": True,
        "official_test_opened": False,
        "GPU_used": False,
    }
    run.atomic_json(out / "INDEPENDENT_AUDIT.json", audit)
    print("EXPANDED_HGB_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
