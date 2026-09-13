"""Independent replay of target-domain weighting without reopening cal labels."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semantic_window_v2_target_domain_weighted_v1 as run  # noqa: E402


def main() -> None:
    out = run.OUT
    prep = run.read_json(out / "preparation_complete.json")
    provenance = run.read_json(out / "PROVENANCE_GATE.json")
    fit = run.read_json(out / "fit_complete.json")
    summary = run.read_json(out / "summary.json")
    protocol = run.read_json(out / "protocol.json")
    source = run.read_json(out / "source_snapshot.json")
    assert prep["status"] == "target_provenance_passed_and_weight_grid_frozen_before_fit"
    assert provenance["status"] == "passed_calibration_and_native_fit_share_target_generator"
    assert protocol == run.protocol()
    assert source == run.source_snapshot()
    assert source["formal_baseline_files_sha256"] == run.expanded_v2.baseline_snapshot()
    for name, expected in fit["artifacts_sha256"].items():
        path = Path(run.__file__) if name == "runner" else out / name
        assert run.sha(path) == expected
    assert run.sha(out / "calibration_scores.npz") == summary["artifacts_sha256"]["calibration_scores.npz"]

    proof, rows, answer_generators, generator_names = run.provenance_gate()
    assert proof == provenance
    assert generator_names == prep["generator_names_in_weight_order"]
    assert [record["alpha"] for record in fit["candidate_table"]] == list(run.AUXILIARY_ALPHAS)

    _, projection = run.combined.load_projection()
    labels = projection["window_labels"].astype(np.int8)
    answer_labels = projection["response_labels"].astype(np.int8)
    groups = run.combined.group_indices("window")
    answers = projection["window_response_index"].astype(np.int32)
    mapping = {name: index for index, name in enumerate(generator_names)}
    answer_codes = np.asarray([mapping[name] for name in answer_generators], dtype=np.int8)
    generator_codes = answer_codes[answers]
    splits = run.native_splits(labels, groups)
    coverage = np.zeros(run.FIT_WINDOWS, dtype=np.int8)
    for _, _, _, held_all in splits:
        coverage[held_all] += 1
    assert np.all(coverage == 1)

    with np.load(out / "fit_scores.npz", allow_pickle=False) as loaded:
        assert np.array_equal(loaded["alphas"], np.asarray(run.AUXILIARY_ALPHAS))
        oof = loaded["window_scores"].copy()
        answer_scores = loaded["answer_scores"].copy()
        assert np.array_equal(loaded["window_labels"], labels)
        assert np.array_equal(loaded["answer_labels"], answer_labels)
        assert int(loaded["selected_index"]) == fit["selected_index"]
    assert oof.shape == (4, run.FIT_WINDOWS) and np.isfinite(oof).all()
    assert answer_scores.shape == (4, run.FIT_ANSWERS) and np.isfinite(answer_scores).all()

    crossfit = pickle.loads((out / "crossfit_models.pkl").read_bytes())
    assert crossfit["version"] == run.VERSION
    assert crossfit["feature_width"] == run.FEATURE_WIDTH
    assert crossfit["generator_identity_feature_count"] == 0
    assert len(crossfit["models"]) == run.FOLDS * len(run.AUXILIARY_ALPHAS)
    model_lookup = {(record["fold"], record["alpha"]): record for record in crossfit["models"]}
    x = np.load(run.DESIGN, mmap_mode="r")
    with threadpool_limits(limits=run.THREADS):
        for fold_number, (train_native, _, train_all, held_all) in enumerate(splits):
            components = run.generator_components(
                train_all, labels, groups, answers, generator_codes, generator_names
            )
            for candidate_index, alpha in enumerate(run.AUXILIARY_ALPHAS):
                _, loss, active, weight_audit = run.compose_weights(
                    alpha, components, labels, len(train_native)
                )
                saved_fold = fit["crossfit"][fold_number]["candidates"][candidate_index]
                assert weight_audit == saved_fold["weight_audit"]
                if alpha == 0:
                    assert np.array_equal(active, train_native)
                    assert np.all(loss[run.NATIVE_WINDOWS:] == 0)
                saved_model = model_lookup[(fold_number, alpha)]
                assert saved_model["held_groups"] == np.unique(groups[held_all]).astype(int).tolist()
                replay = run.predict_probabilities(
                    saved_model["scaler"], saved_model["model"], x[held_all]
                )
                assert np.array_equal(replay, oof[candidate_index, held_all])

    replay_candidates = []
    for candidate_index, alpha in enumerate(run.AUXILIARY_ALPHAS):
        scores = oof[candidate_index]
        answers_one = run.expanded_v2.answer_scores(scores, projection["response_window_indptr"])
        assert np.array_equal(answers_one, answer_scores[candidate_index])
        thresholds = {
            "window": run.q.choose_threshold(labels[:run.NATIVE_WINDOWS], scores[:run.NATIVE_WINDOWS]),
            "answer": run.q.choose_threshold(
                answer_labels[:run.NATIVE_ANSWERS], answers_one[:run.NATIVE_ANSWERS]
            ),
        }
        record = {
            "candidate": f"auxiliary_total_weight_{alpha:g}",
            "alpha": alpha,
            "native_OOF_thresholds": thresholds,
            "native_OOF": {
                "windows": run.q.count(
                    labels[:run.NATIVE_WINDOWS], scores[:run.NATIVE_WINDOWS], thresholds["window"]["threshold"]
                ),
                "answers": run.q.count(
                    answer_labels[:run.NATIVE_ANSWERS], answers_one[:run.NATIVE_ANSWERS],
                    thresholds["answer"]["threshold"],
                ),
            },
            "all_domain_OOF_at_native_thresholds": {
                "windows": run.q.count(labels, scores, thresholds["window"]["threshold"]),
                "answers": run.q.count(answer_labels, answers_one, thresholds["answer"]["threshold"]),
            },
        }
        assert record == fit["candidate_table"][candidate_index]
        replay_candidates.append(record)
    selected_index = max(range(4), key=lambda index: run.candidate_key(replay_candidates[index]))
    assert selected_index == fit["selected_index"] == 0
    assert fit["selected"]["alpha"] == 0

    with np.load(run.NATIVE_SCORES, allow_pickle=False) as native:
        difference = np.abs(oof[0, :run.NATIVE_WINDOWS] - native["selected_scores"])
    assert float(difference.max()) <= np.finfo(np.float64).eps
    assert fit["alpha_zero_native_v2_replay"]["threshold_decisions_and_confusion_counts_identical"] is True

    components = run.generator_components(
        np.arange(run.FIT_WINDOWS), labels, groups, answers, generator_codes, generator_names
    )
    _, full_loss, full_active, full_weight_audit = run.compose_weights(
        fit["selected"]["alpha"], components, labels, run.NATIVE_WINDOWS
    )
    assert full_weight_audit == fit["full_fit"]["weight_audit"]
    assert np.array_equal(full_active, np.arange(run.NATIVE_WINDOWS))
    assert np.all(full_loss[run.NATIVE_WINDOWS:] == 0)

    payload = pickle.loads((out / "model.pkl").read_bytes())
    assert payload["feature_names"] == list(run.feature_v2.WHITEBOX_NAMES + run.feature_v2.GEOMETRY_NAMES)
    assert payload["feature_width"] == payload["model"].n_features_in_ == payload["scaler"].n_features_in_ == 26
    assert payload["generator_identity_feature_count"] == 0
    forbidden = ("generator", "source_id", "group_id", "answer_id", "model_id")
    assert not any(any(term in name.lower() for term in forbidden) for name in payload["feature_names"])

    tree = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in functions["fit_model"].args.args] == [
        "x", "labels", "active", "base_weight", "loss_weight"
    ]
    assert [arg.arg for arg in functions["predict_probabilities"].args.args] == [
        "scaler", "model", "x_rows"
    ]
    prediction_names = {node.id for node in ast.walk(functions["predict_probabilities"]) if isinstance(node, ast.Name)}
    assert not prediction_names & {"generators", "generator_codes", "groups", "answers"}

    with np.load(out / "calibration_scores.npz", allow_pickle=False) as cal, np.load(
        run.native_v2.DEFAULT_OUT / "calibration_scores.npz", allow_pickle=False
    ) as native_cal:
        assert cal["window_scores"].shape == (run.CAL_WINDOWS,)
        assert cal["answer_scores"].shape == (run.CAL_ANSWERS,)
        assert np.isfinite(cal["window_scores"]).all() and np.isfinite(cal["answer_scores"]).all()
        assert np.array_equal(cal["window_scores"], native_cal["window_scores"])
        assert np.array_equal(cal["answer_scores"], native_cal["answer_scores"])

    native_summary = run.read_json(run.NATIVE_SUMMARY)
    assert summary["calibration_strict_native_OOF_thresholds"] == native_summary["calibration_strict_fit_thresholds"]
    assert summary["calibration_F1Opt_diagnostic"] == native_summary["calibration_F1Opt_diagnostic"]
    assert summary["calibration_evaluations"] == 1
    assert summary["calibration_F1Opt_used_for_selection"] is False
    assert summary["calibration_used_for_alpha_model_or_native_OOF_threshold_selection"] is False
    assert summary["inference_input_audit"]["generator_identity_feature_count"] == 0
    assert summary["formal_baseline_files_unchanged"] is True
    assert summary["official_test_opened"] is False

    audit = {
        "status": "passed_independent_fit_replay_without_reopening_calibration_labels",
        "target_generator_provenance_replayed": True,
        "calibration_identity_proved_from_hash_bound_manifest_and_exporter": True,
        "calibration_answer_or_span_rows_opened_by_audit": False,
        "candidate_grid_exact": list(run.AUXILIARY_ALPHAS),
        "source_connected_five_fold_all_domain_coverage_exactly_once": True,
        "all_20_crossfit_model_predictions_exact": True,
        "all_candidate_native_and_all_domain_metrics_exact": True,
        "selection_native_only": True,
        "selected_alpha": 0.0,
        "alpha_zero_auxiliary_weights_exactly_zero": True,
        "alpha_zero_native_score_replay_max_abs_error": float(difference.max()),
        "alpha_zero_native_threshold_decisions_and_confusion_counts_identical": True,
        "full_fit_weight_mass_replayed": True,
        "generator_identity_feature_count": 0,
        "generator_identity_absent_from_fit_and_predict_interfaces": True,
        "serialized_model_numeric_feature_width": 26,
        "calibration_scores_exact_native_control_without_label_reopen": True,
        "calibration_evaluations": 1,
        "calibration_thresholds_used_for_selection": 0,
        "formal_baseline_files_unchanged": True,
        "official_test_opened": False,
        "GPU_used": False,
    }
    run.atomic_json(out / "INDEPENDENT_AUDIT.json", audit)
    print("TARGET_DOMAIN_WEIGHTING_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
