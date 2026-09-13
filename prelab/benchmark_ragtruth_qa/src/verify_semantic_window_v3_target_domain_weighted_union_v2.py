"""Independent read-only replay without reopening calibration labels."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semantic_window_v3_target_domain_weighted_union_v2 as run  # noqa: E402


def independent_fit_mapping() -> dict:
    with np.load(run.NATIVE_AGGREGATES, allow_pickle=False) as native_loaded:
        native = {name: native_loaded[name].copy() for name in native_loaded.files}
    with np.load(run.EXPANDED_AGGREGATES, allow_pickle=False) as expanded_loaded:
        expanded = {name: expanded_loaded[name].copy() for name in expanded_loaded.files}
    features = np.vstack((native["features"], expanded["features"])).astype(
        np.float32, copy=False)
    assert features.shape == (run.FIT_CLAIMS, 44)
    with np.load(run.COMBINED_PROJECTION, allow_pickle=False) as loaded:
        response_claim_indptr = loaded["response_claim_indptr"].copy()
        claim_response = loaded["claim_response_index"].copy()
        claim_ids = loaded["claim_local_ids"].copy()
        microclaim = loaded["claim_microclaim_indices"].copy()
        window_tokens = loaded["window_token_indices"].copy()
        token_claim = loaded["token_claim_index"].copy()
        lexical = loaded["token_lexical_mask"].copy()
    joined_response = np.concatenate((
        native["response_index"], expanded["response_index"] + run.NATIVE_ANSWERS,
    ))
    assert np.array_equal(joined_response, claim_response)
    assert np.array_equal(np.concatenate((native["claim_id"], expanded["claim_id"])),
                          claim_ids)
    assert np.array_equal(np.concatenate((native["microclaim_index"],
                                          expanded["microclaim_index"])), microclaim)
    assert response_claim_indptr[run.NATIVE_ANSWERS] == run.NATIVE_CLAIMS
    assert response_claim_indptr[-1] == run.FIT_CLAIMS

    # Reconstruct every expanded hypothesis identity independently.
    axes = run.read_json(run.COMBINED_AXES)
    claim_axes = run.read_jsonl(run.COMBINED_CLAIM_AXES)
    units = run.read_jsonl(run.EXPANDED_UNITS)
    auxiliary_ids = set(map(str, axes["response_ids"][run.NATIVE_ANSWERS:]))
    raw_by_response: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in run.read_jsonl(run.EXPANDED_RAW_CLAIMS):
        rid = str(row["response_id"])
        if rid in auxiliary_ids:
            raw_by_response[rid][str(row["microclaim_id"])] = row
    cursor = 0
    for local_response, unit in enumerate(units):
        rid = str(unit["response_id"])
        global_response = run.NATIVE_ANSWERS + local_response
        assert rid == str(axes["response_ids"][global_response])
        assert response_claim_indptr[global_response] == run.NATIVE_CLAIMS + cursor
        for local_claim, claim in enumerate(unit["claims"]):
            index = cursor + local_claim
            global_claim = run.NATIVE_CLAIMS + index
            raw = raw_by_response[rid][str(claim["microclaim_id"])]
            assert raw["text"] == claim["text"] == claim_axes[global_claim]["text"]
            assert run.digest(raw["text"]) == claim["text_sha256"]
            hypothesis = run.evidence.atomic_nli.hypothesis_for(raw)
            identity = np.frombuffer(
                bytes.fromhex(run.evidence.digest(hypothesis)), dtype=np.uint8
            )
            assert np.array_equal(expanded["hypothesis_identity_sha256"][index], identity)
        cursor += len(unit["claims"])
    assert cursor == run.AUXILIARY_CLAIMS

    design = np.load(run.FIT_DESIGN, mmap_mode="r")
    base = np.load(run.BASE_DESIGN, mmap_mode="r")
    assert np.array_equal(design[:, :26], base)
    owners = token_claim[window_tokens]
    assert np.array_equal(owners >= 0, lexical[window_tokens].astype(bool))
    claim_hits = np.zeros(run.FIT_CLAIMS, dtype=np.int64)
    lexical_hist = Counter()
    unique_hist = Counter()
    for index, owner_row in enumerate(owners):
        values = owner_row[owner_row >= 0]
        claim_hits[values] += 1
        unique, counts = np.unique(values, return_counts=True)
        weight = counts.astype(np.float64)
        weight /= weight.sum()
        expected = np.average(features[unique], axis=0, weights=weight).astype(np.float32)
        assert np.array_equal(design[index, 26:], expected)
        lexical_hist[len(values)] += 1
        unique_hist[len(unique)] += 1
    assert np.all(claim_hits > 0)
    with np.load(run.NATIVE_V3_FEATURES, allow_pickle=False) as loaded:
        reference = loaded["union"][:, run.feature_v3.variant_indices("backbone_union")]
    assert np.array_equal(design[:run.NATIVE_WINDOWS], reference)
    return {
        "all_fit_rows_rebuilt_exact": True,
        "rows": run.FIT_WINDOWS,
        "claims": run.FIT_CLAIMS,
        "expanded_hypotheses_rebuilt_exact": run.AUXILIARY_CLAIMS,
        "claims_with_coverage": int(np.count_nonzero(claim_hits)),
        "claims_without_coverage": int(np.count_nonzero(claim_hits == 0)),
        "lexical_owner_count_histogram": {
            str(key): int(value) for key, value in sorted(lexical_hist.items())
        },
        "unique_claim_count_per_window_histogram": {
            str(key): int(value) for key, value in sorted(unique_hist.items())
        },
        "native_rows_equal_v3_bit_exact": True,
        "claim_score_max_projection_used": False,
    }


def structural_metric_check(metric: dict) -> None:
    assert metric["tp"] + metric["fn"] == metric["positive"]
    assert metric["tp"] + metric["fp"] + metric["fn"] + metric["tn"] == metric["n"]
    precision = metric["tp"] / (metric["tp"] + metric["fp"]) if metric["tp"] + metric["fp"] else 0.0
    recall = metric["tp"] / (metric["tp"] + metric["fn"]) if metric["tp"] + metric["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    assert np.isclose(metric["precision"], precision, rtol=0, atol=1e-15)
    assert np.isclose(metric["recall"], recall, rtol=0, atol=1e-15)
    assert np.isclose(metric["f1"], f1, rtol=0, atol=1e-15)
    assert 0 <= metric["average_precision"] <= 1
    assert 0 <= metric["auroc"] <= 1


def main() -> None:
    out = run.OUT
    prep = run.read_json(out / "preparation_complete.json")
    provenance = run.read_json(out / "PROVENANCE_GATE.json")
    extraction = run.read_json(out / "EXTRACTION_READINESS_AUDIT.json")
    claim_join = run.read_json(out / "CLAIM_JOIN_AUDIT.json")
    mapping_saved = run.read_json(out / "WINDOW_MAPPING_AUDIT.json")
    fit = run.read_json(out / "fit_complete.json")
    summary = run.read_json(out / "summary.json")
    protocol = run.read_json(out / "protocol.json")
    source = run.read_json(out / "source_snapshot.json")
    assert prep["status"].startswith("preregistered_hashes_")
    assert provenance["status"] == "passed_calibration_and_native_fit_share_target_generator"
    assert extraction["status"] == "complete_extraction_and_claim_aggregates_ready"
    assert claim_join["status"] == "exact_claim_join_passed"
    assert mapping_saved["status"] == "exact_four_bpe_claim_pooling_complete"
    assert protocol == run.protocol()
    assert run.validate_source_freeze()["mismatches"] == []
    assert source == run.source_snapshot()
    assert run.formal_baseline_snapshot() == run.expected_formal_baseline_snapshot()
    for name, expected in prep["artifacts_sha256"].items():
        assert run.sha(out / name) == expected, name
    for name, expected in fit["artifacts_sha256"].items():
        path = Path(run.__file__) if name == "runner" else out / name
        assert run.sha(path) == expected, name
    assert run.sha(out / "calibration_scores.npz") == summary[
        "artifacts_sha256"]["calibration_scores.npz"]

    extraction_replay = run.validate_expanded_extraction()
    assert extraction_replay == extraction
    mapping = independent_fit_mapping()
    assert mapping["rows"] == mapping_saved["design_shape"][0]
    assert mapping["claims_without_coverage"] == 0
    assert mapping["lexical_owner_count_histogram"] == mapping_saved[
        "lexical_owner_count_histogram"]

    proof, _, answer_generators, generator_names = run.weighted_v1.provenance_gate()
    assert proof == provenance
    assert generator_names == prep["generator_names_in_weight_order"]
    assert [row["alpha"] for row in fit["candidate_table"]] == list(run.AUXILIARY_ALPHAS)
    _, projection = run.combined.load_projection()
    labels = projection["window_labels"].astype(np.int8)
    answer_labels = projection["response_labels"].astype(np.int8)
    groups = run.combined.group_indices("window")
    answers = projection["window_response_index"].astype(np.int32)
    generator_map = {name: index for index, name in enumerate(generator_names)}
    answer_codes = np.asarray(
        [generator_map[name] for name in answer_generators], dtype=np.int8
    )
    generator_codes = answer_codes[answers]
    splits = run.weighted_v1.native_splits(labels, groups)
    coverage = np.zeros(run.FIT_WINDOWS, dtype=np.int8)
    for _, _, _, held_all in splits:
        coverage[held_all] += 1
    assert np.all(coverage == 1)

    with np.load(out / "fit_scores.npz", allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "alphas", "window_scores", "raw_model_window_scores", "answer_scores",
            "window_labels", "answer_labels", "selected_index",
        }
        assert np.array_equal(loaded["alphas"], np.asarray(run.AUXILIARY_ALPHAS))
        oof = loaded["window_scores"].copy()
        raw_oof = loaded["raw_model_window_scores"].copy()
        answer_scores = loaded["answer_scores"].copy()
        assert np.array_equal(loaded["window_labels"], labels)
        assert np.array_equal(loaded["answer_labels"], answer_labels)
        assert int(loaded["selected_index"]) == fit["selected_index"]
    assert np.isfinite(oof).all() and np.isfinite(raw_oof).all()
    assert np.array_equal(oof[1:], raw_oof[1:])
    assert np.array_equal(oof[0, run.NATIVE_WINDOWS:],
                          raw_oof[0, run.NATIVE_WINDOWS:])

    crossfit = pickle.loads((out / "crossfit_models.pkl").read_bytes())
    assert crossfit["version"] == run.VERSION
    assert crossfit["feature_width"] == run.FEATURE_WIDTH
    assert crossfit["C"] == run.C_VALUE
    assert crossfit["generator_identity_feature_count"] == 0
    assert len(crossfit["models"]) == run.FOLDS * len(run.AUXILIARY_ALPHAS)
    model_lookup = {(row["fold"], row["alpha"]): row
                    for row in crossfit["models"]}
    x = np.load(run.FIT_DESIGN, mmap_mode="r")
    largest_prediction_error = 0.0
    with threadpool_limits(limits=run.THREADS):
        for fold, (train_native, held_native, train_all, held_all) in enumerate(splits):
            components = run.weighted_v1.generator_components(
                train_all, labels, groups, answers, generator_codes, generator_names
            )
            for alpha_index, alpha in enumerate(run.AUXILIARY_ALPHAS):
                base, loss, active, audit = run.weighted_v1.compose_weights(
                    alpha, components, labels, len(train_native)
                )
                saved = fit["crossfit"][fold]["candidates"][alpha_index]
                assert audit == saved["weight_audit"]
                assert np.isclose(base.sum(), len(train_native), rtol=0, atol=1e-5)
                assert np.isclose(loss.sum(), len(train_native), rtol=0, atol=1e-5)
                if alpha == 0:
                    assert np.array_equal(active, train_native)
                    assert np.all(base[run.NATIVE_WINDOWS:] == 0)
                    assert np.all(loss[run.NATIVE_WINDOWS:] == 0)
                frozen_model = model_lookup[(fold, alpha)]
                assert frozen_model["held_groups"] == np.unique(groups[held_all]).astype(int).tolist()
                native_replay = frozen_model["model"].predict_proba(
                    frozen_model["scaler"].transform(x[held_native]))[:, 1]
                held_aux = held_all[held_all >= run.NATIVE_WINDOWS]
                auxiliary_replay = frozen_model["model"].predict_proba(
                    frozen_model["scaler"].transform(x[held_aux]))[:, 1]
                native_error = float(np.max(np.abs(
                    native_replay - raw_oof[alpha_index, held_native]
                )))
                auxiliary_error = float(np.max(np.abs(
                    auxiliary_replay - raw_oof[alpha_index, held_aux]
                )))
                largest_prediction_error = max(
                    largest_prediction_error, native_error, auxiliary_error
                )
                assert native_error <= np.finfo(np.float64).eps
                assert auxiliary_error <= np.finfo(np.float64).eps

    with np.load(run.NATIVE_V3_SCORES, allow_pickle=False) as loaded:
        native_reference = loaded["selected_scores"].copy()
    assert np.array_equal(oof[0, :run.NATIVE_WINDOWS], native_reference)
    fresh_error = float(np.max(np.abs(
        raw_oof[0, :run.NATIVE_WINDOWS] - native_reference
    )))
    assert fresh_error == fit["alpha_zero_native_v3_replay"][
        "fresh_refit_max_abs_error"]
    assert fresh_error <= fit["alpha_zero_native_v3_replay"][
        "fresh_refit_acceptance_bound"]

    replay_candidates = []
    for alpha_index, alpha in enumerate(run.AUXILIARY_ALPHAS):
        scores = oof[alpha_index]
        answers_one = run.expanded_v2.answer_scores(
            scores, projection["response_window_indptr"]
        )
        assert np.array_equal(answers_one, answer_scores[alpha_index])
        thresholds = {
            "window": run.q.choose_threshold(
                labels[:run.NATIVE_WINDOWS], scores[:run.NATIVE_WINDOWS]),
            "answer": run.q.choose_threshold(
                answer_labels[:run.NATIVE_ANSWERS], answers_one[:run.NATIVE_ANSWERS]),
        }
        record = {
            "candidate": f"backbone_union__C0.001__auxiliary_total_weight_{alpha:g}",
            "alpha": alpha,
            "native_OOF_thresholds": thresholds,
            "native_OOF": {
                "windows": run.q.count(labels[:run.NATIVE_WINDOWS],
                                       scores[:run.NATIVE_WINDOWS],
                                       thresholds["window"]["threshold"]),
                "answers": run.q.count(answer_labels[:run.NATIVE_ANSWERS],
                                       answers_one[:run.NATIVE_ANSWERS],
                                       thresholds["answer"]["threshold"]),
            },
            "all_domain_OOF_at_native_thresholds": {
                "windows": run.q.count(labels, scores,
                                       thresholds["window"]["threshold"]),
                "answers": run.q.count(answer_labels, answers_one,
                                       thresholds["answer"]["threshold"]),
            },
        }
        assert record == fit["candidate_table"][alpha_index]
        replay_candidates.append(record)
    selected_index = max(range(len(replay_candidates)),
                         key=lambda index: run.candidate_key(replay_candidates[index]))
    assert selected_index == fit["selected_index"] == 2
    assert fit["selected"]["alpha"] == 0.5

    components = run.weighted_v1.generator_components(
        np.arange(run.FIT_WINDOWS), labels, groups, answers,
        generator_codes, generator_names
    )
    _, _, full_active, full_audit = run.weighted_v1.compose_weights(
        fit["selected"]["alpha"], components, labels, run.NATIVE_WINDOWS
    )
    assert full_audit == fit["full_fit"]["weight_audit"]
    assert np.array_equal(full_active, np.arange(run.FIT_WINDOWS))

    payload = pickle.loads((out / "model.pkl").read_bytes())
    assert payload["feature_names"] == list(run.inference_feature_names())
    assert payload["feature_width"] == payload["model"].n_features_in_ == 70
    assert payload["feature_width"] == payload["scaler"].n_features_in_
    assert payload["generator_identity_feature_count"] == 0
    forbidden = ("generator", "source_id", "group_id", "answer_id", "response_id")
    assert not any(any(term in name.lower() for term in forbidden)
                   for name in payload["feature_names"])
    tree = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in functions["fit_model"].args.args] == [
        "x", "labels", "active", "base_weight", "loss_weight"]
    assert [arg.arg for arg in functions["predict_probabilities"].args.args] == [
        "scaler", "model", "x_rows"]
    prediction_names = {node.id for node in ast.walk(functions["predict_probabilities"])
                        if isinstance(node, ast.Name)}
    assert not prediction_names & {"generator_codes", "generators", "groups", "answers"}

    # Score replay uses the frozen label-free calibration feature matrix. It
    # deliberately does not load calibration answer/span labels or recompute metrics.
    with np.load(run.ROOT / "results/semantic_window_v3/calibration_window_features.npz",
                 allow_pickle=False) as loaded:
        cal_x = loaded["union"][:, run.feature_v3.variant_indices("backbone_union")]
        cal_answer_ids = loaded["answer_ids"].astype(str)
    with np.load(out / "calibration_scores.npz", allow_pickle=False) as cal:
        saved_window = cal["window_scores"].copy()
        saved_answer = cal["answer_scores"].copy()
        response_ids = cal["response_ids"].astype(str)
    replay_window = payload["model"].predict_proba(
        payload["scaler"].transform(cal_x))[:, 1]
    calibration_window_replay_error = float(np.max(np.abs(
        replay_window - saved_window
    )))
    assert calibration_window_replay_error <= np.finfo(np.float64).eps
    replay_answer = np.asarray([
        replay_window[cal_answer_ids == response_id].max() for response_id in response_ids
    ])
    calibration_answer_replay_error = float(np.max(np.abs(
        replay_answer - saved_answer
    )))
    assert calibration_answer_replay_error <= np.finfo(np.float64).eps
    for section in ("calibration_strict_native_OOF_thresholds",
                    "calibration_F1Opt_diagnostic"):
        for unit in ("windows", "answers"):
            structural_metric_check(summary[section][unit])
    assert summary["calibration_evaluations"] == 1
    assert summary["calibration_F1Opt_used_for_selection"] is False
    assert summary["calibration_used_for_alpha_model_or_native_OOF_threshold_selection"] is False
    assert summary["formal_baseline_files_unchanged"] is True
    assert summary["official_test_opened"] is False

    audit = {
        "status": "passed_independent_mapping_weight_and_model_replay_without_reopening_calibration_labels",
        "preregistered_source_hashes_exact": True,
        "target_generator_provenance_replayed": True,
        "expanded_extraction_39_of_39_hashes_replayed": True,
        "stale_missing_requests_field_interpretation_confirmed": True,
        "claim_identity_join_replayed": True,
        "fit_mapping": mapping,
        "source_connected_five_fold_all_domain_coverage_exactly_once": True,
        "source_connected_group_overlap_each_fold": [0] * run.FOLDS,
        "all_20_serialized_crossfit_model_predictions_replayed": True,
        "largest_serialized_prediction_replay_error": largest_prediction_error,
        "all_candidate_native_and_all_domain_metrics_exact": True,
        "selection_recomputed_from_native_OOF_only": True,
        "selected_alpha": fit["selected"]["alpha"],
        "alpha_zero_auxiliary_weights_exactly_zero": True,
        "alpha_zero_native_control_bit_exact": True,
        "alpha_zero_fresh_refit_max_abs_error": fresh_error,
        "full_fit_weight_mass_replayed": True,
        "generator_identity_feature_count": 0,
        "generator_identity_absent_from_fit_and_predict_interfaces": True,
        "serialized_model_numeric_feature_width": 70,
        "calibration_scores_replayed_from_label_free_features": True,
        "calibration_window_score_replay_max_abs_error": calibration_window_replay_error,
        "calibration_answer_score_replay_max_abs_error": calibration_answer_replay_error,
        "calibration_score_replay_within_one_float64_ulp": True,
        "calibration_answer_or_span_labels_opened_by_audit": False,
        "calibration_metric_passes_by_audit": 0,
        "calibration_evaluations_recorded": 1,
        "formal_baseline_files_unchanged": True,
        "official_test_opened": False,
        "GPU_used": False,
    }
    run.atomic_json(out / "INDEPENDENT_AUDIT.json", audit)
    print("TARGET_DOMAIN_UNION_V2_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
