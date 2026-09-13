"""Independent read-only audit for semantic-window v3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parents[1]
sys.path.insert(0, str(QA_ROOT / "src"))

import run_semantic_window_v3 as v3  # noqa: E402


def assert_nested_close(observed, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        assert set(observed) == set(expected), path
        for key, value in expected.items():
            assert_nested_close(observed[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(observed) == len(expected), path
        for index, value in enumerate(expected):
            assert_nested_close(observed[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert np.isclose(observed, expected, rtol=0, atol=1e-12), (
            path, observed, expected,
        )
    else:
        assert observed == expected, (path, observed, expected)


def validate_evidence_manifest(partition: str) -> dict:
    cohort = v3.COHORT[partition]
    manifest = v3.read_json(v3.EVIDENCE_OUT / f"manifest_{cohort}.json")
    checked = 0
    for record in manifest["candidate_files"].values():
        assert v3.sha(v3.EVIDENCE_OUT / record["path"]) == record["sha256"]
        checked += 1
    snapshot = manifest["source_snapshot"]
    assert snapshot["runner_sha256"] == v3.sha(v3.EVIDENCE_RUNNER)
    assert snapshot["protocol_sha256"] == v3.sha(v3.EVIDENCE_OUT / "protocol.json")
    assert manifest["missing_cache_instances"] == 0
    assert manifest["existing_cache_hit_instances"] == manifest["candidate_instances"]
    return {
        "cohort": cohort,
        "manifest_declared_files_checked": checked,
        "runner_and_protocol_hashes_exact": True,
        "candidate_instances": manifest["candidate_instances"],
        "cache_hit_fraction": 1.0,
    }


def rebuild_feature_identity(out: Path, partition: str) -> tuple[dict, dict]:
    rebuilt, _, meta, source_audit = v3.build_partition_design(partition)
    path = out / (
        "fit_window_features.npz" if partition == "fit"
        else "calibration_window_features.npz"
    )
    expected_names = v3.feature_v3.feature_names(v3.evidence_feature_names())
    with np.load(path, allow_pickle=False) as loaded:
        assert np.array_equal(loaded["union"], rebuilt)
        assert np.array_equal(loaded["feature_names"], expected_names)
        for name in ("window_ids", "response_ids", "group_ids", "answer_ids"):
            expected = np.asarray([
                row[name[:-1]] for row in meta["windows"]
            ])
            assert np.array_equal(loaded[name], expected), name
        window_ids = loaded["window_ids"].astype(str)
    assert len(np.unique(window_ids)) == v3.EXPECTED_WINDOWS[partition]
    return meta, {
        "rows": len(rebuilt), "width": rebuilt.shape[1],
        "unique_window_ids": len(window_ids),
        "saved_matrix_equals_source_rebuild": True,
        "all_window_identity_columns_exact": True,
        "claim_identity": source_audit["evidence_union"],
        "frozen_v2_backbone_identity": source_audit["backbone"],
    }


def audit_fit(out: Path, fit_done: dict) -> dict:
    meta, feature_audit = rebuild_feature_identity(out, "fit")
    expected_names = [
        f"{variant}__C{c_value:g}"
        for variant in v3.VARIANTS for c_value in v3.C_VALUES
    ]
    with np.load(out / "fit_scores.npz", allow_pickle=False) as loaded:
        names = loaded["candidate_names"].astype(str).tolist()
        scores = loaded["candidate_scores"].copy()
        selected_scores = loaded["selected_scores"].copy()
        labels = loaded["labels"].copy()
        selected_index = int(loaded["selected_index"])
        stored_answer_scores = loaded["answer_scores"].copy()
    assert names == expected_names
    assert scores.shape == (6, v3.EXPECTED_WINDOWS["fit"])
    assert np.isfinite(scores).all()
    expected_labels = np.asarray(
        [row["label"] for row in meta["windows"]], dtype=np.int8
    )
    assert np.array_equal(labels, expected_labels)
    table = fit_done["candidate_table"]
    assert [record["candidate"] for record in table] == names
    for index, record in enumerate(table):
        thresholds = v3.v2.choose_thresholds(meta, scores[index])
        metrics = v3.v2.metric_pair(meta, scores[index], thresholds)
        assert_nested_close(thresholds, record["fit_OOF_thresholds"])
        assert_nested_close(metrics, record["fit_OOF"])
    selected_recomputed = max(
        range(len(table)),
        key=lambda index: v3.candidate_selection_key(table[index], index),
    )
    assert selected_recomputed == selected_index
    assert names[selected_index] == fit_done["selected"]["candidate"]
    assert np.array_equal(selected_scores, scores[selected_index])
    assert np.array_equal(stored_answer_scores, v3.v2.answer_scores(meta, selected_scores))

    groups = np.asarray([row["group_id"] for row in meta["windows"]])
    coverage = np.zeros(len(labels), dtype=np.int8)
    fold_sizes = []
    for fold, (train, held) in enumerate(
        GroupKFold(v3.FOLDS).split(np.arange(len(labels)), labels, groups)
    ):
        assert not set(groups[train]) & set(groups[held])
        coverage[held] += 1
        logged = fit_done["crossfit"][fold]
        assert len(train) == logged["train_windows"]
        assert len(held) == logged["held_windows"]
        assert len(set(groups[train])) == logged["train_groups"]
        assert len(set(groups[held])) == logged["held_groups"]
        fold_sizes.append(len(held))
    assert np.all(coverage == 1)
    upstream = v3.v2.assert_fit_upstream_source_group_oof(meta)
    assert_nested_close(upstream, fit_done["upstream_OOF_audit"])
    return {
        "feature_rebuild": feature_audit,
        "candidates_recomputed": len(table),
        "candidate_names_match_preregistration": True,
        "selected_index": selected_index,
        "selected_candidate": names[selected_index],
        "selection_recomputed_from_fit_OOF_only": True,
        "all_fit_thresholds_and_counts_exact": True,
        "fold_held_windows": fold_sizes,
        "each_fit_window_held_exactly_once": True,
        "all_fold_group_intersections_empty": True,
        "upstream_source_group_OOF_reconstructed": upstream,
    }


def audit_calibration(out: Path, fit_done: dict, summary: dict) -> dict:
    meta, feature_audit = rebuild_feature_identity(out, "calibration")
    with np.load(out / "calibration_window_features.npz", allow_pickle=False) as features:
        union = features["union"].copy()
    with np.load(out / "calibration_scores.npz", allow_pickle=False) as loaded:
        scores = loaded["window_scores"].copy()
        answer_scores = loaded["answer_scores"].copy()
        expected_ids = np.asarray([row["window_id"] for row in meta["windows"]])
        expected_responses = np.asarray([row["response_id"] for row in meta["windows"]])
        assert np.array_equal(loaded["window_ids"], expected_ids)
        assert np.array_equal(loaded["response_ids"], expected_responses)
    with (out / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    assert model["variant"] == fit_done["selected"]["variant"]
    assert model["C"] == fit_done["selected"]["C"]
    assert np.array_equal(
        model["columns"], v3.feature_v3.variant_indices(model["variant"])
    )
    reproduced = model["model"].predict_proba(
        model["scaler"].transform(union[:, model["columns"]])
    )[:, 1]
    assert np.array_equal(reproduced, scores)
    assert np.array_equal(answer_scores, v3.v2.answer_scores(meta, scores))
    strict = v3.v2.metric_pair(
        meta, scores, fit_done["selected"]["fit_OOF_thresholds"]
    )
    optimum = v3.v2.choose_thresholds(meta, scores)
    diagnostic = v3.v2.metric_pair(meta, scores, optimum)
    assert_nested_close(strict, summary["calibration_strict_fit_thresholds"])
    assert_nested_close(optimum, summary["calibration_F1Opt_thresholds"])
    assert_nested_close(diagnostic, summary["calibration_F1Opt_diagnostic"])
    return {
        "feature_rebuild": feature_audit,
        "model_scores_bit_exact": True,
        "answer_scores_exact_max_over_native_windows": True,
        "strict_fit_threshold_counts_exact": True,
        "F1Opt_diagnostic_thresholds_and_counts_exact": True,
        "strict_window_f1": strict["windows"]["f1"],
        "strict_answer_f1": strict["answers"]["f1"],
        "diagnostic_window_f1": diagnostic["windows"]["f1"],
        "diagnostic_answer_f1": diagnostic["answers"]["f1"],
    }


def audit(out: Path) -> dict:
    fit_done = v3.read_json(out / "fit_complete.json")
    summary = v3.read_json(out / "summary.json")
    complete = v3.read_json(out / "complete.json")
    assert fit_done["status"] == "model_features_C_and_thresholds_frozen_before_calibration"
    assert summary["status"] == "development_calibration_evaluated_once"
    assert complete["status"] == "complete_native634_development_only"
    for name, expected in fit_done["artifacts_sha256"].items():
        path = {
            "runner": Path(v3.__file__),
            "feature_module": Path(v3.feature_v3.__file__),
        }.get(name, out / name)
        assert v3.sha(path) == expected, name
    for name, key in {
        "fit_complete.json": "fit_complete_sha256",
        "summary.json": "summary_sha256",
        "REPORT.md": "report_sha256",
        "calibration_scores.npz": "calibration_scores_sha256",
    }.items():
        assert v3.sha(out / name) == complete[key], name
    assert summary["fit_complete_sha256"] == complete["fit_complete_sha256"]
    for name, expected in summary["artifacts_sha256"].items():
        assert v3.sha(out / name) == expected, name
    snapshot = v3.read_json(out / "source_snapshot.json")
    for path, expected in snapshot["files_sha256"].items():
        assert v3.sha(Path(path)) == expected, path
    for path, expected in snapshot["formal_baseline_files_sha256"].items():
        assert v3.sha(Path(path)) == expected, path
    assert v3.formal_baseline_snapshot() == snapshot["formal_baseline_files_sha256"]
    for path, expected in summary["calibration_source_files_sha256"].items():
        assert v3.sha(Path(path)) == expected, path

    fit_audit = audit_fit(out, fit_done)
    cal_audit = audit_calibration(out, fit_done, summary)
    manifests = {
        partition: validate_evidence_manifest(partition)
        for partition in ("fit", "calibration")
    }
    source = Path(v3.__file__).read_text(encoding="utf-8")
    assert 'assert not (out / "summary.json").exists()' in source
    assert snapshot["calibration_artifacts_opened"] is False
    assert snapshot["calibration_labels_opened"] is False
    assert fit_done["calibration_artifacts_opened"] is False
    assert fit_done["calibration_labels_opened"] is False
    assert fit_done["calibration_used_for_model_feature_C_or_threshold_selection"] is False
    assert summary["calibration_used_for_model_feature_C_or_fit_threshold_selection"] is False
    assert summary["calibration_evaluations"] == complete["calibration_evaluations"] == 1
    assert summary["incumbent_replaced"] is complete["incumbent_replaced"] is False
    for record in (fit_done, summary, complete):
        assert record["GPU_used"] is False
        assert record["formal_baselines_modified"] is False
        assert record["official_test_opened"] is False
    return {
        "status": "passed",
        "fit": fit_audit,
        "calibration": cal_audit,
        "evidence_union_manifests": manifests,
        "freeze_boundary": {
            "fit_complete_hash_bound_into_summary_and_complete": True,
            "calibration_artifacts_opened_before_fit_freeze": False,
            "calibration_used_for_selection": False,
            "single_use_evaluate_guard_present": True,
            "recorded_calibration_evaluations": 1,
        },
        "artifact_hashes_exact": True,
        "source_input_hashes_exact": True,
        "formal_baseline_hashes_unchanged": True,
        "formal_baseline_file_count": len(snapshot["formal_baseline_files_sha256"]),
        "incumbent_replaced": False,
        "GPU_used": False,
        "official_test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=QA_ROOT / "results/semantic_window_v3"
    )
    args = parser.parse_args()
    result = audit(args.output_dir.resolve())
    target = args.output_dir / "INDEPENDENT_AUDIT.json"
    pending = target.with_suffix(target.suffix + ".tmp")
    pending.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.replace(target)
    print("SEMANTIC_WINDOW_V3_AUDIT_PASSED", result["fit"]["selected_candidate"])


if __name__ == "__main__":
    main()
