"""Independent, read-only integrity audit for the frozen semantic-window v2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parents[1]
sys.path.insert(0, str(QA_ROOT / "src"))

import run_semantic_window_v2 as v2  # noqa: E402


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


def meta_arrays(meta: dict) -> dict[str, np.ndarray]:
    return {
        "window_ids": np.asarray([row["window_id"] for row in meta["windows"]]),
        "response_ids": np.asarray([row["response_id"] for row in meta["windows"]]),
        "group_ids": np.asarray([row["group_id"] for row in meta["windows"]]),
        "answer_ids": np.asarray([row["answer_id"] for row in meta["windows"]]),
    }


def assert_feature_identity(path: Path, meta: dict, expected_rows: int) -> dict:
    expected = meta_arrays(meta)
    with np.load(path, allow_pickle=False) as loaded:
        assert loaded["union"].shape == (expected_rows, len(v2.feature_v2.UNION_NAMES))
        assert np.isfinite(loaded["union"]).all()
        assert np.array_equal(loaded["feature_names"], v2.feature_v2.UNION_NAMES)
        for name, values in expected.items():
            assert np.array_equal(loaded[name], values), name
        window_ids = loaded["window_ids"].astype(str)
    assert len(np.unique(window_ids)) == expected_rows
    return {
        "rows": expected_rows,
        "unique_window_ids": expected_rows,
        "union_width": len(v2.feature_v2.UNION_NAMES),
        "identity_exact": True,
    }


def audit_fit(out: Path, fit_done: dict) -> tuple[dict, dict]:
    meta = v2.source_v1.load_partition_meta("fit")
    identity = assert_feature_identity(
        out / "fit_window_features.npz", meta, v2.EXPECTED_WINDOWS["fit"]
    )
    expected_names = [
        f"{variant}__C{c_value:g}"
        for variant in v2.VARIANTS
        for c_value in v2.C_VALUES
    ]
    with np.load(out / "fit_scores.npz", allow_pickle=False) as loaded:
        names = loaded["candidate_names"].astype(str).tolist()
        scores = loaded["candidate_scores"].copy()
        selected_scores = loaded["selected_scores"].copy()
        labels = loaded["labels"].copy()
        selected_index = int(loaded["selected_index"])
        stored_answer_scores = loaded["answer_scores"].copy()
    assert names == expected_names
    assert scores.shape == (len(expected_names), v2.EXPECTED_WINDOWS["fit"])
    assert np.isfinite(scores).all()
    expected_labels = np.asarray(
        [window["label"] for window in meta["windows"]], dtype=np.int8
    )
    assert np.array_equal(labels, expected_labels)

    table = fit_done["candidate_table"]
    assert [row["candidate"] for row in table] == names
    for index, record in enumerate(table):
        thresholds = v2.choose_thresholds(meta, scores[index])
        metrics = v2.metric_pair(meta, scores[index], thresholds)
        assert_nested_close(thresholds, record["fit_OOF_thresholds"])
        assert_nested_close(metrics, record["fit_OOF"])
    recomputed_selected = max(
        range(len(table)),
        key=lambda index: v2.candidate_selection_key(table[index], index),
    )
    assert recomputed_selected == selected_index
    assert table[selected_index]["candidate"] == fit_done["selected"]["candidate"]
    assert np.array_equal(selected_scores, scores[selected_index])
    assert np.array_equal(stored_answer_scores, v2.answer_scores(meta, selected_scores))

    groups = meta_arrays(meta)["group_ids"]
    coverage = np.zeros(len(labels), dtype=np.int8)
    fold_rows = []
    for fold, (train, held) in enumerate(
        GroupKFold(v2.FOLDS).split(np.arange(len(labels)), labels, groups)
    ):
        assert not set(groups[train]) & set(groups[held])
        coverage[held] += 1
        logged = fit_done["crossfit"][fold]
        assert len(train) == logged["train_windows"]
        assert len(held) == logged["held_windows"]
        assert len(set(groups[train])) == logged["train_groups"]
        assert len(set(groups[held])) == logged["held_groups"]
        fold_rows.append(len(held))
    assert np.all(coverage == 1)
    upstream = v2.assert_fit_upstream_source_group_oof(meta)
    assert_nested_close(upstream, fit_done["upstream_OOF_audit"])
    return {
        "identity": identity,
        "candidates_recomputed": len(table),
        "candidate_names_match_preregistration": True,
        "selected_index": selected_index,
        "selected_candidate": names[selected_index],
        "selection_recomputed_from_fit_OOF_only": True,
        "all_fit_thresholds_and_counts_exact": True,
        "group_folds": v2.FOLDS,
        "fold_held_windows": fold_rows,
        "each_fit_window_held_exactly_once": True,
        "all_fold_group_intersections_empty": True,
        "upstream_source_group_OOF_reconstructed": upstream,
    }, meta


def audit_calibration(out: Path, fit_done: dict, summary: dict) -> dict:
    meta = v2.source_v1.load_partition_meta("calibration")
    identity = assert_feature_identity(
        out / "calibration_window_features.npz",
        meta,
        v2.EXPECTED_WINDOWS["calibration"],
    )
    expected = meta_arrays(meta)
    with np.load(out / "calibration_scores.npz", allow_pickle=False) as loaded:
        scores = loaded["window_scores"].copy()
        answer_scores = loaded["answer_scores"].copy()
        assert np.array_equal(loaded["window_ids"], expected["window_ids"])
        assert np.array_equal(loaded["response_ids"], expected["response_ids"])
    assert scores.shape == (v2.EXPECTED_WINDOWS["calibration"],)
    assert np.isfinite(scores).all()
    assert np.array_equal(answer_scores, v2.answer_scores(meta, scores))

    strict = v2.metric_pair(meta, scores, fit_done["selected"]["fit_OOF_thresholds"])
    optimum = v2.choose_thresholds(meta, scores)
    diagnostic = v2.metric_pair(meta, scores, optimum)
    assert_nested_close(strict, summary["calibration_strict_fit_thresholds"])
    assert_nested_close(optimum, summary["calibration_F1Opt_thresholds"])
    assert_nested_close(diagnostic, summary["calibration_F1Opt_diagnostic"])
    return {
        "identity": identity,
        "answer_scores_exact_max_over_native_windows": True,
        "strict_fit_threshold_counts_exact": True,
        "F1Opt_diagnostic_thresholds_and_counts_exact": True,
        "strict_window_f1": strict["windows"]["f1"],
        "strict_answer_f1": strict["answers"]["f1"],
        "diagnostic_window_f1": diagnostic["windows"]["f1"],
        "diagnostic_answer_f1": diagnostic["answers"]["f1"],
    }


def audit(out: Path) -> dict:
    fit_done = v2.read_json(out / "fit_complete.json")
    summary = v2.read_json(out / "summary.json")
    complete = v2.read_json(out / "complete.json")
    assert fit_done["status"] == "model_features_C_and_thresholds_frozen_before_calibration"
    assert summary["status"] == "development_calibration_evaluated_once"
    assert complete["status"] == "complete_native634_development_only"

    for name, expected in fit_done["artifacts_sha256"].items():
        path = {
            "runner": Path(v2.__file__),
            "feature_module": Path(v2.feature_v2.__file__),
        }.get(name, out / name)
        assert v2.sha(path) == expected, name
    complete_hashes = {
        "fit_complete.json": "fit_complete_sha256",
        "summary.json": "summary_sha256",
        "REPORT.md": "report_sha256",
        "calibration_scores.npz": "calibration_scores_sha256",
    }
    for name, key in complete_hashes.items():
        assert v2.sha(out / name) == complete[key], name
    assert summary["fit_complete_sha256"] == complete["fit_complete_sha256"]
    for name, expected in summary["artifacts_sha256"].items():
        assert v2.sha(out / name) == expected, name

    snapshot = v2.read_json(out / "source_snapshot.json")
    for path, expected in snapshot["files_sha256"].items():
        assert v2.sha(Path(path)) == expected, path
    for path, expected in snapshot["formal_baseline_files_sha256"].items():
        assert v2.sha(Path(path)) == expected, path
    assert v2.baseline_snapshot() == snapshot["formal_baseline_files_sha256"]

    fit_audit, _ = audit_fit(out, fit_done)
    cal_audit = audit_calibration(out, fit_done, summary)
    source = Path(v2.__file__).read_text(encoding="utf-8")
    assert 'assert not (out / "summary.json").exists()' in source
    assert fit_done["calibration_labels_opened"] is False
    assert fit_done["calibration_used_for_model_feature_C_or_threshold_selection"] is False
    assert summary["calibration_used_for_model_feature_C_or_fit_threshold_selection"] is False
    assert summary["calibration_evaluations"] == complete["calibration_evaluations"] == 1
    assert summary["formal_baseline_files_unchanged"] is True
    for record in (fit_done, summary, complete):
        assert record["GPU_used"] is False
        assert record["formal_baselines_modified"] is False
        assert record["official_test_opened"] is False

    return {
        "status": "passed",
        "fit": fit_audit,
        "calibration": cal_audit,
        "freeze_boundary": {
            "fit_complete_hash_bound_into_summary_and_complete": True,
            "calibration_used_for_selection": False,
            "single_use_evaluate_guard_present": True,
            "recorded_calibration_evaluations": 1,
        },
        "artifact_hashes_exact": True,
        "source_inputs_hashes_exact": True,
        "formal_baseline_hashes_unchanged": True,
        "formal_baseline_file_count": len(snapshot["formal_baseline_files_sha256"]),
        "GPU_used": False,
        "official_test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=QA_ROOT / "results/semantic_window_v2"
    )
    args = parser.parse_args()
    result = audit(args.output_dir.resolve())
    target = args.output_dir / "INDEPENDENT_AUDIT.json"
    pending = target.with_suffix(target.suffix + ".tmp")
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)
    print("SEMANTIC_WINDOW_V2_AUDIT_PASSED", result["fit"]["selected_candidate"])


if __name__ == "__main__":
    main()
