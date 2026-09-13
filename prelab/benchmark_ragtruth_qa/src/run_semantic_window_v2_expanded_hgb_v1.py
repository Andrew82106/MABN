"""Preregistered nonlinear readout over the audited expanded window-v2 design.

This is our method, not a modification of any formal baseline.  ``preregister``
freezes three HistGradientBoosting recipes before model fitting.  ``fit`` uses
only the 3,680-answer fit split and source-connected GroupKFold OOF scores for
model and threshold selection.  ``evaluate`` is the sole calibration path and
can run once, after the selected full-fit model and both thresholds are frozen.
There is deliberately no official-test loader or command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402
import run_semantic_source_attribution_combined_fit_v1 as combined  # noqa: E402
import run_semantic_window_v2 as native_v2  # noqa: E402
import run_semantic_window_v2_expanded_fit_v1 as linear  # noqa: E402
import semantic_window_v2_features as feature_v2  # noqa: E402


VERSION = "semantic-window-v2-expanded-hgb-v1"
OUT = ROOT / "results/semantic_window_v2_expanded_hgb_v1"
LINEAR_OUT = linear.OUT
VERIFY_RUNNER = HERE / "verify_semantic_window_v2_expanded_hgb_v1.py"

FIT_ANSWERS = linear.FIT_ANSWERS
FIT_GROUPS = linear.FIT_GROUPS
FIT_WINDOWS = linear.FIT_WINDOWS
FIT_POSITIVE_WINDOWS = linear.FIT_POSITIVE_WINDOWS
CAL_ANSWERS = linear.CAL_ANSWERS
CAL_WINDOWS = linear.CAL_WINDOWS
NATIVE_ANSWERS = linear.NATIVE_ANSWERS
NATIVE_WINDOWS = linear.NATIVE_WINDOWS
FOLDS = linear.DOWNSTREAM_FOLDS
SEED = linear.SEED
THREADS = linear.THREADS
FEATURES = LINEAR_OUT / "fit_whitebox_geometry.npy"
VARIANT = "whitebox_geometry"
LEAF_NODES = (7, 15, 31)
COMMON_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "min_samples_leaf": 100,
    "l2_regularization": 2.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": SEED,
}

INCUMBENT_PATH = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"
LOOKBACK_PATH = ROOT / "results/lookback_k4_evaluation_adapter_v1/summary.json"
LINEAR_SUMMARY = LINEAR_OUT / "summary.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read(path) == value, f"Frozen file changed: {path}"
    else:
        atomic_json(path, value)


def atomic_pickle(path: Path, value) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(pickle.dumps(value, protocol=5))
    pending.replace(path)


def atomic_npz(path: Path, **values) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **values)
    pending.replace(path)


def candidate_name(max_leaf_nodes: int) -> str:
    return f"whitebox_geometry__hgb_leaf{max_leaf_nodes}"


def model_params(max_leaf_nodes: int) -> dict:
    return {**COMMON_PARAMS, "max_leaf_nodes": int(max_leaf_nodes)}


def make_model(max_leaf_nodes: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**model_params(max_leaf_nodes))


def protocol() -> dict:
    recipes = []
    for leaf in LEAF_NODES:
        params = model_params(leaf)
        recipes.append({
            "candidate": candidate_name(leaf),
            "parameters": params,
            "effective_parameters": make_model(leaf).get_params(),
        })
    return {
        "version": VERSION,
        "identity": "Our nonlinear semantic-window readout; it does not modify or relabel any baseline.",
        "preregistration": {
            "candidate_count": len(recipes),
            "candidates_in_fixed_order": recipes,
            "only_varied_parameter": "max_leaf_nodes",
            "parameters_frozen_before_any_HGB_fit": True,
            "no_additional_recipe_after_fit": True,
        },
        "inputs": {
            "feature_family": VARIANT,
            "blocks": ["whitebox", "geometry"],
            "width": 26,
            "source": "Byte-identical audited fit_whitebox_geometry.npy from semantic_window_v2_expanded_fit_v1.",
            "windows": "The same 653,979 eligible stride-one four-original-BPE windows and unchanged any-error labels.",
            "weights": "The exact semantic_window_v2 hierarchy_weights rule: equal group/answer/window base mass, followed by fit-only binary class balance.",
            "preprocessing": "No scaling. HistGradientBoosting learns bins from each fold's fit rows only.",
            "answer_score": "Maximum eligible-window probability per answer, unchanged.",
        },
        "fit": {
            "answers": FIT_ANSWERS,
            "source_connected_groups": FIT_GROUPS,
            "windows": FIT_WINDOWS,
            "positive_windows": FIT_POSITIVE_WINDOWS,
            "crossfit": "Five-fold sklearn GroupKFold over the same locked source-connected group IDs as the expanded LR run.",
            "internal_validation": "Disabled: early_stopping=False.",
            "thresholds": "For each candidate, window and answer thresholds independently maximize pooled all-domain fit OOF F1, then precision, then higher cutoff.",
            "selection": "All-domain fit OOF only: maximize min(window F1, answer F1), then window F1, answer F1, window AP, answer AP, fewer leaves, earlier preregistered recipe.",
            "full_fit": "Refit only the selected recipe once on all fit windows with the same hierarchy loss weights; freeze model and both OOF thresholds before calibration.",
        },
        "native_diagnostic": {
            "scope": "The first 634 native Llama2-7B answers and first 168,123 windows, using their already-held all-domain OOF predictions.",
            "reported_for_every_candidate": [
                "metrics at each candidate's frozen all-domain OOF thresholds",
                "native-subset F1-opt metrics and thresholds",
            ],
            "selection_or_threshold_effect": "None. Native diagnostics cannot alter candidate selection, full fit, or frozen all-domain thresholds.",
        },
        "calibration": {
            "answers": CAL_ANSWERS,
            "windows": CAL_WINDOWS,
            "rule": "Single reporting-only evaluation of the selected full-fit model at its frozen all-domain fit-OOF thresholds; no calibration threshold fitting.",
        },
        "comparison": {
            "expanded_LR": "Same design, weights, folds, windows and answer max; strict calibration at LR's frozen fit-OOF thresholds.",
            "incumbent": "Historical calibration-selected semantic_claim tree plus large score; calibration F1-opt, so threshold provenance is less strict.",
            "Lookback": "Frozen baseline adapted to project 4-BPE windows; calibration F1-opt, so threshold provenance is less strict.",
        },
        "limitations": [
            "The upstream Lookback/large fit columns are frozen three-fold source-group OOF predictions, not retrained inside each downstream five-fold split.",
            "The 3,680 fit answers cover only 615 source-connected groups and mix native Llama2-7B with five auxiliary generators.",
            "Calibration has been reused in project development and is not fresh final-test evidence.",
        ],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scikit_learn_distribution": importlib.metadata.version("scikit-learn"),
        },
        "CPU_threads": THREADS,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def comparison_paths() -> tuple[Path, ...]:
    return (
        LINEAR_OUT / "protocol.json",
        LINEAR_OUT / "preparation_complete.json",
        LINEAR_OUT / "PROVENANCE_AUDIT.json",
        LINEAR_OUT / "INDEPENDENT_AUDIT.json",
        LINEAR_OUT / "fit_complete.json",
        LINEAR_OUT / "fit_scores.npz",
        LINEAR_OUT / "model.pkl",
        LINEAR_SUMMARY,
        FEATURES,
        INCUMBENT_PATH,
        LOOKBACK_PATH,
        combined.OUT / "projection.npz",
        combined.OUT / "axes.json",
        Path(linear.__file__),
        Path(native_v2.__file__),
        HERE / "semantic_window_v2_features.py",
        Path(__file__),
        VERIFY_RUNNER,
    )


def current_source_snapshot() -> dict:
    paths = comparison_paths()
    assert all(path.is_file() for path in paths)
    return {
        "files_sha256": {str(path.resolve()): sha(path) for path in paths},
        "formal_baseline_files_sha256": linear.baseline_snapshot(),
        "expanded_LR_design_sha256": sha(FEATURES),
        "expanded_LR_summary_sha256": sha(LINEAR_SUMMARY),
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def assert_sources_unchanged() -> dict:
    expected = read(OUT / "source_snapshot.json")
    assert expected == current_source_snapshot()
    return expected


def preregister() -> None:
    assert not OUT.exists() or not any(OUT.iterdir()), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / "protocol.json", protocol())
    frozen_json(OUT / "source_snapshot.json", current_source_snapshot())
    receipt = {
        "status": "three_HGB_recipes_frozen_before_fit",
        "written_at_unix_ns": time.time_ns(),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "runner_sha256": sha(Path(__file__)),
        "candidate_names": [candidate_name(leaf) for leaf in LEAF_NODES],
        "fit_started": False,
        "calibration_opened": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / "PREREGISTRATION.json", receipt)
    print("EXPANDED_HGB_PREREGISTERED", receipt["protocol_sha256"], flush=True)


def check_preregistered() -> dict:
    assert read(OUT / "protocol.json") == protocol()
    receipt = read(OUT / "PREREGISTRATION.json")
    assert receipt["status"] == "three_HGB_recipes_frozen_before_fit"
    assert receipt["protocol_sha256"] == sha(OUT / "protocol.json")
    assert receipt["source_snapshot_sha256"] == sha(OUT / "source_snapshot.json")
    assert receipt["runner_sha256"] == sha(Path(__file__))
    assert receipt["candidate_names"] == [candidate_name(leaf) for leaf in LEAF_NODES]
    assert_sources_unchanged()
    return receipt


def verify_upstream_and_design() -> tuple[dict, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    axes, projection = combined.load_projection()
    assert len(axes["response_ids"]) == FIT_ANSWERS
    assert len(set(axes["response_group_ids"])) == FIT_GROUPS
    assert projection["window_labels"].shape == (FIT_WINDOWS,)
    assert int(projection["window_labels"].sum()) == FIT_POSITIVE_WINDOWS
    assert int(projection["response_window_indptr"][NATIVE_ANSWERS]) == NATIVE_WINDOWS

    _, direct = linear.audit_upstream(axes, projection)
    saved = read(LINEAR_OUT / "PROVENANCE_AUDIT.json")
    for key in (
        "status", "source_groups", "answer_coverage_exactly_once",
        "lookback_models_fit_only_on_complement_groups",
        "large_models_fit_only_on_complement_groups",
        "generation_nll_label_blind", "in_sample_supervised_columns",
        "calibration_opened", "official_test_opened",
    ):
        assert direct[key] == saved[key]
    assert saved["status"] == "passed_all_supervised_whitebox_inputs_are_source_group_OOF"
    assert saved["in_sample_supervised_columns"] == 0
    linear_prep = read(LINEAR_OUT / "preparation_complete.json")
    linear_fit = read(LINEAR_OUT / "fit_complete.json")
    assert linear_prep["native_feature_replay"]["native_whitebox_geometry_exact"] is True
    assert linear_fit["artifacts_sha256"]["fit_whitebox_geometry.npy"] == sha(FEATURES)

    x = np.load(FEATURES, mmap_mode="r")
    labels = projection["window_labels"].astype(np.int8)
    groups = combined.group_indices("window")
    answers = projection["window_response_index"].astype(np.int32)
    assert x.shape == (FIT_WINDOWS, 26) and np.isfinite(x).all()
    assert groups.shape == labels.shape == answers.shape
    return axes, projection, labels, groups, answers


def answer_scores(scores: np.ndarray, response_window_indptr: np.ndarray) -> np.ndarray:
    return linear.answer_scores(scores, response_window_indptr)


def thresholds_and_metrics(
    window_labels: np.ndarray,
    window_scores: np.ndarray,
    answer_labels: np.ndarray,
    pooled_answer_scores: np.ndarray,
) -> tuple[dict, dict]:
    thresholds = {
        "window": q.choose_threshold(window_labels, window_scores),
        "answer": q.choose_threshold(answer_labels, pooled_answer_scores),
    }
    metrics = {
        "windows": q.count(window_labels, window_scores, thresholds["window"]["threshold"]),
        "answers": q.count(answer_labels, pooled_answer_scores, thresholds["answer"]["threshold"]),
    }
    return thresholds, metrics


def metrics_at_thresholds(
    window_labels: np.ndarray,
    window_scores: np.ndarray,
    answer_labels: np.ndarray,
    pooled_answer_scores: np.ndarray,
    thresholds: dict,
) -> dict:
    return {
        "windows": q.count(window_labels, window_scores, thresholds["window"]["threshold"]),
        "answers": q.count(answer_labels, pooled_answer_scores, thresholds["answer"]["threshold"]),
    }


def selection_key(record: dict, prereg_index: int) -> tuple:
    windows = record["fit_OOF"]["windows"]
    answers = record["fit_OOF"]["answers"]
    return (
        min(windows["f1"], answers["f1"]),
        windows["f1"],
        answers["f1"],
        windows["average_precision"],
        answers["average_precision"],
        -record["max_leaf_nodes"],
        -prereg_index,
    )


def fit() -> None:
    receipt = check_preregistered()
    assert not (OUT / "fit_complete.json").exists()
    assert not (OUT / "summary.json").exists()
    started_ns = time.time_ns()
    frozen_json(OUT / "fit_started.json", {
        "status": "fit_started_after_preregistration",
        "started_at_unix_ns": started_ns,
        "preregistered_at_unix_ns": receipt["written_at_unix_ns"],
        "protocol_sha256": sha(OUT / "protocol.json"),
        "preregistration_sha256": sha(OUT / "PREREGISTRATION.json"),
        "candidate_names": receipt["candidate_names"],
        "calibration_opened": False,
        "official_test_opened": False,
    })
    assert receipt["written_at_unix_ns"] < started_ns
    started = time.perf_counter()

    _, projection, labels, groups, answers = verify_upstream_and_design()
    x = np.load(FEATURES, mmap_mode="r")
    response_labels = projection["response_labels"].astype(np.int8)
    response_ptr = projection["response_window_indptr"]
    assert response_labels.shape == (FIT_ANSWERS,)

    splits = list(GroupKFold(FOLDS).split(np.arange(FIT_WINDOWS), labels, groups))
    oof = np.full((len(LEAF_NODES), FIT_WINDOWS), np.nan, dtype=np.float64)
    fold_log = []
    with threadpool_limits(limits=THREADS):
        for fold, (train, held) in enumerate(splits):
            combined.assert_source_connected_disjoint(train, held, "window")
            base, loss, class_mass = linear.hierarchy_weights(groups, answers, labels, train)
            x_train = np.asarray(x[train], dtype=np.float32, order="C")
            x_held = np.asarray(x[held], dtype=np.float32, order="C")
            one_fold = {
                "fold": fold,
                "train_windows": len(train), "held_windows": len(held),
                "train_answers": len(np.unique(answers[train])),
                "held_answers": len(np.unique(answers[held])),
                "train_groups": len(np.unique(groups[train])),
                "held_groups": len(np.unique(groups[held])),
                "group_overlap": 0,
                "train_positive": int(labels[train].sum()),
                "held_positive": int(labels[held].sum()),
                "final_loss_mass_by_class": class_mass.tolist(),
                "models": [],
            }
            for candidate_index, leaf in enumerate(LEAF_NODES):
                tick = time.perf_counter()
                model = make_model(leaf)
                model.fit(x_train, labels[train], sample_weight=loss[train])
                oof[candidate_index, held] = model.predict_proba(x_held)[:, 1]
                assert model.n_iter_ == COMMON_PARAMS["max_iter"]
                assert not model.do_early_stopping_
                one_fold["models"].append({
                    "candidate": candidate_name(leaf),
                    "max_leaf_nodes": leaf,
                    "n_iter": int(model.n_iter_),
                    "seconds": time.perf_counter() - tick,
                })
                print("EXPANDED_HGB_FOLD", fold + 1, FOLDS, "LEAF", leaf, flush=True)
            fold_log.append(one_fold)
            del x_train, x_held, base, loss
    assert np.isfinite(oof).all()

    table = []
    all_answer_scores = []
    for candidate_index, leaf in enumerate(LEAF_NODES):
        scores = oof[candidate_index]
        pooled = answer_scores(scores, response_ptr)
        all_answer_scores.append(pooled)
        thresholds, metrics = thresholds_and_metrics(labels, scores, response_labels, pooled)

        native_thresholds, native_opt = thresholds_and_metrics(
            labels[:NATIVE_WINDOWS], scores[:NATIVE_WINDOWS],
            response_labels[:NATIVE_ANSWERS], pooled[:NATIVE_ANSWERS],
        )
        native_at_global = metrics_at_thresholds(
            labels[:NATIVE_WINDOWS], scores[:NATIVE_WINDOWS],
            response_labels[:NATIVE_ANSWERS], pooled[:NATIVE_ANSWERS], thresholds,
        )
        record = {
            "candidate": candidate_name(leaf),
            "max_leaf_nodes": leaf,
            "parameters": model_params(leaf),
            "fit_OOF_thresholds": thresholds,
            "fit_OOF": metrics,
            "native_held_OOF_diagnostic": {
                "selection_effect": "none",
                "at_all_domain_thresholds": native_at_global,
                "native_subset_F1Opt_thresholds": native_thresholds,
                "native_subset_F1Opt": native_opt,
            },
        }
        record["selection_key"] = list(selection_key(record, candidate_index))
        table.append(record)

    selected_index = max(
        range(len(table)), key=lambda index: selection_key(table[index], index)
    )
    selected = table[selected_index]
    active = np.arange(FIT_WINDOWS)
    base, loss, class_mass = linear.hierarchy_weights(groups, answers, labels, active)
    full_model = make_model(selected["max_leaf_nodes"])
    tick = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        full_model.fit(x, labels, sample_weight=loss)
    full_seconds = time.perf_counter() - tick
    assert full_model.n_iter_ == COMMON_PARAMS["max_iter"]
    assert not full_model.do_early_stopping_
    payload = {
        "version": VERSION,
        "candidate": selected["candidate"],
        "variant": VARIANT,
        "feature_names": list(feature_v2.WHITEBOX_NAMES + feature_v2.GEOMETRY_NAMES),
        "parameters": selected["parameters"],
        "model": full_model,
    }
    atomic_pickle(OUT / "model.pkl", payload)
    atomic_npz(
        OUT / "fit_scores.npz",
        candidate_names=np.asarray([row["candidate"] for row in table]),
        candidate_window_scores=oof,
        candidate_answer_scores=np.asarray(all_answer_scores),
        selected_index=np.asarray(selected_index, dtype=np.int32),
        window_labels=labels,
        answer_labels=response_labels,
    )
    source = assert_sources_unchanged()
    provenance = {
        "status": "passed_independent_revalidation_of_expanded_OOF_inputs",
        "source_audit_status": read(LINEAR_OUT / "PROVENANCE_AUDIT.json")["status"],
        "upstream_supervised_columns_source_group_OOF": True,
        "in_sample_supervised_columns": 0,
        "feature_matrix_byte_identical_to_audited_expanded_LR": True,
        "feature_matrix_sha256": sha(FEATURES),
        "native_168123_feature_rows_previously_replayed_exact": True,
        "downstream_five_fold_group_disjoint": True,
        "downstream_five_fold_window_coverage_exactly_once": True,
        "calibration_opened": False,
        "official_test_opened": False,
    }
    atomic_json(OUT / "PROVENANCE_AUDIT.json", provenance)
    fit_complete = {
        "status": "all_domain_fit_OOF_selection_and_full_model_frozen_before_calibration",
        "selected": selected,
        "selected_index": selected_index,
        "selection_rule": protocol()["fit"]["selection"],
        "candidates": table,
        "crossfit": fold_log,
        "full_fit": {
            "windows": FIT_WINDOWS,
            "positive_windows": FIT_POSITIVE_WINDOWS,
            "answers": FIT_ANSWERS,
            "groups": FIT_GROUPS,
            "final_loss_mass_by_class": class_mass.tolist(),
            "n_iter": int(full_model.n_iter_),
            "seconds": full_seconds,
        },
        "native_diagnostics_used_for_selection": False,
        "upstream_provenance": provenance,
        "preregistration_precedes_fit": True,
        "preregistered_at_unix_ns": receipt["written_at_unix_ns"],
        "fit_started_at_unix_ns": started_ns,
        "artifacts_sha256": {
            "model.pkl": sha(OUT / "model.pkl"),
            "fit_scores.npz": sha(OUT / "fit_scores.npz"),
            "PROVENANCE_AUDIT.json": sha(OUT / "PROVENANCE_AUDIT.json"),
            "fit_started.json": sha(OUT / "fit_started.json"),
            "PREREGISTRATION.json": sha(OUT / "PREREGISTRATION.json"),
            "protocol.json": sha(OUT / "protocol.json"),
            "source_snapshot.json": sha(OUT / "source_snapshot.json"),
            "runner": sha(Path(__file__)),
        },
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "formal_baseline_files_sha256": source["formal_baseline_files_sha256"],
        "fit_labels_opened": True,
        "calibration_labels_opened": False,
        "calibration_used_for_recipe_model_or_threshold_selection": False,
        "official_test_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "model_fits": len(LEAF_NODES) * FOLDS + 1,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "fit_complete.json", fit_complete)
    print(
        "EXPANDED_HGB_FIT_FROZEN", selected["candidate"],
        selected["fit_OOF"]["windows"]["f1"],
        selected["fit_OOF"]["answers"]["f1"], flush=True,
    )


def reference_rows() -> list[dict]:
    lr = read(LINEAR_SUMMARY)
    incumbent = read(INCUMBENT_PATH)
    lookback = read(LOOKBACK_PATH)
    return [
        {
            "name": "expanded whitebox+geometry LR",
            "role": "our linear comparator",
            "threshold_provenance": "frozen expanded-fit OOF",
            "windows": lr["calibration_at_frozen_fit_thresholds"]["windows"],
            "answers": lr["calibration_at_frozen_fit_thresholds"]["answers"],
        },
        {
            "name": incumbent["candidate"],
            "role": "historical incumbent",
            "threshold_provenance": "calibration F1Opt",
            "windows": incumbent["metrics"]["calibration"]["windows"],
            "answers": incumbent["metrics"]["calibration"]["answers"],
        },
        {
            "name": "Lookback adapted project-4 window",
            "role": "frozen baseline",
            "threshold_provenance": "calibration F1Opt",
            "windows": lookback["adapted_project4_calibration"]["windows4"],
            "answers": lookback["adapted_project4_calibration"]["answers"],
        },
    ]


def evaluate() -> None:
    check_preregistered()
    fit_done = read(OUT / "fit_complete.json")
    assert fit_done["status"] == "all_domain_fit_OOF_selection_and_full_model_frozen_before_calibration"
    assert not (OUT / "summary.json").exists(), "Calibration evaluation is single-use"
    for name, expected in fit_done["artifacts_sha256"].items():
        path = Path(__file__) if name == "runner" else OUT / name
        assert sha(path) == expected
    started = time.perf_counter()

    # This is the only point where calibration rows and labels are opened.
    union, _, meta, whitebox_audit = native_v2.build_partition_design("calibration")
    columns = feature_v2.variant_indices(VARIANT)
    x = union[:, columns]
    assert x.shape == (CAL_WINDOWS, 26) and np.isfinite(x).all()
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    selected = fit_done["selected"]
    assert payload["version"] == VERSION and payload["candidate"] == selected["candidate"]
    assert payload["parameters"] == selected["parameters"]
    scores = payload["model"].predict_proba(x)[:, 1]
    assert scores.shape == (CAL_WINDOWS,) and np.isfinite(scores).all()
    pooled = native_v2.answer_scores(meta, scores)
    thresholds = selected["fit_OOF_thresholds"]
    strict = native_v2.metric_pair(meta, scores, thresholds)
    atomic_npz(
        OUT / "calibration_scores.npz",
        window_scores=scores,
        answer_scores=pooled,
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["windows"]]),
    )
    references = reference_rows()
    assert_sources_unchanged()
    summary = {
        "status": "single_reporting_only_calibration_evaluation_complete",
        "method_identity": "our nonlinear semantic-window readout",
        "selected_candidate": selected["candidate"],
        "selected_parameters": selected["parameters"],
        "fit_OOF": selected["fit_OOF"],
        "fit_OOF_thresholds": thresholds,
        "calibration_at_frozen_fit_OOF_thresholds": strict,
        "candidate_fit_OOF_table": fit_done["candidates"],
        "native_held_OOF_diagnostics": [
            {
                "candidate": row["candidate"],
                **row["native_held_OOF_diagnostic"],
            }
            for row in fit_done["candidates"]
        ],
        "calibration_comparisons": references,
        "comparison_caveat": "HGB and expanded LR use frozen fit-OOF thresholds; incumbent and Lookback numbers use calibration-F1Opt thresholds and are descriptive, not protocol-matched superiority tests.",
        "calibration_thresholds_fitted": 0,
        "calibration_evaluations": 1,
        "calibration_used_for_recipe_model_or_threshold_selection": False,
        "calibration_whitebox_reconstruction_audit": whitebox_audit,
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "artifacts_sha256": {"calibration_scores.npz": sha(OUT / "calibration_scores.npz")},
        "formal_baseline_files_unchanged": True,
        "calibration_labels_opened": True,
        "official_test_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "summary.json", summary)

    candidate_lines = []
    for row in fit_done["candidates"]:
        full = row["fit_OOF"]
        native = row["native_held_OOF_diagnostic"]["native_subset_F1Opt"]
        candidate_lines.append(
            f"| {row['candidate']} | {full['windows']['f1']:.6f} | {full['answers']['f1']:.6f} | "
            f"{native['windows']['f1']:.6f} | {native['answers']['f1']:.6f} |"
        )
    comparison_lines = [
        f"| selected HGB | fit-OOF frozen | {strict['windows']['f1']:.6f} | {strict['answers']['f1']:.6f} |"
    ]
    for row in references:
        comparison_lines.append(
            f"| {row['name']} | {row['threshold_provenance']} | "
            f"{row['windows']['f1']:.6f} | {row['answers']['f1']:.6f} |"
        )
    report = (
        "# Expanded semantic window HGB v1\n\n"
        "这是我们的非线性读出：复用已审计的 26 维 `whitebox+geometry`、同一 653,979 个窗口、同一层级权重和 answer-max 聚合；没有改动 baseline。\n\n"
        "三档 HGB 在拟合前已预注册，仅用 3,680 答案/615 个 source-connected groups 的五折 OOF 选择。选择规则先最大化 window/answer F1 的较小值，再比较 window F1、answer F1、两级 AP 与模型复杂度。\n\n"
        "| candidate | all-domain OOF window F1 | all-domain OOF answer F1 | native held-OOF window F1Opt | native held-OOF answer F1Opt |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(candidate_lines) + "\n\n"
        f"选中 `{selected['candidate']}`。全量拟合与两个 OOF 阈值冻结后，cal159 只评估一次。\n\n"
        "| calibration comparison | threshold provenance | window F1 | answer F1 |\n"
        "|---|---|---:|---:|\n" + "\n".join(comparison_lines) + "\n\n"
        "HGB/LR 使用 fit-OOF 冻结阈值；incumbent 与 Lookback 是 calibration-F1Opt，阈值来源不匹配，只作描述性参照。native held-OOF 是诊断，不参与本轮选择或阈值冻结。官方 test 未读取。\n"
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("EXPANDED_HGB_CALIBRATION", strict["windows"]["f1"], strict["answers"]["f1"], flush=True)


def status() -> None:
    value = {
        "preregistered": (OUT / "PREREGISTRATION.json").exists(),
        "fit_frozen": (OUT / "fit_complete.json").exists(),
        "calibration_evaluated": (OUT / "summary.json").exists(),
        "official_test_opened": False,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "fit", "evaluate", "status"))
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        {"preregister": preregister, "fit": fit, "evaluate": evaluate, "status": status}[args.command]()


if __name__ == "__main__":
    main()
