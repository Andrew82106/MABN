"""Target-domain weighting for the audited expanded semantic-window-v2 design.

The only searched quantity is the preregistered total auxiliary training mass
alpha in {0, .25, .5, 1}.  Model and threshold selection use native
llama-2-7b-chat OOF predictions only.  Calibration is available solely through
the separate, single-use ``evaluate`` command.  There is no test-data path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_development as q  # noqa: E402
import run_semantic_source_attribution_combined_fit_v1 as combined  # noqa: E402
import run_semantic_window_v2 as native_v2  # noqa: E402
import run_semantic_window_v2_expanded_fit_v1 as expanded_v2  # noqa: E402
import semantic_window_v2_features as feature_v2  # noqa: E402


VERSION = "semantic-window-v2-target-domain-weighted-v1"
OUT = ROOT / "results/semantic_window_v2_target_domain_weighted_v1"
UPSTREAM = ROOT / "results/semantic_window_v2_expanded_fit_v1"
DESIGN = UPSTREAM / "fit_whitebox_geometry.npy"
EXPANDED_FIT = ROOT / "fit_expansion/data/fit.jsonl"
EXPANSION_PROVENANCE = ROOT / "fit_expansion/data/answer_provenance.jsonl"
EXPANSION_FREEZE = ROOT / "fit_expansion/data/export_freeze.json"
DEVELOPMENT_MANIFEST = ROOT / "data/development_manifest.json"
GOLD_MANIFEST = ROOT / "data/gold_manifest.json"
DEVELOPMENT_EXPORTER = ROOT / "src/prepare_development.py"
NATIVE_FIT = ROOT / "results/semantic_window_v2/fit_complete.json"
NATIVE_SUMMARY = ROOT / "results/semantic_window_v2/summary.json"
NATIVE_SCORES = ROOT / "results/semantic_window_v2/fit_scores.npz"
EXPANDED_SUMMARY = UPSTREAM / "summary.json"
INCUMBENT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"

TARGET_GENERATOR = "llama-2-7b-chat"
AUXILIARY_ALPHAS = (0.0, 0.25, 0.5, 1.0)
FIT_ANSWERS = 3680
NATIVE_ANSWERS = 634
AUXILIARY_ANSWERS = 3046
FIT_WINDOWS = 653979
NATIVE_WINDOWS = 168123
CAL_ANSWERS = 159
CAL_WINDOWS = 42241
FIT_GROUPS = 615
FOLDS = 5
C_VALUE = 0.1
FEATURE_WIDTH = 26
THREADS = 4
SEED = 20260913
SKLEARN_VERSION = "1.6.1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, f"Frozen file changed: {path}"
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


def protocol() -> dict:
    return {
        "version": VERSION,
        "preregistered_candidates": {
            "auxiliary_total_mass_relative_to_native": list(AUXILIARY_ALPHAS),
            "candidate_count": len(AUXILIARY_ALPHAS),
            "target_generator": TARGET_GENERATOR,
            "native_total_mass": 1.0,
            "each_of_five_auxiliary_generator_masses": "alpha / 5",
            "C": C_VALUE,
            "feature_variant": "whitebox_geometry",
            "feature_width": FEATURE_WIDTH,
        },
        "provenance_gate": {
            "must_pass_before_fit": True,
            "evidence": "Hash-bound development_manifest plus its exact exporter code; the exporter selects and asserts llama-2-7b-chat for every fit/calibration row.",
            "on_failure": "Stop before any fit; do not infer calibration generator identity.",
        },
        "folds": {
            "count": FOLDS,
            "definition": "The exact native-634 sklearn GroupKFold source-connected splits used by semantic_window_v2.",
            "training": "For each fold, any generator is eligible only when its source-connected group is outside the held native groups.",
            "selection_and_thresholds": "Only held native-634 target-generator windows and answers.",
            "all_domain_OOF": "Saved and reported only as a diagnostic at target-native thresholds; never used for selection.",
        },
        "training_weights": {
            "hierarchy_within_each_generator": "Equal source-connected group mass; equal answer mass within source group; equal eligible-window mass within answer.",
            "class_balance": "Binary window classes balanced separately inside every generator after hierarchy weighting.",
            "domain_mass": "Native generator mass 1; all auxiliaries together alpha, split equally across the five auxiliary generators.",
            "global_normalization": "Every candidate/fold has total sample-weight mass equal to its native training-window count, preserving C=.1 effective regularization across alpha.",
            "scaler": "Uses generator-domain hierarchy weights before class balance.",
            "logistic_loss": "Uses generator-domain hierarchy weights after per-generator binary class balance.",
            "alpha_zero": "Every auxiliary row has exact zero weight; native OOF scores must replay semantic_window_v2 whitebox_geometry C=.1 within one float64 ULP, with identical threshold decisions and confusion counts.",
        },
        "model": "StandardScaler plus liblinear L2 LogisticRegression; fixed C=.1 and seed.",
        "selection_rule": "Maximize native OOF min(window F1, answer F1), then window F1, answer F1, window AP, answer AP, then prefer smaller alpha.",
        "answer_score": "Maximum eligible-window probability per answer.",
        "inference_identity_boundary": {
            "feature_names": list(feature_v2.WHITEBOX_NAMES + feature_v2.GEOMETRY_NAMES),
            "generator_identity_feature_count": 0,
            "generator_identity_use": "Training-weight construction and reporting only.",
            "prediction_inputs": "Exactly 26 numeric whitebox+geometry columns; no generator/source/group/answer identifier.",
        },
        "calibration": "One evaluation after selected alpha, full-fit model, and native-OOF thresholds are frozen; cal-F1Opt is reported only as a post-hoc diagnostic.",
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "GPU_used": False,
    }


def expected_frozen_hash(mapping: dict, path: Path) -> str:
    resolved = str(path.resolve())
    for name, value in mapping.items():
        if str(Path(name).resolve()) == resolved:
            return value
    raise AssertionError(f"Missing frozen hash for {path}")


def provenance_gate() -> tuple[dict, list[dict], np.ndarray, list[str]]:
    """Prove target identity without opening calibration answer/span rows."""
    development = read_json(DEVELOPMENT_MANIFEST)
    gold = read_json(GOLD_MANIFEST)
    assert development["native_response_model"] == TARGET_GENERATOR
    assert development["eligible_answers_by_partition"] == {
        "fit": NATIVE_ANSWERS, "calibration": CAL_ANSWERS,
    }
    assert development["code_sha256"] == sha(DEVELOPMENT_EXPORTER)
    assert gold["signature"]["files_sha256"]["development_manifest"] == sha(DEVELOPMENT_MANIFEST)
    assert gold["counts"]["fit"]["answers"] == NATIVE_ANSWERS
    assert gold["counts"]["calibration"]["answers"] == CAL_ANSWERS
    assert development["development_exporter_parsed_official_test_rows"] is False
    assert development["official_test_response_text_or_labels_inspected_or_exported"] is False

    freeze = read_json(EXPANSION_FREEZE)
    expected = expected_frozen_hash(freeze["output_files_sha256"], EXPANDED_FIT)
    assert sha(EXPANDED_FIT) == expected
    assert freeze["new_answers"] == AUXILIARY_ANSWERS
    assert freeze["retained_answers"] == NATIVE_ANSWERS
    assert freeze["combined_answers"] == FIT_ANSWERS
    assert freeze["calibration_unchanged_and_not_parsed"] is True
    assert freeze["test_not_opened"] is True
    assert freeze["generator_identity_metadata_only"] is True

    rows = read_jsonl(EXPANDED_FIT)
    assert len(rows) == FIT_ANSWERS
    axes, _ = combined.load_projection()
    response_ids = [str(value) for value in axes["response_ids"]]
    assert [str(row["response_id"]) for row in rows] == response_ids
    generators = np.asarray([str(row["model"]) for row in rows])
    assert np.all(generators[:NATIVE_ANSWERS] == TARGET_GENERATOR)
    assert np.all(generators[NATIVE_ANSWERS:] != TARGET_GENERATOR)
    counts = Counter(generators.tolist())
    auxiliary_names = sorted(set(generators.tolist()) - {TARGET_GENERATOR})
    assert len(auxiliary_names) == 5
    assert counts[TARGET_GENERATOR] == NATIVE_ANSWERS
    assert sum(counts[name] for name in auxiliary_names) == AUXILIARY_ANSWERS

    proof = {
        "status": "passed_calibration_and_native_fit_share_target_generator",
        "target_generator": TARGET_GENERATOR,
        "native_fit_answers": NATIVE_ANSWERS,
        "calibration_answers": CAL_ANSWERS,
        "calibration_generator_count": {TARGET_GENERATOR: CAL_ANSWERS},
        "calibration_identity_source": {
            "development_manifest_sha256": sha(DEVELOPMENT_MANIFEST),
            "development_exporter_sha256": sha(DEVELOPMENT_EXPORTER),
            "gold_manifest_sha256": sha(GOLD_MANIFEST),
            "manifest_field": "native_response_model",
            "exporter_rule": "Regex-select llama-2-7b-chat then assert row model equals MODEL for every allowed fit/calibration source.",
            "calibration_answer_or_span_rows_opened_by_gate": False,
        },
        "expanded_generator_counts": dict(sorted(counts.items())),
        "auxiliary_generators": auxiliary_names,
        "auxiliary_answers": AUXILIARY_ANSWERS,
        "expanded_fit_sha256": sha(EXPANDED_FIT),
        "expansion_provenance_sha256": sha(EXPANSION_PROVENANCE),
        "official_test_opened": False,
    }
    return proof, rows, generators, [TARGET_GENERATOR] + auxiliary_names


def source_snapshot() -> dict:
    files = {
        "runner": Path(__file__),
        "upstream_protocol": UPSTREAM / "protocol.json",
        "upstream_preparation": UPSTREAM / "preparation_complete.json",
        "upstream_provenance_audit": UPSTREAM / "PROVENANCE_AUDIT.json",
        "upstream_independent_audit": UPSTREAM / "INDEPENDENT_AUDIT.json",
        "audited_design": DESIGN,
        "combined_projection": combined.OUT / "projection.npz",
        "combined_axes": combined.OUT / "axes.json",
        "expanded_fit": EXPANDED_FIT,
        "expansion_provenance": EXPANSION_PROVENANCE,
        "expansion_freeze": EXPANSION_FREEZE,
        "development_manifest": DEVELOPMENT_MANIFEST,
        "development_exporter": DEVELOPMENT_EXPORTER,
        "gold_manifest": GOLD_MANIFEST,
        "native_fit": NATIVE_FIT,
        "native_summary": NATIVE_SUMMARY,
        "native_scores": NATIVE_SCORES,
        "expanded_summary": EXPANDED_SUMMARY,
        "incumbent": INCUMBENT,
    }
    return {
        "files_sha256": {name: sha(path) for name, path in files.items()},
        "formal_baseline_files_sha256": expanded_v2.baseline_snapshot(),
    }


def prepare() -> None:
    assert sklearn.__version__ == SKLEARN_VERSION
    assert not (OUT / "fit_complete.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / "protocol.json", protocol())
    proof, _, _, generator_names = provenance_gate()
    upstream_prep = read_json(UPSTREAM / "preparation_complete.json")
    upstream_audit = read_json(UPSTREAM / "INDEPENDENT_AUDIT.json")
    upstream_provenance = read_json(UPSTREAM / "PROVENANCE_AUDIT.json")
    assert upstream_prep["native_feature_replay"]["native_whitebox_geometry_exact"] is True
    assert upstream_audit["status"] == "passed_without_reopening_calibration_labels"
    assert upstream_provenance["in_sample_supervised_columns"] == 0
    design = np.load(DESIGN, mmap_mode="r")
    assert design.shape == (FIT_WINDOWS, FEATURE_WIDTH)
    assert np.isfinite(design).all()
    frozen_json(OUT / "PROVENANCE_GATE.json", proof)
    frozen_json(OUT / "source_snapshot.json", source_snapshot())
    complete = {
        "status": "target_provenance_passed_and_weight_grid_frozen_before_fit",
        "generator_names_in_weight_order": generator_names,
        "candidate_alphas": list(AUXILIARY_ALPHAS),
        "feature_design_shape": list(design.shape),
        "feature_design_sha256": sha(DESIGN),
        "native_feature_rows_exact": NATIVE_WINDOWS,
        "generator_identity_feature_count": 0,
        "calibration_answer_or_span_rows_opened": False,
        "fit_labels_opened": False,
        "official_test_opened": False,
        "formal_baselines_modified": False,
        "artifacts_sha256": {
            "protocol.json": sha(OUT / "protocol.json"),
            "PROVENANCE_GATE.json": sha(OUT / "PROVENANCE_GATE.json"),
            "source_snapshot.json": sha(OUT / "source_snapshot.json"),
        },
    }
    atomic_json(OUT / "preparation_complete.json", complete)
    print("TARGET_DOMAIN_WEIGHTING_PREPARED", generator_names, flush=True)


def check_prepared() -> dict:
    done = read_json(OUT / "preparation_complete.json")
    assert done["status"] == "target_provenance_passed_and_weight_grid_frozen_before_fit"
    assert read_json(OUT / "protocol.json") == protocol()
    for name, expected in done["artifacts_sha256"].items():
        assert sha(OUT / name) == expected
    snapshot = read_json(OUT / "source_snapshot.json")
    assert snapshot == source_snapshot()
    assert snapshot["formal_baseline_files_sha256"] == expanded_v2.baseline_snapshot()
    assert read_json(OUT / "PROVENANCE_GATE.json")["status"].startswith("passed_")
    return done


def native_splits(labels: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    native_index = np.arange(NATIVE_WINDOWS, dtype=np.int64)
    splits = []
    all_group_set = set(np.unique(groups).tolist())
    native_group_set = set(np.unique(groups[:NATIVE_WINDOWS]).tolist())
    assert all_group_set == native_group_set and len(native_group_set) == FIT_GROUPS
    for train_native, held_native in GroupKFold(FOLDS).split(
        native_index, labels[:NATIVE_WINDOWS], groups[:NATIVE_WINDOWS]
    ):
        held_groups = np.unique(groups[held_native])
        held_all = np.flatnonzero(np.isin(groups, held_groups))
        train_all = np.flatnonzero(~np.isin(groups, held_groups))
        assert np.array_equal(train_native, train_all[train_all < NATIVE_WINDOWS])
        assert np.array_equal(held_native, held_all[held_all < NATIVE_WINDOWS])
        assert not set(groups[train_all].tolist()) & set(groups[held_all].tolist())
        splits.append((train_native, held_native, train_all, held_all))
    return splits


def generator_components(
    train_all: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    answers: np.ndarray,
    generator_codes: np.ndarray,
    generator_names: list[str],
) -> list[dict]:
    components = []
    for code, name in enumerate(generator_names):
        indices = train_all[generator_codes[train_all] == code]
        assert len(indices) > 0
        base, loss, class_mass = native_v2.window_hierarchy_weights(
            groups, answers, labels, indices
        )
        # Each local generator is first normalized to mean-one over its own rows.
        assert np.isclose(base[indices].sum(), len(indices), rtol=0, atol=1e-6)
        assert np.isclose(loss[indices].sum(), len(indices), rtol=0, atol=1e-6)
        assert np.isclose(class_mass[0], class_mass[1], rtol=0, atol=1e-6)
        components.append({
            "code": code,
            "generator": name,
            "indices": indices,
            "base": base[indices],
            "loss": loss[indices],
            "windows": len(indices),
            "answers": len(np.unique(answers[indices])),
            "groups": len(np.unique(groups[indices])),
        })
    return components


def compose_weights(
    alpha: float,
    components: list[dict],
    labels: np.ndarray,
    native_train_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    assert alpha in AUXILIARY_ALPHAS
    assert components[0]["generator"] == TARGET_GENERATOR and len(components) == 6
    base = np.zeros(FIT_WINDOWS, dtype=np.float64)
    loss = np.zeros(FIT_WINDOWS, dtype=np.float64)
    desired = [native_train_windows / (1.0 + alpha)]
    desired += [native_train_windows * alpha / (5.0 * (1.0 + alpha))] * 5
    rows = []
    for component, target_mass in zip(components, desired):
        indices = component["indices"]
        scale = target_mass / len(indices)
        base[indices] = component["base"] * scale
        loss[indices] = component["loss"] * scale
        by_class = np.bincount(labels[indices], weights=loss[indices], minlength=2)
        if target_mass:
            assert np.isclose(base[indices].sum(), target_mass, rtol=0, atol=1e-5)
            assert np.isclose(loss[indices].sum(), target_mass, rtol=0, atol=1e-5)
            assert np.isclose(by_class[0], by_class[1], rtol=0, atol=1e-5)
        else:
            assert np.all(base[indices] == 0) and np.all(loss[indices] == 0)
        rows.append({
            "generator": component["generator"],
            "windows": component["windows"],
            "answers": component["answers"],
            "groups": component["groups"],
            "target_mass": target_mass,
            "scaler_mass": float(base[indices].sum()),
            "loss_mass": float(loss[indices].sum()),
            "loss_mass_by_class": by_class.astype(float).tolist(),
        })
    assert np.isclose(base.sum(), native_train_windows, rtol=0, atol=1e-5)
    assert np.isclose(loss.sum(), native_train_windows, rtol=0, atol=1e-5)
    active = np.flatnonzero(loss > 0)
    return base, loss, active, {
        "alpha": alpha,
        "normalization_total_mass": native_train_windows,
        "native_to_all_auxiliary_mass_ratio": [1.0, alpha],
        "by_generator": rows,
        "auxiliary_zero_weight_exact": bool(alpha != 0 or all(
            row["loss_mass"] == 0 for row in rows[1:]
        )),
    }


def fit_model(x, labels, active, base_weight, loss_weight):
    """Generator identity is intentionally absent from this fit interface."""
    scaler = StandardScaler()
    scaler.fit(x[active], sample_weight=base_weight[active])
    model = LogisticRegression(
        C=C_VALUE, solver="liblinear", penalty="l2", max_iter=3000,
        random_state=SEED,
    )
    model.fit(scaler.transform(x[active]), labels[active], sample_weight=loss_weight[active])
    assert scaler.n_features_in_ == model.n_features_in_ == FEATURE_WIDTH
    return scaler, model


def predict_probabilities(scaler, model, x_rows: np.ndarray) -> np.ndarray:
    """Inference consumes the 26-column numeric design and nothing else."""
    assert x_rows.shape[1] == FEATURE_WIDTH
    return model.predict_proba(scaler.transform(x_rows))[:, 1]


def candidate_key(record: dict) -> tuple:
    windows = record["native_OOF"]["windows"]
    answers = record["native_OOF"]["answers"]
    return (
        min(windows["f1"], answers["f1"]),
        windows["f1"], answers["f1"],
        windows["average_precision"], answers["average_precision"],
        -record["alpha"],
    )


def fit() -> None:
    assert sklearn.__version__ == SKLEARN_VERSION
    check_prepared()
    assert not (OUT / "fit_complete.json").exists()
    started = time.perf_counter()
    proof, rows, answer_generators, generator_names = provenance_gate()
    assert proof == read_json(OUT / "PROVENANCE_GATE.json")
    axes, projection = combined.load_projection()
    x = np.load(DESIGN, mmap_mode="r")
    labels = projection["window_labels"].astype(np.int8)
    answer_labels = projection["response_labels"].astype(np.int8)
    groups = combined.group_indices("window")
    answers = projection["window_response_index"].astype(np.int32)
    assert x.shape == (FIT_WINDOWS, FEATURE_WIDTH)
    assert labels.shape == (FIT_WINDOWS,) and answer_labels.shape == (FIT_ANSWERS,)
    assert projection["response_window_indptr"][NATIVE_ANSWERS] == NATIVE_WINDOWS
    name_to_code = {name: index for index, name in enumerate(generator_names)}
    answer_codes = np.asarray([name_to_code[name] for name in answer_generators], dtype=np.int8)
    generator_codes = answer_codes[answers]
    assert np.all(generator_codes[:NATIVE_WINDOWS] == 0)

    splits = native_splits(labels, groups)
    oof = np.full((len(AUXILIARY_ALPHAS), FIT_WINDOWS), np.nan, dtype=np.float64)
    fold_models = []
    fold_log = []
    with threadpool_limits(limits=THREADS):
        for fold, (train_native, held_native, train_all, held_all) in enumerate(splits):
            components = generator_components(
                train_all, labels, groups, answers, generator_codes, generator_names
            )
            fold_record = {
                "fold": fold,
                "held_groups": len(np.unique(groups[held_all])),
                "train_native_windows": len(train_native),
                "held_native_windows": len(held_native),
                "train_all_domain_windows": len(train_all),
                "held_all_domain_windows": len(held_all),
                "candidates": [],
            }
            for candidate_index, alpha in enumerate(AUXILIARY_ALPHAS):
                base_weight, loss_weight, active, weight_audit = compose_weights(
                    alpha, components, labels, len(train_native)
                )
                if alpha == 0:
                    assert np.array_equal(active, train_native)
                    assert np.all(base_weight[NATIVE_WINDOWS:] == 0)
                    assert np.all(loss_weight[NATIVE_WINDOWS:] == 0)
                scaler, model = fit_model(x, labels, active, base_weight, loss_weight)
                oof[candidate_index, held_all] = predict_probabilities(
                    scaler, model, x[held_all]
                )
                fold_models.append({
                    "fold": fold, "alpha": alpha,
                    "held_groups": np.unique(groups[held_all]).astype(int).tolist(),
                    "scaler": scaler, "model": model,
                })
                fold_record["candidates"].append({
                    "alpha": alpha,
                    "active_train_windows": len(active),
                    "n_iter": int(model.n_iter_[0]),
                    "weight_audit": weight_audit,
                })
                print("TARGET_DOMAIN_WEIGHTING_FOLD", fold + 1, "ALPHA", alpha, flush=True)
            fold_log.append(fold_record)
    assert np.isfinite(oof).all()

    with np.load(NATIVE_SCORES, allow_pickle=False) as native_saved:
        native_reference = native_saved["selected_scores"]
    alpha_zero_error = np.abs(oof[0, :NATIVE_WINDOWS] - native_reference)
    alpha_zero_max_error = float(alpha_zero_error.max())
    assert alpha_zero_max_error <= np.finfo(np.float64).eps, alpha_zero_max_error

    candidate_records = []
    all_answer_scores = np.empty((len(AUXILIARY_ALPHAS), FIT_ANSWERS), dtype=np.float64)
    for candidate_index, alpha in enumerate(AUXILIARY_ALPHAS):
        scores = oof[candidate_index]
        response_scores = expanded_v2.answer_scores(scores, projection["response_window_indptr"])
        all_answer_scores[candidate_index] = response_scores
        thresholds = {
            "window": q.choose_threshold(labels[:NATIVE_WINDOWS], scores[:NATIVE_WINDOWS]),
            "answer": q.choose_threshold(
                answer_labels[:NATIVE_ANSWERS], response_scores[:NATIVE_ANSWERS]
            ),
        }
        native_metrics = {
            "windows": q.count(
                labels[:NATIVE_WINDOWS], scores[:NATIVE_WINDOWS], thresholds["window"]["threshold"]
            ),
            "answers": q.count(
                answer_labels[:NATIVE_ANSWERS], response_scores[:NATIVE_ANSWERS],
                thresholds["answer"]["threshold"],
            ),
        }
        all_metrics = {
            "windows": q.count(labels, scores, thresholds["window"]["threshold"]),
            "answers": q.count(answer_labels, response_scores, thresholds["answer"]["threshold"]),
        }
        candidate_records.append({
            "candidate": f"auxiliary_total_weight_{alpha:g}",
            "alpha": alpha,
            "native_OOF_thresholds": thresholds,
            "native_OOF": native_metrics,
            "all_domain_OOF_at_native_thresholds": all_metrics,
        })

    native_control = read_json(NATIVE_FIT)["selected"]
    alpha_zero = candidate_records[0]
    for unit in ("window", "answer"):
        ours = alpha_zero["native_OOF"][unit + "s"]
        reference = native_control["fit_OOF"][unit + "s"]
        assert (ours["tp"], ours["fp"], ours["fn"], ours["tn"]) == (
            reference["tp"], reference["fp"], reference["fn"], reference["tn"]
        )
        assert np.isclose(ours["f1"], reference["f1"], rtol=0, atol=1e-15)
        threshold_error = abs(
            alpha_zero["native_OOF_thresholds"][unit]["threshold"]
            - native_control["fit_OOF_thresholds"][unit]["threshold"]
        )
        assert threshold_error <= np.finfo(np.float64).eps

    selected_index = max(range(len(candidate_records)), key=lambda i: candidate_key(candidate_records[i]))
    selected = candidate_records[selected_index]
    selected_alpha = float(selected["alpha"])

    all_indices = np.arange(FIT_WINDOWS, dtype=np.int64)
    components = generator_components(
        all_indices, labels, groups, answers, generator_codes, generator_names
    )
    base_weight, loss_weight, active, full_weight_audit = compose_weights(
        selected_alpha, components, labels, NATIVE_WINDOWS
    )
    scaler, model = fit_model(x, labels, active, base_weight, loss_weight)
    model_payload = {
        "version": VERSION,
        "feature_variant": "whitebox_geometry",
        "feature_names": list(feature_v2.WHITEBOX_NAMES + feature_v2.GEOMETRY_NAMES),
        "feature_width": FEATURE_WIDTH,
        "C": C_VALUE,
        "selected_alpha": selected_alpha,
        "generator_identity_feature_count": 0,
        "scaler": scaler,
        "model": model,
    }
    atomic_pickle(OUT / "model.pkl", model_payload)
    atomic_pickle(OUT / "crossfit_models.pkl", {
        "version": VERSION,
        "feature_width": FEATURE_WIDTH,
        "generator_identity_feature_count": 0,
        "models": fold_models,
    })
    atomic_npz(
        OUT / "fit_scores.npz",
        alphas=np.asarray(AUXILIARY_ALPHAS, dtype=np.float64),
        window_scores=oof,
        answer_scores=all_answer_scores,
        window_labels=labels,
        answer_labels=answer_labels,
        selected_index=np.asarray(selected_index, dtype=np.int32),
    )
    assert read_json(OUT / "source_snapshot.json")["formal_baseline_files_sha256"] == expanded_v2.baseline_snapshot()
    complete = {
        "status": "target_native_OOF_selected_model_and_thresholds_frozen_before_calibration",
        "selection_rule": protocol()["selection_rule"],
        "selected": selected,
        "selected_index": selected_index,
        "candidate_table": candidate_records,
        "alpha_zero_native_v2_replay": {
            "score_array_bit_exact": bool(alpha_zero_max_error == 0),
            "within_one_float64_ulp": True,
            "threshold_decisions_and_confusion_counts_identical": True,
            "rows": NATIVE_WINDOWS,
            "max_abs_error": alpha_zero_max_error,
            "native_scores_sha256": sha(NATIVE_SCORES),
        },
        "crossfit": fold_log,
        "full_fit": {
            "alpha": selected_alpha,
            "active_windows": len(active),
            "all_windows": FIT_WINDOWS,
            "native_windows": NATIVE_WINDOWS,
            "groups": FIT_GROUPS,
            "C": C_VALUE,
            "n_iter": int(model.n_iter_[0]),
            "weight_audit": full_weight_audit,
        },
        "inference_input_audit": {
            "feature_width": FEATURE_WIDTH,
            "feature_names": model_payload["feature_names"],
            "serialized_estimator_n_features_in": int(model.n_features_in_),
            "serialized_scaler_n_features_in": int(scaler.n_features_in_),
            "generator_identity_feature_count": 0,
            "serialized_generator_encoder": False,
            "predict_function_arguments": ["scaler", "model", "x_rows"],
            "generator_identity_used_only_for_training_weight_construction": True,
        },
        "artifacts_sha256": {
            "model.pkl": sha(OUT / "model.pkl"),
            "crossfit_models.pkl": sha(OUT / "crossfit_models.pkl"),
            "fit_scores.npz": sha(OUT / "fit_scores.npz"),
            "protocol.json": sha(OUT / "protocol.json"),
            "preparation_complete.json": sha(OUT / "preparation_complete.json"),
            "source_snapshot.json": sha(OUT / "source_snapshot.json"),
            "runner": sha(Path(__file__)),
        },
        "fit_labels_opened": True,
        "calibration_labels_opened": False,
        "calibration_used_for_alpha_model_or_threshold_selection": False,
        "all_domain_OOF_used_for_selection": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "GPU_used": False,
        "sklearn_version": sklearn.__version__,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "fit_complete.json", complete)
    print(
        "TARGET_DOMAIN_WEIGHTING_FROZEN", selected_alpha,
        selected["native_OOF"]["windows"]["f1"],
        selected["native_OOF"]["answers"]["f1"], flush=True,
    )


def comparison_references() -> dict:
    native_fit = read_json(NATIVE_FIT)
    native_summary = read_json(NATIVE_SUMMARY)
    expanded_summary = read_json(EXPANDED_SUMMARY)
    incumbent = read_json(INCUMBENT)
    return {
        "native_only_v2": {
            "native_OOF": native_fit["selected"]["fit_OOF"],
            "calibration_strict_fit_thresholds": native_summary["calibration_strict_fit_thresholds"],
            "calibration_F1Opt_diagnostic": native_summary["calibration_F1Opt_diagnostic"],
            "selection_scope": "fit-native-only",
        },
        "old_unweighted_expanded_v1": {
            "all_domain_OOF": expanded_summary["fit_OOF"],
            "calibration_strict_fit_thresholds": expanded_summary["calibration_at_frozen_fit_thresholds"],
            "selection_scope": "fixed feature/C but thresholds fitted on all-domain expanded OOF",
        },
        "historical_incumbent_cal_selected": {
            "candidate": incumbent["candidate"],
            "calibration": incumbent["metrics"]["calibration"],
            "selection_scope": "historical repeated-calibration selection; descriptive reference only",
        },
    }


def evaluate() -> None:
    assert sklearn.__version__ == SKLEARN_VERSION
    fit_done = read_json(OUT / "fit_complete.json")
    assert fit_done["status"] == "target_native_OOF_selected_model_and_thresholds_frozen_before_calibration"
    assert not (OUT / "summary.json").exists(), "Calibration evaluation is single-use"
    for name, expected in fit_done["artifacts_sha256"].items():
        path = Path(__file__) if name == "runner" else OUT / name
        assert sha(path) == expected
    snapshot = read_json(OUT / "source_snapshot.json")
    assert snapshot["formal_baseline_files_sha256"] == expanded_v2.baseline_snapshot()
    started = time.perf_counter()

    # This is the sole calibration metric pass, after all selection is frozen.
    union, _, meta, whitebox_audit = native_v2.build_partition_design("calibration")
    columns = feature_v2.variant_indices("whitebox_geometry")
    x = union[:, columns]
    assert x.shape == (CAL_WINDOWS, FEATURE_WIDTH)
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    assert payload["version"] == VERSION
    assert payload["feature_width"] == FEATURE_WIDTH
    assert payload["generator_identity_feature_count"] == 0
    scores = predict_probabilities(payload["scaler"], payload["model"], x)
    answer_score = native_v2.answer_scores(meta, scores)
    assert answer_score.shape == (CAL_ANSWERS,)
    thresholds = fit_done["selected"]["native_OOF_thresholds"]
    strict = native_v2.metric_pair(meta, scores, thresholds)
    diagnostic_thresholds = native_v2.choose_thresholds(meta, scores)
    diagnostic = native_v2.metric_pair(meta, scores, diagnostic_thresholds)
    atomic_npz(
        OUT / "calibration_scores.npz",
        window_scores=scores,
        answer_scores=answer_score,
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["answers"]]),
    )

    references = comparison_references()
    selected_native = fit_done["selected"]["native_OOF"]
    selected_all = fit_done["selected"]["all_domain_OOF_at_native_thresholds"]
    native_strict = references["native_only_v2"]["calibration_strict_fit_thresholds"]
    incumbent_cal = references["historical_incumbent_cal_selected"]["calibration"]
    deltas = {
        "vs_native_only_v2_strict": {
            "windows_f1": strict["windows"]["f1"] - native_strict["windows"]["f1"],
            "answers_f1": strict["answers"]["f1"] - native_strict["answers"]["f1"],
        },
        "vs_historical_incumbent_cal_selected": {
            "windows_f1": strict["windows"]["f1"] - incumbent_cal["windows"]["f1"],
            "answers_f1": strict["answers"]["f1"] - incumbent_cal["answers"]["f1"],
        },
    }
    assert snapshot["formal_baseline_files_sha256"] == expanded_v2.baseline_snapshot()
    summary = {
        "status": "single_strict_calibration_evaluation_complete",
        "candidate": fit_done["selected"]["candidate"],
        "selected_alpha": fit_done["selected"]["alpha"],
        "native_OOF": selected_native,
        "native_OOF_thresholds": thresholds,
        "all_domain_OOF_at_native_thresholds": selected_all,
        "calibration_strict_native_OOF_thresholds": strict,
        "calibration_F1Opt_diagnostic": diagnostic,
        "calibration_F1Opt_thresholds": diagnostic_thresholds,
        "comparison_references": references,
        "strict_deltas": deltas,
        "calibration_evaluations": 1,
        "calibration_F1Opt_used_for_selection": False,
        "calibration_thresholds_fitted_for_deployment": 0,
        "calibration_whitebox_reconstruction_audit": whitebox_audit,
        "inference_input_audit": fit_done["inference_input_audit"],
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "artifacts_sha256": {"calibration_scores.npz": sha(OUT / "calibration_scores.npz")},
        "formal_baseline_files_unchanged": True,
        "calibration_labels_opened": True,
        "calibration_used_for_alpha_model_or_native_OOF_threshold_selection": False,
        "official_test_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "summary.json", summary)

    candidate_lines = []
    for record in fit_done["candidate_table"]:
        candidate_lines.append(
            f"| {record['alpha']:g} | {record['native_OOF']['windows']['f1']:.6f} | "
            f"{record['native_OOF']['windows']['average_precision']:.6f} | "
            f"{record['native_OOF']['answers']['f1']:.6f} | "
            f"{record['native_OOF']['answers']['average_precision']:.6f} |"
        )
    report = f"""# Target-domain weighted expanded semantic window v2

Provenance gate passed: native fit634 and calibration159 are both released `llama-2-7b-chat` answers. The expanded-only 3046 answers come from five other generators. The gate is bound to the development manifest and its exact exporter; it did not open calibration answer/span rows.

The audited 26-column `whitebox_geometry` design and C=0.1 were fixed. Native training mass is 1; total auxiliary mass is alpha, split equally across five generators. Within each generator, source groups, answers, and windows are hierarchically equalized before separate binary window-class balance. Every fold/candidate is finally normalized to the native training-window mass.

| auxiliary alpha | native OOF window F1 | window AP | native OOF answer F1 | answer AP |
|---:|---:|---:|---:|---:|
{chr(10).join(candidate_lines)}

Selected alpha: **{fit_done['selected']['alpha']:g}**, using only held native OOF scores. Alpha=0 reproduced native-only v2 OOF within one float64 ULP, with identical threshold decisions and confusion counts (max absolute score error {fit_done['alpha_zero_native_v2_replay']['max_abs_error']:.3g}).

| evaluation | window F1 | answer F1 |
|---|---:|---:|
| selected native OOF / native-F1Opt | {selected_native['windows']['f1']:.6f} | {selected_native['answers']['f1']:.6f} |
| selected all-domain OOF / frozen native thresholds | {selected_all['windows']['f1']:.6f} | {selected_all['answers']['f1']:.6f} |
| calibration strict / frozen native OOF thresholds | {strict['windows']['f1']:.6f} | {strict['answers']['f1']:.6f} |
| calibration F1Opt diagnostic only | {diagnostic['windows']['f1']:.6f} | {diagnostic['answers']['f1']:.6f} |
| native-only v2 strict | {native_strict['windows']['f1']:.6f} | {native_strict['answers']['f1']:.6f} |
| historical cal-selected incumbent | {incumbent_cal['windows']['f1']:.6f} | {incumbent_cal['answers']['f1']:.6f} |

Strict deltas versus native-only v2 are {deltas['vs_native_only_v2_strict']['windows_f1']:+.6f} window and {deltas['vs_native_only_v2_strict']['answers_f1']:+.6f} answer F1. Versus the historical cal-selected incumbent they are {deltas['vs_historical_incumbent_cal_selected']['windows_f1']:+.6f} and {deltas['vs_historical_incumbent_cal_selected']['answers_f1']:+.6f}.

Generator identity has zero inference columns. It is used only to construct training sample weights; every scaler and classifier consumes the same 26 numeric whitebox+geometry columns. Calibration was evaluated once after alpha, the full-fit model, and both thresholds were frozen. Cal-F1Opt is diagnostic only. Formal baseline hashes stayed unchanged and the official test remained unopened.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        "TARGET_DOMAIN_WEIGHTING_CALIBRATION", strict["windows"]["f1"],
        strict["answers"]["f1"], flush=True,
    )


def status() -> None:
    print(json.dumps({
        "prepared": (OUT / "preparation_complete.json").exists(),
        "fit_frozen": (OUT / "fit_complete.json").exists(),
        "calibration_evaluated": (OUT / "summary.json").exists(),
        "official_test_opened": False,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "evaluate", "status"))
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        {"prepare": prepare, "fit": fit, "evaluate": evaluate, "status": status}[args.command]()


if __name__ == "__main__":
    main()
