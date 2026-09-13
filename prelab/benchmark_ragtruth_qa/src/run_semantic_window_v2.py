"""CPU-only direct 4-BPE window readout for semantic source attribution v2.

Stages are intentionally separated. ``fit`` is the only stage that reads fit
labels and freezes the feature variant, C, two thresholds, and full-fit model.
``evaluate`` is the only stage that reads calibration rows and may run once per
output directory.  No withheld-data path is defined by this runner.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_development as q  # noqa: E402
import run_semantic_source_attribution_v1_score as source_v1  # noqa: E402
import semantic_window_v2_features as feature_v2  # noqa: E402


VERSION = "semantic-window-v2-native634"
DEFAULT_OUT = ROOT / "results/semantic_window_v2"
PROTOCOL_DOCUMENT = ROOT / "research/semantic_window_v2/PROTOCOL.md"
DIAGNOSIS_DIR = ROOT / "research/semantic_source_attribution_v1_failure_diagnosis"
CLAIM_CACHE = {
    "fit": DIAGNOSIS_DIR / "fit_feature_cache.npz",
    "calibration": DIAGNOSIS_DIR / "calibration_feature_cache.npz",
}
CLAIM_CACHE_META = {
    "fit": DIAGNOSIS_DIR / "fit_feature_cache.json",
    "calibration": DIAGNOSIS_DIR / "calibration_feature_cache.json",
}
LOCAL_PROBABILITIES = ROOT / "results/group_crossfit_lb_large_v1/oof_input_scores.npy"
GENERATION_MATRIX = ROOT / "results/development_v1/matrices/base.npy"
UPSTREAM_OOF_DIR = ROOT / "results/group_crossfit_lb_large_v1"
UPSTREAM_OOF_PROTOCOL = UPSTREAM_OOF_DIR / "protocol.json"
UPSTREAM_OOF_FOLDS = UPSTREAM_OOF_DIR / "folds.json"
UPSTREAM_OOF_COMPLETE = UPSTREAM_OOF_DIR / "complete.json"
UPSTREAM_OOF_AUDIT = UPSTREAM_OOF_DIR / "INDEPENDENT_AUDIT.json"
UPSTREAM_OLD_INPUTS = UPSTREAM_OOF_DIR / "old_input_scores.npy"
ATOMIC_PROTOCOL = ROOT / "results/atomic_microclaim_nli_v1/protocol.json"
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_CLAIMS = {"fit": 9055, "calibration": 2267}
EXPECTED_GROUPS = {"fit": 615, "calibration": 154}
WINDOW_BOUNDS = {"fit": (0, 168123), "calibration": (168123, 210364)}
FOLDS = 5
THREADS = 4
SEED = 20260913
C_VALUES = (0.001, 0.01, 0.1)
VARIANTS = tuple(feature_v2.VARIANT_BLOCKS)

LOOKBACK_REFERENCE = {"windows": 0.6008821240162587, "answers": 0.8454545454545455}
INCUMBENT_REFERENCE = {"windows": 0.6902813989031736, "answers": 0.8910891089108911}

BASELINE_FILES = (
    ROOT / "results/lookback_official_span_v1/lookback_lr.pkl",
    ROOT / "results/lookback_official_span_v1/protocol.json",
    ROOT / "results/lookback_official_span_v1/summary.json",
    ROOT / "results/lookback_k4_evaluation_adapter_v1/protocol.json",
    ROOT / "results/lookback_k4_evaluation_adapter_v1/summary.json",
    ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json",
    ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, f"Frozen file changed: {path}"
    else:
        atomic_json(path, value)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(pickle.dumps(value, protocol=5))
    pending.replace(path)


def protocol() -> dict:
    widths = {
        name: int(len(feature_v2.variant_indices(name))) for name in VARIANTS
    }
    return {
        "version": VERSION,
        "scope": {
            "fit": "Native QA fit only: 634 answers in 615 source-connected groups.",
            "calibration": "159 answers; opened once only after the complete fit freeze.",
            "expanded_fit": "Not consumed in this native-634 pilot; feature builder is a separate reusable module.",
            "official_test": "No path or loader for official test exists in this runner.",
            "formal_baselines": "Read-only hash references; no baseline file is written.",
        },
        "repair": {
            "training_unit": "Eligible native 4-original-BPE stride-one window and its unchanged any-error label.",
            "claim_mapping": "Claim features are averaged by exact lexical-token ownership inside each window.",
            "disabled": ["conflict-only head", "max(any, conflict)", "semantic add-gate", "claim-score max propagation"],
            "answer_score": "Maximum of all eligible window probabilities in the answer.",
        },
        "features": {
            "union_width": len(feature_v2.UNION_NAMES),
            "blocks": {
                name: list(values) for name, values in feature_v2.BLOCK_NAMES.items()
            },
            "attribution": "Four fixed contiguous eight-layer bands. Raw logmass keeps mean and across-head/layer std; compact routing keeps band means and passage HHI. No learned layer/head selection.",
            "whitebox": "Frozen 12-D claim summaries. Their fit Lookback/large inputs are upstream group-OOF; calibration inputs are upstream full-fit predictions.",
            "local_clean": "Optional direct window Lookback/large plus generation NLL. Fit Lookback/large are upstream group-OOF; calibration is full-fit.",
            "labels_used_to_build_features": False,
        },
        "candidates": {
            "variants_in_order": list(VARIANTS),
            "variant_blocks": {
                name: list(feature_v2.VARIANT_BLOCKS[name]) for name in VARIANTS
            },
            "widths": widths,
            "C": list(C_VALUES),
            "model": "StandardScaler fitted with hierarchy base weights, then L2 logistic regression/liblinear.",
            "target": "any-error window only",
        },
        "fit": {
            "crossfit": "Five-fold sklearn GroupKFold on source-connected group_id.",
            "weights": "Within each fold: group equal, answer equal within group, window equal within answer; one binary-class balance step; no group reset after class balance.",
            "thresholds": "Window and answer thresholds separately use fit OOF F1, then precision, then higher cutoff.",
            "selection": "Maximize min(window F1, answer F1), window F1, answer F1, window AP, answer AP, fewer dimensions, smaller C, earlier preregistered variant.",
            "full_fit": "Only the selected variant/C is refit on all fit windows after selection.",
        },
        "evaluation": {
            "strict": "Apply the two frozen fit-OOF thresholds to calibration.",
            "diagnostic": "Report separate calibration F1-opt thresholds without changing the frozen method.",
            "references": {
                "Lookback_common_cal_F1Opt": LOOKBACK_REFERENCE,
                "historical_incumbent_cal_selected": INCUMBENT_REFERENCE,
            },
        },
        "seed": SEED,
        "cpu_threads": THREADS,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def baseline_snapshot() -> dict[str, str]:
    assert all(path.is_file() for path in BASELINE_FILES)
    return {str(path.resolve()): sha(path) for path in BASELINE_FILES}


def source_snapshot() -> dict:
    paths = (
        Path(__file__),
        HERE / "semantic_window_v2_features.py",
        PROTOCOL_DOCUMENT,
        source_v1.ROWS_PATH,
        source_v1.LAYOUTS_PATH,
        source_v1.ATTR_MANIFEST_PATH,
        CLAIM_CACHE["fit"],
        CLAIM_CACHE_META["fit"],
        CLAIM_CACHE["calibration"],
        CLAIM_CACHE_META["calibration"],
        LOCAL_PROBABILITIES,
        GENERATION_MATRIX,
        UPSTREAM_OOF_PROTOCOL,
        UPSTREAM_OOF_FOLDS,
        UPSTREAM_OOF_COMPLETE,
        UPSTREAM_OOF_AUDIT,
        UPSTREAM_OLD_INPUTS,
        ATOMIC_PROTOCOL,
    )
    assert all(path.is_file() for path in paths)
    return {
        "files_sha256": {str(path.resolve()): sha(path) for path in paths},
        "formal_baseline_files_sha256": baseline_snapshot(),
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def validate_label_free_inputs() -> dict:
    score_hash = sha(HERE / "run_semantic_source_attribution_v1_score.py")
    manifest_hash = sha(source_v1.ATTR_MANIFEST_PATH)
    claim_shapes = {}
    for partition in ("fit", "calibration"):
        metadata = read_json(CLAIM_CACHE_META[partition])
        assert metadata["partition"] == partition
        assert metadata["score_runner_sha256"] == score_hash
        assert metadata["attribution_manifest_sha256"] == manifest_hash
        with np.load(CLAIM_CACHE[partition], allow_pickle=False) as loaded:
            assert loaded["semantic"].shape == (
                EXPECTED_CLAIMS[partition], feature_v2.SEMANTIC_WIDTH
            )
            assert loaded["response_ids"].shape == (EXPECTED_CLAIMS[partition],)
            assert np.isfinite(loaded["semantic"]).all()
            claim_shapes[partition] = list(loaded["semantic"].shape)
    local = np.load(LOCAL_PROBABILITIES, mmap_mode="r")
    old_local = np.load(UPSTREAM_OLD_INPUTS, mmap_mode="r")
    generation = np.load(GENERATION_MATRIX, mmap_mode="r")
    assert local.shape == (sum(EXPECTED_WINDOWS.values()), 2)
    assert old_local.shape == local.shape
    assert generation.shape == (sum(EXPECTED_WINDOWS.values()), 1025)
    assert np.isfinite(local).all() and np.isfinite(generation[:, -1]).all()
    # Calibration columns are the frozen full-fit inputs in both files.  Fit
    # columns must differ because LOCAL_PROBABILITIES replaces them by OOF.
    fit_right = WINDOW_BOUNDS["fit"][1]
    assert np.array_equal(local[fit_right:], old_local[fit_right:])
    assert np.any(local[:fit_right, 0] != old_local[:fit_right, 0])
    assert np.any(local[:fit_right, 1] != old_local[:fit_right, 1])

    upstream = read_json(UPSTREAM_OOF_PROTOCOL)
    audit = read_json(UPSTREAM_OOF_AUDIT)
    complete = read_json(UPSTREAM_OOF_COMPLETE)
    assert upstream["version"] == "group-crossfit-lb-large-v1"
    assert upstream["scope"].startswith("All3680 fit answers retained across folds")
    assert upstream["lookback"]["calibration"].startswith("No fold calibration")
    assert upstream["large"]["saved"].startswith("Three online training logs")
    assert upstream["official_test_opened"] is False
    assert audit["status"] == "passed_independent_arithmetic_and_wiring_replay"
    assert audit["fold_answer_coverage_exactly_once"] is True
    assert audit["fold_window_coverage_exactly_once"] is True
    assert audit["fit_hold_groups_disjoint"] is True
    assert audit["oof_columns_reconstructed_exactly"] is True
    assert complete["upstream_LR_fits"] == complete["upstream_large_fits"] == 3
    assert complete["official_test_opened"] is False
    atomic_protocol = read_json(ATOMIC_PROTOCOL)
    assert "group-OOF Lookback/large" in atomic_protocol["features"]["whitebox"]
    return {
        "claim_cache_shapes": claim_shapes,
        "local_probability_shape": list(local.shape),
        "generation_shape": list(generation.shape),
        "fit_local_columns_declared_and_independently_audited_source_group_OOF": True,
        "calibration_local_columns_equal_frozen_full_fit_inputs": True,
    }


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "fit_complete.json").exists()
    assert not (out / "summary.json").exists()
    frozen_json(out / "protocol.json", protocol())
    selfcheck = feature_v2.synthetic_selfcheck()
    checked = validate_label_free_inputs()
    snapshot = source_snapshot()
    frozen_json(out / "source_snapshot.json", snapshot)
    frozen_json(out / "CPU_SELFCHECK.json", {
        **selfcheck,
        **checked,
        "fit_labels_opened": False,
        "calibration_labels_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    })
    frozen_json(out / "PREPARATION.json", {
        "status": "prepared_label_free_inputs_only",
        "protocol_sha256": sha(out / "protocol.json"),
        "source_snapshot_sha256": sha(out / "source_snapshot.json"),
        "selfcheck_sha256": sha(out / "CPU_SELFCHECK.json"),
        "fit_labels_opened": False,
        "calibration_labels_opened": False,
        "model_trained": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    })
    print("SEMANTIC_WINDOW_V2_PREPARED", len(feature_v2.UNION_NAMES), flush=True)


def check_prepared(out: Path) -> dict:
    preparation = read_json(out / "PREPARATION.json")
    assert preparation["status"] == "prepared_label_free_inputs_only"
    assert read_json(out / "protocol.json") == protocol()
    assert preparation["protocol_sha256"] == sha(out / "protocol.json")
    assert preparation["source_snapshot_sha256"] == sha(out / "source_snapshot.json")
    snapshot = read_json(out / "source_snapshot.json")
    for path, expected in snapshot["files_sha256"].items():
        assert sha(Path(path)) == expected, path
    assert baseline_snapshot() == snapshot["formal_baseline_files_sha256"]
    return preparation


def load_claim_cache(partition: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(CLAIM_CACHE[partition], allow_pickle=False) as loaded:
        semantic = loaded["semantic"].astype(np.float32, copy=True)
        response_ids = loaded["response_ids"].astype(str, copy=True)
    assert semantic.shape == (EXPECTED_CLAIMS[partition], feature_v2.SEMANTIC_WIDTH)
    return semantic, response_ids


def local_signals(partition: str) -> np.ndarray:
    left, right = WINDOW_BOUNDS[partition]
    probability = np.load(LOCAL_PROBABILITIES, mmap_mode="r")
    generation = np.load(GENERATION_MATRIX, mmap_mode="r")
    result = np.column_stack((
        np.asarray(probability[left:right], dtype=np.float32),
        np.asarray(generation[left:right, -1], dtype=np.float32),
    ))
    assert result.shape == (EXPECTED_WINDOWS[partition], len(feature_v2.LOCAL_NAMES))
    assert np.isfinite(result).all()
    return result


def assert_fit_upstream_source_group_oof(meta: dict) -> dict:
    """Rebuild both supervised fit columns from fold-held artifacts exactly."""
    folds = read_json(UPSTREAM_OOF_FOLDS)
    assert len(folds) == 3
    rebuilt = np.full((EXPECTED_WINDOWS["fit"], 2), np.nan, dtype=np.float64)
    coverage = np.zeros(EXPECTED_WINDOWS["fit"], dtype=np.int8)
    held_group_union: set[str] = set()
    for fold in folds:
        train_groups = set(map(str, fold["fit_groups"]))
        held_groups = set(map(str, fold["hold_groups"]))
        assert not train_groups & held_groups
        assert not held_group_union & held_groups
        held_group_union |= held_groups
        held = np.asarray(fold["hold_window_indices"], dtype=np.int64)
        assert len(held) and held.min() >= 0 and held.max() < EXPECTED_WINDOWS["fit"]
        actual_groups = {str(meta["windows"][index]["group_id"]) for index in held}
        assert actual_groups == held_groups
        folder = UPSTREAM_OOF_DIR / f"fold_{int(fold['fold'])}"
        for kind in ("lb", "large"):
            completion = read_json(folder / f"{kind}_complete.json")
            for name, expected in completion["files_sha256"].items():
                assert sha(folder / name) == expected
            if kind == "large":
                assert completion["no_hold_gold_or_calibration_used_for_training_or_epoch_selection"] is True
        with np.load(folder / "lb_hold_scores.npz", allow_pickle=False) as loaded:
            assert np.array_equal(loaded["window_indices"], held)
            rebuilt[held, 0] = loaded["scores"]
        with np.load(folder / "large_hold_token_probabilities.npz", allow_pickle=False) as loaded:
            probability = {name: loaded[name].copy() for name in loaded.files}
        for index in held:
            window = meta["windows"][int(index)]
            assert str(window["group_id"]) in held_groups
            rebuilt[index, 1] = max(
                probability[window["response_id"]][window["lexical_token_indices"]]
            )
        coverage[held] += 1
    expected = np.asarray(
        np.load(LOCAL_PROBABILITIES, mmap_mode="r")[:EXPECTED_WINDOWS["fit"]]
    )
    assert len(held_group_union) == EXPECTED_GROUPS["fit"]
    assert np.all(coverage == 1)
    assert np.array_equal(rebuilt, expected)
    return {
        "upstream_folds": len(folds),
        "source_groups": len(held_group_union),
        "fit_windows_reconstructed": len(rebuilt),
        "fit_window_coverage_exactly_once": True,
        "fit_hold_groups_disjoint": True,
        "lookback_column_exact": True,
        "large_column_exact": True,
        "calibration_used_by_upstream_folds": False,
    }


def assert_whitebox_pooling_from_local_signals(
    partition: str, rows: list[dict], meta: dict
) -> dict:
    """Recompute all 12 claim summaries from the declared OOF/full-fit rows."""
    semantic, response_ids = load_claim_cache(partition)
    observed = semantic[:, feature_v2.WHITEBOX]
    signals = local_signals(partition)
    expected_rows = []
    cursor = 0
    for row in rows:
        count = len(row["claims"])
        assert np.all(response_ids[cursor:cursor + count] == row["response_id"])
        for claim in row["claims"]:
            token_ids = set(map(int, claim["lexical_token_indices"]))
            indices = [
                window_index
                for window_index in meta["answer_windows"][row["response_id"]]
                if token_ids.intersection(meta["windows"][window_index]["token_indices"])
            ]
            assert indices
            one = signals[indices]
            expected_rows.append(np.concatenate((
                one.max(axis=0), one.mean(axis=0), one.min(axis=0), one.std(axis=0)
            )))
        cursor += count
    expected = np.asarray(expected_rows, dtype=np.float32)
    assert cursor == EXPECTED_CLAIMS[partition]
    assert expected.shape == observed.shape
    max_abs = float(np.max(np.abs(expected.astype(np.float64) - observed.astype(np.float64))))
    assert np.array_equal(expected, observed), max_abs
    return {
        "partition": partition,
        "claims_recomputed": len(expected),
        "whitebox_columns": expected.shape[1],
        "max_abs_error": max_abs,
        "fit_inputs_source_group_OOF": partition == "fit",
        "calibration_inputs_full_fit": partition == "calibration",
    }


def build_partition_design(partition: str) -> tuple[np.ndarray, list[dict], dict, dict]:
    """Load one partition and invoke the reusable label-free window builder."""
    rows, _ = source_v1.partition_rows_and_layouts(partition)
    semantic, response_ids = load_claim_cache(partition)
    # This metadata call is deliberately inside fit/evaluate.  The builder does
    # not read labels, but the shared JSON rows carry them.
    meta = source_v1.load_partition_meta(partition)
    whitebox_audit = assert_whitebox_pooling_from_local_signals(partition, rows, meta)
    matrix = feature_v2.build_window_feature_matrix(
        semantic, response_ids, rows, meta, local_signals(partition)
    )
    assert matrix.shape == (EXPECTED_WINDOWS[partition], len(feature_v2.UNION_NAMES))
    assert len(rows) == EXPECTED_ANSWERS[partition]
    return matrix, rows, meta, whitebox_audit


def window_hierarchy_weights(
    groups: np.ndarray,
    answer_ids: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal group/answer/window base mass, followed by one class balance."""
    tree: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in active:
        tree[str(groups[index])][str(answer_ids[index])].append(int(index))
    base = np.zeros(len(labels), dtype=np.float64)
    for answers in tree.values():
        for indices in answers.values():
            base[indices] = 1.0 / (len(answers) * len(indices))
    mask = base > 0
    base[mask] /= base[mask].mean()
    class_mass = np.bincount(labels[mask], weights=base[mask], minlength=2)
    assert np.all(class_mass > 0)
    loss = base.copy()
    loss[mask] *= (class_mass.sum() / (2 * class_mass))[labels[mask]]
    loss[mask] *= mask.sum() / loss[mask].sum()
    final_mass = np.bincount(labels[mask], weights=loss[mask], minlength=2)
    assert np.isclose(final_mass[0], final_mass[1], rtol=0, atol=1e-7)
    assert np.isclose(final_mass.sum(), mask.sum(), rtol=0, atol=1e-7)
    return base, loss, final_mass


def fit_scaled_lr(
    x: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
    base_weight: np.ndarray,
    loss_weight: np.ndarray,
    c_value: float,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    scaler.fit(x[active], sample_weight=base_weight[active])
    transformed = scaler.transform(x[active])
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        penalty="l2",
        max_iter=3000,
        random_state=SEED,
    )
    model.fit(transformed, y[active], sample_weight=loss_weight[active])
    return scaler, model


def answer_scores(meta: dict, scores: np.ndarray) -> np.ndarray:
    return np.asarray([
        scores[meta["answer_windows"][answer["response_id"]]].max()
        for answer in meta["answers"]
    ], dtype=np.float64)


def choose_thresholds(meta: dict, scores: np.ndarray) -> dict:
    return {
        "window": q.choose_threshold(
            [window["label"] for window in meta["windows"]], scores
        ),
        "answer": q.choose_threshold(
            [answer["label"] for answer in meta["answers"]],
            answer_scores(meta, scores),
        ),
    }


def metric_pair(meta: dict, scores: np.ndarray, thresholds: dict) -> dict:
    return {
        "windows": q.count(
            [window["label"] for window in meta["windows"]],
            scores,
            thresholds["window"]["threshold"],
        ),
        "answers": q.count(
            [answer["label"] for answer in meta["answers"]],
            answer_scores(meta, scores),
            thresholds["answer"]["threshold"],
        ),
    }


def candidate_selection_key(record: dict, prereg_index: int) -> tuple:
    windows = record["fit_OOF"]["windows"]
    answers = record["fit_OOF"]["answers"]
    return (
        min(windows["f1"], answers["f1"]),
        windows["f1"],
        answers["f1"],
        windows["average_precision"],
        answers["average_precision"],
        -record["width"],
        -record["C"],
        -prereg_index,
    )


def crossfit_grid(
    union: np.ndarray,
    meta: dict,
) -> tuple[np.ndarray, list[str], list[dict], list[dict]]:
    labels = np.asarray([window["label"] for window in meta["windows"]], dtype=np.int8)
    groups = np.asarray([window["group_id"] for window in meta["windows"]])
    answer_ids = np.asarray([window["answer_id"] for window in meta["windows"]])
    indices = np.arange(len(labels))
    splits = list(GroupKFold(FOLDS).split(indices, labels, groups))
    candidate_names = [
        f"{variant}__C{c_value:g}" for variant in VARIANTS for c_value in C_VALUES
    ]
    candidate_lookup = {name: index for index, name in enumerate(candidate_names)}
    oof = np.full((len(candidate_names), len(labels)), np.nan, dtype=np.float64)
    coefficients: dict[str, list[list[float]]] = {name: [] for name in candidate_names}
    fold_log = []

    with threadpool_limits(limits=THREADS):
        for fold, (train, held) in enumerate(splits):
            assert not set(groups[train]) & set(groups[held])
            base, loss, final_mass = window_hierarchy_weights(
                groups, answer_ids, labels, train
            )
            record = {
                "fold": fold,
                "train_windows": len(train),
                "held_windows": len(held),
                "train_answers": len(set(answer_ids[train])),
                "held_answers": len(set(answer_ids[held])),
                "train_groups": len(set(groups[train])),
                "held_groups": len(set(groups[held])),
                "group_overlap": 0,
                "train_positive": int(labels[train].sum()),
                "held_positive": int(labels[held].sum()),
                "final_loss_mass_by_class": final_mass.tolist(),
            }
            for variant in VARIANTS:
                columns = feature_v2.variant_indices(variant)
                x = union[:, columns]
                scaler = StandardScaler()
                scaler.fit(x[train], sample_weight=base[train])
                z_train = scaler.transform(x[train])
                z_held = scaler.transform(x[held])
                for c_value in C_VALUES:
                    name = f"{variant}__C{c_value:g}"
                    model = LogisticRegression(
                        C=c_value,
                        solver="liblinear",
                        penalty="l2",
                        max_iter=3000,
                        random_state=SEED,
                    )
                    model.fit(z_train, labels[train], sample_weight=loss[train])
                    oof[candidate_lookup[name], held] = model.predict_proba(z_held)[:, 1]
                    coefficients[name].append(model.coef_[0].astype(float).tolist())
            fold_log.append(record)
            print("SEMANTIC_WINDOW_V2_FOLD", fold + 1, FOLDS, flush=True)
    assert np.isfinite(oof).all()

    table = []
    for prereg_index, name in enumerate(candidate_names):
        variant, c_text = name.rsplit("__C", 1)
        c_value = float(c_text)
        scores = oof[prereg_index]
        thresholds = choose_thresholds(meta, scores)
        metrics = metric_pair(meta, scores, thresholds)
        table.append({
            "candidate": name,
            "variant": variant,
            "blocks": list(feature_v2.VARIANT_BLOCKS[variant]),
            "width": int(len(feature_v2.variant_indices(variant))),
            "C": c_value,
            "fit_OOF_thresholds": thresholds,
            "fit_OOF": metrics,
            "fold_standardized_coefficients": coefficients[name],
        })
    return oof, candidate_names, table, fold_log


def fit(out: Path) -> None:
    check_prepared(out)
    assert not (out / "fit_complete.json").exists()
    assert not (out / "summary.json").exists()
    started = time.perf_counter()
    union, rows, meta, whitebox_audit = build_partition_design("fit")
    upstream_oof_audit = assert_fit_upstream_source_group_oof(meta)
    labels = np.asarray([window["label"] for window in meta["windows"]], dtype=np.int8)
    groups = np.asarray([window["group_id"] for window in meta["windows"]])
    answer_ids = np.asarray([window["answer_id"] for window in meta["windows"]])
    assert len(set(groups)) == EXPECTED_GROUPS["fit"]

    feature_path = out / "fit_window_features.npz"
    atomic_npz(
        feature_path,
        union=union,
        feature_names=np.asarray(feature_v2.UNION_NAMES),
        window_ids=np.asarray([window["window_id"] for window in meta["windows"]]),
        response_ids=np.asarray([window["response_id"] for window in meta["windows"]]),
        group_ids=groups,
        answer_ids=answer_ids,
    )
    oof, candidate_names, table, fold_log = crossfit_grid(union, meta)
    selected_index = max(
        range(len(table)),
        key=lambda index: candidate_selection_key(table[index], index),
    )
    selected = table[selected_index]
    columns = feature_v2.variant_indices(selected["variant"])
    active = np.arange(len(labels))
    base, loss, final_mass = window_hierarchy_weights(
        groups, answer_ids, labels, active
    )
    scaler, model = fit_scaled_lr(
        union[:, columns], labels, active, base, loss, selected["C"]
    )
    model_path = out / "model.pkl"
    atomic_pickle(model_path, {
        "version": VERSION,
        "variant": selected["variant"],
        "C": selected["C"],
        "columns": columns,
        "feature_names": [feature_v2.UNION_NAMES[index] for index in columns],
        "scaler": scaler,
        "model": model,
    })
    scores_path = out / "fit_scores.npz"
    atomic_npz(
        scores_path,
        candidate_names=np.asarray(candidate_names),
        candidate_scores=oof,
        selected_scores=oof[selected_index],
        labels=labels,
        selected_index=np.asarray(selected_index, dtype=np.int32),
        answer_scores=answer_scores(meta, oof[selected_index]),
    )
    assert baseline_snapshot() == read_json(out / "source_snapshot.json")[
        "formal_baseline_files_sha256"
    ]
    fit_complete = {
        "status": "model_features_C_and_thresholds_frozen_before_calibration",
        "selected": {
            key: selected[key]
            for key in (
                "candidate", "variant", "blocks", "width", "C",
                "fit_OOF_thresholds", "fit_OOF",
            )
        },
        "selection_rule": protocol()["fit"]["selection"],
        "candidate_table": [
            {key: record[key] for key in (
                "candidate", "variant", "blocks", "width", "C",
                "fit_OOF_thresholds", "fit_OOF",
            )}
            for record in table
        ],
        "crossfit": fold_log,
        "full_fit": {
            "windows": len(labels),
            "positive_windows": int(labels.sum()),
            "answers": len(meta["answers"]),
            "groups": len(set(groups)),
            "final_loss_mass_by_class": final_mass.tolist(),
            "n_iter": int(model.n_iter_[0]),
        },
        "upstream_OOF_audit": upstream_oof_audit,
        "whitebox_reconstruction_audit": whitebox_audit,
        "artifacts_sha256": {
            "model.pkl": sha(model_path),
            "fit_scores.npz": sha(scores_path),
            "fit_window_features.npz": sha(feature_path),
            "protocol.json": sha(out / "protocol.json"),
            "source_snapshot.json": sha(out / "source_snapshot.json"),
            "runner": sha(Path(__file__)),
            "feature_module": sha(HERE / "semantic_window_v2_features.py"),
        },
        "fit_labels_opened": True,
        "calibration_labels_opened": False,
        "calibration_used_for_model_feature_C_or_threshold_selection": False,
        "conflict_head_enabled": False,
        "add_gate_enabled": False,
        "claim_score_max_projection_used": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(out / "fit_complete.json", fit_complete)
    print(
        "SEMANTIC_WINDOW_V2_FIT_FROZEN",
        selected["candidate"],
        selected["fit_OOF"]["windows"]["f1"],
        selected["fit_OOF"]["answers"]["f1"],
        flush=True,
    )


def format_metric_row(scope: str, metric: dict, threshold: float) -> str:
    return (
        f"| {scope} | {metric['n']} | {metric['tp']}/{metric['fp']}/{metric['fn']}/{metric['tn']} | "
        f"{metric['precision']:.6f} | {metric['recall']:.6f} | {metric['f1']:.6f} | "
        f"{metric['auroc']:.6f} | {metric['average_precision']:.6f} | {threshold:.9f} |"
    )


def render_report(fit_done: dict, summary: dict) -> str:
    selected = fit_done["selected"]
    strict = summary["calibration_strict_fit_thresholds"]
    diagnostic = summary["calibration_F1Opt_diagnostic"]
    fit_metrics = selected["fit_OOF"]
    fit_thresholds = selected["fit_OOF_thresholds"]
    cal_thresholds = summary["calibration_F1Opt_thresholds"]
    lines = [
        "# Semantic window v2 — native-634 development pilot",
        "",
        "v2 修复了 v1 的主要评分路径：直接在项目 4-BPE 窗口训练 any-error 读出，停用 conflict-max 和 add-gate，也不再把 claim 风险分数 max 铺到整条 claim。claim 特征只按窗口内真实 lexical token 归属做加权平均。",
        "",
        f"fit OOF 选择 `{selected['candidate']}`，宽度 {selected['width']}。四个特征变体、三个 C、两级阈值和选择规则均在读取 calibration 前冻结。",
        "",
        "## Fit OOF 候选",
        "",
        "| 候选 | 维度 | 窗口F1 | 窗口AP | 整答F1 | 整答AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in fit_done["candidate_table"]:
        wm = record["fit_OOF"]["windows"]
        am = record["fit_OOF"]["answers"]
        lines.append(
            f"| `{record['candidate']}` | {record['width']} | {wm['f1']:.6f} | "
            f"{wm['average_precision']:.6f} | {am['f1']:.6f} | {am['average_precision']:.6f} |"
        )
    lines += [
        "",
        "选择键依次为两级 F1 的较小值、窗口 F1、整答 F1、两级 AP、更少维、更小 C、预注册顺序。calibration 未参与选择。",
        "",
        "## 选中模型完整指标",
        "",
        "| 口径 | n | TP/FP/FN/TN | P | R | F1 | AUROC | AP | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        format_metric_row("fit OOF 窗口", fit_metrics["windows"], fit_thresholds["window"]["threshold"]),
        format_metric_row("fit OOF 整答", fit_metrics["answers"], fit_thresholds["answer"]["threshold"]),
        format_metric_row("cal 严格窗口", strict["windows"], fit_thresholds["window"]["threshold"]),
        format_metric_row("cal 严格整答", strict["answers"], fit_thresholds["answer"]["threshold"]),
        format_metric_row("cal F1-opt 窗口诊断", diagnostic["windows"], cal_thresholds["window"]["threshold"]),
        format_metric_row("cal F1-opt 整答诊断", diagnostic["answers"], cal_thresholds["answer"]["threshold"]),
        "",
        "## 与冻结参照比较",
        "",
        "| 方法/口径 | 窗口F1 | 整答F1 |",
        "|---|---:|---:|",
        f"| v2 严格 fit 阈值 | {strict['windows']['f1']:.6f} | {strict['answers']['f1']:.6f} |",
        f"| v2 cal F1-opt 诊断 | {diagnostic['windows']['f1']:.6f} | {diagnostic['answers']['f1']:.6f} |",
        f"| Lookback 统一 cal F1-opt | {LOOKBACK_REFERENCE['windows']:.6f} | {LOOKBACK_REFERENCE['answers']:.6f} |",
        f"| 历史 incumbent（cal 选型） | {INCUMBENT_REFERENCE['windows']:.6f} | {INCUMBENT_REFERENCE['answers']:.6f} |",
        "",
        f"v2 严格结果相对 Lookback 为窗口 {strict['windows']['f1'] - LOOKBACK_REFERENCE['windows']:+.6f}、整答 {strict['answers']['f1'] - LOOKBACK_REFERENCE['answers']:+.6f}；相对 incumbent 为窗口 {strict['windows']['f1'] - INCUMBENT_REFERENCE['windows']:+.6f}、整答 {strict['answers']['f1'] - INCUMBENT_REFERENCE['answers']:+.6f}。共同 cal-F1Opt 诊断相对 Lookback 为窗口 {diagnostic['windows']['f1'] - LOOKBACK_REFERENCE['windows']:+.6f}、整答 {diagnostic['answers']['f1'] - LOOKBACK_REFERENCE['answers']:+.6f}。",
        "",
        "## 边界与审计",
        "",
        "- fit 的五折均按 source-connected group 隔离；每个 fit 窗口恰有一个外折预测。",
        "- whitebox 和可选 local 的 fit Lookback/large 输入来自既有上游 group-OOF；本层没有重训上游。它们不是为本层五折重新嵌套生成，故 OOF 身份限于冻结上游加本层来源组留出。",
        "- calibration 只在完整 fit freeze 后执行一次；cal-F1Opt 只作诊断。",
        "- 正式 baseline 文件的前后哈希一致，official test 未打开，GPU 未使用。",
        "- 本轮归因训练范围只有 native 634 答；正式 Lookback 已用 3680 fit。分数可按相同 cal 口径点对点查看，但不能解释为同训练预算胜出。",
        "- calibration 已被项目反复用于开发；本报告不是独立测试结论。",
        "",
        "可复跑入口：`python src/run_semantic_window_v2.py run-all --output-dir results/<fresh-directory>`。",
    ]
    return "\n".join(lines) + "\n"


def evaluate(out: Path) -> None:
    check_prepared(out)
    fit_done = read_json(out / "fit_complete.json")
    assert fit_done["status"] == "model_features_C_and_thresholds_frozen_before_calibration"
    assert not (out / "summary.json").exists(), "Calibration evaluation is single-use per output directory"
    for name in ("model.pkl", "fit_scores.npz", "fit_window_features.npz", "protocol.json", "source_snapshot.json"):
        assert sha(out / name) == fit_done["artifacts_sha256"][name]
    assert sha(Path(__file__)) == fit_done["artifacts_sha256"]["runner"]
    assert sha(HERE / "semantic_window_v2_features.py") == fit_done["artifacts_sha256"]["feature_module"]
    started = time.perf_counter()

    union, _, meta, whitebox_audit = build_partition_design("calibration")
    payload = pickle.loads((out / "model.pkl").read_bytes())
    assert payload["version"] == VERSION
    assert payload["variant"] == fit_done["selected"]["variant"]
    assert payload["C"] == fit_done["selected"]["C"]
    columns = np.asarray(payload["columns"], dtype=np.int32)
    assert np.array_equal(columns, feature_v2.variant_indices(payload["variant"]))
    x = union[:, columns]
    scores = payload["model"].predict_proba(payload["scaler"].transform(x))[:, 1]
    assert scores.shape == (EXPECTED_WINDOWS["calibration"],) and np.isfinite(scores).all()

    # Every score is fixed before these label-bearing metrics are computed.
    fit_thresholds = fit_done["selected"]["fit_OOF_thresholds"]
    strict = metric_pair(meta, scores, fit_thresholds)
    cal_thresholds = choose_thresholds(meta, scores)
    diagnostic = metric_pair(meta, scores, cal_thresholds)
    feature_path = out / "calibration_window_features.npz"
    atomic_npz(
        feature_path,
        union=union,
        feature_names=np.asarray(feature_v2.UNION_NAMES),
        window_ids=np.asarray([window["window_id"] for window in meta["windows"]]),
        response_ids=np.asarray([window["response_id"] for window in meta["windows"]]),
        group_ids=np.asarray([window["group_id"] for window in meta["windows"]]),
        answer_ids=np.asarray([window["answer_id"] for window in meta["windows"]]),
    )
    scores_path = out / "calibration_scores.npz"
    atomic_npz(
        scores_path,
        window_scores=scores,
        answer_scores=answer_scores(meta, scores),
        window_ids=np.asarray([window["window_id"] for window in meta["windows"]]),
        response_ids=np.asarray([window["response_id"] for window in meta["windows"]]),
    )
    before = read_json(out / "source_snapshot.json")["formal_baseline_files_sha256"]
    after = baseline_snapshot()
    assert before == after
    summary = {
        "status": "development_calibration_evaluated_once",
        "selected_candidate": fit_done["selected"]["candidate"],
        "fit_OOF": fit_done["selected"]["fit_OOF"],
        "fit_OOF_thresholds": fit_thresholds,
        "calibration_strict_fit_thresholds": strict,
        "calibration_F1Opt_diagnostic": diagnostic,
        "calibration_F1Opt_thresholds": cal_thresholds,
        "references": {
            "Lookback_common_cal_F1Opt": LOOKBACK_REFERENCE,
            "historical_incumbent_cal_selected": INCUMBENT_REFERENCE,
        },
        "strict_deltas": {
            "versus_Lookback": {
                "windows": strict["windows"]["f1"] - LOOKBACK_REFERENCE["windows"],
                "answers": strict["answers"]["f1"] - LOOKBACK_REFERENCE["answers"],
            },
            "versus_historical_incumbent": {
                "windows": strict["windows"]["f1"] - INCUMBENT_REFERENCE["windows"],
                "answers": strict["answers"]["f1"] - INCUMBENT_REFERENCE["answers"],
            },
        },
        "calibration_evaluations": 1,
        "calibration_whitebox_reconstruction_audit": whitebox_audit,
        "calibration_used_for_model_feature_C_or_fit_threshold_selection": False,
        "fit_complete_sha256": sha(out / "fit_complete.json"),
        "artifacts_sha256": {
            "calibration_scores.npz": sha(scores_path),
            "calibration_window_features.npz": sha(feature_path),
        },
        "formal_baseline_files_unchanged": before == after,
        "calibration_labels_opened": True,
        "conflict_head_enabled": False,
        "add_gate_enabled": False,
        "claim_score_max_projection_used": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(out / "summary.json", summary)
    report_path = out / "REPORT.md"
    report_path.write_text(render_report(fit_done, summary), encoding="utf-8")
    atomic_json(out / "complete.json", {
        "status": "complete_native634_development_only",
        "fit_complete_sha256": sha(out / "fit_complete.json"),
        "summary_sha256": sha(out / "summary.json"),
        "report_sha256": sha(report_path),
        "calibration_scores_sha256": sha(scores_path),
        "calibration_evaluations": 1,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    })
    print(
        "SEMANTIC_WINDOW_V2_EVALUATED",
        strict["windows"]["f1"],
        strict["answers"]["f1"],
        diagnostic["windows"]["f1"],
        diagnostic["answers"]["f1"],
        flush=True,
    )


def status(out: Path) -> None:
    print(json.dumps({
        "output": str(out.resolve()),
        "prepared": (out / "PREPARATION.json").is_file(),
        "fit_frozen": (out / "fit_complete.json").is_file(),
        "calibration_evaluated": (out / "summary.json").is_file(),
        "complete": (out / "complete.json").is_file(),
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "run-all", "status"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(out)
    elif args.stage == "fit":
        fit(out)
    elif args.stage == "evaluate":
        evaluate(out)
    elif args.stage == "run-all":
        prepare(out)
        fit(out)
        evaluate(out)
    else:
        status(out)


if __name__ == "__main__":
    main()
