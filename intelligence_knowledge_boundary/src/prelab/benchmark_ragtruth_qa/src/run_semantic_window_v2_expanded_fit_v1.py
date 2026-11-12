"""Frozen expanded-window-v2 experiment.

The feature family (whitebox_geometry) and C=.1 are inherited verbatim from
the native-634 fit-only selection.  This runner first reconstructs genuinely
source-group-OOF Lookback/large signals for all 3,680 fit answers, adds the
label-free generation NLL, and directly supervises the unchanged eligible
four-BPE windows.  Calibration is a separate, single-use command and the
official test has no path in this module.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402
import run_semantic_source_attribution_combined_fit_v1 as combined  # noqa: E402
import semantic_window_v2_features as feature_v2  # noqa: E402


VERSION = "semantic-window-v2-expanded-fit-v1"
OUT = ROOT / "results/semantic_window_v2_expanded_fit_v1"
UPSTREAM = ROOT / "results/group_crossfit_lb_large_v1"
EXPANDED_MATRICES = ROOT / "fit_expansion/llama_baselines_v1/matrices"
EXPANDED_MATRIX_PREP = ROOT / "fit_expansion/llama_baselines_v1/preparation_complete.json"
EXPANDED_REPLAY_PROTOCOL = ROOT / "fit_expansion/llama_features_v3/protocol.json"
LB_MATRIX = EXPANDED_MATRICES / "lb_prefix_pre_header.npy"
BASE_MATRIX = EXPANDED_MATRICES / "base.npy"
NATIVE_LB_MATRIX = ROOT / "results/lookback_controls_v2/matrices/lb_prefix_pre_header.npy"
NATIVE_BASE_MATRIX = ROOT / "results/development_v1/matrices/base.npy"
NATIVE_LOCAL = UPSTREAM / "oof_input_scores.npy"
NATIVE_V2 = ROOT / "results/semantic_window_v2"
NATIVE_V2_FIT = NATIVE_V2 / "fit_complete.json"
NATIVE_V2_FEATURES = NATIVE_V2 / "fit_window_features.npz"

FIT_ANSWERS = 3680
FIT_GROUPS = 615
FIT_CLAIMS = 34919
FIT_WINDOWS = 653979
FIT_POSITIVE_WINDOWS = 58433
EXCLUDED_WINDOWS = 692
NATIVE_ANSWERS = 634
NATIVE_CLAIMS = 9055
NATIVE_WINDOWS = 168123
CAL_ANSWERS = 159
CAL_WINDOWS = 42241
UPSTREAM_FOLDS = 3
DOWNSTREAM_FOLDS = 5
VARIANT = "whitebox_geometry"
C_VALUE = 0.1
SEED = 20260913
THREADS = 4
LB_BLOCK = 4096


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


def atomic_npy(path: Path, value: np.ndarray) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    pending.replace(path)


def atomic_npz(path: Path, **values) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **values)
    pending.replace(path)


def protocol() -> dict:
    return {
        "version": VERSION,
        "selection_freeze": {
            "source": "semantic_window_v2 native-634 fit-only winner",
            "variant": VARIANT,
            "blocks": ["whitebox", "geometry"],
            "width": 26,
            "C": C_VALUE,
            "reselection_on_expanded_fit": False,
        },
        "cohort": {
            "fit_answers": FIT_ANSWERS,
            "fit_source_connected_groups": FIT_GROUPS,
            "fit_claims": FIT_CLAIMS,
            "fit_windows": FIT_WINDOWS,
            "positive_fit_windows": FIT_POSITIVE_WINDOWS,
            "excluded_nonlexical_windows": EXCLUDED_WINDOWS,
            "calibration_answers": CAL_ANSWERS,
            "calibration_windows": CAL_WINDOWS,
        },
        "whitebox": {
            "columns": list(feature_v2.WHITEBOX_NAMES),
            "Lookback": "Frozen upstream three-fold source-group-OOF lb_prefix_pre_header model applied only to answers whose source group was held from that model.",
            "large": "Frozen upstream three-fold source-group-OOF ModernBERT token probabilities; each source group predicted exactly once by a model that excluded it.",
            "generation_nll": "Label-blind teacher-forced token NLL, pooled by the frozen four-BPE window construction.",
            "claim_pooling": "For each claim and each of the three window signals: max, mean, min, population standard deviation.",
            "window_pooling": "Claim summaries averaged by exact lexical-token ownership inside each window.",
            "in_sample_supervised_columns_allowed": False,
        },
        "geometry": "The exact 14 fixed semantic_window_v2 window geometry columns.",
        "fit": {
            "folds": DOWNSTREAM_FOLDS,
            "split": "sklearn GroupKFold over locked source-connected group IDs",
            "model": "StandardScaler plus liblinear L2 LogisticRegression",
            "weights": "Exact semantic_window_v2 equal group/answer/window base mass, then fit-only binary class balance.",
            "thresholds": "Separate F1-optimal window and answer thresholds from five-fold expanded-fit OOF scores, frozen before calibration.",
            "answer_score": "Maximum eligible-window probability per answer.",
        },
        "calibration": "Separate single-use evaluation after full model and both thresholds are frozen; no calibration threshold fitting.",
        "limitations": [
            "The upstream source-group OOF folds are frozen three-fold predictions and are not retrained inside each downstream five-fold split.",
            "The 3,680 answers still cover only 615 source-connected groups.",
            "Calibration has been used repeatedly elsewhere in the project and is development evidence, not a fresh final test.",
        ],
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def assert_native_selection() -> dict:
    fit = read(NATIVE_V2_FIT)
    selected = fit["selected"]
    assert fit["calibration_labels_opened"] is False
    assert selected["candidate"] == "whitebox_geometry__C0.1"
    assert selected["variant"] == VARIANT and selected["C"] == C_VALUE
    assert selected["width"] == 26 and selected["blocks"] == ["whitebox", "geometry"]
    return {
        "native_fit_complete_sha256": sha(NATIVE_V2_FIT),
        "selected_candidate": selected["candidate"],
        "native_selection_used_calibration": False,
    }


def baseline_snapshot() -> dict[str, str]:
    # Imported only for its fixed, already-declared baseline file list.
    import run_semantic_window_v2 as native_v2
    return native_v2.baseline_snapshot()


def audit_upstream(axes: dict, projection: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    upstream_protocol = read(UPSTREAM / "protocol.json")
    upstream_complete = read(UPSTREAM / "complete.json")
    upstream_audit = read(UPSTREAM / "INDEPENDENT_AUDIT.json")
    folds = read(UPSTREAM / "folds.json")
    assert upstream_protocol["folds"] == UPSTREAM_FOLDS and len(folds) == UPSTREAM_FOLDS
    assert upstream_complete["official_test_opened"] is False
    assert upstream_audit["status"] == "passed_independent_arithmetic_and_wiring_replay"
    assert upstream_audit["fit_hold_groups_disjoint"] is True
    assert upstream_audit["fold_answer_coverage_exactly_once"] is True
    assert read(EXPANDED_REPLAY_PROTOCOL)["labels_interface"].startswith("Extractor reads only frozen new_token_plans")

    response_groups = np.asarray(axes["response_group_ids"], dtype=str)
    response_ids = np.asarray(axes["response_ids"], dtype=str)
    token_ptr = projection["response_token_indptr"]
    all_groups = set(response_groups.tolist())
    held_union: set[str] = set()
    answer_coverage = np.zeros(FIT_ANSWERS, dtype=np.int8)
    records = []
    for fold in folds:
        number = int(fold["fold"])
        folder = UPSTREAM / f"fold_{number}"
        train_groups = set(map(str, fold["fit_groups"]))
        held_groups = set(map(str, fold["hold_groups"]))
        assert len(train_groups) == 410 and len(held_groups) == 205
        assert not train_groups & held_groups and train_groups | held_groups == all_groups
        assert not held_union & held_groups
        held_union |= held_groups
        held_answers = np.flatnonzero(np.isin(response_groups, list(held_groups)))
        assert np.array_equal(held_answers, np.asarray(fold["hold_answer_indices"], dtype=np.int64))
        assert np.all(answer_coverage[held_answers] == 0)
        answer_coverage[held_answers] += 1

        for kind in ("lb", "large"):
            done = read(folder / f"{kind}_complete.json")
            for name, expected in done["files_sha256"].items():
                assert sha(folder / name) == expected
            assert done["official_test_opened"] is False
            if kind == "large":
                assert done["no_hold_gold_or_calibration_used_for_training_or_epoch_selection"] is True

        lb = pickle.loads((folder / "lb.pkl").read_bytes())
        assert set(map(str, lb["fit_groups"])) == train_groups
        assert set(map(str, lb["hold_groups"])) == held_groups
        assert np.array_equal(lb["fit_indices"], np.asarray(fold["fit_window_indices"]))
        assert np.array_equal(lb["hold_indices"], np.asarray(fold["hold_window_indices"]))
        assert lb["C"] == 0.0001

        with np.load(folder / "large_hold_token_probabilities.npz", allow_pickle=False) as large:
            expected_ids = set(response_ids[held_answers].tolist())
            assert set(large.files) == expected_ids
            token_rows = 0
            for answer_index in held_answers:
                rid = response_ids[answer_index]
                expected = int(token_ptr[answer_index + 1] - token_ptr[answer_index])
                assert large[rid].shape == (expected,) and np.isfinite(large[rid]).all()
                token_rows += expected
        records.append({
            "fold": number,
            "fit_groups": len(train_groups),
            "held_groups": len(held_groups),
            "held_answers": len(held_answers),
            "held_tokens": token_rows,
            "lookback_training_excludes_all_held_groups": True,
            "large_training_excludes_all_held_groups": True,
        })
    assert held_union == all_groups and np.all(answer_coverage == 1)

    lb_matrix = np.load(LB_MATRIX, mmap_mode="r")
    base_matrix = np.load(BASE_MATRIX, mmap_mode="r")
    old_lb = np.load(NATIVE_LB_MATRIX, mmap_mode="r")
    old_base = np.load(NATIVE_BASE_MATRIX, mmap_mode="r")
    assert lb_matrix.shape == (FIT_WINDOWS + CAL_WINDOWS, 1024)
    assert base_matrix.shape == (FIT_WINDOWS + CAL_WINDOWS, 1025)
    # Only fit prefixes are touched before the calibration command.
    assert np.array_equal(lb_matrix[:NATIVE_WINDOWS], old_lb[:NATIVE_WINDOWS])
    assert np.array_equal(base_matrix[:NATIVE_WINDOWS], old_base[:NATIVE_WINDOWS])
    matrix_prep = read(EXPANDED_MATRIX_PREP)
    assert matrix_prep["old793_all_windows_exact"] is True
    return folds, {
        "status": "passed_all_supervised_whitebox_inputs_are_source_group_OOF",
        "upstream_folds": UPSTREAM_FOLDS,
        "source_groups": len(held_union),
        "answer_coverage_exactly_once": True,
        "lookback_models_fit_only_on_complement_groups": True,
        "large_models_fit_only_on_complement_groups": True,
        "generation_nll_label_blind": True,
        "expanded_raw_lookback_definition_matches_native_fit_exactly": True,
        "expanded_raw_nll_definition_matches_native_fit_exactly": True,
        "in_sample_supervised_columns": 0,
        "folds": records,
        "calibration_opened": False,
        "official_test_opened": False,
    }


def held_window_indices(projection: dict[str, np.ndarray], held_answers: np.ndarray) -> np.ndarray:
    ptr = projection["response_window_indptr"]
    return np.concatenate([
        np.arange(int(ptr[index]), int(ptr[index + 1]), dtype=np.int64)
        for index in held_answers
    ])


def reconstruct_fit_local_signals(axes: dict, projection: dict[str, np.ndarray], folds: list[dict]) -> tuple[np.ndarray, dict]:
    response_ids = np.asarray(axes["response_ids"], dtype=str)
    response_groups = np.asarray(axes["response_group_ids"], dtype=str)
    token_ptr = projection["response_token_indptr"]
    window_tokens = projection["window_token_indices"]
    lexical_mask = projection["token_lexical_mask"].astype(bool)
    lb_matrix = np.load(LB_MATRIX, mmap_mode="r")
    base_matrix = np.load(BASE_MATRIX, mmap_mode="r")

    signals = np.full((FIT_WINDOWS, 3), np.nan, dtype=np.float32)
    token_large = np.full(len(projection["token_ids"]), np.nan, dtype=np.float32)
    window_coverage = np.zeros(FIT_WINDOWS, dtype=np.int8)
    answer_coverage = np.zeros(FIT_ANSWERS, dtype=np.int8)
    fold_rows = []
    for fold in folds:
        number = int(fold["fold"])
        held_groups = set(map(str, fold["hold_groups"]))
        held_answers = np.flatnonzero(np.isin(response_groups, list(held_groups)))
        held_windows = held_window_indices(projection, held_answers)
        assert np.all(window_coverage[held_windows] == 0)
        window_coverage[held_windows] += 1
        answer_coverage[held_answers] += 1

        folder = UPSTREAM / f"fold_{number}"
        lb = pickle.loads((folder / "lb.pkl").read_bytes())
        for start in range(0, len(held_windows), LB_BLOCK):
            indices = held_windows[start:start + LB_BLOCK]
            design = lb["scaler"].transform(np.asarray(lb_matrix[indices])).astype(np.float32)
            signals[indices, 0] = lb["model"].predict_proba(design)[:, 1].astype(np.float32)
        with np.load(folder / "large_hold_token_probabilities.npz", allow_pickle=False) as large:
            for answer_index in held_answers:
                left, right = map(int, token_ptr[answer_index:answer_index + 2])
                probability = np.asarray(large[response_ids[answer_index]], dtype=np.float32)
                assert probability.shape == (right - left,)
                token_large[left:right] = probability
        fold_rows.append({"fold": number, "held_answers": len(held_answers), "held_windows": len(held_windows)})
        print("EXPANDED_WINDOW_V2_UPSTREAM_OOF", number + 1, UPSTREAM_FOLDS, len(held_windows), flush=True)
    assert np.all(answer_coverage == 1) and np.all(window_coverage == 1)
    assert np.isfinite(signals[:, 0]).all() and np.isfinite(token_large).all()

    safe = np.where(window_tokens >= 0, window_tokens, 0)
    lexical = (window_tokens >= 0) & lexical_mask[safe]
    values = np.where(lexical, token_large[safe], -np.inf)
    signals[:, 1] = values.max(axis=1)
    signals[:, 2] = np.asarray(base_matrix[:FIT_WINDOWS, -1], dtype=np.float32)
    assert np.isfinite(signals).all()

    native_expected = np.column_stack((
        np.asarray(np.load(NATIVE_LOCAL, mmap_mode="r")[:NATIVE_WINDOWS], dtype=np.float32),
        np.asarray(np.load(NATIVE_BASE_MATRIX, mmap_mode="r")[:NATIVE_WINDOWS, -1], dtype=np.float32),
    ))
    error = np.abs(signals[:NATIVE_WINDOWS].astype(np.float64) - native_expected.astype(np.float64))
    assert np.array_equal(signals[:NATIVE_WINDOWS], native_expected), float(error.max())
    audit = {
        "fit_windows": FIT_WINDOWS,
        "window_source_group_OOF_coverage_exactly_once": True,
        "response_source_group_OOF_coverage_exactly_once": True,
        "native_168123_three_signal_replay_exact": True,
        "native_max_abs_error": float(error.max()),
        "in_sample_supervised_columns": 0,
        "columns": ["oof_lookback", "oof_large", "generation_nll"],
        "folds": fold_rows,
    }
    return signals, audit


def claim_whitebox(signals: np.ndarray, projection: dict[str, np.ndarray]) -> np.ndarray:
    owners = projection["window_claim_indices"]
    claim_ids, window_ids = [], []
    window_axis = np.arange(FIT_WINDOWS, dtype=np.int64)
    for column in range(owners.shape[1]):
        valid = owners[:, column] >= 0
        claim_ids.append(owners[valid, column].astype(np.int64))
        window_ids.append(window_axis[valid])
    claim_ids = np.concatenate(claim_ids)
    window_ids = np.concatenate(window_ids)
    order = np.lexsort((window_ids, claim_ids))
    claim_ids, window_ids = claim_ids[order], window_ids[order]
    counts = np.bincount(claim_ids, minlength=FIT_CLAIMS)
    assert len(counts) == FIT_CLAIMS and np.all(counts > 0)
    indptr = np.r_[0, np.cumsum(counts)]
    output = np.empty((FIT_CLAIMS, len(feature_v2.WHITEBOX_NAMES)), dtype=np.float32)
    for claim in range(FIT_CLAIMS):
        one = signals[window_ids[indptr[claim]:indptr[claim + 1]]]
        output[claim] = np.concatenate((one.max(0), one.mean(0), one.min(0), one.std(0)))
    return output


def build_geometry_and_design(claim_features: np.ndarray, projection: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    window_tokens = projection["window_token_indices"]
    valid = window_tokens >= 0
    safe_tokens = np.where(valid, window_tokens, 0)
    lexical = valid & projection["token_lexical_mask"][safe_tokens].astype(bool)
    token_owners = np.where(lexical, projection["token_claim_index"][safe_tokens], -1)
    assert np.all(np.any(lexical, axis=1)) and np.all(token_owners[lexical] >= 0)
    owner_safe = np.where(lexical, token_owners, 0)
    lexical_count = lexical.sum(axis=1)

    # Reproduce semantic_window_v2's normalized, sorted claim-count average.
    whitebox = np.empty((FIT_WINDOWS, len(feature_v2.WHITEBOX_NAMES)), dtype=np.float32)
    for index in range(FIT_WINDOWS):
        owners, counts = np.unique(token_owners[index, lexical[index]], return_counts=True)
        weights = counts.astype(np.float64)
        weights /= weights.sum()
        whitebox[index] = np.average(claim_features[owners], axis=0, weights=weights)
        if (index + 1) % 100000 == 0:
            print("EXPANDED_WINDOW_V2_POOL", index + 1, FIT_WINDOWS, flush=True)

    claim_lengths = np.diff(projection["claim_token_indptr"]).astype(np.int32)
    assert claim_lengths.shape == (FIT_CLAIMS,) and np.all(claim_lengths > 0)
    token_rank = np.full(len(projection["token_ids"]), -1, dtype=np.int32)
    claim_indices = projection["claim_token_indices"]
    claim_ptr = projection["claim_token_indptr"]
    for claim in range(FIT_CLAIMS):
        left, right = map(int, claim_ptr[claim:claim + 2])
        token_rank[claim_indices[left:right]] = np.arange(right - left, dtype=np.int32)
    assert np.all(token_rank[projection["token_lexical_mask"].astype(bool)] >= 0)

    lengths = claim_lengths[owner_safe]
    ranks = token_rank[safe_tokens]
    owner_counts = np.zeros_like(owner_safe, dtype=np.int32)
    for column in range(4):
        owner_counts[:, column] = np.sum(
            lexical & (owner_safe == owner_safe[:, column, None]), axis=1)
    maximum_count = np.max(np.where(lexical, owner_counts, -1), axis=1)
    candidates = np.where(
        lexical & (owner_counts == maximum_count[:, None]),
        owner_safe, np.iinfo(np.int32).max,
    )
    dominant = candidates.min(axis=1)
    dominant_length = claim_lengths[dominant]
    first = np.zeros_like(lexical)
    for column in range(4):
        first[:, column] = lexical[:, column] & ~np.any(
            lexical[:, :column] & (owner_safe[:, :column] == owner_safe[:, column, None]), axis=1)
    distinct = first.sum(axis=1)
    denominator = lexical_count.astype(np.float64)
    position = np.where(lengths == 1, .5, ranks / np.maximum(lengths - 1, 1))
    geometry = np.column_stack((
        lexical_count / 4,
        np.zeros(FIT_WINDOWS),
        np.log1p(distinct),
        (distinct > 1).astype(float),
        maximum_count / denominator,
        np.sum(np.where(lexical, owner_counts / np.maximum(lengths, 1), 0), axis=1) / denominator,
        maximum_count / dominant_length,
        np.sum(np.where(lexical, position, 0), axis=1) / denominator,
        np.sum(np.where(lexical, np.abs(position - .5) * 2, 0), axis=1) / denominator,
        np.sum(lexical & (ranks == 0), axis=1) / denominator,
        np.sum(lexical & (ranks == lengths - 1), axis=1) / denominator,
        np.sum(lexical & (ranks < 2), axis=1) / denominator,
        np.sum(lexical & (ranks >= lengths - 2), axis=1) / denominator,
        np.log1p(dominant_length),
    )).astype(np.float32)
    assert geometry.shape == (FIT_WINDOWS, len(feature_v2.GEOMETRY_NAMES))
    design = np.column_stack((whitebox, geometry)).astype(np.float32)
    assert design.shape == (FIT_WINDOWS, 26) and np.isfinite(design).all()

    with np.load(NATIVE_V2_FEATURES, allow_pickle=False) as native:
        columns = feature_v2.variant_indices(VARIANT)
        expected = native["union"][:, columns].astype(np.float32)
    difference = np.abs(design[:NATIVE_WINDOWS].astype(np.float64) - expected.astype(np.float64))
    assert np.array_equal(design[:NATIVE_WINDOWS], expected), float(difference.max())
    return design, {
        "native_windows_replayed": NATIVE_WINDOWS,
        "native_whitebox_geometry_exact": True,
        "native_max_abs_error": float(difference.max()),
        "claim_features": list(feature_v2.WHITEBOX_NAMES),
        "geometry_features": list(feature_v2.GEOMETRY_NAMES),
    }


def answer_scores(scores: np.ndarray, response_window_indptr: np.ndarray) -> np.ndarray:
    result = np.empty(len(response_window_indptr) - 1, dtype=np.float64)
    for answer in range(len(result)):
        left, right = map(int, response_window_indptr[answer:answer + 2])
        result[answer] = np.max(scores[left:right])
    return result


def hierarchy_weights(groups: np.ndarray, answers: np.ndarray, labels: np.ndarray, active: np.ndarray):
    tree: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in active:
        tree[int(groups[index])][int(answers[index])].append(int(index))
    base = np.zeros(len(labels), dtype=np.float64)
    for group_answers in tree.values():
        for indices in group_answers.values():
            base[indices] = 1.0 / (len(group_answers) * len(indices))
    mask = base > 0
    base[mask] /= base[mask].mean()
    class_mass = np.bincount(labels[mask], weights=base[mask], minlength=2)
    assert np.all(class_mass > 0)
    loss = base.copy()
    loss[mask] *= (class_mass.sum() / (2 * class_mass))[labels[mask]]
    loss[mask] *= mask.sum() / loss[mask].sum()
    final_mass = np.bincount(labels[mask], weights=loss[mask], minlength=2)
    # With >500k rows, float64 bincount accumulation differs by about 2e-7
    # although the analytical class masses are identical.
    assert np.isclose(final_mass[0], final_mass[1], rtol=0, atol=1e-6)
    return base, loss, final_mass


def make_model():
    return LogisticRegression(
        C=C_VALUE, solver="liblinear", penalty="l2", max_iter=3000,
        random_state=SEED,
    )


def prepare() -> None:
    assert not (OUT / "fit_complete.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / "protocol.json", protocol())
    native_selection = assert_native_selection()
    axes, projection = combined.load_projection()
    assert len(axes["response_ids"]) == FIT_ANSWERS
    assert len(set(axes["response_group_ids"])) == FIT_GROUPS
    assert projection["window_labels"].shape == (FIT_WINDOWS,)
    assert int(projection["window_labels"].sum()) == FIT_POSITIVE_WINDOWS
    assert projection["claim_labels"].shape == (FIT_CLAIMS,)
    assert projection["response_window_indptr"][NATIVE_ANSWERS] == NATIVE_WINDOWS
    assert read(combined.OUT / "PREPARATION.json")["excluded_windows"] == EXCLUDED_WINDOWS

    folds, provenance = audit_upstream(axes, projection)
    provenance.update(native_selection)
    frozen_json(OUT / "PROVENANCE_AUDIT.json", provenance)
    signals, signal_audit = reconstruct_fit_local_signals(axes, projection, folds)
    claims = claim_whitebox(signals, projection)
    design, replay = build_geometry_and_design(claims, projection)
    atomic_npy(OUT / "fit_local_signals.npy", signals)
    atomic_npy(OUT / "fit_claim_whitebox.npy", claims)
    atomic_npy(OUT / "fit_whitebox_geometry.npy", design)

    sources = {
        "runner_sha256": sha(Path(__file__)),
        "feature_module_sha256": sha(HERE / "semantic_window_v2_features.py"),
        "combined_projection_sha256": sha(combined.OUT / "projection.npz"),
        "combined_axes_sha256": sha(combined.OUT / "axes.json"),
        "upstream_protocol_sha256": sha(UPSTREAM / "protocol.json"),
        "upstream_folds_sha256": sha(UPSTREAM / "folds.json"),
        "upstream_complete_sha256": sha(UPSTREAM / "complete.json"),
        "native_v2_fit_complete_sha256": sha(NATIVE_V2_FIT),
        "native_v2_fit_features_sha256": sha(NATIVE_V2_FEATURES),
        "expanded_matrix_declared_hashes": read(EXPANDED_MATRIX_PREP)["files_sha256"],
        "formal_baseline_files_sha256": baseline_snapshot(),
    }
    frozen_json(OUT / "source_snapshot.json", sources)
    complete = {
        "status": "fit_design_prepared_with_safe_source_group_OOF_whitebox",
        "counts": {
            "answers": FIT_ANSWERS, "groups": FIT_GROUPS,
            "claims": FIT_CLAIMS, "windows": FIT_WINDOWS,
            "positive_windows": FIT_POSITIVE_WINDOWS,
            "excluded_windows": EXCLUDED_WINDOWS,
        },
        "selection_frozen": "whitebox_geometry__C0.1",
        "provenance_audit": provenance,
        "local_signal_audit": signal_audit,
        "native_feature_replay": replay,
        "artifacts_sha256": {
            "fit_local_signals.npy": sha(OUT / "fit_local_signals.npy"),
            "fit_claim_whitebox.npy": sha(OUT / "fit_claim_whitebox.npy"),
            "fit_whitebox_geometry.npy": sha(OUT / "fit_whitebox_geometry.npy"),
            "protocol.json": sha(OUT / "protocol.json"),
            "source_snapshot.json": sha(OUT / "source_snapshot.json"),
        },
        "fit_labels_opened": True,
        "calibration_opened": False,
        "official_test_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
    }
    atomic_json(OUT / "preparation_complete.json", complete)
    print("EXPANDED_WINDOW_V2_PREPARED", FIT_WINDOWS, FIT_POSITIVE_WINDOWS, flush=True)


def check_prepared() -> dict:
    done = read(OUT / "preparation_complete.json")
    assert done["status"] == "fit_design_prepared_with_safe_source_group_OOF_whitebox"
    for name, expected in done["artifacts_sha256"].items():
        assert sha(OUT / name) == expected
    source = read(OUT / "source_snapshot.json")
    assert source["runner_sha256"] == sha(Path(__file__))
    assert source["feature_module_sha256"] == sha(HERE / "semantic_window_v2_features.py")
    assert source["formal_baseline_files_sha256"] == baseline_snapshot()
    assert read(OUT / "PROVENANCE_AUDIT.json")["in_sample_supervised_columns"] == 0
    return done


def fit() -> None:
    check_prepared()
    assert not (OUT / "fit_complete.json").exists()
    started = time.perf_counter()
    _, projection = combined.load_projection()
    x = np.load(OUT / "fit_whitebox_geometry.npy", mmap_mode="r")
    labels = projection["window_labels"].astype(np.int8)
    groups = combined.group_indices("window")
    answers = projection["window_response_index"].astype(np.int32)
    assert x.shape == (FIT_WINDOWS, 26)
    assert len(np.unique(groups)) == FIT_GROUPS and int(labels.sum()) == FIT_POSITIVE_WINDOWS

    splitter = GroupKFold(DOWNSTREAM_FOLDS)
    splits = list(splitter.split(np.arange(FIT_WINDOWS), labels, groups))
    oof = np.full(FIT_WINDOWS, np.nan, dtype=np.float64)
    fold_log = []
    with threadpool_limits(limits=THREADS):
        for fold, (train, held) in enumerate(splits):
            combined.assert_source_connected_disjoint(train, held, "window")
            base, loss, class_mass = hierarchy_weights(groups, answers, labels, train)
            scaler = StandardScaler()
            scaler.fit(x[train], sample_weight=base[train])
            z_train = scaler.transform(x[train])
            model = make_model()
            model.fit(z_train, labels[train], sample_weight=loss[train])
            oof[held] = model.predict_proba(scaler.transform(x[held]))[:, 1]
            fold_log.append({
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
                "n_iter": int(model.n_iter_[0]),
                "standardized_coefficients": model.coef_[0].astype(float).tolist(),
            })
            print("EXPANDED_WINDOW_V2_FOLD", fold + 1, DOWNSTREAM_FOLDS, flush=True)
    assert np.isfinite(oof).all()
    response_scores = answer_scores(oof, projection["response_window_indptr"])
    response_labels = projection["response_labels"].astype(np.int8)
    thresholds = {
        "window": q.choose_threshold(labels, oof),
        "answer": q.choose_threshold(response_labels, response_scores),
    }
    fit_metrics = {
        "windows": q.count(labels, oof, thresholds["window"]["threshold"]),
        "answers": q.count(response_labels, response_scores, thresholds["answer"]["threshold"]),
    }

    active = np.arange(FIT_WINDOWS)
    base, loss, class_mass = hierarchy_weights(groups, answers, labels, active)
    scaler = StandardScaler()
    scaler.fit(x, sample_weight=base)
    model = make_model()
    model.fit(scaler.transform(x), labels, sample_weight=loss)
    payload = {
        "version": VERSION,
        "variant": VARIANT,
        "C": C_VALUE,
        "feature_names": list(feature_v2.WHITEBOX_NAMES + feature_v2.GEOMETRY_NAMES),
        "scaler": scaler,
        "model": model,
    }
    atomic_pickle(OUT / "model.pkl", payload)
    atomic_npz(
        OUT / "fit_scores.npz", window_scores=oof, answer_scores=response_scores,
        window_labels=labels, answer_labels=response_labels,
    )
    source = read(OUT / "source_snapshot.json")
    assert source["formal_baseline_files_sha256"] == baseline_snapshot()
    complete = {
        "status": "expanded_fit_model_and_thresholds_frozen_before_calibration",
        "selected": {
            "candidate": "whitebox_geometry__C0.1", "variant": VARIANT,
            "blocks": ["whitebox", "geometry"], "width": 26, "C": C_VALUE,
            "fit_OOF_thresholds": thresholds, "fit_OOF": fit_metrics,
        },
        "selection_source": "Frozen native-634 fit-only selection; no expanded-fit feature/C search.",
        "crossfit": fold_log,
        "full_fit": {
            "windows": FIT_WINDOWS, "positive_windows": FIT_POSITIVE_WINDOWS,
            "answers": FIT_ANSWERS, "groups": FIT_GROUPS,
            "final_loss_mass_by_class": class_mass.tolist(),
            "n_iter": int(model.n_iter_[0]),
        },
        "upstream_whitebox_provenance": read(OUT / "PROVENANCE_AUDIT.json")["status"],
        "artifacts_sha256": {
            "model.pkl": sha(OUT / "model.pkl"),
            "fit_scores.npz": sha(OUT / "fit_scores.npz"),
            "fit_whitebox_geometry.npy": sha(OUT / "fit_whitebox_geometry.npy"),
            "preparation_complete.json": sha(OUT / "preparation_complete.json"),
            "protocol.json": sha(OUT / "protocol.json"),
            "source_snapshot.json": sha(OUT / "source_snapshot.json"),
            "runner": sha(Path(__file__)),
        },
        "fit_labels_opened": True,
        "calibration_labels_opened": False,
        "calibration_used_for_model_feature_C_or_threshold_selection": False,
        "official_test_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "fit_complete.json", complete)
    print("EXPANDED_WINDOW_V2_FIT_FROZEN", thresholds["window"]["f1"], thresholds["answer"]["f1"], flush=True)


def evaluate() -> None:
    fit_done = read(OUT / "fit_complete.json")
    assert fit_done["status"] == "expanded_fit_model_and_thresholds_frozen_before_calibration"
    assert not (OUT / "summary.json").exists(), "Calibration evaluation is single-use"
    for name, expected in fit_done["artifacts_sha256"].items():
        path = Path(__file__) if name == "runner" else OUT / name
        assert sha(path) == expected
    source = read(OUT / "source_snapshot.json")
    assert source["formal_baseline_files_sha256"] == baseline_snapshot()
    started = time.perf_counter()

    # Calibration is first opened here, after the model and thresholds exist.
    import run_semantic_window_v2 as native_v2
    union, _, meta, whitebox_audit = native_v2.build_partition_design("calibration")
    columns = feature_v2.variant_indices(VARIANT)
    x = union[:, columns]
    assert x.shape == (CAL_WINDOWS, 26)
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    assert payload["version"] == VERSION and payload["variant"] == VARIANT and payload["C"] == C_VALUE
    scores = payload["model"].predict_proba(payload["scaler"].transform(x))[:, 1]
    assert scores.shape == (CAL_WINDOWS,) and np.isfinite(scores).all()
    answer_score = native_v2.answer_scores(meta, scores)
    thresholds = fit_done["selected"]["fit_OOF_thresholds"]
    strict = native_v2.metric_pair(meta, scores, thresholds)
    atomic_npz(
        OUT / "calibration_scores.npz", window_scores=scores,
        answer_scores=answer_score,
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["windows"]]),
    )
    assert source["formal_baseline_files_sha256"] == baseline_snapshot()
    summary = {
        "status": "single_development_calibration_evaluation_complete",
        "candidate": "whitebox_geometry__C0.1",
        "fit_OOF": fit_done["selected"]["fit_OOF"],
        "fit_OOF_thresholds": thresholds,
        "calibration_at_frozen_fit_thresholds": strict,
        "calibration_thresholds_fitted": 0,
        "calibration_evaluations": 1,
        "calibration_whitebox_reconstruction_audit": whitebox_audit,
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "artifacts_sha256": {"calibration_scores.npz": sha(OUT / "calibration_scores.npz")},
        "formal_baseline_files_unchanged": True,
        "calibration_labels_opened": True,
        "calibration_used_for_model_feature_C_or_fit_threshold_selection": False,
        "official_test_opened": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "summary.json", summary)
    report = f"""# Expanded semantic window v2\n\n`whitebox_geometry`, C=0.1 was copied from the native-634 fit-only winner before this run. No expanded-fit feature or C search was performed.\n\nAll 3,680 fit answers (615 source-connected groups) supplied 653,979 directly supervised four-BPE windows, including 58,433 positive windows. Lookback and large inputs were reconstructed only from each source group's frozen upstream held fold; generation NLL was label-blind. The first 168,123 native feature rows replayed exactly.\n\n| partition / threshold | window F1 | answer F1 |\n|---|---:|---:|\n| expanded fit 5-fold OOF / fit-F1Opt | {fit_done['selected']['fit_OOF']['windows']['f1']:.6f} | {fit_done['selected']['fit_OOF']['answers']['f1']:.6f} |\n| calibration / frozen fit thresholds | {strict['windows']['f1']:.6f} | {strict['answers']['f1']:.6f} |\n\nCalibration was scored once after the full model and thresholds were frozen. No calibration threshold was fitted, the official test remained unopened, GPU was unused, and formal baseline files were unchanged.\n"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("EXPANDED_WINDOW_V2_CALIBRATION", strict["windows"]["f1"], strict["answers"]["f1"], flush=True)


def status() -> None:
    value = {
        "prepared": (OUT / "preparation_complete.json").exists(),
        "fit_frozen": (OUT / "fit_complete.json").exists(),
        "calibration_evaluated": (OUT / "summary.json").exists(),
        "official_test_opened": False,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "evaluate", "status"))
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        {"prepare": prepare, "fit": fit, "evaluate": evaluate, "status": status}[args.command]()


if __name__ == "__main__":
    main()
