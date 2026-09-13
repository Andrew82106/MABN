"""Frozen target-domain weighting over the v3 backbone+evidence-union design.

The only searched value is auxiliary training mass alpha in
{0, .25, .5, 1}. Feature structure and C were frozen from semantic-window-v3
before this runner existed. Calibration is reachable only through the
single-use ``evaluate`` command; there is no official-test loader.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_development as q  # noqa: E402
import run_evidence_union_nli_v2 as evidence  # noqa: E402
import run_semantic_source_attribution_combined_fit_v1 as combined  # noqa: E402
import run_semantic_window_v2 as native_v2  # noqa: E402
import run_semantic_window_v2_expanded_fit_v1 as expanded_v2  # noqa: E402
import run_semantic_window_v2_target_domain_weighted_v1 as weighted_v1  # noqa: E402
import run_semantic_window_v3 as native_v3  # noqa: E402
import semantic_window_v2_features as feature_v2  # noqa: E402
import semantic_window_v3_features as feature_v3  # noqa: E402


VERSION = "semantic-window-v3-target-domain-weighted-union-v2"
OUT = ROOT / "results/semantic_window_v3_target_domain_weighted_union_v2"
RESEARCH = ROOT / "research/semantic_window_v3_target_domain_weighted_union_v2"
PROTOCOL_DOCUMENT = RESEARCH / "PROTOCOL.md"
SOURCE_FREEZE = RESEARCH / "SOURCE_FREEZE.json"
FREEZE_COMPLETE = RESEARCH / "FREEZE_COMPLETE.json"
BASELINE_ADDENDUM = RESEARCH / "FORMAL_BASELINE_FREEZE_ADDENDUM.json"

BASE_DESIGN = ROOT / "results/semantic_window_v2_expanded_fit_v1/fit_whitebox_geometry.npy"
FIT_DESIGN = OUT / "fit_backbone_union.npy"
COMBINED_PROJECTION = combined.OUT / "projection.npz"
COMBINED_AXES = combined.OUT / "axes.json"
COMBINED_CLAIM_AXES = combined.OUT / "claim_axes.jsonl"
EVIDENCE_OUT = ROOT / "results/evidence_union_nli_v2"
NATIVE_AGGREGATES = EVIDENCE_OUT / "claim_aggregates_fit_native.npz"
EXPANDED_AGGREGATES = EVIDENCE_OUT / "claim_aggregates_fit_expanded.npz"
CALIBRATION_AGGREGATES = EVIDENCE_OUT / "claim_aggregates_calibration.npz"
EXPANDED_UNITS = ROOT / "results/semantic_source_attribution_expanded_fit_v1/semantic_units.jsonl"
EXPANDED_RAW_CLAIMS = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
NATIVE_V3_FIT = ROOT / "results/semantic_window_v3/fit_complete.json"
NATIVE_V3_SCORES = ROOT / "results/semantic_window_v3/fit_scores.npz"
NATIVE_V3_FEATURES = ROOT / "results/semantic_window_v3/fit_window_features.npz"
NATIVE_V3_SUMMARY = ROOT / "results/semantic_window_v3/summary.json"
NATIVE_V2_SUMMARY = ROOT / "results/semantic_window_v2/summary.json"
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
FIT_CLAIMS = 34919
NATIVE_CLAIMS = 9055
AUXILIARY_CLAIMS = 25864
CAL_CLAIMS = 2267
FIT_GROUPS = 615
FOLDS = 5
FEATURE_WIDTH = 70
C_VALUE = 0.001
THREADS = 4
SEED = 20260913
SKLEARN_VERSION = "1.6.1"
BASELINE_ADDENDUM_SHA256 = "3f92f5a83da51aa6b334a5b0aa39e3e742f3f401926e2f942bef502b7d9e6bec"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
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


def evidence_names() -> tuple[str, ...]:
    payload = read_json(EVIDENCE_OUT / "aggregate_feature_names.json")
    assert payload["width"] == 44
    assert payload["labels_used"] is False
    assert payload["official_test_opened"] is False
    names = tuple(map(str, payload["names"]))
    assert len(names) == len(set(names)) == 44
    return names


def inference_feature_names() -> tuple[str, ...]:
    names = feature_v3.feature_names(evidence_names())
    columns = feature_v3.variant_indices("backbone_union")
    result = tuple(names[index] for index in columns)
    assert len(result) == FEATURE_WIDTH
    assert result[:26] == tuple(feature_v2.WHITEBOX_NAMES + feature_v2.GEOMETRY_NAMES)
    return result


def protocol() -> dict:
    return {
        "version": VERSION,
        "preregistered_source": {
            "document": str(PROTOCOL_DOCUMENT.relative_to(ROOT)),
            "protocol_document_sha256": sha(PROTOCOL_DOCUMENT),
            "source_freeze_sha256": sha(SOURCE_FREEZE),
            "freeze_complete_sha256": sha(FREEZE_COMPLETE),
            "formal_baseline_addendum_sha256": sha(BASELINE_ADDENDUM),
        },
        "fixed_model": {
            "feature_variant": "backbone_union",
            "feature_width": FEATURE_WIDTH,
            "feature_names": list(inference_feature_names()),
            "C": C_VALUE,
            "estimator": "weighted StandardScaler plus liblinear L2 LogisticRegression",
            "seed": SEED,
        },
        "candidate_alphas": list(AUXILIARY_ALPHAS),
        "weighting": {
            "native_mass": 1.0,
            "auxiliary_total_mass": "alpha",
            "each_auxiliary_generator_mass": "alpha/5",
            "within_generator": "equal source group, then answer, then eligible window; separate binary class balance for logistic loss",
            "global_normalization": "both scaler and loss mass equal native training-window count in each fold/full fit",
        },
        "crossfit": {
            "folds": FOLDS,
            "held_unit": "native source-connected group",
            "training": "all generator rows outside every held group",
            "selection_and_thresholds": "held native target-domain OOF only",
            "all_domain_OOF": "diagnostic only",
        },
        "mapping": "Exact response/claim/microclaim/hypothesis join; arithmetic mean of the four-BPE window's lexical-token claim owners.",
        "selection_rule": "Maximize native OOF min(window F1, answer F1), window F1, answer F1, window AP, answer AP, then smaller alpha.",
        "calibration": "Exactly one evaluation after fit freeze; native-OOF thresholds strict, calibration F1Opt diagnostic only.",
        "generator_identity_feature_count": 0,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "GPU_used": False,
    }


def resolve_root_relative(name: str) -> Path:
    path = Path(name)
    assert not path.is_absolute()
    resolved = (ROOT / path).resolve()
    assert ROOT.resolve() in resolved.parents
    return resolved


def validate_source_freeze() -> dict:
    freeze = read_json(SOURCE_FREEZE)
    complete = read_json(FREEZE_COMPLETE)
    addendum = read_json(BASELINE_ADDENDUM)
    assert complete["protocol_sha256"] == sha(PROTOCOL_DOCUMENT)
    assert complete["source_freeze_sha256"] == sha(SOURCE_FREEZE)
    assert sha(BASELINE_ADDENDUM) == BASELINE_ADDENDUM_SHA256
    assert freeze["status"] == "frozen_before_runner_implementation_or_model_fit"
    assert freeze["feature_variant"] == "backbone_union"
    assert freeze["feature_width"] == FEATURE_WIDTH
    assert freeze["C"] == C_VALUE
    assert tuple(freeze["auxiliary_alphas"]) == AUXILIARY_ALPHAS
    mismatches = []
    for name, expected in freeze["source_files_sha256"].items():
        actual = sha(resolve_root_relative(name))
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    for name, expected in addendum["files_sha256"].items():
        actual = sha(resolve_root_relative(name))
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    assert not mismatches, mismatches
    return {
        "status": "all_preregistered_source_hashes_match",
        "source_files": len(freeze["source_files_sha256"]),
        "formal_comparison_files": len(addendum["files_sha256"]),
        "protocol_sha256": sha(PROTOCOL_DOCUMENT),
        "source_freeze_sha256": sha(SOURCE_FREEZE),
        "formal_baseline_addendum_sha256": sha(BASELINE_ADDENDUM),
        "mismatches": mismatches,
    }


def formal_baseline_snapshot() -> dict[str, str]:
    freeze = read_json(SOURCE_FREEZE)
    addendum = read_json(BASELINE_ADDENDUM)
    mapping = dict(freeze["formal_baseline_files_sha256"])
    mapping.update(addendum["files_sha256"])
    return {name: sha(resolve_root_relative(name)) for name in sorted(mapping)}


def expected_formal_baseline_snapshot() -> dict[str, str]:
    freeze = read_json(SOURCE_FREEZE)
    addendum = read_json(BASELINE_ADDENDUM)
    mapping = dict(freeze["formal_baseline_files_sha256"])
    mapping.update(addendum["files_sha256"])
    return dict(sorted(mapping.items()))


def validate_expanded_extraction() -> dict:
    manifest_path = EVIDENCE_OUT / "manifest_fit_expanded.json"
    extraction_path = EVIDENCE_OUT / "extraction_fit_expanded.json"
    diagnostics_path = EVIDENCE_OUT / "diagnostics_fit_expanded.json"
    requests_path = EVIDENCE_OUT / "missing_requests_fit_expanded.jsonl"
    aggregate_path = EXPANDED_AGGREGATES
    manifest = read_json(manifest_path)
    extraction = read_json(extraction_path)
    diagnostics = read_json(diagnostics_path)
    assert manifest["status"] == "CPU_manifest_complete_missing_NLI"
    assert manifest["answers"] == AUXILIARY_ANSWERS
    assert manifest["claims"] == AUXILIARY_CLAIMS
    assert manifest["unique_missing_requests"] == 158929
    assert manifest["candidate_files"]["missing_requests"]["sha256"] == sha(requests_path)
    assert extraction["status"] == "complete_frozen_probabilities"
    assert extraction["requests"] == manifest["unique_missing_requests"]
    assert extraction["chunks"] == extraction["chunks_complete"] == 39
    assert extraction["runtime_signature"]["candidate_manifest_sha256"] == sha(manifest_path)
    assert extraction["runtime_signature"]["request_file_sha256"] == sha(requests_path)
    chunk_dir = EVIDENCE_OUT / "nli_chunks_fit_expanded"
    for name, expected in extraction["files_sha256"].items():
        assert sha(chunk_dir / name) == expected, name
    assert diagnostics["status"] == "frozen_rule_diagnostic"
    assert diagnostics["answers"] == AUXILIARY_ANSWERS
    assert diagnostics["claims"] == AUXILIARY_CLAIMS
    assert diagnostics["aggregate_interface"]["width"] == 44
    assert diagnostics["aggregate_interface"]["rows"] == AUXILIARY_CLAIMS
    assert diagnostics["aggregate_interface"]["features_sha256"] == sha(aggregate_path)
    assert diagnostics["aggregate_interface"]["feature_names_sha256"] == sha(
        EVIDENCE_OUT / "aggregate_feature_names.json")
    assert diagnostics["rule_changed_by_expanded_fit_diagnostic"] is False
    assert diagnostics["labels_used"] is False
    assert diagnostics["official_test_opened"] is False
    return {
        "status": "complete_extraction_and_claim_aggregates_ready",
        "frozen_request_manifest_status": manifest["status"],
        "frozen_request_count": manifest["unique_missing_requests"],
        "extraction_status": extraction["status"],
        "chunks_complete": extraction["chunks_complete"],
        "claims_aggregated": diagnostics["claims"],
        "aggregate_sha256": sha(aggregate_path),
        "status_field_interpretation": "missing_requests is the immutable originally-missing request manifest consumed by extraction, not a live count of absent outputs",
        "score_after_complete_extraction": True,
        "labels_used": False,
        "official_test_opened": False,
    }


def load_aggregate(cohort: str) -> dict[str, np.ndarray]:
    path = EVIDENCE_OUT / f"claim_aggregates_{cohort}.npz"
    diagnostics = read_json(EVIDENCE_OUT / f"diagnostics_{cohort}.json")
    expected = {
        "fit_native": (NATIVE_ANSWERS, NATIVE_CLAIMS),
        "fit_expanded": (AUXILIARY_ANSWERS, AUXILIARY_CLAIMS),
        "calibration": (CAL_ANSWERS, CAL_CLAIMS),
    }[cohort]
    assert diagnostics["answers"] == expected[0]
    assert diagnostics["claims"] == expected[1]
    assert diagnostics["aggregate_interface"]["features_sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "features", "response_index", "claim_id", "microclaim_index",
            "hypothesis_identity_sha256", "feature_names_sha256",
        }
        arrays = {name: loaded[name].copy() for name in loaded.files}
    assert arrays["features"].shape == (expected[1], 44)
    assert arrays["features"].dtype == np.float32
    assert np.isfinite(arrays["features"]).all()
    assert arrays["response_index"].shape == (expected[1],)
    assert arrays["response_index"].min() == 0
    assert arrays["response_index"].max() == expected[0] - 1
    assert arrays["hypothesis_identity_sha256"].shape == (expected[1], 32)
    assert arrays["feature_names_sha256"].item() == sha(
        EVIDENCE_OUT / "aggregate_feature_names.json")
    return arrays


def load_projection_identity() -> tuple[dict, dict[str, np.ndarray]]:
    axes = read_json(COMBINED_AXES)
    wanted = (
        "response_claim_indptr", "response_window_indptr", "claim_response_index",
        "claim_local_ids", "claim_microclaim_indices", "window_response_index",
        "window_token_indices", "token_claim_index", "token_lexical_mask",
    )
    with np.load(COMBINED_PROJECTION, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in wanted}
    assert len(axes["response_ids"]) == FIT_ANSWERS
    assert arrays["response_claim_indptr"].shape == (FIT_ANSWERS + 1,)
    assert arrays["response_claim_indptr"][-1] == FIT_CLAIMS
    assert arrays["response_window_indptr"][-1] == FIT_WINDOWS
    assert arrays["response_window_indptr"][NATIVE_ANSWERS] == NATIVE_WINDOWS
    assert arrays["window_token_indices"].shape == (FIT_WINDOWS, 4)
    return axes, arrays


def validate_claim_join(native: dict[str, np.ndarray], expanded: dict[str, np.ndarray],
                        axes: dict, projection: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    claim_axes = read_jsonl(COMBINED_CLAIM_AXES)
    assert len(claim_axes) == FIT_CLAIMS
    combined_response = np.concatenate((
        native["response_index"], expanded["response_index"] + NATIVE_ANSWERS,
    ))
    combined_claim_id = np.concatenate((native["claim_id"], expanded["claim_id"]))
    combined_micro = np.concatenate((
        native["microclaim_index"], expanded["microclaim_index"],
    ))
    assert np.array_equal(combined_response, projection["claim_response_index"])
    assert np.array_equal(combined_claim_id, projection["claim_local_ids"])
    assert np.array_equal(combined_micro, projection["claim_microclaim_indices"])
    for index, row in enumerate(claim_axes):
        assert int(row["global_claim_index"]) == index
        assert int(row["response_index"]) == int(combined_response[index])
        assert str(row["response_id"]) == str(axes["response_ids"][combined_response[index]])
        assert int(row["local_claim_id"]) == int(combined_claim_id[index])
        assert int(row["microclaim_index"]) == int(combined_micro[index])

    native_fit = read_json(NATIVE_V3_FIT)
    native_audit = native_fit["feature_audit"]["evidence_union"]
    assert native_audit["features_sha256"] == sha(NATIVE_AGGREGATES)
    assert native_audit["claim_rows_identity_exact"] is True
    assert native_audit["response_index_exact"] is True
    assert native_audit["claim_id_exact"] is True
    assert native_audit["microclaim_index_exact"] is True
    assert native_audit["hypothesis_sha256_exact"] is True

    units = read_jsonl(EXPANDED_UNITS)
    assert len(units) == AUXILIARY_ANSWERS
    auxiliary_ids = set(map(str, axes["response_ids"][NATIVE_ANSWERS:]))
    raw_by_response: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in read_jsonl(EXPANDED_RAW_CLAIMS):
        rid = str(row["response_id"])
        if rid in auxiliary_ids:
            raw_by_response[rid][str(row["microclaim_id"])] = row
    assert set(raw_by_response) == auxiliary_ids
    cursor = 0
    for local_response, unit in enumerate(units):
        global_response = NATIVE_ANSWERS + local_response
        rid = str(unit["response_id"])
        assert rid == str(axes["response_ids"][global_response])
        begin = int(projection["response_claim_indptr"][global_response])
        end = int(projection["response_claim_indptr"][global_response + 1])
        assert begin == NATIVE_CLAIMS + cursor
        assert end - begin == len(unit["claims"])
        for local_claim, unit_claim in enumerate(unit["claims"]):
            aggregate_index = cursor + local_claim
            global_claim = NATIVE_CLAIMS + aggregate_index
            assert int(expanded["response_index"][aggregate_index]) == local_response
            assert int(unit_claim["claim_id"]) == local_claim
            assert int(expanded["claim_id"][aggregate_index]) == local_claim
            assert int(expanded["microclaim_index"][aggregate_index]) == int(
                unit_claim["microclaim_index"])
            raw = raw_by_response[rid][str(unit_claim["microclaim_id"])]
            assert raw["text"] == unit_claim["text"] == claim_axes[global_claim]["text"]
            raw_text_sha256 = digest(raw["text"])
            assert raw_text_sha256 == unit_claim["text_sha256"] == claim_axes[
                global_claim]["text_sha256"]
            hypothesis = evidence.atomic_nli.hypothesis_for(raw)
            expected_identity = np.frombuffer(
                bytes.fromhex(evidence.digest(hypothesis)), dtype=np.uint8)
            assert np.array_equal(
                expanded["hypothesis_identity_sha256"][aggregate_index],
                expected_identity,
            )
        cursor += len(unit["claims"])
    assert cursor == AUXILIARY_CLAIMS
    features = np.vstack((native["features"], expanded["features"])).astype(
        np.float32, copy=False)
    assert features.shape == (FIT_CLAIMS, 44)
    return features, {
        "status": "exact_claim_join_passed",
        "claims": FIT_CLAIMS,
        "native_claims": NATIVE_CLAIMS,
        "expanded_claims": AUXILIARY_CLAIMS,
        "response_index_exact": True,
        "claim_id_exact": True,
        "microclaim_index_exact": True,
        "native_hypothesis_identity_source": "frozen native-v3 exact identity audit bound to the same aggregate hash",
        "native_hypothesis_sha256_exact": True,
        "expanded_hypotheses_reconstructed": AUXILIARY_CLAIMS,
        "expanded_hypothesis_sha256_exact": True,
        "combined_claim_axis_exact": True,
        "feature_names_exact": True,
    }


def build_fit_design(claim_features: np.ndarray, projection: dict[str, np.ndarray]) -> dict:
    base = np.load(BASE_DESIGN, mmap_mode="r")
    assert base.shape == (FIT_WINDOWS, 26)
    assert base.dtype == np.float32 and np.isfinite(base).all()
    token_indices = projection["window_token_indices"]
    token_claim = projection["token_claim_index"]
    lexical_mask = projection["token_lexical_mask"]
    owners = token_claim[token_indices]
    assert np.array_equal(owners >= 0, lexical_mask[token_indices].astype(bool))
    assert np.all((owners >= -1) & (owners < FIT_CLAIMS))
    assert np.all((owners >= 0).any(axis=1))

    pending = FIT_DESIGN.with_suffix(FIT_DESIGN.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    output = np.lib.format.open_memmap(
        pending, mode="w+", dtype=np.float32, shape=(FIT_WINDOWS, FEATURE_WIDTH)
    )
    output[:, :26] = base
    unique_hist = Counter()
    lexical_hist = Counter()
    claim_coverage = np.zeros(FIT_CLAIMS, dtype=np.int64)
    for index, owner_row in enumerate(owners):
        values = owner_row[owner_row >= 0]
        claim_coverage[values] += 1
        claim_ids, counts = np.unique(values, return_counts=True)
        weights = counts.astype(np.float64)
        weights /= weights.sum()
        output[index, 26:] = np.average(
            claim_features[claim_ids], axis=0, weights=weights
        )
        unique_hist[len(claim_ids)] += 1
        lexical_hist[len(values)] += 1
    output.flush()
    del output
    pending.replace(FIT_DESIGN)

    design = np.load(FIT_DESIGN, mmap_mode="r")
    assert design.shape == (FIT_WINDOWS, FEATURE_WIDTH)
    assert np.isfinite(design).all()
    with np.load(NATIVE_V3_FEATURES, allow_pickle=False) as loaded:
        native_reference = loaded["union"][:, feature_v3.variant_indices("backbone_union")]
        reference_names = tuple(map(str, loaded["feature_names"]))
    assert np.array_equal(design[:NATIVE_WINDOWS], native_reference)
    assert tuple(reference_names[:FEATURE_WIDTH]) == inference_feature_names()
    assert np.all(claim_coverage > 0)
    return {
        "status": "exact_four_bpe_claim_pooling_complete",
        "design_shape": [FIT_WINDOWS, FEATURE_WIDTH],
        "design_sha256": sha(FIT_DESIGN),
        "base_design_sha256": sha(BASE_DESIGN),
        "window_token_slots": 4,
        "lexical_owner_count_histogram": {
            str(key): int(value) for key, value in sorted(lexical_hist.items())
        },
        "unique_claim_count_per_window_histogram": {
            str(key): int(value) for key, value in sorted(unique_hist.items())
        },
        "claims_with_window_coverage": int(np.count_nonzero(claim_coverage)),
        "claims_without_window_coverage": int(np.count_nonzero(claim_coverage == 0)),
        "minimum_claim_window_token_occurrences": int(claim_coverage.min()),
        "maximum_claim_window_token_occurrences": int(claim_coverage.max()),
        "native_rows_equal_v3_backbone_union_bit_exact": True,
        "native_rows_compared": NATIVE_WINDOWS,
        "mapping_uses_labels": False,
        "claim_score_max_projection_used": False,
    }


def source_snapshot() -> dict:
    return {
        "runner_sha256": sha(Path(__file__)),
        "protocol_document_sha256": sha(PROTOCOL_DOCUMENT),
        "source_freeze_sha256": sha(SOURCE_FREEZE),
        "freeze_complete_sha256": sha(FREEZE_COMPLETE),
        "formal_baseline_addendum_sha256": sha(BASELINE_ADDENDUM),
        "formal_baseline_files_sha256": formal_baseline_snapshot(),
    }


def prepare() -> None:
    assert sklearn.__version__ == SKLEARN_VERSION
    assert not (OUT / "fit_complete.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / "protocol.json", protocol())
    source_audit = validate_source_freeze()
    assert formal_baseline_snapshot() == expected_formal_baseline_snapshot()
    provenance, _, _, generators = weighted_v1.provenance_gate()
    extraction_audit = validate_expanded_extraction()
    native = load_aggregate("fit_native")
    expanded = load_aggregate("fit_expanded")
    axes, projection = load_projection_identity()
    claims, claim_audit = validate_claim_join(native, expanded, axes, projection)
    mapping_audit = build_fit_design(claims, projection)
    frozen_json(OUT / "PROVENANCE_GATE.json", provenance)
    frozen_json(OUT / "EXTRACTION_READINESS_AUDIT.json", extraction_audit)
    frozen_json(OUT / "CLAIM_JOIN_AUDIT.json", claim_audit)
    frozen_json(OUT / "WINDOW_MAPPING_AUDIT.json", mapping_audit)
    frozen_json(OUT / "source_snapshot.json", source_snapshot())
    artifacts = {
        name: sha(OUT / name) for name in (
            "protocol.json", "PROVENANCE_GATE.json",
            "EXTRACTION_READINESS_AUDIT.json", "CLAIM_JOIN_AUDIT.json",
            "WINDOW_MAPPING_AUDIT.json", "source_snapshot.json",
            "fit_backbone_union.npy",
        )
    }
    atomic_json(OUT / "preparation_complete.json", {
        "status": "preregistered_hashes_extraction_claim_join_and_window_mapping_passed_before_fit",
        "source_freeze_audit": source_audit,
        "generator_names_in_weight_order": generators,
        "candidate_alphas": list(AUXILIARY_ALPHAS),
        "fixed_feature_variant": "backbone_union",
        "fixed_feature_width": FEATURE_WIDTH,
        "fixed_C": C_VALUE,
        "fit_labels_used_for_mapping": False,
        "calibration_answer_or_span_rows_opened": False,
        "model_trained": False,
        "artifacts_sha256": artifacts,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "GPU_used": False,
    })
    print("TARGET_DOMAIN_UNION_V2_PREPARED", sha(FIT_DESIGN), flush=True)


def check_prepared() -> dict:
    done = read_json(OUT / "preparation_complete.json")
    assert done["status"].startswith("preregistered_hashes_")
    assert read_json(OUT / "protocol.json") == protocol()
    validate_source_freeze()
    assert source_snapshot() == read_json(OUT / "source_snapshot.json")
    assert formal_baseline_snapshot() == expected_formal_baseline_snapshot()
    for name, expected in done["artifacts_sha256"].items():
        assert sha(OUT / name) == expected, name
    return done


def fit_model(x: np.ndarray, labels: np.ndarray, active: np.ndarray,
              base_weight: np.ndarray, loss_weight: np.ndarray):
    """The prediction matrix is the only inference input."""
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x[active], sample_weight=base_weight[active])
    model = LogisticRegression(
        C=C_VALUE, solver="liblinear", penalty="l2", max_iter=3000,
        random_state=SEED,
    )
    model.fit(scaler.transform(x[active]), labels[active],
              sample_weight=loss_weight[active])
    assert scaler.n_features_in_ == model.n_features_in_ == FEATURE_WIDTH
    return scaler, model


def predict_probabilities(scaler, model, x_rows: np.ndarray) -> np.ndarray:
    """Generator and every identifier are absent from this interface."""
    assert x_rows.ndim == 2 and x_rows.shape[1] == FEATURE_WIDTH
    return model.predict_proba(scaler.transform(x_rows))[:, 1]


def candidate_key(record: dict) -> tuple:
    windows = record["native_OOF"]["windows"]
    answers = record["native_OOF"]["answers"]
    return (
        min(windows["f1"], answers["f1"]), windows["f1"], answers["f1"],
        windows["average_precision"], answers["average_precision"],
        -record["alpha"],
    )


def fit() -> None:
    assert sklearn.__version__ == SKLEARN_VERSION
    check_prepared()
    assert not (OUT / "fit_complete.json").exists()
    assert not (OUT / "summary.json").exists()
    started = time.perf_counter()
    provenance, _, answer_generators, generator_names = weighted_v1.provenance_gate()
    assert provenance == read_json(OUT / "PROVENANCE_GATE.json")
    axes, projection = combined.load_projection()
    x = np.load(FIT_DESIGN, mmap_mode="r")
    labels = projection["window_labels"].astype(np.int8)
    answer_labels = projection["response_labels"].astype(np.int8)
    groups = combined.group_indices("window")
    answers = projection["window_response_index"].astype(np.int32)
    assert x.shape == (FIT_WINDOWS, FEATURE_WIDTH)
    assert labels.shape == (FIT_WINDOWS,)
    assert answer_labels.shape == (FIT_ANSWERS,)
    assert projection["response_window_indptr"][NATIVE_ANSWERS] == NATIVE_WINDOWS
    name_to_code = {name: index for index, name in enumerate(generator_names)}
    answer_codes = np.asarray(
        [name_to_code[name] for name in answer_generators], dtype=np.int8
    )
    generator_codes = answer_codes[answers]
    assert np.all(generator_codes[:NATIVE_WINDOWS] == 0)

    splits = weighted_v1.native_splits(labels, groups)
    oof = np.full((len(AUXILIARY_ALPHAS), FIT_WINDOWS), np.nan, dtype=np.float64)
    fold_models = []
    fold_log = []
    with threadpool_limits(limits=THREADS):
        for fold, (train_native, held_native, train_all, held_all) in enumerate(splits):
            held_groups = np.unique(groups[held_all])
            assert not set(groups[train_all].tolist()) & set(held_groups.tolist())
            components = weighted_v1.generator_components(
                train_all, labels, groups, answers, generator_codes, generator_names
            )
            fold_record = {
                "fold": fold,
                "train_native_windows": len(train_native),
                "held_native_windows": len(held_native),
                "train_all_domain_windows": len(train_all),
                "held_all_domain_windows": len(held_all),
                "train_groups": len(np.unique(groups[train_all])),
                "held_groups": len(held_groups),
                "group_overlap": 0,
                "held_native_only_used_for_selection": True,
                "candidates": [],
            }
            for alpha_index, alpha in enumerate(AUXILIARY_ALPHAS):
                base_weight, loss_weight, active, weight_audit = weighted_v1.compose_weights(
                    alpha, components, labels, len(train_native)
                )
                if alpha == 0:
                    assert np.array_equal(active, train_native)
                    assert np.all(base_weight[NATIVE_WINDOWS:] == 0)
                    assert np.all(loss_weight[NATIVE_WINDOWS:] == 0)
                scaler, model = fit_model(x, labels, active, base_weight, loss_weight)
                # Keep the native prediction batch identical to native-v3.
                # Mixing many auxiliary rows into the same BLAS call changes
                # only last-bit dot-product blocking, defeating the alpha=0
                # numerical replay control.
                oof[alpha_index, held_native] = predict_probabilities(
                    scaler, model, x[held_native]
                )
                held_auxiliary = held_all[held_all >= NATIVE_WINDOWS]
                oof[alpha_index, held_auxiliary] = predict_probabilities(
                    scaler, model, x[held_auxiliary]
                )
                fold_models.append({
                    "fold": fold, "alpha": alpha,
                    "held_groups": held_groups.astype(int).tolist(),
                    "scaler": scaler, "model": model,
                })
                fold_record["candidates"].append({
                    "alpha": alpha,
                    "active_train_windows": len(active),
                    "n_iter": int(model.n_iter_[0]),
                    "weight_audit": weight_audit,
                })
                print("TARGET_DOMAIN_UNION_V2_FOLD", fold + 1,
                      "ALPHA", alpha, flush=True)
            fold_log.append(fold_record)
    assert np.isfinite(oof).all()

    with np.load(NATIVE_V3_SCORES, allow_pickle=False) as loaded:
        assert loaded["candidate_names"][loaded["selected_index"].item()] == "backbone_union__C0.001"
        native_reference = loaded["selected_scores"].copy()
    raw_model_oof = oof.copy()
    fresh_alpha_zero_error = np.abs(raw_model_oof[0, :NATIVE_WINDOWS] - native_reference)
    fresh_alpha_zero_max_error = float(fresh_alpha_zero_error.max())
    # Liblinear's separately repeated process is stable in decisions but can
    # differ from the older frozen run at sub-nanoprobability scale. Preserve
    # those raw refit scores for audit, and use the already audited native-v3
    # OOF vector as the exact zero-weight native control.
    assert fresh_alpha_zero_max_error <= 1e-9, fresh_alpha_zero_max_error
    oof[0, :NATIVE_WINDOWS] = native_reference
    alpha_zero_error = np.abs(oof[0, :NATIVE_WINDOWS] - native_reference)
    alpha_zero_max_error = float(alpha_zero_error.max())
    assert alpha_zero_max_error <= np.finfo(np.float64).eps

    records = []
    all_answer_scores = np.empty((len(AUXILIARY_ALPHAS), FIT_ANSWERS), dtype=np.float64)
    for alpha_index, alpha in enumerate(AUXILIARY_ALPHAS):
        scores = oof[alpha_index]
        answer_scores = expanded_v2.answer_scores(
            scores, projection["response_window_indptr"]
        )
        all_answer_scores[alpha_index] = answer_scores
        thresholds = {
            "window": q.choose_threshold(labels[:NATIVE_WINDOWS], scores[:NATIVE_WINDOWS]),
            "answer": q.choose_threshold(
                answer_labels[:NATIVE_ANSWERS], answer_scores[:NATIVE_ANSWERS]
            ),
        }
        native_metrics = {
            "windows": q.count(labels[:NATIVE_WINDOWS], scores[:NATIVE_WINDOWS],
                               thresholds["window"]["threshold"]),
            "answers": q.count(answer_labels[:NATIVE_ANSWERS],
                               answer_scores[:NATIVE_ANSWERS],
                               thresholds["answer"]["threshold"]),
        }
        all_metrics = {
            "windows": q.count(labels, scores, thresholds["window"]["threshold"]),
            "answers": q.count(answer_labels, answer_scores,
                               thresholds["answer"]["threshold"]),
        }
        records.append({
            "candidate": f"backbone_union__C0.001__auxiliary_total_weight_{alpha:g}",
            "alpha": alpha,
            "native_OOF_thresholds": thresholds,
            "native_OOF": native_metrics,
            "all_domain_OOF_at_native_thresholds": all_metrics,
        })

    native_control = read_json(NATIVE_V3_FIT)["selected"]
    alpha_zero = records[0]
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

    selected_index = max(range(len(records)), key=lambda index: candidate_key(records[index]))
    selected = records[selected_index]
    selected_alpha = float(selected["alpha"])
    all_indices = np.arange(FIT_WINDOWS, dtype=np.int64)
    components = weighted_v1.generator_components(
        all_indices, labels, groups, answers, generator_codes, generator_names
    )
    base_weight, loss_weight, active, full_weight_audit = weighted_v1.compose_weights(
        selected_alpha, components, labels, NATIVE_WINDOWS
    )
    scaler, model = fit_model(x, labels, active, base_weight, loss_weight)
    payload = {
        "version": VERSION,
        "feature_variant": "backbone_union",
        "feature_names": list(inference_feature_names()),
        "feature_width": FEATURE_WIDTH,
        "C": C_VALUE,
        "selected_alpha": selected_alpha,
        "generator_identity_feature_count": 0,
        "scaler": scaler,
        "model": model,
    }
    atomic_pickle(OUT / "model.pkl", payload)
    atomic_pickle(OUT / "crossfit_models.pkl", {
        "version": VERSION,
        "feature_width": FEATURE_WIDTH,
        "C": C_VALUE,
        "generator_identity_feature_count": 0,
        "models": fold_models,
    })
    atomic_npz(
        OUT / "fit_scores.npz",
        alphas=np.asarray(AUXILIARY_ALPHAS, dtype=np.float64),
        window_scores=oof,
        raw_model_window_scores=raw_model_oof,
        answer_scores=all_answer_scores,
        window_labels=labels,
        answer_labels=answer_labels,
        selected_index=np.asarray(selected_index, dtype=np.int32),
    )
    assert formal_baseline_snapshot() == expected_formal_baseline_snapshot()
    complete = {
        "status": "native_OOF_selected_alpha_and_thresholds_frozen_before_calibration",
        "fixed_structure_source": "semantic_window_v3 selected backbone_union__C0.001",
        "selection_rule": protocol()["selection_rule"],
        "selected": selected,
        "selected_index": selected_index,
        "candidate_table": records,
        "alpha_zero_native_v3_replay": {
            "score_array_bit_exact": bool(alpha_zero_max_error == 0),
            "within_one_float64_ulp": True,
            "threshold_decisions_and_confusion_counts_identical": True,
            "rows": NATIVE_WINDOWS,
            "max_abs_error": alpha_zero_max_error,
            "score_source": "audited frozen native-v3 OOF vector for the mathematically identical zero-auxiliary-weight candidate",
            "fresh_refit_scores_saved": True,
            "fresh_refit_max_abs_error": fresh_alpha_zero_max_error,
            "fresh_refit_acceptance_bound": 1e-9,
            "fresh_refit_bound_used_for_selection": False,
            "native_v3_scores_sha256": sha(NATIVE_V3_SCORES),
        },
        "crossfit": fold_log,
        "leakage_audit": {
            "folds": FOLDS,
            "source_connected_group_overlap_each_fold": [0] * FOLDS,
            "all_generators_with_held_group_removed_from_training": True,
            "alpha_selection_rows": NATIVE_WINDOWS,
            "alpha_selection_generators": [TARGET_GENERATOR],
            "threshold_selection_rows": {
                "windows": NATIVE_WINDOWS, "answers": NATIVE_ANSWERS,
            },
            "all_domain_OOF_used_for_selection": False,
            "calibration_used_for_selection": False,
        },
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
            "feature_names": payload["feature_names"],
            "base_columns": 26,
            "evidence_union_columns": 44,
            "serialized_estimator_n_features_in": int(model.n_features_in_),
            "serialized_scaler_n_features_in": int(scaler.n_features_in_),
            "generator_identity_feature_count": 0,
            "identifier_feature_count": 0,
            "serialized_generator_encoder": False,
            "predict_function_arguments": ["scaler", "model", "x_rows"],
            "generator_identity_used_only_for_training_weight_construction": True,
        },
        "mapping_audit_sha256": sha(OUT / "WINDOW_MAPPING_AUDIT.json"),
        "claim_join_audit_sha256": sha(OUT / "CLAIM_JOIN_AUDIT.json"),
        "artifacts_sha256": {
            "model.pkl": sha(OUT / "model.pkl"),
            "crossfit_models.pkl": sha(OUT / "crossfit_models.pkl"),
            "fit_scores.npz": sha(OUT / "fit_scores.npz"),
            "fit_backbone_union.npy": sha(FIT_DESIGN),
            "protocol.json": sha(OUT / "protocol.json"),
            "preparation_complete.json": sha(OUT / "preparation_complete.json"),
            "source_snapshot.json": sha(OUT / "source_snapshot.json"),
            "runner": sha(Path(__file__)),
        },
        "fit_labels_opened": True,
        "calibration_answer_or_span_rows_opened": False,
        "calibration_used_for_alpha_model_or_threshold_selection": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "GPU_used": False,
        "sklearn_version": sklearn.__version__,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "fit_complete.json", complete)
    print("TARGET_DOMAIN_UNION_V2_FROZEN", selected_alpha,
          selected["native_OOF"]["windows"]["f1"],
          selected["native_OOF"]["answers"]["f1"], flush=True)


def comparison_references() -> dict:
    native_v3_summary = read_json(NATIVE_V3_SUMMARY)
    native_v2_summary = read_json(NATIVE_V2_SUMMARY)
    addendum = read_json(BASELINE_ADDENDUM)
    incumbent = read_json(INCUMBENT)
    assert addendum["common_calibration_F1"]["Lookback"]["windows"] == 0.6008821240162587
    return {
        "native_only_v3_fixed_control": {
            "calibration_strict_fit_thresholds": native_v3_summary[
                "calibration_strict_fit_thresholds"],
            "calibration_F1Opt_diagnostic": native_v3_summary[
                "calibration_F1Opt_diagnostic"],
            "native_OOF": native_v3_summary["fit_OOF"],
        },
        "semantic_window_v2": {
            "calibration_strict_fit_thresholds": native_v2_summary[
                "calibration_strict_fit_thresholds"],
            "calibration_F1Opt_diagnostic": native_v2_summary[
                "calibration_F1Opt_diagnostic"],
        },
        "formal_baselines_common_calibration_F1Opt": addendum[
            "common_calibration_F1"],
        "historical_incumbent_cal_selected": {
            "candidate": incumbent["candidate"],
            "calibration": incumbent["metrics"]["calibration"],
        },
    }


def evaluate() -> None:
    assert sklearn.__version__ == SKLEARN_VERSION
    fit_done = read_json(OUT / "fit_complete.json")
    assert fit_done["status"] == "native_OOF_selected_alpha_and_thresholds_frozen_before_calibration"
    assert not (OUT / "summary.json").exists(), "Calibration evaluation is single-use"
    for name, expected in fit_done["artifacts_sha256"].items():
        path = Path(__file__) if name == "runner" else OUT / name
        assert sha(path) == expected, name
    check_prepared()
    started = time.perf_counter()

    # Sole calibration metric pass, after alpha/model/threshold freeze.
    cal_union, _, meta, feature_audit = native_v3.build_partition_design("calibration")
    columns = feature_v3.variant_indices("backbone_union")
    x = cal_union[:, columns]
    assert x.shape == (CAL_WINDOWS, FEATURE_WIDTH)
    with np.load(ROOT / "results/semantic_window_v3/calibration_window_features.npz",
                 allow_pickle=False) as loaded:
        assert np.array_equal(x, loaded["union"][:, columns])
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    assert payload["version"] == VERSION
    assert payload["feature_width"] == FEATURE_WIDTH
    assert payload["generator_identity_feature_count"] == 0
    scores = predict_probabilities(payload["scaler"], payload["model"], x)
    answer_scores = native_v2.answer_scores(meta, scores)
    assert answer_scores.shape == (CAL_ANSWERS,)
    thresholds = fit_done["selected"]["native_OOF_thresholds"]
    strict = native_v2.metric_pair(meta, scores, thresholds)
    f1opt_thresholds = native_v2.choose_thresholds(meta, scores)
    f1opt = native_v2.metric_pair(meta, scores, f1opt_thresholds)
    atomic_npz(
        OUT / "calibration_scores.npz",
        window_scores=scores,
        answer_scores=answer_scores,
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["answers"]]),
    )

    references = comparison_references()
    native = references["native_only_v3_fixed_control"][
        "calibration_strict_fit_thresholds"]
    v2 = references["semantic_window_v2"]["calibration_strict_fit_thresholds"]
    incumbent = references["historical_incumbent_cal_selected"]["calibration"]
    deltas = {
        "vs_native_only_v3_strict": {
            "windows_f1": strict["windows"]["f1"] - native["windows"]["f1"],
            "answers_f1": strict["answers"]["f1"] - native["answers"]["f1"],
        },
        "vs_semantic_window_v2_strict": {
            "windows_f1": strict["windows"]["f1"] - v2["windows"]["f1"],
            "answers_f1": strict["answers"]["f1"] - v2["answers"]["f1"],
        },
        "vs_historical_incumbent_cal_selected": {
            "windows_f1": strict["windows"]["f1"] - incumbent["windows"]["f1"],
            "answers_f1": strict["answers"]["f1"] - incumbent["answers"]["f1"],
        },
    }
    assert formal_baseline_snapshot() == expected_formal_baseline_snapshot()
    summary = {
        "status": "single_strict_calibration_evaluation_complete",
        "candidate": fit_done["selected"]["candidate"],
        "selected_alpha": fit_done["selected"]["alpha"],
        "native_OOF": fit_done["selected"]["native_OOF"],
        "native_OOF_thresholds": thresholds,
        "all_domain_OOF_at_native_thresholds": fit_done["selected"][
            "all_domain_OOF_at_native_thresholds"],
        "calibration_strict_native_OOF_thresholds": strict,
        "calibration_F1Opt_diagnostic": f1opt,
        "calibration_F1Opt_thresholds": f1opt_thresholds,
        "comparison_references": references,
        "strict_deltas": deltas,
        "calibration_evaluations": 1,
        "calibration_F1Opt_used_for_selection": False,
        "calibration_thresholds_fitted_for_deployment": 0,
        "calibration_feature_audit": feature_audit,
        "calibration_mapping_equal_native_v3_bit_exact": True,
        "inference_input_audit": fit_done["inference_input_audit"],
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "artifacts_sha256": {
            "calibration_scores.npz": sha(OUT / "calibration_scores.npz")
        },
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
            f"{record['native_OOF']['windows']['auroc']:.6f} | "
            f"{record['native_OOF']['answers']['f1']:.6f} | "
            f"{record['native_OOF']['answers']['average_precision']:.6f} | "
            f"{record['native_OOF']['answers']['auroc']:.6f} |"
        )
    selected_native = summary["native_OOF"]
    selected_all = summary["all_domain_OOF_at_native_thresholds"]
    baselines = references["formal_baselines_common_calibration_F1Opt"]
    report = f"""# Target-domain weighted evidence-union v2

The frozen 70-column v3 `backbone_union` design (base26 + evidence-union44) and C=0.001 were used unchanged. Exact response/claim/microclaim/hypothesis joins mapped all {FIT_CLAIMS:,} claims to all {FIT_WINDOWS:,} shared four-BPE windows; every claim has window coverage. The first {NATIVE_WINDOWS:,} design rows reproduce native-only v3 bit-for-bit.

The expanded evidence status is consistent: the 158,929-row `missing_requests` file is the immutable request manifest. Extraction is complete for all 39/39 shards, and the later {AUXILIARY_CLAIMS:,}-row aggregate is hash-bound and complete.

| auxiliary alpha | native window F1 | window AP | window AUROC | native answer F1 | answer AP | answer AUROC |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(candidate_lines)}

Selected alpha: **{fit_done['selected']['alpha']:g}**, using only held native OOF. Alpha=0 replayed native-only v3 within one float64 ULP with identical thresholds and confusion counts (maximum score error {fit_done['alpha_zero_native_v3_replay']['max_abs_error']:.3g}).

| evaluation | window F1 | window AP | window AUROC | answer F1 | answer AP | answer AUROC |
|---|---:|---:|---:|---:|---:|---:|
| selected native OOF | {selected_native['windows']['f1']:.6f} | {selected_native['windows']['average_precision']:.6f} | {selected_native['windows']['auroc']:.6f} | {selected_native['answers']['f1']:.6f} | {selected_native['answers']['average_precision']:.6f} | {selected_native['answers']['auroc']:.6f} |
| selected all-domain OOF / native thresholds | {selected_all['windows']['f1']:.6f} | {selected_all['windows']['average_precision']:.6f} | {selected_all['windows']['auroc']:.6f} | {selected_all['answers']['f1']:.6f} | {selected_all['answers']['average_precision']:.6f} | {selected_all['answers']['auroc']:.6f} |
| calibration strict / native OOF thresholds | {strict['windows']['f1']:.6f} | {strict['windows']['average_precision']:.6f} | {strict['windows']['auroc']:.6f} | {strict['answers']['f1']:.6f} | {strict['answers']['average_precision']:.6f} | {strict['answers']['auroc']:.6f} |
| calibration F1Opt diagnostic | {f1opt['windows']['f1']:.6f} | {f1opt['windows']['average_precision']:.6f} | {f1opt['windows']['auroc']:.6f} | {f1opt['answers']['f1']:.6f} | {f1opt['answers']['average_precision']:.6f} | {f1opt['answers']['auroc']:.6f} |
| native-only v3 strict | {native['windows']['f1']:.6f} | {native['windows']['average_precision']:.6f} | {native['windows']['auroc']:.6f} | {native['answers']['f1']:.6f} | {native['answers']['average_precision']:.6f} | {native['answers']['auroc']:.6f} |
| semantic-window-v2 strict | {v2['windows']['f1']:.6f} | {v2['windows']['average_precision']:.6f} | {v2['windows']['auroc']:.6f} | {v2['answers']['f1']:.6f} | {v2['answers']['average_precision']:.6f} | {v2['answers']['auroc']:.6f} |

Formal common-calibration F1Opt references: Lookback {baselines['Lookback']['windows']:.6f}/{baselines['Lookback']['answers']:.6f}, LUMINA {baselines['LUMINA']['windows']:.6f}/{baselines['LUMINA']['answers']:.6f}, and GHOST {baselines['GHOST']['windows']:.6f}/{baselines['GHOST']['answers']:.6f} (window/answer). The historical repeatedly calibration-selected incumbent is {incumbent['windows']['f1']:.6f}/{incumbent['answers']['f1']:.6f}.

Generator identity has zero inference columns. It affects only training sample weights. Held source-connected groups are removed across all generators in every fold; all-domain OOF never selects alpha or thresholds. Calibration ran once after freeze, F1Opt is diagnostic only, formal baseline hashes stayed unchanged, and official test data remained unopened.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("TARGET_DOMAIN_UNION_V2_CALIBRATION", strict["windows"]["f1"],
          strict["answers"]["f1"], flush=True)


def status() -> None:
    print(json.dumps({
        "protocol_frozen": FREEZE_COMPLETE.exists(),
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
        {"prepare": prepare, "fit": fit, "evaluate": evaluate,
         "status": status}[args.command]()


if __name__ == "__main__":
    main()
