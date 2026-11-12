"""Independent CPU audit of the native exact evidence-union artifacts."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/evidence_union_nli_v2"
RUNNER = ROOT / "src/run_evidence_union_nli_v2.py"
COHORTS = {"fit_native": (634, 9055), "calibration": (159, 2267)}
MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_scope(feature: np.ndarray, probability: np.ndarray,
                 mask: np.ndarray, offset: int) -> None:
    chosen = probability[mask]
    maxima = chosen.max(axis=0)
    means = chosen.mean(axis=0, dtype=np.float64)
    best_e = chosen[int(np.argmax(chosen[:, 0]))]
    best_c = chosen[int(np.argmax(chosen[:, 2]))]
    expected = np.asarray([
        len(chosen), *maxima, *means, 1 - maxima[0],
        max(maxima[2], 1 - maxima[0]), maxima[0] - maxima[2],
        best_e[0] - max(best_e[1], best_e[2]),
        best_c[2] - max(best_c[0], best_c[1]),
    ], dtype=np.float32)
    assert np.array_equal(feature[offset:offset + 12], expected)


def main() -> None:
    names = read(OUT / "aggregate_feature_names.json")
    assert names["width"] == len(names["names"]) == len(set(names["names"])) == 44
    runner_sha = sha(RUNNER)
    cohort_results = {}
    for cohort, (answers_expected, claims_expected) in COHORTS.items():
        manifest = read(OUT / f"manifest_{cohort}.json")
        diagnostic = read(OUT / f"diagnostics_{cohort}.json")
        candidate_path = OUT / f"candidates_{cohort}.jsonl"
        array_path = OUT / f"candidate_arrays_{cohort}.npz"
        aggregate_path = OUT / f"claim_aggregates_{cohort}.npz"
        assert manifest["answers"] == answers_expected
        assert manifest["claims"] == diagnostic["claims"] == claims_expected
        assert manifest["existing_cache_hit_instances"] == manifest["candidate_instances"]
        assert manifest["missing_cache_instances"] == manifest["unique_missing_requests"] == 0
        assert manifest["source_snapshot"]["runner_sha256"] == runner_sha
        assert manifest["source_snapshot"]["model"]["revision"] == MODEL_REVISION
        assert manifest["source_snapshot"]["model"]["weights_sha256"] == MODEL_SHA256
        assert manifest["candidate_files"]["jsonl"]["sha256"] == sha(candidate_path)
        assert manifest["candidate_files"]["arrays"]["sha256"] == sha(array_path)
        assert diagnostic["aggregate_interface"]["features_sha256"] == sha(aggregate_path)
        rows = lines(candidate_path)
        assert len(rows) == answers_expected
        with np.load(array_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key].copy() for key in loaded.files}
        with np.load(aggregate_path, allow_pickle=False) as loaded:
            aggregate = {key: loaded[key].copy() for key in loaded.files}
        assert arrays["probabilities"].shape == (manifest["candidate_instances"], 3)
        assert np.isfinite(arrays["probabilities"]).all()
        assert np.allclose(arrays["probabilities"].sum(1), 1, rtol=0, atol=2e-6)
        assert aggregate["features"].shape == (claims_expected, 44)
        assert aggregate["features"].dtype == np.float32
        assert aggregate["feature_names_sha256"].item() == sha(
            OUT / "aggregate_feature_names.json")
        claim_cursor = candidate_cursor = 0
        histogram = Counter()
        portable_conflicts_bound_to_occurrence = 0
        for answer_index, answer in enumerate(rows):
            assert answer["answer_index"] == answer_index
            assert answer["labels_used"] is False and answer["official_test_opened"] is False
            for claim in answer["claims"]:
                left, right = map(int, arrays["claim_indptr"][[claim_cursor, claim_cursor + 1]])
                assert left == candidate_cursor and right - left == claim["candidate_count"]
                assert aggregate["response_index"][claim_cursor] == answer_index
                assert aggregate["claim_id"][claim_cursor] == claim["claim_id"]
                assert aggregate["microclaim_index"][claim_cursor] == claim["microclaim_index"]
                assert aggregate["hypothesis_identity_sha256"][claim_cursor].tobytes().hex() == claim["hypothesis_sha256"]
                local = arrays["probabilities"][left:right]
                attention = arrays["attention_rank"][left:right] > 0
                bm25 = arrays["bm25_rank"][left:right] > 0
                assert attention.sum() == min(3, len(attention))
                assert bm25.any()
                feature = aggregate["features"][claim_cursor]
                verify_scope(feature, local, np.ones(len(local), dtype=bool), 0)
                verify_scope(feature, local, attention, 12)
                verify_scope(feature, local, bm25, 24)
                expected_tail = np.asarray([
                    feature[1] - feature[13], feature[1] - feature[25],
                    feature[3] - feature[15], feature[3] - feature[27],
                    feature[1] == feature[13], feature[1] == feature[25],
                    feature[3] == feature[15], feature[3] == feature[27],
                ], dtype=np.float32)
                assert np.array_equal(feature[36:], expected_tail)
                for local_index, candidate in enumerate(claim["candidates"]):
                    index = left + local_index
                    assert arrays["request_identity_sha256"][index].tobytes().hex() == candidate["request_id"]
                    assert candidate["cache_status"] == "existing_occurrence_or_exact"
                    if candidate["portable_reuse_status"] == "portable_conflict_recompute":
                        portable_conflicts_bound_to_occurrence += 1
                        assert "portable_exact_existing_cache" not in candidate["cache_sources"]
                        assert candidate["cache_sources"]
                histogram[len(local)] += 1
                candidate_cursor = right
                claim_cursor += 1
        assert claim_cursor == claims_expected
        assert candidate_cursor == manifest["candidate_instances"]
        assert {str(key): value for key, value in sorted(histogram.items())} == manifest[
            "candidate_count_per_claim_histogram"]
        assert sum(manifest["portable_reuse_instance_status"].values()) == candidate_cursor
        assert portable_conflicts_bound_to_occurrence == manifest[
            "portable_reuse_instance_status"].get("portable_conflict_recompute", 0)
        cohort_results[cohort] = {
            "answers": answers_expected, "claims": claims_expected,
            "candidate_instances": candidate_cursor,
            "candidate_histogram": manifest["candidate_count_per_claim_histogram"],
            "portable_conflicts_not_reused": portable_conflicts_bound_to_occurrence,
            "aggregate_shape": list(aggregate["features"].shape),
        }
    freeze = read(OUT / "fit_rule_freeze.json")
    calibration = read(OUT / "diagnostics_calibration.json")
    assert freeze["status"] == "frozen_on_fit_before_calibration_diagnostic"
    assert calibration["fit_rule_freeze_sha256"] == sha(OUT / "fit_rule_freeze.json")
    assert calibration["rule_selected_or_changed_on_calibration"] is False
    result = {
        "status": "passed",
        "runner_sha256": runner_sha,
        "cohorts": cohort_results,
        "candidate_identity_and_ENC_alignment": True,
        "all_44_aggregate_columns_recomputed": True,
        "portable_conflicts_not_reused": True,
        "fit_rule_precedes_calibration_diagnostic": True,
        "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
        "model_loaded": False, "GPU_used": False, "labels_used": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    }
    destination = OUT / "INDEPENDENT_CPU_AUDIT.json"
    assert not destination.exists()
    pending = destination.with_suffix(".json.pending")
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
