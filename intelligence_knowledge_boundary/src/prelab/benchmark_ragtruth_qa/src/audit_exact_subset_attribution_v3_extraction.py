"""Independent read-only audit of all v3 exact-subset pilot GPU caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
SELECTED = OUT / "selected_prepared_inputs.jsonl"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def shapley_efficiency(logp):
    values = np.asarray(logp, dtype=np.float64)
    phi = np.zeros((3, values.shape[1]), dtype=np.float64)
    # n=3 Shapley weights: |S|=0 -> 1/3, |S|=1 -> 1/6, |S|=2 -> 1/3.
    for source in range(3):
        bit = 1 << source
        for mask in range(8):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = (1 / 3, 1 / 6, 1 / 3)[size]
            phi[source] += weight * (values[mask | bit] - values[mask])
    return float(np.max(np.abs(phi.sum(axis=0) - (values[7] - values[0]))))


def main():
    complete = read(OUT / "extraction_complete.json")
    smoke = read(OUT / "GPU_SMOKE.json")
    preparation = read(OUT / "preparation_complete.json")
    rows = list(lines(SELECTED))
    assert complete["status"] == "complete_frozen_fit_pilot_logprob_not_scored"
    assert complete["answers"] == len(rows) == 256
    assert complete["selected_prepared_sha256"] == sha(SELECTED)
    assert complete["runtime_signature_sha256"] == smoke["runtime_signature_sha256"]
    records = complete["records"]
    assert [item["response_id"] for item in records] == [row["response_id"] for row in rows]
    assert len({item["response_id"] for item in records}) == 256
    max_efficiency = 0.0
    logp_min = float("inf")
    logp_max = float("-inf")
    raw_tokens = 0
    total_values = 0
    for row, record in zip(rows, records):
        response_id = row["response_id"]
        path = OUT / "token_logprob" / f"{response_id}.npz"
        metadata_path = path.with_suffix(".json")
        metadata = read(metadata_path)
        assert sha(path) == record["npz_sha256"] == metadata["npz_sha256"]
        assert sha(metadata_path) == record["metadata_sha256"]
        assert metadata["complete"] is True and metadata["response_id"] == response_id
        expected_signature = digest({
            "runtime": smoke["runtime_signature"],
            "response_id": response_id,
            "answer_token_ids_sha256": row["answer_token_ids_sha256"],
            "citation_mask_sha256": row["citation_geometry"]["citation_mask_sha256"],
            "view_input_sha256": [view["input_ids_sha256"] for view in row["views"]],
        })
        assert metadata["cache_signature"] == expected_signature
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {"selected_logprob", "answer_token_ids", "citation_mask"}
            logp = archive["selected_logprob"].copy()
            answer_ids = archive["answer_token_ids"].copy()
            citation_mask = archive["citation_mask"].copy()
        tokens = len(row["answer_token_ids"])
        assert logp.shape == (8, tokens) and logp.dtype == np.float32
        assert np.isfinite(logp).all() and np.all(logp <= 1e-5)
        assert answer_ids.dtype == np.int32 and answer_ids.tolist() == row["answer_token_ids"]
        assert citation_mask.dtype == np.uint8
        assert citation_mask.tolist() == row["citation_geometry"]["citation_mask"]
        efficiency = shapley_efficiency(logp)
        assert efficiency <= 1e-12
        assert abs(efficiency - metadata["shapley_efficiency_max_abs"]) <= 1e-12
        assert metadata["fit_gold_files_opened"] is False
        assert metadata["labels_accessed"] is False
        assert metadata["official_test_opened"] is False
        max_efficiency = max(max_efficiency, efficiency)
        logp_min = min(logp_min, float(logp.min()))
        logp_max = max(logp_max, float(logp.max()))
        raw_tokens += tokens
        total_values += logp.size
    assert raw_tokens == read(OUT / "preparation_statistics.json")["raw_answer_tokens"] == 46481
    assert total_values == raw_tokens * 8
    assert complete["fit_gold_files_opened"] is False
    assert complete["labels_accessed"] is False
    assert complete["official_test_opened"] is False
    assert preparation["selected_answers"] == 256
    report = {
        "status": "passed_independent_read_only_full_extraction_audit",
        "audit_source_sha256": sha(__file__),
        "extraction_complete_sha256": sha(OUT / "extraction_complete.json"),
        "selected_prepared_sha256": sha(SELECTED),
        "counts": {
            "answers": len(rows),
            "raw_answer_tokens": raw_tokens,
            "subset_logprob_values": total_values,
            "cache_npz": len(list((OUT / "token_logprob").glob("*.npz"))),
            "cache_json": len(list((OUT / "token_logprob").glob("*.json"))),
        },
        "numeric": {
            "selected_logprob_min": logp_min,
            "selected_logprob_max": logp_max,
            "independent_shapley_efficiency_max_abs": max_efficiency,
        },
        "checks": {
            "all_256_record_hashes_and_metadata_hashes_exact": True,
            "all_cache_signatures_independently_recomputed": True,
            "all_answer_ids_and_citation_masks_exact": True,
            "all_eight_view_arrays_finite": True,
            "Shapley_efficiency_independently_recomputed": True,
            "runtime_signature_matches_smoke": True,
        },
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "official_test_opened": False,
        "paper_baseline_modified_or_scored": False,
    }
    save(OUT / "EXTRACTION_INDEPENDENT_AUDIT.json", report)
    print("EXACT_SUBSET_V3_FULL_EXTRACTION_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
