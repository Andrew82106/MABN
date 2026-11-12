"""Independent CPU/data-axis audit for atomic_evidence_router_v1."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/atomic_evidence_router_v1"
UPSTREAM = ROOT / "results/atomic_microclaim_nli_v1"
EXPECTED = 11322


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def manifest_map(path, key):
    value = read(path)
    return {str(row["response_id"]): row for row in value[key]}


def main():
    assert not torch.cuda.is_initialized()
    complete = read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_WAIT_UPSTREAM"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    binding = read(OUT / "source_binding.json")
    for name, expected in binding["source_sha256"].items():
        assert sha(Path(name)) == expected, name

    with np.load(OUT / "pooled_token_views.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    assert arrays["harp"].shape == (EXPECTED, 410)
    assert arrays["trace_base"].shape == (EXPECTED, 39)
    assert arrays["nll_claim_ptr"].shape == (EXPECTED + 1,)
    assert arrays["nll_claim_ptr"][-1] == len(arrays["nll_values"]) == 174518
    assert Counter(arrays["partitions"].tolist()) == {"fit": 9055, "calibration": 2267}
    assert all(np.isfinite(arrays[name]).all() for name in ("harp", "trace_base", "nll_values"))

    rows = lines(UPSTREAM / "inputs.jsonl")
    identities = [(str(row["response_id"]), int(claim["claim_id"]), str(claim["microclaim_id"]),
                   str(row["group_id"]), row["partition"])
                  for row in rows for claim in row["claims"]]
    pooled_identities = list(zip(arrays["response_ids"].tolist(), arrays["claim_ids"].tolist(),
                                 arrays["microclaim_ids"].tolist(), arrays["group_ids"].tolist(),
                                 arrays["partitions"].tolist()))
    assert identities == pooled_identities and len(set((x[0], x[1]) for x in identities)) == EXPECTED

    nll_map = manifest_map(ROOT / "data/feature_manifest.json", "records")
    harp_map = manifest_map(ROOT / "data/harp_manifest.json", "records")
    ghost_map = manifest_map(ROOT / "results/ghost_geometry_features_v1/feature_manifest.json", "entries")
    lumina_map = manifest_map(ROOT / "results/lumina_qa_features_v1/feature_manifest.json", "entries")
    answer_ptr = arrays["response_claim_ptr"]
    repooled = []
    for answer_index in (0, len(rows) // 2, len(rows) - 1):
        row = rows[answer_index]; rid = str(row["response_id"])
        claim_index = len(row["claims"]) // 2
        claim = row["claims"][claim_index]
        global_index = int(answer_ptr[answer_index]) + claim_index
        paths = [
            ROOT / "data/features" / f"{rid}.npz",
            ROOT / "data/harp_features" / f"{rid}.npz",
            ROOT / "results/ghost_geometry_features_v1/features" / ghost_map[rid]["file"],
            ROOT / "results/lumina_qa_features_v1/features" / lumina_map[rid]["file"],
        ]
        expected_hashes = [nll_map[rid]["npz_sha256"], harp_map[rid]["npz_sha256"],
                           ghost_map[rid]["npz_sha256"], lumina_map[rid]["npz_sha256"]]
        assert all(sha(path) == expected for path, expected in zip(paths, expected_hashes))
        with np.load(paths[0], allow_pickle=False) as z:
            nll = z["nll"].copy(); ids = z["token_ids"].copy(); offsets = z["response_token_offsets"].copy()
        with np.load(paths[1], allow_pickle=False) as z:
            h = z["harp"][:, :205].copy()
            assert np.array_equal(z["token_ids"], ids) and np.array_equal(z["response_token_offsets"], offsets)
        with np.load(paths[2], allow_pickle=False) as z:
            ghost = z["ghost_features"].copy(); assert np.array_equal(z["token_ids"], ids)
        with np.load(paths[3], allow_pickle=False) as z:
            lumina = z["lumina_features"].copy(); assert np.array_equal(z["token_ids"], ids)
        index = np.asarray(claim["lexical_token_indices"], dtype=np.int64)
        local_h = h[index]
        signed = local_h[np.argmax(np.abs(local_h), axis=0), np.arange(205)]
        expected_harp = np.concatenate((local_h.mean(0), signed)).astype(np.float32)
        one_nll = nll[index]
        expected_head = np.asarray([one_nll.mean(), one_nll.max(), one_nll.min(), one_nll.std()], np.float32)
        left, right = arrays["nll_claim_ptr"][global_index:global_index + 2]
        assert np.array_equal(expected_harp, arrays["harp"][global_index])
        assert np.array_equal(expected_head, arrays["trace_base"][global_index, :4])
        assert np.array_equal(one_nll, arrays["nll_values"][left:right])
        assert ghost[index].shape[1] == 4 and lumina[index].shape[1] == 7
        repooled.append({"response_id": rid, "claim_id": claim_index, "BPEs": len(index)})

    # Independently exercise the exact max-over-owner map for every fit window.
    fit_windows = lines(ROOT / "data/windows_k4_fit.jsonl")
    rows_by_id = {str(row["response_id"]): row for row in rows}
    claim_offset = {}
    cursor = 0
    for row in rows:
        claim_offset[str(row["response_id"])] = cursor; cursor += len(row["claims"])
    dummy = np.linspace(0, 1, EXPECTED, dtype=np.float64)
    projected = np.empty(len(fit_windows), dtype=np.float64)
    for wi, window in enumerate(fit_windows):
        rid = str(window["response_id"]); owners = rows_by_id[rid]["lexical_token_microclaims"]
        claim_ids = {int(cid) for token in window["token_indices"] for cid in owners[token]}
        assert claim_ids
        projected[wi] = max(dummy[claim_offset[rid] + cid] for cid in claim_ids)
    answer_windows = defaultdict(list)
    for wi, window in enumerate(fit_windows):
        answer_windows[str(window["response_id"])].append(wi)
    fit_answers = lines(ROOT / "data/answers_fit.jsonl")
    answer_scores = [max(projected[answer_windows[str(row["response_id"])]]) for row in fit_answers]
    assert len(projected) == 168123 and len(answer_scores) == 634 and np.isfinite(projected).all()

    readiness = read(OUT / "readiness.json")
    missing = [name for name in ("complete.json", "evidence_features.npy", "feature_names.json",
                                 "microclaim_labels.npy", "summary.json") if not (UPSTREAM / name).exists()]
    assert readiness["status"] == "WAIT_UPSTREAM" and readiness["missing"] == missing
    report = {
        "status": "passed_WAIT_UPSTREAM", "pooled_claims": EXPECTED,
        "independent_exact_repool": repooled,
        "fit_four_BPE_windows_mapped": len(projected), "fit_answers_mapped": len(answer_scores),
        "claim_identity_order_exact": True, "cross_cache_token_axes_checked": True,
        "upstream_missing": missing, "metrics_emitted": False,
        "labels_used_for_audit_decision": False, "GPU_used": False, "official_test_opened": False,
    }
    atomic_json(OUT / "INDEPENDENT_CPU_AUDIT.json", report)
    assert not torch.cuda.is_initialized()
    print("ATOMIC_EVIDENCE_ROUTER_INDEPENDENT_CPU_AUDIT_PASSED_WAIT_UPSTREAM")


if __name__ == "__main__":
    main()
