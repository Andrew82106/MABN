"""Independent CPU audit for microclaim_crossencoder_v1 preparation."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_development as q
import run_atomic_microclaim_nli_v1 as atomic
import run_microclaim_crossencoder_v1 as candidate


def main():
    assert not __import__("torch").cuda.is_initialized()
    records, arrays = candidate.check_prepared()
    source_rows = q.lines(candidate.ATOMIC_INPUT)
    by_id = {row["response_id"]: row for row in source_rows}
    response_ids = np.asarray([row["response_id"] for row in records])
    labels = arrays["labels"]
    fit = np.arange(candidate.EXPECTED_CLAIMS["fit"], dtype=np.int64)
    assertions = Counter()

    # Exact one-to-one claim ordering and geometry provenance.
    cursor = 0
    for answer in source_rows:
        for claim in answer["claims"]:
            record = records[cursor]
            assert record["response_id"] == answer["response_id"]
            assert record["claim_id"] == claim["claim_id"]
            assert record["microclaim_id"] == claim["microclaim_id"]
            assert record["lexical_token_indices"] == claim["lexical_token_indices"]
            cursor += 1; assertions["claim_alignment"] += 1
    assert cursor == len(records)

    # Rebuild the pre-hardness fold weights. Positive weights must be bitwise
    # untouched; each answer's total clean mass must be preserved.
    max_positive_delta = max_negative_answer_mass_delta = 0.0
    changed_clean = unchanged_positive = 0
    for fold in range(candidate.FOLDS + 1):
        active = fit if fold == candidate.FOLDS else fit[arrays["fold_assignment"][fit] != fold]
        base = atomic.claim_weights(by_id, response_ids, labels, active)
        final = arrays["weights"][fold].astype(np.float64)
        pos = active[labels[active] == 1]
        delta = float(np.max(np.abs(final[pos] - base[pos])))
        max_positive_delta = max(max_positive_delta, delta)
        # Prepared training weights are intentionally stored as float32.
        assert delta < 1e-6; unchanged_positive += len(pos)
        tree = defaultdict(list)
        for index in active: tree[response_ids[index]].append(int(index))
        for indices in tree.values():
            neg = [i for i in indices if labels[i] == 0]
            if not neg: continue
            mass_delta = abs(float(final[neg].sum() - base[neg].sum()))
            max_negative_answer_mass_delta = max(max_negative_answer_mass_delta, mass_delta)
            assert mass_delta < 2e-6
            changed_clean += sum(abs(final[i] - base[i]) > 1e-8 for i in neg)
    assert changed_clean > 0

    # Every fold is genuinely source-connected OOF.
    leakage = q.read(candidate.OUT / "preparation.json")["fold_leakage_audit"]
    assert len(leakage) == candidate.FOLDS
    assert all(all(row[key] == 0 for key in ("groups_overlap", "sources_overlap", "responses_overlap",
                                              "answer_hashes_overlap", "passage_body_hash_overlap")) for row in leakage)

    # Sequence storage is complete and truncation-free.
    assert int(arrays["lengths"].max()) == 697 < candidate.MAX_LENGTH
    assert arrays["flat_input_ids"].size == int(arrays["lengths"].sum())
    assert np.all(arrays["flat_input_ids"][arrays["bounds"][:, 0]] == 50281)
    assert np.all(arrays["flat_input_ids"][arrays["bounds"][:, 1] - 1] == 50282)

    output = {
        "status": "passed",
        "claims_checked": len(records),
        "fold_weight_sets_checked": candidate.FOLDS + 1,
        "positive_weight_entries_unchanged": int(unchanged_positive),
        "max_positive_weight_abs_delta_after_hard_negative_redistribution": max_positive_delta,
        "clean_weight_entries_changed": int(changed_clean),
        "max_per_answer_clean_mass_abs_delta": max_negative_answer_mass_delta,
        "source_connected_fold_overlap_zero": True,
        "all_full_evidence_pair_lengths_below_768": True,
        "official_test_opened": False,
        "candidate_source_sha256": q.sha(candidate.__file__),
        "preparation_sha256": q.sha(candidate.OUT / "preparation_complete.json"),
    }
    path = candidate.OUT / "INDEPENDENT_CPU_AUDIT.json"
    assert not path.exists()
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MICROCLAIM_CROSSENCODER_INDEPENDENT_CPU_AUDIT_PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()
