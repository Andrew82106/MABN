"""Independent CPU replay of the within-answer contrastive-v5 pair manifest."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "results/atomic_microclaim_relation_expanded_v4/examples.jsonl"
OUT = ROOT / "results/microclaim_within_answer_contrastive_v5"
FIT = 34_919
FOLDS = 5
STOP = set(
    "a an the and or but if to of in on at for from with by as is are was were be been being "
    "it its this that these those some any may can could would should has have had do does did "
    "according based passage passages provided given seems here there their they them he she his "
    "her we us our you your i me my also however therefore thus".split()
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_fit():
    rows = []
    with SOURCE.open(encoding="utf-8") as handle:
        for index in range(FIT):
            row = json.loads(handle.readline())
            assert row["partition"] == "fit" and row["example_index"] == index
            rows.append(row)
    return rows


def words(text):
    values = (token.lower().replace("’", "'") for token in WORD_RE.findall(text))
    return frozenset(token for token in values if token not in STOP)


def trigrams(text):
    value = " ".join(token.lower().replace("’", "'") for token in WORD_RE.findall(text))
    return frozenset(value[index:index + 3] for index in range(max(0, len(value) - 2)))


def dice(left, right):
    denominator = len(left) + len(right)
    return 2 * len(left & right), denominator if denominator else 1


def evidence(row):
    return frozenset(sentence["text_sha256"] for passage in row["evidence"]
                     for sentence in passage["selected"])


def key_and_values(positive, negative):
    wn, wd = dice(words(positive["hypothesis"]), words(negative["hypothesis"]))
    cn, cd = dice(trigrams(positive["hypothesis"]), trigrams(negative["hypothesis"]))
    same = int(positive["parent_claim_index"] == negative["parent_claim_index"])
    shared = len(evidence(positive) & evidence(negative))
    gap = max(0, max(int(negative["char_start"]) - int(positive["char_end"]),
                     int(positive["char_start"]) - int(negative["char_end"])))
    key = (same, Fraction(wn, wd), Fraction(cn, cd), shared, -gap,
           -int(negative["example_index"]))
    return key, (same, wn, wd, cn, cd, shared, gap)


def main():
    assert not (OUT / "INDEPENDENT_AUDIT.json").exists()
    prep = json.loads((OUT / "preparation_complete.json").read_text(encoding="utf-8"))
    for name, expected in prep["files_sha256"].items():
        assert hash_file(OUT / name) == expected
    assert prep["calibration_rows_parsed"] == 0 and prep["GPU_used"] is False
    snapshot = json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))
    for path, expected in snapshot.items():
        assert hash_file(path) == expected, path
    rows = read_fit(); by_answer = defaultdict(list)
    for row in rows:
        by_answer[row["response_id"]].append(row)
    pairs = [json.loads(line) for line in (OUT / "pairs.jsonl").open(encoding="utf-8")]
    with np.load(OUT / "pair_arrays.npz", allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    assert [row["pair_index"] for row in pairs] == list(range(len(pairs)))
    assert len(set((row["positive_index"], row["negative_index"]) for row in pairs)) == len(pairs)
    artifact_by_positive = defaultdict(list)
    for pair in pairs:
        positive, negative = rows[pair["positive_index"]], rows[pair["negative_index"]]
        assert positive["gold_label"] == 1 and negative["gold_label"] == 0
        assert len(words(negative["hypothesis"])) >= 2
        for field in ("response_id", "source_id", "group_id", "held_fold"):
            assert positive[field] == negative[field] == pair[field]
        key, detail = key_and_values(positive, negative)
        same, wn, wd, cn, cd, shared, gap = detail
        assert (pair["same_parent"], pair["word_dice_numerator"], pair["word_dice_denominator"],
                pair["char_trigram_dice_numerator"], pair["char_trigram_dice_denominator"],
                pair["shared_selected_evidence_sentences"], pair["answer_span_gap_chars"]) == detail
        assert abs(pair["word_dice"] - wn / wd) < 1e-15
        assert abs(pair["char_trigram_dice"] - cn / cd) < 1e-15
        artifact_by_positive[pair["positive_index"]].append((pair["rank_within_positive"],
                                                               pair["negative_index"], key))
    # Replay top-2 selection without importing the producer.
    expected_pairable = set()
    for claims in by_answer.values():
        negatives = [row for row in claims if row["gold_label"] == 0 and len(words(row["hypothesis"])) >= 2]
        for positive in (row for row in claims if row["gold_label"] == 1):
            if not negatives:
                assert positive["example_index"] not in artifact_by_positive
                continue
            expected_pairable.add(positive["example_index"])
            candidates = sorted(((key_and_values(positive, negative)[0], negative["example_index"])
                                 for negative in negatives), reverse=True)
            expected = [index for _, index in candidates[:2]]
            actual_rows = sorted(artifact_by_positive[positive["example_index"]])
            actual = [index for _, index, _ in actual_rows]
            assert actual == expected, (positive["example_index"], actual, expected)
            assert [rank for rank, _, _ in actual_rows] == list(range(1, len(expected) + 1))
    assert set(artifact_by_positive) == expected_pairable
    assert np.array_equal(arrays["positive_index"], [row["positive_index"] for row in pairs])
    assert np.array_equal(arrays["negative_index"], [row["negative_index"] for row in pairs])
    assert np.array_equal(arrays["held_fold"], [row["held_fold"] for row in pairs])
    assert arrays["rank_weights"].shape == (FOLDS + 1, len(pairs))
    group_mass_ranges = []
    for fold in range(FOLDS + 1):
        active = np.ones(len(pairs), dtype=bool) if fold == FOLDS else arrays["held_fold"] != fold
        weights = arrays["rank_weights"][fold]
        assert np.all(weights[~active] == 0) and np.isclose(weights[active].sum(), 1, atol=2e-7, rtol=0)
        by_group_weight = defaultdict(float); by_group_positive = defaultdict(lambda: defaultdict(float))
        for index in np.flatnonzero(active):
            pair = pairs[int(index)]; weight = float(weights[index])
            by_group_weight[pair["group_id"]] += weight
            by_group_positive[pair["group_id"]][pair["positive_index"]] += weight
        group_values = list(by_group_weight.values())
        assert max(group_values) - min(group_values) < 3e-8
        for positive_values in by_group_positive.values():
            values = list(positive_values.values())
            assert max(values) - min(values) < 3e-8
        group_mass_ranges.append(max(group_values) - min(group_values))
    stats = json.loads((OUT / "statistics.json").read_text(encoding="utf-8"))
    assert stats["pairs"] == len(pairs)
    assert stats["pairable_positive_claims"] == len(expected_pairable)
    assert stats["fold_pair_counts"] == {str(key): value for key, value in sorted(
        Counter(row["held_fold"] for row in pairs).items())}
    tiny = json.loads((OUT / "CPU_TINY_TRAIN.json").read_text(encoding="utf-8"))
    check = json.loads((OUT / "CPU_CHECK.json").read_text(encoding="utf-8"))
    assert tiny["status"] == "CPU_tiny_combined_loss_passed" and tiny["GPU_used"] is False
    assert check["status"] == "passed" and check["formal_baselines_modified"] is False
    result = {
        "status": "independent_CPU_audit_passed", "pairs": len(pairs),
        "pairable_positive_claims": len(expected_pairable),
        "exact_top2_replay": True, "same_answer_source_group_fold": True,
        "opposite_fit_labels": True, "maximum_endpoint_index": int(max(
            max(arrays["positive_index"]), max(arrays["negative_index"]))),
        "rank_weight_sum_each_split": True,
        "maximum_group_mass_range": max(group_mass_ranges),
        "calibration_rows_parsed": 0, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
        "audited_files_sha256": {name: hash_file(OUT / name) for name in
                                 ("pairs.jsonl", "pair_arrays.npz", "statistics.json",
                                  "training_plan.json", "CPU_CHECK.json", "CPU_TINY_TRAIN.json")},
    }
    pending = OUT / "INDEPENDENT_AUDIT.json.pending"
    pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUT / "INDEPENDENT_AUDIT.json")
    print("WITHIN_ANSWER_CONTRASTIVE_V5_INDEPENDENT_AUDIT_PASSED", len(pairs))


if __name__ == "__main__":
    main()
