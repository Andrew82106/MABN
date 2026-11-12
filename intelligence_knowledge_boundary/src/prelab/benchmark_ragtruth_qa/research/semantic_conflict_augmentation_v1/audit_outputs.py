"""Independent structural replay of the fit-only silver-pair artifacts."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import build_fit_only as build


HERE = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    source = {}
    for owner in build.read_fit():
        for passage in build.parse_passages(owner["retrieved_passages"]):
            for sid, sentence in enumerate(passage["sentences"]):
                source[(owner["group_id"], str(owner["response_id"]), passage["passage_id"], sid)] = sentence

    full = rows(HERE / "pairs_fit_silver.jsonl")
    strict = rows(HERE / "pairs_fit_silver_strict.jsonl")
    qc = rows(HERE / "RULE_QC_AUDIT_MAX100.jsonl")
    qualitative = rows(HERE / "QUALITATIVE_QC_AUDIT.jsonl")
    assert len(full) == 5067 and len(strict) == 4109 and len(qc) == len(qualitative) == 92
    assert len({x["pair_id"] for x in full}) == len(full)
    full_index = {x["pair_id"]: x for x in full}
    strict_ids = {x["pair_id"] for x in strict}
    assert strict_ids <= set(full_index)
    assert {x["pair_id"] for x in qc} <= strict_ids
    assert {x["pair_id"] for x in qualitative} == {x["pair_id"] for x in qc}

    hashes_to_components = defaultdict(set)
    type_counts = Counter()
    for pair in full:
        assert pair["partition"] == "fit"
        key = (pair["group_id"], pair["response_id"], pair["passage_id"], pair["sentence_id"])
        assert source[key] == pair["source_sentence"]
        edit = pair["edit"]
        replay = build.canonical(pair["supported_claim"][:edit["start"]] + edit["replacement"] + pair["supported_claim"][edit["end"]:])
        assert replay == pair["corrupted_claim"]
        assert pair["labels"]["label_level"] == "silver_not_human_gold"
        assert pair["supported_claim"] != pair["corrupted_claim"]
        assert all(pair["checks"].values())
        assert pair["group_id"] in pair["split_component_groups"]
        donor = pair["extra"].get("donor_group_id")
        if donor:
            assert donor in pair["split_component_groups"]
        if pair["strict_rule_eligible"]:
            assert not pair["strict_exclusion_reasons"]
        else:
            assert pair["strict_exclusion_reasons"]
    for pair in strict:
        assert pair["strict_rule_eligible"]
        type_counts[pair["corruption_type"]] += 1
        hashes_to_components[pair["source_sentence_sha256"]].add(pair["split_group_id"])
    assert sum(len(v) > 1 for v in hashes_to_components.values()) == 0

    result = {
        "status": "pass_fit_only_structural_replay",
        "full_pairs": len(full),
        "strict_pairs": len(strict),
        "strict_by_type": dict(type_counts),
        "post_freeze_qc_rows": len(qc),
        "qualitative_decisions": dict(Counter(x["decision"] for x in qualitative)),
        "duplicate_support_hashes_across_components": 0,
        "calibration_read": False,
        "test_read": False,
        "gpu_used": False,
        "trained": False,
        "sha256": {
            name: digest(HERE / name) for name in (
                "pairs_fit_silver.jsonl", "pairs_fit_silver_strict.jsonl",
                "RULE_QC_AUDIT_MAX100.jsonl", "QUALITATIVE_QC_AUDIT.jsonl",
                "STATS.json", "FIT_CONFLICT_AUDIT.json", "PROTOCOL.json",
            )
        },
    }
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
