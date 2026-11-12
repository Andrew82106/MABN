"""Independent CPU-only audit of the relation-only FAVA transfer preparation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/relation_pair_fast_v1"
OLD = ROOT / "results/local_pair_transfer_v1"
TOKENS = ROOT / "auxiliary_fava_local_pairs_v1/tokenization_v1"
BASE = ROOT / "results/full_context_encoder_v2"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    prepared = read(OUT / "preparation_complete.json")
    protocol = read(OUT / "protocol.json")
    resources = read(OUT / "WEIGHTS_AND_RESOURCE_PLAN.json")
    old = read(OLD / "preparation_complete.json")
    index = read(OUT / "pair_index.json")["pairs"]

    assert prepared["status"] == "prepared_not_trained"
    assert prepared["variants"] == ["relation_rank025"]
    assert len(index) == prepared["cohort"]["eligible_pairs"] == 4704
    assert all(row["type"] == "relation" for row in index)
    assert all(row["eligible_for_future_local_pair_training"] for row in index)
    assert len({row["pair_id"] for row in index}) == len(index)

    with np.load(OUT / "pair_weights.npz", allow_pickle=False) as z:
        weights = z["base"].copy()
    assert len(weights) == len(index) and np.isfinite(weights).all() and (weights > 0).all()
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    tree = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(index):
        tree[row["group_id"]][row["source_response_id"]].append(i)
    group_mass = []
    answer_mass_error = 0.0
    for answers in tree.values():
        ix_group = [i for ix in answers.values() for i in ix]
        group_mass.append(float(weights[ix_group].sum()))
        for ix in answers.values():
            expected = 1.0 / len(tree) / len(answers)
            answer_mass_error = max(answer_mass_error, abs(float(weights[ix].sum()) - expected))
    assert max(group_mass) - min(group_mass) < 1e-12
    assert answer_mass_error < 1e-12

    # Descriptive edit cues only. They never select rows or set a loss weight.
    cues = {
        "negation": r"\b(?:no|not|never|none|without|cannot|neither|nor)\b",
        "quantifier": r"\b(?:all|both|each|every|only|any|some|none|most|least|more|less|fewer)\b",
        "temporal_order": r"\b(?:before|after|again|another|first|second|third|then|next|previous|prior|later|earlier|during|while|until|step)\b",
        "comparison": r"\b(?:more|less|fewer|greater|smaller|larger|higher|lower|better|worse|faster|slower|same|different)\b",
        "causal": r"\b(?:cause|causes|caused|because|prevent|prevents|prevented|lead|leads|led|result|results|due)\b",
        "numeric": r"\d",
    }
    cue_counts = Counter()
    for row in index:
        target_text = " ".join(row[side]["target"]["text"] for side in ("original", "preferred")).casefold()
        hits = [name for name, pattern in cues.items() if re.search(pattern, target_text)]
        for name in hits:
            cue_counts[name] += 1
        if not hits:
            cue_counts["other_author_relation_edit"] += 1

    for name, digest in prepared["files_sha256"].items():
        assert sha(OUT / name) == digest, name
    qa_reuse = {}
    for new_name, base_name in (("qa_inputs.jsonl", "inputs.jsonl"),
                                ("qa_training_weights.npz", "training_weights.npz"),
                                ("qa_answer_orders.npy", "answer_orders.npy")):
        qa_reuse[new_name] = sha(OUT / new_name) == sha(BASE / base_name)
    assert all(qa_reuse.values())

    old_pairs = old["cohort"]["eligible_pairs"]
    old_tokens = old["cohort"]["encoder_input_tokens"]
    report = {
        "status": "passed_CPU_only",
        "cohort": {
            "eligible_relation_pairs": len(index),
            "material_groups": len(tree),
            "original_answers": sum(len(v) for v in tree.values()),
            "auxiliary_input_tokens": prepared["cohort"]["encoder_input_tokens"],
            "fraction_of_old_pairs": len(index) / old_pairs,
            "fraction_of_old_aux_tokens": prepared["cohort"]["encoder_input_tokens"] / old_tokens,
            "cue_counts_descriptive_only": dict(cue_counts),
        },
        "weights": {
            "sum": float(weights.sum()),
            "equal_group_mass_max_minus_min": max(group_mass) - min(group_mass),
            "answer_mass_max_error": answer_mass_error,
        },
        "QA_inputs_weights_orders_byte_identical": qa_reuse,
        "single_variant": protocol["variants"],
        "runtime_minutes_empirical_estimate": resources["expected_runtime_minutes"],
        "runtime_basis": resources["same_host_completed_run_basis"],
        "GPU_used": False,
        "trained": False,
        "official_test_opened": False,
        "baseline_files_modified": False,
    }
    (OUT / "INDEPENDENT_CPU_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("RELATION_PAIR_FAST_INDEPENDENT_CPU_AUDIT_PASSED")


if __name__ == "__main__":
    main()
