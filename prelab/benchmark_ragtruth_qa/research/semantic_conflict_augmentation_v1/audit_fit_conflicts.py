"""Exact fit-only census of RAGTruth QA conflict supervision."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import build_fit_only as rules


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIT = ROOT / "data/fit.jsonl"
TOKENS = ROOT / "data/tokens_fit.jsonl"
WINDOWS = ROOT / "data/windows_k4_fit.jsonl"
OUT = HERE / "FIT_CONFLICT_AUDIT.json"
CONFLICT = {"Evident Conflict", "Subtle Conflict"}
BASELESS = {"Evident Baseless Info", "Subtle Baseless Info"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    labels = {}
    groups = set()
    span_counts = Counter()
    conflict_surface_cues = Counter()
    answer_types = defaultdict(set)
    with FIT.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["partition"] == "fit"
            labels[str(row["response_id"])] = row["labels"]
            groups.add(row["group_id"])
            span_counts.update(x["label_type"] for x in row["labels"])
            answer_types[str(row["response_id"])].update(x["label_type"] for x in row["labels"])
            for label in row["labels"]:
                if label["label_type"] not in CONFLICT:
                    continue
                text = label["text"]
                conflict_surface_cues["number"] += bool(rules.NUMBER.search(text))
                conflict_surface_cues["temporal"] += bool(rules.YEAR.search(text) or rules.TIME.search(text) or rules.MONTH.search(text) or rules.WEEKDAY.search(text) or rules.DURATION.search(text))
                conflict_surface_cues["negation"] += bool(rules.NEG_ANY.search(text))
                conflict_surface_cues["passage_attribution"] += bool(re.search(r"\bpassage\s*[123]\b", text, re.I))
                conflict_surface_cues["proper_entity_proxy"] += rules.has_surface_entity_proxy(text)
    assert len(labels) == 634

    typed_tokens = {}
    with TOKENS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["partition"] == "fit"
            rid = str(row["response_id"])
            assert len(row["original_labels"]) == len(row["span_token_mapping"]) == len(labels[rid])
            by_type = defaultdict(set)
            for mapping, label in zip(row["span_token_mapping"], labels[rid]):
                assert mapping["span_index"] == label["span_index"] if "span_index" in label else True
                by_type[label["label_type"]].update(mapping["risk_token_indices"])
            typed_tokens[rid] = by_type
    assert len(typed_tokens) == 634

    window_counts = Counter()
    answer_window_types = defaultdict(set)
    n = 0
    with WINDOWS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["partition"] == "fit"
            if not row["eligible"]:
                continue
            n += 1
            rid = str(row["response_id"])
            ids = set(row["token_indices"])
            present = {kind for kind, values in typed_tokens[rid].items() if ids & values}
            for kind in present:
                window_counts[kind] += 1
            is_conflict = bool(present & CONFLICT)
            is_baseless = bool(present & BASELESS)
            window_counts["any_conflict"] += is_conflict
            window_counts["any_baseless"] += is_baseless
            window_counts["conflict_and_baseless"] += is_conflict and is_baseless
            window_counts["any_risk_rebuilt"] += bool(present)
            window_counts["positive_label_field"] += int(row["label"])
            if is_conflict:
                answer_window_types[rid].add("conflict")
            if is_baseless:
                answer_window_types[rid].add("baseless")
    assert n == 168123
    assert window_counts["any_risk_rebuilt"] == window_counts["positive_label_field"] == 21477

    out = {
        "status": "complete_fit_only_exact_census",
        "scope": {"fit_answers": 634, "fit_groups": len(groups), "calibration_read": False, "test_read": False, "gpu_used": False, "trained": False},
        "gold_spans": dict(span_counts),
        "conflict_span_surface_cues_not_mutually_exclusive": dict(conflict_surface_cues),
        "gold_answers": {
            "any_conflict": sum(bool(v & CONFLICT) for v in answer_types.values()),
            "any_baseless": sum(bool(v & BASELESS) for v in answer_types.values()),
            "both": sum(bool(v & CONFLICT) and bool(v & BASELESS) for v in answer_types.values()),
            "any_risk": sum(bool(v) for v in answer_types.values()),
        },
        "gold_4bpe_windows": {"eligible": n, **dict(window_counts)},
        "imbalance": {
            "subtle_conflict_share_of_all_error_spans": span_counts["Subtle Conflict"] / sum(span_counts.values()),
            "all_conflict_share_of_all_error_spans": sum(span_counts[x] for x in CONFLICT) / sum(span_counts.values()),
            "conflict_window_share_of_all_eligible_windows": window_counts["any_conflict"] / n,
            "conflict_window_share_of_all_positive_windows": window_counts["any_conflict"] / window_counts["any_risk_rebuilt"],
        },
        "sha256": {"fit.jsonl": digest(FIT), "tokens_fit.jsonl": digest(TOKENS), "windows_k4_fit.jsonl": digest(WINDOWS)},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
