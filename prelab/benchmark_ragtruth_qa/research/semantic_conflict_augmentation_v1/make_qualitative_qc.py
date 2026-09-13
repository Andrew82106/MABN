"""Attach a fixed qualitative read-through to the <=100 rule-QC sample.

The decisions below are researcher screening decisions, not independent human
gold labels.  They only decide whether a synthetic pair is suitable as silver
auxiliary training material.
"""
from collections import Counter
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "RULE_QC_AUDIT_MAX100.jsonl"
OUT = HERE / "QUALITATIVE_QC_AUDIT.jsonl"
SUMMARY = HERE / "QUALITATIVE_QC.json"

AMBIGUOUS = {
    "sca1_049002111172dc0432c8": "The target fact is readable, but the source has a broken sentence boundary ('set up.ou can'), creating a formatting shortcut.",
    "sca1_16cc75da15558b68428b": "The edit reverses 'very simple', an evaluative predicate rather than a sharply checkable fact.",
    "sca1_42c0a700f8aeb016625b": "The duration is readable, but the source contains a corrupted word boundary ('ab … out').",
    "sca1_5fc3b40a6e823cc1de79": "The attribution target contains an unresolved possessive referent ('its origins') outside the sentence.",
    "sca1_83242a58d7fd8771dcf4": "The main instruction is readable, but a broken trailing fragment ('ips & Warnings') may create a formatting shortcut.",
}

REJECT = {
    "sca1_11f4e47ce918a4398cce": "The source statement is truncated ('turn off the.'), so it cannot be a complete attribution target.",
    "sca1_c14159cd33393537193f": "This is an article title rather than a proposition whose factual relation can be learned.",
    "sca1_e7e15e33ffdab38b3983": "This is an article-title topic label rather than a declarative temporal fact.",
}


def main():
    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    assert len(rows) == 92
    assert not (set(AMBIGUOUS) & set(REJECT))
    assert set(AMBIGUOUS) | set(REJECT) <= {x["pair_id"] for x in rows}
    reviewed = []
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in REJECT:
            decision, reason = "reject", REJECT[pair_id]
        elif pair_id in AMBIGUOUS:
            decision, reason = "ambiguous", AMBIGUOUS[pair_id]
        else:
            decision = "clear_silver"
            reason = "One contiguous edit preserves readable syntax and changes the exact source-supported proposition; this remains synthetic silver, not verified real-world gold."
        reviewed.append({
            "pair_id": pair_id,
            "corruption_type": row["corruption_type"],
            "decision": decision,
            "reason": reason,
            "supported_claim": row["supported_claim"],
            "corrupted_claim": row["corrupted_claim"],
        })
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in reviewed:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "status": "complete_fixed_qualitative_readthrough",
        "rows": len(reviewed),
        "decision_counts": dict(Counter(x["decision"] for x in reviewed)),
        "by_type_and_decision": {
            kind: dict(Counter(x["decision"] for x in reviewed if x["corruption_type"] == kind))
            for kind in ("entity", "number", "negation", "temporal", "attribution")
        },
        "strict_sample_acceptance": sum(x["decision"] == "clear_silver" for x in reviewed) / len(reviewed),
        "review_boundary": "Research-agent qualitative screening of synthetic-pair fitness; not independent human annotation and not factual gold.",
        "recommended_use": "Only clear_silver rows inform the strict-rule precision estimate; ambiguous/reject rows motivate filters or lower weights.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
