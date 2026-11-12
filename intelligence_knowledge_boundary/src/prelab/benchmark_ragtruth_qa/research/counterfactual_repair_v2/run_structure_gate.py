"""CERP-v2.0 label-blind, CPU-only structural candidate gate.

The only data input is the frozen v1.1 label-blind candidate pool.  This file
contains no loader for QA labels, spans, windows, or held-out partitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
PARENT_GATE = HERE.parent / "counterfactual_repair_v1_design" / "fit_cpu_gate"
INPUT = PARENT_GATE / "candidates_fit_label_blind.jsonl"
PARENT_PREPARE = PARENT_GATE / "PREPARE.json"
PROTOCOL = HERE / "protocol.json"
OUT = HERE / "structure_gate"

EXPECTED_INPUT_SHA = "387de4fd7d0fb7299c20df88bcb414f10c93d27c12c15e3e24ad97100c8e77aa"
EXPECTED_PARENT_PREPARE_SHA = "111c9bc097e3113ca13d6361dd17bcdd050746a58cfd103b97ced6b805c15c8c"
QC_SALT = "cerp-v2-structure-qc-20260913"

LEX_RE = re.compile(
    r"[$€£]?\d+(?:[.,:/-]\d+)*(?:\s?(?:%|percent(?:age)?|degrees?|°[CFcf]?|"
    r"km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
    r"million|billion))?|[A-Za-z]+(?:[-'][A-Za-z]+)*",
    re.IGNORECASE,
)
CITATION_RE = re.compile(
    r"\b(?:passage|source|document|article|reference)s?\s*#?\s*(\d+)\b",
    re.IGNORECASE,
)

STOP = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her hers him his how i if in into is it its may might more most must my
no nor not of on or our ours she should so than that the their theirs them then
there these they this those to too under up us was we were what when where which
who why will with would you your according passage source document article
reference information provided given also however some any each other
""".split())

MONTHS = frozenset("january february march april may june july august september october november december".split())
WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday".split())
POLARITY_GROUPS = (
    frozenset(("increase", "decrease")), frozenset(("increased", "decreased")),
    frozenset(("increases", "decreases")), frozenset(("higher", "lower")),
    frozenset(("more", "less")), frozenset(("larger", "smaller")),
    frozenset(("before", "after")), frozenset(("earlier", "later")),
    frozenset(("above", "below")), frozenset(("over", "under")),
    frozenset(("include", "exclude")), frozenset(("includes", "excludes")),
    frozenset(("included", "excluded")), frozenset(("allow", "prevent")),
    frozenset(("allows", "prevents")), frozenset(("allowed", "prevented")),
    frozenset(("positive", "negative")), frozenset(("true", "false")),
    frozenset(("supports", "opposes")), frozenset(("supported", "opposed")),
    frozenset(("male", "female")), frozenset(("men", "women")),
    frozenset(("first", "last")), frozenset(("maximum", "minimum")),
    frozenset(("with", "without")), frozenset(("yes", "no")),
)
POLARITY = frozenset(x for group in POLARITY_GROUPS for x in group) | frozenset(
    ("not", "never", "cannot", "can't", "won't", "didn't", "doesn't", "isn't", "wasn't", "weren't")
)
UNIT_RE = re.compile(r"\b(%|percent(?:age)?|degrees?|°[CFcf]?|km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|million|billion)\b", re.I)
NUMBER_FULL = re.compile(r"[$€£]?\d+(?:[.,:/-]\d+)*(?:\s?(?:%|percent(?:age)?|degrees?|°[CFcf]?|km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|million|billion))?", re.I)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, values) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    tmp.replace(path)
    return count


def lex(text: str):
    return [
        {"text": match.group(), "norm": match.group().lower(), "start": match.start(), "end": match.end()}
        for match in LEX_RE.finditer(text)
    ]


def is_factual(norm: str) -> bool:
    return (
        norm not in STOP
        or any(char.isdigit() for char in norm)
        or norm in POLARITY
        or norm in MONTHS
        or norm in WEEKDAYS
    ) and (len(norm) > 1 or any(char.isdigit() for char in norm))


def factual_counter(text: str) -> Counter:
    return Counter(tok["norm"] for tok in lex(text) if is_factual(tok["norm"]))


def coverage(claim: str, source: str) -> tuple[float, int, int]:
    left, right = factual_counter(claim), factual_counter(source)
    total = sum(left.values())
    shared = sum(min(count, right[token]) for token, count in left.items())
    return (shared / total if total else 0.0), shared, total


def seq_similarity(left: str, right: str) -> float:
    a = [tok["norm"] for tok in lex(left)]
    b = [tok["norm"] for tok in lex(right)]
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def number_kind(text: str) -> str | None:
    value = text.strip()
    if not NUMBER_FULL.fullmatch(value):
        return None
    low = value.lower().replace(" ", "")
    digits = re.sub(r"\D", "", low)
    if re.fullmatch(r"(?:18|19|20|21)\d{2}", digits) and len(digits) == 4:
        return "year"
    if "%" in low or "percent" in low:
        return "percent"
    if low[:1] in "$€£":
        return "currency"
    unit = UNIT_RE.search(value)
    if unit:
        return "measure:" + unit.group(1).lower()
    if "/" in low or ":" in low or "-" in low:
        return "compound"
    if "." in low or "," in low:
        return "decimal"
    return "integer"


def temporal_kind(text: str) -> str | None:
    tokens = [tok["norm"] for tok in lex(text)]
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0] in MONTHS:
        return "month"
    if len(tokens) == 1 and tokens[0] in WEEKDAYS:
        return "weekday"
    numeric = number_kind(text)
    if numeric == "year":
        return "year"
    if numeric == "compound" and re.search(r"\d[/-]\d", text):
        return "date"
    return None


def proper_name(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'.-]*", text)
    if not words:
        return False
    if len(words) == 1:
        word = words[0]
        return 2 <= len(word) <= 12 and word.isupper()
    meaningful = [word for word in words if word.lower() not in {"of", "the", "and", "for", "de"}]
    return len(meaningful) >= 2 and all(word[0].isupper() or word.isupper() for word in meaningful)


def opposite_pair(left: str, right: str) -> bool:
    a = " ".join(tok["norm"] for tok in lex(left))
    b = " ".join(tok["norm"] for tok in lex(right))
    return any(a in group and b in group and a != b for group in POLARITY_GROUPS)


def classify_and_validate(old: str, new: str) -> tuple[str | None, str]:
    old_time, new_time = temporal_kind(old), temporal_kind(new)
    if old_time or new_time:
        return (("temporal", old_time) if old_time and old_time == new_time else (None, "temporal_subtype_mismatch"))
    old_num, new_num = number_kind(old), number_kind(new)
    if old_num or new_num:
        return (("number", old_num) if old_num and old_num == new_num else (None, "numeric_subtype_mismatch"))
    old_pol = any(tok["norm"] in POLARITY for tok in lex(old))
    new_pol = any(tok["norm"] in POLARITY for tok in lex(new))
    if old_pol or new_pol:
        return (("negation_direction", "frozen_opposite_pair") if opposite_pair(old, new) else (None, "unregistered_polarity_edit"))
    old_name, new_name = proper_name(old), proper_name(new)
    if old_name or new_name:
        return (("entity", "strict_proper_name") if old_name and new_name else (None, "one_sided_proper_name"))
    return "aligned_relation", "local_sequence_alignment"


def find_aligned_block(row: dict):
    claim, source = row["claim_text"], row["source_sentence"]
    a, b = lex(claim), lex(source)
    if not a or not b:
        return None
    matcher = SequenceMatcher(None, [tok["norm"] for tok in a], [tok["norm"] for tok in b], autojunk=False)
    opcodes = matcher.get_opcodes()
    for pos, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag != "replace" or not (i1 < i2 and j1 < j2):
            continue
        start, end = a[i1]["start"], a[i2 - 1]["end"]
        replacement = source[b[j1]["start"]:b[j2 - 1]["end"]]
        if start != int(row["local_edit_start"]) or end != int(row["local_edit_end"]):
            continue
        if claim[start:end] != row["original"] or replacement != row["replacement"]:
            continue
        if i2 - i1 > 4 or j2 - j1 > 4:
            return None
        before = opcodes[pos - 1] if pos else None
        after = opcodes[pos + 1] if pos + 1 < len(opcodes) else None
        if not before or before[0] != "equal" or not after or after[0] != "equal":
            return None
        left_tokens = a[max(before[2] - 6, before[1]):before[2]]
        right_tokens = a[after[1]:min(after[1] + 6, after[2])]
        left_content = sum(is_factual(tok["norm"]) for tok in left_tokens)
        right_content = sum(is_factual(tok["norm"]) for tok in right_tokens)
        if left_content < 1 or right_content < 1 or left_content + right_content < 3:
            return None
        return {
            "left_content_anchors": left_content,
            "right_content_anchors": right_content,
            "total_content_anchors": left_content + right_content,
            "old_lexical_tokens": i2 - i1,
            "new_lexical_tokens": j2 - j1,
            "source_edit_start": b[j1]["start"],
            "source_edit_end": b[j2 - 1]["end"],
        }
    return None


def validate_aligned(row: dict) -> tuple[dict | None, str]:
    block = find_aligned_block(row)
    if block is None:
        return None, "alignment_or_anchor_rule"
    repaired = row["repaired_claim"]
    source = row["source_sentence"]
    before_cov, _, _ = coverage(row["claim_text"], source)
    after_cov, shared, total = coverage(repaired, source)
    gain = after_cov - before_cov
    if after_cov < 0.60:
        return None, "repair_coverage_below_0.60"
    if gain < 0.05:
        return None, "coverage_gain_below_0.05"
    before_sim = seq_similarity(row["claim_text"], source)
    after_sim = seq_similarity(repaired, source)
    if after_sim <= before_sim:
        return None, "sequence_similarity_not_improved"
    kind, subtype = classify_and_validate(row["original"], row["replacement"])
    if kind is None:
        return None, subtype
    result = dict(row)
    result.update(block)
    result.update({
        "parent_candidate_id": row["candidate_id"],
        "candidate_id": "cerp2_" + digest(row["candidate_id"] + "|" + kind)[:20],
        "candidate_type": kind,
        "candidate_subtype": subtype,
        "claim_coverage_before": before_cov,
        "claim_coverage_after": after_cov,
        "claim_coverage_gain": gain,
        "shared_factual_tokens_after": shared,
        "factual_tokens_after": total,
        "sequence_similarity_before": before_sim,
        "sequence_similarity_after": after_sim,
        "structure_rule": "two_sided_single_local_replace",
        "requires_nli_support_gate": True,
        "labels_used": False,
    })
    return result, "accepted"


def validate_citation(row: dict) -> tuple[dict | None, str]:
    match = CITATION_RE.search(row["claim_text"])
    if not match or match.span(1) != (int(row["local_edit_start"]), int(row["local_edit_end"])):
        return None, "citation_interval_mismatch"
    if str(row["replacement"]) != str(row["passage_id"]) or row["original"] == row["replacement"]:
        return None, "citation_target_mismatch"
    stripped = (row["claim_text"][:match.start()] + " " + row["claim_text"][match.end():]).strip()
    cov, shared, total = coverage(stripped, row["source_sentence"])
    if cov < 0.60 or shared < 3:
        return None, "citation_content_support_too_low"
    result = dict(row)
    result.update({
        "parent_candidate_id": row["candidate_id"],
        "candidate_id": "cerp2_" + digest(row["candidate_id"] + "|citation")[:20],
        "candidate_type": "citation",
        "candidate_subtype": "passage_id_with_local_content_support",
        "left_content_anchors": 0,
        "right_content_anchors": 0,
        "total_content_anchors": shared,
        "claim_coverage_before": cov,
        "claim_coverage_after": cov,
        "claim_coverage_gain": 0.0,
        "shared_factual_tokens_after": shared,
        "factual_tokens_after": total,
        "sequence_similarity_before": seq_similarity(stripped, row["source_sentence"]),
        "sequence_similarity_after": seq_similarity(stripped, row["source_sentence"]),
        "structure_rule": "citation_stripped_content_support",
        "requires_nli_support_gate": True,
        "labels_used": False,
    })
    return result, "accepted"


def rank_key(row: dict):
    rank = row.get("union_rank")
    rank = int(rank) if isinstance(rank, int) else 99
    edit_len = int(row.get("old_lexical_tokens", row.get("original_lexical_tokens", 99))) + int(row.get("new_lexical_tokens", row.get("replacement_lexical_tokens", 99)))
    return (
        -int(row["total_content_anchors"]),
        -float(row["claim_coverage_gain"]),
        -float(row["claim_coverage_after"]),
        rank,
        edit_len,
        row["candidate_id"],
    )


def select(candidates: list[dict]) -> list[dict]:
    by_claim = defaultdict(list)
    for row in candidates:
        by_claim[(row["response_id"], row["microclaim_id"])].append(row)
    selected = []
    for key in sorted(by_claim):
        used_intervals = set()
        kept = []
        for row in sorted(by_claim[key], key=rank_key):
            interval = (int(row["local_edit_start"]), int(row["local_edit_end"]))
            if interval in used_intervals:
                continue
            used_intervals.add(interval)
            kept.append(row)
            if len(kept) == 4:
                break
        selected.extend(kept)
    return selected


def stable_qc_sample(candidates: list[dict], protocol: dict) -> list[dict]:
    target = int(protocol["qc"]["target_per_type"])
    total = int(protocol["qc"]["sample_size"])
    by_type = defaultdict(list)
    for row in candidates:
        by_type[row["candidate_type"]].append(row)
    score = lambda row: digest(QC_SALT + "|" + row["candidate_id"])
    chosen, ids = [], set()
    for kind in sorted(by_type):
        for row in sorted(by_type[kind], key=score)[:target]:
            chosen.append(row)
            ids.add(row["candidate_id"])
    if len(chosen) < total:
        remaining = (row for row in candidates if row["candidate_id"] not in ids)
        chosen.extend(sorted(remaining, key=score)[:total - len(chosen)])
    chosen = sorted(chosen, key=lambda row: (row["candidate_type"], score(row)))[:total]
    return [{
        "qc_id": f"cerp2_qc_{index:03d}",
        "candidate_id": row["candidate_id"],
        "candidate_type": row["candidate_type"],
        "response_id": row["response_id"],
        "microclaim_id": row["microclaim_id"],
        "original_claim": row["claim_text"],
        "source_sentence": row["source_sentence"],
        "original_span": row["original"],
        "replacement_span": row["replacement"],
        "repaired_claim": row["repaired_claim"],
        "left_content_anchors": row["left_content_anchors"],
        "right_content_anchors": row["right_content_anchors"],
        "coverage_before": row["claim_coverage_before"],
        "coverage_after": row["claim_coverage_after"],
        "coverage_gain": row["claim_coverage_gain"],
        "agent_proxy_label": None,
        "agent_proxy_reason": None,
    } for index, row in enumerate(chosen, 1)]


def self_check() -> None:
    fake = {
        "candidate_id": "x", "response_id": "r", "microclaim_id": "m",
        "claim_text": "The rate increased to 35 percent in 2020 after the reform.",
        "source_sentence": "The rate increased to 42 percent in 2020 after the reform.",
        "local_edit_start": 22, "local_edit_end": 32, "original": "35 percent",
        "replacement": "42 percent", "repaired_claim": "The rate increased to 42 percent in 2020 after the reform.",
        "candidate_type": "aligned_relation", "union_rank": 1,
    }
    accepted, reason = validate_aligned(fake)
    assert reason == "accepted" and accepted["candidate_type"] == "number"
    assert classify_and_validate("2020", "2021") == ("temporal", "year")
    assert classify_and_validate("35%", "$35")[0] is None
    assert opposite_pair("higher", "lower")
    print("self-check: ok")


def prepare() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_label_blind_structure_gate"
    assert sha(INPUT) == EXPECTED_INPUT_SHA == protocol["input"]["sha256"]
    assert sha(PARENT_PREPARE) == EXPECTED_PARENT_PREPARE_SHA == protocol["input"]["required_parent_prepare_sha256"]
    parent = json.loads(PARENT_PREPARE.read_text(encoding="utf-8"))
    assert parent["selection_used_labels"] is False
    assert parent["artifacts_sha256"][INPUT.name] == EXPECTED_INPUT_SHA

    rejection = Counter()
    raw_eligible, accepted = 0, []
    forbidden_keys = {"label", "labels", "gold", "error_type", "spans", "window_label"}
    for row in rows(INPUT):
        assert row.get("labels_used") is False
        assert not (forbidden_keys & set(row))
        if row["candidate_type"] not in protocol["input"]["allowed_parent_candidate_types"]:
            continue
        raw_eligible += 1
        if row["candidate_type"] == "citation":
            result, reason = validate_citation(row)
        else:
            result, reason = validate_aligned(row)
        rejection[reason] += 1
        if result is not None:
            accepted.append(result)

    selected = select(accepted)
    type_counts = Counter(row["candidate_type"] for row in selected)
    claims = {(row["response_id"], row["microclaim_id"]) for row in selected}
    groups = {row["group_id"] for row in selected}
    candidate_path = OUT / "candidates_fit_label_blind.jsonl"
    atomic_jsonl(candidate_path, selected)
    qc_rows = stable_qc_sample(selected, protocol)
    qc_path = OUT / "QC_SAMPLE_UNREVIEWED.jsonl"
    atomic_jsonl(qc_path, qc_rows)

    gate = protocol["capacity_gate"]
    checks = {
        "candidate_count_min": len(selected) >= int(gate["candidate_count_min"]),
        "candidate_count_max": len(selected) <= int(gate["candidate_count_max"]),
        "claims_with_candidates_min": len(claims) >= int(gate["claims_with_candidates_min"]),
        "groups_with_candidates_min": len(groups) >= int(gate["groups_with_candidates_min"]),
        "candidate_types_with_at_least_100_min": sum(value >= 100 for value in type_counts.values()) >= int(gate["candidate_types_with_at_least_100_min"]),
    }
    summary = {
        "status": "structure_capacity_pass_pending_agent_proxy_qc" if all(checks.values()) else "stopped_structure_capacity_failed",
        "protocol_id": protocol["protocol_id"],
        "input_sha256": EXPECTED_INPUT_SHA,
        "parent_prepare_sha256": EXPECTED_PARENT_PREPARE_SHA,
        "selection_used_labels": False,
        "model_loaded": False,
        "gpu_used": False,
        "raw_parent_rows_eligible": raw_eligible,
        "accepted_before_caps": len(accepted),
        "candidates": len(selected),
        "claims_with_candidates": len(claims),
        "groups_with_candidates": len(groups),
        "candidate_counts": dict(sorted(type_counts.items())),
        "rejection_counts": dict(sorted(rejection.items())),
        "capacity_checks": checks,
        "capacity_pass": all(checks.values()),
        "qc_sample_rows": len(qc_rows),
        "qc_sample_salt": QC_SALT,
        "artifacts_sha256": {
            candidate_path.name: sha(candidate_path),
            qc_path.name: sha(qc_path),
        },
    }
    atomic_json(OUT / "PREPARE.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "prepare"))
    args = parser.parse_args()
    self_check()
    if args.command == "prepare":
        prepare()


if __name__ == "__main__":
    main()
