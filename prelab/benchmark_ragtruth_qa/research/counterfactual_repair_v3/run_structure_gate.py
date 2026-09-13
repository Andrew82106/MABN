"""CERP-v3.0 native-opcode, label-blind, CPU-only structural gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
INPUT = HERE / "input/claim_source_pairs_fit_label_blind.jsonl"
INPUT_MANIFEST = HERE / "input/PAIR_MANIFEST.json"
PROTOCOL = HERE / "protocol.json"
OUT = HERE / "structure_gate"

EXPECTED_INPUT_SHA = "961dee33b57376bfd6e311fc0933e416076bbf28036026d9bd32ad0f04f8eba6"
EXPECTED_MANIFEST_SHA = "8ab57ecaf5c31486059ce1920b4b5221657e569fe6e441d9e176ffeb03a780b4"
QC_SALT = "cerp-v3-native-opcode-qc-20260913"

LEX_RE = re.compile(
    r"[$€£]?\d+(?:[.,:/-]\d+)*(?:\s?(?:%|percent(?:age)?|degrees?|°[CFcf]?|"
    r"km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
    r"million|billion))?|[A-Za-z]+(?:[-'][A-Za-z]+)*",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\b(?:passage|source|document|article|reference)s?\s*#?\s*\d+\b", re.I)
NUMBER_FULL = re.compile(
    r"[$€£]?\d+(?:[.,:/-]\d+)*(?:\s?(?:%|percent(?:age)?|degrees?|°[CFcf]?|"
    r"km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
    r"million|billion))?",
    re.I,
)
UNIT_RE = re.compile(r"\b(%|percent(?:age)?|degrees?|°[CFcf]?|km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|million|billion)\b", re.I)

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
NEGATIONS = frozenset(("not", "no", "never", "without", "cannot", "can't", "won't", "didn't", "doesn't", "isn't", "wasn't", "weren't"))
DIRECTION_GROUPS = (
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
)
DIRECTION = frozenset(word for group in DIRECTION_GROUPS for word in group)
GENERIC_CAPITALIZED = frozenset(("the", "this", "that", "these", "those", "based", "according", "however", "therefore", "additionally", "overall"))


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
        norm not in STOP or any(char.isdigit() for char in norm) or norm in NEGATIONS
        or norm in DIRECTION or norm in MONTHS or norm in WEEKDAYS
    ) and (len(norm) > 1 or any(char.isdigit() for char in norm))


def factual_counter(text: str) -> Counter:
    return Counter(tok["norm"] for tok in lex(text) if is_factual(tok["norm"]))


def coverage(claim: str, source: str) -> tuple[float, int, int]:
    left, right = factual_counter(claim), factual_counter(source)
    total = sum(left.values())
    source_total = sum(right.values())
    shared = sum(min(count, right[token]) for token, count in left.items())
    return (2 * shared / (total + source_total) if total + source_total else 0.0), shared, total


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
    values = [tok["norm"] for tok in lex(text)]
    if len(values) == 1 and values[0] in MONTHS:
        return "month"
    if len(values) == 1 and values[0] in WEEKDAYS:
        return "weekday"
    numeric = number_kind(text)
    if numeric == "year":
        return "year"
    if numeric == "compound" and re.search(r"\d[/-]\d", text):
        return "date"
    return None


def valid_may_context(text: str, start: int, end: int) -> bool:
    if text[start:end] != "May":
        return False
    tokens = lex(text)
    index = next((i for i, tok in enumerate(tokens) if tok["start"] == start and tok["end"] == end), None)
    if index is None:
        return False
    previous = tokens[index - 1]["norm"] if index else ""
    following = tokens[index + 1]["norm"] if index + 1 < len(tokens) else ""
    return previous in {"in", "on", "by"} or any(char.isdigit() for char in previous + following)


def proper_name(text: str, container: str, start: int, end: int) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'.-]*", text)
    if not words:
        return False
    meaningful = [word for word in words if word.lower() not in {"of", "the", "and", "for", "de"}]
    if not meaningful:
        return False
    if len(meaningful) >= 2:
        return all(word[0].isupper() or word.isupper() for word in meaningful)
    word = meaningful[0]
    if 2 <= len(word) <= 12 and word.isupper():
        return True
    tokens = lex(container)
    token_index = next((i for i, tok in enumerate(tokens) if tok["start"] == start), None)
    return bool(token_index and word[0].isupper() and word.lower() not in GENERIC_CAPITALIZED | MONTHS | WEEKDAYS)


def direction_pair(left: str, right: str) -> bool:
    a = " ".join(tok["norm"] for tok in lex(left))
    b = " ".join(tok["norm"] for tok in lex(right))
    return any(a in group and b in group and a != b for group in DIRECTION_GROUPS)


def classify_replace(claim: str, source: str, old: str, new: str, old_start: int, old_end: int, new_start: int, new_end: int):
    old_tokens = [tok["norm"] for tok in lex(old)]
    new_tokens = [tok["norm"] for tok in lex(new)]
    if any(token in NEGATIONS for token in old_tokens + new_tokens):
        return None, "negation_replace_disallowed"
    old_time, new_time = temporal_kind(old), temporal_kind(new)
    if old_time or new_time:
        if not old_time or old_time != new_time:
            return None, "temporal_subtype_mismatch"
        if old.lower() == "may" and not valid_may_context(claim, old_start, old_end):
            return None, "modal_may_rejected"
        if new.lower() == "may" and not valid_may_context(source, new_start, new_end):
            return None, "modal_may_rejected"
        return "temporal", old_time
    old_num, new_num = number_kind(old), number_kind(new)
    if old_num or new_num or any(char.isdigit() for char in old + new):
        return (("number", old_num) if old_num and old_num == new_num else (None, "numeric_subtype_mismatch"))
    old_direction = any(token in DIRECTION for token in old_tokens)
    new_direction = any(token in DIRECTION for token in new_tokens)
    if old_direction or new_direction:
        return (("direction", "frozen_opposite_pair") if direction_pair(old, new) else (None, "unregistered_direction_edit"))
    old_name = proper_name(old, claim, old_start, old_end)
    new_name = proper_name(new, source, new_start, new_end)
    if old_name or new_name:
        return (("entity", "proper_name_pair") if old_name and new_name else (None, "one_sided_proper_name"))
    if not factual_counter(old) or not factual_counter(new):
        return None, "nonfactual_relation_block"
    return "relation", "local_lexical_relation"


def stitch(text: str, start: int, end: int, replacement: str) -> str:
    left, right = text[:start], text[end:]
    if replacement:
        if left and left[-1].isalnum() and replacement[0].isalnum():
            replacement = " " + replacement
        if right and right[0].isalnum() and replacement[-1].isalnum():
            replacement = replacement + " "
        return left + replacement + right
    if left.endswith(" ") and right.startswith(" "):
        right = right.lstrip()
    elif left and right and left[-1].isalnum() and right[0].isalnum():
        left += " "
    return left + right


def in_citation(text: str, start: int, end: int) -> bool:
    for match in CITATION_RE.finditer(text):
        if start == end:
            if match.start() <= start <= match.end():
                return True
        elif match.start() < end and match.end() > start:
            return True
    return False


def generate_pair(pair: dict):
    claim, source = pair["claim_text"], pair["source_sentence"]
    a, b = lex(claim), lex(source)
    if not a or not b:
        return [], Counter({"empty_lexical_side": 1})
    matcher = SequenceMatcher(None, [tok["norm"] for tok in a], [tok["norm"] for tok in b], autojunk=False)
    opcodes = matcher.get_opcodes()
    generated, rejected = [], Counter()
    non_equal_count = sum(tag != "equal" for tag, *_ in opcodes)
    for opcode_index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag not in {"replace", "insert", "delete"}:
            continue
        before = opcodes[opcode_index - 1] if opcode_index else None
        after = opcodes[opcode_index + 1] if opcode_index + 1 < len(opcodes) else None
        if not before or before[0] != "equal" or not after or after[0] != "equal":
            rejected["missing_two_sided_equal_anchor"] += 1
            continue
        left_tokens = a[max(before[2] - 6, before[1]):before[2]]
        right_tokens = a[after[1]:min(after[1] + 6, after[2])]
        left_anchors, right_anchors = len(left_tokens), len(right_tokens)
        factual_anchors = sum(is_factual(tok["norm"]) for tok in left_tokens + right_tokens)
        if left_anchors < 1 or right_anchors < 1 or left_anchors + right_anchors < 3 or factual_anchors < 1:
            rejected["anchor_rule"] += 1
            continue

        if tag in {"replace", "delete"}:
            local_start, local_end = a[i1]["start"], a[i2 - 1]["end"]
            old = claim[local_start:local_end]
        else:
            local_start = local_end = a[i1]["start"] if i1 < len(a) else len(claim)
            old = ""
        if tag in {"replace", "insert"}:
            source_start, source_end = b[j1]["start"], b[j2 - 1]["end"]
            new = source[source_start:source_end]
        else:
            source_start = source_end = b[j1]["start"] if j1 < len(b) else len(source)
            new = ""
        if in_citation(claim, local_start, local_end):
            rejected["citation_interval"] += 1
            continue

        old_count, new_count = i2 - i1, j2 - j1
        if tag == "replace":
            if not (1 <= old_count <= 4 and 1 <= new_count <= 4):
                rejected["replace_length"] += 1
                continue
            kind, subtype = classify_replace(claim, source, old, new, local_start, local_end, source_start, source_end)
            if kind is None:
                rejected[subtype] += 1
                continue
        else:
            changed = [tok["norm"] for tok in (b[j1:j2] if tag == "insert" else a[i1:i2])]
            if not (1 <= len(changed) <= 2 and all(token in NEGATIONS for token in changed)):
                rejected["nonnegation_insert_delete"] += 1
                continue
            kind, subtype = "negation", "insert" if tag == "insert" else "delete"

        repaired = stitch(claim, local_start, local_end, new)
        before_cov, _, _ = coverage(claim, source)
        after_cov, shared, total = coverage(repaired, source)
        gain = after_cov - before_cov
        if after_cov < 0.55:
            rejected["repair_coverage_below_0.55"] += 1
            continue
        if gain < 0.05:
            rejected["coverage_gain_below_0.05"] += 1
            continue
        before_sim, after_sim = seq_similarity(claim, source), seq_similarity(repaired, source)
        if after_sim <= before_sim:
            rejected["sequence_similarity_not_improved"] += 1
            continue

        candidate_id = "cerp3_" + digest("|".join((pair["pair_id"], str(opcode_index), tag, str(local_start), str(local_end), new)))[:20]
        generated.append({
            "candidate_id": candidate_id,
            "pair_id": pair["pair_id"],
            "response_id": pair["response_id"],
            "source_id": pair["source_id"],
            "group_id": pair["group_id"],
            "microclaim_id": pair["microclaim_id"],
            "microclaim_index": pair["microclaim_index"],
            "claim_start": pair["claim_start"],
            "claim_end": pair["claim_end"],
            "claim_text": claim,
            "edit_start": int(pair["claim_start"]) + local_start,
            "edit_end": int(pair["claim_start"]) + local_end,
            "local_edit_start": local_start,
            "local_edit_end": local_end,
            "original": old,
            "replacement": new,
            "repaired_claim": repaired,
            "operation": tag,
            "candidate_type": kind,
            "candidate_subtype": subtype,
            "opcode_index": opcode_index,
            "non_equal_opcode_count": non_equal_count,
            "old_lexical_tokens": old_count,
            "new_lexical_tokens": new_count,
            "left_lexical_anchors": left_anchors,
            "right_lexical_anchors": right_anchors,
            "total_lexical_anchors": left_anchors + right_anchors,
            "total_factual_anchors": factual_anchors,
            "source_edit_start": source_start,
            "source_edit_end": source_end,
            "sentence_index": pair["sentence_index"],
            "passage_id": pair["passage_id"],
            "sentence_id": pair["sentence_id"],
            "sentence_sha256": pair["sentence_sha256"],
            "source_sentence": source,
            "attention_rank": pair.get("attention_rank"),
            "bm25_rank": pair.get("bm25_rank"),
            "attention_scalar": pair.get("attention_scalar"),
            "bm25": pair.get("bm25"),
            "query_term_coverage": pair.get("query_term_coverage"),
            "union_rank": pair.get("union_rank", 99),
            "original_nli_request_id": pair.get("original_nli_request_id"),
            "claim_coverage_before": before_cov,
            "claim_coverage_after": after_cov,
            "claim_coverage_gain": gain,
            "shared_factual_tokens_after": shared,
            "factual_tokens_after": total,
            "sequence_similarity_before": before_sim,
            "sequence_similarity_after": after_sim,
            "sequence_similarity_gain": after_sim - before_sim,
            "requires_nli_support_gate": True,
            "labels_used": False,
        })
    return generated, rejected


def rank_key(row: dict):
    return (
        -int(row["total_factual_anchors"]),
        -int(row["total_lexical_anchors"]),
        -float(row["claim_coverage_gain"]),
        -float(row["claim_coverage_after"]),
        -float(row["sequence_similarity_gain"]),
        int(row.get("union_rank", 99)),
        int(row["old_lexical_tokens"]) + int(row["new_lexical_tokens"]),
        row["candidate_id"],
    )


def select(candidates: list[dict]) -> list[dict]:
    by_claim = defaultdict(list)
    for row in candidates:
        by_claim[(row["response_id"], row["microclaim_id"])].append(row)
    selected = []
    for key in sorted(by_claim):
        used_intervals, used_repairs, kept = set(), set(), []
        for row in sorted(by_claim[key], key=rank_key):
            interval = (int(row["local_edit_start"]), int(row["local_edit_end"]))
            repair = (interval, row["replacement"].lower())
            if interval in used_intervals or repair in used_repairs:
                continue
            used_intervals.add(interval)
            used_repairs.add(repair)
            kept.append(row)
            if len(kept) == 4:
                break
        selected.extend(kept)
    return selected


def stable_qc_sample(candidates: list[dict], protocol: dict) -> list[dict]:
    target, total = int(protocol["qc"]["target_per_type"]), int(protocol["qc"]["sample_size"])
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
        "qc_id": f"cerp3_qc_{index:03d}",
        "candidate_id": row["candidate_id"],
        "candidate_type": row["candidate_type"],
        "operation": row["operation"],
        "response_id": row["response_id"],
        "microclaim_id": row["microclaim_id"],
        "original_claim": row["claim_text"],
        "source_sentence": row["source_sentence"],
        "original_span": row["original"],
        "replacement_span": row["replacement"],
        "repaired_claim": row["repaired_claim"],
        "lexical_anchors": row["total_lexical_anchors"],
        "factual_anchors": row["total_factual_anchors"],
        "coverage_before": row["claim_coverage_before"],
        "coverage_after": row["claim_coverage_after"],
        "coverage_gain": row["claim_coverage_gain"],
        "agent_proxy_label": None,
        "agent_proxy_reason": None,
    } for index, row in enumerate(chosen, 1)]


def self_check() -> None:
    base = {
        "pair_id": "p", "response_id": "r", "source_id": "s", "group_id": "g",
        "microclaim_id": "m", "microclaim_index": 0, "claim_start": 10, "claim_end": 80,
        "sentence_index": 0, "passage_id": 1, "sentence_id": 0, "sentence_sha256": "z",
        "union_rank": 1, "labels_used": False,
    }
    number = dict(base, claim_text="The rate rose to 35 percent after the reform.", source_sentence="The rate rose to 42 percent after the reform.")
    generated, _ = generate_pair(number)
    assert len(generated) == 1 and generated[0]["candidate_type"] == "number"
    insert = dict(base, pair_id="p2", claim_text="The policy does permit exports after review.", source_sentence="The policy does not permit exports after review.")
    generated, _ = generate_pair(insert)
    assert len(generated) == 1 and generated[0]["candidate_type"] == "negation" and generated[0]["operation"] == "insert"
    delete = dict(base, pair_id="p3", claim_text="The policy does not permit exports after review.", source_sentence="The policy does permit exports after review.")
    generated, _ = generate_pair(delete)
    assert len(generated) == 1 and generated[0]["candidate_type"] == "negation" and generated[0]["operation"] == "delete"
    print("self-check: ok")


def prepare() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "CERP-v3.0-native-opcodes"
    assert protocol["status"] == "frozen_before_label_blind_structure_gate"
    assert sha(INPUT) == EXPECTED_INPUT_SHA == protocol["input"]["sha256"] == manifest["artifact_sha256"]
    assert sha(INPUT_MANIFEST) == EXPECTED_MANIFEST_SHA == protocol["input"]["manifest_sha256"]
    assert manifest["selection_used_labels"] is False and manifest["official_test_opened"] is False

    generated, rejection, pair_count = [], Counter(), 0
    forbidden = {"label", "labels", "gold", "error_type", "spans", "window_label"}
    for pair in rows(INPUT):
        assert pair["labels_used"] is False and not (forbidden & set(pair))
        pair_count += 1
        values, rejected = generate_pair(pair)
        generated.extend(values)
        rejection.update(rejected)
    selected = select(generated)
    type_counts = Counter(row["candidate_type"] for row in selected)
    operation_counts = Counter(row["operation"] for row in selected)
    claims = {(row["response_id"], row["microclaim_id"]) for row in selected}
    groups = {row["group_id"] for row in selected}

    candidate_path = OUT / "candidates_fit_label_blind.jsonl"
    atomic_jsonl(candidate_path, selected)
    qc_path = OUT / "QC_SAMPLE_UNREVIEWED.jsonl"
    qc_rows = stable_qc_sample(selected, protocol)
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
        "status": "structure_capacity_pass_pending_agent_proxy_qc" if all(checks.values()) else "stopped_structure_capacity_failed_route_terminated",
        "protocol_id": protocol["protocol_id"],
        "input_sha256": EXPECTED_INPUT_SHA,
        "input_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "selection_used_labels": False,
        "official_test_opened": False,
        "model_loaded": False,
        "gpu_used": False,
        "pairs_seen": pair_count,
        "accepted_before_caps": len(generated),
        "candidates": len(selected),
        "claims_with_candidates": len(claims),
        "groups_with_candidates": len(groups),
        "candidate_counts": dict(sorted(type_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "rejection_counts": dict(sorted(rejection.items())),
        "capacity_checks": checks,
        "capacity_pass": all(checks.values()),
        "qc_sample_rows": len(qc_rows),
        "qc_sample_salt": QC_SALT,
        "artifacts_sha256": {candidate_path.name: sha(candidate_path), qc_path.name: sha(qc_path)},
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
