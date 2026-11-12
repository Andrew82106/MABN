"""Fit-only CPU feasibility gate for CERP-v1.

`prepare` is label blind and freezes candidate edits before any fit label is
opened. `audit` may then read only the fit annotations and fit k=4 windows.
No model weights, CUDA API, published baseline, or held-out dataset is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "fit_cpu_gate"

FIT_ANSWERS = ROOT / "fit_expansion/data/fit.jsonl"
FIT_MICROCLAIMS = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
NATIVE_FIT = ROOT / "data/fit.jsonl"
EXPANDED_UNITS = ROOT / "results/semantic_source_attribution_expanded_fit_v1/semantic_units.jsonl"
NATIVE_EVIDENCE = ROOT / "results/evidence_union_nli_v2/candidates_fit_native.jsonl"
EXPANDED_EVIDENCE = ROOT / "results/evidence_union_nli_v2/candidates_fit_expanded.jsonl"
FIT_WINDOWS = ROOT / "fit_expansion/data/windows_k4_fit.jsonl"
PROTOCOL = HERE / "protocol.json"

MAX_PER_CLAIM = 12
MAX_PER_INTERVAL = 3
MIN_RETAIN = 0.45
MIN_UNCHANGED_CONTENT = 2
QC_PER_TYPE = 10
QC_SALT = "cerp-qc-v1.1"

TYPES = (
    "citation", "number", "temporal", "negation_direction", "entity",
    "aligned_relation",
)
TYPE_PRIORITY = {name: i for i, name in enumerate(TYPES)}

STOP = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her hers him his how i if in into is it its may might more most must my
no nor not of on or our ours she should so than that the their theirs them then
there these they this those to too under up us was we were what when where which
who why will with would you your according passage source document article
reference information provided given also however some any each other
""".split())

LEX_RE = re.compile(
    r"[$€£]?\d+(?:[.,:/-]\d+)*(?:\s?(?:%|percent(?:age)?|degrees?|°[CFcf]?|"
    r"km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
    r"million|billion))?|[A-Za-z]+(?:[-'][A-Za-z]+)*",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[$€£]?\d+(?:[.,:/-]\d+)*(?:\s?(?:%|percent(?:age)?|"
    r"degrees?|°[CFcf]?|km|cm|mm|kg|lbs?|hours?|minutes?|seconds?|days?|"
    r"weeks?|months?|years?|million|billion))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
CITATION_RE = re.compile(
    r"\b(?:passage|source|document|article|reference)s?\s*#?\s*(\d+)\b",
    re.IGNORECASE,
)
TEMPORAL_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|yesterday|today|tomorrow|earlier|later|before|after)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(?:not|no|never|without|neither|nor|cannot|can't|won't|didn't|doesn't|isn't|wasn't|weren't)\b", re.IGNORECASE)

DIRECTION_TERMS = (
    "bigger", "larger", "smaller", "higher", "lower", "more", "less",
    "increase", "increased", "decrease", "decreased", "before", "after",
    "earlier", "later", "cause", "causes", "caused", "prevent", "prevents",
    "prevented", "include", "includes", "included", "exclude", "excludes",
    "excluded", "with", "without", "above", "below", "over", "under",
    "first", "last", "maximum", "minimum",
)
DIRECTION_RE = re.compile(
    r"\b(?:" + "|".join(map(re.escape, sorted(DIRECTION_TERMS, key=len, reverse=True))) + r")\b",
    re.IGNORECASE,
)
CAP_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+(?:(?:[A-Z][A-Za-z0-9&'.-]*)|of|the|and)){0,4}\b"
)

PASSAGE_HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, values) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False,
                               separators=(",", ":")) + "\n")
            count += 1
    tmp.replace(path)
    return count


def lex(text: str):
    return [
        {"text": m.group(), "norm": m.group().lower(),
         "start": m.start(), "end": m.end()}
        for m in LEX_RE.finditer(text)
    ]


def content(tokens):
    return [t for t in tokens if t["norm"] not in STOP and
            (len(t["norm"]) > 1 or any(ch.isdigit() for ch in t["norm"]))]


def sim_stats(left: str, right: str):
    a = {t["norm"] for t in content(lex(left))}
    b = {t["norm"] for t in content(lex(right))}
    shared = len(a & b)
    union = len(a | b)
    return shared, shared / union if union else 0.0


def sentence_spans(text: str):
    """Byte-for-byte copy of the already frozen project sentence rule."""
    assert text and any(char.isalnum() for char in text)
    cuts, i, line_start = [], 0, 0
    while i < len(text):
        char = text[i]
        if char in "\r\n":
            if char == "\r" and i + 1 < len(text) and text[i + 1] == "\n":
                i += 1
            cuts.append(i + 1)
            line_start = i + 1
        elif char in TERMINAL:
            end = i + 1
            while end < len(text) and (text[end] in TERMINAL or text[end] in CLOSERS):
                end += 1
            prefix = text[line_start:i + 1].strip()
            followed_by_break = end == len(text) or text[end].isspace()
            match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            final_token = match.group(0).lower() if match else ""
            dotted = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", final_token))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", final_token))
            protected = end < len(text) and (final_token in ABBREVIATIONS or dotted or initial)
            if followed_by_break and not LIST_PREFIX.fullmatch(prefix) and not protected:
                cuts.append(end)
                i = end - 1
        i += 1
    cuts.append(len(text))
    spans, start = [], 0
    for stop in sorted(set(cuts)):
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and any(char.isalnum() for char in text[left:right]):
            spans.append((left, right))
        start = stop
    assert spans
    return spans


def parse_passages(text: str):
    matches = list(PASSAGE_HEADER.finditer(text))
    assert [int(match.group(1)) for match in matches] == [1, 2, 3]
    result = []
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        while body_start < body_end and text[body_start].isspace():
            body_start += 1
        while body_end > body_start and text[body_end - 1].isspace():
            body_end -= 1
        body = text[body_start:body_end]
        for sentence_id, (left, right) in enumerate(sentence_spans(body)):
            value = body[left:right]
            result.append({
                "sentence_id": sentence_id, "passage_id": int(match.group(1)),
                "sentence_index": len(result), "text": value,
                "text_sha256": digest(value),
            })
    return result


def number_kind(text: str) -> str:
    low = text.lower().replace(" ", "")
    digits = re.sub(r"\D", "", low)
    if "%" in low or "percent" in low:
        return "percent"
    if low[:1] in "$€£":
        return "currency"
    if "°" in low or "degree" in low:
        return "temperature"
    if any(unit in low for unit in ("hour", "minute", "second", "day", "week", "month", "year")):
        return "duration"
    if ":" in low:
        return "clock"
    if len(digits) == 4 and 1000 <= int(digits) <= 2100:
        return "year"
    return "number"


def regex_spans(pattern: re.Pattern, text: str, group: int | None = None):
    out = []
    for m in pattern.finditer(text):
        start, end = m.span(group or 0)
        out.append({"start": start, "end": end, "text": text[start:end]})
    return out


def entity_spans(text: str, structured=None):
    values = []
    if structured:
        for row in structured:
            start, end = int(row["start"]), int(row["end"])
            values.append({"start": start, "end": end, "text": row["text"]})
    else:
        for m in CAP_RE.finditer(text):
            value = m.group().strip()
            words = value.split()
            if (m.start() == 0 and len(words) == 1 and not value.isupper()):
                continue
            if value.lower() in STOP or value.isdigit():
                continue
            values.append({"start": m.start(), "end": m.end(), "text": value})
    unique = {}
    for row in values:
        unique[(row["start"], row["end"], row["text"].lower())] = row
    return list(unique.values())


def direction_spans(text: str):
    found = regex_spans(NEGATION_RE, text) + regex_spans(DIRECTION_RE, text)
    return sorted({(r["start"], r["end"], r["text"].lower()): r for r in found}.values(),
                  key=lambda r: (r["start"], r["end"]))


def retrieval_rank(meta: dict) -> int:
    values = []
    if isinstance(meta.get("attention_rank"), int):
        values.append(meta["attention_rank"])
    if isinstance(meta.get("bm25_rank"), int):
        values.append(meta["bm25_rank"])
    return min(values) if values else 99


def candidate_record(answer: dict, claim: dict, source: dict, meta: dict,
                     kind: str, start: int, end: int, replacement: str,
                     anchors: int, subtype: str = ""):
    text = claim["text"]
    if not (0 <= start <= end <= len(text)):
        return None
    original = text[start:end]
    replacement = replacement.strip()
    if not replacement or original.lower() == replacement.lower():
        return None
    old_tokens, new_tokens, all_tokens = lex(original), lex(replacement), lex(text)
    if len(old_tokens) > 5 or len(new_tokens) > 6:
        return None
    kept = [t for t in content(all_tokens) if not (t["start"] < end and t["end"] > start)]
    retention = (len(all_tokens) - len(old_tokens)) / max(1, len(all_tokens))
    if len(kept) < MIN_UNCHANGED_CONTENT or retention < MIN_RETAIN:
        return None
    shared, jac = sim_stats(text, source["text"])
    strong = kind in {"citation", "number", "temporal"}
    if strong:
        if shared < 1 and jac < 0.08:
            return None
    elif shared < 2 or jac < 0.12:
        return None
    repaired = text[:start] + replacement + text[end:]
    if re.sub(r"\W+", " ", repaired).strip().lower() == re.sub(r"\W+", " ", source["text"]).strip().lower():
        return None
    absolute_start = int(claim["start"]) + start
    absolute_end = int(claim["start"]) + end
    identity = {
        "response_id": answer["response_id"], "microclaim_id": claim["microclaim_id"],
        "edit_start": absolute_start, "edit_end": absolute_end,
        "replacement": replacement, "sentence_sha256": source["text_sha256"],
    }
    return {
        "candidate_id": "cerp_" + digest(identity)[:20],
        "response_id": answer["response_id"], "source_id": answer["source_id"],
        "group_id": answer["group_id"], "microclaim_id": claim["microclaim_id"],
        "microclaim_index": int(claim["microclaim_index"]),
        "claim_start": int(claim["start"]), "claim_end": int(claim["end"]),
        "claim_text": text, "edit_start": absolute_start, "edit_end": absolute_end,
        "local_edit_start": start, "local_edit_end": end,
        "original": original, "replacement": replacement,
        "repaired_claim": repaired, "candidate_type": kind,
        "candidate_subtype": subtype,
        "passage_id": int(source["passage_id"]),
        "sentence_id": int(source["sentence_id"]),
        "sentence_index": int(source["sentence_index"]),
        "sentence_sha256": source["text_sha256"], "source_sentence": source["text"],
        "attention_rank": meta.get("attention_rank"),
        "bm25_rank": meta.get("bm25_rank"),
        "attention_scalar": meta.get("attention_scalar"),
        "bm25": meta.get("bm25"),
        "query_term_coverage": meta.get("query_term_coverage"),
        "union_rank": retrieval_rank(meta), "shared_content_tokens": shared,
        "content_jaccard": jac, "anchor_tokens": anchors,
        "original_lexical_tokens": len(old_tokens),
        "replacement_lexical_tokens": len(new_tokens),
        "original_lexical_retention": retention,
        "single_contiguous_edit": True, "replacement_exact_in_source": replacement in source["text"],
        "whole_sentence_copy": False, "labels_used": False,
    }


def aligned_edits(claim_text: str, source_text: str):
    a, b = lex(claim_text), lex(source_text)
    matcher = SequenceMatcher(None, [x["norm"] for x in a], [x["norm"] for x in b], autojunk=False)
    opcodes = matcher.get_opcodes()
    for pos, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag != "replace" or not (0 < i2 - i1 <= 5 and 0 < j2 - j1 <= 6):
            continue
        before = 0
        after = 0
        if pos and opcodes[pos - 1][0] == "equal":
            before = min(3, opcodes[pos - 1][2] - opcodes[pos - 1][1])
        if pos + 1 < len(opcodes) and opcodes[pos + 1][0] == "equal":
            after = min(3, opcodes[pos + 1][2] - opcodes[pos + 1][1])
        anchors = before + after
        if anchors < 2:
            continue
        yield a[i1]["start"], a[i2 - 1]["end"], source_text[b[j1]["start"]:b[j2 - 1]["end"]], anchors


def construct(answer: dict, claim: dict, structured: dict,
              source: dict, meta: dict):
    text, evidence = claim["text"], source["text"]
    proposals = []

    # Explicit evidence-source attribution.
    for left in regex_spans(CITATION_RE, text):
        m = CITATION_RE.search(left["text"])
        if m and int(m.group(1)) != int(source["passage_id"]):
            s, e = m.span(1)
            proposals.append(("citation", left["start"] + s, left["start"] + e,
                              str(source["passage_id"]), 3, "passage_id"))

    # Number + unit is edited as one slot.
    left_numbers = regex_spans(NUMBER_RE, text)
    right_numbers = regex_spans(NUMBER_RE, evidence)
    for left in left_numbers:
        lk = number_kind(left["text"])
        ordered = sorted(right_numbers, key=lambda x: (number_kind(x["text"]) != lk, x["start"]))
        for right in ordered[:4]:
            if left["text"].lower() != right["text"].lower():
                proposals.append(("number", left["start"], left["end"], right["text"],
                                  2, f"{lk}->{number_kind(right['text'])}"))

    # Month/day/direction words not already captured as numeric slots.
    for left in regex_spans(TEMPORAL_RE, text):
        for right in regex_spans(TEMPORAL_RE, evidence)[:4]:
            if left["text"].lower() != right["text"].lower():
                proposals.append(("temporal", left["start"], left["end"], right["text"], 2, "word"))

    left_direction = direction_spans(text)
    right_direction = direction_spans(evidence)
    for left in left_direction:
        for right in right_direction[:4]:
            if left["text"].lower() != right["text"].lower():
                proposals.append(("negation_direction", left["start"], left["end"],
                                  right["text"], 2, "cue"))

    # Claim entities use the existing deterministic atomizer coordinates.
    structured_entities = []
    for item in structured.get("entity_candidates", []):
        structured_entities.append({
            "start": int(item["start"]) - int(claim["start"]),
            "end": int(item["end"]) - int(claim["start"]),
            "text": item["text"],
        })
    for left in entity_spans(text, structured_entities):
        if not (0 <= left["start"] < left["end"] <= len(text)):
            continue
        for right in entity_spans(evidence)[:5]:
            if left["text"].lower() != right["text"].lower():
                proposals.append(("entity", left["start"], left["end"], right["text"], 2, "proper_chunk"))

    for start, end, replacement, anchors in aligned_edits(text, evidence):
        proposals.append(("aligned_relation", start, end, replacement, anchors, "sequence_alignment"))

    result = []
    for kind, start, end, replacement, anchors, subtype in proposals:
        row = candidate_record(answer, claim, source, meta, kind, start, end,
                               replacement, anchors, subtype)
        if row is not None:
            result.append(row)
    return result


def load_answers_label_blind():
    values = {}
    for row in rows(FIT_ANSWERS):
        assert row["partition"] == "fit"
        values[row["response_id"]] = {
            "response_id": row["response_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "question": row["question"],
            "answer_sha256": row["answer_sha256"],
        }
    assert len(values) == 3680 and len({r["group_id"] for r in values.values()}) == 615
    return values


def load_structured():
    values = {}
    for row in rows(FIT_MICROCLAIMS):
        assert row["partition"] == "fit"
        values[(row["response_id"], int(row["microclaim_index"]))] = row
    assert len(values) == 34941
    return values


def load_units(structured):
    values = {}
    # This source is a physically separate 634-row fit file.  The prior v1
    # attempt used a mixed upstream artifact and stopped at its first non-fit
    # row; v1.1 never opens that artifact.
    for row in rows(NATIVE_FIT):
        assert row["partition"] == "fit"
        rid = row["response_id"]
        claims = [value for (response_id, _), value in structured.items()
                  if response_id == rid and any(ch.isalnum() for ch in value["text"])]
        claims.sort(key=lambda value: int(value["microclaim_index"]))
        sentences = parse_passages(row["retrieved_passages"])
        values[row["response_id"]] = {
            "claims": {int(c["microclaim_index"]): c for c in claims},
            "sentences": {int(s["sentence_index"]): s for s in sentences},
        }
    native = len(values)
    for row in rows(EXPANDED_UNITS):
        assert row["partition"] == "fit" and row["labels_used"] is False
        assert row["response_id"] not in values
        values[row["response_id"]] = {
            "claims": {int(c["microclaim_index"]): c for c in row["claims"]},
            "sentences": {int(s["sentence_index"]): s for s in row["sentences"]},
        }
    assert native == 634 and len(values) == 3680
    return values


def iter_evidence_rows():
    yield from rows(NATIVE_EVIDENCE)
    yield from rows(EXPANDED_EVIDENCE)


def prepare():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_fit_cpu_coverage_audit_v1.1"
    answers = load_answers_label_blind()
    structured = load_structured()
    units = load_units(structured)
    counts = Counter()

    def generated():
        for evidence_row in iter_evidence_rows():
            rid = evidence_row["response_id"]
            assert evidence_row["partition"] == "fit" and evidence_row["labels_used"] is False
            answer, unit = answers[rid], units[rid]
            for evidence_claim in evidence_row["claims"]:
                idx = int(evidence_claim["microclaim_index"])
                claim = unit["claims"][idx]
                struct = structured[(rid, idx)]
                pool = []
                for meta in evidence_claim["candidates"]:
                    source = unit["sentences"][int(meta["sentence_index"])]
                    assert source["text_sha256"] == meta["sentence_sha256"]
                    pool.extend(construct(answer, claim, struct, source, meta))
                # Stable label-blind ranking and exact deduplication.
                dedup = {}
                for row in pool:
                    key = (row["edit_start"], row["edit_end"],
                           row["replacement"].lower(), row["sentence_sha256"])
                    old = dedup.get(key)
                    score = (-row["shared_content_tokens"], -row["content_jaccard"],
                             row["union_rank"], TYPE_PRIORITY[row["candidate_type"]],
                             row["original_lexical_tokens"] + row["replacement_lexical_tokens"],
                             row["candidate_id"])
                    if old is None or score < old[0]:
                        dedup[key] = (score, row)
                ordered = [x[1] for x in sorted(dedup.values(), key=lambda x: x[0])]
                interval_count = Counter()
                kept = []
                for row in ordered:
                    interval = (row["edit_start"], row["edit_end"])
                    if interval_count[interval] >= MAX_PER_INTERVAL:
                        continue
                    interval_count[interval] += 1
                    kept.append(row)
                    if len(kept) == MAX_PER_CLAIM:
                        break
                counts["claims"] += 1
                counts["claims_with_candidates"] += bool(kept)
                for row in kept:
                    counts[f"type_{row['candidate_type']}"] += 1
                    yield row

    candidate_path = OUT / "candidates_fit_label_blind.jsonl"
    n = atomic_jsonl(candidate_path, generated())
    assert counts["claims"] == 34919
    assert n == sum(v for k, v in counts.items() if k.startswith("type_"))

    candidates = list(rows(candidate_path))
    selected = []
    for kind in TYPES:
        subset = [row for row in candidates if row["candidate_type"] == kind]
        subset.sort(key=lambda row: digest(f"{QC_SALT}|{row['candidate_id']}"))
        selected.extend(subset[:QC_PER_TYPE])
    qc_path = OUT / "QC_SAMPLE_UNREVIEWED.jsonl"
    atomic_jsonl(qc_path, selected)
    manifest = {
        "status": "label_blind_candidates_frozen_before_fit_audit",
        "answers": len(answers), "groups": len({r["group_id"] for r in answers.values()}),
        "claims_seen": counts["claims"],
        "claims_with_candidates": counts["claims_with_candidates"],
        "candidates": n,
        "candidate_counts": {kind: counts[f"type_{kind}"] for kind in TYPES},
        "qc_sample_rows": len(selected), "qc_sample_salt": QC_SALT,
        "selection_used_labels": False, "model_loaded": False, "gpu_used": False,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            FIT_ANSWERS, FIT_MICROCLAIMS, NATIVE_FIT, EXPANDED_UNITS,
            NATIVE_EVIDENCE, EXPANDED_EVIDENCE, PROTOCOL)},
        "artifacts_sha256": {
            candidate_path.name: sha(candidate_path), qc_path.name: sha(qc_path),
        },
    }
    atomic_json(OUT / "PREPARE.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def intersects(left: int, right: int, start: int, end: int) -> bool:
    if left == right:
        return start <= left <= end
    return left < end and right > start


def audit():
    prepared = json.loads((OUT / "PREPARE.json").read_text(encoding="utf-8"))
    candidate_path = OUT / "candidates_fit_label_blind.jsonl"
    assert prepared["artifacts_sha256"][candidate_path.name] == sha(candidate_path)
    candidates = list(rows(candidate_path))
    by_response = defaultdict(list)
    for row in candidates:
        by_response[row["response_id"]].append((row["edit_start"], row["edit_end"], row["candidate_type"]))

    answer_labels = {}
    spans = []
    clean_answers = set()
    for row in rows(FIT_ANSWERS):
        labels = [{"start": int(s["start"]), "end": int(s["end"]),
                   "label_type": s["label_type"]} for s in row["labels"]]
        answer_labels[row["response_id"]] = labels
        if not labels:
            clean_answers.add(row["response_id"])
        for i, span in enumerate(labels):
            if span["label_type"] not in {"Evident Conflict", "Subtle Conflict"}:
                continue
            hit_types = sorted({kind for left, right, kind in by_response[row["response_id"]]
                                if intersects(left, right, span["start"], span["end"])})
            spans.append({
                "response_id": row["response_id"], "span_index": i,
                "label_type": span["label_type"], "start": span["start"], "end": span["end"],
                "covered": bool(hit_types), "candidate_types": hit_types,
            })
    assert len(spans) == 301
    atomic_jsonl(OUT / "conflict_span_coverage_fit.jsonl", spans)

    window_counts = Counter()
    for window in rows(FIT_WINDOWS):
        assert window["partition"] == "fit" and window["eligible"] is True
        rid = window["response_id"]
        start, end = int(window["char_start"]), int(window["char_end"])
        touched = any(intersects(left, right, start, end)
                      for left, right, _ in by_response.get(rid, ()))
        conflict = any(s["label_type"] in {"Evident Conflict", "Subtle Conflict"}
                       and intersects(s["start"], s["end"], start, end)
                       for s in answer_labels[rid])
        baseless = any("Baseless" in s["label_type"]
                       and intersects(s["start"], s["end"], start, end)
                       for s in answer_labels[rid])
        category = ("conflict" if conflict else "baseless" if baseless else
                    "clean_answer_negative" if rid in clean_answers else "risky_answer_negative")
        window_counts[f"{category}_total"] += 1
        window_counts[f"{category}_touched"] += touched
        window_counts["total"] += 1
        window_counts["touched"] += touched
    assert window_counts["total"] == 653979

    def rate(prefix):
        return window_counts[f"{prefix}_touched"] / max(1, window_counts[f"{prefix}_total"])

    ec = [s for s in spans if s["label_type"] == "Evident Conflict"]
    sc = [s for s in spans if s["label_type"] == "Subtle Conflict"]
    span_metrics = {
        "Evident Conflict": {"total": len(ec), "covered": sum(s["covered"] for s in ec),
                              "coverage": sum(s["covered"] for s in ec) / len(ec)},
        "Subtle Conflict": {"total": len(sc), "covered": sum(s["covered"] for s in sc),
                             "coverage": sum(s["covered"] for s in sc) / len(sc)},
        "all_conflict": {"total": len(spans), "covered": sum(s["covered"] for s in spans),
                         "coverage": sum(s["covered"] for s in spans) / len(spans)},
    }
    window_metrics = {}
    for prefix in ("conflict", "baseless", "clean_answer_negative", "risky_answer_negative"):
        window_metrics[prefix] = {
            "total": window_counts[f"{prefix}_total"],
            "touched": window_counts[f"{prefix}_touched"],
            "touch_rate": rate(prefix),
        }
    clean_rate = rate("clean_answer_negative")
    enrichment = rate("conflict") / max(clean_rate, 1e-12)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = protocol["cpu_gate"]
    checks = {
        "ec_span_coverage": span_metrics["Evident Conflict"]["coverage"] >= gate["ec_span_coverage_min"],
        "all_conflict_span_coverage": span_metrics["all_conflict"]["coverage"] >= gate["all_conflict_span_coverage_min"],
        "conflict_window_touch": rate("conflict") >= gate["conflict_window_touch_min"],
        "clean_answer_window_touch": clean_rate <= gate["clean_answer_window_touch_max"],
        "conflict_to_clean_enrichment": enrichment >= gate["conflict_to_clean_touch_enrichment_min"],
    }
    result = {
        "status": "fit_cpu_automatic_gate_complete_pending_frozen_qc_review",
        "span_metrics": span_metrics, "window_metrics": window_metrics,
        "conflict_to_clean_touch_enrichment": enrichment,
        "automatic_gate_checks": checks,
        "automatic_gate_pass": all(checks.values()),
        "negative_pollution_interpretation": (
            "Candidate presence is never a positive pseudo-label. Clean and risky-answer negative windows "
            "remain explicit human-negative training controls; NLI repair margin must suppress them."
        ),
        "fit_labels_read_only_after_candidate_sha_freeze": True,
        "model_loaded": False, "gpu_used": False,
        "candidate_sha256": sha(candidate_path),
        "fit_windows_sha256": sha(FIT_WINDOWS),
    }
    atomic_json(OUT / "AUDIT_AUTOMATIC.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def self_check():
    fake_answer = {"response_id": "r", "source_id": "s", "group_id": "g"}
    claim = {"text": "The rate was $35 in 2020.", "start": 10, "end": 39,
             "microclaim_id": "m", "microclaim_index": 0}
    source = {"text": "The rate was 35% in 2021.", "passage_id": 2,
              "sentence_id": 0, "sentence_index": 0,
              "text_sha256": digest("The rate was 35% in 2021.")}
    meta = {"attention_rank": 0, "bm25_rank": 0}
    built = construct(fake_answer, claim, {"entity_candidates": []}, source, meta)
    assert built and all(row["replacement_exact_in_source"] for row in built)
    assert any(row["original"] == "$35" and row["replacement"] == "35%" for row in built)
    assert all(row["edit_start"] >= claim["start"] for row in built)
    print("CERP_CPU_SELF_CHECK_PASSED", len(built))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "prepare", "audit"))
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "prepare":
        prepare()
    else:
        audit()


if __name__ == "__main__":
    main()
