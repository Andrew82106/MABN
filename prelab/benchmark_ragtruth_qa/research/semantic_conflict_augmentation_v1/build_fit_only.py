"""Build and audit fit-only source-sentence minimal pairs.

This script is deliberately bounded to ``data/fit.jsonl``.  It does not read
calibration or test data, does not load a model, and does not train anything.
The generated corruption labels are synthetic silver labels, never RAGTruth
human labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIT = ROOT / "data/fit.jsonl"

PASSAGE_HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})
WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
NUMBER = re.compile(r"(?<![A-Za-z0-9])(?P<prefix>[$€£])?(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)(?P<suffix>%?)(?![A-Za-z0-9])")
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
TIME = re.compile(r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})(?:\s*(?P<ampm>a\.?m\.?|p\.?m\.?))?\b", re.I)
DURATION = re.compile(r"\b(?P<num>\d+(?:\.\d+)?)\s+(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", re.I)
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH = re.compile(r"\b(" + "|".join(MONTHS) + r")\b")
WEEKDAY = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b")
HEDGE = re.compile(r"\b(?:about|around|approximately|roughly|nearly|almost|up to|at least|at most|more than|less than|over|under|may|might|can|could|simply)\b", re.I)
QUESTION_START = re.compile(r"^(?:who|what|when|where|why|how|which|is|are|was|were|do|does|did|can|could|would|should|will)\b", re.I)

CONTRACTION_MAP = {
    "isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were",
    "can't": "can", "cannot": "can", "couldn't": "could", "won't": "will",
    "wouldn't": "would", "shouldn't": "should", "mustn't": "must",
    "doesn't": "does", "don't": "do", "didn't": "did", "hasn't": "has",
    "haven't": "have", "hadn't": "had",
}
CONTRACTION = re.compile(r"\b(" + "|".join(re.escape(x) for x in CONTRACTION_MAP) + r")\b", re.I)
AUX_NOT = re.compile(r"\b(is|are|was|were|will|would|can|could|should|must|does|do|did|has|have|had)\s+not\b", re.I)
AUX_INSERT = re.compile(r"\b(is|are|was|were|will|would|can|should|must)\b", re.I)
NEG_ANY = re.compile(r"\b(?:not|never|no|cannot|isn't|aren't|wasn't|weren't|can't|couldn't|won't|wouldn't|shouldn't|mustn't|doesn't|don't|didn't|hasn't|haven't|hadn't)\b", re.I)

CAP_SEQUENCE = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]{2,})(?:\s+(?:(?:of|the|and|de|van|von)\s+)?(?:[A-Z]{2,}|[A-Z][a-z]{2,})){0,3}\b")
ENTITY_STOP = frozenset({
    "The", "This", "That", "These", "Those", "There", "Here", "It", "I", "We", "You", "He", "She", "They",
    "A", "An", "In", "On", "At", "For", "From", "To", "By", "With", "Without", "If", "When", "While", "After", "Before",
    "First", "Second", "Third", "Finally", "Next", "Then", "However", "Therefore", "Additionally", "Based", "Passage",
    "Directions", "Ingredients", "Instructions", "Method", "Step", "Steps", "Note", "Notes", "Tip", "Tips",
} | set(MONTHS) | set(WEEKDAYS))
ORG_CUES = frozenset({"Inc", "Corp", "Corporation", "Company", "University", "College", "Institute", "Association", "Agency", "Committee", "Council", "Bank", "Foundation", "Department", "Ministry", "Ltd"})
PLACE_PREP = frozenset({"in", "at", "from", "to", "near", "across", "within", "outside"})
META_CUE = re.compile(r"\b(?:not mentioned|source states|correct answer|should be|generative|original:)\b", re.I)
QUANTITY_UNIT = re.compile(r"^\s*(?:degrees?|°[CF]?|percent(?:age)?|million|billion|thousand|hundred|psi\b|mph\b|km\b|kg\b|g\b|mg\b|lb\b|lbs\b|oz\b|inches?\b|feet\b|foot\b|yards?\b|miles?\b|meters?\b|liters?\b|ml\b|calories?\b|people\b|workers?\b|items?\b|times?\b|points?\b|votes?\b|seats?\b|dollars?\b)", re.I)
QUANTITY_LEFT = re.compile(r"\b(?:cost|price|amount|total|average|mean|median|maximum|minimum|rate|rank|top|age|score|weight|height|length|width|income|salary|tax|revenue|population|number|count|including|contains?|collected|generated|won|lost|played)\D{0,18}$", re.I)
INDEX_LEFT = re.compile(r"\b(?:fact|step|passage|chapter|verse|figure|fig|item|option|section|part)\s*$", re.I)
EVALUATIVE = re.compile(r"\b(?:great|good|better|best|right|easy|helpful?|reminder|eligible|prefer(?:red)?|daunting|beautiful|wonderful)\b", re.I)
ANAPHORIC_START = re.compile(r"^(?:It|They|He|She|This|That|These|Those)\b")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [x.group(0).casefold() for x in WORD.finditer(text)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
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
            match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            final_token = match.group(0).lower() if match else ""
            dotted = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", final_token))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", final_token))
            protected = end < len(text) and (final_token in ABBREVIATIONS or dotted or initial)
            if (end == len(text) or text[end].isspace()) and not LIST_PREFIX.fullmatch(prefix) and not protected:
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
        if left < right and any(c.isalnum() for c in text[left:right]):
            spans.append((left, right))
        start = stop
    return spans


def parse_passages(text: str) -> list[dict]:
    matches = list(PASSAGE_HEADER.finditer(text))
    assert [int(x.group(1)) for x in matches] == [1, 2, 3]
    result = []
    for i, match in enumerate(matches):
        left = match.end()
        right = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[left:right].strip()
        sentences = [canonical(body[a:b]) for a, b in sentence_spans(body)]
        assert sentences
        result.append({"passage_id": int(match.group(1)), "text": body, "sentences": sentences})
    return result


def entity_kind(text: str, sentence: str, start: int, end: int) -> str | None:
    words_ = text.split()
    if text in ENTITY_STOP or any(x in ENTITY_STOP for x in words_) and len(words_) == 1:
        return None
    if len(words_) == 1 and start == next((m.start() for m in WORD.finditer(sentence)), -1):
        return None
    clean_words = [re.sub(r"[^A-Za-z]", "", x) for x in words_]
    if len(words_) >= 2 and (clean_words[-1] in ORG_CUES or ("of" in [x.casefold() for x in clean_words] and any(x in ORG_CUES for x in clean_words))):
        return "organization"
    prefix = sentence[max(0, start - 12):start].casefold()
    suffix = sentence[end:min(len(sentence), end + 32)]
    if re.match(r"\s*\((?:born|died)\b", suffix, re.I):
        return "person"
    if re.search(r"\b(?:mr|mrs|ms|dr|prof|president|senator|governor)\.?\s+$", prefix, re.I):
        return "person"
    return None


def extract_entities(sentence: str) -> list[dict]:
    out = []
    for match in CAP_SEQUENCE.finditer(sentence):
        text = match.group(0)
        kind = entity_kind(text, sentence, match.start(), match.end())
        if kind and sentence.count(text) == 1:
            out.append({"start": match.start(), "end": match.end(), "text": text, "kind": kind, "word_count": len(text.split())})
    return out


def entity_subtype(kind: str, text: str) -> str:
    if kind == "person":
        return "person"
    low = text.casefold()
    if any(x in low for x in ("university", "college", "institute")):
        return "academic"
    if any(x in low for x in ("agency", "council", "department", "ministry", "bureau")):
        return "public_body"
    if any(x in low for x in ("bank", "company", "corporation", "corp", " inc")):
        return "company"
    if "association" in low:
        return "association"
    return "other_organization"


def has_surface_entity_proxy(text: str) -> bool:
    first = next((m.start() for m in WORD.finditer(text)), -1)
    for match in CAP_SEQUENCE.finditer(text):
        if match.group(0) in ENTITY_STOP:
            continue
        if len(match.group(0).split()) > 1 or match.start() != first:
            return True
    return False


def replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return canonical(text[:start] + replacement + text[end:])


def changed_number(raw: str) -> str:
    plain = raw.replace(",", "")
    if "." in plain:
        places = len(plain.split(".", 1)[1])
        value = float(plain) + 10 ** (-places)
        return f"{value:.{places}f}"
    value = int(plain)
    changed = value + 1 if not str(value).endswith("9") else max(0, value - 1)
    value_text = str(changed)
    if "," in raw:
        value_text = f"{changed:,}"
    return value_text


def preserve_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    return replacement


def jaccard(a: str, b: str) -> float:
    aa, bb = set(tokens(a)), set(tokens(b))
    return len(aa & bb) / len(aa | bb) if aa | bb else 0.0


def base_sentence_ok(sentence: str) -> bool:
    n = len(tokens(sentence))
    first_alpha = next((c for c in sentence if c.isalpha()), "")
    return (6 <= n <= 60 and len(sentence) <= 500 and "?" not in sentence
            and not QUESTION_START.match(sentence) and bool(first_alpha) and first_alpha.isupper())


def make_row(owner: dict, sentence: dict, kind: str, supported: str, corrupted: str,
             start: int, end: int, old: str, new: str, premise: str | None = None,
             extra: dict | None = None) -> dict:
    payload = "|".join([owner["group_id"], str(sentence["passage_id"]), sentence["text"], kind, str(start), old, new])
    pair_id = f"sca1_{sha_text(payload)[:20]}"
    changed = supported[:start] + new + supported[end:] if kind != "attribution" else corrupted
    assert canonical(changed) == corrupted
    return {
        "schema_version": "semantic-conflict-augmentation-v1",
        "pair_id": pair_id,
        "partition": "fit",
        "response_id": str(owner["response_id"]),
        "source_id": str(owner["source_id"]),
        "group_id": owner["group_id"],
        "question": owner["question"],
        "passage_id": sentence["passage_id"],
        "sentence_id": sentence["sentence_id"],
        "source_sentence": sentence["text"],
        "source_sentence_sha256": sha_text(sentence["text"]),
        "nli_premise": premise or sentence["text"],
        "supported_claim": supported,
        "corrupted_claim": corrupted,
        "corruption_type": kind,
        "edit": {"start": start, "end": end, "original": old, "replacement": new},
        "labels": {
            "supported_claim": "entailed_by_exact_source_sentence",
            "corrupted_claim": "controlled_corruption_silver",
            "label_level": "silver_not_human_gold",
        },
        "checks": {
            "single_contiguous_edit": True,
            "source_sentence_exact_in_owner_passage": True,
            "supported_and_corrupt_differ": supported != corrupted,
            "corrupted_claim_exact_absent_from_all_owner_passages": canonical(corrupted) not in owner["all_source_sentences"],
            "metadata_cue_absent": not META_CUE.search(supported + " " + corrupted),
        },
        "extra": extra or {},
    }


def number_pair(owner: dict, sentence: dict, temporal_spans: list[tuple[int, int]]) -> dict | None:
    text = sentence["text"]
    for match in NUMBER.finditer(text):
        a, b = match.span()
        if any(a < y and b > x for x, y in temporal_spans):
            continue
        before = text[max(0, a - 24):a]
        after = text[b:min(len(text), b + 16)]
        if not before.strip() or INDEX_LEFT.search(before):
            continue
        if (len(match.group("num").replace(",", "")) == 1
                and re.search(r"(?:^|[.!?:])\s*$", before)
                and re.match(r"\s+[A-Z]", after)):
            continue
        if (a > 0 and text[a - 1] in ".:/-") or (b < len(text) and text[b] in ".:/-"):
            continue
        if HEDGE.search(before) or re.match(r"\s*(?:-|–|—|to\b)", after, re.I) or re.search(r"(?:-|–|—|\bto)\s*$", before, re.I):
            continue
        explicit_quantity = bool(match.group("prefix") or match.group("suffix") or QUANTITY_UNIT.search(after) or QUANTITY_LEFT.search(before))
        if not explicit_quantity:
            continue
        raw = match.group("num")
        new_num = changed_number(raw)
        new = f"{match.group('prefix') or ''}{new_num}{match.group('suffix') or ''}"
        old = match.group(0)
        corrupted = replace_span(text, a, b, new)
        if canonical(corrupted) in owner["all_source_sentences"]:
            continue
        return make_row(owner, sentence, "number", text, corrupted, a, b, old, new)
    return None


def temporal_pair(owner: dict, sentence: dict) -> tuple[dict | None, list[tuple[int, int]]]:
    text = sentence["text"]
    candidates = []
    years = list(YEAR.finditer(text))
    years = [m for m in years if not ((m.start() > 0 and text[m.start() - 1] in "-/") or (m.end() < len(text) and text[m.end()] in "-/"))]
    if len(years) == 1:
        m = years[0]
        new = str(int(m.group(0)) + 1)
        candidates.append((m.start(), m.end(), m.group(0), new, "year_shift"))
    times = list(TIME.finditer(text))
    if len(times) == 1:
        m = times[0]
        hour = int(m.group("hour"))
        new_hour = (hour % 12) + 1 if m.group("ampm") else (hour + 1) % 24
        new = f"{new_hour}:{m.group('minute')}" + (m.group("ampm") or "")
        candidates.append((m.start(), m.end(), m.group(0), new, "clock_shift"))
    months = list(MONTH.finditer(text))
    if len(months) == 1:
        m = months[0]
        idx = [x.casefold() for x in MONTHS].index(m.group(0).casefold())
        new = preserve_case(MONTHS[(idx + 1) % len(MONTHS)], m.group(0))
        candidates.append((m.start(), m.end(), m.group(0), new, "month_shift"))
    weekdays = list(WEEKDAY.finditer(text))
    if len(weekdays) == 1:
        m = weekdays[0]
        idx = [x.casefold() for x in WEEKDAYS].index(m.group(0).casefold())
        new = preserve_case(WEEKDAYS[(idx + 1) % len(WEEKDAYS)], m.group(0))
        candidates.append((m.start(), m.end(), m.group(0), new, "weekday_shift"))
    durations = list(DURATION.finditer(text))
    if len(durations) == 1:
        m = durations[0]
        if not HEDGE.search(text[max(0, m.start() - 24):m.start()]):
            n = m.group("num")
            new_n = changed_number(n)
            new = new_n + text[m.start("num") + len(n):m.end()]
            candidates.append((m.start(), m.end(), m.group(0), new, "duration_shift"))
    occupied = [(a, b) for a, b, _, _, _ in candidates]
    for a, b, old, new, subtype in candidates:
        corrupted = replace_span(text, a, b, new)
        if canonical(corrupted) in owner["all_source_sentences"]:
            continue
        return make_row(owner, sentence, "temporal", text, corrupted, a, b, old, new, extra={"temporal_subtype": subtype}), occupied
    return None, occupied


def negation_pair(owner: dict, sentence: dict) -> dict | None:
    text = sentence["text"]
    match = CONTRACTION.search(text)
    if match:
        replacement = preserve_case(CONTRACTION_MAP[match.group(0).casefold()], match.group(0))
        corrupted = replace_span(text, match.start(), match.end(), replacement)
        if canonical(corrupted) not in owner["all_source_sentences"]:
            return make_row(owner, sentence, "negation", text, corrupted, match.start(), match.end(), match.group(0), replacement, extra={"negation_operation": "remove"})
    match = AUX_NOT.search(text)
    if match:
        not_match = re.search(r"\bnot\b", match.group(0), re.I)
        assert not_match
        a = match.start() + not_match.start()
        b = match.start() + not_match.end()
        left = a - 1 if a > 0 and text[a - 1].isspace() else a
        corrupted = replace_span(text, left, b, "")
        if canonical(corrupted) not in owner["all_source_sentences"]:
            return make_row(owner, sentence, "negation", text, corrupted, left, b, text[left:b], "", extra={"negation_operation": "remove"})
    if NEG_ANY.search(text) or re.search(r"\b(?:may|might|possibly|perhaps|sometimes|often)\b", text, re.I):
        return None
    match = AUX_INSERT.search(text)
    if match:
        clause_prefix = text[max(text.rfind(".", 0, match.start()), text.rfind(";", 0, match.start()), text.rfind(",", 0, match.start())) + 1:match.start()]
        if re.search(r"\b(?:if|when|unless)\b", clause_prefix, re.I):
            return None
        a = b = match.end()
        corrupted = replace_span(text, a, b, " not")
        if canonical(corrupted) not in owner["all_source_sentences"]:
            return make_row(owner, sentence, "negation", text, corrupted, a, b, "", " not", extra={"negation_operation": "insert"})
    return None


def build_entity_pairs(owners: list[dict]) -> list[dict]:
    """Create disjoint, type-matched swaps; donor groups remain traceable."""
    occurrences = []
    for owner in owners:
        for sentence in owner["sentences"]:
            if not base_sentence_ok(sentence["text"]):
                continue
            for entity in sentence["entities"]:
                if entity["kind"] not in {"person", "organization"}:
                    continue
                if ((entity["start"] > 0 and sentence["text"][entity["start"] - 1] == "-")
                        or (entity["end"] < len(sentence["text"]) and sentence["text"][entity["end"]] == "-")
                        or re.match(r"\s*\([A-Z]{2,8}\)", sentence["text"][entity["end"]:])):
                    continue
                occurrences.append({"owner": owner, "sentence": sentence, **entity})
    # One source occurrence per surface identity prevents a frequent entity from
    # dominating.  Exact word-count matching keeps the edit syntactically close.
    identities = {}
    for item in occurrences:
        item["entity_subtype"] = entity_subtype(item["kind"], item["text"])
        key = (item["kind"], item["entity_subtype"], item["word_count"], item["text"].casefold().startswith("the "), item["text"].casefold())
        identities.setdefault(key, item)
    buckets = defaultdict(list)
    for item in identities.values():
        buckets[(item["kind"], item["entity_subtype"], item["word_count"], item["text"].casefold().startswith("the "))].append(item)
    result = []
    for bucket_key, items in sorted(buckets.items()):
        items.sort(key=lambda x: sha_text("entity-pair-v1|" + x["owner"]["group_id"] + "|" + x["text"]))
        pending = list(items)
        while len(pending) >= 2:
            left = pending.pop(0)
            partner_index = next((i for i, x in enumerate(pending)
                                  if x["text"] not in left["sentence"]["text"]
                                  and left["text"] not in x["sentence"]["text"]
                                  and x["text"] not in left["owner"]["all_source_sentences"]
                                  and left["text"] not in x["owner"]["all_source_sentences"]), None)
            if partner_index is None:
                continue
            right = pending.pop(partner_index)
            for target, donor in ((left, right), (right, left)):
                text = target["sentence"]["text"]
                corrupted = replace_span(text, target["start"], target["end"], donor["text"])
                if canonical(corrupted) in target["owner"]["all_source_sentences"]:
                    continue
                row = make_row(target["owner"], target["sentence"], "entity", text, corrupted,
                               target["start"], target["end"], target["text"], donor["text"],
                               extra={"entity_kind": target["kind"], "donor_sentence_key": donor["sentence"]["sentence_key"],
                                      "entity_subtype": target["entity_subtype"], "donor_entity_subtype": donor["entity_subtype"],
                                      "donor_group_id": donor["owner"]["group_id"], "donor_source_id": str(donor["owner"]["source_id"])})
                if all(row["checks"].values()):
                    result.append(row)
    return result


def assign_split_components(pairs: list[dict]) -> None:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[max(aa, bb)] = min(aa, bb)

    sentence_owner = {}
    for pair in pairs:
        find(pair["group_id"])
        sentence_hash = pair["source_sentence_sha256"]
        if sentence_hash in sentence_owner:
            union(pair["group_id"], sentence_owner[sentence_hash])
        else:
            sentence_owner[sentence_hash] = pair["group_id"]
        donor = pair["extra"].get("donor_group_id")
        if donor:
            union(pair["group_id"], donor)
    members = defaultdict(list)
    for group in list(parent):
        members[find(group)].append(group)
    ids = {root: "sca_component_" + sha_text("|".join(sorted(group_list)))[:16] for root, group_list in members.items()}
    for pair in pairs:
        root = find(pair["group_id"])
        pair["split_group_id"] = ids[root]
        pair["split_component_groups"] = sorted(members[root])


def strict_rule_reasons(pair: dict) -> list[str]:
    source = pair["source_sentence"]
    reasons = []
    if "..." in source or "....." in source:
        reasons.append("malformed_or_incomplete_source")
    if pair["corruption_type"] == "negation":
        if EVALUATIVE.search(source):
            reasons.append("evaluative_predicate")
        if re.search(r"\bcan\b", source, re.I):
            reasons.append("broad_ability_modal")
    if pair["corruption_type"] == "number":
        if re.search(r"\b(?:blog|menu|heading|index)\b", source, re.I) and not re.search(r"\b(?:is|are|was|were|has|have|will|costs?|contains?)\b", source, re.I):
            reasons.append("title_or_navigation_fragment")
    if pair["corruption_type"] == "attribution":
        if ANAPHORIC_START.match(source):
            reasons.append("sentence_external_referent")
        if len(tokens(source)) <= 8 and not re.search(r"\b(?:is|are|was|were|has|have|had|will|can|could|should|must|want|need|states?|reports?|shows?|uses?|makes?|contains?)\b", source, re.I):
            reasons.append("title_or_navigation_fragment")
    if pair["corruption_type"] == "entity":
        if pair["extra"].get("entity_subtype") != pair["extra"].get("donor_entity_subtype"):
            reasons.append("entity_subtype_mismatch")
    return reasons


def attribution_pair(owner: dict, sentence: dict, passages: dict[int, list[str]]) -> dict | None:
    text, p = sentence["text"], sentence["passage_id"]
    if not base_sentence_ok(text):
        return None
    if sum(canonical(text) in {canonical(x) for x in sents} for sents in passages.values()) != 1:
        return None
    options = []
    for q, sents in passages.items():
        if q == p:
            continue
        best = max(sents, key=lambda x: jaccard(text, x))
        score = jaccard(text, best)
        if score < 0.35:
            options.append((score, q, best))
    if not options:
        return None
    _, q, distractor = min(options, key=lambda x: (x[0], x[1]))
    supported = f"Passage {p} states: {text}"
    corrupted = f"Passage {q} states: {text}"
    digit = supported.index(str(p))
    premise = f"Passage {p}: {text}\nPassage {q}: {distractor}"
    return make_row(owner, sentence, "attribution", supported, corrupted, digit, digit + 1, str(p), str(q), premise=premise,
                    extra={"corrupted_passage_id": q, "distractor_sentence": distractor, "max_target_passage_jaccard": jaccard(text, distractor)})


def read_fit() -> list[dict]:
    assert FIT.resolve() == (ROOT / "data/fit.jsonl").resolve()
    rows = []
    with FIT.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["partition"] == "fit"
            rows.append({
                "partition": "fit",
                "response_id": str(row["response_id"]),
                "source_id": str(row["source_id"]),
                "group_id": row["group_id"],
                "question": row["question"],
                "retrieved_passages": row["retrieved_passages"],
            })
    assert len(rows) == 634
    return rows


def build() -> tuple[list[dict], dict, list[dict]]:
    raw = read_fit()
    owners = []
    for row in raw:
        passages = parse_passages(row["retrieved_passages"])
        passage_map = {x["passage_id"]: x["sentences"] for x in passages}
        sentences = []
        for passage in passages:
            for sentence_id, text in enumerate(passage["sentences"]):
                sentence = {
                    "passage_id": passage["passage_id"], "sentence_id": sentence_id,
                    "text": text, "sentence_key": f"{passage['passage_id']}:{sentence_id}:{sha_text(text)[:12]}",
                }
                sentence["entities"] = extract_entities(text)
                sentences.append(sentence)
        owner = dict(row)
        owner["passages"] = passage_map
        owner["sentences"] = sentences
        owner["all_source_sentences"] = {canonical(x["text"]) for x in sentences}
        owners.append(owner)

    pairs = []
    rejected = Counter()
    per_group_attr = Counter()
    for owner in owners:
        for sentence in owner["sentences"]:
            if not base_sentence_ok(sentence["text"]):
                rejected["base_sentence_filter"] += 1
                continue
            temporal, temporal_spans = temporal_pair(owner, sentence)
            candidates = [number_pair(owner, sentence, temporal_spans), temporal, negation_pair(owner, sentence)]
            if per_group_attr[(owner["group_id"], sentence["passage_id"])] == 0:
                attr = attribution_pair(owner, sentence, owner["passages"])
                if attr:
                    candidates.append(attr)
                    per_group_attr[(owner["group_id"], sentence["passage_id"])] += 1
            for pair in candidates:
                if pair is None:
                    continue
                if not all(pair["checks"].values()):
                    rejected[f"failed_pair_checks__{pair['corruption_type']}"] += 1
                    continue
                pairs.append(pair)

    pairs.extend(build_entity_pairs(owners))

    unique = {}
    for pair in pairs:
        key = (pair["group_id"], pair["source_sentence"], pair["corruption_type"], pair["corrupted_claim"])
        unique.setdefault(key, pair)
    pairs = sorted(unique.values(), key=lambda x: (x["group_id"], x["passage_id"], x["sentence_id"], x["corruption_type"], x["pair_id"]))
    assign_split_components(pairs)

    for pair in pairs:
        reasons = strict_rule_reasons(pair)
        pair["strict_rule_eligible"] = not reasons
        pair["strict_exclusion_reasons"] = reasons
    strict_pairs = [x for x in pairs if x["strict_rule_eligible"]]

    by_type = Counter(x["corruption_type"] for x in pairs)
    strict_by_type = Counter(x["corruption_type"] for x in strict_pairs)
    strict_groups_by_type = {kind: len({x["group_id"] for x in strict_pairs if x["corruption_type"] == kind}) for kind in sorted(strict_by_type)}
    hash_components = defaultdict(set)
    for pair in strict_pairs:
        hash_components[pair["source_sentence_sha256"]].add(pair["split_group_id"])
    groups_by_type = {kind: len({x["group_id"] for x in pairs if x["corruption_type"] == kind}) for kind in sorted(by_type)}
    qc = []
    for kind in ("entity", "number", "negation", "temporal", "attribution"):
        candidates = [x for x in strict_pairs if x["corruption_type"] == kind]
        candidates.sort(key=lambda x: sha_text("qc-audit-v2|" + x["pair_id"]))
        qc.extend(candidates[:20])
    assert len(qc) <= 100

    stats = {
        "status": "complete_fit_only_generation_and_rule_qc_sample",
        "scope": {"input": str(FIT.relative_to(ROOT)), "fit_rows": len(raw), "groups": len({x["group_id"] for x in raw}), "pair_selection_used_human_labels": False, "calibration_read": False, "test_read": False, "gpu_used": False, "trained": False},
        "silver_pair_pool": {"pairs": len(pairs), "by_type": dict(by_type), "groups_by_type": groups_by_type, "unique_groups": len({x["group_id"] for x in pairs}), "split_components": len({x["split_group_id"] for x in pairs}), "largest_split_component_groups": max(len(x["split_component_groups"]) for x in pairs), "rejected": dict(rejected)},
        "strict_silver_subset": {"pairs": len(strict_pairs), "by_type": dict(strict_by_type), "groups_by_type": strict_groups_by_type, "groups": len({x["group_id"] for x in strict_pairs}), "all_structural_checks_pass": all(all(x["checks"].values()) for x in strict_pairs), "duplicate_source_sentence_hashes_across_split_components": sum(len(v) > 1 for v in hash_components.values()), "excluded_by_reason": dict(Counter(reason for x in pairs for reason in x["strict_exclusion_reasons"]))},
        "rule_qc_sample": {"rows": len(qc), "by_type": dict(Counter(x["corruption_type"] for x in qc)), "selection": "post-freeze audit: 20 smallest sha256(qc-audit-v2|pair_id) per type"},
        "label_boundary": {
            "positive": "Exact source sentence (or exact passage attribution) supplies mechanically supported supervision.",
            "negative": "Controlled corruption is synthetic silver, not a human-verified false fact and not a RAGTruth label.",
            "evaluation": "These pairs are auxiliary fit-only training/stress data; never an evaluation set.",
        },
        "input_sha256": file_sha256(FIT),
    }
    return pairs, strict_pairs, stats, qc


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    pairs, strict_pairs, stats, qc = build()
    write_jsonl(HERE / "pairs_fit_silver.jsonl", pairs)
    write_jsonl(HERE / "pairs_fit_silver_strict.jsonl", strict_pairs)
    write_jsonl(HERE / "RULE_QC_AUDIT_MAX100.jsonl", qc)
    write_json(HERE / "STATS.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
