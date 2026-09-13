"""Deterministic CPU-only atomic-microclaim and relation-feature audit.

The preparation stage is label blind.  It refines the frozen sentence-like
claim geometry already used by the QA development pipeline, extracts only
surface-form relation metadata, and writes new standalone artifacts.  The
audit stage may read fit/calibration gold solely to measure coverage.  No
model is fitted, no threshold is selected, and no official test artifact is
opened.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
PARTITIONS = ("fit", "calibration")
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}

RAW = {part: DATA / f"{part}.jsonl" for part in PARTITIONS}
ANSWERS = {part: DATA / f"answers_{part}.jsonl" for part in PARTITIONS}
TOKENS = {part: DATA / f"tokens_{part}.jsonl" for part in PARTITIONS}
WINDOWS = {part: DATA / f"windows_k4_{part}.jsonl" for part in PARTITIONS}
INCUMBENT_PLANS = ROOT / "semantic_baseline" / "cuda_variant" / "plans.jsonl"
INCUMBENT_PLAN_MANIFEST = ROOT / "semantic_baseline" / "cuda_variant" / "plan_manifest.json"
CURRENT_ERROR_AUDIT = ROOT / "research" / "current_candidate_error_audit_r32_v1" / "AUDIT.json"
CURRENT_SPANS = ROOT / "research" / "current_candidate_error_audit_r32_v1" / "span_records.jsonl"

SCHEMA_VERSION = "ragtruth-qa-atomic-relation-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def write_json_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite existing artifact: {path}")
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def write_jsonl_new(path: Path, rows) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite existing artifact: {path}")
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def ratio(a: int | float, b: int | float) -> float | None:
    return float(a / b) if b else None


def mean(values) -> float | None:
    values = list(values)
    return float(statistics.fmean(values)) if values else None


def quantile(values, q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")
LEADING_CONNECTOR = re.compile(r"^\s*(?:(?:and|but|yet|or|whereas|while|which|who)\b[,;:]?\s*)+", re.I)
ATTRIBUTION_PREFIXES = (
    re.compile(r"^\s*(?:according\s+to|as\s+(?:stated|noted|described|reported)\s+in)\s+passages?\s+(?:no\.?\s*)?\d+(?:\s*(?:[,/&]|and)\s*\d+)*\s*[,;:]?\s*", re.I),
    re.compile(r"^\s*passage\s+\d+\s+(?:states?|says?|mentions?|notes?|reports?|explains?|indicates?|suggests?|provides?)\s+(?:that\s+)?", re.I),
    re.compile(r"^\s*(?:based\s+on|from)\s+(?:the\s+)?(?:provided\s+)?passages?(?:\s+\d+)?\s*[,;:]?\s*", re.I),
)
DISCOURSE_PREFIX = re.compile(
    r"^\s*(?:(?:therefore|thus|however|additionally|moreover|finally|overall|in summary|for example|for instance)\b[,;:]?\s*)+",
    re.I,
)

AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "cannot", "need", "needs", "needed",
}
COMMON_VERBS = {
    "act", "acts", "allow", "allows", "appear", "appears", "apply", "applies", "become",
    "becomes", "begin", "begins", "belong", "belongs", "cause", "causes", "change", "changes",
    "come", "comes", "contain", "contains", "continue", "continues", "cost", "costs", "create",
    "creates", "cry", "cries", "depend", "depends", "describe", "describes", "determine",
    "determines", "differ", "differs", "end", "ends", "equal", "equals", "exceed", "exceeds",
    "fall", "falls", "feel", "feels", "find", "finds", "follow", "follows", "form", "forms",
    "give", "gives", "go", "goes", "grow", "grows", "happen", "happens", "help", "helps",
    "hold", "holds", "include", "includes", "indicate", "indicates", "involve", "involves",
    "lack", "lacks", "lead", "leads", "leave", "leaves", "make", "makes", "mean", "means",
    "mention", "mentions", "occur", "occurs", "offer", "offers", "pay", "pays", "provide",
    "provides", "reach", "reaches", "refer", "refers", "represent", "represents", "require",
    "requires", "result", "results", "rise", "rises", "rose", "say", "says", "seem", "seems", "show",
    "shows", "sleep", "sleeps", "slept", "start", "starts", "state", "states", "stop", "stops",
    "suggest", "suggests", "take", "takes", "use", "uses", "vary", "varies", "win", "wins",
    "work", "works",
}


def strip_scaffold(text: str) -> tuple[str, int]:
    """Return assertion-like suffix and its local offset without changing source text."""
    cursor = 0
    for pattern in (LIST_MARKER, LEADING_CONNECTOR, DISCOURSE_PREFIX):
        match = pattern.match(text[cursor:])
        if match:
            cursor += match.end()
    changed = True
    while changed:
        changed = False
        for pattern in ATTRIBUTION_PREFIXES:
            match = pattern.match(text[cursor:])
            if match:
                cursor += match.end()
                changed = True
                break
    match = DISCOURSE_PREFIX.match(text[cursor:])
    if match:
        cursor += match.end()
    return text[cursor:], cursor


def is_predicate_token(token: str) -> bool:
    lower = token.lower().replace("’", "'")
    if lower in AUXILIARIES or lower in COMMON_VERBS or lower.endswith("n't"):
        return True
    if len(lower) >= 5 and lower.endswith(("ed", "ing", "ize", "izes", "ized", "ise", "ises", "ised", "ify", "ifies", "ified")):
        return True
    return False


def predicate_tokens(text: str):
    core, offset = strip_scaffold(text)
    return [(offset + m.start(), offset + m.end(), m.group()) for m in WORD_TOKEN.finditer(core) if is_predicate_token(m.group())]


def predicate_like(text: str) -> bool:
    words = WORD.findall(text)
    return len(words) >= 2 and bool(predicate_tokens(text))


def right_independent_enough(text: str) -> bool:
    """Conservative guard against splitting noun/object coordination."""
    core, _ = strip_scaffold(text)
    tokens = list(WORD_TOKEN.finditer(core))
    if not tokens:
        return False
    first = tokens[0].group().lower()
    if first in AUXILIARIES or first in COMMON_VERBS or first.endswith(("ed", "ing", "s")):
        return predicate_like(text)
    if first in {"he", "she", "it", "they", "we", "i", "you", "this", "that", "these", "those", "who", "which"}:
        return predicate_like(text)
    # Explicit noun phrase plus a later predicate.
    predicates = predicate_tokens(text)
    return bool(predicates and predicates[0][0] > 0)


SEMICOLON = re.compile(r";")
CONTRAST = re.compile(r",?\s*\b(?:but|yet|whereas|while)\b\s+", re.I)
COORDINATION = re.compile(r",?\s*\b(?:and|or)\b\s+", re.I)
RELATIVE = re.compile(r",\s*\b(?:which|who)\b\s+", re.I)
RESULT_PARTICIPLE = re.compile(r",\s*\b(?:suggesting|indicating|meaning|resulting|making|causing)\b\s+", re.I)


def split_candidate(text: str, start: int, end: int):
    fragment = text[start:end]
    candidates = []
    for match in SEMICOLON.finditer(fragment):
        candidates.append((0, match.start(), match.end(), "semicolon"))
    for priority, pattern, reason in (
        (1, CONTRAST, "contrast_clause"),
        (2, RELATIVE, "relative_clause"),
        (3, RESULT_PARTICIPLE, "result_participle"),
        (4, COORDINATION, "coordinated_clause"),
    ):
        for match in pattern.finditer(fragment):
            connector = re.search(r"[A-Za-z]+", match.group())
            assert connector is not None
            candidates.append((priority, match.start(), match.start() + connector.start(), reason))
    for priority, split_at, right_at, reason in sorted(candidates, key=lambda x: (x[0], x[1])):
        left = fragment[:split_at].rstrip(" \t\r\n,")
        right = fragment[right_at:].lstrip()
        if len(WORD.findall(left)) < 2 or len(WORD.findall(right)) < 2:
            continue
        if reason == "coordinated_clause":
            left_tail = left[-80:].lower()
            right_head = right[:30].lower()
            if re.search(r"\bbetween\b[^,;]{0,60}$", left_tail):
                continue
            if re.search(r"\b(?:both|either|neither)\b[^,;]{0,60}$", left_tail):
                continue
            if right_head.startswith("or not"):
                continue
            valid_right = right_independent_enough(right)
        else:
            valid_right = predicate_like(right)
        if predicate_like(left) and valid_right:
            absolute_left_end = start + len(fragment[:split_at].rstrip(" \t\r\n,"))
            absolute_right_start = start + right_at + len(fragment[right_at:]) - len(fragment[right_at:].lstrip())
            if absolute_left_end > start and end > absolute_right_start:
                return absolute_left_end, absolute_right_start, reason
    return None


def atomic_split(text: str, start: int = 0, end: int | None = None):
    """Split one frozen parent claim into conservative relation-sized spans."""
    end = len(text) if end is None else end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    reasons = []

    def recurse(a: int, b: int, depth: int):
        if depth >= 12:
            return [(a, b)]
        selected = split_candidate(text, a, b)
        if selected is None:
            return [(a, b)]
        left_end, right_start, reason = selected
        reasons.append(reason)
        return recurse(a, left_end, depth + 1) + recurse(right_start, b, depth + 1)

    spans = recurse(start, end, 0)
    assert spans and all(a < b for a, b in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    parent_alnum = {i for i in range(start, end) if text[i].isalnum()}
    child_alnum = {i for a, b in spans for i in range(a, b) if text[i].isalnum()}
    assert child_alnum == parent_alnum, "Atomic splitting must preserve every alphanumeric character"
    return spans, reasons


CITATION_EXPR = r"\d+(?:\s*(?:[-–—/,;]|\band\b|\bto\b)\s*(?:and\s+)?\d+)*"
CITATION_PATTERNS = (
    ("named", re.compile(r"\bpassages?\s+(?:no\.?\s*)?(" + CITATION_EXPR + r")", re.I)),
    ("bracket", re.compile(r"\[\s*(" + CITATION_EXPR + r")\s*\]", re.I)),
)


def parse_citations(text: str, base: int = 0):
    references = []
    for kind, pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            expression = match.group(1)
            numbers = [int(v) for v in re.findall(r"\d+", expression)]
            ids = set(numbers)
            status = "recognized"
            for raw_a, raw_b in re.findall(r"(\d+)\s*(?:[-–—]|\bto\b)\s*(\d+)", expression, re.I):
                a, b = int(raw_a), int(raw_b)
                if a > b or b > 99:
                    status = "unknown"
                else:
                    ids.update(range(a, b + 1))
            if not ids or any(v < 1 or v > 99 for v in ids):
                status = "unknown"
            references.append({
                "start": base + match.start(), "end": base + match.end(), "text": match.group(),
                "kind": kind, "ids": sorted(ids), "status": status,
            })
    references.sort(key=lambda row: (row["start"], row["end"], row["kind"]))
    ids = sorted({value for row in references if row["status"] == "recognized" for value in row["ids"]})
    all_ids = sorted({value for row in references for value in row["ids"]})
    return {
        "references": references,
        "passage_ids": [value for value in ids if value in (1, 2, 3)],
        "invalid_passage_ids": [value for value in all_ids if value not in (1, 2, 3)],
        "has_unknown_syntax": any(row["status"] == "unknown" for row in references),
    }


NEGATION_PATTERNS = (
    ("not", re.compile(r"\bnot\b", re.I)),
    ("contracted_not", re.compile(r"\b[A-Za-z]+n['’]t\b", re.I)),
    ("never", re.compile(r"\bnever\b", re.I)),
    ("no", re.compile(r"\bno\b", re.I)),
    ("without", re.compile(r"\bwithout\b", re.I)),
    ("neither_nor", re.compile(r"\b(?:neither|nor)\b", re.I)),
    ("unable", re.compile(r"\bunable\b", re.I)),
    ("lack", re.compile(r"\b(?:lack|lacks|lacked|lacking)\b", re.I)),
    ("fail", re.compile(r"\b(?:fail|fails|failed|failing)\s+to\b", re.I)),
)

COMPARATOR_PATTERNS = (
    ("greater_or_equal", ">=", re.compile(r">=|\b(?:at\s+least|no\s+less\s+than)\b", re.I)),
    ("less_or_equal", "<=", re.compile(r"<=|\b(?:at\s+most|no\s+more\s+than)\b", re.I)),
    ("greater", ">", re.compile(r"(?<![<>=])>(?![=])|\b(?:more|greater|higher|larger|bigger|older|longer|faster)\s+than\b|\bexceeds?\b", re.I)),
    ("less", "<", re.compile(r"(?<![<>=])<(?![=])|\b(?:less|fewer|lower|smaller|younger|shorter|slower)\s+than\b|\bbelow\b", re.I)),
    ("equal", "=", re.compile(r"(?<![<>=])=(?![=])|\b(?:equal\s+to|the\s+same\s+as|exactly)\b", re.I)),
    ("range", "range", re.compile(r"\bbetween\b|\bfrom\b(?=.{0,40}\bto\b)", re.I)),
    ("contrast", "contrast", re.compile(r"\b(?:compared\s+(?:with|to)|versus|vs\.?|whereas|while)\b", re.I)),
    ("increase", "up", re.compile(r"\b(?:increase[sd]?|increasing|rise[sd]?|rising|grow(?:s|ing|n)?|gain(?:s|ed|ing)?)\b", re.I)),
    ("decrease", "down", re.compile(r"\b(?:decrease[sd]?|decreasing|decline[sd]?|declining|fall(?:s|ing)?|fell|drop(?:s|ped|ping)?)\b", re.I)),
)

TEMPORAL_PATTERNS = (
    ("before", re.compile(r"\b(?:before|prior\s+to|earlier\s+than)\b", re.I)),
    ("after", re.compile(r"\b(?:after|following|later\s+than)\b", re.I)),
    ("sequence", re.compile(r"\b(?:first|firstly|then|next|subsequently|finally|previously|later|earlier)\b", re.I)),
    ("boundary", re.compile(r"\b(?:until|since|during|by\s+(?:\d{1,2}(?::\d{2})?|\d{4}|the\s+time))\b", re.I)),
)

CONDITION_PATTERNS = (
    ("if", re.compile(r"\bif\b", re.I)),
    ("unless", re.compile(r"\bunless\b", re.I)),
    ("only_if", re.compile(r"\bonly\s+if\b", re.I)),
    ("provided", re.compile(r"\bprovided\s+that\b", re.I)),
    ("as_long_as", re.compile(r"\bas\s+long\s+as\b", re.I)),
    ("in_case", re.compile(r"\bin\s+case\b", re.I)),
    ("when", re.compile(r"\bwhen\b", re.I)),
    ("depending", re.compile(r"\bdepend(?:s|ed|ing)?\s+on\b", re.I)),
)


def collect_cues(text: str, patterns, base: int = 0, comparator: bool = False):
    rows = []
    for entry in patterns:
        if comparator:
            kind, operator, pattern = entry
        else:
            kind, pattern = entry
            operator = None
        for match in pattern.finditer(text):
            row = {"kind": kind, "start": base + match.start(), "end": base + match.end(), "text": match.group()}
            if operator is not None:
                row["operator"] = operator
            rows.append(row)
    # Longer cue wins for exact-overlap duplicates (e.g. only if vs if).
    rows.sort(key=lambda row: (row["start"], -(row["end"] - row["start"]), row["kind"]))
    kept = []
    for row in rows:
        if any(row["start"] >= old["start"] and row["end"] <= old["end"] for old in kept):
            continue
        kept.append(row)
    return sorted(kept, key=lambda row: (row["start"], row["end"], row["kind"]))


DATE = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
NUMBER = re.compile(
    r"(?<![\w])(?P<currency>[$€£])?\s*(?P<a>\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)(?P<ordinal>st|nd|rd|th)?"
    r"(?:\s*(?P<range>[-–—]|\bto\b)\s*(?P<b>\d+(?:,\d{3})*(?:\.\d+)?))?",
    re.I,
)
UNIT = re.compile(
    r"^\s*(?P<unit>%|percent(?:age)?|degrees?\s*[CF]?|°\s*[CF]?|milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"mm|cm|km|meters?|metres?|inches?|feet|foot|ft|yards?|miles?|mg|grams?|kg|kilograms?|ounces?|oz|pounds?|lbs?|"
    r"mph|km/h|kph|m/s|dollars?|euros?|pounds?\s+sterling|people|persons?|items?|seats?|votes?|times?)\b",
    re.I,
)
UNIT_NORMALIZATION = {
    "%": "percent", "percentage": "percent", "percent": "percent",
    "sec": "second", "secs": "second", "seconds": "second", "second": "second",
    "minutes": "minute", "minute": "minute", "hours": "hour", "hour": "hour",
    "days": "day", "day": "day", "weeks": "week", "week": "week", "months": "month", "month": "month",
    "years": "year", "year": "year", "meters": "meter", "metres": "meter", "meter": "meter", "metre": "meter",
    "inches": "inch", "inch": "inch", "feet": "foot", "ft": "foot", "yards": "yard", "yard": "yard",
    "miles": "mile", "mile": "mile", "grams": "gram", "gram": "gram", "kilograms": "kilogram", "kg": "kilogram",
    "ounces": "ounce", "ounce": "ounce", "oz": "ounce", "pounds": "pound", "lbs": "pound",
    "dollars": "usd", "dollar": "usd", "euros": "eur", "euro": "eur",
}


def normalize_unit(raw: str | None, currency: str | None) -> str | None:
    if currency:
        return {"$": "usd", "€": "eur", "£": "gbp"}[currency]
    if not raw:
        return None
    compact = " ".join(raw.lower().split())
    if compact.startswith("degree") or compact.startswith("°"):
        return "temperature"
    return UNIT_NORMALIZATION.get(compact, compact.rstrip("s"))


def quantities(text: str, base: int, citations):
    excluded = [(row["start"] - base, row["end"] - base) for row in citations["references"]]
    marker = LIST_MARKER.match(text)
    if marker:
        excluded.append((marker.start(), marker.end()))
    rows = []
    date_spans = []
    for match in DATE.finditer(text):
        rows.append({"start": base + match.start(), "end": base + match.end(), "text": match.group(),
                     "values": [match.group()], "unit": "date", "is_range": False})
        date_spans.append((match.start(), match.end()))
    for match in NUMBER.finditer(text):
        a, b = match.start(), match.end()
        if any(a < right and left < b for left, right in excluded + date_spans):
            continue
        unit_match = UNIT.match(text[b:])
        end = b + (unit_match.end() if unit_match else 0)
        raw_unit = unit_match.group("unit") if unit_match else None
        values = [match.group("a").replace(",", "")]
        if match.group("b"):
            values.append(match.group("b").replace(",", ""))
        rows.append({
            "start": base + a, "end": base + end, "text": text[a:end], "values": values,
            "unit": normalize_unit(raw_unit, match.group("currency")), "is_range": bool(match.group("b")),
            "ordinal": bool(match.group("ordinal")),
        })
    return sorted(rows, key=lambda row: (row["start"], row["end"]))


ENTITY = re.compile(r"\b(?:[A-Z][A-Za-z0-9'’.-]*)(?:\s+(?:[A-Z][A-Za-z0-9'’.-]*|of|the|and|for|in)){0,5}\b")
QUOTED = re.compile(r"[\"“](?P<value>[^\"”\n]{2,80})[\"”]")
ENTITY_STOP = {
    "a", "an", "and", "according", "additionally", "based", "but", "finally", "for", "from", "here",
    "however", "if", "in", "it", "overall", "passage", "passages", "the", "therefore", "these", "this",
    "thus", "unable", "when", "while",
}


def entity_candidates(text: str, base: int):
    rows = []
    for match in ENTITY.finditer(text):
        surface = match.group().strip()
        words = surface.split()
        if len(words) == 1 and surface.lower() in ENTITY_STOP:
            continue
        if re.fullmatch(r"Passages?", surface, re.I):
            continue
        rows.append({"start": base + match.start(), "end": base + match.end(), "text": surface, "kind": "capitalized"})
    for match in QUOTED.finditer(text):
        rows.append({"start": base + match.start("value"), "end": base + match.end("value"),
                     "text": match.group("value"), "kind": "quoted"})
    rows.sort(key=lambda row: (row["start"], -(row["end"] - row["start"]), row["kind"]))
    kept = []
    for row in rows:
        if any(row["start"] >= old["start"] and row["end"] <= old["end"] for old in kept):
            continue
        kept.append(row)
    return sorted(kept, key=lambda row: (row["start"], row["end"]))


def subject_predicate(text: str, base: int):
    core, local_offset = strip_scaffold(text)
    core_start = local_offset
    # Label-value statements such as "Blue: water and sky" have an implicit copula.
    colon = core.find(":")
    if 0 < colon <= 80 and 1 <= len(WORD.findall(core[:colon])) <= 8 and WORD.search(core[colon + 1:]):
        subject_text = core[:colon].strip(" \t\r\n-*+•,;:")
        subject_local = core_start + core[:colon].find(subject_text)
        return (
            {"text": subject_text, "start": base + subject_local, "end": base + subject_local + len(subject_text), "kind": "label_subject"},
            {"text": ":", "start": base + core_start + colon, "end": base + core_start + colon + 1, "kind": "label_value"},
            len(predicate_tokens(text)),
        )
    predicates = predicate_tokens(text)
    if not predicates:
        return None, None, 0
    pred_start, pred_end, pred_text = predicates[0]
    raw_subject = text[core_start:pred_start].strip(" \t\r\n-*+•,;:")
    if not raw_subject:
        subject = None
    else:
        # Prefer the clause after an attributional "that" if one remains.
        that_matches = list(re.finditer(r"\bthat\b\s+", raw_subject, re.I))
        if that_matches:
            raw_subject = raw_subject[that_matches[-1].end():].strip(" ,;:")
        if not raw_subject or len(WORD.findall(raw_subject)) > 14 or len(raw_subject) > 140:
            subject = None
        else:
            search_start = max(core_start, pred_start - len(raw_subject) - 8)
            subject_local = text.find(raw_subject, search_start, pred_start)
            assert subject_local >= 0
            subject = {"text": raw_subject, "start": base + subject_local, "end": base + subject_local + len(raw_subject), "kind": "surface_subject"}
    predicate = {"text": pred_text, "start": base + pred_start, "end": base + pred_end, "kind": "surface_predicate"}
    return subject, predicate, len(predicates)


def surface_features(text: str, base: int = 0):
    citations = parse_citations(text, base)
    negation = collect_cues(text, NEGATION_PATTERNS, base)
    comparator = collect_cues(text, COMPARATOR_PATTERNS, base, comparator=True)
    temporal = collect_cues(text, TEMPORAL_PATTERNS, base)
    condition = collect_cues(text, CONDITION_PATTERNS, base)
    number_unit = quantities(text, base, citations)
    entities = entity_candidates(text, base)
    subject, predicate, predicate_count = subject_predicate(text, base)
    return {
        "negation": negation,
        "comparator": comparator,
        "quantities": number_unit,
        "temporal_order": temporal,
        "condition": condition,
        "citations": citations,
        "local_subject": subject,
        "predicate": predicate,
        "predicate_count": predicate_count,
        "entity_candidates": entities,
    }


COREFERENCE_SUBJECTS = {"he", "she", "it", "they", "we", "i", "you", "this", "that", "these", "those", "who", "which"}
FEATURE_NAMES = (
    "has_negation", "has_comparator", "has_number_or_unit", "has_temporal_order", "has_condition",
    "has_explicit_passage_id", "has_parent_passage_context", "has_local_subject", "has_effective_subject",
    "has_entity_candidate", "has_any_core_relation", "has_any_relation_or_source",
)
CORE_FEATURES = ("has_negation", "has_comparator", "has_number_or_unit", "has_temporal_order", "has_condition")
RELATION_OR_SOURCE = CORE_FEATURES + ("has_explicit_passage_id", "has_parent_passage_context")


def make_microclaims(raw_row, plan_row):
    text = raw_row["original_response"]
    assert plan_row["response_id"] == raw_row["response_id"]
    assert plan_row["answer_sha256"] == raw_row["answer_sha256"]
    records = []
    split_parent_claims = 0
    split_reason_counts = Counter()
    global_index = 0
    for parent_index, parent in enumerate(plan_row["claims"]):
        assert text[parent["start"]:parent["end"]] == parent["text"]
        spans, reasons = atomic_split(text, parent["start"], parent["end"])
        second_spans, second_reasons = atomic_split(text, parent["start"], parent["end"])
        assert spans == second_spans and reasons == second_reasons
        split_parent_claims += int(len(spans) > 1)
        split_reason_counts.update(reasons)
        parent_citations = parse_citations(parent["text"], parent["start"])
        previous_effective_subject = None
        for child_index, (start, end) in enumerate(spans):
            claim_text = text[start:end]
            features = surface_features(claim_text, start)
            local_subject = features["local_subject"]
            local_key = local_subject["text"].strip(" ,;:").lower() if local_subject else None
            antecedent = None
            if previous_effective_subject is not None and (local_subject is None or local_key in COREFERENCE_SUBJECTS):
                antecedent = {**previous_effective_subject, "kind": "previous_atomic_sibling_antecedent"}
            effective = local_subject or antecedent
            if local_subject is not None and local_key in COREFERENCE_SUBJECTS and antecedent is not None:
                effective = local_subject
            if effective is not None and local_key not in COREFERENCE_SUBJECTS:
                previous_effective_subject = effective
            elif previous_effective_subject is None and effective is not None:
                previous_effective_subject = effective
            explicit_ids = features["citations"]["passage_ids"]
            parent_ids = parent_citations["passage_ids"]
            feature_vector = {
                "has_negation": bool(features["negation"]),
                "has_comparator": bool(features["comparator"]),
                "has_number_or_unit": bool(features["quantities"]),
                "has_temporal_order": bool(features["temporal_order"]),
                "has_condition": bool(features["condition"]),
                "has_explicit_passage_id": bool(explicit_ids),
                "has_parent_passage_context": bool(parent_ids),
                "has_local_subject": local_subject is not None,
                "has_effective_subject": effective is not None,
                "has_entity_candidate": bool(features["entity_candidates"]),
            }
            feature_vector["has_any_core_relation"] = any(feature_vector[name] for name in CORE_FEATURES)
            feature_vector["has_any_relation_or_source"] = any(feature_vector[name] for name in RELATION_OR_SOURCE)
            record = {
                "schema_version": SCHEMA_VERSION,
                "partition": raw_row["partition"],
                "response_id": raw_row["response_id"],
                "source_id": raw_row["source_id"],
                "group_id": raw_row["group_id"],
                "microclaim_id": f"{raw_row['response_id']}__atomic_{global_index:03d}",
                "microclaim_index": global_index,
                "parent_claim_index": parent_index,
                "parent_sentence_index": parent.get("sentence_index", parent_index),
                "child_index": child_index,
                "children_in_parent": len(spans),
                "parent_split_reasons": reasons,
                "start": start,
                "end": end,
                "text": claim_text,
                "word_count": len(WORD.findall(claim_text)),
                "available_passage_ids": [1, 2, 3],
                "explicit_passage_ids": explicit_ids,
                "invalid_explicit_passage_ids": features["citations"]["invalid_passage_ids"],
                "parent_passage_ids": parent_ids,
                "passage_context_inherited": bool(not explicit_ids and parent_ids),
                "local_subject": local_subject,
                "antecedent_subject": antecedent,
                "effective_subject": effective,
                "predicate": features["predicate"],
                "predicate_count": features["predicate_count"],
                "entity_candidates": features["entity_candidates"],
                "negation_cues": features["negation"],
                "comparator_cues": features["comparator"],
                "quantities": features["quantities"],
                "temporal_order_cues": features["temporal_order"],
                "condition_cues": features["condition"],
                "citation_parse": features["citations"],
                "feature_vector": feature_vector,
            }
            forbidden = {"label", "labels", "risk", "gold", "score", "prediction"} & set(record)
            assert not forbidden
            records.append(record)
            global_index += 1
    answer_alnum = {i for i, ch in enumerate(text) if ch.isalnum()}
    covered_alnum = {i for row in records for i in range(row["start"], row["end"]) if text[i].isalnum()}
    assert answer_alnum == covered_alnum
    return records, split_parent_claims, split_reason_counts


def feature_prevalence(records):
    result = {"microclaims": len(records)}
    for name in FEATURE_NAMES:
        count = sum(bool(row["feature_vector"][name]) for row in records)
        result[name] = {"count": count, "fraction": ratio(count, len(records))}
    result["explicit_passage_id_counts"] = dict(sorted(Counter(
        value for row in records for value in row["explicit_passage_ids"]
    ).items()))
    result["parent_passage_id_counts"] = dict(sorted(Counter(
        value for row in records for value in row["parent_passage_ids"]
    ).items()))
    signatures = Counter()
    for row in records:
        active = [name.removeprefix("has_") for name in RELATION_OR_SOURCE if row["feature_vector"][name]]
        signatures["+".join(active) if active else "none"] += 1
    result["relation_signature_top20"] = [
        {"signature": signature, "count": count, "fraction": ratio(count, len(records))}
        for signature, count in signatures.most_common(20)
    ]
    return result


def protocol():
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "RAGTruth QA original fit634 + calibration159 only; official test excluded by explicit path allowlist.",
        "parent_geometry": "Exact frozen 8,852 sentence-like answer claims from semantic_baseline/cuda_variant/plans.jsonl; used only as label-blind parent spans.",
        "atomic_split": {
            "boundaries": ["semicolon", "but/yet/whereas/while", "comma which/who", "result participle", "and/or"],
            "guard": "Both sides require a deterministic surface predicate; and/or additionally requires an independent-looking right clause. Between/both/either/neither coordination is protected.",
            "invariant": "Every answer alphanumeric character remains in exactly one ordered microclaim; whitespace/punctuation gaps may remain outside claims.",
            "determinism": "No model, parser download, random seed, corpus statistic, label, score, or threshold participates.",
        },
        "relation_features": {
            "negation": "Fixed lexical cues including not/n't/never/no/without/neither/nor/unable/lack/fail to.",
            "comparator": "Fixed symbols and ordered phrases for >, >=, <, <=, equality, range, contrast, increase, decrease.",
            "number_unit": "Digit/currency/date/range extraction with a fixed unit vocabulary; list markers and recognized passage citations excluded from quantities.",
            "temporal_order": "Fixed before/after/sequence/boundary cues.",
            "condition": "Fixed if/unless/only-if/provided/as-long-as/in-case/when/depending cues.",
            "source_passage": "Explicit claim IDs and parent-claim passage context kept separately; available IDs are 1,2,3 and dataset source_id stays categorical metadata.",
            "subject_entity": "Surface subject before first predicate, previous atomic sibling as marked antecedent, plus capitalized/quoted entity candidates. These are candidates, not linguistic gold.",
        },
        "prepare": "Writes label-free microclaim records and prevalence only.",
        "audit": "Reads original fit/cal gold after preparation to measure span/window coverage and relation-feature activation; gold never enters exported features.",
        "incumbent_context": "Calibration missed Evident Conflict spans are joined only for post-hoc diagnosis against the frozen R32 error audit.",
        "prohibitions": ["no training", "no threshold selection", "no baseline changes", "no GPU", "no official test"],
        "limits": [
            "Rules approximate syntax and coreference; candidate subjects/entities are not a dependency parse or NER.",
            "Feature presence is coverage, not proof that a feature predicts conflict.",
            "Parent passage context is exported separately from an explicit citation to avoid hiding inheritance.",
            "Calibration findings are post-selection diagnostics and not independent-test evidence.",
        ],
    }


def self_test():
    examples = {
        "contrast": "The two-toed sloth is larger, but the three-toed sloth is faster.",
        "noun_coordination": "Fish and chips are common.",
        "condition": "If rain falls, the game stops.",
        "source_quantity": "Passage 2 states that the box weighs 10 kg and costs $5.",
        "relative": "The value rose to 20%, which was higher than 2019.",
    }
    contrast, reasons = atomic_split(examples["contrast"])
    assert len(contrast) == 2 and reasons == ["contrast_clause"]
    noun, noun_reasons = atomic_split(examples["noun_coordination"])
    assert len(noun) == 1 and not noun_reasons
    conditional, conditional_reasons = atomic_split(examples["condition"])
    assert len(conditional) == 1 and not conditional_reasons
    assert surface_features(examples["condition"])["condition"]
    source_features = surface_features(examples["source_quantity"])
    assert source_features["citations"]["passage_ids"] == [2]
    assert [row["text"].strip() for row in source_features["quantities"]] == ["10 kg", "$5"]
    relation = surface_features("It is not less than 5 cm after 2020 unless reduced.")
    assert relation["negation"] and relation["comparator"] and relation["temporal_order"] and relation["condition"]
    assert len(relation["quantities"]) == 2
    relative, relative_reasons = atomic_split(examples["relative"])
    assert len(relative) == 2 and relative_reasons == ["relative_clause"]
    parsed = parse_citations("See passages 1, 2 and 3; not [2024].")
    assert parsed["passage_ids"] == [1, 2, 3] and parsed["invalid_passage_ids"] == [2024]
    assert json_digest(atomic_split(examples["contrast"])) == json_digest(atomic_split(examples["contrast"]))
    return {
        "passed": True,
        "tests": list(examples),
        "alphanumeric_coverage_checked": True,
        "citation_not_quantity_checked": True,
        "deterministic_repeat_checked": True,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }


def allowed_source_paths():
    paths = [Path(__file__), INCUMBENT_PLANS, INCUMBENT_PLAN_MANIFEST, CURRENT_ERROR_AUDIT, CURRENT_SPANS]
    for part in PARTITIONS:
        paths.extend((RAW[part], ANSWERS[part], TOKENS[part], WINDOWS[part]))
    for path in paths:
        name = path.name.lower()
        assert name not in {"test.jsonl", "answers_test.jsonl", "tokens_test.jsonl", "windows_k4_test.jsonl"}
        assert path.exists(), path
    return paths


def prepare():
    HERE.mkdir(parents=True, exist_ok=True)
    outputs = [HERE / "protocol.json", HERE / "CPU_SELFCHECK.json", HERE / "design_freeze.json",
               HERE / "microclaims_fit.jsonl", HERE / "microclaims_calibration.jsonl", HERE / "preparation.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preparation artifacts already exist; preserving them. Use verify instead.")
    checks = self_test()
    source_paths = allowed_source_paths()
    plan_rows = {row["response_id"]: row for row in read_jsonl(INCUMBENT_PLANS)}
    manifest = read_json(INCUMBENT_PLAN_MANIFEST)
    assert manifest["counts"]["answers"] == 793 and manifest["counts"]["claims"] == 8852
    all_by_partition = {}
    summaries = {}
    for part in PARTITIONS:
        raw_rows = list(read_jsonl(RAW[part]))
        answer_rows = {row["response_id"]: row for row in read_jsonl(ANSWERS[part])}
        assert len(raw_rows) == len(answer_rows) == EXPECTED_ANSWERS[part]
        records = []
        split_parents = 0
        split_reasons = Counter()
        parent_count = 0
        for raw_row in raw_rows:
            assert raw_row["partition"] == part
            answer = answer_rows[raw_row["response_id"]]
            assert answer["original_response"] == raw_row["original_response"]
            assert answer["answer_sha256"] == raw_row["answer_sha256"]
            plan = plan_rows[raw_row["response_id"]]
            parent_count += len(plan["claims"])
            made, split_count, reasons = make_microclaims(raw_row, plan)
            records.extend(made)
            split_parents += split_count
            split_reasons.update(reasons)
        assert len({row["microclaim_id"] for row in records}) == len(records)
        all_by_partition[part] = records
        words = [row["word_count"] for row in records]
        residual = sum(split_candidate(row["text"], 0, len(row["text"])) is not None for row in records)
        assert residual == 0
        summaries[part] = {
            "answers": len(raw_rows),
            "incumbent_parent_claims": parent_count,
            "atomic_microclaims": len(records),
            "added_microclaims": len(records) - parent_count,
            "split_parent_claims": split_parents,
            "split_parent_fraction": ratio(split_parents, parent_count),
            "split_reason_counts": dict(sorted(split_reasons.items())),
            "mean_words": mean(words),
            "p95_words": quantile(words, .95),
            "max_words": max(words),
            "residual_self_rule_split_candidates": residual,
            "feature_prevalence": feature_prevalence(records),
        }
    assert sum(value["incumbent_parent_claims"] for value in summaries.values()) == 8852
    protocol_value = protocol()
    input_hashes = {str(path.resolve()): sha(path) for path in source_paths}
    write_json_new(HERE / "protocol.json", protocol_value)
    write_json_new(HERE / "CPU_SELFCHECK.json", checks)
    write_json_new(HERE / "design_freeze.json", {
        "schema_version": SCHEMA_VERSION,
        "script_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "protocol.json"),
        "input_sha256": input_hashes,
        "label_blind_preparation": True,
        "trained": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    for part in PARTITIONS:
        write_jsonl_new(HERE / f"microclaims_{part}.jsonl", all_by_partition[part])
    preparation = {
        "status": "complete_label_blind_atomic_relation_preparation",
        "schema_version": SCHEMA_VERSION,
        "partitions": summaries,
        "totals": {
            "answers": sum(value["answers"] for value in summaries.values()),
            "incumbent_parent_claims": sum(value["incumbent_parent_claims"] for value in summaries.values()),
            "atomic_microclaims": sum(value["atomic_microclaims"] for value in summaries.values()),
            "added_microclaims": sum(value["added_microclaims"] for value in summaries.values()),
        },
        "artifacts_sha256": {
            "protocol.json": sha(HERE / "protocol.json"),
            "CPU_SELFCHECK.json": sha(HERE / "CPU_SELFCHECK.json"),
            "design_freeze.json": sha(HERE / "design_freeze.json"),
            **{f"microclaims_{part}.jsonl": sha(HERE / f"microclaims_{part}.jsonl") for part in PARTITIONS},
        },
        "gold_fields_used": False,
        "model_loaded": False,
        "trained": False,
        "thresholds_selected": 0,
        "GPU_used": False,
        "official_test_opened": False,
    }
    write_json_new(HERE / "preparation.json", preparation)
    print("ATOMIC_RELATION_PREPARATION_COMPLETE", preparation["totals"], flush=True)


def verify_preparation():
    prep = read_json(HERE / "preparation.json")
    assert prep["status"] == "complete_label_blind_atomic_relation_preparation"
    for name, expected in prep["artifacts_sha256"].items():
        assert sha(HERE / name) == expected, name
    freeze = read_json(HERE / "design_freeze.json")
    assert freeze["script_sha256"] == sha(Path(__file__))
    assert freeze["protocol_sha256"] == sha(HERE / "protocol.json")
    assert read_json(HERE / "protocol.json") == protocol()
    for raw_path, expected in freeze["input_sha256"].items():
        assert sha(Path(raw_path)) == expected, raw_path
    assert self_test()["passed"]
    return prep


def feature_flags(rows):
    return {name: any(row["feature_vector"][name] for row in rows) for name in FEATURE_NAMES}


def alnum_intersection(text: str, a: int, b: int, c: int, d: int) -> int:
    lo, hi = max(a, c), min(b, d)
    return sum(text[index].isalnum() for index in range(lo, max(lo, hi)))


def map_gold_spans(part: str, claims_by_response, answers_by_response):
    records = []
    for rid, answer in answers_by_response.items():
        text = answer["original_response"]
        claims = claims_by_response[rid]
        for span_index, label in enumerate(answer["original_labels"]):
            start, end = int(label["start"]), int(label["end"])
            assert text[start:end] == label["text"]
            gold_chars = sum(ch.isalnum() for ch in text[start:end])
            overlaps = []
            for claim in claims:
                overlap = alnum_intersection(text, start, end, claim["start"], claim["end"])
                if overlap:
                    claim_chars = sum(ch.isalnum() for ch in claim["text"])
                    overlaps.append((claim, overlap, claim_chars))
            covered = sum(value for _, value, _ in overlaps)
            if gold_chars:
                assert covered == gold_chars
            flags = feature_flags([row for row, _, _ in overlaps]) if overlaps else {name: False for name in FEATURE_NAMES}
            direct = surface_features(label["text"], start)
            direct_flags = {
                "has_negation": bool(direct["negation"]),
                "has_comparator": bool(direct["comparator"]),
                "has_number_or_unit": bool(direct["quantities"]),
                "has_temporal_order": bool(direct["temporal_order"]),
                "has_condition": bool(direct["condition"]),
                "has_explicit_passage_id": bool(direct["citations"]["passage_ids"]),
                "has_local_subject": direct["local_subject"] is not None,
                "has_entity_candidate": bool(direct["entity_candidates"]),
            }
            direct_flags["has_any_core_relation"] = any(direct_flags[name] for name in CORE_FEATURES)
            direct_flags["has_any_relation_or_source"] = direct_flags["has_any_core_relation"] or direct_flags["has_explicit_passage_id"]
            best_gold_coverage = max((value / gold_chars for _, value, _ in overlaps), default=0.0) if gold_chars else 1.0
            best_claim_purity = max((value / claim_chars for _, value, claim_chars in overlaps if claim_chars), default=0.0)
            records.append({
                "partition": part,
                "response_id": rid,
                "span_index": span_index,
                "label_type": label["label_type"],
                "start": start,
                "end": end,
                "text": label["text"],
                "gold_alphanumeric_characters": gold_chars,
                "covered_alphanumeric_characters": covered,
                "microclaims_touched": len(overlaps),
                "microclaim_ids": [row["microclaim_id"] for row, _, _ in overlaps],
                "best_microclaim_gold_coverage": best_gold_coverage,
                "best_microclaim_purity": best_claim_purity,
                "single_microclaim_full_gold_coverage": best_gold_coverage == 1.0,
                "context_feature_flags": flags,
                "direct_span_feature_flags": direct_flags,
                "explicit_passage_ids_in_context": sorted({value for row, _, _ in overlaps for value in row["explicit_passage_ids"]}),
                "parent_passage_ids_in_context": sorted({value for row, _, _ in overlaps for value in row["parent_passage_ids"]}),
            })
    return records


def summarize_gold(records):
    result = {
        "spans": len(records),
        "mapped_spans": sum(row["covered_alphanumeric_characters"] == row["gold_alphanumeric_characters"] for row in records),
        "single_microclaim_full_gold_coverage": sum(row["single_microclaim_full_gold_coverage"] for row in records),
        "single_microclaim_full_gold_coverage_rate": ratio(sum(row["single_microclaim_full_gold_coverage"] for row in records), len(records)),
        "mean_best_microclaim_gold_coverage": mean(row["best_microclaim_gold_coverage"] for row in records),
        "mean_best_microclaim_purity": mean(row["best_microclaim_purity"] for row in records),
        "mean_microclaims_touched": mean(row["microclaims_touched"] for row in records),
    }
    context_names = FEATURE_NAMES
    direct_names = tuple(CORE_FEATURES) + ("has_explicit_passage_id", "has_local_subject", "has_entity_candidate", "has_any_core_relation", "has_any_relation_or_source")
    result["context_feature_coverage"] = {
        name: {"count": sum(row["context_feature_flags"][name] for row in records),
               "fraction": ratio(sum(row["context_feature_flags"][name] for row in records), len(records))}
        for name in context_names
    }
    result["direct_span_feature_coverage"] = {
        name: {"count": sum(row["direct_span_feature_flags"][name] for row in records),
               "fraction": ratio(sum(row["direct_span_feature_flags"][name] for row in records), len(records))}
        for name in direct_names
    }
    return result


def assign_tokens_to_claims(token_row, claims):
    text = token_row["original_response"]
    assignment = [-1] * token_row["token_count"]
    for token_index, (a, b) in enumerate(token_row["response_token_offsets"]):
        chars = [index for index in range(a, b) if text[index].isalnum()]
        if not chars:
            continue
        overlaps = [sum(claim["start"] <= index < claim["end"] for index in chars) for claim in claims]
        chosen = max(range(len(claims)), key=lambda index: (overlaps[index], -index))
        assert overlaps[chosen] > 0
        assignment[token_index] = chosen
    return assignment


def token_gold_types(token_row):
    types = [set() for _ in range(token_row["token_count"])]
    assert len(token_row["span_token_mapping"]) == len(token_row["original_labels"])
    for mapping, label in zip(token_row["span_token_mapping"], token_row["original_labels"]):
        assert int(mapping["span_index"]) >= 0
        for token_index in mapping["risk_token_indices"]:
            types[token_index].add(label["label_type"])
    return types


def window_prevalence(part: str, claims_by_response, token_rows):
    assignments = {}
    gold_types = {}
    for rid, token in token_rows.items():
        assignments[rid] = assign_tokens_to_claims(token, claims_by_response[rid])
        gold_types[rid] = token_gold_types(token)
    all_counts = Counter()
    ec_counts = Counter()
    windows = 0
    ec_windows = 0
    assigned = 0
    ec_assigned = 0
    cross_claim = 0
    ec_cross_claim = 0
    for window in read_jsonl(WINDOWS[part]):
        windows += 1
        rid = window["response_id"]
        claim_ids = sorted({assignments[rid][index] for index in window["token_indices"] if assignments[rid][index] >= 0})
        rows = [claims_by_response[rid][index] for index in claim_ids]
        flags = feature_flags(rows) if rows else {name: False for name in FEATURE_NAMES}
        assigned += int(bool(rows))
        cross_claim += int(len(rows) > 1)
        for name, value in flags.items():
            all_counts[name] += int(value)
        types = set().union(*(gold_types[rid][index] for index in window["token_indices"]))
        if "Evident Conflict" in types:
            ec_windows += 1
            ec_assigned += int(bool(rows))
            ec_cross_claim += int(len(rows) > 1)
            for name, value in flags.items():
                ec_counts[name] += int(value)
    assert windows == EXPECTED_WINDOWS[part]
    return {
        "windows": windows,
        "assigned_to_at_least_one_microclaim": assigned,
        "assignment_rate": ratio(assigned, windows),
        "cross_microclaim_windows": cross_claim,
        "cross_microclaim_fraction": ratio(cross_claim, windows),
        "feature_prevalence": {name: {"count": all_counts[name], "fraction": ratio(all_counts[name], windows)} for name in FEATURE_NAMES},
        "evident_conflict_windows": ec_windows,
        "evident_conflict_assigned": ec_assigned,
        "evident_conflict_assignment_rate": ratio(ec_assigned, ec_windows),
        "evident_conflict_cross_microclaim_windows": ec_cross_claim,
        "evident_conflict_cross_microclaim_fraction": ratio(ec_cross_claim, ec_windows),
        "evident_conflict_feature_coverage": {name: {"count": ec_counts[name], "fraction": ratio(ec_counts[name], ec_windows)} for name in FEATURE_NAMES},
    }


def audit():
    verify_preparation()
    outputs = [HERE / "AUDIT.json", HERE / "evident_conflict_records.jsonl", HERE / "REPORT.md", HERE / "complete.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Audit artifacts already exist; preserving them. Use verify instead.")
    claims_by_part = {}
    claims_by_response = {}
    answers_by_part = {}
    token_rows_by_part = {}
    for part in PARTITIONS:
        records = list(read_jsonl(HERE / f"microclaims_{part}.jsonl"))
        by_response = defaultdict(list)
        for row in records:
            assert row["partition"] == part
            by_response[row["response_id"]].append(row)
        for rows in by_response.values():
            rows.sort(key=lambda row: row["microclaim_index"])
            assert [row["microclaim_index"] for row in rows] == list(range(len(rows)))
        claims_by_part[part] = records
        claims_by_response[part] = dict(by_response)
        answers = {row["response_id"]: row for row in read_jsonl(ANSWERS[part])}
        tokens = {row["response_id"]: row for row in read_jsonl(TOKENS[part])}
        assert set(answers) == set(tokens) == set(by_response)
        answers_by_part[part] = answers
        token_rows_by_part[part] = tokens

    all_gold = []
    ec_by_part = {}
    all_types_by_part = {}
    window_results = {}
    for part in PARTITIONS:
        mapped = map_gold_spans(part, claims_by_response[part], answers_by_part[part])
        all_gold.extend(mapped)
        types = defaultdict(list)
        for row in mapped:
            types[row["label_type"]].append(row)
        all_types_by_part[part] = {name: summarize_gold(rows) for name, rows in sorted(types.items())}
        ec_rows = types["Evident Conflict"]
        assert len(ec_rows) == (109 if part == "fit" else 43)
        ec_by_part[part] = summarize_gold(ec_rows)
        window_results[part] = window_prevalence(part, claims_by_response[part], token_rows_by_part[part])
    assert window_results["calibration"]["evident_conflict_windows"] == 997

    ec_records = [row for row in all_gold if row["label_type"] == "Evident Conflict"]
    current_audit = read_json(CURRENT_ERROR_AUDIT)
    assert current_audit["candidate"] == "semantic_claim__old_tree__large_weight0.4"
    assert current_audit["human_spans"]["by_label_type"]["Evident Conflict"]["spans"] == 43
    incumbent_spans = {
        (row["response_id"], int(row["span_index"])): row
        for row in read_jsonl(CURRENT_SPANS) if row["label_type"] == "Evident Conflict"
    }
    cal_ec = [row for row in ec_records if row["partition"] == "calibration"]
    assert len(incumbent_spans) == len(cal_ec) == 43
    for row in cal_ec:
        incumbent = incumbent_spans[(row["response_id"], row["span_index"])]
        row["incumbent_any_window_detected"] = incumbent["any_window_detected"]
        row["incumbent_fully_missed"] = incumbent["fully_missed"]
        row["incumbent_window_recall"] = incumbent["window_recall"]
    missed = [row for row in cal_ec if row["incumbent_fully_missed"]]
    assert len(missed) == 30

    preparation = read_json(HERE / "preparation.json")
    result = {
        "status": "complete_cpu_only_atomic_relation_coverage_audit",
        "schema_version": SCHEMA_VERSION,
        "preparation": preparation["totals"],
        "atomic_geometry": {
            part: {key: value for key, value in preparation["partitions"][part].items() if key != "feature_prevalence"}
            for part in PARTITIONS
        },
        "candidate_feature_prevalence_microclaims": {
            part: preparation["partitions"][part]["feature_prevalence"] for part in PARTITIONS
        },
        "candidate_feature_prevalence_windows": window_results,
        "gold_span_coverage_by_type": all_types_by_part,
        "gold_evident_conflict_span_coverage": {
            **ec_by_part,
            "combined": summarize_gold(ec_records),
        },
        "current_incumbent_calibration_evident_conflict": {
            "candidate": current_audit["candidate"],
            "risk_windows_detected_total": current_audit["target_arithmetic"]["evident_conflict_current_hit_total"],
            "span_any_hit": 43 - len(missed),
            "span_fully_missed": len(missed),
            "all_43_relation_schema_coverage": summarize_gold(cal_ec),
            "fully_missed_30_relation_schema_coverage": summarize_gold(missed),
            "interpretation": "Feature activation inside an incumbent miss is diagnostic candidate coverage, not a rescued prediction.",
        },
        "design_integrity": {
            "prepare_was_label_blind": preparation["gold_fields_used"] is False,
            "gold_used_only_in_audit": True,
            "new_fits": 0,
            "new_thresholds": 0,
            "baseline_files_modified": False,
            "GPU_used": False,
            "official_test_opened": False,
        },
        "limits": protocol()["limits"],
        "source_sha256": {
            str(path.resolve()): sha(path)
            for path in (Path(__file__), HERE / "preparation.json", CURRENT_ERROR_AUDIT, CURRENT_SPANS,
                         *ANSWERS.values(), *TOKENS.values(), *WINDOWS.values())
        },
    }
    write_json_new(HERE / "AUDIT.json", result)
    write_jsonl_new(HERE / "evident_conflict_records.jsonl", ec_records)

    def pct(value):
        return "N/A" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# RAGTruth QA 原子微主张与关系特征审计", "",
        "仅处理原始 fit634 + calibration159。未训练、未调阈值、未改 baseline、未用 GPU、未打开官方 test。", "",
        "## 原子切分", "",
        "| 分区 | 旧句级主张 | 原子微主张 | 新增 | 被拆旧主张 | P95词数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for part in PARTITIONS:
        row = preparation["partitions"][part]
        lines.append(f"| {part} | {row['incumbent_parent_claims']} | {row['atomic_microclaims']} | {row['added_microclaims']} | {row['split_parent_claims']} ({pct(row['split_parent_fraction'])}) | {row['p95_words']:.1f} |")
    lines += ["", "切分后仍覆盖每个答案的全部字母数字字符；按同一规则复查，残余可切边界为 0。", "",
              "## 候选特征在微主张中的出现率", "",
              "| 特征 | fit | calibration |", "|---|---:|---:|"]
    display = {
        "has_negation": "否定", "has_comparator": "比较", "has_number_or_unit": "数字/单位",
        "has_temporal_order": "时序", "has_condition": "条件", "has_explicit_passage_id": "显式 Passage ID",
        "has_parent_passage_context": "父主张 Passage 上下文", "has_effective_subject": "可用主体",
        "has_entity_candidate": "实体候选", "has_any_core_relation": "任一核心关系特征",
        "has_any_relation_or_source": "任一关系或来源特征",
    }
    for name, label in display.items():
        fit_value = preparation["partitions"]["fit"]["feature_prevalence"][name]
        cal_value = preparation["partitions"]["calibration"]["feature_prevalence"][name]
        lines.append(f"| {label} | {fit_value['count']}/{preparation['partitions']['fit']['atomic_microclaims']} ({pct(fit_value['fraction'])}) | {cal_value['count']}/{preparation['partitions']['calibration']['atomic_microclaims']} ({pct(cal_value['fraction'])}) |")
    lines += ["", "## Gold Evident Conflict 覆盖", "",
              "| 分区 | span | 全字符映射 | 单一微主张完整容纳 | 任一核心关系特征 | 任一关系或来源 | 可用主体 | 实体候选 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for part in (*PARTITIONS, "combined"):
        row = result["gold_evident_conflict_span_coverage"][part]
        context = row["context_feature_coverage"]
        lines.append(
            f"| {part} | {row['spans']} | {row['mapped_spans']}/{row['spans']} | "
            f"{row['single_microclaim_full_gold_coverage']}/{row['spans']} ({pct(row['single_microclaim_full_gold_coverage_rate'])}) | "
            f"{context['has_any_core_relation']['count']}/{row['spans']} ({pct(context['has_any_core_relation']['fraction'])}) | "
            f"{context['has_any_relation_or_source']['count']}/{row['spans']} ({pct(context['has_any_relation_or_source']['fraction'])}) | "
            f"{context['has_effective_subject']['count']}/{row['spans']} ({pct(context['has_effective_subject']['fraction'])}) | "
            f"{context['has_entity_candidate']['count']}/{row['spans']} ({pct(context['has_entity_candidate']['fraction'])}) |"
        )
    missed_summary = result["current_incumbent_calibration_evident_conflict"]["fully_missed_30_relation_schema_coverage"]
    missed_context = missed_summary["context_feature_coverage"]
    lines += [
        "", "## 对当前 R32 漏检的含义", "",
        f"当前候选 calibration 的 Evident Conflict 风险窗仅检出 195/997；43 个 EC span 中 30 个完全漏检。",
        f"这 30 个漏检 span 所在微主张中，{missed_context['has_any_core_relation']['count']}/30 含否定、比较、数字/单位、时序或条件；"
        f"{missed_context['has_any_relation_or_source']['count']}/30 含关系或 Passage 来源特征；"
        f"{missed_context['has_effective_subject']['count']}/30 有可用主体。",
        "这些数字说明静态关系槽覆盖了多少漏检案例，不代表已经修复或能达到某个 F1。", "",
        "## 边界", "",
        "该脚本是确定性候选生成器，不是句法分析器。主体、实体和指代均为表面候选；特征出现不等于冲突。后续若训练关系专头，只能用 fit 选模型与阈值，cal 仅作开发检查。",
    ]
    report_text = "\n".join(lines) + "\n"
    if (HERE / "REPORT.md").exists():
        raise FileExistsError(HERE / "REPORT.md")
    (HERE / "REPORT.md").write_text(report_text, encoding="utf-8", newline="\n")
    complete = {
        "status": "complete",
        "artifacts_sha256": {name: sha(HERE / name) for name in (
            "protocol.json", "CPU_SELFCHECK.json", "design_freeze.json", "preparation.json",
            "microclaims_fit.jsonl", "microclaims_calibration.jsonl", "AUDIT.json",
            "evident_conflict_records.jsonl", "REPORT.md",
        )},
        "new_fits": 0,
        "new_thresholds": 0,
        "baseline_files_modified": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    write_json_new(HERE / "complete.json", complete)
    print("ATOMIC_RELATION_AUDIT_COMPLETE", len(ec_records), window_results["calibration"]["evident_conflict_windows"], flush=True)


def verify():
    verify_preparation()
    complete_path = HERE / "complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        assert complete["status"] == "complete"
        for name, expected in complete["artifacts_sha256"].items():
            assert sha(HERE / name) == expected, name
        audit_value = read_json(HERE / "AUDIT.json")
        assert audit_value["design_integrity"]["official_test_opened"] is False
        assert audit_value["design_integrity"]["new_fits"] == 0
        print("ATOMIC_RELATION_ALL_ARTIFACTS_VERIFIED", flush=True)
    else:
        print("ATOMIC_RELATION_PREPARATION_VERIFIED", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "prepare", "audit", "verify"))
    args = parser.parse_args(argv)
    if args.action == "self-test":
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
    elif args.action == "prepare":
        prepare()
    elif args.action == "audit":
        audit()
    else:
        verify()


if __name__ == "__main__":
    main()
