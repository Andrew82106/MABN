"""Mine fit-only in-domain contrastive microclaim pairs for a v5 rank loss.

Every pair shares the exact frozen source/question/material and source-connected
fold, but has opposite existing gold projections.  Pair construction is purely
structural: it never creates or changes a factual label and never reads the
calibration or official-test partitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "results/atomic_microclaim_relation_expanded_v4"
EXAMPLES = V4 / "examples.jsonl"
ANSWERS = V4 / "answers.jsonl"
V4_MANIFEST = V4 / "manifest.json"
V4_COMPLETE = V4 / "complete.json"
OUT = ROOT / "results/microclaim_indomain_pairs_v5"
LOSS_CODE = HERE / "microclaim_pairwise_rank_loss_v5.py"

FIT_EXAMPLES = 34_919
FIT_ANSWERS = 3_680
FIT_SOURCES = 634
FIT_GROUPS = 615
FOLDS = 5
MAX_PAIRS_PER_POSITIVE = 2
MAX_USES_PER_NEGATIVE = 3
MIN_SURFACE = 0.52
MIN_OBJECT_SURFACE = 0.72
PAIR_TOKEN_BUDGET = 1536
PAIR_MAX_EXAMPLES = 4
REFERENCE_TOKENS_PER_SECOND = 8920.754982150334

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*")
LEADING_ENUM_RE = re.compile(r"^\s*(?:[-*•]+|\(?\d{1,3}\s*[.)])\s*")
LEADING_STEP_RE = re.compile(r"^\s*(?:step|method|option)\s+\d{1,3}\s*[:.)-]\s*", re.I)
CITATION_RE = re.compile(
    r"\(?\b(?:passage|source|document|article|reference)s?\s*#?\s*\d+"
    r"(?:\s*[:.,-]\s*\d+)?\)?",
    re.I,
)
SOURCE_RE = re.compile(
    r"\b(passages?|sources?|documents?|articles?|references?)\s*#?\s*(\d+)",
    re.I,
)
NUMBER_RE = re.compile(r"(?<!\w)(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*\d+)?)(?!\w)")

STOP = set(
    "a an the and or but if then than that this these those it its they them their he she his her "
    "we us our you your i me my of in on at to from for with by as into over under about based "
    "provided given according passage passages source sources evidence information also however "
    "therefore thus additionally moreover here there some any all each other more most very such "
    "please note important approximately roughly around likely possibly perhaps generally typically"
    .split()
)
DISCOURSE = set(
    "step option method answer following follows include includes including firstly secondly thirdly "
    "finally next then now again simply specifically essentially"
    .split()
)
NEGATION = {
    "no", "not", "never", "neither", "nor", "without", "cannot", "can't", "unable",
    "unlikely", "hardly", "rarely", "isn't", "aren't", "wasn't", "weren't", "doesn't",
    "don't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't",
}
TEMPORAL_WORDS = set(
    "before after during while when until since previously currently later earlier recent recently "
    "today yesterday tomorrow annually daily weekly monthly yearly initially eventually immediately "
    "morning afternoon evening night overnight spring summer autumn fall winter january february "
    "march april may june july august september october november december monday tuesday wednesday "
    "thursday friday saturday sunday century centuries decade decades"
    .split()
)
TIME_UNITS = set(
    "second seconds minute minutes hour hours day days week weeks month months year years decade "
    "decades century centuries"
    .split()
)
NUMBER_WORDS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0,
    "nineteen": 19.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0,
    "fifty": 50.0, "sixty": 60.0, "seventy": 70.0, "eighty": 80.0,
    "ninety": 90.0, "hundred": 100.0, "thousand": 1000.0, "million": 1e6,
    "billion": 1e9, "half": 0.5, "quarter": 0.25,
}
ORDINAL_WORDS = {
    "first": 1.0, "second": 2.0, "third": 3.0, "fourth": 4.0, "fifth": 5.0,
    "sixth": 6.0, "seventh": 7.0, "eighth": 8.0, "ninth": 9.0, "tenth": 10.0,
}
VERB_ROOTS = set(
    "be have do say state mention indicate show suggest report find include contain consist become "
    "make use provide require allow cause lead mean refer occur happen begin start end remain seem "
    "appear know believe think claim describe explain support contradict confirm deny increase "
    "decrease reduce improve produce create develop establish identify locate live die bear born "
    "win lose cost weigh measure take give get go come work serve play write publish release open "
    "close call name own hold reach involve affect result cook heat add remove cut place mix stir "
    "pour bake boil simmer drain wash clean click select choose press enter install connect apply "
    "keep grow form react burn turn move run build fill cover set put let leave bring carry send "
    "receive return spend last need want recommend follow prevent help enable represent comprise "
    "offer sell buy learn teach become remain feel look sound taste smell produce perform rest cool "
    "tax grow check ensure expose divide award soak grill preheat weigh transfer experience"
    .split()
)
BE_FORMS = set(
    "am is are was were be been being has have had do does did can could may might must should "
    "would will shall"
    .split()
)
BE_NORMALIZE = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "been": "be", "being": "be", "has": "have", "had": "have",
    "does": "do", "did": "do",
}
EQUIVALENCE = {
    "approximate": "about", "approximately": "about", "roughly": "about", "around": "about",
    "given": "provide", "provided": "provide", "stated": "state", "states": "state",
    "mentioned": "mention", "mentions": "mention", "shown": "show", "shows": "show",
    "indicated": "indicate", "indicates": "indicate", "grey": "gray",
    "individual": "unique", "another": "additional", "skillet": "pan",
    "periodically": "regularly", "every": "regularly", "pulse": "process",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_json(path: Path, value):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def save_jsonl(path: Path, rows):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def read_fit_prefix(path: Path, expected: int):
    """Read exactly the leading fit block; calibration lines are never parsed."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for _ in range(expected):
            line = handle.readline()
            assert line, (path, len(rows), expected)
            row = json.loads(line)
            assert row["partition"] == "fit"
            rows.append(row)
    return rows


def raw_tokens(text: str):
    text = LEADING_STEP_RE.sub("", LEADING_ENUM_RE.sub("", text))
    return WORD_RE.findall(text)


def stem(token: str):
    token = token.lower().replace("’", "'")
    if token.endswith("'s"):
        token = token[:-2]
    token = EQUIVALENCE.get(token, token)
    if token in BE_NORMALIZE:
        return BE_NORMALIZE[token]
    if token in BE_FORMS or token in NEGATION:
        return token
    irregular = {
        "giving": "give", "gave": "give", "given": "give", "making": "make",
        "made": "make", "having": "have", "taking": "take", "took": "take",
        "taken": "take", "bearing": "bear", "bears": "bear", "bore": "bear",
        "brought": "bring", "went": "go", "gone": "go", "came": "come",
        "written": "write", "wrote": "write", "shown": "show", "found": "find",
        "led": "lead", "held": "hold", "felt": "feel", "left": "leave",
    }
    if token in irregular:
        return irregular[token]
    for suffix, replacement in (
        ("ization", "ize"), ("isation", "ise"), ("ational", "ate"),
        ("ments", ""), ("ment", ""), ("ingly", ""), ("edly", ""),
        ("ies", "y"), ("ing", ""), ("ied", "y"), ("ed", ""),
        ("es", ""), ("s", ""),
    ):
        if token.endswith(suffix) and len(token) >= len(suffix) + 4:
            token = token[:-len(suffix)] + replacement
            break
    return EQUIVALENCE.get(token, token)


def lexical_tokens(text: str, strip_citations: bool = True):
    if strip_citations:
        text = CITATION_RE.sub(" ", text)
    return [stem(token) for token in raw_tokens(text)]


def content_tokens(tokens):
    return [token for token in tokens
            if token not in STOP and token not in DISCOURSE and len(token) > 1]


def is_verb(raw: str, lemma: str):
    lower = raw.lower().replace("’", "'")
    return (
        lemma in VERB_ROOTS or lower in BE_FORMS or
        (len(lower) >= 5 and lower.endswith(("ing", "ed", "ize", "ise", "ify", "ate")))
    )


def anchors(text: str):
    raw = raw_tokens(CITATION_RE.sub(" ", text))
    lemmas = [stem(token) for token in raw]
    keep = [(r, l) for r, l in zip(raw, lemmas)
            if l not in STOP and l not in DISCOURSE and len(l) > 1]
    verb_index = next((index for index, (r, l) in enumerate(keep) if is_verb(r, l)), None)
    if verb_index is None:
        return set(lemma for _, lemma in keep[:3]), set(), "fragment"
    subject = {"<implicit-you>"} if verb_index == 0 else {
        lemma for _, lemma in keep[max(0, verb_index - 3):verb_index]
    }
    predicate = {lemma for raw_token, lemma in keep[verb_index:verb_index + 3]
                 if is_verb(raw_token, lemma) or raw_token.lower() in {
                     "in", "on", "at", "to", "from", "with", "for", "of", "into", "by"
                 }}
    return subject, predicate, "clause"


def set_dice(left, right):
    left, right = set(left), set(right)
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


def surface_features(left: str, right: str):
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)
    left_content = content_tokens(left_tokens)
    right_content = content_tokens(right_tokens)
    sequence = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()
    token_dice = set_dice(left_tokens, right_tokens)
    content_dice = set_dice(left_content, right_content)
    char = SequenceMatcher(
        None, " ".join(left_tokens), " ".join(right_tokens), autojunk=False
    ).ratio()
    score = 0.35 * sequence + 0.25 * token_dice + 0.25 * content_dice + 0.15 * char
    return {
        "score": score, "sequence": sequence, "token_dice": token_dice,
        "content_dice": content_dice, "char": char,
        "left_tokens": left_tokens, "right_tokens": right_tokens,
        "left_content": left_content, "right_content": right_content,
    }


def canonical_number(raw: str):
    value = raw.replace(",", "").replace(" ", "")
    if "/" in value:
        left, right = value.split("/", 1)
        if left.replace(".", "", 1).isdigit() and right.replace(".", "", 1).isdigit():
            denominator = float(right)
            return None if denominator == 0 else round(float(left) / denominator, 8)
    try:
        return float(value)
    except ValueError:
        return None


def number_signature(text: str):
    text = LEADING_STEP_RE.sub("", LEADING_ENUM_RE.sub("", text))
    text = CITATION_RE.sub(" ", text)
    values = []
    for raw in NUMBER_RE.findall(text):
        value = canonical_number(raw)
        if value is not None:
            values.append(value)
    for token in raw_tokens(text):
        lower = token.lower()
        if lower in NUMBER_WORDS:
            values.append(NUMBER_WORDS[lower])
        elif lower in ORDINAL_WORDS:
            values.append(ORDINAL_WORDS[lower])
    return tuple(sorted(set(values)))


def quantity_signature(text: str):
    """Map comparable local units to canonical numeric values.

    This deliberately ignores list numbers, passage citations, model/category
    identifiers such as ``Type 2``, and verse-like ``14:6`` references.
    """
    text = LEADING_STEP_RE.sub("", LEADING_ENUM_RE.sub("", text))
    text = CITATION_RE.sub(" ", text)
    raw = [token.lower() for token in raw_tokens(text)]
    result = defaultdict(set)
    for index, token in enumerate(raw):
        previous = raw[index - 1] if index else ""
        if previous in {"type", "model", "version", "chapter", "john", "step"}:
            continue
        if ":" in token:
            continue
        values = []
        if token in NUMBER_WORDS:
            values = [NUMBER_WORDS[token]]
        elif token in ORDINAL_WORDS:
            values = [ORDINAL_WORDS[token]]
        elif re.fullmatch(r"\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", token):
            value = canonical_number(token)
            if value is not None:
                values = [value]
        elif re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", token):
            values = [float(part) for part in token.split("-")]
        if not values:
            continue
        unit = None
        for distance in (1, 2, 3):
            if index + distance < len(raw):
                candidate = stem(raw[index + distance])
                if (candidate not in STOP and candidate not in DISCOURSE and
                        not candidate.replace(".", "", 1).isdigit() and
                        candidate not in {"to", "and", "or", "per", "about"}):
                    unit = candidate.rstrip("s")
                    break
        if unit is None:
            for distance in (1, 2):
                if index >= distance:
                    candidate = stem(raw[index - distance])
                    if candidate not in STOP and candidate not in DISCOURSE:
                        unit = candidate.rstrip("s")
                        break
        if unit:
            result[unit].update(values)
    return {key: tuple(sorted(values)) for key, values in sorted(result.items())}


def source_signature(text: str):
    return tuple(sorted({(kind.lower().rstrip("s"), int(number))
                         for kind, number in SOURCE_RE.findall(text)}))


def negation_signature(text: str):
    lowered = text.lower().replace("’", "'")
    return tuple(sorted({token for token in NEGATION if re.search(
        rf"(?<!\w){re.escape(token)}(?!\w)", lowered
    )}))


def predicate_negation(text: str):
    lowered = text.lower().replace("’", "'")
    if "whether or not" in lowered or "not only" in lowered:
        return {}
    raw = raw_tokens(CITATION_RE.sub(" ", lowered))
    lemmas = [stem(token) for token in raw]
    predicates = {lemma for token, lemma in zip(raw, lemmas) if is_verb(token, lemma)}
    negated = set()
    for index, lemma in enumerate(lemmas):
        if lemma not in NEGATION:
            continue
        for distance in (1, 2, 3):
            if index + distance < len(raw) and is_verb(raw[index + distance], lemmas[index + distance]):
                negated.add(lemmas[index + distance])
                break
        else:
            for distance in (1, 2):
                if index >= distance and is_verb(raw[index - distance], lemmas[index - distance]):
                    negated.add(lemmas[index - distance])
                    break
    return {predicate: predicate in negated for predicate in predicates}


def temporal_signature(text: str):
    raw = [token.lower() for token in raw_tokens(CITATION_RE.sub(" ", text))]
    values = {token for token in raw if token in TEMPORAL_WORDS}
    for index, token in enumerate(raw):
        if token in TIME_UNITS:
            previous = raw[index - 1] if index else ""
            canonical = canonical_number(previous)
            if previous in NUMBER_WORDS:
                canonical = NUMBER_WORDS[previous]
            values.add(f"{canonical if canonical is not None else '*'}:{token.rstrip('s')}")
        elif re.fullmatch(r"(?:1[5-9]|20)\d{2}", token):
            values.add(f"year:{token}")
    return tuple(sorted(values))


def temporal_contrast(left: str, right: str, left_quantities, right_quantities):
    common_time_units = (set(left_quantities) & set(right_quantities) &
                         {stem(unit).rstrip("s") for unit in TIME_UNITS})
    if any(left_quantities[unit] != right_quantities[unit] for unit in common_time_units):
        return True
    a, b = set(temporal_signature(left)), set(temporal_signature(right))
    relative = {"before", "after", "previously", "currently", "later", "earlier",
                "today", "yesterday", "tomorrow", "initially", "eventually"}
    calendar = {stem(value) for value in (
        "spring summer autumn fall winter january february march april may june july august "
        "september october november december monday tuesday wednesday thursday friday saturday sunday"
    ).split()}
    a_relative, b_relative = a & relative, b & relative
    a_calendar, b_calendar = a & calendar, b & calendar
    years_a = {value for value in a if value.startswith("year:")}
    years_b = {value for value in b if value.startswith("year:")}
    opposite_pairs = {
        frozenset(("before", "after")), frozenset(("previously", "currently")),
        frozenset(("later", "earlier")), frozenset(("today", "yesterday")),
        frozenset(("today", "tomorrow")), frozenset(("initially", "eventually")),
    }
    relative_opposition = any(
        frozenset((left_value, right_value)) in opposite_pairs
        for left_value in a_relative for right_value in b_relative
    )
    return bool(
        relative_opposition or
        (a_calendar and b_calendar and a_calendar != b_calendar) or
        (years_a and years_b and years_a != years_b)
    )


def differing_content(left_tokens, right_tokens):
    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
    left_diff, right_diff, replacements = [], [], 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 > i1 and j2 > j1:
            replacements += 1
        left_diff.extend(left_tokens[i1:i2])
        right_diff.extend(right_tokens[j1:j2])
    ignored = STOP | DISCOURSE | NEGATION | TEMPORAL_WORDS | TIME_UNITS
    left_diff = [token for token in left_diff if token not in ignored and len(token) > 1]
    right_diff = [token for token in right_diff if token not in ignored and len(token) > 1]
    return left_diff, right_diff, replacements


def strict_object_contrast(left_tokens, right_tokens, left_diff, right_diff):
    ignored = set(NUMBER_WORDS) | set(ORDINAL_WORDS) | NEGATION | TEMPORAL_WORDS | TIME_UNITS
    left_diff = [token for token in left_diff if token not in ignored]
    right_diff = [token for token in right_diff if token not in ignored]
    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
    changes = [(tag, left_tokens[i1:i2], right_tokens[j1:j2])
               for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]
    replacements = [row for row in changes if row[0] == "replace" and row[1] and row[2]]
    other_content_changes = [row for row in changes if row[0] != "replace" and
                             (any(x not in STOP | DISCOURSE for x in row[1]) or
                              any(x not in STOP | DISCOURSE for x in row[2]))]
    if len(replacements) != 1 or other_content_changes:
        return False
    if not left_diff or not right_diff or len(left_diff) > 3 or len(right_diff) > 3:
        return False
    if any(token in VERB_ROOTS or token in BE_FORMS for token in left_diff + right_diff):
        return False
    if len(left_diff) == len(right_diff) == 1 and SequenceMatcher(
            None, left_diff[0], right_diff[0], autojunk=False).ratio() >= 0.80:
        return False
    _, i1, i2, j1, j2 = next(row for row in matcher.get_opcodes() if row[0] == "replace")
    shared_before = len(set(left_tokens[:i1]) & set(right_tokens[:j1]))
    shared_after = len(set(left_tokens[i2:]) & set(right_tokens[j2:]))
    return shared_before + shared_after >= 2 and bool(shared_before and shared_after)


def risk_profile(example, answer):
    offsets = answer["response_token_offsets"]
    text = answer["original_response"]
    pieces = [
        text[max(example["char_start"], span["start"]):min(example["char_end"], span["end"])]
        for span in example["overlapping_gold_spans"]
        if max(example["char_start"], span["start"]) < min(example["char_end"], span["end"])
    ]
    if not pieces:
        for token_index in example["risk_bpe_indices"]:
            left, right = offsets[token_index]
            pieces.append(text[left:right])
    text = " ".join(pieces)
    return {
        "text": text,
        "content": set(content_tokens(lexical_tokens(text, strip_citations=False))),
        "lexical": set(lexical_tokens(text, strip_citations=False)),
        "numbers": set(number_signature(text)),
        "sources": set(source_signature(text)),
        "temporal": set(temporal_signature(text)),
    }


def slot_analysis(positive, negative, sf, positive_risk):
    pos_text, neg_text = positive["hypothesis"], negative["hypothesis"]
    pos_num, neg_num = number_signature(pos_text), number_signature(neg_text)
    pos_quant, neg_quant = quantity_signature(pos_text), quantity_signature(neg_text)
    pos_neg, neg_neg = negation_signature(pos_text), negation_signature(neg_text)
    pos_time, neg_time = temporal_signature(pos_text), temporal_signature(neg_text)
    pos_source, neg_source = source_signature(pos_text), source_signature(neg_text)
    pos_diff, neg_diff, replacements = differing_content(
        sf["left_content"], sf["right_content"]
    )

    raw_slots = []
    common_quantity_units = set(pos_quant) & set(neg_quant)
    quantity_changed_values = {
        value for unit in common_quantity_units if pos_quant[unit] != neg_quant[unit]
        for value in pos_quant[unit]
    }
    if quantity_changed_values:
        raw_slots.append("quantity")
    pos_pred_neg, neg_pred_neg = predicate_negation(pos_text), predicate_negation(neg_text)
    common_predicates = set(pos_pred_neg) & set(neg_pred_neg)
    lexical_predicates = common_predicates - BE_FORMS - {
        "be", "have", "do", "can", "could", "may", "might", "must", "should",
        "would", "will", "shall",
    }
    negation_changed_predicates = {
        predicate for predicate in lexical_predicates
        if pos_pred_neg[predicate] != neg_pred_neg[predicate]
    }
    if negation_changed_predicates:
        raw_slots.append("negation")
    if temporal_contrast(pos_text, neg_text, pos_quant, neg_quant):
        raw_slots.append("temporal")
    if (pos_source and neg_source and pos_source != neg_source and
            (len(pos_diff) + len(neg_diff) <= 3 or sf["score"] >= 0.75)):
        raw_slots.append("source")

    special = (set(str(v) for v in pos_num + neg_num) | NEGATION | TEMPORAL_WORDS |
               TIME_UNITS | set(NUMBER_WORDS) | set(ORDINAL_WORDS))
    object_pos = [token for token in pos_diff if token not in special and not any(char.isdigit() for char in token)]
    object_neg = [token for token in neg_diff if token not in special and not any(char.isdigit() for char in token)]
    if (replacements > 0 and object_pos and object_neg and sf["score"] >= MIN_OBJECT_SURFACE and
            strict_object_contrast(sf["left_content"], sf["right_content"], object_pos, object_neg)):
        raw_slots.append("object")

    pos_subject, pos_predicate, pos_form = anchors(pos_text)
    neg_subject, neg_predicate, neg_form = anchors(neg_text)
    subject_dice = set_dice(pos_subject, neg_subject)
    predicate_match = bool(pos_predicate & neg_predicate)
    explicit_anchor = subject_dice >= 0.5 and predicate_match
    implicit_anchor = (
        pos_subject == {"<implicit-you>"} and neg_subject == {"<implicit-you>"} and predicate_match
    )
    skeleton_anchor = sf["content_dice"] >= 0.58 and sf["sequence"] >= 0.56
    anchor_match = explicit_anchor or implicit_anchor or skeleton_anchor

    full_positive = not positive["partially_positive"]
    slot_alignment = {}
    if "quantity" in raw_slots:
        slot_alignment["quantity"] = full_positive or bool(
            quantity_changed_values & positive_risk["numbers"]
        )
    if "negation" in raw_slots:
        slot_alignment["negation"] = full_positive or bool(
            negation_changed_predicates & positive_risk["content"]
        ) or bool(set(pos_neg) & positive_risk["lexical"])
    if "temporal" in raw_slots:
        slot_alignment["temporal"] = full_positive or bool(
            set(pos_time) & positive_risk["temporal"]
        ) or bool(quantity_changed_values & positive_risk["numbers"])
    if "source" in raw_slots:
        slot_alignment["source"] = full_positive or bool(
            set(pos_source) & positive_risk["sources"]
        )
    if "object" in raw_slots:
        slot_alignment["object"] = full_positive or bool(
            set(object_pos) & positive_risk["content"]
        )
    slots = [name for name in raw_slots if slot_alignment.get(name, False)]
    risk_aligned = bool(slots)

    details = {
        "slots": slots,
        "positive_numbers": pos_num, "negative_numbers": neg_num,
        "positive_quantities": pos_quant, "negative_quantities": neg_quant,
        "positive_negation": pos_neg, "negative_negation": neg_neg,
        "positive_temporal": pos_time, "negative_temporal": neg_time,
        "positive_sources": pos_source, "negative_sources": neg_source,
        "positive_difference_terms": pos_diff,
        "negative_difference_terms": neg_diff,
        "positive_risk_terms": sorted(positive_risk["content"]),
        "raw_slots_before_gold_scope_alignment": raw_slots,
        "slot_gold_scope_alignment": slot_alignment,
        "risk_difference_aligned": risk_aligned,
        "subject_dice": subject_dice, "predicate_match": predicate_match,
        "anchor_match": anchor_match,
        "anchor_mode": (
            "explicit_subject_predicate" if explicit_anchor else
            "implicit_subject_predicate" if implicit_anchor else
            "ordered_lexical_skeleton" if skeleton_anchor else "none"
        ),
        "positive_form": pos_form, "negative_form": neg_form,
    }
    return details


def quality(example):
    tokens = lexical_tokens(example["hypothesis"])
    content = content_tokens(tokens)
    if len(tokens) < 4 or len(content) < 2:
        return False
    normalized = " ".join(tokens)
    if normalized in {
        "unable to answer base on provide passage", "insufficient information",
        "here are possible interpretation", "here are the step",
    }:
        return False
    meta_patterns = (
        "unable to answer", "no information", "not enough information", "insufficient information",
        "do not provide information", "does not provide information", "not specifically mention",
        "no specific mention", "cannot be determine", "cannot answer",
    )
    if any(pattern in normalized for pattern in meta_patterns):
        return False
    token_set = set(tokens)
    if (("information" in token_set and token_set & NEGATION and
         token_set & {"provide", "contain", "mention", "answer", "determine"}) or
            "not mention" in normalized):
        return False
    return True


def normalized_claim(text: str):
    return " ".join(lexical_tokens(text))


def primary_slot(slots):
    for value in ("negation", "quantity", "temporal", "source", "object"):
        if value in slots:
            return value
    raise AssertionError(slots)


def material_signature(answer):
    payload = {
        "question": answer["question"],
        "passages": [(p["passage_id"], p["body_sha256"]) for p in answer["passages"]],
    }
    return digest(payload)


def evidence_signature(example):
    return tuple(
        (source["passage_id"], tuple(item["text_sha256"] for item in source["selected"]))
        for source in example["evidence"]
    )


def evidence_overlap(left, right):
    a = {(p, item) for p, values in evidence_signature(left) for item in values}
    b = {(p, item) for p, values in evidence_signature(right) for item in values}
    return len(a & b) / len(a | b) if a or b else 1.0


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(values[low])
    return float(values[low] * (high - position) + values[high] * (position - low))


def mine_pairs(examples, answers):
    answer_by_id = {row["response_id"]: row for row in answers}
    by_source = defaultdict(list)
    for row in examples:
        by_source[row["source_id"]].append(row)

    exclusion = Counter()
    ambiguous = defaultdict(list)
    eligible_candidates = []
    source_meta = {}
    for source_id, rows in sorted(by_source.items()):
        source_answers = {row["response_id"] for row in rows}
        signatures = {material_signature(answer_by_id[rid]) for rid in source_answers}
        questions = {answer_by_id[rid]["question"] for rid in source_answers}
        groups = {row["group_id"] for row in rows}
        folds = {row["held_fold"] for row in rows}
        assert len(signatures) == len(questions) == len(groups) == len(folds) == 1
        source_meta[source_id] = {
            "material_sha256": next(iter(signatures)), "group_id": next(iter(groups)),
            "held_fold": next(iter(folds)), "answers": len(source_answers), "claims": len(rows),
        }
        positives = [row for row in rows if row["gold_label"] == 1]
        negatives = [row for row in rows if row["gold_label"] == 0]
        for positive in positives:
            if not quality(positive):
                exclusion["positive_low_content"] += len(negatives)
                continue
            positive_risk = risk_profile(positive, answer_by_id[positive["response_id"]])
            for negative in negatives:
                if positive["response_id"] == negative["response_id"]:
                    exclusion["same_response"] += 1
                    continue
                if not quality(negative):
                    exclusion["negative_low_content"] += 1
                    continue
                if normalized_claim(positive["hypothesis"]) == normalized_claim(negative["hypothesis"]):
                    exclusion["normalized_exact_opposite_gold"] += 1
                    if len(ambiguous["normalized_exact_opposite_gold"]) < 40:
                        ambiguous["normalized_exact_opposite_gold"].append((positive, negative))
                    continue
                sf = surface_features(positive["hypothesis"], negative["hypothesis"])
                if sf["score"] < MIN_SURFACE:
                    exclusion["surface_below_threshold"] += 1
                    continue
                slot = slot_analysis(positive, negative, sf, positive_risk)
                if not slot["slots"]:
                    reason = ("gold_scope_slot_mismatch" if
                              slot["raw_slots_before_gold_scope_alignment"] else
                              "no_explicit_slot_contrast")
                    exclusion[reason] += 1
                    if sf["score"] >= 0.80 and len(ambiguous["high_similarity_no_slot"]) < 40:
                        ambiguous["high_similarity_no_slot"].append((positive, negative))
                    continue
                if not slot["anchor_match"]:
                    exclusion["subject_predicate_or_skeleton_mismatch"] += 1
                    continue
                if not slot["risk_difference_aligned"]:
                    exclusion["partial_gold_difference_not_aligned"] += 1
                    continue
                packed_same = evidence_signature(positive) == evidence_signature(negative)
                overlap = evidence_overlap(positive, negative)
                candidate = {
                    "positive": positive, "negative": negative, "surface": sf, "slot": slot,
                    "primary_slot": primary_slot(slot["slots"]),
                    "packed_premise_same": packed_same,
                    "selected_evidence_overlap": overlap,
                }
                eligible_candidates.append(candidate)

    # Rare structural types receive first access; each positive can retain two
    # alternatives and no clean endpoint can dominate more than three pairs.
    by_positive = defaultdict(list)
    for candidate in eligible_candidates:
        by_positive[candidate["positive"]["example_index"]].append(candidate)
    selected = []
    negative_uses = Counter()
    type_order = {name: index for index, name in enumerate(
        ("negation", "quantity", "temporal", "source", "object")
    )}
    for positive_index in sorted(by_positive):
        candidates = sorted(
            by_positive[positive_index],
            key=lambda row: (
                type_order[row["primary_slot"]],
                -int(row["packed_premise_same"]),
                -row["surface"]["score"],
                -row["selected_evidence_overlap"],
                row["negative"]["example_index"],
            ),
        )
        chosen, used_primary = [], set()
        # First pass keeps structurally distinct contrasts for one positive.
        for candidate in candidates:
            negative_index = candidate["negative"]["example_index"]
            if negative_uses[negative_index] >= MAX_USES_PER_NEGATIVE:
                continue
            if candidate["primary_slot"] in used_primary:
                continue
            chosen.append(candidate)
            used_primary.add(candidate["primary_slot"])
            negative_uses[negative_index] += 1
            if len(chosen) == MAX_PAIRS_PER_POSITIVE:
                break
        # Second pass fills an unused slot if only one type was available.
        if len(chosen) < MAX_PAIRS_PER_POSITIVE:
            chosen_ids = {row["negative"]["example_index"] for row in chosen}
            for candidate in sorted(candidates, key=lambda row: (
                    -int(row["packed_premise_same"]), -row["surface"]["score"],
                    -row["selected_evidence_overlap"], row["negative"]["example_index"])):
                negative_index = candidate["negative"]["example_index"]
                if negative_index in chosen_ids or negative_uses[negative_index] >= MAX_USES_PER_NEGATIVE:
                    continue
                chosen.append(candidate)
                chosen_ids.add(negative_index)
                negative_uses[negative_index] += 1
                if len(chosen) == MAX_PAIRS_PER_POSITIVE:
                    break
        selected.extend(chosen)

    selected.sort(key=lambda row: (
        row["positive"]["held_fold"], row["positive"]["group_id"],
        row["positive"]["example_index"], row["negative"]["example_index"],
    ))
    pair_rows = []
    selected_group_counts = Counter(row["positive"]["group_id"] for row in selected)
    selected_groups = len(selected_group_counts)
    for index, candidate in enumerate(selected):
        positive, negative = candidate["positive"], candidate["negative"]
        source = source_meta[positive["source_id"]]
        assert positive["group_id"] == negative["group_id"] == source["group_id"]
        assert positive["held_fold"] == negative["held_fold"] == source["held_fold"]
        row = {
            "schema_version": "microclaim-indomain-pair-v5",
            "pair_index": index,
            "pair_id": digest([positive["microclaim_id"], negative["microclaim_id"]])[:24],
            "partition": "fit",
            "source_id": positive["source_id"], "group_id": positive["group_id"],
            "held_fold": positive["held_fold"],
            "question_sha256": digest(positive["question"]),
            "material_sha256": source["material_sha256"],
            "positive": {
                "example_index": positive["example_index"], "response_id": positive["response_id"],
                "microclaim_id": positive["microclaim_id"], "gold_label": positive["gold_label"],
                "partially_positive": positive["partially_positive"],
                "gold_label_types": sorted({span["label_type"] for span in positive["overlapping_gold_spans"]}),
                "input_token_length": positive["input_token_length"],
                "hypothesis": positive["hypothesis"],
            },
            "negative": {
                "example_index": negative["example_index"], "response_id": negative["response_id"],
                "microclaim_id": negative["microclaim_id"], "gold_label": negative["gold_label"],
                "input_token_length": negative["input_token_length"],
                "hypothesis": negative["hypothesis"],
            },
            "contrast": {
                "primary_slot": candidate["primary_slot"],
                "all_slots": candidate["slot"]["slots"],
                "surface_similarity": candidate["surface"]["score"],
                "token_sequence_similarity": candidate["surface"]["sequence"],
                "content_dice": candidate["surface"]["content_dice"],
                "anchor_mode": candidate["slot"]["anchor_mode"],
                "subject_dice": candidate["slot"]["subject_dice"],
                "predicate_match": candidate["slot"]["predicate_match"],
                "positive_difference_terms": candidate["slot"]["positive_difference_terms"],
                "negative_difference_terms": candidate["slot"]["negative_difference_terms"],
                "risk_difference_aligned": candidate["slot"]["risk_difference_aligned"],
                "packed_top2_premise_same": candidate["packed_premise_same"],
                "selected_evidence_jaccard": candidate["selected_evidence_overlap"],
                "quantity": {
                    "positive": candidate["slot"]["positive_quantities"],
                    "negative": candidate["slot"]["negative_quantities"],
                },
                "negation": {
                    "positive": candidate["slot"]["positive_negation"],
                    "negative": candidate["slot"]["negative_negation"],
                },
                "temporal": {
                    "positive": candidate["slot"]["positive_temporal"],
                    "negative": candidate["slot"]["negative_temporal"],
                },
                "source_reference": {
                    "positive": candidate["slot"]["positive_sources"],
                    "negative": candidate["slot"]["negative_sources"],
                },
            },
            "supervision": "endpoint labels copied unchanged from frozen v4 gold projection; pair mining makes no factual claim",
            "full_fit_pair_training_weight": (
                len(selected) / (selected_groups * selected_group_counts[positive["group_id"]])
            ),
        }
        pair_rows.append(row)

    ambiguity_rows = []
    for reason in sorted(ambiguous):
        for positive, negative in ambiguous[reason]:
            ambiguity_rows.append({
                "reason": reason, "source_id": positive["source_id"],
                "group_id": positive["group_id"], "held_fold": positive["held_fold"],
                "positive_example_index": positive["example_index"],
                "negative_example_index": negative["example_index"],
                "positive_hypothesis": positive["hypothesis"],
                "negative_hypothesis": negative["hypothesis"],
                "action": "excluded_from_pair_training",
            })
    return pair_rows, eligible_candidates, exclusion, ambiguity_rows, source_meta


def pair_batches(rows):
    ordered = sorted(rows, key=lambda row: (
        max(row["positive"]["input_token_length"], row["negative"]["input_token_length"]),
        row["pair_index"],
    ))
    batches, current = [], []
    for row in ordered:
        trial = current + [row]
        width = max(max(x["positive"]["input_token_length"], x["negative"]["input_token_length"])
                    for x in trial)
        if current and (len(trial) > PAIR_MAX_EXAMPLES or 2 * len(trial) * width > PAIR_TOKEN_BUDGET):
            batches.append(current)
            current = [row]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def training_cost(rows):
    plans = []
    for fold in list(range(FOLDS)) + ["full_fit"]:
        active = rows if fold == "full_fit" else [row for row in rows if row["held_fold"] != fold]
        batches = pair_batches(active)
        logical = sum(row["positive"]["input_token_length"] + row["negative"]["input_token_length"]
                      for row in active)
        padded = sum(
            2 * len(batch) * max(
                max(row["positive"]["input_token_length"], row["negative"]["input_token_length"])
                for row in batch
            ) for batch in batches
        )
        plans.append({
            "model": fold, "training_pairs": len(active), "logical_endpoint_tokens": logical,
            "padded_endpoint_tokens": padded, "pair_microbatches": len(batches),
            "estimated_pair_pass_seconds": padded / REFERENCE_TOKENS_PER_SECOND,
        })
    total_padded = sum(row["padded_endpoint_tokens"] for row in plans)
    base_planned = 35_513_024
    return {
        "batching": {
            "padded_token_budget": PAIR_TOKEN_BUDGET, "maximum_pairs": PAIR_MAX_EXAMPLES,
            "each_pair_has_two_endpoints": True,
        },
        "per_model": plans,
        "all_six_models_padded_endpoint_tokens": total_padded,
        "base_v4_all_six_models_planned_train_padded_tokens": base_planned,
        "estimated_rank_pass_overhead_fraction": total_padded / base_planned,
        "estimated_rank_pass_minutes_all_six_models": total_padded / REFERENCE_TOKENS_PER_SECOND / 60,
        "estimate_basis": "same-host reference throughput from expanded-v4 preparation; planning estimate only",
        "checkpoint_storage_increment": 0,
    }


def audit(pair_rows, examples, answers, eligible_candidates, exclusion, ambiguity_rows, source_meta):
    by_index = {row["example_index"]: row for row in examples}
    answer_by_id = {row["response_id"]: row for row in answers}
    assert len(by_index) == FIT_EXAMPLES
    assert len({row["pair_id"] for row in pair_rows}) == len(pair_rows)
    groups_by_fold = defaultdict(set)
    sources_by_fold = defaultdict(set)
    for row in pair_rows:
        positive = by_index[row["positive"]["example_index"]]
        negative = by_index[row["negative"]["example_index"]]
        assert positive["gold_label"] == 1 and negative["gold_label"] == 0
        assert row["partition"] == positive["partition"] == negative["partition"] == "fit"
        assert positive["response_id"] != negative["response_id"]
        assert row["source_id"] == positive["source_id"] == negative["source_id"]
        assert row["group_id"] == positive["group_id"] == negative["group_id"]
        assert row["held_fold"] == positive["held_fold"] == negative["held_fold"]
        assert answer_by_id[positive["response_id"]]["question"] == answer_by_id[negative["response_id"]]["question"]
        assert material_signature(answer_by_id[positive["response_id"]]) == row["material_sha256"]
        assert material_signature(answer_by_id[negative["response_id"]]) == row["material_sha256"]
        assert row["positive"]["gold_label"] == 1 and row["negative"]["gold_label"] == 0
        assert row["contrast"]["all_slots"] and row["contrast"]["risk_difference_aligned"]
        assert row["contrast"]["surface_similarity"] >= MIN_SURFACE
        assert normalized_claim(positive["hypothesis"]) != normalized_claim(negative["hypothesis"])
        groups_by_fold[row["held_fold"]].add(row["group_id"])
        sources_by_fold[row["held_fold"]].add(row["source_id"])
    for left in range(FOLDS):
        for right in range(left + 1, FOLDS):
            assert not (groups_by_fold[left] & groups_by_fold[right])
            assert not (sources_by_fold[left] & sources_by_fold[right])

    similarities = [row["contrast"]["surface_similarity"] for row in pair_rows]
    evidence_overlaps = [row["contrast"]["selected_evidence_jaccard"] for row in pair_rows]
    primary = Counter(row["contrast"]["primary_slot"] for row in pair_rows)
    all_slots = Counter(slot for row in pair_rows for slot in row["contrast"]["all_slots"])
    anchor = Counter(row["contrast"]["anchor_mode"] for row in pair_rows)
    label_types = Counter(kind for row in pair_rows for kind in row["positive"]["gold_label_types"])
    pair_by_fold = Counter(row["held_fold"] for row in pair_rows)
    train_by_model = {
        str(fold): sum(row["held_fold"] != fold for row in pair_rows) for fold in range(FOLDS)
    }
    endpoint_positive = Counter(row["positive"]["example_index"] for row in pair_rows)
    endpoint_negative = Counter(row["negative"]["example_index"] for row in pair_rows)
    selected_groups = {row["group_id"] for row in pair_rows}
    selected_sources = {row["source_id"] for row in pair_rows}
    fit_positive_total = sum(row["gold_label"] == 1 for row in examples)
    fit_conflict_positive_total = sum(
        row["gold_label"] == 1 and any("Conflict" in span["label_type"]
                                      for span in row["overlapping_gold_spans"])
        for row in examples
    )
    selected_conflict_positive = {
        row["positive"]["example_index"] for row in pair_rows
        if any("Conflict" in kind for kind in row["positive"]["gold_label_types"])
    }
    post_surface = (len(eligible_candidates) + exclusion["no_explicit_slot_contrast"] +
                    exclusion["gold_scope_slot_mismatch"] +
                    exclusion["subject_predicate_or_skeleton_mismatch"])
    surface_screened = post_surface + exclusion["surface_below_threshold"]
    pair_group_mass = defaultdict(float)
    for row in pair_rows:
        pair_group_mass[row["group_id"]] += row["full_fit_pair_training_weight"]
    return {
        "status": "CPU_pair_mining_and_integrity_audit_passed",
        "fit_examples_read": len(examples), "fit_answers_read": len(answers),
        "calibration_rows_read": 0, "official_test_rows_read": 0,
        "sources": len(source_meta), "groups": len({v["group_id"] for v in source_meta.values()}),
        "eligible_candidates_before_endpoint_caps": len(eligible_candidates),
        "selected_pairs": len(pair_rows),
        "selected_positive_endpoints": len(endpoint_positive),
        "selected_negative_endpoints": len(endpoint_negative),
        "selected_sources": len(selected_sources), "selected_groups": len(selected_groups),
        "positive_endpoint_coverage": len(endpoint_positive) / fit_positive_total,
        "conflict_positive_endpoint_coverage": (
            len(selected_conflict_positive) / fit_conflict_positive_total
        ),
        "maximum_pairs_per_positive_observed": max(endpoint_positive.values(), default=0),
        "maximum_uses_per_negative_observed": max(endpoint_negative.values(), default=0),
        "pairs_by_held_fold": {str(k): pair_by_fold[k] for k in range(FOLDS)},
        "training_pairs_by_OOF_model": train_by_model,
        "full_fit_training_pairs": len(pair_rows),
        "primary_slot_counts": dict(sorted(primary.items())),
        "all_slot_counts": dict(sorted(all_slots.items())),
        "anchor_mode_counts": dict(sorted(anchor.items())),
        "positive_gold_type_counts": dict(sorted(label_types.items())),
        "partial_positive_pairs": sum(row["positive"]["partially_positive"] for row in pair_rows),
        "surface_similarity": {
            "min": min(similarities, default=None), "p10": percentile(similarities, 0.10),
            "median": percentile(similarities, 0.50), "p90": percentile(similarities, 0.90),
            "max": max(similarities, default=None),
            "opposite_gold_candidates_screened_after_content_filters": surface_screened,
            "candidates_passing_similarity_threshold": post_surface,
            "similarity_screen_pass_rate": (
                post_surface / surface_screened if surface_screened else None
            ),
        },
        "selected_evidence_jaccard": {
            "min": min(evidence_overlaps, default=None), "median": percentile(evidence_overlaps, 0.50),
            "mean": statistics.fmean(evidence_overlaps) if evidence_overlaps else None,
            "packed_top2_premise_exact_pairs": sum(
                row["contrast"]["packed_top2_premise_same"] for row in pair_rows
            ),
        },
        "excluded_candidate_reasons": dict(sorted(exclusion.items())),
        "saved_ambiguity_examples": len(ambiguity_rows),
        "structural_ambiguity_flags_on_retained_pairs": {
            "partial_positive_projection": sum(row["positive"]["partially_positive"] for row in pair_rows),
            "multiple_slot_changes": sum(len(row["contrast"]["all_slots"]) > 1 for row in pair_rows),
            "ordered_skeleton_instead_of_explicit_subject_predicate": sum(
                row["contrast"]["anchor_mode"] == "ordered_lexical_skeleton" for row in pair_rows
            ),
            "different_claim_selected_top2_pack": sum(
                not row["contrast"]["packed_top2_premise_same"] for row in pair_rows
            ),
            "surface_similarity_below_0.60": sum(
                row["contrast"]["surface_similarity"] < 0.60 for row in pair_rows
            ),
            "interpretation": "Flags disclose possible pair confounds; they do not alter or reinterpret gold truth labels.",
        },
        "full_fit_pair_weight": {
            "mean": statistics.fmean(row["full_fit_pair_training_weight"] for row in pair_rows),
            "group_mass_min": min(pair_group_mass.values()),
            "group_mass_max": max(pair_group_mass.values()),
            "rule": "equal group mass, then equal pair mass; recompute normalization after excluding each OOF held fold",
        },
        "integrity": {
            "same_exact_source": True, "same_exact_question": True,
            "same_exact_underlying_material": True, "different_responses": True,
            "opposite_frozen_gold_projection": True, "same_group_and_held_fold": True,
            "cross_fold_pairs": 0, "normalized_exact_opposite_pairs": 0,
            "pair_ids_unique": True, "calibration_absent": True, "official_test_absent": True,
        },
    }


def protocol():
    return {
        "version": "microclaim-indomain-pairs-v5",
        "purpose": "Add in-domain relation-sensitive ranking supervision to expanded-v4 without changing any gold label.",
        "source": {
            "examples": "Only the leading 34,919 fit rows of frozen expanded-v4 examples.jsonl.",
            "answers": "Only the leading 3,680 fit rows of frozen expanded-v4 answers.jsonl.",
            "calibration": "Never parsed or used.", "official_test": "No path; never opened.",
        },
        "pair_definition": {
            "same_context": "Both endpoints have the exact same source_id, question and three underlying passage hashes, and different generated responses.",
            "opposite_labels": "Positive endpoint has frozen v4 gold_label=1; negative endpoint has frozen v4 gold_label=0.",
            "surface": f"Deterministic token/character similarity >= {MIN_SURFACE}; object-only contrasts require >= {MIN_OBJECT_SURFACE}.",
            "relation": "Require matched subject/predicate anchors or a strong ordered lexical skeleton plus a differing quantity, negation, temporal, source-reference or object slot.",
            "gold_alignment": "For partial positives, a positive-side changed term must overlap the unchanged risk-BPE terms; no new factual judgment is made.",
            "ambiguity_filters": "Exclude same-response pairs, normalized exact text with opposite projected labels, low-content scaffolds, pairs without explicit slot contrast, anchor mismatch and partial-label scope mismatch.",
            "caps": {"per_positive": MAX_PAIRS_PER_POSITIVE, "per_negative": MAX_USES_PER_NEGATIVE},
        },
        "crossfit": "Each pair inherits its single source-connected held_fold. OOF model f may train only on pairs whose held_fold != f; full-fit uses all pairs.",
        "proposed_loss": {
            "base": "Unchanged expanded-v4 weighted BCE over all training microclaims.",
            "ranking": "Weighted RankNet softplus(risk_negative - risk_positive), with equal source-group mass then equal pair mass; weights are recomputed inside each OOF training split.",
            "single_candidate_weight": 0.25,
            "epoch_objective": "mean_weighted_BCE + 0.25 * mean_group_balanced_pair_rank_loss",
            "endpoint_inputs": "Reuse each endpoint's unchanged expanded-v4 top-2 cross-encoder input.",
        },
        "evaluation": "Unchanged v4 fit-OOF thresholding and calibration evaluation; this artifact performs no training or scoring.",
        "baselines": "Untouched.", "GPU_used": False,
    }


def build_report(audit_data, cost):
    slot = audit_data["primary_slot_counts"]
    sim = audit_data["surface_similarity"]
    ev = audit_data["selected_evidence_jaccard"]
    return "\n".join([
        "# Expanded-v5 in-domain microclaim pairs", "",
        "本阶段只用 expanded-v4 的 fit 数据构造训练对；没有读取 calibration/test、没有训练模型、没有改 v4 或 baseline。", "",
        "## 结果", "",
        f"- 选出 **{audit_data['selected_pairs']:,}** 对，覆盖 {audit_data['selected_positive_endpoints']:,} 个错误微主张、{audit_data['selected_negative_endpoints']:,} 个安全微主张。",
        f"- 五折 pair 数：{audit_data['pairs_by_held_fold']}；每个 OOF 模型只用 held_fold 不等于自己的 pair。",
        f"- 主槽位：{slot}。",
        f"- 表面相似度：最小 {sim['min']:.3f}，中位 {sim['median']:.3f}，P90 {sim['p90']:.3f}；阈值只放行内容合格候选的 {sim['similarity_screen_pass_rate']:.2%}。",
        f"- 所有 pair 的 source、问题和底层三篇资料完全一致；top-2 打包证据也完全相同的有 {ev['packed_top2_premise_exact_pairs']:,} 对。",
        f"- 只覆盖全部错误微主张的 {audit_data['positive_endpoint_coverage']:.2%} 和 {audit_data['selected_groups']}/{audit_data['groups']} 个 group。",
        "", "## 歧义控制", "",
        f"剔除原因见 AUDIT.json。最关键的是剔除了 {audit_data['excluded_candidate_reasons'].get('normalized_exact_opposite_gold', 0):,} 个“文本相同但投影标签相反”的候选，"
        f"以及 {audit_data['excluded_candidate_reasons'].get('gold_scope_slot_mismatch', 0):,} 个局部 gold 与实际差异槽位对不上的候选。"
        "这些剔除只判断配对结构，不重新判断事实真假。", "",
        "## v5 训练接法", "",
        "保留 v4 的全部 BCE；另加 `0.25 * softplus(risk_safe - risk_error)`。pair 按 source group 等权，"
        "OOF 训练严格排除本折 pair。这样直接惩罚“同资料、同关系骨架，只改一个关键槽位却仍判安全”的情况。", "",
        f"额外 pair pass 预计为 v4 训练 token 的 **{cost['estimated_rank_pass_overhead_fraction']:.1%}**，"
        f"六个模型合计约 **{cost['estimated_rank_pass_minutes_all_six_models']:.1f} 分钟**（规划值，尚未跑 GPU）。", "",
        "## 限制", "",
        "自然严格 pair 很少，因此它只能作为低权重定向修正，不能替代 34,919 条 BCE 主监督。"
        "RAGTruth 没有 gold 支持句或关系三元组。这里的“主体/谓词/槽位”由确定性词法规则识别；"
        "标签仍完全来自原 gold span 投影，所以这些 pair 是训练监督，不是新的事实证书。", "",
    ])


def prepare(preview=False):
    assert sha(V4_MANIFEST) == "35791a3b2881b098713d947273d7293f912f6f1a84ed2a50bedfe022a2c74370"
    assert sha(V4_COMPLETE) == "d6d6658a6aa82cc10fd56116ebd938b92b896155d85c706c2282939f29cd6c4e"
    examples = read_fit_prefix(EXAMPLES, FIT_EXAMPLES)
    answers = read_fit_prefix(ANSWERS, FIT_ANSWERS)
    assert len({row["source_id"] for row in examples}) == FIT_SOURCES
    assert len({row["group_id"] for row in examples}) == FIT_GROUPS
    pair_rows, eligible, exclusion, ambiguity_rows, source_meta = mine_pairs(examples, answers)
    audit_data = audit(pair_rows, examples, answers, eligible, exclusion, ambiguity_rows, source_meta)
    cost = training_cost(pair_rows)
    loss_spec = importlib.util.spec_from_file_location("microclaim_pairwise_rank_loss_v5", LOSS_CODE)
    loss_module = importlib.util.module_from_spec(loss_spec)
    assert loss_spec.loader is not None
    loss_spec.loader.exec_module(loss_module)
    loss_check = loss_module.cpu_check()
    if preview:
        print(json.dumps({"audit": audit_data, "training_cost": cost, "loss_check": loss_check},
                         ensure_ascii=False, indent=2))
        return
    if OUT.exists():
        assert OUT.is_dir() and not any(OUT.iterdir()), f"Refuse overwrite: {OUT}"
    else:
        OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol())
    save_jsonl(OUT / "pairs.jsonl", pair_rows)
    save_jsonl(OUT / "ambiguity_examples.jsonl", ambiguity_rows)
    save_json(OUT / "AUDIT.json", audit_data)
    save_json(OUT / "TRAINING_COST.json", cost)
    save_json(OUT / "LOSS_CPU_CHECK.json", loss_check)
    (OUT / "REPORT.md").write_text(build_report(audit_data, cost), encoding="utf-8")
    files = ("protocol.json", "pairs.jsonl", "ambiguity_examples.jsonl", "AUDIT.json",
             "TRAINING_COST.json", "LOSS_CPU_CHECK.json", "REPORT.md")
    manifest = {
        "version": "microclaim-indomain-pairs-v5",
        "created_utc": utc_now(), "source_code_sha256": sha(Path(__file__)),
        "loss_code_sha256": sha(LOSS_CODE),
        "v4_manifest_sha256": sha(V4_MANIFEST), "v4_complete_sha256": sha(V4_COMPLETE),
        "files_sha256": {name: sha(OUT / name) for name in files},
        "selected_pairs": len(pair_rows), "calibration_rows_read": 0,
        "official_test_opened": False, "GPU_used": False, "baselines_modified": False,
    }
    save_json(OUT / "manifest.json", manifest)
    complete = {
        "status": "complete_CPU_pair_preparation_and_audit",
        "manifest_sha256": sha(OUT / "manifest.json"), "selected_pairs": len(pair_rows),
        "source_code_sha256": sha(Path(__file__)), "loss_code_sha256": sha(LOSS_CODE),
        "GPU_used": False,
        "calibration_rows_read": 0, "official_test_opened": False,
        "v4_modified": False, "baselines_modified": False,
    }
    save_json(OUT / "complete.json", complete)
    print("MICROCLAIM_INDOMAIN_PAIRS_V5_PREPARED", len(pair_rows), flush=True)


def verify():
    required = ("protocol.json", "pairs.jsonl", "ambiguity_examples.jsonl", "AUDIT.json",
                "TRAINING_COST.json", "LOSS_CPU_CHECK.json", "REPORT.md", "manifest.json", "complete.json")
    assert all((OUT / name).is_file() for name in required)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
    assert manifest["source_code_sha256"] == sha(Path(__file__))
    assert complete["source_code_sha256"] == sha(Path(__file__))
    assert manifest["loss_code_sha256"] == complete["loss_code_sha256"] == sha(LOSS_CODE)
    assert complete["manifest_sha256"] == sha(OUT / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    examples = read_fit_prefix(EXAMPLES, FIT_EXAMPLES)
    answers = read_fit_prefix(ANSWERS, FIT_ANSWERS)
    with (OUT / "pairs.jsonl").open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    # Re-run all saved-pair invariants without re-mining or reading later partitions.
    audit_data = audit(pairs, examples, answers, [], Counter(), [], {
        source_id: {
            "group_id": rows[0]["group_id"], "held_fold": rows[0]["held_fold"],
            "material_sha256": material_signature(next(
                answer for answer in answers if answer["source_id"] == source_id
            )),
        }
        for source_id, rows in ((sid, [x for x in examples if x["source_id"] == sid])
                                for sid in sorted({x["source_id"] for x in examples}))
    })
    saved_audit = json.loads((OUT / "AUDIT.json").read_text(encoding="utf-8"))
    assert audit_data["selected_pairs"] == saved_audit["selected_pairs"] == manifest["selected_pairs"]
    assert audit_data["integrity"] == saved_audit["integrity"]
    print("MICROCLAIM_INDOMAIN_PAIRS_V5_VERIFIED", len(pairs), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "prepare", "verify"))
    args = parser.parse_args()
    if args.command == "preview":
        prepare(preview=True)
    elif args.command == "prepare":
        prepare(preview=False)
    else:
        verify()


if __name__ == "__main__":
    main()
