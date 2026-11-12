"""Prepare exact-span token supervision for the auxiliary conflict pilot.

Only auxiliary fit and QA fit are read.  Calibration/test paths are absent.
No model weights are loaded and no GPU operation occurs.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/auxiliary_conflict_token_pilot_v1"
AUX = ROOT / "auxiliary_human_v1"
V4_DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4_RUN = ROOT / "results/microclaim_crossencoder_expanded_v4"
MODERN_TOKENIZER = ROOT.parent / "models/ModernBERT-base-nli"
MINICHECK_TOKENIZER = ROOT / "semantic_baseline/model"
ATOMIC_CODE = ROOT / "research/atomic_relation_audit_r32_v1/audit_atomic_relations.py"
NLI_CODE = HERE / "run_retrieved_evidence_nli_v1.py"
NLTK_DATA = ROOT / "semantic_baseline/nltk_data"

N_AUX = 9_678
N_AUX_SOURCES = 1_614
N_AUX_GROUPS = 1_558
N_CONFLICT_SPANS = 4_381
N_QA_FIT = 34_919
N_QA_ANSWERS = 3_680
QA_FOLD = 0
WINDOW_K = 4
PARENT_BPE_LIMIT = 96
MAX_PAIR_TOKENS = 2_048
KNOWN_TYPES = frozenset((
    "Evident Conflict", "Subtle Conflict",
    "Evident Baseless Info", "Subtle Baseless Info",
))
CONFLICT_TYPES = frozenset(("Evident Conflict", "Subtle Conflict"))
BASELESS_TYPES = KNOWN_TYPES - CONFLICT_TYPES
STAGES = {"aux": 0, "qa_train": 1, "qa_held": 2}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


atomic = load_module("atomic_conflict_aux_v1", ATOMIC_CODE)
nli = load_module("nli_retrieval_conflict_aux_v1", NLI_CODE)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def read_prefix(path, count):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for index in range(count):
            line = handle.readline()
            assert line, (path, index)
            rows.append(json.loads(line))
    return rows


def trim(text, left, right):
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def sentence_parents(text, punkt, tokenizer):
    """Exact parent-span routine used by semantic_baseline/run_semantic.py."""
    sentence_spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        blocks = list(punkt.span_tokenize(line))
        index = 0
        while index < len(blocks):
            left, right = blocks[index]
            if re.fullmatch(r"\s*(?:\d+[.)]|[-*•])\s*", line[left:right]) and index + 1 < len(blocks):
                right = blocks[index + 1][1]
                index += 1
            left, right = trim(text, offset + left, offset + right)
            if left < right:
                sentence_spans.append((left, right))
            index += 1
        offset += len(line)
    assert offset == len(text)

    pieces = []
    for outer_left, outer_right in sentence_spans:
        left, right = outer_left, outer_right
        while left < right:
            left, right = trim(text, left, right)
            if left >= right:
                break
            encoded = tokenizer(text[left:right], add_special_tokens=False,
                                return_offsets_mapping=True, truncation=False)
            if len(encoded["input_ids"]) <= PARENT_BPE_LIMIT:
                pieces.append((left, right))
                break
            stop = left + encoded["offset_mapping"][PARENT_BPE_LIMIT - 1][1]
            lower = left + encoded["offset_mapping"][max(0, 3 * PARENT_BPE_LIMIT // 4 - 1)][1]
            gaps = [match.start() + left for match in re.finditer(r"\s+", text[left:stop])
                    if match.start() + left >= lower]
            if gaps:
                stop = gaps[-1]
            while stop > left and len(tokenizer.encode(text[left:stop], add_special_tokens=False)) > PARENT_BPE_LIMIT:
                stop -= 1
            assert stop > left
            a, b = trim(text, left, stop)
            if a < b:
                pieces.append((a, b))
            left = stop
    covered = np.zeros(len(text), dtype=bool)
    for left, right in pieces:
        covered[left:right] = True
    assert all(char.isspace() or covered[index] for index, char in enumerate(text))
    return pieces


def microclaims(text, punkt, tokenizer, prefix):
    output = []
    parents = sentence_parents(text, punkt, tokenizer)
    for parent_index, (left, right) in enumerate(parents):
        spans, reasons = atomic.atomic_split(text, left, right)
        for child_index, (start, end) in enumerate(spans):
            if not any(char.isalnum() for char in text[start:end]):
                continue
            output.append({
                "microclaim_id": f"{prefix}__atomic_{len(output):03d}",
                "microclaim_index": len(output), "parent_claim_index": parent_index,
                "child_index": child_index, "children_in_parent": len(spans),
                "parent_split_reasons": reasons, "start": start, "end": end,
                "text": text[start:end],
                "word_count": len(atomic.WORD.findall(text[start:end])),
            })
    assert output
    covered = np.zeros(len(text), dtype=bool)
    for row in output:
        covered[row["start"]:row["end"]] = True
    assert all(not char.isalnum() or covered[index] for index, char in enumerate(text))
    return output


class EvidenceIndex:
    def __init__(self, text):
        self.text = text
        self.rows = []
        for sentence_id, (left, right) in enumerate(nli.sentence_spans(text)):
            value = text[left:right]
            tokens = nli.words(value)
            assert tokens
            self.rows.append({"sentence_id": sentence_id, "start": left, "end": right,
                              "text": value, "sha256": digest(value),
                              "tokens": tokens, "frequency": Counter(tokens)})
        self.average = sum(len(row["tokens"]) for row in self.rows) / len(self.rows)
        self.df = Counter()
        for row in self.rows:
            self.df.update(set(row["tokens"]))

    def select(self, query_text):
        query = sorted(set(nli.words(query_text)))
        assert query
        query_set = set(query)
        ranked = []
        count = len(self.rows)
        for row in self.rows:
            score = 0.0
            for term in query:
                tf = row["frequency"][term]
                if not tf:
                    continue
                df = self.df[term]
                idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                denominator = tf + nli.BM25_K1 * (
                    1.0 - nli.BM25_B + nli.BM25_B * len(row["tokens"]) / self.average)
                score += idf * tf * (nli.BM25_K1 + 1.0) / denominator
            coverage = len(query_set & set(row["tokens"])) / len(query_set)
            ranked.append((score, row["sentence_id"], coverage, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        chosen = []
        union = set()
        for rank, (score, _sid, coverage, row) in enumerate(ranked[:2], 1):
            union.update(row["tokens"])
            chosen.append({"rank": rank, "sentence_id": row["sentence_id"],
                           "start": row["start"], "end": row["end"],
                           "text": row["text"], "text_sha256": row["sha256"],
                           "bm25": float(score), "query_term_coverage": float(coverage)})
        return chosen, len(query_set & union) / len(query_set)


def masks_for(text, labels):
    conflict = np.zeros(len(text), dtype=bool)
    baseless = np.zeros(len(text), dtype=bool)
    for label in labels:
        assert label["label_type"] in KNOWN_TYPES
        left, right = int(label["start"]), int(label["end"])
        assert 0 <= left <= right <= len(text) and text[left:right] == label["text"]
        target = conflict if label["label_type"] in CONFLICT_TYPES else baseless
        for at in range(left, right):
            if text[at].isalnum():
                target[at] = True
    return conflict, baseless


def classify_claim(claim, conflict, baseless):
    left, right = claim["start"], claim["end"]
    lexical = np.fromiter((char.isalnum() for char in claim["text"]), dtype=bool)
    c = int(conflict[left:right].sum())
    b = int(baseless[left:right].sum())
    total = int(lexical.sum())
    assert total > 0
    if c and not b:
        kind = "conflict"
    elif not c and not b:
        kind = "safe"
    elif b and not c:
        kind = "baseless"
    else:
        kind = "mixed"
    return kind, c, b, total


def make_premise(question, evidence):
    lines = [f"Question: {question}", "Evidence:"]
    lines.extend(f"[{index}] {row['text']}" for index, row in enumerate(evidence, 1))
    return "\n".join(lines)


def matching_select(rows, stage):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    selected = []
    stats = Counter()
    for group_id in sorted(by_group):
        members = by_group[group_id]
        positives = [row for row in members if row["kind"] == "conflict"]
        safe = [row for row in members if row["kind"] == "safe"]
        if not positives:
            continue
        target_word = statistics.median(row["word_count"] for row in positives)
        target_coverage = statistics.median(row["evidence_query_coverage"] for row in positives)
        safe.sort(key=lambda row: (
            abs(row["word_count"] - target_word),
            abs(row["evidence_query_coverage"] - target_coverage),
            digest([row["response_id"], row["microclaim_id"]]),
        ))
        chosen_safe = safe[:len(positives)]
        for row in positives + chosen_safe:
            copy = dict(row)
            copy["stage"] = stage
            copy["selection"] = "all_clean_conflict" if row["kind"] == "conflict" else "within_group_matched_safe"
            selected.append(copy)
        stats["groups"] += 1
        stats["conflict"] += len(positives)
        stats["safe"] += len(chosen_safe)
        stats["groups_without_enough_safe"] += int(len(chosen_safe) < len(positives))
        stats["unmatched_conflict"] += max(0, len(positives) - len(chosen_safe))
    selected.sort(key=lambda row: (row["group_id"], row["response_id"], row["microclaim_index"], row["kind"]))
    return selected, dict(stats)


def verify_aux_token_row(source, token):
    assert str(token["response_id"]) == str(source["response_id"])
    for key in ("source_id", "group_id", "task_type", "answer_sha256"):
        assert str(token[key]) == str(source[key])
    text = source["original_response"]
    conflict, baseless = masks_for(text, source["labels"])
    risk_char = conflict | baseless
    expected = []
    for left, right in token["response_token_offsets"]:
        expected.append(int(risk_char[int(left):int(right)].any()))
    assert expected == token["risk_mask"]
    assert token["answer_risk"] == int(bool(source["labels"]))
    return conflict, baseless


def prepare_aux(punkt, mini_tokenizer):
    manifest = json.loads((AUX / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha(AUX / name) == expected, name
    rows, traces = [], []
    inventory = Counter()
    evidence_cache = {}
    with (AUX / "candidate_fit.jsonl").open(encoding="utf-8") as source_handle, \
            (AUX / "token_inputs.jsonl").open(encoding="utf-8") as token_handle:
        for answer_index in range(N_AUX):
            source_line, token_line = source_handle.readline(), token_handle.readline()
            assert source_line and token_line
            source, token = json.loads(source_line), json.loads(token_line)
            assert source["partition"] == "auxiliary_candidate_fit" and source["official_split"] == "train"
            assert source["quality"] == "good" and not source["new_labels_generated"]
            conflict, baseless = verify_aux_token_row(source, token)
            material = source["retrieved_passages"]
            source_id = str(source["source_id"])
            if source_id not in evidence_cache:
                evidence_cache[source_id] = EvidenceIndex(material)
            else:
                assert evidence_cache[source_id].text == material
            claims = microclaims(source["original_response"], punkt, mini_tokenizer,
                                 f"aux_{source['response_id']}")
            local_records = []
            for claim in claims:
                kind, c_chars, b_chars, lexical_chars = classify_claim(claim, conflict, baseless)
                evidence, coverage = evidence_cache[source_id].select(claim["text"])
                row = {
                    "origin": "auxiliary_human_v1", "source_id": source_id,
                    "group_id": source["group_id"], "response_id": str(source["response_id"]),
                    "task_type": source["task_type"], "question": source["question"],
                    **claim, "kind": kind, "conflict_alnum_chars": c_chars,
                    "baseless_alnum_chars": b_chars, "lexical_alnum_chars": lexical_chars,
                    "evidence": evidence, "evidence_query_coverage": float(coverage),
                    "premise": make_premise(source["question"], evidence),
                    "labels": [{"span_index": index, "label_type": label["label_type"],
                                "start": int(label["start"]), "end": int(label["end"]),
                                "text_sha256": digest(label["text"]),
                                "meta_sha256": digest(label.get("meta", "")),
                                "implicit_true": bool(label.get("implicit_true", False)),
                                "due_to_null": bool(label.get("due_to_null", False))}
                               for index, label in enumerate(source["labels"])
                               if max(claim["start"], int(label["start"])) < min(claim["end"], int(label["end"]))],
                    "answer_sha256": source["answer_sha256"],
                }
                rows.append(row); local_records.append(row); inventory[kind] += 1
            for span_index, label in enumerate(source["labels"]):
                if label["label_type"] not in CONFLICT_TYPES:
                    continue
                overlapping = [row["microclaim_id"] for row in local_records
                               if max(row["start"], int(label["start"])) < min(row["end"], int(label["end"]))]
                alnum = sum(char.isalnum() for char in label["text"])
                traces.append({
                    "origin": "auxiliary_human_v1", "source_id": source_id,
                    "group_id": source["group_id"], "response_id": str(source["response_id"]),
                    "span_index": span_index, "label_type": label["label_type"],
                    "start": int(label["start"]), "end": int(label["end"]),
                    "text": label["text"], "text_sha256": digest(label["text"]),
                    "meta": label.get("meta", ""), "meta_sha256": digest(label.get("meta", "")),
                    "alnum_chars": alnum, "overlapping_microclaim_ids": overlapping,
                })
            if (answer_index + 1) % 1000 == 0:
                print("AUX_CONFLICT_PREP", answer_index + 1, N_AUX, len(rows), flush=True)
        assert not source_handle.readline(), "unexpected auxiliary source rows"
        assert not token_handle.readline(), "unexpected auxiliary token rows"
    assert len(traces) == N_CONFLICT_SPANS
    assert len({row["source_id"] for row in rows}) == N_AUX_SOURCES
    assert len({row["group_id"] for row in rows}) == N_AUX_GROUPS
    selected, selection = matching_select(rows, "aux")
    selected_ids = {row["microclaim_id"] for row in selected}
    for trace in traces:
        trace["selected_microclaim_ids"] = [value for value in trace["overlapping_microclaim_ids"]
                                             if value in selected_ids]
    return selected, traces, {"inventory": dict(inventory), "selection": selection,
                              "all_microclaims": len(rows)}


def qa_coverage(row):
    query = set(nli.words(row["claim_text"]))
    evidence_words = set()
    for passage in row["evidence"]:
        for selected in passage["selected"]:
            evidence_words.update(nli.words(selected["text"]))
    return len(query & evidence_words) / len(query) if query else 0.0


def prepare_qa():
    examples = read_prefix(V4_DATA / "examples.jsonl", N_QA_FIT)
    answers = read_prefix(V4_DATA / "answers.jsonl", N_QA_ANSWERS)
    assert all(row["partition"] == "fit" and row["example_index"] == index
               for index, row in enumerate(examples))
    assert all(row["partition"] == "fit" and row["response_index"] == index
               for index, row in enumerate(answers))
    answer_by_id = {str(row["response_id"]): row for row in answers}
    candidates, held = [], []
    inventory = Counter()
    for row in examples:
        text = answer_by_id[str(row["response_id"])]["original_response"]
        labels = row["overlapping_gold_spans"]
        conflict, baseless = masks_for(text, [dict(label, text=text[int(label["start"]):int(label["end"])])
                                              for label in labels])
        claim = {"start": int(row["char_start"]), "end": int(row["char_end"]),
                 "text": row["claim_text"]}
        kind, c_chars, b_chars, lexical_chars = classify_claim(claim, conflict, baseless)
        record = {
            "origin": "expanded_v4_fit", "source_id": str(row["source_id"]),
            "group_id": row["group_id"], "response_id": str(row["response_id"]),
            "task_type": "QA", "question": row["question"],
            "microclaim_id": row["microclaim_id"], "microclaim_index": int(row["microclaim_index"]),
            "example_index": int(row["example_index"]), "held_fold": int(row["held_fold"]),
            "start": claim["start"], "end": claim["end"], "text": claim["text"],
            "word_count": len(atomic.WORD.findall(claim["text"])),
            "kind": kind, "conflict_alnum_chars": c_chars,
            "baseless_alnum_chars": b_chars, "lexical_alnum_chars": lexical_chars,
            "evidence": row["evidence"], "evidence_query_coverage": qa_coverage(row),
            "premise": row["premise"],
            "labels": [{"span_index": int(label["span_index"]), "label_type": label["label_type"],
                        "start": int(label["start"]), "end": int(label["end"]),
                        "text_sha256": digest(text[int(label["start"]):int(label["end"])]),
                        "meta_sha256": None,
                        "implicit_true": bool(label["implicit_true"]),
                        "due_to_null": bool(label["due_to_null"])} for label in labels],
            "answer_sha256": answer_by_id[str(row["response_id"])]["answer_sha256"],
            "all_overlapping_bpe_indices": list(map(int, row["all_overlapping_bpe_indices"])),
            "mapped_eligible_window_starts": list(map(int, row["mapped_eligible_window_starts"])),
        }
        inventory[("held" if row["held_fold"] == QA_FOLD else "train", kind)] += 1
        (held if row["held_fold"] == QA_FOLD else candidates).append(record)
    train, selection = matching_select(candidates, "qa_train")
    for row in held:
        row["stage"] = "qa_held"
        row["selection"] = "all_held_microclaims"
    held.sort(key=lambda row: row["example_index"])
    return train, held, answers, examples, {"inventory": {f"{part}_{kind}": value for (part, kind), value in inventory.items()},
                                            "selection": selection}


def token_targets(tokenizer, row):
    encoded = tokenizer(row["premise"], row["text"], add_special_tokens=True,
                        padding=False, truncation=False, return_offsets_mapping=True)
    assert len(encoded["input_ids"]) <= MAX_PAIR_TOKENS
    sequence_ids = encoded.sequence_ids()
    text = row["text"]
    conflict = np.zeros(len(text), dtype=bool)
    baseless = np.zeros(len(text), dtype=bool)
    for label in row["labels"]:
        left = max(row["start"], int(label["start"])) - row["start"]
        right = min(row["end"], int(label["end"])) - row["start"]
        if left >= right:
            continue
        target = conflict if label["label_type"] in CONFLICT_TYPES else baseless
        for at in range(left, right):
            if text[at].isalnum():
                target[at] = True
    targets, char_counts, hypothesis_intervals = [], [], []
    sequence_codes, hypothesis_offset_start, hypothesis_offset_end = [], [], []
    positive_char_mass = supervised_char_mass = 0.0
    for token_index, (seq, offset) in enumerate(zip(sequence_ids, encoded["offset_mapping"])):
        target, count = -1.0, 0
        sequence_codes.append(-1 if seq is None else int(seq))
        hypothesis_offset_start.append(int(offset[0]) if seq == 1 else -1)
        hypothesis_offset_end.append(int(offset[1]) if seq == 1 else -1)
        if seq == 1:
            left, right = map(int, offset)
            chars = [at for at in range(left, right) if text[at].isalnum()]
            if chars:
                count = len(chars)
                if not baseless[chars].any():
                    target = float(conflict[chars].sum() / len(chars))
                    positive_char_mass += count * target
                    supervised_char_mass += count
                hypothesis_intervals.append((token_index, row["start"] + left,
                                             row["start"] + right, count))
        targets.append(target); char_counts.append(count if target >= 0 else 0)
    if row["stage"] != "qa_held":
        assert supervised_char_mass > 0
        if row["kind"] == "conflict":
            assert positive_char_mass > 0
        else:
            assert row["kind"] == "safe" and positive_char_mass == 0
    return (encoded["input_ids"], targets, char_counts, sequence_codes,
            hypothesis_offset_start, hypothesis_offset_end, hypothesis_intervals,
            positive_char_mass, supervised_char_mass)


def example_weights(metadata, stage):
    indices = [index for index, row in enumerate(metadata) if row["stage"] == stage]
    tree = defaultdict(lambda: defaultdict(list))
    for index in indices:
        row = metadata[index]
        tree[row["group_id"]][row["response_id"]].append(index)
    base = np.zeros(len(metadata), dtype=np.float64)
    for answers in tree.values():
        for claims in answers.values():
            base[claims] = 1.0 / (len(tree) * len(answers) * len(claims))
    p = np.asarray([row.get("positive_char_fraction", 0.0) for row in metadata])
    pos_mass = float(np.sum(base[indices] * p[indices]))
    neg_mass = float(np.sum(base[indices] * (1.0 - p[indices])))
    assert pos_mass > 0 and neg_mass > 0
    total = pos_mass + neg_mass
    positive_factor, negative_factor = total / (2 * pos_mass), total / (2 * neg_mass)
    group_scale = {}
    desired = total / len(tree)
    for group, answers in tree.items():
        members = [index for claims in answers.values() for index in claims]
        mass = np.sum(base[members] * (positive_factor * p[members] +
                                       negative_factor * (1.0 - p[members])))
        group_scale[group] = desired / mass
    weights = np.zeros(len(metadata), dtype=np.float64)
    for index in indices:
        weights[index] = base[index] * group_scale[metadata[index]["group_id"]]
    weights[indices] *= len(indices) / weights[indices].sum()
    return weights, {"positive_factor": positive_factor, "negative_factor": negative_factor,
                     "groups": len(tree), "examples": len(indices)}


def build_encoded(aux_rows, qa_train, qa_held, tokenizer):
    rows = aux_rows + qa_train + qa_held
    input_ids = array("i")
    targets = array("f")
    char_counts = array("H")
    sequence_codes = array("b")
    hypothesis_offset_start = array("i")
    hypothesis_offset_end = array("i")
    indptr = [0]
    metadata, intervals = [], []
    for index, row in enumerate(rows):
        (ids, local_targets, local_counts, local_sequence_codes,
         local_offset_start, local_offset_end, local_intervals,
         pos_mass, supervised_mass) = token_targets(tokenizer, row)
        flat_start = len(input_ids)
        input_ids.extend(ids); targets.extend(local_targets); char_counts.extend(local_counts)
        sequence_codes.extend(local_sequence_codes)
        hypothesis_offset_start.extend(local_offset_start)
        hypothesis_offset_end.extend(local_offset_end)
        indptr.append(len(input_ids))
        record = {key: row[key] for key in (
            "origin", "stage", "source_id", "group_id", "response_id", "task_type",
            "microclaim_id", "microclaim_index", "start", "end", "text", "kind",
            "word_count", "evidence_query_coverage", "selection", "answer_sha256")}
        if "example_index" in row:
            record["example_index"] = row["example_index"]
            record["held_fold"] = row["held_fold"]
            record["all_overlapping_bpe_indices"] = row["all_overlapping_bpe_indices"]
            record["mapped_eligible_window_starts"] = row["mapped_eligible_window_starts"]
        record.update({
            "input_index": index, "flat_start": flat_start, "flat_end": len(input_ids),
            "input_tokens": len(ids),
            "supervised_tokens": sum(value >= 0 for value in local_targets),
            "positive_char_mass": pos_mass, "supervised_char_mass": supervised_mass,
            "positive_char_fraction": pos_mass / supervised_mass if supervised_mass else 0.0,
            "premise_sha256": digest(row["premise"]), "hypothesis_sha256": digest(row["text"]),
            "trace_labels": row["labels"],
        })
        metadata.append(record)
        intervals.append([(flat_start + ti, left, right, count)
                          for ti, left, right, count in local_intervals])
        if (index + 1) % 2000 == 0:
            print("CONFLICT_TOKEN_ENCODE", index + 1, len(rows), len(input_ids), flush=True)
    aux_weight, aux_balance = example_weights(metadata, "aux")
    qa_weight, qa_balance = example_weights(metadata, "qa_train")
    example_weight = aux_weight + qa_weight
    return rows, metadata, intervals, {
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "targets": np.asarray(targets, dtype=np.float32),
        "char_counts": np.asarray(char_counts, dtype=np.uint16),
        "sequence_code": np.asarray(sequence_codes, dtype=np.int8),
        "hypothesis_offset_start": np.asarray(hypothesis_offset_start, dtype=np.int32),
        "hypothesis_offset_end": np.asarray(hypothesis_offset_end, dtype=np.int32),
        "indptr": np.asarray(indptr, dtype=np.int64),
        "stage_code": np.asarray([STAGES[row["stage"]] for row in rows], dtype=np.int8),
        "example_weight": example_weight.astype(np.float32),
    }, {"aux": aux_balance, "qa_train": qa_balance}


def build_held_windows(qa_held, metadata, intervals, arrays, qa_answers, qa_examples):
    held_offset = next(index for index, row in enumerate(metadata) if row["stage"] == "qa_held")
    record_by_example = {row["example_index"]: (held_offset + local, row)
                         for local, row in enumerate(qa_held)}
    assert len(record_by_example) == len(qa_held)
    examples_by_response = defaultdict(list)
    for row in qa_examples:
        if row["held_fold"] == QA_FOLD:
            examples_by_response[str(row["response_id"])].append(row)

    with np.load(V4_RUN / "fold_0/predictions.npz", allow_pickle=False) as z:
        v4_indices = z["held_indices"].copy()
        v4_scores = z["held_scores"].astype(np.float64)
    assert set(map(int, v4_indices)) == set(record_by_example)
    v4_by_example = dict(zip(map(int, v4_indices), map(float, v4_scores)))

    window_values, window_indptr = array("q"), [0]
    labels, ec, sc, answer_local, max_coverage, v4_window = [], [], [], [], [], []
    answer_labels, answer_ids = [], []
    held_answer_index = 0
    for answer in qa_answers:
        rid = str(answer["response_id"])
        local = examples_by_response.get(rid)
        if not local:
            continue
        local.sort(key=lambda row: row["example_index"])
        answer_ids.append(rid); answer_labels.append(int(answer["answer_risk"]))
        lexical = np.asarray(answer["lexical_mask"], dtype=bool)
        risk = np.asarray(answer["risk_mask"], dtype=bool)
        offsets = np.asarray(answer["response_token_offsets"], dtype=np.int64)
        text = answer["original_response"]
        eligible = [start for start in range(max(0, len(lexical) - WINDOW_K + 1))
                    if lexical[start:start + WINDOW_K].any()]
        span_keys = {}
        for row in local:
            for span in row["overlapping_gold_spans"]:
                key = (span["label_type"], int(span["start"]), int(span["end"]))
                span_keys[key] = True
        type_masks = {name: np.zeros(len(lexical), dtype=bool) for name in CONFLICT_TYPES}
        for (name, left, right) in span_keys:
            if name not in CONFLICT_TYPES:
                continue
            for bpe, (a, b) in enumerate(offsets):
                lo, hi = max(int(a), left), min(int(b), right)
                if lo < hi and any(text[at].isalnum() for at in range(lo, hi)):
                    type_masks[name][bpe] = True

        bpe_tokens = defaultdict(set)
        claim_by_start = defaultdict(list)
        for example in local:
            global_record, _ = record_by_example[int(example["example_index"])]
            claim_by_start.update({})
            for start in example["mapped_eligible_window_starts"]:
                claim_by_start[int(start)].append(int(example["example_index"]))
            for flat_position, token_left, token_right, _count in intervals[global_record]:
                for bpe in example["all_overlapping_bpe_indices"]:
                    a, b = offsets[int(bpe)]
                    lo, hi = max(int(a), token_left), min(int(b), token_right)
                    if lo < hi and any(text[at].isalnum() for at in range(lo, hi)):
                        bpe_tokens[int(bpe)].add(int(flat_position))
        for bpe in np.flatnonzero(lexical):
            assert bpe_tokens[int(bpe)], (rid, int(bpe))

        for start in eligible:
            values = sorted({position for bpe in range(start, start + WINDOW_K)
                             for position in bpe_tokens.get(bpe, ())})
            assert values
            window_values.extend(values); window_indptr.append(len(window_values))
            labels.append(int(risk[start:start + WINDOW_K].any()))
            ec.append(int(type_masks["Evident Conflict"][start:start + WINDOW_K].any()))
            sc.append(int(type_masks["Subtle Conflict"][start:start + WINDOW_K].any()))
            answer_local.append(held_answer_index)
            owner_examples = claim_by_start[start]
            assert owner_examples
            max_coverage.append(max(metadata[record_by_example[index][0]]["evidence_query_coverage"]
                                    for index in owner_examples))
            v4_window.append(max(v4_by_example[index] for index in owner_examples))
        held_answer_index += 1
    assert len(labels) and held_answer_index == len(answer_ids)
    return {
        "held_window_token_indptr": np.asarray(window_indptr, dtype=np.int64),
        "held_window_token_values": np.asarray(window_values, dtype=np.int64),
        "held_window_label": np.asarray(labels, dtype=np.int8),
        "held_window_ec": np.asarray(ec, dtype=np.int8),
        "held_window_sc": np.asarray(sc, dtype=np.int8),
        "held_window_answer_index": np.asarray(answer_local, dtype=np.int32),
        "held_window_max_evidence_coverage": np.asarray(max_coverage, dtype=np.float32),
        "held_v4_window_score": np.asarray(v4_window, dtype=np.float32),
        "held_answer_label": np.asarray(answer_labels, dtype=np.int8),
        "held_answer_ids": np.asarray(answer_ids),
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "preparation_complete.json").exists()
    import nltk
    from nltk.tokenize.punkt import PunktTokenizer
    nltk.data.path.insert(0, str(NLTK_DATA))
    punkt = PunktTokenizer("english")
    mini_tokenizer = AutoTokenizer.from_pretrained(MINICHECK_TOKENIZER, local_files_only=True,
                                                   use_fast=True, trust_remote_code=False)
    modern_tokenizer = AutoTokenizer.from_pretrained(MODERN_TOKENIZER, local_files_only=True,
                                                     use_fast=True, trust_remote_code=False)

    aux_rows, traces, aux_stats = prepare_aux(punkt, mini_tokenizer)
    qa_train, qa_held, qa_answers, qa_examples, qa_stats = prepare_qa()
    all_rows, metadata, intervals, encoded, balances = build_encoded(
        aux_rows, qa_train, qa_held, modern_tokenizer)
    held = build_held_windows(qa_held, metadata, intervals, encoded, qa_answers, qa_examples)
    encoded.update(held)
    write_jsonl(OUT / "examples.jsonl", metadata)
    write_jsonl(OUT / "conflict_span_trace.jsonl", traces)
    write_npz(OUT / "arrays.npz", **encoded)

    stage_counts = Counter(row["stage"] for row in metadata)
    kind_counts = Counter((row["stage"], row["kind"]) for row in metadata)
    input_lengths = np.diff(encoded["indptr"])
    train_tokens = {}
    for stage in ("aux", "qa_train"):
        active = [index for index, row in enumerate(metadata) if row["stage"] == stage]
        train_tokens[stage] = {
            "examples": len(active),
            "logical_input_tokens": int(input_lengths[active].sum()),
            "supervised_hypothesis_tokens": int(sum(metadata[index]["supervised_tokens"] for index in active)),
            "positive_character_mass": float(sum(metadata[index]["positive_char_mass"] for index in active)),
            "supervised_character_mass": float(sum(metadata[index]["supervised_char_mass"] for index in active)),
        }
    result = {
        "status": "cpu_preparation_complete",
        "auxiliary": aux_stats, "qa": qa_stats,
        "selected_stage_counts": dict(stage_counts),
        "selected_kind_counts": {f"{stage}_{kind}": value for (stage, kind), value in kind_counts.items()},
        "conflict_span_trace": {
            "released_spans": len(traces),
            "with_alphanumeric_characters": sum(row["alnum_chars"] > 0 for row in traces),
            "with_atomic_overlap": sum(bool(row["overlapping_microclaim_ids"]) for row in traces),
            "with_selected_conflict_microclaim": sum(bool(row["selected_microclaim_ids"]) for row in traces),
            "type_counts": dict(Counter(row["label_type"] for row in traces)),
        },
        "encoded": {
            "examples": len(metadata), "flat_tokens": len(encoded["input_ids"]),
            "max_pair_tokens": int(input_lengths.max()),
            "p50_pair_tokens": float(np.median(input_lengths)),
            "p95_pair_tokens": float(np.quantile(input_lengths, 0.95)),
            "no_truncation": True,
        },
        "training_mass": train_tokens, "stage_balance": balances,
        "held_evaluation": {
            "answers": len(encoded["held_answer_label"]),
            "windows": len(encoded["held_window_label"]),
            "positive_windows": int(encoded["held_window_label"].sum()),
            "ec_windows": int(encoded["held_window_ec"].sum()),
            "sc_windows": int(encoded["held_window_sc"].sum()),
            "window_token_edges": len(encoded["held_window_token_values"]),
            "high_overlap_windows": int(np.sum(encoded["held_window_max_evidence_coverage"] >= 0.5)),
        },
        "isolation": {
            "auxiliary_material_groups": N_AUX_GROUPS,
            "qa_train_groups": len({row["group_id"] for row in qa_train}),
            "qa_held_groups": len({row["group_id"] for row in qa_held}),
            "qa_train_held_group_intersection": len(
                {row["group_id"] for row in qa_train} & {row["group_id"] for row in qa_held}),
            "upstream_quarantined_sources": 62,
            "upstream_isolation_audit_sha256": sha(AUX / "INDEPENDENT_AUDIT.json"),
        },
        "scope": {"calibration_rows_read": 0, "test_rows_read": 0,
                  "model_weights_loaded": False, "GPU_used": False,
                  "published_baselines_modified": False},
        "sources_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            AUX / "manifest.json", AUX / "candidate_fit.jsonl", AUX / "token_inputs.jsonl",
            AUX / "INDEPENDENT_AUDIT.json", ATOMIC_CODE, NLI_CODE,
            V4_RUN / "fold_0/predictions.npz", OUT / "PROTOCOL.md")},
    }
    write_json(OUT / "preparation_complete.json", result)
    write_json(OUT / "manifest.json", {
        "status": "cpu_only_ready_for_review",
        "files_sha256": {name: sha(OUT / name) for name in
                          ("PROTOCOL.md", "examples.jsonl", "conflict_span_trace.jsonl",
                           "arrays.npz", "preparation_complete.json")},
        "calibration_rows_read": 0, "test_rows_read": 0,
        "model_weights_loaded": False, "GPU_used": False,
    })
    print("AUXILIARY_CONFLICT_TOKEN_CPU_PREPARED", len(metadata), flush=True)


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    complete = json.loads((OUT / "preparation_complete.json").read_text(encoding="utf-8"))
    with np.load(OUT / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["indptr"][0] == 0 and arrays["indptr"][-1] == len(arrays["input_ids"])
        assert len(arrays["targets"]) == len(arrays["char_counts"]) == len(arrays["input_ids"])
        assert len(arrays["sequence_code"]) == len(arrays["hypothesis_offset_start"]) == len(arrays["input_ids"])
        assert len(arrays["hypothesis_offset_end"]) == len(arrays["input_ids"])
        assert len(arrays["held_window_token_indptr"]) == len(arrays["held_window_label"]) + 1
        assert arrays["held_window_token_indptr"][-1] == len(arrays["held_window_token_values"])
        assert arrays["held_window_token_values"].min() >= 0
        assert arrays["held_window_token_values"].max() < len(arrays["input_ids"])
        assert not np.any(np.isnan(arrays["targets"]))
    assert complete["scope"] == {"calibration_rows_read": 0, "test_rows_read": 0,
                                  "model_weights_loaded": False, "GPU_used": False,
                                  "published_baselines_modified": False}
    print("AUXILIARY_CONFLICT_TOKEN_CPU_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify}[args.stage]()
