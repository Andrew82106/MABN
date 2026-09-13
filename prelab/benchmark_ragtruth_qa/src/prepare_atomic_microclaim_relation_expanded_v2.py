"""Prepare expanded, gold-labelled microclaim relation examples on CPU only.

The fit side uses the frozen 3,680-answer expansion and its label-blind atomic
microclaims.  Calibration reuses exactly the 2,267 scored microclaims already
frozen by atomic_microclaim_nli_v1.  No official-test path is present here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/atomic_microclaim_relation_expanded_v2"
FIT_RAW = ROOT / "fit_expansion/data/fit.jsonl"
FIT_TOKENS = ROOT / "fit_expansion/data/tokens_fit.jsonl"
FIT_MICROCLAIMS = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
FIT_MICROCLAIM_COMPLETE = ROOT / "research/atomic_relation_expanded_fit_v1/complete.json"
FIT_EXPORT_FREEZE = ROOT / "fit_expansion/data/export_freeze.json"
CURRENT_ATOMIC_INPUT = ROOT / "results/atomic_microclaim_nli_v1/inputs.jsonl"
CURRENT_ATOMIC_COMPLETE = ROOT / "results/atomic_microclaim_nli_v1/preparation_complete.json"
CURRENT_RETRIEVED_INPUT = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
CURRENT_RETRIEVED_COMPLETE = ROOT / "results/retrieved_evidence_nli_v1/preparation_complete.json"
CAL_TOKENS = ROOT / "data/tokens_calibration.jsonl"
GOLD_MANIFEST = ROOT / "data/gold_manifest.json"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
AUDIT_CODE = ROOT / "research/atomic_microclaim_relation_expanded_v2_audit/audit.py"

EXPECTED = {
    "fit_answers": 3680,
    "fit_groups": 615,
    "fit_sources": 634,
    "fit_atomic": 34941,
    "fit_scored": 34919,
    "fit_raw_bpes": 665708,
    "cal_answers": 159,
    "cal_groups": 154,
    "cal_scored": 2267,
    "cal_raw_bpes": 42798,
    "fit_windows": 653979,
    "fit_positive_windows": 58433,
    "cal_windows": 42241,
    "cal_positive_windows": 5984,
}
FOLDS = 5
WINDOW_K = 4
PAIR_LIMIT = 2048
CONNECTOR = re.compile(r"^\s*(?:(?:and|or|but|yet|whereas|while|which|who)\b[,;:]?\s*)+", re.I)
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


old_nli = load_module("retrieved_nli_for_expanded_relation", HERE / "run_retrieved_evidence_nli_v1.py")
atomic = load_module("atomic_rules_for_expanded_relation", ROOT / "research/atomic_relation_audit_r32_v1/audit_atomic_relations.py")


def lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_json(path: Path, value):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def write_jsonl(path: Path, rows):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return {
        "version": "atomic-microclaim-relation-expanded-v2",
        "purpose": "CPU data preparation for direct microclaim-level evidence-relation training",
        "partitions": {
            "fit": "Frozen deduplicated 3,680-answer fit expansion; 34,941 label-blind atomic records, of which 34,919 contain Unicode alphanumeric text.",
            "calibration": "Reuse exactly the 2,267 already-scored calibration microclaims in atomic_microclaim_nli_v1; do not reconstruct or expand calibration claims.",
            "official_test": "Absent and unopened.",
        },
        "input": {
            "retrieval": "For each microclaim, use the existing deterministic claim-only Unicode BM25 top-2 independently inside passages 1, 2 and 3. Calibration retrieval is copied exactly from the frozen current input.",
            "premise": "Question followed by passage-labelled retrieved top-2 evidence from all three passages, in passage/rank order.",
            "hypothesis": "The deterministic contextualized atomic microclaim; no answer-wide text is supplied.",
            "tokenizer": "Local tasksource/ModernBERT-base-nli tokenizer; paired encoding, no truncation; hard limit 2,048 tokens.",
        },
        "gold": {
            "microclaim": "Positive iff at least one lexical original answer BPE assigned to the microclaim has the unchanged human risk mask.",
            "window": "Original four-BPE stride-one window is eligible iff it contains a lexical BPE and positive iff any of its four BPEs has the unchanged risk mask.",
            "projection": "A window receives the maximum score among microclaims owning a lexical BPE in that window. Character spans, BPE ownership and ragged window-to-example indices are frozen.",
        },
        "groups": "The released source-connected group_id is the indivisible unit. Fit and calibration group/source sets must be disjoint.",
        "folds": "Five deterministic GroupKFold folds on fit examples; every source-connected group occurs in exactly one held fold.",
        "weights": "Within each partition: equal group mass, then equal answer mass, then equal microclaim mass. Fit training weights additionally balance binary classes and finally restore equal group loss mass; calibration training weight is null.",
        "baselines": "No baseline file, structure, parameter, threshold or result is modified.",
        "stage": "initialize -> CPU prepare -> CPU check -> independent CPU audit. No model load, model fit, inference or GPU use.",
    }


def hypothesis_for(claim):
    raw = claim["text"].strip()
    if int(claim["children_in_parent"]) == 1:
        return raw
    core, _ = atomic.strip_scaffold(raw)
    core = CONNECTOR.sub("", core).strip(" \t\r\n,;:") or raw
    antecedent = claim.get("antecedent_subject")
    if int(claim["child_index"]) > 0 and antecedent:
        subject = antecedent["text"].strip(" \t\r\n,;:")
        if subject and not re.match(rf"^{re.escape(subject)}\b", core, re.I):
            core = f"{subject} {core}"
    return core


def assign_tokens(text, offsets, claims):
    owners = [[] for _ in offsets]
    inverse = [[] for _ in claims]
    for token_index, (left, right) in enumerate(offsets):
        chars = [position for position in range(left, right) if text[position].isalnum()]
        if not chars:
            continue
        overlap = [sum(claim["start"] <= position < claim["end"] for position in chars) for claim in claims]
        owners[token_index] = [claim_index for claim_index, count in enumerate(overlap) if count]
        assert owners[token_index]
        for owner in owners[token_index]:
            inverse[owner].append(token_index)
    lexical = [any(text[position].isalnum() for position in range(left, right)) for left, right in offsets]
    assert list(map(bool, owners)) == lexical
    assert all(inverse)
    return owners, inverse


def annotation_overlap(labels, left, right):
    rows = []
    for index, label in enumerate(labels):
        if max(left, int(label["start"])) < min(right, int(label["end"])):
            rows.append({"span_index": index, "label_type": label["label_type"],
                         "start": int(label["start"]), "end": int(label["end"]),
                         "implicit_true": bool(label.get("implicit_true", False)),
                         "due_to_null": bool(label.get("due_to_null", False))})
    return rows


def eligible_windows(lexical, risk):
    eligible, positive = [], []
    for start in range(max(0, len(lexical) - WINDOW_K + 1)):
        if any(lexical[start:start + WINDOW_K]):
            eligible.append(start)
            if any(risk[start:start + WINDOW_K]):
                positive.append(start)
    return eligible, positive


def hierarchy_weights(partitions, response_ids, group_ids):
    n = len(partitions)
    output = np.zeros(n, dtype=np.float64)
    for partition in ("fit", "calibration"):
        active = [index for index, value in enumerate(partitions) if value == partition]
        tree = defaultdict(lambda: defaultdict(list))
        for index in active:
            tree[group_ids[index]][response_ids[index]].append(index)
        for answers in tree.values():
            for indices in answers.values():
                output[indices] = 1.0 / (len(answers) * len(indices))
        values = output[active]
        output[active] *= len(active) / values.sum()
    return output


def training_weights(partitions, response_ids, group_ids, labels):
    active = np.flatnonzero(np.asarray(partitions) == "fit")
    output = np.zeros(len(labels), dtype=np.float64)
    tree = defaultdict(lambda: defaultdict(list))
    for index in active:
        tree[group_ids[index]][response_ids[index]].append(int(index))
    for answers in tree.values():
        for indices in answers.values():
            output[indices] = 1.0 / (len(answers) * len(indices))
    output[active] *= len(active) / output[active].sum()
    y = np.asarray(labels, dtype=np.int8)
    mass = np.bincount(y[active], weights=output[active], minlength=2)
    assert np.all(mass > 0)
    output[active] *= (mass.sum() / (2.0 * mass))[y[active]]
    target_group_mass = len(active) / len(tree)
    for answers in tree.values():
        indices = [index for values in answers.values() for index in values]
        output[indices] *= target_group_mass / output[indices].sum()
    output[active] *= len(active) / output[active].sum()
    return output


def group_folds(partitions, group_ids):
    result = np.full(len(partitions), -1, dtype=np.int8)
    active = np.flatnonzero(np.asarray(partitions) == "fit")
    groups = np.asarray(group_ids, dtype=object)
    splitter = GroupKFold(FOLDS)
    for fold, (_, held_local) in enumerate(splitter.split(np.zeros(len(active)), groups=groups[active])):
        result[active[held_local]] = fold
    assert np.all(result[active] >= 0)
    assert all(len(set(result[active][groups[active] == group])) == 1 for group in set(groups[active]))
    return result


def selected_evidence(passages, retrieval):
    lookup = {(passage["passage_id"], sentence["sentence_id"]): sentence
              for passage in passages for sentence in passage["sentences"]}
    output = []
    for source in retrieval:
        assert source["passage_id"] in (1, 2, 3)
        chosen = []
        for selected in source["selected"]:
            sentence = lookup[(source["passage_id"], selected["sentence_id"])]
            assert sentence["text_sha256"] == selected["evidence_sha256"]
            chosen.append({
                "rank": int(selected["rank"]), "sentence_id": int(selected["sentence_id"]),
                "text": sentence["text"], "text_sha256": sentence["text_sha256"],
                "passage_char_start": int(sentence["passage_char_start"]),
                "passage_char_end": int(sentence["passage_char_end"]),
                "bm25": float(selected["bm25"]),
                "query_term_coverage": float(selected["query_term_coverage"]),
            })
        assert 1 <= len(chosen) <= 2
        output.append({"passage_id": int(source["passage_id"]), "selected": chosen,
                       "selected_union_query_coverage": float(source["selected_union_query_coverage"])})
    assert [row["passage_id"] for row in output] == [1, 2, 3]
    return output


def retrieval_signature(retrieval):
    return [(source["passage_id"], source["selected_union_query_coverage"], [
        (row["rank"], row["sentence_id"], row["bm25"], row["query_term_coverage"], row["evidence_sha256"])
        for row in source["selected"]]) for source in retrieval]


def make_premise(question, evidence):
    sections = [f"Question: {question}", "Evidence:"]
    for source in evidence:
        text = " ".join(item["text"] for item in source["selected"])
        sections.append(f"[Passage {source['passage_id']}] {text}")
    return "\n".join(sections)


def source_snapshot():
    paths = (
        Path(__file__), HERE / "run_retrieved_evidence_nli_v1.py",
        ROOT / "research/atomic_relation_audit_r32_v1/audit_atomic_relations.py",
        FIT_RAW, FIT_TOKENS, FIT_MICROCLAIMS, FIT_MICROCLAIM_COMPLETE, FIT_EXPORT_FREEZE,
        CURRENT_ATOMIC_INPUT, CURRENT_ATOMIC_COMPLETE, CURRENT_RETRIEVED_INPUT,
        CURRENT_RETRIEVED_COMPLETE, CAL_TOKENS, GOLD_MANIFEST,
        MODEL / "config.json", MODEL / "tokenizer.json", MODEL / "tokenizer_config.json",
        MODEL / "special_tokens_map.json", MODEL / "model.safetensors",
        MODEL / "download_manifest.json", AUDIT_CODE,
    )
    return {str(path.resolve()): sha(path) for path in paths}


def self_test():
    text = "Alpha rose, but beta did not."
    claims = [{"start": 0, "end": 11}, {"start": 12, "end": len(text)}]
    offsets = [[0, 5], [6, 10], [10, 11], [12, 15], [16, 20], [21, 24], [25, 28], [28, 29]]
    owners, inverse = assign_tokens(text, offsets, claims)
    assert inverse == [[0, 1], [3, 4, 5, 6]] and owners[2] == []
    eligible, positive = eligible_windows([1, 1, 0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0])
    assert eligible == [0, 1, 2, 3, 4] and positive == [0, 1]
    return {"status": "passed", "token_ownership": True, "window_k": WINDOW_K,
            "model_loaded": False, "GPU_used": False, "official_test_opened": False}


def initialize():
    assert not OUT.exists(), f"Preserve existing output: {OUT}"
    OUT.mkdir(parents=True)
    save_json(OUT / "PREFLIGHT.json", self_test())
    save_json(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Expanded atomic-microclaim relation data v2\n\n"
        "Build one supervised example per factual atomic microclaim. Each paired input contains the question, "
        "the claim-selected top-2 evidence from every released passage, and the contextualized claim. Gold comes "
        "only from the unchanged human answer spans projected through original BPE coordinates.\n\n"
        "Fit uses 3,680 answers. Calibration reuses exactly the existing 2,267 scored microclaims. Group folds and "
        "weights keep source-connected groups indivisible. This stage performs CPU preparation and audit only.\n",
        encoding="utf-8")
    (OUT / "FAILURE_HISTORY.md").write_text(
        "# Failure history\n\nNo failed stage at initialization. Any later exception is appended as a separate immutable `FAILURE_*.json`.\n",
        encoding="utf-8")
    print("EXPANDED_MICROCLAIM_RELATION_PROTOCOL_FROZEN", flush=True)


def build_claim_infos(fit_rows, fit_tokens, fit_claims, current_atomic, cal_tokens):
    claim_infos = {}
    answer_owners = {}
    response_ids, group_ids, partitions, labels = [], [], [], []
    example_cursor = 0
    for partition, answer_rows in (("fit", fit_rows), ("calibration", current_atomic)):
        for answer_index, answer in enumerate(answer_rows):
            rid = answer["response_id"]
            token = (fit_tokens if partition == "fit" else cal_tokens)[rid]
            text = token["original_response"]
            assert digest(text) == answer["answer_sha256"] == token["answer_sha256"]
            if partition == "fit":
                all_claims = sorted(fit_claims[rid], key=lambda row: row["microclaim_index"])
                claims = [claim for claim in all_claims if any(char.isalnum() for char in claim["text"])]
                owners, inverse = assign_tokens(text, token["response_token_offsets"], claims)
            else:
                claims = answer["claims"]
                owners = answer["lexical_token_microclaims"]
                inverse = [claim["lexical_token_indices"] for claim in claims]
                assert len(claims) and all(any(char.isalnum() for char in claim["text"]) for claim in claims)
            assert len(owners) == token["token_count"]
            local = []
            risk = np.asarray(token["risk_mask"], dtype=bool)
            for claim_id, (claim, lexical_bpes) in enumerate(zip(claims, inverse)):
                assert claim_id == (claim.get("claim_id", claim_id) if partition == "calibration" else claim_id)
                values = risk[lexical_bpes]
                label = int(values.any())
                info = {
                    "example_index": example_cursor,
                    "claim_id": claim_id,
                    "claim": claim,
                    "lexical_bpes": list(map(int, lexical_bpes)),
                    "risk_bpes": [int(index) for index in lexical_bpes if risk[index]],
                    "gold_label": label,
                    "risk_fraction": float(values.mean()),
                }
                local.append(info)
                response_ids.append(rid); group_ids.append(answer["group_id"])
                partitions.append(partition); labels.append(label)
                example_cursor += 1
            claim_infos[rid] = local
            answer_owners[rid] = owners
            if (answer_index + 1) % 500 == 0:
                print("EXPANDED_RELATION_LABEL_MAP", partition, answer_index + 1, len(answer_rows), flush=True)
    return claim_infos, answer_owners, response_ids, group_ids, partitions, labels


def prepare():
    assert OUT.exists() and json.loads((OUT / "protocol.json").read_text(encoding="utf-8")) == protocol()
    assert not (OUT / "prepare_started.json").exists()
    snapshot = source_snapshot()
    save_json(OUT / "prepare_started.json", {
        "status": "CPU_gold_projection_started", "started_at_utc": utc_now(),
        "source_sha256": snapshot, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False,
    })
    started = time.perf_counter()
    fit_rows = lines(FIT_RAW)
    fit_tokens = {row["response_id"]: row for row in lines(FIT_TOKENS)}
    fit_claims = defaultdict(list)
    for claim in lines(FIT_MICROCLAIMS):
        fit_claims[claim["response_id"]].append(claim)
    current_all = lines(CURRENT_ATOMIC_INPUT)
    current_fit = {row["response_id"]: row for row in current_all if row["partition"] == "fit"}
    current_cal = [row for row in current_all if row["partition"] == "calibration"]
    retrieved_all = lines(CURRENT_RETRIEVED_INPUT)
    retrieved_cal = {row["response_id"]: row for row in retrieved_all if row["partition"] == "calibration"}
    cal_tokens = {row["response_id"]: row for row in lines(CAL_TOKENS)}
    assert len(fit_rows) == len(fit_tokens) == len(fit_claims) == EXPECTED["fit_answers"]
    assert sum(map(len, fit_claims.values())) == EXPECTED["fit_atomic"]
    assert sum(row["token_count"] for row in fit_tokens.values()) == EXPECTED["fit_raw_bpes"]
    assert sum(row["token_count"] for row in cal_tokens.values()) == EXPECTED["cal_raw_bpes"]
    assert len(current_cal) == len(retrieved_cal) == len(cal_tokens) == EXPECTED["cal_answers"]
    assert sum(len(row["claims"]) for row in current_cal) == EXPECTED["cal_scored"]
    assert not ({row["group_id"] for row in fit_rows} & {row["group_id"] for row in current_cal})
    assert not ({row["source_id"] for row in fit_rows} & {row["source_id"] for row in current_cal})

    discarded = []
    for source in fit_rows:
        for claim in sorted(fit_claims[source["response_id"]], key=lambda row: row["microclaim_index"]):
            if not any(char.isalnum() for char in claim["text"]):
                discarded.append({
                    "partition": "fit", "response_id": claim["response_id"],
                    "source_id": claim["source_id"], "group_id": claim["group_id"],
                    "microclaim_id": claim["microclaim_id"], "microclaim_index": claim["microclaim_index"],
                    "char_start": claim["start"], "char_end": claim["end"], "text": claim["text"],
                    "reason": "no_unicode_alphanumeric_character; retained for 34,941-record accounting; excluded from supervised scoring",
                })
    assert len(discarded) == EXPECTED["fit_atomic"] - EXPECTED["fit_scored"]
    write_jsonl(OUT / "discarded_microclaims.jsonl", discarded)

    claim_infos, answer_owners, response_ids, group_ids, partitions, labels = build_claim_infos(
        fit_rows, fit_tokens, fit_claims, current_cal, cal_tokens)
    assert Counter(partitions) == {"fit": EXPECTED["fit_scored"], "calibration": EXPECTED["cal_scored"]}
    labels_array = np.asarray(labels, dtype=np.int8)
    hierarchy = hierarchy_weights(partitions, response_ids, group_ids)
    train_weight = training_weights(partitions, response_ids, group_ids, labels_array)
    folds = group_folds(partitions, group_ids)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast
    examples_path = OUT / "examples.jsonl"
    answers_path = OUT / "answers.jsonl"
    examples_pending = examples_path.with_suffix(".jsonl.pending")
    answers_pending = answers_path.with_suffix(".jsonl.pending")
    input_lengths = np.zeros(len(labels), dtype=np.int32)
    premise_lengths = np.zeros(len(labels), dtype=np.int32)
    hypothesis_lengths = np.zeros(len(labels), dtype=np.int32)
    response_index_array = np.zeros(len(labels), dtype=np.int32)
    group_index_array = np.zeros(len(labels), dtype=np.int32)
    claim_char_start = np.zeros(len(labels), dtype=np.int32)
    claim_char_end = np.zeros(len(labels), dtype=np.int32)
    group_names = sorted(set(group_ids))
    group_number = {name: index for index, name in enumerate(group_names)}
    response_order = []
    window_response_index, window_token_start, window_label = [], [], []
    window_claim_indptr, window_claim_values = [0], []
    old_fit_exact = 0
    input_token_total = Counter()
    label_types = Counter()
    partial = Counter()
    all_answer_rows = [("fit", row) for row in fit_rows] + [("calibration", row) for row in current_cal]
    with examples_pending.open("w", encoding="utf-8", newline="\n") as example_handle, \
            answers_pending.open("w", encoding="utf-8", newline="\n") as answer_handle:
        for response_index, (partition, source) in enumerate(all_answer_rows):
            rid = source["response_id"]
            token = (fit_tokens if partition == "fit" else cal_tokens)[rid]
            text = token["original_response"]
            question = source["question"] if partition == "fit" else retrieved_cal[rid]["question"]
            if partition == "fit":
                passages = old_nli.parse_passages(source["retrieved_passages"])
            else:
                passages = source["passages"]
                assert passages == retrieved_cal[rid]["passages"]
            infos = claim_infos[rid]
            owners = answer_owners[rid]
            lexical = list(map(bool, token["lexical_mask"]))
            risk = list(map(bool, token["risk_mask"]))
            eligible, positive = eligible_windows(lexical, risk)
            eligible_set = set(eligible)
            local_examples = []
            premises, hypotheses = [], []
            for info in infos:
                claim = info["claim"]
                hypothesis = claim["hypothesis"] if partition == "calibration" else hypothesis_for(claim)
                retrieval = claim["retrieval"] if partition == "calibration" else [
                    old_nli.select_evidence(hypothesis, passage) for passage in passages]
                evidence = selected_evidence(passages, retrieval)
                premise = make_premise(question, evidence)
                premises.append(premise); hypotheses.append(hypothesis)
                local_examples.append((info, claim, hypothesis, evidence, premise, retrieval))
            encoded = tokenizer(premises, hypotheses, add_special_tokens=True, padding=False, truncation=False)
            encoded_premise = tokenizer(premises, add_special_tokens=True, padding=False, truncation=False)
            encoded_hypothesis = tokenizer(hypotheses, add_special_tokens=True, padding=False, truncation=False)
            lengths = list(map(len, encoded["input_ids"]))
            assert max(lengths) <= PAIR_LIMIT
            example_start = local_examples[0][0]["example_index"]
            assert example_start == sum(len(claim_infos[old]) for old in response_order)
            for local_id, (packed, pair_length, premise_length, hypothesis_length) in enumerate(zip(
                    local_examples, lengths, map(len, encoded_premise["input_ids"]), map(len, encoded_hypothesis["input_ids"]))):
                info, claim, hypothesis, evidence, premise, retrieval = packed
                index = info["example_index"]
                assert index == example_start + local_id
                start, end = int(claim["start"]), int(claim["end"])
                assert text[start:end] == claim["text"]
                all_bpes = [token_index for token_index, (left, right) in enumerate(token["response_token_offsets"])
                            if max(start, left) < min(end, right)]
                mapped_windows = sorted({window_start for token_index in info["lexical_bpes"]
                                         for window_start in range(max(0, token_index - WINDOW_K + 1),
                                                                   min(token_index, len(lexical) - WINDOW_K) + 1)
                                         if window_start in eligible_set})
                overlaps = annotation_overlap(token["original_labels"], start, end)
                if info["gold_label"]:
                    assert overlaps
                for row in overlaps:
                    label_types[row["label_type"]] += 1
                partial[partition] += int(info["gold_label"] and info["risk_fraction"] < 1.0)
                input_lengths[index] = pair_length
                premise_lengths[index] = premise_length
                hypothesis_lengths[index] = hypothesis_length
                response_index_array[index] = response_index
                group_index_array[index] = group_number[source["group_id"]]
                claim_char_start[index] = start; claim_char_end[index] = end
                input_token_total[partition] += pair_length
                example = {
                    "schema_version": "atomic-microclaim-relation-expanded-v2",
                    "example_index": index, "partition": partition,
                    "response_id": rid, "source_id": source["source_id"], "group_id": source["group_id"],
                    "held_fold": int(folds[index]) if partition == "fit" else None,
                    "claim_id": local_id, "microclaim_id": claim["microclaim_id"],
                    "microclaim_index": int(claim["microclaim_index"]),
                    "parent_claim_index": int(claim["parent_claim_index"]),
                    "child_index": int(claim["child_index"]), "children_in_parent": int(claim["children_in_parent"]),
                    "char_start": start, "char_end": end, "claim_text": claim["text"],
                    "hypothesis": hypothesis, "question": question,
                    "evidence": evidence, "premise": premise,
                    "input_pair_sha256": digest([premise, hypothesis]),
                    "input_token_length": pair_length, "premise_token_length": premise_length,
                    "hypothesis_token_length": hypothesis_length,
                    "lexical_bpe_indices": info["lexical_bpes"], "all_overlapping_bpe_indices": all_bpes,
                    "risk_bpe_indices": info["risk_bpes"], "mapped_eligible_window_starts": mapped_windows,
                    "gold_label": info["gold_label"], "risk_bpe_fraction": info["risk_fraction"],
                    "partially_positive": bool(info["gold_label"] and info["risk_fraction"] < 1.0),
                    "overlapping_gold_spans": overlaps,
                    "hierarchy_evaluation_weight": float(hierarchy[index]),
                    "fit_training_weight": float(train_weight[index]) if partition == "fit" else None,
                    "calibration_reused_from": "atomic_microclaim_nli_v1/inputs.jsonl" if partition == "calibration" else None,
                }
                example_handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
                if partition == "fit" and rid in current_fit:
                    old = current_fit[rid]["claims"][local_id]
                    assert (old["microclaim_id"], old["text"], old["hypothesis"], retrieval_signature(old["retrieval"]),
                            old["lexical_token_indices"]) == (
                                claim["microclaim_id"], claim["text"], hypothesis, retrieval_signature(retrieval), info["lexical_bpes"])
                    old_fit_exact += 1

            for start in eligible:
                local_claims = sorted({claim_id for token_index in range(start, start + WINDOW_K)
                                       for claim_id in owners[token_index]})
                assert local_claims
                window_response_index.append(response_index)
                window_token_start.append(start)
                window_label.append(int(any(risk[start:start + WINDOW_K])))
                window_claim_values.extend(example_start + claim_id for claim_id in local_claims)
                window_claim_indptr.append(len(window_claim_values))
            answer_record = {
                "schema_version": "atomic-microclaim-relation-answer-map-v1",
                "response_index": response_index, "partition": partition,
                "response_id": rid, "source_id": source["source_id"], "group_id": source["group_id"],
                "question": question, "question_sha256": digest(question),
                "original_response": text, "answer_sha256": digest(text),
                "token_ids": token["token_ids"], "response_token_offsets": token["response_token_offsets"],
                "lexical_mask": list(map(int, lexical)), "risk_mask": list(map(int, risk)),
                "token_count": len(lexical), "answer_risk": int(any(risk)),
                "example_start": example_start, "example_end": example_start + len(infos),
                "eligible_window_count": len(eligible), "positive_window_count": len(positive),
                "window_array_start": len(window_label) - len(eligible), "window_array_end": len(window_label),
                "passages": passages,
                "calibration_claims_reused": partition == "calibration",
            }
            answer_handle.write(json.dumps(answer_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            response_order.append(rid)
            if (response_index + 1) % 250 == 0:
                print("EXPANDED_RELATION_INPUTS", response_index + 1, len(all_answer_rows), flush=True)
    examples_pending.replace(examples_path); answers_pending.replace(answers_path)
    del tokenizer
    assert old_fit_exact == 9061 - 6  # six old fit punctuation-only atomic records
    assert Counter(window_label) == {
        0: EXPECTED["fit_windows"] + EXPECTED["cal_windows"] - EXPECTED["fit_positive_windows"] - EXPECTED["cal_positive_windows"],
        1: EXPECTED["fit_positive_windows"] + EXPECTED["cal_positive_windows"],
    }
    fit_window_count = sum(1 for index in window_response_index if all_answer_rows[index][0] == "fit")
    fit_positive_window_count = sum(label for index, label in zip(window_response_index, window_label)
                                    if all_answer_rows[index][0] == "fit")
    assert (fit_window_count, fit_positive_window_count) == (EXPECTED["fit_windows"], EXPECTED["fit_positive_windows"])

    arrays_path = OUT / "arrays.npz"
    with arrays_path.with_suffix(".npz.pending").open("wb") as handle:
        np.savez_compressed(
            handle, labels=labels_array, partition=np.asarray([0 if x == "fit" else 1 for x in partitions], dtype=np.int8),
            response_index=response_index_array, group_index=group_index_array, held_fold=folds,
            hierarchy_evaluation_weight=hierarchy, fit_training_weight=train_weight,
            input_token_length=input_lengths, premise_token_length=premise_lengths,
            hypothesis_token_length=hypothesis_lengths, claim_char_start=claim_char_start,
            claim_char_end=claim_char_end, window_response_index=np.asarray(window_response_index, dtype=np.int32),
            window_token_start=np.asarray(window_token_start, dtype=np.int32),
            window_label=np.asarray(window_label, dtype=np.int8),
            window_claim_indptr=np.asarray(window_claim_indptr, dtype=np.int64),
            window_claim_example_index=np.asarray(window_claim_values, dtype=np.int32),
        )
    arrays_path.with_suffix(".npz.pending").replace(arrays_path)

    example_rows = lines(examples_path)
    group_rows = []
    by_group = defaultdict(list)
    for row in example_rows:
        by_group[(row["partition"], row["group_id"])].append(row)
    for (partition, group_id), members in sorted(by_group.items()):
        group_rows.append({
            "partition": partition, "group_id": group_id,
            "group_index": group_number[group_id],
            "held_fold": members[0]["held_fold"],
            "source_ids": sorted({row["source_id"] for row in members}),
            "response_ids": sorted({row["response_id"] for row in members}, key=lambda x: int(x)),
            "answers": len({row["response_id"] for row in members}), "microclaims": len(members),
            "positive_microclaims": sum(row["gold_label"] for row in members),
            "hierarchy_weight_mass": sum(row["hierarchy_evaluation_weight"] for row in members),
            "fit_training_weight_mass": (sum(row["fit_training_weight"] for row in members)
                                         if partition == "fit" else None),
        })
    write_jsonl(OUT / "groups.jsonl", group_rows)

    measured_tokens = 7457373
    measured_seconds = 835.9576083999127
    measured_rate = measured_tokens / measured_seconds
    one_epoch_seconds = input_token_total["fit"] / measured_rate
    estimate = {
        "basis": "Same-host relation_pair_fast_v1 auxiliary phase: 7,457,373 logical tokens / 835.958 s. This is a planning estimate, not a benchmark of the new loader.",
        "reference_logical_tokens_per_second": measured_rate,
        "fit_logical_input_tokens_per_epoch": input_token_total["fit"],
        "single_full_fit_epoch_minutes_point": one_epoch_seconds / 60,
        "single_full_fit_epoch_minutes_range": [one_epoch_seconds / 60 * 0.8, one_epoch_seconds / 60 * 1.5],
        "five_fold_OOF_plus_full_fit_one_epoch_each_minutes_point": one_epoch_seconds / 60 * 5,
        "five_fold_OOF_plus_full_fit_one_epoch_each_minutes_range": [one_epoch_seconds / 60 * 4, one_epoch_seconds / 60 * 7.5],
        "excludes": "validation inference, checkpoint I/O and any hyperparameter variants",
    }
    fit_indices = np.flatnonzero(np.asarray(partitions) == "fit")
    cal_indices = np.flatnonzero(np.asarray(partitions) == "calibration")
    statistics = {
        "status": "CPU_prepared_waiting_for_check", "answers": len(all_answer_rows),
        "answers_by_partition": {"fit": len(fit_rows), "calibration": len(current_cal)},
        "groups_by_partition": {"fit": len({row["group_id"] for row in fit_rows}),
                                "calibration": len({row["group_id"] for row in current_cal})},
        "sources_by_partition": {"fit": len({row["source_id"] for row in fit_rows}),
                                 "calibration": len({row["source_id"] for row in current_cal})},
        "microclaims": len(labels),
        "microclaims_by_partition": dict(Counter(partitions)),
        "positive_microclaims": {"fit": int(labels_array[fit_indices].sum()),
                                  "calibration": int(labels_array[cal_indices].sum())},
        "negative_microclaims": {"fit": int(len(fit_indices) - labels_array[fit_indices].sum()),
                                  "calibration": int(len(cal_indices) - labels_array[cal_indices].sum())},
        "positive_rate": {"fit": float(labels_array[fit_indices].mean()),
                           "calibration": float(labels_array[cal_indices].mean())},
        "partial_positive_microclaims": dict(partial),
        "discarded_nonlexical_atomic_records": {"fit": EXPECTED["fit_atomic"] - EXPECTED["fit_scored"],
                                                "calibration": 1},
        "label_type_claim_overlaps": dict(sorted(label_types.items())),
        "input_token_length": {
            "min": int(input_lengths.min()), "median": float(np.median(input_lengths)),
            "p90": float(np.quantile(input_lengths, .90)), "p95": float(np.quantile(input_lengths, .95)),
            "p99": float(np.quantile(input_lengths, .99)), "max": int(input_lengths.max()),
            "over_512": int((input_lengths > 512).sum()), "over_1024": int((input_lengths > 1024).sum()),
        },
        "logical_input_tokens": dict(input_token_total),
        "windows": {"fit": fit_window_count, "calibration": len(window_label) - fit_window_count},
        "positive_windows": {"fit": fit_positive_window_count,
                             "calibration": int(sum(window_label) - fit_positive_window_count)},
        "window_to_claim_edges": len(window_claim_values),
        "raw_BPEs_preserved": {"fit": sum(row["token_count"] for row in fit_tokens.values()),
                               "calibration": sum(row["token_count"] for row in cal_tokens.values())},
        "fit_old634_exact_reuse_check_microclaims": old_fit_exact,
        "fit_calibration_group_overlap": 0, "fit_calibration_source_overlap": 0,
        "weights": {
            "fit_training_mean": float(train_weight[fit_indices].mean()),
            "fit_training_sum": float(train_weight[fit_indices].sum()),
            "fit_group_mass_min": float(min(sum(train_weight[i] for i, group in enumerate(group_ids) if group == name and partitions[i] == "fit") for name in {group_ids[i] for i in fit_indices})),
            "fit_group_mass_max": float(max(sum(train_weight[i] for i, group in enumerate(group_ids) if group == name and partitions[i] == "fit") for name in {group_ids[i] for i in fit_indices})),
            "calibration_training_nonzero": int(np.count_nonzero(train_weight[cal_indices])),
        },
        "GPU_time_estimate": estimate,
        "input_sha256": {"examples.jsonl": sha(examples_path), "answers.jsonl": sha(answers_path),
                         "groups.jsonl": sha(OUT / "groups.jsonl"),
                         "discarded_microclaims.jsonl": sha(OUT / "discarded_microclaims.jsonl"),
                         "arrays.npz": sha(arrays_path)},
        "source_sha256": snapshot,
        "seconds_CPU_prepare": time.perf_counter() - started,
        "model_loaded": False, "model_trained": False, "GPU_used": False,
        "calibration_claim_geometry_reused": True, "official_test_opened": False,
        "formal_baselines_modified": False,
    }
    save_json(OUT / "statistics.json", statistics)
    save_json(OUT / "source_snapshot.json", {"files_sha256": snapshot, "official_test_opened": False})
    save_json(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_not_checked", "examples": len(labels),
        "fit_examples": len(fit_indices), "calibration_examples": len(cal_indices),
        "core_sha256": {name: sha(OUT / name) for name in
                        ("examples.jsonl", "answers.jsonl", "groups.jsonl", "discarded_microclaims.jsonl",
                         "arrays.npz", "statistics.json", "source_snapshot.json")},
        "model_loaded": False, "GPU_used": False, "official_test_opened": False,
    })
    assert snapshot == source_snapshot()
    print("EXPANDED_MICROCLAIM_RELATION_CPU_PREPARED", len(labels), flush=True)


def load_prepared():
    complete = json.loads((OUT / "preparation_complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "CPU_prepared_not_checked"
    for name, expected in complete["core_sha256"].items():
        assert sha(OUT / name) == expected, name
    for name, expected in json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))["files_sha256"].items():
        assert sha(Path(name)) == expected, name
    return lines(OUT / "examples.jsonl"), lines(OUT / "answers.jsonl"), lines(OUT / "groups.jsonl"), np.load(OUT / "arrays.npz")


def check():
    examples, answers, groups, arrays = load_prepared()
    assert len(examples) == EXPECTED["fit_scored"] + EXPECTED["cal_scored"]
    assert len(answers) == EXPECTED["fit_answers"] + EXPECTED["cal_answers"]
    assert Counter(row["partition"] for row in examples) == {"fit": EXPECTED["fit_scored"], "calibration": EXPECTED["cal_scored"]}
    assert Counter(row["partition"] for row in groups) == {"fit": EXPECTED["fit_groups"], "calibration": EXPECTED["cal_groups"]}
    assert np.array_equal(arrays["labels"], np.asarray([row["gold_label"] for row in examples], dtype=np.int8))
    assert np.array_equal(arrays["input_token_length"], np.asarray([row["input_token_length"] for row in examples], dtype=np.int32))
    assert np.all(arrays["input_token_length"] <= PAIR_LIMIT)
    assert np.count_nonzero(arrays["fit_training_weight"][arrays["partition"] == 1]) == 0
    assert len(arrays["window_label"]) == EXPECTED["fit_windows"] + EXPECTED["cal_windows"]
    assert int(arrays["window_label"].sum()) == EXPECTED["fit_positive_windows"] + EXPECTED["cal_positive_windows"]
    assert len(arrays["window_claim_indptr"]) == len(arrays["window_label"]) + 1
    assert int(arrays["window_claim_indptr"][-1]) == len(arrays["window_claim_example_index"])
    assert all(examples[index]["input_pair_sha256"] == digest([examples[index]["premise"], examples[index]["hypothesis"]])
               for index in (0, EXPECTED["fit_scored"] - 1, len(examples) - 1))
    check_record = {
        "status": "CPU_check_passed_waiting_for_independent_audit",
        "examples": len(examples), "answers": len(answers), "groups": len(groups),
        "fit_positive_microclaims": int(arrays["labels"][arrays["partition"] == 0].sum()),
        "calibration_positive_microclaims": int(arrays["labels"][arrays["partition"] == 1].sum()),
        "windows": len(arrays["window_label"]), "window_claim_edges": len(arrays["window_claim_example_index"]),
        "exact_hashes": True, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    }
    save_json(OUT / "CPU_CHECK.json", check_record)
    stats = json.loads((OUT / "statistics.json").read_text(encoding="utf-8"))
    estimate = stats["GPU_time_estimate"]
    report = [
        "# Expanded atomic-microclaim relation data v2", "",
        "已生成可直接训练的 `question + top evidence + claim` 微主张级数据。每条主张保留字符跨度、原始BPE归属、人工标签、4-BPE窗口回投关系、来源组、五折编号和训练权重。", "",
        "| 划分 | 回答 | 来源组 | 微主张 | 正例 | 负例 | 正例率 | 4-BPE窗口 | 正窗口 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| fit | {EXPECTED['fit_answers']} | {EXPECTED['fit_groups']} | {stats['microclaims_by_partition']['fit']} | {stats['positive_microclaims']['fit']} | {stats['negative_microclaims']['fit']} | {stats['positive_rate']['fit']:.2%} | {stats['windows']['fit']} | {stats['positive_windows']['fit']} |",
        f"| calibration | {EXPECTED['cal_answers']} | {EXPECTED['cal_groups']} | {stats['microclaims_by_partition']['calibration']} | {stats['positive_microclaims']['calibration']} | {stats['negative_microclaims']['calibration']} | {stats['positive_rate']['calibration']:.2%} | {stats['windows']['calibration']} | {stats['positive_windows']['calibration']} |", "",
        f"配对输入长度：中位数 {stats['input_token_length']['median']:.0f}，P95 {stats['input_token_length']['p95']:.0f}，最大 {stats['input_token_length']['max']}；未截断。",
        f"按同机历史吞吐估算，一次完整fit训练约 {estimate['single_full_fit_epoch_minutes_point']:.1f} 分钟；五折OOF加最终全量模型、每份各1轮约 {estimate['five_fold_OOF_plus_full_fit_one_epoch_each_minutes_point']:.1f} 分钟。该数只用于排期。", "",
        "fit 的34,941条原子记录和665,708个原始BPE均已保留并核对；其中22条只有标点、无可监督的词面BPE，单列留档，34,919条进入训练。", "",
        "校准集没有重新切分：精确复用当前 2,267 个 scored microclaims。fit 与 calibration 的 source/group 均无交集。没有读取 official test，没有训练或加载模型，没有使用GPU，也没有修改任何 baseline。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    core_names = (
        "PREFLIGHT.json", "protocol.json", "PLAN.md", "FAILURE_HISTORY.md", "prepare_started.json",
        "examples.jsonl", "answers.jsonl", "groups.jsonl", "discarded_microclaims.jsonl", "arrays.npz", "statistics.json",
        "source_snapshot.json", "preparation_complete.json", "CPU_CHECK.json", "REPORT.md",
    )
    manifest = {
        "status": "frozen_CPU_dataset_waiting_for_independent_audit",
        "version": "atomic-microclaim-relation-expanded-v2",
        "files_sha256": {name: sha(OUT / name) for name in core_names},
        "source_sha256": json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))["files_sha256"],
        "expected": EXPECTED, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    }
    save_json(OUT / "manifest.json", manifest)
    print("EXPANDED_MICROCLAIM_RELATION_CPU_CHECK_PASSED", len(examples), flush=True)


def record_failure(stage, exc):
    if not OUT.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUT / f"FAILURE_{stage}_{stamp}.json"
    path.write_text(json.dumps({"status": "failed", "stage": stage, "error_type": type(exc).__name__,
                                "error": str(exc), "time_utc": utc_now(), "record_retained": True,
                                "GPU_used": False, "official_test_opened": False},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("self-test", "initialize", "prepare", "check"))
    args = parser.parse_args()
    try:
        if args.stage == "self-test":
            print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        elif args.stage == "initialize":
            initialize()
        elif args.stage == "prepare":
            prepare()
        else:
            check()
    except Exception as exc:
        record_failure(args.stage, exc)
        raise


if __name__ == "__main__":
    main()
