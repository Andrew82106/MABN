"""CPU-only sanitizer for the forced-evidence quote probe.

This program materializes the sole GPU-visible input.  It never opens a
calibration/test path, never copies answer text or annotation fields, and never
loads model weights or initializes CUDA.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT / "research/forced_evidence_quote_probe_v1"
OUT = ROOT / "results/forced_evidence_quote_probe_v1"
PLAN_PATH = RESEARCH / "PLAN.json"
PROTOCOL_PATH = RESEARCH / "PROTOCOL.md"
REVIEW_PATH = RESEARCH / "INDEPENDENT_PROTOCOL_REVIEW.json"
FIT_PATH = ROOT / "fit_expansion/data/fit.jsonl"
CLAIM_PATH = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
CLAIM_CODE = ROOT / "src/prepare_atomic_microclaim_relation_expanded_v4.py"
BM25_CODE = ROOT / "src/run_retrieved_evidence_nli_v1.py"
MODEL = ROOT.parent / "models/Llama-2-7b-chat-hf"

SALT = "forced_evidence_quote_probe_v1"
MODEL_VALUE = "llama-2-7b-chat"
N_FIT = 3_680
N_GROUPS = 615
N_NATIVE = 634
N_SELECTED = 256
N_CLAIM_ROWS = 34_941
N_SELECTED_RAW_CLAIMS = 3_777
N_CLAIMS = 3_776
MAX_NEW_TOKENS = 160
EXPECTED_RESPONSE_DIGEST = "bf1fd595bc5bc1b6700d54404023f42cc230b4406e0f12c6539d1931dc4691e3"
EXPECTED_GROUP_DIGEST = "e24c27298c7604208bba9d6cbbb25d3e7aba4ff802d73446244c2c911ffb1f27"
EXPECTED_CLAIM_PROMPT_DIGEST = "a59b8b526cc93d17ab07c0f7ae2e9cf8151c46c16728ed27cb91d517e45f4247"

FIT_READ_ALLOWLIST = frozenset((
    "response_id", "source_id", "group_id", "partition", "official_split",
    "model", "question", "retrieved_passages",
))
OUTPUT_FIELDS = (
    "response_id", "source_id", "group_id", "model", "question",
    "passage_1", "passage_2", "passage_3", "microclaim_id",
    "microclaim_index", "claim_start", "claim_end", "claim_text_raw",
    "claim_prompt_text",
)
FORBIDDEN = frozenset((
    "original_response", "quality", "labels", "gold_label", "risk_mask",
    "risk_bpe_indices", "risk_bpe_fraction", "label_type", "answer_risk",
    "existing_scores",
))
CONTROL_MARKERS = ("<quote>", "</quote>", "[/INST]")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json_new(path: Path, value) -> None:
    assert not path.exists(), f"Refuse overwrite: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def write_jsonl_new(path: Path, rows) -> None:
    assert not path.exists(), f"Refuse overwrite: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    pending.replace(path)


def selection_hash(kind: str, value: str) -> str:
    return digest(f"{SALT}\0{kind}\0{value}")


def id_set_digest(values) -> str:
    return digest("".join(f"{value}\n" for value in sorted(values)))


def fold_for(group_id: str) -> int:
    raw = hashlib.sha256(f"{SALT}\0fold\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % 5


def recursively_reject_forbidden(value) -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN & set(value)
        assert not overlap, f"Forbidden field in sanitized value: {sorted(overlap)}"
        for child in value.values():
            recursively_reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            recursively_reject_forbidden(child)


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "sum": int(array.sum()), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()), "min": int(array.min()),
    }


def render_prompt(template: str, row: dict) -> str:
    return template.format(claim=row["claim_prompt_text"],
                           p1=row["passage_1"], p2=row["passage_2"],
                           p3=row["passage_3"])


def verify_protocol_locks(plan: dict) -> dict:
    assert not torch.cuda.is_initialized()
    review = load_json(REVIEW_PATH)
    assert review["execution_decision"]["cpu_manifest_implementation_may_proceed"] is True
    assert review["execution_decision"]["gpu_smoke_allowed_now"] is False
    assert review["execution_decision"]["scoring_allowed_now"] is False
    locks = plan["input_locks"]
    paths = {
        "fit": FIT_PATH,
        "claims": CLAIM_PATH,
        "claim_code": CLAIM_CODE,
        "bm25_code": BM25_CODE,
    }
    expected = {
        "fit": locks["fit_jsonl"]["sha256"],
        "claims": locks["label_blind_microclaims"]["sha256"],
        "claim_code": locks["claim_prompt_function"]["file_sha256"],
        "bm25_code": locks["label_blind_sentence_bm25_code"]["sha256"],
    }
    for name, path in paths.items():
        assert sha(path) == expected[name], (name, path)
    assert plan["scope"]["selected_answers"] == N_SELECTED
    assert plan["scope"]["selected_groups"] == N_SELECTED
    assert plan["scope"]["selected_atomic_claims"] == N_CLAIMS
    assert plan["sanitized_gpu_manifest"]["allowed_fields"] == list(OUTPUT_FIELDS)
    assert set(plan["sanitized_gpu_manifest"]["forbidden_fields"]) == FORBIDDEN
    assert plan["sanitized_gpu_manifest"]["required_model_value"] == MODEL_VALUE
    assert plan["quote_prompt"]["sha256"] == digest(plan["quote_prompt"]["utf8_template"])
    assert plan["decoding"]["max_new_tokens_including_close"] == MAX_NEW_TOKENS
    return {name: {"path": str(path.resolve()), "sha256": expected[name]}
            for name, path in paths.items()}


def read_and_select_fit(bm25_module):
    rows, by_group = [], defaultdict(list)
    with FIT_PATH.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            visible = {key: raw[key] for key in FIT_READ_ALLOWLIST}
            del raw
            assert visible["partition"] == "fit"
            assert visible["official_split"] == "train"
            assert all(not marker.lower() in str(FIT_PATH).lower()
                       for marker in ("calibration", "official_test", "test150"))
            rows.append(visible)
            if visible["model"] == MODEL_VALUE:
                by_group[visible["group_id"]].append(visible)
    assert len(rows) == N_FIT and len({row["group_id"] for row in rows}) == N_GROUPS
    native = sum(map(len, by_group.values()))
    assert native == N_NATIVE and len(by_group) == N_GROUPS
    native_hist = Counter(map(len, by_group.values()))
    selected_groups = sorted(by_group, key=lambda group_id:
        (selection_hash("group", group_id), group_id))[:N_SELECTED]
    selected = []
    for group_id in selected_groups:
        candidates = sorted(by_group[group_id], key=lambda row:
            (selection_hash("answer", row["response_id"]), row["response_id"]))
        selected.append(candidates[0])
    assert len({row["response_id"] for row in selected}) == N_SELECTED
    assert all(row["model"] == MODEL_VALUE for row in selected)
    assert id_set_digest(row["response_id"] for row in selected) == EXPECTED_RESPONSE_DIGEST
    assert id_set_digest(row["group_id"] for row in selected) == EXPECTED_GROUP_DIGEST
    materialized = {}
    for row in selected:
        passages = bm25_module.parse_passages(row["retrieved_passages"])
        assert [passage["passage_id"] for passage in passages] == [1, 2, 3]
        materialized[row["response_id"]] = {
            "response_id": row["response_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "model": row["model"],
            "question": row["question"],
            "passage_1": row["retrieved_passages"][passages[0]["body_char_start"]:passages[0]["body_char_end"]],
            "passage_2": row["retrieved_passages"][passages[1]["body_char_start"]:passages[1]["body_char_end"]],
            "passage_3": row["retrieved_passages"][passages[2]["body_char_start"]:passages[2]["body_char_end"]],
            "passage_rows": passages,
        }
    return materialized, selected_groups, native_hist


def best_mechanical_quote(claim_text: str, answer: dict, bm25_module):
    candidates = []
    for passage in answer["passage_rows"]:
        ranked = bm25_module.bm25_rank(claim_text, passage["sentences"])
        top = ranked[0]
        sentence = passage["sentences"][top["sentence_id"]]
        candidates.append({
            "passage_id": passage["passage_id"],
            "sentence_id": sentence["sentence_id"], "text": sentence["text"],
            "text_sha256": sentence["text_sha256"], "bm25": float(top["bm25"]),
        })
    return min(candidates, key=lambda item:
        (-item["bm25"], item["passage_id"], item["sentence_id"], item["text_sha256"]))


def build_rows(plan, answers, claim_module, bm25_module, tokenizer):
    selected_ids = set(answers)
    raw_selected = 0
    excluded = []
    rows = []
    with CLAIM_PATH.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            claim = json.loads(line)
            assert claim["partition"] == "fit"
            recursively_reject_forbidden(claim)
            if claim["response_id"] not in selected_ids:
                continue
            raw_selected += 1
            if not any(char.isalnum() for char in claim["text"]):
                excluded.append({"response_id": claim["response_id"],
                                 "microclaim_index": claim["microclaim_index"],
                                 "text": claim["text"]})
                continue
            answer = answers[claim["response_id"]]
            assert (claim["source_id"], claim["group_id"]) == (
                answer["source_id"], answer["group_id"])
            prompt_claim = claim_module.hypothesis_for(claim)
            assert prompt_claim and any(char.isalnum() for char in prompt_claim)
            row = {
                "response_id": answer["response_id"],
                "source_id": answer["source_id"], "group_id": answer["group_id"],
                "model": answer["model"], "question": answer["question"],
                "passage_1": answer["passage_1"], "passage_2": answer["passage_2"],
                "passage_3": answer["passage_3"],
                "microclaim_id": claim["microclaim_id"],
                "microclaim_index": int(claim["microclaim_index"]),
                "claim_start": int(claim["start"]), "claim_end": int(claim["end"]),
                "claim_text_raw": claim["text"], "claim_prompt_text": prompt_claim,
            }
            assert tuple(row) == OUTPUT_FIELDS
            recursively_reject_forbidden(row)
            rows.append(row)
    assert raw_selected == N_SELECTED_RAW_CLAIMS
    assert excluded == [{"response_id": "14757", "microclaim_index": 1, "text": '\":'}]
    assert len(rows) == N_CLAIMS
    rows.sort(key=lambda row: (row["response_id"], row["microclaim_index"]))
    prompt_claim_serial = "".join(
        f"{row['microclaim_id']}\t{row['claim_prompt_text']}\n" for row in rows)
    assert digest(prompt_claim_serial) == EXPECTED_CLAIM_PROMPT_DIGEST

    template = plan["quote_prompt"]["utf8_template"]
    prompt_lengths, claim_lengths, quote_lengths = [], [], []
    prompt_fingerprints, quote_fingerprints = [], []
    marker_occurrences = 0
    per_answer = Counter()
    for row in rows:
        prompt = render_prompt(template, row)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        claim_ids = tokenizer.encode(row["claim_prompt_text"], add_special_tokens=False)
        mechanical = best_mechanical_quote(row["claim_prompt_text"], answers[row["response_id"]], bm25_module)
        quote_ids = tokenizer.encode(mechanical["text"] + "</quote>", add_special_tokens=False)
        prompt_lengths.append(len(prompt_ids)); claim_lengths.append(len(claim_ids))
        quote_lengths.append(len(quote_ids)); per_answer[row["response_id"]] += 1
        prompt_fingerprints.append(
            f"{row['microclaim_id']}\t{digest(prompt)}\t{len(prompt_ids)}\t{digest(prompt_ids)}\n")
        quote_fingerprints.append(
            f"{row['microclaim_id']}\t{mechanical['passage_id']}\t{mechanical['sentence_id']}\t"
            f"{mechanical['text_sha256']}\t{mechanical['bm25']:.17g}\t{len(quote_ids)}\n")
        marker_occurrences += sum(text.count(marker) for text in (
            row["claim_prompt_text"], row["passage_1"], row["passage_2"], row["passage_3"])
            for marker in CONTROL_MARKERS)
    assert marker_occurrences == 0
    assert max(prompt_lengths) + MAX_NEW_TOKENS < int(plan["model"]["max_positions"])
    assert max(quote_lengths) <= MAX_NEW_TOKENS
    return rows, excluded, {
        "claims_per_answer": quantiles(list(per_answer.values())),
        "prompt_input_tokens": quantiles(prompt_lengths),
        "claim_tokens": quantiles(claim_lengths),
        "mechanical_quote_tokens_including_close": quantiles(quote_lengths),
        "prompt_rows_sha256": digest("".join(prompt_fingerprints)),
        "mechanical_quote_rows_sha256": digest("".join(quote_fingerprints)),
        "claim_prompt_rows_sha256": digest(prompt_claim_serial),
        "control_marker_occurrences": marker_occurrences,
    }


def verify_measured(plan, stats):
    frozen = plan["measured_cpu_inventory"]
    for section in ("claims_per_answer", "prompt_input_tokens", "claim_tokens",
                    "mechanical_quote_tokens_including_close"):
        for key, value in frozen[section].items():
            if key == "over_160":
                assert sum(1 for _ in ()) == value  # fixed zero checked by max <= 160
                continue
            actual = stats[section][key]
            assert np.isclose(actual, value, rtol=0, atol=1e-12), (section, key, actual, value)


def prepare():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "MANIFEST.json").exists()
    plan = load_json(PLAN_PATH)
    locked_inputs = verify_protocol_locks(plan)
    claim_module = load_module("forced_quote_claim_prompt_v1", CLAIM_CODE)
    bm25_module = load_module("forced_quote_bm25_v1", BM25_CODE)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    assert tokenizer.is_fast
    assert sha(MODEL / "tokenizer.json") == plan["model"]["tokenizer_json_sha256"]
    answers, selected_groups, native_hist = read_and_select_fit(bm25_module)
    rows, excluded, stats = build_rows(
        plan, answers, claim_module, bm25_module, tokenizer)
    verify_measured(plan, stats)

    input_path = OUT / "label_free_inputs.jsonl"
    write_jsonl_new(input_path, rows)
    resource = {
        "status": "CPU_estimate_only_no_GPU_measurement",
        "measured_label_free_inventory": stats,
        "frozen_planning_estimate": plan["resources"],
        "model_weight_download_required": False,
        "GPU_peak_memory": "N/A until an authorized smoke",
        "GPU_used": False,
    }
    write_json_new(OUT / "RESOURCE_ESTIMATE.json", resource)
    preparation = {
        "status": "label_free_GPU_input_materialized_CPU_only",
        "counts": {
            "fit_rows_allowlist_projected": N_FIT,
            "native_answers": N_NATIVE, "native_groups": N_GROUPS,
            "selected_answers": len(answers), "selected_groups": len(selected_groups),
            "selected_claim_rows_before_filter": N_SELECTED_RAW_CLAIMS,
            "selected_claims": len(rows), "excluded_non_alphanumeric": excluded,
            "fold_answer_counts": {str(key): value for key, value in sorted(
                Counter(fold_for(answer["group_id"]) for answer in answers.values()).items())},
            "native_answers_per_group_histogram": {
                str(key): value for key, value in sorted(native_hist.items())},
        },
        "digests": {
            "selected_response_ids": id_set_digest(answers),
            "selected_group_ids": id_set_digest(selected_groups),
            "claim_prompt_rows": stats["claim_prompt_rows_sha256"],
            "rendered_prompt_rows": stats["prompt_rows_sha256"],
            "mechanical_quote_rows": stats["mechanical_quote_rows_sha256"],
        },
        "schema": {
            "allowed_fields_exact_order": list(OUTPUT_FIELDS),
            "forbidden_fields": sorted(FORBIDDEN),
            "all_rows_exact_schema": True,
            "all_rows_native_model": True,
            "answer_text_present": False,
            "gold_or_label_payload_present": False,
        },
        "scope": {
            "files_read": [str(path.resolve()) for path in (
                PLAN_PATH, PROTOCOL_PATH, REVIEW_PATH, FIT_PATH, CLAIM_PATH,
                CLAIM_CODE, BM25_CODE, MODEL / "tokenizer.json")],
            "calibration_read": False, "official_test_read": False,
            "source_JSON_fully_parsed_and_contains_gold_fields": True,
            "gold_fields_referenced_copied_or_used_in_selection_or_features": False,
            "model_weights_loaded": False,
            "GPU_used": False, "scoring_run": False, "baseline_modified": False,
        },
        "source_locks": locked_inputs,
        "protocol_sha256": sha(PROTOCOL_PATH), "plan_sha256": sha(PLAN_PATH),
        "independent_protocol_review_sha256": sha(REVIEW_PATH),
    }
    write_json_new(OUT / "PREPARATION.json", preparation)
    manifest = {
        "status": "CPU_inputs_ready_GPU_and_scoring_blocked",
        "sole_GPU_input": "label_free_inputs.jsonl",
        "files_sha256": {name: sha(OUT / name) for name in (
            "label_free_inputs.jsonl", "RESOURCE_ESTIMATE.json", "PREPARATION.json")},
        "counts": {"groups": N_SELECTED, "answers": N_SELECTED, "claims": N_CLAIMS},
        "calibration_read": False, "official_test_read": False,
        "model_weights_loaded": False, "GPU_used": False, "scoring_run": False,
    }
    write_json_new(OUT / "MANIFEST.json", manifest)
    assert not torch.cuda.is_initialized()
    print("FORCED_QUOTE_LABEL_FREE_PREPARED", N_SELECTED, N_CLAIMS,
          sha(input_path), flush=True)


def verify():
    assert not torch.cuda.is_initialized()
    manifest = load_json(OUT / "MANIFEST.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    seen_answers, seen_groups, count = set(), set(), 0
    with (OUT / "label_free_inputs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); count += 1
            assert tuple(row) == OUTPUT_FIELDS
            recursively_reject_forbidden(row)
            assert row["model"] == MODEL_VALUE
            seen_answers.add(row["response_id"]); seen_groups.add(row["group_id"])
    assert (len(seen_groups), len(seen_answers), count) == (N_SELECTED, N_SELECTED, N_CLAIMS)
    assert id_set_digest(seen_answers) == EXPECTED_RESPONSE_DIGEST
    assert id_set_digest(seen_groups) == EXPECTED_GROUP_DIGEST
    assert not torch.cuda.is_initialized()
    print("FORCED_QUOTE_LABEL_FREE_VERIFIED", count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify}[args.stage]()
