"""Label-blind semantic source-attribution extraction for 3,046 extra fit answers.

This runner is intentionally separate from ``run_semantic_source_attribution_v1``
and its cache.  It imports the frozen native Q/K/V attribution implementation,
binds that source and the frozen sentence splitter by SHA-256, and changes only
the input layout: the 3,046 additional fit-only teacher-forced Llama replays.

Stages
------
prepare
    Build exact source-sentence and scored-microclaim token layouts without
    opening answer labels, token-label files, window-label files, calibration,
    or official test data.
cpu-check
    Check every prepared membership geometry and compare the imported hook to
    an explicit tiny-Llama ``output_attentions``/value-projection oracle.
extract
    Explicit, resumable NF4 GPU extraction into an independent output folder.
status
    Validate and report committed cache coverage without loading a model.

No training or scoring command exists here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

VERSION = "semantic-source-attribution-expanded-fit-v1"
OUT = ROOT / "results/semantic_source_attribution_expanded_fit_v1"
PLANS_PATH = ROOT / "fit_expansion/data/new_token_plans.jsonl"
MICROCLAIMS_PATH = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
MICROCLAIMS_COMPLETE_PATH = ROOT / "research/atomic_relation_expanded_fit_v1/complete.json"
FEASIBILITY_PATH = ROOT / "research/semantic_source_attribution_expansion_feasibility_v1/REPORT.json"
NATIVE_RUNNER_PATH = HERE / "run_semantic_source_attribution_v1.py"
SENTENCE_RUNNER_PATH = HERE / "run_retrieved_evidence_nli_v1.py"
GLOBAL_GPU_LOCK = ROOT / "results/.exclusive_gpu_runner.lock"

# These values bind the exact implementations already checked by the native v1
# runner.  Any later code drift must create a new version rather than silently
# changing this expanded cache.
EXPECTED_NATIVE_RUNNER_SHA256 = "a87b4c8bcc232cead05cdaa23deb67642c8551f11870b5fd1a640513e78b977b"
EXPECTED_NATIVE_FORMULA_SHA256 = "3e555bdda24dcdca861bc2a5913cf9d4150357c73858100ff5c3d60cdbf0a4ce"
EXPECTED_SENTENCE_RUNNER_SHA256 = "353129f81e194b87542ec60aa74a6db012cb437108384805ceaa1dd13d1bd64a"
EXPECTED_SENTENCE_FORMULA_SHA256 = "218d3bb57be64fbaf12c0b8fd8e75f076100cc4c80ee3839a187eabb7531791f"
EXPECTED_PLANS_SHA256 = "f22efc2cb7bcee00a9bbb6c5ef0a305b4fde40f92c2622b4fd67a8390734326c"
EXPECTED_MICROCLAIMS_SHA256 = "c1731ab6ca68569f7811db46f594431751995d65d2468cdec51f8d63f37d585a"
EXPECTED_MICROCLAIMS_COMPLETE_SHA256 = "3374bba512d6bdda37828a7f0e0ee66ce67b831e2a67e9f0db9b314d95f5c048"
EXPECTED_FEASIBILITY_SHA256 = "ec5aa6d3decee8ae027d5037dc120479726e7c82da1b1d01b29bdc2d8faafddc"

EXPECTED_ANSWERS = 3046
EXPECTED_RAW_CLAIMS = 25880
EXPECTED_SCORED_CLAIMS = 25864
EXPECTED_NONLEXICAL_CLAIMS = 16
EXPECTED_SOURCE_SENTENCE_INSTANCES = 43867
EXPECTED_UNIQUE_SOURCE_SENTENCES = 9093
EXPECTED_CLAIM_SENTENCE_EDGES = 390683
EXPECTED_LEXICAL_BPE = 420782
EXPECTED_NEW_WINDOWS = 485856
EXPECTED_NEW_EXCLUDED_WINDOWS = 356
EXPECTED_RETAINED_WINDOWS = 168123
EXPECTED_RETAINED_EXCLUDED_WINDOWS = 336
EXPECTED_COMBINED_WINDOWS = 653979
EXPECTED_COMBINED_EXCLUDED_WINDOWS = 692
EXPECTED_MODEL = {
    "model_type": "llama", "layers": 32, "heads": 32,
    "hidden_size": 4096, "head_dim": 128,
}
TOP_K = 15
THREADS = 4
SEED = 20260913
FORBIDDEN_ANNOTATION_KEYS = {
    "label", "labels", "gold", "risk", "risk_mask", "original_labels",
    "hallucination", "hallucination_label", "token_labels",
    "risk_token_indices", "risk_character_spans",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = load_module("semantic_source_attribution_native_v1_frozen", NATIVE_RUNNER_PATH)
sentence_parser = load_module("retrieved_evidence_sentence_parser_v1_frozen", SENTENCE_RUNNER_PATH)


def formula_sha256() -> str:
    source = "\n".join(inspect.getsource(item) for item in (
        native.SemanticAttributionHooks, native.extract_one,
        native.validate_arrays,
    ))
    return digest(source)


def sentence_formula_sha256() -> str:
    source = "\n".join(inspect.getsource(item) for item in (
        sentence_parser.sentence_spans, sentence_parser.parse_passages,
    ))
    return digest(source)


def assert_bindings() -> None:
    assert sha(NATIVE_RUNNER_PATH) == EXPECTED_NATIVE_RUNNER_SHA256
    assert formula_sha256() == EXPECTED_NATIVE_FORMULA_SHA256
    assert sha(SENTENCE_RUNNER_PATH) == EXPECTED_SENTENCE_RUNNER_SHA256
    assert sentence_formula_sha256() == EXPECTED_SENTENCE_FORMULA_SHA256
    assert sha(PLANS_PATH) == EXPECTED_PLANS_SHA256
    assert sha(MICROCLAIMS_PATH) == EXPECTED_MICROCLAIMS_SHA256
    assert sha(MICROCLAIMS_COMPLETE_PATH) == EXPECTED_MICROCLAIMS_COMPLETE_SHA256
    assert sha(FEASIBILITY_PATH) == EXPECTED_FEASIBILITY_SHA256
    assert native.VERSION == "semantic-source-attribution-v1"
    assert native.TOP_K == TOP_K and native.QUERY_BATCH == 8


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, ("Frozen JSON changed", str(path))
    else:
        atomic_json(path, value)


def frozen_text(path: Path, value: str) -> None:
    if path.exists():
        assert path.read_text(encoding="utf-8") == value
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".pending")
        pending.write_text(value, encoding="utf-8")
        pending.replace(path)


def frozen_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    if path.exists():
        assert sha(path) == sha(pending), ("Frozen JSONL changed", str(path))
        pending.unlink()
    else:
        pending.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def frozen_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(arrays)
            assert all(np.array_equal(loaded[key], value)
                       for key, value in arrays.items())
    else:
        atomic_npz(path, arrays)


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "CPU stage must not initialize CUDA"


def reject_annotation_keys(value) -> None:
    if isinstance(value, dict):
        found = FORBIDDEN_ANNOTATION_KEYS & set(value)
        assert not found, f"Annotation-bearing keys are forbidden: {sorted(found)}"
        for child in value.values():
            reject_annotation_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_annotation_keys(child)


def _window_geometry(plan: dict) -> dict:
    answer = plan["original_response"]
    view = plan["original"]
    offsets = [tuple(map(int, pair)) for pair in view["response_token_offsets"]]
    token_ids = list(map(int, view["answer_token_ids"]))
    absolute = list(map(int, view["answer_token_positions"]))
    assert len(offsets) == len(token_ids) == len(absolute) > 0
    lexical = [int(any(char.isalnum() for char in answer[left:right]))
               for left, right in offsets]
    count = len(offsets)
    candidates = [(start, min(start + 4, count))
                  for start in range(max(1, count - 4 + 1))]
    hasher = hashlib.sha256()
    eligible = excluded = 0
    for start, end in candidates:
        local_lexical = [index for index in range(start, end) if lexical[index]]
        eligible += int(bool(local_lexical))
        excluded += int(not local_lexical)
        row = {
            "response_id": plan["response_id"], "k": 4, "stride": 1,
            "token_start": start, "token_end": end,
            "token_indices": list(range(start, end)),
            "answer_token_positions": absolute[start:end],
            "token_ids": token_ids[start:end],
            "character_intervals": offsets[start:end],
            "lexical_token_indices": local_lexical,
            "eligible": bool(local_lexical),
        }
        hasher.update((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("utf-8"))
    return {
        "lexical_mask": lexical,
        "lexical_tokens": sum(lexical),
        "candidate_windows": len(candidates),
        "eligible_windows": eligible,
        "excluded_nonlexical_windows": excluded,
        "geometry_sha256": hasher.hexdigest(),
    }


def _group_new_microclaims(new_ids: set[str]) -> tuple[dict[str, list[dict]], int]:
    grouped = defaultdict(list)
    total_fit_rows = 0
    with MICROCLAIMS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["partition"] == "fit"
            reject_annotation_keys(row)
            total_fit_rows += 1
            if row["response_id"] in new_ids:
                grouped[row["response_id"]].append(row)
    assert total_fit_rows == 34941
    assert set(grouped) == new_ids
    return grouped, total_fit_rows


def _build_one(plan: dict, raw_claims: list[dict]) -> tuple[dict, dict, list[dict], dict]:
    assert plan["partition"] == "fit" and plan["official_split"] == "train"
    assert plan["labels_used"] is False
    assert digest(plan["original_response"]) == plan["answer_sha256"]
    view = plan["original"]
    answer = plan["original_response"]
    offsets = [tuple(map(int, pair)) for pair in view["response_token_offsets"]]
    assert len(offsets) == len(view["answer_token_positions"])
    assert [row["microclaim_index"] for row in raw_claims] == list(range(len(raw_claims)))

    claims = []
    exclusions = []
    inverse = [[] for _ in offsets]
    for raw in raw_claims:
        assert raw["response_id"] == plan["response_id"]
        assert raw["source_id"] == plan["source_id"]
        assert raw["group_id"] == plan["group_id"]
        start, end = int(raw["start"]), int(raw["end"])
        assert 0 <= start < end <= len(answer)
        assert answer[start:end] == raw["text"]
        lexical_indices = []
        for index, (left, right) in enumerate(offsets):
            begin, finish = max(left, start), min(right, end)
            if begin < finish and any(char.isalnum() for char in answer[begin:finish]):
                lexical_indices.append(index)
        if not lexical_indices:
            assert not any(char.isalnum() for char in raw["text"])
            exclusions.append({
                "schema_version": VERSION,
                "response_id": plan["response_id"],
                "source_id": plan["source_id"],
                "group_id": plan["group_id"],
                "partition": "fit",
                "microclaim_id": raw["microclaim_id"],
                "microclaim_index": int(raw["microclaim_index"]),
                "character_range": [start, end],
                "text": raw["text"],
                "text_sha256": digest(raw["text"]),
                "reason": "no_alphanumeric_character_and_no_lexical_bpe",
                "labels_used": False,
            })
            continue
        claim_id = len(claims)
        for index in lexical_indices:
            inverse[index].append(claim_id)
        claims.append({
            "claim_id": claim_id,
            "response_id": plan["response_id"],
            "microclaim_id": raw["microclaim_id"],
            "microclaim_index": int(raw["microclaim_index"]),
            "start": start, "end": end, "text": raw["text"],
            "lexical_token_indices": lexical_indices,
        })
    assert claims

    geometry = _window_geometry(plan)
    assert all((len(inverse[index]) == 1) if is_lexical else
               (len(inverse[index]) == 0)
               for index, is_lexical in enumerate(geometry["lexical_mask"]))
    assert sum(len(row["lexical_token_indices"]) for row in claims) == geometry["lexical_tokens"]

    rendered = native.fq.WRAPPER_LEFT + plan["released_prompt"] + native.fq.WRAPPER_RIGHT + answer
    ref_left, ref_right = map(int, view["rendered_reference_character_range"])
    assert digest(rendered) == view["rendered_text_sha256"]
    document = rendered[ref_left:ref_right]
    passages = sentence_parser.parse_passages(document)
    assert len(passages) == 3
    safe_atomic = {
        "schema_version": VERSION,
        "partition": "fit", "response_id": plan["response_id"],
        "source_id": plan["source_id"], "group_id": plan["group_id"],
        "answer_sha256": plan["answer_sha256"],
        "response_token_offsets_sha256": digest(view["response_token_offsets"]),
        "passages": passages, "claims": claims,
        "lexical_token_microclaims": inverse,
        "labels_used": False, "official_test_opened": False,
    }
    layout = native._prepare_layout(plan, safe_atomic)
    layout["schema_version"] = VERSION
    layout["native_formula_sha256"] = EXPECTED_NATIVE_FORMULA_SHA256
    layout["window_geometry_sha256"] = geometry["geometry_sha256"]
    layout["excluded_nonlexical_microclaims"] = len(exclusions)

    units = {
        "schema_version": VERSION,
        "response_id": plan["response_id"], "source_id": plan["source_id"],
        "group_id": plan["group_id"], "partition": "fit",
        "answer_sha256": plan["answer_sha256"],
        "document_sha256": digest(document),
        "sentences": [{
            "sentence_index": sentence_index,
            "passage_id": passage["passage_id"],
            "sentence_id": sentence["sentence_id"],
            "document_char_start": sentence["document_char_start"],
            "document_char_end": sentence["document_char_end"],
            "text": sentence["text"], "text_sha256": sentence["text_sha256"],
        } for sentence_index, (passage, sentence) in enumerate(
            (pair for passage in passages for pair in
             ((passage, sentence) for sentence in passage["sentences"])))],
        "claims": [{
            "claim_id": row["claim_id"],
            "microclaim_id": row["microclaim_id"],
            "microclaim_index": row["microclaim_index"],
            "start": row["start"], "end": row["end"],
            "text": row["text"], "text_sha256": digest(row["text"]),
        } for row in claims],
        "labels_used": False, "official_test_opened": False,
    }
    reject_annotation_keys({key: value for key, value in units.items()
                            if key not in ("labels_used", "official_test_opened")})
    return layout, units, exclusions, geometry


def _global_csr_index(layouts: list[dict]) -> dict[str, np.ndarray]:
    indptr = [0]
    claim_response_indices, claim_local_ids, claim_microclaim_indices = [], [], []
    edge_response_indices, edge_sentence_indices = [], []
    edge_passage_ids, edge_sentence_ids, edge_sentence_hashes = [], [], []
    for response_index, layout in enumerate(layouts):
        sentences = layout["sentences"]
        for claim in layout["claims"]:
            claim_response_indices.append(response_index)
            claim_local_ids.append(claim["claim_id"])
            claim_microclaim_indices.append(claim["microclaim_index"])
            for sentence in sentences:
                edge_response_indices.append(response_index)
                edge_sentence_indices.append(sentence["sentence_index"])
                edge_passage_ids.append(sentence["passage_id"])
                edge_sentence_ids.append(sentence["sentence_id"])
                edge_sentence_hashes.append(np.frombuffer(
                    bytes.fromhex(sentence["text_sha256"]), dtype=np.uint8))
            indptr.append(indptr[-1] + len(sentences))
    assert len(indptr) == EXPECTED_SCORED_CLAIMS + 1
    assert indptr[-1] == EXPECTED_CLAIM_SENTENCE_EDGES
    result = {
        "claim_indptr": np.asarray(indptr, dtype=np.int64),
        "claim_response_indices": np.asarray(claim_response_indices, dtype=np.int32),
        "claim_local_ids": np.asarray(claim_local_ids, dtype=np.int32),
        "claim_microclaim_indices": np.asarray(claim_microclaim_indices, dtype=np.int32),
        "edge_response_indices": np.asarray(edge_response_indices, dtype=np.int32),
        "edge_sentence_indices": np.asarray(edge_sentence_indices, dtype=np.int32),
        "edge_sentence_passage_ids": np.asarray(edge_passage_ids, dtype=np.int8),
        "edge_sentence_ids": np.asarray(edge_sentence_ids, dtype=np.int32),
        "edge_sentence_identity_sha256": np.stack(edge_sentence_hashes),
        "logical_claim_sentence_mass_shape": np.asarray(
            [EXPECTED_CLAIM_SENTENCE_EDGES,
             EXPECTED_MODEL["layers"] * EXPECTED_MODEL["heads"]], dtype=np.int64),
    }
    assert result["claim_indptr"].shape == (EXPECTED_SCORED_CLAIMS + 1,)
    assert result["edge_sentence_identity_sha256"].shape == (
        EXPECTED_CLAIM_SENTENCE_EDGES, 32)
    return result


PROTOCOL = {
    "version": VERSION,
    "role": "ours expanded fit feature cache; formal baselines and native v1 are untouched",
    "scope": {
        "answers": "Exactly the frozen 3,046 additional fit-only replay plans.",
        "claims": "Exactly 25,864 lexical microclaims; 16 punctuation-only records are deterministically excluded.",
        "calibration": "No calibration path is present or opened.",
        "official_test": "No official-test path is present or opened.",
    },
    "input_allowlist": [
        str(PLANS_PATH.relative_to(ROOT)),
        str(MICROCLAIMS_PATH.relative_to(ROOT)),
        str(MICROCLAIMS_COMPLETE_PATH.relative_to(ROOT)),
        str(FEASIBILITY_PATH.relative_to(ROOT)),
    ],
    "input_denylist": [
        "fit_expansion/data/new_fit.jsonl",
        "fit_expansion/data/tokens_fit.jsonl",
        "fit_expansion/data/windows_k4_fit.jsonl",
        "fit_expansion/data/windows_excluded_fit.jsonl",
        "data/calibration.jsonl", "data/*calibration*", "*test*",
    ],
    "native_formula_binding": {
        "runner_sha256": EXPECTED_NATIVE_RUNNER_SHA256,
        "formula_objects": ["SemanticAttributionHooks", "extract_one", "validate_arrays"],
        "formula_sha256": EXPECTED_NATIVE_FORMULA_SHA256,
        "primitive": "float32 causal softmax attention(q,s) * repeated V-head L2 norm",
        "fragment": "token maximum inside each head",
        "claim": "mean over unique lexical post-token query positions",
        "previous_answer": "strict position < query; current query token excluded",
        "head_readout": "mean over heads; full claim-sentence layer-head CSR retained as float16",
    },
    "sentence_splitter_binding": {
        "runner_sha256": EXPECTED_SENTENCE_RUNNER_SHA256,
        "formula_sha256": EXPECTED_SENTENCE_FORMULA_SHA256,
    },
    "window_geometry": {
        "k": 4, "stride": 1,
        "new_eligible": EXPECTED_NEW_WINDOWS,
        "retained_native_fit_eligible": EXPECTED_RETAINED_WINDOWS,
        "combined_fit_eligible": EXPECTED_COMBINED_WINDOWS,
        "statement": "Each new answer geometry is independently hashed from exact token IDs, absolute positions, offsets, and lexical eligibility. 653,979 is the frozen retained+new arithmetic; label-bearing window files are not opened here.",
    },
    "outputs": {
        "folder": str(OUT.relative_to(ROOT)),
        "layout": "One label-blind exact coordinate row per answer.",
        "semantic_units": "Exact source-sentence and scored-claim text for later frozen NLI scoring.",
        "exclusions": "The 16 punctuation-only microclaims with deterministic reason.",
        "cache": "One independently committed/resumable NPZ per answer plus global CSR index.",
    },
    "model": {
        "checkpoint": native.llama_runner.REPO,
        "revision": native.llama_runner.REVISION,
        "load_config": native.llama_runner.LOAD_CONFIG,
        "teacher_forced_replay": True,
        "limitation": "These are common Llama-2-7B replay representations, not native states of the five original generators.",
    },
    "stage_gate": "prepare -> cpu-check -> separately invoked extract -> status; no training/scoring command",
}


def _protocol_markdown() -> str:
    return """# Expanded-fit semantic source attribution v1\n\nThis runner maps only the 3,046 additional fit answers. It imports the exact frozen native v1 Q/K/V attribution implementation and writes to a separate cache.\n\n- Inputs: frozen replay plans plus label-free expanded microclaims.\n- Excluded: 16 punctuation-only microclaims with no lexical BPE.\n- Geometry: 485,856 new eligible 4-BPE windows; with the retained 168,123 native-fit windows this is 653,979.\n- Data isolation: no answer labels, token/window labels, calibration, or official test data are opened.\n- GPU policy: `prepare`, `cpu-check`, and `status` are CPU-only; `extract` is explicit and resumable.\n- Method: unchanged post-token attention-times-value-norm source attribution from native v1.\n"""


def prepare() -> dict:
    assert_cpu_only()
    assert_bindings()
    plans = json_lines(PLANS_PATH)
    assert len(plans) == EXPECTED_ANSWERS
    assert len({row["response_id"] for row in plans}) == EXPECTED_ANSWERS
    assert all(row["partition"] == "fit" and row["official_split"] == "train"
               and row["labels_used"] is False for row in plans)
    reject_annotation_keys(plans)
    plan_ids = {row["response_id"] for row in plans}
    microclaims, total_fit_microclaims = _group_new_microclaims(plan_ids)

    layouts, units, exclusions = [], [], []
    geometry_hasher = hashlib.sha256()
    totals = Counter()
    unique_sentences = set()
    raw_claim_count = 0
    claim_length_counts = Counter()
    for index, plan in enumerate(plans, 1):
        raw = sorted(microclaims[plan["response_id"]],
                     key=lambda row: row["microclaim_index"])
        raw_claim_count += len(raw)
        layout, one_units, one_exclusions, geometry = _build_one(plan, raw)
        layouts.append(layout)
        units.append(one_units)
        exclusions.extend(one_exclusions)
        totals.update({
            "scored_claims": len(layout["claims"]),
            "source_sentences": len(layout["sentences"]),
            "claim_sentence_edges": layout["claim_sentence_edges"],
            "lexical_bpe": geometry["lexical_tokens"],
            "candidate_windows": geometry["candidate_windows"],
            "eligible_windows": geometry["eligible_windows"],
            "excluded_windows": geometry["excluded_nonlexical_windows"],
            "boundary_crossing_sentence_instances": sum(
                bool(row["boundary_crossing_token_positions"])
                for row in layout["sentences"]),
            "boundary_crossing_bpe_instances": sum(
                len(row["boundary_crossing_token_positions"])
                for row in layout["sentences"]),
        })
        claim_length_counts.update(len(row["lexical_absolute_token_positions"])
                                   for row in layout["claims"])
        for row in layout["sentences"]:
            unique_sentences.add((layout["source_id"], row["passage_id"],
                                  row["sentence_id"], row["text_sha256"]))
        geometry_hasher.update((plan["response_id"] + ":" +
                                geometry["geometry_sha256"] + "\n").encode())
        if index % 250 == 0:
            print("EXPANDED_SEMANTIC_ATTRIBUTION_PREPARED", index,
                  EXPECTED_ANSWERS, flush=True)

    assert total_fit_microclaims == 34941
    assert raw_claim_count == EXPECTED_RAW_CLAIMS
    assert totals["scored_claims"] == EXPECTED_SCORED_CLAIMS
    assert len(exclusions) == EXPECTED_NONLEXICAL_CLAIMS
    assert totals["source_sentences"] == EXPECTED_SOURCE_SENTENCE_INSTANCES
    assert len(unique_sentences) == EXPECTED_UNIQUE_SOURCE_SENTENCES
    assert totals["claim_sentence_edges"] == EXPECTED_CLAIM_SENTENCE_EDGES
    assert totals["lexical_bpe"] == EXPECTED_LEXICAL_BPE
    assert totals["eligible_windows"] == EXPECTED_NEW_WINDOWS
    assert totals["excluded_windows"] == EXPECTED_NEW_EXCLUDED_WINDOWS
    assert EXPECTED_RETAINED_WINDOWS + totals["eligible_windows"] == EXPECTED_COMBINED_WINDOWS
    assert (EXPECTED_RETAINED_EXCLUDED_WINDOWS + totals["excluded_windows"] ==
            EXPECTED_COMBINED_EXCLUDED_WINDOWS)

    OUT.mkdir(parents=True, exist_ok=True)
    layouts_path = OUT / "layouts.jsonl"
    units_path = OUT / "semantic_units.jsonl"
    exclusions_path = OUT / "nonlexical_microclaim_exclusions.jsonl"
    csr_path = OUT / "global_csr_index.npz"
    frozen_jsonl(layouts_path, layouts)
    frozen_jsonl(units_path, units)
    frozen_jsonl(exclusions_path, exclusions)
    frozen_npz(csr_path, _global_csr_index(layouts))
    frozen_json(OUT / "protocol.json", PROTOCOL)
    frozen_text(OUT / "PROTOCOL.md", _protocol_markdown())

    signature = {
        "version": VERSION,
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "protocol_markdown_sha256": sha(OUT / "PROTOCOL.md"),
        "plans_sha256": sha(PLANS_PATH),
        "microclaims_sha256": sha(MICROCLAIMS_PATH),
        "microclaims_complete_sha256": sha(MICROCLAIMS_COMPLETE_PATH),
        "feasibility_report_sha256": sha(FEASIBILITY_PATH),
        "native_runner_sha256": sha(NATIVE_RUNNER_PATH),
        "native_formula_sha256": formula_sha256(),
        "sentence_runner_sha256": sha(SENTENCE_RUNNER_PATH),
        "sentence_formula_sha256": sentence_formula_sha256(),
        "layouts_sha256": sha(layouts_path),
        "semantic_units_sha256": sha(units_path),
        "exclusions_sha256": sha(exclusions_path),
        "global_csr_index_sha256": sha(csr_path),
        "new_window_geometry_sha256": geometry_hasher.hexdigest(),
        "model": native.model_asset_binding(),
        "load_config": native.llama_runner.LOAD_CONFIG,
        "software": native._software_signature(),
        "response_ids_in_order": [row["response_id"] for row in layouts],
        "labels_used": False, "calibration_opened": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / "signature.json", signature)
    report = {
        "status": "prepared_waiting_for_cpu_check",
        "answers": len(layouts), "partition": "fit",
        "raw_microclaims": raw_claim_count,
        "scored_microclaims": totals["scored_claims"],
        "excluded_nonlexical_microclaims": len(exclusions),
        "source_sentence_instances": totals["source_sentences"],
        "unique_source_sentences": len(unique_sentences),
        "claim_sentence_edges": totals["claim_sentence_edges"],
        "lexical_bpe": totals["lexical_bpe"],
        "window_geometry": {
            "new_candidate": totals["candidate_windows"],
            "new_eligible": totals["eligible_windows"],
            "new_excluded_nonlexical": totals["excluded_windows"],
            "retained_eligible": EXPECTED_RETAINED_WINDOWS,
            "combined_eligible": EXPECTED_COMBINED_WINDOWS,
            "combined_excluded_nonlexical": EXPECTED_COMBINED_EXCLUDED_WINDOWS,
            "new_geometry_sha256": signature["new_window_geometry_sha256"],
            "label_bearing_window_files_opened": False,
        },
        "expected_csr_shape": [EXPECTED_CLAIM_SENTENCE_EDGES, 1024],
        "global_claim_indptr_shape": [EXPECTED_SCORED_CLAIMS + 1],
        "expected_csr_float16_bytes": EXPECTED_CLAIM_SENTENCE_EDGES * 1024 * 2,
        "claim_lexical_token_length": {
            "min": min(claim_length_counts), "max": max(claim_length_counts),
            "histogram": {str(key): value for key, value in
                          sorted(claim_length_counts.items())},
        },
        "boundary_crossing_source_sentence_instances":
            totals["boundary_crossing_sentence_instances"],
        "boundary_crossing_bpe_instances": totals["boundary_crossing_bpe_instances"],
        "all_lexical_bpe_owned_exactly_once": True,
        "native_cache_modified": False,
        "gpu_started": False, "model_loaded": False,
        "labels_used": False, "calibration_opened": False,
        "official_test_opened": False,
        "signature_sha256": digest(signature),
    }
    frozen_json(OUT / "PREPARATION.json", report)
    preparation_markdown = f"""# Expanded-fit attribution preparation report\n\nCPU preparation passed for all {EXPECTED_ANSWERS:,} additional fit answers.\n\n- Scored microclaims: {EXPECTED_SCORED_CLAIMS:,}\n- Punctuation-only exclusions: {EXPECTED_NONLEXICAL_CLAIMS}\n- Source-sentence instances: {EXPECTED_SOURCE_SENTENCE_INSTANCES:,}\n- Claim-sentence edges: {EXPECTED_CLAIM_SENTENCE_EDGES:,}\n- New eligible 4-BPE windows: {EXPECTED_NEW_WINDOWS:,}\n- Combined retained + new eligible windows: {EXPECTED_COMBINED_WINDOWS:,}\n\nEvery lexical answer BPE has exactly one scored microclaim owner. The native v1 attribution code and sentence splitter are SHA-bound. No label-bearing token/window file, calibration row, official test row, pretrained model, or GPU was opened by preparation. GPU extraction has not started.\n"""
    frozen_text(OUT / "PREPARATION_REPORT.md", preparation_markdown)
    assert_cpu_only()
    print("EXPANDED_SEMANTIC_ATTRIBUTION_PREPARATION_COMPLETE",
          EXPECTED_SCORED_CLAIMS, EXPECTED_CLAIM_SENTENCE_EDGES, flush=True)
    return report


def load_prepared():
    assert_bindings()
    signature = read_json(OUT / "signature.json")
    preparation = read_json(OUT / "PREPARATION.json")
    assert signature["runner_sha256"] == sha(Path(__file__))
    assert signature["protocol_sha256"] == sha(OUT / "protocol.json")
    assert signature["plans_sha256"] == sha(PLANS_PATH)
    assert signature["microclaims_sha256"] == sha(MICROCLAIMS_PATH)
    assert signature["native_formula_sha256"] == formula_sha256()
    assert signature["layouts_sha256"] == sha(OUT / "layouts.jsonl")
    assert signature["semantic_units_sha256"] == sha(OUT / "semantic_units.jsonl")
    assert signature["exclusions_sha256"] == sha(OUT / "nonlexical_microclaim_exclusions.jsonl")
    assert signature["global_csr_index_sha256"] == sha(OUT / "global_csr_index.npz")
    assert preparation["signature_sha256"] == digest(signature)
    plans = json_lines(PLANS_PATH)
    layouts = json_lines(OUT / "layouts.jsonl")
    assert len(plans) == len(layouts) == EXPECTED_ANSWERS
    assert [row["response_id"] for row in plans] == signature["response_ids_in_order"]
    assert [row["response_id"] for row in layouts] == signature["response_ids_in_order"]
    with np.load(OUT / "global_csr_index.npz", allow_pickle=False) as index:
        assert index["claim_indptr"].shape == (EXPECTED_SCORED_CLAIMS + 1,)
        assert int(index["claim_indptr"][-1]) == EXPECTED_CLAIM_SENTENCE_EDGES
        assert np.array_equal(index["logical_claim_sentence_mass_shape"],
                              [EXPECTED_CLAIM_SENTENCE_EDGES, 1024])
    return plans, layouts, signature, preparation


def cpu_check() -> dict:
    assert_cpu_only()
    plans, layouts, signature, _ = load_prepared()
    assert sum(len(row["claims"]) for row in layouts) == EXPECTED_SCORED_CLAIMS
    assert sum(row["claim_sentence_edges"] for row in layouts) == EXPECTED_CLAIM_SENTENCE_EDGES
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).cpu().eval()

    membership_max_error = 0.0
    claim_lengths = Counter()
    for layout in layouts:
        geometry = native.SemanticAttributionHooks(model, layout, query_batch=2)
        sums = geometry.membership.cpu().numpy().sum(axis=1, dtype=np.float32)
        membership_max_error = max(membership_max_error,
                                   float(np.max(np.abs(sums - 1.0))))
        claim_lengths.update(len(row["lexical_absolute_token_positions"])
                             for row in layout["claims"])
        del geometry
    assert membership_max_error <= 5e-7

    ids = [1] + [2 + ((index * 7) % 120) for index in range(47)]
    answer_positions = list(range(18, 48))
    tiny = {
        "response_id": "tiny", "sequence_length": len(ids),
        "answer_token_positions": answer_positions,
        "source_sentence_union_token_positions": [2, 3, 4, 6, 7],
        "passage_token_positions": [[2, 3], [4], [6, 7]],
        "other_context_token_positions": [0, 1, 5, 8, 9],
        "sentences": [
            {"sentence_index": 0, "passage_id": 1, "sentence_id": 0,
             "token_positions": [2, 3], "text_sha256": "00" * 32},
            {"sentence_index": 1, "passage_id": 2, "sentence_id": 0,
             "token_positions": [4], "text_sha256": "11" * 32},
            {"sentence_index": 2, "passage_id": 3, "sentence_id": 0,
             "token_positions": [6, 7], "text_sha256": "22" * 32},
        ],
        "claims": [
            {"claim_id": 0, "microclaim_index": 0,
             "lexical_absolute_token_positions": [18, 20]},
            {"claim_id": 1, "microclaim_index": 1,
             "lexical_absolute_token_positions": [20, 21, 23]},
            {"claim_id": 2, "microclaim_index": 2,
             "lexical_absolute_token_positions": answer_positions[:26]},
        ],
    }
    input_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        with native.SemanticAttributionHooks(model, tiny, query_batch=2) as hooks:
            hooked_hidden = model.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_attentions=False, output_hidden_states=False,
                return_dict=True).last_hidden_state.detach().clone()
        actual = hooks.finish()
        capture = native._ValueCapture(model)
        try:
            oracle_output = model.model(
                input_ids=input_ids, attention_mask=mask, use_cache=False,
                output_attentions=True, output_hidden_states=False,
                return_dict=True)
        finally:
            capture.close()
    oracle = native._dense_oracle(model, tiny, oracle_output.attentions, capture.values)
    errors = {
        "claim_sentence_mass_float16": float(np.max(np.abs(
            actual["claim_sentence_mass"].astype(np.float32).reshape(3, 3, 2, 4) -
            oracle["claim_sentence"]))),
    }
    for key in ("source_total_relevance", "previous_answer_relevance",
                "other_context_relevance", "passage_relevance",
                "sentence_relevance"):
        errors[key] = float(np.max(np.abs(actual[key] - oracle[key])))
    assert errors["claim_sentence_mass_float16"] <= 0.002
    assert max(value for key, value in errors.items()
               if key != "claim_sentence_mass_float16") <= 2e-7
    arrays = {
        **actual,
        "claim_ids": np.arange(3, dtype=np.int32),
        "microclaim_indices": np.arange(3, dtype=np.int32),
        "sentence_indices": np.arange(3, dtype=np.int32),
        "sentence_passage_ids": np.asarray([1, 2, 3], dtype=np.int8),
        "sentence_ids": np.zeros(3, dtype=np.int32),
        "sentence_identity_sha256": np.stack([
            np.frombuffer(bytes.fromhex(row["text_sha256"]), dtype=np.uint8)
            for row in tiny["sentences"]]),
        "answer_token_positions": np.asarray(answer_positions, dtype=np.int64),
    }
    validation = native.validate_arrays(arrays, tiny, 2, 4)
    with tempfile.TemporaryDirectory(prefix="expanded_semantic_attr_resume_cpu_") as directory:
        orphan_root = Path(directory)
        feature = orphan_root / "features/tiny.npz"
        feature.parent.mkdir(parents=True)
        feature.write_bytes(b"uncommitted")
        moved = _quarantine_paths(orphan_root, "tiny")
        assert len(moved) == 1 and not feature.exists()
    assert torch.equal(hooked_hidden, oracle_output.last_hidden_state)

    result = {
        "status": "passed_ready_for_separately_invoked_gpu_extract",
        "runner_sha256": sha(Path(__file__)),
        "signature_sha256": digest(signature),
        "preparation_sha256": sha(OUT / "PREPARATION.json"),
        "native_runner_sha256": sha(NATIVE_RUNNER_PATH),
        "native_formula_sha256": formula_sha256(),
        "all_real_layout_membership_check": {
            "answers": len(layouts), "claims": sum(claim_lengths.values()),
            "minimum_lexical_tokens": min(claim_lengths),
            "maximum_lexical_tokens": max(claim_lengths),
            "max_float32_row_sum_abs_error": membership_max_error,
        },
        "tiny_model": {"layers": 2, "heads": 4, "kv_heads": 2,
                       "sequence_tokens": len(ids)},
        "oracle": "explicit output_attentions rows times independently captured repeated V-head norms",
        "max_abs_errors": errors, "validation": validation,
        "post_token_query": True,
        "previous_answer_strictly_less_than_query": True,
        "query_self_excluded_from_previous_answer": True,
        "token_sentence_head_tensor_materialized": False,
        "interrupted_uncommitted_record_quarantine_checked": True,
        "pretrained_model_loaded": False, "gpu_used": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "labels_used": False, "calibration_opened": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / "CPU_SELFCHECK.json", result)
    assert_cpu_only()
    print("EXPANDED_SEMANTIC_ATTRIBUTION_CPU_SELFCHECK_PASSED",
          json.dumps(errors), flush=True)
    return result


def record_paths(response_id) -> tuple[Path, Path, Path]:
    feature = OUT / "features" / f"{response_id}.npz"
    return feature, feature.with_suffix(".json"), feature.with_suffix(".commit.json")


def read_committed(plan: dict, layout: dict, signature: dict, load_arrays=False):
    feature, metadata_path, commit_path = record_paths(plan["response_id"])
    present = [path.exists() for path in (feature, metadata_path, commit_path)]
    if not any(present):
        return None
    if not all(present):
        assert not commit_path.exists(), "Commit exists for an incomplete shard"
        return None
    metadata, commit = read_json(metadata_path), read_json(commit_path)
    assert commit["response_id"] == metadata["response_id"] == plan["response_id"]
    assert commit["metadata_sha256"] == sha(metadata_path)
    assert commit["npz_sha256"] == metadata["npz_sha256"] == sha(feature)
    assert metadata["signature_sha256"] == digest(signature)
    assert metadata["plan_sha256"] == digest(plan)
    assert metadata["layout_sha256"] == digest(layout)
    assert metadata["native_formula_sha256"] == EXPECTED_NATIVE_FORMULA_SHA256
    assert metadata["labels_used"] is False
    assert metadata["calibration_opened"] is False
    assert metadata["official_test_opened"] is False
    if not load_arrays:
        return metadata
    with np.load(feature, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    native.validate_arrays(arrays, layout, 32, 32)
    return arrays, metadata


def _quarantine_paths(root: Path, response_id) -> list[str]:
    feature = root / "features" / f"{response_id}.npz"
    metadata = feature.with_suffix(".json")
    commit = feature.with_suffix(".commit.json")
    assert not commit.exists()
    candidates = [feature, metadata,
                  feature.with_suffix(feature.suffix + ".pending"),
                  metadata.with_suffix(metadata.suffix + ".pending"),
                  commit.with_suffix(commit.suffix + ".pending")]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return []
    quarantine = root / "uncommitted" / str(response_id)
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in existing:
        destination = quarantine / f"{path.name}.{sha(path)}"
        if destination.exists():
            assert sha(destination) == sha(path)
            path.unlink()
        else:
            path.replace(destination)
        moved.append(str(destination.relative_to(root)))
    return moved


def quarantine_uncommitted(response_id) -> list[str]:
    return _quarantine_paths(OUT, response_id)


def save_record(plan: dict, layout: dict, signature: dict,
                arrays: dict[str, np.ndarray], seconds: float) -> dict:
    audit = native.validate_arrays(arrays, layout, 32, 32)
    feature, metadata_path, commit_path = record_paths(plan["response_id"])
    assert not any(path.exists() for path in (feature, metadata_path, commit_path))
    atomic_npz(feature, arrays)
    metadata = {
        "status": "complete", "version": VERSION,
        "response_id": plan["response_id"], "source_id": plan["source_id"],
        "group_id": plan["group_id"], "partition": "fit",
        "signature_sha256": digest(signature),
        "plan_sha256": digest(plan), "layout_sha256": digest(layout),
        "native_formula_sha256": EXPECTED_NATIVE_FORMULA_SHA256,
        "npz_sha256": sha(feature), "seconds": seconds,
        **audit, "labels_used": False, "calibration_opened": False,
        "official_test_opened": False,
    }
    atomic_json(metadata_path, metadata)
    commit = {
        "status": "committed", "response_id": plan["response_id"],
        "npz_sha256": metadata["npz_sha256"],
        "metadata_sha256": sha(metadata_path),
    }
    atomic_json(commit_path, commit)
    assert read_committed(plan, layout, signature) == metadata
    return metadata


@contextmanager
def exclusive_gpu():
    descriptor = None
    acquired = False
    try:
        descriptor = os.open(str(GLOBAL_GPU_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        acquired = True
        os.write(descriptor, f"{VERSION}:{os.getpid()}".encode())
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired and GLOBAL_GPU_LOCK.exists():
            GLOBAL_GPU_LOCK.unlink()


def extract(limit: int | None = None) -> dict:
    plans, layouts, signature, _ = load_prepared()
    cpu = read_json(OUT / "CPU_SELFCHECK.json")
    assert cpu["status"] == "passed_ready_for_separately_invoked_gpu_extract"
    assert cpu["runner_sha256"] == sha(Path(__file__))
    assert cpu["signature_sha256"] == digest(signature)
    assert torch.cuda.is_available(), "CUDA GPU is required for NF4 extraction"
    assert shutil.disk_usage(OUT).free > 2 * 1024 ** 3
    native.verify_model_assets(signature)
    existing = [read_committed(plan, layout, signature)
                for plan, layout in zip(plans, layouts)]
    missing = [index for index, value in enumerate(existing) if value is None]
    recovered = {str(plans[index]["response_id"]):
                 quarantine_uncommitted(plans[index]["response_id"])
                 for index in missing}
    recovered = {key: value for key, value in recovered.items() if value}
    if limit is not None:
        assert limit > 0
        missing = missing[:limit]
    model = None
    started = time.perf_counter()
    with exclusive_gpu():
        try:
            if missing:
                model, load_metadata = native.llama_runner.load_nf4()
                spec = native.fq.architecture(model)
                assert spec["layers"] == 32 and spec["heads"] == 32
                assert spec["head_dim"] == 128
                atomic_json(OUT / "extract_started.json", {
                    "status": "running", "version": VERSION,
                    "signature_sha256": digest(signature),
                    "missing_at_start": len(missing), "limit": limit,
                    "quarantined_uncommitted_records": recovered,
                    "model_load": load_metadata,
                    "native_formula_sha256": EXPECTED_NATIVE_FORMULA_SHA256,
                    "labels_used": False, "calibration_opened": False,
                    "official_test_opened": False,
                })
                for completed, index in enumerate(missing, 1):
                    one_started = time.perf_counter()
                    arrays = native.extract_one(model, plans[index], layouts[index])
                    save_record(plans[index], layouts[index], signature, arrays,
                                time.perf_counter() - one_started)
                    atomic_json(OUT / "progress.json", {
                        "status": "running",
                        "completed_this_invocation": completed,
                        "scheduled_this_invocation": len(missing),
                        "records_committed_total": sum(
                            record_paths(row["response_id"])[2].exists()
                            for row in plans),
                        "last_response_id": plans[index]["response_id"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "labels_used": False, "calibration_opened": False,
                        "official_test_opened": False,
                    })
                    print("EXPANDED_SEMANTIC_ATTRIBUTION_EXTRACTED", completed,
                          len(missing), plans[index]["response_id"], flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    committed = [read_committed(plan, layout, signature)
                 for plan, layout in zip(plans, layouts)]
    complete = all(value is not None for value in committed)
    report = {
        "status": "complete" if complete else "partial_resumable",
        "records_complete": sum(value is not None for value in committed),
        "records_total": EXPECTED_ANSWERS,
        "claims_complete": sum(value["claims"] for value in committed if value),
        "claim_sentence_edges_complete": sum(value["edges"] for value in committed if value),
        "signature_sha256": digest(signature),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "native_cache_modified": False,
        "labels_used": False, "calibration_opened": False,
        "official_test_opened": False, "training_or_scoring_run": False,
    }
    atomic_json(OUT / "status.json", report)
    if complete:
        manifest = {
            "status": "complete", "version": VERSION,
            "records_complete": EXPECTED_ANSWERS,
            "claims_complete": EXPECTED_SCORED_CLAIMS,
            "claim_sentence_edges_complete": EXPECTED_CLAIM_SENTENCE_EDGES,
            "logical_claim_sentence_mass_shape": [EXPECTED_CLAIM_SENTENCE_EDGES, 1024],
            "global_claim_indptr_shape": [EXPECTED_SCORED_CLAIMS + 1],
            "global_csr_index_sha256": sha(OUT / "global_csr_index.npz"),
            "signature_sha256": digest(signature),
            "native_formula_sha256": EXPECTED_NATIVE_FORMULA_SHA256,
            "labels_used": False, "calibration_opened": False,
            "official_test_opened": False, "training_or_scoring_run": False,
            "records": committed,
            "files": {plan["response_id"]: {
                "npz_sha256": item["npz_sha256"],
                "metadata_sha256": sha(record_paths(plan["response_id"])[1]),
                "commit_sha256": sha(record_paths(plan["response_id"])[2]),
            } for plan, item in zip(plans, committed)},
        }
        frozen_json(OUT / "feature_manifest.json", manifest)
    print("EXPANDED_SEMANTIC_ATTRIBUTION_STATUS", json.dumps(report), flush=True)
    return report


def status() -> dict:
    assert_cpu_only()
    plans, layouts, signature, preparation = load_prepared()
    cpu_path = OUT / "CPU_SELFCHECK.json"
    cpu_valid = False
    if cpu_path.exists():
        cpu = read_json(cpu_path)
        cpu_valid = (
            cpu.get("status") == "passed_ready_for_separately_invoked_gpu_extract" and
            cpu.get("runner_sha256") == sha(Path(__file__)) and
            cpu.get("signature_sha256") == digest(signature) and
            cpu.get("native_formula_sha256") == EXPECTED_NATIVE_FORMULA_SHA256)
    committed = [read_committed(plan, layout, signature)
                 for plan, layout in zip(plans, layouts)]
    report = {
        "version": VERSION,
        "prepared": preparation["status"] == "prepared_waiting_for_cpu_check",
        "cpu_selfcheck_valid": cpu_valid,
        "records_complete": sum(value is not None for value in committed),
        "records_total": EXPECTED_ANSWERS,
        "claims_complete": sum(value["claims"] for value in committed if value),
        "claim_sentence_edges_complete": sum(value["edges"] for value in committed if value),
        "gpu_started": (OUT / "extract_started.json").exists(),
        "feature_manifest_complete": (OUT / "feature_manifest.json").exists(),
        "native_cache_modified": False,
        "labels_used": False, "calibration_opened": False,
        "official_test_opened": False, "training_or_scoring_run": False,
    }
    atomic_json(OUT / "status.json", report)
    assert_cpu_only()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("cpu-check")
    extraction = sub.add_parser("extract")
    extraction.add_argument("--limit", type=int)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "cpu-check":
        cpu_check()
    elif args.command == "extract":
        extract(args.limit)
    else:
        status()


if __name__ == "__main__":
    main()
