"""Frozen ModernBERT NLI paired stress test on fit-only synthetic silver pairs.

CPU stages prepare and validate exact endpoint requests without loading model
weights or touching calibration/test data.  GPU inference is separately gated.
The endpoint roles are synthetic silver supervision, never human gold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/semantic_conflict_pair_nli_stress_v1"
SOURCE_DIR = ROOT / "research/semantic_conflict_augmentation_v1"
PAIR_SOURCE = SOURCE_DIR / "pairs_fit_silver_strict.jsonl"
QC_SOURCE = SOURCE_DIR / "QUALITATIVE_QC_AUDIT.jsonl"
QC_SUMMARY_SOURCE = SOURCE_DIR / "QUALITATIVE_QC.json"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"

# This is the only reusable NLI source.  Every path explicitly says fit; the
# paired runner never opens the neighboring calibration artifacts.
FIT_CACHE_DIR = ROOT / "results/semantic_source_attribution_v1_score"
FIT_CACHE_MANIFEST = FIT_CACHE_DIR / "selected_nli_manifest_fit.json"
FIT_EXISTING = FIT_CACHE_DIR / "selected_nli_existing_fit.npz"
FIT_MISSING_REQUESTS = FIT_CACHE_DIR / "selected_nli_requests_fit.jsonl"
FIT_MISSING_META = FIT_CACHE_DIR / "selected_nli_missing_scores_fit.json"
FIT_MISSING = FIT_CACHE_DIR / "selected_nli_missing_scores_fit.npz"

VERSION = "semantic-conflict-pair-nli-stress-v1"
MODEL_ID = "tasksource/ModernBERT-base-nli"
MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"
PAIR_SOURCE_SHA256 = "99e317925e3399bc77684aa9706a5538fbdbec4f766502ac6909647eb58985ea"
QC_SOURCE_SHA256 = "fa5d80fca72f1d659f990b27c8f8fe4c850c240d8a9e14273b9bbc4d4966695a"
QC_SUMMARY_SHA256 = "927ee0b93e47f737df049c34ada576be162be39815ad5a1d5c25f7f9db7981ba"
CLASSES = ("entailment", "neutral", "contradiction")
TYPES = ("entity", "number", "negation", "temporal", "attribution")
EXPECTED_BY_TYPE = {
    "entity": 12,
    "number": 291,
    "negation": 1944,
    "temporal": 348,
    "attribution": 1514,
}
EXPECTED_PAIRS = sum(EXPECTED_BY_TYPE.values())
PAIR_TOKEN_LIMIT = 2048
INFERENCE_BATCH = 16
THREADS = 4
SEED = 20260913
MIN_FREE_GPU_BYTES = 3 * 1024**3
PEAK_GPU_BYTES = int(6.5 * 1024**3)
CACHE_REPLAY_ATOL = 1e-5
BACKEND = "cuda_fp32_sdpa_math_batch16_pair_stress_v1"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def json_lines(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in rows)


def frozen_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"Frozen artifact differs: {path}"
        return
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json_text(value), encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    frozen_text(path, json_text(value))


def frozen_jsonl(path: Path, rows: list[dict]) -> None:
    frozen_text(path, jsonl_text(rows))


def frozen_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            assert set(old.files) == set(arrays), f"Frozen NPZ keys differ: {path}"
            for key, value in arrays.items():
                assert np.array_equal(old[key], value, equal_nan=True), (path, key)
        return
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def request_id(premise: str, hypothesis: str) -> str:
    # Exact identity used by the existing fit-only semantic NLI cache.
    return digest({
        "checkpoint": MODEL_ID,
        "revision": MODEL_REVISION,
        "premise": premise,
        "hypothesis": hypothesis,
    })


def load_source_pairs() -> list[dict]:
    assert sha(PAIR_SOURCE) == PAIR_SOURCE_SHA256
    rows = json_lines(PAIR_SOURCE)
    assert len(rows) == EXPECTED_PAIRS
    assert Counter(row["corruption_type"] for row in rows) == Counter(EXPECTED_BY_TYPE)
    assert len({row["pair_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["schema_version"] == "semantic-conflict-augmentation-v1"
        assert row["partition"] == "fit"
        assert row["corruption_type"] in TYPES
        assert row["strict_rule_eligible"] is True
        assert row["strict_exclusion_reasons"] == []
        assert row["labels"] == {
            "supported_claim": "entailed_by_exact_source_sentence",
            "corrupted_claim": "controlled_corruption_silver",
            "label_level": "silver_not_human_gold",
        }
        assert row["nli_premise"] and row["supported_claim"] and row["corrupted_claim"]
        assert row["supported_claim"] != row["corrupted_claim"]
        assert row["source_sentence"] in row["nli_premise"]
        assert row["checks"] and all(value is True for value in row["checks"].values())
        edit = row["edit"]
        start, end = int(edit["start"]), int(edit["end"])
        assert 0 <= start <= end <= len(row["supported_claim"])
        assert row["supported_claim"][start:end] == edit["original"]
        rebuilt = (row["supported_claim"][:start] + edit["replacement"]
                   + row["supported_claim"][end:])
        assert rebuilt == row["corrupted_claim"]
        assert row["split_group_id"] and row["split_component_groups"]
        assert row["group_id"] in row["split_component_groups"]
    return rows


def tokenizer_lengths(pairs: list[tuple[str, str]]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, use_fast=True, trust_remote_code=False)
    lengths = []
    for left in range(0, len(pairs), 256):
        block = pairs[left:left + 256]
        encoded = tokenizer(
            [pair[0] for pair in block], [pair[1] for pair in block],
            add_special_tokens=True, padding=False, truncation=False)
        lengths.extend(map(len, encoded["input_ids"]))
    assert len(lengths) == len(pairs)
    assert lengths and max(lengths) <= PAIR_TOKEN_LIMIT
    return lengths


def build_request_records(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    occurrences: dict[str, dict] = {}
    pair_records = []
    for row in rows:
        endpoint_ids = {}
        for role, field in (("supported", "supported_claim"),
                            ("corrupted", "corrupted_claim")):
            premise, hypothesis = row["nli_premise"], row[field]
            identity = request_id(premise, hypothesis)
            endpoint_ids[role] = identity
            item = occurrences.setdefault(identity, {
                "request_id": identity,
                "premise": premise,
                "hypothesis": hypothesis,
                "occurrence_count": 0,
                "roles": set(),
                "corruption_types": set(),
            })
            assert item["premise"] == premise and item["hypothesis"] == hypothesis
            item["occurrence_count"] += 1
            item["roles"].add(role)
            item["corruption_types"].add(row["corruption_type"])
        pair_records.append({
            "pair_id": row["pair_id"],
            "corruption_type": row["corruption_type"],
            "response_id": row["response_id"],
            "source_id": row["source_id"],
            "group_id": row["group_id"],
            "split_group_id": row["split_group_id"],
            "supported_request_id": endpoint_ids["supported"],
            "corrupted_request_id": endpoint_ids["corrupted"],
        })
    identities = sorted(occurrences)
    pair_text = [(occurrences[key]["premise"], occurrences[key]["hypothesis"])
                 for key in identities]
    lengths = tokenizer_lengths(pair_text)
    requests = []
    for identity, token_length in zip(identities, lengths):
        item = occurrences[identity]
        requests.append({
            "request_id": identity,
            "premise": item["premise"],
            "hypothesis": item["hypothesis"],
            "token_length": token_length,
            "occurrence_count": item["occurrence_count"],
            "roles": sorted(item["roles"]),
            "corruption_types": sorted(item["corruption_types"]),
        })
    length_by_id = {row["request_id"]: row["token_length"] for row in requests}
    for row in pair_records:
        row["supported_token_length"] = length_by_id[row["supported_request_id"]]
        row["corrupted_token_length"] = length_by_id[row["corrupted_request_id"]]
    return requests, pair_records


def checked_fit_cache() -> tuple[dict[str, np.ndarray], set[str], dict]:
    manifest = read_json(FIT_CACHE_MANIFEST)
    missing_meta = read_json(FIT_MISSING_META)
    assert manifest["partition"] == missing_meta["partition"] == "fit"
    assert manifest["existing_npz_sha256"] == sha(FIT_EXISTING)
    assert manifest["requests_sha256"] == sha(FIT_MISSING_REQUESTS)
    assert missing_meta["request_file_sha256"] == sha(FIT_MISSING_REQUESTS)
    assert missing_meta["npz_sha256"] == sha(FIT_MISSING)
    assert missing_meta["checkpoint"] == MODEL_ID
    assert missing_meta["revision"] == MODEL_REVISION
    assert missing_meta["model_sha256"] == MODEL_SHA256
    assert missing_meta["trained"] is False
    assert manifest["formal_baselines_modified"] is False
    assert missing_meta["formal_baselines_modified"] is False
    assert manifest["official_test_opened"] is False
    assert missing_meta["official_test_opened"] is False

    mapping: dict[str, np.ndarray] = {}
    ambiguous: set[str] = set()
    finite_occurrences = 0
    for path in (FIT_EXISTING, FIT_MISSING):
        with np.load(path, allow_pickle=False) as loaded:
            identities = loaded["request_identity_sha256"].reshape(-1, 32)
            probabilities = loaded["probabilities"].reshape(-1, 3)
        assert identities.dtype == np.uint8 and probabilities.dtype == np.float32
        assert len(identities) == len(probabilities)
        for identity_bytes, probability in zip(identities, probabilities):
            if not np.isfinite(probability).all():
                continue
            assert np.allclose(probability.sum(), 1, rtol=0, atol=2e-6)
            finite_occurrences += 1
            identity = bytes(identity_bytes).hex()
            if identity in ambiguous:
                continue
            if identity not in mapping:
                mapping[identity] = probability.copy()
            elif not np.array_equal(mapping[identity], probability):
                mapping.pop(identity)
                ambiguous.add(identity)
    audit = {
        "finite_cache_occurrences": finite_occurrences,
        "unambiguous_unique_requests": len(mapping),
        "ambiguous_unique_requests_excluded": len(ambiguous),
        "reuse_rule": (
            "Exact checkpoint+revision+premise+hypothesis identity; repeated cached "
            "probabilities must be bit-identical. Ambiguous repeats are not reused."
        ),
        "source_sha256": {
            str(FIT_CACHE_MANIFEST.relative_to(ROOT)): sha(FIT_CACHE_MANIFEST),
            str(FIT_EXISTING.relative_to(ROOT)): sha(FIT_EXISTING),
            str(FIT_MISSING_REQUESTS.relative_to(ROOT)): sha(FIT_MISSING_REQUESTS),
            str(FIT_MISSING_META.relative_to(ROOT)): sha(FIT_MISSING_META),
            str(FIT_MISSING.relative_to(ROOT)): sha(FIT_MISSING),
        },
        "calibration_cache_opened": False,
        "official_test_opened": False,
    }
    return mapping, ambiguous, audit


def seed_cache(requests: list[dict], pair_records: list[dict]):
    mapping, ambiguous, audit = checked_fit_cache()
    request_ids, probabilities = [], []
    for row in requests:
        if row["request_id"] in mapping:
            request_ids.append(np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8))
            probabilities.append(mapping[row["request_id"]])
    id_array = (np.stack(request_ids).astype(np.uint8, copy=False) if request_ids
                else np.empty((0, 32), dtype=np.uint8))
    probability_array = (np.stack(probabilities).astype(np.float32, copy=False)
                         if probabilities else np.empty((0, 3), dtype=np.float32))
    hits = {bytes(value).hex() for value in id_array}
    by_type_role = Counter()
    complete_pairs = 0
    for row in pair_records:
        supported = row["supported_request_id"] in hits
        corrupted = row["corrupted_request_id"] in hits
        by_type_role[(row["corruption_type"], "supported", supported)] += 1
        by_type_role[(row["corruption_type"], "corrupted", corrupted)] += 1
        complete_pairs += supported and corrupted
    audit.update({
        "target_unique_requests": len(requests),
        "reusable_target_unique_requests": len(hits),
        "missing_target_unique_requests": len(requests) - len(hits),
        "fully_cached_target_pairs": int(complete_pairs),
        "partial_cache_cannot_produce_pair_metrics": complete_pairs == 0,
        "reusable_endpoint_occurrences_by_type_and_role": {
            kind: {
                role: by_type_role[kind, role, True]
                for role in ("supported", "corrupted")
            } for kind in TYPES
        },
        "ambiguous_target_requests": sum(row["request_id"] in ambiguous for row in requests),
    })
    return id_array, probability_array, audit


def protocol() -> dict:
    return {
        "version": VERSION,
        "status": "frozen_before_gpu_inference",
        "purpose": "Paired sensitivity stress test of an unchanged frozen NLI checkpoint.",
        "scope": {
            "input": "research/semantic_conflict_augmentation_v1/pairs_fit_silver_strict.jsonl",
            "partition": "fit only",
            "pair_count": EXPECTED_PAIRS,
            "calibration_read": False,
            "official_test_read": False,
            "human_gold_read": False,
            "training": False,
            "formal_baseline_modified": False,
        },
        "label_boundary": {
            "supported_role": "Mechanically supported by an exact source sentence/passage identity.",
            "corrupted_role": "Controlled synthetic corruption.",
            "human_gold": False,
            "interpretation": (
                "All pair metrics diagnose frozen-NLI sensitivity on fit-only silver; "
                "they are not factual accuracy, held-out evaluation, or a RAGTruth score."
            ),
        },
        "nli": {
            "checkpoint": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SHA256,
            "class_order": list(CLASSES),
            "pair": "Use each row's exact nli_premise as premise and each endpoint claim as hypothesis.",
            "frozen": True,
            "trained": False,
            "precision": "CUDA FP32; autocast and TF32 disabled",
            "attention": "SDPA math; reference_compile=False",
            "batch": INFERENCE_BATCH,
            "truncation": False,
            "maximum_pair_tokens": PAIR_TOKEN_LIMIT,
            "gpu_gate": "gpu-smoke and extract are separate explicit commands after CPU check.",
        },
        "scores": {
            "contradiction": "P(contradiction), class index 2.",
            "risk": "max(P(contradiction), 1-P(entailment)); under normalized E/N/C this equals 1-P(entailment).",
            "pair_accuracy": "Fraction with corrupted endpoint score strictly greater than its supported endpoint; ties count as incorrect and are reported.",
            "paired_concordance": "Same comparison with a tie worth 0.5.",
            "endpoint_AUROC": "Across the 2N endpoints, corrupted=1 and supported=0; threshold-free and unpaired across endpoint rows.",
            "hard_pair_accuracy": "Both supported argmax=entailment and corrupted argmax=contradiction.",
            "reporting": "Overall and separately for entity, number, negation, temporal, attribution.",
        },
        "qualitative_qc": {
            "source": "Frozen 92-row research-agent review produced before this stress test.",
            "use": "Descriptive stratification only after primary all-4109 metrics are computed.",
            "selection_or_filtering": False,
            "ambiguous_or_reject_removed_from_primary": False,
            "independent_human_gold": False,
        },
        "cache_reuse": {
            "source": "Existing fit-only semantic-attribution NLI cache with the exact same checkpoint/revision.",
            "identity": "SHA256 of checkpoint, revision, exact premise, exact hypothesis.",
            "ambiguity": "Reuse only if every finite repeated cached value is bit-identical.",
            "gpu_smoke_replay_tolerance": CACHE_REPLAY_ATOL,
        },
        "runner_sha256": sha(Path(__file__)),
    }


def runbook_text() -> str:
    return """# Frozen pair NLI stress v1 runbook

The CPU preparation is complete before GPU admission. Run these commands from
`prelab/benchmark_ragtruth_qa` only after the shared GPU is free:

```powershell
& '..\\.venv\\Scripts\\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py gpu-smoke
& '..\\.venv\\Scripts\\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py extract
& '..\\.venv\\Scripts\\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py score
& '..\\.venv\\Scripts\\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py status
```

`gpu-smoke` does not chain into extraction. The score stage is CPU-only and
reports the full strict silver set before the small qualitative-QC strata.
Silver endpoint roles are synthetic and are not human gold.
"""


def prepare() -> dict:
    import torch

    assert not torch.cuda.is_initialized()
    rows = load_source_pairs()
    assert sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert read_json(MODEL / "download_manifest.json")["revision"] == MODEL_REVISION
    requests, pair_records = build_request_records(rows)
    seed_ids, seed_probabilities, cache_audit = seed_cache(requests, pair_records)

    frozen_json(OUT / "protocol.json", protocol())
    frozen_jsonl(OUT / "requests.jsonl", requests)
    frozen_jsonl(OUT / "pair_index.jsonl", pair_records)
    frozen_npz(OUT / "seed_fit_cache.npz",
               request_identity_sha256=seed_ids,
               probabilities=seed_probabilities)
    frozen_json(OUT / "CACHE_REUSE_AUDIT.json", cache_audit)
    frozen_text(OUT / "RUNBOOK.md", runbook_text())

    lengths = np.asarray([row["token_length"] for row in requests])
    preparation = {
        "status": "prepared_waiting_for_cpu_check",
        "pairs": len(rows),
        "endpoint_occurrences": 2 * len(rows),
        "unique_requests": len(requests),
        "deduplicated_endpoint_occurrences": 2 * len(rows) - len(requests),
        "by_type": dict(Counter(row["corruption_type"] for row in rows)),
        "token_lengths": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "over_limit": int((lengths > PAIR_TOKEN_LIMIT).sum()),
        },
        "cache": {
            "reusable_unique_requests": len(seed_ids),
            "missing_unique_requests": len(requests) - len(seed_ids),
            "fully_cached_pairs": cache_audit["fully_cached_target_pairs"],
        },
        "sha256": {
            "source_pairs": sha(PAIR_SOURCE),
            "protocol": sha(OUT / "protocol.json"),
            "requests": sha(OUT / "requests.jsonl"),
            "pair_index": sha(OUT / "pair_index.jsonl"),
            "seed_fit_cache": sha(OUT / "seed_fit_cache.npz"),
            "cache_reuse_audit": sha(OUT / "CACHE_REUSE_AUDIT.json"),
            "runner": sha(Path(__file__)),
            "model": sha(MODEL / "model.safetensors"),
        },
        "model_loaded": False,
        "gpu_used": False,
        "cuda_initialized": False,
        "trained": False,
        "human_gold_read": False,
        "calibration_read": False,
        "official_test_read": False,
        "formal_baseline_modified": False,
        "silver_not_human_gold": True,
    }
    frozen_json(OUT / "PREPARATION.json", preparation)
    assert not torch.cuda.is_initialized()
    print("SEMANTIC_CONFLICT_PAIR_NLI_PREPARED", len(rows), len(requests), len(seed_ids), flush=True)
    return preparation


def load_seed() -> tuple[dict[str, np.ndarray], str]:
    path = OUT / "seed_fit_cache.npz"
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {"request_identity_sha256", "probabilities"}
        ids = loaded["request_identity_sha256"]
        probabilities = loaded["probabilities"]
    assert ids.dtype == np.uint8 and ids.ndim == 2 and ids.shape[1] == 32
    assert probabilities.dtype == np.float32 and probabilities.shape == (len(ids), 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1), 1, rtol=0, atol=2e-6)
    return {bytes(key).hex(): value.copy() for key, value in zip(ids, probabilities)}, sha(path)


def validate_prepared(deep: bool) -> tuple[list[dict], list[dict], list[dict], dict]:
    preparation = read_json(OUT / "PREPARATION.json")
    saved_protocol = read_json(OUT / "protocol.json")
    assert saved_protocol == protocol()
    assert preparation["sha256"]["runner"] == sha(Path(__file__))
    assert preparation["sha256"]["source_pairs"] == sha(PAIR_SOURCE) == PAIR_SOURCE_SHA256
    assert preparation["sha256"]["model"] == sha(MODEL / "model.safetensors") == MODEL_SHA256
    for name, filename in (("protocol", "protocol.json"),
                           ("requests", "requests.jsonl"),
                           ("pair_index", "pair_index.jsonl"),
                           ("seed_fit_cache", "seed_fit_cache.npz"),
                           ("cache_reuse_audit", "CACHE_REUSE_AUDIT.json")):
        assert preparation["sha256"][name] == sha(OUT / filename), name
    rows = load_source_pairs()
    requests = json_lines(OUT / "requests.jsonl")
    pair_records = json_lines(OUT / "pair_index.jsonl")
    assert len(rows) == len(pair_records) == EXPECTED_PAIRS
    assert len(requests) == preparation["unique_requests"]
    assert [row["pair_id"] for row in pair_records] == [row["pair_id"] for row in rows]
    assert len({row["request_id"] for row in requests}) == len(requests)
    assert [row["request_id"] for row in requests] == sorted(row["request_id"] for row in requests)
    for row in requests:
        assert request_id(row["premise"], row["hypothesis"]) == row["request_id"]
        assert 0 < row["token_length"] <= PAIR_TOKEN_LIMIT
    request_ids = {row["request_id"] for row in requests}
    for source, index in zip(rows, pair_records):
        assert request_id(source["nli_premise"], source["supported_claim"]) == index["supported_request_id"]
        assert request_id(source["nli_premise"], source["corrupted_claim"]) == index["corrupted_request_id"]
        assert index["supported_request_id"] in request_ids
        assert index["corrupted_request_id"] in request_ids
    seed, seed_sha = load_seed()
    assert seed_sha == preparation["sha256"]["seed_fit_cache"]
    assert len(seed) == preparation["cache"]["reusable_unique_requests"]
    if deep:
        rebuilt_requests, rebuilt_pairs = build_request_records(rows)
        assert rebuilt_requests == requests
        assert rebuilt_pairs == pair_records
        ids, probabilities, audit = seed_cache(requests, pair_records)
        assert set(seed) == {bytes(value).hex() for value in ids}
        assert all(np.array_equal(seed[bytes(key).hex()], value)
                   for key, value in zip(ids, probabilities))
        assert audit == read_json(OUT / "CACHE_REUSE_AUDIT.json")
    return rows, requests, pair_records, preparation


def risk(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.maximum(probabilities[..., 2], 1.0 - probabilities[..., 0])


def metric_block(supported: np.ndarray, corrupted: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    supported = np.asarray(supported, dtype=np.float64)
    corrupted = np.asarray(corrupted, dtype=np.float64)
    assert supported.shape == corrupted.shape and supported.ndim == 2
    assert supported.shape[1] == corrupted.shape[1] == 3 and len(supported) > 0
    supported_risk, corrupted_risk = risk(supported), risk(corrupted)

    def score_metrics(supported_score, corrupted_score):
        margin = corrupted_score - supported_score
        wins = int((margin > 0).sum())
        ties = int((margin == 0).sum())
        endpoint_y = np.r_[np.zeros(len(margin), dtype=np.int8),
                           np.ones(len(margin), dtype=np.int8)]
        endpoint_score = np.r_[supported_score, corrupted_score]
        return {
            "corrupted_gt_supported_pairs": wins,
            "ties": ties,
            "pair_accuracy": wins / len(margin),
            "paired_concordance": (wins + 0.5 * ties) / len(margin),
            "endpoint_AUROC": float(roc_auc_score(endpoint_y, endpoint_score)),
            "mean_supported": float(np.mean(supported_score)),
            "mean_corrupted": float(np.mean(corrupted_score)),
            "mean_margin_corrupted_minus_supported": float(np.mean(margin)),
            "median_margin_corrupted_minus_supported": float(np.median(margin)),
        }

    return {
        "pairs": len(supported),
        "mean_endpoint_probabilities": {
            "supported": dict(zip(CLASSES, map(float, supported.mean(0)))),
            "corrupted": dict(zip(CLASSES, map(float, corrupted.mean(0)))),
        },
        "contradiction": score_metrics(supported[:, 2], corrupted[:, 2]),
        "risk": score_metrics(supported_risk, corrupted_risk),
        "hard_pair_accuracy": float(np.mean(
            (supported.argmax(1) == 0) & (corrupted.argmax(1) == 2))),
    }


def metric_selfcheck() -> dict:
    supported = np.asarray([
        [.8, .1, .1], [.7, .2, .1], [.4, .2, .4],
    ], dtype=np.float32)
    corrupted = np.asarray([
        [.1, .2, .7], [.7, .2, .1], [.5, .1, .4],
    ], dtype=np.float32)
    result = metric_block(supported, corrupted)
    assert result["contradiction"]["corrupted_gt_supported_pairs"] == 1
    assert result["contradiction"]["ties"] == 2
    assert np.isclose(result["contradiction"]["pair_accuracy"], 1 / 3)
    assert np.isclose(result["contradiction"]["paired_concordance"], 2 / 3)
    # Normalized three-way probabilities make the fixed OR risk equal 1-E.
    assert np.allclose(risk(supported), 1 - supported[:, 0])
    assert np.allclose(risk(corrupted), 1 - corrupted[:, 0])
    return {
        "strict_tie_handling_checked": True,
        "paired_concordance_tie_half_checked": True,
        "risk_formula_checked": True,
        "endpoint_AUROC_checked": True,
    }


def cpu_check() -> dict:
    import torch

    assert not torch.cuda.is_initialized()
    rows, requests, pair_records, preparation = validate_prepared(deep=True)
    check = {
        "status": "passed_ready_for_separately_invoked_gpu_smoke",
        "pairs": len(rows),
        "unique_requests": len(requests),
        "maximum_pair_tokens": max(row["token_length"] for row in requests),
        "cache": {
            "reusable_unique_requests": preparation["cache"]["reusable_unique_requests"],
            "missing_unique_requests": preparation["cache"]["missing_unique_requests"],
            "fully_cached_pairs": preparation["cache"]["fully_cached_pairs"],
            "pair_results_available_from_cache": preparation["cache"]["fully_cached_pairs"] > 0,
        },
        "metric_selfcheck": metric_selfcheck(),
        "all_source_rows_fit": all(row["partition"] == "fit" for row in rows),
        "all_endpoint_roles_silver": all(
            row["labels"]["label_level"] == "silver_not_human_gold" for row in rows),
        "requests_rebuilt_exactly": True,
        "pair_index_rebuilt_exactly": True,
        "cache_audit_rebuilt_exactly": True,
        "pretrained_model_loaded": False,
        "gpu_used": False,
        "cuda_initialized": False,
        "trained": False,
        "human_gold_read": False,
        "calibration_read": False,
        "official_test_read": False,
        "formal_baseline_modified": False,
        "sha256": {
            "runner": sha(Path(__file__)),
            "protocol": sha(OUT / "protocol.json"),
            "preparation": sha(OUT / "PREPARATION.json"),
            "requests": sha(OUT / "requests.jsonl"),
            "pair_index": sha(OUT / "pair_index.jsonl"),
            "seed_fit_cache": sha(OUT / "seed_fit_cache.npz"),
        },
    }
    frozen_json(OUT / "CPU_SELFCHECK.json", check)
    assert not torch.cuda.is_initialized()
    write_status()
    print("SEMANTIC_CONFLICT_PAIR_NLI_CPU_CHECK_PASSED", len(rows), len(requests), flush=True)
    return check


def runtime_signature() -> dict:
    import torch

    return {
        "backend": BACKEND,
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "requests_sha256": sha(OUT / "requests.jsonl"),
        "model_sha256": MODEL_SHA256,
        "revision": MODEL_REVISION,
        "batch": INFERENCE_BATCH,
        "precision": "float32",
        "software": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "numpy", "tokenizers")},
        "cuda_runtime": torch.version.cuda,
    }


def load_cuda():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    assert sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert read_json(MODEL / "download_manifest.json")["revision"] == MODEL_REVISION
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info(0)
    assert free >= MIN_FREE_GPU_BYTES, f"Insufficient free GPU memory: {free}"
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, use_fast=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", reference_compile=False,
        trust_remote_code=False).eval().requires_grad_(False).to("cuda:0")
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert all(parameter.dtype == torch.float32
               and parameter.device.type == "cuda"
               and not parameter.requires_grad for parameter in model.parameters())
    return tokenizer, model, {
        "device": torch.cuda.get_device_name(0),
        "free_before_load": free,
        "total_memory": total,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def infer_pairs(tokenizer, model, pairs: list[tuple[str, str]]):
    import torch

    outputs, shapes = [], []
    device = torch.device("cuda:0")
    for left in range(0, len(pairs), INFERENCE_BATCH):
        batch = pairs[left:left + INFERENCE_BATCH]
        encoded = tokenizer(
            [pair[0] for pair in batch], [pair[1] for pair in batch],
            add_special_tokens=True, padding=True, truncation=False,
            return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= PAIR_TOKEN_LIMIT
        shapes.append([len(batch), int(encoded["input_ids"].shape[1])])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), context:
            logits = model(**encoded).logits
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            outputs.append(logits.softmax(-1).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    assert result.shape == (len(pairs), 3)
    assert np.isfinite(result).all()
    assert np.allclose(result.sum(1), 1, rtol=0, atol=2e-6)
    return result, shapes


def gpu_memory() -> dict:
    import torch

    result = {
        "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(0)),
        "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    assert max(result.values()) <= PEAK_GPU_BYTES, result
    return result


def clean_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_smoke() -> dict:
    check = read_json(OUT / "CPU_SELFCHECK.json")
    assert check["status"] == "passed_ready_for_separately_invoked_gpu_smoke"
    _, requests, _, _ = validate_prepared(deep=False)
    assert not (OUT / "GPU_SMOKE.json").exists()
    seed, _ = load_seed()
    index_by_id = {row["request_id"]: index for index, row in enumerate(requests)}
    selected = [0, len(requests) // 2, len(requests) - 1]
    selected.extend(index_by_id[identity] for identity in sorted(seed))
    selected = sorted(set(selected))
    pairs = [(requests[index]["premise"], requests[index]["hypothesis"])
             for index in selected]
    tokenizer = model = None
    started = time.perf_counter()
    try:
        tokenizer, model, loaded = load_cuda()
        first, shapes = infer_pairs(tokenizer, model, pairs)
        second, shapes_again = infer_pairs(tokenizer, model, pairs)
        assert shapes == shapes_again and np.array_equal(first, second)
        replay_errors = []
        for local, index in enumerate(selected):
            identity = requests[index]["request_id"]
            if identity in seed:
                replay_errors.append(float(np.max(np.abs(first[local] - seed[identity]))))
        maximum_replay_error = max(replay_errors, default=0.0)
        assert maximum_replay_error <= CACHE_REPLAY_ATOL, maximum_replay_error
        frozen_npz(OUT / "GPU_SMOKE_PROBABILITIES.npz",
                   request_indices=np.asarray(selected, dtype=np.int32),
                   probabilities=first)
        result = {
            "status": "passed_no_automatic_extract",
            "requests": len(selected),
            "reused_cache_requests_replayed": len(replay_errors),
            "maximum_cache_replay_abs_error": maximum_replay_error,
            "cache_replay_atol": CACHE_REPLAY_ATOL,
            "same_path_repeat_exact": True,
            "batch_shapes": shapes,
            "model_load": loaded,
            "runtime_signature": runtime_signature(),
            "runtime_signature_sha256": digest(runtime_signature()),
            **gpu_memory(),
            "seconds": time.perf_counter() - started,
            "trained": False,
            "human_gold_read": False,
            "calibration_read": False,
            "official_test_read": False,
            "formal_baseline_modified": False,
            "gpu_used": True,
        }
        frozen_json(OUT / "GPU_SMOKE.json", result)
        write_status()
        print("SEMANTIC_CONFLICT_PAIR_NLI_GPU_SMOKE_PASSED", len(selected), flush=True)
        return result
    finally:
        del model, tokenizer
        clean_gpu()


def validate_scores(requests: list[dict]):
    path = OUT / "scores.npz"
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "request_identity_sha256", "probabilities", "source_code", "token_lengths"}
        ids = loaded["request_identity_sha256"].copy()
        probabilities = loaded["probabilities"].copy()
        source_code = loaded["source_code"].copy()
        token_lengths = loaded["token_lengths"].copy()
    expected_ids = np.stack([
        np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8) for row in requests])
    assert np.array_equal(ids, expected_ids)
    assert probabilities.dtype == np.float32 and probabilities.shape == (len(requests), 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1), 1, rtol=0, atol=2e-6)
    assert source_code.dtype == np.int8 and set(source_code.tolist()) <= {1, 2}
    assert np.array_equal(token_lengths,
                          np.asarray([row["token_length"] for row in requests], dtype=np.int32))
    return probabilities, source_code


def extract() -> dict:
    smoke = read_json(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_no_automatic_extract"
    assert smoke["runtime_signature_sha256"] == digest(runtime_signature())
    _, requests, _, preparation = validate_prepared(deep=False)
    assert not (OUT / "scores.npz").exists()
    seed, _ = load_seed()
    probabilities = np.empty((len(requests), 3), dtype=np.float32)
    source_code = np.zeros(len(requests), dtype=np.int8)
    missing = []
    for index, row in enumerate(requests):
        if row["request_id"] in seed:
            probabilities[index] = seed[row["request_id"]]
            source_code[index] = 1
        else:
            missing.append(index)
    assert len(missing) == preparation["cache"]["missing_unique_requests"]

    tokenizer = model = None
    started = time.perf_counter()
    shape_histogram = Counter()
    try:
        tokenizer, model, loaded = load_cuda()
        for left in range(0, len(missing), 512):
            indices = missing[left:left + 512]
            pairs = [(requests[index]["premise"], requests[index]["hypothesis"])
                     for index in indices]
            values, shapes = infer_pairs(tokenizer, model, pairs)
            probabilities[indices] = values
            source_code[indices] = 2
            shape_histogram.update((batch, length) for batch, length in shapes)
            progress = {
                "status": "running_uncommitted",
                "new_requests_complete": min(left + len(indices), len(missing)),
                "new_requests_total": len(missing),
                "reused_requests": len(seed),
                "pid": os.getpid(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(OUT / "progress.json", progress)
            print("SEMANTIC_CONFLICT_PAIR_NLI_EXTRACT", progress["new_requests_complete"],
                  len(missing), flush=True)
        assert np.all(source_code > 0)
        frozen_npz(
            OUT / "scores.npz",
            request_identity_sha256=np.stack([
                np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8)
                for row in requests]),
            probabilities=probabilities,
            source_code=source_code,
            token_lengths=np.asarray([row["token_length"] for row in requests], dtype=np.int32),
        )
        validate_scores(requests)
        result = {
            "status": "complete_frozen_probabilities_not_scored",
            "unique_requests": len(requests),
            "reused_fit_cache_requests": int((source_code == 1).sum()),
            "new_gpu_requests": int((source_code == 2).sum()),
            "scores_sha256": sha(OUT / "scores.npz"),
            "shape_histogram": {
                f"batch_{batch}_tokens_{length}": count
                for (batch, length), count in sorted(shape_histogram.items())
            },
            "model_load": loaded,
            "runtime_signature": runtime_signature(),
            "runtime_signature_sha256": digest(runtime_signature()),
            **gpu_memory(),
            "seconds": time.perf_counter() - started,
            "trained": False,
            "human_gold_read": False,
            "calibration_read": False,
            "official_test_read": False,
            "formal_baseline_modified": False,
            "gpu_used": True,
            "silver_not_human_gold": True,
        }
        frozen_json(OUT / "extraction_complete.json", result)
        write_status()
        print("SEMANTIC_CONFLICT_PAIR_NLI_EXTRACTION_COMPLETE", len(requests), flush=True)
        return result
    finally:
        del model, tokenizer
        clean_gpu()


def qualitative_qc_description(pair_ids: list[str], supported: np.ndarray,
                               corrupted: np.ndarray) -> dict:
    assert sha(QC_SOURCE) == QC_SOURCE_SHA256
    assert sha(QC_SUMMARY_SOURCE) == QC_SUMMARY_SHA256
    audit = json_lines(QC_SOURCE)
    summary = read_json(QC_SUMMARY_SOURCE)
    assert len(audit) == summary["rows"] == 92
    assert Counter(row["decision"] for row in audit) == Counter(summary["decision_counts"])
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    assert len({row["pair_id"] for row in audit}) == len(audit)
    assert all(row["pair_id"] in pair_index for row in audit)
    by_decision = {}
    for decision in ("clear_silver", "ambiguous", "reject"):
        indices = [pair_index[row["pair_id"]] for row in audit if row["decision"] == decision]
        by_decision[decision] = metric_block(supported[indices], corrupted[indices])
    return {
        "role": "descriptive_only; never filtering, tuning, weighting, or primary evaluation",
        "review_boundary": summary["review_boundary"],
        "independent_human_gold": False,
        "rows": len(audit),
        "decision_counts": summary["decision_counts"],
        "metrics_by_decision_descriptive_only": by_decision,
        "ambiguous_or_reject_removed_from_primary": False,
        "source_sha256": {
            "QUALITATIVE_QC_AUDIT.jsonl": sha(QC_SOURCE),
            "QUALITATIVE_QC.json": sha(QC_SUMMARY_SOURCE),
        },
    }


def report_text(summary: dict) -> str:
    rows = []
    for kind in ("overall", *TYPES):
        item = summary["primary_all_strict_silver"][kind]
        c, r = item["contradiction"], item["risk"]
        rows.append(
            f"| {kind} | {item['pairs']} | {c['pair_accuracy']:.4f} | "
            f"{c['mean_margin_corrupted_minus_supported']:.4f} | {c['endpoint_AUROC']:.4f} | "
            f"{r['pair_accuracy']:.4f} | {r['mean_margin_corrupted_minus_supported']:.4f} | "
            f"{r['endpoint_AUROC']:.4f} | {item['hard_pair_accuracy']:.4f} |")
    qc = summary["qualitative_qc_descriptive_only"]
    return """# Frozen ModernBERT pair stress test v1

These are fit-only synthetic silver sensitivity results, not human-gold factual
accuracy and not held-out RAGTruth evaluation. The checkpoint is unchanged and
no model was trained.

| type | pairs | C pair acc | mean ΔC | C endpoint AUROC | risk pair acc | mean Δrisk | risk endpoint AUROC | hard E→C pair acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

Pair accuracy uses the strict within-pair comparison `corrupted > supported`;
ties count as incorrect and are available in `summary.json`. Risk is
`max(C, 1-E)`, which equals `1-E` for normalized E/N/C probabilities. Endpoint
AUROC instead pools the two endpoint roles and labels corrupted endpoints 1.

The separate 92-row research-agent QC contains {qc['decision_counts']['clear_silver']}
clear silver, {qc['decision_counts']['ambiguous']} ambiguous, and
{qc['decision_counts']['reject']} reject. Its strata are descriptive only;
ambiguous/reject rows were not removed from the primary 4,109-pair result.
"""


def score() -> dict:
    import torch

    assert not torch.cuda.is_initialized()
    rows, requests, pair_records, _ = validate_prepared(deep=False)
    extraction = read_json(OUT / "extraction_complete.json")
    assert extraction["status"] == "complete_frozen_probabilities_not_scored"
    assert extraction["scores_sha256"] == sha(OUT / "scores.npz")
    probabilities, source_code = validate_scores(requests)
    index_by_id = {row["request_id"]: index for index, row in enumerate(requests)}
    supported = np.stack([
        probabilities[index_by_id[row["supported_request_id"]]] for row in pair_records])
    corrupted = np.stack([
        probabilities[index_by_id[row["corrupted_request_id"]]] for row in pair_records])
    pair_ids = [row["pair_id"] for row in pair_records]
    kinds = np.asarray([row["corruption_type"] for row in pair_records])

    # Primary all-strict metrics are complete before the QC file is opened.
    primary = {"overall": metric_block(supported, corrupted)}
    for kind in TYPES:
        selected = np.flatnonzero(kinds == kind)
        primary[kind] = metric_block(supported[selected], corrupted[selected])
    assert primary["overall"]["pairs"] == EXPECTED_PAIRS
    assert {kind: primary[kind]["pairs"] for kind in TYPES} == EXPECTED_BY_TYPE

    qc = qualitative_qc_description(pair_ids, supported, corrupted)
    summary = {
        "status": "complete_fit_only_silver_stress_test",
        "interpretation": (
            "Frozen-NLI paired sensitivity on synthetic silver controlled corruptions; "
            "not human gold, factual accuracy, held-out evaluation, or RAGTruth performance."
        ),
        "primary_all_strict_silver": primary,
        "qualitative_qc_descriptive_only": qc,
        "inference": {
            "checkpoint": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA256,
            "unique_requests": len(requests),
            "endpoint_occurrences": 2 * len(rows),
            "reused_fit_cache_requests": int((source_code == 1).sum()),
            "new_gpu_requests": int((source_code == 2).sum()),
        },
        "trained": False,
        "human_gold_read": False,
        "calibration_read": False,
        "official_test_read": False,
        "formal_baseline_modified": False,
        "silver_not_human_gold": True,
        "sha256": {
            "source_pairs": sha(PAIR_SOURCE),
            "scores": sha(OUT / "scores.npz"),
            "extraction_complete": sha(OUT / "extraction_complete.json"),
            "protocol": sha(OUT / "protocol.json"),
            "runner": sha(Path(__file__)),
        },
    }
    frozen_npz(
        OUT / "pair_scores.npz",
        pair_ids=np.asarray(pair_ids),
        corruption_types=kinds,
        supported_probabilities=supported.astype(np.float32),
        corrupted_probabilities=corrupted.astype(np.float32),
        supported_risk=risk(supported).astype(np.float32),
        corrupted_risk=risk(corrupted).astype(np.float32),
    )
    summary["sha256"]["pair_scores"] = sha(OUT / "pair_scores.npz")
    frozen_json(OUT / "summary.json", summary)
    frozen_text(OUT / "REPORT.md", report_text(summary))
    frozen_json(OUT / "complete.json", {
        "status": summary["status"],
        "summary_sha256": sha(OUT / "summary.json"),
        "report_sha256": sha(OUT / "REPORT.md"),
        "pair_scores_sha256": sha(OUT / "pair_scores.npz"),
        "silver_not_human_gold": True,
        "calibration_read": False,
        "official_test_read": False,
        "formal_baseline_modified": False,
    })
    assert not torch.cuda.is_initialized()
    write_status()
    print("SEMANTIC_CONFLICT_PAIR_NLI_SCORE_COMPLETE", EXPECTED_PAIRS, flush=True)
    return summary


def write_status() -> dict:
    if (OUT / "complete.json").exists():
        state = "complete"
    elif (OUT / "extraction_complete.json").exists():
        state = "ready_for_cpu_score"
    elif (OUT / "GPU_SMOKE.json").exists():
        state = "ready_for_gpu_extract"
    elif (OUT / "CPU_SELFCHECK.json").exists():
        state = "ready_waiting_for_gpu_smoke"
    elif (OUT / "PREPARATION.json").exists():
        state = "prepared_waiting_for_cpu_check"
    else:
        state = "not_prepared"
    preparation = (read_json(OUT / "PREPARATION.json")
                   if (OUT / "PREPARATION.json").exists() else {})
    result = {
        "version": VERSION,
        "status": state,
        "pairs": preparation.get("pairs"),
        "unique_requests": preparation.get("unique_requests"),
        "reusable_fit_cache_requests": preparation.get("cache", {}).get("reusable_unique_requests"),
        "missing_gpu_requests": preparation.get("cache", {}).get("missing_unique_requests"),
        "fully_cached_pairs": preparation.get("cache", {}).get("fully_cached_pairs"),
        "pair_results_available": (OUT / "summary.json").exists(),
        "gpu_started_by_this_runner": (OUT / "GPU_SMOKE.json").exists(),
        "trained": False,
        "human_gold_read": False,
        "calibration_read": False,
        "official_test_read": False,
        "formal_baseline_modified": False,
        "silver_not_human_gold": True,
    }
    atomic_json(OUT / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare", "cpu-check", "gpu-smoke", "extract", "score", "status"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "cpu-check":
        cpu_check()
    elif args.command == "gpu-smoke":
        gpu_smoke()
    elif args.command == "extract":
        extract()
    elif args.command == "score":
        score()
    else:
        print(json.dumps(write_status(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
