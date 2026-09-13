"""Train and evaluate the semantic source-attribution v1 detector.

The feature extractor is deliberately separate from this runner.  ``prepare``
and ``cpu-check`` are label-free CPU stages.  ``link-nli`` may run only after
the attribution cache exists and deterministically binds the three most
attributed source sentences to the frozen ModernBERT NLI cache.  ``fit`` opens
fit labels and produces source-group OOF predictions; ``evaluate`` is the only
stage allowed to open calibration labels.  No official baseline artifact is
ever written by this file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_development as q  # noqa: E402
import run_atomic_microclaim_nli_v1 as atomic_nli  # noqa: E402
import run_aligned_evidence_head_v1 as aligned  # noqa: E402


VERSION = "semantic-source-attribution-v1-score"
OUT = ROOT / "results/semantic_source_attribution_v1_score"
ATTR_OUT = ROOT / "results/semantic_source_attribution_v1"
ATOMIC_OUT = ROOT / "results/atomic_microclaim_nli_v1"
ROWS_PATH = ATOMIC_OUT / "inputs.jsonl"
PAIR_DIR = ATOMIC_OUT / "pair_scores"
WHITEBOX_PATH = ATOMIC_OUT / "whitebox_features.npy"
LAYOUTS_PATH = ATTR_OUT / "layouts.jsonl"
ATTR_MANIFEST_PATH = ATTR_OUT / "feature_manifest.json"
ATTR_RUNNER_PATH = HERE / "run_semantic_source_attribution_v1.py"
ATTR_SIGNATURE_PATH = ATTR_OUT / "signature.json"
ATTR_GLOBAL_INDEX_PATH = ATTR_OUT / "global_csr_index.npz"

EXPECTED_ATTR_RUNNER_SHA256 = "a87b4c8bcc232cead05cdaa23deb67642c8551f11870b5fd1a640513e78b977b"
EXPECTED_ATTR_SIGNATURE_FILE_SHA256 = "1c74873c6c6bd9882879309aa889a09d9990d9a91b5e888eaf56ac385d15aafa"
EXPECTED_ATTR_GLOBAL_INDEX_SHA256 = "207e892248143c029070668755994f7acf3611f0ea5a97399c3149abe01b9467"

GOLD_FILES = tuple(
    ROOT / "data" / f"{kind}_{partition}.jsonl"
    for partition in ("fit", "calibration")
    for kind in ("answers", "tokens", "windows_k4")
)

EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_CLAIMS = {"fit": 9055, "calibration": 2267}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
EXPECTED_GROUPS = {"fit": 615, "calibration": 154}
LAYERS = 32
HEADS = 32
ATTR_WIDTH = LAYERS * HEADS
TOP_SENTENCES = 3
FOLDS = 5
THREADS = 4
SEED = 20260913

# One fixed regularization strength per component.  There is no C/layer/head
# search, so calibration cannot quietly become a feature selector.
CLEAN_BASE_C = 0.01
SEMANTIC_C = 0.001

RAW_ATTR_NAMES = tuple(
    f"log1p_source_sentence_mass__layer_{layer:02d}__head_{head:02d}"
    for layer in range(LAYERS) for head in range(HEADS)
)
REGION_NAMES = tuple(
    f"{kind}__layer_{layer:02d}"
    for kind in (
        "source_share", "strict_previous_answer_share", "other_context_share",
        "passage_1_within_source_share", "passage_2_within_source_share",
        "passage_3_within_source_share", "top1_sentence_concentration",
        "top3_sentence_concentration", "sentence_relevance_entropy",
    )
    for layer in range(LAYERS)
)
NLI_BLOCK_NAMES = tuple(
    f"{scope}__{name}"
    for scope in ("top1", "top3_relevance_weighted")
    for name in ("entailment", "neutral", "contradiction", *atomic_nli.PAIR_RELATION_NAMES)
)
SELECTION_NAMES = (
    "top1_global_sentence_share", "top3_global_sentence_share",
    "top1_minus_top2_global_share", "log1p_source_sentence_count",
)
SEMANTIC_FEATURE_NAMES = (
    RAW_ATTR_NAMES + REGION_NAMES + NLI_BLOCK_NAMES + SELECTION_NAMES
    + tuple(f"whitebox__{name}" for name in atomic_nli.WHITEBOX_FEATURE_NAMES)
)

assert tuple(aligned.FEATURE_NAMES[-len(aligned.INCUMBENT_NAMES):]) == tuple(aligned.INCUMBENT_NAMES)
CLEAN_BASE_FEATURE_NAMES = tuple(aligned.FEATURE_NAMES[:-len(aligned.INCUMBENT_NAMES)])


class MissingSelectedNLI(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, f"Frozen file changed: {path}"
    else:
        atomic_json(path, value)


def frozen_jsonl(path: Path, rows) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            old = [json.loads(line) for line in handle if line.strip()]
        assert old == rows, f"Frozen file changed: {path}"
        return
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    pending.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def prefix_partition_rows(path: Path, partition: str) -> list[dict]:
    """Read one contiguous partition without scanning later partitions."""
    rows, seen = [], False
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["partition"] == partition:
                seen = True
                rows.append(row)
            elif seen:
                break
    return rows


def all_partition_rows(path: Path, partition: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [row for line in handle if line.strip()
                for row in [json.loads(line)] if row["partition"] == partition]


def protocol() -> dict:
    return {
        "version": VERSION,
        "role": "ours; formal baseline models and artifacts are read-only and unchanged",
        "task_interface": {
            "data": "Frozen native RAGTruth QA: fit 634 answers, calibration 159 answers.",
            "split": "Five-fold source-connected GroupKFold on fit; calibration is opened only by evaluate, once the method is frozen.",
            "labels": "Unchanged human factual-risk spans; any-error and Evident/Subtle Conflict claim heads are trained separately.",
            "localization": "Unchanged eligible 4-original-BPE stride-one windows. A window takes the maximum score of its overlapping microclaims; an answer takes its maximum window.",
            "thresholds": "Fit-OOF thresholds are frozen for strict calibration. Final evaluate may additionally report the common calibration F1-opt diagnostic without changing the model.",
            "official_test": "No official-test path occurs in this runner.",
        },
        "semantic_source_attribution": {
            "primitive": "Frozen extractor attention times per-head V-vector norm at post-token query states.",
            "extractor_binding": {
                "runner_sha256": EXPECTED_ATTR_RUNNER_SHA256,
                "signature_file_sha256": EXPECTED_ATTR_SIGNATURE_FILE_SHA256,
                "global_csr_index_sha256": EXPECTED_ATTR_GLOBAL_INDEX_SHA256,
            },
            "raw_1024": "For every layer/head, sum the float16 claim-to-source-sentence masses, apply log1p, retain all 32x32 coordinates; no layer/head selection.",
            "sentence_selection": "Rank every exact source sentence by the arithmetic mean of all 1024 unrounded attribution coordinates; stable ties use passage/order sentence_index. Keep top 1 and top 3.",
            "compact": "Per-layer source/strict-previous-answer/other shares, three within-source passage shares, top1/top3 sentence concentration, and normalized sentence entropy.",
            "nli": "Top1 and attribution-weighted top3 exact-sentence ModernBERT E/N/C plus same-pair relation features. Reuse exact frozen atomic cache; missing exact pairs must be listed and scored by the same frozen checkpoint before fit.",
            "whitebox": "Reuse the 12 clean fit-group-OOF/full-fit-cal Lookback/large/NLL summaries; no historical calibration-selected incumbent score is an input.",
        },
        "models": {
            "clean_aligned_base": f"Pair-aligned claim/evidence/whitebox features with the six historical-incumbent aggregates removed; StandardScaler plus L2 LR C={CLEAN_BASE_C}.",
            "semantic_head": f"All {len(SEMANTIC_FEATURE_NAMES)} frozen semantic-attribution features; StandardScaler plus L2 LR C={SEMANTIC_C}.",
            "targets": ["any_error", "conflict_only"],
            "weights": "First assign equal source-group, answer-within-group, and claim-within-answer mass. Then apply one exact binary-class balancing step and one global scale only; never re-equalize groups after class balancing.",
            "fusion": "Within each component take max(any-error, conflict-only). A fit-OOF add gate may add semantic-positive windows to the clean-base decisions; no-add is represented by enabled=false and is handled explicitly. Base positives are retained.",
            "selection_freedom": "No tree/model-family, layer, head, C, or feature-subset search. Only fit-OOF decision thresholds are learned.",
        },
        "dimensions": {
            "raw_attribution": len(RAW_ATTR_NAMES),
            "compact_region": len(REGION_NAMES),
            "selected_nli": len(NLI_BLOCK_NAMES),
            "selection": len(SELECTION_NAMES),
            "whitebox": len(atomic_nli.WHITEBOX_FEATURE_NAMES),
            "semantic_total": len(SEMANTIC_FEATURE_NAMES),
            "clean_aligned_base": len(CLEAN_BASE_FEATURE_NAMES),
        },
        "stage_gate": "prepare -> cpu-check -> extractor completion -> link-nli -> fill missing frozen NLI -> fit -> evaluate",
        "calibration_labels_opened_in_prepare": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def individual_pair_indices(row: dict) -> dict[tuple[int, int, int], int]:
    _, owners = atomic_nli.row_pairs(row)
    result = {}
    for pair_index, owner in enumerate(owners):
        claim_id, passage_id, rank, sentence_id, second_sentence_id = owner
        if int(rank) == 0:
            continue
        assert int(second_sentence_id) == -1
        key = (int(claim_id), int(passage_id), int(sentence_id))
        assert key not in result
        result[key] = pair_index
    return result


def build_fit_index(rows: list[dict], layouts: list[dict]) -> tuple[list[dict], dict]:
    indexed, total_edges, cached_edges = [], 0, 0
    for global_answer_index, (row, layout) in enumerate(zip(rows, layouts)):
        assert row["response_id"] == layout["response_id"]
        assert row["partition"] == layout["partition"] == "fit"
        assert row["group_id"] == layout["group_id"]
        assert row["labels_used"] is False and layout["labels_used"] is False
        sentence_lookup = {
            (int(passage["passage_id"]), int(sentence["sentence_id"])): sentence
            for passage in row["passages"] for sentence in passage["sentences"]
        }
        pair_map = individual_pair_indices(row)
        sentences = []
        for sentence in layout["sentences"]:
            key = (int(sentence["passage_id"]), int(sentence["sentence_id"]))
            atomic_sentence = sentence_lookup[key]
            assert atomic_sentence["text_sha256"] == sentence["text_sha256"]
            sentences.append({
                "sentence_index": int(sentence["sentence_index"]),
                "passage_id": key[0], "sentence_id": key[1],
                "text_sha256": sentence["text_sha256"],
            })
        claims = []
        for claim, layout_claim in zip(row["claims"], layout["claims"]):
            assert int(claim["claim_id"]) == int(layout_claim["claim_id"])
            assert claim["microclaim_id"] == layout_claim["microclaim_id"]
            available = []
            for sentence in sentences:
                key = (int(claim["claim_id"]), sentence["passage_id"],
                       sentence["sentence_id"])
                pair_index = pair_map.get(key)
                available.append(pair_index)
                total_edges += 1
                cached_edges += pair_index is not None
            claims.append({
                "claim_id": int(claim["claim_id"]),
                "microclaim_id": claim["microclaim_id"],
                "microclaim_index": int(claim["microclaim_index"]),
                "cached_individual_pair_indices_by_sentence": available,
            })
        assert len(claims) * len(sentences) == int(layout["claim_sentence_edges"])
        cache_path = PAIR_DIR / f"{row['response_id']}.npz"
        assert cache_path.is_file()
        indexed.append({
            "global_answer_index": global_answer_index,
            "response_id": row["response_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "partition": "fit",
            "claims": claims, "sentences": sentences,
            "atomic_pair_cache_sha256": sha(cache_path),
        })
    stats = {
        "answers": len(indexed),
        "groups": len({row["group_id"] for row in indexed}),
        "claims": sum(len(row["claims"]) for row in indexed),
        "claim_sentence_edges": total_edges,
        "cached_individual_nli_edges": cached_edges,
        "uncached_individual_nli_edges": total_edges - cached_edges,
        "cached_individual_nli_fraction": cached_edges / total_edges,
    }
    return indexed, stats


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha(ATTR_RUNNER_PATH) == EXPECTED_ATTR_RUNNER_SHA256
    assert sha(ATTR_SIGNATURE_PATH) == EXPECTED_ATTR_SIGNATURE_FILE_SHA256
    assert sha(ATTR_GLOBAL_INDEX_PATH) == EXPECTED_ATTR_GLOBAL_INDEX_SHA256
    frozen_json(OUT / "protocol.json", protocol())
    rows = prefix_partition_rows(ROWS_PATH, "fit")
    layouts = prefix_partition_rows(LAYOUTS_PATH, "fit")
    assert len(rows) == len(layouts) == EXPECTED_ANSWERS["fit"]
    assert sum(len(row["claims"]) for row in rows) == EXPECTED_CLAIMS["fit"]
    assert len({row["group_id"] for row in rows}) == EXPECTED_GROUPS["fit"]
    index, stats = build_fit_index(rows, layouts)
    assert stats["answers"] == EXPECTED_ANSWERS["fit"]
    assert stats["claims"] == EXPECTED_CLAIMS["fit"]
    assert stats["groups"] == EXPECTED_GROUPS["fit"]
    frozen_jsonl(OUT / "fit_claim_sentence_index.jsonl", index)
    source_snapshot = {
        "runner_sha256": sha(Path(__file__)),
        "run_development_sha256": sha(Path(q.__file__)),
        "atomic_nli_runner_sha256": sha(Path(atomic_nli.__file__)),
        "aligned_evidence_runner_sha256": sha(Path(aligned.__file__)),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "atomic_inputs_sha256": sha(ROWS_PATH),
        "atomic_preparation_sha256": sha(ATOMIC_OUT / "preparation_complete.json"),
        "attribution_layouts_sha256": sha(LAYOUTS_PATH),
        "attribution_runner_sha256": sha(ATTR_RUNNER_PATH),
        "attribution_signature_sha256": sha(ATTR_SIGNATURE_PATH),
        "attribution_global_index_sha256": sha(ATTR_GLOBAL_INDEX_PATH),
        "fit_index_sha256": sha(OUT / "fit_claim_sentence_index.jsonl"),
        "gold_files_sha256_hash_only": {
            str(path.relative_to(ROOT)): sha(path) for path in GOLD_FILES
        },
        "calibration_label_files_opened": False,
        "fit_label_files_opened": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / "source_snapshot.json", source_snapshot)
    frozen_json(OUT / "PREPARATION.json", {
        "status": "fit_metadata_prepared_waiting_for_extractor",
        **stats,
        "semantic_feature_width": len(SEMANTIC_FEATURE_NAMES),
        "clean_base_feature_width": len(CLEAN_BASE_FEATURE_NAMES),
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "calibration_labels_opened": False, "fit_labels_opened": False,
        "model_trained": False, "GPU_used": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("SEMANTIC_ATTRIBUTION_SCORE_PREPARED", stats["answers"],
          stats["claims"], stats["claim_sentence_edges"], flush=True)


def stable_top_sentences(claim_sentence_mass: np.ndarray,
                         top_k: int = TOP_SENTENCES,
                         unrounded_scalar: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray(claim_sentence_mass, dtype=np.float32)
    assert mass.ndim == 2 and mass.shape[1] == ATTR_WIDTH and len(mass) > 0
    if unrounded_scalar is None:
        scalar = mass.mean(axis=1, dtype=np.float32)
    else:
        scalar = np.asarray(unrounded_scalar, dtype=np.float32)
        assert scalar.shape == (len(mass),) and np.isfinite(scalar).all()
    order = np.lexsort((np.arange(len(scalar)), -scalar))[:min(top_k, len(scalar))]
    return order.astype(np.int32), scalar


def relation_vector(claim: dict, sentence_text: str) -> np.ndarray:
    value = atomic_nli.relation_pair_features(claim, sentence_text).astype(np.float32)
    assert value.shape == (len(atomic_nli.PAIR_RELATION_NAMES),)
    return value


def semantic_feature_block(arrays: dict[str, np.ndarray], claim_index: int,
                           nli: dict[int, tuple[np.ndarray, np.ndarray]],
                           whitebox: np.ndarray) -> np.ndarray:
    """Construct one frozen feature row; ``nli`` is keyed by sentence_index."""
    c = len(arrays["claim_ids"])
    s = len(arrays["sentence_indices"])
    assert 0 <= claim_index < c and s > 0
    left, right = map(int, arrays["claim_indptr"][[claim_index, claim_index + 1]])
    assert right - left == s
    assert np.array_equal(arrays["claim_sentence_indices"][left:right],
                          np.arange(s, dtype=np.int32))
    mass = arrays["claim_sentence_mass"][left:right].astype(np.float32)
    assert mass.shape == (s, ATTR_WIDTH) and np.isfinite(mass).all()
    raw = np.log1p(mass.sum(axis=0, dtype=np.float32))

    source = arrays["source_total_relevance"][claim_index].astype(np.float64)
    previous = arrays["previous_answer_relevance"][claim_index].astype(np.float64)
    other = arrays["other_context_relevance"][claim_index].astype(np.float64)
    passage = arrays["passage_relevance"][claim_index].astype(np.float64)
    sentence = arrays["sentence_relevance"][claim_index].astype(np.float64)
    assert source.shape == previous.shape == other.shape == (LAYERS,)
    assert passage.shape == (LAYERS, 3) and sentence.shape == (LAYERS, s)
    eps = np.finfo(np.float64).eps
    context_total = source + previous + other
    source_safe = np.maximum(source, eps)
    sentence_total = sentence.sum(axis=1)
    sentence_safe = np.maximum(sentence_total, eps)
    ordered = np.sort(sentence, axis=1)[:, ::-1]
    top1 = ordered[:, 0] / sentence_safe
    top3 = ordered[:, :min(3, s)].sum(axis=1) / sentence_safe
    probability = sentence / sentence_safe[:, None]
    entropy = -(probability * np.log(np.maximum(probability, eps))).sum(axis=1)
    if s > 1:
        entropy /= np.log(s)
    else:
        entropy[:] = 0
    region = np.concatenate((
        source / np.maximum(context_total, eps),
        previous / np.maximum(context_total, eps),
        other / np.maximum(context_total, eps),
        passage[:, 0] / source_safe, passage[:, 1] / source_safe,
        passage[:, 2] / source_safe, top1, top3, entropy,
    )).astype(np.float32)

    # sentence_relevance was head-averaged in float32 before the CSR's float16
    # storage, so its layer mean is the least-quantized exact 1024 mean.
    global_sentence_scalar = sentence.mean(axis=0, dtype=np.float64).astype(np.float32)
    selected, scalar = stable_top_sentences(
        mass, unrounded_scalar=global_sentence_scalar)
    missing = [int(index) for index in selected if int(index) not in nli]
    if missing:
        raise MissingSelectedNLI(f"Missing selected NLI for sentence indices {missing}")
    blocks = []
    for index in selected:
        probability_row, relation_row = nli[int(index)]
        probability_row = np.asarray(probability_row, dtype=np.float32)
        relation_row = np.asarray(relation_row, dtype=np.float32)
        assert probability_row.shape == (3,) and relation_row.shape == (len(atomic_nli.PAIR_RELATION_NAMES),)
        assert np.isclose(probability_row.sum(), 1, atol=2e-6)
        blocks.append(np.concatenate((probability_row, relation_row)))
    top1_nli = blocks[0]
    selected_scalar = np.maximum(scalar[selected].astype(np.float64), 0)
    if selected_scalar.sum() == 0:
        selected_weight = np.full(len(selected), 1 / len(selected), dtype=np.float64)
    else:
        selected_weight = selected_scalar / selected_scalar.sum()
    top3_nli = np.average(np.vstack(blocks), axis=0, weights=selected_weight)
    total_scalar = max(float(np.maximum(scalar, 0).sum()), eps)
    first = float(max(scalar[selected[0]], 0))
    second = float(max(scalar[selected[1]], 0)) if len(selected) > 1 else 0.0
    selection = np.asarray([
        first / total_scalar,
        float(np.maximum(scalar[selected], 0).sum()) / total_scalar,
        (first - second) / total_scalar,
        np.log1p(s),
    ], dtype=np.float32)
    result = np.concatenate((raw, region, top1_nli, top3_nli, selection,
                             np.asarray(whitebox, dtype=np.float32)))
    assert result.shape == (len(SEMANTIC_FEATURE_NAMES),)
    assert np.isfinite(result).all()
    return result.astype(np.float32, copy=False)


def synthetic_selfcheck() -> dict:
    rng = np.random.default_rng(SEED)
    c, s = 2, 4
    mass = rng.random((c, s, LAYERS, HEADS), dtype=np.float32)
    compact_sentence = mass.mean(axis=-1).transpose(0, 2, 1)
    arrays = {
        "claim_sentence_mass": mass.reshape(c * s, ATTR_WIDTH).astype(np.float16),
        "claim_indptr": np.arange(c + 1, dtype=np.int64) * s,
        "claim_sentence_indices": np.tile(np.arange(s, dtype=np.int32), c),
        "source_total_relevance": rng.random((c, LAYERS), dtype=np.float32),
        "previous_answer_relevance": rng.random((c, LAYERS), dtype=np.float32),
        "other_context_relevance": rng.random((c, LAYERS), dtype=np.float32),
        "passage_relevance": rng.random((c, LAYERS, 3), dtype=np.float32),
        "sentence_relevance": compact_sentence.astype(np.float32),
        "claim_ids": np.arange(c, dtype=np.int32),
        "sentence_indices": np.arange(s, dtype=np.int32),
    }
    probability = {
        index: (np.asarray([.6, .3, .1], dtype=np.float32),
                np.zeros(len(atomic_nli.PAIR_RELATION_NAMES), dtype=np.float32))
        for index in range(s)
    }
    feature = semantic_feature_block(
        arrays, 0, probability,
        np.zeros(len(atomic_nli.WHITEBOX_FEATURE_NAMES), dtype=np.float32),
    )
    tied = np.ones((4, ATTR_WIDTH), dtype=np.float32)
    order, _ = stable_top_sentences(tied)
    assert order.tolist() == [0, 1, 2]
    groups = np.repeat(np.arange(10), 2)
    for train, held in GroupKFold(5).split(np.arange(20), groups=groups):
        assert not set(groups[train]) & set(groups[held])
    synthetic_rows = {
        "r0": {"group_id": "g0"}, "r1": {"group_id": "g0"},
        "r2": {"group_id": "g1"}, "r3": {"group_id": "g2"},
    }
    synthetic_response_ids = np.asarray(["r0", "r0", "r1", "r2", "r3", "r3"])
    synthetic_labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    synthetic_active = np.arange(len(synthetic_labels))
    synthetic_weights = claim_weights(
        synthetic_rows, synthetic_response_ids, synthetic_labels, synthetic_active)
    synthetic_mass = np.bincount(
        synthetic_labels, weights=synthetic_weights, minlength=2)
    assert np.isclose(synthetic_mass[0], synthetic_mass[1], rtol=0, atol=1e-9)
    no_add_labels = np.asarray([0, 1], dtype=np.int8)
    no_add_base = np.asarray([.1, .9], dtype=np.float64)
    no_add_gate = choose_add_threshold(
        no_add_labels, no_add_base >= .5, np.asarray([.9, .1]))
    assert no_add_gate["enabled"] is False and no_add_gate["threshold"] is None
    no_add_result, no_add_hits = gated_scores(
        no_add_base, np.asarray([.9, .1]), no_add_gate,
        {"window": {"threshold": .5}, "answer": {"threshold": .8}},
    )
    assert np.array_equal(no_add_result, no_add_base) and not no_add_hits.any()
    assert len(feature) == len(SEMANTIC_FEATURE_NAMES)
    assert len(CLEAN_BASE_FEATURE_NAMES) + len(aligned.INCUMBENT_NAMES) == len(aligned.FEATURE_NAMES)
    return {
        "passed": True,
        "semantic_feature_width": len(feature),
        "raw_1024_retained": len(RAW_ATTR_NAMES) == 1024,
        "stable_tie_order": order.tolist(),
        "group_fold_isolation": True,
        "final_binary_weight_mass": synthetic_mass.tolist(),
        "no_add_explicitly_disabled": True,
        "historical_incumbent_features_removed": list(aligned.INCUMBENT_NAMES),
        "calibration_labels_opened": False, "fit_labels_opened": False,
        "GPU_used": False, "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def check_prepared() -> dict:
    protocol_path = OUT / "protocol.json"
    preparation = read_json(OUT / "PREPARATION.json")
    snapshot = read_json(OUT / "source_snapshot.json")
    assert read_json(protocol_path) == protocol()
    assert snapshot["runner_sha256"] == sha(Path(__file__))
    assert snapshot["run_development_sha256"] == sha(Path(q.__file__))
    assert snapshot["atomic_nli_runner_sha256"] == sha(Path(atomic_nli.__file__))
    assert snapshot["aligned_evidence_runner_sha256"] == sha(Path(aligned.__file__))
    assert snapshot["protocol_sha256"] == sha(protocol_path)
    assert snapshot["atomic_inputs_sha256"] == sha(ROWS_PATH)
    assert snapshot["atomic_preparation_sha256"] == sha(
        ATOMIC_OUT / "preparation_complete.json")
    assert snapshot["attribution_layouts_sha256"] == sha(LAYOUTS_PATH)
    assert snapshot["attribution_runner_sha256"] == EXPECTED_ATTR_RUNNER_SHA256 == sha(ATTR_RUNNER_PATH)
    assert snapshot["attribution_signature_sha256"] == EXPECTED_ATTR_SIGNATURE_FILE_SHA256 == sha(ATTR_SIGNATURE_PATH)
    assert snapshot["attribution_global_index_sha256"] == EXPECTED_ATTR_GLOBAL_INDEX_SHA256 == sha(ATTR_GLOBAL_INDEX_PATH)
    assert snapshot["fit_index_sha256"] == sha(OUT / "fit_claim_sentence_index.jsonl")
    assert snapshot["gold_files_sha256_hash_only"] == {
        str(path.relative_to(ROOT)): sha(path) for path in GOLD_FILES
    }
    assert preparation["answers"] == EXPECTED_ANSWERS["fit"]
    assert preparation["claims"] == EXPECTED_CLAIMS["fit"]
    assert preparation["groups"] == EXPECTED_GROUPS["fit"]
    assert preparation["source_snapshot_sha256"] == sha(
        OUT / "source_snapshot.json")
    assert not preparation["calibration_labels_opened"]
    return preparation


def cpu_check() -> None:
    preparation = check_prepared()
    result = synthetic_selfcheck()
    frozen_json(OUT / "CPU_SELFCHECK.json", {
        "status": "passed_waiting_for_attribution_extraction",
        "preparation_sha256": sha(OUT / "PREPARATION.json"),
        "source_snapshot_sha256": sha(OUT / "source_snapshot.json"),
        "fit_answers": preparation["answers"],
        "fit_claims": preparation["claims"],
        **result,
    })
    print("SEMANTIC_ATTRIBUTION_SCORE_CPU_CHECK_PASSED",
          len(SEMANTIC_FEATURE_NAMES), flush=True)


def partition_rows_and_layouts(partition: str) -> tuple[list[dict], list[dict]]:
    assert partition in EXPECTED_ANSWERS
    rows = all_partition_rows(ROWS_PATH, partition)
    layouts = all_partition_rows(LAYOUTS_PATH, partition)
    assert len(rows) == len(layouts) == EXPECTED_ANSWERS[partition]
    assert [row["response_id"] for row in rows] == [row["response_id"] for row in layouts]
    assert sum(len(row["claims"]) for row in rows) == EXPECTED_CLAIMS[partition]
    assert len({row["group_id"] for row in rows}) == EXPECTED_GROUPS[partition]
    return rows, layouts


def validate_attribution_arrays(arrays: dict[str, np.ndarray], row: dict,
                                layout: dict) -> None:
    required = {
        "claim_sentence_mass", "claim_indptr", "claim_sentence_indices",
        "source_total_relevance", "previous_answer_relevance",
        "other_context_relevance", "passage_relevance", "sentence_relevance",
        "top_sentence_indices", "top_sentence_relevance", "claim_ids",
        "microclaim_indices", "sentence_indices", "sentence_passage_ids",
        "sentence_ids", "sentence_identity_sha256", "answer_token_positions",
    }
    assert set(arrays) == required
    c, s = len(row["claims"]), len(layout["sentences"])
    assert arrays["claim_sentence_mass"].shape == (c * s, ATTR_WIDTH)
    assert arrays["claim_sentence_mass"].dtype == np.float16
    assert np.array_equal(arrays["claim_indptr"], np.arange(c + 1, dtype=np.int64) * s)
    assert np.array_equal(arrays["claim_sentence_indices"], np.tile(np.arange(s, dtype=np.int32), c))
    assert arrays["source_total_relevance"].shape == (c, LAYERS)
    assert arrays["previous_answer_relevance"].shape == (c, LAYERS)
    assert arrays["other_context_relevance"].shape == (c, LAYERS)
    assert arrays["passage_relevance"].shape == (c, LAYERS, 3)
    assert arrays["sentence_relevance"].shape == (c, LAYERS, s)
    assert arrays["top_sentence_indices"].shape == (c, LAYERS, 15)
    assert arrays["top_sentence_relevance"].shape == (c, LAYERS, 15)
    assert np.array_equal(arrays["claim_ids"], np.arange(c, dtype=np.int32))
    assert np.array_equal(arrays["microclaim_indices"],
                          [claim["microclaim_index"] for claim in row["claims"]])
    assert np.array_equal(arrays["sentence_indices"], np.arange(s, dtype=np.int32))
    assert np.array_equal(arrays["sentence_passage_ids"],
                          [sentence["passage_id"] for sentence in layout["sentences"]])
    assert np.array_equal(arrays["sentence_ids"],
                          [sentence["sentence_id"] for sentence in layout["sentences"]])
    assert arrays["sentence_identity_sha256"].shape == (s, 32)
    assert np.array_equal(arrays["answer_token_positions"],
                          layout["answer_token_positions"])
    for index, sentence in enumerate(layout["sentences"]):
        assert arrays["sentence_identity_sha256"][index].tobytes().hex() == sentence["text_sha256"]
    for value in arrays.values():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all() and np.all(value >= 0)


def load_attribution(row: dict, layout: dict, manifest: dict) -> dict[str, np.ndarray]:
    rid = row["response_id"]
    feature = ATTR_OUT / "features" / f"{rid}.npz"
    metadata_path = feature.with_suffix(".json")
    commit_path = feature.with_suffix(".commit.json")
    expected = manifest["files"][rid]
    assert sha(feature) == expected["npz_sha256"]
    assert sha(metadata_path) == expected["metadata_sha256"]
    assert sha(commit_path) == expected["commit_sha256"]
    metadata = read_json(metadata_path)
    assert metadata["response_id"] == rid and metadata["partition"] == row["partition"]
    assert metadata["npz_sha256"] == expected["npz_sha256"]
    with np.load(feature, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    validate_attribution_arrays(arrays, row, layout)
    return arrays


def checked_attribution_manifest() -> dict:
    assert ATTR_MANIFEST_PATH.is_file(), "Attribution extraction is not complete"
    manifest = read_json(ATTR_MANIFEST_PATH)
    assert manifest["status"] == "complete"
    assert manifest["records_complete"] == manifest["records_total"] == sum(EXPECTED_ANSWERS.values())
    assert manifest["claims_complete"] == sum(EXPECTED_CLAIMS.values())
    assert manifest["labels_used"] is False and manifest["official_test_opened"] is False
    assert len(manifest["files"]) == sum(EXPECTED_ANSWERS.values())
    return manifest


def sentence_text_lookup(row: dict) -> dict[tuple[int, int], str]:
    return {
        (int(passage["passage_id"]), int(sentence["sentence_id"])): sentence["text"]
        for passage in row["passages"] for sentence in passage["sentences"]
    }


def nli_request_id(premise: str, hypothesis: str) -> str:
    return digest({
        "checkpoint": "tasksource/ModernBERT-base-nli",
        "revision": atomic_nli.MODEL_REVISION,
        "premise": premise, "hypothesis": hypothesis,
    })


def link_nli(partition: str) -> None:
    check_prepared()
    manifest = checked_attribution_manifest()
    rows, layouts = partition_rows_and_layouts(partition)
    links, requests = [], {}
    selected_indices, selected_weights, probabilities, request_bytes = [], [], [], []
    cached_count = missing_count = 0
    global_claim = 0
    for answer_index, (row, layout) in enumerate(zip(rows, layouts)):
        arrays = load_attribution(row, layout, manifest)
        cache_probabilities, _ = atomic_nli.validate_cache(
            PAIR_DIR / f"{row['response_id']}.npz", row)
        pair_map = individual_pair_indices(row)
        texts = sentence_text_lookup(row)
        answer_links = []
        for claim_index, claim in enumerate(row["claims"]):
            left, right = map(int, arrays["claim_indptr"][[claim_index, claim_index + 1]])
            mass = arrays["claim_sentence_mass"][left:right].astype(np.float32)
            scalar_unrounded = arrays["sentence_relevance"][claim_index].mean(
                axis=0, dtype=np.float64).astype(np.float32)
            chosen, scalar = stable_top_sentences(
                mass, unrounded_scalar=scalar_unrounded)
            chosen_probability, chosen_request, claim_links = [], [], []
            chosen_mass = np.maximum(scalar[chosen].astype(np.float64), 0)
            if chosen_mass.sum() == 0:
                weights = np.full(len(chosen), 1 / len(chosen), dtype=np.float64)
            else:
                weights = chosen_mass / chosen_mass.sum()
            for rank, sentence_index in enumerate(chosen, 1):
                sentence = layout["sentences"][int(sentence_index)]
                key = (int(claim["claim_id"]), int(sentence["passage_id"]),
                       int(sentence["sentence_id"]))
                pair_index = pair_map.get(key)
                premise = texts[key[1], key[2]]
                assert digest(premise) == sentence["text_sha256"]
                request_id = nli_request_id(premise, claim["hypothesis"])
                if pair_index is None:
                    source = "missing_exact_pair"
                    probability = np.full(3, np.nan, dtype=np.float32)
                    requests.setdefault(request_id, {
                        "request_id": request_id, "premise": premise,
                        "hypothesis": claim["hypothesis"],
                        "premise_sha256": digest(premise),
                        "hypothesis_sha256": digest(claim["hypothesis"]),
                    })
                    missing_count += 1
                else:
                    source = "frozen_atomic_individual_pair"
                    probability = cache_probabilities[pair_index].astype(np.float32)
                    cached_count += 1
                chosen_probability.append(probability)
                chosen_request.append(bytes.fromhex(request_id))
                claim_links.append({
                    "rank": rank, "sentence_index": int(sentence_index),
                    "passage_id": key[1], "sentence_id": key[2],
                    "attribution_scalar": float(scalar[sentence_index]),
                    "top3_weight": float(weights[rank - 1]),
                    "nli_source": source,
                    "atomic_pair_index": pair_index,
                    "request_id": request_id,
                })
            selected_indices.append(chosen)
            selected_weights.append(weights)
            probabilities.append(np.vstack(chosen_probability))
            request_bytes.append(np.stack([
                np.frombuffer(value, dtype=np.uint8) for value in chosen_request
            ]))
            answer_links.append({
                "global_claim_index": global_claim,
                "claim_id": int(claim["claim_id"]),
                "microclaim_id": claim["microclaim_id"],
                "selected": claim_links,
            })
            global_claim += 1
        links.append({
            "answer_index": answer_index, "response_id": row["response_id"],
            "group_id": row["group_id"], "partition": partition,
            "claims": answer_links,
        })
        if (answer_index + 1) % 100 == 0:
            print("SEMANTIC_NLI_LINK", partition, answer_index + 1, len(rows), flush=True)
    assert global_claim == EXPECTED_CLAIMS[partition]
    links_path = OUT / f"selected_nli_links_{partition}.jsonl"
    requests_path = OUT / f"selected_nli_requests_{partition}.jsonl"
    cache_path = OUT / f"selected_nli_existing_{partition}.npz"
    frozen_jsonl(links_path, links)
    frozen_jsonl(requests_path, [requests[key] for key in sorted(requests)])
    arrays = {
        "selected_sentence_indices": np.asarray(selected_indices, dtype=np.int32),
        "selected_weights": np.asarray(selected_weights, dtype=np.float32),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "request_identity_sha256": np.asarray(request_bytes, dtype=np.uint8),
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as old:
            assert set(old.files) == set(arrays)
            assert all(np.array_equal(old[key], value, equal_nan=True)
                       for key, value in arrays.items())
    else:
        atomic_npz(cache_path, **arrays)
    frozen_json(OUT / f"selected_nli_manifest_{partition}.json", {
        "status": "linked_missing_pairs_not_inferred" if requests else "complete_no_missing_pairs",
        "partition": partition, "answers": len(rows), "claims": global_claim,
        "selected_links": global_claim * TOP_SENTENCES,
        "cached_links": cached_count, "missing_links": missing_count,
        "unique_missing_requests": len(requests),
        "links_sha256": sha(links_path), "requests_sha256": sha(requests_path),
        "existing_npz_sha256": sha(cache_path),
        "attribution_manifest_sha256": sha(ATTR_MANIFEST_PATH),
        "labels_opened": False, "model_loaded": False, "GPU_used": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("SEMANTIC_NLI_LINK_COMPLETE", partition, cached_count,
          missing_count, len(requests), flush=True)


def infer_missing_nli(partition: str) -> None:
    """Explicit GPU stage; uses the same frozen NLI checkpoint as atomic v1."""
    manifest_path = OUT / f"selected_nli_manifest_{partition}.json"
    manifest = read_json(manifest_path)
    requests_path = OUT / f"selected_nli_requests_{partition}.jsonl"
    assert manifest["requests_sha256"] == sha(requests_path)
    with requests_path.open(encoding="utf-8") as handle:
        requests = [json.loads(line) for line in handle if line.strip()]
    output_path = OUT / f"selected_nli_missing_scores_{partition}.npz"
    output_meta = OUT / f"selected_nli_missing_scores_{partition}.json"
    assert not output_path.exists() and not output_meta.exists()
    assert len(requests) == manifest["unique_missing_requests"]
    if requests:
        tokenizer, model, load_metadata = atomic_nli.load_cuda()
        try:
            pairs = [(row["premise"], row["hypothesis"]) for row in requests]
            probability, shapes = atomic_nli.infer_pairs(tokenizer, model, pairs,
                                                         atomic_nli.torch.device("cuda:0"))
        finally:
            del model, tokenizer
            atomic_nli.clean_gpu()
    else:
        load_metadata, shapes = {}, []
        probability = np.empty((0, 3), dtype=np.float32)
    identities = np.stack([
        np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8)
        for row in requests
    ]) if requests else np.empty((0, 32), dtype=np.uint8)
    atomic_npz(output_path, request_identity_sha256=identities,
               probabilities=probability)
    atomic_json(output_meta, {
        "status": "complete_frozen_probabilities_not_trained",
        "partition": partition, "requests": len(requests),
        "checkpoint": "tasksource/ModernBERT-base-nli",
        "revision": atomic_nli.MODEL_REVISION,
        "model_sha256": atomic_nli.MODEL_SHA256,
        "request_file_sha256": sha(requests_path),
        "npz_sha256": sha(output_path), "batch_shapes": shapes,
        "model_load": load_metadata, "labels_opened": False,
        "GPU_used": bool(requests), "trained": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("SEMANTIC_MISSING_NLI_COMPLETE", partition, len(requests), flush=True)


def load_selected_nli(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = read_json(OUT / f"selected_nli_manifest_{partition}.json")
    existing_path = OUT / f"selected_nli_existing_{partition}.npz"
    assert manifest["existing_npz_sha256"] == sha(existing_path)
    with np.load(existing_path, allow_pickle=False) as loaded:
        selected = loaded["selected_sentence_indices"].copy()
        weights = loaded["selected_weights"].copy()
        probability = loaded["probabilities"].copy()
        identities = loaded["request_identity_sha256"].copy()
    assert selected.shape == weights.shape == (EXPECTED_CLAIMS[partition], TOP_SENTENCES)
    assert probability.shape == (EXPECTED_CLAIMS[partition], TOP_SENTENCES, 3)
    if np.isnan(probability).any():
        scores_path = OUT / f"selected_nli_missing_scores_{partition}.npz"
        scores_meta = read_json(OUT / f"selected_nli_missing_scores_{partition}.json")
        assert scores_meta["npz_sha256"] == sha(scores_path)
        with np.load(scores_path, allow_pickle=False) as missing:
            keys = missing["request_identity_sha256"]
            values = missing["probabilities"]
        mapping = {key.tobytes(): values[index] for index, key in enumerate(keys)}
        for claim, rank in zip(*np.where(np.isnan(probability).any(axis=2))):
            probability[claim, rank] = mapping[identities[claim, rank].tobytes()]
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=2), 1, rtol=0, atol=2e-6)
    return selected, weights, probability


def build_partition_features(partition: str) -> dict[str, np.ndarray]:
    """Build label-free claim matrices for one already-linked partition."""
    manifest = checked_attribution_manifest()
    rows, layouts = partition_rows_and_layouts(partition)
    selected, selected_weights, selected_probability = load_selected_nli(partition)
    whitebox = np.load(WHITEBOX_PATH, mmap_mode="r")
    wb_offset = 0 if partition == "fit" else EXPECTED_CLAIMS["fit"]
    clean_rows, semantic_rows, response_ids, groups = [], [], [], []
    claim_cursor = 0
    for answer_index, (row, layout) in enumerate(zip(rows, layouts)):
        arrays = load_attribution(row, layout, manifest)
        pair_probability, _ = atomic_nli.validate_cache(
            PAIR_DIR / f"{row['response_id']}.npz", row)
        pairs, owners = atomic_nli.row_pairs(row)
        texts = sentence_text_lookup(row)
        for claim_index, claim in enumerate(row["claims"]):
            whitebox_row = np.asarray(whitebox[wb_offset + claim_cursor], dtype=np.float32)
            pair_features, _, _ = aligned.select_slots(
                claim, row, pair_probability, pairs, owners)
            clean = np.concatenate((
                atomic_nli.claim_self_features(claim), pair_features, whitebox_row,
            )).astype(np.float32)
            assert clean.shape == (len(CLEAN_BASE_FEATURE_NAMES),)

            chosen = selected[claim_cursor]
            left, right = map(int, arrays["claim_indptr"][[claim_index, claim_index + 1]])
            mass = arrays["claim_sentence_mass"][left:right].astype(np.float32)
            scalar_unrounded = arrays["sentence_relevance"][claim_index].mean(
                axis=0, dtype=np.float64).astype(np.float32)
            recomputed, scalar = stable_top_sentences(
                mass, unrounded_scalar=scalar_unrounded)
            assert np.array_equal(chosen, recomputed)
            chosen_mass = np.maximum(scalar[chosen].astype(np.float64), 0)
            expected_weight = (chosen_mass / chosen_mass.sum() if chosen_mass.sum()
                               else np.full(len(chosen), 1 / len(chosen)))
            assert np.allclose(selected_weights[claim_cursor], expected_weight,
                               rtol=0, atol=2e-7)
            nli = {}
            for rank, sentence_index in enumerate(chosen):
                sentence = layout["sentences"][int(sentence_index)]
                text = texts[int(sentence["passage_id"]), int(sentence["sentence_id"])]
                nli[int(sentence_index)] = (
                    selected_probability[claim_cursor, rank],
                    relation_vector(claim, text),
                )
            semantic = semantic_feature_block(arrays, claim_index, nli, whitebox_row)
            clean_rows.append(clean); semantic_rows.append(semantic)
            response_ids.append(row["response_id"]); groups.append(row["group_id"])
            claim_cursor += 1
        if (answer_index + 1) % 100 == 0:
            print("SEMANTIC_MATRIX", partition, answer_index + 1, len(rows), flush=True)
    assert claim_cursor == EXPECTED_CLAIMS[partition]
    clean = np.vstack(clean_rows).astype(np.float32)
    semantic = np.vstack(semantic_rows).astype(np.float32)
    assert clean.shape == (claim_cursor, len(CLEAN_BASE_FEATURE_NAMES))
    assert semantic.shape == (claim_cursor, len(SEMANTIC_FEATURE_NAMES))
    assert np.isfinite(clean).all() and np.isfinite(semantic).all()
    return {
        "clean": clean, "semantic": semantic,
        "response_ids": np.asarray(response_ids), "groups": np.asarray(groups),
    }


def load_partition_meta(partition: str) -> dict:
    data = ROOT / "data"
    answers = q.lines(data / f"answers_{partition}.jsonl")
    tokens = q.lines(data / f"tokens_{partition}.jsonl")
    windows = q.lines(data / f"windows_k4_{partition}.jsonl")
    assert len(answers) == len(tokens) == EXPECTED_ANSWERS[partition]
    assert len(windows) == EXPECTED_WINDOWS[partition]
    assert [row["response_id"] for row in answers] == [row["response_id"] for row in tokens]
    by_response = {answer["response_id"]: {"answer": answer, "tokens": token}
                   for answer, token in zip(answers, tokens)}
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        assert window["partition"] == partition and window["eligible"]
        answer_windows[window["response_id"]].append(index)
    assert all(answer_windows[answer["response_id"]] for answer in answers)
    return {
        "answers": answers, "tokens": tokens, "windows": windows,
        "by_response": by_response, "answer_windows": dict(answer_windows),
        "partition": partition,
    }


def labels_for_claims(meta: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    any_labels, conflict_labels = [], []
    for row in rows:
        token = meta["by_response"][row["response_id"]]["tokens"]
        risk = np.asarray(token["risk_mask"], dtype=bool)
        lexical = np.asarray(token["lexical_mask"], dtype=bool)
        offsets = token["response_token_offsets"]
        conflict = np.zeros(len(risk), dtype=bool)
        for label in token["original_labels"]:
            if "Conflict" not in label["label_type"]:
                continue
            begin, end = int(label["start"]), int(label["end"])
            for token_index, (left, right) in enumerate(offsets):
                if max(left, begin) < min(right, end):
                    conflict[token_index] = True
        assert np.all(~conflict | risk | ~lexical)
        for claim in row["claims"]:
            ids = claim["lexical_token_indices"]
            any_labels.append(int(risk[ids].any()))
            conflict_labels.append(int(conflict[ids].any()))
    any_y = np.asarray(any_labels, dtype=np.int8)
    conflict_y = np.asarray(conflict_labels, dtype=np.int8)
    assert len(any_y) == EXPECTED_CLAIMS[meta["partition"]]
    assert np.all(conflict_y <= any_y)
    return any_y, conflict_y


def claim_weights(rows_by_id: dict, response_ids: np.ndarray,
                  labels: np.ndarray, active: np.ndarray) -> np.ndarray:
    tree = defaultdict(lambda: defaultdict(list))
    for index in active:
        rid = response_ids[index]
        tree[rows_by_id[rid]["group_id"]][rid].append(int(index))
    weights = np.zeros(len(labels), dtype=np.float64)
    for answers in tree.values():
        for indices in answers.values():
            weights[indices] = 1 / (len(answers) * len(indices))
    mask = weights > 0
    weights[mask] /= weights[mask].mean()
    mass = np.bincount(labels[mask], weights=weights[mask], minlength=2)
    assert np.all(mass > 0)
    weights[mask] *= (mass.sum() / (2 * mass))[labels[mask]]
    weights[mask] *= mask.sum() / weights[mask].sum()
    final_mass = np.bincount(labels[mask], weights=weights[mask], minlength=2)
    assert np.isclose(final_mass[0], final_mass[1], rtol=0, atol=1e-9), final_mass
    assert np.isclose(final_mass.sum(), mask.sum(), rtol=0, atol=1e-9), final_mass
    return weights


def make_lr(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, solver="liblinear", penalty="l2",
                           max_iter=3000, random_state=SEED),
    )


def fit_lr(model, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    model.fit(x, y, logisticregression__sample_weight=weights)
    return model


def project_claim_scores(meta: dict, rows: list[dict],
                         claim_scores: np.ndarray) -> np.ndarray:
    output = np.empty(len(meta["windows"]), dtype=np.float64)
    cursor = 0
    for row in rows:
        local = claim_scores[cursor:cursor + len(row["claims"])]
        cursor += len(row["claims"])
        owners = row["lexical_token_microclaims"]
        for window_index in meta["answer_windows"][row["response_id"]]:
            claim_ids = {
                int(claim_id)
                for token_index in meta["windows"][window_index]["token_indices"]
                for claim_id in owners[token_index]
            }
            assert claim_ids
            output[window_index] = max(local[claim_id] for claim_id in claim_ids)
    assert cursor == len(claim_scores) and np.isfinite(output).all()
    return output


def answer_scores(meta: dict, window_scores: np.ndarray) -> np.ndarray:
    return np.asarray([
        window_scores[meta["answer_windows"][answer["response_id"]]].max()
        for answer in meta["answers"]
    ], dtype=np.float64)


def choose_thresholds(meta: dict, window_scores: np.ndarray) -> dict:
    return {
        "window": q.choose_threshold(
            [row["label"] for row in meta["windows"]], window_scores),
        "answer": q.choose_threshold(
            [row["label"] for row in meta["answers"]],
            answer_scores(meta, window_scores)),
    }


def metric_pair(meta: dict, window_scores: np.ndarray, thresholds: dict) -> dict:
    return {
        "windows": q.count(
            [row["label"] for row in meta["windows"]], window_scores,
            thresholds["window"]["threshold"]),
        "answers": q.count(
            [row["label"] for row in meta["answers"]],
            answer_scores(meta, window_scores), thresholds["answer"]["threshold"]),
    }


def choose_add_threshold(labels: np.ndarray, base_prediction: np.ndarray,
                         semantic_scores: np.ndarray) -> dict:
    active = np.flatnonzero(~base_prediction)
    if not len(active):
        return {
            "enabled": False, "threshold": None,
            "union_f1": float(q.count(labels, base_prediction.astype(float), .5)["f1"]),
            "incremental_precision": 1.0, "additions": 0, "tp": 0, "fp": 0,
        }
    order = np.argsort(-semantic_scores[active], kind="stable")
    ids = active[order]
    scores = semantic_scores[ids]
    assert len(scores) > 0
    last = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    add_tp = np.cumsum(labels[ids])[last]
    additions = last + 1
    add_fp = additions - add_tp
    base_tp = int(np.count_nonzero(base_prediction & (labels == 1)))
    base_fp = int(np.count_nonzero(base_prediction & (labels == 0)))
    positives = int(labels.sum())
    denominator = 2 * (base_tp + add_tp) + base_fp + add_fp + positives - base_tp - add_tp
    f1 = 2 * (base_tp + add_tp) / denominator
    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + positives - base_tp)
    # The no-add option is a real disabled state, never an artificial threshold
    # just above the largest observed fit score.
    options = [(base_f1, 1.0, 0, float("inf"), 0, 0, False, None)]
    for index in range(len(last)):
        precision = float(add_tp[index] / additions[index])
        options.append((float(f1[index]), precision, -int(additions[index]),
                        float(scores[last[index]]), int(add_tp[index]),
                        int(add_fp[index]), True, float(scores[last[index]])))
    best = max(options, key=lambda item: item[:4])
    return {
        "enabled": best[6], "threshold": best[7], "union_f1": best[0],
        "incremental_precision": best[1], "additions": -best[2],
        "tp": best[4], "fp": best[5],
    }


def gated_scores(base: np.ndarray, semantic: np.ndarray, gate: dict,
                 thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    window_cut = float(thresholds["window"]["threshold"])
    answer_cut = float(thresholds["answer"]["threshold"])
    result = base.astype(np.float64, copy=True)
    if not gate["enabled"]:
        assert gate["threshold"] is None and gate["additions"] == 0
        return result, np.zeros(len(result), dtype=bool)
    assert answer_cut > window_cut, (
        "The preregistered preserve-answer gate needs answer threshold above window threshold",
        window_cut, answer_cut,
    )
    assert gate["threshold"] is not None
    hit = (base < window_cut) & (semantic >= float(gate["threshold"]))
    if hit.any():
        selected = semantic[hit]
        span = max(float(selected.max() - gate["threshold"]), np.finfo(np.float64).eps)
        fraction = np.clip((selected - gate["threshold"]) / span, 0, 1)
        upper = np.nextafter(answer_cut, -np.inf)
        result[hit] = window_cut + (upper - window_cut) * (.25 + .74 * fraction)
    assert np.array_equal(result >= window_cut, (base >= window_cut) | hit)
    return result, hit


def crossfit_components(features: dict[str, np.ndarray], rows: list[dict],
                        any_y: np.ndarray, conflict_y: np.ndarray) -> tuple[dict, list]:
    count = len(any_y)
    indices = np.arange(count)
    folds = list(GroupKFold(FOLDS).split(indices, any_y, features["groups"]))
    row_by_id = {row["response_id"]: row for row in rows}
    specifications = {
        "clean_any": (features["clean"], any_y, CLEAN_BASE_C),
        "clean_conflict": (features["clean"], conflict_y, CLEAN_BASE_C),
        "semantic_any": (features["semantic"], any_y, SEMANTIC_C),
        "semantic_conflict": (features["semantic"], conflict_y, SEMANTIC_C),
    }
    outputs = {name: np.full(count, np.nan, dtype=np.float64) for name in specifications}
    fold_log = []
    with threadpool_limits(limits=THREADS):
        for fold, (train, held) in enumerate(folds):
            record = {
                "fold": fold, "train_claims": len(train), "held_claims": len(held),
                "train_groups": len(set(features["groups"][train])),
                "held_groups": len(set(features["groups"][held])),
            }
            assert not set(features["groups"][train]) & set(features["groups"][held])
            for name, (matrix, target, c_value) in specifications.items():
                weights = claim_weights(row_by_id, features["response_ids"], target, train)
                final_mass = np.bincount(target[train], weights=weights[train], minlength=2)
                assert np.isclose(final_mass[0], final_mass[1], rtol=0, atol=1e-9)
                model = fit_lr(make_lr(c_value), matrix[train], target[train], weights[train])
                outputs[name][held] = model.predict_proba(matrix[held])[:, 1]
                record[name + "_train_positive"] = int(target[train].sum())
                record[name + "_held_positive"] = int(target[held].sum())
                record[name + "_final_weight_mass_by_class"] = final_mass.tolist()
            fold_log.append(record)
            print("SEMANTIC_CROSSFIT_FOLD", fold + 1, FOLDS, flush=True)
    assert all(np.isfinite(value).all() for value in outputs.values())
    return outputs, fold_log


def fit_full_components(features: dict[str, np.ndarray], rows: list[dict],
                        any_y: np.ndarray, conflict_y: np.ndarray) -> tuple[dict, dict]:
    indices = np.arange(len(any_y))
    row_by_id = {row["response_id"]: row for row in rows}
    specifications = {
        "clean_any": (features["clean"], any_y, CLEAN_BASE_C),
        "clean_conflict": (features["clean"], conflict_y, CLEAN_BASE_C),
        "semantic_any": (features["semantic"], any_y, SEMANTIC_C),
        "semantic_conflict": (features["semantic"], conflict_y, SEMANTIC_C),
    }
    models, weight_mass_log = {}, {}
    with threadpool_limits(limits=THREADS):
        for name, (matrix, target, c_value) in specifications.items():
            weights = claim_weights(row_by_id, features["response_ids"], target, indices)
            final_mass = np.bincount(target, weights=weights, minlength=2)
            assert np.isclose(final_mass[0], final_mass[1], rtol=0, atol=1e-9)
            weight_mass_log[name] = final_mass.tolist()
            models[name] = fit_lr(make_lr(c_value), matrix, target, weights)
    return models, weight_mass_log


def fit() -> None:
    check_prepared()
    assert (OUT / "CPU_SELFCHECK.json").is_file()
    assert not (OUT / "fit_complete.json").exists()
    started = time.perf_counter()
    rows, _ = partition_rows_and_layouts("fit")
    features = build_partition_features("fit")
    meta = load_partition_meta("fit")  # First label-bearing read in this runner.
    any_y, conflict_y = labels_for_claims(meta, rows)
    oof, fold_log = crossfit_components(features, rows, any_y, conflict_y)
    clean_claim = np.maximum(oof["clean_any"], oof["clean_conflict"])
    semantic_claim = np.maximum(oof["semantic_any"], oof["semantic_conflict"])
    clean_window = project_claim_scores(meta, rows, clean_claim)
    semantic_window = project_claim_scores(meta, rows, semantic_claim)
    thresholds = choose_thresholds(meta, clean_window)
    labels = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    gate = choose_add_threshold(
        labels, clean_window >= thresholds["window"]["threshold"], semantic_window)
    combined_window, additions = gated_scores(
        clean_window, semantic_window, gate, thresholds)
    fit_metrics = metric_pair(meta, combined_window, thresholds)
    clean_metrics = metric_pair(meta, clean_window, thresholds)
    assert abs(fit_metrics["windows"]["f1"] - gate["union_f1"]) < 1e-15
    models, full_weight_mass = fit_full_components(features, rows, any_y, conflict_y)
    model_path = OUT / "model.pkl"
    model_path.write_bytes(pickle.dumps({
        "version": VERSION, "models": models,
        "clean_feature_names": CLEAN_BASE_FEATURE_NAMES,
        "semantic_feature_names": SEMANTIC_FEATURE_NAMES,
    }, protocol=5))
    scores_path = OUT / "fit_scores.npz"
    atomic_npz(scores_path, **oof, clean_claim=clean_claim,
               semantic_claim=semantic_claim, clean_window=clean_window,
               semantic_window=semantic_window, combined_window=combined_window,
               additions=additions.astype(np.int8), any_labels=any_y,
               conflict_labels=conflict_y)
    fit_source = {
        "runner_sha256": sha(Path(__file__)),
        "attribution_manifest_sha256": sha(ATTR_MANIFEST_PATH),
        "fit_nli_manifest_sha256": sha(OUT / "selected_nli_manifest_fit.json"),
        "fit_nli_scores_sha256": sha(OUT / "selected_nli_missing_scores_fit.npz"),
        "whitebox_sha256": sha(WHITEBOX_PATH),
    }
    atomic_json(OUT / "fit_complete.json", {
        "status": "models_and_thresholds_frozen_before_calibration_labels",
        "crossfit": fold_log, "base_fit_thresholds": thresholds,
        "full_fit_final_weight_mass_by_class": full_weight_mass,
        "semantic_add_gate": gate, "clean_fit_OOF_metrics": clean_metrics,
        "combined_fit_OOF_metrics": fit_metrics,
        "fit_additions": {
            "count": int(additions.sum()),
            "tp": int(np.count_nonzero(additions & (labels == 1))),
            "fp": int(np.count_nonzero(additions & (labels == 0))),
        },
        "claim_labels": {"any_error": int(any_y.sum()),
                         "conflict_only": int(conflict_y.sum())},
        "model_sha256": sha(model_path), "fit_scores_sha256": sha(scores_path),
        "feature_widths": {"clean": features["clean"].shape[1],
                           "semantic": features["semantic"].shape[1]},
        "source_sha256": fit_source,
        "fit_labels_opened": True, "calibration_labels_opened": False,
        "calibration_used_for_model_or_threshold_selection": False,
        "GPU_used_for_training": False, "formal_baselines_modified": False,
        "official_test_opened": False, "seconds": time.perf_counter() - started,
    })
    print("SEMANTIC_ATTRIBUTION_SCORE_FIT_FROZEN",
          fit_metrics["windows"]["f1"], fit_metrics["answers"]["f1"], flush=True)


def evaluate() -> None:
    fit_done = read_json(OUT / "fit_complete.json")
    assert fit_done["status"] == "models_and_thresholds_frozen_before_calibration_labels"
    assert fit_done["model_sha256"] == sha(OUT / "model.pkl")
    assert fit_done["source_sha256"]["runner_sha256"] == sha(Path(__file__))
    assert fit_done["source_sha256"]["attribution_manifest_sha256"] == sha(ATTR_MANIFEST_PATH)
    assert fit_done["source_sha256"]["fit_nli_manifest_sha256"] == sha(
        OUT / "selected_nli_manifest_fit.json")
    assert fit_done["source_sha256"]["fit_nli_scores_sha256"] == sha(
        OUT / "selected_nli_missing_scores_fit.npz")
    assert fit_done["source_sha256"]["whitebox_sha256"] == sha(WHITEBOX_PATH)
    assert not (OUT / "summary.json").exists()
    started = time.perf_counter()
    rows, _ = partition_rows_and_layouts("calibration")
    features = build_partition_features("calibration")
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    assert tuple(payload["clean_feature_names"]) == CLEAN_BASE_FEATURE_NAMES
    assert tuple(payload["semantic_feature_names"]) == SEMANTIC_FEATURE_NAMES
    model = payload["models"]
    predictions = {
        "clean_any": model["clean_any"].predict_proba(features["clean"])[:, 1],
        "clean_conflict": model["clean_conflict"].predict_proba(features["clean"])[:, 1],
        "semantic_any": model["semantic_any"].predict_proba(features["semantic"])[:, 1],
        "semantic_conflict": model["semantic_conflict"].predict_proba(features["semantic"])[:, 1],
    }
    clean_claim = np.maximum(predictions["clean_any"], predictions["clean_conflict"])
    semantic_claim = np.maximum(predictions["semantic_any"], predictions["semantic_conflict"])
    # Calibration label-bearing files are first opened after every score above is fixed.
    meta = load_partition_meta("calibration")
    clean_window = project_claim_scores(meta, rows, clean_claim)
    semantic_window = project_claim_scores(meta, rows, semantic_claim)
    thresholds = fit_done["base_fit_thresholds"]
    combined_window, additions = gated_scores(
        clean_window, semantic_window,
        fit_done["semantic_add_gate"], thresholds)
    strict = metric_pair(meta, combined_window, thresholds)
    strict_clean = metric_pair(meta, clean_window, thresholds)
    # Required common development diagnostic; it does not alter the frozen model.
    calibration_f1opt_thresholds = choose_thresholds(meta, combined_window)
    calibration_f1opt = metric_pair(meta, combined_window,
                                    calibration_f1opt_thresholds)
    labels = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    scores_path = OUT / "calibration_scores.npz"
    atomic_npz(scores_path, **predictions, clean_claim=clean_claim,
               semantic_claim=semantic_claim, clean_window=clean_window,
               semantic_window=semantic_window, combined_window=combined_window,
               answer_scores=answer_scores(meta, combined_window),
               additions=additions.astype(np.int8))
    summary = {
        "status": "development_calibration_evaluated_once",
        "strict_fit_thresholds": strict,
        "strict_clean_base_same_thresholds": strict_clean,
        "unified_calibration_F1Opt_diagnostic": calibration_f1opt,
        "unified_calibration_F1Opt_thresholds": calibration_f1opt_thresholds,
        "calibration_additions": {
            "count": int(additions.sum()),
            "tp": int(np.count_nonzero(additions & (labels == 1))),
            "fp": int(np.count_nonzero(additions & (labels == 0))),
        },
        "model_or_feature_selected_on_calibration": False,
        "calibration_evaluations": 1,
        "scores_sha256": sha(scores_path),
        "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "calibration_labels_opened": True, "formal_baselines_modified": False,
        "official_test_opened": False, "seconds": time.perf_counter() - started,
    }
    atomic_json(OUT / "summary.json", summary)
    report = [
        "# Semantic source attribution v1 score", "",
        "模型保留完整 32×32 source-attribution 坐标，并用注意力选中的精确来源句做 NLI；clean Aligned Evidence 基座不含历史 calibration 优化分数。两个部件都按 source-connected group 五折交叉预测。", "",
        "| 口径 | 窗口F1 | 整答F1 |",
        "|---|---:|---:|",
        f"| strict fit阈值 | {strict['windows']['f1']:.6f} | {strict['answers']['f1']:.6f} |",
        f"| 统一cal F1-opt诊断 | {calibration_f1opt['windows']['f1']:.6f} | {calibration_f1opt['answers']['f1']:.6f} |",
        "",
        f"语义门在 calibration 新增 TP {summary['calibration_additions']['tp']}、FP {summary['calibration_additions']['fp']}。",
        "",
        "未改正式 baseline，未打开 official test。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(OUT / "complete.json", {
        "status": "complete_development_only",
        "summary_sha256": sha(OUT / "summary.json"),
        "scores_sha256": sha(scores_path), "report_sha256": sha(OUT / "REPORT.md"),
        "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("SEMANTIC_ATTRIBUTION_SCORE_EVALUATED",
          strict["windows"]["f1"], strict["answers"]["f1"], flush=True)


def status() -> None:
    payload = {
        "prepared": (OUT / "PREPARATION.json").exists(),
        "cpu_check": (OUT / "CPU_SELFCHECK.json").exists(),
        "attribution_complete": ATTR_MANIFEST_PATH.exists(),
        "fit_nli_linked": (OUT / "selected_nli_links_fit.jsonl").exists(),
        "fit_frozen": (OUT / "fit_complete.json").exists(),
        "evaluated": (OUT / "summary.json").exists(),
        "formal_baselines_modified": False, "official_test_opened": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(
        "prepare", "cpu-check", "link-nli", "infer-nli", "fit",
        "evaluate", "status",
    ))
    parser.add_argument("--partition", choices=tuple(EXPECTED_ANSWERS),
                        default="fit")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "cpu-check":
        cpu_check()
    elif args.stage == "link-nli":
        link_nli(args.partition)
    elif args.stage == "infer-nli":
        infer_missing_nli(args.partition)
    elif args.stage == "fit":
        assert args.partition == "fit"
        fit()
    elif args.stage == "evaluate":
        assert args.partition == "fit", "evaluate has a fixed calibration target"
        evaluate()
    else:
        status()


if __name__ == "__main__":
    main()
