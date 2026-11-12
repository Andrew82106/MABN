"""Exact attention-top3 union per-passage-BM25-top2 NLI candidate pipeline.

CPU preparation is label blind.  It binds exact source-sentence instances to
the frozen atomic microclaim hypotheses, forms the requested union, and audits
both occurrence-bound cache hits and portable exact-pair reuse.  The original
634 fit and 159 calibration answers can be prepared immediately.  A separate
expanded-fit cohort becomes available only after its 3,046-record attribution
manifest is complete.

This runner never addresses official-test data or baseline artifacts and has
no detector-training command.  ``gpu-smoke`` and ``extract`` only fill frozen
NLI probabilities for exact pairs missing from existing caches.  ``score`` is
a CPU evidence-coverage diagnostic; the fit invocation freezes the prescribed
union/readout rule and calibration can only apply that frozen rule.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_atomic_microclaim_nli_v1 as atomic_nli  # noqa: E402
import run_retrieved_evidence_nli_v1 as retrieved_nli  # noqa: E402


VERSION = "attention-bm25-exact-evidence-union-nli-v2"
OUT = ROOT / "results/evidence_union_nli_v2"

NATIVE_ATOMIC_OUT = ROOT / "results/atomic_microclaim_nli_v1"
NATIVE_ROWS = NATIVE_ATOMIC_OUT / "inputs.jsonl"
NATIVE_PAIRS = NATIVE_ATOMIC_OUT / "pair_scores"
NATIVE_ATTR_OUT = ROOT / "results/semantic_source_attribution_v1"
NATIVE_ATTR_LAYOUTS = NATIVE_ATTR_OUT / "layouts.jsonl"
NATIVE_ATTR_MANIFEST = NATIVE_ATTR_OUT / "feature_manifest.json"
NATIVE_SEMANTIC_SCORE_OUT = ROOT / "results/semantic_source_attribution_v1_score"

EXPANDED_ATTR_OUT = ROOT / "results/semantic_source_attribution_expanded_fit_v1"
EXPANDED_ATTR_LAYOUTS = EXPANDED_ATTR_OUT / "layouts.jsonl"
EXPANDED_ATTR_UNITS = EXPANDED_ATTR_OUT / "semantic_units.jsonl"
EXPANDED_ATTR_MANIFEST = EXPANDED_ATTR_OUT / "feature_manifest.json"
EXPANDED_RAW_CLAIMS = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"

MODEL = ROOT.parent / "models/ModernBERT-base-nli"
MODEL_ID = "tasksource/ModernBERT-base-nli"
MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"
CLASSES = ("entailment", "neutral", "contradiction")

ATTENTION_TOP_K = 3
BM25_TOP_K = 2
PASSAGE_IDS = (1, 2, 3)
PAIR_LIMIT = 2048
CHUNK_SIZE = 4096

COHORTS = ("fit_native", "calibration", "fit_expanded")
EXPECTED = {
    "fit_native": {"answers": 634, "claims": 9055, "partition": "fit"},
    "calibration": {"answers": 159, "claims": 2267, "partition": "calibration"},
    "fit_expanded": {"answers": 3046, "claims": 25864, "partition": "fit"},
}


def aggregate_feature_names() -> tuple[str, ...]:
    names = []
    for scope in ("union", "attention", "bm25"):
        names += [
            f"{scope}__candidate_count",
            f"{scope}__max_entailment", f"{scope}__max_neutral",
            f"{scope}__max_contradiction",
            f"{scope}__mean_entailment", f"{scope}__mean_neutral",
            f"{scope}__mean_contradiction",
            f"{scope}__lack_of_entailment",
            f"{scope}__or_risk",
            f"{scope}__max_entailment_minus_max_contradiction",
            f"{scope}__best_entailment_class_margin",
            f"{scope}__best_contradiction_class_margin",
        ]
    names += [
        "union_gain_max_entailment_over_attention",
        "union_gain_max_entailment_over_bm25",
        "union_gain_max_contradiction_over_attention",
        "union_gain_max_contradiction_over_bm25",
        "attention_covers_union_best_entailment",
        "bm25_covers_union_best_entailment",
        "attention_covers_union_best_contradiction",
        "bm25_covers_union_best_contradiction",
    ]
    return tuple(names)


AGGREGATE_FEATURE_NAMES = aggregate_feature_names()


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


def request_id(premise: str, hypothesis: str) -> str:
    return digest({
        "checkpoint": MODEL_ID, "revision": MODEL_REVISION,
        "premise": premise, "hypothesis": hypothesis,
    })


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
        assert read_json(path) == value, f"Frozen JSON changed: {path}"
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
        assert sha(path) == sha(pending), f"Frozen JSONL changed: {path}"
        pending.unlink()
    else:
        pending.replace(path)


def frozen_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            assert set(old.files) == set(arrays), f"Frozen NPZ keys changed: {path}"
            for key, value in arrays.items():
                assert np.array_equal(old[key], value, equal_nan=True), (path, key)
        return
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "CPU stage must not initialize CUDA"


def model_binding() -> dict:
    assert atomic_nli.MODEL_REVISION == retrieved_nli.MODEL_REVISION == MODEL_REVISION
    assert atomic_nli.MODEL_SHA256 == retrieved_nli.MODEL_SHA256 == MODEL_SHA256
    manifest = read_json(MODEL / "download_manifest.json")
    assert manifest["revision"] == MODEL_REVISION
    assert sha(MODEL / "model.safetensors") == MODEL_SHA256
    return {
        "checkpoint": MODEL_ID, "revision": MODEL_REVISION,
        "weights_sha256": MODEL_SHA256, "class_order": list(CLASSES),
        "frozen": True, "maximum_pair_tokens": PAIR_LIMIT,
        "truncation": False,
    }


def protocol() -> dict:
    return {
        "version": VERSION,
        "role": "new exact-evidence NLI candidate; formal baselines remain read-only",
        "candidate_rule": {
            "identity": "A candidate is one exact (passage_id, sentence_id) source-sentence instance.",
            "attention": "Top min(3,N) source-sentence instances by the arithmetic mean of the exact float32 32-layer sentence relevance; stable ties use sentence_index.",
            "bm25": "Top min(2,N_passage) exact sentences independently in passages 1, 2 and 3, using the frozen Unicode BM25 implementation.",
            "union": "Set union of attention top3 and the three per-passage BM25 top2 sets. Canonical output order is sentence_index.",
            "nli": model_binding(),
        },
        "cache_reuse": {
            "native_occurrence": "For the existing 634+159 rows, use the exact probability bound to that answer/claim/sentence occurrence in atomic NLI or the completed attention-top3 cache.",
            "portable": "For another occurrence, reuse an exact checkpoint+revision+premise+hypothesis request only if every existing cached occurrence is bit-identical. Conflicting FP32 repeats are missing and must be recomputed.",
        },
        "cohorts": {
            "fit_native": "Existing 634 fit answers and 9,055 scored atomic microclaims.",
            "calibration": "Existing 159 calibration answers and 2,267 scored atomic microclaims.",
            "fit_expanded": "Optional 3,046 additional fit answers and 25,864 scored microclaims; attach only after its attribution manifest is complete.",
            "official_test": "No path or loader exists in this runner.",
        },
        "score": {
            "outputs": "Per-claim candidate count and best-entailment/best-contradiction coverage by attention, BM25, overlap and union.",
            "aggregate_interface": f"A label-free {len(AGGREGATE_FEATURE_NAMES)}-column float32 matrix supplies union/attention/BM25 max and mean E/N/C, class margins, lack-of-entailment, OR-risk, union gains and best-score coverage flags.",
            "fit_rule": "The prescribed union and per-class maximum/tie rule are frozen by the fit_native score command.",
            "calibration": "Calibration score refuses to run before the fit rule freeze and reports diagnostics without selecting or changing a rule.",
            "training": "No detector/model training is implemented.",
        },
        "stage_gate": "initialize -> prepare fit_native/calibration -> score fit_native -> score calibration; later attach fit_expanded -> gpu-smoke -> extract -> score",
        "labels_used_by_prepare_or_score": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def self_test() -> dict:
    attention = {0, 2, 4}
    bm25 = {1, 2, 3, 4}
    assert sorted(attention | bm25) == [0, 1, 2, 3, 4]
    scalar = np.asarray([0.2, 0.7, 0.7, 0.1], dtype=np.float32)
    order = np.lexsort((np.arange(len(scalar)), -scalar))[:3]
    assert order.tolist() == [1, 2, 0]
    one = np.asarray([0.1, 0.2, 0.7], dtype=np.float32)
    same = one.copy()
    changed = one.copy(); changed[0] = np.nextafter(changed[0], np.float32(1))
    assert np.array_equal(one, same) and not np.array_equal(one, changed)
    assert len(AGGREGATE_FEATURE_NAMES) == 44
    return {
        "status": "passed", "instance_union": True,
        "stable_attention_tie": True, "bit_exact_cache_conflict": True,
        "model": model_binding(), "model_loaded": False, "GPU_used": False,
        "labels_used": False, "official_test_opened": False,
    }


def runbook() -> str:
    script = "prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py"
    python = "& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe'"
    return f"""# Evidence-union NLI v2 runbook

CPU work available now:

```powershell
{python} {script} prepare --cohort fit_native
{python} {script} prepare --cohort calibration
{python} {script} score --cohort fit_native
{python} {script} score --cohort calibration
```

After the separate 3,046-answer attribution extractor has written a complete
`semantic_source_attribution_expanded_fit_v1/feature_manifest.json`:

```powershell
{python} {script} prepare --cohort fit_expanded
{python} {script} gpu-smoke --cohort fit_expanded
{python} {script} extract --cohort fit_expanded
{python} {script} score --cohort fit_expanded
```

`extract` is resumable in immutable {CHUNK_SIZE:,}-request chunks.  Use
`--limit-chunks 1` for a bounded extraction invocation.  Neither `score` nor
any other command trains the final detector.
"""


def initialize() -> None:
    assert_cpu_only()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_json(OUT / "protocol.json", protocol())
    frozen_json(OUT / "CPU_SELFCHECK.json", self_test())
    frozen_text(OUT / "RUNBOOK.md", runbook())
    print("EVIDENCE_UNION_NLI_V2_INITIALIZED_CPU_ONLY", flush=True)


def stable_top3(arrays: dict[str, np.ndarray], claim_index: int) -> tuple[np.ndarray, np.ndarray]:
    sentence_relevance = arrays["sentence_relevance"][claim_index]
    assert sentence_relevance.ndim == 2 and sentence_relevance.shape[0] == 32
    scalar = sentence_relevance.mean(axis=0, dtype=np.float64).astype(np.float32)
    assert np.isfinite(scalar).all() and np.all(scalar >= 0)
    order = np.lexsort((np.arange(len(scalar)), -scalar))[:min(ATTENTION_TOP_K, len(scalar))]
    return order.astype(np.int32), scalar


def checked_attr_manifest(path: Path, cohort: str) -> dict:
    assert path.is_file(), f"Attribution manifest is not complete: {path}"
    manifest = read_json(path)
    expected = (EXPECTED[cohort] if cohort == "fit_expanded" else
                {"answers": (EXPECTED["fit_native"]["answers"] +
                             EXPECTED["calibration"]["answers"]),
                 "claims": (EXPECTED["fit_native"]["claims"] +
                            EXPECTED["calibration"]["claims"])})
    assert manifest["status"] == "complete"
    assert manifest["records_complete"] == expected["answers"]
    if "records_total" in manifest:
        assert manifest["records_total"] == expected["answers"]
    assert manifest["claims_complete"] == expected["claims"]
    assert len(manifest["files"]) == expected["answers"]
    assert manifest["labels_used"] is False
    assert manifest["official_test_opened"] is False
    if cohort == "fit_expanded":
        assert manifest["calibration_opened"] is False
    return manifest


def load_attr_arrays(out: Path, manifest: dict, layout: dict,
                     claims: list[dict], sentences: list[dict]) -> dict[str, np.ndarray]:
    rid = layout["response_id"]
    feature = out / "features" / f"{rid}.npz"
    metadata = feature.with_suffix(".json")
    commit = feature.with_suffix(".commit.json")
    expected = manifest["files"][rid]
    assert sha(feature) == expected["npz_sha256"]
    assert sha(metadata) == expected["metadata_sha256"]
    assert sha(commit) == expected["commit_sha256"]
    with np.load(feature, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    c, s = len(claims), len(sentences)
    assert arrays["claim_ids"].tolist() == [int(row["claim_id"]) for row in claims]
    assert arrays["microclaim_indices"].tolist() == [int(row["microclaim_index"]) for row in claims]
    assert arrays["sentence_indices"].tolist() == list(range(s))
    assert arrays["sentence_passage_ids"].tolist() == [int(row["passage_id"]) for row in sentences]
    assert arrays["sentence_ids"].tolist() == [int(row["sentence_id"]) for row in sentences]
    assert arrays["sentence_relevance"].shape == (c, 32, s)
    assert arrays["claim_indptr"].tolist() == (np.arange(c + 1) * s).tolist()
    assert arrays["claim_sentence_mass"].shape == (c * s, 1024)
    assert [row.tobytes().hex() for row in arrays["sentence_identity_sha256"]] == [
        row["text_sha256"] for row in sentences]
    return arrays


def load_semantic_selected(partition: str) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_manifest_{partition}.json"
    existing_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_existing_{partition}.npz"
    manifest = read_json(manifest_path)
    assert manifest["existing_npz_sha256"] == sha(existing_path)
    with np.load(existing_path, allow_pickle=False) as loaded:
        selected = loaded["selected_sentence_indices"].copy()
        probability = loaded["probabilities"].copy()
        identities = loaded["request_identity_sha256"].copy()
    if np.isnan(probability).any():
        score_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_missing_scores_{partition}.npz"
        score_meta = read_json(score_path.with_suffix(".json"))
        assert score_meta["checkpoint"] == MODEL_ID
        assert score_meta["revision"] == MODEL_REVISION
        assert score_meta["model_sha256"] == MODEL_SHA256
        assert score_meta["npz_sha256"] == sha(score_path)
        with np.load(score_path, allow_pickle=False) as loaded:
            mapping = {key.tobytes(): value.copy() for key, value in zip(
                loaded["request_identity_sha256"], loaded["probabilities"])}
        for claim, rank in zip(*np.where(np.isnan(probability).any(axis=2))):
            probability[claim, rank] = mapping[identities[claim, rank].tobytes()]
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=2), 1, rtol=0, atol=2e-6)
    return selected, probability


class CacheCatalog:
    def __init__(self):
        self.values: dict[str, np.ndarray] = {}
        self.ambiguous: set[str] = set()
        self.sources: dict[str, set[str]] = defaultdict(set)
        self.occurrences: Counter = Counter()
        self.source_occurrences: Counter = Counter()

    def add(self, key: str, value: np.ndarray, source: str) -> None:
        value = np.asarray(value, dtype=np.float32)
        assert value.shape == (3,) and np.isfinite(value).all()
        assert np.isclose(value.sum(), 1, rtol=0, atol=2e-6)
        self.sources[key].add(source)
        self.occurrences[key] += 1
        self.source_occurrences[source] += 1
        if key in self.ambiguous:
            return
        if key not in self.values:
            self.values[key] = value.copy()
        elif not np.array_equal(self.values[key], value):
            del self.values[key]
            self.ambiguous.add(key)

    def status(self, key: str) -> str:
        if key in self.ambiguous:
            return "portable_conflict_recompute"
        if key in self.values:
            return "portable_exact_hit"
        return "portable_missing"


def fast_atomic_cache(cache: Path, row: dict, complete: dict,
                      input_sha256: str, protocol_sha256: str
                      ) -> tuple[np.ndarray, list[tuple], list[tuple]]:
    """Validate one atomic shard without rehashing the 43 MB input file."""
    pairs, owners = atomic_nli.row_pairs(row)
    with np.load(cache, allow_pickle=False) as loaded:
        expected = {
            "response_id", "input_row_sha256", "pair_definition_sha256",
            "input_file_sha256", "protocol_sha256", "model_sha256",
            "execution_signature_sha256", "execution_backend", "method_role",
            "probabilities", "claim_ids", "passage_ids", "pair_kinds",
            "sentence_a", "sentence_b", "reused_old_pair",
        }
        assert set(loaded.files) == expected
        assert loaded["response_id"].item() == row["response_id"]
        assert loaded["input_row_sha256"].item() == digest(row)
        assert loaded["pair_definition_sha256"].item() == digest(
            {"pairs": pairs, "owners": owners})
        assert loaded["input_file_sha256"].item() == input_sha256
        assert loaded["protocol_sha256"].item() == protocol_sha256
        assert loaded["model_sha256"].item() == MODEL_SHA256
        assert loaded["execution_signature_sha256"].item() == complete["execution_signature_sha256"]
        assert loaded["execution_backend"].item() == atomic_nli.BACKEND
        assert loaded["method_role"].item() == atomic_nli.ROLE
        probability = loaded["probabilities"].copy()
        assert loaded["claim_ids"].tolist() == [int(x[0]) for x in owners]
        assert loaded["passage_ids"].tolist() == [int(x[1]) for x in owners]
        assert loaded["pair_kinds"].tolist() == [int(x[2]) for x in owners]
        assert loaded["sentence_a"].tolist() == [int(x[3]) for x in owners]
        assert loaded["sentence_b"].tolist() == [int(x[4]) for x in owners]
    assert probability.shape == (len(pairs), 3) and probability.dtype == np.float32
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
    return probability, pairs, owners


def existing_cache_catalog() -> tuple[CacheCatalog, dict, dict[str, dict]]:
    catalog = CacheCatalog()
    direct_by_response = {}
    rows = json_lines(NATIVE_ROWS)
    complete = read_json(NATIVE_ATOMIC_OUT / "extraction_complete.json")
    assert complete["status"] == "complete_frozen_probabilities_not_scored"
    assert complete["model_sha256"] == MODEL_SHA256
    input_sha256 = sha(NATIVE_ROWS)
    protocol_sha256 = sha(NATIVE_ATOMIC_OUT / "protocol.json")
    assert complete["input_sha256"] == input_sha256
    assert complete["protocol_sha256"] == protocol_sha256
    for row in rows:
        cache = NATIVE_PAIRS / f"{row['response_id']}.npz"
        assert complete["files_sha256"][cache.name] == sha(cache)
        probabilities, pairs, owners = fast_atomic_cache(
            cache, row, complete, input_sha256, protocol_sha256)
        direct = {}
        for pair, owner, probability in zip(pairs, owners, probabilities):
            if int(owner[2]) == 0:  # joint top-2 evidence is outside this exact-sentence candidate
                continue
            catalog.add(request_id(pair[0], pair[1]), probability,
                        "atomic_microclaim_nli_v1_individual")
            key = (int(owner[0]), int(owner[1]), int(owner[3]))
            assert key not in direct
            direct[key] = probability.copy()
        direct_by_response[row["response_id"]] = direct
    semantic_inputs = {}
    for partition in ("fit", "calibration"):
        requests_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_requests_{partition}.jsonl"
        scores_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_missing_scores_{partition}.npz"
        link_manifest_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_manifest_{partition}.json"
        existing_path = NATIVE_SEMANTIC_SCORE_OUT / f"selected_nli_existing_{partition}.npz"
        metadata = read_json(scores_path.with_suffix(".json"))
        link_manifest = read_json(link_manifest_path)
        assert link_manifest["requests_sha256"] == sha(requests_path)
        assert link_manifest["existing_npz_sha256"] == sha(existing_path)
        assert metadata["checkpoint"] == MODEL_ID
        assert metadata["revision"] == MODEL_REVISION
        assert metadata["model_sha256"] == MODEL_SHA256
        assert metadata["request_file_sha256"] == sha(requests_path)
        assert metadata["npz_sha256"] == sha(scores_path)
        requests = json_lines(requests_path)
        with np.load(scores_path, allow_pickle=False) as loaded:
            identities = loaded["request_identity_sha256"]
            probabilities = loaded["probabilities"]
        assert len(requests) == len(identities) == len(probabilities)
        for row, identity, probability in zip(requests, identities, probabilities):
            assert row["request_id"] == identity.tobytes().hex()
            assert row["request_id"] == request_id(row["premise"], row["hypothesis"])
            catalog.add(row["request_id"], probability,
                        f"semantic_top3_missing_{partition}")
        semantic_inputs[partition] = {
            "link_manifest_sha256": sha(link_manifest_path),
            "existing_probabilities_sha256": sha(existing_path),
            "missing_requests_sha256": sha(requests_path),
            "missing_probabilities_sha256": sha(scores_path),
            "missing_metadata_sha256": sha(scores_path.with_suffix(".json")),
        }
    audit = {
        "status": "complete_existing_cache_catalog",
        "unique_requests_seen": len(catalog.occurrences),
        "portable_exact_requests": len(catalog.values),
        "portable_conflicting_requests": len(catalog.ambiguous),
        "cached_occurrences": int(sum(catalog.occurrences.values())),
        "source_occurrences": dict(sorted(catalog.source_occurrences.items())),
        "conflict_occurrences": int(sum(catalog.occurrences[key] for key in catalog.ambiguous)),
        "reuse_policy": "bit-identical across every cached occurrence",
        "source_binding": {
            "atomic_input_sha256": input_sha256,
            "atomic_protocol_sha256": protocol_sha256,
            "atomic_extraction_sha256": sha(NATIVE_ATOMIC_OUT / "extraction_complete.json"),
            "semantic_top3": semantic_inputs,
        },
        "model": model_binding(), "model_loaded": False, "GPU_used": False,
        "labels_used": False, "official_test_opened": False,
    }
    return catalog, audit, direct_by_response


def native_rows_layouts(cohort: str):
    partition = EXPECTED[cohort]["partition"]
    rows = [row for row in json_lines(NATIVE_ROWS) if row["partition"] == partition]
    layouts = [row for row in json_lines(NATIVE_ATTR_LAYOUTS) if row["partition"] == partition]
    assert len(rows) == len(layouts) == EXPECTED[cohort]["answers"]
    return rows, layouts


def passages_from_sentences(sentences: list[dict]) -> list[dict]:
    output = []
    for passage_id in PASSAGE_IDS:
        selected = [dict(row) for row in sentences if int(row["passage_id"]) == passage_id]
        selected.sort(key=lambda row: int(row["sentence_id"]))
        assert selected and [int(row["sentence_id"]) for row in selected] == list(range(len(selected)))
        output.append({"passage_id": passage_id, "sentences": selected})
    return output


def bm25_map(hypothesis: str, passages: list[dict], recorded=None) -> dict[tuple[int, int], dict]:
    rebuilt = [retrieved_nli.select_evidence(hypothesis, passage) for passage in passages]
    if recorded is not None:
        assert len(recorded) == len(rebuilt) == 3
        for old, new in zip(recorded, rebuilt):
            assert int(old["passage_id"]) == int(new["passage_id"])
            assert [(int(x["rank"]), int(x["sentence_id"]), float(x["bm25"]),
                     float(x["query_term_coverage"]), x["evidence_sha256"])
                    for x in old["selected"]] == [
                   (int(x["rank"]), int(x["sentence_id"]), float(x["bm25"]),
                    float(x["query_term_coverage"]), x["evidence_sha256"])
                    for x in new["selected"]]
    output = {}
    for passage in rebuilt:
        for selected in passage["selected"]:
            key = (int(passage["passage_id"]), int(selected["sentence_id"]))
            assert key not in output
            output[key] = selected
    return output


def native_sources(cohort: str, attr_manifest: dict,
                   direct_by_response: dict[str, dict]):
    rows, layouts = native_rows_layouts(cohort)
    selected, attention_probability = load_semantic_selected(EXPECTED[cohort]["partition"])
    claim_cursor = 0
    for row, layout in zip(rows, layouts):
        assert row["response_id"] == layout["response_id"]
        sentences = []
        text_lookup = {(int(p["passage_id"]), int(s["sentence_id"])): s
                       for p in row["passages"] for s in p["sentences"]}
        for sentence in layout["sentences"]:
            key = (int(sentence["passage_id"]), int(sentence["sentence_id"]))
            text = text_lookup[key]
            assert text["text_sha256"] == sentence["text_sha256"]
            sentences.append({**sentence, "text": text["text"]})
        arrays = load_attr_arrays(NATIVE_ATTR_OUT, attr_manifest, layout,
                                  row["claims"], sentences)
        direct_atomic = direct_by_response[row["response_id"]]
        claims = []
        for local_claim, claim in enumerate(row["claims"]):
            chosen, scalar = stable_top3(arrays, local_claim)
            assert selected[claim_cursor].tolist() == chosen.tolist()
            direct_attention = {
                int(sentence_index): attention_probability[claim_cursor, rank].copy()
                for rank, sentence_index in enumerate(chosen)
            }
            passages = row["passages"]
            claims.append({
                "claim": claim, "sentences": sentences, "passages": passages,
                "attention_indices": chosen, "attention_scalar": scalar,
                "bm25": bm25_map(claim["hypothesis"], passages, claim["retrieval"]),
                "direct_atomic": direct_atomic,
                "direct_attention": direct_attention,
            })
            claim_cursor += 1
        yield row, layout, claims
    assert claim_cursor == EXPECTED[cohort]["claims"]


def expanded_raw_claims(response_ids: set[str]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in json_lines(EXPANDED_RAW_CLAIMS):
        if row["response_id"] in response_ids:
            grouped[row["response_id"]][row["microclaim_id"]] = row
    assert set(grouped) == response_ids
    return grouped


def expanded_sources(attr_manifest: dict):
    layouts = json_lines(EXPANDED_ATTR_LAYOUTS)
    units = json_lines(EXPANDED_ATTR_UNITS)
    assert len(layouts) == len(units) == EXPECTED["fit_expanded"]["answers"]
    ids = {row["response_id"] for row in units}
    assert len(ids) == len(units)
    raw = expanded_raw_claims(ids)
    total = 0
    for unit, layout in zip(units, layouts):
        rid = unit["response_id"]
        assert rid == layout["response_id"]
        assert unit["partition"] == layout["partition"] == "fit"
        assert unit["labels_used"] is False and layout["labels_used"] is False
        sentences = unit["sentences"]
        claims = unit["claims"]
        arrays = load_attr_arrays(EXPANDED_ATTR_OUT, attr_manifest, layout,
                                  claims, sentences)
        passages = passages_from_sentences(sentences)
        packed = []
        for local_claim, unit_claim in enumerate(claims):
            claim = raw[rid][unit_claim["microclaim_id"]]
            assert claim["text"] == unit_claim["text"]
            hypothesis = atomic_nli.hypothesis_for(claim)
            chosen, scalar = stable_top3(arrays, local_claim)
            packed.append({
                "claim": {**unit_claim, "hypothesis": hypothesis,
                          "hypothesis_sha256": digest(hypothesis)},
                "sentences": sentences, "passages": passages,
                "attention_indices": chosen, "attention_scalar": scalar,
                "bm25": bm25_map(hypothesis, passages),
                "direct_atomic": {}, "direct_attention": {},
            })
            total += 1
        yield unit, layout, packed
    assert total == EXPECTED["fit_expanded"]["claims"]


def prepare(cohort: str) -> None:
    assert_cpu_only()
    assert cohort in COHORTS
    assert read_json(OUT / "protocol.json") == protocol()
    model_binding()
    if cohort == "fit_expanded":
        attr_path, attr_out = EXPANDED_ATTR_MANIFEST, EXPANDED_ATTR_OUT
    else:
        attr_path, attr_out = NATIVE_ATTR_MANIFEST, NATIVE_ATTR_OUT
    attr_manifest = checked_attr_manifest(attr_path, cohort)
    catalog, catalog_audit, direct_by_response = existing_cache_catalog()
    frozen_json(OUT / "existing_cache_catalog_audit.json", catalog_audit)
    source_iterator = (expanded_sources(attr_manifest) if cohort == "fit_expanded"
                       else native_sources(cohort, attr_manifest,
                                           direct_by_response))

    output_rows, missing = [], {}
    indptr = [0]
    probabilities, attention_ranks, bm25_ranks = [], [], []
    sentence_indices, passage_ids, sentence_ids, identities = [], [], [], []
    candidate_hist = Counter()
    hit_hist = Counter()
    portable_hist = Counter()
    overlap_total = 0
    claim_total = answer_total = 0
    request_occurrences = Counter()
    for answer_index, (source, layout, packed_claims) in enumerate(source_iterator):
        answer_claims = []
        for one in packed_claims:
            claim = one["claim"]
            hypothesis = claim["hypothesis"]
            sentences = one["sentences"]
            by_index = {int(row["sentence_index"]): row for row in sentences}
            by_key = {(int(row["passage_id"]), int(row["sentence_id"])):
                      int(row["sentence_index"]) for row in sentences}
            attention = {int(value): rank for rank, value in
                         enumerate(one["attention_indices"], 1)}
            bm25 = {by_key[key]: value for key, value in one["bm25"].items()}
            union = sorted(set(attention) | set(bm25))
            overlap_total += len(set(attention) & set(bm25))
            candidate_hist[len(union)] += 1
            candidates = []
            direct_hits = portable_hits = portable_conflicts = 0
            for sentence_index in union:
                sentence = by_index[sentence_index]
                pid, sid = int(sentence["passage_id"]), int(sentence["sentence_id"])
                premise = sentence["text"]
                assert digest(premise) == sentence["text_sha256"]
                identity = request_id(premise, hypothesis)
                request_occurrences[identity] += 1
                portable = catalog.status(identity)
                portable_hist[portable] += 1
                value = None
                sources = []
                atomic_key = (int(claim["claim_id"]), pid, sid)
                if atomic_key in one["direct_atomic"]:
                    value = one["direct_atomic"][atomic_key].copy()
                    sources.append("atomic_microclaim_nli_v1")
                if sentence_index in one["direct_attention"]:
                    attention_value = one["direct_attention"][sentence_index]
                    if value is not None:
                        assert np.array_equal(value, attention_value), (
                            source["response_id"], claim["claim_id"], sentence_index)
                    else:
                        value = attention_value.copy()
                    sources.append("semantic_source_attribution_v1_score")
                if value is None and portable == "portable_exact_hit":
                    value = catalog.values[identity].copy()
                    sources.append("portable_exact_existing_cache")
                if value is None:
                    probability = np.full(3, np.nan, dtype=np.float32)
                    hit_hist["missing"] += 1
                    if portable == "portable_conflict_recompute":
                        portable_conflicts += 1
                    missing.setdefault(identity, {
                        "request_id": identity, "premise": premise,
                        "hypothesis": hypothesis,
                        "premise_sha256": digest(premise),
                        "hypothesis_sha256": digest(hypothesis),
                        "reason": portable,
                    })
                else:
                    probability = value
                    direct_hits += int(any(x != "portable_exact_existing_cache" for x in sources))
                    portable_hits += int(sources == ["portable_exact_existing_cache"])
                    hit_hist["existing"] += 1
                bm = bm25.get(sentence_index)
                candidate = {
                    "sentence_index": sentence_index, "passage_id": pid,
                    "sentence_id": sid, "sentence_sha256": sentence["text_sha256"],
                    "attention_rank": attention.get(sentence_index),
                    "attention_scalar": (float(one["attention_scalar"][sentence_index])
                                         if sentence_index in attention else None),
                    "bm25_rank": (int(bm["rank"]) if bm is not None else None),
                    "bm25": (float(bm["bm25"]) if bm is not None else None),
                    "query_term_coverage": (float(bm["query_term_coverage"])
                                            if bm is not None else None),
                    "selected_by": (["attention"] if sentence_index in attention else [])
                                   + (["bm25"] if sentence_index in bm25 else []),
                    "request_id": identity, "cache_status": (
                        "existing_occurrence_or_exact" if value is not None else "missing"),
                    "cache_sources": sources,
                    "portable_reuse_status": portable,
                }
                candidates.append(candidate)
                probabilities.append(probability)
                attention_ranks.append(attention.get(sentence_index, 0))
                bm25_ranks.append(int(bm["rank"]) if bm is not None else 0)
                sentence_indices.append(sentence_index)
                passage_ids.append(pid); sentence_ids.append(sid)
                identities.append(np.frombuffer(bytes.fromhex(identity), dtype=np.uint8))
            indptr.append(indptr[-1] + len(candidates))
            answer_claims.append({
                "global_claim_index": claim_total,
                "claim_id": int(claim["claim_id"]),
                "microclaim_id": claim["microclaim_id"],
                "microclaim_index": int(claim["microclaim_index"]),
                "hypothesis_sha256": digest(hypothesis),
                "candidate_count": len(candidates),
                "attention_count": len(attention), "bm25_count": len(bm25),
                "attention_bm25_overlap": len(set(attention) & set(bm25)),
                "existing_cache_hits": len(candidates) - sum(
                    row["cache_status"] == "missing" for row in candidates),
                "missing_cache_candidates": sum(
                    row["cache_status"] == "missing" for row in candidates),
                "direct_occurrence_hits": direct_hits,
                "portable_existing_hits": portable_hits,
                "portable_conflicts_requiring_recompute": portable_conflicts,
                "candidates": candidates,
            })
            claim_total += 1
        output_rows.append({
            "answer_index": answer_index, "response_id": source["response_id"],
            "source_id": source["source_id"], "group_id": source["group_id"],
            "partition": source["partition"], "claims": answer_claims,
            "labels_used": False, "official_test_opened": False,
        })
        answer_total += 1
        if answer_total % 100 == 0:
            print("EVIDENCE_UNION_PREPARE", cohort, answer_total,
                  EXPECTED[cohort]["answers"], flush=True)

    assert answer_total == EXPECTED[cohort]["answers"]
    assert claim_total == EXPECTED[cohort]["claims"]
    candidates_path = OUT / f"candidates_{cohort}.jsonl"
    arrays_path = OUT / f"candidate_arrays_{cohort}.npz"
    requests_path = OUT / f"missing_requests_{cohort}.jsonl"
    frozen_jsonl(candidates_path, output_rows)
    frozen_jsonl(requests_path, [missing[key] for key in sorted(missing)])
    arrays = {
        "claim_indptr": np.asarray(indptr, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "attention_rank": np.asarray(attention_ranks, dtype=np.int8),
        "bm25_rank": np.asarray(bm25_ranks, dtype=np.int8),
        "sentence_index": np.asarray(sentence_indices, dtype=np.int32),
        "passage_id": np.asarray(passage_ids, dtype=np.int8),
        "sentence_id": np.asarray(sentence_ids, dtype=np.int32),
        "request_identity_sha256": np.stack(identities),
    }
    frozen_npz(arrays_path, arrays)
    source_snapshot = {
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "attribution_manifest_sha256": sha(attr_path),
        "attribution_layouts_sha256": sha(EXPANDED_ATTR_LAYOUTS if cohort == "fit_expanded" else NATIVE_ATTR_LAYOUTS),
        "atomic_input_sha256": sha(NATIVE_ROWS),
        "atomic_extraction_sha256": sha(NATIVE_ATOMIC_OUT / "extraction_complete.json"),
        "existing_cache_catalog_audit_sha256": sha(OUT / "existing_cache_catalog_audit.json"),
        "model": model_binding(),
    }
    if cohort != "fit_expanded":
        partition = EXPECTED[cohort]["partition"]
        for stem in (
                f"selected_nli_manifest_{partition}.json",
                f"selected_nli_existing_{partition}.npz",
                f"selected_nli_requests_{partition}.jsonl",
                f"selected_nli_missing_scores_{partition}.npz",
                f"selected_nli_missing_scores_{partition}.json"):
            source_snapshot[f"semantic_top3_{stem}_sha256"] = sha(
                NATIVE_SEMANTIC_SCORE_OUT / stem)
    if cohort == "fit_expanded":
        source_snapshot.update({
            "semantic_units_sha256": sha(EXPANDED_ATTR_UNITS),
            "expanded_raw_claims_sha256": sha(EXPANDED_RAW_CLAIMS),
        })
    manifest = {
        "status": "CPU_manifest_complete_missing_NLI" if missing else "CPU_manifest_complete_all_candidates_cached",
        "version": VERSION, "cohort": cohort,
        "answers": answer_total, "claims": claim_total,
        "candidate_instances": len(probabilities),
        "candidate_count_per_claim_histogram": {str(k): v for k, v in sorted(candidate_hist.items())},
        "attention_bm25_overlap_instances": overlap_total,
        "existing_cache_hit_instances": int(hit_hist["existing"]),
        "missing_cache_instances": int(hit_hist["missing"]),
        "unique_candidate_requests": len(request_occurrences),
        "unique_missing_requests": len(missing),
        "portable_reuse_instance_status": dict(sorted(portable_hist.items())),
        "candidate_files": {
            "jsonl": {"path": candidates_path.name, "sha256": sha(candidates_path)},
            "arrays": {"path": arrays_path.name, "sha256": sha(arrays_path)},
            "missing_requests": {"path": requests_path.name, "sha256": sha(requests_path)},
        },
        "source_snapshot": source_snapshot,
        "model_loaded": False, "GPU_used": False, "labels_used": False,
        "rule_selected": False, "formal_baselines_modified": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / f"manifest_{cohort}.json", manifest)
    assert_cpu_only()
    print("EVIDENCE_UNION_MANIFEST_COMPLETE", cohort,
          json.dumps({key: manifest[key] for key in (
              "answers", "claims", "candidate_instances",
              "existing_cache_hit_instances", "missing_cache_instances",
              "unique_missing_requests")}), flush=True)


def runtime_signature(cohort: str) -> dict:
    return {
        "version": VERSION, "cohort": cohort,
        "runner_sha256": sha(Path(__file__)),
        "candidate_manifest_sha256": sha(OUT / f"manifest_{cohort}.json"),
        "request_file_sha256": sha(OUT / f"missing_requests_{cohort}.jsonl"),
        "model": model_binding(), "backend": atomic_nli.BACKEND,
        "batch": atomic_nli.INFERENCE_BATCH, "precision": "float32",
    }


def load_requests(cohort: str) -> list[dict]:
    manifest = read_json(OUT / f"manifest_{cohort}.json")
    path = OUT / f"missing_requests_{cohort}.jsonl"
    assert manifest["candidate_files"]["missing_requests"]["sha256"] == sha(path)
    rows = json_lines(path)
    assert len(rows) == manifest["unique_missing_requests"]
    assert [row["request_id"] for row in rows] == sorted(row["request_id"] for row in rows)
    return rows


def gpu_smoke(cohort: str) -> None:
    requests = load_requests(cohort)
    assert requests, f"{cohort} has no missing NLI requests; GPU smoke is unnecessary"
    signature = runtime_signature(cohort)
    selected = sorted(set([0, len(requests) // 2, len(requests) - 1]))
    pairs = [(requests[index]["premise"], requests[index]["hypothesis"])
             for index in selected]
    tokenizer = model = None
    started = time.perf_counter()
    try:
        tokenizer, model, loaded = atomic_nli.load_cuda()
        first, shapes = atomic_nli.infer_pairs(tokenizer, model, pairs,
                                               torch.device("cuda:0"))
        second, shapes2 = atomic_nli.infer_pairs(tokenizer, model, pairs,
                                                 torch.device("cuda:0"))
        assert shapes == shapes2 and np.array_equal(first, second)
        frozen_npz(OUT / f"GPU_SMOKE_{cohort}.npz", {
            "request_identity_sha256": np.stack([
                np.frombuffer(bytes.fromhex(requests[index]["request_id"]), dtype=np.uint8)
                for index in selected]),
            "probabilities": first,
        })
        frozen_json(OUT / f"GPU_SMOKE_{cohort}.json", {
            "status": "passed_no_automatic_extract", "cohort": cohort,
            "selected_request_indices": selected, "batch_shapes": shapes,
            "same_path_repeat_exact": True, "model_load": loaded,
            "runtime_signature": signature,
            "probabilities_sha256": sha(OUT / f"GPU_SMOKE_{cohort}.npz"),
            "seconds": time.perf_counter() - started,
            "trained": False, "GPU_used": True, "labels_used": False,
            "formal_baselines_modified": False, "official_test_opened": False,
        })
        print("EVIDENCE_UNION_GPU_SMOKE_PASSED", cohort, flush=True)
    finally:
        del model, tokenizer
        atomic_nli.clean_gpu()


def validate_chunk(path: Path, requests: list[dict], begin: int, end: int,
                   signature_sha256: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "request_identity_sha256", "probabilities", "begin", "end",
            "runtime_signature_sha256"}
        assert int(loaded["begin"]) == begin and int(loaded["end"]) == end
        assert loaded["runtime_signature_sha256"].item() == signature_sha256
        expected = [requests[index]["request_id"] for index in range(begin, end)]
        assert [row.tobytes().hex() for row in loaded["request_identity_sha256"]] == expected
        probability = loaded["probabilities"].copy()
    assert probability.shape == (end - begin, 3) and probability.dtype == np.float32
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
    return probability


def extract(cohort: str, limit_chunks: int | None) -> None:
    requests = load_requests(cohort)
    assert requests, f"{cohort} has no missing NLI requests; extraction is unnecessary"
    smoke = read_json(OUT / f"GPU_SMOKE_{cohort}.json")
    assert smoke["status"] == "passed_no_automatic_extract"
    signature = runtime_signature(cohort)
    assert smoke["runtime_signature"] == signature
    signature_sha256 = digest(signature)
    chunks = [(begin, min(begin + CHUNK_SIZE, len(requests)))
              for begin in range(0, len(requests), CHUNK_SIZE)]
    folder = OUT / f"nli_chunks_{cohort}"
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for begin, end in chunks:
        path = folder / f"{begin:07d}_{end:07d}.npz"
        if path.exists():
            validate_chunk(path, requests, begin, end, signature_sha256)
        else:
            pending.append((begin, end))
    if limit_chunks is not None:
        assert limit_chunks > 0
        pending = pending[:limit_chunks]
    tokenizer = model = None
    started = time.perf_counter()
    try:
        if pending:
            tokenizer, model, _ = atomic_nli.load_cuda()
            for completed, (begin, end) in enumerate(pending, 1):
                pairs = [(row["premise"], row["hypothesis"])
                         for row in requests[begin:end]]
                probability, _ = atomic_nli.infer_pairs(
                    tokenizer, model, pairs, torch.device("cuda:0"))
                path = folder / f"{begin:07d}_{end:07d}.npz"
                frozen_npz(path, {
                    "request_identity_sha256": np.stack([
                        np.frombuffer(bytes.fromhex(row["request_id"]), dtype=np.uint8)
                        for row in requests[begin:end]]),
                    "probabilities": probability,
                    "begin": np.asarray(begin, dtype=np.int64),
                    "end": np.asarray(end, dtype=np.int64),
                    "runtime_signature_sha256": np.asarray(signature_sha256),
                })
                validate_chunk(path, requests, begin, end, signature_sha256)
                print("EVIDENCE_UNION_EXTRACT_CHUNK", cohort, completed,
                      len(pending), begin, end, flush=True)
    finally:
        del model, tokenizer
        atomic_nli.clean_gpu()
    files, complete = {}, True
    for begin, end in chunks:
        path = folder / f"{begin:07d}_{end:07d}.npz"
        if not path.exists():
            complete = False
            continue
        validate_chunk(path, requests, begin, end, signature_sha256)
        files[path.name] = sha(path)
    atomic_json(OUT / f"extraction_{cohort}.json", {
        "status": "complete_frozen_probabilities" if complete else "partial_resumable",
        "cohort": cohort, "requests": len(requests), "chunks": len(chunks),
        "chunks_complete": len(files), "files_sha256": files,
        "runtime_signature": signature, "runtime_signature_sha256": signature_sha256,
        "seconds_this_invocation": time.perf_counter() - started,
        "trained": False, "GPU_used": bool(pending), "labels_used": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    })


def resolved_probabilities(cohort: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays_path = OUT / f"candidate_arrays_{cohort}.npz"
    manifest = read_json(OUT / f"manifest_{cohort}.json")
    assert manifest["candidate_files"]["arrays"]["sha256"] == sha(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    probability = arrays["probabilities"]
    if np.isnan(probability).any():
        requests = load_requests(cohort)
        extraction = read_json(OUT / f"extraction_{cohort}.json")
        assert extraction["status"] == "complete_frozen_probabilities"
        assert extraction["runtime_signature"] == runtime_signature(cohort)
        mapping = {}
        folder = OUT / f"nli_chunks_{cohort}"
        for filename, expected_sha in extraction["files_sha256"].items():
            path = folder / filename
            assert sha(path) == expected_sha
            with np.load(path, allow_pickle=False) as loaded:
                for key, value in zip(loaded["request_identity_sha256"],
                                      loaded["probabilities"]):
                    mapping[key.tobytes()] = value.copy()
        assert len(mapping) == len(requests)
        missing_rows = np.flatnonzero(np.isnan(probability).any(axis=1))
        for index in missing_rows:
            probability[index] = mapping[arrays["request_identity_sha256"][index].tobytes()]
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
    arrays["probabilities"] = probability
    return arrays, manifest


def basic_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()), "median": float(np.median(array)),
        "mean": float(array.mean()), "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def scope_aggregate(probability: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = probability[mask]
    assert selected.ndim == 2 and selected.shape[1] == 3 and len(selected) > 0
    maxima = selected.max(axis=0)
    means = selected.mean(axis=0, dtype=np.float64)
    best_entailment = selected[int(np.argmax(selected[:, 0]))]
    best_contradiction = selected[int(np.argmax(selected[:, 2]))]
    values = [
        float(len(selected)), *map(float, maxima), *map(float, means),
        float(1.0 - maxima[0]),
        float(max(maxima[2], 1.0 - maxima[0])),
        float(maxima[0] - maxima[2]),
        float(best_entailment[0] - max(best_entailment[1], best_entailment[2])),
        float(best_contradiction[2] - max(best_contradiction[0], best_contradiction[1])),
    ]
    assert len(values) == 12
    return np.asarray(values, dtype=np.float32)


def score(cohort: str) -> None:
    assert_cpu_only()
    if cohort == "calibration":
        freeze_path = OUT / "fit_rule_freeze.json"
        assert freeze_path.is_file(), "Run fit_native score to freeze the rule before calibration"
        freeze = read_json(freeze_path)
        assert freeze["status"] == "frozen_on_fit_before_calibration_diagnostic"
    arrays, manifest = resolved_probabilities(cohort)
    candidate_rows = json_lines(OUT / f"candidates_{cohort}.jsonl")
    assert len(candidate_rows) == EXPECTED[cohort]["answers"]
    indptr = arrays["claim_indptr"]
    probability = arrays["probabilities"]
    diagnostic_rows = []
    aggregate_rows = []
    aggregate_response_indices, aggregate_claim_ids = [], []
    aggregate_microclaim_indices, aggregate_hypothesis_ids = [], []
    aggregate = {
        name: {"attention_score_coverage": 0, "bm25_score_coverage": 0,
               "overlap_score_coverage": 0, "attention_canonical_winner": 0,
               "bm25_canonical_winner": 0, "union_gain_over_attention": [],
               "union_gain_over_bm25": []}
        for name in ("best_entailment", "best_contradiction")
    }
    claim_cursor = 0
    for answer_index, answer in enumerate(candidate_rows):
        claims = []
        for claim in answer["claims"]:
            left, right = map(int, indptr[[claim_cursor, claim_cursor + 1]])
            local = probability[left:right]
            attention = arrays["attention_rank"][left:right] > 0
            bm25 = arrays["bm25_rank"][left:right] > 0
            assert len(local) == claim["candidate_count"]
            output = {
                "global_claim_index": claim_cursor,
                "claim_id": claim["claim_id"], "microclaim_id": claim["microclaim_id"],
                "candidate_count": len(local),
            }
            scope_values = {
                "union": scope_aggregate(local, np.ones(len(local), dtype=bool)),
                "attention": scope_aggregate(local, attention),
                "bm25": scope_aggregate(local, bm25),
            }
            max_e = {key: float(value[1]) for key, value in scope_values.items()}
            max_c = {key: float(value[3]) for key, value in scope_values.items()}
            coverage_flags = {}
            for label, class_index in (("best_entailment", 0),
                                       ("best_contradiction", 2)):
                values = local[:, class_index]
                winner = int(np.argmax(values))
                union_best = float(values[winner])
                attention_best = float(values[attention].max())
                bm25_best = float(values[bm25].max())
                overlap = attention & bm25
                overlap_best = float(values[overlap].max()) if overlap.any() else None
                attention_coverage = attention_best == union_best
                bm25_coverage = bm25_best == union_best
                overlap_coverage = overlap_best == union_best if overlap_best is not None else False
                aggregate[label]["attention_score_coverage"] += int(attention_coverage)
                aggregate[label]["bm25_score_coverage"] += int(bm25_coverage)
                aggregate[label]["overlap_score_coverage"] += int(overlap_coverage)
                aggregate[label]["attention_canonical_winner"] += int(attention[winner])
                aggregate[label]["bm25_canonical_winner"] += int(bm25[winner])
                aggregate[label]["union_gain_over_attention"].append(union_best - attention_best)
                aggregate[label]["union_gain_over_bm25"].append(union_best - bm25_best)
                candidate = claim["candidates"][winner]
                output[label] = {
                    "probability": union_best,
                    "candidate_offset": winner,
                    "sentence_index": candidate["sentence_index"],
                    "passage_id": candidate["passage_id"],
                    "sentence_id": candidate["sentence_id"],
                    "selected_by": candidate["selected_by"],
                    "attention_best": attention_best,
                    "bm25_best": bm25_best,
                    "overlap_best": overlap_best,
                    "attention_score_coverage": attention_coverage,
                    "bm25_score_coverage": bm25_coverage,
                    "overlap_score_coverage": overlap_coverage,
                }
                coverage_flags[label] = (attention_coverage, bm25_coverage)
            feature = np.concatenate([
                scope_values["union"], scope_values["attention"], scope_values["bm25"],
                np.asarray([
                    max_e["union"] - max_e["attention"],
                    max_e["union"] - max_e["bm25"],
                    max_c["union"] - max_c["attention"],
                    max_c["union"] - max_c["bm25"],
                    *coverage_flags["best_entailment"],
                    *coverage_flags["best_contradiction"],
                ], dtype=np.float32),
            ]).astype(np.float32, copy=False)
            assert feature.shape == (len(AGGREGATE_FEATURE_NAMES),)
            aggregate_rows.append(feature)
            aggregate_response_indices.append(answer_index)
            aggregate_claim_ids.append(int(claim["claim_id"]))
            aggregate_microclaim_indices.append(int(claim["microclaim_index"]))
            aggregate_hypothesis_ids.append(np.frombuffer(
                bytes.fromhex(claim["hypothesis_sha256"]), dtype=np.uint8))
            claims.append(output)
            claim_cursor += 1
        diagnostic_rows.append({
            "response_id": answer["response_id"], "partition": answer["partition"],
            "claims": claims,
        })
    assert claim_cursor == EXPECTED[cohort]["claims"]
    summary = {}
    for label, values in aggregate.items():
        summary[label] = {
            key: {"count": int(value), "fraction": float(value / claim_cursor)}
            for key, value in values.items() if not isinstance(value, list)
        }
        summary[label]["union_gain_over_attention"] = basic_stats(
            values["union_gain_over_attention"])
        summary[label]["union_gain_over_bm25"] = basic_stats(
            values["union_gain_over_bm25"])
    diagnostics_path = OUT / f"coverage_{cohort}.jsonl"
    aggregate_path = OUT / f"claim_aggregates_{cohort}.npz"
    feature_names_path = OUT / "aggregate_feature_names.json"
    frozen_jsonl(diagnostics_path, diagnostic_rows)
    frozen_json(feature_names_path, {
        "version": VERSION, "width": len(AGGREGATE_FEATURE_NAMES),
        "names": list(AGGREGATE_FEATURE_NAMES),
        "row_order": "answer order then claim_id; identities are stored beside the matrix",
        "probability_class_order": list(CLASSES),
        "labels_used": False, "official_test_opened": False,
    })
    frozen_npz(aggregate_path, {
        "features": np.vstack(aggregate_rows).astype(np.float32),
        "response_index": np.asarray(aggregate_response_indices, dtype=np.int32),
        "claim_id": np.asarray(aggregate_claim_ids, dtype=np.int32),
        "microclaim_index": np.asarray(aggregate_microclaim_indices, dtype=np.int32),
        "hypothesis_identity_sha256": np.stack(aggregate_hypothesis_ids),
        "feature_names_sha256": np.asarray(sha(feature_names_path)),
    })
    result = {
        "status": "fit_rule_frozen" if cohort == "fit_native" else "frozen_rule_diagnostic",
        "cohort": cohort, "answers": len(candidate_rows), "claims": claim_cursor,
        "candidate_count_per_claim_histogram": manifest["candidate_count_per_claim_histogram"],
        "coverage": summary,
        "coverage_file_sha256": sha(diagnostics_path),
        "aggregate_interface": {
            "width": len(AGGREGATE_FEATURE_NAMES),
            "features_sha256": sha(aggregate_path),
            "feature_names_sha256": sha(feature_names_path),
            "rows": claim_cursor,
        },
        "probability_source": "occurrence-bound frozen cache plus exact portable reuse plus separately extracted missing requests",
        "model": model_binding(), "labels_used": False, "model_trained": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    }
    if cohort == "fit_native":
        frozen_json(OUT / "fit_rule_freeze.json", {
            "status": "frozen_on_fit_before_calibration_diagnostic",
            "source_cohort": "fit_native",
            "candidate_rule": "attention_top3_union_bm25_top2_per_passage",
            "identity": "passage_id+sentence_id exact source-sentence instance",
            "aggregation": "per NLI class maximum across the union",
            "tie_break": "canonical sentence_index",
            "selection_note": "Rule prescribed before diagnostics; no calibration value or label selected it.",
            "fit_coverage_sha256": sha(diagnostics_path),
            "labels_used": False, "model_trained": False,
            "official_test_opened": False,
        })
    elif cohort == "calibration":
        result["fit_rule_freeze_sha256"] = sha(OUT / "fit_rule_freeze.json")
        result["rule_selected_or_changed_on_calibration"] = False
    else:
        if (OUT / "fit_rule_freeze.json").exists():
            result["fit_rule_freeze_sha256"] = sha(OUT / "fit_rule_freeze.json")
            result["rule_changed_by_expanded_fit_diagnostic"] = False
    frozen_json(OUT / f"diagnostics_{cohort}.json", result)
    if cohort == "calibration":
        fit = read_json(OUT / "diagnostics_fit_native.json")
        fit_manifest = read_json(OUT / "manifest_fit_native.json")
        def pct(value):
            return f"{100.0 * value:.2f}%"
        report = f"""# Exact evidence-union NLI v2: native CPU audit

The manifest contains the exact sentence-instance union of attention top 3 and
BM25 top 2 independently in each passage.  All {fit_manifest['candidate_instances']:,}
fit and {manifest['candidate_instances']:,} calibration candidate instances hit
their occurrence-bound frozen NLI caches; neither cohort has a missing request.

| Cohort | Claims | Candidate instances | Candidate count histogram |
|---|---:|---:|---|
| fit | {fit['claims']:,} | {fit_manifest['candidate_instances']:,} | {fit_manifest['candidate_count_per_claim_histogram']} |
| calibration | {result['claims']:,} | {manifest['candidate_instances']:,} | {manifest['candidate_count_per_claim_histogram']} |

| Coverage of union best | fit attention | fit BM25 | cal attention | cal BM25 |
|---|---:|---:|---:|---:|
| entailment | {pct(fit['coverage']['best_entailment']['attention_score_coverage']['fraction'])} | {pct(fit['coverage']['best_entailment']['bm25_score_coverage']['fraction'])} | {pct(result['coverage']['best_entailment']['attention_score_coverage']['fraction'])} | {pct(result['coverage']['best_entailment']['bm25_score_coverage']['fraction'])} |
| contradiction | {pct(fit['coverage']['best_contradiction']['attention_score_coverage']['fraction'])} | {pct(fit['coverage']['best_contradiction']['bm25_score_coverage']['fraction'])} | {pct(result['coverage']['best_contradiction']['attention_score_coverage']['fraction'])} | {pct(result['coverage']['best_contradiction']['bm25_score_coverage']['fraction'])} |

The reusable label-free claim interface has {len(AGGREGATE_FEATURE_NAMES)} float32 columns:
union, attention and BM25 subset max/mean E/N/C, class margins, lack-E,
OR-risk, union gains and best-score coverage flags.  Exact candidate E/N/C is
retained separately in each cohort's `candidate_arrays_*.npz`.

The portable cache catalog contains {catalog_audit_placeholder()}.
Calibration used the rule frozen by the fit score command and did not select or
change a rule.  No labels, pretrained model, GPU, official test data, detector
training, or baseline write was used by this audit.
"""
        frozen_text(OUT / "REPORT.md", report)
    assert_cpu_only()
    print("EVIDENCE_UNION_COVERAGE_COMPLETE", cohort,
          json.dumps(summary, ensure_ascii=False), flush=True)


def catalog_audit_placeholder() -> str:
    audit = read_json(OUT / "existing_cache_catalog_audit.json")
    return (f"{audit['portable_exact_requests']:,} bit-identical reusable requests "
            f"and {audit['portable_conflicting_requests']:,} conflicting requests; "
            "the latter are deliberately marked for recomputation when attaching new answers")


def status() -> None:
    assert_cpu_only()
    result = {"version": VERSION, "cohorts": {}, "GPU_used_by_status": False,
              "official_test_opened": False}
    for cohort in COHORTS:
        manifest_path = OUT / f"manifest_{cohort}.json"
        result["cohorts"][cohort] = {
            "manifest": manifest_path.exists(),
            "diagnostics": (OUT / f"diagnostics_{cohort}.json").exists(),
            "missing_requests": (read_json(manifest_path)["unique_missing_requests"]
                                 if manifest_path.exists() else None),
            "gpu_smoke": (OUT / f"GPU_SMOKE_{cohort}.json").exists(),
            "extraction": (read_json(OUT / f"extraction_{cohort}.json")["status"]
                           if (OUT / f"extraction_{cohort}.json").exists() else None),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("initialize")
    sub.add_parser("self-test")
    for name in ("prepare", "gpu-smoke", "score"):
        one = sub.add_parser(name)
        one.add_argument("--cohort", choices=COHORTS, required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--cohort", choices=COHORTS, required=True)
    extraction.add_argument("--limit-chunks", type=int)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.stage == "initialize":
        initialize()
    elif args.stage == "self-test":
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
    elif args.stage == "prepare":
        prepare(args.cohort)
    elif args.stage == "gpu-smoke":
        gpu_smoke(args.cohort)
    elif args.stage == "extract":
        extract(args.cohort, args.limit_chunks)
    elif args.stage == "score":
        score(args.cohort)
    else:
        status()


if __name__ == "__main__":
    main()
