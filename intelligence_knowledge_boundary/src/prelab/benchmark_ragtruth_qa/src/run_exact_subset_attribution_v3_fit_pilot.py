"""Hash-selected, fit-only pilot for exact three-passage subset attribution.

CPU preparation selects 256 distinct fit groups without reading labels, answer
lengths, or model scores.  Explicit later GPU commands reuse the frozen v2 WDDM
admission gate and the unchanged v1 Llama/eight-subset extraction math, while
writing independent v3 smoke/cache lineage.  The final explicit score command
opens only the expanded fit gold files and reports five-fold group OOF metrics.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_exact_subset_attribution_v1 as core
import run_exact_subset_attribution_v2 as wddm_gate


ROOT = core.ROOT
V1_OUT = ROOT / "results/exact_subset_attribution_v1"
V2_OUT = ROOT / "results/exact_subset_attribution_v2"
OUT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
SOURCE_PREPARED = V1_OUT / "prepared_inputs.jsonl"
SELECTED_PREPARED = OUT / "selected_prepared_inputs.jsonl"
FIT_DATA = ROOT / "fit_expansion/data"
FIT_ANSWERS = FIT_DATA / "answers_fit.jsonl"
FIT_TOKENS = FIT_DATA / "tokens_fit.jsonl"
FIT_WINDOWS = FIT_DATA / "windows_k4_fit.jsonl"
FIT_EXPORT_FREEZE = FIT_DATA / "export_freeze.json"

VERSION = "exact-three-passage-subset-attribution-v3-fit-pilot"
ROLE = "ours_method_internal_ablation_pilot; not a paper baseline"
SELECTION_SEED = "exact-subset-attribution-v3-fit-pilot-20260912"
EXPECTED_ALL_GROUPS = 615
SELECTED_GROUPS = 256
FOLDS = 5
LR_C = 0.1
THREADS = core.THREADS
EXPECTED_SOURCE_PREPARED_SHA256 = "55753d10aa79f2069acfed7161c9b64fa8bcc4ed21da97c1a403ec73e1ff1f2a"
EXPECTED_FIT_GOLD_SHA256 = {
    "answers_fit.jsonl": "8adef0c27d5021acdf559c1566db2d6d949ecccb88ad2532ae626381add0aa33",
    "tokens_fit.jsonl": "18501c2a53a22620f24827d3181d831f2aee77da2b335bd50270fbe939048ec2",
    "windows_k4_fit.jsonl": "cfaf5af088eaac422ee2685c9150a774171d930666426e936f86d0cee905eec3",
}
EXPECTED_FIT_COUNTS = {"answers": 3680, "windows": 653979}
HISTORICAL_RUNTIME_BASIS = {
    "artifact": "results/lumina_qa_features_v1/progress.json",
    "two_view_input_tokens": 4789996,
    "measured_seconds": 4703.901522000087,
    "note": "Different extraction workload; used only for a broad pre-smoke range.",
}

READOUTS = {
    "A_full_nll_citation": (
        "full_nll", "citation_any", "citation_source_fraction",
    ),
    "B_full_empty_delta_citation": (
        "full_nll", "empty_nll", "full_minus_empty",
        "citation_any", "citation_source_fraction",
    ),
    "C_all_29": tuple(core.FEATURE_NAMES),
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selection_hash(identifier):
    """Exact sha256(seed||str(identifier)), with no delimiter."""
    return hashlib.sha256((SELECTION_SEED + str(identifier)).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    pending.replace(path)


def select_identities(identities, expected_groups=EXPECTED_ALL_GROUPS,
                      selected_groups=SELECTED_GROUPS):
    """Select only from partition/group_id/response_id identity fields."""
    grouped = defaultdict(list)
    for identity in identities:
        if identity["partition"] == "fit":
            grouped[identity["group_id"]].append(identity)
    assert len(grouped) == expected_groups
    ranked_groups = sorted(
        grouped,
        key=lambda group_id: (selection_hash(group_id), str(group_id)),
    )[:selected_groups]
    selected = []
    for rank, group_id in enumerate(ranked_groups):
        candidates = grouped[group_id]
        chosen = min(
            candidates,
            key=lambda item: (selection_hash(item["response_id"]),
                              str(item["response_id"])),
        )
        selected.append({
            "selection_rank": rank,
            "group_id": group_id,
            "group_hash": selection_hash(group_id),
            "response_id": chosen["response_id"],
            "response_hash": selection_hash(chosen["response_id"]),
            "source_id": chosen["source_id"],
            "partition": "fit",
        })
    assert len(selected) == selected_groups
    assert len({item["group_id"] for item in selected}) == selected_groups
    assert len({item["response_id"] for item in selected}) == selected_groups
    return selected


def assign_folds(selected):
    groups = np.asarray([str(item["group_id"]) for item in selected], dtype=object)
    indices = np.arange(len(selected), dtype=np.int64)
    held_fold = np.full(len(selected), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=FOLDS)
    for fold, (_, held) in enumerate(splitter.split(indices, groups=groups)):
        held_fold[held] = fold
    assert np.all(held_fold >= 0)
    assert Counter(held_fold.tolist()) == {0: 52, 1: 51, 2: 51, 3: 51, 4: 51}
    return held_fold.tolist()


def protocol():
    return {
        "version": VERSION,
        "role": ROLE,
        "selection": {
            "population": "All 615 fit source-connected groups in the frozen label-free v1 prepared inputs.",
            "group_rule": "Sort by sha256(UTF8(seed||str(group_id))); take the smallest 256.",
            "response_rule": "Within each selected group, take the smallest sha256(UTF8(seed||str(response_id))).",
            "seed": SELECTION_SEED,
            "selected_groups": SELECTED_GROUPS,
            "one_answer_per_group": True,
            "forbidden_selection_inputs": ["labels", "answer length", "input length", "model score", "risk type"],
        },
        "extraction": {
            "model": f"{core.REPO}@{core.REVISION}",
            "backend": core.BACKEND,
            "views": "The unchanged v1 masks 000..111 and same published answer axis.",
            "features": list(core.FEATURE_NAMES),
            "feature_width": len(core.FEATURE_NAMES),
            "GPU_gate": "Reuse the frozen v2 WDDM process gate; write independent v3 smoke/cache/lineage.",
        },
        "fit_only_scoring": {
            "gold_paths": [
                "fit_expansion/data/answers_fit.jsonl",
                "fit_expansion/data/tokens_fit.jsonl",
                "fit_expansion/data/windows_k4_fit.jsonl",
            ],
            "gold_open_time": "Only the explicit score command, after all 256 GPU caches are frozen.",
            "folds": "Fixed label-free five-fold GroupKFold over the 256 distinct source-connected groups.",
            "training": "Lexical raw-BPE labels; same v1 group/answer/token weights, StandardScaler, L2 liblinear LR C=0.1.",
            "projection": "Unified eligible stride-one 4-original-BPE windows take max lexical-token risk; answer takes max window risk.",
            "thresholds": "Each readout chooses window and answer thresholds on its complete fit OOF predictions only.",
            "metrics": ["precision", "recall", "F1", "AUROC", "average precision"],
            "readouts": {name: list(features) for name, features in READOUTS.items()},
            "identity": "A/B/C are internal ablations of our method, not paper baselines.",
            "limitations": "Same OOF cohort supplies threshold choice and reporting; pilot evidence is developmental, not independent test evidence.",
        },
        "common_evaluation_policy": {
            "owner": "Our dataset, group split, labels, 4-BPE windows, answer labels, and metrics define the common exam.",
            "paper_baselines": "Any future paper baseline must retain author architecture, features, training, and native readout unchanged.",
            "allowed_baseline_mapping": "Only predeclared parameter-free label-independent output mapping.",
            "current_A_B_C": "Internal method ablations; no paper baseline is run or modified here.",
        },
        "stage_gate": "initialize -> prepare -> check -> audit; review; gpu-smoke -> extract; review; score",
        "prohibitions": [
            "CPU stages do not read fit label files.",
            "Calibration and official test are never addressed.",
            "v1/v2 artifacts are read-only and never overwritten.",
            "No baseline is modified or scored.",
        ],
    }


def upstream_hashes():
    paths = {
        "v1_runner": Path(core.__file__),
        "v2_gate_runner": Path(wddm_gate.__file__),
        "v1_prepared_inputs": SOURCE_PREPARED,
        "v1_preparation_complete": V1_OUT / "preparation_complete.json",
        "v2_design_freeze": V2_OUT / "design_freeze.json",
        "v2_protocol": V2_OUT / "protocol.json",
        "v2_CPU_CHECK": V2_OUT / "CPU_CHECK.json",
        "v2_CODE_AUDIT": V2_OUT / "CODE_AUDIT.json",
        "v2_INDEPENDENT_AUDIT": V2_OUT / "INDEPENDENT_AUDIT.json",
        "fit_export_freeze_manifest": FIT_EXPORT_FREEZE,
    }
    return {name: {"path": str(path.resolve()), "sha256": sha(path)}
            for name, path in paths.items()}


def verify_upstreams(expected=None):
    current = upstream_hashes()
    assert current["v1_prepared_inputs"]["sha256"] == EXPECTED_SOURCE_PREPARED_SHA256
    v1 = read(V1_OUT / "preparation_complete.json")
    assert v1["labels_accessed"] is False and v1["official_test_opened"] is False
    v2 = read(V2_OUT / "INDEPENDENT_AUDIT.json")
    assert v2["status"] == "passed_independent_stdlib_CPU_audit"
    assert v2["GPU_used"] is False and v2["labels_accessed"] is False
    export = read(FIT_EXPORT_FREEZE)
    serialized = json.dumps(export, ensure_ascii=False)
    for name, expected_hash in EXPECTED_FIT_GOLD_SHA256.items():
        assert name in serialized and expected_hash in serialized
    if expected is not None:
        assert current == expected
    return current


def synthetic_selfcheck():
    identities = []
    for group in ("g0", "g1", "g2", "g3"):
        for response in ("a", "b"):
            identities.append({
                "partition": "fit", "group_id": group,
                "response_id": group + response, "source_id": group,
            })
    identities.append({
        "partition": "calibration", "group_id": "never",
        "response_id": "never", "source_id": "never",
    })
    chosen = select_identities(identities, expected_groups=4, selected_groups=2)
    assert all(item["partition"] == "fit" for item in chosen)
    assert len({item["group_id"] for item in chosen}) == 2
    expected_groups = sorted(("g0", "g1", "g2", "g3"),
                             key=lambda value: (selection_hash(value), value))[:2]
    assert [item["group_id"] for item in chosen] == expected_groups
    for item in chosen:
        candidates = (item["group_id"] + "a", item["group_id"] + "b")
        assert item["response_id"] == min(
            candidates, key=lambda value: (selection_hash(value), value))
    assert set(READOUTS["A_full_nll_citation"]) == {
        "full_nll", "citation_any", "citation_source_fraction",
    }
    assert set(READOUTS["B_full_empty_delta_citation"]) == set(
        READOUTS["A_full_nll_citation"]
    ) | {"empty_nll", "full_minus_empty"}
    assert READOUTS["C_all_29"] == tuple(core.FEATURE_NAMES)
    assert len(core.FEATURE_NAMES) == 29
    assert not torch.cuda.is_initialized()
    return {
        "status": "passed",
        "selection_uses_only_partition_group_response": True,
        "calibration_identity_excluded_before_ranking": True,
        "one_answer_per_selected_group": True,
        "three_fixed_readouts_exact": True,
        "feature_width": 29,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), f"Refuse to overwrite {OUT}"
    OUT.mkdir(parents=True)
    save(OUT / "PREFLIGHT.json", synthetic_selfcheck())
    save(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Exact subset attribution v3：256-group fit-only pilot\n\n"
        "先在无标签 prepared inputs 中按固定 SHA-256 规则选 256 个不同 fit group，每组一答。"
        "CPU 阶段不看长度、分数或标签。审核后，以同一 Llama 和同一 8 个资料子集建立独立缓存。\n\n"
        "冻结缓存后，`score` 只打开三份 expanded-fit 金标；按固定五折 GroupKFold 报统一 4-BPE 窗口"
        "和整答指标。A/B/C 是我们方法内部消融，不是论文 baseline。\n",
        encoding="utf-8",
    )
    upstream = verify_upstreams()
    save(OUT / "design_freeze.json", {
        "status": "frozen_before_label_free_selection",
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "upstream_sha256": upstream,
        "selection_seed": SELECTION_SEED,
        "selection_seed_sha256": digest(SELECTION_SEED),
        "readouts_sha256": digest({name: list(value) for name, value in READOUTS.items()}),
        "future_fit_gold_expected_sha256_from_label_free_export_manifest": EXPECTED_FIT_GOLD_SHA256,
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    print("EXACT_SUBSET_V3_FIT_PILOT_PROTOCOL_FROZEN", flush=True)


def verify_freeze():
    freeze = read(OUT / "design_freeze.json")
    assert freeze["source_sha256"] == sha(__file__)
    assert freeze["protocol_sha256"] == sha(OUT / "protocol.json")
    assert read(OUT / "protocol.json") == protocol()
    assert freeze["selection_seed"] == SELECTION_SEED
    assert freeze["selection_seed_sha256"] == digest(SELECTION_SEED)
    assert freeze["readouts_sha256"] == digest(
        {name: list(value) for name, value in READOUTS.items()}
    )
    verify_upstreams(freeze["upstream_sha256"])
    return freeze


def prepare():
    assert not torch.cuda.is_initialized()
    freeze = verify_freeze()
    assert not (OUT / "preparation_complete.json").exists()
    identities = []
    for row in lines(SOURCE_PREPARED):
        allowed_flag = row["labels_used"]
        assert allowed_flag is False
        identities.append({
            "partition": row["partition"],
            "group_id": row["group_id"],
            "response_id": row["response_id"],
            "source_id": row["source_id"],
        })
    assert len(identities) == 3839
    selected = select_identities(identities)
    held_folds = assign_folds(selected)
    for item, fold in zip(selected, held_folds):
        item["held_fold"] = fold
    selected_ids = {item["response_id"] for item in selected}
    selected_rows_by_id = {}
    for row in lines(SOURCE_PREPARED):
        if row["response_id"] not in selected_ids:
            continue
        allowed_flag = row.pop("labels_used")
        assert allowed_flag is False
        core.reject_annotation_keys(row)
        row["labels_used"] = False
        selected_rows_by_id[row["response_id"]] = row
    assert set(selected_rows_by_id) == selected_ids
    selected_rows = [selected_rows_by_id[item["response_id"]] for item in selected]
    atomic_jsonl(SELECTED_PREPARED, selected_rows)
    save(OUT / "selection_manifest.json", {
        "status": "frozen_label_free_selection",
        "rule": {
            "seed": SELECTION_SEED,
            "concatenation": "UTF8(seed + str(identifier)); no delimiter",
            "group": "smallest 256 of 615 group hashes",
            "response": "smallest response hash within each selected group",
        },
        "population": {"answers": len(identities), "fit_groups": EXPECTED_ALL_GROUPS},
        "selected": selected,
        "selection_order_sha256": digest([
            [item["group_id"], item["response_id"]] for item in selected
        ]),
        "fold_assignment_sha256": digest([
            [item["response_id"], item["held_fold"]] for item in selected
        ]),
        "selection_inputs": ["partition", "group_id", "response_id"],
        "source_id_use": "recorded after selection; not ranked",
        "labels_or_lengths_or_scores_accessed": False,
        "official_test_opened": False,
    })

    raw_tokens = sum(len(row["answer_token_ids"]) for row in selected_rows)
    input_tokens = np.zeros(8, dtype=np.int64)
    max_tokens = np.zeros(8, dtype=np.int64)
    answer_token_counts = []
    for row in selected_rows:
        answer_token_counts.append(len(row["answer_token_ids"]))
        for view in row["views"]:
            mask = view["mask"]
            input_tokens[mask] += view["input_token_count"]
            max_tokens[mask] = max(max_tokens[mask], view["input_token_count"])
    proportional = (input_tokens.sum() /
                    HISTORICAL_RUNTIME_BASIS["two_view_input_tokens"] *
                    HISTORICAL_RUNTIME_BASIS["measured_seconds"])
    estimated_range = [max(300, int(proportional * 0.75)),
                       max(900, int(proportional * 1.75))]
    statistics = {
        "status": "CPU_selected_waiting_for_GPU_review",
        "selected_answers": len(selected_rows),
        "selected_groups": len({row["group_id"] for row in selected_rows}),
        "selected_source_ids": len({row["source_id"] for row in selected_rows}),
        "fold_answer_counts": dict(sorted(Counter(held_folds).items())),
        "raw_answer_tokens": raw_tokens,
        "answer_token_count": {
            "min": int(np.min(answer_token_counts)),
            "median": float(np.median(answer_token_counts)),
            "p95": float(np.percentile(answer_token_counts, 95)),
            "max": int(np.max(answer_token_counts)),
        },
        "input_tokens_by_subset_mask": input_tokens.tolist(),
        "all_eight_view_input_tokens": int(input_tokens.sum()),
        "max_input_tokens_by_subset_mask": max_tokens.tolist(),
        "max_any_view_input_tokens": int(max_tokens.max()),
        "GPU_forward_calls_full_extract": len(selected_rows) * 8,
        "GPU_forward_calls_smoke": 17,
        "raw_logprob_float32_bytes": raw_tokens * 8 * 4,
        "derived_feature_float32_bytes": raw_tokens * len(core.FEATURE_NAMES) * 4,
        "pre_smoke_runtime_estimate_seconds": estimated_range,
        "runtime_estimate_basis": HISTORICAL_RUNTIME_BASIS,
        "runtime_estimate_warning": "Broad cross-workload extrapolation; replace with v3 smoke measurement.",
        "selection_was_frozen_before_length_statistics": True,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    save(OUT / "preparation_statistics.json", statistics)
    names = (
        "PREFLIGHT.json", "protocol.json", "PLAN.md", "design_freeze.json",
        "selection_manifest.json", "selected_prepared_inputs.jsonl",
        "preparation_statistics.json",
    )
    save(OUT / "preparation_complete.json", {
        "status": "CPU_ready_waiting_for_GPU_review",
        "files_sha256": {name: sha(OUT / name) for name in names},
        "source_prepared_sha256": sha(SOURCE_PREPARED),
        "selected_prepared_sha256": sha(SELECTED_PREPARED),
        "selected_answers": SELECTED_GROUPS,
        "selected_distinct_groups": SELECTED_GROUPS,
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "new_fits": 0,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    assert freeze["upstream_sha256"] == upstream_hashes()
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_V3_FIT_PILOT_CPU_PREPARED", flush=True)


def check_ready():
    freeze = verify_freeze()
    complete = read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_ready_waiting_for_GPU_review"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected
    assert complete["source_prepared_sha256"] == sha(SOURCE_PREPARED)
    assert complete["selected_prepared_sha256"] == sha(SELECTED_PREPARED)
    assert freeze["upstream_sha256"] == upstream_hashes()
    return complete


def check():
    assert not torch.cuda.is_initialized()
    check_ready()
    identities = []
    source_by_id = {}
    for row in lines(SOURCE_PREPARED):
        identities.append({
            "partition": row["partition"], "group_id": row["group_id"],
            "response_id": row["response_id"], "source_id": row["source_id"],
        })
        source_by_id[row["response_id"]] = row
    expected = select_identities(identities)
    expected_folds = assign_folds(expected)
    for item, fold in zip(expected, expected_folds):
        item["held_fold"] = fold
    manifest = read(OUT / "selection_manifest.json")
    assert manifest["selected"] == expected
    selected_rows = list(lines(SELECTED_PREPARED))
    assert len(selected_rows) == SELECTED_GROUPS
    for row, item in zip(selected_rows, expected):
        assert row == source_by_id[item["response_id"]]
        assert row["partition"] == "fit" and row["labels_used"] is False
        clean = {key: value for key, value in row.items() if key != "labels_used"}
        core.reject_annotation_keys(clean)
        assert len(row["views"]) == 8
        assert [view["mask"] for view in row["views"]] == list(range(8))
        assert all(view["answer_positions"][-1] == view["input_token_count"] - 1
                   for view in row["views"])
    stats = read(OUT / "preparation_statistics.json")
    assert stats["selected_answers"] == stats["selected_groups"] == SELECTED_GROUPS
    assert stats["raw_answer_tokens"] == sum(
        len(row["answer_token_ids"]) for row in selected_rows
    )
    assert stats["GPU_forward_calls_full_extract"] == 2048
    report = {
        "status": "passed_CPU_label_free_selection_and_geometry_check",
        "population_answers": len(identities),
        "population_fit_groups": len({item["group_id"] for item in identities
                                      if item["partition"] == "fit"}),
        "selected_answers": len(selected_rows),
        "selected_groups": len({row["group_id"] for row in selected_rows}),
        "selected_source_ids": len({row["source_id"] for row in selected_rows}),
        "fold_answer_counts": dict(sorted(Counter(expected_folds).items())),
        "selection_manifest_sha256": sha(OUT / "selection_manifest.json"),
        "selected_prepared_sha256": sha(SELECTED_PREPARED),
        "selection_recomputed_exactly": True,
        "selected_rows_equal_upstream_objects": True,
        "eight_views_and_same_answer_geometry_checked": True,
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    save(OUT / "CPU_CHECK.json", report)
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_V3_FIT_PILOT_CPU_CHECK_PASSED", flush=True)


def runtime_signature():
    complete = read(OUT / "preparation_complete.json")
    return {
        "version": VERSION,
        "source_sha256": sha(__file__),
        "v1_core_source_sha256": sha(core.__file__),
        "v2_gate_source_sha256": sha(wddm_gate.__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "selection_manifest_sha256": complete["files_sha256"]["selection_manifest.json"],
        "selected_prepared_sha256": complete["selected_prepared_sha256"],
        "model_manifest_sha256": sha(core.MODEL_MANIFEST),
        "repo": core.REPO,
        "revision": core.REVISION,
        "backend": core.BACKEND,
        "feature_names_sha256": digest(core.FEATURE_NAMES),
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "accelerate", "numpy")},
        "cuda_runtime": torch.version.cuda,
    }


def cache_signature(row):
    return digest({
        "runtime": runtime_signature(),
        "response_id": row["response_id"],
        "answer_token_ids_sha256": row["answer_token_ids_sha256"],
        "citation_mask_sha256": row["citation_geometry"]["citation_mask_sha256"],
        "view_input_sha256": [view["input_ids_sha256"] for view in row["views"]],
    })


def validate_cache(path, row):
    metadata = read(path.with_suffix(".json"))
    assert metadata["complete"] is True
    assert metadata["response_id"] == row["response_id"]
    assert metadata["cache_signature"] == cache_signature(row)
    assert metadata["npz_sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as archive:
        logp = archive["selected_logprob"].copy()
        answer_ids = archive["answer_token_ids"].copy()
        citation_mask = archive["citation_mask"].copy()
    assert logp.shape == (8, len(row["answer_token_ids"]))
    assert logp.dtype == np.float32 and np.isfinite(logp).all()
    assert answer_ids.tolist() == row["answer_token_ids"]
    assert citation_mask.tolist() == row["citation_geometry"]["citation_mask"]
    _, efficiency = core.derive_features(logp, citation_mask)
    assert abs(efficiency - metadata["shapley_efficiency_max_abs"]) <= 1e-12
    return logp


def gpu_smoke():
    check_ready()
    assert not (OUT / "GPU_SMOKE.json").exists(), "No silent smoke overwrite"
    rows = list(lines(SELECTED_PREPARED))
    chosen = (rows[0], rows[-1])
    with wddm_gate.exclusive_gpu(f"{VERSION}:gpu-smoke") as gate_lease:
        lease = dict(gate_lease, consumer_version=VERSION,
                     v2_gate_source_sha256=sha(wddm_gate.__file__))
        model, model_meta = core.load_nf4()
        try:
            started = time.perf_counter()
            records = []
            first_full = None
            for row in chosen:
                logp, efficiency = core.extract_row(model, row)
                records.append({
                    "response_id": row["response_id"],
                    "tokens": logp.shape[1],
                    "all_eight_input_tokens": sum(view["input_token_count"]
                                                  for view in row["views"]),
                    "logprob_sha256": digest(logp.tolist()),
                    "shapley_efficiency_max_abs": efficiency,
                })
                if first_full is None:
                    first_full = logp[7].copy()
            repeated = core.selected_token_logprob(
                model, chosen[0]["views"][7]["prefix_ids"],
                chosen[0]["answer_token_ids"],
                chosen[0]["views"][7]["answer_positions"],
            )
            repeat_error = float(np.max(np.abs(first_full - repeated)))
            assert repeat_error <= 1e-6
            save(OUT / "GPU_SMOKE.json", {
                "status": "passed_not_full_extraction",
                "runtime_signature": runtime_signature(),
                "runtime_signature_sha256": digest(runtime_signature()),
                "gate_lease": lease,
                "model": model_meta,
                "records": records,
                "forward_calls": 17,
                "full_view_repeat_max_abs": repeat_error,
                "seconds_after_model_load": time.perf_counter() - started,
                "fit_gold_files_opened": False,
                "labels_accessed": False,
                "official_test_opened": False,
            })
        finally:
            del model
            core.clean_gpu()
    print("EXACT_SUBSET_V3_FIT_PILOT_GPU_SMOKE_PASSED", flush=True)


def extract():
    check_ready()
    smoke = read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_not_full_extraction"
    assert smoke["runtime_signature_sha256"] == digest(runtime_signature())
    assert not (OUT / "extraction_complete.json").exists()
    cache_dir = OUT / "token_logprob"
    cache_dir.mkdir(exist_ok=True)
    records = []
    with wddm_gate.exclusive_gpu(f"{VERSION}:extract") as gate_lease:
        lease = dict(gate_lease, consumer_version=VERSION,
                     v2_gate_source_sha256=sha(wddm_gate.__file__))
        model, model_meta = core.load_nf4()
        started = time.perf_counter()
        try:
            for index, row in enumerate(lines(SELECTED_PREPARED)):
                path = cache_dir / f"{row['response_id']}.npz"
                metadata_path = path.with_suffix(".json")
                if path.exists() and metadata_path.exists():
                    validate_cache(path, row)
                else:
                    assert not path.exists() and not metadata_path.exists()
                    logp, efficiency = core.extract_row(model, row)
                    core.atomic_npz(
                        path,
                        selected_logprob=logp,
                        answer_token_ids=np.asarray(row["answer_token_ids"], dtype=np.int32),
                        citation_mask=np.asarray(row["citation_geometry"]["citation_mask"], dtype=np.uint8),
                    )
                    save(metadata_path, {
                        "complete": True,
                        "response_id": row["response_id"],
                        "cache_signature": cache_signature(row),
                        "npz_sha256": sha(path),
                        "shapley_efficiency_max_abs": efficiency,
                        "fit_gold_files_opened": False,
                        "labels_accessed": False,
                        "official_test_opened": False,
                    })
                records.append({
                    "response_id": row["response_id"],
                    "npz_sha256": sha(path),
                    "metadata_sha256": sha(metadata_path),
                })
                if (index + 1) % 16 == 0:
                    print("EXACT_SUBSET_V3_GPU_EXTRACTED", index + 1,
                          SELECTED_GROUPS, flush=True)
        finally:
            del model
            core.clean_gpu()
    assert len(records) == SELECTED_GROUPS
    save(OUT / "extraction_complete.json", {
        "status": "complete_frozen_fit_pilot_logprob_not_scored",
        "answers": len(records),
        "records": records,
        "gate_lease": lease,
        "model": model_meta,
        "runtime_signature_sha256": digest(runtime_signature()),
        "selected_prepared_sha256": sha(SELECTED_PREPARED),
        "seconds_after_model_load": time.perf_counter() - started,
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "official_test_opened": False,
    })
    print("EXACT_SUBSET_V3_FIT_PILOT_EXTRACTION_COMPLETE", flush=True)


def check_extracted(prepared):
    complete = read(OUT / "extraction_complete.json")
    assert complete["status"] == "complete_frozen_fit_pilot_logprob_not_scored"
    assert complete["runtime_signature_sha256"] == digest(runtime_signature())
    assert complete["selected_prepared_sha256"] == sha(SELECTED_PREPARED)
    records = {item["response_id"]: item for item in complete["records"]}
    assert len(records) == len(prepared) == SELECTED_GROUPS
    for row in prepared:
        path = OUT / "token_logprob" / f"{row['response_id']}.npz"
        record = records[row["response_id"]]
        assert sha(path) == record["npz_sha256"]
        assert sha(path.with_suffix(".json")) == record["metadata_sha256"]
        validate_cache(path, row)
    return complete


def verify_fit_gold_files():
    paths = {
        "answers_fit.jsonl": FIT_ANSWERS,
        "tokens_fit.jsonl": FIT_TOKENS,
        "windows_k4_fit.jsonl": FIT_WINDOWS,
    }
    actual = {name: sha(path) for name, path in paths.items()}
    assert actual == EXPECTED_FIT_GOLD_SHA256
    return actual


def load_selected_fit_gold(selected):
    """Gold-opening function reachable only from explicit score."""
    selected_ids = {row["response_id"] for row in selected}
    answers_by_id = {}
    tokens_by_id = {}
    answer_count = token_count = 0
    for answer in lines(FIT_ANSWERS):
        answer_count += 1
        if answer["response_id"] in selected_ids:
            answers_by_id[answer["response_id"]] = answer
    for token in lines(FIT_TOKENS):
        token_count += 1
        if token["response_id"] in selected_ids:
            tokens_by_id[token["response_id"]] = token
    assert answer_count == token_count == EXPECTED_FIT_COUNTS["answers"]
    assert set(answers_by_id) == set(tokens_by_id) == selected_ids
    answers = [answers_by_id[row["response_id"]] for row in selected]
    tokens = [tokens_by_id[row["response_id"]] for row in selected]
    by_response = {}
    for prepared, answer, token in zip(selected, answers, tokens):
        assert prepared["partition"] == answer["partition"] == token["partition"] == "fit"
        assert prepared["response_id"] == answer["response_id"] == token["response_id"]
        assert prepared["group_id"] == answer["group_id"]
        assert prepared["answer_token_ids"] == token["token_ids"]
        assert prepared["response_token_offsets"] == token["response_token_offsets"]
        assert answer["eligible"] and answer["quality"] == "good"
        assert answer["label"] == int(len(answer["original_labels"]) > 0)
        assert token["answer_risk"] == answer["label"]
        assert token["answer_sha256"] == answer["answer_sha256"] == prepared["answer_sha256"]
        by_response[answer["response_id"]] = {"answer": answer, "tokens": token}
    windows = []
    full_window_count = 0
    for window in lines(FIT_WINDOWS):
        full_window_count += 1
        if window["response_id"] in selected_ids:
            token = by_response[window["response_id"]]["tokens"]
            indices = window["token_indices"]
            assert window["partition"] == "fit" and window["eligible"]
            assert indices == list(range(window["token_start"], window["token_end"]))
            assert len(indices) == min(4, token["token_count"])
            assert any(token["lexical_mask"][index] for index in indices)
            assert window["label"] == int(any(token["risk_mask"][index]
                                               for index in indices))
            windows.append(window)
    assert full_window_count == EXPECTED_FIT_COUNTS["windows"]
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[window["response_id"]].append(index)
    assert all(answer_windows[answer["response_id"]] for answer in answers)
    return {
        "answers": answers,
        "tokens": tokens,
        "windows": windows,
        "by_response": by_response,
        "answer_windows": dict(answer_windows),
    }


def score():
    assert not torch.cuda.is_initialized()
    check_ready()
    prepared = list(lines(SELECTED_PREPARED))
    check_extracted(prepared)
    assert not (OUT / "score_started.json").exists(), "No silent score overwrite"
    fit_hashes = verify_fit_gold_files()
    save(OUT / "score_started.json", {
        "status": "explicit_fit_only_gold_open_after_frozen_GPU_cache",
        "fit_gold_sha256": fit_hashes,
        "calibration_opened": False,
        "official_test_opened": False,
    })
    meta = load_selected_fit_gold(prepared)
    selection = read(OUT / "selection_manifest.json")
    selected_meta = {item["response_id"]: item for item in selection["selected"]}
    assert [row["response_id"] for row in prepared] == [
        answer["response_id"] for answer in meta["answers"]
    ]
    total_tokens = sum(len(row["answer_token_ids"]) for row in prepared)
    assert total_tokens == read(OUT / "preparation_statistics.json")["raw_answer_tokens"]
    feature_path = OUT / "token_features_29.npy"
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32,
        shape=(total_tokens, len(core.FEATURE_NAMES)),
    )
    token_starts = {}
    cursor = 0
    for row in prepared:
        logp = validate_cache(OUT / "token_logprob" / f"{row['response_id']}.npz", row)
        derived, _ = core.derive_features(
            logp, row["citation_geometry"]["citation_mask"]
        )
        right = cursor + len(derived)
        features[cursor:right] = derived
        token_starts[row["response_id"]] = (cursor, right)
        cursor = right
    features.flush()
    y_token = np.empty(total_tokens, dtype=np.int8)
    lexical = np.empty(total_tokens, dtype=bool)
    for answer in meta["answers"]:
        response_id = answer["response_id"]
        left, right = token_starts[response_id]
        token = meta["by_response"][response_id]["tokens"]
        y_token[left:right] = token["risk_mask"]
        lexical[left:right] = token["lexical_mask"]
    held_folds = np.asarray([
        selected_meta[answer["response_id"]]["held_fold"]
        for answer in meta["answers"]
    ], dtype=np.int8)
    recomputed = assign_folds(selection["selected"])
    assert held_folds.tolist() == recomputed
    all_answer_indices = np.arange(len(meta["answers"]), dtype=np.int64)
    scores_to_save = {}
    summaries = {}
    saved_models = {}
    for readout_name, names in READOUTS.items():
        columns = np.asarray([core.FEATURE_NAMES.index(name) for name in names],
                             dtype=np.int64)
        oof_token = np.full(total_tokens, np.nan, dtype=np.float64)
        fold_models = []
        for fold in range(FOLDS):
            train_answers = all_answer_indices[held_folds != fold]
            held_answers = all_answer_indices[held_folds == fold]
            weights = core.fold_token_weights(
                meta, train_answers, token_starts, y_token, lexical
            )
            train_tokens = np.flatnonzero(weights > 0)
            held_tokens = np.concatenate([
                np.arange(*token_starts[meta["answers"][index]["response_id"]])
                for index in held_answers
            ])
            assert set(y_token[train_tokens]) == {0, 1}
            scaler = StandardScaler().fit(
                features[train_tokens][:, columns],
                sample_weight=weights[train_tokens],
            )
            model = LogisticRegression(
                C=LR_C, solver="liblinear", penalty="l2", max_iter=2000,
                random_state=core.SEED,
            )
            model.fit(
                scaler.transform(features[train_tokens][:, columns]),
                y_token[train_tokens], sample_weight=weights[train_tokens],
            )
            assert model.n_iter_.max() < 2000
            oof_token[held_tokens] = model.predict_proba(
                scaler.transform(features[held_tokens][:, columns])
            )[:, 1]
            fold_models.append({
                "fold": fold, "scaler": scaler, "model": model,
                "train_answers": len(train_answers),
                "held_answers": len(held_answers),
                "train_tokens": len(train_tokens),
                "held_raw_tokens": len(held_tokens),
            })
        assert np.isfinite(oof_token).all()
        window_scores = core.project_token_scores(meta, oof_token, token_starts)
        answer_scores = core.q.answer_scores(meta, window_scores)
        window_y = np.asarray([window["label"] for window in meta["windows"]],
                              dtype=np.int8)
        answer_y = np.asarray([answer["label"] for answer in meta["answers"]],
                              dtype=np.int8)
        thresholds = {
            "window": core.q.choose_threshold(window_y, window_scores),
            "answer": core.q.choose_threshold(answer_y, answer_scores),
        }
        metrics = {
            "windows": core.q.count(
                window_y, window_scores, thresholds["window"]["threshold"]
            ),
            "answers": core.q.count(
                answer_y, answer_scores, thresholds["answer"]["threshold"]
            ),
        }
        summaries[readout_name] = {
            "identity": "ours_internal_ablation_not_paper_baseline",
            "feature_names": list(names),
            "feature_columns": columns.tolist(),
            "C": LR_C,
            "thresholds": thresholds,
            "fit_OOF_metrics": metrics,
            "folds": [{key: value for key, value in item.items()
                       if key not in {"scaler", "model"}}
                      for item in fold_models],
        }
        saved_models[readout_name] = {
            "feature_names": names, "columns": columns,
            "fold_models": fold_models,
        }
        scores_to_save[f"{readout_name}__token"] = oof_token
        scores_to_save[f"{readout_name}__window"] = window_scores
        scores_to_save[f"{readout_name}__answer"] = answer_scores
    np.savez_compressed(OUT / "oof_scores.npz", **scores_to_save,
                        y_token=y_token, lexical_mask=lexical)
    (OUT / "readout_models.pkl").write_bytes(pickle.dumps({
        "role": ROLE, "C": LR_C, "folds": FOLDS,
        "readouts": saved_models,
    }, protocol=5))
    save(OUT / "summary.json", {
        "status": "fit_only_internal_ablation_pilot_complete",
        "role": ROLE,
        "selected_answers_and_distinct_groups": SELECTED_GROUPS,
        "selected_windows": len(meta["windows"]),
        "selected_raw_tokens": total_tokens,
        "readouts": summaries,
        "threshold_and_report_cohort_same_fit_OOF": True,
        "calibration_opened": False,
        "official_test_opened": False,
        "final_test_claim": False,
        "paper_baseline_modified_or_scored": False,
    })
    rows = []
    for name, item in summaries.items():
        rows.append(
            f"| {name} | {len(item['feature_names'])} | "
            f"{item['fit_OOF_metrics']['windows']['f1']:.4f} | "
            f"{item['fit_OOF_metrics']['answers']['f1']:.4f} |"
        )
    (OUT / "REPORT.md").write_text(
        "# Exact subset attribution v3：fit-only pilot\n\n"
        "A/B/C 均为我们方法内部消融，不是论文 baseline。全部数字来自固定 256-group fit OOF；"
        "同一批 OOF 标签也用于选阈值，因此只能作开发筛查。\n\n"
        "| 读出 | 特征数 | 4-BPE window F1 | answer F1 |\n"
        "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    files = ("score_started.json", "token_features_29.npy", "oof_scores.npz",
             "readout_models.pkl", "summary.json", "REPORT.md")
    save(OUT / "complete.json", {
        "status": "complete_fit_only_pilot_not_independent_test",
        "files_sha256": {name: sha(OUT / name) for name in files},
        "calibration_opened": False,
        "official_test_opened": False,
        "paper_baseline_modified_or_scored": False,
    })
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_V3_FIT_PILOT_SCORE_COMPLETE", flush=True)


def audit():
    assert not torch.cuda.is_initialized()
    check_ready()
    assert (OUT / "CPU_CHECK.json").exists()
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = {
        "select_identities", "assign_folds", "prepare", "check", "gpu_smoke",
        "extract", "load_selected_fit_gold", "score",
    }
    assert required <= set(functions)
    selection_source = ast.get_source_segment(source, functions["select_identities"])
    prepare_source = ast.get_source_segment(source, functions["prepare"])
    check_source = ast.get_source_segment(source, functions["check"])
    score_source = ast.get_source_segment(source, functions["score"])
    assert all(term not in selection_source for term in (
        "label", "length", "token_count", "risk", "score", "answer_token_ids",
        "input_token_count",
    ))
    assert "selection_hash(group_id)" in selection_source
    assert 'selection_hash(item["response_id"])' in selection_source
    assert "FIT_ANSWERS" not in prepare_source + check_source
    assert "FIT_TOKENS" not in prepare_source + check_source
    assert "FIT_WINDOWS" not in prepare_source + check_source
    assert "load_nf4(" not in prepare_source + check_source
    assert "load_selected_fit_gold(prepared)" in score_source
    assert "GroupKFold" in source and "core.fold_token_weights" in score_source
    assert "core.project_token_scores" in score_source
    assert "core.q.choose_threshold" in score_source
    assertions = {
        "exact_hash_group_then_response_selection": True,
        "256_distinct_groups_one_answer_each": read(OUT / "CPU_CHECK.json")["selected_groups"] == 256,
        "selection_does_not_use_labels_lengths_or_scores": True,
        "five_label_free_GroupKFold_assignments": read(OUT / "CPU_CHECK.json")["fold_answer_counts"] == {
            "0": 52, "1": 51, "2": 51, "3": 51, "4": 51,
        },
        "same_v1_Llama_eight_views_and_29_features": len(core.FEATURE_NAMES) == 29,
        "v2_WDDM_gate_reused": "wddm_gate.exclusive_gpu" in source,
        "independent_v3_smoke_cache_lineage": str(OUT) != str(V1_OUT) != str(V2_OUT),
        "three_fixed_C_point1_LR_readouts": LR_C == 0.1 and [len(value) for value in READOUTS.values()] == [3, 5, 29],
        "internal_ablations_not_paper_baselines": ROLE.endswith("not a paper baseline"),
        "unified_4BPE_and_answer_max": "core.project_token_scores" in score_source and "core.q.answer_scores" in score_source,
        "fit_only_gold_loader": all(path.name.endswith("_fit.jsonl") for path in (FIT_ANSWERS, FIT_TOKENS, FIT_WINDOWS)),
        "calibration_and_test_paths_absent": True,
        "paper_baselines_unmodified": True,
        "GPU_not_run": not (OUT / "GPU_SMOKE.json").exists(),
    }
    assert all(assertions.values())
    report = {
        "status": "passed_CPU_only_fit_pilot_protocol_selection_and_code_audit",
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "selection_manifest_sha256": sha(OUT / "selection_manifest.json"),
        "selected_prepared_sha256": sha(SELECTED_PREPARED),
        "CPU_CHECK_sha256": sha(OUT / "CPU_CHECK.json"),
        "assertions": assertions,
        "statistics": read(OUT / "preparation_statistics.json"),
        "next_commands_after_root_review": ["gpu-smoke", "extract"],
        "score_command_requires_frozen_complete_extraction": True,
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
        "paper_baseline_modified": False,
    }
    save(OUT / "CODE_AUDIT.json", report)
    (OUT / "CODE_AUDIT.md").write_text(
        "# Exact subset attribution v3 fit-only pilot：CPU 审计\n\n"
        "通过。固定 SHA-256 规则从 615 个 fit group 选择 256 组，每组一答；选择代码不读取"
        "标签、长度、风险类型或模型分数。五折也在无标签阶段冻结。\n\n"
        "GPU 将复用 v2 WDDM 门禁，但 smoke、256 条 cache 和 lineage 全部写 v3。评分只允许读取"
        " expanded-fit 的 answer/token/4-BPE-window 三份金标。A/B/C 是我们方法内部消融；"
        "论文 baseline 未运行、未修改。当前未使用 GPU 或标签。\n",
        encoding="utf-8",
    )
    print("EXACT_SUBSET_V3_FIT_PILOT_CODE_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(
        "initialize", "self-test", "prepare", "check", "audit",
        "gpu-smoke", "extract", "score",
    ))
    arguments = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if arguments.stage == "initialize":
            initialize()
        elif arguments.stage == "self-test":
            print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
        elif arguments.stage == "prepare":
            prepare()
        elif arguments.stage == "check":
            check()
        elif arguments.stage == "audit":
            audit()
        elif arguments.stage == "gpu-smoke":
            gpu_smoke()
        elif arguments.stage == "extract":
            extract()
        else:
            score()
