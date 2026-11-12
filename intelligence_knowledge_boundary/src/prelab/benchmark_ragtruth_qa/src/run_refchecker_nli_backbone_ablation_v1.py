"""RefChecker NLIChecker-backbone substitution on retrieved-evidence NLI v1.

This is an in-house backbone ablation, never a complete RefChecker baseline.
It reuses v1's exact 51,953 claim/evidence pairs and exact supervised readout,
changing only the frozen NLI checkpoint/tokenizer to the checkpoint used by
RefChecker's official NLIChecker.  CPU preparation/checks are label-free;
GPU extraction and CPU scoring remain explicit separate stages.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import run_development as q
import run_retrieved_evidence_nli_v1 as v1


ROOT = q.ROOT
OUT = ROOT / "results/refchecker_nli_backbone_ablation_v1"
V1_OUT = ROOT / "results/retrieved_evidence_nli_v1"
MODEL = ROOT.parent / "models/RefChecker-NLIChecker-roberta-large"

ROLE = "ours_backbone_ablation; not_complete_RefChecker; never_a_formal_baseline"
MODEL_ID = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
MODEL_REVISION = "5b605abab9b75bc87ab66cfc049ef58d9d64b8ed"
MODEL_SHA256 = "a51c55cb7dc41c4c862f06a3d9c413a8aa0619222bc56f06cb6f2f5b2d3ee14a"
MODEL_MANIFEST_SHA256 = "fd40f0eb0a68123064cd97e770640604788f591b6da9d42bde048d29963ff6f2"
# The downloaded state dict also contains RoBERTa's unused pooler
# (1,049,600 parameters).  AutoModelForSequenceClassification correctly drops
# those tensors, so keep checkpoint and instantiated-model counts separate.
CHECKPOINT_PARAMETERS = 356_412_419
RUNTIME_PARAMETERS = 355_362_819
PARAMETER_BYTES = 1_425_649_676

REFCHECKER_COMMIT = "1df1b25cee792ba2b171302e31ca4f768bd67703"
OFFICIAL_FILES = {
    "nli_checker.py": {
        "repo_path": "refchecker/checker/nli_checker.py",
        "sha256": "d3d9ed0318f04a1e68c0149b4ad413a38dfbad05a6e0b96ea72a3e15a236883e",
    },
    "checker_base.py": {
        "repo_path": "refchecker/checker/checker_base.py",
        "sha256": "64eb4329bffaf1ccf54bcfc5cf25e4cd112c7ae76fee6a702b167bf6750ef7f5",
    },
}
MODEL_FILE_SHA256 = {
    "README.md": "bb8affaa741b518e536d09ac6ff8f85d7ea3aee215c56e111379c78eaed8a17c",
    "config.json": "9d074204b668902ea51bd8ecb323f01cd37040cb54e8601d872c8fdb0687131e",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "pytorch_model.bin": MODEL_SHA256,
    "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
    "tokenizer_config.json": "0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8",
    "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
}
V1_EXPECTED = {
    "runner": "353129f81e194b87542ec60aa74a6db012cb437108384805ceaa1dd13d1bd64a",
    "protocol": "a7e55e74cf87e734b6948f49aae47bc8ec8e8418fc742e81c797412598c56c52",
    "inputs": "aaad043a5ecae7ff881758cc7b0717b2771168c8473a3f84096045f7fe151408",
    "preparation": "4f5f34fd6404c33a03c46b52b6a8abc331a404143f36498bee1e16bf4cd334ec",
}

CLASSES = ("entailment", "neutral", "contradiction")
PAIR_LIMIT = 512
BATCH = 16
THREADS = 4
SEED = 20261015
BACKEND = "refchecker_roberta_large_cuda_fp32_sdpa_math_batch16_v1"
PEAK_LIMIT = int(6.5 * 1024 ** 3)
MIN_FREE = 4 * 1024 ** 3


def protocol():
    original = copy.deepcopy(v1.protocol())
    return {
        "version": "retrieved-evidence-refchecker-nli-backbone-ablation-v1",
        "status": "frozen_before_GPU_extraction_and_any_label_access",
        "role": ROLE,
        "controlled_change": {
            "unchanged": "Byte-identical v1 inputs: 793 answers, 8845 scored claims, 51953 selected claim/evidence-sentence pairs, pair order/provenance, BM25 retrieval, claim/token/window geometry, 35 feature formulas, source-group OOF LR readout, C/weights/labels/projection/thresholds/metrics.",
            "changed": f"Only frozen NLI tokenizer/checkpoint: v1 tasksource/ModernBERT-base-nli -> official NLIChecker default {MODEL_ID}.",
            "v1_expected_sha256": V1_EXPECTED,
            "v1_semantics": original,
        },
        "official_identity": {
            "RefChecker_repository": "https://github.com/amazon-science/RefChecker",
            "commit": REFCHECKER_COMMIT,
            "source_files": OFFICIAL_FILES,
            "NLIChecker_default_model": MODEL_ID,
            "official_label_order": ["Entailment", "Neutral", "Contradiction"],
            "official_pair_orientation": "NLIChecker._check tokenizes batch_references first and batch_claims second: reference is premise, claim is hypothesis.",
            "is_joint_false": "CheckerBase.check flattens each claim across each reference passage and each optional reference segment, then calls _check(..., is_joint=False). Our already flattened evidence-sentence/claim pairs match only this individual-pair orientation/order concept.",
            "official_tokenization": "max_length=512, truncation=True, padding=True, return_token_type_ids=True, batch_size default16.",
            "official_nonjoint_segment_merge": "Per reference passage, official merge_ret gives Entailment priority, then Contradiction, else Neutral. This ablation does not use that discrete merge.",
        },
        "checkpoint": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "pytorch_weights_sha256": MODEL_SHA256,
            "download_manifest_sha256": MODEL_MANIFEST_SHA256,
            "files_sha256": MODEL_FILE_SHA256,
            "architecture": "RobertaForSequenceClassification; 24 layers, hidden1024, 16 heads, 355362819 instantiated parameters; checkpoint additionally stores the unused 1049600-parameter pooler; FP32 weights.",
            "config_class_order": list(CLASSES),
            "frozen": True,
            "training": False,
        },
        "execution": {
            "pair": "Exact v1 evidence sentence as reference/premise and exact v1 automatic answer claim as claim/hypothesis.",
            "pair_count": 51953,
            "tokenization": "Use official max_length512/truncation/padding/token-type setting. CPU prepare first proves every untruncated pair <=512 and official encoding is therefore lossless.",
            "GPU": f"Explicit later CUDA0 FP32 eval/inference, SDPA math, batch{BATCH}, TF32/autocast/compile disabled; no automatic fallback.",
            "readout": "Private import of v1 reuses its learning/projection/score functions; only output path and validated probability-cache loader are rebound.",
            "stages": "initialize -> CPU prepare -> CPU check; after review only gpu-smoke -> extract; score is a separate CPU process and only then opens development gold.",
        },
        "not_full_RefChecker": [
            "No RefChecker LLM extractor or its triplet/subsentence RCClaim representation; uses the pre-existing v1 automatic sentence claims.",
            "No RefChecker reference splitting/retriever or full-reference checking; uses v1 BM25 top-2 evidence sentences.",
            "No official argmax merge_ret, multi-passage merge, strict/major/soft response aggregator, localization component, or RefChecker benchmark labels/metrics.",
            "Uses our supervised RAGTruth claim-risk LR and original four-BPE window projection. Therefore report only as an in-house checkpoint/backbone ablation, never as RefChecker reproduction or formal baseline.",
        ],
        "resource_estimate": {
            "weights_on_disk_bytes": 1_425_814_813,
            "FP32_tensor_bytes": PARAMETER_BYTES,
            "RTX3070_8GB_peak_GiB_before_smoke": [2.0, 5.0],
            "full_51953_pair_minutes_before_measurement": [10, 25],
            "hard_peak_limit_GiB": 6.5,
            "warning": "Estimate only; gpu-smoke must measure before extraction. No fit/score or GPU has run at protocol freeze.",
        },
        "official_test_opened": False,
    }


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def private_v1(name):
    spec = importlib.util.spec_from_file_location(name, Path(v1.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_v1():
    assert q.sha(v1.__file__) == V1_EXPECTED["runner"]
    assert q.sha(V1_OUT / "protocol.json") == V1_EXPECTED["protocol"]
    assert q.sha(V1_OUT / "inputs.jsonl") == V1_EXPECTED["inputs"]
    assert q.sha(V1_OUT / "preparation_complete.json") == V1_EXPECTED["preparation"]
    rows, complete = v1.check_prepared()
    assert len(rows) == 793 and complete["scored_claims"] == 8845 and complete["pairs"] == 51953
    return rows, complete


def validate_model_files(load_state_metadata=False):
    manifest = q.read(MODEL / "download_manifest.json")
    assert q.sha(MODEL / "download_manifest.json") == MODEL_MANIFEST_SHA256
    assert manifest["status"] == "complete" and manifest["repo_id"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    listed = {row["filename"]: row for row in manifest["files"]}
    assert set(listed) == set(MODEL_FILE_SHA256)
    for name, expected in MODEL_FILE_SHA256.items():
        path = MODEL / name
        assert path.stat().st_size == listed[name]["size"]
        assert q.sha(path) == listed[name]["sha256"] == expected, name
    config = q.read(MODEL / "config.json")
    assert config["architectures"] == ["RobertaForSequenceClassification"]
    assert config["id2label"] == {"0": "entailment", "1": "neutral", "2": "contradiction"}
    assert config["label2id"] == {"entailment": 0, "neutral": 1, "contradiction": 2}
    assert config["num_hidden_layers"] == 24 and config["hidden_size"] == 1024
    assert config["num_attention_heads"] == 16 and config["max_position_embeddings"] == 514
    state_report = None
    if load_state_metadata:
        state = torch.load(MODEL / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
        state_report = {
            "state_dict_tensors": len(state),
            "parameters": sum(tensor.numel() for tensor in state.values()),
            "tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in state.values()),
            "classifier_weight_shape": list(state["classifier.out_proj.weight"].shape),
            "classifier_bias_shape": list(state["classifier.out_proj.bias"].shape),
        }
        assert state_report == {"state_dict_tensors": 395, "parameters": CHECKPOINT_PARAMETERS,
                                "tensor_bytes": PARAMETER_BYTES, "classifier_weight_shape": [3, 1024],
                                "classifier_bias_shape": [3]}
        del state
    return manifest, config, state_report


def fetch_official_sources():
    folder = OUT / "official_refchecker_source"
    folder.mkdir()
    for name, record in OFFICIAL_FILES.items():
        url = f"https://raw.githubusercontent.com/amazon-science/RefChecker/{REFCHECKER_COMMIT}/{record['repo_path']}"
        data = urllib.request.urlopen(url, timeout=30).read()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        (folder / name).write_bytes(data)
    return folder


def verify_official_semantics():
    folder = OUT / "official_refchecker_source"
    nli = (folder / "nli_checker.py").read_text(encoding="utf-8")
    base = (folder / "checker_base.py").read_text(encoding="utf-8")
    assert "LABELS = [\"Entailment\", \"Neutral\", \"Contradiction\"]" in nli
    assert f"model='{MODEL_ID}'" in nli
    assert "batch_size=16" in nli
    assert "batch_references, batch_claims, max_length=512, truncation=True" in nli
    assert "return_tensors=\"pt\", padding=True, return_token_type_ids=True" in nli
    assert "if is_joint:" in base and "else:" in base
    assert "for c_idx, claim in enumerate(claims):" in base
    assert "for idx_psg, seg_psg in enumerate(segments_all_psg):" in base
    assert "input_flattened.append([claim, seg, questions])" in base
    assert "claims=[inp[0] for inp in input_flattened]" in base
    assert "references=[inp[1] for inp in input_flattened]" in base
    assert "is_joint=False" in base
    assert 'if "Entailment" in ret:' in base and 'if "Contradiction" in ret:' in base
    return {
        "status": "official_source_semantics_verified",
        "commit": REFCHECKER_COMMIT,
        "source_sha256": {name: q.sha(folder / name) for name in OFFICIAL_FILES},
        "label_order": list(CLASSES),
        "pair_orientation": "reference/premise first; claim/hypothesis second",
        "is_joint_false_individual_flattening": True,
        "official_discrete_segment_merge_not_used_by_ablation": True,
    }


def nonjoint_flatten(claims, references):
    """Toy mirror of the official is_joint=False nested pair order."""
    flattened = []
    for claim_index, claim in enumerate(claims):
        for passage_index, segments in enumerate(references):
            for segment_index, segment in enumerate(segments):
                flattened.append(((segment, claim), (claim_index, passage_index, segment_index)))
    return flattened


def synthetic_selfcheck():
    flattened = nonjoint_flatten(["c0", "c1"], [["p0s0", "p0s1"], ["p1s0"]])
    assert flattened == [
        (("p0s0", "c0"), (0, 0, 0)), (("p0s1", "c0"), (0, 0, 1)),
        (("p1s0", "c0"), (0, 1, 0)), (("p0s0", "c1"), (1, 0, 0)),
        (("p0s1", "c1"), (1, 0, 1)), (("p1s0", "c1"), (1, 1, 0)),
    ]
    toy = {"response_id": "toy", "passages": [
        {"passage_id": passage_id, "sentences": [
            {"sentence_id": 0, "text": f"p{passage_id}a", "text_sha256": v1.digest(f"p{passage_id}a")},
            {"sentence_id": 1, "text": f"p{passage_id}b", "text_sha256": v1.digest(f"p{passage_id}b")},
        ]} for passage_id in v1.PASSAGE_IDS], "claims": []}
    for claim_id in range(2):
        toy["claims"].append({"claim_id": claim_id, "text": f"c{claim_id}", "retrieval": [
            {"passage_id": passage_id, "selected": [
                {"rank": rank + 1, "sentence_id": rank, "evidence_sha256": v1.digest(f"p{passage_id}{'ab'[rank]}")}
                for rank in range(2)]} for passage_id in v1.PASSAGE_IDS]})
    pairs, owners = v1.row_pairs(toy)
    assert pairs[0] == ("p1a", "c0") and pairs[-1] == ("p3b", "c1")
    assert owners[0] == (0, 1, 1, 0) and owners[-1] == (1, 3, 2, 1)
    return {
        "status": "passed", "official_nonjoint_pair_order_mirrored": True,
        "reference_first_claim_second": True, "v1_pair_order_compatible_at_individual_pair_level": True,
        "complete_RefChecker_claim_extraction_or_merge_reproduced": False,
        "model_loaded": False, "GPU_initialized": torch.cuda.is_initialized(),
        "labels_accessed": False, "official_test_opened": False,
    }


def learning_source_identity():
    names = ("aggregate_claim_features", "nested_claim_weights", "project_claim_scores", "score")
    fresh = private_v1("_refchecker_ablation_v1_learning_identity")
    result = {name: inspect.getsource(getattr(fresh, name)) == inspect.getsource(getattr(v1, name)) for name in names}
    assert all(result.values())
    return {"functions": result, "v1_runner_sha256": q.sha(v1.__file__),
            "no_readout_code_modified": True}


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), "Do not overwrite an existing ablation path"
    validate_v1()
    validate_model_files(load_state_metadata=False)
    OUT.mkdir(parents=True)
    fetch_official_sources()
    q.save(OUT / "protocol.json", protocol())
    q.save(OUT / "PREFLIGHT.json", synthetic_selfcheck())
    q.save(OUT / "OFFICIAL_SEMANTICS_CHECK.json", verify_official_semantics())
    q.save(OUT / "READOUT_IDENTITY.json", learning_source_identity())
    (OUT / "IDENTITY.md").write_text(
        "# 身份说明：仅为官方 NLIChecker 骨干替换消融\n\n"
        "本实验只把 retrieved-evidence NLI v1 的冻结 NLI checkpoint 换成 RefChecker NLIChecker 默认的 RoBERTa-large。"
        "v1 的 51,953 个 pair、检索、claim、训练标签、LR 和 4-BPE 评测全部保持。\n\n"
        "它没有复现 RefChecker 的 triplet/subsentence 提取、reference 分段、argmax merge_ret、多 passage/整答聚合、定位器或官方 benchmark，"
        "所以不能叫 RefChecker baseline。`is_joint=False` 的核对范围仅是：官方逐 claim×reference 展开，且 reference 在前、claim 在后。\n",
        encoding="utf-8")
    print("REFCHECKER_NLI_BACKBONE_ABLATION_PROTOCOL_FROZEN", flush=True)


def source_paths():
    paths = [
        Path(__file__), Path(v1.__file__), Path(q.__file__), OUT / "protocol.json",
        OUT / "official_refchecker_source/nli_checker.py", OUT / "official_refchecker_source/checker_base.py",
        V1_OUT / "protocol.json", V1_OUT / "inputs.jsonl", V1_OUT / "preparation_complete.json",
        MODEL / "download_manifest.json",
    ]
    paths += [MODEL / name for name in MODEL_FILE_SHA256]
    return paths


def prepare():
    assert not torch.cuda.is_initialized()
    assert q.read(OUT / "protocol.json") == protocol()
    assert not (OUT / "prepare_started.json").exists(), "No silent preparation overwrite"
    rows, complete = validate_v1()
    _, config, state = validate_model_files(load_state_metadata=True)
    assert state["parameters"] == CHECKPOINT_PARAMETERS and state["tensor_bytes"] == PARAMETER_BYTES
    verify_official_semantics()
    snapshot = {str(path.resolve()): q.sha(path) for path in source_paths()}
    q.save(OUT / "prepare_started.json", {"status": "CPU_only_started", "source_sha256": snapshot,
           "model_instantiated": False, "GPU_used": False, "labels_accessed": False,
           "official_test_opened": False})
    copied = OUT / "inputs.jsonl"
    pending = copied.with_suffix(".jsonl.pending")
    pending.write_bytes((V1_OUT / "inputs.jsonl").read_bytes())
    pending.replace(copied)
    assert q.sha(copied) == V1_EXPECTED["inputs"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    assert tokenizer.is_fast and tokenizer.model_max_length >= PAIR_LIMIT
    total_pairs = 0
    length_histogram = {}
    maximum = 0
    longest = None
    for answer_index, row in enumerate(rows):
        pairs, owners = v1.row_pairs(row)
        raw = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                        add_special_tokens=True, padding=False, truncation=False,
                        return_token_type_ids=True)
        official = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                             add_special_tokens=True, padding=False, max_length=PAIR_LIMIT,
                             truncation=True, return_token_type_ids=True)
        assert raw["input_ids"] == official["input_ids"]
        assert raw["attention_mask"] == official["attention_mask"]
        assert raw["token_type_ids"] == official["token_type_ids"]
        assert all(set(values) <= {0} for values in raw["token_type_ids"])
        lengths = [len(ids) for ids in raw["input_ids"]]
        assert max(lengths) <= PAIR_LIMIT
        for length, owner in zip(lengths, owners):
            length_histogram[str(length)] = length_histogram.get(str(length), 0) + 1
            if length > maximum:
                maximum = length
                longest = {"answer_index": answer_index, "response_id": row["response_id"],
                           "owner": list(owner), "length": length}
        total_pairs += len(pairs)
        if (answer_index + 1) % 100 == 0:
            print("REFCHECKER_BACKBONE_TOKENIZED", answer_index + 1, len(rows), flush=True)
    del tokenizer
    assert total_pairs == complete["pairs"] == 51953
    expanded_lengths = np.repeat(np.asarray(sorted(map(int, length_histogram)), dtype=np.int32),
                                 np.asarray([length_histogram[str(key)] for key in sorted(map(int, length_histogram))], dtype=np.int64))
    q.save(OUT / "pair_length_statistics.json", {
        "answers": len(rows), "claims": complete["scored_claims"], "pairs": total_pairs,
        "RoBERTa_pair_tokens": {"min": int(expanded_lengths.min()),
                                "median": float(np.median(expanded_lengths)), "max": int(maximum)},
        "longest_pair": longest, "all_official_truncation_outputs_exactly_equal_untruncated": True,
        "all_token_type_ids_zero_as_RoBERTa_type_vocab_size1": True,
        "model_instantiated": False, "GPU_used": False, "labels_accessed": False,
        "official_test_opened": False,
    })
    q.save(OUT / "model_verification.json", {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "config": config,
        "state_dict": state, "weights_sha256": MODEL_SHA256,
        "download_manifest_sha256": MODEL_MANIFEST_SHA256,
        "class_order": list(CLASSES), "class_order_exact": True,
        "full_model_instantiated": False, "GPU_used": False,
    })
    q.save(OUT / "resource_plan.json", {
        "GPU": "RTX 3070 8GB", "pairs": total_pairs, "batch": BATCH,
        "maximum_RoBERTa_pair_tokens": maximum,
        "checkpoint_parameters_including_unused_pooler": CHECKPOINT_PARAMETERS,
        "instantiated_classifier_parameters": RUNTIME_PARAMETERS,
        "FP32_parameter_GiB": PARAMETER_BYTES / 1024 ** 3,
        "weights_on_disk_GiB": (MODEL / "pytorch_model.bin").stat().st_size / 1024 ** 3,
        "estimated_peak_GiB_before_smoke": [2.0, 5.0],
        "hard_stop_peak_GiB": PEAK_LIMIT / 1024 ** 3,
        "estimated_full_extraction_minutes_before_measurement": [10, 25],
        "estimate_basis": "Frozen FP32 356M RoBERTa-large, inference-only batch16 and observed max pair length; gpu-smoke required before extraction.",
        "GPU_measured": False, "GPU_used": False,
    })
    q.save(OUT / "source_snapshot.json", {"files_sha256": snapshot, "official_test_opened": False})
    assert snapshot == {str(path.resolve()): q.sha(path) for path in source_paths()}
    names = ["protocol.json", "PREFLIGHT.json", "OFFICIAL_SEMANTICS_CHECK.json",
             "READOUT_IDENTITY.json", "IDENTITY.md", "inputs.jsonl", "pair_length_statistics.json",
             "model_verification.json", "resource_plan.json", "source_snapshot.json"]
    q.save(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_waiting_for_GPU_review", "answers": 793, "claims": 8845,
        "pairs": 51953, "files_sha256": {name: q.sha(OUT / name) for name in names},
        "model_instantiated": False, "real_fits": 0, "GPU_used": False,
        "labels_accessed": False, "official_test_opened": False,
    })
    assert not torch.cuda.is_initialized()
    print("REFCHECKER_NLI_BACKBONE_ABLATION_CPU_PREPARED", flush=True)


def check_prepared():
    assert not torch.cuda.is_initialized()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_waiting_for_GPU_review"
    assert complete["answers"] == 793 and complete["claims"] == 8845 and complete["pairs"] == 51953
    for name, expected in complete["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    for name, expected in q.read(OUT / "source_snapshot.json")["files_sha256"].items():
        assert q.sha(name) == expected, name
    assert q.read(OUT / "protocol.json") == protocol()
    assert q.sha(OUT / "inputs.jsonl") == q.sha(V1_OUT / "inputs.jsonl") == V1_EXPECTED["inputs"]
    rows, _ = validate_v1()
    return rows, complete


def check():
    rows, complete = check_prepared()
    validate_model_files(load_state_metadata=False)
    official = verify_official_semantics()
    identity = learning_source_identity()
    assert q.read(OUT / "PREFLIGHT.json") == synthetic_selfcheck()
    q.save(OUT / "CPU_CHECK.json", {
        "status": "passed_waiting_for_GPU_review", "answers": len(rows),
        "claims": complete["claims"], "pairs": complete["pairs"],
        "official_semantics": official, "readout_identity": identity,
        "class_order_exact": True, "same_v1_pair_bytes": True,
        "full_RefChecker_reproduced": False, "model_instantiated": False,
        "real_fits": 0, "GPU_used": False, "labels_accessed": False,
        "official_test_opened": False,
    })
    assert not torch.cuda.is_initialized()
    print("REFCHECKER_NLI_BACKBONE_ABLATION_CPU_CHECK_PASSED", flush=True)


def runtime_signature():
    return {
        "backend": BACKEND, "source_sha256": q.sha(__file__),
        "protocol_sha256": q.sha(OUT / "protocol.json"), "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "model_sha256": MODEL_SHA256, "model_revision": MODEL_REVISION,
        "official_commit": REFCHECKER_COMMIT, "batch": BATCH, "precision": "float32",
        "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
        "cuda_runtime": torch.version.cuda, "role": ROLE,
    }


def infer_pairs(tokenizer, model, pairs, device):
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    output, shapes = [], []
    for left in range(0, len(pairs), BATCH):
        batch = pairs[left:left + BATCH]
        encoded = tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                            add_special_tokens=True, max_length=PAIR_LIMIT, truncation=True,
                            padding=True, return_token_type_ids=True, return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= PAIR_LIMIT
        assert not encoded["token_type_ids"].count_nonzero()
        shapes.append([len(batch), int(encoded["input_ids"].shape[1])])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                   if device.type == "cuda" else nullcontext())
        with torch.inference_mode(), torch.autocast(device.type, enabled=False), context:
            logits = model(**encoded).logits
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            output.append(logits.softmax(-1).cpu().numpy())
    result = np.concatenate(output).astype(np.float32, copy=False)
    assert result.shape == (len(pairs), 3) and np.isfinite(result).all()
    assert np.allclose(result.sum(1), 1, rtol=0, atol=2e-6)
    return result, shapes


def load_cuda():
    validate_model_files(load_state_metadata=False)
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info(0)
    assert free >= MIN_FREE
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa").eval().requires_grad_(False).to("cuda:0")
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert sum(parameter.numel() for parameter in model.parameters()) == RUNTIME_PARAMETERS
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cuda" and
               not parameter.requires_grad for parameter in model.parameters())
    return tokenizer, model, {"device": torch.cuda.get_device_name(0), "free_before_load": free,
                              "total_memory": total,
                              "checkpoint_parameters_including_unused_pooler": CHECKPOINT_PARAMETERS,
                              "instantiated_classifier_parameters": RUNTIME_PARAMETERS}


def memory_gate():
    values = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
              "reserved_peak_bytes": torch.cuda.max_memory_reserved(0)}
    assert max(values.values()) <= PEAK_LIMIT, values
    return values


def clean_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cache_signature(row):
    pairs, owners = v1.row_pairs(row)
    return {
        "response_id": row["response_id"], "input_row_sha256": v1.digest(row),
        "pair_definition_sha256": v1.digest({"pairs": pairs, "owners": owners}),
        "input_file_sha256": q.sha(OUT / "inputs.jsonl"), "protocol_sha256": q.sha(OUT / "protocol.json"),
        "model_sha256": MODEL_SHA256, "model_revision": MODEL_REVISION,
        "execution_signature_sha256": digest(runtime_signature()), "execution_backend": BACKEND,
        "method_role": ROLE,
    }


def validate_cache(path, row):
    pairs, owners = v1.row_pairs(row)
    signature = cache_signature(row)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(signature) | {"probabilities", "claim_ids", "passage_ids", "ranks", "sentence_ids"}
        for key, value in signature.items():
            assert data[key].item() == value, (path.name, key)
        probability = data["probabilities"]
        assert probability.dtype == np.float32 and probability.shape == (len(pairs), 3)
        assert np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
        assert data["claim_ids"].tolist() == [owner[0] for owner in owners]
        assert data["passage_ids"].tolist() == [owner[1] for owner in owners]
        assert data["ranks"].tolist() == [owner[2] for owner in owners]
        assert data["sentence_ids"].tolist() == [owner[3] for owner in owners]
        return probability.copy()


def gpu_smoke():
    rows, _ = check_prepared()
    assert q.read(OUT / "CPU_CHECK.json")["status"] == "passed_waiting_for_GPU_review"
    assert not (OUT / "GPU_SMOKE.json").exists()
    longest_id = q.read(OUT / "pair_length_statistics.json")["longest_pair"]["response_id"]
    selected = list(dict.fromkeys([0, len(rows) // 2, len(rows) - 1,
                                   next(i for i, row in enumerate(rows) if row["response_id"] == longest_id)]))
    tokenizer = model = None
    started = time.perf_counter()
    try:
        tokenizer, model, loaded = load_cuda()
        pairs, owners = [], []
        for index in selected:
            one, one_owners = v1.row_pairs(rows[index])
            pairs.extend(one)
            owners.extend((rows[index]["response_id"], *owner) for owner in one_owners)
        first, shapes = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
        second, shapes2 = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
        assert shapes == shapes2 and np.array_equal(first, second)
        np.savez_compressed(OUT / "GPU_SMOKE_PROBABILITIES.npz", probabilities=first,
                            owners_json=np.asarray(json.dumps(owners, separators=(",", ":"))),
                            execution_signature_sha256=np.asarray(digest(runtime_signature())))
        q.save(OUT / "GPU_SMOKE.json", {
            "status": "passed_no_automatic_extract", "selected_response_ids": [rows[i]["response_id"] for i in selected],
            "pairs": len(pairs), "batch_shapes": shapes, "same_path_repeat_exact": True,
            "loaded": loaded, **memory_gate(), "seconds": time.perf_counter() - started,
            "probability_sha256": q.sha(OUT / "GPU_SMOKE_PROBABILITIES.npz"),
            "execution_signature_sha256": digest(runtime_signature()),
            "GPU_used": True, "trained": False, "official_test_opened": False,
        })
        print("REFCHECKER_NLI_BACKBONE_GPU_SMOKE_PASSED_NO_AUTOMATIC_EXTRACT", flush=True)
    finally:
        del model, tokenizer
        clean_gpu()


def extract():
    rows, complete = check_prepared()
    smoke = q.read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_no_automatic_extract"
    assert smoke["execution_signature_sha256"] == digest(runtime_signature())
    folder = OUT / "pair_scores"
    folder.mkdir(exist_ok=True)
    missing = []
    for index, row in enumerate(rows):
        path = folder / f"{row['response_id']}.npz"
        if path.exists():
            validate_cache(path, row)
        else:
            missing.append(index)
    tokenizer = model = None
    started = time.perf_counter()
    try:
        if missing:
            tokenizer, model, loaded = load_cuda()
            for completed, index in enumerate(missing, 1):
                row = rows[index]
                pairs, owners = v1.row_pairs(row)
                probability, _ = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
                path = folder / f"{row['response_id']}.npz"
                pending = path.with_suffix(".npz.pending")
                with pending.open("wb") as handle:
                    np.savez_compressed(handle, **{key: np.asarray(value) for key, value in cache_signature(row).items()},
                        probabilities=probability,
                        claim_ids=np.asarray([owner[0] for owner in owners], np.int32),
                        passage_ids=np.asarray([owner[1] for owner in owners], np.int8),
                        ranks=np.asarray([owner[2] for owner in owners], np.int8),
                        sentence_ids=np.asarray([owner[3] for owner in owners], np.int32))
                pending.replace(path)
                validate_cache(path, row)
                memory_gate()
                if completed == 1 or completed % 25 == 0 or completed == len(missing):
                    progress = {"complete": len(rows) - len(missing) + completed, "total": len(rows),
                                "seconds_this_run": time.perf_counter() - started, "pid": os.getpid()}
                    q.save(OUT / "progress.json", progress)
                    print("REFCHECKER_NLI_BACKBONE_EXTRACT", progress, flush=True)
        files = {}
        for row in rows:
            path = folder / f"{row['response_id']}.npz"
            validate_cache(path, row)
            files[path.name] = q.sha(path)
        q.save(OUT / "extraction_complete.json", {
            "status": "complete_frozen_probabilities_not_scored", "answers": len(rows),
            "claims": complete["claims"], "pairs": complete["pairs"], "files_sha256": files,
            "input_sha256": q.sha(OUT / "inputs.jsonl"), "protocol_sha256": q.sha(OUT / "protocol.json"),
            "model_sha256": MODEL_SHA256, "model_revision": MODEL_REVISION,
            "execution_signature_sha256": digest(runtime_signature()),
            "new_rows_this_run": len(missing), "seconds_this_run": time.perf_counter() - started,
            "GPU_used": bool(missing), "trained": False, "annotation_values_accessed": False,
            "official_test_opened": False,
        })
        print("REFCHECKER_NLI_BACKBONE_EXTRACTION_COMPLETE_NO_AUTOMATIC_SCORE", flush=True)
    finally:
        del model, tokenizer
        clean_gpu()


def check_extracted(rows):
    complete = q.read(OUT / "extraction_complete.json")
    assert complete["status"] == "complete_frozen_probabilities_not_scored"
    assert complete["answers"] == 793 and complete["claims"] == 8845 and complete["pairs"] == 51953
    assert complete["input_sha256"] == q.sha(OUT / "inputs.jsonl")
    assert complete["protocol_sha256"] == q.sha(OUT / "protocol.json")
    assert complete["model_sha256"] == MODEL_SHA256 and complete["model_revision"] == MODEL_REVISION
    assert complete["execution_signature_sha256"] == digest(runtime_signature())
    for row in rows:
        path = OUT / "pair_scores" / f"{row['response_id']}.npz"
        assert q.sha(path) == complete["files_sha256"][path.name]
        validate_cache(path, row)
    return complete


def bound_readout():
    bound = private_v1("_refchecker_nli_backbone_exact_v1_readout")
    bound.OUT = OUT
    bound.protocol = protocol
    bound.check_prepared = check_prepared
    bound.check_extracted = check_extracted
    bound.validate_cache = validate_cache
    for name in ("aggregate_claim_features", "nested_claim_weights", "project_claim_scores", "score"):
        assert inspect.getsource(getattr(bound, name)) == inspect.getsource(getattr(v1, name))
    return bound


def score():
    assert not torch.cuda.is_initialized()
    bound_readout().score()
    q.save(OUT / "ABLATION_IDENTITY_AFTER_SCORE.json", {
        "complete_RefChecker_baseline": False,
        "identity": "Only official NLIChecker checkpoint backbone substituted into our v1 retrieval/readout.",
        "official_test_opened": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "gpu-smoke", "extract", "score"))
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
        elif arguments.stage == "gpu-smoke":
            gpu_smoke()
        elif arguments.stage == "extract":
            extract()
        else:
            score()
