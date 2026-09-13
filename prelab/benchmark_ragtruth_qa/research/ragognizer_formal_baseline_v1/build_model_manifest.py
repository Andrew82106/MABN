"""Generate byte and tensor metadata for the pinned RAGognizer baseline."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = HERE.parents[3]
MODEL = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf"
BASE = WORKSPACE / "prelab" / "models" / "Llama-2-7b-chat-hf"
BASE_META = HERE / "official_base_tokenizer_meta_f5db"
RAG_REPO = PROJECT / "third_party" / "RAGognizer"
HEAD_REPO = PROJECT / "third_party" / "transformer-heads"
PAPER = HERE / "sources" / "RAGognizer_arXiv_2604.15945v1.pdf"
OUTPUT = HERE / "MODEL_MANIFEST.json"

MODEL_REVISION = "2b58ab9aa73f6d499ec72a38ced0976caf78b26f"
RAG_COMMIT = "1d3e8fdd6de638dc2d06265829a4d9ced387e0ba"
HEAD_COMMIT = "6a2ca2a12a25035ea290b0b0e03839fc16348e44"
BASE_META_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"

MODEL_FILES = (
    ".gitattributes", "README.md", "ft_llm/README.md", "ft_llm/adapter_config.json",
    "ft_llm/adapter_model.safetensors", "ft_llm/hallu_head_neg_16.safetensors",
    "ft_llm/head_configs.json", "mlp_config.json", "mlp_other_data.json", "mlp_state.pt",
)
RAG_SOURCE_FILES = (
    "README.md", "ragognizer/pyproject.toml",
    "ragognizer/ragognizer/detectors/RAGognizer.py",
    "ragognizer/ragognizer/detectors/detector.py",
    "fine-tuning/ft.py", "fine-tuning/requirements.txt", "fine-tuning/inference_demo.py",
)
HEAD_SOURCE_FILES = (
    "pyproject.toml", "transformer_heads/util/load_model.py",
    "transformer_heads/util/prepare_model.py", "transformer_heads/model/model.py",
    "transformer_heads/model/head.py", "transformer_heads/config.py",
    "transformer_heads/constants.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(root: Path, name: str) -> dict:
    path = root / name
    return {"path": name.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def safetensor_summary(path: Path) -> dict:
    shapes: dict[str, list[int]] = {}
    dtype_tensors: Counter[str] = Counter()
    parameters = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = list(map(int, handle.get_slice(key).get_shape()))
            tensor = handle.get_tensor(key)
            shapes[key] = shape
            parameters += math.prod(shape)
            dtype_tensors[str(tensor.dtype)] += 1
    return {
        "tensor_count": len(shapes),
        "parameter_count": parameters,
        "dtype_tensor_counts": dict(sorted(dtype_tensors.items())),
        "shapes": shapes,
    }


def main() -> None:
    if git_head(RAG_REPO) != RAG_COMMIT or git_head(HEAD_REPO) != HEAD_COMMIT:
        raise RuntimeError("pinned source commit changed")
    present = sorted(
        path.relative_to(MODEL).as_posix()
        for path in MODEL.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    )
    if present != sorted(MODEL_FILES):
        raise RuntimeError(f"official model file set changed: {present}")

    adapter = safetensor_summary(MODEL / "ft_llm" / "adapter_model.safetensors")
    head = safetensor_summary(MODEL / "ft_llm" / "hallu_head_neg_16.safetensors")
    mlp_state = torch.load(MODEL / "mlp_state.pt", map_location="cpu", weights_only=True)
    mlp = {
        "tensor_count": len(mlp_state),
        "parameter_count": sum(int(tensor.numel()) for tensor in mlp_state.values()),
        "dtype_tensor_counts": dict(sorted(Counter(str(tensor.dtype) for tensor in mlp_state.values()).items())),
        "shapes": {key: list(map(int, tensor.shape)) for key, tensor in mlp_state.items()},
    }
    adapter_config = json.loads((MODEL / "ft_llm" / "adapter_config.json").read_text(encoding="utf-8"))
    head_config = json.loads((MODEL / "ft_llm" / "head_configs.json").read_text(encoding="utf-8"))
    mlp_config = json.loads((MODEL / "mlp_config.json").read_text(encoding="utf-8"))
    other_data = json.loads((MODEL / "mlp_other_data.json").read_text(encoding="utf-8"))

    base_weights = [record(BASE, name) for name in (
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    )]
    official_index_hash = sha256_file(BASE_META / "model.safetensors.index.json")
    if base_weights[-1]["sha256"] != official_index_hash:
        raise RuntimeError("local base index differs from official Meta revision")

    manifest = {
        "version": "ragognizer-formal-model-manifest-v1",
        "date": "2026-09-13",
        "method_status": "2026 arXiv preprint; no conference or journal venue claimed",
        "official_model": {
            "repo": "F4biian/RAGognizer-Llama-2-7b-chat-hf",
            "revision": MODEL_REVISION,
            "revision_date": "2026-04-20",
            "local_path": str(MODEL.relative_to(WORKSPACE)),
            "download_complete": True,
            "remote_revision_sibling_count": 10,
            "remote_lfs_sha256_verified_against_local": True,
            "files": [record(MODEL, name) for name in MODEL_FILES],
            "total_bytes": sum((MODEL / name).stat().st_size for name in MODEL_FILES),
        },
        "learned_components": {
            "lora_adapter": adapter,
            "integrated_transformer_heads_detection_head": head,
            "separate_mlp_diagnostic": mlp,
            "integrated_head_config": head_config,
            "lora_config": adapter_config,
            "separate_mlp_config": mlp_config,
            "released_other_data": other_data,
            "integrated_and_separate_heads_same_architecture": False,
            "integrated_and_separate_heads_same_parameter_count": False,
        },
        "official_source": {
            "repo": "F4biian/RAGognizer",
            "commit": RAG_COMMIT,
            "files": [record(RAG_REPO, name) for name in RAG_SOURCE_FILES],
        },
        "resolved_unpinned_transformer_heads_dependency": {
            "repo": "F4biian/transformer-heads",
            "commit_used_for_this_audit": HEAD_COMMIT,
            "author_package_pins_commit": False,
            "files": [record(HEAD_REPO, name) for name in HEAD_SOURCE_FILES],
        },
        "base_model": {
            "adapter_declared_repo": adapter_config["base_model_name_or_path"],
            "adapter_declared_revision": adapter_config.get("revision"),
            "official_metadata_revision_used": BASE_META_REVISION,
            "local_weight_origin": "NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee",
            "weight_and_index_files": base_weights,
            "weight_shard_bytes": sum(item["bytes"] for item in base_weights[:2]),
            "official_metadata_files": [record(BASE_META, name) for name in (
                "config.json", "generation_config.json", "model.safetensors.index.json",
                "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
            )],
            "local_weight_shards_match_official_meta_revision_lfs_hashes": True,
            "base_revision_provenance_limit": "adapter_config revision is null; the released adapter does not pin a base commit",
        },
        "paper": {
            "title": "RAGognizer: Hallucination-Aware Fine-Tuning via Detection Head Integration",
            "arxiv": "2604.15945v1",
            "submitted": "2026-04-17",
            "local_path": str(PAPER.relative_to(WORKSPACE)),
            "bytes": PAPER.stat().st_size,
            "sha256": sha256_file(PAPER),
        },
        "safety_boundary": {
            "ragognize_test_downloaded": False,
            "ragognize_test_opened": False,
            "gpu_workload_started": False,
        },
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_files": len(manifest["official_model"]["files"]),
        "model_bytes": manifest["official_model"]["total_bytes"],
        "adapter_parameters": adapter["parameter_count"],
        "integrated_head_parameters": head["parameter_count"],
        "separate_mlp_parameters": mlp["parameter_count"],
    }))


if __name__ == "__main__":
    main()
