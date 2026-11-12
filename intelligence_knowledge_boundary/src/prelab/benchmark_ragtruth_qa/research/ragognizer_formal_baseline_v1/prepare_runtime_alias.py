"""Create offline path aliases without changing any official model tensor.

Large base tensors and the two released learned files are hard-linked, so the
runtime view is byte-identical and consumes no second copy of their contents.
Only PEFT's locator string is rewritten in a copied adapter_config.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
BASE_SOURCE = WORKSPACE / "prelab" / "models" / "Llama-2-7b-chat-hf"
BASE_META = HERE / "official_base_tokenizer_meta_f5db"
MODEL_SOURCE = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf"
BASE_RUNTIME = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf-base-f5db-runtime"
CHECKPOINT_RUNTIME = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf-offline-runtime"
AUDIT = HERE / "RUNTIME_ALIAS_AUDIT.json"

BASE_WEIGHT_HASHES = {
    "model-00001-of-00002.safetensors": "66dec18c9f1705b9387d62f8485f4e7d871ca388718786737ed3c72dbfaac9fb",
    "model-00002-of-00002.safetensors": "0fd6895090da1b2ccffdb93964847709a3b31e6b69fe7dc5a480dce37c811b1d",
}
BASE_META_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
)
CHECKPOINT_HARDLINKS = (
    "adapter_model.safetensors", "hallu_head_neg_16.safetensors",
)
CHECKPOINT_COPIES = ("head_configs.json", "README.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        destination.unlink()
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"hard-link identity check failed: {destination}")


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"copy hash mismatch: {destination}")


def main() -> None:
    BASE_RUNTIME.mkdir(parents=True, exist_ok=True)
    runtime_files: list[dict] = []
    for name, expected in BASE_WEIGHT_HASHES.items():
        source = BASE_SOURCE / name
        if sha256_file(source) != expected:
            raise RuntimeError(f"base shard hash mismatch: {name}")
        destination = BASE_RUNTIME / name
        replace_hardlink(source, destination)
        runtime_files.append({
            "path": str(destination.relative_to(WORKSPACE)), "source": str(source.relative_to(WORKSPACE)),
            "mode": "hardlink", "bytes": destination.stat().st_size, "sha256": expected,
        })
    for name in BASE_META_FILES:
        source = BASE_META / name
        destination = BASE_RUNTIME / name
        copy_exact(source, destination)
        runtime_files.append({
            "path": str(destination.relative_to(WORKSPACE)), "source": str(source.relative_to(WORKSPACE)),
            "mode": "exact_copy", "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
        })

    runtime_ft = CHECKPOINT_RUNTIME / "ft_llm"
    runtime_ft.mkdir(parents=True, exist_ok=True)
    original_ft = MODEL_SOURCE / "ft_llm"
    for name in CHECKPOINT_HARDLINKS:
        source = original_ft / name
        destination = runtime_ft / name
        replace_hardlink(source, destination)
        runtime_files.append({
            "path": str(destination.relative_to(WORKSPACE)), "source": str(source.relative_to(WORKSPACE)),
            "mode": "hardlink", "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
        })
    for name in CHECKPOINT_COPIES:
        source = original_ft / name
        destination = runtime_ft / name
        copy_exact(source, destination)
        runtime_files.append({
            "path": str(destination.relative_to(WORKSPACE)), "source": str(source.relative_to(WORKSPACE)),
            "mode": "exact_copy", "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
        })

    original_config_path = original_ft / "adapter_config.json"
    original_config = json.loads(original_config_path.read_text(encoding="utf-8"))
    runtime_config = dict(original_config)
    runtime_config["base_model_name_or_path"] = str(BASE_RUNTIME.resolve())
    runtime_config_path = runtime_ft / "adapter_config.json"
    runtime_config_path.write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    differing_keys = sorted(
        key for key in set(original_config) | set(runtime_config)
        if original_config.get(key) != runtime_config.get(key)
    )
    if differing_keys != ["base_model_name_or_path"]:
        raise RuntimeError(f"unexpected adapter config changes: {differing_keys}")

    # These files are metadata for the diagnostic separate-MLP path and native
    # threshold appendix.  They are copied verbatim; route A does not load the MLP.
    for name in ("mlp_config.json", "mlp_other_data.json", "mlp_state.pt"):
        source = MODEL_SOURCE / name
        destination = CHECKPOINT_RUNTIME / name
        if name == "mlp_state.pt":
            replace_hardlink(source, destination)
            mode = "hardlink"
        else:
            copy_exact(source, destination)
            mode = "exact_copy"
        runtime_files.append({
            "path": str(destination.relative_to(WORKSPACE)), "source": str(source.relative_to(WORKSPACE)),
            "mode": mode, "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
        })

    audit = {
        "version": "ragognizer-offline-runtime-alias-v1",
        "status": "ready",
        "purpose": "offline path resolution only; no model architecture, tensor, head, LoRA, loss, or score change",
        "base_revision": "meta-llama/Llama-2-7b-chat-hf@f5db02db724555f92da89c216ac04704f23d4590",
        "base_runtime_path": str(BASE_RUNTIME.relative_to(WORKSPACE)),
        "checkpoint_revision": "F4biian/RAGognizer-Llama-2-7b-chat-hf@2b58ab9aa73f6d499ec72a38ced0976caf78b26f",
        "checkpoint_runtime_path": str(CHECKPOINT_RUNTIME.relative_to(WORKSPACE)),
        "adapter_config": {
            "original_sha256": sha256_file(original_config_path),
            "runtime_sha256": sha256_file(runtime_config_path),
            "only_changed_key": "base_model_name_or_path",
            "original_value": original_config["base_model_name_or_path"],
            "runtime_value": runtime_config["base_model_name_or_path"],
        },
        "files": runtime_files,
        "all_learned_files_byte_identical": all(
            item["sha256"] == sha256_file(WORKSPACE / item["source"])
            for item in runtime_files
            if item["source"].endswith((".safetensors", ".pt"))
        ),
        "labels_read": False,
        "gpu_workload_started": False,
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "files": len(runtime_files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
