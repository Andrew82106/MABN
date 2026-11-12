"""Rebuild the train-only intake manifest from audited artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_ROWS = 1842


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    api = json.loads((ROOT / "HF_DATASET_API.json").read_text(encoding="utf-8-sig"))
    tree = json.loads((ROOT / "HF_TREE_API.json").read_text(encoding="utf-8-sig"))
    audit = json.loads((ROOT / "AUDIT.json").read_text(encoding="utf-8"))
    required_count_fields = (
        "question_present",
        "documents_present",
        "documents_str_present",
        "rag_prompt_present",
        "llama_response_present",
        "response_text_present",
        "full_prompt_present",
    )
    required_field_failures = sum(
        EXPECTED_ROWS - int(audit["required_fields"][name])
        for name in required_count_fields
    )
    assert required_field_failures == 0
    assert audit["required_fields"]["span_list_present_in_schema"] is True

    names = [
        "HF_DATASET_API.json",
        "HF_TREE_API.json",
        "OFFICIAL_README.md",
        "train-00000-of-00001.parquet",
        "PROTOCOL.json",
        "audit_train.py",
        "build_manifest.py",
        "AUDIT.json",
        "SCHEMA.json",
        "GROUP_INDEX.jsonl",
        "ALL_TRAIN_INDEX.jsonl",
        "NO_UNANSWERABILITY_HINT_INDEX.jsonl",
        "DATA_QUALITY_REPORT.md",
        "REPORT.md",
    ]
    artifacts = {
        name: {"bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)}
        for name in names
    }
    forbidden = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("*test*.parquet", "*validation*.parquet")
        for path in ROOT.rglob(pattern)
    )
    assert not forbidden
    train_hash = artifacts["train-00000-of-00001.parquet"]["sha256"]
    official_train_hash = "6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44"
    assert train_hash == official_train_hash

    manifest = {
        "version": "ragognize-train-only-intake-v1",
        "status": "complete_train_only_audited_not_trained",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Hugging Face official dataset API",
            "repository": api["id"],
            "revision": api["sha"],
            "last_modified": api.get("lastModified"),
            "private": api.get("private"),
            "gated": api.get("gated"),
            "license": api.get("cardData", {}).get("license"),
            "dataset_api_url": "https://huggingface.co/api/datasets/F4biian/RAGognize",
            "pinned_tree_api_url": "https://huggingface.co/api/datasets/F4biian/RAGognize/tree/aab54518c2a7c0d25fff8bffbf5337d0321de142?recursive=true&expand=true",
            "pinned_train_url": "https://huggingface.co/datasets/F4biian/RAGognize/resolve/aab54518c2a7c0d25fff8bffbf5337d0321de142/data/train-00000-of-00001.parquet?download=true",
        },
        "physical_split_check": {
            "separable": True,
            "repository_files": [
                {
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size"),
                    "oid": item.get("oid"),
                    "lfs": item.get("lfs"),
                }
                for item in tree
            ],
            "downloaded_content_files": [
                "README.md",
                "data/train-00000-of-00001.parquet",
            ],
            "test_or_validation_content_downloaded": False,
            "test_or_validation_content_read": False,
            "forbidden_local_parquet_matches": forbidden,
        },
        "train": {
            "file": "train-00000-of-00001.parquet",
            "bytes": artifacts["train-00000-of-00001.parquet"]["bytes"],
            "sha256": train_hash,
            "official_lfs_oid": official_train_hash,
            "rows": audit["row_counts"]["train_prompt_rows"],
            "model_subset": "Llama-2-7b-chat-hf",
            "model_response_rows": audit["row_counts"]["llama_2_response_rows"],
        },
        "frozen_label_blind_subsets": {
            "all_train": {
                "index": "ALL_TRAIN_INDEX.jsonl",
                "rows": 1842,
                "questions": 943,
                "recommended_groups": 917,
                "index_sha256": artifacts["ALL_TRAIN_INDEX.jsonl"]["sha256"],
            },
            "no_unanswerability_hint": {
                "index": "NO_UNANSWERABILITY_HINT_INDEX.jsonl",
                "selection": "template_details.settings.unanswerability_hint == false only",
                "rows": 971,
                "questions": 498,
                "recommended_groups": 491,
                "index_sha256": artifacts["NO_UNANSWERABILITY_HINT_INDEX.jsonl"]["sha256"],
            },
            "selection_used_labels_or_model_performance": False,
        },
        "quality_summary": {
            "required_count_fields": list(required_count_fields),
            "required_field_failures": required_field_failures,
            "span_list_present_in_schema": True,
            "valid_spans": audit["annotation_integrity"]["total_valid_spans"],
            "span_coordinate_or_text_failures": (
                audit["issues"].get("span_out_of_bounds", 0)
                + audit["issues"].get("span_text_coordinate_mismatch", 0)
            ),
            "four_bpe_mapping_ready": audit["four_bpe_mapping"]["mapping_ready"],
            "canonical_official_token_start_mismatch_rows": audit["issues"].get(
                "canonical_official_token_starts_differ_from_local_tokenizer", 0
            ),
            "overlapping_span_rows": audit["issues"].get("overlapping_gold_spans", 0),
            "human_gold": False,
        },
        "operations": {
            "model_forward": False,
            "training": False,
            "GPU_used": False,
            "baseline_modified": False,
        },
        "artifacts": artifacts,
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(ROOT / "MANIFEST.json"),
        "manifest_sha256": sha256(ROOT / "MANIFEST.json"),
        "required_field_failures": required_field_failures,
        "artifact_count": len(artifacts),
    }, indent=2))


if __name__ == "__main__":
    main()
