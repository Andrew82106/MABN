"""Build paired blind annotator files from public MAS episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))

from paperalpha_runtime.mas_annotation import (
    DEFAULT_CLASSES,
    build_annotation_rows,
    load_json_records,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path, help="public episode JSONL/JSON")
    parser.add_argument("--projection", type=Path, help="optional public monitor projection JSONL/JSON")
    parser.add_argument("--output", required=True, type=Path, help="new packet directory")
    parser.add_argument("--salt", default="paperalpha-mas-blind-v1", help="non-secret reproducibility salt")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="optional deterministic prefix for a pilot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes = load_json_records(args.episodes)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        episodes = episodes[: args.limit]
    projections = load_json_records(args.projection) if args.projection else []
    rows, source_index = build_annotation_rows(
        episodes,
        projections,
        salt=args.salt,
        shuffle_seed=args.shuffle_seed,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "annotator_a.jsonl", rows)
    write_jsonl(output / "annotator_b.jsonl", rows)
    write_jsonl(output / "source_index.jsonl", source_index)
    manifest = {
        "schema": "paperalpha.mas_blind_annotation.v1",
        "n_items": len(rows),
        "classes": list(DEFAULT_CLASSES),
        "annotator_files": ["annotator_a.jsonl", "annotator_b.jsonl"],
        "coordinator_file": "source_index.jsonl",
        "blindness": {
            "labels_removed": True,
            "hidden_fields_removed": True,
            "model_scores_removed": True,
            "source_episode_ids_removed": True,
        },
        "instructions": "Annotators must cite evidence_id values; first_violation_event_id must be one cited evidence_id or null.",
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

