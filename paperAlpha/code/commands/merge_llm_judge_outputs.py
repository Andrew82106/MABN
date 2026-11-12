"""Merge a baseline output with a disjoint retry output without overwriting either."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--recovery", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    first = rows(args.source)
    second = rows(args.recovery)
    merged = first + second
    ids = [str(row.get("episode_id", "")) for row in merged]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate episode_id across source and recovery outputs")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged), encoding="utf-8"
    )
    (args.out / "manifest.json").write_text(json.dumps({
        "protocol_version": "llm-judge-baseline-merged-v1",
        "source": str(args.source),
        "recovery": str(args.recovery),
        "source_predictions": len(first),
        "recovery_predictions": len(second),
        "merged_predictions": len(merged),
        "unique_episode_ids": len(set(ids)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_predictions": len(first), "recovery_predictions": len(second), "merged_predictions": len(merged), "unique_episode_ids": len(set(ids)), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
