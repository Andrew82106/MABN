"""Assemble a complete judge run, preferring later retry predictions on overlap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_predictions(directory: Path) -> list[dict]:
    path = directory / "predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_ids(source: Path) -> set[str]:
    result: set[str] = set()
    for prompt in source.glob("prompt_*.json"):
        payload = json.loads(prompt.read_text(encoding="utf-8").split("\n", 1)[1])
        result.update(str(row["episode_id"]) for row in payload)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--retry", type=Path, action="append", required=True, help="retry output; later arguments take precedence")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    by_id: dict[str, dict] = {}
    for directory in [args.source, *args.retry]:
        for row in read_predictions(directory):
            episode_id = str(row.get("episode_id", ""))
            if not episode_id:
                raise SystemExit(f"prediction without episode_id in {directory}")
            by_id[episode_id] = row
    expected = expected_ids(args.source)
    missing = sorted(expected - set(by_id))
    extra = sorted(set(by_id) - expected)
    if missing or extra:
        raise SystemExit(json.dumps({"missing": missing, "extra": extra}, ensure_ascii=False))
    merged = [by_id[key] for key in sorted(expected)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged), encoding="utf-8"
    )
    (args.out / "manifest.json").write_text(json.dumps({
        "protocol_version": "llm-judge-baseline-assembled-v1",
        "source": str(args.source),
        "retries": [str(path) for path in args.retry],
        "expected_predictions": len(expected),
        "assembled_predictions": len(merged),
        "duplicate_policy": "later retry output replaces earlier output by episode_id",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"expected_predictions": len(expected), "assembled_predictions": len(merged), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
