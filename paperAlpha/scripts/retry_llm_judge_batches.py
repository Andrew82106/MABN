"""Retry only failed batches from a completed llm_judge_baseline run.

The source run is never modified.  Prompts are reused verbatim so the retry
has exactly the same public input and batch assignment as the original run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_baseline_module():
    path = Path(__file__).with_name("llm_judge_baseline.py")
    spec = importlib.util.spec_from_file_location("llm_judge_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="completed baseline output")
    ap.add_argument("--out", type=Path, required=True, help="new recovery output")
    ap.add_argument("--endpoint", default=os.getenv("OPENAI_BASE_URL", "http://localhost:58661/v1"))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--key-env", default="OPENAI_API_KEY")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, action="append", dest="requested_batches", help="explicit batch id; may be repeated")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    key = os.getenv(args.key_env, "")
    if not key:
        raise SystemExit(f"{args.key_env} is required")

    baseline = load_baseline_module()
    errors = read_jsonl(args.source / "errors.jsonl")
    batch_ids = sorted({int(row["batch"]) for row in errors})
    if args.requested_batches:
        batch_ids = sorted(set(args.requested_batches))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps({
            "protocol_version": "llm-judge-baseline-recovery-v1",
            "source": str(args.source),
            "batch_ids": batch_ids,
            "batch_count": len(batch_ids),
            "endpoint": args.endpoint,
            "model": args.model,
            "temperature": 0,
            "prompt_policy": "prompts copied verbatim from source run",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def run_one(batch: int):
        prompt_path = args.source / f"prompt_{batch:04d}.json"
        if not prompt_path.exists():
            return batch, [], {"batch": batch, "error_type": "FileNotFoundError", "error": str(prompt_path)}
        prompt = prompt_path.read_text(encoding="utf-8")
        (args.out / f"prompt_{batch:04d}.json").write_text(prompt, encoding="utf-8")
        try:
            value = baseline.call_judge(args.endpoint, args.model, key, prompt, args.timeout, args.retries)
            return batch, value if isinstance(value, list) else [value], None
        except Exception as exc:  # preserve the terminal error for auditability
            return batch, [], {"batch": batch, "error_type": type(exc).__name__, "error": str(exc)}

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, batch) for batch in batch_ids]
        for future in as_completed(futures):
            completed.append(future.result())
    predictions: list[dict] = []
    retry_errors: list[dict] = []
    for _, batch_predictions, error in sorted(completed, key=lambda item: item[0]):
        predictions.extend(batch_predictions)
        if error is not None:
            retry_errors.append(error)
    (args.out / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    (args.out / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in retry_errors), encoding="utf-8"
    )
    print(json.dumps({
        "source_failed_batches": len(batch_ids),
        "recovered_batches": len(batch_ids) - len(retry_errors),
        "remaining_failed_batches": len(retry_errors),
        "predictions": len(predictions),
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
