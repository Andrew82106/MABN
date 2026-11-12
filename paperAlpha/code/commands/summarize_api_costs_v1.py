"""Summarize recorded API request cost/latency without reading credentials.

The raw Chat Completions pilot records OpenAI-style ``prompt_tokens`` and
``completion_tokens``.  Older ledgers used ``input_tokens``/``output_tokens``;
both names are accepted.  Failed and incomplete runs stay visible in the
report and are never converted into safe observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    index = (len(xs) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (index - lo))


def summarize(root: Path) -> dict:
    runs = [p for p in sorted((root / "runs").iterdir()) if p.is_dir() and (p / "RUN.json").exists()]
    totals = {"runs": len(runs), "completed_runs": 0, "failed_runs": 0,
              "requests_started": 0, "requests_finished": 0, "requests_failed": 0,
              "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    request_latency: list[float] = []
    run_records = []
    for run_dir in runs:
        run = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
        status = str(run.get("status"))
        totals["completed_runs" if status == "completed" else "failed_runs"] += 1
        req_started = req_finished = req_failed = 0
        in_tok = out_tok = 0
        latencies: list[float] = []
        response_file = run_dir / "api_responses.jsonl"
        if response_file.exists():
            for line in response_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "error_type" in row:
                    req_failed += 1
                else:
                    req_finished += 1
                    response = row.get("response") or {}
                    usage = response.get("usage") or {}
                    in_tok += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                    out_tok += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
                if row.get("elapsed_seconds") is not None:
                    latencies.append(float(row["elapsed_seconds"]))
                    request_latency.append(float(row["elapsed_seconds"]))
        request_file = run_dir / "api_requests.jsonl"
        if request_file.exists():
            req_started = sum(bool(line.strip()) for line in request_file.read_text(encoding="utf-8").splitlines())
        totals["requests_started"] += req_started
        totals["requests_finished"] += req_finished
        totals["requests_failed"] += req_failed
        totals["input_tokens"] += in_tok
        totals["output_tokens"] += out_tok
        run_records.append({"run": run_dir.name, "status": status, "requests_started": req_started,
                            "requests_finished": req_finished, "requests_failed": req_failed,
                            "input_tokens": in_tok, "output_tokens": out_tok,
                            "total_tokens": in_tok + out_tok,
                            "latency_seconds": {"p50": _percentile(latencies, .5),
                                                "p95": _percentile(latencies, .95),
                                                "mean": mean(latencies) if latencies else None}})
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return {"schema": "api-cost-latency-summary-v1", "source": str(root), "totals": totals,
            "request_latency_seconds": {"p50": _percentile(request_latency, .5),
                                         "p95": _percentile(request_latency, .95),
                                         "mean": mean(request_latency) if request_latency else None,
                                         "n": len(request_latency)},
            "runs": run_records,
            "scope": "Recorded API transport only; excludes monitor fitting, semantic extraction and network billing. Failed runs are retained."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.source.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
