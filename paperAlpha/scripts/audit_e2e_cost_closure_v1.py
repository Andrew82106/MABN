"""Join existing cost ledgers into one auditable API-to-monitor budget.

No new API call is made and no label/evaluator field is read.  The report keeps
transport, semantic extraction and monitor inference as separate components;
it intentionally does not invent a single end-to-end latency for unmatched
cohorts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--monitor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    api, semantic, monitor = load(args.api), load(args.semantic), load(args.monitor)
    api_totals = api["totals"]
    api_runs = max(int(api_totals.get("runs", 0)), 1)
    semantic_counts = semantic["counts"]
    monitor_scope = monitor.get("scope", "")
    result = {
        "schema": "paperalpha-e2e-cost-closure-v1",
        "status": "componentwise_closed_not_single_cohort_end_to_end",
        "components": {
            "api_transport": {
                "source": str(args.api),
                "runs": api_totals.get("runs"), "completed_runs": api_totals.get("completed_runs"),
                "failed_runs": api_totals.get("failed_runs"),
                "requests_started": api_totals.get("requests_started"),
                "requests_finished": api_totals.get("requests_finished"),
                "requests_failed": api_totals.get("requests_failed"),
                "input_tokens": api_totals.get("input_tokens"), "output_tokens": api_totals.get("output_tokens"),
                "total_tokens": api_totals.get("total_tokens"),
                "tokens_per_planned_run": float(api_totals.get("total_tokens", 0)) / api_runs,
                "latency_seconds": api.get("request_latency_seconds"),
            },
            "semantic_extraction": {
                "source": str(args.semantic),
                "source_records": semantic_counts.get("source_records"),
                "requests_started": semantic_counts.get("requests_started"),
                "api_completed": semantic_counts.get("api_completed"),
                "api_failed": semantic_counts.get("api_failed"),
                "parsed_final": semantic_counts.get("parsed_final"),
                "final_abstain": semantic_counts.get("final_abstain"),
                "final_abstain_rate": semantic_counts.get("final_abstain_rate"),
                "latency_seconds": semantic.get("latency_seconds"),
                "tokens": semantic.get("tokens"),
            },
            "monitor_inference": {
                "source": str(args.monitor),
                "latency": monitor.get("cpu_inference_ms"),
                "peak_tracemalloc_mb": monitor.get("peak_tracemalloc_mb"),
                "scope": monitor_scope,
            },
        },
        "boundary": [
            "The three ledgers use different cohorts and are not added into one claimed episode latency.",
            "API transport includes failed requests; semantic failures become explicit abstention.",
            "Monitor latency excludes API generation, network transfer, semantic parsing and training.",
            "No semantic accuracy or risk metric is inferred from this cost closure.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
