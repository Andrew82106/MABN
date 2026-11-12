"""Summarize a complete or interrupted frozen confirmation queue.

The collector intentionally writes each episode before moving to the next one.
This utility therefore produces a truthful report even when an API provider
becomes unavailable. It never imputes missing episodes or treats a failed call
as a negative prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    plan_path = args.input / "PLAN.json"
    if not plan_path.exists():
        raise SystemExit("PLAN.json is required")
    plan = read_json(plan_path)
    episodes = []
    for episode_dir in sorted((args.input / "episodes").glob("*-*")):
        if not episode_dir.is_dir():
            continue
        monitor_path, evaluator_path = episode_dir / "monitor.json", episode_dir / "evaluator.json"
        # A partially running episode has request/response logs but no monitor.
        monitor = read_json(monitor_path) if monitor_path.exists() else None
        evaluator = read_json(evaluator_path) if evaluator_path.exists() else None
        response_path = episode_dir / "api_responses.jsonl"
        requests = completed = failed = input_tokens = output_tokens = 0
        latencies = []
        if response_path.exists():
            for line in response_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                requests += 1
                if item.get("status") == "completed":
                    completed += 1
                    usage = item.get("usage") or {}
                    input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                    output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                    if item.get("meta", {}).get("elapsed_seconds") is not None:
                        latencies.append(float(item["meta"]["elapsed_seconds"]))
                elif item.get("status") == "failed":
                    failed += 1
        episode_id = episode_dir.name
        topology = episode_id.rsplit("-", 1)[0]
        order = next((x for x in plan.get("scenario_order", []) if x.get("episode_id") == episode_id), {})
        episodes.append({"episode_id": episode_id, "topology": topology, "variant": order.get("variant"),
                         "expected_in_plan": bool(order), "monitor_written": monitor is not None,
                         "evaluator_written": evaluator is not None, "requests": requests,
                         "completed": completed, "failed": failed, "input_tokens": input_tokens,
                         "output_tokens": output_tokens, "latencies": latencies,
                         "policy_intent_accuracy": None if not evaluator else evaluator.get("correct")})
    latencies = [v for e in episodes for v in e["latencies"]]
    correct = [e["policy_intent_accuracy"] for e in episodes if e["policy_intent_accuracy"] is not None]
    planned = int(plan.get("total_episodes", 0))
    report = {"schema": "paperalpha-frozen-confirmation-summary-v1", "plan": plan,
              "observed_episodes": len(episodes), "planned_episodes": planned,
              "missing_episodes": max(0, planned - len(episodes)), "episodes": episodes,
              "total_api_requests": sum(e["requests"] for e in episodes),
              "completed_api_requests": sum(e["completed"] for e in episodes),
              "failed_api_requests": sum(e["failed"] for e in episodes),
              "input_tokens": sum(e["input_tokens"] for e in episodes),
              "output_tokens": sum(e["output_tokens"] for e in episodes),
              "policy_intent_accuracy": sum(correct) / len(correct) if correct else None,
              "latency_seconds": {"p50": median(latencies) if latencies else None,
                                  "p95": sorted(latencies)[max(0, int(len(latencies) * .95) - 1)] if latencies else None},
              "complete": len(episodes) == planned and all(e["monitor_written"] and e["evaluator_written"] for e in episodes),
              "scope": "summary of frozen integration/policy-intent collection; no human security gold"}
    out_json = args.input / "SUMMARY.json"
    out_md = args.input / "SUMMARY.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text("# Frozen confirmation queue summary\n\n" +
        f"- Planned/observed episodes: {planned}/{len(episodes)}\n" +
        f"- Missing episodes: {report['missing_episodes']}\n" +
        f"- API requests completed/failed: {report['completed_api_requests']}/{report['failed_api_requests']}\n" +
        f"- Tokens input/output: {report['input_tokens']}/{report['output_tokens']}\n" +
        f"- Policy-intent evaluator accuracy: {report['policy_intent_accuracy']}\n\n" +
        ("Collection is complete.\n" if report["complete"] else "Collection is incomplete; missing episodes are not imputed.\n") +
        "Labels are hand-authored evaluator targets, not independent human security gold.\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("planned_episodes", "observed_episodes", "missing_episodes", "completed_api_requests", "failed_api_requests", "complete")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
