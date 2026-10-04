"""Low-cost LANYUN DAG extraction smoke test; no credential is persisted."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_model_test_v1 import (  # noqa: E402
    CANONICAL,
    WORKFLOWS,
    api_chat,
    dag_quality,
    normalize_dag,
    parse_dag,
    strategy_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://maas-api.lanyun.net/v1")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--out", type=Path, default=ROOT / "results/submission/final_eval/lanyun_dag_smoke_v1")
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    key = os.environ.get("LANYUN_API_KEY", "")
    if not key:
        raise SystemExit("LANYUN_API_KEY is required in the process environment")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in ("direct", "decompose", "edge_audit"):
        for workflow, description in WORKFLOWS:
            text, error, meta = api_chat(
                args.endpoint, key, args.model,
                strategy_prompt(strategy, workflow, description),
                timeout=args.timeout,
            )
            parsed = parse_dag(text)
            dag, fallback = normalize_dag(parsed)
            quality = dag_quality(dag, CANONICAL)
            rows.append({
                "strategy": strategy,
                "workflow": workflow,
                "api_returned": error is None and parsed is not None,
                "fallback": fallback,
                "http_status": meta.get("status"),
                "attempts": meta.get("attempts"),
                "error_type": None if error is None else error.split(":", 1)[0],
                "quality": quality,
            })
    successful = [r for r in rows if r["api_returned"]]
    def mean(key: str) -> float | None:
        values = [float(r["quality"].get(key, 0.0)) for r in successful]
        return round(statistics.mean(values), 4) if values else None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "model": args.model,
        "requests": len(rows),
        "successful_parsed": len(successful),
        "success_rate": round(len(successful) / len(rows), 4),
        "mean_exact_node_f1": mean("node_f1"),
        "mean_exact_edge_f1": mean("edge_f1"),
        "mean_semantic_node_f1": mean("semantic_node_f1"),
        "mean_semantic_edge_f1": mean("semantic_edge_f1"),
        "acyclic_rate": round(sum(bool(r["quality"].get("acyclic", 0.0)) for r in successful) / len(successful), 4) if successful else None,
        "scope": "transport/DAG extraction smoke test; not a risk-monitoring result and not a journal comparison",
        "rows": rows,
    }
    (args.out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# LANYUN DAG smoke test",
        "",
        f"- Model: `{args.model}`",
        f"- Requests: {len(rows)}; parsed successfully: {len(successful)} ({report['success_rate']:.1%})",
        f"- Exact node/edge F1: {report['mean_exact_node_f1']} / {report['mean_exact_edge_f1']}",
        f"- Semantic node/edge F1: {report['mean_semantic_node_f1']} / {report['mean_semantic_edge_f1']}",
        f"- Acyclic rate among parsed DAGs: {report['acyclic_rate']}",
        "",
        "This is a low-cost API transport and DAG-format smoke test. It is not used as the independent MAS risk-monitoring cohort or as a superiority claim.",
    ]
    (args.out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("requests", "successful_parsed", "success_rate", "mean_exact_node_f1", "mean_exact_edge_f1", "mean_semantic_node_f1", "mean_semantic_edge_f1", "acyclic_rate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
