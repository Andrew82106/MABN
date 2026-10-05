"""Run the local fixture for the unified chain/fork/join/review queue.

The output directory is intentionally under ``results/`` (ignored by the
repository).  ``monitor.json`` contains the public projection; the optional
``evaluator.json`` is a separate evaluator-only artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paperalpha_runtime.unified_topology_queue import run_topology  # noqa: E402


def fixture_handler(request, parents):
    payload = request.payload if isinstance(request.payload, dict) else {}
    if payload.get("fail") is True:
        raise RuntimeError("fixture requested failure")
    return {"node": request.node, "accepted": True,
            "parent_statuses": {key: value["status"] for key, value in parents.items()},
            "payload": payload}


def fixture_evaluator(private):
    failed = [row["request_id"] for row in private["records"] if row["status"] == "failed"]
    return {"schema": "paperalpha-unified-topology-evaluator-v1", "label": int(bool(failed)),
            "failed_request_ids": failed,
            "meaning": "1 means at least one API call failed in this run; evaluator-only."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("chain", "fork", "join", "review"), default="chain")
    parser.add_argument("--nodes", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "unified_topology_queue_local")
    parser.add_argument("--no-evaluator", action="store_true")
    args = parser.parse_args(argv)
    defaults = {"chain": ["a", "b"], "fork": ["root", "left", "right"],
                "join": ["left", "right", "join"], "review": ["work", "review"]}
    nodes = args.nodes or defaults[args.topology]
    payloads = {node: {"node": node} for node in nodes}
    result = run_topology(args.topology, nodes, fixture_handler, payloads,
                          evaluator=None if args.no_evaluator else fixture_evaluator,
                          run_id="local-fixture")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "monitor.json").write_text(json.dumps(result.monitor_projection(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_evaluator:
        (args.output / "evaluator.json").write_text(json.dumps(result.evaluation_view(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.snapshot(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

