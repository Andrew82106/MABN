"""Small offline CLI for scenario inspection and receipt verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .provenance import load_receipt, verify_receipt
from .scenario import load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(prog="shared-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-scenario")
    inspect_parser.add_argument("scenario", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("receipt", type=Path)
    verify_parser.add_argument("scenario", type=Path)
    verify_parser.add_argument("ledger", type=Path)
    verify_parser.add_argument("--code-root", action="append", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect-scenario":
        scenario = load_scenario(args.scenario)
        result = {
            "scenario_id": scenario.config["scenario_id"],
            "agents": len(scenario.config["agents"]),
            "tree_hash": scenario.tree_sha256,
        }
    else:
        receipt = load_receipt(args.receipt)
        result = verify_receipt(
            receipt, scenario_root=args.scenario, code_roots=args.code_root,
            ledger_path=args.ledger,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
