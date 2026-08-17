"""Public offline-only Q2-D CLI.  There is deliberately no live command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from q2d_contract.replay import replay_run
from q2d_contract.runner import run_fake
from q2d_contract.validation import validate_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="q2d-offline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fake", "validate", "replay"):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "fake":
        result = run_fake(args.run_id)
        code = 0 if result.get("passed") else 1
    elif args.command == "validate":
        result = validate_run(args.run_id, write_outputs=True)
        code = 0 if result.get("passed") else 1
    else:
        result = replay_run(args.run_id, write_output=True)
        code = 0 if result.get("passed") else 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

