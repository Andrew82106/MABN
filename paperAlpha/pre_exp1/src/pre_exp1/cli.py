"""Command-line entry points for P0 smoke, validation, and replay."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .replay import replay_run
from .sandbox import run_p0_smoke
from .validation import validate_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pre-exp1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("run-smoke", help="run four P0 trajectories")
    smoke.add_argument("--run-id")
    validate = subparsers.add_parser("validate-run", help="validate a run")
    validate.add_argument("--run-id", required=True)
    replay = subparsers.add_parser("replay-run", help="replay a run transcript")
    replay.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run-smoke":
        result = run_p0_smoke(args.run_id)
    elif args.command == "validate-run":
        result = validate_run(args.run_id)
    else:
        result = replay_run(args.run_id, write_output=True)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

