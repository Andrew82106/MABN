"""Small fail-closed command-line interface for qualification evidence."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .core import DEFAULT_CONTEXT, safe_error
from .replay import replay_run
from .reporting import generate_qualification_report
from .validation import validate_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p1-model-qualification")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "replay"):
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
    summary = commands.add_parser("summary")
    summary.add_argument("--run-id", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_run(args.run_id, context=DEFAULT_CONTEXT, write_outputs=True)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0 if result["passed"] else 1
        if args.command == "replay":
            result = replay_run(args.run_id, context=DEFAULT_CONTEXT, write_output=True)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0 if result["passed"] else 1
        result = generate_qualification_report(args.run_id, context=DEFAULT_CONTEXT)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"passed": False, "error": safe_error(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
