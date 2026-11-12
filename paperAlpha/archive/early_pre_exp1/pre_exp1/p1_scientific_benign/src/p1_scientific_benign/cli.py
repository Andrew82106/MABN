"""Fail-closed CLI for the scientific P1 benign gate pilot."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .audit import write_repair_audit
from .core import DEFAULT_CONTEXT, EXECUTION_DRY, EXECUTION_REAL, route_context, safe_error
from .replay import replay_run
from .runner import run_batch
from .validation import validate_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p1-scientific-benign")
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry")
    dry.add_argument("--run-id")
    real = commands.add_parser("pilot")
    real.add_argument("--run-id")
    real.add_argument("--allow-local-pilot", action="store_true")
    for name in ("validate", "replay"):
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.add_argument("--repair-id", help="write a versioned no-provider audit under acceptance_repair/<REPAIR_ID>")
    args = parser.parse_args(argv)
    try:
        if args.command == "dry":
            result = run_batch(context=DEFAULT_CONTEXT, execution_mode=EXECUTION_DRY, run_id=args.run_id)
        elif args.command == "pilot":
            result = run_batch(context=DEFAULT_CONTEXT, execution_mode=EXECUTION_REAL, run_id=args.run_id, allow_local_pilot=args.allow_local_pilot)
        elif args.command == "validate":
            routed = route_context(args.run_id, context=DEFAULT_CONTEXT)
            result = validate_run(args.run_id, context=routed, write_outputs=False)
            if args.repair_id:
                replay = replay_run(args.run_id, context=routed, write_output=False)
                result = {"validation": result, "replay": replay, "repair_audit": write_repair_audit(args.run_id, context=routed, repair_id=args.repair_id, validation=result, replay=replay), "passed": result.get("passed") is True}
        else:
            routed = route_context(args.run_id, context=DEFAULT_CONTEXT)
            result = replay_run(args.run_id, context=routed, write_output=False)
            if args.repair_id:
                validation = validate_run(args.run_id, context=routed, write_outputs=False)
                result = {"replay": result, "validation": validation, "repair_audit": write_repair_audit(args.run_id, context=routed, repair_id=args.repair_id, validation=validation, replay=result), "passed": result.get("passed") is True}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if args.command in {"dry", "pilot"}:
            return 0 if result.get("validation", {}).get("passed") and result.get("replay", {}).get("passed") else 1
        return 0 if result.get("passed") is True else 1
    except Exception as exc:
        print(json.dumps({"passed": False, "error": safe_error(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
