"""Safe public command-line interface for offline P1v2-A artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import replay
from .core import DEFAULT_DATA_ROOT, DEFAULT_RUN_ID, canonical_json_bytes
from .inventory import capture_baseline, compare_baselines
from .runner import create_dry_run
from .validation import validate_run


def _emit(value: object) -> None:
    print(canonical_json_bytes(value).decode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p1v2-readiness")
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry")
    dry.add_argument("--run-id", default=DEFAULT_RUN_ID)
    for name in ("validate", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    baseline = commands.add_parser("capture-baseline")
    baseline.add_argument("--phase", choices=("before", "after"), required=True)
    baseline.add_argument("--label", default="rework1")
    compare = commands.add_parser("compare-baselines")
    compare.add_argument("--label", default="rework1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "dry":
            result = create_dry_run(run_id=args.run_id)
            _emit(result)
            return 0
        if args.command == "validate":
            result = validate_run(data_root=args.data_root, run_id=args.run_id)
            _emit(result)
            return 0 if result["passed"] is True else 1
        if args.command == "replay":
            result = replay.replay_run(data_root=args.data_root, run_id=args.run_id)
            _emit(result)
            return 0 if result["passed"] is True else 1
        if args.command == "capture-baseline":
            _emit(capture_baseline(phase=args.phase, label=args.label))
            return 0
        if args.command == "compare-baselines":
            result = compare_baselines(label=args.label)
            _emit(result)
            return 0 if result["passed"] is True else 1
    except Exception:
        _emit({"error": {"code": "command_failed"}, "passed": False})
        return 2
    _emit({"error": {"code": "unsupported_command"}, "passed": False})
    return 2
