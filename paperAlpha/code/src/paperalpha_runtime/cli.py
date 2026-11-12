"""Single reproducible entry point for the active paperAlpha evaluations."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'code/commands'


def _run(script: str, args: list[str]) -> None:
    target = SCRIPTS / script
    if not target.exists():
        raise SystemExit(f"missing active evaluator: {target}")
    old = sys.argv
    try:
        sys.argv = [str(target), *args]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        print("usage: paperalpha-evaluate {independent|binding|costs} [evaluator options]")
        return
    command, forwarded = values[0], values[1:]
    scripts = {"independent": "evaluate_independent_mas_journal_v1.py",
               "binding": "evaluate_binding_consistency_extension_v1.py",
               "costs": "summarize_api_costs_v1.py"}
    if command not in scripts:
        raise SystemExit(f"unknown command {command!r}; choose independent, binding or costs")
    _run(scripts[command], forwarded)


if __name__ == "__main__":
    main()
