"""Portable entry point for the A2ASecBench per-case recovery runner."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    implementation = (
        Path(__file__).resolve().parents[2] / 'experiments/a2asecbench/runs/a2asecbench_v1/co_eval/isolated_runner.py'
    )
    runpy.run_path(str(implementation), run_name="__main__")
