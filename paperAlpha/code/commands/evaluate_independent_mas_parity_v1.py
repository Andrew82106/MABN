"""Run the frozen journal evaluator with one public-field parity ablation.

The main runtime vector contains a `permission_mismatch` count while the
learned graph controls do not.  This wrapper reruns the same strict protocol
with that component set to zero, without changing the frozen main result.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load():
    path = Path(__file__).with_name("evaluate_independent_mas_journal_v1.py")
    spec = importlib.util.spec_from_file_location("journal_parity_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load()
    original = module.runtime_vector

    def parity_runtime_vector(ep):
        row = original(ep).copy()
        # runtime_vector layout: observed_edges, ..., unexpected, mismatch,
        # path, inverse_distance.  Remove only permission_mismatch.
        row[5] = 0.0
        return row

    module.runtime_vector = parity_runtime_vector
    sys.argv = [sys.argv[0], "--out", 'paperAlpha/experiments/synthetic_mas/runs/independent_mas_journal_parity_v1']
    module.main()


if __name__ == "__main__":
    main()
