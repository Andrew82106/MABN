"""CPU inference latency/memory sanity for the inspectable runtime BN."""
from __future__ import annotations

import importlib.util
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("two", root / 'code/commands/evaluate_two_layer_v2.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    eps = read_jsonl(root / 'experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl')[:1000]
    y = np.asarray([0 if i % 2 else 1 for i in range(len(eps))], dtype=int)
    bn = mod.fit_bn(eps, y)
    theta = np.array([-2., 3., .2, .8, -.4, .1, .1, -2., 4., .2, 1., 1., .3], dtype=float)
    # Warm-up avoids counting import and one-time allocation overhead.
    for ep in eps[:10]: mod._propagate(theta, mod.event_features(ep), bn)
    times = []; tracemalloc.start(); start_peak = tracemalloc.get_traced_memory()[1]
    for ep in eps:
        t0 = time.perf_counter(); parsed = mod.event_features(ep); mod._propagate(theta, parsed, bn); times.append((time.perf_counter() - t0) * 1000)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out = {"n": len(eps), "cpu_inference_ms": {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)), "mean": float(np.mean(times)), "max": float(np.max(times))}, "peak_tracemalloc_mb": float(max(0, peak - start_peak) / (1024**2)), "scope": "feature extraction plus runtime BN propagation; excludes API generation, network, semantic parsing and training"}
    result = root / 'experiments/synthetic_mas/runs/independent_mas_journal_v1/latency.json'; result.write_text(json.dumps(out, indent=2), encoding="utf-8"); print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
