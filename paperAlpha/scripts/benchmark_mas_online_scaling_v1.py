"""Measure monitor CPU scaling over observable MAS trace width.

The benchmark intentionally excludes API generation and semantic LLM parsing;
it measures the monitor's own JSON-to-event-feature and BN propagation path.
"""
from __future__ import annotations

import importlib.util
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - the bundled environment includes psutil
    psutil = None

ROOT = Path(__file__).resolve().parents[2]


def main():
    import argparse

    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, default=ROOT / "paperAlpha/results/independent_mas_v3/traces_public.jsonl"); ap.add_argument("--out", type=Path, default=ROOT / "paperAlpha/results/submission/final_eval/online_scaling_v1"); args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("two_scaling", ROOT / "paperAlpha/scripts/evaluate_two_layer_v2.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    eps = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    y = np.asarray([i % 2 for i in range(len(eps))], dtype=int); bn = mod.fit_bn(eps, y)
    theta = np.array([-2., 3., .2, .8, -.4, .1, .1, -2., 4., .2, 1., 1., .3], dtype=float)
    bins = [("1-3", lambda n: n <= 3), ("4-6", lambda n: 4 <= n <= 6), ("7-10", lambda n: 7 <= n <= 10), ("11+", lambda n: n >= 11)]
    records = []
    process = psutil.Process() if psutil is not None else None
    for name, predicate in bins:
        subset = [e for e in eps if predicate(len(e.get("events", [])))]
        if not subset: continue
        for e in subset[:20]:
            mod._propagate(theta, mod.event_features(e), bn)
        times = []; tracemalloc.start(); start = tracemalloc.get_traced_memory()[1]
        rss_before = process.memory_info().rss if process is not None else None
        rss_peak = rss_before
        for e in subset:
            t0 = time.perf_counter(); parsed = mod.event_features(e); mod._propagate(theta, parsed, bn); times.append((time.perf_counter() - t0) * 1000)
            if process is not None:
                rss_peak = max(rss_peak, process.memory_info().rss)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rss_after = process.memory_info().rss if process is not None else None
        records.append({"event_bin": name, "n": len(subset), "median_events": float(np.median([len(e.get("events", [])) for e in subset])), "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times)), "peak_tracemalloc_mb": float(max(0, peak - start) / (1024**2)), "process_rss_before_mb": None if rss_before is None else float(rss_before / (1024**2)), "process_rss_after_mb": None if rss_after is None else float(rss_after / (1024**2)), "process_rss_peak_delta_mb": None if rss_before is None else float(max(0, rss_peak - rss_before) / (1024**2))})
    result = {"protocol": {"scope": "JSON-to-event-feature extraction plus runtime BN propagation", "excluded": ["API generation", "network transfer", "semantic LLM extraction", "training"]}, "records": records}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "REPORT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# MAS monitor online scaling audit v1", "", "Monitor CPU path only; API generation/network/semantic extraction are excluded. Process RSS is an observational batch-level measurement, not a per-request allocation guarantee.", "", "| event bin | n | median events | p50 ms | p95 ms | tracemalloc peak MB | RSS before MB | RSS after MB | RSS peak delta MB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records: lines.append(f"| {r['event_bin']} | {r['n']} | {r['median_events']:.1f} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['peak_tracemalloc_mb']:.3f} | {r['process_rss_before_mb'] if r['process_rss_before_mb'] is not None else 'NA'} | {r['process_rss_after_mb'] if r['process_rss_after_mb'] is not None else 'NA'} | {r['process_rss_peak_delta_mb'] if r['process_rss_peak_delta_mb'] is not None else 'NA'} |")
    lines += ["", "These numbers establish monitor-side scaling only; a journal deployment table must add semantic extraction and API transport separately.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
