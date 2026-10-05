"""Join API, semantic-extraction and monitor cost on shared episode IDs.

This audit consumes frozen, label-free monitor/API records and the separately
recorded semantic-extraction attempts.  It does not score risk or read the
evaluator file.  Only episodes present in both frozen ledgers are reported;
missing episodes are not silently imputed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(len(values) - 1, int(pos) + 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def usage(item: dict) -> tuple[int, int]:
    u = item.get("usage") or {}
    return int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0), int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0)


def semantic_records(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by.setdefault(str(row["episode"]), []).append(row)
    return by


def monitor_episode(row: dict) -> dict:
    """Build a label-free event projection from a frozen monitor record."""
    nodes = list(row.get("nodes") or [])
    events = []
    if nodes:
        events.append({"actor": nodes[0], "event_type": "input_received", "data_class": "unknown", "content": "workflow input"})
    for record in row.get("records") or []:
        if record.get("status") != "completed":
            continue
        response = (record.get("response") or {}).get("text", "")
        parents = record.get("parent_ids") or []
        target = record.get("node")
        if parents:
            source = str(parents[0]).rsplit(":", 1)[-1]
            events.append({"actor": source, "target": target, "event_type": "message_sent", "data_class": "unknown", "edge_kind": "workflow", "edge_confidence": 0.0, "content": response})
        else:
            events.append({"actor": target, "event_type": "observation", "data_class": "unknown", "content": response})
    return {"agents": nodes, "events": events, "workflow_dag": {"nodes": nodes, "edges": row.get("edges") or []}, "task_text": " ".join((r.get("request") or {}).get("task", "") for r in row.get("records") or [])}


def monitor_latency(monitor_root: Path, episodes: list[str], monitor_module) -> dict:
    point, full = [], []
    for episode in episodes:
        path = monitor_root / "episodes" / episode / "monitor.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        ep = monitor_episode(raw)
        parsed = monitor_module.event_features(ep)
        bn = monitor_module.fit_bn([ep], __import__("numpy").zeros(1))
        theta = __import__("numpy").zeros(13)
        for _ in range(10):
            monitor_module.semantic_policy_evidence(ep)
            monitor_module._propagate(theta, parsed, bn)
        for _ in range(50):
            started = time.perf_counter_ns()
            semantic = monitor_module.semantic_policy_evidence(ep)
            parsed = monitor_module.event_features(ep)
            parsed["semantic"] = semantic
            monitor_module._propagate(theta, parsed, bn)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            full.append(elapsed)
        started = time.perf_counter_ns(); monitor_module.event_features(ep); point.append((time.perf_counter_ns() - started) / 1e6)
    return {"episode_n": len(episodes), "point_ms": {"p50": percentile(point, .5), "p95": percentile(point, .95)}, "semantic_plus_graph_bn_ms": {"p50": percentile(full, .5), "p95": percentile(full, .95), "n": len(full)}, "scope": "local semantic pattern matching + graph feature extraction + BN propagation; excludes network and training"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-report", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--monitor-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    api = json.loads(args.api_report.read_text(encoding="utf-8"))
    sem = semantic_records(args.semantic)
    api_by = {str(row["episode_id"]): row for row in api.get("episodes", [])}
    shared = sorted(set(api_by) & set(sem))
    episodes = []
    for episode in shared:
        a = api_by[episode]
        api_latency = [float(x) for x in (a.get("stats") or {}).get("latencies", [])]
        api_in = int((a.get("stats") or {}).get("input_tokens", 0)); api_out = int((a.get("stats") or {}).get("output_tokens", 0))
        sem_elapsed, sem_in, sem_out, attempts = [], 0, 0, 0
        for record in sem[episode]:
            for attempt in record.get("attempts", []):
                attempts += 1
                if isinstance(attempt.get("elapsed_seconds"), (int, float)):
                    sem_elapsed.append(float(attempt["elapsed_seconds"]))
                i, o = usage(attempt); sem_in += i; sem_out += o
        episodes.append({"episode_id": episode, "api": {"requests": int((a.get("stats") or {}).get("requests", 0)), "completed": int((a.get("stats") or {}).get("completed", 0)), "failed": int((a.get("stats") or {}).get("failed", 0)), "latency_seconds": sum(api_latency), "input_tokens": api_in, "output_tokens": api_out}, "semantic": {"records": len(sem[episode]), "attempts": attempts, "latency_seconds": sum(sem_elapsed), "input_tokens": sem_in, "output_tokens": sem_out}})
    monitor = load_module(ROOT / "scripts/evaluate_two_layer_v2.py", "e2e_cost_monitor")
    monitor_cost = monitor_latency(args.monitor_root, shared, monitor)
    for row in episodes:
        row["measured_serial_service_seconds"] = row["api"]["latency_seconds"] + row["semantic"]["latency_seconds"]
    serial = [r["measured_serial_service_seconds"] for r in episodes]
    totals = {"api_input_tokens": sum(r["api"]["input_tokens"] for r in episodes), "api_output_tokens": sum(r["api"]["output_tokens"] for r in episodes), "semantic_input_tokens": sum(r["semantic"]["input_tokens"] for r in episodes), "semantic_output_tokens": sum(r["semantic"]["output_tokens"] for r in episodes)}
    report = {"schema": "paperalpha-e2e-episode-cost-v1", "status": "shared_episode_cost_measured_with_partial_semantic_coverage", "api_report": str(args.api_report), "semantic_report": str(args.semantic), "shared_episode_count": len(shared), "api_report_episode_count": len(api_by), "semantic_report_episode_count": len(sem), "episodes": episodes, "serial_service_latency_seconds": {"p50": percentile(serial, .5), "p95": percentile(serial, .95), "mean": statistics.mean(serial) if serial else None}, "monitor": monitor_cost, "token_totals": totals, "boundaries": ["Only shared episode IDs are joined; the remaining API episodes are not imputed.", "The serial value sums recorded API and semantic request latencies; it is a service-time estimate, not a claim that both requests were on one wall-clock critical path.", "The monitor timing is measured locally on the same shared monitor projections and excludes network, semantic API calls and training.", "No labels, evaluator files or risk metrics are read."]}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# Same-episode API-to-monitor cost audit", "", f"- Shared episodes: **{len(shared)} / {len(api_by)} API episodes", f"- Serial service-time estimate p50/p95: **{report['serial_service_latency_seconds']['p50']:.3f} / {report['serial_service_latency_seconds']['p95']:.3f} s**", f"- Monitor local semantic+graph+BN p50/p95: **{monitor_cost['semantic_plus_graph_bn_ms']['p50']:.3f} / {monitor_cost['semantic_plus_graph_bn_ms']['p95']:.3f} ms**", f"- Token totals: API in/out {totals['api_input_tokens']}/{totals['api_output_tokens']}; semantic in/out {totals['semantic_input_tokens']}/{totals['semantic_output_tokens']}", "", "The report joins only episode IDs present in both frozen ledgers. It is a cost measurement, not an accuracy result; failed requests and semantic attempts remain visible."]
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"shared_episode_count": len(shared), "serial_service_latency_seconds": report["serial_service_latency_seconds"], "monitor": monitor_cost}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
