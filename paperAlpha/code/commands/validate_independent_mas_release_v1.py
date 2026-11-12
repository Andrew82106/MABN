"""Validate trace/label identity and public-field boundaries for the frozen MAS release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FORBIDDEN = {"label", "label_source", "mechanism", "scenario_family", "external_effect", "hidden_fields"}


def keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in keys(child)}
    return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl"))
    ap.add_argument("--labels", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl"))
    ap.add_argument("--out", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/submission_audit/runs/final_eval/release_validator_v1.json"))
    args = ap.parse_args()
    traces = [json.loads(line) for line in args.traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    trace_ids = [r.get("episode_id") for r in traces]
    label_ids = [r.get("episode_id") for r in labels]
    forbidden = sorted({key for row in traces for key in keys(row).intersection(FORBIDDEN)})
    report = {"schema": "independent-mas-release-validator-v1", "trace_count": len(traces), "label_count": len(labels), "trace_duplicates": sorted({x for x in trace_ids if trace_ids.count(x) > 1}), "label_duplicates": sorted({x for x in label_ids if label_ids.count(x) > 1}), "labels_without_trace": sorted(set(label_ids) - set(trace_ids)), "traces_without_label": sorted(set(trace_ids) - set(label_ids)), "forbidden_public_keys": forbidden, "pass": len(traces) == len(set(trace_ids)) == len(labels) == len(set(label_ids)) and set(trace_ids) == set(label_ids) and not forbidden}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
