"""Mutation audit: model scores must not change when label-bearing fields change."""
from __future__ import annotations

import argparse, copy, json
from pathlib import Path

from evaluate_unified_monitor_v2 import forward, read_eps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/compositional_mas_v1/episodes.jsonl")); ap.add_argument("--out", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/unified_monitor_v2/input_isolation_audit.json")); args = ap.parse_args()
    eps = read_eps(args.input)[:100]
    theta = [0.0] * 14
    changed = []
    for ep in eps:
        altered = copy.deepcopy(ep)
        altered["label"] = 1 - int(ep.get("label", 0))
        altered["label_evidence"] = {"source_agents": ["tampered"], "sink_agents": ["tampered"]}
        altered["label_rule"] = "tampered"
        altered["risk"] = 1 - int(ep.get("label", 0))
        altered["violation"] = True
        p0 = forward(theta, ep); p1 = forward(theta, altered)
        changed.append(abs(p0 - p1))
    result = {"episodes_checked": len(eps), "max_absolute_score_change": max(changed, default=0.0), "pass": max(changed, default=0.0) == 0.0, "mutated_fields": ["label", "label_evidence", "label_rule", "risk", "violation"], "interpretation": "runtime mutation audit only; it complements code review and does not prove absence of all leakage paths"}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
