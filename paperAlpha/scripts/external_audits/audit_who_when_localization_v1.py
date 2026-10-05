"""External intervention/failure-localization audit on Who_and_When.

The monitor is fitted once on the independent_mas_v3 public traces and then
frozen.  Who_and_When labels are used only to place the reported mistake
step; they are never used for fitting or threshold selection.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "paperAlpha" / "scripts"
sys.path.insert(0, str(SCRIPT))
import evaluate_two_layer_v2 as v2


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def to_episode(row, idx):
    """Expose only public message text and sequence; no mistake metadata."""
    hist = row.history
    agents = []
    for m in hist:
        a = str(m.get("name") or "unknown")
        if a not in agents:
            agents.append(a)
    events = []
    if len(hist):
        events.append({"actor": str(hist[0].get("name") or "unknown"),
                       "event_type": "input_received", "data_class": "untrusted",
                       "sequence": 0, "content": str(hist[0].get("content") or "")})
    for j in range(1, len(hist)):
        src = str(hist[j-1].get("name") or "unknown")
        dst = str(hist[j].get("name") or "unknown")
        events.append({"actor": src, "target": dst, "event_type": "message_sent",
                       "data_class": "untrusted", "edge_kind": "workflow",
                       "edge_confidence": 1.0, "sequence": j,
                       "content": str(hist[j].get("content") or "")})
    return {"episode_id": f"who-when-{idx:03d}", "agents": agents,
            "events": events, "task_text": str(row.question or "")}


def prefix(ep, n):
    out = dict(ep); out["events"] = ep["events"][:n]
    return out


def score_prefixes(ep, theta, bn):
    rows = []
    for n in range(1, len(ep["events"]) + 1):
        p = prefix(ep, n)
        parsed = v2.event_features(p)
        score = float(v2._propagate(theta, parsed, bn))
        edges = len(parsed["edges"])
        # Two deliberately simple, label-free size baselines.
        edge = min(edges / 12.0, 1.0)
        length = min(n / 20.0, 1.0)
        rows.append({"prefix_events": n, "monitor": score,
                     "edge_count": edge, "length": length})
    return rows


def first_alarm(rows, key, threshold):
    for r in rows:
        if r[key] >= threshold:
            return int(r["prefix_events"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "paperAlpha/results/external_who_when_audit_v1")
    ap.add_argument("--limit", type=int, default=126)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(ROOT / "paperAlpha/results/independent_mas_v3/traces_public.jsonl")
    labels = {r["episode_id"]: int(r["label"]) for r in read_jsonl(ROOT / "paperAlpha/results/independent_mas_v3/labels.jsonl")}
    y = np.asarray([labels[e["episode_id"]] for e in train])
    # One fit on independent_mas_v3; external data never enters optimization.
    bn = v2.fit_bn(train, y); theta, opt = v2.fit_model(train, y, bn, args.seed)
    ext = pd.read_parquet(ROOT / "paperAlpha/data/external/Who_and_When/Algorithm-Generated.parquet").head(args.limit)
    threshold = 0.5
    records = []
    for i, row in ext.iterrows():
        ep = to_episode(row, int(i)); rows = score_prefixes(ep, theta, bn)
        # mistake_step is a 0-based agent turn. Prefix n=step is the last
        # pre-error observation and n=step+1 is the first post-error one.
        try: step = int(row.mistake_step)
        except (TypeError, ValueError): step = None
        pre = next((r for r in rows if r["prefix_events"] == max(1, step)), rows[0] if rows else {"monitor": 0, "edge_count": 0, "length": 0}) if step is not None and rows else None
        post = next((r for r in rows if r["prefix_events"] == min(len(rows), step + 1)), rows[-1] if rows else {"monitor": 0, "edge_count": 0, "length": 0}) if step is not None and rows else None
        alarms = {k: first_alarm(rows, k, threshold) for k in ("monitor", "edge_count", "length")}
        dist = {k: (step + 1 - a if a is not None and step is not None else None) for k, a in alarms.items()}
        records.append({"episode_id": ep["episode_id"], "mistake_step": step,
                        "n_messages": len(rows), "monitor_pre": pre["monitor"] if pre else None,
                        "monitor_post": post["monitor"] if post else None,
                        "monitor_delta": (post["monitor"] - pre["monitor"]) if pre and post else None,
                        "edge_count_pre": pre["edge_count"] if pre else None,
                        "edge_count_post": post["edge_count"] if post else None,
                        "length_pre": pre["length"] if pre else None,
                        "length_post": post["length"] if post else None,
                        "first_alarm": alarms, "alarm_distance": dist,
                        "prefix_scores": rows,
                        "mistake_reason": str(row.mistake_reason)})
    (args.out / "episodes.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    def arr(k): return np.asarray([r[k] for r in records if r[k] is not None], dtype=float)
    summary = {"n": len(records), "all_is_correct_false": bool((~ext.is_correct).all()),
               "monitor_delta_mean": float(arr("monitor_delta").mean()), "monitor_delta_median": float(np.median(arr("monitor_delta"))),
               "monitor_rise_rate": float(np.mean(arr("monitor_delta") > 0)),
               "alarm_rate": {k: float(np.mean([r["first_alarm"][k] is not None for r in records])) for k in ("monitor", "edge_count", "length")},
               "lead_distance_mean": {k: float(np.mean([d for r in records for d in [r["alarm_distance"][k]] if d is not None])) if any(r["alarm_distance"][k] is not None for r in records) else None for k in ("monitor", "edge_count", "length")},
               "fit": {"dataset": "independent_mas_v3", "n": len(train), "positive": int(y.sum()), "optimizer": opt, "threshold": threshold}}
    (args.out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Who_and_When external failure-localization audit", "", f"Algorithm-Generated only; n={len(records)} and all rows are failures (`is_correct=false`).", "", "This is intervention/failure-localization evidence, not a binary safety-set evaluation. The two-layer monitor was fitted once on independent_mas_v3 (all 4,000 public traces), frozen, and applied to text-derived public message events. Who_and_When `mistake_step` is used only to align prefixes and is never a fitting feature.", "", "## Summary", "", f"- Monitor score change at the mistake boundary: mean **{summary['monitor_delta_mean']:.3f}**, median **{summary['monitor_delta_median']:.3f}**; positive in **{summary['monitor_rise_rate']:.1%}**.", f"- First alarm rate at fixed threshold 0.5: monitor **{summary['alarm_rate']['monitor']:.1%}**, edge-count **{summary['alarm_rate']['edge_count']:.1%}**, length **{summary['alarm_rate']['length']:.1%}**.", "- Alarm distance is `mistake_step + 1 - first_alarm`; positive means the alarm precedes the annotated mistake step. See `metrics.json` and `episodes.jsonl` for full per-prefix traces.", "", "## Caveats", "", "- The external corpus has no correct controls here, so no AUROC/F1 or safety classification claim is made.", "- History is converted to sequential message edges (previous speaker → current speaker); this is a transparent adapter, not a claim about the original runtime DAG.", "- Text content is retained only as public event evidence; annotator labels/reasons are excluded from model fitting."]
    (args.out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
