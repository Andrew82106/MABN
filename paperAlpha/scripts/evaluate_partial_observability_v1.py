"""Safety-oriented interval monitoring under missing API provenance.

For an observed event graph, compute a lower risk bound using only explicitly
untrusted edges and an upper bound treating ``unknown``/redacted edges as
potentially tainted.  When the bounds disagree the monitor abstains instead
of silently treating missing evidence as clean.  The benchmark labels are
latent structural labels from ``evaluate_generalization_stress_v1``.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import evaluate_generalization_stress_v1 as base


def reach(ep, allow_unknown=False):
    names = ep["agents"]
    src = {e["actor"] for e in ep["events"] if e["event_type"] == "input_received" and e.get("data_class") == "untrusted"}
    sink = {e["actor"] for e in ep["events"] if e["event_type"] == "tool_call" and e.get("tool") in base.PRIV_TOOLS}
    adj = {a: set() for a in names}
    accepted = {"untrusted"} | ({"unknown"} if allow_unknown else set())
    for e in ep["events"]:
        if e["event_type"] in {"message_sent", "delegation"} and e.get("target") in adj and e.get("data_class") in accepted:
            if e["actor"] != e["target"]: adj[e["actor"]].add(e["target"])
    seen, q = set(src), list(src)
    while q:
        for v in adj[q.pop()]:
            if v not in seen: seen.add(v); q.append(v)
    return float(bool(seen & sink))


def assess(eps):
    y=np.array([e["label"] for e in eps],int); lo=np.array([reach(e) for e in eps]); hi=np.array([reach(e,True) for e in eps]); certain=lo==hi
    # Selective metrics only on non-abstained predictions; all intervals that
    # are certain must agree with the latent label for an ideal monitor.
    sel_y=y[certain]; sel_p=lo[certain]
    sel_f1 = float(f1_score(sel_y,sel_p,zero_division=0)) if len(sel_y) and len(np.unique(sel_y)) > 1 else None
    return {"n":len(eps),"positive":int(y.sum()),"coverage":float(np.mean(certain)),"abstention_rate":float(1-np.mean(certain)),"lower_recall":float(recall_score(y,lo,zero_division=0)),"lower_precision":float(precision_score(y,lo,zero_division=0)),"selective_f1":sel_f1,"certain_error_rate":float(np.mean(sel_p!=sel_y)) if len(sel_y) else None,"positive_lower_bound_rate":float(np.mean(lo[y==1])) if np.any(y==1) else None,"positive_upper_bound_rate":float(np.mean(hi[y==1])) if np.any(y==1) else None}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("paperAlpha/results/partial_observability_v1")); ap.add_argument("--seed",type=int,default=20260914); ap.add_argument("--n",type=int,default=400); args=ap.parse_args()
    cfg={"n_agents":6,"path_len":4,"decoy_multiplier":1}
    rows=[]
    for j,p in enumerate([1.,.8,.6,.4,.2]):
        c={**cfg,"reliability":p}; eps=base.make_set(args.n,c,args.seed+10+j); rows.append({"reliability":p,"metrics":assess(eps)})
    args.out.mkdir(parents=True,exist_ok=True); (args.out/"metrics.json").write_text(json.dumps({"protocol":"lower bound=observed untrusted paths; upper bound=untrusted or unknown paths; abstain when bounds disagree","scenarios":rows},indent=2),encoding="utf-8")
    md=["# Partial-observability interval monitor v1","","The monitor emits lower/upper path-risk bounds and abstains when an unknown edge can change the decision. This avoids calling missing API provenance clean.","","| reliability | coverage | abstention | lower recall | selective F1 | certain error |","|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m=r["metrics"]; sf1="n/a" if m["selective_f1"] is None else f"{m['selective_f1']:.3f}"; md.append(f"| {r['reliability']:.1f} | {m['coverage']:.3f} | {m['abstention_rate']:.3f} | {m['lower_recall']:.3f} | {sf1} | {m['certain_error_rate']:.3f} |")
    md += ["","## Interpretation","- Lower-bound recall declines with missing provenance, reflecting an API observability limit.","- Selective F1 is computed only when lower and upper bounds agree; the monitor trades coverage for honest uncertainty.","- Labels are latent structural properties in a controlled benchmark; this does not establish causal discovery or deployment performance."]
    (args.out/"REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8"); print(json.dumps(rows,indent=2))

if __name__ == "__main__": main()
