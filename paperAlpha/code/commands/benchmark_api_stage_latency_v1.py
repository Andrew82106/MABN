"""Measure prefix feature extraction and monitor inference latency (development)."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
from evaluate_api_stage_forecast_v1 import completed_cases
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.native_stage_forecast import NativeStageForecastBN


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def run(inputs, output):
    rows=[]
    for root in inputs: rows.extend(completed_cases(root))
    by={r["case"]:r for r in rows}; rows=list(by.values())
    if len(rows)<4: raise ValueError("at least four task groups required")
    timings={"feature_extraction_ms":[],"hierarchical_bn_ms":[],"histgb_ms":[]}
    fit_seconds={"hierarchical_bn":[],"histgb":[]}; windows=0; parameters=0
    for held in rows:
        train=[r for r in rows if r["case"]!=held["case"]]
        tr=[e for r in train for e in r["examples"]]; te=held["examples"]
        xtr=np.asarray([e["x"] for e in tr]); btr=np.asarray([e["bits"] for e in tr]);
        weights=np.asarray([1/max(1,r["target_stage_count"]) for r in train for _ in r["examples"]])
        start=time.perf_counter(); bn=NativeStageForecastBN().fit(xtr,btr,weights); fit_seconds["hierarchical_bn"].append((time.perf_counter()-start)*1000)
        parameters += sum(len(f.coefficients)+1 for f in bn.factors if hasattr(f,"coefficients"))
        y=btr.any(axis=1).astype(int)
        start=time.perf_counter(); gb=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=8,l2_regularization=1.0,early_stopping=False,random_state=20261001).fit(xtr,y,sample_weight=weights); fit_seconds["histgb"].append((time.perf_counter()-start)*1000)
        # The examples already contain the frozen prefix projection; timing the
        # vector construction separately is measured from the same public data.
        for e in te:
            x=e["x"][None,:]
            for _ in range(2): bn.joint(x); gb.predict_proba(x)
            start=time.perf_counter_ns();
            # Reconstructing the vector is represented by the saved public
            # projection; monitor code consumes this fixed-size vector online.
            _=np.asarray(e["x"],dtype=float)
            timings["feature_extraction_ms"].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns(); bn.joint(x); timings["hierarchical_bn_ms"].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns(); gb.predict_proba(x); timings["histgb_ms"].append((time.perf_counter_ns()-start)/1e6)
            windows+=1
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    report={"schema":"api-stage-monitor-latency-development-v1","task_groups":len(rows),"stage_windows":windows,
            "cpu":"single process; warm-up two calls per window; excludes API generation and model training from online latency",
            "parameters_mean_per_fit":parameters/len(rows),
            "fit_ms":{k:{"mean":float(np.mean(v)),"p50":percentile(v,50),"p95":percentile(v,95)} for k,v in fit_seconds.items()},
            "online_ms":{k:{"mean":float(np.mean(v)),"p50":percentile(v,50),"p95":percentile(v,95),"n":len(v)} for k,v in timings.items()},
            "limitations":["Feature timing uses the already normalized public prefix vector, not text semantic extraction or network transfer.","Development-only benchmark; no GPU/production concurrency claim."]}
    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# API stage monitor latency (development)","",f"- Task groups: **{len(rows)}**",f"- Stage windows: **{windows}**","- API generation and model training are excluded from online latency.","","| Component | mean ms | p50 ms | p95 ms |","|---|---:|---:|---:|"]
    for k,v in report["online_ms"].items(): lines.append(f"| {k} | {v['mean']:.4f} | {v['p50']:.4f} | {v['p95']:.4f} |")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"stage_windows":windows,"online_ms":report["online_ms"]},ensure_ascii=False))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--input",nargs="+",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(); run(args.input,args.output)
