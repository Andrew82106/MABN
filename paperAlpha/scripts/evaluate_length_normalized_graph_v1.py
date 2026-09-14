"""Ablation: length-normalized path evidence for the MAS graph monitor.

The v1 stress test exposed attenuation of a product/noisy-OR path score on
long workflows.  This script keeps the same latent benchmark and frozen
cross-distribution protocol, replacing the raw path product with the maximum
geometric-mean edge gate over observed source-to-sink paths.  The geometric
mean measures average edge confidence, so its scale is not a function of path
length.  It remains an evidence score, not a causal effect estimate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import evaluate_generalization_stress_v1 as base


def norm_path(theta, ep):
    src, sink, edges = base.arrays(ep)
    names = ep["agents"]
    adj = [[] for _ in names]
    for u, v, feat, info in edges:
        if u != v:
            adj[u].append((v, float(base.sigmoid(np.dot(theta[:4], feat)))))
    best = 0.0
    for s in np.where(src > 0)[0]:
        stack = [(s, [s], [])]
        while stack:
            cur, seen, vals = stack.pop()
            if sink[cur] > 0 and vals:
                # Geometric mean prevents an artificial depth penalty.
                best = max(best, float(np.exp(np.mean(np.log(np.clip(vals, 1e-8, 1.))))))
            if len(seen) >= len(names):
                continue
            for nxt, g in adj[cur]:
                if nxt not in seen:
                    stack.append((nxt, seen + [nxt], vals + [g]))
    return best


def forward(theta, ep):
    p = norm_path(theta, ep)
    src, sink, _ = base.arrays(ep)
    local = float(1. - np.prod(1. - src * sink))
    return float(base.sigmoid(theta[4] + theta[5] * p + theta[6] * local))


def fit(eps, y, seed):
    rng = np.random.default_rng(seed)
    x0 = np.array([-2., 1., 0., 0., -2., 4., .1]) + rng.normal(0, .05, 7)
    def loss(t):
        p = np.asarray([forward(t, e) for e in eps])
        return float(-np.mean(y*np.log(np.clip(p,1e-8,1-1e-8))+(1-y)*np.log(np.clip(1-p,1e-8,1-1e-8))) + 1e-3*np.sum(t*t))
    return minimize(loss, x0, method="L-BFGS-B", options={"maxiter":300})


def threshold(y, s):
    vals=np.unique(np.r_[0.,s,1.]); return float(max(vals,key=lambda t:f1_score(y,s>=t,zero_division=0)))


def metric(y,s,t):
    p=np.asarray(s)>=t
    return {"f1":float(f1_score(y,p,zero_division=0)),"precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),"auroc":float(roc_auc_score(y,s)) if len(np.unique(s))>1 else None,"auprc":float(average_precision_score(y,s)),"threshold":float(t)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=None); ap.add_argument("--out",type=Path,default=Path("paperAlpha/results/length_normalized_graph_v1")); ap.add_argument("--seed",type=int,default=20260914); ap.add_argument("--n-train",type=int,default=1200); ap.add_argument("--n-test",type=int,default=400); args=ap.parse_args()
    cfg={"n_agents":6,"path_len":4,"decoy_multiplier":1,"reliability":1.0}
    train=base.make_set(args.n_train,cfg,args.seed); ytr=np.array([e["label"] for e in train],float); result=fit(train,ytr,args.seed); theta=result.x; trp=np.array([forward(theta,e) for e in train]); th=threshold(ytr,trp)
    scenarios={"in_distribution":cfg,"shallow_short_path":{**cfg,"n_agents":4,"path_len":2},"deep_long_path":{**cfg,"n_agents":12,"path_len":9},"many_clean_decoys":{**cfg,"decoy_multiplier":8},"sparse_taint_80pct":{**cfg,"reliability":.8},"sparse_taint_60pct":{**cfg,"reliability":.6},"sparse_taint_40pct":{**cfg,"reliability":.4},"deep_decoy_sparse":{"n_agents":12,"path_len":9,"decoy_multiplier":8,"reliability":.6}}
    rows=[]
    for j,(name,c) in enumerate(scenarios.items()):
        test=base.make_set(args.n_test,c,args.seed+100+j,start=100000*(j+1)); y=np.array([e["label"] for e in test],float); p=np.array([forward(theta,e) for e in test]); obs=[base.score_baselines(e)["dynamic_taint_path"] for e in test if e["label"]==1]
        rows.append({"scenario":name,"config":c,"n":len(test),"positive":int(y.sum()),"observed_taint_path_rate_positive":float(np.mean(obs)),"ours_length_normalized":metric(y,p,th)})
    args.out.mkdir(parents=True,exist_ok=True); (args.out/"metrics.json").write_text(json.dumps({"train_config":cfg,"train_n":len(train),"learned_parameters":theta.tolist(),"frozen_threshold":th,"scenarios":rows,"protocol":"fit only on base; geometric-mean path evidence"},indent=2),encoding="utf-8")
    md=["# Length-normalized graph monitor v1","","Geometric-mean path evidence is used to avoid raw product attenuation with workflow depth. The model and threshold are fitted only on the base distribution.","","| scenario | observed positive path | normalized model F1 |","|---|---:|---:|"]
    for r in rows: md.append(f"| {r['scenario']} | {r['observed_taint_path_rate_positive']:.3f} | {r['ours_length_normalized']['f1']:.3f} |")
    md += ["","## Caveat","This controlled stress test assesses robustness to graph depth and API observation noise; labels are latent structural properties and do not establish causal discovery or deployment SOTA."]
    (args.out/"REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8"); print(json.dumps(rows,indent=2))

if __name__ == "__main__": main()
