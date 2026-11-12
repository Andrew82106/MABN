"""Grouped holdout evaluation for the independent MAS v3 benchmark.

The public traces and the hidden label file are joined only by episode_id.
scenario_family is used solely to create group folds, never as a feature.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
import evaluate_two_layer_v2 as v2


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def fpr_recall_at(y, p, fpr_target=0.05):
    y = np.asarray(y); p = np.asarray(p)
    neg = p[y == 0]
    if not len(neg) or not np.any(y == 1): return {"threshold": None, "fpr": None, "recall": None}
    # Highest threshold permitting the requested empirical FPR.
    threshold = float(np.quantile(neg, 1.0 - fpr_target, method="higher"))
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "fpr": float(fp / max(1, fp + tn)), "recall": float(tp / max(1, tp + fn))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=ROOT / 'paperAlpha/experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl')
    ap.add_argument("--labels", type=Path, default=ROOT / 'paperAlpha/experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    traces = read_jsonl(args.traces); labels = {r["episode_id"]: r for r in read_jsonl(args.labels)}
    eps = []; y = []; groups = []; mechanisms = []
    for e in traces:
        r = labels[e["episode_id"]]
        # label and family are attached only for evaluation bookkeeping. v2's
        # feature extractor reads events/agents and cannot consume either key.
        eps.append(dict(e)); y.append(int(r["label"])); groups.append(r["scenario_family"]); mechanisms.append(r["mechanism"])
    y = np.asarray(y); groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=args.folds)
    fold_rows = []; all_pred = {"ours": [], "layer1_bn": [], "topology_only": [], "edge_count": [], "single_agent": [], "dynamic_rule": [], "flat_logistic": []}; all_y=[]
    for fold, (tr, te) in enumerate(splitter.split(eps, y, groups)):
        tr_eps, te_eps = [eps[i] for i in tr], [eps[i] for i in te]
        res = v2.evaluate_split(tr_eps, te_eps, y[tr], y[te], args.seed + fold)
        mm = res["metrics"]; preds = res["predictions"]
        # Recompute baseline arrays so every fold contributes OOF probabilities.
        trparsed=[v2.event_features(e) for e in tr_eps]; teparsed=[v2.event_features(e) for e in te_eps]
        trainagg=np.asarray([[p["aggregate"][k] for k in ["event_count","agent_count","message_count","privileged_calls","untrusted_inputs"]] for p in trparsed])
        bs=v2.baseline_scores(te_eps,res["bn"],(trainagg,y[tr]))
        pmap={"ours":np.asarray([q["risk_probability"] for q in preds]), **{k:np.asarray(v) for k,v in bs.items() if v is not None}}
        fold_metrics={"fold":fold,"train":len(tr),"test":len(te),"train_groups":int(len(set(groups[tr]))),"test_groups":int(len(set(groups[te]))),"positive_test":int(y[te].sum()),"groups":sorted(set(groups[te]))}
        fold_metrics["metrics"]={"ours":mm["ours"],"baselines":mm["baselines"]}
        fold_metrics["fixed_fpr_5pct"]={k:fpr_recall_at(y[te],p) for k,p in pmap.items()}
        fold_rows.append(fold_metrics)
        all_y.extend(y[te].tolist())
        for k,p in pmap.items(): all_pred[k].extend(p.tolist())
    all_y=np.asarray(all_y); pooled={}
    for k,p in all_pred.items():
        p=np.asarray(p); pooled[k]={**v2.metric(all_y,p,.5),"ece":v2.ece(all_y,p),"fixed_fpr_5pct":fpr_recall_at(all_y,p)}
    report={"protocol":{"dataset":"independent_mas_v3","n":len(eps),"positive":int(y.sum()),"groups":int(len(set(groups))),"folds":args.folds,"split":"GroupKFold by scenario_family","seed":args.seed,"features":"public traces only; labels/mechanism/scenario_family excluded from features"},"model":{"name":"TwoLayerBNGraphMonitor-v2","layer1":"train-only Laplace CPT + node noisy-OR","layer2":"train-only sigmoid edge gates + noisy-OR propagation","causal_claim":False},"pooled_oof":pooled,"folds":fold_rows,"label_mechanism_counts":{m:int(sum(x==m for x in mechanisms)) for m in sorted(set(mechanisms))},"leakage_audit":{"label_join":"episode_id only","excluded_from_features":["label","mechanism","scenario_family","terminal outcome fields"],"grouping":"scenario_family used only for holdout","caveat":"scenario_family is available in the label file and groups may share generator conventions; this is a robustness holdout, not an external deployment test."}}
    (args.out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Independent MAS v3: grouped two-layer evaluation","",f"GroupKFold={args.folds}; groups are scenario_family; n={len(eps)}, positives={int(y.sum())}.","","## Pooled out-of-fold metrics","","| method | F1 | AUROC | AUPRC | Brier | ECE | recall at empirical 5% FPR |","|---|---:|---:|---:|---:|---:|---:|"]
    for k,m in pooled.items(): lines.append(f"| {k} | {m['f1']:.3f} | {m['auroc'] if m['auroc'] is not None else 'n/a'} | {m['auprc']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | {m['fixed_fpr_5pct']['recall']:.3f} |")
    lines += ["","## Protocol notes","- The public trace contains agents, events, and workflow DAG only. Labels are joined by episode_id after the split metadata is read.","- scenario_family never enters Layer 1 or Layer 2 features; it only defines the grouped holdout.","- The result is an OOF robustness estimate. It does not claim causal discovery or external deployment validity.","- See metrics.json for fold-level results and the leakage audit."]
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(args.out),"pooled_oof":pooled},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
