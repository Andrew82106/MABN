"""Three-part interpretable MAS monitor fusion.

Layer 1: local Laplace BN; Layer 2a: semantic message propagation; Layer 2b:
global graph-context logistic head.  A train-only meta head fuses the two
probabilities.  All features come from the public trace; labels are joined
only for fitting/evaluation and scenario_family is split metadata only.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def metric(y, p, threshold):
    pred = np.asarray(p) >= threshold
    return {"f1": float(f1_score(y,pred,zero_division=0)), "precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)), "auroc": float(roc_auc_score(y,p)) if len(np.unique(p))>1 else None, "auprc": float(average_precision_score(y,p)), "brier": float(brier_score_loss(y,p)), "threshold": float(threshold)}


def best_threshold(y, p):
    vals = np.unique(np.r_[0., p, 1.]); return float(max(vals, key=lambda t: f1_score(y,p>=t,zero_division=0)))


def logit(p):
    p = np.clip(np.asarray(p,dtype=float), 1e-5, 1-1e-5); return np.log(p/(1-p))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path('experiments/synthetic_mas/runs/independent_mas_v2/traces_public.jsonl')); ap.add_argument("--labels",type=Path,default=Path('experiments/synthetic_mas/runs/independent_mas_v2/labels.jsonl')); ap.add_argument("--out",type=Path,default=Path('experiments/synthetic_mas/runs/fusion_v3')); ap.add_argument("--seed",type=int,default=20260914); args=ap.parse_args()
    fair=load("fair_v3",Path(__file__).with_name("evaluate_independent_fair_baselines_v1.py")); two=load("two_v3",Path(__file__).with_name("evaluate_two_layer_v2.py"))
    traces=fair.read_jsonl(args.input); rows=fair.read_jsonl(args.labels); labels={r["episode_id"]:{"label":int(r["label"]),"scenario_family":r.get("scenario_family","unknown")} for r in rows}; eps=[fair.public_episode(t,labels) for t in traces]; y=np.asarray([e["label"] for e in eps],dtype=int); tr,te,split=fair.group_split(eps,.2)
    # Graph context head.
    graph_x=np.asarray([fair.graph_features(e) for e in eps]); graph_model=fair.fit_logistic(graph_x[tr],y[tr],args.seed+1); pg_tr=fair.predict_logistic(graph_model,graph_x[tr]); pg_te=fair.predict_logistic(graph_model,graph_x[te])
    # Local BN + message propagation head.
    train_eps=[eps[i] for i in tr]; test_eps=[eps[i] for i in te]; bn=two.fit_bn(train_eps,y[tr]); theta,opt=two.fit_model(train_eps,y[tr],bn,args.seed+2); pt_tr=np.asarray([two._propagate(theta,two.event_features(e),bn) for e in train_eps]); pt_te=np.asarray([two._propagate(theta,two.event_features(e),bn) for e in test_eps])
    # Train-only fusion head.  Input columns have named semantic meaning and
    # therefore remain auditable: local/message score, global graph score and
    # observability coverage.
    def coverage(e):
        p=two.event_features(e); return float(sum(f[4]==0 for _,_,f,_ in p["edges"])/max(1,len(p["edges"])))
    ztr=np.c_[logit(pt_tr),logit(pg_tr),np.asarray([coverage(e) for e in train_eps])]; zte=np.c_[logit(pt_te),logit(pg_te),np.asarray([coverage(e) for e in test_eps])]
    fusion=fair.fit_logistic(ztr,y[tr],args.seed+3); pf_tr=fair.predict_logistic(fusion,ztr); pf_te=fair.predict_logistic(fusion,zte); th=best_threshold(y[tr],pf_tr)
    results={"global_graph":metric(y[te],pg_te,best_threshold(y[tr],pg_tr)),"local_message":metric(y[te],pt_te,best_threshold(y[tr],pt_tr)),"fusion_v3":metric(y[te],pf_te,th)}
    args.out.mkdir(parents=True,exist_ok=True); report={"model":{"name":"FusionV3","components":["Layer1 Laplace local BN","Layer2 semantic message propagation","global graph-context logistic head","train-only interpretable fusion head"],"fusion_inputs":["logit(local_message_score)","logit(global_graph_score)","observed_edge_coverage"],"forbidden_fields":["label","external_effect","hidden_fields","label_source","scenario_family as feature"]},"dataset":{"n":len(eps),"positive":int(y.sum()),"negative":int(len(y)-y.sum())},"split":{**split,"train_count":int(len(tr)),"test_count":int(len(te))},"metrics":results,"fusion_parameters":[float(x) for x in fusion[0]],"limitations":["This is a synthetic hidden-effect benchmark.","Fusion improves or degrades only according to the held-out group split; no claim of causal discovery is made."]}; (args.out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); (args.out/"REPORT.md").write_text("# Fusion v3\n\n三部分监测器：局部 BN、显式消息传播和全图上下文统计；最终用训练折内的可解释融合头组合。\n\n| method | F1 | precision | recall | AUROC | AUPRC | Brier |\n|---|---:|---:|---:|---:|---:|---:|\n"+"\n".join(f"| {k} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['auroc']:.3f} | {v['auprc']:.3f} | {v['brier']:.3f} |" for k,v in results.items())+"\n\n`scenario_family` 只用于 group split，未作为特征。\n",encoding="utf-8"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
