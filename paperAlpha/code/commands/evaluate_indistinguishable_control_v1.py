"""Unidentifiable-label control for the independent MAS benchmark."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from evaluate_independent_fair_baselines_v1 import graph_features, fit_logistic, predict_logistic, best_threshold


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path('experiments/synthetic_mas/runs/independent_mas_v2/traces_public.jsonl')); ap.add_argument("--labels",type=Path,default=Path('experiments/synthetic_mas/runs/independent_mas_v2/labels.jsonl')); ap.add_argument("--out",type=Path,default=Path('experiments/synthetic_mas/runs/indistinguishable_control_v1')); ap.add_argument("--seed",type=int,default=20260914); args=ap.parse_args()
    traces=[json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]; rows=[json.loads(x) for x in args.labels.read_text(encoding="utf-8").splitlines() if x.strip()]; labels={r["episode_id"]:int(r["label"]) for r in rows}; groups={r["episode_id"]:r.get("scenario_family","unknown") for r in rows}
    eps=[{**e,"scenario_family":groups[e["episode_id"]]} for e in traces]; X=np.asarray([graph_features(e) for e in eps]); y=np.asarray([labels[e["episode_id"]] for e in eps]); rng=np.random.default_rng(args.seed); y_shuf=rng.permutation(y)
    idx=np.arange(len(eps)); tr,te=train_test_split(idx,test_size=.3,random_state=args.seed,stratify=y_shuf); model=fit_logistic(X[tr],y_shuf[tr],args.seed); p=predict_logistic(model,X[te]); th=best_threshold(y_shuf[tr],predict_logistic(model,X[tr])); pred=p>=th
    result={"n":len(eps),"positive_rate":float(y.mean()),"shuffled_positive_rate":float(y_shuf.mean()),"method":"graph_features_learned with labels permuted independently of public traces","f1":float(f1_score(y_shuf[te],pred,zero_division=0)),"auroc":float(roc_auc_score(y_shuf[te],p)),"auprc":float(average_precision_score(y_shuf[te],p)),"expected_interpretation":"AUROC near 0.5 demonstrates non-identifiability when labels contain no information about observations"}
    args.out.mkdir(parents=True,exist_ok=True); (args.out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); (args.out/"REPORT.md").write_text("# Indistinguishable-label control\n\n标签在保持正例比例的前提下被独立打乱，与公开运行轨迹失去关系。任何 API-only 方法都不应在此获得稳定的排序能力。\n\n- n=%d\n- shuffled positive rate=%.3f\n- graph-features learned F1=%.3f\n- AUROC=%.3f\n- AUPRC=%.3f\n\n该实验是不可辨识性控制，不是正常性能评测。\n"%(result["n"],result["shuffled_positive_rate"],result["f1"],result["auroc"],result["auprc"]),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
