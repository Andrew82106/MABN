"""Split-conformal prediction sets for the TwoLayer MAS monitor."""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np


def load_two():
    p=Path(__file__).with_name("evaluate_two_layer_v2.py"); s=importlib.util.spec_from_file_location("two_conf",p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m


def read(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def group_split(eps, fraction=.2, seed=20260914):
    groups=sorted({e.get("scenario_family","unknown") for e in eps}); rng=np.random.default_rng(seed); rng.shuffle(groups); held=set(groups[:max(1,int(len(groups)*fraction))]); te=np.asarray([i for i,e in enumerate(eps) if e.get("scenario_family","unknown") in held]); tr=np.asarray([i for i in range(len(eps)) if i not in set(te)]); return tr,te


def conformal_sets(y_cal, p_cal, p_test, alpha=.1):
    y_cal=np.asarray(y_cal); p_cal=np.asarray(p_cal); p_test=np.asarray(p_test); pos=np.sort(1-p_cal[y_cal==1]); neg=np.sort(p_cal[y_cal==0]); out=[]
    for p in p_test:
        # Smoothed finite-sample p-values; class-conditional calibration.
        pv1=(1+np.sum(pos >= 1-p))/(len(pos)+1) if len(pos) else 0.0
        pv0=(1+np.sum(neg >= p))/(len(neg)+1) if len(neg) else 0.0
        allowed=tuple(k for k,pv in ((0,pv0),(1,pv1)) if pv > alpha); out.append((allowed,pv0,pv1))
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path('experiments/synthetic_mas/runs/independent_mas_v2/traces_public.jsonl')); ap.add_argument("--labels",type=Path,default=Path('experiments/synthetic_mas/runs/independent_mas_v2/labels.jsonl')); ap.add_argument("--out",type=Path,default=Path('experiments/synthetic_mas/runs/conformal_monitor_v1')); ap.add_argument("--alpha",type=float,default=.1); ap.add_argument("--seed",type=int,default=20260914); args=ap.parse_args()
    two=load_two(); traces=read(args.input); rows=read(args.labels); meta={r["episode_id"]:r for r in rows}; eps=[]
    for e in traces:
        r=meta[e["episode_id"]]; eps.append({**e,"label":int(r["label"]),"scenario_family":r.get("scenario_family","unknown")})
    y=np.asarray([e["label"] for e in eps]); tr,te=group_split(eps,.2,args.seed); groups=[eps[i].get("scenario_family","unknown") for i in tr]; unique=sorted(set(groups)); rng=np.random.default_rng(args.seed+1); rng.shuffle(unique); cal_groups=set(unique[:max(1,int(.2*len(unique)))]); cal=np.asarray([i for i in tr if eps[i].get("scenario_family","unknown") in cal_groups]); fit=np.asarray([i for i in tr if i not in set(cal)])
    fit_eps=[eps[i] for i in fit]; cal_eps=[eps[i] for i in cal]; test_eps=[eps[i] for i in te]; bn=two.fit_bn(fit_eps,y[fit]); theta,opt=two.fit_model(fit_eps,y[fit],bn,args.seed); pc=np.asarray([two._propagate(theta,two.event_features(e),bn) for e in cal_eps]); pt=np.asarray([two._propagate(theta,two.event_features(e),bn) for e in test_eps]); sets=conformal_sets(y[cal],pc,pt,args.alpha); sizes=np.asarray([len(s[0]) for s in sets]); covered=np.asarray([int(y[i] in sets[j][0]) for j,i in enumerate(te)]); singleton=np.asarray([len(s[0])==1 for s in sets]); singleton_correct=np.asarray([len(s[0])==1 and y[i] in s[0] for j,(s,i) in enumerate(zip(sets,te))]); empty=sizes==0; both=sizes==2
    result={"alpha":args.alpha,"fit_cal_test":{"fit":int(len(fit)),"calibration":int(len(cal)),"test":int(len(te)),"fit_groups":len(set(eps[i].get("scenario_family") for i in fit)),"calibration_groups":len(cal_groups)},"metrics":{"marginal_coverage":float(covered.mean()),"mean_prediction_set_size":float(sizes.mean()),"singleton_rate":float(singleton.mean()),"singleton_accuracy":float(singleton_correct.sum()/max(1,singleton.sum())),"abstention_empty_set_rate":float(empty.mean()),"uncertain_both_classes_rate":float(both.mean())},"optimizer":opt,"interpretation":"Conformal coverage is a finite-sample guarantee under exchangeability; it is not a causal or deployment guarantee."}
    args.out.mkdir(parents=True,exist_ok=True); (args.out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); (args.out/"REPORT.md").write_text("# Conformal MAS monitor v1\n\nTwoLayer BN+graph 的 split-conformal 输出集合；校准集按拓扑族与测试集隔离。\n\n- alpha=%.2f\n- marginal coverage=%.3f\n- mean set size=%.3f\n- singleton rate=%.3f\n- singleton accuracy=%.3f\n- empty-set abstention=%.3f\n- both-class uncertainty=%.3f\n\n覆盖率解释依赖校准/测试交换性，不能外推为真实部署保证。\n"%(args.alpha,result["metrics"]["marginal_coverage"],result["metrics"]["mean_prediction_set_size"],result["metrics"]["singleton_rate"],result["metrics"]["singleton_accuracy"],result["metrics"]["abstention_empty_set_rate"],result["metrics"]["uncertain_both_classes_rate"]),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
