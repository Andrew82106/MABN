"""Independent bounded R15 audit: source/selection/sampling/weights/counts.

No fitting or GPU operations. Original training labels only. Audits use explicit
linear-model arithmetic and independent counting, never import the runner.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
R14=ROOT.parent/"round14_window_detection"
spec=importlib.util.spec_from_file_location("independent_audit14",R14/"results/audit14.py")
m14=importlib.util.module_from_spec(spec);spec.loader.exec_module(m14)
m13=m14.prior;m12=m13.previous;a=m14.a
METHODS=("baseline_c01","ridge_c003","ridge_c001","heads64_c01","heads128_c01","ensemble_c01","ensemble_c003")
CS=dict(zip(METHODS,(.01,.003,.001,.01,.01,.01,.003)))


def single_score(wrapper,x):
    model=wrapper["model"]
    # Preserve the selected-column array's memory layout and the LR sigmoid
    # kernel so exact score/threshold equality is not changed by audit rounding.
    z=np.array(x[:,wrapper["columns"]],copy=True,order="K")
    z-=model["scaler"].mean_;z/=model["scaler"].scale_
    z=z.astype(np.float32)
    return expit((z@model["model"].coef_.T+model["model"].intercept_).ravel())


def check_single(wrapper,x,rows,indices,columns,c,target):
    np.testing.assert_array_equal(wrapper["fit_indices"],indices)
    np.testing.assert_array_equal(wrapper["columns"],columns)
    fit_rows=[rows[j] for j in indices]
    a.require(wrapper["fit_groups"]==sorted({r["group_id"] for r in fit_rows}),"Member fitted wrong groups")
    b,w,factors=m12.direct_weights(fit_rows,"token");w*=target/w.sum()
    model=wrapper["model"]
    a.require(wrapper["kind"]=="single" and model["C"]==model["model"].C==c,"Member C differs")
    a.require(model["feature"]=="lb" and model["components"] is None and model["target_loss_mass"]==target,"Member feature/weight policy differs")
    a.require(model["model"].coef_.shape==(1,len(columns)),"Wrong selected-head width")
    np.testing.assert_allclose(model["base_weights"],b,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(model["loss_weights"],w,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(model["class_factors"],factors,atol=1e-12,rtol=1e-12)
    m12.check_scaler(model["scaler"],x[indices][:,columns],b,"member fit-only scaler")


def head_order(x,rows):
    base,_,_=m12.direct_weights(rows,"token");base/=base.sum()
    y=np.asarray([r["gold"] for r in rows],float)
    mean,var=m12.weighted_stats(x,base)
    bound=base.sum()*np.finfo(float).eps*var+(base.sum()*mean*np.finfo(float).eps)**2
    scale=np.where(var<=bound,1.,np.sqrt(var))
    z=x.copy();z-=mean;z/=scale;z=z.astype(float)
    yc=y-np.dot(base,y)
    corr=(z.T@(base*yc))/np.sqrt(np.dot(base,yc**2))
    corr[var<=np.finfo(float).eps]=0.
    order=np.lexsort((np.arange(784),-np.abs(corr)))
    return order,corr


def bootstrap(rows):
    groups=sorted({r["group_id"] for r in rows});counts=np.zeros((len(groups),len(METHODS),3),int)
    for j,g in enumerate(groups):
        rr=[r for r in rows if r["group_id"]==g]
        for k,name in enumerate(METHODS):
            c=m14.binary_metrics([r["gold"] for r in rr],[r["predictions"][name] for r in rr])
            counts[j,k]=c["tp"],c["fp"],c["fn"]
    samples=np.random.default_rng(20260915).integers(0,len(groups),(2000,len(groups)))
    total=np.asarray([counts[ix].sum(0) for ix in samples]);tp,fp,fn=(total[:,:,k] for k in range(3))
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.full(tp.shape,np.nan,float),where=tp+fn>0)
    def ci(v):
        good=v[np.isfinite(v)]
        return {"ci95":np.quantile(good,[.025,.975]).tolist() if len(good) else None,"defined_draws":len(good)}
    return {"f1":{n:ci(f1[:,j]) for j,n in enumerate(METHODS)},
            "difference_from_baseline_f1":{n:ci(f1[:,j]-f1[:,0]) for j,n in enumerate(METHODS) if j}}


def run(root):
    out=root/"results";done=a.read_json(out/"complete15.json")
    a.require(done["new_fits"]==260 and done["reused_models"]==5 and not done["original_validation_or_test_labels_used"],"Wrong scope")
    for name,digest in done["files_sha256"].items():a.require(a.sha(out/name)==digest,"Completed artifact changed: "+name)
    snap=a.read_json(out/"source_snapshot15.json");started=a.read_json(out/"started15.json")
    a.require(started["source_snapshot_sha256"]==a.sha(out/"source_snapshot15.json"),"Start snapshot changed")
    a.require(snap["prior"]==a.read_json(R14/"results/source_snapshot14.json"),"R14 source graph differs")
    for base,files in ((root,snap["local_files_sha256"]),(R14,snap["prior"]["local_files_sha256"]),
                       (m14.R13,snap["prior"]["prior"]["local_files_sha256"]),
                       (m13.R12,snap["prior"]["prior"]["prior"]["local_files_sha256"])):
        for name,digest in files.items():a.require(a.sha(base/name)==digest,"Frozen source changed: "+name)
    for name,digest in snap["prior_results_sha256"].items():a.require(a.sha(R14/"results"/name)==digest,"R14 baseline artifact changed")
    config=a.read_json(root/"protocol.json")
    a.require(config["methods"]==list(METHODS) and config["C"]==CS and config["width"]==4,"Candidate protocol differs")
    a.require((config["ensemble_members"],config["ensemble_fit_groups"],config["ensemble_seed"])==(24,54,20260915),"Sampling budget differs")
    items,tokens,gold=m13.training_gold();windows,matrices=m14.training_windows(items,tokens)
    rows=[m14.canonical(w) for w in windows[4]];x=matrices[4]
    a.require(rows==a.read_lines(R14/"results/window_catalog_k4.jsonl"),"Window data changed")
    saved=a.read_lines(out/"window_scores_k4.jsonl");lookup={r["window_key"]:r for r in saved}
    old_scores={r["window_key"]:r for r in a.read_lines(R14/"results/window_scores_k4.jsonl")}
    a.require(len(rows)==len(saved)==len(lookup) and set(lookup)==set(old_scores),"Missing/duplicate windows")
    max_baseline_error=0.
    for row in rows:
        s=lookup[row["window_key"]];old=old_scores[row["window_key"]]
        a.require(all(s[k]==v for k,v in row.items()),"Window geometry/gold differs")
        a.require(set(s["scores"])==set(s["predictions"])==set(METHODS),"Missing predictor")
        a.require(all(a.finite(v) for v in s["scores"].values()),"Nonfinite score")
        delta=abs(s["scores"]["baseline_c01"]-old["scores"]["window_lr"]);max_baseline_error=max(max_baseline_error,delta)
        a.require(delta<=1e-12 and s["predictions"]["baseline_c01"]==old["predictions"]["window_lr"] and s["fold"]==old["fold"],"Original baseline not reproduced")
    models=pickle.loads((out/"fitted_folds.pkl").read_bytes());old_models=pickle.loads((R14/"results/fitted_folds.pkl").read_bytes())
    summary=a.read_json(out/"summary.json");a.require(summary["coverage"]==gold["coverage"],"Coverage changed")
    ids=np.asarray([r["group_id"] for r in rows]);assigned=np.zeros(len(rows),int)
    report={"schema":"round15-independent-audit-v1","status":"passed","no_refitting_or_gpu":True,
            "gold":gold,"source_and_result_hashes":"passed","folds":{},"metrics":{},
            "baseline_replication":{"max_score_difference":max_baseline_error,"all_predictions_and_folds_identical":True}}
    checked_members=0
    for fold,block in models.items():
        old=old_models[fold]
        for field in ("fit_groups","calibration_groups","evaluation_groups"):a.require(block[field]==old[field],"Wrong group split")
        fit,cal,evgroups=map(set,(block["fit_groups"],block["calibration_groups"],block["evaluation_groups"]))
        a.require((len(fit),len(cal),len(evgroups))==(72,24,24) and not fit&cal and not fit&evgroups and not cal&evgroups,"Group leakage")
        ix,ca,ev=[np.flatnonzero(np.isin(ids,block[name])) for name in ("fit_groups","calibration_groups","evaluation_groups")]
        assigned[ev]+=1;target=sum(w["group_id"] in fit for w in windows[1])
        a.require(target==block["target_loss_mass"],"Anchor loss mass differs")
        order,corr=head_order(x[ix],[rows[j] for j in ix])
        np.testing.assert_allclose(corr,block["head_correlations"],atol=1e-7,rtol=1e-7)
        np.testing.assert_array_equal(order,block["head_ranking"])
        rng=np.random.default_rng(20260915+fold)
        subsets=[sorted(str(g) for g in rng.choice(sorted(fit),54,replace=False)) for _ in range(24)]
        a.require(subsets==block["subsets"],"Random subsets not reproduced")
        sub_ix=[np.flatnonzero(np.isin(ids,ss)) for ss in subsets]
        a.require(set(block["models"])==set(METHODS),"Wrong method inventory")
        detail={}
        for name,wrapper in block["models"].items():
            member_values=None
            if name.startswith("ensemble"):
                a.require(wrapper["kind"]=="ensemble" and len(wrapper["members"])==24 and wrapper["C"]==CS[name],"Wrong ensemble")
                values=[]
                for member,indices in zip(wrapper["members"],sub_ix):
                    check_single(member,x,rows,indices,np.arange(784),CS[name],target);checked_members+=1
                    values.append(single_score(member,x))
                member_values=np.asarray(values);score=member_values.mean(0)
            else:
                cols=order[:64] if name=="heads64_c01" else order[:128] if name=="heads128_c01" else np.arange(784)
                check_single(wrapper,x,rows,ix,cols,CS[name],target)
                if name.startswith("heads"):
                    np.testing.assert_allclose(wrapper["selection_correlation"],corr,atol=1e-7,rtol=1e-7)
                score=single_score(wrapper,x)
            threshold=wrapper["threshold"]
            chosen=m13.independent_threshold([{"gold":rows[j]["gold"],"scores":{name:float(score[j])}} for j in ca],name)
            a.compare_values(chosen,wrapper["threshold_selection"],"Calibration-only threshold")
            a.require(threshold==wrapper["threshold_selection"]["threshold"],"Stored threshold mismatch")
            stated=summary["folds"][str(fold)][name]
            a.require(stated["threshold"]==wrapper["threshold_selection"],"Summary threshold provenance differs")
            if name=="baseline_c01":a.require(wrapper["threshold_selection"]==old["models"][4]["thresholds"]["window_lr"],"Baseline threshold changed")
            for stage,indices in (("fit_pool",ix),("calibration",ca),("evaluation",ev)):
                m14.compare_metrics([rows[j] for j in indices],name,stated[stage],score[indices],threshold)
            if member_values is not None:
                a.require(len(stated["member_metrics_at_ensemble_threshold"])==24,"Member diagnostic coverage differs")
                for j,indices in enumerate(sub_ix):
                    mm=stated["member_metrics_at_ensemble_threshold"][j]
                    m14.compare_metrics([rows[k] for k in indices],name,mm["fit"],member_values[j,indices],threshold)
                    m14.compare_metrics([rows[k] for k in ev],name,mm["evaluation"],member_values[j,ev],threshold)
                a.require("out-of-bag" in stated["fit_pool_note"],"Ensemble training score mislabelled")
            errors=[]
            for j in ev:
                row=lookup[rows[j]["window_key"]]
                delta=abs(row["scores"][name]-score[j]);errors.append(delta)
                a.require(delta<=1e-9 and row["fold"]==fold,"OOF score does not match frozen model")
                a.require(row["predictions"][name]==(row["scores"][name]>=threshold),"Wrong fold threshold prediction")
            detail[name]={"max_outer_score_error":max(errors),"threshold":threshold,"fit_only_preprocessing_and_weights":"passed"}
        report["folds"][str(fold)]={"disjoint_groups":[72,24,24],"shared_24_subsets_reproduced":True,
            "fit_only_head_ranking_reproduced":True,"target_loss_mass":target,"methods":detail}
    a.require(checked_members==240 and len(models)==5 and (assigned==1).all(),"OOF/member coverage incomplete")
    a.require(summary["window_count"]==len(rows) and summary["positive_windows"]==sum(r["gold"] for r in rows),"Window totals differ")
    for name in METHODS:
        report["metrics"][name]=m14.compare_metrics(saved,name,summary["methods"][name])
        for stage in ("fit_pool","calibration","evaluation"):
            for metric in ("f1","average_precision","auroc"):
                expected=np.mean([summary["folds"][str(f)][name][stage][metric] for f in range(5)])
                np.testing.assert_allclose(expected,summary["fold_mean"][name][stage][metric],atol=1e-12,rtol=0)
        for category,methods in summary["categories"].items():
            m14.compare_metrics([r for r in saved if r["category"]==category],name,methods[name])
    boot=bootstrap(saved);stated=summary["paired_bootstrap"]
    a.require((stated["groups"],stated["draws"],stated["seed"])==(120,2000,20260915),"Bootstrap scope differs")
    for field in ("f1","difference_from_baseline_f1"):
        for name,interval in boot[field].items():m12.prior.check_interval(interval,stated[field][name],name)
    records=a.read_lines(out/"highlight_regions.jsonl")
    a.require(len(records)==len({(r['width'],r['method'],r['item_id']) for r in records})==195*7,"Coarse highlight coverage differs")
    original=m14.METHODS
    try:
        m14.METHODS=METHODS
        coarse=m14.audit_highlights(4,saved,items,tokens,summary["coarse_highlight"],records)
    finally:m14.METHODS=original
    report.update(paired_bootstrap=boot,coarse_highlight=coarse,checked_ensemble_members=240,summary_sha256=a.sha(out/"summary.json"))
    a.write_json(out/"INDEPENDENT_AUDIT15.json",report)
    print(json.dumps({"status":"passed","new_models":260,"ensemble_members":240,
        "baseline_max_score_difference":max_baseline_error,"f1":{n:v["f1"] for n,v in report["metrics"].items()}},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["run"]);parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):run(args.root.resolve())
