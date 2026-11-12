"""Freeze and run same-label, same-effective-feature graph controls.

Existing exposed development data only. All predeclared candidates are reported;
no baseline/model selection, new API requests, or old held-out test access.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import shutil
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_compiled_multiworkflow_study import read,write,sha,make_record,risk_metrics
from compiled_multiworkflow_data import load_data
from paperalpha_monitor.structured_event_baseline import StructuredEventBaseline,TREE_CONFIG

CONFIG={"regularization":[.001,.03],"histgb":TREE_CONFIG,"max_iter":500,"probability_digits":8,
    "inference":["exact","mean_field","same_marginals_independent"],
    "no_parent_control":"refit with the two parent columns disabled; same labels and other features",
    "selection":"All predeclared candidates; no new default or validation-based threshold selection",
    "scope":"Same-supervision decomposition controls, themselves graphical conditional models; not original MAS baselines."}


def now(): return datetime.now(timezone.utc).isoformat()


def source_data(source):
    plan=read(source/"PLAN.json"); report=read(source/"REPORT.json"); fit=read(source/"FIT.json")
    if plan["schema"]!="compiled-multiworkflow-development-v1": raise ValueError("expected typed multiworkflow source")
    for n,h in plan["source_sha256"].items():
        if sha(source/"source"/n)!=h: raise ValueError("original frozen source changed: "+n)
    for n,h in plan["compiled_sha256"].items():
        if sha(source/n)!=h: raise ValueError("original compiled evidence changed")
    for n,h in fit["model_sha256"].items():
        if sha(source/"models"/n)!=h: raise ValueError("original fitted model changed")
    if report["plan_sha256"]!=sha(source/"PLAN.json") or report["fit_sha256"]!=sha(source/"FIT.json") or report["predictions_sha256"]!=sha(source/"PREDICTIONS.json"):
        raise ValueError("original report binding changed")
    data,verification=load_data(Path(plan["branch_run"]),Path(plan["review_run"]))
    partition={r["id"]:{"split":r["split"],"template_id":r["template_id"],"source_hash":r["source_hash"]} for r in data}
    if verification!=plan["verification"] or partition!=plan["partition"]: raise ValueError("source replay differs")
    for row in data:
        for phase,sample in row["samples"].items():
            if sample.event_dag.specification()!=read(source/"compiled"/row["id"].replace(":","_")/(phase+"_graph.json")):
                raise ValueError("effective graph differs from original experiment")
    return plan,data


def prepare(out,source):
    old,data=source_data(source)
    names=set(old["source_sha256"])
    names.update(['code/src/paperalpha_monitor/structured_event_baseline.py','code/commands/run_matched_event_controls.py',
                  'code/commands/verify_matched_event_controls.py','code/tests/test_structured_event_baseline.py',
                  'code/tests/test_message_workflow_compiler.py'])
    out.mkdir(parents=True,exist_ok=False)
    for name in sorted(names):
        dest=out/"source"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    plan={"schema":"matched-event-controls-v1","created_at_utc":now(),"config":CONFIG,
        "source_run":str(source),"original_hashes":{n:sha(source/n) for n in ("PLAN.json","FIT.json","REPORT.json","PREDICTIONS.json")},
        "partition":old["partition"],"source_sha256":{n:sha(out/"source"/n) for n in sorted(names)},
        "supervision":"Same 20 earliest training prefixes and all 58 actual future event labels for every new estimator and original event BN.",
        "new_api_calls":0,"old_test_read":False,"development_only":True}
    write(out/"PLAN.json",plan)
    print(json.dumps({"prepared":str(out),"flows":len(data),"plan_sha256":sha(out/"PLAN.json")}),flush=True)


def load(out):
    plan=read(out/"PLAN.json")
    if ROOT!=(out/"source").resolve() or plan["config"]!=CONFIG: raise ValueError("run the unchanged frozen source")
    for n,h in plan["source_sha256"].items():
        if sha(ROOT/n)!=h: raise ValueError("frozen control source changed: "+n)
    source=Path(plan["source_run"])
    for n,h in plan["original_hashes"].items():
        if sha(source/n)!=h: raise ValueError("original comparison artifacts changed")
    original,data=source_data(source)
    if plan["partition"]!=original["partition"]: raise ValueError("partition changed")
    return plan,data


def summarise(predictions):
    result={}
    for name,records in predictions.items():
        metrics={}
        for split in ("train","validation"):
            for cohort,phase in (("review","early"),("review","late"),("branch","late")):
                rows=[r for r in records if r["split"]==split and r["cohort"]==cohort and r["phase"]==phase]
                metrics[f"{split}/{cohort}/{phase}"]=risk_metrics(rows)
            for moment in ("first","last"):
                rows=[r for r in records if r["split"]==split and r["phase"]==("early" if moment=="first" and r["cohort"]=="review" else "late")]
                metrics[f"{split}/pooled_{moment}"]=risk_metrics(rows)
        result[name]=metrics
    return result


def run(out):
    plan,data=load(out); source=Path(plan["source_run"])
    if (out/"models").exists(): raise ValueError("do not overwrite existing fits")
    train=[r for r in data if r["split"]=="train"]
    dags=[r["samples"][r["training_phase"]].event_dag for r in train]; labels=[r["labels"] for r in train]
    models={}
    for relations in (True,False):
        suffix="" if relations else "_no_parent"
        for penalty in CONFIG["regularization"]:
            name=f"conditional_glm_{penalty}{suffix}"
            models[name]=StructuredEventBaseline(relations=relations,penalty=penalty).fit(dags,labels,max_iter=CONFIG["max_iter"])
        name="conditional_histgb"+suffix
        models[name]=StructuredEventBaseline(kind="histgb",relations=relations,tree_config=CONFIG["histgb"]).fit(dags,labels)
    for name,model in models.items():
        write(out/"models"/(name+".json"),model.metadata())
        if model.tree is not None: joblib.dump(model.tree,out/"models"/(name+".joblib"))
        print(json.dumps({"fit":name,"details":model.fit_report}),flush=True)
    write(out/"FIT.json",{"created_at_utc":now(),"train_ids":[r["id"] for r in train],
        "train_graph_hashes":[d.sha256 for d in dags],"plan_sha256":sha(out/"PLAN.json"),
        "model_sha256":{p.name:sha(p) for p in sorted((out/"models").iterdir())}})
    old=read(source/"PREDICTIONS.json")
    original_lookup={name:{(r["id"],r["phase"]):r for r in records} for name,records in old.items()}
    predictions={}; equivalence={}
    for name,model in models.items():
        modes=CONFIG["inference"] if model.relations else ["exact"]
        for row in data:
            for phase,sample in row["samples"].items():
                # Reuse the exact same prefix policy, not a weaker policy baseline.
                previous=original_lookup["event_bn_0.001"][row["id"],phase]
                if previous["sample_sha256"]!=sample.sha256: raise ValueError("policy/event binding differs")
                targets=[sample.queries[a]["event_id"] for a in sample.action_ids]
                for mode in modes:
                    joint=model.predict_joint(sample.event_dag,targets,inference=mode)
                    predictions.setdefault(name+"/"+mode,[]).append(make_record(row,phase,joint,previous["policy_truth"]))
        if model.kind=="glm" and model.relations:
            original=read(source/"models"/(f"event_bn_{model.penalty}.json"))
            records=predictions[name+"/exact"]; reference=original_lookup[f"event_bn_{model.penalty}"]
            equivalence[name]={"max_coefficient_difference":float(np.max(np.abs(model.coef-np.asarray(original["theta"][:25])))),
                "objective_difference":abs(model.fit_report["loss"]-original["optimizer"]["loss"]),
                "max_joint_difference":max(float(np.max(np.abs(np.asarray(r["joint"])-reference[r["id"],r["phase"]]["joint"]))) for r in records),
                "max_operational_risk_difference":max(abs(r["risk_probability"]-reference[r["id"],r["phase"]]["risk_probability"]) for r in records)}
    write(out/"PREDICTIONS.json",predictions)
    report={"completed_at_utc":now(),"plan_sha256":sha(out/"PLAN.json"),"fit_sha256":sha(out/"FIT.json"),
        "predictions_sha256":sha(out/"PREDICTIONS.json"),"metrics":summarise(predictions),"glm_bn_equivalence":equivalence,
        "original_metrics":read(source/"REPORT.json")["metrics"],"models_fitted":len(models),
        "scoring_methods":len(predictions),"predictions":sum(map(len,predictions.values())),
        "new_default_selected":False,"new_api_calls":0,"old_test_read":False,
        "scope":CONFIG["scope"],"supervision":plan["supervision"],
        "limitations":["Previously exposed authored 20-train/20-validation pilot; only 2 validation positives.",
            "Conditional GLM/tree controls use the same DAG and therefore are themselves BN factorizations, not competing original MAS monitors.",
            "Mean-field is an explicitly labeled approximation; same-marginal independence removes only terminal dependence.",
            "Tree and GLM regularization/optimization differ; equal evidence and labels do not mean equal inductive bias.",
            "The earlier terminal-only direct classifiers remain unequal-supervision comparisons; no retrospective replacement."]}
    write(out/"REPORT.json",report)
    print(json.dumps({"completed":True,"models":len(models),"methods":len(predictions),"predictions":report["predictions"],"equivalence":equivalence}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("prepare","run")); parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source-run",type=Path)
    args=parser.parse_args()
    if args.mode=="prepare": prepare(args.out.resolve(),args.source_run.resolve())
    else: run(args.out.resolve())
