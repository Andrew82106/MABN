"""Frozen message-factor development study; no old test or new API calls."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read,write,sha,load_data,frozen_plan,metric
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN,compile_message_training,message_objective


def prepare(out,parent):
    old=read(parent/"PLAN.json"); report=read(parent/"REPORT.json")
    if sha(parent/"PLAN.json")!=report["plan_sha256"]: raise ValueError("parent plan changed")
    out.mkdir(parents=True,exist_ok=False)
    for filename,expected in old["inputs_sha256"].items():
        source=parent/"inputs"/filename
        if sha(source)!=expected: raise ValueError("parent input changed")
        dest=out/"inputs"/filename; dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,dest)
    names=sorted(set(old["source_sha256"])|{str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT / 'code/src').rglob("*.py")}
        |{'code/commands/run_message_hurdle_study.py','code/commands/verify_message_hurdle.py','code/commands/audit_message_dependence.py','code/tests/test_message_hurdle.py'})
    for name in names:
        dest=out/"source"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    shutil.copyfile(parent/"REPORT.json",out/"REFERENCE_REPORT.json")
    plan={**old,"schema":"observable-message-hurdle-development-v1","parent":str(parent),
        "parent_plan_sha256":sha(parent/"PLAN.json"),"reference_report_sha256":sha(out/"REFERENCE_REPORT.json"),
        "source_sha256":{n:sha(out/"source"/n) for n in names},
        "training":"Same complete future Worker and final literal labels as parent; joint message likelihood, OR gate label derived, not duplicated.",
        "inference":"Exact grouped variable elimination including all fields; known authorization union. Full joint queries may exceed factor width.",
        "comparison":"Both fixed penalties .001/.03, parent baseline metrics unchanged. Same-supervision local gate Logistic check. No new default selection.",
        "regularization":[.001,.03],"max_iter":500,"new_api_calls":0,"old_test_read":False,
        "references":[*old["references"],"https://mc-stan.org/docs/2_32/stan-users-guide/zero-inflated.html"]}
    write(out/"PLAN.json",plan)
    print(json.dumps({"prepared":str(out),"source_files":len(names),"plan_sha256":sha(out/"PLAN.json")}))


def run(out):
    plan=frozen_plan(out)
    if sha(out/"REFERENCE_REPORT.json")!=plan["reference_report_sha256"]: raise ValueError("reference results changed")
    data,projection=load_data(out)
    write(out/"PROJECTION.json",projection)
    train=[r for r in data if r["split"]=="train"]
    report={"plan_sha256":sha(out/"PLAN.json"),"reference_metrics":read(out/"REFERENCE_REPORT.json")["metrics"],
        "metrics":{},"field_metrics":{},"optimizer":{},"gate_logistic_equivalence":{},"prediction_sha256":{},"model_sha256":{},
        "new_api_calls":0,"old_test_read":False,"new_default_selected":False,
        "scope":"Previously used public development only; fixed AgentLeak workflow, literal release, same per-value/intermediate supervision as event BN."}
    for penalty in plan["regularization"]:
        name=f"message_hurdle_{penalty}"
        model=MessageHurdleWorkflowRiskBN(regularization=penalty)
        start=time.monotonic()
        model.fit_messages([r["early"].event_dag for r in train],[r["labels"] for r in train],max_iter=plan["max_iter"])
        write(out/"models"/(name+".json"),model.parameters())
        report["model_sha256"][name+".json"]=sha(out/"models"/(name+".json"))
        report["optimizer"][name]=model.optimizer
        batches,_=compile_message_training(model.event_law,[r["early"].event_dag for r in train],[r["labels"] for r in train])
        comparisons={}
        for offset,stage in ((29,"work"),(37,"publish")):
            selected=[b for b in batches if b["offset"]==offset]
            x=np.concatenate([b["g"] for b in selected]); y=np.concatenate([b["y"].any(axis=1) for b in selected])
            baseline=LogisticRegression(C=1/(2*penalty*len(train)),fit_intercept=False,max_iter=1000,tol=1e-11).fit(x,y)
            comparisons[stage]={"max_training_logit_difference":float(np.max(np.abs(baseline.decision_function(x)-x@model.theta[offset:offset+4]))),
                "coefficient":baseline.coef_[0].tolist(),"design_rank":int(np.linalg.matrix_rank(x)),"gate_positive_messages":int(y.sum())}
        report["gate_logistic_equivalence"][name]=comparisons
        print(json.dumps({"model":name,"fitted":model.optimizer,"seconds":time.monotonic()-start}),flush=True)
        records=[]; fields={}
        for i,row in enumerate(data):
            for phase in ("early","late"):
                sample=row[phase]
                p=model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
                record={"id":row["id"],"split":row["split"],"template_id":row["template_id"],"phase":phase,
                    "input_sha256":sample.sha256,"probability":p,"y":row["y"]}
                if phase=="late":
                    members=[q["event_id"] for q in sample.queries.values()]
                    summary=model.event_law.message_summary(sample.event_dag,members,model.theta[:45])
                    assert abs(summary["any_appearance_probability"]-p)<1e-10
                    record["field_probabilities"]=[summary["appearance_probabilities"][n] for n in members]
                    record["field_labels"]=[row["labels"][n]["value"] for n in members]
                    fp,fy=np.array(record["field_probabilities"]),np.array(record["field_labels"])
                    fields.setdefault(row["split"],[]).append({"mean_brier":float(np.mean((fp-fy)**2)),"sq_sum":float(np.sum((fp-fy)**2)),
                        "n":len(fp),"observed_count":int(fy.sum()),"expected_count":float(fp.sum())})
                records.append(record)
            if (i+1)%250==0:
                print(json.dumps({"model":name,"scored_flows":i+1,"elapsed_seconds":time.monotonic()-start}),flush=True)
        write(out/"predictions"/(name+".json"),records)
        report["prediction_sha256"][name+".json"]=sha(out/"predictions"/(name+".json"))
        report["metrics"][name]={split+"/"+phase:metric([r["y"] for r in records if r["split"]==split and r["phase"]==phase],
            [r["probability"] for r in records if r["split"]==split and r["phase"]==phase]) for split in ("train","validation") for phase in ("early","late")}
        report["field_metrics"][name]={split:{"flow_mean_field_brier":float(np.mean([r["mean_brier"] for r in items])),
            "pooled_field_brier":sum(r["sq_sum"] for r in items)/sum(r["n"] for r in items),
            "mean_observed_count":float(np.mean([r["observed_count"] for r in items])),
            "mean_expected_count":float(np.mean([r["expected_count"] for r in items]))} for split,items in fields.items()}
    write(out/"REPORT.json",report)
    print(json.dumps({"completed":True,"flows":len(data),"predictions":sum(2*len(data) for _ in plan["regularization"])}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["prepare","run"])
    parser.add_argument("--out",type=Path,required=True); parser.add_argument("--parent",type=Path)
    args=parser.parse_args()
    if args.mode=="prepare": prepare(args.out.resolve(),args.parent.resolve())
    else: run(args.out.resolve())
