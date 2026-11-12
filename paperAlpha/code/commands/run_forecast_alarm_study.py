"""Fit/calibration/validation-separated development of episode alarm decisions.

All models refit without calibration templates. Fixed policies are frozen
before this run scores validation. Prior development exposure is disclosed.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import shutil
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read,write,sha,load_data,frozen_plan,metric,baseline_features
from paperalpha_runtime.canonical import digest
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import fit_event_traces
from paperalpha_monitor.forecast_alarm import calibrate_forecast,fixed_policy,forecast_alarm_metrics


def now(): return datetime.now(timezone.utc).isoformat()


def split_templates(rows,config):
    names=sorted({r["template_id"] for r in rows},key=lambda t:hashlib.sha256((config["split_seed"]+t).encode()).hexdigest())
    n=config["calibration_template_count"]
    if type(n) is not int or not 0<n<len(names): raise ValueError("nonempty fit/calibration templates required")
    selected=set(names[:n])
    return {r["id"]:("calibration" if r["template_id"] in selected else "fit") for r in rows},names


def prepare(out,parent,config_path):
    config=read(config_path); old=read(parent/"PLAN.json"); report=read(parent/"REPORT.json")
    if sha(parent/"PLAN.json")!=report["plan_sha256"]: raise ValueError("parent changed")
    for filename,expected in old["inputs_sha256"].items():
        if sha(parent/"inputs"/filename)!=expected: raise ValueError("parent data changed")
    assignment,ordered=split_templates(read(parent/"inputs/train.json"),config)
    for row in read(parent/"inputs/validation.json"):
        if row["id"] in assignment or row["template_id"] in ordered: raise ValueError("original split overlap")
        assignment[row["id"]]="validation"
    out.mkdir(parents=True,exist_ok=False)
    for filename in old["inputs_sha256"]:
        dest=out/"inputs"/filename; dest.parent.mkdir(exist_ok=True); shutil.copyfile(parent/"inputs"/filename,dest)
    names=sorted(set(old["source_sha256"])|{str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT / 'code/src').rglob("*.py")}
        |{'code/commands/run_forecast_alarm_study.py','code/commands/verify_forecast_alarm_study.py','code/tests/test_forecast_alarm.py','code/tests/test_forecast_alarm_study.py',
            'code/tests/test_message_policy_composition.py','code/tests/test_phase_alarm_policy.py','code/commands/evaluate_phase_alarm_policy.py',
            'code/commands/evaluate_prefix_forecast_v1.py','experiments/runtime_alarm/configs/forecast_alarm_calibration_v1.json'})
    for name in names:
        dest=out/"source"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    plan={**old,"schema":config["schema"],"parent":str(parent),"parent_plan_sha256":sha(parent/"PLAN.json"),
        "config":config,"partition_by_id":assignment,"ordered_original_train_templates":ordered,
        "source_sha256":{n:sha(out/"source"/n) for n in names},"created_at_utc":now(),
        "partition_counts":{p:sum(v==p for v in assignment.values()) for p in ("fit","calibration","validation")},
        "old_test_read":False,"new_api_calls":0,
        "scope":"Previously exposed development data. Refit on six templates; calibrate on two excluded templates; report four other validation templates. No new held-out or equal-population-FPR claim."}
    write(out/"PLAN.json",plan)
    print(json.dumps({"prepared":str(out),"plan_sha256":sha(out/"PLAN.json"),"counts":plan["partition_counts"]}),flush=True)


def bind_partition(data,plan):
    expected=plan["partition_by_id"]
    if set(expected)!={r["id"] for r in data}: raise ValueError("full original data inventory required")
    partitions={p:[] for p in ("fit","calibration","validation")}
    seen={}
    for row in data:
        p=expected[row["id"]]
        if (row["split"]=="validation")!=(p=="validation"): raise ValueError("parent validation boundary changed")
        if row["template_id"] in seen and seen[row["template_id"]]!=p: raise ValueError("template contamination")
        seen[row["template_id"]]=p; partitions[p].append(row)
    if any(len(partitions[p])!=plan["partition_counts"][p] for p in partitions): raise ValueError("split count changed")
    return partitions


def fit_models(fit,config,out):
    models={}; paths={}; artifacts={}; provenance={"fit_ids":[r["id"] for r in fit],"fit_templates":sorted({r["template_id"] for r in fit}),"models":{}}
    labels=[r["labels"] for r in fit]; dags=[r["early"].event_dag for r in fit]
    for kind,cls in (("event",EliminatingDAGWorkflowRiskBN),("message",MessageHurdleWorkflowRiskBN)):
        for regularization in config["regularization"]:
            start=time.monotonic(); name=f"{kind}_{regularization}"
            if kind=="event": model=fit_event_traces(dags,labels,regularization=regularization,model_class=cls)
            else: model=cls(regularization=regularization).fit_messages(dags,labels)
            models[name]=model; artifacts[name]=model.parameters()
            write(out/"models"/(name+".json"),artifacts[name]); paths[name]=name+".json"
            provenance["models"][name]={"kind":kind,"optimizer":model.optimizer,"fingerprint":digest(artifacts[name])}
            print(json.dumps({"fitted":name,"seconds":time.monotonic()-start,"success":model.optimizer["success"]}),flush=True)
    ys=np.asarray([r["y"] for r in fit]); settings=config["baseline"]
    for name in ("logistic","histgb"):
        estimators={}; files={}
        for phase in config["phases"]:
            x=np.asarray([baseline_features(r,phase) for r in fit])
            estimator=(LogisticRegression(C=settings["logistic_C"],max_iter=2000,tol=1e-10) if name=="logistic" else
                HistGradientBoostingClassifier(max_leaf_nodes=settings["histgb_leaves"],max_iter=settings["histgb_iterations"],
                    l2_regularization=settings["histgb_l2"],random_state=settings["random_state"]))
            estimator.fit(x,ys); estimators[phase]=estimator
            filename=name+"_"+phase+".joblib"; joblib.dump(estimator,out/"models"/filename); files[filename]=sha(out/"models"/filename)
        artifacts[name]={"model":name,"files_sha256":files,"feature_contract":"agentleak-eight-prefix-counts-v1","fitted_flows":len(fit)}
        models[name]=estimators; write(out/"models"/(name+".json"),artifacts[name]); paths[name]=name+".json"
        provenance["models"][name]={"kind":name,"fingerprint":digest(artifacts[name])}
    for name in ("train_prevalence","visible_persistence"):
        artifacts[name]={"model":name,"fit_prevalence":float(ys.mean()),"feature_contract":"agentleak-eight-prefix-counts-v1"}
        models[name]=None; write(out/"models"/(name+".json"),artifacts[name]); paths[name]=name+".json"
        provenance["models"][name]={"kind":name,"fingerprint":digest(artifacts[name])}
    provenance["saved_at_utc"]=now()
    provenance["model_file_sha256"]={p.name:sha(p) for p in sorted((out/"models").iterdir())}
    write(out/"FIT.json",provenance)
    return models,artifacts,provenance


def score_partition(rows,models,artifacts,config,partition,out):
    result={}
    for name,model in models.items():
        start=time.monotonic(); predictions=[]
        arrays={p:np.asarray([baseline_features(r,p) for r in rows]) for p in config["phases"]}
        if name in ("logistic","histgb"):
            probabilities={p:model[p].predict_proba(arrays[p])[:,1] for p in config["phases"]}
        for i,row in enumerate(rows):
            prefixes=[]
            for phase in config["phases"]:
                sample=row[phase]
                if name.startswith(("event_","message_")):
                    p=model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
                elif name in ("logistic","histgb"): p=float(probabilities[phase][i])
                elif name=="train_prevalence": p=artifacts[name]["fit_prevalence"]
                else: p=float(arrays[phase][i,2 if phase=="early" else 4]>0)
                prefixes.append({"phase":phase,"sequence":sample.event_dag.prefix_sequence,"probability":p,"input_sha256":sample.sha256})
            predictions.append({"episode_id":row["id"],"model_fingerprint":digest(artifacts[name]),"target_id":config["target_id"],"prefixes":prefixes})
            if name.startswith(("event_","message_")) and (i+1)%500==0:
                print(json.dumps({"partition":partition,"model":name,"scored":i+1}),flush=True)
        write(out/"predictions"/partition/(name+".json"),predictions); result[name]=predictions
        print(json.dumps({"partition":partition,"model":name,"completed":len(rows),"seconds":time.monotonic()-start}),flush=True)
    return result


def freeze_calibration(rows,predictions,artifacts,config,out):
    ids={r["id"] for r in rows}; labels={r["id"]:r["y"] for r in rows}; negative={i for i in ids if labels[i]==0}
    result={"calibration_ids":sorted(ids),"calibration_negative_ids":sorted(negative),"calibration_templates":sorted({r["template_id"] for r in rows}),"models":{}}
    for name,episodes in predictions.items():
        if {e["episode_id"] for e in episodes}!=ids: raise ValueError("calibration-only score inventory required")
        fingerprint=digest(artifacts[name]); only_negative=[e for e in episodes if e["episode_id"] in negative]
        candidates={}
        for rule in config["policies"]:
            if rule=="fixed_half":
                candidates[rule]={"policy":fixed_policy(config["phases"],dict.fromkeys(config["phases"],.5),fingerprint,config["target_id"]),"calibrated":False}
            else:
                extra={"phase_budgets":config["fixed_phase_budgets"]} if rule=="fixed_phase_budget" else {}
                candidates[rule]=calibrate_forecast(only_negative,phases=config["phases"],model_fingerprint=fingerprint,target_id=config["target_id"],
                    total_budget=config["total_empirical_fpr_budget"],mode=rule,**extra)
        result["models"][name]=candidates
    result["prediction_sha256"]={name+".json":sha(out/"predictions/calibration"/(name+".json")) for name in predictions}
    result["fit_sha256"]=sha(out/"FIT.json"); result["saved_at_utc"]=now()
    write(out/"CALIBRATION.json",result)
    return result


def run(out):
    plan=frozen_plan(out); config=plan["config"]
    data,projection=load_data(out); write(out/"PROJECTION.json",projection)
    parts=bind_partition(data,plan)
    models,artifacts,fit=fit_models(parts["fit"],config,out)
    cal_scores=score_partition(parts["calibration"],models,artifacts,config,"calibration",out)
    calibration=freeze_calibration(parts["calibration"],cal_scores,artifacts,config,out)
    frozen_calibration_sha=sha(out/"CALIBRATION.json")
    write(out/"VALIDATION_START.json",{"calibration_sha256":frozen_calibration_sha,"started_at_utc":now(),"models_frozen":fit["model_file_sha256"]})
    val_scores=score_partition(parts["validation"],models,artifacts,config,"validation",out)
    report={"plan_sha256":sha(out/"PLAN.json"),"calibration_sha256":frozen_calibration_sha,"metrics":{},"phase_probability_metrics":{},"template_metrics":{},
        "partition_counts":{p:{"flows":len(rs),"positive":sum(r["y"] for r in rs),"templates":len({r["template_id"] for r in rs})} for p,rs in parts.items()},
        "new_api_calls":0,"old_test_read":False,"new_default_selected":False,"scope":plan["scope"]}
    for partition,scores in (("calibration",cal_scores),("validation",val_scores)):
        labels={r["id"]:r["y"] for r in parts[partition]}
        templates={r["id"]:r["template_id"] for r in parts[partition]}
        for name,episodes in scores.items():
            key=partition+"/"+name
            report["phase_probability_metrics"][key]={phase:metric([labels[e["episode_id"]] for e in episodes],
                [next(p["probability"] for p in e["prefixes"] if p["phase"]==phase) for e in episodes]) for phase in config["phases"]}
            report["metrics"][key]={}; decisions={}
            for rule,candidate in calibration["models"][name].items():
                summary,actual=forecast_alarm_metrics(episodes,labels,candidate["policy"])
                report["metrics"][key][rule]=summary
                decisions[rule]=[{"episode_id":e["episode_id"],"decisions":d} for e,d in zip(episodes,actual)]
                report["template_metrics"][key+"/"+rule]={t:forecast_alarm_metrics([e for e in episodes if templates[e["episode_id"]]==t],
                    {i:y for i,y in labels.items() if templates[i]==t},candidate["policy"])[0] for t in sorted(set(templates.values()))}
            write(out/"decisions"/partition/(name+".json"),decisions)
    if sha(out/"CALIBRATION.json")!=frozen_calibration_sha: raise ValueError("policy changed during validation")
    for filename,expected in fit["model_file_sha256"].items():
        if sha(out/"models"/filename)!=expected: raise ValueError("model changed during evaluation")
    report["prediction_sha256"]={str(p.relative_to(out)):sha(p) for p in sorted((out/"predictions").rglob("*.json"))}
    report["decision_sha256"]={str(p.relative_to(out)):sha(p) for p in sorted((out/"decisions").rglob("*.json"))}
    report["completed_at_utc"]=now(); write(out/"REPORT.json",report)
    print(json.dumps({"completed":True,"models":len(models),"calibration_flows":len(parts["calibration"]),"validation_flows":len(parts["validation"]),
        "prefix_probabilities":2*len(models)*(len(parts["calibration"])+len(parts["validation"]))}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["prepare","run"]); parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--parent",type=Path); parser.add_argument("--config",type=Path,default=ROOT / 'experiments/runtime_alarm/configs/forecast_alarm_calibration_v1.json')
    args=parser.parse_args()
    if args.mode=="prepare": prepare(args.out.resolve(),args.parent.resolve(),args.config.resolve())
    else: run(args.out.resolve())
