"""Independent probability/threshold/first-alert audit; never refit a model."""
from pathlib import Path
import argparse
import json
import math
import shutil
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read,write,sha,load_data,frozen_plan,baseline_features,metric
from run_forecast_alarm_study import bind_partition,split_templates
from verify_agentleak_event_dag import independent_fields
from verify_message_hurdle import independent as message_reference
from paperalpha_runtime.canonical import digest
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN


def independent_threshold(values,budget):
    if not values: raise ValueError("nonempty calibration negatives required")
    permitted=math.floor(len(values)*budget)
    candidates=sorted({0.,math.nextafter(1.,math.inf),*[math.nextafter(v,math.inf) for v in values]})
    return next(c for c in candidates if sum(v>=c for v in values)<=permitted)


def audit_decisions(episodes,labels,policy,stored):
    assert [r["episode_id"] for r in stored]==[e["episode_id"] for e in episodes]
    firsts=[]; compared=0
    for episode,record in zip(episodes,stored):
        first=None
        assert len(episode["prefixes"])==len(record["decisions"])
        for prefix,decision in zip(episode["prefixes"],record["decisions"]):
            threshold=policy["thresholds"][prefix["phase"]]; exceeded=prefix["probability"]>=threshold
            emit=exceeded and first is None
            if emit: first={k:prefix[k] for k in ("phase","sequence","input_sha256","probability")}
            assert decision=={"phase":prefix["phase"],"sequence":prefix["sequence"],"probability":prefix["probability"],
                "threshold":threshold,"threshold_exceeded":exceeded,"alert_issued":emit,"ever_alerted":first is not None,
                "first_alert":first,"input_sha256":prefix["input_sha256"],"policy_sha256":digest(policy)}
            compared+=1
        firsts.append(first)
    pos=[i for i,e in enumerate(episodes) if labels[e["episode_id"]]==1]; neg=[i for i,e in enumerate(episodes) if labels[e["episode_id"]]==0]
    tp=sum(firsts[i] is not None for i in pos); fp=sum(firsts[i] is not None for i in neg)
    result={"positive_episode_n":len(pos),"negative_episode_n":len(neg),"true_positive":tp,"false_positive":fp,
        "any_alarm_recall":tp/len(pos) if pos else None,"any_alarm_fpr":fp/len(neg) if neg else None,
        "precision":tp/(tp+fp) if tp+fp else None,"f1":2*tp/(len(pos)+tp+fp) if len(pos)+tp+fp else None,
        "first_phase_recall":sum(firsts[i] is not None and firsts[i]["phase"]==policy["phases"][0] for i in pos)/len(pos) if pos else None,
        "first_alert_phase_counts":{p:sum(f is not None and f["phase"]==p for f in firsts) for p in policy["phases"]},
        "observed_phase_episode_counts":{p:sum(any(s["phase"]==p for s in e["prefixes"]) for e in episodes) for p in policy["phases"]},
        "wall_clock_lead_time":None}
    return result,compared


def run(out,destination):
    plan=frozen_plan(out); config=plan["config"]; report=read(out/"REPORT.json"); fit=read(out/"FIT.json"); calibration=read(out/"CALIBRATION.json")
    assert report["plan_sha256"]==sha(out/"PLAN.json") and report["calibration_sha256"]==sha(out/"CALIBRATION.json")
    start=read(out/"VALIDATION_START.json")
    assert start["calibration_sha256"]==sha(out/"CALIBRATION.json") and fit["saved_at_utc"]<=calibration["saved_at_utc"]<=start["started_at_utc"]<=report["completed_at_utc"]
    for name,expected in fit["model_file_sha256"].items(): assert sha(out/"models"/name)==expected
    for key in ("prediction_sha256","decision_sha256"):
        for name,expected in report[key].items(): assert sha(out/name)==expected
    data,projection=load_data(out,verbose=False); assert projection==read(out/"PROJECTION.json")
    parts=bind_partition(data,plan); lookup={r["id"]:r for r in data}
    assignment,ordered=split_templates(read(out/"inputs/train.json"),config)
    assert ordered==plan["ordered_original_train_templates"]
    assert all(plan["partition_by_id"][i]==p for i,p in assignment.items())
    fit_ids={r["id"] for r in parts["fit"]}; cal_ids={r["id"] for r in parts["calibration"]}; val_ids={r["id"] for r in parts["validation"]}
    assert fit_ids==set(fit["fit_ids"]) and cal_ids==set(calibration["calibration_ids"]) and not fit_ids&cal_ids and not (fit_ids|cal_ids)&val_ids
    negatives={r["id"] for r in parts["calibration"] if r["y"]==0}
    assert negatives==set(calibration["calibration_negative_ids"])
    error,count,decisions_count,thresholds=0.,0,0,0
    for name,meta in fit["models"].items():
        saved=read(out/"models"/(name+".json")); assert digest(saved)==meta["fingerprint"]
        event=EliminatingDAGWorkflowRiskBN.restore(saved) if name.startswith("event_") else None
        classifiers={phase:joblib.load(out/"models"/(name+"_"+phase+".joblib")) for phase in config["phases"]} if name in ("logistic","histgb") else None
        cal_episodes=read(out/"predictions/calibration"/(name+".json"))
        neg_episodes=[e for e in cal_episodes if e["episode_id"] in negatives]
        for rule,entry in calibration["models"][name].items():
            policy=entry["policy"]
            assert policy["model_fingerprint"]==meta["fingerprint"] and policy["target_id"]==config["target_id"]
            for phase in config["phases"]:
                values=[max(p["probability"] for p in e["prefixes"] if rule=="episode_max" or p["phase"]==phase) for e in neg_episodes]
                expected=.5 if rule=="fixed_half" else independent_threshold(values,config["total_empirical_fpr_budget"] if rule=="episode_max" else config["fixed_phase_budgets"][phase])
                assert policy["thresholds"][phase]==expected; thresholds+=1
            if rule!="fixed_half": assert set(entry["calibration_negative_ids"])==negatives
        for partition in ("calibration","validation"):
            episodes=read(out/"predictions"/partition/(name+".json")); stored=read(out/"decisions"/partition/(name+".json"))
            assert {e["episode_id"] for e in episodes}==(cal_ids if partition=="calibration" else val_ids)
            if classifiers:
                batch_p={p:classifiers[p].predict_proba(np.array([baseline_features(lookup[e["episode_id"]],p) for e in episodes]))[:,1] for p in config["phases"]}
            for i,e in enumerate(episodes):
                row=lookup[e["episode_id"]]
                assert e["model_fingerprint"]==meta["fingerprint"] and e["target_id"]==config["target_id"]
                assert [p["phase"] for p in e["prefixes"]]==config["phases"]
                for prefix in e["prefixes"]:
                    phase=prefix["phase"]; sample=row[phase]
                    assert prefix["sequence"]==sample.event_dag.prefix_sequence and prefix["input_sha256"]==sample.sha256
                    if event: expected=independent_fields(event,sample,phase)[1]
                    elif name.startswith("message_"): expected=message_reference(saved,sample,phase)[0]
                    elif classifiers: expected=float(batch_p[phase][i])
                    elif name=="train_prevalence": expected=sum(r["y"] for r in parts["fit"])/len(parts["fit"])
                    else: expected=float(baseline_features(row,phase)[2 if phase=="early" else 4]>0)
                    error=max(error,abs(expected-prefix["probability"])); count+=1
            labels={r["id"]:r["y"] for r in parts[partition]}
            for phase in config["phases"]:
                result=metric([labels[e["episode_id"]] for e in episodes],[next(p["probability"] for p in e["prefixes"] if p["phase"]==phase) for e in episodes])
                assert result==report["phase_probability_metrics"][partition+"/"+name][phase]
            for rule,entry in calibration["models"][name].items():
                summary,compared=audit_decisions(episodes,labels,entry["policy"],stored[rule]); decisions_count+=compared
                assert summary==report["metrics"][partition+"/"+name][rule]
                if partition=="calibration" and rule!="fixed_half":
                    assert summary["any_alarm_fpr"]<=config["total_empirical_fpr_budget"]+1e-15
    assert error<1e-10
    destination.mkdir(parents=True,exist_ok=False); shutil.copyfile(Path(__file__),destination/"AUDITOR.py")
    result={"reconstructed_flows":len(data),"partition_counts":plan["partition_counts"],"probabilities_recomputed":count,
        "maximum_probability_error":error,"thresholds_recomputed":thresholds,"stream_decisions_recomputed":decisions_count,
        "all_calibration_budgets_met":True,"validation_fpr_guaranteed":False,"new_fit":False,"new_api_calls":0,"old_test_read":False,
        "plan_sha256":sha(out/"PLAN.json"),"report_sha256":sha(out/"REPORT.json"),"auditor_sha256":sha(destination/"AUDITOR.py")}
    write(destination/"REPORT.json",result); print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args(); run(args.out.resolve(),args.destination.resolve())
