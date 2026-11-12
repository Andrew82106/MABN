"""Independent feature construction, world sums and GLM objectives.

Reuses raw-data contract validation and sklearn's saved tree evaluator, not the
new baseline's feature builder, probability wrapper or joint inference code.
"""
from pathlib import Path
import argparse
from itertools import product
import json
import math
import shutil
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_matched_event_controls import load,read,write,sha,summarise


def reference_features(event, parent_values, relations, *, soft=False):
    families=["prepare","work","review","publish","dispatch"]
    values=[1.,float(event["cues"][0]),float(event["cues"][1]),0.,0.]
    if relations and parent_values:
        values[3]=sum(parent_values)/len(parent_values)
        if len(parent_values)>1:
            values[4]=math.prod(parent_values) if soft else float(all(parent_values))
    result=[0.]*25; start=5*families.index(event["family"])
    result[start:start+5]=values
    return result


def reference_joint(saved,dag,targets,mode,tree=None):
    fixed={k:v["value"] for k,v in dag.observations.items()}
    free=[k for k in dag.order if k not in fixed]
    assert not dag.missing_past and len(free)<=18
    assert all(set(dag.parents[k])<=set(fixed) for k in fixed)
    cache={}
    def probability(n,values,soft=False):
        x=reference_features(dag.lookup[n],[values[p] for p in dag.parents[n]],saved["relations"],soft=soft)
        key=tuple(x)
        if key not in cache:
            if saved["kind"]=="glm":
                z=sum(a*b for a,b in zip(x,saved["coef"]))
                cache[key]=1/(1+math.exp(-z))
            else:
                q=tree.predict_proba(np.asarray([x]))[0]
                cache[key]=sum(float(p) for p,label in zip(q,tree.classes_) if label==1)
        return cache[key]
    worlds=list(product((0,1),repeat=len(targets)))
    if mode=="mean_field":
        values=dict(fixed)
        for n in free: values[n]=probability(n,values,soft=True)
        return np.asarray([math.prod(values[n] if bit else 1-values[n] for n,bit in zip(targets,bits)) for bits in worlds])
    result=np.zeros(len(worlds))
    for bits in product((0,1),repeat=len(free)):
        values={**fixed,**dict(zip(free,bits))}; mass=1.
        for n in free:
            p=probability(n,values); mass*=p if values[n] else 1-p
        index=sum(values[n]*2**(len(targets)-j-1) for j,n in enumerate(targets))
        result[index]+=mass
    assert abs(result.sum()-1)<1e-11
    if mode=="same_marginals_independent":
        marginals=[sum(p for p,bits in zip(result,worlds) if bits[j]) for j in range(len(targets))]
        result=np.asarray([math.prod(p if bit else 1-p for p,bit in zip(marginals,bits)) for bits in worlds])
    return result


def audit(out,destination):
    plan,data=load(out); fit=read(out/"FIT.json"); report=read(out/"REPORT.json"); predictions=read(out/"PREDICTIONS.json")
    assert report["plan_sha256"]==sha(out/"PLAN.json") and report["fit_sha256"]==sha(out/"FIT.json")
    assert report["predictions_sha256"]==sha(out/"PREDICTIONS.json")
    assert plan["created_at_utc"]<=fit["created_at_utc"]<=report["completed_at_utc"]
    for n,h in fit["model_sha256"].items(): assert sha(out/"models"/n)==h
    train=[r for r in data if r["split"]=="train"]; lookup={r["id"]:r for r in data}
    assert fit["train_ids"]==[r["id"] for r in train]
    assert fit["train_graph_hashes"]==[r["samples"][r["training_phase"]].event_dag.sha256 for r in train]
    count=0; probability_error=objective_error=gradient_error=0.; objectives=gradients=0
    models={}
    for name in fit["model_sha256"]:
        if name.endswith(".json"):
            identity=name[:-5]; saved=read(out/"models"/name)
            tree=joblib.load(out/"models"/(identity+".joblib")) if saved["kind"]=="histgb" else None
            models[identity]=(saved,tree)
    for name,records in predictions.items():
        identity,mode=name.split("/"); saved,tree=models[identity]
        assert len(records)==sum(len(r["samples"]) for r in data)
        assert len({(r["id"],r["phase"]) for r in records})==len(records)
        for record in records:
            row=lookup[record["id"]]; sample=row["samples"][record["phase"]]
            assert record["sample_sha256"]==sample.sha256
            targets=[sample.queries[a]["event_id"] for a in sample.action_ids]
            expected=reference_joint(saved,sample.event_dag,targets,mode,tree)
            error=float(np.max(np.abs(expected-record["joint"]))); probability_error=max(probability_error,error); assert error<1e-11
            policy=[int(any(f*b for f,b in zip(row["authority_failure"],bits))) for bits in product((0,1),repeat=2)]
            assert policy==record["policy_truth"] and record["label"]==2*row["bits"][0]+row["bits"][1]
            assert record["risk_label"]==policy[record["label"]]==row["risk"]
            raw=sum(p*t for p,t in zip(record["joint"],policy))
            assert abs(raw-record["raw_risk_probability"])<1e-12 and round(raw,8)==record["risk_probability"]
            count+=1
    for identity,(saved,tree) in models.items():
        rows=[]; labels=[]
        for row in train:
            d=row["samples"][row["training_phase"]].event_dag
            values={k:v["value"] for k,v in {**d.observations,**row["labels"]}.items()}
            for n in d.order:
                if n not in d.observations:
                    rows.append(reference_features(d.lookup[n],[values[p] for p in d.parents[n]],saved["relations"]))
                    labels.append(values[n])
        assert len(labels)==saved["fit_report"]["label_count"]==58
        if saved["kind"]!="glm": continue
        def objective(theta):
            terms=[]
            for x,y in zip(rows,labels):
                z=sum(a*b for a,b in zip(theta,x)); terms.append(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)
            return sum(terms)/len(train)+saved["penalty"]*sum(t*t for t in theta)
        t=np.asarray(saved["coef"]); observed=objective(t); err=abs(observed-saved["fit_report"]["loss"])
        objective_error=max(objective_error,err); assert err<1e-10; objectives+=1
        from paperalpha_monitor.structured_event_baseline import glm_objective
        actual=glm_objective(t,np.asarray(rows),np.asarray(labels),len(train),saved["penalty"])[1]
        for i in range(25):
            h=np.zeros(25); h[i]=1e-5; numeric=(objective(t+h)-objective(t-h))/2e-5
            err=abs(numeric-actual[i]); gradient_error=max(gradient_error,err); assert err<1e-8; gradients+=1
    assert summarise(predictions)==report["metrics"]
    # Independently recompute the central squared-error statistic.
    for name,records in predictions.items():
        selected=[r for r in records if r["split"]=="validation" and r["phase"]==("early" if r["cohort"]=="review" else "late")]
        brier=sum((r["risk_probability"]-r["risk_label"])**2 for r in selected)/len(selected)
        assert abs(brier-report["metrics"][name]["validation/pooled_first"]["brier"])<1e-12
    destination.mkdir(parents=True,exist_ok=False); shutil.copyfile(Path(__file__),destination/"AUDITOR.py")
    result={"flows":len(data),"probability_vectors":count,"max_joint_error":probability_error,
        "independent_glm_objectives":objectives,"max_objective_error":objective_error,
        "gradient_directions":gradients,"max_gradient_error":gradient_error,
        "plan_sha256":sha(out/"PLAN.json"),"report_sha256":sha(out/"REPORT.json"),"auditor_sha256":sha(destination/"AUDITOR.py"),
        "new_api_calls":0,"new_fit":False,"old_test_read":False,
        "scope":"Independent features/world sums/GLM objectives; sklearn tree prediction and summary helper reused. Not semantic or out-of-sample validation."}
    write(destination/"REPORT.json",result); print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args(); audit(args.out.resolve(),args.destination.resolve())
