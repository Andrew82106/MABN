"""Independent scalar event/message worlds and metric checks, using frozen data.

The probability reference spells out named coefficients without calling the
production law, design matrices, variable elimination or exported CPTs.
"""
from pathlib import Path
import argparse
import copy
from itertools import product
import json
import math
import shutil
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_compiled_multiworkflow_study import load,read,write,sha,features,risk_metrics,rule_joint,CONFIG
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from paperalpha_monitor.contextual_message_bn import ContextualMessageWorkflowRiskBN


def reference(saved,sample,labels=None):
    d=sample.event_dag; weights=dict(zip(saved["names"],saved["theta"])); groups={}
    for identity in d.order:
        e=d.lookup[identity]; groups.setdefault((e["node"],e["actor"],e["family"]),[]).append(identity)
    observed={k:v["value"] for k,v in d.observations.items()}; free=[k for k in d.order if k not in observed]
    assert not d.missing_past and len(free)<=16
    assert all(set(d.parents[n])<=set(observed) for n in observed)
    targets=[sample.queries[a]["event_id"] for a in sample.action_ids]
    result=np.zeros(4); label_mass=0.
    def sigmoid(z): return 1/(1+math.exp(-z))
    def local(identity,values):
        e=d.lookup[identity]; parent=[values[p] for p in d.parents[identity]]
        x=[1,*e["cues"],sum(parent)/len(parent) if parent else 0,int(len(parent)>1 and all(parent))]
        names=("bias","visible_support","visible_opposition","parent_mean","multiple_parents_all")
        return sigmoid(sum(weights[f"event:{e['family']}:{k}"]*v for k,v in zip(names,x)))
    is_message=saved["model"] in (SemanticArrivalWorkflowRiskBN.MODEL,ContextualMessageWorkflowRiskBN.MODEL)
    for bits in product((0,1),repeat=len(free)):
        values={**observed,**dict(zip(free,bits))}; mass=1.
        for (_,_,family),group in groups.items():
            if all(k in observed for k in group): continue
            assert all(k not in observed for k in group)
            probabilities=[local(k,values) for k in group]
            ys=[values[k] for k in group]
            base=math.prod(p if y else 1-p for p,y in zip(probabilities,ys))
            if is_message:
                parent=list(dict.fromkeys(p for k in group for p in d.parents[k])); pv=[values[p] for p in parent]; n=len(group)
                cues=[d.lookup[k]["cues"] for k in group]
                means=[sum(values[p] for p in d.parents[k])/len(d.parents[k]) if d.parents[k] else 0 for k in group]
                x=[1,math.log1p(n),sum(pv)/len(pv) if pv else 0,int(any(pv)),sum(c[0] for c in cues)/n,sum(c[1] for c in cues)/n,
                    sum(c[0]*m for c,m in zip(cues,means))/n,sum(c[1]*m for c,m in zip(cues,means))/n]
                names=("bias","log_group_size","parent_fraction","any_parent","cue_support_fraction","cue_opposition_fraction","supported_arrival_fraction","opposed_arrival_fraction")
                q=sigmoid(sum(weights[f"message:{family}:{k}"]*v for k,v in zip(names,x)))
                factor=q*base/(1-math.prod(1-p for p in probabilities)) if any(ys) else 1-q
            else: factor=base
            mass*=factor
        result[2*values[targets[0]]+values[targets[1]]]+=mass
        if labels is not None and all(values[k]==v["value"] for k,v in labels.items()): label_mass+=mass
    assert abs(result.sum()-1)<1e-11
    return result,label_mass


def audit(out,destination):
    plan,data=load(out); report=read(out/"REPORT.json"); fit=read(out/"FIT.json"); predictions=read(out/"PREDICTIONS.json")
    assert report["plan_sha256"]==sha(out/"PLAN.json") and report["fit_sha256"]==sha(out/"FIT.json")
    assert report["predictions_sha256"]==sha(out/"PREDICTIONS.json")
    assert plan["created_at_utc"]<=fit["saved_at_utc"]<=report["completed_at_utc"]
    for name,h in fit["model_sha256"].items(): assert sha(out/"models"/name)==h
    lookup={r["id"]:r for r in data}; train=[r for r in data if r["split"]=="train"]
    assert set(fit["train_ids"])=={r["id"] for r in train}
    p_error=objective_error=gradient_error=0.; p_count=gradient_count=0
    # The reference policy uses separately reconstructed source failure flags,
    # not production probability queries or future response labels.
    truth={r["id"]:[int(any(f*b for f,b in zip(r["authority_failure"],bits))) for bits in product((0,1),repeat=2)] for r in data}
    for name,records in predictions.items():
        bn=(name.startswith("message_bn") or name.startswith("event_bn")) and not name.endswith("_independent")
        saved=read(out/"models"/(name+".json")) if bn or name.startswith("direct") else None
        model=(ContextualMessageWorkflowRiskBN if name.startswith("message") else EliminatingDAGWorkflowRiskBN).restore(saved) if bn else None
        for record in records:
            row=lookup[record["id"]]; phase=record["phase"]; sample=row["samples"][phase]; policy=truth[row["id"]]
            assert record["sample_sha256"]==sample.sha256 and record["policy_truth"]==policy
            assert record["label"]==2*row["bits"][0]+row["bits"][1] and record["risk_label"]==policy[record["label"]]
            if bn:
                expected,_=reference(saved,sample)
                assert model.predict(sample)["probabilities"]==record["policy_outputs"]
            elif name.endswith("_independent"):
                peer=next(r for r in predictions[name[:-12]] if r["id"]==record["id"] and r["phase"]==phase)
                p=peer["joint"]; a,b=p[2]+p[3],p[1]+p[3]; expected=np.array([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
            elif name.startswith("direct"):
                z=features(sample,policy)@np.asarray(saved[phase]["coef"]); z-=z.max(); expected=np.exp(z); expected/=expected.sum()
            elif name=="histgb":
                tree=joblib.load(out/"models"/("histgb_"+phase+".joblib")); expected=np.zeros(4)
                expected[tree.classes_.astype(int)]=tree.predict_proba(features(sample,policy)[None,:])[0]
            else: expected=rule_joint(sample,name)
            error=float(np.max(np.abs(expected-record["joint"]))); p_error=max(p_error,error); assert error<1e-11
            raw=float(np.asarray(record["joint"])@policy)
            assert abs(raw-record["raw_risk_probability"])<1e-12 and round(raw,8)==record["risk_probability"]
            p_count+=1
        if bn:
            # Independent full-message likelihood, not just endpoint marginals.
            def objective(theta):
                artifact=copy.deepcopy(saved); artifact["theta"][:model.event_count]=theta.tolist()
                losses=[-math.log(reference(artifact,r["samples"][r["training_phase"]],r["labels"])[1]) for r in train]
                return sum(losses)/len(losses)+saved["regularization"]*float(theta@theta)
            actual=objective(model.theta[:model.event_count]); error=abs(actual-saved["optimizer"]["loss"])
            objective_error=max(objective_error,error); assert error<1e-10
            if name.startswith("message"):
                from paperalpha_monitor.message_hurdle import compile_message_training,message_objective
                compiled=compile_message_training(model.event_law,[r["samples"][r["training_phase"]].event_dag for r in train],[r["labels"] for r in train])
                gradient=message_objective(model.theta[:model.event_count],compiled,model.regularization)[1]
            else:
                from paperalpha_monitor.observable_event_learning import complete_trace_design,complete_trace_objective
                x,y=complete_trace_design(model.event_law,[r["samples"][r["training_phase"]].event_dag for r in train],[r["labels"] for r in train])
                gradient=complete_trace_objective(model.theta[:model.event_count],x,y,len(train),model.regularization)[1]
            for i in range(model.event_count):
                theta=model.theta[:model.event_count]; h=np.zeros(model.event_count); h[i]=1e-5
                numeric=(objective(theta+h)-objective(theta-h))/2e-5
                error=abs(numeric-gradient[i]); gradient_error=max(gradient_error,error); assert error<1e-8; gradient_count+=1
            print(json.dumps({"verified_bn":name}),flush=True)
        for split in ("train","validation"):
            for cohort,phase in (("review","early"),("review","late"),("branch","late")):
                subset=[r for r in records if r["split"]==split and r["cohort"]==cohort and r["phase"]==phase]
                metric=risk_metrics(subset); assert metric==report["metrics"][name][f"{split}/{cohort}/{phase}"]
                assert abs(metric["brier"]-sum((r["risk_probability"]-r["risk_label"])**2 for r in subset)/len(subset))<1e-12
            for moment in ("first","last"):
                subset=[r for r in records if r["split"]==split and r["phase"]==("early" if moment=="first" and r["cohort"]=="review" else "late")]
                assert risk_metrics(subset)==report["metrics"][name][f"{split}/pooled_{moment}"]
    destination.mkdir(parents=True,exist_ok=False); shutil.copyfile(Path(__file__),destination/"AUDITOR.py")
    result={"flows_reconstructed":len(data),"prefixes":sum(len(r["samples"]) for r in data),"probability_vectors":p_count,
        "maximum_joint_probability_error":p_error,"independent_training_objectives":4,"maximum_objective_error":objective_error,
        "training_gradient_directions":gradient_count,"maximum_gradient_error":gradient_error,
        "report_sha256":sha(out/"REPORT.json"),"plan_sha256":sha(out/"PLAN.json"),"auditor_sha256":sha(destination/"AUDITOR.py"),
        "new_api_calls":0,"old_test_read":False,"new_fit":False,"scope":"Numeric/source-binding audit of existing development pilots, not a deployment or semantic-truth validation."}
    write(destination/"REPORT.json",result); print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args(); audit(args.out.resolve(),args.destination.resolve())
