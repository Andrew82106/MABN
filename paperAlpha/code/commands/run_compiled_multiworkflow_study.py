"""Frozen cross-workflow development fit with explicit shared policy targets.

No collection, test-set reading, default replacement or validation tuning.
Two small previously exposed authored API pilots, not a fresh public benchmark.
"""
from pathlib import Path
import argparse
import copy
from itertools import product
import json
import shutil
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, write, sha
from compiled_multiworkflow_data import load_data
from paperalpha_monitor.observable_event_dag import FAMILIES
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import fit_event_traces
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from paperalpha_monitor.contextual_message_bn import ContextualMessageWorkflowRiskBN,PROFILE
from paperalpha_runtime.canonical import digest

CONFIG = {"regularization":[.001,.03],"max_iter":500,"probability_digits":8,"feature_event_capacity":16,
    "message_model":ContextualMessageWorkflowRiskBN.MODEL,"typed_cue_profile":PROFILE,
    "histgb":{"max_leaf_nodes":7,"max_iter":100,"min_samples_leaf":3,"l2_regularization":.1,"random_state":17},
    "selection":"Report every predeclared configuration. No default selected and no threshold tuning.",
    "supervision":"BN: one earliest available complete-future-message likelihood per workflow. Direct classifiers: final two-bit labels, separate phase heads; reviewer intermediate labels are not supplied to them.",
    "alarm":"Fixed .5 descriptive scores only. No calibration split in this small pilot; not a deployment false-positive guarantee."}


def now(): return datetime.now(timezone.utc).isoformat()


def policy_truth(sample):
    """Prefix-only deterministic policy table; unknown current facts disallowed."""
    if len(sample.action_ids)!=2 or sample.authority.facts:
        raise ValueError("this comparison requires two explicit targets and fully observed current policy")
    model=ContextualMessageWorkflowRiskBN()
    result=[]
    for bits in product((0,1),repeat=2):
        p=model.predict(sample,event_conditions=dict(zip(sample.action_ids,bits)))["probabilities"]["system:any"]
        if min(abs(p),abs(p-1))>1e-12: raise ValueError("policy table is not deterministic")
        result.append(int(p>.5))
    return result


def features(sample, truth, capacity=16):
    """Bounded full effective graph, no text IDs, task labels or future outcomes.

    Preserves event family/cues/observed state/target membership, adjacency,
    message membership and literal equivalence. Positional representation can
    be order-sensitive; it does not claim permutation-invariant generalization.
    """
    d=sample.event_dag; n=len(d.order)
    if n>capacity or len(truth)!=4: raise ValueError("explicit feature capacity exceeded; never truncate")
    node=np.zeros((capacity,len(FAMILIES)+6)); adjacency=np.zeros((capacity,capacity))
    same_message=np.zeros_like(adjacency); same_literal=np.zeros_like(adjacency)
    target=[sample.queries[a]["event_id"] for a in sample.action_ids]
    for i,identity in enumerate(d.order):
        e=d.lookup[identity]; node[i,FAMILIES.index(e["family"])]=1
        node[i,5:7]=e["cues"]
        node[i,7]=1 if identity in d.observations else 0
        node[i,8]=d.observations[identity]["value"] if identity in d.observations else 0
        node[i,9:11]=[int(identity==t) for t in target]
        for j,other in enumerate(d.order):
            f=d.lookup[other]
            adjacency[j,i]=int(other in d.parents[identity])
            same_message[i,j]=int((e["node"],e["actor"],e["family"])==(f["node"],f["actor"],f["family"]))
            same_literal[i,j]=int(e["subject"].casefold()==f["subject"].casefold())
    return np.r_[1.,node.ravel(),adjacency.ravel(),same_message.ravel(),same_literal.ravel(),np.asarray(truth,float)]


def prepare(out, branch_run, review_run):
    data,verification=load_data(branch_run,review_run)
    out.mkdir(parents=True,exist_ok=False)
    names=[str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json','code/commands/run_compiled_multiworkflow_study.py','code/commands/compiled_multiworkflow_data.py',
        'code/commands/run_agentleak_event_dag.py','code/commands/inspect_agentleak_release.py','code/commands/harnessaudit_cohort.py',
        'code/commands/scalar_binary_graph.py','code/commands/verify_compiled_multiworkflow_study.py',
        'code/tests/test_branched_message_compiler.py','code/tests/test_compiled_multiworkflow_study.py','code/tests/test_contextual_message_bn.py']
    for name in names:
        dest=out/"source"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    for row in data:
        for phase,sample in row["samples"].items():
            write(out/"compiled"/row["id"].replace(":","_")/(phase+"_graph.json"),sample.event_dag.specification())
            write(out/"compiled"/row["id"].replace(":","_")/(phase+"_manifest.json"),row["manifests"][phase])
    plan={"schema":"compiled-multiworkflow-development-v1","created_at_utc":now(),"config":CONFIG,
        "branch_run":str(branch_run.resolve()),"review_run":str(review_run.resolve()),"verification":verification,
        "partition":{r["id"]:{"split":r["split"],"template_id":r["template_id"],"source_hash":r["source_hash"]} for r in data},
        "source_sha256":{n:sha(out/"source"/n) for n in sorted(names)},
        "compiled_sha256":{p.relative_to(out).as_posix():sha(p) for p in sorted((out/"compiled").rglob("*.json"))},
        "scope":"Existing 20 train/20 validation from two authored API pilots, three task templates per split; not new independent or broad MAS benchmark evidence.",
        "new_api_calls":0,"old_test_read":False}
    write(out/"PLAN.json",plan)
    print(json.dumps({"prepared":str(out),"flows":len(data),"plan_sha256":sha(out/"PLAN.json")}),flush=True)


def load(out):
    plan=read(out/"PLAN.json")
    if ROOT!=(out/"source").resolve() or plan["config"]!=CONFIG: raise ValueError("execute the unchanged frozen source")
    for n,h in plan["source_sha256"].items():
        if sha(ROOT/n)!=h: raise ValueError("frozen source changed: "+n)
    for n,h in plan["compiled_sha256"].items():
        if sha(out/n)!=h: raise ValueError("compiled evidence changed")
    data,verification=load_data(Path(plan["branch_run"]),Path(plan["review_run"]))
    if verification!=plan["verification"]: raise ValueError("source verification changed")
    if {r["id"]:{"split":r["split"],"template_id":r["template_id"],"source_hash":r["source_hash"]} for r in data}!=plan["partition"]:
        raise ValueError("original source split changed")
    for r in data:
        for phase,sample in r["samples"].items():
            if sample.event_dag.specification()!=read(out/"compiled"/r["id"].replace(":","_")/(phase+"_graph.json")):
                raise ValueError("compiled graph replay differs")
    return plan,data


def risk_metrics(rows):
    if not rows: return None
    y=np.asarray([r["risk_label"] for r in rows]); p=np.asarray([r["risk_probability"] for r in rows]); positive=int(y.sum())
    joint=np.asarray([r["joint"] for r in rows]); cls=np.asarray([r["label"] for r in rows])
    return {"n":len(rows),"positives":positive,"brier":float(np.mean((p-y)**2)),
        "auroc":float(roc_auc_score(y,p)) if 0<positive<len(rows) else None,
        "recall_at_half":float(((p>=.5)&(y==1)).sum()/positive) if positive else None,
        "fpr_at_half":float(((p>=.5)&(y==0)).sum()/(len(rows)-positive)) if positive<len(rows) else None,
        "f1_at_half":float(f1_score(y,p>=.5,zero_division=0)),"joint_brier":float(np.mean(((joint-np.eye(4)[cls])**2).sum(axis=1))),
        "joint_nll_clipped_1e12":float(-np.log(np.clip(joint[np.arange(len(rows)),cls],1e-12,1.)).mean())}


def rule_joint(sample,rule):
    d=sample.event_dag; values={}
    for identity in d.order:
        e=d.lookup[identity]
        if identity in d.observations: values[identity]=d.observations[identity]["value"]
        else: values[identity]=int(any(values[p] for p in d.parents[identity]))*int(not e["cues"][1])
    targets=[sample.queries[a]["event_id"] for a in sample.action_ids]
    bits=[0 if rule=="always_absent" else values[t]*int(d.lookup[t]["cues"][0]) if rule=="visible_support_only" else values[t] for t in targets]
    return np.eye(4)[2*bits[0]+bits[1]]


def make_record(row,phase,p,truth):
    p=np.asarray(p,float)
    if p.shape!=(4,) or not np.isfinite(p).all() or np.min(p)<-1e-14 or abs(p.sum()-1)>1e-12:
        raise ValueError("proper complete target joint distribution required")
    raw=float(p@truth); operational=round(raw,CONFIG["probability_digits"])
    label=2*row["bits"][0]+row["bits"][1]
    if truth[label]!=row["risk"]: raise ValueError("independent prefix policy does not match source outcome")
    return {"id":row["id"],"cohort":row["cohort"],"split":row["split"],"template_id":row["template_id"],"phase":phase,
        "sample_sha256":row["samples"][phase].sha256,"joint":p.tolist(),"policy_truth":truth,"label":label,
        "risk_label":row["risk"],"raw_risk_probability":raw,"risk_probability":operational}


def run(out):
    plan,data=load(out)
    if (out/"models").exists(): raise ValueError("do not overwrite partial or completed fits")
    train=[r for r in data if r["split"]=="train"]
    dags=[r["samples"][r["training_phase"]].event_dag for r in train]; labels=[r["labels"] for r in train]
    truth={(r["id"],p):policy_truth(s) for r in data for p,s in r["samples"].items()}
    fitted={}; stored={}
    for kind in ("event","message"):
        for penalty in CONFIG["regularization"]:
            name=f"{kind}_bn_{penalty}"
            if kind=="event": model=fit_event_traces(dags,labels,regularization=penalty,max_iter=CONFIG["max_iter"],model_class=EliminatingDAGWorkflowRiskBN)
            else: model=ContextualMessageWorkflowRiskBN(regularization=penalty).fit_messages(dags,labels,max_iter=CONFIG["max_iter"])
            if not model.optimizer["success"]: raise ValueError("fit did not converge; preserve artifacts and investigate")
            fitted[name]=model; stored[name]=model.parameters(); write(out/"models"/(name+".json"),stored[name])
            print(json.dumps({"fitted":name,"optimizer":model.optimizer}),flush=True)
    # Different prefix heads match the actually available phase, no branch early
    # snapshot is fabricated across an unobserved authority change.
    for penalty in CONFIG["regularization"]:
        name=f"direct_joint_{penalty}"; heads={}
        for phase in ("early","late"):
            subset=[r for r in train if phase in r["samples"]]
            x=np.asarray([features(r["samples"][phase],truth[r["id"],phase]) for r in subset]); y=np.asarray([2*r["bits"][0]+r["bits"][1] for r in subset])
            def objective(flat):
                c=flat.reshape(x.shape[1],4); z=x@c; logs=z-logsumexp(z,axis=1,keepdims=True)
                gradient=x.T@(np.exp(logs)-np.eye(4)[y])/len(y)+2*penalty*c
                return float(-logs[np.arange(len(y)),y].mean()+penalty*np.sum(c*c)),gradient.ravel()
            opt=minimize(objective,np.zeros(x.shape[1]*4),jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-12,"gtol":1e-8})
            heads[phase]={"coef":opt.x.reshape(x.shape[1],4).tolist(),"loss":float(opt.fun),"success":bool(opt.success),"iterations":int(opt.nit),
                "train_prefixes":len(y),"columns":x.shape[1],"input_rank":int(np.linalg.matrix_rank(x))}
            if not opt.success: raise ValueError("direct model did not converge")
        stored[name]=heads; write(out/"models"/(name+".json"),heads)
    trees={}
    for phase in ("early","late"):
        subset=[r for r in train if phase in r["samples"]]
        x=np.asarray([features(r["samples"][phase],truth[r["id"],phase]) for r in subset]); y=np.asarray([2*r["bits"][0]+r["bits"][1] for r in subset])
        trees[phase]=HistGradientBoostingClassifier(**CONFIG["histgb"]).fit(x,y)
        joblib.dump(trees[phase],out/"models"/("histgb_"+phase+".joblib"))
    write(out/"FIT.json",{"saved_at_utc":now(),"train_ids":[r["id"] for r in train],
        "source_plan_sha256":sha(out/"PLAN.json"),"model_sha256":{p.name:sha(p) for p in sorted((out/"models").iterdir())}})
    predictions={}
    for row in data:
        for phase,sample in row["samples"].items():
            table=truth[row["id"],phase]; x=features(sample,table); targets=[sample.queries[a]["event_id"] for a in sample.action_ids]
            for name,model in fitted.items():
                law=model.event_law.conditional(sample.event_dag,targets,model.theta[:model.event_count]); p=np.exp(law["log_probabilities"])
                result=model.predict(sample)["probabilities"]
                if abs(result["system:any"]-p@table)>1e-12: raise ValueError("joint target/policy mismatch")
                record=make_record(row,phase,p,table); record["policy_outputs"]=result
                predictions.setdefault(name,[]).append(record)
                a,b=p[2]+p[3],p[1]+p[3]; independent=[(1-a)*(1-b),(1-a)*b,a*(1-b),a*b]
                predictions.setdefault(name+"_independent",[]).append(make_record(row,phase,independent,table))
            for name,heads in stored.items():
                if name.startswith("direct"):
                    z=x@np.asarray(heads[phase]["coef"]); p=np.exp(z-logsumexp(z))
                    predictions.setdefault(name,[]).append(make_record(row,phase,p,table))
            p=np.zeros(4); p[trees[phase].classes_.astype(int)]=trees[phase].predict_proba(x[None,:])[0]
            predictions.setdefault("histgb",[]).append(make_record(row,phase,p,table))
            for rule in ("always_absent","preserve_unless_removed","visible_support_only"):
                predictions.setdefault(rule,[]).append(make_record(row,phase,rule_joint(sample,rule),table))
    write(out/"PREDICTIONS.json",predictions)
    metrics={}
    for name,rows in predictions.items():
        metrics[name]={}
        for split in ("train","validation"):
            for cohort,phase in (("review","early"),("review","late"),("branch","late")):
                metrics[name][f"{split}/{cohort}/{phase}"]=risk_metrics([r for r in rows if r["split"]==split and r["cohort"]==cohort and r["phase"]==phase])
            for moment in ("first","last"):
                selected=[r for r in rows if r["split"]==split and r["phase"]==("early" if moment=="first" and r["cohort"]=="review" else "late")]
                metrics[name][f"{split}/pooled_{moment}"]=risk_metrics(selected)
    report={"completed_at_utc":now(),"plan_sha256":sha(out/"PLAN.json"),"fit_sha256":sha(out/"FIT.json"),
        "predictions_sha256":sha(out/"PREDICTIONS.json"),"metrics":metrics,"verification":plan["verification"],
        "source_scope":plan["scope"],"scoring_methods":len(predictions),"predictions":sum(map(len,predictions.values())),
        "new_default_selected":False,"new_api_calls":0,"old_test_read":False,
        "supervision":CONFIG["supervision"],"alarm":CONFIG["alarm"],"all_cohorts_share_bn_parameters":True}
    write(out/"REPORT.json",report); print(json.dumps({"completed":True,"methods":len(predictions),"predictions":report["predictions"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("prepare","run")); parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--branch-run",type=Path); parser.add_argument("--review-run",type=Path)
    args=parser.parse_args()
    if args.mode=="prepare": prepare(args.out.resolve(),args.branch_run.resolve(),args.review_run.resolve())
    else: run(args.out.resolve())
