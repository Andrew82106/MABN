"""Frozen finite-choice forecast experiment plus independent numerical checks."""
from pathlib import Path
import argparse
import copy
from itertools import product
import json
import math
import shutil
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from collect_exclusive_purchase import read,write,sha,now
from paperalpha_runtime import exclusive_purchase as study
from paperalpha_runtime.privacy_transfer import delivered_text
from paperalpha_monitor.exclusive_dispatch_bn import ExclusiveDispatchWorkflowRiskBN
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN,WorkflowRiskInput
from scalar_binary_graph import evaluate


def validate_record_order(records):
    if [r["sequence"] for r in records]!=list(range(1,len(records)+1)):
        raise ValueError("event ordering changed")
    if any(a["monotonic_ns"]>b["monotonic_ns"] for a,b in zip(records,records[1:])):
        raise ValueError("event time reversed")
    return sum(a["monotonic_ns"]==b["monotonic_ns"] for a,b in zip(records,records[1:]))


def reconstruct(out):
    plan=read(out/"PLAN.json"); collection=read(out/"COLLECTION.json")
    for n,h in plan["source_sha256"].items():
        if sha(out/"source"/n)!=h: raise ValueError("original collection source changed")
    if plan["cases"]!=study.cases(): raise ValueError("task definitions changed")
    if collection["plan_sha256"]!=sha(out/"PLAN.json"): raise ValueError("collection binding changed")
    rows=[]; failed=[]
    for case in plan["cases"]:
        directory=out/"runs"/case["id"]; report=read(directory/"REPORT.json")
        if sha(directory/"REPORT.json")!=collection["reports_sha256"][case["id"]]: raise ValueError("case report changed")
        for n,h in report["files_sha256"].items():
            if sha(directory/n)!=h: raise ValueError("raw evidence changed")
        if not report["complete"]:
            failed.append({"id":case["id"],"failure":report["failure_type"]}); continue
        records=[read(p) for p in sorted(directory.glob("record_*.json"))]
        validate_record_order(records)
        planner=delivered_text(next(r["response"] for r in records if r["kind"]=="response" and r["stage"]=="planner"))["text"]
        requests=[r for r in records if r["kind"]=="request"]
        if [r["request"] for r in requests]!=[study.request(case,"planner")]+[study.request(case,b["actor"],planner) for b in case["branches"]]:
            raise ValueError("actual delivered API inputs differ")
        pre=read(directory/"PREFIX.json")
        if pre!=next(r for r in records if r["kind"]=="forecast_prefix") or pre["value"]!=study.prefix(case,planner):
            raise ValueError("prefix reconstructed differently")
        if not pre["sequence"]<requests[1]["sequence"]: raise ValueError("buyer future leaked into prefix")
        engine=study.SimulatedDispatchSlots(case); events=[]
        for b in case["branches"]:
            response=next(r for r in records if r["kind"]=="response" and r["stage"]==b["actor"])
            command=study.parse_buyer(case,b["actor"],delivered_text(response["response"])["text"])
            parsed=next(r for r in records if r["kind"]=="parsed_proposal" and r["actor"]==b["actor"])
            event=next(r for r in records if r["kind"]=="runtime_event" and r["event"]["actor"]==b["actor"])
            if parsed["command"]!=command or not response["sequence"]<parsed["sequence"]<event["sequence"]:
                raise ValueError("proposal and actual simulated dispatch order differ")
            actual=engine.execute(b["actor"],command)
            if actual!=event["event"]: raise ValueError("dispatch replay differs")
            events.append(actual)
        label=study.outcome(case,events)
        if label!=read(directory/"OUTCOME.json"): raise ValueError("source outcome differs")
        sample=study.compile_input(pre["value"])
        rows.append({"case":case,"sample":sample,"label":label,"prefix_sha256":sha(directory/"PREFIX.json")})
    return plan,rows,failed


def plain(s):
    return WorkflowRiskInput(workflow=s.workflow,authority=s.authority,proposals=s.proposals,
        contracts=s.contracts,combinations=s.combinations,cutoff=s.cutoff,horizon=s.horizon)


def local_x(sample):
    # Independent spelling of the payment-family features, not production _design.
    return np.asarray([[1.,float(p["visible"]=="support"),float(p["visible"]=="opposition")] for p in sample.proposals])


def standalone_softmax_objective(theta,train,penalty):
    loss=0.; gradient=np.zeros(3)
    for row in train:
        s=row["sample"]; x=local_x(s)
        for group in s.choice_groups:
            indices=[s.action_ids.index(a) for a in group["action_ids"]]
            design=np.vstack([np.zeros(3),x[indices]])
            z=design@theta; logp=z-logsumexp(z)
            chosen=[j+1 for j,a in enumerate(group["action_ids"]) if row["label"]["dispatch"][a]]
            if len(chosen)>1: raise ValueError("invalid mutually exclusive label")
            k=chosen[0] if chosen else 0
            loss-=logp[k]; gradient+=(np.exp(logp)@design-design[k])
    return float(loss/len(train)+penalty*(theta@theta)),gradient/len(train)+2*penalty*theta


def scalar_choice_worlds(sample,theta):
    # Scalar group softmax, independent of BN _design/_occurrence_worlds.
    worlds=list(product((0,1),repeat=len(sample.action_ids))); probabilities=[]
    weights=[]
    for group in sample.choice_groups:
        logits=[0.]
        for aid in group["action_ids"]:
            p=sample.proposals[sample.action_ids.index(aid)]
            z=theta[0]+(theta[1] if p["visible"]=="support" else theta[2] if p["visible"]=="opposition" else 0.)
            logits.append(float(z))
        shifted=[math.exp(z-max(logits)) for z in logits]; total=sum(shifted)
        weights.append([z/total for z in shifted])
    for bits in worlds:
        mass=1.
        for group,p in zip(sample.choice_groups,weights):
            chosen=[j+1 for j,a in enumerate(group["action_ids"]) if bits[sample.action_ids.index(a)]]
            if len(chosen)>1: mass=0.; break
            mass*=p[chosen[0] if chosen else 0]
        probabilities.append(mass)
    return np.asarray(worlds),np.asarray(probabilities)


def policy_truth(row):
    # Independent original treasury policy for every (even infeasible) bit vector.
    from paperalpha_runtime.mas_purchase_study import packet,reference_verdict
    case=row["case"]; actions=row["sample"].proposals; truth=[]
    for bits in product((0,1),repeat=len(actions)):
        chosen=[p["binding"] for p,bit in zip(actions,bits) if bit]
        truth.append(int(not reference_verdict(packet(case,chosen))["allowed"]))
    return np.asarray(truth)


def metrics(records,split):
    chosen=[r for r in records if r["split"]==split]
    if not chosen: return None
    p=np.asarray([r["risk_probability"] for r in chosen]); y=np.asarray([r["risk_label"] for r in chosen]); positives=int(y.sum())
    return {"n":len(chosen),"positives":positives,"brier":float(np.mean((p-y)**2)),
        "tp":int(((p>=.5)&(y==1)).sum()),"fp":int(((p>=.5)&(y==0)).sum()),
        "fn":int(((p<.5)&(y==1)).sum()),"tn":int(((p<.5)&(y==0)).sum()),
        "mean_invalid_choice_mass":float(np.mean([r["invalid_choice_mass"] for r in chosen]))}


def prepare_evaluation(source,destination):
    plan,rows,failed=reconstruct(source)
    destination.mkdir(parents=True,exist_ok=False)
    for name in plan["source_sha256"]:
        # Only the evaluator and its explicit regression test may differ.
        if sha(ROOT/name)!=plan["source_sha256"][name] and name not in {
                'code/commands/evaluate_exclusive_purchase.py','code/tests/test_exclusive_dispatch_bn.py'}:
            raise ValueError("model, collector or task changed since original freeze: "+name)
        target=destination/"source"/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target)
    evaluation_plan={"schema":"exclusive-purchase-evaluation-v1","created_at_utc":now(),"source_run":str(source),
        "collection_sha256":sha(source/"COLLECTION.json"),"collection_plan_sha256":sha(source/"PLAN.json"),
        "source_sha256":{n:sha(destination/"source"/n) for n in plan["source_sha256"]},"fit":plan["fit"],
        "reason":"Original strict-increase monotonic-time check incorrectly rejected equal timestamps. Require consecutive event sequence plus nondecreasing clock; raw records, tasks, models and fit settings unchanged.",
        "complete_flows":len(rows),"failed_flows":len(failed),"new_api_calls":0}
    write(destination/"EVAL_PLAN.json",evaluation_plan)
    print(json.dumps({"prepared_evaluation":str(destination),"plan_sha256":sha(destination/"EVAL_PLAN.json")}),flush=True)


def run(destination):
    frozen=read(destination/"EVAL_PLAN.json"); out=Path(frozen["source_run"])
    if ROOT!=(destination/"source").resolve(): raise ValueError("run the corrected frozen evaluator")
    for n,h in frozen["source_sha256"].items():
        if sha(ROOT/n)!=h: raise ValueError("evaluation source changed")
    if sha(out/"COLLECTION.json")!=frozen["collection_sha256"] or sha(out/"PLAN.json")!=frozen["collection_plan_sha256"]:
        raise ValueError("collection changed after evaluation freeze")
    plan,rows,failed=reconstruct(out)
    if plan["fit"]!=frozen["fit"]: raise ValueError("predeclared model settings changed")
    if (destination/"FIT.json").exists() or (destination/"models").exists(): raise ValueError("do not overwrite an evaluation")
    train=[r for r in rows if r["case"]["split"]=="train"]
    if not train: raise ValueError("no completed training flows; no default fabricated")
    models={}; softmax={}; audits={"graphs":0,"max_graph_error":0.,"max_scalar_probability_error":0.,"gradient_checks":0,"max_gradient_error":0.}
    for penalty in plan["fit"]["regularization"]:
        for kind,cls in (("exclusive",ExclusiveDispatchWorkflowRiskBN),("independent_ablation",WorkflowRiskBN)):
            samples=[r["sample"] if kind=="exclusive" else plain(r["sample"]) for r in train]
            labels=[{"dispatch":r["label"]["dispatch"]} for r in train]
            model=cls(regularization=penalty).fit(samples,labels,max_iter=plan["fit"]["max_iter"])
            name=f"{kind}_{penalty}"; models[name]=model; write(destination/"models"/(name+".json"),model.parameters())
        result=minimize(lambda t:standalone_softmax_objective(t,train,penalty),np.zeros(3),jac=True,method="L-BFGS-B",
            bounds=[(-12.,12.)]*3,options={"maxiter":500,"ftol":1e-12,"gtol":1e-8})
        if not result.success: raise ValueError("standalone softmax did not converge")
        softmax[f"standalone_softmax_{penalty}"]=result.x
        write(destination/"models"/f"standalone_softmax_{penalty}.json",{"theta":result.x.tolist(),"loss":float(result.fun),"success":bool(result.success)})
    write(destination/"FIT.json",{"at_utc":now(),"train_ids":[r["case"]["id"] for r in train],
        "model_sha256":{p.name:sha(p) for p in sorted((destination/"models").iterdir())}})
    predictions={}; equivalence={}
    for row in rows:
        s=row["sample"]; bits=np.asarray(list(product((0,1),repeat=len(s.action_ids)))); truth=policy_truth(row)
        invalid=np.zeros(len(bits),bool)
        for g in s.choice_groups: invalid|=bits[:,[s.action_ids.index(a) for a in g["action_ids"]]].sum(axis=1)>1
        def save(name,p):
            if abs(p.sum()-1)>1e-11 or min(p)<0: raise ValueError("invalid forecast distribution")
            raw=float(p@truth); label=row["label"]["risk"]
            predictions.setdefault(name,[]).append({"id":row["case"]["id"],"split":row["case"]["split"],"domain":row["case"]["domain"],
                "sample_sha256":s.sha256,"prefix_sha256":row["prefix_sha256"],"joint":p.tolist(),"policy_truth":truth.tolist(),
                "risk_probability":round(raw,8),"raw_risk_probability":raw,"risk_label":label,"invalid_choice_mass":float(p[invalid].sum())})
        for name,model in models.items():
            inp=s if name.startswith("exclusive") else plain(s)
            worlds=model._worlds(inp,model.theta); p=np.exp(worlds["log_mass"]).sum(axis=1)
            result=model.predict(inp,derivatives=True); save(name,p)
            assert abs(result["probabilities"]["system:any"]-p@truth)<1e-12
            if name.startswith("exclusive"):
                _,expected=scalar_choice_worlds(s,model.theta[:3]); error=float(np.max(abs(p-expected)))
                audits["max_scalar_probability_error"]=max(audits["max_scalar_probability_error"],error); assert error<1e-12
                for query,probability in result["probabilities"].items():
                    error=abs(evaluate(model.probability_graph(s,output=query))-probability)
                    audits["graphs"]+=1; audits["max_graph_error"]=max(audits["max_graph_error"],error); assert error<1e-12
                independent=np.prod(np.where(bits,p@bits,1-p@bits),axis=1)
                save(name+"_same_marginals_independent",independent)
        for name,t in softmax.items(): save(name,scalar_choice_worlds(s,t)[1])
        # Strong, predeclared deterministic priority rule; balanced defaults to basic.
        chosen=[0]*len(s.action_ids)
        priority=study.parse_planner(read(out/"runs"/row["case"]["id"]/"PREFIX.json")["value"]["actual_planner_text"])["priority"]
        for g in s.choice_groups:
            candidates=[a for a in g["action_ids"] if a.endswith("enhanced" if priority=="quality" else "basic")]
            if len(candidates)!=1: raise ValueError("predeclared basic/enhanced rule requires complete catalogue")
            chosen[s.action_ids.index(candidates[0])]=1
        p=np.zeros(len(bits)); p[sum(v*2**(len(chosen)-i-1) for i,v in enumerate(chosen))]=1; save("priority_rule",p)
    for penalty in plan["fit"]["regularization"]:
        model=models[f"exclusive_{penalty}"]; t=softmax[f"standalone_softmax_{penalty}"]
        equivalence[str(penalty)]={"coefficient_difference":float(max(abs(model.theta[:3]-t))),
            "objective_difference":abs(model.loss_gradient(model.theta,[r["sample"] for r in train],[{"dispatch":r["label"]["dispatch"]} for r in train])[0]-standalone_softmax_objective(model.theta[:3],train,penalty)[0])}
        probe=model.theta.copy()+np.random.default_rng(141).normal(0,.1,len(model.theta))
        analytic=model.loss_gradient(probe,[r["sample"] for r in train],[{"dispatch":r["label"]["dispatch"]} for r in train])[1]
        numeric=approx_derivative(lambda full:standalone_softmax_objective(full[:3],train,penalty)[0]+penalty*float(full[3:]@full[3:]),probe).ravel()
        err=float(max(abs(numeric-analytic))); assert err<1e-8
        audits["gradient_checks"]+=len(probe); audits["max_gradient_error"]=max(audits["max_gradient_error"],err)
    write(destination/"PREDICTIONS.json",predictions)
    report={"completed_at_utc":now(),"evaluation_plan_sha256":sha(destination/"EVAL_PLAN.json"),"collection_sha256":sha(out/"COLLECTION.json"),"fit_sha256":sha(destination/"FIT.json"),
        "predictions_sha256":sha(destination/"PREDICTIONS.json"),"methods":len(predictions),"predictions":sum(map(len,predictions.values())),
        "metrics":{n:{s:metrics(r,s) for s in ("train","validation")} for n,r in predictions.items()},"independent_audit":audits,
        "softmax_equivalence":equivalence,"failed_flows":failed,"all_planned_flows":len(plan["cases"]),
        "scope":"New small authored API development cohort; offline future-dispatch forecast, not native-MAS superiority or a new distribution.",
        "limits":["Only payment preference coefficients have supervision; other action families and authority priors remain untrained.",
            "Current planning reply is observed before prediction; shared future planner uncertainty is not included.",
            "Independent-dispatch methods deliberately ignore runtime exclusivity and are structural ablations, not strong valid-choice competitors.",
            "Standalone softmax shares the same conditional-choice hypothesis; priority rule is retained regardless of score.",
            "No held-out calibration split; .5 is descriptive. The BN does not use raw work_order nuance beyond the parsed priority."],
        "new_default_selected":False,"old_test_read":False}
    write(destination/"REPORT.json",report); print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("prepare","run")); p.add_argument("--out",type=Path,required=True); p.add_argument("--source-run",type=Path)
    args=p.parse_args()
    if args.mode=="prepare": prepare_evaluation(args.source_run.resolve(),args.out.resolve())
    else: run(args.out.resolve())
