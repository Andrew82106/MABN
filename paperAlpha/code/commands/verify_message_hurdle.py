"""Independent count-polynomial audit of the public message-hurdle study.

The reference calculation uses scalar named coefficients and Bernoulli count
convolution, not the production hurdle kernel or group-factor VE. It is exact
for the explicitly checked pinned workflow, not a generic MAS independence rule.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys

import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read,write,sha,load_data,frozen_plan,metric
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN
from scalar_binary_graph import evaluate


def independent(saved,sample,phase):
    t=dict(zip(saved["names"],saved["theta"])); dag=sample.event_dag
    assert dag.workflow["nodes"]==["vault","prepare","work","publish"]
    assert dag.workflow["edges"]==[["vault","prepare"],["vault","work"],["prepare","work"],["work","publish"]]
    assert all(e["cues"]==[0,0] for e in dag.events)
    n=len(sample.action_ids)
    v=np.array([dag.observations[f"value-{j}:vault"]["value"] for j in range(n)])
    c=np.array([dag.observations[f"value-{j}:prepare"]["value"] for j in range(n)])
    k=np.arange(n+1)
    qf=expit(t["message:publish:bias"]+t["message:publish:log_group_size"]*np.log1p(n)
        +t["message:publish:parent_fraction"]*k/n+t["message:publish:any_parent"]*(k>0))
    if phase=="early":
        pw=expit(t["event:work:bias"]+t["event:work:parent_mean"]*(v+c)/2+t["event:work:multiple_parents_all"]*v*c)
        qw=float(expit(t["message:work:bias"]+t["message:work:log_group_size"]*np.log1p(n)
            +t["message:work:parent_fraction"]*(v.sum()+c.sum())/(2*n)+t["message:work:any_parent"]*bool(v.any() or c.any())))
        counts=np.array([1.])
        for p in pw: counts=np.convolve(counts,[1-p,p])
        counts[1:]*=qw/counts[1:].sum(); counts[0]=1-qw
        return float(counts@qf),None
    w=np.array([dag.observations[f"value-{j}:work"]["value"] for j in range(n)])
    pf=expit(t["event:publish:bias"]+t["event:publish:parent_mean"]*w)
    q=float(qf[int(w.sum())]); z=-np.expm1(np.log1p(-pf).sum())
    return q,q*pf/z


def run(out,destination):
    plan=frozen_plan(out); report=read(out/"REPORT.json")
    assert sha(out/"PLAN.json")==report["plan_sha256"]
    assert sha(out/"REFERENCE_REPORT.json")==plan["reference_report_sha256"]
    for k,v in report["model_sha256"].items(): assert sha(out/"models"/k)==v
    for k,v in report["prediction_sha256"].items(): assert sha(out/"predictions"/k)==v
    data,projection=load_data(out,verbose=False)
    assert projection==read(out/"PROJECTION.json")
    by_id={r["id"]:r for r in data}
    maximum,field_error,grad_error,count,gradient_count=0.,0.,0.,0,0
    for name in report["prediction_sha256"]:
        saved=read(out/"models"/name); records=read(out/"predictions"/name)
        model=MessageHurdleWorkflowRiskBN.restore(saved)
        for i,r in enumerate(records):
            row=by_id[r["id"]]; sample=row[r["phase"]]
            assert r["input_sha256"]==sample.sha256 and r["y"]==row["y"] and r["split"]==row["split"]
            p,fields=independent(saved,sample,r["phase"])
            maximum=max(maximum,abs(p-r["probability"])); count+=1
            if fields is not None:
                field_error=max(field_error,float(np.max(np.abs(fields-r["field_probabilities"]))))
                assert r["field_labels"]==[row["labels"][q["event_id"]]["value"] for q in sample.queries.values()]
            if i%1600==0:
                actual=model.predict_known_authorization_union(sample,derivatives=True)
                for j in range(45):
                    old=saved["theta"][j]
                    saved["theta"][j]=old+1e-5; plus=independent(saved,sample,r["phase"])[0]
                    saved["theta"][j]=old-1e-5; minus=independent(saved,sample,r["phase"])[0]
                    saved["theta"][j]=old
                    grad_error=max(grad_error,abs((plus-minus)/2e-5-actual["gradients"]["system:any"][j])); gradient_count+=1
        for split in ("train","validation"):
            for phase in ("early","late"):
                group=[r for r in records if r["split"]==split and r["phase"]==phase]
                assert metric([r["y"] for r in group],[r["probability"] for r in group])==report["metrics"][name[:-5]][split+"/"+phase]
    assert maximum<1e-10 and field_error<1e-10 and grad_error<2e-8
    destination.mkdir(parents=True,exist_ok=False); shutil.copyfile(Path(__file__),destination/"AUDITOR.py")
    result={"source_flows_reconstructed":len(data),"predictions_recomputed":count,"maximum_probability_error":maximum,
        "maximum_field_probability_error":field_error,"gradient_directions":gradient_count,"maximum_gradient_error":grad_error,
        "new_api_calls":0,"old_test_read":False,"new_fit":False,"plan_sha256":sha(out/"PLAN.json"),
        "report_sha256":sha(out/"REPORT.json"),"auditor_sha256":sha(destination/"AUDITOR.py")}
    write(destination/"REPORT.json",result); print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args(); run(args.out.resolve(),args.destination.resolve())
