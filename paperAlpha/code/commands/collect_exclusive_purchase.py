"""Freeze and collect finite-choice API MAS traces, without scores or refitting."""
from pathlib import Path
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'code/src'))
from paperalpha_runtime import exclusive_purchase as study
from paperalpha_runtime.privacy_transfer import delivered_text


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False); f.write("\n")


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/src').rglob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json','code/commands/collect_exclusive_purchase.py','code/commands/evaluate_exclusive_purchase.py',
              'code/commands/scalar_binary_graph.py','code/tests/test_exclusive_dispatch_bn.py']
    for n in names:
        dest=out/"source"/n; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/n,dest)
    plan={"schema":"exclusive-purchase-collection-v1","created_at_utc":now(),"cases":study.cases(),
        "source_sha256":{n:sha(out/"source"/n) for n in sorted(names)},
        "api":{"endpoint":"http://localhost:58661/v1","model":"gpt-5.5","effort":"low","max_output_tokens":3200,
               "timeout_seconds":60,"transport_retries":0,"concurrent_flows":2,"target_calls":72},
        "fit":{"regularization":[.001,.03],"max_iter":500,"fixed_alarm_threshold":.5,"probability_digits":8},
        "protocol":["24 predeclared authored flows: two train and two validation domains; no post-result resampling or threshold changes.",
            "Each flow makes one planner and two buyer API requests, with separate buyer contexts sharing only the actual planner reply.",
            "Prefix is saved before either buyer is called; it contains finite candidate commands and application-enforced exclusive dispatch slots.",
            "An actual single command is parsed before a separate local simulated tool-dispatch record; proposal text is not a dispatch event.",
            "No suppliers, actual orders, payments, external messages, defense blocks, retries or model-setting changes.",
            "After collection, offline prefix replay fits only training domains. Same-supervision softmax and independent-dispatch controls must be reported.",
            "Failures and abstentions remain in collection reports; incomplete flows are not negative labels.",
            "Initial data is also saved before planner generation for future upstream-state research, not current early-warning claims."],
        "openai_docs":"https://developers.openai.com/api/docs/guides/structured-outputs#json-mode",
        "old_test_read":False,"agent_outputs_must_be_api":True,"new_default_selected":False}
    write(out/"PLAN.json",plan); print(json.dumps({"prepared":str(out),"flows":24,"plan_sha256":sha(out/"PLAN.json")}),flush=True)


def load(out):
    plan=read(out/"PLAN.json")
    if ROOT!=(out/"source").resolve(): raise ValueError("run the frozen collector/evaluator")
    for n,h in plan["source_sha256"].items():
        if sha(ROOT/n)!=h: raise ValueError("frozen source changed: "+n)
    if plan["cases"]!=study.cases(): raise ValueError("predeclared task set changed")
    return plan


async def collect(out):
    plan=load(out)
    from openai import AsyncOpenAI
    key=os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key: raise RuntimeError("PAPERALPHA_LOCAL_API_KEY is required; never write it to an artifact")
    api=AsyncOpenAI(base_url=plan["api"]["endpoint"],api_key=key,timeout=60,max_retries=0)
    try:
        models=await api.models.list()
        write(out/"API_PREFLIGHT.json",{"at_utc":now(),"authenticated":True,"models":[m.id for m in models.data],"generation_calls":0})
    except Exception as exc:
        write(out/"API_PREFLIGHT.json",{"at_utc":now(),"authenticated":False,"error_type":type(exc).__name__,"generation_calls":0})
        await api.close(); raise RuntimeError("Local endpoint preflight failed; no generation attempted") from None
    semaphore=asyncio.Semaphore(plan["api"]["concurrent_flows"])
    async def flow(case):
        async with semaphore:
            directory=out/"runs"/case["id"]; directory.mkdir(parents=True,exist_ok=False)
            records=[]; api_calls=0; complete=False; failure=None
            def record(value):
                row={"sequence":len(records)+1,"at_utc":now(),"monotonic_ns":time.monotonic_ns(),**value}
                write(directory/f"record_{len(records)+1:04d}.json",row); records.append(row); return row
            async def ask(stage,request):
                nonlocal api_calls
                record({"kind":"request","stage":stage,"request":request}); api_calls+=1
                raw=(await api.responses.create(**request)).model_dump(mode="json")
                public={k:raw.get(k) for k in ("id","model","status","incomplete_details","usage")}
                public["output"]=[m for m in raw.get("output",[]) if m.get("type")=="message"]
                record({"kind":"response","stage":stage,"response":public})
                reply=delivered_text(public)
                if reply["native_refusal"]: raise RuntimeError("native refusal")
                return reply["text"]
            try:
                write(directory/"INITIAL.json",{"case":case,"at_utc":now(),"phase":"before_planner"})
                planner=await ask("planner",study.request(case,"planner")); study.parse_planner(planner)
                row=record({"kind":"forecast_prefix","value":study.prefix(case,planner)})
                write(directory/"PREFIX.json",row)
                engine=study.SimulatedDispatchSlots(case); dispatches=[]
                for branch in case["branches"]:
                    actor=branch["actor"]; text=await ask(actor,study.request(case,actor,planner))
                    command=study.parse_buyer(case,actor,text)
                    record({"kind":"parsed_proposal","actor":actor,"command":command})
                    event=engine.execute(actor,command)
                    record({"kind":"runtime_event","event":event}); dispatches.append(event)
                write(directory/"OUTCOME.json",study.outcome(case,dispatches)); complete=True
            except Exception as exc:
                failure=type(exc).__name__; record({"kind":"failure","error_type":failure})
            report={"case_id":case["id"],"completed_at_utc":now(),"complete":complete,"api_calls":api_calls,
                "failure_type":failure,"files_sha256":{p.name:sha(p) for p in sorted(directory.glob("*.json"))}}
            write(directory/"REPORT.json",report)
            print(json.dumps({"flow":case["id"],"complete":complete,"api_calls":api_calls,"failure":failure}),flush=True)
            return report
    try:
        reports=await asyncio.gather(*(flow(c) for c in plan["cases"]))
        report={"plan_sha256":sha(out/"PLAN.json"),"completed_at_utc":now(),"planned_flows":len(plan["cases"]),
            "completed_flows":sum(r["complete"] for r in reports),"api_calls":sum(r["api_calls"] for r in reports),
            "reports_sha256":{r["case_id"]:sha(out/"runs"/r["case_id"]/"REPORT.json") for r in reports},
            "new_fits":0,"old_test_read":False,"accuracy_claim":False}
        write(out/"COLLECTION.json",report); print(json.dumps(report),flush=True)
    finally:
        await api.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("prepare","collect")); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.mode=="prepare": prepare(args.out.resolve())
    else: asyncio.run(collect(args.out.resolve()))
