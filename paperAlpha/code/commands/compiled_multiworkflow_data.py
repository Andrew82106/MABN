"""Offline source verification and typed compilation of two existing API pilots.

Only existing train/validation records are selected. Future replies are kept
in a separate label mapping and never passed to runtime compiler entry points.
"""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, sha
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.privacy_transfer import delivered_text
from paperalpha_runtime import relational_forecast_study as relations
from paperalpha_runtime import branched_publisher_data as branches
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput
from paperalpha_monitor.branched_message_compiler import compile_branch_prefix
from paperalpha_monitor.message_workflow_compiler import MessageWorkflowCompiler
from paperalpha_monitor.message_policy_binding import bind_message_policy
from paperalpha_monitor.contextual_message_bn import typed_cue_evidence
from paperalpha_monitor.observable_event_learning import validated_labels


def compile_review_packet(packet, prefix_records, compiler):
    """Exact declared review layout plus actual prior requests; no outcomes."""
    previous = relations.compile_input(packet)
    old = previous.event_dag; cutoff = old.prefix_sequence
    if any(r["sequence"] > cutoff for r in prefix_records):
        raise ValueError("future journal records cannot enter the compiler")
    def ev(identity,text,seq=0):
        return {"record_id":identity,"text":text,"sequence":seq,"source":"bound-review-api-prefix"}
    stages,records,cues,deliveries = [],{},{},[]
    for node in old.workflow["nodes"]:
        e = old.lookup[node]
        template = "configured_source" if node == "source" else e["family"] + "_message"
        stages.append({"node":node,"actor":e["actor"],"template":template,"subjects":["marker"],"definition":copy.deepcopy(e["cue_evidence"])})
        if node in old.observations:
            o = old.observations[node]
            records[node] = {"kind":"complete","evidence":{k:o[k] for k in ("record_id","text","sequence","source")}}
        cues[node] = {"node":node,"subject":"marker","category":"support" if e["cues"] == [1,0] else "opposition" if e["cues"] == [0,1] else "unknown",
            "evidence":copy.deepcopy(e["cue_evidence"])}
        if e["family"]=="review":
            cues[node]["evidence"]=typed_cue_evidence(cues[node]["evidence"],kind="task_retention",category=cues[node]["category"],event_id="marker:"+node)
    for a,b in old.workflow["edges"]:
        seen = b in old.observations
        request = next((r for r in prefix_records if r["kind"] == "request" and r["stage"] == b),None)
        if seen:
            if request is None: raise ValueError("observed output needs its actual input record")
            payload = json.loads(request["request"]["input"][0]["content"])
            field = "source_record" if b.startswith("review-") else "reviewed_packet"
            if payload[field] != old.observations[a]["text"]:
                raise ValueError("actual received parent differs from compiled route")
            evidence = ev("record_"+f"{request['sequence']:04d}"+".json",request["request"]["input"][0]["content"],request["sequence"])
        else:
            evidence = ev("planned:"+a+":"+b,"Declared complete-message route "+a+" -> "+b)
        deliveries.append({"source":a,"target":b,"subjects":["marker"],"basis":"observed_delivery" if seen else "declared_delivery","evidence":evidence})
    dag,manifest = compiler.compile(workflow=old.workflow,subjects=[{"id":"marker","literal":old.lookup["source"]["subject"]}],
        stages=stages,deliveries=deliveries,records=records,cues=list(cues.values()),prefix_sequence=cutoff)
    targets = [{"event_id":"marker:"+p["node"],"binding":copy.deepcopy(p["binding"]),
        "field_literals":{"source_reference":old.lookup[p["node"]]["subject"]}} for p in previous.proposals]
    sample = bind_message_policy(dag,authority=previous.authority,targets=targets,cutoff=previous.cutoff,horizon=previous.horizon)
    manifest["policy_sample_sha256"] = sample.sha256
    return sample,manifest


def load_relations(run,compiler):
    plan = read(run / "PLAN.json"); rows=[]; checked=0
    assert plan["cases"] == relations.cases()
    assert sha(ROOT / 'code/src/paperalpha_runtime/relational_forecast_study.py') == plan["source_sha256"]['code/src/paperalpha_runtime/relational_forecast_study.py']
    for case in plan["cases"]:
        if case["split"] not in ("train","validation"): continue
        folder=run/"runs"/case["id"]; report=read(folder/"REPORT.json")
        assert report["plan_sha256"] == sha(run/"PLAN.json") and report["completed"]
        for name,expected in report["files_sha256"].items(): assert sha(folder/name)==expected
        records=[read(p) for p in sorted(folder.glob("record_*.json"))]
        assert [r["sequence"] for r in records] == list(range(1,len(records)+1))
        assert all(a["monotonic_ns"]<=b["monotonic_ns"] for a,b in zip(records,records[1:]))
        messages={"source":case["source"]}; replies={}; samples={}; manifests={}
        assert records[0]["kind"]=="source" and records[0]["text"]==messages["source"]
        completed={"source":{"record_id":"record_0001.json","sequence":1,"text":messages["source"]}}
        for i,record in enumerate(records):
            if record["kind"]=="prefix":
                assert record["packet"]==relations.prefix(case,completed,record["sequence"])
                assert record["packet"]==read(folder/(record["phase"].upper()+".json"))
                samples[record["phase"]],manifests[record["phase"]] = compile_review_packet(record["packet"],records[:i+1],compiler)
            elif record["kind"]=="request":
                node=relations.workflow(case)["nodes"][1+len(replies)]
                assert record["stage"]==node and record["request"]==relations.request(case,node,messages)
                checked+=1
            elif record["kind"]=="response":
                before=records[i-1]
                assert before["kind"]=="request" and before["stage"]==record["stage"]
                reply=delivered_text(record["response"]); assert not reply["native_refusal"]
                node=record["stage"]; item={"record_id":f"record_{record['sequence']:04d}.json","sequence":record["sequence"],"text":reply["text"]}
                replies[node]=item; messages[node]=item["text"]
                if node.startswith("review-"): completed[node]=item
        labels=relations.outcome(case,replies); assert labels==read(folder/"OUTCOME.json")
        labels={"marker:"+k:v for k,v in labels.items()}
        assert set(samples)=={"early","late"}
        validated_labels(samples["early"].event_dag,labels)
        bits=[labels[samples["early"].queries[a]["event_id"]]["value"] for a in samples["early"].action_ids]
        rows.append({"id":"review:"+case["id"],"split":case["split"],"template_id":"review:"+case["topic"],"cohort":"review",
            "samples":samples,"manifests":manifests,"training_phase":"early","labels":labels,"bits":bits,
            "authority_failure":[1,1],"risk":int(any(bits)),"source_hash":sha(folder/"REPORT.json")})
    assert len(rows)==24 and checked==84
    return rows,{"flows":len(rows),"verified_existing_api_calls":checked,"plan_sha256":sha(run/"PLAN.json")}


def load_branches(run,compiler):
    plan=read(run/"PLAN.json"); dev=run/"development"; collection=read(dev/"COLLECTION.json")
    assert collection["plan_sha256"]==plan["sha256"]
    assert plan["cases"]==branches.cases()
    assert sha(ROOT / 'code/src/paperalpha_runtime/branched_publisher_data.py')==plan["sources_sha256"]['code/src/paperalpha_runtime/branched_publisher_data.py']
    for name,expected in collection["artifact_sha256"].items(): assert sha(dev/name)==expected
    journal=[json.loads(line) for line in (dev/"journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["seq"] for r in journal]==list(range(1,len(journal)+1))
    assert all(a["monotonic_ns"]<=b["monotonic_ns"] for a,b in zip(journal,journal[1:]))
    cases=[c for c in plan["cases"] if c["split"] in ("train","validation")]
    assert {r["case_id"] for r in journal if r["case_id"] is not None}=={c["id"] for c in cases}
    rows=[]; checked=0
    for case in cases:
        local=[r for r in journal if r["case_id"]==case["id"]]
        record=read(dev/"cases"/(case["id"]+".json")); assert record["status"]=="completed"
        ready=next(r for r in local if r["kind"]=="prefix_ready")
        requests={r["stage"]:r for r in local if r["kind"]=="api_request"}
        responses={r["stage"]:r for r in local if r["kind"]=="api_response"}
        assert set(requests)==set(responses)==set(branches.STAGES)
        texts={}
        for node,response in responses.items():
            assert response["request_seq"]==requests[node]["seq"]<response["seq"]
            value=delivered_text(response["response"]); assert not value["native_refusal"]
            texts[node]=value["text"]
        workers={w:texts[w] for w in ("worker-a","worker-b")}
        for node,r in requests.items():
            options={} if node=="planner" else {"workers":workers} if node=="publisher" else {"planner_text":texts["planner"]}
            expected=branches.stage_request(case,node,**options)
            assert r["request"]=={**expected,**{k:plan["settings"][k] for k in ("model","reasoning","max_output_tokens","store")}}
            checked+=1
        initial=next(r for r in local if r["kind"]=="authority_initialized")
        update=next(r for r in local if r["kind"]=="authority_transition")
        assert initial["records"]==branches.issuance(case) and update["records"]==branches.transitions(case)
        assert max(responses[w]["seq"] for w in workers)<update["seq"]<ready["seq"]<requests["publisher"]["seq"]
        actual=initial["records"]+update["records"]
        prepared=branches.stage_request(case,"publisher",workers=workers)
        assert ready["observation"]==branches.prefix(case,texts["planner"],workers,actual,prepared)
        branch=BranchPrivacyInput(ready["observation"],available_journal_sequence=ready["seq"],prefix_journal_sequence=ready["seq"])
        calls={node:{"request_id":"journal:"+str(requests[node]["seq"]),"request_sequence":requests[node]["seq"],
            "response_id":"journal:"+str(responses[node]["seq"]),"response_sequence":responses[node]["seq"],
            "request":{k:requests[node]["request"][k] for k in ("instructions","input")},"text":texts[node]}
            for node in ("planner","worker-a","worker-b")}
        sample,manifest=compile_branch_prefix(branch,calls=calls,publisher_request=prepared,compiler=compiler)
        outcome=branches.outcome(case,actual,texts["publisher"])
        assert outcome==next(r["outcome"] for r in local if r["kind"]=="outcome")
        labels={}
        for aid in sample.action_ids:
            identity=sample.queries[aid]["event_id"]; value=int(sample.event_dag.lookup[identity]["subject"].casefold() in texts["publisher"].casefold())
            labels[identity]={"value":value,"record_id":"journal:"+str(responses["publisher"]["seq"]),
                "sequence":responses["publisher"]["seq"],"text":texts["publisher"],"source":"actual-completed-branch-api-response","kind":"complete_message"}
        validated_labels(sample.event_dag,labels)
        bits=[labels[sample.queries[a]["event_id"]]["value"] for a in sample.action_ids]
        assert bits==outcome["appearance"]
        rows.append({"id":"branch:"+case["id"],"split":case["split"],"template_id":"branch:"+case["template_id"],"cohort":"branch",
            "samples":{"late":sample},"manifests":{"late":manifest},"training_phase":"late","labels":labels,"bits":bits,
            "authority_failure":outcome["authority_failure"],"risk":outcome["targets"]["1"],"source_hash":sha(dev/"journal.jsonl")})
    assert len(rows)==16 and checked==64
    return rows,{"flows":len(rows),"verified_existing_api_calls":checked,"plan_sha256":sha(run/"PLAN.json")}


def load_data(branch_run,review_run):
    compiler=MessageWorkflowCompiler()
    a,va=load_branches(branch_run,compiler); b,vb=load_relations(review_run,compiler)
    rows=a+b
    train={r["template_id"] for r in rows if r["split"]=="train"}
    validation={r["template_id"] for r in rows if r["split"]=="validation"}
    assert not train & validation and len(train)==len(validation)==3
    assert len({r["id"] for r in rows})==40
    return rows,{"branch":va,"review":vb,"new_api_calls":0,"old_test_read":False,
        "scope":"Two authored existing API pilots; original 20 train/20 validation. Not new independent test data or broad task coverage."}
