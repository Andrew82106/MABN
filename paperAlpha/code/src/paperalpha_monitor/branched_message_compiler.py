"""Actual fixed-branch prefix -> source-complete knowledge message subgraphs.

Unlike the old four-node projection, includes each worker's assigned source.
Only completed calls and the already prepared publication request are accepted.
Current policy proofs and publisher-visible cache remain distinct inputs.
"""
import copy
import json

from paperalpha_runtime.canonical import digest
from .branch_privacy_input import BranchPrivacyInput
from .branch_workflow_risk import BranchWorkflowInput
from .message_workflow_compiler import MessageWorkflowCompiler
from .message_policy_binding import bind_message_policy
from .contextual_message_bn import typed_cue_evidence


def compile_branch_prefix(branch, *, calls, publisher_request, compiler=None):
    if not isinstance(branch, BranchPrivacyInput):
        raise ValueError("verified branch prefix required")
    branch.validate(); cutoff = branch.prefix_journal_sequence
    if not isinstance(calls, dict) or set(calls) != {"planner", "worker-a", "worker-b"}:
        raise ValueError("only the three completed prefix calls are accepted")
    raw = branch.observation
    def ev(identity, text, seq):
        return {"record_id": identity, "text": text, "sequence": seq, "source": "bound-branch-api-prefix"}
    payloads = {}
    for node, call in calls.items():
        if not isinstance(call, dict) or set(call) != {"request_id", "request_sequence", "response_id", "response_sequence", "request", "text"}:
            raise ValueError("strict completed call with request/response binding required")
        start,end = call["request_sequence"],call["response_sequence"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= cutoff:
            raise ValueError("call must be complete inside the declared prefix")
        request = call["request"]
        if set(request) != {"instructions","input"} or not isinstance(request["instructions"],str) or not request["instructions"]:
            raise ValueError("public instructions and actual input required, not API settings or labels")
        if not isinstance(request["input"],list) or len(request["input"]) != 1 or set(request["input"][0]) != {"role","content"} or request["input"][0]["role"] != "user":
            raise ValueError("one actual user payload required")
        payload = json.loads(request["input"][0]["content"]); payloads[node] = payload
        expected_text = raw["pooled_scope"]["coordinator_text"] if node == "planner" else raw["worker_messages"][node]
        if call["text"] != expected_text or not isinstance(call["text"], str) or not call["text"]:
            raise ValueError("completed text must match the verified prefix observation")
        base = {"user_request", "task_id"}
        fields = base | ({"branches"} if node == "planner" else {"planner_message","public_branch_label","assigned_record"})
        if set(payload) != fields or payload["user_request"] != raw["publisher_payload"]["user_request"] or payload["task_id"] != raw["publisher_payload"]["task_id"]:
            raise ValueError("unexpected context or task identity in branch input")
        if node != "planner":
            if payload["planner_message"] != calls["planner"]["text"] or start <= calls["planner"]["response_sequence"]:
                raise ValueError("worker input must contain the actual prior planner output")
            if not isinstance(payload["assigned_record"], dict):
                raise ValueError("actual assigned source record required")
    if not isinstance(publisher_request, dict) or set(publisher_request) != {"instructions","input"}:
        raise ValueError("prepared publisher request only; no future output")
    if not isinstance(publisher_request["instructions"],str) or not publisher_request["instructions"]:
        raise ValueError("source-bound publisher instructions required")
    p_input = publisher_request["input"]
    if not isinstance(p_input,list) or len(p_input)!=1 or p_input[0].get("role")!="user" or set(p_input[0])!={"role","content"}:
        raise ValueError("one bound publisher payload required")
    if json.loads(p_input[0]["content"]) != raw["publisher_payload"]:
        raise ValueError("prepared publisher input differs from the captured prefix")
    previous = BranchWorkflowInput.from_branch(branch)
    registry = raw["pooled_scope"]["tracked_fields"]
    subjects = [{"id":f"field-{j}", "literal":next(f["literal"] for f in registry if set(f["field_names"]) == set(p["binding"]["arguments"]["field_names"]))}
        for j,p in enumerate(previous.proposals)]
    ids = [s["id"] for s in subjects]
    nodes = ["planner", "source-a", "source-b", "worker-a", "worker-b", "publisher"]
    edges = [["planner","worker-a"], ["planner","worker-b"], ["source-a","worker-a"], ["source-b","worker-b"],
        ["worker-a","publisher"], ["worker-b","publisher"]]
    workflow = {"nodes":nodes, "edges":edges, "actors":{n:[n if not n.startswith("source-") else "host-record-adapter"] for n in nodes}}
    stages, records, cues, routes = [], {}, [], []
    for node in nodes:
        source = node.startswith("source-")
        if source:
            owner = "worker-" + node[-1]; call = calls[owner]
            # Derived JSON of the exact assigned_record component, with its
            # full source request identified. Not an extra Agent response.
            text = json.dumps(payloads[owner]["assigned_record"], ensure_ascii=False, sort_keys=True)
            definition = ev("component:"+call["request_id"], "assigned_record component of the actual worker request; no other branch supplied.", call["request_sequence"])
            records[node] = {"kind":"complete", "evidence":ev("assigned:"+call["request_id"],text,call["request_sequence"])}
            template = "configured_source"
        elif node == "publisher":
            definition = ev("prepared-publisher:"+digest(publisher_request),publisher_request["instructions"],cutoff)
            template = "publish_message"
        else:
            call = calls[node]
            definition = ev("instruction:"+call["request_id"],call["request"]["instructions"],call["request_sequence"])
            records[node] = {"kind":"complete", "evidence":ev(call["response_id"],call["text"],call["response_sequence"])}
            template = "prepare_message" if node == "planner" else "work_message"
        stages.append({"node":node,"actor":workflow["actors"][node][0],"template":template,"subjects":ids,"definition":definition})
    for j,subject in enumerate(subjects):
        values = branch.release.visible_features[j].astype(int).tolist()
        category = "support" if values == [1,0] else "opposition" if values == [0,1] else "unknown"
        cues.append({"node":"publisher","subject":subject["id"],"category":category,
            "evidence":typed_cue_evidence(ev("visible-cache:"+digest(raw["publisher_payload"]),p_input[0]["content"],cutoff),
                kind="visible_authority",category=category,event_id=subject["id"]+":publisher")})
    for a,b in edges:
        future = b == "publisher"
        call = None if future else calls[b]
        routes.append({"source":a,"target":b,"subjects":ids,
            "basis":"declared_delivery" if future else "observed_delivery",
            "evidence":ev("prepared-publication" if future else "input:"+call["request_id"],
                p_input[0]["content"] if future else call["request"]["input"][0]["content"],cutoff if future else call["request_sequence"])})
    dag, manifest = (compiler or MessageWorkflowCompiler()).compile(workflow=workflow, subjects=subjects, stages=stages,
        deliveries=routes, records=records, prefix_sequence=cutoff, cues=cues)
    targets = [{"event_id":f"field-{j}:publisher","binding":copy.deepcopy(p["binding"]),
        "field_literals":{f:subjects[j]["literal"] for f in p["binding"]["arguments"]["field_names"]}}
        for j,p in enumerate(previous.proposals)]
    sample = bind_message_policy(dag, authority=previous.authority, targets=targets,
        cutoff=previous.cutoff,horizon=previous.horizon)
    manifest["source_calls_sha256"] = digest(calls)
    manifest["prepared_publisher_sha256"] = digest(publisher_request)
    manifest["policy_sample_sha256"] = sample.sha256
    manifest["visible_cache_is_actual_authority"] = False
    return sample, manifest
