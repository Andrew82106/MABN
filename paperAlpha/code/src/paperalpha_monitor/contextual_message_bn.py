"""Typed visible-context cues, with the same normalized arrival message law.

This explicit profile covers review retention instructions and publisher-visible
authority assessments. Neither kind is the monitor's actual authorization.
It must not be mislabeled as an initial-task-relevance-only classifier model.
"""
import copy
import json

from paperalpha_runtime.canonical import digest
from .task_cue_message import SemanticArrivalMessageLaw
from .message_hurdle import MessageHurdleWorkflowRiskBN

CATEGORIES={"support":[1,0],"opposition":[0,1],"unknown":[0,0]}
PROFILE={"prepare":[],"work":[],"review":["task_retention"],"publish":["visible_authority"],"dispatch":[]}


def typed_cue_evidence(basis,*,kind,category,event_id):
    if kind not in {k for values in PROFILE.values() for k in values} or category not in CATEGORIES:
        raise ValueError("explicit supported evidence kind and category required")
    if not isinstance(basis,dict) or set(basis)!={"record_id","sequence","source","text"} or not isinstance(event_id,str) or not event_id:
        raise ValueError("bound underlying evidence and event identity required")
    payload={"schema":"typed-visible-message-cue-v1","kind":kind,"category":category,"event_id":event_id,"basis":copy.deepcopy(basis)}
    return {"record_id":"typed-cue:"+digest(payload),"sequence":basis["sequence"],"source":"typed-visible-message-cue-v1",
        "text":json.dumps(payload,sort_keys=True,ensure_ascii=False)}


def validate_context_cue(event):
    evidence=event["cue_evidence"]
    if evidence["source"]!="typed-visible-message-cue-v1":
        if event["cues"]!=[0,0]: raise ValueError("nonzero context cue requires an explicit typed evidence binding")
        return
    payload=json.loads(evidence["text"])
    if set(payload)!={"schema","kind","category","event_id","basis"} or payload["schema"]!="typed-visible-message-cue-v1":
        raise ValueError("supported typed cue envelope required")
    if payload["kind"] not in PROFILE[event["family"]] or payload["event_id"]!=event["id"]:
        raise ValueError("cue kind must match its declared stage probability profile and exact event")
    if payload["category"] not in CATEGORIES or CATEGORIES[payload["category"]]!=event["cues"]:
        raise ValueError("cue category and numerical flags disagree")
    basis=payload["basis"]
    if set(basis)!={"record_id","sequence","source","text"} or evidence["sequence"]!=basis["sequence"]:
        raise ValueError("underlying cue evidence and availability must be retained")
    if type(basis["sequence"]) is not int or any(not isinstance(basis[k],str) or not basis[k] for k in ("record_id","source","text")):
        raise ValueError("underlying cue evidence must be a complete source record")
    if evidence["record_id"]!="typed-cue:"+digest(payload): raise ValueError("typed cue evidence changed")


class ContextualMessageLaw(SemanticArrivalMessageLaw):
    message_spec={**copy.deepcopy(SemanticArrivalMessageLaw.message_spec),"schema":"typed-context-arrival-message-v1",
        "cue_meaning":"Typed externally visible retention instructions or agent-visible authority assessments. Not actual authority, internal belief or a task-relevance-only classifier.",
        "cue_profile_by_family":copy.deepcopy(PROFILE),
        "profile_scope":"Shared parameters within each declared family; not evidence that different cue meanings are interchangeable. Unsupported nonzero kinds are rejected."}

    @staticmethod
    def groups(dag):
        for event in dag.events: validate_context_cue(event)
        return SemanticArrivalMessageLaw.groups(dag)

    def group_parts(self,dag,group,values,theta):
        for identity in group: validate_context_cue(dag.lookup[identity])
        return super().group_parts(dag,group,values,theta)


class ContextualMessageWorkflowRiskBN(MessageHurdleWorkflowRiskBN):
    MODEL="ContextualMessageWorkflowRiskBN-v1"
    law_class=ContextualMessageLaw
