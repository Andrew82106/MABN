import copy
import json

import numpy as np
import pytest

from test_branched_message_compiler import inputs
from paperalpha_monitor.branched_message_compiler import compile_branch_prefix
from paperalpha_monitor.contextual_message_bn import ContextualMessageWorkflowRiskBN,typed_cue_evidence,validate_context_cue
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from paperalpha_runtime.canonical import digest


def test_context_profile_is_explicit_and_does_not_change_probability_math():
    branch,calls,publisher=inputs(); sample,_=compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    old,new=SemanticArrivalWorkflowRiskBN(),ContextualMessageWorkflowRiskBN()
    theta=np.random.default_rng(908).normal(0,.3,len(old.theta)); old.theta=theta.copy(); new.theta=theta.copy()
    a,b=[m.predict(sample,derivatives=True) for m in (old,new)]
    assert a["probabilities"]==b["probabilities"]
    assert np.array_equal(a["gradients"]["system:any"],b["gradients"]["system:any"])
    artifact=new.parameters()
    assert artifact["model"]!=old.parameters()["model"]
    profile=artifact["event_law"]["message_hurdle"]["cue_profile_by_family"]
    assert profile["publish"]==["visible_authority"] and profile["review"]==["task_retention"]
    assert ContextualMessageWorkflowRiskBN.restore(artifact).parameters()==artifact
    with pytest.raises(ValueError): ContextualMessageWorkflowRiskBN.restore(old.parameters())


@pytest.mark.parametrize("mutation",["untyped","wrong_kind","wrong_event","wrong_category","changed_basis"])
def test_misidentified_cues_are_rejected(mutation):
    branch,calls,publisher=inputs(); sample,_=compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    event=copy.deepcopy(sample.event_dag.lookup["field-0:publisher"]); body=json.loads(event["cue_evidence"]["text"])
    if mutation=="untyped": event["cue_evidence"]=body["basis"]
    else:
        if mutation=="wrong_kind": body["kind"]="task_retention"
        elif mutation=="wrong_event": body["event_id"]="field-1:publisher"
        elif mutation=="wrong_category": body["category"]="opposition"
        else: body["basis"]["sequence"]+=1
        event["cue_evidence"]["text"]=json.dumps(body,sort_keys=True)
        event["cue_evidence"]["record_id"]="typed-cue:"+digest(body)
    with pytest.raises(ValueError): validate_context_cue(event)


def test_review_instruction_is_a_distinct_kind_not_a_permission():
    basis={"record_id":"instruction","source":"actual-review-request","sequence":1,"text":"Remove source reference identifiers."}
    event={"id":"marker:review-a","family":"review","cues":[0,1],
        "cue_evidence":typed_cue_evidence(basis,kind="task_retention",category="opposition",event_id="marker:review-a")}
    validate_context_cue(event)
    event["family"]="publish"
    with pytest.raises(ValueError): validate_context_cue(event)
