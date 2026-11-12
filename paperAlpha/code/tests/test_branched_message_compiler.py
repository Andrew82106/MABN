"""Constructed multi-source and shared-policy checks, not API accuracy."""
import copy
import json

import numpy as np
import pytest

from test_message_workflow_compiler import ROOT
from branch_publisher_fixtures import fixture
from paperalpha_monitor.branched_message_compiler import compile_branch_prefix
from paperalpha_monitor.message_policy_binding import bind_message_policy
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from scalar_binary_graph import evaluate


def inputs(current="revoked", shared=True):
    branch = fixture(current=current,shared=shared)
    raw = branch.observation; payload = raw["publisher_payload"]
    base = {k:payload[k] for k in ("user_request","task_id")}
    calls = {}
    for node,start,end in (("planner",1,2),("worker-a",3,5),("worker-b",4,6)):
        value = {**base, "branches":[{"actor":"worker-a"},{"actor":"worker-b"}]} if node == "planner" else {
            **base, "planner_message":raw["pooled_scope"]["coordinator_text"], "public_branch_label":node,
            "assigned_record":{"reference":raw["pooled_scope"]["tracked_fields"][0 if node.endswith("a") else 1]["literal"]}}
        calls[node] = {"request_id":"request:"+node,"request_sequence":start,"response_id":"response:"+node,"response_sequence":end,
            "request":{"instructions":"Constructed explicit instruction for "+node,"input":[{"role":"user","content":json.dumps(value)}]},
            "text":raw["pooled_scope"]["coordinator_text"] if node == "planner" else raw["worker_messages"][node]}
    publisher = {"instructions":"Constructed publisher instructions", "input":[{"role":"user","content":json.dumps(payload)}]}
    return branch,calls,publisher


@pytest.mark.parametrize("current", ["valid","revoked","root_unknown","delegation_unknown","opaque"])
def test_full_source_graph_and_shared_authority_are_executable(current):
    branch,calls,publisher = inputs(current)
    sample,manifest = compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    d = sample.event_dag
    assert len(d.workflow["nodes"]) == 6 and len(d.events) == 12
    assert set(d.parents["field-0:worker-a"]) == {"field-0:planner","field-0:source-a"}
    assert "field-0:source-b" not in d.parents["field-0:worker-a"]
    assert d.observations["field-0:source-a"]["value"] == 1 and d.observations["field-0:source-b"]["value"] == 0
    assert len(d.parents["field-0:publisher"]) == 2
    assert sample.authority.specification() == branch.release.graph.specification()
    assert manifest["visible_cache_is_actual_authority"] is False
    assert all(e["basis"] == "declared_delivery" for e in d.edges if e["target"].endswith(":publisher"))
    model = SemanticArrivalWorkflowRiskBN(); model.theta += np.random.default_rng(99).normal(0,.25,len(model.theta))
    prediction = model.predict(sample,derivatives=True)
    for name,value in prediction["probabilities"].items():
        assert evaluate(model.probability_graph(sample,output=name)) == pytest.approx(value,abs=1e-12)
    for i in range(len(model.theta)):
        theta = model.theta.copy(); h = np.zeros_like(theta); h[i] = 1e-5
        numeric = (model.predict(sample,theta=theta+h)["probabilities"]["system:any"]-model.predict(sample,theta=theta-h)["probabilities"]["system:any"])/2e-5
        assert numeric == pytest.approx(prediction["gradients"]["system:any"][i],abs=1e-8)


def test_scope_binding_cannot_be_changed_by_readable_target_names():
    branch,calls,publisher = inputs()
    sample,_ = compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    targets = [{"event_id":sample.queries[p["action_id"]]["event_id"],"binding":copy.deepcopy(p["binding"]),
        "field_literals":sample.queries[p["action_id"]]["field_literals"]} for p in sample.proposals]
    targets[0]["binding"]["arguments"]["recipient"] = "another-recipient"
    with pytest.raises(ValueError):
        bind_message_policy(sample.event_dag,authority=sample.authority,targets=targets,cutoff=sample.cutoff,horizon=sample.horizon)


@pytest.mark.parametrize("mutation",["future_output","future_record","wrong_planner","wrong_worker","wrong_publisher","missing_source","label_field"])
def test_future_or_unbound_context_is_rejected(mutation):
    branch,calls,publisher = inputs()
    if mutation == "future_output": calls["publisher"] = copy.deepcopy(calls["worker-a"])
    elif mutation == "future_record": calls["worker-a"]["response_sequence"] = 21
    elif mutation == "wrong_worker": calls["worker-a"]["text"] = "unrelated response"
    elif mutation == "wrong_publisher": publisher["input"][0]["content"] = "{}"
    elif mutation == "label_field": calls["planner"]["risk"] = 1
    else:
        payload = json.loads(calls["worker-a"]["request"]["input"][0]["content"])
        if mutation == "wrong_planner": payload["planner_message"] = "not the previous response"
        else: payload.pop("assigned_record")
        calls["worker-a"]["request"]["input"][0]["content"] = json.dumps(payload)
    with pytest.raises(ValueError): compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
