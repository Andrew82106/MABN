"""Constructed workflow/learning tests, not detection performance evidence."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.message_workflow_compiler import MessageWorkflowCompiler
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from paperalpha_monitor.message_hurdle import compile_message_training, message_objective
from paperalpha_monitor.agentleak_message_compiler import compile_prefix, rebind_existing_policy
from paperalpha_monitor.agentleak_event_projection import project, system_prompts
from paperalpha_runtime.canonical import digest
from scalar_binary_graph import evaluate


def ev(identity, text="Explicit constructed definition", sequence=0):
    return {"record_id": identity, "source": "constructed-compiler-test", "text": text, "sequence": sequence}


def specification(layout="fork"):
    nodes = ["source", "left", "right", "review", "publish"]
    edges = [["source", "left"], ["source", "right"], ["left", "review"], ["right", "review"], ["review", "publish"]]
    if layout == "chain":
        edges = [[nodes[i], nodes[i+1]] for i in range(len(nodes)-1)]
    workflow = {"nodes": nodes, "edges": edges, "actors": {n: [n + "-agent"] for n in nodes}}
    templates = ["configured_source", "work_message", "work_message", "review_message", "publish_message"]
    return {"workflow": workflow, "subjects": [{"id": "secret", "literal": "EXACT-SECRET"}],
        "stages": [{"node": n, "actor": n + "-agent", "template": t, "subjects": ["secret"], "definition": ev(n)} for n,t in zip(nodes, templates)],
        "deliveries": [{"source": a, "target": b, "subjects": ["secret"], "basis": "declared_delivery", "evidence": ev(a + b)} for a,b in edges],
        "records": {"source": {"kind": "complete", "evidence": ev("initial", "EXACT-SECRET", 1)}}, "prefix_sequence": 2}


@pytest.mark.parametrize("layout", ["chain", "fork"])
def test_compiles_actual_subgraphs_and_runs_the_learnable_joint_law(layout):
    inputs = specification(layout); original = copy.deepcopy(inputs)
    compiler = MessageWorkflowCompiler(); dag, manifest = compiler.compile(**inputs)
    assert inputs == original and len(dag.events) == 5 and len(dag.edges) == len(inputs["deliveries"])
    assert manifest["event_dag_sha256"] == dag.sha256
    assert manifest["stage_templates"]["review"] == "review_message"
    assert manifest["knowledge"]["risk_ids"] == ["RB013", "RB014"]
    model = SemanticArrivalWorkflowRiskBN(); model.theta[:65] = np.random.default_rng(6).normal(0, .4, 65)
    result = model.event_law.conditional(dag, ["secret:publish"], model.theta[:65])
    assert np.exp(result["log_probabilities"]).sum() == pytest.approx(1.)
    target = "secret:publish"
    nodes = model.event_law.graph_nodes(dag, model.theta[:65], {i:i for i in dag.ids})
    graph = {"nodes": nodes, "output_id": target}
    assert evaluate(graph) == pytest.approx(np.exp(result["log_probabilities"])[1], abs=1e-12)
    assert "review" in {n["family"] for n in manifest["nodes"]}


def test_shared_function_parameters_fit_multiple_workflow_shapes():
    compiler = MessageWorkflowCompiler(); dags, labels = [], []
    for i in range(8):
        dag, _ = compiler.compile(**specification("fork" if i%2 else "chain"))
        future = {}
        for j, identity in enumerate(dag.order[1:]):
            value = int((i+j)%3 == 0)
            future[identity] = {**ev("output:"+str(i)+identity, "EXACT-SECRET" if value else "redacted", 3+j),
                "value": value, "kind": "complete_message"}
        dags.append(dag); labels.append(future)
    model = SemanticArrivalWorkflowRiskBN(regularization=.01)
    compiled = compile_message_training(model.event_law, dags, labels)
    before = message_objective(model.theta[:65], compiled, .01)[0]
    model.fit_messages(dags, labels)
    assert model.optimizer["success"] and model.optimizer["trace_count"] == 8
    assert message_objective(model.theta[:65], compiled, .01)[0] < before-.05
    assert np.isfinite(model.theta).all()


def test_subset_delivery_does_not_invent_a_value_dependency():
    inputs = specification(); inputs["subjects"].append({"id": "other", "literal": "ANOTHER-SECRET"})
    for stage in inputs["stages"]: stage["subjects"].append("other")
    dag, _ = MessageWorkflowCompiler().compile(**inputs)
    assert dag.parents["secret:left"] == ("secret:source",)
    assert dag.parents["other:left"] == ()
    # No route for this tracked value is not proof it cannot be generated.
    assert "other:left" not in dag.observations


def test_missing_past_message_remains_unknown_and_can_be_marginalized():
    inputs = specification(); inputs["records"]["left"] = {"kind":"missing", "evidence":ev("lost", "Capture records the message but text is missing", 2)}
    dag, _ = MessageWorkflowCompiler().compile(**inputs)
    assert dag.missing_past == ("secret:left",) and "secret:left" not in dag.observations
    model = SemanticArrivalWorkflowRiskBN()
    law = model.event_law.conditional(dag, ["secret:publish"], model.theta[:65])
    assert np.isfinite(law["log_probabilities"]).all()


def test_cue_category_is_bound_to_the_correct_stage_and_field():
    inputs = specification(); inputs["cues"] = [{"node":"publish", "subject":"secret", "category":"opposition", "evidence":ev("cue", "Return only aggregate statistics",1)}]
    dag, manifest = MessageWorkflowCompiler().compile(**inputs)
    assert dag.lookup["secret:publish"]["cues"] == [0,1]
    assert dag.lookup["secret:review"]["cues"] == [0,0]
    assert manifest["nodes"][-1]["cue_sha256"] == digest(inputs["cues"][0])


@pytest.mark.parametrize("mutation", ["cycle", "future", "past_source", "unowned", "unknown_template", "duplicate_literal",
    "missing_route", "bad_subject", "duplicate_record", "late_delivery", "invented_delivery", "renamed_operator", "unknown_knowledge", "registry_changed"])
def test_invalid_or_unsupported_compilations_are_not_silently_guessed(mutation):
    inputs = specification(); compiler = MessageWorkflowCompiler()
    if mutation == "cycle": inputs["workflow"]["edges"].append(["publish","source"])
    elif mutation == "future": inputs["records"]["source"]["evidence"]["sequence"] = 3
    elif mutation == "past_source": inputs["records"] = {}
    elif mutation == "unowned": inputs["stages"][1]["actor"] = "some-other-agent"
    elif mutation == "unknown_template": inputs["stages"][1]["template"] = "unspecified_magic_agent"
    elif mutation == "duplicate_literal": inputs["subjects"].append({"id":"alias", "literal":"exact-secret"})
    elif mutation == "missing_route": inputs["deliveries"].pop()
    elif mutation == "bad_subject": inputs["deliveries"][0]["subjects"] = ["untracked"]
    elif mutation == "duplicate_record": inputs["records"]["left"] = copy.deepcopy(inputs["records"]["source"])
    elif mutation in ("late_delivery", "invented_delivery"):
        r = inputs["deliveries"][0 if mutation == "late_delivery" else 2]; r["basis"] = "observed_delivery"
        if mutation == "late_delivery":
            inputs["records"]["left"] = {"kind":"complete", "evidence":ev("left-message", "redacted", 1)}
            r["evidence"]["sequence"] = 2
    elif mutation == "registry_changed": compiler.templates["work_message"]["family"] = "publish"
    elif mutation in ("renamed_operator", "unknown_knowledge"):
        spec = copy.deepcopy(compiler.specification)
        if mutation == "renamed_operator": spec["templates"][0]["family"] = "publish"
        else: spec["knowledge"]["risk_ids"] = ["NONEXISTENT"]
        with pytest.raises(ValueError): MessageWorkflowCompiler(spec)
        return
    with pytest.raises(ValueError): compiler.compile(**inputs)


def prefix_fixture(late):
    initial = {"vault":{"field":"EXACT-SECRET"}, "request":"Produce a concise answer", "allowed_set":{"fields":[]}}
    prompts = system_prompts(initial)
    coordinator = {"call_id":"c", "system_prompt":prompts["prepare"], "user_prompt":"User request: Produce a concise answer\nPrepare instructions for Worker agent.", "response":"EXACT-SECRET"}
    worker = {"call_id":"w", "system_prompt":prompts["work"], "user_prompt":"Instructions: EXACT-SECRET", "response":"EXACT-SECRET"} if late else None
    return initial, coordinator, worker


@pytest.mark.parametrize("late", [False,True])
def test_existing_agentleak_policy_and_scores_use_compiled_messages(late):
    initial,c,w = prefix_fixture(late)
    old = project(initial, coordinator_text=c["response"], coordinator_record_id=c["call_id"],
        **({"worker_text":w["response"], "worker_record_id":w["call_id"]} if late else {}))
    dag, manifest = compile_prefix(initial, coordinator=c, worker=w)
    assert dag.events == old.event_dag.events and dag.observations == old.event_dag.observations
    assert dag.parents == old.event_dag.parents and dag.sha256 != old.event_dag.sha256
    for edge in dag.edges:
        if edge["basis"] == "observed_delivery":
            assert edge["evidence"]["sequence"] >= dag.observations[edge["source"]]["sequence"]
    new = rebind_existing_policy(old, dag)
    assert new.authority.specification() == old.authority.specification()
    model = SemanticArrivalWorkflowRiskBN(); model.theta[:65] = np.random.default_rng(77).normal(0,.3,65)
    a,b = [model.predict_known_authorization_union(x, derivatives=True) for x in (old,new)]
    assert a["probabilities"] == b["probabilities"]
    assert np.array_equal(a["gradients"]["system:any"], b["gradients"]["system:any"])


@pytest.mark.parametrize("mutation", ["extra_label", "wrong_delivery", "different_source", "empty_response"])
def test_prefix_adapter_rejects_unverified_context_and_future_label_fields(mutation):
    initial,c,w = prefix_fixture(True)
    if mutation == "extra_label": c["leaked"] = True
    elif mutation == "wrong_delivery": w["user_prompt"] = "Instructions from somebody else"
    elif mutation == "different_source": w["system_prompt"] = "No vault access"
    else: c["response"] = ""
    with pytest.raises(ValueError): compile_prefix(initial, coordinator=c, worker=w)
