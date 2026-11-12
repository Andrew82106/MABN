"""Constructed dependency checks, kept separate from real API performance."""
import copy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.canonical import digest
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, ObservableEventLaw, DAGWorkflowInput, DAGWorkflowRiskBN
from paperalpha_monitor.event_dag_projection import from_verified_branch
from branch_publisher_fixtures import fixture as branch_fixture
from workflow_risk_fixtures import fixture as policy_fixture
from scalar_binary_graph import evaluate


def evidence(name, text="Constructed explicit prefix evidence", sequence=0):
    return {"record_id": name, "sequence": sequence, "source": "constructed-mechanism-test", "text": text}


def coupled_tool_fixture(n=2, authority="known", layout="fork"):
    policy = policy_fixture(families=("payment",) * n, visible=("unknown",) * n, amounts=("60",) * n, authority=authority, layout=layout,
                            assessment_start=max(10, 3 * n + 1))
    events = [{"id": "future-plan", "node": "plan", "actor": "planner", "family": "prepare", "semantics": "literal_appearance",
               "subject": "BATCH_APPROVED", "cues": [0, 0], "cue_evidence": evidence("plan") }]
    edges, queries = [], {}
    for i, proposal in enumerate(policy.proposals):
        identity = "future-" + proposal["action_id"]
        events.append({"id": identity, "node": proposal["node"], "actor": proposal["binding"]["actor"], "family": "dispatch", "semantics": "tool_dispatch",
            "subject": digest(proposal["binding"]), "cues": [int(i == 0), int(i == 1)], "cue_evidence": evidence(identity)})
        parent = events[-2]["id"] if layout == "chain" else "future-plan"
        edges.append({"source": parent, "target": identity, "basis": "declared_delivery", "evidence": evidence("planned:" + identity)})
        queries[proposal["action_id"]] = {"event_id": identity, "binding_sha256": digest(proposal["binding"]), "field_literals": None}
    dag = ObservableEventDAG(workflow=policy.workflow, events=events, edges=edges, observations={}, prefix_sequence=20)
    sample = DAGWorkflowInput(event_dag=dag, queries=queries, workflow=policy.workflow, authority=policy.authority, proposals=policy.proposals,
                              contracts=policy.contracts, combinations=policy.combinations, cutoff=policy.cutoff, horizon=policy.horizon)
    model = DAGWorkflowRiskBN()
    model.theta[model.names.index("event:dispatch:bias")] = -2.
    model.theta[model.names.index("event:dispatch:parent_mean")] = 4.
    return sample, model


@pytest.mark.parametrize("layout,n", [("fork", 1), ("fork", 2), ("fork", 3), ("fork", 5), ("chain", 2), ("chain", 4)])
def test_variable_workflow_shapes_have_executable_joint_graphs(layout, n):
    sample, model = coupled_tool_fixture(n, layout=layout)
    prediction = model.predict(sample)
    assert prediction["enumerated_future_event_worlds"] == 2 ** (n + 1)
    assert len(model.theta) == 28
    for query, probability in prediction["probabilities"].items():
        assert evaluate(model.probability_graph(sample, output=query)) == pytest.approx(probability, abs=1e-12)


def test_unobserved_shared_predecessor_is_not_replaced_by_independent_marginals():
    sample, model = coupled_tool_fixture()
    law = model.event_law.conditional(sample.event_dag, [q["event_id"] for q in sample.queries.values()], model.theta[:25])
    p = np.exp(law["log_probabilities"])
    marginals = p @ law["states"]
    assert p[-1] > np.prod(marginals) + .1
    assert model.predict(sample)["probabilities"]["channel:cumulative_budget"] == pytest.approx(p[-1])
    # Observing a queued event as a scenario updates the other event via the
    # shared predecessor. It is NOT a causal intervention severing parent edges.
    conditioned = model.predict(sample, event_conditions={"a-0": 1}, derivatives=True)
    assert conditioned["dispatch_probabilities"]["a-1"] > marginals[1] + .1
    for query, probability in conditioned["probabilities"].items():
        assert evaluate(model.probability_graph(sample, output=query, event_conditions={"a-0": 1})) == pytest.approx(probability, abs=1e-12)


@pytest.mark.parametrize("authority", ["known", "invalid", "root", "delegation", "opaque"])
def test_full_joint_risk_gradients_and_authority_identity(authority):
    sample, model = coupled_tool_fixture(authority=authority)
    theta = model.theta + np.random.default_rng(4).normal(0, .2, len(model.theta))
    model.theta = theta.copy()
    prediction = model.predict(sample, derivatives=True, event_conditions={"a-0": 1})
    actual = prediction["gradients"]["system:any"]
    for i in range(len(theta)):
        d = np.zeros_like(theta); d[i] = 1e-5
        numeric = (model.predict(sample, theta=theta + d, event_conditions={"a-0": 1})["probabilities"]["system:any"] -
                   model.predict(sample, theta=theta - d, event_conditions={"a-0": 1})["probabilities"]["system:any"]) / 2e-5
        assert numeric == pytest.approx(actual[i], abs=2e-8)
    graph = model.probability_graph(sample)
    assert len([n for n in graph["nodes"] if n["id"].startswith("fact:")]) == len(sample.authority.facts)


def test_likelihood_gradients_marginalize_missing_labels():
    sample, model = coupled_tool_fixture(authority="root")
    model.theta += np.random.default_rng(9).normal(0, .15, len(model.theta))
    labels = [{"events": {"a-0": 1}, "risks": {"system:any": 1}}]
    loss, gradient = model.loss_gradient(model.theta, [sample], labels)
    for i in range(len(model.theta)):
        d = np.zeros_like(model.theta); d[i] = 1e-5
        numeric = (model.loss_gradient(model.theta + d, [sample], labels)[0] - model.loss_gradient(model.theta - d, [sample], labels)[0]) / 2e-5
        assert numeric == pytest.approx(gradient[i], abs=2e-8)
    assert np.isfinite(loss)


@pytest.mark.parametrize("current", ["valid", "revoked", "root_unknown"])
def test_existing_branch_prefix_is_bound_without_future_output(current):
    branch = branch_fixture(current=current)
    sample = from_verified_branch(branch)
    model = DAGWorkflowRiskBN()
    result = model.predict(sample)
    assert len(sample.event_dag.events) == 8 and len(sample.event_dag.observations) == 6
    for query, p in result["probabilities"].items():
        assert evaluate(model.probability_graph(sample, output=query)) == pytest.approx(p, abs=1e-12)
    assert result["dispatch_probabilities"] == {}
    assert len(result["appearance_probabilities"]) == 2
    assert set(sample.event_dag.observations).isdisjoint(q["event_id"] for q in sample.queries.values())


def test_rename_actor_and_event_ids_does_not_change_the_event_law():
    sample, model = coupled_tool_fixture()
    dag = sample.event_dag
    spec = dag.specification()
    for event in spec["events"]:
        event["id"] = "renamed:" + event["id"]
    for edge in spec["edges"]:
        edge["source"], edge["target"] = "renamed:" + edge["source"], "renamed:" + edge["target"]
    other = ObservableEventDAG(**spec)
    a = model.event_law.conditional(dag, ["future-a-0", "future-a-1"], model.theta[:25])
    b = model.event_law.conditional(other, ["renamed:future-a-0", "renamed:future-a-1"], model.theta[:25])
    assert np.allclose(a["log_probabilities"], b["log_probabilities"], atol=1e-14)


@pytest.mark.parametrize("mutation", ["future_evidence", "cycle", "wrong_actor", "different_literal", "invented_observation", "future_observed_delivery", "missing_parent"])
def test_invalid_or_leaking_event_evidence_rejected(mutation):
    branch = from_verified_branch(branch_fixture())
    spec = branch.event_dag.specification()
    if mutation == "future_evidence": spec["events"][0]["cue_evidence"]["sequence"] = 21
    elif mutation == "cycle": spec["workflow"]["edges"].append(["publisher", "planner"])
    elif mutation == "wrong_actor": spec["events"][0]["actor"] = "publisher"
    elif mutation == "different_literal": spec["events"][1]["subject"] = "DIFFERENT"
    elif mutation == "invented_observation": spec["observations"][next(iter(spec["observations"]))]["value"] ^= 1
    elif mutation == "future_observed_delivery": spec["observations"].pop("field-0:planner")
    elif mutation == "missing_parent":
        spec["observations"].pop("field-0:planner")
        for edge in spec["edges"]: edge["basis"] = "declared_delivery"
    with pytest.raises(ValueError):
        ObservableEventDAG(**spec)


def test_capacity_counts_unobserved_intermediate_events_not_only_targets():
    sample, model = coupled_tool_fixture()
    model.max_binary_variables = 2
    with pytest.raises(ValueError, match="intermediate"):
        model.predict(sample)


def test_restore_and_knowledge_identity_roundtrip():
    sample, model = coupled_tool_fixture()
    restored = DAGWorkflowRiskBN.restore(model.parameters())
    assert restored.predict(sample) == model.predict(sample)
    bad = model.parameters()
    bad["event_law"]["features"].append("label")
    with pytest.raises(ValueError):
        DAGWorkflowRiskBN.restore(bad)


def test_executed_event_cannot_be_relabelled_as_a_future_policy_target():
    sample, model = coupled_tool_fixture()
    # Existing instance mutation is rejected before any probability query.
    sample.queries["a-0"]["event_id"] = "future-plan"
    with pytest.raises(ValueError):
        model.predict(sample)


def missing_past_fixture():
    sample, model = coupled_tool_fixture()
    spec = sample.event_dag.specification()
    spec["workflow"]["nodes"].append("receipt")
    spec["workflow"]["actors"]["receipt"] = ["reader"]
    spec["workflow"]["edges"].append(["plan", "receipt"])
    spec["events"].append({"id": "receipt-mentions-approval", "node": "receipt", "actor": "reader", "family": "review",
        "semantics": "literal_appearance", "subject": "BATCH_APPROVED", "cues": [0, 0], "cue_evidence": evidence("receipt-cue")})
    spec["edges"].append({"source": "future-plan", "target": "receipt-mentions-approval", "basis": "observed_delivery", "evidence": evidence("delivered-but-source-content-missing")})
    spec["observations"] = {"receipt-mentions-approval": {**evidence("receipt-message", "The message says BATCH_APPROVED", sequence=10), "value": 1, "kind": "complete_message"}}
    # The historical ID is intentionally unchanged. Phase comes from the
    # explicit declaration, never heuristics applied to the identifier string.
    spec["missing_past"] = ["future-plan"]
    dag = ObservableEventDAG(**spec)
    combined = DAGWorkflowInput(event_dag=dag, queries=sample.queries, workflow=dag.workflow, authority=sample.authority, proposals=sample.proposals,
        contracts=sample.contracts, combinations=sample.combinations, cutoff=sample.cutoff, horizon=sample.horizon)
    model.theta[model.names.index("event:review:bias")] = -2.
    model.theta[model.names.index("event:review:parent_mean")] = 4.
    return combined, model


def test_missing_past_is_integrated_and_descendant_evidence_updates_it():
    sample, model = missing_past_fixture()
    prediction = model.predict(sample, derivatives=True)
    assert prediction["dispatch_probabilities"]["a-0"] > .7
    assert prediction["enumerated_future_event_worlds"] == 4
    assert prediction["enumerated_unobserved_event_worlds"] == 8
    for query, p in prediction["probabilities"].items():
        assert evaluate(model.probability_graph(sample, output=query)) == pytest.approx(p, abs=1e-12)
    for i in range(len(model.theta)):
        d = np.zeros_like(model.theta); d[i] = 1e-5
        numeric = (model.predict(sample, theta=model.theta + d)["probabilities"]["system:any"] -
                   model.predict(sample, theta=model.theta - d)["probabilities"]["system:any"]) / 2e-5
        assert numeric == pytest.approx(prediction["gradients"]["system:any"][i], abs=2e-8)


def test_missing_past_cannot_be_scored_as_a_new_future_target():
    sample, model = missing_past_fixture()
    with pytest.raises(ValueError, match="future targets"):
        model.event_law.conditional(sample.event_dag, ["future-plan"], model.theta[:25])
