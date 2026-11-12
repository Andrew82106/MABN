"""Elimination must reproduce enumeration, not silently assume independence."""
import copy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, ObservableEventLaw
from paperalpha_monitor.event_elimination import EliminatingEventLaw, EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import trace_loss_gradient, complete_trace_design, complete_trace_objective, fit_event_traces
from paperalpha_monitor.agentleak_event_projection import project, later_labels
from test_observable_event_dag import coupled_tool_fixture, missing_past_fixture
from scalar_binary_graph import evaluate


def many_fields(n=13):
    workflow = {"nodes": ["source", "work", "publish"], "edges": [["source", "work"], ["work", "publish"]],
        "actors": {"source": ["source-adapter"], "work": ["worker"], "publish": ["publisher"]}}
    def evidence(name, text="Constructed local factor fixture"):
        return {"record_id": name, "sequence": 1, "source": "constructed-not-performance", "text": text}
    events, edges, observed = [], [], {}
    for j in range(n):
        literal = f"CANARY_{j:04d}"
        for node in workflow["nodes"]:
            identity = f"{j}:{node}"
            events.append({"id": identity, "node": node, "actor": workflow["actors"][node][0],
                "family": "prepare" if node == "source" else node, "semantics": "literal_appearance", "subject": literal,
                "cues": [int(j % 3 == 0), 0], "cue_evidence": evidence("cue:" + identity)})
            if node == "source":
                observed[identity] = {**evidence(identity, literal if j % 2 else "Complete source without this value"),
                    "kind": "complete_message", "value": j % 2}
        for a, b in workflow["edges"]:
            edges.append({"source": f"{j}:{a}", "target": f"{j}:{b}", "basis": "declared_delivery", "evidence": evidence(f"route:{j}:{a}:{b}")})
    return ObservableEventDAG(workflow=workflow, events=events, edges=edges, observations=observed, prefix_sequence=2)


@pytest.mark.parametrize("n,layout,authority", [(1, "fork", "known"), (2, "fork", "root"), (3, "fork", "delegation"), (4, "chain", "opaque"), (5, "fork", "invalid")])
def test_conditional_law_risk_and_gradients_match_full_enumeration(n, layout, authority):
    sample, old = coupled_tool_fixture(n=n, layout=layout, authority=authority)
    old.theta += np.random.default_rng(9).normal(0, .3, len(old.theta))
    new = EliminatingDAGWorkflowRiskBN()
    new.theta = old.theta.copy()
    targets = [sample.queries[a]["event_id"] for a in reversed(sample.action_ids)]
    a = old.event_law.conditional(sample.event_dag, targets, old.theta[:25])
    b = new.event_law.conditional(sample.event_dag, targets, new.theta[:25])
    assert np.allclose(a["log_probabilities"], b["log_probabilities"], atol=1e-12)
    assert np.allclose(a["scores"], b["scores"], atol=1e-12)
    assert a["log_evidence"] == pytest.approx(b["log_evidence"], abs=1e-12)
    conditions = {sample.action_ids[0]: 1}
    a, b = old.predict(sample, derivatives=True, event_conditions=conditions), new.predict(sample, derivatives=True, event_conditions=conditions)
    assert a["probabilities"] == pytest.approx(b["probabilities"], abs=1e-12)
    for k in a["gradients"]:
        assert np.allclose(a["gradients"][k], b["gradients"][k], atol=1e-12)
    assert "enumerated_future_event_worlds" not in b
    for k, probability in b["probabilities"].items():
        assert evaluate(new.probability_graph(sample, output=k, event_conditions=conditions)) == pytest.approx(probability, abs=1e-12)


def test_missing_past_descendant_evidence_keeps_normalization_and_gradients():
    sample, old = missing_past_fixture()
    law = EliminatingEventLaw()
    theta = old.theta[:25] + np.random.default_rng(7).normal(0, .3, 25)
    targets = [sample.queries[a]["event_id"] for a in sample.action_ids]
    a = old.event_law.conditional(sample.event_dag, targets, theta)
    b = law.conditional(sample.event_dag, targets, theta)
    assert np.allclose(a["log_probabilities"], b["log_probabilities"], atol=1e-12)
    assert np.allclose(a["scores"], b["scores"], atol=1e-12)
    result = law.label_log_likelihood(sample.event_dag, {targets[0]: 1}, theta)
    for i in range(25):
        d = np.zeros(25); d[i] = 1e-5
        diff = (law.label_log_likelihood(sample.event_dag, {targets[0]: 1}, theta+d)["log_probability"] -
                law.label_log_likelihood(sample.event_dag, {targets[0]: 1}, theta-d)["log_probability"]) / 2e-5
        assert result["score"][i] == pytest.approx(diff, abs=2e-8)


def test_full_13_field_joint_survives_26_unobserved_events_and_matches_closed_form():
    dag = many_fields()
    theta = np.random.default_rng(3).normal(0, .6, 25)
    law = EliminatingEventLaw()
    targets = [f"{j}:publish" for j in range(13)]
    with pytest.raises(ValueError, match="capacity"):
        ObservableEventLaw().conditional(dag, targets, theta)
    result = law.conditional(dag, targets, theta)
    marginals = []
    for j in range(13):
        worker = dag.lookup[f"{j}:work"]
        publisher = dag.lookup[f"{j}:publish"]
        w = expit(law.design(worker, [j % 2]) @ theta)
        f0, f1 = [expit(law.design(publisher, [bit]) @ theta) for bit in (0, 1)]
        marginals.append((1-w)*f0+w*f1)
    expected = np.prod(np.where(result["states"] == 1, marginals, 1-np.asarray(marginals)), axis=1)
    assert len(expected) == 8192 and result["free_events"] == 26
    assert result["inference"]["peak_factor_variables"] == 13
    assert np.allclose(expected, np.exp(result["log_probabilities"]), atol=1e-14)
    all_absent = law.label_log_likelihood(dag, dict.fromkeys(targets, 0), theta)
    assert all_absent["log_probability"] == pytest.approx(np.log(expected[0]), abs=1e-12)
    assert all_absent["inference"]["labelled"]["peak_factor_variables"] <= 1


def test_joint_supervision_of_26_future_labels_has_no_26_bit_output_table():
    dag = many_fields()
    law = EliminatingEventLaw(max_factor_variables=3)
    theta = np.random.default_rng(4).normal(0, .5, 25)
    labels = {n: int(i % 3 == 0) for i, n in enumerate(dag.order) if n not in dag.observations}
    p = law.label_log_likelihood(dag, labels, theta)
    values = {**{n: o["value"] for n, o in dag.observations.items()}, **labels}
    expected = 0.
    for n in dag.order:
        if n in labels:
            z = law.design(dag.lookup[n], [values[v] for v in dag.parents[n]]) @ theta
            expected += labels[n]*z-np.logaddexp(0., z)
    assert p["log_probability"] == pytest.approx(expected, abs=1e-12)
    assert p["inference"]["labelled"]["peak_factor_variables"] == 0


def test_dense_factor_is_rejected_even_when_output_has_only_one_bit():
    dag = many_fields(6)
    spec = dag.specification()
    # All work events are same-literal records from the same functional stage;
    # one publisher receives all six of them. This is a dense conditional table.
    for e in spec["events"]:
        e["subject"] = "SHARED_LITERAL"
    for o in spec["observations"].values():
        o["text"] = "SHARED_LITERAL" if o["value"] else "No shared literal supplied"
    for j in range(1, 6):
        edge = copy.deepcopy(spec["edges"][1]); edge["source"] = f"{j}:work"; edge["target"] = "0:publish"
        spec["edges"].append(edge)
    wide = ObservableEventDAG(**spec)
    with pytest.raises(ValueError, match="factor width"):
        EliminatingEventLaw(max_factor_variables=3).conditional(wide, ["0:publish"], np.zeros(25))


def test_restore_preserves_inference_settings_and_rejects_tampering():
    sample, _ = coupled_tool_fixture()
    model = EliminatingDAGWorkflowRiskBN(max_factor_variables=5)
    saved = model.parameters()
    assert EliminatingDAGWorkflowRiskBN.restore(saved).predict(sample) == model.predict(sample)
    saved["max_factor_variables"] = 100
    with pytest.raises(ValueError): EliminatingDAGWorkflowRiskBN.restore(saved)


def test_known_authorization_union_retains_shared_dependencies_and_gradients():
    from test_relational_forecast_study import fixture
    sample = fixture()[1]
    model = EliminatingDAGWorkflowRiskBN()
    model.theta[model.names.index("event:publish:bias")] = -2
    model.theta[model.names.index("event:publish:parent_mean")] = 4
    fast = model.predict_known_authorization_union(sample, derivatives=True)
    full = model.predict(sample, derivatives=True)
    assert fast["probabilities"]["system:any"] == pytest.approx(full["probabilities"]["system:any"], abs=1e-12)
    assert fast["probabilities"]["system:any"] < .65  # independent .5/.5 would incorrectly produce .75
    assert np.allclose(fast["gradients"]["system:any"], full["gradients"]["system:any"], atol=1e-12)
    payment, _ = coupled_tool_fixture()
    with pytest.raises(ValueError): model.predict_known_authorization_union(payment)


def test_agentleak_full_field_projection_includes_direct_vault_and_distinct_clocks():
    initial = {"vault": {f"field-{i}": f"PRIVATE_VALUE_{i:03d}" for i in range(13)}, "request": "Write a notice.", "allowed_set": {"fields": []}}
    sample = project(initial, coordinator_text="Only PUBLIC facts", coordinator_record_id="coord-actual")
    dag = sample.event_dag
    assert len(dag.ids) == 52 and len(dag.ids)-len(dag.observations) == 26
    assert dag.parents["value-0:work"] == ("value-0:vault", "value-0:prepare")
    assert dag.prefix_sequence == 3 and sample.cutoff == 30
    assert dag.workflow["actors"]["prepare"] == dag.workflow["actors"]["publish"]
    labels = later_labels(initial, worker_text="PRIVATE_VALUE_000", worker_record_id="worker-actual", final_text="No references.", final_record_id="final-actual")
    model = EliminatingDAGWorkflowRiskBN()
    x, y = complete_trace_design(model.event_law, [dag], [labels])
    theta = np.random.default_rng(9).normal(0, .4, 25)
    a = complete_trace_objective(theta, x, y, 1, .01)
    b = trace_loss_gradient(model.event_law, theta, [dag], [labels], .01)
    assert a[0] == pytest.approx(b[0], abs=1e-12) and np.allclose(a[1], b[1], atol=1e-12)
    fitted = fit_event_traces([dag], [labels], model_class=EliminatingDAGWorkflowRiskBN)
    assert fitted.optimizer["objective_evaluation"] == "exact complete-observation local factorization"
    assert fitted.optimizer["success"]
    p = fitted.predict_known_authorization_union(sample)["probabilities"]["system:any"]
    q = fitted.predict(sample)["probabilities"]["system:any"]
    assert p == pytest.approx(q, abs=1e-12)


def test_complete_trace_fast_objective_rejects_partial_or_missing_history_labels():
    dag = many_fields(2)
    label = {"0:publish": {"record_id": "later", "sequence": 5, "source": "fixture", "text": "No value", "kind": "complete_message", "value": 0}}
    with pytest.raises(ValueError): complete_trace_design(EliminatingEventLaw(), [dag], [label])
