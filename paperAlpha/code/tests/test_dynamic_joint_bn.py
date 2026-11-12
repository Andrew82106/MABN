import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.dynamic_joint_bn import (
    ADOPT, AUTHORIZE, CONFIDENCE, HEAD, N_PARAMETERS, ROLES, TRANSMIT,
    DynamicJointBN, compile_trace, state_bits, transition, transition_pairs,
)


def event(sequence, actor, kind, **kw):
    return {"sequence": sequence, "actor": actor, "event_type": kind, **kw}


def chain():
    return [event(0, "a", "input_received", data_class="untrusted", source_id="doc"),
            event(1, "a", "message_sent", target="b", data_class="untrusted", edge_confidence=.8),
            event(2, "b", "tool_call", tool="shell.exec", permission_mismatch=True)]


def transparent_model():
    model = DynamicJointBN()
    model.theta[1] = np.log(.4 / .6)
    model.theta[ADOPT:AUTHORIZE] = 24
    model.theta[TRANSMIT:CONFIDENCE] = 24
    return model


def marginal(model, trace, name):
    index = next(i for i, variable in enumerate(trace.variables) if variable["id"] == name)
    distribution = model._forward(model.theta, trace)[2]
    return float(distribution @ state_bits(trace.n_variables)[:, index])


def test_transition_preserves_mass_and_reverse_derivative():
    distribution = np.array([.1, .2, .3, .4])
    source, target = transition_pairs(2, 1, 0)
    after = transition(distribution, .3, source, target)
    np.testing.assert_allclose(after, [.1, .14, .3, .46])
    assert after.sum() == pytest.approx(1)
    assert (after >= 0).all()


def test_diamond_preserves_common_origin_dependence():
    events = chain()[:1] + [
        event(1, "a", "message_sent", target="b", data_class="untrusted"),
        event(2, "a", "message_sent", target="c", data_class="untrusted"),
        event(3, "b", "message_sent", target="d", data_class="untrusted"),
        event(4, "c", "message_sent", target="d", data_class="untrusted"),
    ]
    compiled = compile_trace({"events": events})
    model = transparent_model()
    assert marginal(model, compiled, "d") == pytest.approx(.4, abs=1e-8)
    # An independent-marginal OR would instead count the same origin twice.
    assert 1 - (1 - marginal(model, compiled, "b")) * (1 - marginal(model, compiled, "c")) == pytest.approx(.64, abs=1e-8)


def test_same_material_entering_two_agents_is_one_shared_latent_origin():
    events = [chain()[0], event(1, "b", "input_received", data_class="untrusted", source_id="doc"),
              event(2, "a", "message_sent", target="c", data_class="untrusted"),
              event(3, "b", "message_sent", target="c", data_class="untrusted")]
    compiled = compile_trace({"events": events})
    assert sum(v["kind"] == "origin_hypothesis" for v in compiled.variables) == 1
    assert marginal(transparent_model(), compiled, "c") == pytest.approx(.4, abs=1e-8)


def test_normative_dag_does_not_invent_risk_transmission():
    events = [chain()[0], event(1, "b", "tool_call", tool="search.read")]
    compiled = compile_trace({"events": events}, workflow={"nodes": ["a", "b"], "edges": [["a", "b"]]})
    assert marginal(transparent_model(), compiled, "b") == 0


def test_message_before_input_does_not_transmit_future_source():
    events = [event(0, "a", "message_sent", target="b", data_class="untrusted"),
              event(1, "a", "input_received", data_class="untrusted")]
    compiled = compile_trace({"events": events})
    assert marginal(transparent_model(), compiled, "b") == 0


@pytest.mark.parametrize("workflow", [None, {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}])
def test_future_events_and_actor_inventory_cannot_change_prefix_predictions(workflow):
    original = chain()
    appended = original + [event(3, "c", "input_received", data_class="untrusted", source_id="new"),
                           event(4, "c", "tool_call", tool="payments.write")]
    model = DynamicJointBN()
    actual = model.predict([compile_trace({"agents": ["future", "secret"], "events": appended}, workflow=workflow)])[0]
    for length in range(1, len(original) + 1):
        prefix = compile_trace({"events": original[:length]}, workflow=workflow)
        np.testing.assert_allclose(model.predict([prefix])[0], actual[:length], atol=1e-12, rtol=1e-12)


def test_runtime_only_masks_normative_derived_edge_kind():
    original, altered = chain(), chain()
    altered[1]["edge_kind"] = "unexpected"
    model = DynamicJointBN()
    np.testing.assert_equal(*model.predict([compile_trace({"events": original}), compile_trace({"events": altered})]))


def test_permission_masking_affects_all_local_and_global_paths():
    first, second = chain(), chain()
    second[-1]["permission_mismatch"] = False
    model = DynamicJointBN()
    np.testing.assert_equal(*model.predict([compile_trace({"events": events}, permission_mode="masked") for events in (first, second)]))


def test_roles_are_explicit_and_select_different_local_cpts():
    model = transparent_model()
    model.theta[ADOPT + ROLES.index("reader") * 3 + 1] = -20
    reader = compile_trace({"events": chain()[:1]}, roles={"a": "reader"})
    executor = compile_trace({"events": chain()[:1]}, roles={"a": "executor"})
    assert marginal(model, reader, "a") < 1e-8
    assert marginal(model, executor, "a") > .39


def test_grouped_workflow_instantiates_nodes_not_a_latent_per_alias():
    workflow = {"nodes": ["collect", "act"], "edges": [["collect", "act"]],
                "roles": {"collect": "reader", "act": "executor"},
                "agent_to_node": {"a": "collect", "a2": "collect", "b": "act"}}
    events = chain()[:2]
    events.insert(1, event(1, "a2", "input_received", data_class="untrusted", source_id="doc"))
    events[-1]["sequence"] = 2
    compiled = compile_trace({"events": events}, workflow=workflow)
    assert sum(v["kind"] == "workflow_node" for v in compiled.variables) == 2
    adoption = [op for step in compiled.steps for op in step.operations if op.evidence["template"] == "adoption"]
    assert len(adoption) == 1
    assert compiled.steps[-1].operations[0].evidence["expected"] == "expected"


def test_retry_and_matching_action_are_not_new_latent_injections():
    events = chain()
    events[-1].update(event_type="tool_proposed", event_id="proposal", action_id="action")
    events += [copy.deepcopy(events[-1]), event(3, "b", "tool_call", tool="shell.exec", action_id="action")]
    compiled = compile_trace({"events": events})
    model = DynamicJointBN()
    scores = model.predict([compiled])[0]
    assert scores[2] == scores[3]
    assert sum(op.evidence["template"] == "authorization" for step in compiled.steps for op in step.operations) == 1


def test_graph_size_limit_is_explicit_not_silent_approximation():
    with pytest.raises(ValueError, match="exact joint state exceeds"):
        compile_trace({"events": chain()}, max_variables=2)


def test_declared_node_cannot_be_remapped_to_another_node():
    workflow = {"nodes": ["a", "b"], "edges": [["a", "b"]], "agent_to_node": {"a": "b"}}
    with pytest.raises(ValueError, match="cannot alias"):
        compile_trace({"events": chain()}, workflow=workflow)


@pytest.mark.parametrize("limit", [True, 0, -1, 2.5])
def test_invalid_state_limit_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        compile_trace({"events": chain()}, max_variables=limit)


@pytest.mark.parametrize("prefix_training", [True, False])
def test_exact_end_to_end_gradient_matches_finite_differences(prefix_training):
    trace = compile_trace({"events": chain()}, workflow={"nodes": ["a", "b"], "edges": [["a", "b"]]})
    model = DynamicJointBN()
    theta = np.linspace(-1.2, 1.1, N_PARAMETERS)
    loss, gradient = model.loss_gradient(theta, [trace], np.array([1]), prefix_training)
    numeric = approx_derivative(lambda values: model.loss_gradient(values, [trace], np.array([1]), prefix_training)[0], theta).ravel()
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-5)
    assert np.linalg.norm(gradient[:TRANSMIT]) > 1e-5
    assert np.linalg.norm(gradient[TRANSMIT:HEAD]) > 1e-5
    assert np.linalg.norm(gradient[HEAD:]) > 1e-5


def test_explanation_is_exact_probability_mixture():
    model = DynamicJointBN()
    trace = compile_trace({"events": chain()})
    explanation = model.explain(trace)
    assert explanation["mass_sum"] == pytest.approx(1)
    total = sum(row["risk_contribution"] for row in explanation["system_mixture_terms"])
    total += explanation["remaining_risk_contribution"]
    assert total == pytest.approx(model.predict([trace])[0][-1])


def test_small_joint_fit_reduces_prefix_loss():
    safe = [event(0, "a", "input_received", data_class="clean"),
            event(1, "a", "tool_call", tool="shell.exec", permission_mismatch=False)]
    traces = [compile_trace({"events": events}) for events in (chain(), safe)]
    y = np.asarray([1, 0])
    model = DynamicJointBN(max_iter=10)
    before = model.loss_gradient(model.theta, traces, y)[0]
    model.fit(traces, y)
    assert model.loss_gradient(model.theta, traces, y)[0] < before
