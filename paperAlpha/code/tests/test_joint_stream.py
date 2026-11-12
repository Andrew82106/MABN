import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor import DynamicJointBN, JointRiskStream
import paperalpha_monitor.dynamic_joint_bn as core


def event(sequence, actor, kind, **kwargs):
    return {"sequence": sequence, "actor": actor, "event_type": kind, **kwargs}


def events():
    return [event(0, "a", "input_received", data_class="untrusted", source_id="doc"),
            event(1, "a", "message_sent", target="b", data_class="untrusted", edge_confidence=.7),
            event(2, "a", "message_sent", target="c", data_class="unknown"),
            event(3, "b", "delegation", target="d", data_class="clean"),
            event(4, "c", "message_sent", target="d", data_class="untrusted"),
            event(5, "d", "tool_proposed", tool="shell.exec", action_id="run", permission_mismatch=False),
            event(6, "d", "tool_call", tool="shell.exec", action_id="run", permission_mismatch=True)]


@pytest.mark.parametrize("mode", [{}, {"use_messages": False}, {"use_latent_head": False}])
@pytest.mark.parametrize("context", [{}, {"permission_mode": "masked"}, {
    "workflow": {"nodes": ["a", "b", "c", "d"], "edges": [["a", "b"], ["b", "d"]]},
    "roles": {"a": "reader", "b": "reasoner", "c": "reviewer", "d": "executor"},
    "allowed_tools": {"d": ["shell.exec"]},
}])
def test_online_matches_batch_probabilities_and_exact_marginals(mode, context):
    model = DynamicJointBN(**mode)
    model.theta = np.random.default_rng(712).normal(size=core.N_PARAMETERS)
    stream = JointRiskStream(model, **context)
    rows = events()
    scores = []
    for i, row in enumerate(rows):
        result = stream.push(row)
        scores.append(result["risk_probability"])
        prefix = core.compile_trace({"events": rows[:i + 1]}, **context)
        expected = model.explain(prefix)
        actual = stream.explain()
        assert actual["risk_probability"] == pytest.approx(expected["risk_probability"], abs=1e-12)
        np.testing.assert_allclose([v["marginal"] for v in actual["variables"]], [v["marginal"] for v in expected["variables"]], atol=1e-12)
        total = sum(v["risk_contribution"] for v in actual["system_mixture_terms"]) + actual["remaining_risk_contribution"]
        assert total == pytest.approx(actual["risk_probability"], abs=1e-12)
    np.testing.assert_allclose(scores, model.predict([core.compile_trace({"events": rows}, **context)])[0], atol=1e-12)


def test_online_never_calls_batch_compile_or_replays_model_forward(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("online updates must not replay batch compilation/forward")
    monkeypatch.setattr(core, "compile_trace", forbidden)
    monkeypatch.setattr(DynamicJointBN, "_forward", forbidden)
    stream = JointRiskStream(DynamicJointBN())
    for row in events():
        assert 0 <= stream.push(row)["risk_probability"] <= 1


def test_undeclared_future_nodes_and_sources_do_not_exist_in_explanation():
    stream = JointRiskStream(DynamicJointBN())
    stream.push(events()[0])
    assert {v["id"] for v in stream.explain()["variables"]} == {"a", "source:doc"}
    assert stream.explain()["state_count"] == 4
    stream.push(events()[1])
    assert stream.explain()["state_count"] == 8


def test_context_and_parameters_are_frozen_at_episode_start():
    model = DynamicJointBN()
    context = {"roles": {"a": "reader"}, "allowed_tools": {"b": ["shell.exec"]}}
    stream = JointRiskStream(model, **context)
    reference = JointRiskStream(copy.deepcopy(model), **copy.deepcopy(context))
    context["roles"]["a"] = "executor"
    context["allowed_tools"]["b"].clear()
    model.theta[:] = 7
    for row in events():
        assert stream.push(row) == reference.push(row)


def test_duplicates_do_not_reinject_count_or_restart_first_alarm():
    stream = JointRiskStream(DynamicJointBN(), threshold=0.)
    first = {**events()[0], "event_id": "event-0"}
    one = stream.push(first)
    second = stream.push(copy.deepcopy(first))
    assert second["duplicate_event"]
    assert second["accepted_event_count"] == one["accepted_event_count"] == 1
    assert second["first_alarm_sequence"] == one["first_alarm_sequence"] == 0
    assert second["risk_probability"] == one["risk_probability"]
    assert stream.explain()["latest_event_transitions"] == []


@pytest.mark.parametrize("bad", [
    event(1, "a", "input_received", source_id=[]),
    event(1, "a", "message_sent", target="b", data_class=[]),
    event(1, "a", "message_sent", target="b", edge_confidence=-1),
    event(1, "a", "tool_call", tool="shell.exec", permission_mismatch="yes"),
    event(1, "a", "message_sent", target="b", payload={"label": 1}),
    event(0, "a", "message_sent", target="b"),
])
def test_rejected_event_does_not_change_probability_or_compiler_state(bad):
    stream = JointRiskStream(DynamicJointBN())
    stream.push(events()[0])
    before = stream.explain()
    with pytest.raises(ValueError):
        stream.push(bad)
    assert stream.explain() == before
    assert stream.push(events()[1])["accepted_event_count"] == 2


def test_state_capacity_failure_is_atomic_and_never_reported_as_safety():
    stream = JointRiskStream(DynamicJointBN(), max_variables=2)
    stream.push(events()[0])
    before = stream.explain()
    with pytest.raises(ValueError, match="event was not accepted"):
        stream.push(events()[1])
    assert stream.explain() == before
    # Same sequence can still be accepted for an event within the capacity.
    stream.push(event(1, "a", "tool_call", tool="search.read"))
    assert stream.snapshot()["accepted_event_count"] == 2


def test_no_event_has_no_prediction_and_absent_threshold_is_not_no_alarm():
    stream = JointRiskStream(DynamicJointBN())
    with pytest.raises(ValueError, match="no event observed"):
        stream.snapshot()
    assert stream.push(events()[0])["alarm"] is None


def test_returned_explanations_and_input_mutation_do_not_change_state():
    stream = JointRiskStream(DynamicJointBN())
    row = events()[0]
    stream.push(row)
    row["actor"] = "changed"
    explanation = stream.explain()
    explanation["variables"][0]["id"] = "changed-again"
    explanation["observed_features"]["bias"] = 99
    assert stream.explain()["variables"][0]["id"] == "a"
    assert stream.explain()["observed_features"]["bias"] == 1


@pytest.mark.parametrize("threshold", [True, -1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_is_not_accepted(threshold):
    with pytest.raises(ValueError, match="threshold"):
        JointRiskStream(DynamicJointBN(), threshold=threshold)
