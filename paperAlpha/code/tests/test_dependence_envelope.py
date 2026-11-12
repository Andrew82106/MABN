"""Analytic bounds, complete authority coupling and original BN agreement."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.dependence_envelope import FiniteJointRiskEnvelope, dependence_envelope
from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN
from paperalpha_runtime.exclusive_purchase import cases, prefix
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input


def table(p, q):
    states = np.asarray(list(product((0, 1), repeat=2)))
    mass = np.asarray([(p if a else 1-p) * (q if b else 1-q) for a, b in states])
    return FiniteJointRiskEnvelope(states, mass, ["a", "b"],
                                  {"AND": states.prod(axis=1), "OR": states.max(axis=1)})


@pytest.mark.parametrize("p,q", [(0., 0.), (1., 1.), (.5, .5), (.2, .7), (.8, .7), (0., .9), (1e-6, .9)])
def test_analytic_frechet_and_or_bounds_and_dual_witness(p, q):
    model = table(p, q)
    for query, expected in (("AND", (max(0, p+q-1), min(p, q))),
                            ("OR", (max(p, q), min(1, p+q)))):
        result = model.bound(query, include_witness=True)
        assert (result["lower"], result["upper"]) == pytest.approx(expected, abs=1e-9)
        for sign, w in zip((1, -1), result["endpoint_witnesses"]):
            x, y = np.asarray(w["mass"]), np.asarray(w["equality_dual"])
            assert np.allclose(model.a @ x, model.b, atol=1e-9)
            assert np.max(model.a.T @ y - sign * model.events[query]) <= 1e-9
            assert sign * model.events[query] @ x == pytest.approx(model.b @ y, abs=1e-9)


def test_same_marginals_allow_opposite_synchronization():
    result = table(.5, .5).bound("AND", include_witness=True)
    assert result["reference"] == .25
    assert result["lower"] == 0 and result["upper"] == .5
    assert result["endpoint_witnesses"][0]["mass"] == pytest.approx([0, .5, .5, 0])
    assert result["endpoint_witnesses"][1]["mass"] == pytest.approx([.5, 0, 0, .5])


@pytest.mark.parametrize("complete,revoked", [(True, False), (False, False), (True, True)])
def test_workflow_queries_preserve_original_point_and_full_authority_marginal(complete, revoked):
    case = cases()[0]; case["revoke_parent_after_proposals"] = revoked
    value = prefix(case, json.dumps({"priority": "balanced", "work_order": "Choose supplies"}))
    sample = compile_attribute_input(value, authority_complete=complete)
    model = AttributeChoiceWorkflowRiskBN()
    model.theta = np.random.default_rng(571).normal(0, .5, len(model.theta))
    before = model.parameters()
    report = dependence_envelope(model, sample, include_witness=True)
    point = model.predict(sample)["probabilities"]
    problem = report["witness_problem"]
    assert model.parameters() == before
    for name, row in report["queries"].items():
        assert row["reference"] == pytest.approx(point[name], abs=1e-12)
        assert row["lower"] - 1e-8 <= point[name] <= row["upper"] + 1e-8
        states = np.asarray(problem["states"])
        for w in row["endpoint_witnesses"]:
            for j in range(states.shape[1]):
                for k in np.unique(states[:, j]):
                    ix = states[:, j] == k
                    assert np.asarray(w["mass"])[ix].sum() == pytest.approx(
                        np.asarray(problem["reference_mass"])[ix].sum(), abs=1e-8)
    if not complete:
        assert report["blocks"]["authority:joint_facts"]["kind"] == "joint_authority"
    assert not report["is_confidence_interval"] and not report["proves_safety"]


def test_equivalence_requires_same_other_worlds_and_same_query_truth():
    states = np.asarray(list(product((0, 1, 2), (0, 1))))
    e = FiniteJointRiskEnvelope(states, np.ones(6)/6, ["choice", "other"],
                               {"risk": (states[:, 0] > 0) * states[:, 1]})
    assert e.equivalence_classes("risk")["choice"] == [[0], [1, 2]]
    # Missing contexts must not be assumed safe or equivalent.
    e = FiniteJointRiskEnvelope(states[:5], np.ones(5)/5, ["choice", "other"],
                               {"risk": np.zeros(5)})
    assert e.equivalence_classes("risk")["choice"] == [[0, 1], [2]]


def test_three_blocks_union_is_not_sum_of_channel_risks():
    states = np.asarray(list(product((0, 1), repeat=3)))
    e = FiniteJointRiskEnvelope(states, np.ones(8)/8, ["a", "b", "c"],
                               {"union": states.max(axis=1)})
    r = e.bound("union")
    assert (r["lower"], r["reference"], r["upper"]) == pytest.approx((.5, .875, 1))


@pytest.mark.parametrize("kind", ["mass", "duplicate", "negative", "capacity", "event"])
def test_invalid_problem_rejected(kind):
    states = np.asarray([[0], [1]]); mass = [.5, .5]; events = {"risk": [0, 1]}; cap = 4096
    if kind == "mass": mass = [.5, .6]
    if kind == "duplicate": states[1] = 0
    if kind == "negative": mass = [-.5, 1.5]
    if kind == "capacity": cap = 1
    if kind == "event": events = {"risk": [0, np.nan]}
    with pytest.raises(ValueError): FiniteJointRiskEnvelope(states, mass, ["a"], events, max_worlds=cap)
