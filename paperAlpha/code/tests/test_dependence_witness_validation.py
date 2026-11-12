"""Independent witness audit rejects malformed rather than silently zipped data."""
import copy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from diagnose_dependence_envelope import inspect_witness
from paperalpha_monitor.dependence_envelope import FiniteJointRiskEnvelope


@pytest.fixture
def certificate():
    states = np.asarray(list(product((0, 1), repeat=2)))
    e = FiniteJointRiskEnvelope(states, [.21, .49, .09, .21], ["a", "b"],
                               {"AND": states.prod(axis=1), "OR": states.max(axis=1)})
    return {"queries": {q: e.bound(q, include_witness=True) for q in e.events},
            "witness_problem": {"states": e.states.tolist(), "blocks": list(e.blocks),
                "reference_mass": e.mass.tolist(), "events": {q: v.tolist() for q, v in e.events.items()},
                "constraint_rows": e.constraints, "marginals": e.b.tolist()}}


def test_valid_and_tiny_residual_certificates(certificate):
    assert inspect_witness(certificate) < 1e-12
    changed = copy.deepcopy(certificate)
    changed["queries"]["AND"]["endpoint_witnesses"][0]["mass"][0] += 1e-11
    assert 0 < inspect_witness(changed) < 1e-8


@pytest.mark.parametrize("target", ["mass", "dual", "reference", "event", "marginal", "bound"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_witness_rejected(certificate, target, bad):
    p = certificate["witness_problem"]
    w = certificate["queries"]["AND"]["endpoint_witnesses"][0]
    if target == "mass": w["mass"] = [bad] * len(w["mass"])
    if target == "dual": w["equality_dual"] = [bad] * len(w["equality_dual"])
    if target == "reference": p["reference_mass"][0] = bad
    if target == "event": p["events"]["AND"][0] = bad
    if target == "marginal": p["marginals"][0] = bad
    if target == "bound": certificate["queries"]["AND"]["lower"] = bad
    with pytest.raises(ValueError): inspect_witness(certificate)


@pytest.mark.parametrize("target", ["mass", "dual", "reference", "event", "marginal", "endpoints", "state"])
@pytest.mark.parametrize("extra", [False, True])
def test_mismatched_lengths_rejected(certificate, target, extra):
    p = certificate["witness_problem"]
    w = certificate["queries"]["AND"]["endpoint_witnesses"][0]
    values = {"mass": w["mass"], "dual": w["equality_dual"], "reference": p["reference_mass"],
        "event": p["events"]["AND"], "marginal": p["marginals"],
        "endpoints": certificate["queries"]["AND"]["endpoint_witnesses"], "state": p["states"][0]}[target]
    if extra: values.append(copy.deepcopy(values[-1]))
    else: values.pop()
    with pytest.raises(ValueError): inspect_witness(certificate)


@pytest.mark.parametrize("corruption", ["duplicate_state", "negative_state", "float_state", "bool_state",
    "empty_states", "duplicate_block", "negative_reference", "unnormalized_reference", "bad_event",
    "different_queries", "wrong_constraint", "wrong_marginal", "wrong_point", "wrong_width", "wrong_endpoint",
    "wrong_mass", "wrong_dual", "overflow_dual"])
def test_structural_and_numerical_corruption_rejected(certificate, corruption):
    p = certificate["witness_problem"]; row = certificate["queries"]["AND"]
    if corruption == "duplicate_state": p["states"][0] = p["states"][1][:]
    if corruption == "negative_state": p["states"][0][0] = -1
    if corruption == "float_state": p["states"][0][0] = .5
    if corruption == "bool_state": p["states"][0][0] = False
    if corruption == "empty_states": p["states"] = []
    if corruption == "duplicate_block": p["blocks"] = ["a", "a"]
    if corruption == "negative_reference": p["reference_mass"] = [-.1, .8, .1, .2]
    if corruption == "unnormalized_reference": p["reference_mass"][0] += .1
    if corruption == "bad_event": p["events"]["AND"][0] = -1
    if corruption == "different_queries": del p["events"]["OR"]
    if corruption == "wrong_constraint": p["constraint_rows"][1]["category"] = 2
    if corruption == "wrong_marginal": p["marginals"][0] += .1
    if corruption == "wrong_point": row["reference"] += .1
    if corruption == "wrong_width": row["width"] += .1
    if corruption == "wrong_endpoint": row["lower"] += .1; row["width"] -= .1
    if corruption == "wrong_mass": row["endpoint_witnesses"][0]["mass"] = [1., 0., 0., 0.]
    if corruption == "wrong_dual": row["endpoint_witnesses"][0]["equality_dual"][0] += 1.
    if corruption == "overflow_dual": row["endpoint_witnesses"][0]["equality_dual"] = [1e308] * len(p["marginals"])
    with pytest.raises(ValueError): inspect_witness(certificate)
