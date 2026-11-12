"""Independent feature/likelihood checks and the observable confounding witness."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.exclusive_purchase import cases, prefix
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input
from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN, attribute_rows
from evaluate_choice_attributes import reference_features, reference_objective, reference_worlds, support_diagnosis


def rows():
    out = []
    for priority in ("economy", "balanced", "quality"):
        pre = prefix(cases()[0], json.dumps({"priority": priority, "work_order": "Constructed contract test."}))
        s = compile_attribute_input(pre)
        selected = {g["action_ids"][0 if priority == "economy" else 1] for g in s.choice_groups}
        out.append({"sample": s, "prefix": pre, "x": reference_features(pre, s),
                    "label": {"dispatch": {a: int(a in selected) for a in s.action_ids}}})
    return out


def test_independent_features_likelihood_and_joint_law():
    data = rows(); m = AttributeChoiceWorkflowRiskBN(regularization=.03)
    m.theta[:10] = np.random.default_rng(747).normal(0, .3, 10)
    actual = m.loss_gradient(m.theta, [r["sample"] for r in data], [{"dispatch": r["label"]["dispatch"]} for r in data])
    expected = reference_objective(m.theta[:10], data, .03)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert np.allclose(actual[1][:10], expected[1], atol=1e-12)
    for r in data:
        assert np.array_equal(r["x"], attribute_rows(r["sample"])[0])
        states, p = reference_worlds(r["sample"], r["x"], m.theta[:10])
        observed, logp, _ = m._occurrence_worlds(r["sample"], m.theta)
        assert np.array_equal(states, observed) and np.allclose(p, np.exp(logp), atol=1e-12)


def test_no_alias_does_not_mean_separate_cost_warranty_identification():
    data = rows(); report = support_diagnosis(data)
    assert report["still_identical_alternative_pairs"] == 0
    assert report["full_choice_design_rank"] == 9
    assert report["conditional_on_dispatch_contrast_rank"] == 3
    assert report["observed_abstentions"] == 0 and report["max_null_contrast"] < 1e-12
    theta = np.random.default_rng(723).normal(0, .4, 10)
    shifted = theta + np.asarray(report["conditional_cost_warranty_null_direction"])
    for r in data:
        for g in r["sample"].choice_groups:
            x = r["x"][[r["sample"].action_ids.index(a) for a in g["action_ids"]]]
            # Conditional on dispatch, preference shifts exactly cancel. The
            # outside-option probability need NOT be invariant and is not claimed.
            a, b = np.exp(x @ theta), np.exp(x @ shifted)
            assert np.allclose(a / a.sum(), b / b.sum(), atol=1e-12)
