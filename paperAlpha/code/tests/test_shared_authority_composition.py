from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit, logit

from test_shared_authority_bn import bundle
from paperalpha_monitor.action_mechanism_knowledge import MechanismFactor
from paperalpha_monitor.shared_authority_graph import SharedAuthorityTrace
from paperalpha_monitor.shared_authority_joint_bn import SharedAuthorityJointBN
from paperalpha_monitor.capability_action_bn import CAP_HEAD


def test_shared_authority_remains_joint_with_other_knowledge_fragment():
    model, data = SharedAuthorityJointBN(), bundle(True)
    model.theta[model.base_count] = logit(.2)
    model.theta += np.random.default_rng(1911).normal(0, .1, len(model.theta))
    final = data.trace.steps[-1]
    indices = model.registry.numeric_slices["declared_bound"]
    values = (1., *([.3] * (len(indices) - 1)))
    factor = MechanismFactor(final.base.dispatches[0].action_id, final.base.dispatches[0].node, 1,
        "declared_bound", indices, values, {"scope": "Explicit component fixture, not observed API quantity evidence"})
    final = replace(final, mechanism_factors=(*final.mechanism_factors, factor))
    trace = SharedAuthorityTrace(replace(data.trace, steps=(*data.trace.steps[:-1], final)), data.graph)
    result = model.forward(trace, derivatives=True)
    p = expit(model.theta[model.base_count])
    q = expit(model.theta[list(indices)] @ values)
    head, _ = model.registry.head_indices["declared_bound"]
    bias = final.base.observables @ model.theta[CAP_HEAD:CAP_HEAD + 15]
    expected = 0.
    for shared in (0, 1):
        for numeric in (0, 1):
            weight = (p if shared else 1 - p) * (q if numeric else 1 - q)
            # Two actions share one failure fact, so fraction=pair=shared.
            expected += weight * expit(bias + model.theta[CAP_HEAD + 20] * shared
                + model.theta[head] * numeric + sum(model.theta[-2:]) * shared)
    assert result["probabilities"][-1] == pytest.approx(expected, abs=3e-15)
    assert abs(result["gradient"][-1, model.base_count]) > 1e-6
    assert np.max(np.abs(result["gradient"][-1, list(indices)])) > 1e-6


def test_frequency_weights_equal_explicit_repetition_not_independent_sample_claim():
    model = SharedAuthorityJointBN()
    a, b = bundle(True), bundle(False)
    weighted = model.loss_gradient(model.theta, [a, b], [0, 1], weights=[3, 2])
    repeated = model.loss_gradient(model.theta, [a, a, a, b, b], [0, 0, 0, 1, 1])
    assert weighted[0] == pytest.approx(repeated[0], abs=2e-15)
    np.testing.assert_allclose(weighted[1], repeated[1], atol=2e-15)


@pytest.mark.parametrize("weights", [[0], [-1], [float("nan")], [1, 2]])
def test_invalid_frequency_weights_are_rejected(weights):
    model = SharedAuthorityJointBN()
    with pytest.raises(ValueError, match="weights"):
        model.loss_gradient(model.theta, [bundle()], [0], weights=weights)
