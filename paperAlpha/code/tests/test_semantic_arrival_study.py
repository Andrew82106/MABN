import copy

import numpy as np
import pytest

from test_semantic_arrival import fixture
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from run_task_cue_message_study import transform_probability, score_artifact, task_features
from verify_task_cue_message_study import independent_probability, independent_features


@pytest.mark.parametrize("late,n", [(False,1), (False,3), (False,5), (True,3)])
def test_independent_exhaustive_worlds_agree_with_joint_inference(late,n):
    sample, _ = fixture("literal-unique-value-00" if late else None, n)
    model = SemanticArrivalWorkflowRiskBN()
    model.theta[:65] = np.random.default_rng(184).normal(0, .5, 65)
    phase = "late" if late else "early"
    expected = independent_probability(model.parameters(), sample, phase)
    assert expected == pytest.approx(model.predict_known_authorization_union(sample)["probabilities"]["system:any"], abs=1e-12)
    assert np.array_equal(task_features({phase:sample},phase), independent_features(sample, phase, True))


def test_one_fixed_score_resolution_and_fingerprint_metadata():
    assert transform_probability(.321234560000001, 8) == transform_probability(.321234559999999, 8) == .32123456
    assert transform_probability(.321234561, None) == .321234561
    assert transform_probability(0., 8) == 0. and transform_probability(1.,8) == 1.
    original = {"model":"demo", "theta":[1.]}
    assert score_artifact(copy.deepcopy(original), {}) == original
    tagged = score_artifact(copy.deepcopy(original), {"probability_digits":8})
    assert tagged["score_transform"] == {"mode":"round_nearest_even_decimal", "probability_digits":8}


@pytest.mark.parametrize("p,d", [(float("nan"),8),(float("inf"),8),(-.1,8),(1.1,8),(.2,True),(.2,0),(.2,13)])
def test_bad_operational_probabilities_rejected(p,d):
    with pytest.raises(ValueError): transform_probability(p,d)
