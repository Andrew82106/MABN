import copy
import json

import numpy as np
import pytest

from test_task_cue_message import fixture
from paperalpha_runtime.task_field_semantics import bind_task_cues
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN
from paperalpha_monitor.task_cue_message import TaskCueMessageWorkflowRiskBN
from run_task_cue_message_study import task_features, verify_request_replay
from paperalpha_runtime.task_field_semantics import request_payload
from verify_task_cue_message_study import independent_features, independent_probability, audit_cue_binding


def test_api_replay_accepts_json_key_order_but_rejects_content_or_setting_change():
    _, context, _, _, _ = fixture()
    sorted_context = json.loads(json.dumps(context, sort_keys=True))
    payload = request_payload(sorted_context)
    assert payload != request_payload(context)
    verify_request_replay(context, payload)
    broken = copy.deepcopy(payload); broken["model"] = "other"
    with pytest.raises(ValueError): verify_request_replay(context, broken)
    broken = copy.deepcopy(payload); altered = dict(sorted_context, outcome=1)
    broken["input"][0]["content"] = json.dumps(altered)
    with pytest.raises(ValueError): verify_request_replay(context, broken)


@pytest.mark.parametrize("phase", ["early", "late"])
def test_task_histogram_matches_independent_and_ignores_labels(phase):
    raw, context, _, annotation, _ = fixture(phase == "late")
    sample = bind_task_cues(raw, context, annotation)
    row = {phase: sample, "y": 1, "vertical": "unused", "template_id": "unused", "id": "unused"}
    features = task_features(row, phase)
    assert len(features) == 80 and features[8:].sum() == 2
    assert np.array_equal(features, independent_features(sample, phase, True))
    changed = {**row, "y": 0, "vertical": "other", "template_id": "other", "id": "other"}
    assert np.array_equal(features, task_features(changed, phase))
    assert audit_cue_binding(raw, sample, annotation) == 4


@pytest.mark.parametrize("phase", ["early", "late"])
@pytest.mark.parametrize("cls", [MessageHurdleWorkflowRiskBN, TaskCueMessageWorkflowRiskBN])
def test_independent_count_polynomial_matches_production(phase, cls):
    raw, context, _, annotation, _ = fixture(phase == "late")
    sample = bind_task_cues(raw, context, annotation)
    model = cls(); model.theta[:model.event_count] = np.random.default_rng(912).normal(0, .7, model.event_count)
    expected = independent_probability(model.parameters(), sample, phase)
    assert expected == pytest.approx(model.predict_known_authorization_union(sample)["probabilities"]["system:any"], abs=1e-12)
