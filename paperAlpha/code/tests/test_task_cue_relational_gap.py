"""A limitation witness, not a performance or success test.

The v1 message gate knows task-wide relevance fractions but not which relevant
field arrived. Keep this counterexample to guide the next relational model.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.agentleak_event_projection import project
from paperalpha_monitor.task_cue_message import TaskCueMessageWorkflowRiskBN
from paperalpha_runtime.task_field_semantics import context_from_initial, unknown_annotation, bind_task_cues
from run_task_cue_message_study import task_features


def test_v1_any_risk_cannot_distinguish_which_semantic_field_arrived():
    initial = {"vault": {"address": "literal-address-abc", "ssn": "literal-ssn-xyz"},
        "request": "Provide the address, not the SSN", "allowed_set": {"fields": []}}
    context, _ = context_from_initial(initial["request"], initial["vault"])
    annotation = unknown_annotation(context)
    annotation["fields"][0].update(internal_use="needed", outward_form="verbatim")
    annotation["fields"][1].update(internal_use="not_needed", outward_form="not_requested")
    samples = []
    for value in initial["vault"].values():
        original = project(initial, coordinator_text="Please answer the request", coordinator_record_id="coordinator",
            worker_text=value, worker_record_id="worker")
        samples.append(bind_task_cues(original, context, annotation))
    model = TaskCueMessageWorkflowRiskBN()
    model.theta[:55] = np.random.default_rng(423).normal(0, .4, 55)
    a, b = (model.predict_known_authorization_union(s)["probabilities"]["system:any"] for s in samples)
    assert a == pytest.approx(b, abs=1e-13)
    # The fair classifier representation DOES retain this interaction.
    assert not np.array_equal(task_features({"late": samples[0]}, "late"), task_features({"late": samples[1]}, "late"))


def test_complete_late_message_or_uses_exact_gate_and_zero_local_gradient():
    initial = {"vault": {"address": "literal-address-abc", "ssn": "literal-ssn-xyz"},
        "request": "Write a summary", "allowed_set": {"fields": []}}
    sample = project(initial, coordinator_text="literal-address-abc", coordinator_record_id="c",
        worker_text="literal-ssn-xyz", worker_record_id="w")
    model = TaskCueMessageWorkflowRiskBN()
    model.theta[:55] = np.random.default_rng(62).normal(0, .7, 55)
    result = model.predict_known_authorization_union(sample, derivatives=True)
    members = [q["event_id"] for q in sample.queries.values()]
    q = model.event_law.message_summary(sample.event_dag, members, model.theta[:55])["any_appearance_probability"]
    assert result["probabilities"]["system:any"] == q
    assert result["inference"]["exact_message_gate_identity"] is True
    assert np.array_equal(result["gradients"]["system:any"][:25], np.zeros(25))
    model.theta[:25] += 1.7
    assert model.predict_known_authorization_union(sample)["probabilities"]["system:any"] == q
