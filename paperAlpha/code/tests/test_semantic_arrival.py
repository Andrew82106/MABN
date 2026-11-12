import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.agentleak_event_projection import project, later_labels
from paperalpha_monitor.task_cue_message import TaskCueMessageWorkflowRiskBN, SemanticArrivalWorkflowRiskBN
from paperalpha_monitor.message_hurdle import compile_message_training, message_objective
from paperalpha_runtime.task_field_semantics import context_from_initial, unknown_annotation, bind_task_cues
from scalar_binary_graph import evaluate


def fixture(worker=None, n=2):
    initial = {"vault": {f"field-{j:02d}": f"literal-unique-value-{j:02d}" for j in range(n)},
        "request": "Produce a concise answer", "allowed_set": {"fields": []}}
    context, _ = context_from_initial(initial["request"], initial["vault"])
    annotation = unknown_annotation(context)
    for j, item in enumerate(annotation["fields"]):
        item["internal_use"] = ("needed", "not_needed", "unclear")[j % 3]
        item["outward_form"] = ("verbatim", "not_requested", "unclear")[j % 3]
    args = {"coordinator_text": "literal-unique-value-00", "coordinator_record_id": "c"}
    if worker is not None: args.update(worker_text=worker, worker_record_id="w")
    original = project(initial, **args)
    sample = bind_task_cues(original, context, annotation)
    labels = later_labels(initial, worker_text=worker or "literal-unique-value-01", worker_record_id="w",
        final_text="literal-unique-value-00", final_record_id="f")
    if worker is not None: labels = {k: v for k, v in labels.items() if k.endswith(":publish")}
    return sample, labels


def test_arrival_distinguishes_same_counts_different_semantic_fields():
    a, _ = fixture("literal-unique-value-00")
    b, _ = fixture("literal-unique-value-01")
    old, model = TaskCueMessageWorkflowRiskBN(), SemanticArrivalWorkflowRiskBN()
    assert len(model.theta) == 68 and model.event_count == 65
    model.theta[model.names.index("message:publish:supported_arrival_fraction")] = 2.
    model.theta[model.names.index("message:publish:opposed_arrival_fraction")] = -2.
    assert old.predict_known_authorization_union(a)["probabilities"] == old.predict_known_authorization_union(b)["probabilities"]
    assert model.predict_known_authorization_union(a)["probabilities"]["system:any"] == expit(1.)
    assert model.predict_known_authorization_union(b)["probabilities"]["system:any"] == expit(-1.)
    assert a.authority.specification() == b.authority.specification()


@pytest.mark.parametrize("late", [False, True])
def test_zero_arrival_coefficients_preserve_old_joint_law(late):
    sample, _ = fixture("literal-unique-value-01" if late else None, 3)
    old, model = TaskCueMessageWorkflowRiskBN(), SemanticArrivalWorkflowRiskBN()
    old.theta = np.random.default_rng(52).normal(0, .3, len(old.theta))
    mapping = dict(zip(old.names, old.theta)); model.theta = np.array([mapping.get(n, 0.) for n in model.names])
    assert model.predict_known_authorization_union(sample)["probabilities"] == pytest.approx(old.predict_known_authorization_union(sample)["probabilities"], abs=1e-12)
    targets = [q["event_id"] for q in sample.queries.values()]
    a = model.event_law.conditional(sample.event_dag, targets, model.theta[:65])
    b = old.event_law.conditional(sample.event_dag, targets, old.theta[:55])
    assert np.allclose(a["log_probabilities"], b["log_probabilities"], atol=1e-12)
    assert SemanticArrivalWorkflowRiskBN.restore(model.parameters()).parameters() == model.parameters()


@pytest.mark.parametrize("late", [False, True])
def test_joint_graph_and_all_training_and_prediction_gradients(late):
    sample, labels = fixture("literal-unique-value-01" if late else None, 3)
    model = SemanticArrivalWorkflowRiskBN(); model.theta[:65] = np.random.default_rng(537).normal(0, .3, 65)
    prediction = model.predict_known_authorization_union(sample, derivatives=True)
    assert evaluate(model.probability_graph(sample), max_free_nodes=18) == pytest.approx(prediction["probabilities"]["system:any"], abs=1e-12)
    compiled = compile_message_training(model.event_law, [sample.event_dag], [labels]); theta = model.theta[:65].copy()
    loss, gradient = message_objective(theta, compiled, .001)
    direct = model.event_law.label_log_likelihood(sample.event_dag, {k: v["value"] for k, v in labels.items()}, theta)
    assert loss == pytest.approx(-direct["log_probability"] + .001 * (theta @ theta), abs=1e-12)
    for k in range(65):
        plus, minus = theta.copy(), theta.copy(); plus[k] += 1e-5; minus[k] -= 1e-5
        assert (message_objective(plus, compiled, .001)[0]-message_objective(minus, compiled, .001)[0])/2e-5 == pytest.approx(gradient[k], abs=1e-8)
        model.theta[k] = theta[k]+1e-5; pp = model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
        model.theta[k] = theta[k]-1e-5; pm = model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
        model.theta[k] = theta[k]
        assert (pp-pm)/2e-5 == pytest.approx(prediction["gradients"]["system:any"][k], abs=1e-8)


def test_future_input_is_marginalized_not_filled_with_absence():
    early, _ = fixture(n=2)
    model = SemanticArrivalWorkflowRiskBN()
    model.theta[model.names.index("message:publish:supported_arrival_fraction")] = 3.
    assert all("work" not in k for k in early.event_dag.observations)
    p = model.predict_known_authorization_union(early)["probabilities"]["system:any"]
    assert .5 < p < float(expit(1.5))
    late, _ = fixture("No tracked values")
    assert model.predict_known_authorization_union(late)["probabilities"]["system:any"] == .5


def test_message_features_are_permutation_invariant_and_keep_pairing():
    sample, _ = fixture("literal-unique-value-01", 3)
    law = SemanticArrivalWorkflowRiskBN().event_law; dag = sample.event_dag
    group = [q["event_id"] for q in sample.queries.values()]
    values = {n: np.array([o["value"]], float) for n, o in dag.observations.items()}
    a = law.gate_design(dag, group, values)
    assert np.array_equal(a, law.gate_design(dag, list(reversed(group)), values))
    assert a[0, -2] == 0 and a[0, -1] == pytest.approx(1/3)
