"""Grouped likelihood checks use constructed messages, not real API results."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.grouped_event_learning import (GroupedEventTrace, grouped_complete_design,
    grouped_loss_gradient, fit_grouped_event_traces)
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, DAGWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import complete_trace_design, trace_loss_gradient, fit_event_traces
from paperalpha_runtime import relational_forecast_study as study


def fixture(layout="shared", values=((1, 0), (0, 1)), identity="fixture"):
    case = next(c for c in study.cases() if c["split"] == "train" and c["layout"] == layout
                and c["review_policy"] == "summarize" and c["source_marker_present"])
    packet = study.prefix(case, {"source": {"record_id": identity+":source", "sequence": 1, "text": case["source"]}}, 2)
    dag = study.compile_input(packet).event_dag
    seq = 2
    def observation(name, bit):
        nonlocal seq
        seq += 2
        return {"record_id": identity+":"+name, "sequence": seq, "source": "constructed-fixture",
                "kind": "complete_message", "value": bit,
                "text": "Packet " + (case["marker"] if bit else "without tracked reference")}
    shared = {n: observation(n, 1) for n in dag.order if n.startswith("review-")}
    continuations = [{n: observation(str(i)+":"+n, v) for n, v in zip(("publish-a", "publish-b"), bits)}
                     for i, bits in enumerate(values)]
    return GroupedEventTrace(dag, shared, continuations)


@pytest.mark.parametrize("layout", ["shared", "separate"])
def test_shared_labels_once_and_continuations_all_counted(layout):
    group = fixture(layout); law = DAGWorkflowRiskBN().event_law
    x, y, audit = grouped_complete_design(law, [group])
    shared = len(group.shared_labels)
    assert x.shape == (shared+4, 25) and len(y) == shared+4
    assert audit["shared_event_labels"] == shared and audit["continuation_event_labels"] == 4
    assert audit["unique_response_records"] == shared+4
    # Flattening duplicates the shared observation. The exact excess is its
    # log likelihood, not a harmless global rescaling of the objective.
    theta = np.random.default_rng(320).normal(0, .3, 25)
    actual, _ = grouped_loss_gradient(law, theta, [group])
    naive = sum(trace_loss_gradient(law, theta, [group.dag], [{**group.shared_labels, **c}], 0.)[0]
                for c in group.continuations)
    shared_loss = trace_loss_gradient(law, theta, [group.dag], [group.shared_labels], 0.)[0]
    assert naive - actual == pytest.approx(shared_loss, abs=1e-12)


def test_single_continuation_exactly_reduces_to_original_trainer():
    group = fixture(values=((1, 0),)); law = DAGWorkflowRiskBN().event_law
    old_x, old_y = complete_trace_design(law, [group.dag], [{**group.shared_labels, **group.continuations[0]}])
    x, y, _ = grouped_complete_design(law, [group])
    assert np.array_equal(x, old_x) and np.array_equal(y, old_y)
    model = fit_grouped_event_traces([group], regularization=.03)
    old = fit_event_traces([group.dag], [{**group.shared_labels, **group.continuations[0]}], regularization=.03)
    assert np.array_equal(model.theta, old.theta)


def test_full_grouped_gradient_and_model_restore_do_not_change_prefix():
    groups = [fixture(identity="one"), fixture("separate", identity="two")]
    before = [g.specification() for g in groups]; law = DAGWorkflowRiskBN().event_law
    theta = np.random.default_rng(231).normal(0, .4, 25)
    value, grad = grouped_loss_gradient(law, theta, groups, .03)
    numeric = approx_derivative(lambda t: grouped_loss_gradient(law, t, groups, .03)[0], theta).ravel()
    assert np.isfinite(value) and np.allclose(grad, numeric, atol=1e-8)
    model = fit_grouped_event_traces(groups, regularization=.03)
    restored = DAGWorkflowRiskBN.restore(model.parameters())
    assert np.array_equal(restored.theta, model.theta)
    assert model.optimizer["training_audit"]["groups"] == 2
    assert model.optimizer["authority_priors_fitted"] is False
    assert np.array_equal(model.theta[25:], np.zeros(3))
    assert before == [g.specification() for g in groups]


def test_group_likelihood_equals_explicit_unrolled_graph():
    group = fixture(); spec = group.dag.specification()
    suffix = set(group.continuations[0]); shared = set(spec["workflow"]["nodes"])-suffix
    w = spec["workflow"]
    w["nodes"] = [n for n in w["nodes"] if n in shared]
    w["actors"] = {n: v for n, v in w["actors"].items() if n in shared}
    old_edges = copy.deepcopy(spec["edges"]); old_events = copy.deepcopy(spec["events"])
    w["edges"] = [e for e in w["edges"] if e[1] in shared]
    spec["events"] = [e for e in spec["events"] if e["id"] in shared]
    spec["edges"] = [e for e in old_edges if e["target"] in shared]
    labels = copy.deepcopy(group.shared_labels)
    for i, continuation in enumerate(group.continuations):
        for n, label in continuation.items():
            name = n+":"+str(i)
            event = copy.deepcopy(next(e for e in old_events if e["id"] == n))
            event.update(id=name, node=name, actor=name)
            w["nodes"].append(name); w["actors"][name] = [name]; spec["events"].append(event)
            for edge in old_edges:
                if edge["target"] == n:
                    spec["edges"].append({**edge, "target": name}); w["edges"].append([edge["source"], name])
            labels[name] = copy.deepcopy(label)
    unrolled = ObservableEventDAG(**spec); law = DAGWorkflowRiskBN().event_law
    theta = np.random.default_rng(29).normal(0, .2, 25)
    actual = grouped_loss_gradient(law, theta, [group], .01)
    expected = trace_loss_gradient(law, theta, [unrolled], [labels], .01)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert np.allclose(actual[1], expected[1], atol=1e-12)


@pytest.mark.parametrize("mutation", ["missing", "overlap", "same_record", "clock", "shared_descendant", "text", "changed"])
def test_bad_groups_rejected(mutation):
    group = fixture(); shared = copy.deepcopy(group.shared_labels); suffix = copy.deepcopy(group.continuations)
    if mutation == "missing": suffix[0].pop("publish-b")
    elif mutation == "overlap": suffix[0]["review-a"] = shared["review-a"]
    elif mutation == "same_record": suffix[1]["publish-a"]["record_id"] = suffix[0]["publish-a"]["record_id"]
    elif mutation == "clock": suffix[1]["publish-a"]["sequence"] = 3
    elif mutation == "shared_descendant": shared = {"publish-a": suffix[0]["publish-a"]}
    elif mutation == "text": suffix[0]["publish-a"]["value"] ^= 1
    else:
        group.continuations[0]["publish-a"]["text"] += " changed"
        with pytest.raises(ValueError): group.validate()
        return
    with pytest.raises(ValueError): GroupedEventTrace(group.dag, shared, suffix)


def test_reusing_actual_responses_in_multiple_groups_is_rejected():
    group = fixture(); law = DAGWorkflowRiskBN().event_law
    with pytest.raises(ValueError, match="reused across training groups"):
        grouped_complete_design(law, [group, group])


@pytest.mark.parametrize("mutation", ["shared_actor", "suffix_actor", "conflicting_text",
    "prefix_id", "prefix_id_new_source", "same_tick", "new_source_continuation"])
def test_record_aliases_and_journal_collisions_rejected(mutation):
    group = fixture("separate", values=((1, 0), (0, 1)))
    shared, continuations = copy.deepcopy(group.shared_labels), copy.deepcopy(group.continuations)
    if mutation == "shared_actor":
        shared["review-b"] = copy.deepcopy(shared["review-a"])
    elif mutation == "suffix_actor":
        continuations[0]["publish-b"] = copy.deepcopy(continuations[0]["publish-a"])
    elif mutation == "conflicting_text":
        # Both labels individually match their texts, but a response cannot
        # have two incompatible payloads/values and timestamps.
        continuations[0]["publish-b"]["record_id"] = continuations[0]["publish-a"]["record_id"]
    elif mutation in ("prefix_id", "prefix_id_new_source"):
        original = group.dag.observations["source"]
        shared["review-a"]["record_id"] = original["record_id"]
        shared["review-a"]["source"] = original["source"] if mutation == "prefix_id" else "changed-source-type"
    elif mutation == "same_tick":
        continuations[0]["publish-b"]["sequence"] = continuations[0]["publish-a"]["sequence"]
    else:
        continuations[1]["publish-a"]["record_id"] = continuations[0]["publish-a"]["record_id"]
        continuations[1]["publish-a"]["source"] = "changed-source-type"
    with pytest.raises(ValueError):
        GroupedEventTrace(group.dag, shared, continuations)


def test_changing_source_cannot_reuse_response_across_training_groups():
    first, second = fixture(identity="first"), fixture(identity="second")
    shared = copy.deepcopy(second.shared_labels)
    shared["review-a"]["record_id"] = first.shared_labels["review-a"]["record_id"]
    shared["review-a"]["source"] = "changed-source-type"
    second = GroupedEventTrace(second.dag, shared, second.continuations)
    with pytest.raises(ValueError, match="reused across training groups"):
        grouped_complete_design(DAGWorkflowRiskBN().event_law, [first, second])


def multiple_literals():
    group = fixture(values=((1, 0),))
    spec = group.dag.specification()
    extra = copy.deepcopy(group.dag.lookup["review-a"])
    extra.update(id="review-other-literal", subject="OTHER-FICTIONAL-LITERAL")
    spec["events"].append(extra)
    shared = copy.deepcopy(group.shared_labels)
    shared[extra["id"]] = {**copy.deepcopy(shared["review-a"]), "value": 0}
    return spec, shared, group.continuations


def test_distinct_literals_from_same_actual_message_are_allowed_not_new_records():
    spec, shared, continuations = multiple_literals()
    group = GroupedEventTrace(ObservableEventDAG(**spec), shared, continuations)
    _, labels, audit = grouped_complete_design(DAGWorkflowRiskBN().event_law, [group])
    assert len(labels) == 4 and audit["unique_response_records"] == 3
    assert audit["shared_event_labels"] == 2


@pytest.mark.parametrize("mutation", ["subject", "text", "sequence", "source"])
def test_multiliteral_message_must_retain_payload_and_not_duplicate_subject(mutation):
    spec, shared, continuations = multiple_literals()
    label = shared["review-other-literal"]
    if mutation == "subject":
        spec["events"][-1]["subject"] = spec["events"][1]["subject"].lower()
        label["value"] = 1
    elif mutation == "text":
        label["text"] += " otherwise identical"
    elif mutation == "sequence":
        label["sequence"] += 1
    else:
        label["source"] = "changed-source-type"
    with pytest.raises(ValueError):
        GroupedEventTrace(ObservableEventDAG(**spec), shared, continuations)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_bad_regularization_rejected(bad):
    with pytest.raises(ValueError): fit_grouped_event_traces([fixture()], regularization=bad)
