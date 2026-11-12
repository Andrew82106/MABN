"""Independent complete-world mathematics for partial grouped supervision.

The oracle never calls a production design, conditional, likelihood or fitting
helper. It explicitly unrolls continuation variables and sums all missing bits.
These are constructed protocol/math cases, not real model-performance evidence.
"""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, ObservableEventLaw
from paperalpha_monitor.grouped_event_learning import GroupedEventTrace, grouped_loss_gradient
from paperalpha_monitor.partial_grouped_event_learning import PartialGroupedEventTrace, partial_grouped_loss_gradient


def fixture(shared_values, continuation_values, *, separate=False, chain=False, identity="independent"):
    reviewers = ["review-a", "review-b"] if separate else ["review-a"]
    nodes = ["source", *reviewers, "publish-a", "publish-b"]
    edges = [["source", n] for n in reviewers]
    edges += [["review-a", "publish-a"], ["publish-a" if chain else reviewers[-1], "publish-b"]]
    marker = "INDEPENDENT-TEST-CANARY"
    def evidence(record, text, sequence=0):
        return {"record_id": identity + ":" + record, "sequence": sequence,
                "source": "constructed-independent-oracle", "text": text}
    def observation(record, bit, sequence):
        return {**evidence(record, "Message " + (marker if bit else "without the canary"), sequence),
                "kind": "complete_message", "value": bit}
    events = [{"id": n, "node": n, "actor": n,
               "family": "prepare" if n == "source" else "review" if n.startswith("review") else "publish",
               "semantics": "literal_appearance", "subject": marker,
               "cues": [1, 0] if n == "review-a" else [0, 1] if n == "review-b" else [0, 0],
               "cue_evidence": evidence("instruction:" + n, "Constructed prompt evidence")}
              for n in nodes]
    dag = ObservableEventDAG(workflow={"nodes": nodes, "edges": edges, "actors": {n: [n] for n in nodes}},
        events=events, edges=[{"source": a, "target": b, "basis": "declared_delivery",
                              "evidence": evidence("edge:" + a + ":" + b, "Constructed route")} for a, b in edges],
        observations={"source": observation("source", 1, 1)}, prefix_sequence=2)
    sequence = 2
    shared = {}
    for n in reviewers:
        sequence += 2
        if n in shared_values: shared[n] = observation(n, shared_values[n], sequence)
    continuations = []
    for i, values in enumerate(continuation_values):
        labels = {}
        for n in ("publish-a", "publish-b"):
            sequence += 2
            if n in values: labels[n] = observation(str(i) + ":" + n, values[n], sequence)
        continuations.append(labels)
    return PartialGroupedEventTrace(dag, reviewers, shared, continuations)


def enumerated_log_likelihood_and_score(group, theta):
    """P(all actually observed future bits | fixed prefix) and its derivative."""
    dag = group.dag
    prefix = set(dag.observations)
    shared = set(group.shared_event_ids)
    future = set(dag.ids) - prefix
    suffix = future - shared
    vertices = [(n, None) for n in dag.order if n in shared]
    vertices += [(n, i) for i in range(len(group.continuations)) for n in dag.order if n in suffix]
    observed = {(n, None): v["value"] for n, v in group.shared_labels.items()}
    observed.update({(n, i): v["value"] for i, labels in enumerate(group.continuations) for n, v in labels.items()})
    families = ("prepare", "work", "review", "publish", "dispatch")
    probability, derivative = 0., np.zeros(len(theta))
    for assignment in product((0, 1), repeat=len(vertices)):
        values = dict(zip(vertices, assignment))
        if any(values[key] != value for key, value in observed.items()): continue
        mass, score = 1., np.zeros(len(theta))
        for node, continuation in vertices:
            event = dag.lookup[node]
            parent_values = [dag.observations[p]["value"] if p in prefix else
                             values[(p, None if p in shared else continuation)] for p in dag.parents[node]]
            features = np.zeros(len(theta)); start = families.index(event["family"]) * 5
            features[start:start + 5] = [1., *event["cues"],
                sum(parent_values) / len(parent_values) if parent_values else 0.,
                float(len(parent_values) > 1 and all(parent_values))]
            p = 1. / (1. + np.exp(-float(features @ theta)))
            bit = values[(node, continuation)]
            mass *= p if bit else 1. - p
            score += (bit - p) * features
        probability += mass
        derivative += mass * score
    assert 0. < probability <= 1. + 1e-12
    return np.log(probability), derivative / probability


@pytest.mark.parametrize("separate,shared,continuations", [
    (False, {}, [{}, {}]),
    (False, {"review-a": 0}, [{}, {}]),
    (True, {"review-a": 1}, [{}, {}]),
    (True, {"review-b": 0}, [{}, {}]),
    (False, {"review-a": 1}, [{"publish-a": 1}, {"publish-b": 0}]),
    (True, {"review-a": 0, "review-b": 1}, [{"publish-b": 1}, {}]),
    (False, {"review-a": 0}, [{"publish-a": 1, "publish-b": 0}, {"publish-a": 0, "publish-b": 1}]),
    (True, {"review-a": 1, "review-b": 0}, [{"publish-a": 0, "publish-b": 1}, {"publish-a": 1, "publish-b": 0}]),
])
def test_partial_group_matches_independently_unrolled_full_joint(separate, shared, continuations):
    group = fixture(shared, continuations, separate=separate)
    theta = np.random.default_rng(931).normal(0., .45, 25)
    regularization, denominator = .017, 36
    expected_logp, expected_score = enumerated_log_likelihood_and_score(group, theta)
    actual_loss, actual_gradient = partial_grouped_loss_gradient(
        ObservableEventLaw(), theta, [group], regularization=regularization, normalization_groups=denominator)
    assert actual_loss == pytest.approx(-expected_logp / denominator + regularization * (theta @ theta), abs=1e-12)
    assert np.allclose(actual_gradient, -expected_score / denominator + 2 * regularization * theta, atol=1e-12, rtol=0)
    # Finite differences also use only the enumerated oracle, not production loss.
    for i in range(len(theta)):
        plus, minus = theta.copy(), theta.copy(); plus[i] += 1e-5; minus[i] -= 1e-5
        numeric = -(enumerated_log_likelihood_and_score(group, plus)[0] -
                    enumerated_log_likelihood_and_score(group, minus)[0]) / (2e-5 * denominator)
        assert actual_gradient[i] == pytest.approx(numeric + 2 * regularization * theta[i], abs=1e-9)


def test_multiple_groups_use_planned_count_even_with_unobserved_groups():
    groups = [fixture({}, [{}, {}], identity="empty"),
              fixture({"review-a": 1}, [{"publish-b": 1}, {}], identity="one-sided"),
              fixture({"review-a": 0}, [{"publish-a": 1, "publish-b": 0}], identity="complete")]
    theta = np.random.default_rng(731).normal(0., .3, 25)
    expected = [enumerated_log_likelihood_and_score(g, theta) for g in groups]
    loss, gradient = partial_grouped_loss_gradient(ObservableEventLaw(), theta, groups,
                                                  regularization=0., normalization_groups=36)
    assert loss == pytest.approx(-sum(v for v, _ in expected) / 36, abs=1e-12)
    assert np.allclose(gradient, -sum((g for _, g in expected), np.zeros(25)) / 36, atol=1e-12, rtol=0)


def test_all_empty_is_not_observed_negative_labels():
    group = fixture({}, [{}, {}]); theta = np.full(25, .7)
    loss, gradient = partial_grouped_loss_gradient(ObservableEventLaw(), theta, [group], normalization_groups=36)
    assert loss == pytest.approx(0., abs=1e-12)
    assert np.allclose(gradient, 0., atol=1e-12)
    wrong = fixture({"review-a": 0}, [{"publish-a": 0, "publish-b": 0}, {"publish-a": 0, "publish-b": 0}])
    wrong_logp, _ = enumerated_log_likelihood_and_score(wrong, theta)
    assert -wrong_logp / 36 > .1


@pytest.mark.parametrize("separate", [False, True])
def test_complete_case_equals_original_grouped_likelihood(separate):
    shared = {"review-a": 1, **({"review-b": 0} if separate else {})}
    group = fixture(shared, [{"publish-a": 1, "publish-b": 0}, {"publish-a": 0, "publish-b": 1}], separate=separate)
    complete = GroupedEventTrace(group.dag, group.shared_labels, group.continuations)
    theta = np.random.default_rng(651).normal(0., .3, 25); law = ObservableEventLaw()
    got = partial_grouped_loss_gradient(law, theta, [group], regularization=.01, normalization_groups=1)
    old = grouped_loss_gradient(law, theta, [complete], regularization=.01)
    assert got[0] == pytest.approx(old[0], abs=1e-12)
    assert np.allclose(got[1], old[1], atol=1e-12, rtol=0)


def test_suffix_cannot_borrow_parent_from_another_continuation():
    with pytest.raises(ValueError):
        fixture({"review-a": 1}, [{"publish-a": 1}, {"publish-b": 0}], chain=True)


def test_observed_publisher_cannot_fill_failed_shared_parent_with_zero():
    with pytest.raises(ValueError):
        fixture({}, [{"publish-a": 0}, {}])


def fixed_protocol_models(monkeypatch):
    """Do not optimize even synthetic data: inspect evaluator routing instead."""
    from paperalpha_monitor import partial_repeated_message_evaluation as evaluation
    from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN
    from paperalpha_monitor.structured_event_baseline import StructuredEventBaseline
    calls = []

    def record(traces, denominator):
        calls.append({"denominator": denominator,
            "groups": [t.dag.observations["source"]["record_id"].removesuffix(":source") for t in traces],
            "records": [v["record_id"] for t in traces for b in [t.shared_labels, *t.continuations] for v in b.values()]})

    def fixed_bn(traces, *, regularization, max_iter, normalization_groups, model_class=DAGWorkflowRiskBN):
        record(traces, normalization_groups)
        model = model_class(regularization=regularization)
        model.optimizer = {"success": True, "constructed_unfitted_test_double": True,
                           "test_training_ids": calls[-1]["groups"]}
        return model

    def fixed_glm(traces, *, penalty, max_iter, normalization_groups):
        record(traces, normalization_groups)
        model = StructuredEventBaseline(penalty=penalty)
        model.coef = np.zeros(25)
        model.fit_report = {"constructed_unfitted_test_double": True}
        return model

    predict = evaluation.bn_joint
    def checked_prefix(model, dag):
        assert not set(dag.observations) & {"publish-a", "publish-b"}
        assert set(dag.observations) <= {"source", "review", "review-a", "review-b"}
        return predict(model, dag)
    monkeypatch.setattr(evaluation, "fit_partial_grouped_event_traces", fixed_bn)
    monkeypatch.setattr(evaluation, "fit_partial_matched_glm", fixed_glm)
    monkeypatch.setattr(evaluation, "bn_joint", checked_prefix)
    return evaluation, calls


def test_evaluator_routes_same_scoring_records_with_planned_denominator_and_no_heldout_labels(monkeypatch):
    from test_repeated_message_evaluation import cohort
    evaluation, calls = fixed_protocol_models(monkeypatch)
    value = cohort()
    groups = value["grouped_traces"]
    # One shared failure, one suffix-only failure, one upstream-only successful
    # observation and one excluded identity: none becomes a negative outcome.
    groups[0].update(shared_labels={}, upstream_complete=False, complete=False,
                     continuations=[], eligible_for_grouped_fit=False, eligible_for_statistics=False)
    partial = groups[1]["continuations"].pop(0)
    groups[1].update(complete=False, incomplete_continuations=[{
        "index": 0, "completed": False, "labels": {"publish-a": partial["publish-a"]}}])
    groups[2].update(continuations=[], complete=False, eligible_for_grouped_fit=False,
        eligible_for_statistics=False,
        incomplete_continuations=[{"index": i, "completed": False, "labels": {}} for i in range(2)])
    groups[3].update(response_identity_qualified=False, eligible_for_grouped_fit=False, eligible_for_statistics=False)
    report = evaluation.evaluate_partial_collection(value)
    assert all(c["denominator"] == 36 for c in calls)
    known = {g["id"]: g for g in groups}
    folds = {f["held_out_task"]: f for f in report["folds"]}
    for model in report["models"]:
        ids = model["exact_bn"]["optimizer"]["test_training_ids"]
        assert set(ids) <= set(folds[model["held_out_task"]]["planned_train_ids"])
        assert not set(ids) & set(folds[model["held_out_task"]]["score_ids"])
        assert all(known[n]["base_task"] != model["held_out_task"] for n in ids)
    for call in calls:
        allowed_records = {v["record_id"] for n in call["groups"] for b in [known[n]["shared_labels"],
            *known[n]["continuations"], *[r["labels"] for r in known[n]["incomplete_continuations"]]] for v in b.values()}
        assert set(call["records"]) <= allowed_records
        assert len(call["records"]) == len(set(call["records"]))
    complete, observed = [{(r["group_id"], r["method"], r["phase"], r["penalty"]):
                          (r["input_sha256"], r["actual_response_records"], r["actual_outcomes"])
                          for r in report["predictions"] if r["supervision_mode"] == mode}
                          for mode in evaluation.SUPERVISION_MODES]
    assert complete == observed and len(complete) == 45 * 16
    expected_records = {v["record_id"] for g in groups if g["response_identity_qualified"]
        for b in [g["shared_labels"], *g["continuations"], *[r["labels"] for r in g["incomplete_continuations"]]] for v in b.values()}
    assert report["supervision_coverage"]["all_observed"]["effective_labels"]["total"] == len(expected_records)


def test_one_informative_task_cannot_create_an_unpaired_improvement_claim(monkeypatch):
    from test_repeated_message_evaluation import cohort
    evaluation, calls = fixed_protocol_models(monkeypatch)
    value = cohort()
    scored_task = value["grouped_traces"][0]["base_task"]
    for group in value["grouped_traces"]:
        if group["base_task"] == scored_task:
            continue
        group.update(continuations=[], complete=False, eligible_for_grouped_fit=False,
            eligible_for_statistics=False,
            incomplete_continuations=[{"index": i, "completed": False, "labels": {}} for i in range(2)])
    report = evaluation.evaluate_partial_collection(value)
    assert calls and all(call["denominator"] == 36 for call in calls)
    assert {r["supervision_mode"] for r in report["predictions"]} == {"all_observed"}
    assert {r["group_id"] for r in report["predictions"]} == {
        g["id"] for g in value["grouped_traces"] if g["base_task"] == scored_task}
    assert not report["paired_supervision_contrasts"]
    for coverage in report["supervision_comparison_coverage"]:
        assert not coverage["same_scored_group_ids"]
        assert not coverage["direct_summary_comparison_valid"]
        assert not coverage["common_score_ids"]
    assert all(s["groups"] == 0 and s["overall"] is None for s in report["paired_supervision_summaries"])
