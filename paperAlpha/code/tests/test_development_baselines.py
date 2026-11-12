"""Constructed exactness/equivalence checks; never trains a real development set."""
import copy
import json

import numpy as np
import pytest
from scipy.special import logsumexp

from paperalpha_monitor.development_baselines import (
    ConditionalFactorBaseline, equal_group_task_weights, grouped_proper_scores,
    target_vector_log_probability, template_group_folds, weighted_complete_objective,
)
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, ObservableEventLaw
from paperalpha_monitor.observable_event_learning import complete_trace_design, complete_trace_objective, fit_event_traces
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.message_workflow_compiler import MessageWorkflowCompiler
from paperalpha_monitor.relational_event_law import RelationalEventLaw, make_relational_context
from test_message_workflow_compiler import specification, ev
from test_structured_event_baseline import training_examples


def relational_case(index=0, *, prefix_stage=0, literal_count=2):
    nodes = ["vault", "coordinator", "work", "finalizer"]
    edges = [["vault", "coordinator"], ["vault", "work"], ["coordinator", "work"], ["work", "finalizer"]]
    templates = ["configured_source", "prepare_message", "work_message", "publish_message"]
    literals = [f"TOKEN_{i:02d}_CONSTRUCTED" for i in range(literal_count)]
    records = [{"record_id": f"record-{i}", "fields": {"name": "Alice Green" if i == 0 else "Bob Brown", "secret": literal}}
               for i, literal in enumerate(literals)]
    request = "Summarize Alice Green's profile." if index % 2 else "Summarize all profiles."
    initial = json.dumps({"request": request, "records": records})
    values = [[int((index+literal+stage) % 3 == 0) for literal in range(literal_count)] for stage in range(3)]
    replies = [" ".join(literal for literal, bit in zip(literals, row) if bit) or "No tracked value."
               for row in values]
    source = {"kind": "complete", "evidence": ev(f"task-{index}:initial", initial, 1)}
    observed = {"vault": source}
    for stage in range(prefix_stage):
        observed[nodes[stage+1]] = {"kind": "complete", "evidence": ev(f"task-{index}:stage-{stage}", replies[stage], stage+2)}
    spec = {"workflow": {"nodes": nodes, "edges": edges, "actors": {n: [n+"-agent"] for n in nodes}},
            "subjects": [{"id": f"value-{i}", "literal": literal} for i, literal in enumerate(literals)],
            "stages": [{"node": n, "actor": n+"-agent", "template": template,
                        "subjects": [f"value-{i}" for i in range(literal_count)], "definition": ev(n)}
                       for n, template in zip(nodes, templates)],
            "deliveries": [{"source": a, "target": b, "subjects": [f"value-{i}" for i in range(literal_count)],
                            "basis": "declared_delivery", "evidence": ev(a+b)} for a, b in edges],
            "records": observed, "prefix_sequence": 1+prefix_stage}
    dag, _ = MessageWorkflowCompiler().compile(**spec)
    context = make_relational_context(dag, request=request, records=records,
        reference_fields={"name": "individual"}, visibility_basis="initial_api_request",
        initial_evidence=source["evidence"],
        stage_definitions={node: {"template": template, "evidence": ev(node)} for node, template in zip(nodes, templates)})
    declared = dag.specification(); declared["relational_context"] = context
    dag = ObservableEventDAG(**declared)
    outcomes = {f"value-{literal}:{nodes[stage+1]}": {
        **ev(f"task-{index}:stage-{stage}", replies[stage], stage+2), "kind": "complete_message", "value": values[stage][literal]}
        for stage in range(prefix_stage, 3) for literal in range(literal_count)}
    return dag, outcomes


def fit_options(dags):
    tasks = [f"task-{i}" for i in range(len(dags))]
    return {"task_ids": tasks, "earliest_prefix_sequences": {task: dag.prefix_sequence for task, dag in zip(tasks, dags)}}


@pytest.mark.parametrize("relational", [False, True])
def test_glm_same_features_labels_objective_fit_and_exact_predictions(relational):
    if relational:
        examples = [relational_case(i) for i in range(8)]
        dags, labels = map(list, zip(*examples)); law = RelationalEventLaw()
    else:
        dags, labels = training_examples(); law = ObservableEventLaw()
    baseline = ConditionalFactorBaseline(law.names, law.node_design, regularization=.02)
    design = baseline.training_design(dags, labels, **fit_options(dags))
    x, y = complete_trace_design(law, dags, labels)
    assert np.array_equal(design["x"], x) and np.array_equal(design["y"], y)
    theta = np.random.default_rng(721).normal(0, .2, len(law.names))
    actual = weighted_complete_objective(theta, x, y, row_weights=design["row_weights"],
        task_weight_sum=design["task_weight_sum"], regularization=.02)
    expected = complete_trace_objective(theta, x, y, len(dags), .02)
    assert actual[0] == expected[0] and np.array_equal(actual[1], expected[1])
    baseline.fit(dags, labels, **fit_options(dags))
    def constructor(**kwargs):
        return EliminatingDAGWorkflowRiskBN(event_law=RelationalEventLaw() if relational else None, **kwargs)
    model = fit_event_traces(dags, labels, regularization=.02, model_class=constructor)
    np.testing.assert_allclose(baseline.coef, model.theta[:len(law.names)], atol=1e-10, rtol=0)
    assert baseline.fit_report["event_label_count"] == sum(len(row) for row in labels)
    targets = [n for n in dags[0].order if dags[0].lookup[n]["family"] == "publish"]
    prediction = baseline.predict(dags[0], targets, include_dense_joint=True)
    expected_joint = model.event_law.conditional(dags[0], targets, model.theta[:len(law.names)])
    np.testing.assert_allclose(prediction["joint_log_probabilities"], expected_joint["log_probabilities"], atol=1e-10, rtol=0)


@pytest.mark.parametrize("prefix_stage", [0, 1, 2])
def test_relational_three_prefixes_match_exact_law(prefix_stage):
    dag, outcomes = relational_case(2, prefix_stage=prefix_stage)
    law = RelationalEventLaw(); theta = np.random.default_rng(517).normal(0, .5, 70)
    baseline = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(theta)
    targets = ["value-0:finalizer", "value-1:finalizer"]
    result = baseline.predict(dag, targets, include_dense_joint=True)
    reference = law.conditional(dag, targets, theta)["log_probabilities"]
    np.testing.assert_allclose(result["joint_log_probabilities"], reference, atol=1e-12)
    assert result["union_probability"] == pytest.approx(1-np.exp(reference[0]), abs=1e-12)
    bits = {n: outcomes[n]["value"] for n in targets}
    index = 2*bits[targets[0]]+bits[targets[1]]
    assert target_vector_log_probability(result, bits) == pytest.approx(reference[index], abs=1e-12)
    assert result["max_component_free"] == 3-prefix_stage


@pytest.mark.parametrize("literal_count", [8, 38])
def test_many_components_do_not_trigger_global_eighteen_variable_limit(literal_count):
    dag, outcomes = relational_case(literal_count=literal_count)
    law = RelationalEventLaw()
    baseline = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(np.zeros(70))
    targets = [f"value-{i}:finalizer" for i in range(literal_count)]
    result = baseline.predict(dag, targets)
    assert result["component_count"] == literal_count and result["max_component_free"] == 3
    assert len(dag.ids)-len(dag.observations) == 3*literal_count
    assert result["union_probability"] == pytest.approx(1-.5**literal_count)
    assert all(p == pytest.approx(.5) for p in result["target_marginals"].values())
    assert target_vector_log_probability(result, {n: outcomes[n]["value"] for n in targets}) == pytest.approx(literal_count*np.log(.5))
    assert result["independence_fallback"] is False


def test_common_future_parent_is_integrated_not_zero_or_independent():
    spec = specification()
    edges = [["source", "review"], ["review", "left"], ["review", "right"], ["left", "publish"], ["right", "publish"]]
    spec["workflow"]["edges"] = edges
    spec["deliveries"] = [{"source": a, "target": b, "subjects": ["secret"],
                           "basis": "declared_delivery", "evidence": ev(a+b)} for a,b in edges]
    dag, _ = MessageWorkflowCompiler().compile(**spec)
    law = ObservableEventLaw(); theta = np.zeros(25); theta[5] = -2; theta[8] = 4
    baseline = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(theta)
    result = baseline.predict(dag, ["secret:left", "secret:right"], include_dense_joint=True)
    assert np.exp(result["joint_log_probabilities"])[3] > .39
    assert result["target_marginals"]["secret:left"] == pytest.approx(.5)
    assert result["component_count"] == 1
    np.testing.assert_allclose(result["joint_log_probabilities"], law.conditional(dag, result["targets"], theta)["log_probabilities"], atol=1e-12)


def test_per_component_capacity_rejects_no_fallback():
    dag, _ = MessageWorkflowCompiler().compile(**specification())
    law = ObservableEventLaw()
    baseline = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(np.zeros(25))
    with pytest.raises(ValueError, match="capacity.*fallback"):
        baseline.predict(dag, ["secret:publish"], max_free=3)


def test_missing_past_is_marginalized_not_filled_zero():
    spec = specification()
    spec["records"]["left"] = {"kind": "missing", "evidence": ev("missing", "Missing recorded response", 2)}
    dag, _ = MessageWorkflowCompiler().compile(**spec)
    law = ObservableEventLaw(); theta = np.random.default_rng(819).normal(0, .5, 25)
    result = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(theta).predict(dag, ["secret:publish"], include_dense_joint=True)
    np.testing.assert_allclose(result["joint_log_probabilities"], law.conditional(dag, ["secret:publish"], theta)["log_probabilities"], atol=1e-12)


@pytest.mark.parametrize("kind", ["glm", "histgb"])
def test_complete_event_supervision_and_template_weights(kind):
    examples = [relational_case(i) for i in range(4)]
    dags, labels = map(list, zip(*examples)); law = RelationalEventLaw()
    baseline = ConditionalFactorBaseline(law.names, law.node_design, kind=kind,
        tree_config={"max_iter": 10, "min_samples_leaf": 1})
    options = {**fit_options(dags), "group_ids": ["template-a", "template-a", "template-b", "template-c"]}
    design = baseline.training_design(dags, labels, **options)
    assert design["task_weights"].tolist() == [.5, .5, 1., 1.]
    baseline.fit(dags, labels, **options)
    assert baseline.fit_report["event_label_count"] == 24
    result = baseline.predict(dags[0], ["value-0:finalizer", "value-1:finalizer"], include_dense_joint=True)
    assert np.exp(result["joint_log_probabilities"]).sum() == pytest.approx(1.)
    assert 0 <= result["union_probability"] <= 1


def test_weighted_joint_objective_matches_manual_task_sum_and_gradient():
    x = np.asarray([[1., 2.], [1., 1.], [1., 0.]])
    y = np.asarray([1, 0, 1]); weights = np.asarray([.5, .5, 2.]); theta = np.asarray([.3, -.7])
    value, gradient = weighted_complete_objective(theta, x, y, row_weights=weights, task_weight_sum=2.5, regularization=.02)
    manual = (weights*(np.logaddexp(0, x@theta)-y*(x@theta))).sum()/2.5+.02*(theta@theta)
    assert value == pytest.approx(manual)
    for i in range(2):
        delta = np.zeros(2); delta[i] = 1e-5
        plus = weighted_complete_objective(theta+delta,x,y,row_weights=weights,task_weight_sum=2.5,regularization=.02)[0]
        minus = weighted_complete_objective(theta-delta,x,y,row_weights=weights,task_weight_sum=2.5,regularization=.02)[0]
        assert gradient[i] == pytest.approx((plus-minus)/2e-5, abs=1e-9)


@pytest.mark.parametrize("fault", ["duplicate_task", "late_prefix", "missing_label", "false_literal", "wrong_dimension", "future_evidence"])
def test_bad_training_contract_rejected(fault):
    dag, labels = relational_case(); law = RelationalEventLaw()
    dags, outcomes = [dag], [copy.deepcopy(labels)]; options = fit_options(dags)
    feature = law.node_design
    if fault == "duplicate_task":
        dags *= 2; outcomes *= 2; options["task_ids"] *= 2
    elif fault == "late_prefix": options["earliest_prefix_sequences"]["task-0"] = 0
    elif fault == "missing_label": outcomes[0].pop(next(iter(outcomes[0])))
    elif fault == "false_literal":
        first = next(iter(outcomes[0])); outcomes[0][first]["value"] = 1-outcomes[0][first]["value"]
    elif fault == "wrong_dimension": feature = lambda *args: np.zeros(25)
    else: outcomes[0][next(iter(outcomes[0]))]["sequence"] = dag.prefix_sequence
    with pytest.raises(ValueError):
        ConditionalFactorBaseline(law.names, feature).training_design(dags, outcomes, **options)


def test_group_folds_keep_all_fields_and_prefixes_together_without_labels():
    tasks = [f"task-{i}" for i in range(8) for _ in range(3)]
    groups = [f"template-{i//2}" for i in range(8) for _ in range(3)]
    folds = template_group_folds(tasks, groups, n_splits=4)
    assert sorted(i for fold in folds for i in fold["test_indices"]) == list(range(24))
    for fold in folds:
        assert not set(fold["train_groups"]) & set(fold["test_groups"])
        assert not {tasks[i] for i in fold["train_indices"]} & {tasks[i] for i in fold["test_indices"]}


def test_group_metrics_do_not_count_repeated_prefixes_as_independent_tasks():
    result = grouped_proper_scores([.2, .2, .9], [0, 0, 0], ["a", "a", "b"], ["A", "A", "B"])
    assert result["task_count"] == result["group_count"] == 2
    assert result["brier"] == pytest.approx((.04+.81)/2)


def test_extreme_glm_log_probability_remains_finite():
    dag, _ = relational_case(literal_count=1); law = RelationalEventLaw()
    theta = np.zeros(70); theta[law.names.index("event:publish:bias")] = -1000
    result = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(theta).predict(dag, ["value-0:finalizer"])
    assert target_vector_log_probability(result, {"value-0:finalizer": 1}) == pytest.approx(-1000)


def test_no_row_random_early_stopping():
    law = ObservableEventLaw()
    with pytest.raises(ValueError, match="early stopping"):
        ConditionalFactorBaseline(law.names, law.node_design, kind="histgb", tree_config={"early_stopping": True})


@pytest.mark.parametrize("bit", [0, 1])
def test_histgb_single_class_training_retains_positive_class_semantics(bit):
    dag, labels = relational_case(literal_count=1)
    for observation in labels.values():
        observation.update(value=bit, text="TOKEN_00_CONSTRUCTED" if bit else "No tracked value.")
    law = RelationalEventLaw()
    baseline = ConditionalFactorBaseline(law.names, law.node_design, kind="histgb",
        tree_config={"max_iter": 3, "min_samples_leaf": 1}).fit([dag], [labels], **fit_options([dag]))
    result = baseline.predict(dag, ["value-0:finalizer"])
    assert result["union_probability"] == pytest.approx(bit)
    assert target_vector_log_probability(result, {"value-0:finalizer": bit}) == pytest.approx(0.)


def test_changing_final_target_across_prefixes_rejected():
    with pytest.raises(ValueError, match="fixed final target"):
        grouped_proper_scores([.3,.4], [0,1], ["same", "same"], ["template", "template"])
