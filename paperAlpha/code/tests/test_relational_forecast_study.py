"""Authored protocol fixtures, not real API accuracy results."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import relational_forecast_study as study
from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN, ObservableEventDAG
from paperalpha_monitor.observable_event_learning import fit_event_traces, trace_loss_gradient, validated_labels
from paperalpha_monitor.observable_event_qualification import intermediate_semantics, binary_target_coverage
from audit_relational_event_meaning import flip_review_representation
from scalar_binary_graph import evaluate


def fixture(layout="shared", policy="retain", source_present=True, bits=None):
    case = next(c for c in study.cases() if c["split"] == "train" and c["layout"] == layout and c["review_policy"] == policy and c["source_marker_present"] == source_present)
    prefix = {"source": {"record_id": "source", "sequence": 1, "text": case["source"]}}
    early = study.prefix(case, prefix, 2)
    labels = {}
    stages = study.workflow(case)["nodes"][1:]
    bits = bits or [1] * len(stages)
    for i, (node, bit) in enumerate(zip(stages, bits)):
        labels[node] = {"record_id": node, "sequence": 4 + 2 * i, "source": "constructed-api-fixture",
            "text": "Fixture report. " + (case["marker"] if bit else "No reference included."), "kind": "complete_message", "value": bit}
        if node.startswith("review-"):
            prefix[node] = {k: labels[node][k] for k in ("record_id", "sequence", "text")}
    late = study.prefix(case, prefix, max(v["sequence"] for v in prefix.values()) + 1)
    return case, study.compile_input(early), study.compile_input(late), labels


def test_cases_balanced_by_layout_cue_source_and_holdout_unrun():
    cases = study.cases()
    assert len(cases) == 48 and len({c["id"] for c in cases}) == 48
    for split in ("train", "validation"):
        subset = [c for c in cases if c["split"] == split]
        assert len(subset) == 12
        assert sum(len(study.workflow(c)["nodes"]) - 1 for c in subset) == 42
        assert sum(c["source_marker_present"] for c in subset) == 6
    assert len({c["marker"] for c in cases}) == 48


@pytest.mark.parametrize("layout", ["shared", "separate"])
def test_complete_parent_text_and_no_registry_or_other_branch_in_request(layout):
    case = fixture(layout)[0]
    messages = {"source": case["source"], "review-a": "Exact first review", "review-b": "Other review"}
    request = study.request(case, "publish-a", messages)
    body = json.loads(request["input"][0]["content"])
    assert body == {"task": case["task"], "reviewed_packet": messages["review-a"]}
    assert case["marker"] not in json.dumps(request)
    b = json.loads(study.request(case, "publish-b", messages)["input"][0]["content"])
    assert b["reviewed_packet"] == messages["review-a" if layout == "shared" else "review-b"]


@pytest.mark.parametrize("layout", ["shared", "separate"])
@pytest.mark.parametrize("phase", ["early", "late"])
def test_event_joint_drives_actual_knowledge_policy(layout, phase):
    case, early, late, labels = fixture(layout)
    sample = early if phase == "early" else late
    model = DAGWorkflowRiskBN()
    model.theta += np.random.default_rng(42).normal(0, .4, 28)
    p = model.predict(sample)
    joint = model.event_law.conditional(sample.event_dag, ["publish-a", "publish-b"], model.theta[:25])
    assert p["probabilities"]["system:any"] == pytest.approx(1 - np.exp(joint["log_probabilities"][0]))
    for query, probability in p["probabilities"].items():
        assert evaluate(model.probability_graph(sample, output=query)) == pytest.approx(probability, abs=1e-12)


def test_shared_future_review_correlation_disappears_when_its_text_is_observed():
    case, early, late, labels = fixture()
    model = DAGWorkflowRiskBN()
    model.theta[model.names.index("event:publish:bias")] = -2
    model.theta[model.names.index("event:publish:parent_mean")] = 4
    for sample, dependent in ((early, True), (late, False)):
        joint = model.event_law.conditional(sample.event_dag, ["publish-a", "publish-b"], model.theta[:25])
        probabilities = np.exp(joint["log_probabilities"])
        marginal = probabilities @ joint["states"]
        difference = abs(probabilities[-1] - np.prod(marginal))
        assert difference > .1 if dependent else difference < 1e-12


@pytest.mark.parametrize("partial", [False, True])
def test_intermediate_trace_likelihood_gradients_and_no_input_mutation(partial):
    case, early, late, labels = fixture(bits=[1, 1, 0])
    dag = early.event_dag
    if partial:
        labels = {"publish-a": labels["publish-a"]}
    before = dag.specification()
    law = DAGWorkflowRiskBN().event_law
    theta = np.random.default_rng(1).normal(0, .4, 25)
    value, gradient = trace_loss_gradient(law, theta, [dag], [labels], .03)
    for i in range(25):
        d = np.zeros(25); d[i] = 1e-5
        numeric = (trace_loss_gradient(law, theta + d, [dag], [labels], .03)[0] - trace_loss_gradient(law, theta - d, [dag], [labels], .03)[0]) / 2e-5
        assert numeric == pytest.approx(gradient[i], abs=2e-8)
    assert dag.specification() == before


@pytest.mark.parametrize("change", ["past_label", "wrong_literal", "future_in_prefix", "unknown_event"])
def test_future_supervision_never_silently_becomes_prefix_evidence(change):
    case, early, late, labels = fixture()
    labels = copy.deepcopy(labels)
    if change == "past_label": labels["review-a"]["sequence"] = 1
    elif change == "wrong_literal": labels["review-a"]["value"] ^= 1
    elif change == "unknown_event": labels["invented"] = labels.pop("review-a")
    else:
        completed = {"source": {"record_id": "source", "sequence": 1, "text": case["source"]},
                     "publish-a": {"record_id": "future", "sequence": 100, "text": "not yet available"}}
        with pytest.raises(ValueError): study.prefix(case, completed, 2)
        return
    with pytest.raises(ValueError): validated_labels(early.event_dag, labels)


def test_training_fits_event_parameters_but_does_not_claim_authority_fit():
    rows = [fixture(policy=policy, source_present=present, bits=[int(present and policy == "retain")] * 3)
            for policy in ("retain", "remove") for present in (False, True)]
    model = fit_event_traces([r[1].event_dag for r in rows], [r[3] for r in rows])
    assert model.optimizer["success"] and model.optimizer["authority_priors_fitted"] is False
    assert len(model.optimizer["fitted_parameter_names"]) == 25 and np.all(model.theta[25:] == 0)
    restored = DAGWorkflowRiskBN.restore(model.parameters())
    assert restored.predict(rows[0][1]) == model.predict(rows[0][1])


def test_identical_terminal_forecasts_can_hide_opposite_intermediate_meanings():
    case, early, late, labels = fixture()
    model = DAGWorkflowRiskBN()
    model.theta[model.names.index("event:review:bias")] = 2.
    model.theta[model.names.index("event:publish:bias")] = -2.
    model.theta[model.names.index("event:publish:parent_mean")] = 4.
    other = flip_review_representation(model, [early.event_dag])
    targets = ["publish-a", "publish-b"]
    a = model.event_law.conditional(early.event_dag, targets, model.theta[:25])
    b = other.event_law.conditional(early.event_dag, targets, other.theta[:25])
    assert np.allclose(a["log_probabilities"], b["log_probabilities"])
    scores = [intermediate_semantics(m, [early.event_dag], [labels], [targets])["by_family"]["review"] for m in (model, other)]
    assert scores[0]["accuracy_at_half"] == 1 and scores[1]["accuracy_at_half"] == 0
    complete = [trace_loss_gradient(m.event_law, m.theta[:25], [early.event_dag], [labels], 0.)[0] for m in (model, other)]
    assert complete[0] < complete[1]
    terminal = [{n: labels[n] for n in targets}]
    terminal_losses = [trace_loss_gradient(m.event_law, m.theta[:25], [early.event_dag], terminal, 0.)[0] for m in (model, other)]
    assert terminal_losses[0] == pytest.approx(terminal_losses[1], abs=1e-12)
    with pytest.raises(ValueError): flip_review_representation(model, [late.event_dag])


def test_unlabelled_intermediates_are_reported_as_unvalidated_not_safe():
    case, early, late, labels = fixture()
    result = intermediate_semantics(DAGWorkflowRiskBN(), [early.event_dag], [{"publish-a": labels["publish-a"]}], [["publish-a", "publish-b"]])
    scores = result["by_family"]["review"]
    assert scores["future_events"] == 1 and scores["labelled_events"] == 0 and scores["brier"] is None


@pytest.mark.parametrize("values,recall,fpr,both", [([0, 0], False, True, False), ([1, 1], True, False, False), ([0, 1], True, True, True)])
def test_class_coverage_does_not_invent_performance_when_a_class_is_absent(values, recall, fpr, both):
    coverage = binary_target_coverage(values)
    assert coverage["recall_estimable"] == recall
    assert coverage["false_positive_rate_estimable"] == fpr
    assert coverage["class_coverage_for_discrimination"] == both


@pytest.mark.parametrize("values", [[], [0, True], [0, 2], [0, None]])
def test_target_coverage_requires_actual_binary_labels(values):
    with pytest.raises(ValueError): binary_target_coverage(values)


def test_prepare_freezes_standalone_numeric_imports_and_audit_dependencies(tmp_path):
    import run_relational_forecast_study as runner
    destination = tmp_path / "freeze"
    runner.prepare(destination, Path(sys.executable))
    plan = json.loads((destination / "PLAN.json").read_text(encoding="utf-8"))
    assert plan["planned_development_calls"] == 84
    for name in ('code/commands/verify_relational_forecast_study.py', 'code/commands/audit_relational_event_meaning.py',
                 'code/src/paperalpha_monitor/observable_event_qualification.py'):
        assert name in plan["source_sha256"] and (destination / "source" / name).is_file()
    assert json.loads((destination / "IMPORT_CHECK.json").read_text(encoding="utf-8"))["exit_code"] == 0
