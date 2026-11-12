import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.text_conditioned_bn import (TextConditionedBN, JournalTextCompiler, compile_text_journal,
    PARAMETER_NAMES, BASE_SIZE, CHANNEL_WIDTH, FEATURE_NAMES, FEATURE_INDEX, grouped_features)
from paperalpha_monitor.compact_action_bn import CompactCapabilityBN
from paperalpha_monitor.dynamic_joint_bn import JointTrace
from paperalpha_runtime.semantic_evidence import TextEvidenceWindow
from test_capability_action_bn import context


def scenario():
    def record(n, kind, **payload):
        return {"schema_version": "runtime-journal-v1", "sequence": n, "kind": kind, "payload": payload}
    return [record(1, "run_started", agents=["reader", "executor"], external_endpoints=["user"]),
        record(2, "communication", actor="reader", target="executor", content="Do not pay yet."),
        record(3, "tool_handler_started", actor="executor", tool="payment_effect", invocation_id="call-1",
               provider_call_id="provider-1", arguments={"amount": 5}),
        record(4, "tool_handler_returned", actor="executor", tool="payment_effect", invocation_id="call-1",
               provider_call_id="provider-1", result="Payment is planned."),
        record(5, "communication", actor="executor", target="reader", content="I have approval."),
        record(6, "communication", actor="reader", target="user", content="No approval record is available."),
        record(7, "run_finished", status="completed")]


def item(quote, predicate):
    return {"predicate": predicate, "presentation": "direct", "quote": quote, "occurrence": 0,
            "condition_quotes": [], "quantity_quote": None, "references": []}


def annotations():
    cases = [(2, "Do not pay yet.", "action_deferred"), (4, "Payment is planned.", "execution_claimed_planned"),
             (5, "I have approval.", "permission_claimed_granted"), (6, "No approval record is available.", "evidence_claimed_missing")]
    return {f"journal-{n}": {"status": "observations", "more_possible": False, "observations": [item(text, kind)]}
            for n, text, kind in cases}


def trace():
    return compile_text_journal(scenario(), annotations(), **context())


@pytest.mark.parametrize("label,prefix", [(0, True), (1, True), (0, False), (1, False)])
def test_all_326_analytic_gradients_match_finite_differences(label, prefix):
    model, compiled = TextConditionedBN(), trace()
    theta = model.theta + np.random.default_rng(69).normal(0, .4, len(model.theta))
    assert len(theta) == len(PARAMETER_NAMES) == 326
    loss, analytic = model.episode_loss_gradient(theta, compiled, label, prefix)
    numeric = approx_derivative(lambda t: model.episode_loss_gradient(t, compiled, label, prefix)[0], theta).ravel()
    assert math.isfinite(loss)
    np.testing.assert_allclose(analytic, numeric, rtol=3e-5, atol=4e-8)
    assert np.any(np.abs(analytic[BASE_SIZE:]) > 1e-8)


def independent_transition(mass, step, theta):
    """Dictionary enumeration, no production transition/state-bit routines."""
    def move(source, transform):
        output = {}
        for (bits, history), probability in source.items():
            for destination, weight in transform(bits, history):
                output[destination] = output.get(destination, 0.) + probability * weight
        return output
    sigmoid = lambda v: 1 / (1 + math.exp(-v))
    if step.revision is not None:
        revision = step.revision
        start = BASE_SIZE + CHANNEL_WIDTH * revision.channel
        def revise(bits, history):
            previous = (bits >> revision.node) & 1
            logit = theta[start] + theta[start + 1] * previous
            logit += sum(float(a) * float(b) for a, b in zip(theta[start + 2:start + CHANNEL_WIDTH], revision.features))
            q = sigmoid(logit)
            return [((bits & ~(1 << revision.node), history), 1 - q), ((bits | (1 << revision.node), history), q)]
        mass = move(mass, revise)
    from paperalpha_monitor.capability_action_bn import CAP_CONFIDENCE
    for operation in step.base.operations:
        def transmit(bits, history):
            q = sigmoid(theta[operation.parameter] + theta[CAP_CONFIDENCE] * operation.confidence_feature)
            allowed = operation.sender is None or (bits >> operation.sender) & 1
            if (bits >> operation.receiver) & 1 or not allowed:
                return [((bits, history), 1.)]
            return [((bits, history), 1 - q), ((bits | (1 << operation.receiver), history), q)]
        mass = move(mass, transmit)
    for dispatch in step.base.dispatches:
        def do_action(bits, history):
            q = sigmoid(theta[dispatch.parameter])
            next_history = 2 if ((bits >> dispatch.node) & 1) or history == 2 else 1
            return [((bits, history), 1 - q), ((bits, next_history), q)]
        mass = move(mass, do_action)
    return mass


def test_full_joint_state_matches_independent_dictionary_and_preserves_historical_marginal():
    model, compiled = TextConditionedBN(), trace()
    for seed in range(3):
        theta = model.theta + np.random.default_rng(seed).normal(0, .7, len(model.theta))
        engine, mass = model._engine(theta, derivatives=True), {(0, 0): 1.}
        for step in compiled.steps:
            before = engine.distribution.sum(axis=0).copy()
            engine.apply(step, compiled.variables)
            mass = independent_transition(mass, step, theta)
            expected = np.zeros_like(engine.distribution)
            for (bits, history), probability in mass.items():
                expected[bits, history] = probability
            np.testing.assert_allclose(engine.distribution, expected, atol=2e-15, rtol=0)
            np.testing.assert_allclose(engine.jacobian.sum(axis=(0, 1)), 0., atol=1e-15)
            if not step.base.dispatches:
                np.testing.assert_allclose(before, engine.distribution.sum(axis=0), atol=2e-15)


def test_text_disabled_exactly_recovers_v4_on_same_base_events():
    compiled = trace()
    new, old = TextConditionedBN(use_text=False), CompactCapabilityBN()
    old.theta += np.random.default_rng(27).normal(0, .6, len(old.theta))
    new.theta[:BASE_SIZE] = old.theta
    legacy = JointTrace(compiled.variables, tuple(s.base for s in compiled.steps), compiled.workflow)
    np.testing.assert_allclose(new.predict([compiled]), old.predict([legacy]), rtol=0, atol=2e-15)
    _, grad = new.episode_loss_gradient(new.theta, compiled, 1)
    np.testing.assert_array_equal(grad[BASE_SIZE:], 0.)


def test_text_changes_state_and_propagation_without_granting_permission():
    model, compiled = TextConditionedBN(), trace()
    index = PARAMETER_NAMES.index("text:communication:claim:action_deferred:direct")
    model.theta[index] = -2
    low = model.predict([compiled])[0]
    model.theta[index] = 2
    high = model.predict([compiled])[0]
    assert np.max(np.abs(high - low)) > 1e-4
    model.use_messages = False
    no_messages = model.predict([compiled])[0]
    assert np.max(np.abs(high - no_messages)) > 1e-5
    explanation = model.explain(compiled)
    assert explanation["actions"][0]["permission_status"] == "unknown"
    assert explanation["actions"][0]["accepted_check"] is None
    assert explanation["state_interpretation"].startswith("process-concern")


def test_masked_features_leave_only_text_channel_and_previous_state_effects():
    model = TextConditionedBN(mask_text_features=True)
    _, grad = model.episode_loss_gradient(model.theta, trace(), 0)
    for channel in range(2):
        start = BASE_SIZE + channel * CHANNEL_WIDTH
        np.testing.assert_array_equal(grad[start + 2:start + CHANNEL_WIDTH], 0.)
        assert np.any(np.abs(grad[start:start + 2]) > 0)


def test_future_text_and_future_annotations_do_not_change_existing_prefixes():
    rows, proposed = scenario(), annotations()
    model = TextConditionedBN()
    model.theta += np.random.default_rng(4).normal(0, .5, len(model.theta))
    full = model.predict([trace()])[0]
    for stop in range(2, 7):
        prefix = rows[:stop]
        allowed = {f"journal-{r['sequence']}" for r in prefix}
        compiled = compile_text_journal(prefix, {k: v for k, v in proposed.items() if k in allowed}, **context())
        np.testing.assert_allclose(model.predict([compiled])[0], full[:len(compiled.steps)], atol=2e-15, rtol=0)
    with pytest.raises(ValueError, match="outside"):
        compile_text_journal(rows[:2], proposed, **context())


def test_missing_and_invalid_annotations_are_unknown_not_safe_or_authorized():
    missing = compile_text_journal(scenario(), **context())
    for step in missing.steps:
        if step.revision:
            assert step.revision.features[FEATURE_INDEX["extraction_unavailable"]] == 1
    broken = annotations()
    broken["journal-2"]["safe"] = True
    compiled = compile_text_journal(scenario(), broken, **context())
    assert compiled.steps[0].revision.evidence["batch"] is None
    assert compiled.steps[0].revision.features[FEATURE_INDEX["extraction_unavailable"]] == 1


def test_same_predicate_twice_has_presence_not_two_independent_votes():
    rows = scenario()
    rows[1]["payload"]["content"] = "Wait. Wait."
    window = TextEvidenceWindow()
    window.push(rows[0])
    packet = window.push(rows[1])
    first = item("Wait.", "action_deferred")
    second = {**first, "occurrence": 1}
    one = {"status": "observations", "more_possible": False, "observations": [first]}
    two = {**one, "observations": [first, second]}
    x1, _ = grouped_features(packet, one)
    x2, evidence = grouped_features(packet, two)
    np.testing.assert_array_equal(x1, x2)
    assert evidence["batch"]["accepted_items"] == 2
    assert not evidence["batch"]["independent_evidence"]


def test_nontext_annotation_and_bad_runtime_record_fail_atomically():
    compiler = JournalTextCompiler(**context())
    compiler.push(scenario()[0])
    before = copy.deepcopy(compiler.__dict__)
    with pytest.raises(ValueError):
        compiler.push(scenario()[2], annotations()["journal-2"])
    assert compiler.projector.sequence == before["projector"].sequence
    compiler.push(scenario()[1], annotations()["journal-2"])
    with pytest.raises(ValueError, match="non-text"):
        compiler.push(scenario()[2], annotations()["journal-2"])
    assert compiler.projector.sequence == 2 and len(compiler.window.history) == 1


def test_optimizer_can_fit_component_fixture_without_reporting_accuracy():
    model = TextConditionedBN(max_iter=3)
    compiled = trace()
    before = model.loss_gradient(model.theta, [compiled], [1])[0]
    model.fit([compiled], [1])  # Fabricated label solely a numerical training check.
    after = model.loss_gradient(model.theta, [compiled], [1])[0]
    assert after < before and len(model.parameters()["theta"]) == 326


def test_v5_explicitly_exposes_numeric_blind_spot_not_a_budget_comparator():
    def with_amount(amount):
        rows, proposed = scenario(), annotations()
        text = f"Budget cap ${amount}."
        rows[1]["payload"]["content"] = text
        observation = item(text, "normative_upper_bound")
        observation["quantity_quote"] = {"quote": f"${amount}", "occurrence": 0}
        proposed["journal-2"]["observations"] = [observation]
        return compile_text_journal(rows, proposed, **context())
    model = TextConditionedBN()
    model.theta += np.random.default_rng(51).normal(0, .8, len(model.theta))
    a, b = with_amount(400), with_amount(40000)
    np.testing.assert_array_equal(a.steps[0].revision.features, b.steps[0].revision.features)
    np.testing.assert_array_equal(model.predict([a])[0], model.predict([b])[0])
    # This is a documented v5 limitation, not a requirement for future models.


def test_later_new_agent_allocation_does_not_change_earlier_scores():
    rows = scenario()
    rows[0]["payload"]["agents"].append("late-agent")
    rows[-1] = {"schema_version": "runtime-journal-v1", "sequence": 7, "kind": "communication",
                "payload": {"actor": "late-agent", "target": "reader", "content": "No record is available."}}
    model = TextConditionedBN()
    full = compile_text_journal(rows, annotations(), **context())
    prefix = compile_text_journal(rows[:6], annotations(), **context())
    assert len(full.variables) == 3 and len(prefix.variables) == 2
    np.testing.assert_allclose(model.predict([full])[0][:-1], model.predict([prefix])[0], atol=2e-15, rtol=0)


def test_default_weights_are_not_misrepresented_as_semantic_training():
    model, masked = TextConditionedBN(), TextConditionedBN(mask_text_features=True)
    compiled = trace()
    np.testing.assert_array_equal(model.predict([compiled]), masked.predict([compiled]))
    _, gradient = model.episode_loss_gradient(model.theta, compiled, 1)
    feature_gradients = np.r_[gradient[BASE_SIZE + 2:BASE_SIZE + CHANNEL_WIDTH], gradient[BASE_SIZE + CHANNEL_WIDTH + 2:]]
    assert np.any(np.abs(feature_gradients) > 1e-8)
    assert not model.optimizer
