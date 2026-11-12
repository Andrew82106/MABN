import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.action_mechanism_knowledge import ActionMechanismRegistry, load_action_mechanisms
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN, JournalKnowledgeCompiler, compile_knowledge_journal
from paperalpha_monitor.text_conditioned_bn import TextConditionedBN, TextStep, BASE_SIZE, CHANNEL_WIDTH
from paperalpha_monitor.capability_action_bn import CAP_HEAD, CAP_CONFIDENCE
from paperalpha_monitor.dynamic_joint_bn import JointTrace
from test_scoped_budget_bn import scenario, context


def subregistry(indices):
    spec = load_action_mechanisms().specification()
    spec["mechanisms"] = [spec["mechanisms"][i] for i in indices]
    catalog = json.loads((ROOT / 'knowledge/risks/risk_catalog.json').read_text(encoding="utf-8"))
    return ActionMechanismRegistry(spec, catalog)


def independent_enumeration(trace, theta, registry):
    """Keep ALL dispatched binary factors and process snapshots, no compression.

    Uses scalar dict enumeration rather than the production history transition,
    latent feature builder or engine. Only compiled factor/evidence definitions
    are shared, as the object whose probability semantics we check.
    """
    sigmoid = lambda x: 1 / (1 + math.exp(-float(x)))
    mass, scores, distributions = {(0, ()): 1.}, [], []

    def move(choices):
        nonlocal mass
        updated = {}
        for state, value in mass.items():
            for destination, probability in choices(*state):
                updated[destination] = updated.get(destination, 0.) + value * probability
        mass = updated

    for step in trace.steps:
        if step.revision is not None:
            revision = step.revision
            start = BASE_SIZE + CHANNEL_WIDTH * revision.channel
            def revise(bits, actions):
                previous = (bits >> revision.node) & 1
                q = sigmoid(theta[start] + theta[start + 1] * previous + sum(
                    float(a) * float(b) for a, b in zip(theta[start + 2:start + CHANNEL_WIDTH], revision.features)))
                return [((bits & ~(1 << revision.node), actions), 1 - q), ((bits | (1 << revision.node), actions), q)]
            move(revise)
        for op in step.base.operations:
            def transmit(bits, actions):
                q = sigmoid(theta[op.parameter] + theta[CAP_CONFIDENCE] * (op.confidence_feature if op.is_message else 0.))
                active = not ((bits >> op.receiver) & 1) and (op.sender is None or (bits >> op.sender) & 1)
                if not active:
                    return [((bits, actions), 1.)]
                return [((bits, actions), 1 - q), ((bits | (1 << op.receiver), actions), q)]
            move(transmit)
        for factor in step.mechanism_factors:
            def dispatch(bits, actions):
                q = sigmoid(sum(theta[i] * v for i, v in zip(factor.parameter_indices, factor.values)))
                c = (bits >> factor.node) & 1
                return [((bits, actions + ((factor.channel, 0, c),)), 1 - q),
                        ((bits, actions + ((factor.channel, 1, c),)), q)]
            move(dispatch)
        score, collapsed = 0., np.zeros((1 << len(trace.variables), 3 ** len(registry.channels)))
        for (bits, actions), probability in mass.items():
            history = []
            for k in range(len(registry.channels)):
                this = [(r, c) for channel, r, c in actions if channel == k]
                history.append(2 if any(r and c for r, c in this) else int(any(r for r, c in this)))
            index = 0
            for h in history:
                index = index * 3 + h
            collapsed[bits, index] += probability
            count = sum((bits >> i) & 1 for i in step.base.observed_nodes)
            n = len(step.base.observed_nodes)
            info = [count / max(1, n), count * (count - 1) / max(1, n * (n - 1))]
            for nodes in (step.base.capable_nodes, step.base.terminal_nodes):
                info.append(sum((bits >> i) & 1 for i in nodes) / max(1, len(nodes)))
            b = theta[CAP_HEAD:CAP_HEAD + 23]
            logit = sum(a * w for a, w in zip(step.base.observables, b[:15])) + sum(a * w for a, w in zip(info, b[15:19]))
            logit += b[20] * (history[0] > 0) + b[22] * (history[0] == 2)
            for k, name in enumerate(registry.channels[1:], 1):
                i, j = registry.head_indices[name]
                logit += theta[i] * (history[k] > 0) + theta[j] * (history[k] == 2)
            score += probability * sigmoid(logit)
        scores.append(score)
        distributions.append(collapsed)
    return scores, distributions


@pytest.mark.parametrize("label,prefix", [(0, True), (1, True), (0, False), (1, False)])
def test_all_345_gradients_match_numeric_derivatives(label, prefix):
    model = KnowledgeScopedBN()
    assert len(model.theta) == len(model.parameter_names) == 345
    trace = compile_knowledge_journal(scenario(), **context())
    theta = model.theta + np.random.default_rng(312).normal(0, .4, len(model.theta))
    _, analytic = model.episode_loss_gradient(theta, trace, label, prefix)
    numeric = approx_derivative(lambda t: model.episode_loss_gradient(t, trace, label, prefix)[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=5e-8, rtol=3e-5)
    for indices in model.registry.numeric_slices.values():
        assert np.any(np.abs(analytic[list(indices)]) > 1e-8)


@pytest.mark.parametrize("indices", [(0,), (0, 1), (0, 2), (0, 1, 2)])
def test_exact_history_compression_matches_uncompressed_independent_enumeration(indices):
    registry = subregistry(indices)
    trace = compile_knowledge_journal(scenario(), mechanism_registry=registry, **context())
    model = KnowledgeScopedBN(mechanism_registry=registry)
    theta = model.theta + np.random.default_rng(56).normal(0, .6, len(model.theta))
    scores, masses = independent_enumeration(trace, theta, registry)
    engine = model._engine(theta)
    for step, expected, mass in zip(trace.steps, scores, masses):
        assert engine.apply(step, trace.variables) == pytest.approx(expected, abs=3e-15)
        np.testing.assert_allclose(engine.distribution, mass, rtol=0, atol=3e-15)
        assert engine.distribution.sum() == pytest.approx(1.)


def test_numeric_factors_do_not_change_authorization_or_process_marginals():
    trace = compile_knowledge_journal(scenario(), **context())
    model = KnowledgeScopedBN()
    theta = model.theta.copy()
    theta[326:] += np.random.default_rng(49).normal(0, 1, len(theta) - 326)
    a, b = model._engine(model.theta), model._engine(theta)
    for step in trace.steps:
        a.apply(step, trace.variables)
        b.apply(step, trace.variables)
        np.testing.assert_allclose(a.distribution.reshape(-1, 3, 3, 3).sum(axis=(2, 3)),
            b.distribution.reshape(-1, 3, 3, 3).sum(axis=(2, 3)), rtol=0, atol=3e-15)
    assert not np.allclose(a.distribution, b.distribution)
    for x, y in zip(a.explain(trace.steps[-1], trace.variables)["actions"], b.explain(trace.steps[-1], trace.variables)["actions"]):
        assert x["mechanism_factors"][0] == y["mechanism_factors"][0]


def test_workflow_group_and_tool_capability_select_different_actual_subgraphs():
    rows = scenario()
    rows[2]["payload"]["tool"] = rows[3]["payload"]["tool"] = "get_customer_context"
    ctx = context()
    ctx["workflow"] = {"nodes": ["research", "execution"], "edges": [["research", "execution"]],
        "agent_to_node": {"a": "research", "b": "execution"}, "roles": {"research": "reader", "execution": "executor"}}
    ctx.pop("roles")
    trace = compile_knowledge_journal(rows, **ctx)
    explained = KnowledgeScopedBN().explain(trace)
    first, second = explained["actions"]
    assert first["node"] == "research" and first["role"] == "reader"
    assert [f["template"] for f in first["mechanism_factors"]] == ["authorization"]
    assert second["node"] == "execution" and second["role"] == "executor"
    assert [f["template"] for f in second["mechanism_factors"]] == ["authorization", "declared_bound", "cumulative_budget"]
    assert first["mechanism_factors"][0]["parameter_names"] != second["mechanism_factors"][0]["parameter_names"]


def test_no_contract_is_absent_not_a_fake_safe_budget_probability():
    ctx = context()
    ctx.pop("budget_contracts")
    trace = compile_knowledge_journal(scenario(), **ctx)
    result = KnowledgeScopedBN().explain(trace)
    assert result["mechanisms"]["cumulative_budget"]["concern_probability"] is None
    assert not result["mechanisms"]["cumulative_budget"]["instantiated"]
    assert all(len(a["mechanism_factors"]) == 2 for a in result["actions"])
    assert all(not t["instantiated"] for d in result["selection_decisions"] for t in d["templates"] if t["template"] == "cumulative_budget")


def test_numeric_histories_cannot_influence_readout_when_latent_head_is_disabled():
    trace = compile_knowledge_journal(scenario(), **context())
    model = KnowledgeScopedBN(use_latent_head=False)
    model.theta += np.random.default_rng(80).normal(0, .6, len(model.theta))
    _, gradient = model.episode_loss_gradient(model.theta, trace, 1)
    np.testing.assert_allclose(gradient[326:], 0., atol=3e-15)


def test_temporal_text_revision_cannot_rewrite_dispatched_mechanism_histories():
    rows = scenario()
    rows.insert(-1, {"schema_version": "runtime-journal-v1", "sequence": 7, "kind": "communication",
        "payload": {"actor": "a", "target": "b", "content": "Everything is fine now."}})
    rows[-1]["sequence"] = 8
    trace = compile_knowledge_journal(rows, **context())
    model = KnowledgeScopedBN()
    model.theta += np.random.default_rng(11).normal(0, .6, len(model.theta))
    engine = model._engine(model.theta)
    for step in trace.steps[:-1]:
        engine.apply(step, trace.variables)
    history = engine.distribution.sum(axis=0).copy()
    engine.apply(trace.steps[-1], trace.variables)
    np.testing.assert_allclose(engine.distribution.sum(axis=0), history, atol=3e-15, rtol=0)


def test_zero_new_readout_exactly_recovers_v5_without_merging_budget_and_permission():
    trace = compile_knowledge_journal(scenario(), **context())
    legacy = JointTrace(trace.variables, tuple(TextStep(s.base, s.revision) for s in trace.steps), trace.workflow)
    old, new = TextConditionedBN(), KnowledgeScopedBN()
    old.theta += np.random.default_rng(42).normal(0, .7, len(old.theta))
    new.theta[:326] = old.theta
    np.testing.assert_allclose(new.predict([trace]), old.predict([legacy]), atol=3e-15, rtol=0)
    assert new.parameters()["parameter_names"] == list(new.parameter_names)


def test_cumulative_concern_can_affect_head_without_changing_permission_factor():
    model = KnowledgeScopedBN()
    indices = model.registry.numeric_slices["cumulative_budget"]
    feature_index = model.registry.templates[2]["features"].index("global_attempted_exceeded") + 1
    model.theta[indices[feature_index]] = 3.
    model.theta[model.registry.head_indices["cumulative_budget"][0]] = 2.
    below = compile_knowledge_journal(scenario("199"), **context())
    above = compile_knowledge_journal(scenario("300"), **context())
    a, b = model.explain(below), model.explain(above)
    assert b["risk_probability"] > a["risk_probability"]
    assert b["mechanisms"]["cumulative_budget"]["concern_probability"] > a["mechanisms"]["cumulative_budget"]["concern_probability"]
    np.testing.assert_allclose(b["mechanisms"]["authorization"]["history_probabilities"],
        a["mechanisms"]["authorization"]["history_probabilities"], rtol=0, atol=3e-15)
    assert sum(t["risk_contribution"] for t in b["system_mixture_terms"]) == pytest.approx(b["risk_probability"])


def test_incremental_prefix_and_future_isolation():
    rows = scenario()
    model = KnowledgeScopedBN()
    model.theta += np.random.default_rng(81).normal(0, .5, len(model.theta))
    full = compile_knowledge_journal(rows, **context())
    expected = model.predict([full])[0]
    compiler, engine, scores = JournalKnowledgeCompiler(**context()), model._engine(model.theta), []
    for n, row in enumerate(rows):
        step = compiler.push(row)
        if step is not None:
            scores.append(engine.apply(step, compiler.variables))
            partial = compile_knowledge_journal(rows[:n + 1], **context())
            assert scores[-1] == pytest.approx(model.predict([partial])[0][-1], abs=3e-15)
    np.testing.assert_allclose(scores, expected, atol=3e-15, rtol=0)


def test_registry_mismatch_and_atomic_rejected_event():
    compiler = JournalKnowledgeCompiler(**context())
    for row in scenario()[:4]:
        compiler.push(row)
    old = copy.deepcopy(compiler.last_step)
    broken = scenario()[4]
    broken["payload"]["invocation_id"] = "first"
    with pytest.raises(ValueError):
        compiler.push(broken)
    assert compiler.last_step.sequence == old.sequence
    assert compiler.last_step.mechanism_factors == old.mechanism_factors
    bad = replace(old, registry_sha256="not-matching")
    with pytest.raises(ValueError, match="registry mismatch"):
        KnowledgeScopedBN()._engine(KnowledgeScopedBN().theta).apply(bad, compiler.variables)


@pytest.mark.parametrize("mutation", ["source", "feature", "selector", "authorization", "duplicate"])
def test_registry_rejects_false_provenance_and_unsupported_semantics(mutation):
    spec = load_action_mechanisms().specification()
    if mutation == "source":
        spec["mechanisms"][2]["knowledge_links"][0]["source_ids"] = [11]
    elif mutation == "feature":
        spec["mechanisms"][1]["features"].append("test_label")
    elif mutation == "selector":
        spec["mechanisms"][1]["selector"] = "agent_says_approved"
    elif mutation == "authorization":
        spec["mechanisms"][0]["factor"] = "logistic_observed_features"
    else:
        spec["mechanisms"][2]["id"] = spec["mechanisms"][1]["id"]
    with pytest.raises(ValueError):
        ActionMechanismRegistry(spec, json.loads((ROOT / 'knowledge/risks/risk_catalog.json').read_text(encoding="utf-8")))


def test_small_training_objective_decreases_not_a_performance_experiment():
    model = KnowledgeScopedBN(max_iter=5)
    traces = [compile_knowledge_journal(scenario(amount), **context()) for amount in ("100", "900")]
    labels = [0, 1]  # Hand-authored component labels, not independent task outcomes.
    before = model.loss_gradient(model.theta, traces, labels)[0]
    model.fit(traces, labels)
    assert model.loss_gradient(model.theta, traces, labels)[0] < before
