from collections import defaultdict
import copy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/commands'), str(ROOT / 'code/src')]
from audit_delegation_joint_context import shared_root_fixture
from monitor_delegation_pilot import make_compiler
from paperalpha_monitor.delegation_evidence import DelegationLedger
from paperalpha_monitor.shared_authority_graph import AuthorityWindow, AuthorityProofGraph, SharedAuthorityTrace, SharedAuthorityCompiler
from paperalpha_monitor.shared_authority_joint_bn import SharedAuthorityJointBN, SharedAuthorityStream
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeFilter
from paperalpha_runtime.delegation_pilot import configuration, PilotEnvironment
from test_delegation_evidence import grant, action
from test_qualified_delegation_cli import observed


def bundle(shared=True, *, complete=False):
    compiler, _ = shared_root_fixture(shared)
    graph = AuthorityProofGraph(AuthorityWindow("test-window", 0, 100,
        ("child-a", "child-b", "root-a", "root-b") if complete else ("child-a", "child-b")))
    for snapshot in compiler.snapshots.values():
        graph.add(snapshot)
    return SharedAuthorityTrace(compiler.trace(), graph)


def active_bundle():
    config, records = observed("normal")
    config["coverage_complete"] = False
    compiler = SharedAuthorityCompiler(make_compiler(config), AuthorityWindow("active", 0, 100, ("delegation-0001",)))
    events = [*records[:2], {"event_id": "source", "sequence": 3, "actor": "planner", "event_type": "input_received",
        "source_id": "one-observed-source", "data_class": "untrusted"}, {**records[2], "sequence": 4},
        {**records[3], "sequence": 5}, {**records[2], "sequence": 6, "event_id": "second-message"},
        {**records[3], "sequence": 7, "event_id": "second-payment", "action_id": "second-payment"}]
    for event in events:
        compiler.push(event)
    return compiler.trace()


def alternate(shared=False, *, complete=False):
    ledger = DelegationLedger(root_principals=["user"], trusted_observers=["adapter"], coverage_complete=complete)
    records = [grant(1, "root-a", "user", "planner"), grant(2, "root-b", "user", "planner"),
        grant(3, "child-a", "planner", "worker", "root-a"),
        grant(4, "child-b", "planner", "worker", "root-a" if shared else "root-b")]
    for record in records:
        ledger.observe(record)
    graph = AuthorityProofGraph(AuthorityWindow("alternate", 0, 100, ("child-a", "child-b")))
    graph.add(ledger.assess(action(), sequence=5, task_id="task-a", grant_ids=["child-a", "child-b"]))
    return graph


@pytest.mark.parametrize("shared,any_failure,two_failure", [(True, .2, .2), (False, .36, .04)])
def test_shared_root_preserves_joint_dependence_with_identical_action_marginals(shared, any_failure, two_failure):
    graph, priors = bundle(shared).graph, [logit(.2), logit(.6), 0.]
    _, _, mass, _, failed = graph.worlds(priors)
    assert all(float(mass @ values) == pytest.approx(.2) for values in failed.values())
    assert graph.query(priors, at_least=1)[0] == pytest.approx(any_failure)
    assert graph.query(priors, at_least=2)[0] == pytest.approx(two_failure)
    assert graph.query(priors, at_least=3)[0] == 0.


@pytest.mark.parametrize("shared,probability", [(False, .04), (True, .2)])
def test_alternative_paths_are_or_not_independent_evidence_copies(shared, probability):
    graph = alternate(shared)
    assert graph.query([logit(.2), 0, 0])[0] == pytest.approx(probability)


def test_complete_valid_records_are_deterministic_not_a_nonzero_learned_prior():
    graph = alternate(complete=True)
    assert not graph.facts
    for priors in ([0., 0., 0.], [8., 8., 8.], [-8., -8., -8.]):
        p, dp = graph.query(priors)
        assert p == 0 and np.max(np.abs(dp)) == 0


@pytest.mark.parametrize("variant,expected", [("normal", 0.), ("revoked", 1.), ("narrow_scope", 1.)])
def test_known_policy_observations_reduce_to_reference_rule(variant, expected):
    cfg, records = observed(variant)
    compiler = SharedAuthorityCompiler(make_compiler(cfg), AuthorityWindow("known", 0, 100))
    for r in records:
        compiler.push(r)
    assert compiler.graph.query([0, 0, 0])[0] == expected


def test_missing_and_unverified_chain_remains_opaque_not_shared_fake_root():
    cfg, records = observed("missing_root")
    compiler = SharedAuthorityCompiler(make_compiler(cfg), AuthorityWindow("missing", 0, 100))
    for r in records:
        compiler.push(r)
    assert {f.kind for f in compiler.graph.facts.values()} == {2}
    assert compiler.graph.query([logit(.2), logit(.3), logit(.7)])[0] == pytest.approx(.7)
    assert compiler.graph.specification()["proves_safety"] is False


def test_all_383_joint_gradients_match_finite_difference():
    trace, model = bundle(False), SharedAuthorityJointBN(regularization=.001)
    theta = model.theta + np.random.default_rng(731).normal(0, .25, len(model.theta))
    assert len(theta) == 383
    _, analytic = model.loss_gradient(theta, [trace], [1])
    numeric = approx_derivative(lambda t: model.loss_gradient(t, [trace], [1])[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=5e-8, rtol=2e-5)
    assert abs(analytic[model.base_count]) > 1e-6
    assert np.any(np.abs(analytic[:model.base_count]) > 1e-6)


def test_reference_query_gradient_and_mass_match_independent_scalar_enumeration():
    graph = alternate(False)
    theta = np.array([.7, -.8, .3])
    p, derivative = graph.query(theta)
    q = 1 / (1 + np.exp(-theta[0]))
    assert p == pytest.approx(q * q)
    assert derivative[0] == pytest.approx(2 * q * q * (1 - q))
    numeric = approx_derivative(lambda t: graph.query(t)[0], theta).ravel()
    np.testing.assert_allclose(derivative, numeric, atol=1e-8)
    assert graph.worlds(theta)[2].sum() == pytest.approx(1.)


def test_root_relationship_affects_system_head_not_only_explanation():
    model = SharedAuthorityJointBN()
    model.theta[model.base_count] = logit(.2)
    shared = model.forward(bundle(True))["probabilities"]
    independent = model.forward(bundle(False))["probabilities"]
    assert shared[0] == pytest.approx(independent[0])
    assert abs(shared[-1] - independent[-1]) > .01


def test_system_mixture_is_not_independent_marginal_plug_in():
    trace, model = bundle(True), SharedAuthorityJointBN()
    model.theta[model.base_count] = logit(.2)
    result = model.forward(trace)
    # Independent oracle: in one world BOTH actions have q=0; in the other
    # BOTH have q=1. Enumerate history explicitly through immutable snapshots.
    b = model.theta
    from paperalpha_monitor.capability_action_bn import CAP_HEAD
    last = trace.trace.steps[-1].base
    clean = 1 / (1 + np.exp(-(last.observables @ b[CAP_HEAD:CAP_HEAD + 15])))
    failure = 1 / (1 + np.exp(-(last.observables @ b[CAP_HEAD:CAP_HEAD + 15] + b[CAP_HEAD + 20])))
    assert result["probabilities"][-1] == pytest.approx(.8 * clean + .2 * failure)


def test_future_revocation_does_not_rewrite_prior_action_proof_or_emitted_score():
    config, records = observed("normal")
    config["coverage_complete"] = False
    compiler = SharedAuthorityCompiler(make_compiler(config), AuthorityWindow("stream", 0, 100, ("delegation-0001",)))
    model = SharedAuthorityJointBN()
    stream = SharedAuthorityStream(model, compiler, allow_untrained_diagnostic=True)
    for r in records:
        report = stream.push(r)
    old, graph = copy.deepcopy(report), copy.deepcopy(compiler.graph)
    revoke = {"event_id": "future-revoke", "sequence": 20, "kind": "revoke", "observer": config["observer"],
        "authenticated_actor": "user", "payload": {"grant_id": "root"}}
    assert not stream.push(revoke)["dispatches"]
    new_action = {**records[-1], "event_id": "second-action", "sequence": 21, "action_id": "second-action"}
    now = stream.push(new_action)
    assert now["dispatches"][0]["reference_policy_any_failure"] == 1.
    assert compiler.graph.proofs["payment-0001"] == graph.proofs["payment-0001"]
    assert report == old
    batch = model.forward(compiler.trace())["probabilities"]
    assert batch[1] == pytest.approx(old["dispatches"][0]["model_point_probability"])


def test_capacity_and_window_errors_are_transactional():
    config, records = observed("normal")
    config["coverage_complete"] = False
    compiler = SharedAuthorityCompiler(make_compiler(config), AuthorityWindow("small", 0, 30), max_facts=0)
    for r in records[:-1]:
        compiler.push(r)
    previous = compiler.base.sequence
    with pytest.raises(ValueError, match="capacity"):
        compiler.push(records[-1])
    assert compiler.base.sequence == previous and not compiler.graph.proofs
    with pytest.raises(ValueError, match="window"):
        compiler.push({"sequence": 31})
    assert compiler.base.sequence == previous


def test_binding_and_graph_tampering_are_rejected():
    trace = bundle(True)
    key = next(iter(trace.graph.proofs))
    trace.graph.proofs[key] = replace(trace.graph.proofs[key], paths=((),))
    with pytest.raises(ValueError, match="graph changed"):
        SharedAuthorityJointBN().forward(trace)


def test_joint_state_capacity_never_silently_drops_shared_dependence():
    with pytest.raises(ValueError, match="capacity"):
        SharedAuthorityJointBN(max_joint_states=1).forward(bundle(False))


def test_reference_only_training_does_not_qualify_system_output():
    model = SharedAuthorityJointBN(regularization=0)
    traces = [bundle(True)] * 10 + [bundle(False)] * 25
    labels = [1] * 2 + [0] * 8 + [1] * 9 + [0] * 16
    model.fit(traces, labels, target="reference_failure_count")
    assert model.optimizer["success"] and model.optimizer["loss"] < model.optimizer["initial_loss"]
    assert 1 / (1 + np.exp(-model.theta[model.base_count])) == pytest.approx(.2, abs=2e-5)
    np.testing.assert_array_equal(model.theta[:model.base_count], model.initial[:model.base_count])
    assert model.fitted_targets == ["reference_failure_count:1"]
    config, _ = observed("normal")
    with pytest.raises(ValueError, match="own supervision"):
        SharedAuthorityStream(model, SharedAuthorityCompiler(make_compiler(config), AuthorityWindow("unfitted", 0, 100)))


def test_joint_system_training_reduces_endpoint_loss():
    model = SharedAuthorityJointBN(regularization=.001, max_iter=60)
    model.fit([bundle(True), bundle(False)], [0, 1])
    assert model.optimizer["loss"] < model.optimizer["initial_loss"] - .01
    assert model.fitted_targets == ["system_outcome"]
    assert abs(model.theta[model.base_count] - model.initial[model.base_count]) > 1e-4


def test_information_propagation_and_shared_fact_gradients_are_both_active():
    model, trace = SharedAuthorityJointBN(), active_bundle()
    result = model.forward(trace, derivatives=True)
    transmission = [i for i, name in enumerate(model.parameter_names) if name.startswith("transmission:")]
    assert np.max(np.abs(result["gradient"][-1, transmission])) > 1e-6
    assert abs(result["gradient"][-1, model.base_count]) > 1e-6
    # Directional derivatives include active process, shared prior and head.
    rng = np.random.default_rng(1909)
    for _ in range(5):
        direction = rng.normal(0, 1, len(model.theta))
        direction /= np.linalg.norm(direction)
        numeric = (model.forward(trace, model.theta + 1e-5 * direction)["probabilities"][-1]
            - model.forward(trace, model.theta - 1e-5 * direction)["probabilities"][-1]) / 2e-5
        assert numeric == pytest.approx(result["gradient"][-1] @ direction, abs=2e-9)


def test_full_scalar_enumeration_without_history_compression_matches_shared_joint_bn():
    import itertools
    import math
    from paperalpha_monitor.capability_action_bn import CAP_HEAD, CAP_CONFIDENCE
    model, bundle_ = SharedAuthorityJointBN(), active_bundle()
    model.theta += np.random.default_rng(1910).normal(0, .2, len(model.theta))
    theta, trace, graph = model.theta, bundle_.trace, bundle_.graph
    sigmoid = lambda x: 1 / (1 + math.exp(-float(x)))
    result = np.zeros(len(trace.steps))
    facts = list(graph.facts.values())
    for assigned in itertools.product((0, 1), repeat=len(facts)):
        world = dict(zip((f.identity for f in facts), assigned))
        weight = math.prod(sigmoid(theta[model.base_count + f.kind]) if bit else
            1 - sigmoid(theta[model.base_count + f.kind]) for f, bit in zip(facts, assigned))
        mass = {(0, ()): 1.}  # Every action and its C-at-dispatch retained.
        for t, step in enumerate(trace.steps):
            assert step.revision is None
            for op in step.base.operations:
                updated = {}
                for (bits, actions), value in mass.items():
                    q = sigmoid(theta[op.parameter] + theta[CAP_CONFIDENCE] * (op.confidence_feature if op.is_message else 0.))
                    active = not ((bits >> op.receiver) & 1) and (op.sender is None or ((bits >> op.sender) & 1))
                    choices = [(bits, 1.)] if not active else [(bits, 1 - q), (bits | (1 << op.receiver), q)]
                    for new, p in choices:
                        key = (new, actions)
                        updated[key] = updated.get(key, 0.) + value * p
                mass = updated
            for factor in step.mechanism_factors:
                assert factor.template_id == "authorization"
                proof = graph.proofs[factor.action_id]
                failed = not any(all(not world[k] for k in path) for path in proof.paths)
                mass = {(bits, history + ((int(failed), (bits >> factor.node) & 1),)): value
                    for (bits, history), value in mass.items()}
            for (bits, history), value in mass.items():
                count = sum((bits >> i) & 1 for i in step.base.observed_nodes)
                n = len(step.base.observed_nodes)
                info = [count / max(n, 1), count * (count - 1) / max(1, n * (n - 1)),
                    sum((bits >> i) & 1 for i in step.base.capable_nodes) / max(1, len(step.base.capable_nodes)),
                    sum((bits >> i) & 1 for i in step.base.terminal_nodes) / max(1, len(step.base.terminal_nodes))]
                logit_ = step.base.observables @ theta[CAP_HEAD:CAP_HEAD + 15] + np.asarray(info) @ theta[CAP_HEAD + 15:CAP_HEAD + 19]
                logit_ += theta[CAP_HEAD + 20] * any(a for a, c in history)
                logit_ += theta[CAP_HEAD + 22] * any(a and c for a, c in history)
                failed, n = sum(a for a, c in history), len(history)
                logit_ += theta[-2] * failed / max(1, n) + theta[-1] * failed * (failed - 1) / max(1, n * (n - 1))
                result[t] += weight * value * sigmoid(logit_)
    np.testing.assert_allclose(model.forward(bundle_)["probabilities"], result, atol=3e-15, rtol=0.)


def test_unsupported_invalid_claim_cannot_become_deterministic_denial():
    ledger = DelegationLedger(root_principals=["user"], trusted_observers=["adapter"], coverage_complete=False)
    root = grant(1, "root", "user", "planner", can_delegate=False)
    root["observer"] = "untrusted-text"
    ledger.observe(root)
    ledger.observe(grant(2, "child", "planner", "worker", "root"))
    snapshot = ledger.assess(action(), sequence=3, task_id="task-a", grant_ids=["child"])
    # The old ledger's diagnostic invalidity includes an unverified parent.
    assert "unverified_evidence" in snapshot.evidence["candidates"][0]["reasons"]
    graph = AuthorityProofGraph(AuthorityWindow("opaque", 0, 100))
    graph.add(snapshot)
    assert {f.kind for f in graph.facts.values()} == {2}
    assert graph.query([0, 0, 0])[0] == .5


def test_deterministic_reference_label_contradictions_are_not_silently_clipped():
    with pytest.raises(ValueError, match="contradicts"):
        SharedAuthorityJointBN().fit([bundle(complete=True)], [1], target="reference_failure_count")


def test_supported_complete_path_dominates_arbitrarily_many_unknown_alternatives():
    ledger = DelegationLedger(root_principals=["user"], trusted_observers=["adapter"], coverage_complete=True)
    ledger.observe(grant(1, "direct", "user", "worker"))
    snapshot = ledger.assess(action(), sequence=2, task_id="task-a",
        grant_ids=["direct", *("missing-" + str(i) for i in range(20))])
    graph = AuthorityProofGraph(AuthorityWindow("known-alternative", 0, 100), max_facts=0)
    graph.add(snapshot)
    assert not graph.facts and graph.query([0, 0, 0])[0] == 0.


def test_task_and_window_namespaces_do_not_merge_reused_grant_names():
    def snapshot(task, seq):
        ledger = DelegationLedger(root_principals=["user"], trusted_observers=["adapter"], coverage_complete=False)
        event = grant(1, "same-name", "user", "worker")
        event["payload"]["scope"]["task_id"] = task
        ledger.observe(event)
        return ledger.assess({**action(), "action_id": "action-" + task}, sequence=seq, task_id=task, grant_ids=["same-name"])
    graph = AuthorityProofGraph(AuthorityWindow("multi-task", 0, 100))
    graph.add(snapshot("task-a", 2))
    graph.add(snapshot("task-b", 3))
    assert len(graph.facts) == 2 and graph.query([logit(.2), 0, 0])[0] == pytest.approx(.36)
    other = AuthorityProofGraph(AuthorityWindow("different-window", 0, 100))
    other.add(snapshot("task-a", 2))
    assert not set(other.facts) & set(graph.facts)


def test_export_contains_executable_shared_paths_and_no_guessed_probability():
    graph = bundle(True).graph
    spec = graph.specification()
    assert len(spec["facts"]) == 1
    paths = [a["supported_if_any_path_has_no_failed_fact"] for a in spec["actions"]]
    assert paths[0] == paths[1] and spec["calibrated"] is False
    assert "probability" not in spec["facts"][0]


def test_joint_head_can_target_simultaneous_failures_without_reversing_any_failure_semantics():
    from paperalpha_monitor.capability_action_bn import CAP_HEAD
    model = SharedAuthorityJointBN()
    model.theta[model.base_count] = logit(.2)
    model.theta[CAP_HEAD:CAP_HEAD + 23] = 0.
    model.theta[CAP_HEAD] = -2.
    model.theta[-1] = 6.
    shared = model.forward(bundle(True))["probabilities"][-1]
    separate = model.forward(bundle(False))["probabilities"][-1]
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    assert shared == pytest.approx(.8 * sigmoid(-2) + .2 * sigmoid(4))
    assert separate == pytest.approx(.96 * sigmoid(-2) + .04 * sigmoid(4))
    assert shared > separate
    assert bundle(True).graph.query(model.theta[model.prior_slice])[0] < bundle(False).graph.query(model.theta[model.prior_slice])[0]
