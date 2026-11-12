import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.scoped_budget_bn import (ScopedBudgetBN, JournalBudgetCompiler, compile_budget_journal,
    PARAMETER_NAMES, QUANTITY_OFFSET, QUANTITY_FEATURES)
from paperalpha_monitor.text_conditioned_bn import TextConditionedBN, TextStep
from paperalpha_monitor.dynamic_joint_bn import JointTrace
from paperalpha_monitor.tool_capabilities import load_tool_registry
from test_scoped_quantities import payment, contract


def scenario(second_amount="300"):
    first, second = payment(3), payment(5, "second", "b", amount=second_amount, amount_usd=second_amount)
    def returned(row):
        return {"schema_version": "runtime-journal-v1", "sequence": row["sequence"] + 1, "kind": "tool_handler_returned",
                "payload": {k: v for k, v in {**row["payload"], "result": "Recorded."}.items() if k != "arguments"}}
    return [{"schema_version": "runtime-journal-v1", "sequence": 1, "kind": "run_started",
             "payload": {"agents": ["a", "b"], "external_endpoints": ["user"]}},
            {"schema_version": "runtime-journal-v1", "sequence": 2, "kind": "communication",
             "payload": {"actor": "a", "target": "b", "content": "Two branches."}},
            first, returned(first), second, returned(second),
            {"schema_version": "runtime-journal-v1", "sequence": 7, "kind": "run_finished", "payload": {"status": "completed"}}]


def context():
    return {"tool_registry": load_tool_registry(), "budget_contracts": [contract()], "roles": {"a": "executor", "b": "executor"}}


@pytest.mark.parametrize("label,prefix", [(0, True), (1, True), (0, False), (1, False)])
def test_all_340_gradients_include_numeric_action_and_cumulative_factors(label, prefix):
    model = ScopedBudgetBN()
    compiled = compile_budget_journal(scenario(), **context())
    theta = model.theta + np.random.default_rng(93).normal(0, .4, len(model.theta))
    assert len(theta) == 340
    _, analytic = model.episode_loss_gradient(theta, compiled, label, prefix)
    numeric = approx_derivative(lambda t: model.episode_loss_gradient(t, compiled, label, prefix)[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, rtol=3e-5, atol=4e-8)
    assert np.any(np.abs(analytic[QUANTITY_OFFSET:]) > 1e-8)


def test_numeric_path_resolves_v5_magnitude_blind_spot_but_not_as_permission():
    model = ScopedBudgetBN()
    model.theta[PARAMETER_NAMES.index("quantity:self_limit_exceeded")] = 2.
    model.theta[PARAMETER_NAMES.index("quantity:self_margin_bounded")] = 1.
    a = compile_budget_journal(scenario("349"), **context())
    b = compile_budget_journal(scenario("900"), **context())
    assert model.predict([b])[0][-1] > model.predict([a])[0][-1]
    assert all(x["permission_status"] == "unknown" and x["accepted_check"] is None for x in model.explain(b)["actions"])


def test_cross_agent_global_factor_changes_risk_even_when_both_self_limits_pass():
    model = ScopedBudgetBN()
    model.theta[PARAMETER_NAMES.index("quantity:global_attempted_exceeded")] = 3.
    compiled = compile_budget_journal(scenario(), **context())
    disabled = ScopedBudgetBN(use_quantities=False)
    disabled.theta = model.theta.copy()
    a, b = model.predict([compiled])[0], disabled.predict([compiled])[0]
    np.testing.assert_allclose(a[:3], b[:3], atol=2e-15)
    assert a[3] > b[3]
    actions = model.explain(compiled)["actions"]
    assert all(x["quantity_conditioning"]["self_declared_bound_comparison"]["status"] == "below" for x in actions)


def test_zero_numeric_weights_and_disabled_numeric_path_recover_v5():
    compiled = compile_budget_journal(scenario(), **context())
    legacy = JointTrace(compiled.variables, tuple(TextStep(s.base, s.revision) for s in compiled.steps), compiled.workflow)
    old, new = TextConditionedBN(), ScopedBudgetBN()
    old.theta += np.random.default_rng(41).normal(0, .5, len(old.theta))
    new.theta[:QUANTITY_OFFSET] = old.theta
    np.testing.assert_allclose(new.predict([compiled]), old.predict([legacy]), atol=2e-15, rtol=0)
    new.theta[QUANTITY_OFFSET:] = 1.
    new.use_quantities = False
    np.testing.assert_allclose(new.predict([compiled]), old.predict([legacy]), atol=2e-15, rtol=0)
    _, gradient = new.episode_loss_gradient(new.theta, compiled, 1)
    np.testing.assert_array_equal(gradient[QUANTITY_OFFSET:], 0.)


def test_numeric_authorization_risk_does_not_propagate_as_process_state():
    compiled = compile_budget_journal(scenario(), **context())
    a, b = ScopedBudgetBN(), ScopedBudgetBN()
    b.theta[QUANTITY_OFFSET:] = 2.
    ca, cb = a._engine(a.theta), b._engine(b.theta)
    for step in compiled.steps:
        ca.apply(step, compiled.variables)
        cb.apply(step, compiled.variables)
        np.testing.assert_allclose(ca.distribution.sum(axis=1), cb.distribution.sum(axis=1), atol=2e-15, rtol=0)
    assert not np.allclose(ca.distribution.sum(axis=0), cb.distribution.sum(axis=0))


def test_later_branch_cannot_rewrite_earlier_numeric_comparisons_or_score():
    rows = scenario()
    model = ScopedBudgetBN()
    model.theta[QUANTITY_OFFSET:] = np.random.default_rng(3).normal(0, .7, len(QUANTITY_FEATURES))
    full = compile_budget_journal(rows, **context())
    prefix = compile_budget_journal(rows[:4], **context())
    np.testing.assert_allclose(model.predict([full])[0][:3], model.predict([prefix])[0], atol=2e-15, rtol=0)
    assert full.steps[-1].quantity_observations["first"] == prefix.steps[-1].quantity_observations["first"]


def test_compiler_failure_is_atomic_for_ledger_and_text_window():
    compiler = JournalBudgetCompiler(**context())
    for row in scenario()[:4]:
        compiler.push(row)
    total = copy.deepcopy(compiler.quantities.totals)
    broken = scenario()[4]
    broken["payload"]["invocation_id"] = "first"
    with pytest.raises(ValueError):
        compiler.push(broken)
    assert compiler.quantities.totals == total and compiler.projector.sequence == 4


def test_action_explanation_uses_the_numeric_conditional_not_old_base_cell():
    model = ScopedBudgetBN()
    model.theta[PARAMETER_NAMES.index("quantity:global_attempted_exceeded")] = 2.
    compiled = compile_budget_journal(scenario(), **context())
    explanation = model.explain(compiled)
    first, second = explanation["actions"]
    assert second["authorization_risk_hypothesis"] > first["authorization_risk_hypothesis"]
    assert sum(x["risk_contribution"] for x in explanation["system_mixture_terms"]) == pytest.approx(explanation["risk_probability"])
    assert model.parameters()["name"] == "ScopedBudgetTextBN-v6"
