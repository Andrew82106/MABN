import copy
from decimal import Decimal
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.scoped_quantities import (ScopedQuantityLedger, decimal_amount, bounded_margin, load_quantity_profiles)


def payment(sequence=2, identity="first", actor="a", **changes):
    arguments = {"request_id": "r", "customer_id": "c", "bundle_id": "bundle", "selected_option": "bundle",
                 "amount": "300", "amount_usd": "300", "amount_limit": "400", "currency": "USD"}
    arguments.update(changes)
    return {"schema_version": "runtime-journal-v1", "sequence": sequence, "kind": "tool_handler_started",
        "payload": {"invocation_id": identity, "provider_call_id": "provider-" + identity, "actor": actor,
                    "tool": "authorize_payment", "arguments": arguments}}


def contract(**changes):
    return {"id": "shared", "scope": {"request_id": "r"}, "currency": "USD", "limit": "500",
            "aggregation": "cumulative_requested_nonnegative_charges", "source": "hand-authored component constraint", **changes}


@pytest.mark.parametrize("value,expected", [("0", "0"), ("0002.50", "2.50"), (3, "3"), ("0.0001", ".0001")])
def test_plain_decimal_is_exact(value, expected):
    assert decimal_amount(value) == Decimal(expected)


@pytest.mark.parametrize("value", [True, -1, 0.1, "1e2", "-1", "1,000", "NaN", "$300", " 300", "", None, "9" * 129])
def test_unsupported_formats_remain_unknown_not_silently_normalized(value):
    with pytest.raises(ValueError):
        decimal_amount(value)


def test_local_arithmetic_does_not_establish_permission_or_global_budget():
    result = ScopedQuantityLedger().observe(payment(amount="349", amount_usd="349"))
    assert result["self_declared_bound_comparison"]["status"] == "below"
    assert result["self_declared_bound_comparison"]["margin_bounded"] == pytest.approx(-51 / 750)
    assert result["cumulative_contracts"] == []
    assert not result["authorization_established"] and not result["actual_spending_established"]
    assert not result["features"]["scope_identifiers_unavailable"]


def test_alias_disagreement_is_not_resolved_by_choosing_favorable_amount():
    result = ScopedQuantityLedger().observe(payment(amount="349", amount_usd="900"))
    assert result["features"]["amount_alias_conflict"] == 1
    assert result["self_declared_bound_comparison"]["status"] == "unknown"


def test_currency_disagreement_and_default_are_separate_cases():
    result = ScopedQuantityLedger().observe(payment(currency="EUR"))
    assert result["features"]["currency_unknown_or_conflicting"] == 1
    assert result["self_declared_bound_comparison"]["status"] == "unknown"
    default = ScopedQuantityLedger().observe(payment(currency=""))
    assert default["currency_default_applied"] and default["declared_currency"] == "USD"
    assert default["self_declared_bound_comparison"]["status"] == "below"


def test_invalid_alias_is_not_ignored_even_when_other_alias_parses():
    result = ScopedQuantityLedger().observe(payment(amount="approximately 300"))
    assert result["invalid_amount_fields"] == ["amount"]
    assert result["features"]["requested_amount_known"] == 0


def test_two_agents_individually_within_but_cumulatively_exceed_explicit_contract():
    ledger = ScopedQuantityLedger(contracts=[contract()])
    first = ledger.observe(payment())
    second = ledger.observe(payment(3, "second", "b"))
    assert first["self_declared_bound_comparison"]["status"] == second["self_declared_bound_comparison"]["status"] == "below"
    assert first["cumulative_contracts"][0]["comparison"]["status"] == "below"
    assert second["cumulative_contracts"][0]["comparison"]["status"] == "exceeds"
    assert second["cumulative_contracts"][0]["comparison"]["amount"] == "600"
    assert second["features"]["global_attempted_exceeded"] == 1


def test_other_request_does_not_consume_this_contract():
    ledger = ScopedQuantityLedger(contracts=[contract()])
    ledger.observe(payment())
    other = ledger.observe(payment(3, "second", "b", request_id="another"))
    assert other["cumulative_contracts"] == [] and ledger.totals["shared"] == Decimal(300)


def test_missing_scope_or_bad_amount_does_not_prove_remaining_budget():
    ledger = ScopedQuantityLedger(contracts=[contract()])
    ledger.observe(payment())
    gap = ledger.observe(payment(3, "second", "b", request_id=""))
    assert gap["features"]["global_scope_unknown"] == 1
    assert gap["cumulative_contracts"][0]["comparison"]["status"] == "incomplete_subtotal"
    assert gap["features"]["global_margin_bounded"] == 0
    later = ledger.observe(payment(4, "third", "b"))
    assert later["cumulative_contracts"][0]["comparison"]["status"] == "exceeds"
    assert not later["cumulative_contracts"][0]["covered_profiled_calls_complete"]


def test_different_known_scope_dominates_another_missing_dimension():
    ledger = ScopedQuantityLedger(contracts=[contract(scope={"request_id": "r", "customer_id": "c"})])
    result = ledger.observe(payment(request_id="other", customer_id=""))
    assert result["cumulative_contracts"] == [] and ledger.gaps["shared"] == []


def test_wrong_currency_is_unknown_without_exchange_rate_guess():
    ledger = ScopedQuantityLedger(contracts=[contract()])
    result = ledger.observe(payment(currency="EUR", amount_usd=""))
    assert result["self_declared_bound_comparison"]["status"] == "below"
    assert result["cumulative_contracts"][0]["comparison"]["status"] == "incomplete_subtotal"
    assert result["cumulative_contracts"][0]["missing_contributions"][0]["reasons"] == ["currency_not_comparable"]


def test_runtime_identity_deduplicates_retries_not_two_distinct_actions():
    ledger = ScopedQuantityLedger(contracts=[contract()])
    first = ledger.observe(payment())
    assert ledger.observe(payment(3)) == first
    assert ledger.totals["shared"] == Decimal(300)
    with pytest.raises(ValueError, match="conflicting"):
        ledger.observe(payment(4, amount="301"))
    assert ledger.totals["shared"] == Decimal(300)
    ledger.observe(payment(5, "other", "a"))
    assert ledger.totals["shared"] == Decimal(600)


def test_returns_failures_and_unprofiled_tools_do_not_claim_confirmed_spending():
    ledger = ScopedQuantityLedger(contracts=[contract()])
    ledger.observe(payment())
    assert ledger.observe({"kind": "tool_handler_failed"}) is None
    row = payment(3, "other")
    row["payload"]["tool"] = "not-profiled"
    assert ledger.observe(row) is None and ledger.totals["shared"] == Decimal(300)


def test_contract_is_copied_and_self_limit_never_creates_one():
    specification = contract()
    ledger = ScopedQuantityLedger(contracts=[specification])
    specification["limit"] = "1"
    assert ledger.contracts["shared"]["limit"] == "500"
    assert ScopedQuantityLedger().observe(payment())["features"]["global_contract_matches"] == 0


def test_zero_limit_and_small_fraction_are_compared_without_float_epsilon():
    result = ScopedQuantityLedger().observe(payment(amount="0.00000000000000000001", amount_usd="0.00000000000000000001", amount_limit="0"))
    assert result["self_declared_bound_comparison"]["status"] == "exceeds"
    assert 0 < bounded_margin(Decimal("9" * 128), Decimal(0)) <= 1


@pytest.mark.parametrize("changes", [{"aggregation": "actual_spend"}, {"scope": {}}, {"limit": "-1"}, {"currency": "usd"}])
def test_unsupported_contract_semantics_rejected(changes):
    with pytest.raises(ValueError):
        ScopedQuantityLedger(contracts=[contract(**changes)])
