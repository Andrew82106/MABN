"""Declared payment amounts and scoped cumulative attempted exposure.

Exact decimal comparisons establish arithmetic under recorded declarations,
not authorization, successful payment, true balances or complete resource use.
Global limits must be supplied before the run; a tool's own amount_limit is
never promoted to a global or independently authorized constraint.
"""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import json
from pathlib import Path
import re

from paperalpha_runtime.semantic_evidence import digest

from ._resources import static_resource

DEFAULT_PROFILE = static_resource('knowledge/templates/harnessaudit_payment_quantities_v1.json')
FEATURE_NAMES = ("numeric_profile_available", "requested_amount_known", "self_limit_known",
    "self_comparison_unknown", "self_limit_exceeded", "self_margin_bounded", "amount_alias_conflict",
    "currency_unknown_or_conflicting", "global_contract_matches", "global_evidence_incomplete",
    "global_attempted_exceeded", "global_margin_bounded", "global_scope_unknown", "scope_identifiers_unavailable")


def decimal_amount(value):
    """Strict nonnegative decimal tokens; reject silent locale/float repairs."""
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not 0 < len(value) <= 128 or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None:
        raise ValueError("nonnegative plain decimal string/integer required, at most 128 characters")
    return Decimal(value)


def currency_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value) else None


def bounded_margin(amount, limit):
    with localcontext() as context:
        context.prec = 512
        return float((amount - limit) / (Decimal(1) + amount + limit))


def amount_comparison(amount, limit):
    if amount is None or limit is None:
        return {"status": "unknown", "amount": None if amount is None else str(amount),
                "limit": None if limit is None else str(limit), "margin_bounded": 0.}
    return {"status": "exceeds" if amount > limit else "at_limit" if amount == limit else "below",
            "amount": str(amount), "limit": str(limit), "margin_bounded": bounded_margin(amount, limit)}


class QuantityProfiles:
    def __init__(self, specification):
        self.specification = copy.deepcopy(specification)
        if specification.get("schema_version") != "payment-quantity-profile-v1":
            raise ValueError("explicit payment quantity profile required")
        self.tools = {}
        for tool in specification.get("tools", []):
            name = tool.get("name")
            if not isinstance(name, str) or not name or name in self.tools:
                raise ValueError("unique tool names required")
            if (tool.get("limit_semantics") != "self_declared_single_action_bound" or
                tool.get("usage_semantics") != "requested_nonnegative_charge_at_dispatch_not_confirmed_spend"):
                raise ValueError("unsupported quantity/limit semantics")
            fields = tool.get("amount_fields")
            if not isinstance(fields, list) or not fields:
                raise ValueError("amount field descriptions required")
            for field in fields:
                if not isinstance(field.get("name"), str) or not field["name"] or not field.get("description"):
                    raise ValueError("amount field identity and public description required")
                if field.get("unit") != "currency_field" and currency_code(field.get("unit")) is None:
                    raise ValueError("explicit amount units required")
            if len({f["name"] for f in fields}) != len(fields):
                raise ValueError("duplicate amount field mapping")
            if currency_code(tool.get("default_currency")) is None:
                raise ValueError("documented currency default required")
            if any(not isinstance(tool.get(k), str) or not tool[k] for k in ("limit_field", "currency_field", "limit_description", "currency_description")):
                raise ValueError("public limit and currency descriptions required")
            scope = tool.get("scope_fields")
            if not isinstance(scope, list) or not scope or any(not isinstance(k, str) or not k for k in scope) or len(set(scope)) != len(scope):
                raise ValueError("distinct scope dimension names required")
            self.tools[name] = copy.deepcopy(tool)
        self.sha256 = digest(self.specification)


def load_quantity_profiles(path=DEFAULT_PROFILE):
    return QuantityProfiles(json.loads(Path(path).read_text(encoding="utf-8")))


class ScopedQuantityLedger:
    """Prefix ledger for profiled tool dispatches, never post-run effects.

    A contract constrains the sum of requested nonnegative charges for matching
    explicit IDs in this run. Its presence is input configuration, not proof
    that its author has authority. Unknown matching calls make a below-limit
    subtotal incomplete; they do not erase an already exceeded known subtotal.
    """
    def __init__(self, profiles=None, contracts=()):
        self.profiles = copy.deepcopy(profiles or load_quantity_profiles())
        self.contracts, self.totals, self.gaps, self.calls = {}, {}, {}, {}
        for contract in contracts:
            if not isinstance(contract, dict) or set(contract) != {"id", "scope", "currency", "limit", "aggregation", "source"}:
                raise ValueError("explicit pre-run contract fields required")
            identity, scope = contract["id"], contract["scope"]
            if not isinstance(identity, str) or not identity or identity in self.contracts:
                raise ValueError("unique contract identity required")
            if not isinstance(scope, dict) or not scope or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in scope.items()):
                raise ValueError("nonempty exact scope dimensions required")
            if contract["aggregation"] != "cumulative_requested_nonnegative_charges":
                raise ValueError("only explicitly cumulative requested charges are supported")
            if currency_code(contract["currency"]) is None or not isinstance(contract["source"], str) or not contract["source"]:
                raise ValueError("contract currency and declared source required")
            decimal_amount(contract["limit"])
            self.contracts[identity] = copy.deepcopy(contract)
            self.totals[identity], self.gaps[identity] = Decimal(0), []
        self.contracts_sha256 = digest(list(self.contracts.values()))

    @staticmethod
    def _scope_match(scope, contract):
        if any(key in scope and scope[key] != value for key, value in contract.items()):
            return "different"
        return "same" if all(key in scope for key in contract) else "unknown"

    def observe(self, row):
        if row.get("kind") != "tool_handler_started":
            return None
        data = row["payload"]
        identity = data["invocation_id"]
        fingerprint = digest(data)
        if identity in self.calls:
            if self.calls[identity][0] != fingerprint:
                raise ValueError("conflicting invocation identity in quantity ledger")
            return copy.deepcopy(self.calls[identity][1])
        profile = self.profiles.tools.get(data["tool"])
        if profile is None:
            return None  # Explicit profile coverage, not an assertion of safety.
        arguments = data.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("structured call arguments required for quantity observations")
        features = dict.fromkeys(FEATURE_NAMES, 0.)
        features["numeric_profile_available"] = 1.
        currency_raw = arguments.get(profile["currency_field"])
        currency = currency_code(profile["default_currency"] if currency_raw in (None, "") else currency_raw)
        amount_fields, values, invalid = [], [], []
        for field in profile["amount_fields"]:
            raw = arguments.get(field["name"])
            if raw is None or raw == "":
                continue
            unit = currency if field["unit"] == "currency_field" else field["unit"]
            try:
                value = decimal_amount(raw)
            except ValueError:
                invalid.append(field["name"])
                value = None
            amount_fields.append({"field": field["name"], "raw_value": raw, "value": None if value is None else str(value), "currency": unit})
            if value is not None:
                values.append((value, unit))
        units = {unit for _, unit in values}
        currency_problem = currency is None or None in units or len(units) > 1 or bool(units - {currency})
        conflict = len({value for value, _ in values}) > 1 or currency_problem and len(amount_fields) > 1
        amount = values[0][0] if values and not invalid and not conflict and not currency_problem else None
        try:
            limit = decimal_amount(arguments.get(profile["limit_field"])) if currency is not None else None
        except ValueError:
            limit = None
        local = amount_comparison(amount, limit)
        features.update(requested_amount_known=float(amount is not None), self_limit_known=float(limit is not None),
            self_comparison_unknown=float(local["status"] == "unknown"), self_limit_exceeded=float(local["status"] == "exceeds"),
            self_margin_bounded=local["margin_bounded"], amount_alias_conflict=float(conflict),
            currency_unknown_or_conflicting=float(currency_problem))
        scope = {k: arguments[k] for k in profile["scope_fields"] if isinstance(arguments.get(k), str) and arguments[k]}
        features["scope_identifiers_unavailable"] = float(not scope)
        global_results = []
        for contract_id, contract in self.contracts.items():
            match = self._scope_match(scope, contract["scope"])
            if match == "different":
                continue
            reasons = []
            if match == "unknown":
                reasons.append("scope_unknown")
                features["global_scope_unknown"] = 1.
            if currency != contract["currency"]:
                reasons.append("currency_not_comparable")
            if amount is None:
                reasons.append("amount_unavailable_or_conflicting")
            if match == "same":
                features["global_contract_matches"] = 1.
            if reasons:
                self.gaps[contract_id].append({"invocation_id": identity, "reasons": reasons})
            else:
                with localcontext() as context:
                    context.prec = 512
                    self.totals[contract_id] += amount
            total, bound = self.totals[contract_id], decimal_amount(contract["limit"])
            comparison = amount_comparison(total, bound)
            complete = not self.gaps[contract_id]
            if not complete and comparison["status"] != "exceeds":
                comparison["status"] = "incomplete_subtotal"
            # Signed below-limit margins are not used when contributions are missing.
            margin = comparison["margin_bounded"] if complete or total > bound else 0.
            features["global_evidence_incomplete"] = max(features["global_evidence_incomplete"], float(not complete))
            features["global_attempted_exceeded"] = max(features["global_attempted_exceeded"], float(total > bound))
            if not global_results or margin > features["global_margin_bounded"]:
                features["global_margin_bounded"] = margin
            global_results.append({"contract_id": contract_id, "contract": copy.deepcopy(contract), "scope_match": match,
                "currency_comparable": currency == contract["currency"], "current_call_contribution_known": not reasons,
                "comparison": comparison, "missing_contributions": copy.deepcopy(self.gaps[contract_id]),
                "covered_profiled_calls_complete": complete})
        result = {"schema_version": "scoped-quantity-observation-v1", "invocation_id": identity,
            "sequence": row["sequence"], "actor": data["actor"], "tool": data["tool"], "arguments_sha256": digest(arguments),
            "profile_sha256": self.profiles.sha256, "contracts_sha256": self.contracts_sha256,
            "amount_fields": amount_fields, "invalid_amount_fields": invalid, "declared_currency": currency,
            "currency_default_applied": currency_raw in (None, ""), "scope": scope,
            "self_declared_bound_comparison": local, "cumulative_contracts": global_results, "features": features,
            "authorization_established": False, "actual_spending_established": False,
            "meaning": "Arithmetic over declared call arguments and configured scoped attempted-charge limits; not approval or realized harm."}
        self.calls[identity] = fingerprint, copy.deepcopy(result)
        return result
