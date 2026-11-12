"""v6 action factors conditioned on explicit numeric comparisons.

Text claims still cannot grant permission. Numeric comparisons modify the
action risk hypothesis, not the communicated process bit. Global attempted
exposure is deterministic visible context, preserving action elimination's
conditional independence assumption; correlated latent authorization errors
are still outside this model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from paperalpha_runtime.semantic_evidence import digest
from .dynamic_joint_bn import JointTrace
from .text_conditioned_bn import (TextConditionedBN, TextConditionedFilter, JournalTextCompiler, TextStep,
    PARAMETER_NAMES as TEXT_PARAMETERS, MODEL_SHA256 as TEXT_SHA256)
from .scoped_quantities import ScopedQuantityLedger, FEATURE_NAMES as QUANTITY_FEATURES

QUANTITY_OFFSET = len(TEXT_PARAMETERS)
PARAMETER_NAMES = (*TEXT_PARAMETERS, *("quantity:" + name for name in QUANTITY_FEATURES))
MODEL_ID = "ScopedBudgetTextBN-v6"
MODEL_SPEC = {"id": MODEL_ID, "base_text_specification_sha256": TEXT_SHA256, "numeric_features": QUANTITY_FEATURES,
    "parameter_names": PARAMETER_NAMES, "quantity_protocol": "scoped-quantity-observation-v1",
    "action_factor": "sigmoid(base_capability_role_check_cell + numeric_weights @ declared_comparisons)",
    "global_usage": "sum of requested nonnegative charges at observed dispatch, not confirmed spending",
    "condition_scope": "only explicit public field mappings and pre-run configured scoped cumulative contracts",
    "knowledge_links": ["RB008: claims are not approval", "RB009: explicit scope boundaries", "RB039: cross-agent cumulative resources"],
    "boundary": "Engineering formalization of catalog mechanisms, not literature-derived numeric probabilities or novelty proof."}
MODEL_SHA256 = digest(MODEL_SPEC)


@dataclass(frozen=True)
class BudgetTextStep(TextStep):
    quantity_observations: dict


class JournalBudgetCompiler(JournalTextCompiler):
    def __init__(self, *, quantity_profiles=None, budget_contracts=(), **context):
        super().__init__(**context)
        self.quantities = ScopedQuantityLedger(quantity_profiles, budget_contracts)

    def _push(self, row, annotation):
        step = super()._push(row, annotation)
        if step is None:
            return None
        self.quantities.observe(row)
        observations = {identity: copy.deepcopy(pair[1]) for identity, pair in self.quantities.calls.items()}
        self.last_step = BudgetTextStep(step.base, step.revision, observations)
        return copy.deepcopy(self.last_step)


def compile_budget_journal(rows, annotations=None, **context):
    compiler, steps = JournalBudgetCompiler(**context), []
    annotations = {} if annotations is None else annotations
    seen = set()
    for row in rows:
        identity = f"journal-{row['sequence']}"
        if identity in annotations:
            seen.add(identity)
        step = compiler.push(row, annotations.get(identity))
        if step is not None:
            steps.append(step)
    if set(annotations) != seen:
        raise ValueError("annotations outside this journal prefix are not accepted")
    if not steps:
        raise ValueError("a nonempty model-event prefix is required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), tuple(steps), compiler.workflow)


class BudgetFilter(TextConditionedFilter):
    parameter_count, step_type = len(PARAMETER_NAMES), BudgetTextStep

    def __init__(self, theta, *, use_quantities=True, **options):
        super().__init__(theta, **options)
        self.use_quantities = use_quantities
        self.current_quantities, self.dispatched_numeric_risk = {}, {}

    def _dispatch_gate(self, dispatch):
        observation = self.current_quantities.get(dispatch.action_id)
        if not self.use_quantities or observation is None:
            q, gradient = super()._dispatch_gate(dispatch)
        else:
            x = np.asarray([observation["features"][name] for name in QUANTITY_FEATURES], dtype=float)
            q = float(expit(self.theta[dispatch.parameter] + self.theta[QUANTITY_OFFSET:] @ x))
            gradient = None
            if self.derivatives:
                gradient = np.zeros(len(self.theta))
                gradient[dispatch.parameter] = 1
                gradient[QUANTITY_OFFSET:] = x
                gradient *= q * (1 - q)
        self.dispatched_numeric_risk[dispatch.action_id] = q
        return q, gradient

    def apply(self, step, variables):
        if type(step) is not BudgetTextStep:
            raise ValueError("v6 requires a scoped budget step")
        self.current_quantities = copy.deepcopy(step.quantity_observations)
        return super().apply(step, variables)

    def explain(self, step, variables):
        result = super().explain(step, variables)
        for action in result["actions"]:
            identity = action["action_id"]
            if identity in self.dispatched_numeric_risk:
                action["authorization_risk_hypothesis"] = self.dispatched_numeric_risk[identity]
            action["quantity_conditioning"] = copy.deepcopy(self.current_quantities.get(identity))
        result["quantity_observation_scope"] = "Declared numeric consistency and configured cumulative attempted exposure, not trusted permission or actual spending."
        result["quantity_conditioning_enabled"] = self.use_quantities
        return result


class ScopedBudgetBN(TextConditionedBN):
    model_name, step_type, parameter_names = MODEL_ID, BudgetTextStep, PARAMETER_NAMES

    def __init__(self, *, use_quantities=True, **options):
        self.use_quantities = use_quantities
        super().__init__(**options)

    def initial_parameters(self):
        return np.r_[super().initial_parameters(), np.zeros(len(QUANTITY_FEATURES))]

    def _engine(self, theta, *, derivatives=False):
        return BudgetFilter(theta, use_quantities=self.use_quantities, use_text=self.use_text,
            mask_text_features=self.mask_text_features, use_messages=self.use_messages,
            use_latent_head=self.use_latent_head, derivatives=derivatives)

    def parameters(self):
        result = super().parameters()
        result.update(name=MODEL_ID, specification=MODEL_SPEC, specification_sha256=MODEL_SHA256,
            parameter_names=list(PARAMETER_NAMES), use_quantities=self.use_quantities)
        result["limitations"] = [s for s in result["limitations"] if "numeric-scope comparison" not in s]
        result["limitations"] += ["Numeric observations cover explicitly profiled tool arguments, not text-derived amounts/conditions.",
            "No promotion of self-declared bounds to trusted authorization or global budget.",
            "Cumulative contracts are pre-run observable configuration, not inferred from hidden evaluation policies.",
            "Counts attempted nonnegative charges, not confirmed spending; no refund, failure rollback, currency conversion or asynchronous latency model.",
            "Numeric conditioning currently attaches at tool-handler entry/dispatch; no earlier pending-proposal budget model.",
            "Knowledge-driven role subgraphs, automatic DAGs and source identity remain incomplete."]
        return result
