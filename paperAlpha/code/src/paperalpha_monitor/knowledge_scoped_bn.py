"""v7 knowledge-selected action subgraphs with separate mechanism histories.

Conditional on visible observations, dispatch factors are independent
Bernoulli hypotheses, NOT established or independent real-world violations.
Each factor updates its own three-state history, joint with all process bits
and other histories. This explicitly changes v6's conflated authorization
semantics and does not inherit any earlier model's accuracy results.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.special import expit

from .action_mechanism_knowledge import load_action_mechanisms, MechanismFactor
from .capability_action_bn import CAP_HEAD, coverage
from .dynamic_joint_bn import JointTrace, state_bits, transition_pairs, latent_features
from .scoped_budget_bn import JournalBudgetCompiler, BudgetTextStep
from .text_conditioned_bn import TextConditionedBN, TextConditionedFilter

MODEL_ID = "KnowledgeScopedBN-v7"
HISTORY_STATES = ("no_mechanism_concern", "concern_without_dispatch_process_overlap", "concern_with_dispatch_process_overlap")


@dataclass(frozen=True)
class KnowledgeStep(BudgetTextStep):
    mechanism_factors: tuple[MechanismFactor, ...]
    selection_decisions: tuple[dict, ...]
    registry_sha256: str


class JournalKnowledgeCompiler(JournalBudgetCompiler):
    def __init__(self, *, mechanism_registry=None, **context):
        self.mechanisms = copy.deepcopy(mechanism_registry or load_action_mechanisms())
        super().__init__(**context)

    def _push(self, row, annotation):
        step = super()._push(row, annotation)
        if step is None:
            return None
        if step.base.pending:
            raise ValueError("v7 journal model supports dispatch-time observations, not pending action revision")
        actions = {a["action_id"]: a for a in step.base.actions}
        factors, decisions = [], []
        for dispatch in step.base.dispatches:
            action = actions[dispatch.action_id]
            compiled, selected = self.mechanisms.factors(action, step.quantity_observations.get(dispatch.action_id), step.sequence)
            factors.extend(compiled)
            decisions.append({"action_id": dispatch.action_id, "node": action["node"], "templates": list(selected)})
        self.last_step = KnowledgeStep(step.base, step.revision, step.quantity_observations,
            tuple(factors), tuple(decisions), self.mechanisms.sha256)
        return copy.deepcopy(self.last_step)


def compile_knowledge_journal(rows, annotations=None, **context):
    compiler, steps = JournalKnowledgeCompiler(**context), []
    annotations, seen = ({} if annotations is None else annotations), set()
    for row in rows:
        identity = f"journal-{row['sequence']}"
        if identity in annotations:
            seen.add(identity)
        step = compiler.push(row, annotations.get(identity))
        if step is not None:
            steps.append(step)
    if set(annotations) != seen or not steps:
        raise ValueError("nonempty visible prefix with no future annotations required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), tuple(steps), compiler.workflow)


class KnowledgeFilter(TextConditionedFilter):
    step_type = KnowledgeStep

    def __init__(self, theta, *, registry, **options):
        self.registry = copy.deepcopy(registry)
        self.parameter_count = len(registry.parameter_names)
        super().__init__(theta, **options)
        self.history_tuples = np.asarray(list(product(range(3), repeat=len(registry.channels))), dtype=int)
        self.history_width = len(self.history_tuples)
        self.history_powers = 3 ** np.arange(len(registry.channels) - 1, -1, -1)
        self.distribution = np.zeros((1, self.history_width))
        self.distribution[0, 0] = 1.
        self.jacobian = np.zeros((*self.distribution.shape, len(theta))) if self.derivatives else None
        self.factor_records, self.selection_records = {}, []
        self.observed_channels = set()

    def _expand(self, n):
        size = 1 << n
        if size < len(self.distribution):
            raise ValueError("process variable identities must be append-only")
        if size > len(self.distribution):
            fresh = np.zeros((size, self.history_width))
            fresh[:len(self.distribution)] = self.distribution
            self.distribution = fresh
            if self.derivatives:
                jac = np.zeros((size, self.history_width, len(self.theta)))
                jac[:len(self.jacobian)] = self.jacobian
                self.jacobian = jac

    def _factor(self, factor, bits):
        indices, values = np.asarray(factor.parameter_indices), np.asarray(factor.values)
        q = float(expit(self.theta[indices] @ values))
        dq = None
        if self.derivatives:
            dq = np.zeros(len(self.theta))
            np.add.at(dq, indices, q * (1 - q) * values)
        histories = self.history_tuples[:, factor.channel]
        old = np.arange(self.history_width)
        sources, targets = [], []
        for process in (0, 1):
            rows = np.flatnonzero(bits[:, factor.node] == process)
            active = histories < (2 if process else 1)
            target = old[active] + ((2 if process else 1) - histories[active]) * self.history_powers[factor.channel]
            sources.append((rows[:, None] * self.history_width + old[active]).ravel())
            targets.append((rows[:, None] * self.history_width + target).ravel())
        self._move(np.concatenate(sources), np.concatenate(targets), q, dq)
        self.observed_channels.add(factor.channel)
        self.factor_records.setdefault(factor.action_id, []).append({"template": factor.template_id,
            "conditional_concern_probability": q, "parameter_names": [self.registry.parameter_names[i] for i in indices],
            "parameter_values": self.theta[indices].tolist(), "observation_values": values.tolist(),
            "evidence": copy.deepcopy(factor.evidence)})

    def apply(self, step, variables):
        if type(step) is not KnowledgeStep or step.registry_sha256 != self.registry.sha256:
            raise ValueError("knowledge compiler/model registry mismatch")
        if step.base.pending:
            raise ValueError("pending factors require a separately defined v7 model")
        if self.registry_sha256 is not None and self.registry_sha256 != step.base.capability_registry_sha256:
            raise ValueError("tool registry must remain fixed in a run")
        self.registry_sha256 = step.base.capability_registry_sha256
        self._expand(len(variables))
        self.last_revision = None
        if self.use_text and step.revision is not None:
            self._revise(step.revision)
        bits = state_bits(len(variables))
        for op in step.base.operations:
            if op.is_message and not self.use_messages:
                continue
            q, dq = self._gate(op.parameter, op.confidence_feature if op.is_message else 0.)
            sources, targets = transition_pairs(len(variables), op.receiver, op.sender)
            histories = np.arange(self.history_width)
            self._move((sources[:, None] * self.history_width + histories).ravel(),
                (targets[:, None] * self.history_width + histories).ravel(), q, dq)
        for dispatch in step.base.dispatches:
            self.dispatch_information[dispatch.action_id] = float(self.distribution.sum(axis=1) @ bits[:, dispatch.node])
        for factor in step.mechanism_factors:
            self._factor(factor, bits)
        self.selection_records.extend(copy.deepcopy(step.selection_decisions))
        info = latent_features(bits, step.base)
        h = self.history_tuples
        any_concern, overlap = (h > 0).astype(float), (h == 2).astype(float)
        if not self.use_latent_head:
            info, any_concern, overlap = np.zeros_like(info), np.zeros_like(any_concern), np.zeros_like(overlap)
        b = self.theta[CAP_HEAD:CAP_HEAD + 23]
        logits = (step.base.observables @ b[:15] + info @ b[15:19])[:, None]
        logits = logits + b[20] * any_concern[None, :, 0] + b[22] * overlap[None, :, 0]
        for k, name in enumerate(self.registry.channels[1:], 1):
            i, j = self.registry.head_indices[name]
            logits += self.theta[i] * any_concern[None, :, k] + self.theta[j] * overlap[None, :, k]
        self.conditional = expit(logits)
        self.probability = float(np.sum(self.distribution * self.conditional))
        if self.derivatives:
            gradient = np.einsum("shp,sh->p", self.jacobian, self.conditional)
            dh = self.distribution * self.conditional * (1 - self.conditional)
            gradient[CAP_HEAD:CAP_HEAD + 15] += dh.sum() * step.base.observables
            gradient[CAP_HEAD + 15:CAP_HEAD + 19] += dh.sum(axis=1) @ info
            gradient[CAP_HEAD + 20] += np.sum(dh * any_concern[None, :, 0])
            gradient[CAP_HEAD + 22] += np.sum(dh * overlap[None, :, 0])
            for k, name in enumerate(self.registry.channels[1:], 1):
                i, j = self.registry.head_indices[name]
                gradient[i] += np.sum(dh * any_concern[None, :, k])
                gradient[j] += np.sum(dh * overlap[None, :, k])
            self.gradient = gradient
        return self.probability

    def explain(self, step, variables):
        bits = state_bits(len(variables))
        marginal = self.distribution.sum(axis=1) @ bits
        history_mass = self.distribution.sum(axis=0)
        mechanisms = {}
        for k, name in enumerate(self.registry.channels):
            masses = [float(history_mass[self.history_tuples[:, k] == j].sum()) for j in range(3)]
            mechanisms[name] = {"instantiated": k in self.observed_channels, "history_probabilities": masses,
                "concern_probability": sum(masses[1:]) if k in self.observed_channels else None,
                "meaning": "Latent model hypothesis, not a calibrated violation probability. Null means no applicable factor, not verified safety."}
        actions = []
        for action in step.base.actions:
            item = {**copy.deepcopy(action), "process_concern_at_dispatch_hypothesis": self.dispatch_information.get(action["action_id"]),
                "mechanism_factors": copy.deepcopy(self.factor_records.get(action["action_id"], []))}
            actions.append(item)
        return {"risk_probability": self.probability, "sequence": step.sequence, "mass_sum": float(self.distribution.sum()),
            "registry_sha256": self.registry.sha256, "retained_joint_states": int(self.distribution.size),
            "history_state_names": list(HISTORY_STATES), "mechanisms": mechanisms,
            "variables": [{**copy.deepcopy(v), "kind": "process_concern_hypothesis" if v["kind"] == "information_state" else v["kind"],
                "marginal": float(marginal[i])} for i, v in enumerate(variables)],
            "actions": actions, "selection_decisions": copy.deepcopy(self.selection_records),
            "coverage": coverage(step.base), "text_revision": copy.deepcopy(self.last_revision),
            "system_mixture_terms": [{"history": dict(zip(self.registry.channels, (HISTORY_STATES[i] for i in row))),
                "mass": float(history_mass[n]), "risk_contribution": float(np.sum(self.distribution[:, n] * self.conditional[:, n]))}
                for n, row in enumerate(self.history_tuples)],
            "interpretation": "Distinct knowledge-selected action factors, exact joint history under conditional factor independence; not causal discovery or full role-template compilation."}


class KnowledgeScopedBN(TextConditionedBN):
    model_name, step_type = MODEL_ID, KnowledgeStep

    def __init__(self, *, mechanism_registry=None, **options):
        self.registry = copy.deepcopy(mechanism_registry or load_action_mechanisms())
        self.parameter_names = self.registry.parameter_names
        super().__init__(**options)

    def initial_parameters(self):
        return np.r_[super().initial_parameters(), self.registry.additional_initial]

    def _engine(self, theta, *, derivatives=False):
        return KnowledgeFilter(theta, registry=self.registry, use_text=self.use_text,
            mask_text_features=self.mask_text_features, use_messages=self.use_messages,
            use_latent_head=self.use_latent_head, derivatives=derivatives)

    def parameters(self):
        result = super().parameters()
        result.update(name=MODEL_ID, parameter_names=list(self.parameter_names), specification=self.registry.specification(),
            specification_sha256=self.registry.sha256, inference="Joint mechanism histories: 3**M * 2**process/source bits; M<=3.")
        result["limitations"] = [s for s in result["limitations"] if "numeric-scope comparison" not in s]
        result["limitations"] += ["V7 dispatch-only journal interface; numeric factors do not cover earlier proposals.",
            "Knowledge compilation currently selects three action fragments, not every role/risk in the catalog.",
            "No automatic role/workflow extraction, reliable cross-record source identity or textual-condition evaluation.",
            "Mechanism factors are conditionally independent given visible evidence; hidden shared causes are not resolved.",
            "Multiple scoped contracts currently collapse to aggregate features; no per-contract latent risk.",
            "No trained v7 parameters or independent MAS performance result yet."]
        return result
