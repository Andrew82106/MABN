"""Scoped cross-agent evidence as parents of the existing authorization factor.

This opt-in extension preserves v9's outputs and independent action hypotheses.
It does NOT model shared grant uncertainty as a joint latent common cause. A
separate endpoint loss is provided for controlled policy-outcome experiments;
these outcomes must not be relabelled as HarnessAudit forbidden-tool tiers.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from paperalpha_runtime.semantic_evidence import digest
from .delegation_evidence import FEATURE_NAMES, BINDING_FIELDS, DelegationSnapshot
from .duty_conditioned_bn import DutyMechanismRegistry, DutyConditionedBN
from .knowledge_scoped_bn import KnowledgeScopedBN
from .typed_action_supervision import HIGH_NAMES, LOW_NAMES, bce
from .dynamic_joint_bn import JointTrace
from .delegation_evidence import DelegationLedger
from .compact_action_bn import CompactActionCompiler
from .action_mechanism_knowledge import load_action_mechanisms
from .knowledge_scoped_bn import KnowledgeStep


class DelegationMechanismRegistry(DutyMechanismRegistry):
    def __init__(self):
        super().__init__()
        self.duty_registry_sha256 = self.sha256
        start = len(self.parameter_names)
        self.relation_indices = tuple(range(start, start + len(FEATURE_NAMES)))
        self.parameter_names += tuple("delegation_condition:authorization:" + x for x in FEATURE_NAMES)
        self.additional_initial = np.r_[self.additional_initial, np.zeros(len(FEATURE_NAMES))]
        self._spec = {"schema": "delegation-conditioned-mechanisms-v1", "base": self._spec,
            "features": list(FEATURE_NAMES), "knowledge_links": ["RB008", "RB009", "RB011", "RB012"],
            "equation": "sigmoid(v9 authorization logit + relation weights dot observed chain features)",
            "scope": "Externally bound adapter records and preconfigured roots; not text-only authority or enforcement."}
        self.sha256 = digest(self._spec)


class DelegationConditioner:
    def __init__(self):
        self.registry = DelegationMechanismRegistry()

    def step(self, step, snapshots):
        if step.registry_sha256 != self.registry.duty_registry_sha256:
            raise ValueError("unchanged v9 duty-conditioned factors required")
        factors, used = [], set()
        for factor in step.mechanism_factors:
            if factor.template_id != "authorization" or factor.action_id not in snapshots:
                factors.append(factor)
                continue
            item = snapshots[factor.action_id]
            action = next(a for a in step.base.actions if a["action_id"] == factor.action_id)
            bound = {k: action["binding"][k] for k in BINDING_FIELDS}
            if not isinstance(item, DelegationSnapshot) or item.sequence != step.sequence or item.action_id != factor.action_id or item.binding_sha256 != digest(bound):
                raise ValueError("relation evidence must match this exact dispatch binding and prefix")
            if item.evidence["action_binding"] != bound or item.features != tuple(item.evidence["features"][k] for k in FEATURE_NAMES):
                raise ValueError("relation evidence changed")
            used.add(factor.action_id)
            factors.append(replace(factor, parameter_indices=(*factor.parameter_indices, *self.registry.relation_indices),
                values=(*factor.values, *item.features), evidence={**copy.deepcopy(factor.evidence),
                    "delegation_relations": copy.deepcopy(item.evidence), "registry_sha256": self.registry.sha256}))
        if used != set(snapshots):
            raise ValueError("relation snapshots must cover only current dispatches")
        return replace(step, mechanism_factors=tuple(factors), registry_sha256=self.registry.sha256)

    def trace(self, trace, snapshots):
        supplied = set()
        steps = []
        for step in trace.steps:
            selected = {d.action_id: snapshots[d.action_id] for d in step.base.dispatches if d.action_id in snapshots}
            supplied.update(selected)
            steps.append(self.step(step, selected))
        if supplied != set(snapshots):
            raise ValueError("unobserved/future action evidence supplied")
        return JointTrace(trace.variables, tuple(steps), trace.workflow)


class DelegationEvidenceCompiler:
    """Opt-in externally scoped event adapter; old raw journals are unchanged.

    Grant/revoke envelopes come from DelegationLedger's adapter contract. Other
    records follow CompactActionCompiler. Dispatches may reference grant_ids;
    tool arguments, actor, principal and resource retain their original binding.
    """
    def __init__(self, *, duties, task_id, root_principals, trusted_observers, coverage_complete=False, **context):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("public task scope required")
        self.duties, self.task_id = duties, task_id
        self.ledger = DelegationLedger(root_principals=root_principals, trusted_observers=trusted_observers,
            coverage_complete=coverage_complete)
        self.compact = CompactActionCompiler(**context)
        self.base_registry, self.relations = load_action_mechanisms(), DelegationConditioner()
        self.steps, self.snapshots, self.sequence = [], {}, 0

    def push(self, record):
        sequence = record.get("sequence")
        if type(sequence) is not int or sequence <= self.sequence:
            raise ValueError("strict shared observation order required")
        ledger = copy.deepcopy(self.ledger)
        if record.get("kind") in {"grant", "revoke"}:
            ledger.observe(record)
            self.ledger, self.sequence = ledger, sequence
            return None
        compact = copy.deepcopy(self.compact)
        raw = copy.deepcopy(record)
        grant_ids = raw.pop("grant_ids", [])
        base = compact.push(raw)
        if base.pending:
            raise ValueError("this adapter is dispatch-only, not proposal revision")
        factors, decisions, snapshots = [], [], {}
        for dispatch in base.dispatches:
            action = next(a for a in base.actions if a["action_id"] == dispatch.action_id)
            compiled, selected = self.base_registry.factors(action, None, sequence)
            factors.extend(compiled)
            decisions.append({"action_id": dispatch.action_id, "templates": list(selected)})
            snapshots[dispatch.action_id] = ledger.assess(action["binding"], sequence=sequence,
                task_id=self.task_id, grant_ids=grant_ids)
        if grant_ids and not base.dispatches:
            raise ValueError("grant references require an observed dispatch")
        step = KnowledgeStep(base, None, {}, tuple(factors), tuple(decisions), self.base_registry.sha256)
        step = self.relations.step(self.duties.step(step), snapshots)
        self.ledger, self.compact, self.sequence = ledger, compact, sequence
        self.snapshots.update(snapshots)
        self.steps.append(step)
        return copy.deepcopy(step)

    def trace(self):
        if not self.steps:
            raise ValueError("at least one observed model event required")
        return JointTrace(tuple(copy.deepcopy(self.compact.variables)), tuple(copy.deepcopy(self.steps)), self.compact.workflow)


class DelegationConditionedBN(DutyConditionedBN):
    def __init__(self, *, regularization=.02, max_iter=200, use_graph=True, use_messages=True, use_text=True):
        self.registry = DelegationMechanismRegistry()
        self.base = KnowledgeScopedBN(mechanism_registry=self.registry, use_messages=use_messages, use_text=use_text)
        self.base_count = len(self.base.theta)
        self.parameter_names = (*self.base.parameter_names, *("tool_forbidden:" + k for k in HIGH_NAMES),
            *("tool_unnecessary_conditional:" + k for k in LOW_NAMES))
        self.regularization, self.max_iter = regularization, max_iter
        self.use_graph, self.use_messages, self.use_text = use_graph, use_messages, use_text
        self.theta, self.optimizer, self.fitted_targets = self.initial_parameters(), {}, []

    def import_v9(self, model):
        if type(model) is not DutyConditionedBN:
            raise ValueError("exact v9 model required")
        n = model.base_count
        self.theta = np.r_[model.theta[:n], np.zeros(len(FEATURE_NAMES)), model.theta[n:]]
        self.fitted_targets, self.optimizer = list(model.fitted_targets), {}

    def import_v8(self, model):
        raise ValueError("use an explicit v9 transfer")

    def trace_loss_gradient(self, theta, trace, targets):
        # Local tool-tier losses preserve v9. Controlled episode supervision is
        # endpoint-only; it must not label early indistinguishable prefixes as
        # already unsafe or claim a validated forecast horizon.
        local = replace(targets, episode_outcome=None)
        if any(t.forbidden is not None for t in local.actions.values()):
            loss, gradient = super().trace_loss_gradient(theta, trace, local)
        else:
            loss, gradient = 0., np.zeros_like(theta)
            if targets.actions:
                dispatches = {d.action_id for s in trace.steps for d in s.base.dispatches}
                if set(targets.actions) - dispatches:
                    raise ValueError("unknown action targets")
        if targets.episode_outcome is not None:
            engine = self.engine(theta, derivatives=True)
            for step in trace.steps:
                engine.apply(step, trace.variables)
            value, derivative = bce(engine.probability, targets.episode_outcome)
            loss += value
            gradient += derivative * engine.gradient
        elif not any(t.forbidden is not None for t in local.actions.values()):
            raise ValueError("no observed supervision")
        return float(loss), gradient

    def parameters(self):
        value = super().parameters()
        value.update(model="DelegationConditionedBN-research", relation_specification=self.registry.specification(),
            episode_loss="final observed prefix only; task outcome definition is supplied separately",
            scope="Referenced delegation-chain evidence in local factors; not full authority discovery or shared-grant latent modeling.")
        return value

    def delegation_reports(self, trace):
        """Expose evidence qualification next to, never hidden inside, a score.

        This is neither statistical calibration nor an abstention guarantee.
        It reports support for references, not authority from unobserved grants.
        """
        if "episode_outcome" not in self.fitted_targets:
            raise ValueError("endpoint supervision is required for policy-outcome reports")
        engine, result = self.engine(), []
        for step in trace.steps:
            engine.apply(step, trace.variables)
            for factor in step.mechanism_factors:
                relation = factor.evidence.get("delegation_relations")
                if factor.template_id != "authorization" or relation is None:
                    continue
                status = relation["reference_status"]
                insufficient = status in {"unresolved_reference", "no_reference"} or bool(relation["features"]["binding_incomplete"])
                result.append({"action_id": factor.action_id, "sequence": step.sequence,
                    "model_point_probability": engine.probability, "reference_status": status,
                    "evidence_insufficient": insufficient,
                    "conclusion": "insufficient_evidence" if insufficient else "referenced_chain_evaluable",
                    "calibrated": False, "proves_safety": False,
                    "candidates": copy.deepcopy(relation["candidates"]),
                    "scope": "Reference-required configured policy only; not all possible authority or actual harm."})
        return result
