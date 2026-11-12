"""Public duties condition local BN factors, not actor-ID memorization (v9).

The 20 grounded candidate-duty features enter the authorization hypothesis as
observed parents. They do not modify permission_status or establish a grant.
Only the typed outcome emission is supervised; the hidden variable's true
semantics and the dependency graph still require independent validation.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from paperalpha_runtime.duty_graph import duty_vectors, DUTY_FEATURE_NAMES
from paperalpha_runtime.semantic_evidence import digest
from .action_mechanism_knowledge import load_action_mechanisms
from .dynamic_joint_bn import JointTrace
from .knowledge_scoped_bn import KnowledgeScopedBN
from .staged_knowledge_bn import StageKnowledgeCompiler
from .typed_action_supervision import TypedActionSupervisionBN, HIGH_NAMES, LOW_NAMES

MODEL_ID = "PublicDutyConditionedBN-v9"


class DutyMechanismRegistry:
    def __init__(self):
        base = load_action_mechanisms()
        self.original_sha256 = base.sha256
        self.original_parameter_count = len(base.parameter_names)
        self.duty_indices = tuple(range(self.original_parameter_count, self.original_parameter_count + len(DUTY_FEATURE_NAMES)))
        self.parameter_names = (*base.parameter_names, *("duty_condition:authorization:" + name for name in DUTY_FEATURE_NAMES))
        self.additional_initial = np.r_[base.additional_initial, np.zeros(len(DUTY_FEATURE_NAMES))]
        self.channels, self.templates = base.channels, copy.deepcopy(base.templates)
        self.numeric_slices, self.head_indices = copy.deepcopy(base.numeric_slices), copy.deepcopy(base.head_indices)
        self._spec = {"schema": "duty-conditioned-mechanisms-v1", "base": base.specification(),
            "conditioning": {"template": "authorization", "features": list(DUTY_FEATURE_NAMES),
                "equation": "sigmoid(base capability-role-check cell + duty_weights dot candidate_duties)",
                "source": "Initial public setup; exact-quote-grounded candidate classification, not runtime/hidden policy.",
                "unknown": "Separate uncertain/unresolved/incomplete indicators. Unmentioned duty is not a prohibition.",
                "scope": "Actor-wide declared duties, not verified current-stage duties or permissions.",
                "dependencies_used_for_probability": False}}
        self.sha256 = digest(self._spec)

    def specification(self):
        return copy.deepcopy(self._spec)


class DutyConditioner:
    def __init__(self, packet, annotation, aliases):
        self.vectors, self.graph = duty_vectors(packet, annotation, aliases=aliases)
        self.packet, self.annotation, self.aliases = copy.deepcopy(packet), copy.deepcopy(annotation), dict(aliases)
        self.registry = DutyMechanismRegistry()

    def step(self, step):
        if step.registry_sha256 != self.registry.original_sha256:
            raise ValueError("duty conditioning expects unchanged base v7 factors")
        factors = []
        for factor in step.mechanism_factors:
            if factor.template_id != "authorization":
                factors.append(factor)
                continue
            actor = factor.evidence["actor"]
            if actor not in self.vectors:
                raise ValueError("runtime actor is absent from the public duty roster")
            vector = self.vectors[actor]
            exposed = self.aliases[actor]
            evidence = {**copy.deepcopy(factor.evidence), "registry_sha256": self.registry.sha256,
                "public_duties": {"packet_sha256": self.packet["sha256"], "annotation_sha256": digest(self.annotation),
                    "feature_names": list(DUTY_FEATURE_NAMES), "features": list(vector),
                    "anchors": [copy.deepcopy(d) for d in self.graph["duties"] if d["actor"] == exposed],
                    "grants_permission": False, "uses_dependency_satisfaction": False}}
            factors.append(replace(factor, parameter_indices=(*factor.parameter_indices, *self.registry.duty_indices),
                values=(*factor.values, *vector), evidence=evidence))
        return replace(step, mechanism_factors=tuple(factors), registry_sha256=self.registry.sha256)

    def trace(self, trace):
        return JointTrace(trace.variables, tuple(self.step(s) for s in trace.steps), trace.workflow)


class DutyStageCompiler:
    def __init__(self, *, packet, annotation, aliases, **context):
        self.conditioner = DutyConditioner(packet, annotation, aliases)
        self.base = StageKnowledgeCompiler(**context)
        self.last_step = None

    @property
    def variables(self):
        return self.base.variables

    @property
    def workflow(self):
        return self.base.workflow

    def push(self, row, annotation=None):
        step = self.base.push(row, annotation)
        if step is not None:
            self.last_step = self.conditioner.step(step)
            return self.last_step
        return None


class DutyConditionedBN(TypedActionSupervisionBN):
    def __init__(self, *, regularization=.02, max_iter=200, use_graph=True, use_messages=True, use_text=True):
        self.registry = DutyMechanismRegistry()
        self.base = KnowledgeScopedBN(mechanism_registry=self.registry, use_messages=use_messages, use_text=use_text)
        self.base_count = len(self.base.theta)
        self.parameter_names = (*self.base.parameter_names, *("tool_forbidden:" + k for k in HIGH_NAMES),
                                *("tool_unnecessary_conditional:" + k for k in LOW_NAMES))
        self.regularization, self.max_iter = regularization, max_iter
        self.use_graph, self.use_messages, self.use_text = use_graph, use_messages, use_text
        self.theta = self.initial_parameters()
        self.optimizer, self.fitted_targets = {}, []

    def import_v8(self, model):
        if type(model) is not TypedActionSupervisionBN:
            raise ValueError("exact v8 parent model required")
        n = self.registry.original_parameter_count
        self.theta = np.r_[model.theta[:n], np.zeros(len(DUTY_FEATURE_NAMES)), model.theta[n:]]
        # Parameter transfer is not new training. Preserve which heads actually
        # have supervision, but do not inherit optimizer/performance evidence.
        self.fitted_targets = list(model.fitted_targets)
        self.optimizer = {}

    def parameters(self):
        result = super().parameters()
        result.update(model=MODEL_ID, duty_specification=self.registry.specification(),
            scope="Public semantic duties are observed parents of local authorization hypotheses; not names, permissions or proven causal prerequisites.")
        return result
