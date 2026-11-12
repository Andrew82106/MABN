"""Typed action supervision on the v7 joint process model (research v8).

Observed forbidden-tool and unnecessary-tool outcomes are NOT labels for the
latent authorization bit, resource violations, or actual harm. A hierarchical
readout predicts P(forbidden|prefix) and P(unnecessary|not forbidden,prefix).
The latter excludes authorization/history features. Its conditional loss does
not update authorization factors. Neither outcome is fed back as runtime input.

The frozen v7 implementation is reused, not edited. Its final-task head may be
trained separately when eligible labels exist; action-only fitting does not
qualify that head for prediction. Exactness refers to model marginalization,
not calibration, causal correctness, or independence of real-world events.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .dynamic_joint_bn import ROLES, latent_features
from .capability_action_bn import CAP_OBSERVABLES
from .tool_capabilities import CAPABILITIES
from .knowledge_scoped_bn import KnowledgeFilter, KnowledgeScopedBN


MODEL_ID = "TypedActionSupervisionBN-v8"
PROCESS_NAMES = ("own_process", "process_fraction", "process_pair_fraction",
                 "capable_process_fraction", "terminal_process_fraction")
HIGH_NAMES = (*CAP_OBSERVABLES, *PROCESS_NAMES, "current_authorization_hypothesis",
              "current_authorization_process_overlap", "prior_authorization_any", "prior_authorization_overlap")
LOW_CELLS = tuple((cap, role) for cap in CAPABILITIES for role in ROLES)
LOW_NAMES = (*(f"capability_role:{cap}:{role}" for cap, role in LOW_CELLS),
             *CAP_OBSERVABLES[1:], *PROCESS_NAMES)
LOCAL_NAMES = ("bias", *(f"capability_role:{cap}:{role}" for cap, role in LOW_CELLS),
               "permission:no_mismatch", "permission:mismatch", "permission:unknown", "incomplete_scope")
VISIBLE_NAMES = (*LOCAL_NAMES, *CAP_OBSERVABLES[1:])


@dataclass(frozen=True)
class ActionTarget:
    forbidden: int | None
    unnecessary: int | None

    def __post_init__(self):
        for value in (self.forbidden, self.unnecessary):
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError("action labels must be 0, 1 or missing")
        if self.unnecessary is not None and self.forbidden is None:
            raise ValueError("necessity supervision requires observed forbidden status")
        if self.forbidden == 1 and self.unnecessary not in (0, None):
            raise ValueError("forbidden and unnecessary tool tiers are mutually exclusive")


@dataclass(frozen=True)
class TraceTargets:
    actions: dict[str, ActionTarget]
    episode_outcome: int | None = None

    def __post_init__(self):
        if self.episode_outcome is not None and (type(self.episode_outcome) is not int or self.episode_outcome not in (0, 1)):
            raise ValueError("episode outcome must be an eligible binary label or missing")
        if any(not isinstance(k, str) or not k or not isinstance(v, ActionTarget) for k, v in self.actions.items()):
            raise ValueError("explicit action IDs and typed targets required")


def local_features(action):
    cell = LOW_CELLS.index((action["capability_profile"]["capability"], action["role"]))
    status = action["permission_status"]
    if status not in {"no_mismatch", "mismatch", "unknown"}:
        raise ValueError("unknown permission vocabulary")
    values = np.r_[1., np.eye(len(LOW_CELLS))[cell],
                   [float(status == p) for p in ("no_mismatch", "mismatch", "unknown")],
                   float(not action["scope_complete"])]
    return values


def visible_features(step, action):
    return np.r_[local_features(action), step.base.observables[1:]]


class TypedActionFilter(KnowledgeFilter):
    def __init__(self, theta, *, registry, use_graph=True, **options):
        n = len(registry.parameter_names)
        full = np.asarray(theta, dtype=float)
        if full.shape != (n + len(HIGH_NAMES) + len(LOW_NAMES),) or not np.isfinite(full).all():
            raise ValueError("finite v8 parameter vector required")
        super().__init__(full[:n], registry=registry, **options)
        self.theta, self.base_count, self.use_graph = full.copy(), n, use_graph
        if self.derivatives:
            self.jacobian = np.zeros((*self.distribution.shape, len(full)))
        self.current_step = None
        self.action_predictions = {}
        self.action_gradients = {}

    def apply(self, step, variables):
        self.current_step = step
        return super().apply(step, variables)

    def _factor(self, factor, bits):
        # Authorization is the first factor for each dispatch. Read the PRIOR
        # history; do not let an action count itself as previous evidence.
        if factor.template_id == "authorization":
            if factor.action_id in self.action_predictions:
                raise ValueError("duplicate action readout")
            action = next(a for a in self.current_step.base.actions if a["action_id"] == factor.action_id)
            self._read_action(factor, bits, action)
        super()._factor(factor, bits)

    def _read_action(self, factor, bits, action):
        step, n = self.current_step, self.base_count
        obs = step.base.observables.copy()
        process = np.column_stack((bits[:, factor.node], latent_features(bits, step.base)))
        histories = self.history_tuples[:, 0]
        if not self.use_graph:
            obs[1:] = 0.
            process = np.zeros_like(process)
            histories = np.zeros_like(histories)
        indices, values = np.asarray(factor.parameter_indices), np.asarray(factor.values)
        q = float(expit(self.theta[indices] @ values))
        dq = np.zeros(len(self.theta))
        np.add.at(dq, indices, q * (1 - q) * values)
        # Marginalize the new Bernoulli action hypothesis before inserting it
        # into history. It is independent of previous states GIVEN observations,
        # exactly as in v7; hidden common authorization causes remain unsupported.
        conditional, explicit = [], []
        for r in (0., 1.):
            features = np.broadcast_to(np.r_[obs, np.zeros(9)], (*self.distribution.shape, len(HIGH_NAMES))).copy()
            features[:, :, 15:20] = process[:, None, :]
            features[:, :, 20] = r
            features[:, :, 21] = r * process[:, None, 0]
            features[:, :, 22] = (histories > 0)[None, :]
            features[:, :, 23] = (histories == 2)[None, :]
            cond = expit(features @ self.theta[n:n + len(HIGH_NAMES)])
            conditional.append(cond)
            if self.derivatives:
                explicit.append(np.einsum("sh,shk->k", self.distribution * cond * (1 - cond), features))
        mixed = (1 - q) * conditional[0] + q * conditional[1]
        high = float(np.sum(self.distribution * mixed))
        start = n + len(HIGH_NAMES)
        low_features = np.zeros((len(bits), len(LOW_NAMES)))
        cell = LOW_CELLS.index((action["capability_profile"]["capability"], action["role"]))
        low_features[:, cell] = 1.
        low_features[:, len(LOW_CELLS):len(LOW_CELLS) + 14] = obs[None, 1:]
        low_features[:, -5:] = process
        low_cond = expit(low_features @ self.theta[start:])
        marginal = self.distribution.sum(axis=1)
        low = float(marginal @ low_cond)
        self.action_predictions[factor.action_id] = {
            "sequence": step.sequence, "actor": action["binding"]["actor"], "stage": action["node"],
            "forbidden": high, "unnecessary_given_not_forbidden": low,
            "unnecessary": (1 - high) * low, "no_tool_tier_violation": (1 - high) * (1 - low),
            "current_authorization_hypothesis": q,
            "visible_features": visible_features(step, action).tolist(),
            "meaning": "Predicted tool-rule tiers at dispatch; no-tool-tier violation does not mean resource/communication safety or absence of harm."}
        if self.derivatives:
            gh = np.einsum("shp,sh->p", self.jacobian, mixed)
            gh += dq * np.sum(self.distribution * (conditional[1] - conditional[0]))
            gh[n:n + len(HIGH_NAMES)] += (1 - q) * explicit[0] + q * explicit[1]
            # Integrate out ALL histories; necessity cannot supervise A via
            # its own conditional loss. Shared process parameters still learn.
            gl = self.jacobian.sum(axis=1).T @ low_cond
            gl[start:] += (marginal * low_cond * (1 - low_cond)) @ low_features
            self.action_gradients[factor.action_id] = (gh, gl)


def bce(probability, target):
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(-target * np.log(p) - (1 - target) * np.log1p(-p)), (p - target) / (p * (1 - p))


class TypedActionSupervisionBN:
    """A typed observation model attached to the v7 joint filter, not v7 scores."""
    def __init__(self, *, regularization=.02, max_iter=200, use_graph=True, use_messages=True, use_text=True):
        self.base = KnowledgeScopedBN(use_messages=use_messages, use_text=use_text)
        self.registry = self.base.registry
        self.base_count = len(self.base.theta)
        self.parameter_names = (*self.base.parameter_names, *("tool_forbidden:" + k for k in HIGH_NAMES),
                                *("tool_unnecessary_conditional:" + k for k in LOW_NAMES))
        self.regularization, self.max_iter = regularization, max_iter
        self.use_graph, self.use_messages, self.use_text = use_graph, use_messages, use_text
        self.theta = self.initial_parameters()
        self.optimizer, self.fitted_targets = {}, []

    def initial_parameters(self):
        high = np.zeros(len(HIGH_NAMES))
        high[0], high[20] = -2., 1.
        low = np.zeros(len(LOW_NAMES))
        low[:len(LOW_CELLS)] = -2.
        return np.r_[self.base.initial_parameters(), high, low]

    def engine(self, theta=None, *, derivatives=False):
        return TypedActionFilter(self.theta if theta is None else theta, registry=self.registry,
            use_graph=self.use_graph, use_messages=self.use_messages, use_text=self.use_text, derivatives=derivatives)

    def trace_loss_gradient(self, theta, trace, targets):
        dispatches = [d.action_id for s in trace.steps for d in s.base.dispatches]
        if len(dispatches) != len(set(dispatches)) or set(targets.actions) - set(dispatches):
            raise ValueError("labels require unique observed dispatch IDs")
        usable = {k: v for k, v in targets.actions.items() if v.forbidden is not None}
        if not usable and targets.episode_outcome is None:
            raise ValueError("trace has no observed supervision")
        engine, loss, gradient = self.engine(theta, derivatives=True), 0., np.zeros_like(theta)
        for step in trace.steps:
            engine.apply(step, trace.variables)
            if targets.episode_outcome is not None:
                value, derivative = bce(engine.probability, targets.episode_outcome)
                loss += value / len(trace.steps)
                gradient += derivative * engine.gradient / len(trace.steps)
            for d in step.base.dispatches:
                target = usable.get(d.action_id)
                if target is None:
                    continue
                row = engine.action_predictions[d.action_id]
                gh, gl = engine.action_gradients[d.action_id]
                value, derivative = bce(row["forbidden"], target.forbidden)
                loss += value / len(usable)
                gradient += derivative * gh / len(usable)
                if target.forbidden == 0 and target.unnecessary is not None:
                    value, derivative = bce(row["unnecessary_given_not_forbidden"], target.unnecessary)
                    loss += value / len(usable)
                    gradient += derivative * gl / len(usable)
        return float(loss), gradient

    def loss_gradient(self, theta, traces, targets):
        if not traces or len(traces) != len(targets):
            raise ValueError("one typed target bundle per trace required")
        loss, gradient = 0., np.zeros_like(theta)
        for trace, target in zip(traces, targets):
            value, derivative = self.trace_loss_gradient(theta, trace, target)
            loss += value / len(traces)
            gradient += derivative / len(traces)
        centered = theta - self.initial_parameters()
        loss += self.regularization * (centered @ centered)
        gradient += 2 * self.regularization * centered
        return float(loss), gradient

    def fit(self, traces, targets):
        initial, _ = self.loss_gradient(self.theta, traces, targets)
        bounds = [(-8., 8.)] * len(self.theta)
        # An increased authorization hypothesis cannot lower the conditional
        # forbidden probability. Other effects remain unconstrained hypotheses.
        for j in (20, 21):
            bounds[self.base_count + j] = (0., 8.)
        fitted = minimize(self.loss_gradient, self.theta, args=(traces, targets), jac=True,
            method="L-BFGS-B", bounds=bounds, options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6})
        self.theta = fitted.x
        gradient = fitted.jac.copy()
        for i, (lo, hi) in enumerate(bounds):
            if (fitted.x[i] <= lo + 1e-9 and gradient[i] > 0) or (fitted.x[i] >= hi - 1e-9 and gradient[i] < 0):
                gradient[i] = 0.
        self.optimizer = {"success": bool(fitted.success), "iterations": int(fitted.nit),
            "initial_loss": initial, "loss": float(fitted.fun), "projected_gradient_max": float(np.max(np.abs(gradient))),
            "message": str(fitted.message)}
        self.fitted_targets = [name for name, present in (
            ("forbidden_tool_rule", any(v.forbidden is not None for t in targets for v in t.actions.values())),
            ("unnecessary_given_not_forbidden", any(v.forbidden == 0 and v.unnecessary is not None for t in targets for v in t.actions.values())),
            ("episode_outcome", any(t.episode_outcome is not None for t in targets))) if present]
        return self

    def predict_actions(self, trace):
        engine = self.engine()
        for step in trace.steps:
            engine.apply(step, trace.variables)
        return copy.deepcopy(engine.action_predictions)

    def predict_episode(self, trace):
        if "episode_outcome" not in self.fitted_targets:
            raise ValueError("action supervision does not qualify the untrained episode head")
        engine, result = self.engine(), []
        for step in trace.steps:
            result.append(engine.apply(step, trace.variables))
        return result

    def parameters(self):
        return {"model": MODEL_ID, "parameter_names": list(self.parameter_names), "theta": self.theta.tolist(),
            "registry_sha256": self.registry.sha256, "regularization": self.regularization, "max_iter": self.max_iter,
            "use_graph": self.use_graph, "use_messages": self.use_messages, "use_text": self.use_text,
            "optimizer": copy.deepcopy(self.optimizer), "fitted_targets": self.fitted_targets,
            "scope": "Tool-tier observation learning plus optional separately labelled episode forecast; not risk-latent truth, harm or calibrated safety.",
            "factorization": "P(forbidden|E), P(unnecessary|not forbidden,E); conditional necessity excludes authorization history.",
            "training": "Per-trace mean dispatch categorical NLL, optional per-trace mean prefix episode BCE; equal trace weights; fixed L2 around initial parameters."}
