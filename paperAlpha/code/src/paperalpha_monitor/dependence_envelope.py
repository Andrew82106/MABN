"""Finite-world risk bounds with fixed block marginals, unknown dependence.

This is a standard marginal-coupling linear program, not a learned dependence
model, confidence interval or causal discovery. It complements the point BN:
local predictions may be wrong even when these conditional bounds are sharp.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from paperalpha_runtime.canonical import digest
from .exclusive_dispatch_bn import ExclusiveDispatchInput
from .workflow_risk_bn import WorkflowRiskBN


class FiniteJointRiskEnvelope:
    """All distributions on declared support preserving each block marginal.

    A block can contain multiple variables whose *joint* marginal is preserved.
    Rows are complete worlds, not independent risk-channel probabilities.
    """

    def __init__(self, states, reference_mass, blocks, events, *, max_worlds=4096):
        raw = np.asarray(states)
        mass = np.asarray(reference_mass, dtype=float)
        if (raw.ndim != 2 or not raw.shape[0] or not raw.shape[1]
                or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0)):
            raise ValueError("nonempty integer category matrix required")
        if type(max_worlds) is not int or max_worlds < 1 or len(raw) > max_worlds:
            raise ValueError("finite coupling capacity exceeded; no independent fallback")
        if len(np.unique(raw, axis=0)) != len(raw):
            raise ValueError("duplicate joint worlds")
        if (mass.shape != (len(raw),) or not np.isfinite(mass).all()
                or np.any(mass < 0) or abs(mass.sum() - 1) > 1e-12):
            raise ValueError("normalized nonnegative reference mass required")
        if (len(blocks) != raw.shape[1] or any(not isinstance(b, str) or not b for b in blocks)
                or len(set(blocks)) != len(blocks)):
            raise ValueError("unique named blocks required")
        if not isinstance(events, dict) or not events:
            raise ValueError("explicit bounded risk queries required")
        self.states, self.mass, self.blocks = raw.copy(), mass.copy(), tuple(blocks)
        self.events = {}
        for name, values in events.items():
            values = np.asarray(values, dtype=float)
            if (not isinstance(name, str) or not name or values.shape != mass.shape
                    or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("finite world-aligned risks in [0,1] required")
            self.events[name] = values.copy()
        rows = [np.ones(len(mass))]
        self.constraints = [{"block": "normalization", "category": None}]
        for j, block in enumerate(blocks):
            for category in np.unique(raw[:, j]):
                rows.append((raw[:, j] == category).astype(float))
                self.constraints.append({"block": block, "category": int(category)})
        self.a = csr_matrix(np.asarray(rows))
        self.b = self.a @ self.mass

    def bound(self, query, *, include_witness=False):
        values = self.events[query]
        endpoints, audits = [], []
        witnesses = []
        for sign in (1., -1.):
            objective = sign * values
            result = linprog(objective, A_eq=self.a, b_eq=self.b, bounds=(0., None),
                             method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                                       "dual_feasibility_tolerance": 1e-9})
            if not result.success:
                raise RuntimeError("coupling optimization failed: " + result.message)
            x, dual = result.x, result.eqlin.marginals
            audit = {"max_marginal_residual": float(np.max(np.abs(self.a @ x - self.b))),
                     "negative_mass": float(max(0., -np.min(x))),
                     "max_dual_violation": float(max(0., np.max(self.a.T @ dual - objective))),
                     "primal_dual_gap": float(abs(objective @ x - self.b @ dual))}
            if max(audit.values()) > 1e-8:
                raise RuntimeError("coupling primal/dual certificate exceeds numeric tolerance")
            endpoints.append(float(values @ x)); audits.append(audit)
            if include_witness:
                witnesses.append({"mass": x.tolist(), "equality_dual": dual.tolist()})
        point = float(values @ self.mass)
        if not endpoints[0] - 1e-8 <= point <= endpoints[1] + 1e-8:
            raise RuntimeError("reference distribution not enclosed")
        answer = {"lower": endpoints[0], "reference": point, "upper": endpoints[1],
                  "width": endpoints[1] - endpoints[0], "numerical_certificates": audits}
        if include_witness:
            answer["endpoint_witnesses"] = witnesses
        return answer

    def equivalence_classes(self, query):
        """Same policy outcome for every supported setting of OTHER blocks.

        Parameter-free, query-specific equivalence; not action equivalence or
        causal attribution. Unequal context support is never merged.
        """
        values = self.events[query]
        answer = {}
        for j, name in enumerate(self.blocks):
            classes = {}
            for category in np.unique(self.states[:, j]):
                ix = np.flatnonzero(self.states[:, j] == category)
                key = tuple(sorted((tuple(np.delete(self.states[i], j)), float(values[i])) for i in ix))
                classes.setdefault(key, []).append(int(category))
            answer[name] = list(classes.values())
        return answer


def compile_envelope(model, sample, *, queries=None, max_worlds=4096):
    """Keep local choice marginals and the FULL shared-authority distribution.

    Relax dependence across dispatch slots AND between dispatch and uncertain
    authority; never duplicate an authority root per actor. No outcome labels
    or future conditions enter this API. The BN's support/coverage stay fixed.
    """
    if not isinstance(model, WorkflowRiskBN):
        raise ValueError("supported finite workflow BN required")
    worlds = model._worlds(sample, model.theta)
    log_mass = worlds["log_mass"]
    # Finite log mass retains extremely rare worlds; exp underflow is not a
    # new structural impossibility. Explicit -inf excludes impossible slots.
    r, c = np.nonzero(np.isfinite(log_mass))
    if len(r) > max_worlds:
        raise ValueError("finite coupling capacity exceeded; no independent fallback")
    mass = np.exp(log_mass[r, c])
    states, names, definitions = [], [], {}
    if isinstance(sample, ExclusiveDispatchInput):
        groups = sample.choice_groups
    else:
        groups = [{"id": a, "action_ids": [a]} for a in sample.action_ids]
    for group in groups:
        name = "dispatch:" + group["id"]
        indices = [sample.action_ids.index(a) for a in group["action_ids"]]
        bits = worlds["states"][r][:, indices]
        if np.any(bits.sum(axis=1) > 1):
            raise ValueError("exclusive world support violated")
        states.append((bits @ np.arange(1, len(indices) + 1)).astype(int))
        names.append(name)
        definitions[name] = {"kind": "dispatch_slot", "categories": [None] + group["action_ids"]}
    if worlds["fact_ids"]:
        name = "authority:joint_facts"
        states.append(c.astype(int)); names.append(name)
        definitions[name] = {"kind": "joint_authority", "fact_ids": list(worlds["fact_ids"]),
                             "categories": worlds["fact_states"].astype(int).tolist()}
    selected = list(worlds["events"]) if queries is None else list(queries)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(worlds["events"]):
        raise ValueError("nonempty unique configured queries required")
    events = {q: worlds["events"][q][r, c].astype(float) for q in selected}
    envelope = FiniteJointRiskEnvelope(np.asarray(states).T, mass, names, events, max_worlds=max_worlds)
    metadata = {"input_sha256": sample.sha256, "registry_sha256": model.registry.sha256,
                "parameter_sha256": digest(model.parameters()), "blocks": definitions,
                "worlds": len(r), "disabled_templates": sorted(set(("authorization", "single_bound",
                    "cumulative_budget", "prohibited_combination")) - set(model.registry.templates))}
    return envelope, metadata


def dependence_envelope(model, sample, *, queries=None, include_witness=False, max_worlds=4096):
    envelope, metadata = compile_envelope(model, sample, queries=queries, max_worlds=max_worlds)
    result = {"schema": "finite-dependence-risk-envelope-v1", **metadata,
              "queries": {q: {**envelope.bound(q, include_witness=include_witness),
                               "policy_equivalent_categories": envelope.equivalence_classes(q)}
                          for q in envelope.events},
              "assumptions": ["Local slot probabilities are fixed to the current BN, not validated true marginals.",
                  "The full joint distribution of shared authorization facts is preserved as one block.",
                  "Dependence across slots and between dispatch and authorization is otherwise unspecified.",
                  "Only declared finite commands, runtime support and configured policy truth tables are covered."],
              "calibrated": False, "proves_safety": False, "is_confidence_interval": False,
              "not_modeled": ["marginal model estimation error", "missing actions or policies",
                              "observation/extraction error", "future changing workflows or authority"],
              "method": "Standard finite marginal-coupling linear program; not a novel causal or dependence learner.",
              "certificate_tolerance": 1e-8}
    if include_witness:
        result["witness_problem"] = {"states": envelope.states.tolist(),
            "blocks": list(envelope.blocks), "reference_mass": envelope.mass.tolist(),
            "events": {q: v.tolist() for q, v in envelope.events.items()},
            "constraint_rows": envelope.constraints, "marginals": envelope.b.tolist()}
    return result
