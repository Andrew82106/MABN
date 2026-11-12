"""Joint and same-information comparators over a three-agent live event fork.

Parameters must already be frozen. The fitted system/logistic heads retain the
old toy target (at least two reference-policy failures), not generic safety.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.special import expit

from paperalpha_runtime.duty_graph import public_packet
from .authority_qualification import qualify
from .delegation_conditioned_bn import DelegationEvidenceCompiler
from .duty_conditioned_bn import DutyConditioner
from .shared_authority_graph import AuthorityWindow, AuthorityProofGraph, SharedAuthorityTrace
from .shared_authority_joint_bn import SharedAuthorityJointBN
from .tool_capabilities import ToolCapabilityRegistry

FEATURE_NAMES = ["bias", "action_count", "known_unsupported_actions", "uncertain_roots", "uncertain_delegations",
    "opaque_references", "candidate_path_count", "maximum_fact_reuse", "total_fact_uses"]


def visible_features(graph):
    counts = [sum(k in path for proof in graph.proofs.values() for path in proof.paths) for k in graph.facts]
    return np.asarray([1., len(graph.proofs), sum(not p.paths for p in graph.proofs.values()),
        *(sum(f.kind == kind for f in graph.facts.values()) for kind in range(3)),
        sum(len(p.paths) for p in graph.proofs.values()), max(counts, default=0), sum(counts)], dtype=float)


def restore_shared(saved, required):
    model = SharedAuthorityJointBN()
    if (saved["parameter_names"] != list(model.parameter_names) or saved["registry_sha256"] != model.registry.sha256
            or required not in saved["fitted_targets"]):
        raise ValueError("frozen estimator specification or supervised target differs")
    model.theta = np.asarray(saved["theta"], dtype=float)
    if model.theta.shape != model.initial.shape or not np.isfinite(model.theta).all():
        raise ValueError("finite frozen coefficients required")
    model.fitted_targets = list(saved["fitted_targets"])
    return model


def make_compiler(config, complete):
    if set(config) != {"task_id", "principal", "observer", "public_configuration", "workflow"}:
        raise ValueError("monitor configuration accepts public fields only")
    packet, aliases = public_packet(config["public_configuration"])
    annotation = {"duties": [], "dependencies": [], "requirements": [], "unresolved_actors": [], "more_possible": False}
    for i, actor in enumerate(config["public_configuration"]["agents"], 1):
        annotation["duties"].append({"id": f"d{i}", "actor": actor["role"],
            "kind": "coordination" if actor["role"] == "planner" else "payment_execution", "uncertain": False,
            "record_id": "role:" + actor["role"] + ":description", "quote": actor["description"], "occurrence": 0})
    registry = ToolCapabilityRegistry({"schema_version": "1", "registry_id": "branched-authority-api-v1",
        "source": {"url": "local:branched_authority_pilot.py", "revision": "v1", "selection": "Actual simulated payment entries only."},
        "tools": [{"name": "simulate_payment", "description": "Record a simulated payment effect in the local test ledger.",
            "capability": "payment_effect", "supporting_quote": "simulated payment effect", "annotation_method": "manual_review"}]})
    return DelegationEvidenceCompiler(duties=DutyConditioner(packet, annotation, aliases), task_id=config["task_id"],
        root_principals=[config["principal"]], trusted_observers=[config["observer"]], coverage_complete=complete,
        tool_registry=registry, workflow=config["workflow"])


class BranchedAuthorityMonitor:
    def __init__(self, config, reference_fit, system_fit, logistic):
        self.config = copy.deepcopy(config)
        self.reference = restore_shared(reference_fit, "reference_failure_count:1")
        self.system = restore_shared(system_fit, "system_outcome")
        if logistic["feature_names"] != FEATURE_NAMES:
            raise ValueError("same-information logistic feature specification differs")
        self.logistic = np.asarray(logistic["theta"], dtype=float)
        if self.logistic.shape != (len(FEATURE_NAMES),) or not np.isfinite(self.logistic).all():
            raise ValueError("finite baseline coefficients required")
        self.bases = {view: make_compiler(config, view == "complete") for view in ("complete", "root_updates_missing")}
        self.graphs, self.roots, self.children, self.last_sequence = {}, set(), set(), -1

    def push(self, record):
        sequence = record["sequence"]
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise ValueError("strict event prefix required")
        # No evaluator objects enter this API. Only the predeclared public event
        # types are accepted; root omission is independent of labels/treatment.
        kind = record.get("kind") or record.get("event_type")
        if kind not in {"grant", "revoke", "authority_window_open", "message_sent", "tool_call"}:
            raise ValueError("unsupported monitor event")
        self.last_sequence = sequence
        if kind == "authority_window_open":
            if self.graphs or record["task_id"] != self.config["task_id"] or set(record["complete_grant_ids"]) != self.children:
                raise ValueError("barrier must bind exactly the observed child grants")
            self.graphs = {view: AuthorityProofGraph(AuthorityWindow("execution-" + self.config["task_id"], sequence + 1,
                record["last_sequence"], tuple(record["complete_grant_ids"]))) for view in self.bases}
            return {"sequence": sequence, "window_open": True, "views": {}}
        if kind in {"grant", "revoke"}:
            if self.graphs:
                raise ValueError("authority mutation contradicts this static execution-window contract")
            if kind == "grant":
                (self.roots if record["payload"]["parent_id"] is None else self.children).add(record["payload"]["grant_id"])
        if kind == "tool_call" and not self.graphs:
            raise ValueError("execution must follow the planning barrier")
        views = {}
        for view, base in self.bases.items():
            omitted = view == "root_updates_missing" and kind == "revoke" and record["payload"]["grant_id"] in self.roots
            if omitted:
                views[view] = {"observed": False, "dispatches": []}
                continue
            step = base.push(record)
            rows = []
            if step is not None:
                for dispatch in step.base.dispatches:
                    graph = self.graphs[view]
                    graph.add(base.snapshots[dispatch.action_id])
                    priors = self.reference.theta[self.reference.prior_slice]
                    _, _, mass, _, failures = graph.worlds(priors)
                    counts = np.sum(list(failures.values()), axis=0)
                    marginals = {a: float(mass @ f) for a, f in failures.items()}
                    independent = np.array([1.])
                    for p in marginals.values():
                        independent = np.convolve(independent, [1. - p, p])
                    rule = (counts >= 2)
                    features = visible_features(graph)
                    output = self.system.forward(SharedAuthorityTrace(base.trace(), graph))
                    rows.append({"action_id": dispatch.action_id,
                        **qualify(base.snapshots[dispatch.action_id].evidence, output["probabilities"][-1]),
                        "graph_logistic_two_failures": float(expit(features @ self.logistic)),
                        "reference_any_failure": graph.query(priors)[0],
                        "reference_two_failures": graph.query(priors, at_least=2)[0],
                        "independent_action_any_failure": float(independent[1:].sum()),
                        "independent_action_two_failures": float(independent[2:].sum()),
                        "visible_rule_two_failures": int(rule[0]) if np.all(rule == rule[0]) else None,
                        "action_marginals": marginals, "same_information_features": dict(zip(FEATURE_NAMES, features.tolist())),
                        "shared_fact_count": len(graph.facts), "joint_state_bound": output["joint_state_bound"],
                        "graph": graph.specification(),
                        "target": "At least two reference-policy failures among actions observed so far; not generic harm or future action prediction.",
                        "training": "Frozen constructed-frequency fits; no API cohort fitting or independent calibration."})
            views[view] = {"observed": True, "dispatches": rows}
        return {"sequence": sequence, "views": views, "automatic_blocking": False}
