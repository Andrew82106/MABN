"""Shared uncertain facts and deterministic AND/OR authorization fragments.

Conditional model for a DECLARED static uncertainty window. Observed changes
are reflected in each immutable action proof; unobserved state is assumed fixed
within that window. This is not a learned revocation hazard, authentication,
causal discovery, or a calibrated authority decision. Distinct primitive facts
are independent; reuses of one fact are NOT independent.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from paperalpha_runtime.canonical import digest
from .delegation_evidence import DelegationSnapshot, DelegationLedger

PRIOR_NAMES = ("unseen_root_invalidity", "unseen_delegation_invalidity", "opaque_unresolved_reference")


@dataclass(frozen=True)
class AuthorityWindow:
    identity: str
    first_sequence: int
    last_sequence: int
    complete_grant_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.identity, str) or not self.identity or type(self.first_sequence) is not int
                or type(self.last_sequence) is not int or not 0 <= self.first_sequence <= self.last_sequence):
            raise ValueError("explicit bounded authority-uncertainty window required")
        if (not isinstance(self.complete_grant_ids, tuple)
                or any(not isinstance(g, str) or not g for g in self.complete_grant_ids)
                or len(set(self.complete_grant_ids)) != len(self.complete_grant_ids)):
            raise ValueError("explicit unique completely observed grant identities required")


@dataclass(frozen=True)
class UncertainFact:
    identity: str
    kind: int
    provenance: dict


@dataclass(frozen=True)
class ActionProof:
    action_id: str
    sequence: int
    binding_sha256: str
    paths: tuple[tuple[str, ...], ...]
    evidence: dict


class AuthorityProofGraph:
    def __init__(self, window, *, max_facts=8):
        if not isinstance(window, AuthorityWindow) or type(max_facts) is not int or not 0 <= max_facts <= 16:
            raise ValueError("bounded window and explicit exact-enumeration capacity required")
        self.window, self.max_facts = window, max_facts
        self.facts, self.proofs, self.sequence = {}, {}, -1

    def add(self, snapshot):
        if not isinstance(snapshot, DelegationSnapshot):
            raise ValueError("externally scoped delegation snapshot required")
        if not self.window.first_sequence <= snapshot.sequence <= self.window.last_sequence:
            raise ValueError("action is outside the declared uncertainty window")
        if snapshot.sequence <= self.sequence or snapshot.action_id in self.proofs:
            raise ValueError("new action in strict prefix order required")
        e = copy.deepcopy(snapshot.evidence)
        if snapshot.binding_sha256 != digest(e["action_binding"]) or snapshot.action_id != e["action_binding"]["action_id"]:
            raise ValueError("action binding changed")
        if tuple(e["features"][k] for k in e["feature_names"]) != snapshot.features:
            raise ValueError("snapshot features changed")
        check = DelegationLedger(root_principals=e["root_principals"], trusted_observers=e["trusted_observers"],
            coverage_complete=not bool(e["features"]["coverage_incomplete"]))
        for record in e["source_records"]:
            if record["sequence"] >= snapshot.sequence:
                raise ValueError("future authority record in action snapshot")
            check.observe(record)
        recomputed = check.assess(e["action_binding"], sequence=snapshot.sequence, task_id=e["task_id"],
            grant_ids=[c["grant_id"] for c in e["candidates"]]).evidence
        if any(recomputed[k] != e[k] for k in ("candidates", "features", "reference_status")):
            raise ValueError("proof candidates differ from observed authority records")
        records = {r["payload"]["grant_id"]: r for r in e["source_records"] if r["kind"] == "grant"}
        additions, paths = {}, []

        def unknown(kind, basis):
            identity = digest({"window": (self.window.identity, self.window.first_sequence, self.window.last_sequence), "task": e["task_id"],
                "principal": e["action_binding"]["principal"], "kind": kind, "basis": basis})
            fact = UncertainFact(identity, kind, copy.deepcopy(basis))
            if identity in self.facts and self.facts[identity] != fact:
                raise ValueError("shared primitive identity changed")
            additions[identity] = fact
            return identity

        for candidate in e["candidates"]:
            uncertain_basis = bool(set(candidate["reasons"]) & {"unverified_evidence", "missing_parent", "not_yet_observed", "binding_incomplete"})
            if candidate["status"] == "invalid" and not uncertain_basis:
                # Under the explicit reference-required policy, a disproven
                # candidate is false. Another candidate may still support it.
                continue
            if candidate["status"] == "unresolved" or uncertain_basis:
                # Unverified/missing chains do not authorize merging identities.
                # Preserve an opaque unknown, scoped to this exact action.
                paths.append((unknown(2, {"action_binding": e["action_binding"], "candidate": candidate}),))
                continue
            if candidate["status"] != "supported":
                raise ValueError("unsupported candidate status")
            path = []
            for identity in candidate["path"]:
                event = records.get(identity)
                if (event is None or event["sequence"] >= snapshot.sequence
                        or event["observer"] not in e["trusted_observers"]
                        or event["authenticated_actor"] != event["payload"]["issuer"]):
                    raise ValueError("shared paths require actual prior attested records")
                if e["features"]["coverage_incomplete"] and identity not in self.window.complete_grant_ids:
                    path.append(unknown(0 if event["payload"]["parent_id"] is None else 1,
                        {"grant_record": event, "meaning": "Unseen invalidity assumed fixed within the declared window."}))
            paths.append(tuple(path))
        if e["features"]["binding_incomplete"]:
            paths = [(unknown(2, {"action_binding": e["action_binding"], "reason": "binding_incomplete"}),)]
        # No references means failure of this declared reference-required policy,
        # not proof that no unreferenced external authority exists.
        paths = tuple(dict.fromkeys(paths))
        # A proven supporting alternative makes all uncertain alternatives
        # irrelevant for this action; do not spend capacity on irrelevant facts.
        if () in paths:
            paths = ((),)
        needed = {identity for path in paths for identity in path}
        additions = {k: v for k, v in additions.items() if k in needed}
        if len(set(self.facts) | set(additions)) > self.max_facts:
            raise ValueError("shared-uncertainty capacity exceeded; no independent fallback")
        proof = ActionProof(snapshot.action_id, snapshot.sequence, snapshot.binding_sha256, paths, e)
        self.facts.update(additions)
        self.proofs[proof.action_id], self.sequence = proof, snapshot.sequence
        return copy.deepcopy(proof)

    def validate(self):
        rebuilt = AuthorityProofGraph(self.window, max_facts=self.max_facts)
        for proof in self.proofs.values():
            e = proof.evidence
            rebuilt.add(DelegationSnapshot(proof.sequence, proof.action_id, proof.binding_sha256,
                tuple(e["features"][k] for k in e["feature_names"]), e))
        if rebuilt.facts != self.facts or rebuilt.proofs != self.proofs:
            raise ValueError("shared graph changed after evidence compilation")

    def worlds(self, prior_logits):
        self.validate()
        theta = np.asarray(prior_logits, dtype=float)
        if theta.shape != (len(PRIOR_NAMES),) or not np.isfinite(theta).all():
            raise ValueError("finite shared-prior logits required")
        ids = tuple(self.facts)
        n = len(ids)
        states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
        kinds = np.asarray([self.facts[k].kind for k in ids], dtype=int)
        logits = theta[kinds]
        log_mass = states @ (-np.logaddexp(0., -logits)) + (1. - states) @ (-np.logaddexp(0., logits))
        mass = np.exp(log_mass)
        score = np.zeros((len(states), len(theta)))
        for j, kind in enumerate(kinds):
            score[:, kind] += states[:, j] - expit(logits[j])
        lookup = {identity: i for i, identity in enumerate(ids)}
        failures = {}
        for identity, proof in self.proofs.items():
            supported = np.zeros(len(states), dtype=bool)
            for path in proof.paths:
                supported |= np.all(states[:, [lookup[k] for k in path]] == 0, axis=1)
            failures[identity] = ~supported
        return ids, states, mass, score, failures

    def query(self, prior_logits, *, at_least=1):
        if type(at_least) is not int or at_least < 1 or not self.proofs:
            raise ValueError("positive failure count and observed actions required")
        _, _, mass, score, failures = self.worlds(prior_logits)
        count = np.sum(list(failures.values()), axis=0)
        event = (count >= at_least).astype(float)
        probability = float(mass @ event)
        gradient = (mass * (event - probability)) @ score
        return probability, gradient

    def specification(self):
        return {"window": {"id": self.window.identity, "first_sequence": self.window.first_sequence,
                "last_sequence": self.window.last_sequence, "complete_grant_ids": list(self.window.complete_grant_ids)},
            "facts": [{"id": f.identity, "prior": PRIOR_NAMES[f.kind], "provenance": copy.deepcopy(f.provenance)} for f in self.facts.values()],
            "actions": [{"id": p.action_id, "sequence": p.sequence, "binding_sha256": p.binding_sha256,
                "supported_if_any_path_has_no_failed_fact": [list(path) for path in p.paths]} for p in self.proofs.values()],
            "semantics": "Shared Bernoulli invalidity facts; AND along a path, OR across alternatives. Empty path=true; no paths=false.",
            "assumptions": ["Unobserved invalidity is fixed within the externally declared window.",
                "Distinct primitives are conditionally independent; issuer-wide failures are not modeled.",
                "Attestation and per-grant coverage are external adapter contracts, not strings proving trust."],
            "scope": "Referenced-authority policy only, not all possible authority or actual harm.",
            "calibrated": False, "proves_safety": False}


@dataclass(frozen=True)
class SharedAuthorityTrace:
    trace: object
    graph: AuthorityProofGraph


class SharedAuthorityCompiler:
    def __init__(self, base, window, *, max_facts=8):
        self.base, self.graph = base, AuthorityProofGraph(window, max_facts=max_facts)

    def push(self, record):
        if type(record.get("sequence")) is not int or not self.graph.window.first_sequence <= record["sequence"] <= self.graph.window.last_sequence:
            raise ValueError("record outside the declared uncertainty window")
        # Transactional capacity/binding checks; rejected records consume nothing.
        base, graph = copy.deepcopy(self.base), copy.deepcopy(self.graph)
        step = base.push(record)
        if step is not None:
            for dispatch in step.base.dispatches:
                graph.add(base.snapshots[dispatch.action_id])
        self.base, self.graph = base, graph
        return step

    def trace(self):
        return SharedAuthorityTrace(self.base.trace(), copy.deepcopy(self.graph))
