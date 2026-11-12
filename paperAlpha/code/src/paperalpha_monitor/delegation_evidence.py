"""Prefix-only relations between scoped delegation records and an action.

This is a passive evidence compiler, not an authorization service. Authenticated
observer/issuer fields must come from an external adapter, never from parsed
agent text. The output concerns the referenced chain, not every possible grant.
Missing evidence stays unresolved; no grant/check is written into the old BN.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from paperalpha_runtime.canonical import canonical, digest


FEATURE_NAMES = ("support_matched", "reference_invalid", "reference_unresolved", "no_reference",
    "unverified_evidence", "scope_mismatch", "scope_expansion", "issuer_mismatch",
    "delegation_not_allowed", "expired", "revoked", "binding_incomplete", "coverage_incomplete")
BINDING_FIELDS = ("action_id", "actor", "principal", "tool", "resource", "arguments")
SCOPE_FIELDS = {"task_id", "principal", "tools", "resources", "arguments_sha256"}


def _name(value):
    return isinstance(value, str) and bool(value)


def _scope(value):
    if not isinstance(value, dict) or set(value) != SCOPE_FIELDS:
        raise ValueError("explicit task/principal/tool/resource/argument scope required")
    if not all(_name(value[k]) for k in ("task_id", "principal")):
        raise ValueError("grant task and principal must be explicit")
    for key in ("tools", "resources"):
        if not isinstance(value[key], list) or not value[key] or not all(map(_name, value[key])) or len(set(value[key])) != len(value[key]):
            raise ValueError("nonempty finite scope sets required; no wildcard semantics")
    fingerprint = value["arguments_sha256"]
    if fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("argument restriction must be a canonical digest or null")


def scope_narrows(child, parent):
    return (all(child[k] == parent[k] for k in ("task_id", "principal"))
        and all(set(child[k]) <= set(parent[k]) for k in ("tools", "resources"))
        and (parent["arguments_sha256"] is None or child["arguments_sha256"] == parent["arguments_sha256"]))


@dataclass(frozen=True)
class DelegationSnapshot:
    sequence: int
    action_id: str
    binding_sha256: str
    features: tuple[float, ...]
    evidence: dict


class DelegationLedger:
    def __init__(self, *, root_principals, trusted_observers, coverage_complete=False):
        if not root_principals or not trusted_observers or not all(map(_name, [*root_principals, *trusted_observers])):
            raise ValueError("explicit external root and observer identities required")
        if type(coverage_complete) is not bool:
            raise ValueError("coverage completeness must be an explicit boolean")
        self.roots, self.observers = frozenset(root_principals), frozenset(trusted_observers)
        self.coverage_complete = coverage_complete
        self.sequence, self.records, self.revocations, self.event_ids = 0, {}, {}, {}

    def observe(self, event):
        """Record issuance/revocation in arrival order; never sort or repair."""
        value = copy.deepcopy(event)
        if not isinstance(value, dict) or set(value) != {"event_id", "sequence", "kind", "observer", "authenticated_actor", "payload"}:
            raise ValueError("explicit adapter envelope required")
        canonical(value)
        if not _name(value["event_id"]):
            raise ValueError("event identity required")
        if value["event_id"] in self.event_ids:
            if self.event_ids[value["event_id"]] != value:
                raise ValueError("conflicting event identity")
            return False
        seq, data = value["sequence"], value["payload"]
        if type(seq) is not int or seq <= self.sequence or not isinstance(data, dict):
            raise ValueError("strict observed order and object payload required")
        if not _name(value["observer"]) or not _name(value["authenticated_actor"]):
            raise ValueError("adapter origin fields required even for reported claims")
        if value["kind"] == "grant":
            if set(data) != {"grant_id", "parent_id", "issuer", "recipient", "scope", "can_delegate", "valid_until_sequence"}:
                raise ValueError("invalid grant fields")
            if not all(_name(data[k]) for k in ("grant_id", "issuer", "recipient")) or (data["parent_id"] is not None and not _name(data["parent_id"])):
                raise ValueError("explicit grant identities required")
            _scope(data["scope"])
            if type(data["can_delegate"]) is not bool or type(data["valid_until_sequence"]) is not int or data["valid_until_sequence"] < seq:
                raise ValueError("invalid delegation flag or expiry")
            if data["grant_id"] in self.records:
                raise ValueError("grant IDs are immutable; revisions need a new ID")
            self.records[data["grant_id"]] = value
        elif value["kind"] == "revoke":
            if set(data) != {"grant_id"} or not _name(data["grant_id"]):
                raise ValueError("revocation requires a grant identity")
            # Unknown targets are retained, not promoted into valid revocations.
            self.revocations.setdefault(data["grant_id"], []).append(value)
        else:
            raise ValueError("only grant/revoke relation observations supported")
        self.sequence, self.event_ids[value["event_id"]] = seq, value
        return True

    def _chain(self, grant_id, at, seen=()):
        if grant_id in seen:
            return {"status": "invalid", "reasons": ["issuer_mismatch"], "path": [grant_id], "detail": "cyclic reference"}
        event = self.records.get(grant_id)
        if event is None:
            return {"status": "unresolved", "reasons": ["missing_parent"], "path": [grant_id]}
        data, reasons = event["payload"], []
        if event["sequence"] >= at:
            return {"status": "unresolved", "reasons": ["not_yet_observed"], "path": [grant_id]}
        attested = event["observer"] in self.observers and event["authenticated_actor"] == data["issuer"]
        if not attested:
            return {"status": "unresolved", "reasons": ["unverified_evidence"], "path": [grant_id]}
        if data["valid_until_sequence"] < at:
            reasons.append("expired")
        if any(event["sequence"] < r["sequence"] < at and r["observer"] in self.observers
               and r["authenticated_actor"] == data["issuer"] for r in self.revocations.get(grant_id, [])):
            reasons.append("revoked")
        path, parent_status = [grant_id], "supported"
        if data["parent_id"] is None:
            if data["issuer"] not in self.roots:
                reasons.append("issuer_mismatch")
        else:
            parent = self._chain(data["parent_id"], at, (*seen, grant_id))
            path, parent_status = parent["path"] + path, parent["status"]
            reasons += parent["reasons"]
            record = self.records.get(data["parent_id"])
            if record is not None:
                p = record["payload"]
                if record["sequence"] >= event["sequence"] or p["recipient"] != data["issuer"]:
                    reasons.append("issuer_mismatch")
                if not p["can_delegate"]:
                    reasons.append("delegation_not_allowed")
                if not scope_narrows(data["scope"], p["scope"]) or data["valid_until_sequence"] > p["valid_until_sequence"]:
                    reasons.append("scope_expansion")
        invalid = set(reasons) - {"missing_parent", "not_yet_observed", "unverified_evidence"}
        status = "invalid" if invalid else ("unresolved" if parent_status == "unresolved" else "supported")
        return {"status": status, "reasons": sorted(set(reasons)), "path": path}

    def assess(self, action, *, sequence, task_id, grant_ids):
        if type(sequence) is not int or sequence <= self.sequence:
            raise ValueError("action must follow the observed evidence prefix")
        if not isinstance(action, dict) or not set(BINDING_FIELDS) <= set(action) or not isinstance(action["arguments"], dict):
            raise ValueError("action binding required")
        if not all(_name(action[k]) for k in ("action_id", "actor", "tool")):
            raise ValueError("action identities required")
        if not isinstance(grant_ids, (list, tuple)) or any(not _name(x) for x in grant_ids) or len(set(grant_ids)) != len(grant_ids):
            raise ValueError("unique referenced grant IDs required")
        bound = {k: copy.deepcopy(action[k]) for k in BINDING_FIELDS}
        binding_complete = all(_name(x) for x in (task_id, action["principal"], action["resource"]))
        candidates = []
        for identity in grant_ids:
            candidate = self._chain(identity, sequence)
            candidate["grant_id"] = identity
            event = self.records.get(identity)
            if event is not None and candidate["status"] == "supported":
                grant, scope = event["payload"], event["payload"]["scope"]
                if not binding_complete:
                    candidate.update(status="unresolved", reasons=["binding_incomplete"])
                elif (grant["recipient"] != action["actor"] or scope["task_id"] != task_id or scope["principal"] != action["principal"]
                    or action["tool"] not in scope["tools"] or action["resource"] not in scope["resources"]
                    or (scope["arguments_sha256"] is not None and scope["arguments_sha256"] != digest(action["arguments"]))):
                    candidate.update(status="invalid", reasons=["scope_mismatch"])
            candidates.append(candidate)
        reasons = {r for c in candidates for r in c["reasons"]}
        values = {name: float(name in reasons) for name in FEATURE_NAMES}
        values.update(support_matched=float(any(c["status"] == "supported" for c in candidates)),
            reference_invalid=float(any(c["status"] == "invalid" for c in candidates)),
            reference_unresolved=float(any(c["status"] == "unresolved" for c in candidates)),
            no_reference=float(not grant_ids), binding_incomplete=float(not binding_complete),
            coverage_incomplete=float(not self.coverage_complete))
        diagnostics = dict(values)
        # This fragment asks whether at least one referenced chain supports the
        # exact action. An irrelevant/invalid alternative must not negate an
        # independently supported reference. Preserve every candidate for audit.
        if values["support_matched"]:
            for name in FEATURE_NAMES:
                if name not in {"support_matched", "coverage_incomplete"}:
                    values[name] = 0.
        status = ("supported_reference" if values["support_matched"] else "unresolved_reference" if values["reference_unresolved"]
            else "invalid_references" if values["reference_invalid"] else "no_reference")
        evidence = {"candidates": candidates, "task_id": task_id, "action_binding": bound,
            "features": values, "diagnostic_features": diagnostics, "reference_status": status,
            "feature_names": list(FEATURE_NAMES), "observed_through_sequence": self.sequence,
            "root_principals": sorted(self.roots), "trusted_observers": sorted(self.observers),
            "grants_permission": False, "checks_all_possible_grants": False,
            "source_records": [copy.deepcopy(e) for e in self.event_ids.values()],
            "meaning": "Observed referenced-chain support, not a cryptographic verifier, policy decision, denial, or proof of harm."}
        # Advancing the observation cursor prevents later evidence being applied
        # to a historical action. Existing snapshots remain immutable copies.
        self.sequence = sequence
        return DelegationSnapshot(sequence, action["action_id"], digest(bound), tuple(values[k] for k in FEATURE_NAMES), evidence)
