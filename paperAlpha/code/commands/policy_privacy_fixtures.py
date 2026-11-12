"""Explicitly constructed mechanism fixtures; not API traces or test data."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.delegation_evidence import DelegationLedger
from paperalpha_monitor.shared_authority_graph import AuthorityWindow, AuthorityProofGraph
from paperalpha_monitor.policy_gated_privacy import ScopedRelease, ReleaseBinding, ReleaseContext
from paperalpha_monitor.privacy_model_independent import SCHEMA
from paperalpha_runtime.canonical import digest


def fixture(*, shared=True, status="uncertain", stage=2, c=(1, 1), w=(1, 1), aliases=False, unknown_components="root"):
    if status not in {"uncertain", "valid", "revoked", "recipient_mismatch", "mixed", "missing_root"}:
        raise ValueError("explicit fixture mode required")
    fields = [("record_ref", "alias_ref") if aliases else ("record_ref",), ("internal_cost",)]
    resources = ["document-record", "account-ledger"]
    packet = {"schema": SCHEMA, "stage": stage, "vertical": "corporate", "coordinator_chars": 120,
        "groups": [{"kind": kind, "field_names": list(names), "coordinator_present": bool(c[i]),
            "worker_present": bool(w[i]) if stage == 2 else None}
            for i, (kind, names) in enumerate(zip(("canary", "number"), fields))]}
    ledger = DelegationLedger(root_principals=["owner"], trusted_observers=["policy-adapter"],
        coverage_complete=status != "uncertain")
    authorized_recipient, purpose = "authorized-reviewer", "review-records"

    def grant(seq, identity, issuer, recipient, parent=None, arguments=None, resource_subset=None):
        return {"event_id": "grant-" + identity, "sequence": seq, "kind": "grant", "observer": "policy-adapter",
            "authenticated_actor": issuer, "payload": {"grant_id": identity, "parent_id": parent, "issuer": issuer,
                "recipient": recipient, "scope": {"task_id": "task-release", "principal": "owner", "tools": ["release_literal"],
                    "resources": resources if resource_subset is None else resource_subset,
                    "arguments_sha256": None if arguments is None else digest(arguments)},
                "can_delegate": parent is None, "valid_until_sequence": 30}}

    records = [grant(1, "root-a", "owner", "planner"), grant(2, "root-b", "owner", "planner")]
    for i, names in enumerate(fields):
        args = {"recipient": authorized_recipient, "purpose": purpose, "field_names": sorted(names)}
        records.append(grant(3 + i, f"child-{i}", "planner", "publisher", "root-a" if shared or i == 0 else "root-b",
            arguments=args, resource_subset=[resources[i]]))
    if status in {"revoked", "mixed"}:
        records.append({"event_id": "revoke", "sequence": 5, "kind": "revoke", "observer": "policy-adapter",
            "authenticated_actor": "owner" if status == "revoked" else "planner",
            "payload": {"grant_id": "root-a" if status == "revoked" else "child-0"}})
    for record in records:
        if status == "missing_root" and record["kind"] == "grant" and record["payload"]["parent_id"] is None:
            continue
        ledger.observe(record)
    if unknown_components not in {"root", "delegation", "both"}:
        raise ValueError("explicit uncertainty fixture required")
    complete = {"root": ("child-0", "child-1"), "delegation": ("root-a", "root-b"), "both": ()}[unknown_components]
    graph = AuthorityProofGraph(AuthorityWindow("release-window", 0, 30, complete))
    recipient = "unexpected-external-recipient" if status == "recipient_mismatch" else authorized_recipient
    bindings = []
    for i, names in enumerate(fields):
        action = {"action_id": f"release-{i}", "actor": "publisher", "principal": "owner", "tool": "release_literal",
            "resource": resources[i], "arguments": {"recipient": recipient, "purpose": purpose, "field_names": sorted(names)}}
        graph.add(ledger.assess(action, sequence=10 + i, task_id="task-release", grant_ids=[f"child-{i}"]))
        bindings.append(ReleaseBinding(tuple(names), action["action_id"], resources[i]))
    context = ReleaseContext("task-release", "publisher", "owner", recipient, purpose, 12)
    return ScopedRelease(packet, graph, context, bindings)
