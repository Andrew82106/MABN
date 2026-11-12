import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.delegation_evidence import DelegationLedger
from paperalpha_runtime.semantic_evidence import digest


def grant(seq, identity, issuer, recipient, parent=None, **changes):
    payload = {"grant_id": identity, "parent_id": parent, "issuer": issuer, "recipient": recipient,
        "scope": {"task_id": "task-a", "principal": "user", "tools": ["authorize_payment"],
            "resources": ["order-a"], "arguments_sha256": None},
        "can_delegate": True, "valid_until_sequence": 30}
    payload.update(changes)
    return {"event_id": "grant-" + identity, "sequence": seq, "kind": "grant", "observer": "adapter",
        "authenticated_actor": issuer, "payload": payload}


def action():
    return {"action_id": "pay", "actor": "worker", "principal": "user", "tool": "authorize_payment",
        "resource": "order-a", "arguments": {"amount": "50"}}


def ledger():
    return DelegationLedger(root_principals=["user"], trusted_observers=["adapter"], coverage_complete=True)


def chain():
    return [grant(1, "root", "user", "planner"), grant(2, "child", "planner", "worker", "root")]


def observe(events):
    result = ledger()
    for event in events:
        result.observe(event)
    return result


def assess(obj, bound=None, **changes):
    return obj.assess(action() if bound is None else bound, **{"sequence": 10, "task_id": "task-a", "grant_ids": ["child"], **changes})


def test_valid_delegation_has_bound_path_but_does_not_issue_permission():
    result = assess(observe(chain()))
    assert result.evidence["features"]["support_matched"] == 1
    assert result.evidence["candidates"][0]["path"] == ["root", "child"]
    assert not result.evidence["grants_permission"]


@pytest.mark.parametrize("field,value", [("actor", "other"), ("principal", "other"), ("resource", "order-b"), ("tool", "delete")])
def test_other_actor_principal_resource_tool_cannot_reuse_grant(field, value):
    result = assess(observe(chain()), {**action(), field: value})
    assert result.evidence["features"]["scope_mismatch"] == 1
    assert result.evidence["features"]["support_matched"] == 0


def test_cross_task_and_argument_binding():
    assert assess(observe(chain()), task_id="task-b").evidence["features"]["scope_mismatch"] == 1
    events = chain()
    events[1]["payload"]["scope"]["arguments_sha256"] = digest(action()["arguments"])
    assert assess(observe(events)).evidence["features"]["support_matched"] == 1
    assert assess(observe(events), {**action(), "arguments": {"amount": "500"}}).evidence["features"]["scope_mismatch"] == 1


@pytest.mark.parametrize("change,reason", [("issuer", "issuer_mismatch"), ("scope", "scope_expansion"),
    ("permission", "delegation_not_allowed"), ("expiry", "expired")])
def test_upstream_conditions_change_identical_downstream_action(change, reason):
    events = chain()
    if change == "issuer":
        events[0]["payload"]["recipient"] = "different-planner"
    elif change == "scope":
        events[0]["payload"]["scope"]["resources"] = ["another-order"]
    elif change == "permission":
        events[0]["payload"]["can_delegate"] = False
    else:
        for e in events:
            e["payload"]["valid_until_sequence"] = 3
    result = assess(observe(events))
    assert result.evidence["features"][reason] == 1
    assert result.evidence["features"]["support_matched"] == 0


def test_parent_revocation_propagates_but_unauthenticated_claim_does_not():
    events = chain()
    revoke = {"event_id": "r", "sequence": 4, "kind": "revoke", "observer": "adapter",
        "authenticated_actor": "user", "payload": {"grant_id": "root"}}
    assert assess(observe([*events, revoke])).evidence["features"]["revoked"] == 1
    revoke["observer"] = "parsed-agent-text"
    assert assess(observe([*events, revoke])).evidence["features"]["support_matched"] == 1


def test_missing_unverified_or_incomplete_evidence_is_unresolved_not_denied():
    events = chain()
    for records in ([events[1]], [{**events[0], "observer": "parsed-agent-text"}, events[1]]):
        result = assess(observe(records))
        assert result.evidence["features"]["reference_unresolved"] == 1
        assert result.evidence["features"]["reference_invalid"] == 0
    result = assess(observe(events), {**action(), "principal": None})
    assert result.evidence["features"]["binding_incomplete"] == 1
    assert result.evidence["features"]["reference_unresolved"] == 1


def test_future_evidence_cannot_rewrite_snapshot_and_no_retroactive_parent():
    obj = observe(chain())
    old = assess(obj)
    before = copy.deepcopy(old)
    obj.observe({"event_id": "r", "sequence": 11, "kind": "revoke", "observer": "adapter",
        "authenticated_actor": "user", "payload": {"grant_id": "root"}})
    assert old == before
    with pytest.raises(ValueError, match="prefix"):
        assess(obj)
    events = chain()
    events[0]["sequence"] = 3
    result = assess(observe([events[1], events[0]]))
    assert result.evidence["features"]["issuer_mismatch"] == 1


def test_multiple_references_preserve_valid_and_invalid_alternatives():
    events = chain() + [grant(3, "bad", "user", "other")]
    result = assess(observe(events), grant_ids=["child", "bad"])
    assert result.evidence["features"]["support_matched"] == 1
    assert result.evidence["features"]["reference_invalid"] == 0
    assert result.evidence["diagnostic_features"]["reference_invalid"] == 1
    assert result.evidence["reference_status"] == "supported_reference"


def test_bad_records_do_not_consume_sequence_and_duplicate_events_are_idempotent():
    obj = ledger()
    event = chain()[0]
    obj.observe(event)
    assert obj.observe(event) is False
    bad = copy.deepcopy(chain()[1])
    bad["payload"]["scope"]["tools"] = []
    with pytest.raises(ValueError):
        obj.observe(bad)
    assert obj.sequence == 1
    obj.observe(chain()[1])
    assert assess(obj).evidence["features"]["support_matched"] == 1
