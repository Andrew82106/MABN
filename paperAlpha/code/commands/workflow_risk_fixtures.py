"""Constructed policy/graph fixtures only: never real API observations."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.delegation_evidence import DelegationLedger
from paperalpha_monitor.shared_authority_graph import AuthorityWindow, AuthorityProofGraph
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskInput
from paperalpha_runtime.canonical import digest


def fixture(*, families=("payment", "payment"), visible=("support", "opposition"),
            shared=True, authority="root", amounts=("60", "60"), single_limit="100",
            budget="100", prior="0", combination=False, layout="fork", rename="", assessment_start=10):
    n = len(families)
    if len(visible) != n or len(amounts) != n:
        raise ValueError("aligned fixture dimensions")
    actors = [rename + f"worker-{i}" for i in range(n)]
    nodes = [rename + f"step-{i}" for i in range(n)]
    plan, end = rename + "plan", rename + "end"
    chain = [plan, *nodes, end]
    if layout == "chain":
        edges = [[a, b] for a, b in zip(chain, chain[1:])]
    elif layout == "fork":
        edges = [[plan, node] for node in nodes] + [[node, end] for node in nodes]
    else:
        raise ValueError("explicit fixture topology")
    workflow = {"nodes": chain, "edges": edges, "actors": {plan: [rename + "planner"], end: [rename + "publisher"], **dict(zip(nodes, [[a] for a in actors]))}}
    ledger = DelegationLedger(root_principals=["user"], trusted_observers=["fixture-adapter"], coverage_complete=authority in {"known", "invalid"})
    resources = [f"resource-{i}" for i in range(n)]
    tools = ["tool:" + f for f in families]
    def grant(seq, identity, issuer, recipient, parent=None):
        return {"event_id": "grant:" + identity, "sequence": seq, "kind": "grant", "observer": "fixture-adapter", "authenticated_actor": issuer,
            "payload": {"grant_id": identity, "parent_id": parent, "issuer": issuer, "recipient": recipient,
            "scope": {"task_id": "fixture-task", "principal": "user", "tools": sorted(set(tools)), "resources": resources, "arguments_sha256": None},
            "can_delegate": parent is None, "valid_until_sequence": 100}}
    roots = [f"root-{i}" for i in range(n)]
    children = [f"child-{i}" for i in range(n)]
    records = [grant(i + 1, root, "user", "planner") for i, root in enumerate(roots)]
    records += [grant(n + i + 1, child, "planner", actor, roots[0] if shared else roots[i]) for i, (child, actor) in enumerate(zip(children, actors))]
    for record in records:
        if authority != "opaque" or record["payload"]["parent_id"] is not None:
            ledger.observe(record)
    if authority == "invalid":
        for i, root in enumerate(roots):
            ledger.observe({"event_id": "revoke:" + root, "sequence": 2 * n + i + 1, "kind": "revoke", "observer": "fixture-adapter", "authenticated_actor": "user", "payload": {"grant_id": root}})
    complete = {"root": tuple(children), "delegation": tuple(roots), "opaque": (), "known": (), "invalid": ()}[authority]
    graph = AuthorityProofGraph(AuthorityWindow("fixture-static", 0, 100, complete))
    proposals = []
    for i, (actor, node, family, cue) in enumerate(zip(actors, nodes, families, visible)):
        args = {"amount": amounts[i], "currency": "USD"} if family == "payment" else {"record": f"record-{i}"}
        action = {"action_id": f"a-{i}", "actor": actor, "principal": "user", "tool": tools[i], "resource": resources[i], "arguments": args}
        graph.add(ledger.assess(action, sequence=assessment_start + i, task_id="fixture-task", grant_ids=[children[i]]))
        proposals.append({"action_id": action["action_id"], "node": node, "binding": action, "family": family, "visible": cue,
            "evidence": {"record_id": f"fixture-cue-{i}", "sequence": assessment_start + i, "source": "constructed-mechanism-fixture", "text": "Constructed visible cue: " + cue},
            "quantity": {"amount_field": "amount", "currency_field": "currency", "single_limit": single_limit,
                         "limit_source": None if single_limit is None else "fixture-policy", "limit_sequence": None if single_limit is None else 0} if family == "payment" else None})
    contracts = [] if budget is None else [{"id": "budget", "task_id": "fixture-task", "principal": "user", "resources": resources,
        "currency": "USD", "limit": budget, "prior_requested": prior, "source": "constructed-policy", "sequence": 0, "history_complete": True}]
    combinations = [] if not combination else [{"id": "pair", "task_id": "fixture-task", "principal": "user",
        "bindings": {p["action_id"]: digest(p["binding"]) for p in proposals}, "source": "constructed-policy", "sequence": 0}]
    cutoff = max(20, assessment_start + n)
    return WorkflowRiskInput(workflow=workflow, authority=graph, proposals=proposals, contracts=contracts, combinations=combinations, cutoff=cutoff, horizon=cutoff + 5)


def rebuild(sample, **changes):
    args = dict(workflow=sample.workflow, authority=sample.authority, proposals=sample.proposals, contracts=sample.contracts,
                combinations=sample.combinations, cutoff=sample.cutoff, horizon=sample.horizon)
    args.update(changes)
    return WorkflowRiskInput(**args)
