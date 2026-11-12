"""Constructed mixed appearance/tool events; NEVER a real API mixed trace."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from branch_publisher_fixtures import fixture as branch_fixture
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput
from paperalpha_monitor.branch_workflow_risk import BranchWorkflowInput
from paperalpha_monitor.delegation_evidence import DelegationLedger
from paperalpha_runtime.canonical import digest


def combination(branch, identity="literal-pair"):
    return {"id": identity, "task_id": branch.release.context.task_id, "principal": branch.release.context.principal,
        "bindings": {aid: digest(branch.release.graph.proofs[aid].evidence["action_binding"]) for aid in branch.release.action_ids},
        "source": "constructed co-release prohibition; not an observed policy", "sequence": 0}


def fixture(*, current="root_unknown", visible="supported", shared=True, payment=False, prohibit=True,
            w=((1, 0), (0, 1)), kinds=("canary", "number")):
    branch = branch_fixture(current=current, visible=visible, shared=shared, w=w, kinds=kinds)
    if not payment:
        return BranchWorkflowInput.from_branch(branch, combinations=[combination(branch)] if prohibit else [])
    # Add a declared simulated grant before either release proof. Do not alter
    # actual development records or claim this third actor was present there.
    raw = copy.deepcopy(branch.observation)
    scope = raw["pooled_scope"]
    record = {"event_id": "fixture-payment-root", "sequence": 6, "kind": "grant", "observer": scope["trusted_observers"][0],
        "authenticated_actor": "owner", "payload": {"grant_id": "payment-root", "parent_id": None, "issuer": "owner", "recipient": "cashier",
        "scope": {"task_id": scope["context"]["task_id"], "principal": "owner", "tools": ["authorize_payment"], "resources": ["wallet"], "arguments_sha256": None},
        "can_delegate": False, "valid_until_sequence": 30}}
    scope["authority_records"].append(record)
    scope["complete_grant_ids"].append("payment-root")
    branch = BranchPrivacyInput(raw, available_journal_sequence=20, prefix_journal_sequence=20)
    linked = BranchWorkflowInput.from_branch(branch, combinations=[combination(branch)] if prohibit else [])
    ledger = DelegationLedger(root_principals=scope["root_principals"], trusted_observers=scope["trusted_observers"], coverage_complete=scope["coverage_complete"])
    for record in scope["authority_records"]:
        ledger.observe(record)
    graph = copy.deepcopy(linked.authority)
    action = {"action_id": "payment", "actor": "cashier", "principal": "owner", "tool": "authorize_payment", "resource": "wallet",
              "arguments": {"amount": "60", "currency": "USD"}}
    graph.add(ledger.assess(action, sequence=12, task_id=scope["context"]["task_id"], grant_ids=["payment-root"]))
    workflow = copy.deepcopy(linked.workflow)
    workflow["nodes"].append("cashier")
    workflow["actors"]["cashier"] = ["cashier"]
    workflow["edges"].append(["planner", "cashier"])
    proposal = {"action_id": "payment", "node": "cashier", "binding": action, "family": "payment", "visible": "support",
        "evidence": {"record_id": "fixture-payment-cue", "sequence": 12, "source": "constructed-prefix", "text": "Constructed payment support cue"},
        "quantity": {"amount_field": "amount", "currency_field": "currency", "single_limit": "100", "limit_source": "constructed-policy", "limit_sequence": 0}}
    budget = {"id": "mixed-budget", "task_id": scope["context"]["task_id"], "principal": "owner", "resources": ["wallet"],
        "currency": "USD", "limit": "100", "prior_requested": "50", "source": "constructed-policy", "sequence": 0, "history_complete": True}
    return BranchWorkflowInput(branch=branch, workflow=workflow, authority=graph, proposals=[*linked.proposals, proposal],
        contracts=[budget], combinations=linked.combinations, cutoff=12, horizon=30,
        event_semantics={**linked.event_semantics, "payment": "tool_dispatch"})
