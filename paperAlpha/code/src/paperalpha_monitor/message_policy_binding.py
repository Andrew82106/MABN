"""Bind explicit message targets to existing shared policy proofs.

No actor, recipient, resource, permission or target set is inferred here.
The event predictor and policy graph keep different clocks and shared facts.
"""
import copy

from paperalpha_runtime.canonical import digest
from .observable_event_dag import DAGWorkflowInput


def bind_message_policy(dag, *, authority, targets, cutoff, horizon, combinations=()):
    """Connect named future appearances and exact per-recipient action bindings.

    A target contains event_id, binding and field_literals. The binding must
    already be covered by the supplied independently built authority graph.
    This constructor does not mint authority from task-use cues or role names.
    """
    dag.validate(); authority.validate()
    if not isinstance(targets, list) or not targets:
        raise ValueError("explicit nonempty message target list required")
    proposals, queries = [], {}
    for target in copy.deepcopy(targets):
        if not isinstance(target, dict) or set(target) != {"event_id", "binding", "field_literals"}:
            raise ValueError("strict event, binding and field-alias target required")
        event = dag.lookup.get(target["event_id"])
        binding = target["binding"]
        if event is None or event["semantics"] != "literal_appearance" or not isinstance(binding, dict):
            raise ValueError("message target must identify a real literal event")
        aid = binding.get("action_id")
        if not isinstance(aid, str) or not aid or aid in queries:
            raise ValueError("unique action identity required")
        proposals.append({"action_id": aid, "node": event["node"], "binding": binding,
            "family": "release", "visible": None, "quantity": None,
            "evidence": {"event_dag_sha256": dag.sha256, "event_id": event["id"],
                "available_journal_sequence": dag.prefix_sequence, "prefix_journal_sequence": dag.prefix_sequence}})
        queries[aid] = {"event_id": event["id"], "binding_sha256": digest(binding), "field_literals": target["field_literals"]}
    return DAGWorkflowInput(event_dag=dag, queries=queries, workflow=dag.workflow, authority=authority,
        proposals=proposals, combinations=combinations, cutoff=cutoff, horizon=horizon,
        event_semantics={a: "literal_appearance" for a in queries})
