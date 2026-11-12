"""Project already verified branch PREFIXES, never outcomes, into event DAGs.

This is one real-data adapter for the generic graph model, not automatic role
inference. Other workflows need their own source-bound extraction contracts.
"""
import copy
import json

from paperalpha_runtime.canonical import digest
from .branch_privacy_input import BranchPrivacyInput
from .branch_workflow_risk import BranchWorkflowInput
from .observable_event_dag import ObservableEventDAG, DAGWorkflowInput


def from_verified_branch(branch):
    if not isinstance(branch, BranchPrivacyInput):
        raise ValueError("verified branch prefix required")
    branch.validate()
    previous = BranchWorkflowInput.from_branch(branch)
    seq = branch.prefix_journal_sequence
    raw = branch.observation
    def evidence(identity, text):
        return {"record_id": identity, "sequence": seq, "source": "verified-captured-branch-prefix", "text": text}
    events, edges, observations, queries = [], [], {}, {}
    registry = raw["pooled_scope"]["tracked_fields"]
    for j, proposal in enumerate(previous.proposals):
        names = proposal["binding"]["arguments"]["field_names"]
        literal = next(f["literal"] for f in registry if set(f["field_names"]) == set(names))
        for node in previous.workflow["nodes"]:
            identity = f"field-{j}:{node}"
            family = "prepare" if node == "planner" else "publish" if node == "publisher" else "work"
            context = json.dumps(raw["publisher_payload"], ensure_ascii=False, sort_keys=True) if node == "publisher" else "No support/opposition classification used for this completed stage."
            events.append({"id": identity, "node": node, "actor": node, "family": family, "semantics": "literal_appearance", "subject": literal,
                "cues": branch.release.visible_features[j].astype(int).tolist() if node == "publisher" else [0, 0],
                "cue_evidence": evidence("context:" + identity, context)})
            if node != "publisher":
                text = raw["pooled_scope"]["coordinator_text"] if node == "planner" else raw["worker_messages"][node]
                observations[identity] = {**evidence("message:" + node, text), "kind": "complete_message", "value": int(literal.casefold() in text.casefold())}
        for a, b in previous.workflow["edges"]:
            edges.append({"source": f"field-{j}:{a}", "target": f"field-{j}:{b}", "basis": "observed_delivery",
                "evidence": evidence(f"delivery:{a}:{b}", "Verified fixed workflow delivery; publisher payload retains each actual completed worker message.")})
        queries[proposal["action_id"]] = {"event_id": f"field-{j}:publisher", "binding_sha256": digest(proposal["binding"]), "field_literals": {name: literal for name in names}}
    dag = ObservableEventDAG(workflow=previous.workflow, events=events, edges=edges, observations=observations, prefix_sequence=seq)
    proposals = copy.deepcopy(previous.proposals)
    for p in proposals:
        p["evidence"] = {"event_dag_sha256": dag.sha256, "event_id": queries[p["action_id"]]["event_id"],
                         "available_journal_sequence": seq, "prefix_journal_sequence": seq}
    return DAGWorkflowInput(event_dag=dag, queries=queries, workflow=previous.workflow, authority=previous.authority,
        proposals=proposals, contracts=previous.contracts, combinations=previous.combinations,
        cutoff=previous.cutoff, horizon=previous.horizon, event_semantics=previous.event_semantics)
