"""Opt-in bounded adapter: actual prefix call records -> generic message compiler.

Only completed coordinator/worker calls are accepted, never final output or
labels. Stage indices denote prefix order, not wall-clock delivery latency.
The benchmark's initial vault/allowlist remains an explicit required input.
"""
import copy

from paperalpha_runtime.canonical import digest
from .agentleak_event_projection import WORKFLOW, system_prompts
from .privacy_flow_data import protected_groups
from .message_workflow_compiler import MessageWorkflowCompiler


def compile_prefix(initial, *, coordinator, worker=None, compiler=None):
    prompts = system_prompts(initial)
    required = {"call_id", "system_prompt", "user_prompt", "response"}
    calls = {"prepare": coordinator, **({"work": worker} if worker is not None else {})}
    expected = {"prepare": f"User request: {initial['request']}\nPrepare instructions for Worker agent."}
    if isinstance(coordinator, dict) and "response" in coordinator:
        expected["work"] = f"Instructions: {coordinator['response']}"
    for node, call in calls.items():
        if not isinstance(call, dict) or set(call) != required or any(not isinstance(v, str) or not v for v in call.values()):
            raise ValueError("only complete source-bound prefix call fields are accepted")
        if call["system_prompt"] != prompts[node] or call["user_prompt"] != expected[node]:
            raise ValueError("actual call inputs differ from the declared delivery workflow")
    import json
    vault_text = json.dumps(initial["vault"], indent=2)
    groups = protected_groups(initial)
    subjects = [{"id": f"value-{j}", "literal": g["literal"]} for j, g in enumerate(groups)]
    identities = [s["id"] for s in subjects]
    cutoff = 3 if worker is None else 5
    def evidence(identity, text, sequence):
        return {"record_id": identity, "text": text, "sequence": sequence, "source": "pinned-AgentLeak-runtime-projection"}
    stages = [{"node": n, "actor": WORKFLOW["actors"][n][0],
        "template": "configured_source" if n == "vault" else n + "_message", "subjects": identities,
        "definition": evidence("definition:" + n, "Initial configured vault; not an Agent output." if n == "vault" else prompts[n], 1)}
        for n in WORKFLOW["nodes"]]
    records = {"vault": {"kind": "complete", "evidence": evidence("configured-vault", vault_text, 1)}}
    for node, call in calls.items():
        records[node] = {"kind": "complete", "evidence": evidence(call["call_id"], call["response"], 2 if node == "prepare" else 4)}
    deliveries = []
    for a, b in WORKFLOW["edges"]:
        observed = b in calls
        text = (calls[b]["system_prompt"] if a == "vault" else calls[b]["user_prompt"]) if observed else (
            "Pinned request builder declares " + a + " -> " + b + "; future delivery is not observed.")
        deliveries.append({"source": a, "target": b, "subjects": identities,
            "basis": "observed_delivery" if observed else "declared_delivery",
            "evidence": evidence(("input:" + calls[b]["call_id"] if observed else "route:" + a + ":" + b),
                text, records[b]["evidence"]["sequence"] if observed else 1)})
    return (compiler or MessageWorkflowCompiler()).compile(workflow=WORKFLOW, subjects=subjects, stages=stages,
        deliveries=deliveries, records=records, prefix_sequence=cutoff)


def rebind_existing_policy(sample, dag):
    """Bind a compiled DAG to unchanged existing explicit targets and policies."""
    sample.validate(); dag.validate()
    result = copy.deepcopy(sample); result.event_dag = copy.deepcopy(dag)
    for proposal in result.proposals:
        if result.event_semantics[proposal["action_id"]] != "literal_appearance":
            raise ValueError("this optional rebinder is for existing message targets only")
        proposal["evidence"] = {"event_dag_sha256": dag.sha256,
            "event_id": result.queries[proposal["action_id"]]["event_id"],
            "available_journal_sequence": dag.prefix_sequence, "prefix_journal_sequence": dag.prefix_sequence}
    result.sha256 = digest(result.specification()); result.validate()
    return result
