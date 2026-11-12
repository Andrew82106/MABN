import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
from verify_fixed_dag_acquisition import check_execution
from paperalpha_runtime.fixed_workflow import public_daily_life_workflow, node_input, node_instructions
from test_fixed_workflow import agents


def fixture():
    roster = agents()
    workflow = public_daily_life_workflow(roster)
    selected = {"workflow": workflow, "public_configuration": {"goal": "goal", "agents": roster}}
    manifest = {"status": "completed", "model_requested": "example"}
    journal = [{"kind": "run_started", "payload": {"declared_workflow": workflow,
                "public_configuration": selected["public_configuration"]}}]
    rows, requests, outputs = [], [], {}
    actors = {a["role"]: a for a in roster}
    for call, node in enumerate(workflow["nodes"], 1):
        parents = [u for u, v in workflow["edges"] if v == node]
        content = node_input("goal", [{"sender": u, "content": outputs[u]} for u in parents])
        prompt = node_instructions(actors[node])
        journal += [{"kind": "communication", "payload": {"actor": u, "target": node, "content": outputs[u]}} for u in parents]
        rows.append({"stage": "started", "node": node, "first_possible_call": call, "input": content, "instructions": prompt})
        requests.append({"call": call, "input": content, "system_instructions": prompt, "model_requested": "example"})
        outputs[node] = node + " report"
        rows.append({"stage": "completed", "node": node, "calls_started": 1, "last_call": call, "output": outputs[node]})
    journal.append({"kind": "communication", "payload": {"actor": "concierge_coordinator", "target": "user",
                   "content": outputs["concierge_coordinator"]}})
    return selected, manifest, journal, rows, outputs, requests


def test_verifier_checks_the_actual_requests_and_deliveries():
    result = check_execution(*fixture())
    assert result["completed_nodes"] == 4 and result["delivered_internal_edges"] == 6


@pytest.mark.parametrize("change", [
    lambda x: x[3].__setitem__(slice(0, 4), [x[3][0], x[3][2], x[3][1], x[3][3]]),
    lambda x: x[3][0].update(input="future report"),
    lambda x: x[5][0].update(input="not the actual planned input"),
    lambda x: x[5][1].update(call=1),
    lambda x: x[5][0].update(system_instructions="different role"),
    lambda x: x[5][0].update(model_requested="changed"),
    lambda x: x[2].insert(1, copy.deepcopy(x[2][-1])),
    lambda x: x[2][1]["payload"].update(content="edited in transit"),
    lambda x: x[3][1].update(calls_started=0),
    lambda x: x[4].update(extra="fabricated output"),
])
def test_audit_does_not_accept_tampered_or_unassigned_execution(change):
    data = list(fixture())
    change(data)
    with pytest.raises(ValueError):
        check_execution(*data)


def test_failed_first_node_with_no_available_api_budget_is_preserved_as_failure():
    selected, manifest, journal, rows, outputs, requests = fixture()
    manifest["status"] = "failed"
    end = {"stage": "failed", "node": rows[0]["node"], "calls_started": 0, "last_call": 0}
    result = check_execution(selected, manifest, journal[:1], [rows[0], end], {}, [])
    assert result["started_nodes"] == 1 and result["completed_nodes"] == 0
