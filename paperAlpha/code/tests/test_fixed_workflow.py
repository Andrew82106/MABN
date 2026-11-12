import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.fixed_workflow import (topological_order, public_daily_life_workflow,
    node_instructions, execute_workflow)


def agents(insurance=False):
    roles = ["concierge_coordinator", "booking_specialist", "vendor_liaison", "payments_and_coverage_specialist"]
    if insurance:
        roles.append("insurance_coverage_analyst")
    return [{"role": role, "description": "Declared duty.", "system_prompt": "Do your own part."} for role in roles]


@pytest.mark.parametrize("insurance", [False, True])
def test_declared_template_is_acyclic_and_only_uses_public_roster(insurance):
    workflow = public_daily_life_workflow(agents(insurance))
    order = topological_order(workflow)
    assert len(order) == 4 + int(insurance)
    assert order[0] == "booking_specialist" and order[-1] == "concierge_coordinator"
    assert all(order.index(u) < order.index(v) for u, v in workflow["edges"])
    assert set(workflow["roles"]) == set(order)


@pytest.mark.parametrize("edges", [[["A", "A"]], [["A", "B"], ["B", "A"]],
                                  [["A", "B"], ["A", "B"]], [["A", "missing"]]])
def test_invalid_graphs_are_not_repaired_or_silently_made_acyclic(edges):
    with pytest.raises(ValueError):
        topological_order({"nodes": ["A", "B"], "edges": edges})


@pytest.mark.parametrize("nodes", [[], ["A", "A"], ["A", None], ["A"] * 13])
def test_invalid_nodes_rejected(nodes):
    with pytest.raises(ValueError):
        topological_order({"nodes": nodes, "edges": []})


def test_hidden_policies_and_unsupported_roles_cannot_become_a_workflow():
    roster = agents()
    roster[0]["useful_tools"] = ["secret policy"]
    with pytest.raises(ValueError, match="public"):
        public_daily_life_workflow(roster)
    roster = agents() + [{"role": "unknown", "description": "extra", "system_prompt": ""}]
    with pytest.raises(ValueError, match="unsupported"):
        public_daily_life_workflow(roster)


def test_only_declared_predecessors_are_delivered_once_not_serial_siblings():
    workflow = {"nodes": ["A", "B", "C", "D"], "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]]}
    before = copy.deepcopy(workflow)
    inputs, messages = {}, []
    async def run(node, content):
        inputs[node] = json.loads(content)
        return node + " output"
    outputs = asyncio.run(execute_workflow(workflow, "original", run, lambda *parts: messages.append(parts)))
    assert workflow == before and list(outputs) == ["A", "B", "C", "D"]
    assert inputs["A"]["predecessor_reports"] == []
    assert inputs["C"]["predecessor_reports"] == [{"sender": "A", "content": "A output"}]
    assert inputs["D"]["predecessor_reports"] == [{"sender": "B", "content": "B output"}, {"sender": "C", "content": "C output"}]
    assert set(messages) == {(u, v, u + " output") for u, v in workflow["edges"]}
    assert all(value["original_task"] == "original" for value in inputs.values())


@pytest.mark.parametrize("bad_result", ["", " ", None])
def test_missing_outputs_stop_downstream_and_are_never_fabricated(bad_result):
    executed, messages = [], []
    async def run(node, content):
        executed.append(node)
        return bad_result
    with pytest.raises(ValueError, match="nonempty"):
        asyncio.run(execute_workflow({"nodes": ["A", "B"], "edges": [["A", "B"]]}, "goal", run,
                                    lambda *parts: messages.append(parts)))
    assert executed == ["A"] and not messages


def test_node_failure_does_not_retry_or_run_downstream():
    executed = []
    async def run(node, content):
        executed.append(node)
        raise RuntimeError("deliberate")
    with pytest.raises(RuntimeError, match="deliberate"):
        asyncio.run(execute_workflow({"nodes": ["A", "B"], "edges": [["A", "B"]]}, "goal", run, lambda *parts: None))
    assert executed == ["A"]


def test_prompts_keep_assigned_duty_without_exposing_policy_metadata():
    prompt = node_instructions(agents()[1])
    assert "Do your own part." in prompt and "untrusted evidence" in prompt
    assert "useful_tools" not in prompt and "allowed_args" not in prompt


def test_collector_imports_in_existing_sdk_environment_without_numpy():
    sdk = ROOT / '.local/environments/collection/harnessaudit_oai_20260919/Scripts/python.exe'
    if not sdk.exists():
        pytest.skip("optional SDK runtime unavailable")
    check = subprocess.run([str(sdk), "-X", "utf8", "-B", str(ROOT / 'code/commands/collect_fixed_dag_harnessaudit.py'), "--help"],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
