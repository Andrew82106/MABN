import asyncio
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.stage_workflow import (validate_stage_workflow, daily_life_relay_stages,
    execute_stages, StageTextWindow)
from paperalpha_monitor.staged_knowledge_bn import StageKnowledgeCompiler, compile_stage_journal
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN
from paperalpha_monitor.tool_capabilities import load_tool_registry
from test_fixed_workflow import agents
from test_knowledge_scoped_bn import independent_enumeration


def workflow():
    return {"schema": "stage-workflow-v1", "context_mode": "fresh_per_stage",
        "nodes": ["s:first", "s:work", "s:final"], "edges": [["s:first", "s:work"], ["s:work", "s:final"]],
        "owners": {"s:first": "hub", "s:work": "worker", "s:final": "hub"},
        "roles": {"s:first": "reasoner", "s:work": "executor", "s:final": "reviewer"},
        "purposes": {"s:first": "plan", "s:work": "execute", "s:final": "review"}}


def journal():
    result = []
    def emit(kind, **payload):
        result.append({"schema_version": "runtime-journal-v1", "sequence": len(result) + 1,
                       "kind": kind, "payload": payload})
    emit("run_started", agents=["hub", "worker"], external_endpoints=["user"], declared_stage_workflow=workflow())
    def tool(actor, stage, identity):
        binding = dict(actor=actor, stage_id=stage, invocation_id=identity,
                       provider_call_id="provider-" + identity, tool="get_customer_context")
        emit("tool_handler_started", **binding, arguments={"customer_id": "C"})
        emit("tool_handler_returned", **binding, result="result-" + identity)
    tool("hub", "s:first", "one")
    emit("communication", actor="hub", target="worker", stage_id="s:first", target_stage_id="s:work", content="handoff")
    tool("worker", "s:work", "two")
    emit("communication", actor="worker", target="hub", stage_id="s:work", target_stage_id="s:final", content="work report")
    tool("hub", "s:final", "three")
    emit("communication", actor="hub", target="user", stage_id="s:final", content="final report")
    emit("run_finished", status="completed")
    return result


def context():
    return dict(stage_workflow=workflow(), tool_registry=load_tool_registry())


@pytest.mark.parametrize("insurance", [False, True])
def test_relay_template_separates_repeated_owner_from_acyclic_functional_stages(insurance):
    w = daily_life_relay_stages(agents(insurance))
    order = validate_stage_workflow(w)
    assert len(order) == 6 + int(insurance)
    hub = "concierge_coordinator"
    assert list(w["owners"].values()).count(hub) == 3
    assert all(hub in (w["owners"][u], w["owners"][v]) for u, v in w["edges"])
    assert all(order.index(u) < order.index(v) for u, v in w["edges"])
    assert "permissions" not in w


@pytest.mark.parametrize("mutate", [
    lambda w: w.update(context_mode="shared_memory"),
    lambda w: w["owners"].update({"s:final": "s:first"}),
    lambda w: w["owners"].pop("s:work"),
    lambda w: w["edges"].append(["s:final", "s:first"]),
    lambda w: w.update(allowed_edges=[["s:first", "s:work"]]),
])
def test_unsupported_or_confounded_stage_specifications_fail(mutate):
    w = workflow()
    mutate(w)
    with pytest.raises(ValueError):
        validate_stage_workflow(w)


def test_runner_delivers_only_exact_predecessor_outputs_with_both_identities():
    inputs, messages, calls = {}, [], []
    async def execute(stage, actor, content):
        calls.append((stage, actor))
        inputs[stage] = json.loads(content)
        return stage + " report"
    outputs = asyncio.run(execute_stages(workflow(), "goal", execute, messages.append))
    assert calls == [(s, workflow()["owners"][s]) for s in workflow()["nodes"]]
    assert len(outputs) == 3 and len(messages) == 2
    assert inputs["s:final"]["predecessor_reports"] == [{"actor": "worker", "target": "hub",
        "stage_id": "s:work", "target_stage_id": "s:final", "content": "s:work report"}]
    assert "s:first report" not in json.dumps(inputs["s:final"])


def test_runner_failure_stops_without_retry_or_fake_completion():
    calls, messages = [], []
    async def execute(stage, actor, content):
        calls.append(stage)
        if stage == "s:work":
            raise RuntimeError("record this failure")
        return "first report"
    with pytest.raises(RuntimeError):
        asyncio.run(execute_stages(workflow(), "goal", execute, messages.append))
    assert calls == ["s:first", "s:work"] and len(messages) == 1


def test_stage_nodes_and_roles_are_distinct_but_action_actors_and_scope_stay_physical():
    trace = compile_stage_journal(journal(), **context())
    explained = KnowledgeScopedBN().explain(trace)
    assert [v["id"] for v in trace.variables] == workflow()["nodes"]
    actions = explained["actions"]
    assert [a["node"] for a in actions] == workflow()["nodes"]
    assert [a["binding"]["actor"] for a in actions] == ["hub", "worker", "hub"]
    assert [a["role"] for a in actions] == ["reasoner", "executor", "reviewer"]
    assert all(a["permission_status"] == "unknown" and a["binding"]["principal"] is None for a in actions)
    assert actions[0]["authorization_parameter"] != actions[2]["authorization_parameter"]


def test_context_does_not_leak_across_two_fresh_stages_of_the_same_owner():
    window, packets = StageTextWindow(), {}
    for row in journal():
        packet = window.push(row)
        if packet:
            packets[row["sequence"]] = packet
    # Final hub tool return may reference only the worker report delivered to
    # this stage, not the hub's own first-stage tool result or outbound report.
    final = packets[9]
    assert [p["record_id"] for p in final["prior_records"]] == ["journal-7"]
    assert final["current"]["actor"] == "hub" and final["current"]["stage_id"] == "s:final"


@pytest.mark.parametrize("index,mutation", [
    (0, lambda p: p.pop("declared_stage_workflow")),
    (1, lambda p: p.pop("stage_id")),
    (1, lambda p: p.update(stage_id="s:work")),
    (2, lambda p: p.update(stage_id="s:final")),
    (3, lambda p: p.update(target_stage_id="s:first")),
    (9, lambda p: p.update(target_stage_id="s:final")),
])
def test_stage_binding_errors_do_not_mutate_the_accepted_prefix(index, mutation):
    rows = journal()
    compiler = StageKnowledgeCompiler(**context())
    for row in rows[:index]:
        compiler.push(row)
    before = compiler.projector.sequence
    bad = copy.deepcopy(rows[index])
    mutation(bad["payload"])
    with pytest.raises(ValueError):
        compiler.push(bad)
    assert compiler.projector.sequence == before
    compiler.push(rows[index])


def test_same_actor_delivery_uses_distinct_source_target_stages_and_retains_deviation():
    rows = journal()[:4]
    # This is deliberately off the nominal graph, but is still an actual
    # observation. It must not disappear merely because the graph forbids it.
    rows[3]["payload"].update(target="hub", target_stage_id="s:final")
    trace = compile_stage_journal(rows, **context())
    op = trace.steps[-1].base.operations[0]
    assert op.sender != op.receiver
    assert trace.variables[op.sender]["id"] == "s:first"
    assert trace.variables[op.receiver]["id"] == "s:final"
    assert op.evidence["actor"] == op.evidence["target"] == "hub"
    assert op.evidence["expected"] == "unexpected"


def test_new_stage_does_not_implicitly_inherit_same_owner_process_state():
    trace = compile_stage_journal(journal()[:3], **context())
    info = {v["id"]: v["marginal"] for v in KnowledgeScopedBN().explain(trace)["variables"]}
    assert info["s:first"] > 0 and info["s:final"] == 0


def test_incremental_batch_and_independent_full_state_enumeration_agree():
    model = KnowledgeScopedBN()
    model.theta += np.random.default_rng(98).normal(0, .4, len(model.theta))
    rows, trace = journal(), compile_stage_journal(journal(), **context())
    compiler, engine = StageKnowledgeCompiler(**context()), model._engine(model.theta)
    scores = []
    for row in rows:
        step = compiler.push(row)
        if step:
            scores.append(engine.apply(step, compiler.variables))
    expected, _ = independent_enumeration(trace, model.theta, model.registry)
    np.testing.assert_allclose(scores, model.predict([trace])[0], atol=2e-15, rtol=0)
    np.testing.assert_allclose(scores, expected, atol=2e-15, rtol=0)


def test_all_parameter_gradients_still_train_through_stage_communication_and_actions():
    model = KnowledgeScopedBN()
    theta = model.theta + np.random.default_rng(42).normal(0, .4, len(model.theta))
    trace = compile_stage_journal(journal(), **context())
    _, analytic = model.episode_loss_gradient(theta, trace, 1)
    numeric = approx_derivative(lambda t: model.episode_loss_gradient(t, trace, 1)[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=3e-8, rtol=3e-5)


def test_appending_future_stages_cannot_change_existing_prefix_predictions():
    model = KnowledgeScopedBN()
    full = model.predict([compile_stage_journal(journal(), **context())])[0]
    for end in range(2, len(journal()) + 1):
        prefix = model.predict([compile_stage_journal(journal()[:end], **context())])[0]
        np.testing.assert_array_equal(prefix, full[:len(prefix)])
