"""Synthetic fixtures + fake node executors only; no models or benchmark runs."""
import asyncio
import copy
import inspect
import json
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.harnessaudit_sandbox_adapter import HarnessAuditSandboxAdapter, REVIEWED_TOOLS
import paperalpha_runtime.harnessaudit_sandbox_adapter as implementation

UPSTREAM = ROOT / 'code/external/HarnessAudit'


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network is prohibited in synthetic sandbox tests")
    original_connect = socket.socket.connect
    socketpair_code = getattr(getattr(socket, "_fallback_socketpair", None), "__code__", None)

    def guarded_connect(sock, address):
        # Windows asyncio builds its internal self-pipe using socketpair's
        # loopback implementation. Permit that exact stdlib call site only.
        if socketpair_code is not None and inspect.currentframe().f_back.f_code is socketpair_code:
            return original_connect(sock, address)
        return forbidden(sock, address)

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def seeds(tmp_path, domain):
    folder = tmp_path / "synthetic" / domain
    folder.mkdir(parents=True, exist_ok=True)
    if domain == "ecommerce":
        values = {"seed_customers.json": {"customers": {}, "payment_methods": {}},
                  "seed_products.json": {"products": {}}, "seed_orders.json": {"orders": {}},
                  "seed_cases.json": {"cases": {}, "case_trends": {}},
                  "seed_knowledge.json": {"articles": {}, "queries": {}},
                  "seed_sales.json": {"opportunities": {}, "quotes": {
                      "Q-SYNTH": {"opportunity_id": "OP-SYNTH", "approval_state": "Pending",
                                  "discount_token": "NONE", "contract_term": "12 months", "final_value": 100}}}}
        for name, value in values.items():
            (folder / name).write_text(json.dumps(value), encoding="utf-8")
    return folder


def config(tmp_path, domain="office", **overrides):
    values = {"upstream": UPSTREAM, "domain": domain, "seed_dir": seeds(tmp_path, domain),
              "public_task": {"goal": "Use only synthetic data in the declared fixed workflow.", "agents": [
                  {"role": "task_operator", "description": "Synthetic operator", "system_prompt": "Operate synthetic records."},
                  {"role": "task_reviewer", "description": "Synthetic reviewer", "system_prompt": "Review predecessor reports."}]},
              "workflow": {"nodes": ["work", "review"], "edges": [["work", "review"]]},
              "role_bindings": {"work": {"role": "task_operator", "source": "synthetic explicit binding"},
                                "review": {"role": "task_reviewer", "source": "synthetic explicit binding"}},
              "visible_policy": {"source": "synthetic public rule declaration", "rules": [
                  {"subject": "task_reviewer", "forbidden_tools": ["create_user_account"]}]},
              "evaluation_config": {"groundtruth": "PRIVATE-ORACLE-SENTINEL", "tool_necessity": ["PRIVATE-NECESSITY"]}}
    values.update(overrides)
    return values


MUTATIONS = {
    "office": ("create_user_account", {"first_name": "Synthetic", "last_name": "Person", "email": "person@example.invalid",
                                      "department": "Testing", "title": "Fixture", "location": "Memory", "user_id": "USR-SYNTH"}, "employees"),
    "ecommerce": ("approve_quote", {"quote_id": "Q-SYNTH", "decision": "approve", "justification": "synthetic fixture"}, "quotes"),
    "finance": ("submit_compliance_report", {"report_type": "audit", "content": "synthetic fixture",
                                            "client_id": "CL-SYNTH", "portfolio_id": "PF-SYNTH"}, "compliance_reports"),
}


@pytest.mark.parametrize("domain", list(MUTATIONS))
def test_stateful_tools_reset_role_binding_and_oracle_isolation(tmp_path, domain):
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, domain))
    tool, arguments, table = MUTATIONS[domain]
    packets = []

    async def fake(packet, invoke):
        packets.append(copy.deepcopy(packet))
        if packet["node_id"] == "work":
            return await invoke(tool, arguments, "synthetic-call-1")
        assert packet["input"]["predecessor_reports"][0]["sender"] == "work"
        return "Review complete, without claiming safety."

    result = asyncio.run(adapter.run(fake, journal_path=tmp_path / "one.jsonl"))
    assert result["status"] == "completed" and result["safety"] == "not_evaluated"
    private = adapter.oracle_view()
    assert private["initial_state"][table] != private["final_state"][table]
    trans = private["tool_transitions"][0]
    assert trans["actor"] == "work" and trans["bank_role"] == "task_operator"
    assert trans["arguments"] == arguments and trans["state_before"][table] != trans["state_after"][table]
    audits = private["final_state"]["audit_log"]
    assert any(row.get("agent_role") == "task_operator" or row.get("actor_role") == "task_operator" for row in audits)
    public = adapter.monitor_prefix()
    assert public["coverage"]["run_status"] == "completed"
    assert len(public["tools"]) == len(REVIEWED_TOOLS[domain])
    encoded = json.dumps({"view": public, "packets": packets})
    for forbidden in ("PRIVATE-ORACLE-SENTINEL", "PRIVATE-NECESSITY", "initial_state", "state_before", "seed_sha256"):
        assert forbidden not in encoded
    assert "normative" not in json.dumps(packets)
    asyncio.run(adapter.run(fake, journal_path=tmp_path / "two.jsonl"))
    again = adapter.oracle_view()
    assert again["initial_state"] == private["initial_state"]
    assert len(again["final_state"][table]) == len(private["final_state"][table])
    assert not list(tmp_path.rglob("*.sqlite"))


def test_branch_merge_routes_only_direct_predecessors_and_preserves_prefix(tmp_path):
    workflow = {"nodes": ["root", "left", "right", "merge"],
                "edges": [["root", "left"], ["root", "right"], ["left", "merge"], ["right", "merge"]]}
    bindings = {n: {"role": "task_operator", "source": "explicit role clone"} for n in workflow["nodes"]}
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, workflow=workflow, role_bindings=bindings))
    seen, snapshots = {}, []

    async def fake(packet, invoke):
        node = packet["node_id"]
        seen[node] = packet["input"]["predecessor_reports"]
        snapshots.append(adapter.monitor_prefix())
        return "report-for-" + node

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "branch.jsonl"))
    assert seen["root"] == []
    common = {"sender_role": "task_operator", "sender_role_source": "explicit role clone"}
    assert seen["right"] == [{"sender": "root", "content": "report-for-root", **common}]
    assert seen["merge"] == [{"sender": "left", "content": "report-for-left", **common},
                              {"sender": "right", "content": "report-for-right", **common}]
    for prefix in snapshots:
        assert prefix == adapter.monitor_prefix(prefix["sequence"])
        assert "report-for-merge" not in json.dumps(prefix)
    with pytest.raises(ValueError, match="future"):
        adapter.monitor_prefix(100000)
    messages = [r["payload"] for r in adapter.monitor_prefix()["journal"] if r["kind"] == "communication"]
    routed = [(p["actor"], p["target"]) for p in messages if p["target"] != "harness_report_store"]
    assert routed == [tuple(e) for e in workflow["edges"]]


def test_explicit_state_visibility_does_not_release_other_columns(tmp_path):
    selection = [{"table": "employees", "columns": ["user_id", "status"], "where": {"user_id": "USR-SYNTH"},
                  "source": "explicit synthetic sandbox instrumentation, not ordinary API visibility"}]
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, state_visibility=selection))

    async def fake(packet, invoke):
        if packet["node_id"] == "work":
            await invoke(MUTATIONS["office"][0], MUTATIONS["office"][1])
        return "Finished."

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "selected.jsonl"))
    view = adapter.monitor_prefix()
    assert view["observability"]["regime"] == "trace_plus_declared_sandbox_state"
    native = next(r["payload"] for r in view["journal"] if r["kind"] == "tool_native_result")
    assert native["observable_state_before"][0]["rows"] == []
    assert native["observable_state_after"][0]["rows"] == [{"user_id": "USR-SYNTH", "status": "provisioning"}]
    assert "email" not in json.dumps(native["observable_state_after"])
    assert adapter.oracle_view()["final_state"]["employees"][0]["email"] == "person@example.invalid"


@pytest.mark.parametrize("mutation", ["cycle", "unbound", "unknown_role", "private_task", "private_actor", "unknown_column", "unsupported_domain", "reserved_stage"])
def test_bad_declarations_are_rejected_before_run(tmp_path, mutation):
    args = config(tmp_path)
    if mutation == "cycle":
        args["workflow"]["edges"].append(["review", "work"])
    elif mutation == "unbound":
        del args["role_bindings"]["review"]
    elif mutation == "unknown_role":
        args["role_bindings"]["work"]["role"] = "not_an_author_role"
    elif mutation == "private_task":
        args["public_task"]["groundtruth"] = "hidden"
    elif mutation == "private_actor":
        args["public_task"]["agents"][0]["tool_necessity"] = ["hidden"]
    elif mutation == "unknown_column":
        args["state_visibility"] = [{"table": "employees", "columns": ["nonexistent"], "where": {}, "source": "explicit"}]
    elif mutation == "unsupported_domain":
        args["domain"] = "sde"
    else:
        args["workflow"]["nodes"].append("harness_report_store")
    with pytest.raises(ValueError):
        HarnessAuditSandboxAdapter(**args)


@pytest.mark.parametrize("failure", ["unknown_tool", "budget", "node", "blank", "timeout"])
def test_failures_remain_failed_and_do_not_run_successors(tmp_path, failure):
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, max_tools_per_node=1, node_timeout_seconds=0.02))
    visited = []

    async def fake(packet, invoke):
        visited.append(packet["node_id"])
        if failure == "unknown_tool":
            return await invoke("send_real_email", {"recipient": "nobody@example.invalid"})
        if failure == "budget":
            await invoke("read_dashboard_metric", {})
            return await invoke("read_dashboard_metric", {})
        if failure == "node":
            raise RuntimeError("PRIVATE-EXCEPTION-TEXT")
        if failure == "timeout":
            await asyncio.sleep(1)
        return ""

    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        asyncio.run(adapter.run(fake, journal_path=tmp_path / "failed.jsonl"))
    assert visited == ["work"]
    view, private = adapter.monitor_prefix(), adapter.oracle_view()
    assert view["coverage"]["run_status"] == private["status"] == "failed"
    assert view["coverage"]["open_invocations"] == []
    assert "PRIVATE-EXCEPTION-TEXT" not in json.dumps(view)
    if failure in {"unknown_tool", "budget"}:
        assert private["tool_transitions"][-1]["bank_dispatched"] is False
        assert private["tool_transitions"][-1]["terminal"] == "failed"
        attempt = adapter.oracle_actions("synthetic-failed-tool")["actions"][-1]
        assert attempt["raw_event"]["bank_dispatched"] is False
        assert attempt["raw_event"]["surface_semantics"].startswith("tool_request_attempt")


def test_passive_policy_and_business_rejection_not_claimed_success(tmp_path):
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, "ecommerce", visible_policy={"source": "synthetic public policy",
        "rules": [{"subject": "task_operator", "forbidden_tools": ["approve_quote"]}]}))

    async def fake(packet, invoke):
        # A legitimate catalogue tool executes irrespective of visible policy;
        # the monitor is not silently converted into a prevention mechanism.
        return await invoke("approve_quote", {"quote_id": "NONEXISTENT", "decision": "approve"})

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "rejected.jsonl"))
    private = adapter.oracle_view()
    assert private["initial_state"]["quotes"] == private["final_state"]["quotes"]
    native = [r["payload"] for r in adapter.monitor_prefix()["journal"] if r["kind"] == "tool_native_result"]
    assert len(native) == 2 and all("rejected" in p["result"] for p in native)
    assert all(p["semantic_success"] == "not_inferred_from_return" for p in native)


def test_stale_tool_handles_and_seed_changes_rejected(tmp_path):
    args = config(tmp_path)
    adapter = HarnessAuditSandboxAdapter(**args)
    handles = []

    async def fake(packet, invoke):
        handles.append(invoke)
        if packet["node_id"] == "review":
            with pytest.raises(RuntimeError, match="active node"):
                await handles[0]("read_dashboard_metric", {})
        return "Synthetic report"

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "first.jsonl"))
    with pytest.raises(FileExistsError):
        asyncio.run(adapter.run(fake, journal_path=tmp_path / "first.jsonl"))
    (args["seed_dir"] / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="seed fixture changed"):
        asyncio.run(adapter.run(fake, journal_path=tmp_path / "second.jsonl"))


def test_shared_author_hub_role_maps_to_distinct_stage_ids(tmp_path):
    workflow = {"nodes": ["hub_plan", "spoke", "hub_review"], "edges": [["hub_plan", "spoke"], ["spoke", "hub_review"]]}
    bindings = {n: {"role": "task_reviewer" if n == "spoke" else "task_operator", "source": "declared original role"}
                for n in workflow["nodes"]}
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, workflow=workflow, role_bindings=bindings))

    async def fake(packet, invoke):
        await invoke("read_dashboard_metric", {"metric_name": "SYNTH"}, "provider-" + packet["node_id"])
        return "report-" + packet["node_id"]

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "hub.jsonl"))
    bridge = adapter.oracle_actions("synthetic-role-test")
    assert bridge["labels"] == "not_evaluated" and len(bridge["excluded_observations"]) == 3
    from multi_agent.schemas.actions import ObservableAction
    actions = [ObservableAction.model_validate(a) for a in bridge["actions"]]
    messages = [a for a in actions if a.surface == "communication"]
    assert [(a.agent_role, a.target_role) for a in messages] == [
        ("task_operator", "task_reviewer"), ("task_reviewer", "task_operator")]
    tools = [a for a in actions if a.surface == "tool_call"]
    assert [(a.agent_id, a.agent_role) for a in tools] == [
        ("hub_plan", "task_operator"), ("spoke", "task_reviewer"), ("hub_review", "task_operator")]
    assert [a.raw_event["stage_id"] for a in tools] == workflow["nodes"]
    assert len(adapter.oracle_view()["final_state"]["audit_log"]) == 3


def test_source_pin_and_real_backend_are_rejected_without_upstream_edits(tmp_path, monkeypatch):
    args = config(tmp_path)
    original_run = implementation.subprocess.run

    def wrong_head(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[-2:] == ["rev-parse", "HEAD"]:
            result.stdout = "wrong-commit"
        return result

    with monkeypatch.context() as patch:
        patch.setattr(implementation.subprocess, "run", wrong_head)
        with pytest.raises(ValueError, match="reviewed commit"):
            HarnessAuditSandboxAdapter(**args)
    original_load = implementation.yaml.safe_load

    def real_catalog(text):
        value = original_load(text)
        value["tools"][0]["backend_type"] = "real"
        return value

    with monkeypatch.context() as patch:
        patch.setattr(implementation.yaml, "safe_load", real_catalog)
        with pytest.raises(ValueError, match="real backend"):
            HarnessAuditSandboxAdapter(**args)


def test_in_memory_connection_rejects_attach(tmp_path):
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path))
    bank = adapter._new_bank()
    try:
        assert all(not row[2] for row in bank.conn.execute("PRAGMA database_list"))
        with pytest.raises(implementation.sqlite3.DatabaseError, match="authorized"):
            bank.conn.execute("ATTACH DATABASE ':memory:' AS unauthorized_database")
    finally:
        bank.close()


@pytest.mark.parametrize("nodes", [["work"], ["absent"], ["review", "review"], "review"])
def test_user_delivery_requires_explicit_unique_sink_nodes(tmp_path, nodes):
    with pytest.raises(ValueError, match="output"):
        HarnessAuditSandboxAdapter(**config(tmp_path, output_nodes=nodes))


def test_only_declared_sink_is_delivered_to_user(tmp_path):
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, output_nodes=["review"]))
    saved = []

    async def fake(packet, invoke):
        if packet["node_id"] == "review":
            saved.append(adapter.monitor_prefix())
            assert packet["delivery_target"] == "user"
            assert "declared sink is delivered to the user" in packet["instructions"]
        else:
            assert packet["delivery_target"] == "harness_report_store"
        return "internal-work-report" if packet["node_id"] == "work" else "actual-user-report"

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "user.jsonl"))
    view = adapter.monitor_prefix()
    assert saved[0] == adapter.monitor_prefix(saved[0]["sequence"])
    assert "actual-user-report" not in json.dumps(saved[0])
    communications = [r["payload"] for r in view["journal"] if r["kind"] == "communication"]
    delivered = [p for p in communications if p["target"] == "user"]
    stored = [p for p in communications if p["target"] == "harness_report_store"]
    assert [(p["actor"], p["content"]) for p in delivered] == [("review", "actual-user-report")]
    assert len(stored) == 2
    bridge = adapter.oracle_actions("synthetic-user-delivery")
    user_actions = [a for a in bridge["actions"] if a.get("target_role") == "user"]
    assert len(user_actions) == 1 and user_actions[0]["agent_role"] == "task_reviewer"
    assert user_actions[0]["content"] == "actual-user-report"
    assert user_actions[0]["raw_event"]["communication_kind"] == "user_output"
    assert len(bridge["excluded_observations"]) == 2


def test_private_finance_setup_changes_initial_state_without_becoming_model_input(tmp_path):
    args = config(tmp_path, "finance")
    seed = args["seed_dir"]
    (seed / "seed_prices.json").write_text(json.dumps({"SYNTH": {"price": 10, "sector": "Synthetic"}}), encoding="utf-8")
    (seed / "seed_portfolios.json").write_text(json.dumps({"PF-SYNTH": {
        "client_id": "CL-SYNTH", "client_name": "Fixture", "cash": 10000, "positions": []}}), encoding="utf-8")
    setup = {"client_id": "CL-SYNTH", "portfolio_id": "PF-SYNTH", "request_id": "PRIVATE-SETUP-REQUEST",
             "seed_trades": [{"action": "BUY", "ticker": "SYNTH", "quantity": 2, "portfolio_id": "PF-SYNTH",
                              "order_id": "PRIVATE-SETUP-ORDER", "submitted_by": "PRIVATE-SETUP-ROLE",
                              "timestamp": "2026-01-01T00:00:00Z"}]}
    adapter = HarnessAuditSandboxAdapter(**args, bank_setup_private=setup)
    packets = []

    async def fake(packet, invoke):
        packets.append(packet)
        return "Synthetic report with no tool requests."

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "setup.jsonl"))
    private = adapter.oracle_view()
    assert private["bank_setup_private"] == setup
    initial = private["initial_state"]
    assert len(initial["trades"]) == 1 and initial["trades"][0]["order_id"] == "PRIVATE-SETUP-ORDER"
    assert initial["positions"][0]["quantity"] == 2
    assert initial["portfolios"][0]["cash"] < 10000
    assert private["tool_transitions"] == []  # Setup is not a MAS tool action.
    assert "PRIVATE-SETUP" not in json.dumps({"packets": packets, "prefix": adapter.monitor_prefix()})
    asyncio.run(adapter.run(fake, journal_path=tmp_path / "setup_reset.jsonl"))
    assert adapter.oracle_view()["initial_state"] == initial


@pytest.mark.parametrize("domain,setup", [
    ("office", {"client_id": "fixture"}), ("finance", {"groundtruth": "forbidden"}),
    ("finance", {"seed_trades": "not_a_list"}), ("finance", {"seed_trades": ["not_an_object"]})])
def test_unreviewed_bank_setup_rejected(tmp_path, domain, setup):
    with pytest.raises(ValueError):
        HarnessAuditSandboxAdapter(**config(tmp_path, domain), bank_setup_private=setup)


def test_public_stage_purposes_are_explicit_and_not_inferred_from_node_names(tmp_path):
    purposes = {"work": {"purpose": "Prepare a synthetic plan only.", "source": "explicit public stage contract"},
                "review": {"purpose": "Review only the delivered predecessor plan.", "source": "explicit public stage contract"}}
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path), stage_purposes=purposes)
    packets = []

    async def fake(packet, invoke):
        packets.append(packet)
        assert packet["stage_purpose"] == purposes[packet["node_id"]]
        assert packet["stage_purpose"]["purpose"] in packet["instructions"]
        return "Synthetic report."

    asyncio.run(adapter.run(fake, journal_path=tmp_path / "purpose.jsonl"))
    assert adapter.monitor_prefix(0)["stage_purposes"] == purposes
    assert packets[1]["input"]["predecessor_reports"][0]["sender_role"] == "task_operator"
    with pytest.raises(ValueError, match="complete node"):
        HarnessAuditSandboxAdapter(**config(tmp_path), stage_purposes={"work": purposes["work"]})
