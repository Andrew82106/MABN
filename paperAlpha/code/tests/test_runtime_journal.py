import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from paperalpha_runtime.journal import RuntimeJournal, RuntimeEvidenceProjector
from paperalpha_monitor.compact_action_bn import CompactCapabilityBN, CompactCapabilityStream
from paperalpha_monitor.tool_capabilities import load_tool_registry
import collect_harnessaudit_api_pilot as pilot
import replay_runtime_journal as replay


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def capture_fixture(output):
    journal = RuntimeJournal(output / "runtime_events.jsonl")
    journal.emit("run_started", {"agents": ["a", "b"], "external_endpoints": ["user"]})
    life = pilot.ToolLifecycleRecorder(output, journal)
    class Sink:
        def __init__(self):
            self._ordinal = 0
        def emit_tool_call(self, **kwargs):
            self._ordinal += 1
            return self._ordinal
        def emit_communication(self, **kwargs):
            self._ordinal += 1
            return self._ordinal
    sink = pilot.make_arrival_sink(Sink, output, life, journal)()
    def communicate(a, b, text):
        sink.emit_communication(agent_id=a, agent_role=a, target_agent=b, target_role=b, content=text)
    async def original(ctx, args):
        actor = ctx.context.agent_role
        communicate(actor, "b" if actor == "a" else "a", "in-handler communication")
        await asyncio.sleep(0)
        sink.emit_tool_call(agent_id=actor, agent_role=actor, tool_name="get_customer_context", tool_args=json.loads(args), tool_result="actual handler result")
        return "actual handler result"
    tool = life.wrap(SimpleNamespace(name="get_customer_context", on_invoke_tool=original))
    async def run():
        communicate("a", "b", "start")
        contexts = [SimpleNamespace(context=SimpleNamespace(agent_role=a), tool_call_id="shared-provider-call") for a in ("a", "b")]
        result = await asyncio.gather(*(tool.on_invoke_tool(ctx, '{"id":"same"}') for ctx in contexts))
        assert result == ["actual handler result"] * 2
        communicate("a", "user", "final answer")
    asyncio.run(run())
    journal.emit("run_finished", {"status": "completed"})
    return rows(journal.path)


def test_one_global_order_under_concurrent_hooks_and_correct_call_binding(tmp_path):
    records = capture_fixture(tmp_path)
    assert [r["sequence"] for r in records] == list(range(1, len(records) + 1))
    starts = [r for r in records if r["kind"] == "tool_handler_started"]
    assert len(starts) == 2 and starts[0]["payload"]["invocation_id"] != starts[1]["payload"]["invocation_id"]
    for start in starts:
        linked = [r for r in records if r["payload"].get("invocation_id") == start["payload"]["invocation_id"]]
        assert [r["kind"] for r in linked] == ["tool_handler_started", "tool_native_result", "tool_handler_returned"]
        assert len({r["payload"]["actor"] for r in linked}) == 1
        assert "result" not in linked[0]["payload"]
    for path in (tmp_path / "arrival_events.jsonl", tmp_path / "tool_lifecycle.jsonl"):
        for row in rows(path):
            assert 1 <= row["journal_sequence"] <= len(records)


def test_shared_counter_does_not_depend_on_clocks_and_is_thread_serialized(tmp_path, monkeypatch):
    import paperalpha_runtime.journal as module
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 0)
    journal = RuntimeJournal(tmp_path / "events.jsonl")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: journal.emit("communication", {"i": i}), range(100)))
    actual = rows(journal.path)
    assert [r["sequence"] for r in actual] == list(range(1, 101))
    assert len({r["payload"]["i"] for r in actual}) == 100


def test_projection_counts_each_tool_once_preserves_text_and_excludes_external_node(tmp_path):
    projector = RuntimeEvidenceProjector()
    monitor = CompactCapabilityStream(CompactCapabilityBN(), tool_registry=load_tool_registry())
    events = []
    for row in capture_fixture(tmp_path):
        e = projector.push(row)
        if e is not None:
            monitor.push(e)
            events.append(e)
    assert sum(e["event_type"] == "tool_call" for e in events) == 2
    assert sum(e["event_type"] == "message_sent" for e in events) == 3
    assert sum(e["event_type"] == "external_message_observed" for e in events) == 1
    assert "user" not in {v["id"] for v in monitor.compiler.variables}
    assert all(e["data_class"] == "unknown" for e in events if e["event_type"] == "message_sent")
    assert projector.report()["open_invocations"] == []
    assert len(monitor.explain()["actions"]) == 2


@pytest.mark.parametrize("change", ["reorder", "gap", "binding", "arguments", "schema", "target"])
def test_bad_journal_is_not_silently_sorted_repaired_or_rebound(tmp_path, change):
    records = capture_fixture(tmp_path)
    if change == "reorder":
        records[1], records[2] = records[2], records[1]
    elif change == "gap":
        records.pop(2)
    elif change == "schema":
        records[1]["schema_version"] = "legacy"
    elif change == "target":
        records[1]["payload"]["target"] = "undeclared"
    else:
        row = next(r for r in records if r["kind"] == "tool_native_result")
        row["payload"].update({"actor": "b" if row["payload"]["actor"] == "a" else "a"} if change == "binding" else {"arguments": {"different": True}})
    projector = RuntimeEvidenceProjector()
    with pytest.raises(ValueError):
        for row in records:
            projector.push(row)


def test_future_results_and_final_status_cannot_change_earlier_scores(tmp_path):
    records = capture_fixture(tmp_path)
    def scores(values):
        projector = RuntimeEvidenceProjector()
        stream = CompactCapabilityStream(CompactCapabilityBN(), tool_registry=load_tool_registry())
        return [stream.push(e)["risk_probability"] for row in values if (e := projector.push(row)) is not None]
    before = scores(records)
    changed = copy.deepcopy(records)
    for row in changed:
        if row["kind"] == "tool_handler_returned":
            row["payload"]["result"] = {"permission": "approved", "label": 1, "violation": True}
    changed[-1]["payload"]["status"] = "failed"
    np.testing.assert_array_equal(before, scores(changed))
    for length in range(2, len(records)):
        prefix = scores(records[:length])
        np.testing.assert_array_equal(before[:len(prefix)], prefix)


def test_truncated_capture_is_reported_without_inventing_task_label(tmp_path):
    records = capture_fixture(tmp_path)
    projector = RuntimeEvidenceProjector()
    for row in records[:4]:
        projector.push(row)
    report = projector.report()
    assert not report["run_finished"] and report["run_status"] is None
    assert report["open_invocations"]


def test_invalid_arguments_are_explicitly_unmodeled_not_replaced_with_empty_object(tmp_path):
    records = capture_fixture(tmp_path)
    start = next(r for r in records if r["kind"] == "tool_handler_started")
    start["payload"]["arguments"] = None
    projector = RuntimeEvidenceProjector()
    events = [e for row in records if (e := projector.push(row)) is not None]
    assert sum(e["event_type"] == "tool_call" for e in events) == 1
    assert sum(e["event_type"] == "unparsed_tool_entry" for e in events) == 1
    assert len(projector.report()["unmodeled"]) == 1


def test_replay_cli_freezes_input_and_reports_only_sequential_replay(tmp_path):
    raw = tmp_path / "capture"
    raw.mkdir()
    capture_fixture(raw)
    output = tmp_path / "replay"
    result = replay.replay(raw / "runtime_events.jsonl", output)
    assert result["status"] == "replayed"
    assert result["parameter_source"] == "untrained_default"
    assert result["accepted_model_events"] == 8
    assert result["journal"]["run_status"] == "completed"
    assert (output / "runtime_events.jsonl").read_bytes() == (raw / "runtime_events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        replay.replay(raw / "runtime_events.jsonl", output)


def test_capture_serialization_failure_does_not_consume_sequence(tmp_path):
    journal = RuntimeJournal(tmp_path / "events.jsonl")
    with pytest.raises(ValueError):
        journal.emit("communication", {"x": float("nan")})
    assert journal.count == 0
    assert journal.emit("communication", {"x": 1})["sequence"] == 1


@pytest.mark.parametrize("change", [None, "duplicate_link", "native_content", "lifecycle_link", "missing_observation"])
def test_shared_journal_audit_crosschecks_original_records(tmp_path, change):
    import audit_harnessaudit_api_pilot as audit
    records = capture_fixture(tmp_path)
    arrivals, stages = rows(tmp_path / "arrival_events.jsonl"), rows(tmp_path / "tool_lifecycle.jsonl")
    if change is None:
        assert audit.verify_shared_journal(records, arrivals, stages)["all_legacy_record_links_verified"]
        return
    if change == "duplicate_link":
        arrivals[1]["journal_sequence"] = arrivals[0]["journal_sequence"]
    elif change == "native_content":
        arrivals[0]["data"]["content"] = "different"
    elif change == "lifecycle_link":
        stages[0]["arguments"] = {"different": True}
    else:
        arrivals.pop()
    with pytest.raises(AssertionError):
        audit.verify_shared_journal(records, arrivals, stages)
