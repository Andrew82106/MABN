import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from paperalpha_runtime.stage_capture import StageRuntimeJournal, stage_instructions
from paperalpha_runtime.stage_workflow import StageBindings
from collect_stage_harnessaudit import StageLedger, StageLifecycleRecorder
from collect_harnessaudit_api_pilot import make_arrival_sink
from audit_harnessaudit_api_pilot import verify_shared_journal, rows
from test_stage_workflow import workflow


def test_capture_scopes_bind_physical_actors_and_are_reset_after_failure(tmp_path):
    journal = StageRuntimeJournal(tmp_path / "journal.jsonl", workflow())
    with pytest.raises(ValueError, match="no active"):
        journal.active()
    with pytest.raises(RuntimeError):
        with journal.scope("s:first"):
            assert journal.active() == ("s:first", "hub", None)
            with pytest.raises(ValueError, match="mismatch"):
                journal.emit("tool_handler_started", {"actor": "worker"})
            raise RuntimeError("stop")
    assert journal.count == 0
    with pytest.raises(ValueError, match="no active"):
        journal.active()


def test_request_ledger_records_stage_without_changing_physical_actor_or_budget(tmp_path):
    journal = StageRuntimeJournal(tmp_path / "journal.jsonl", workflow())
    ledger = StageLedger(tmp_path, 2, journal)
    for stage in ("s:first", "s:final"):
        with journal.scope(stage):
            ledger.begin({"model_requested": "test"})
    actual = rows(tmp_path / "api_requests.jsonl")
    assert [r["stage_id"] for r in actual] == ["s:first", "s:final"]
    assert [r["actor"] for r in actual] == ["hub", "hub"]
    assert ledger.started == ledger.budget == 2


def test_original_tool_lifecycle_and_arrival_hooks_emit_matching_stage_journal(tmp_path):
    w = workflow()
    journal = StageRuntimeJournal(tmp_path / "runtime_events.jsonl", w)
    journal.emit("run_started", {"agents": ["hub", "worker"], "external_endpoints": ["user"], "declared_stage_workflow": w})
    recorder = StageLifecycleRecorder(tmp_path, journal)
    class Sink:
        def __init__(self):
            self._ordinal = 0
        def emit_tool_call(self, **kwargs):
            self._ordinal += 1
        def emit_communication(self, **kwargs):
            self._ordinal += 1
    sink = make_arrival_sink(Sink, tmp_path, recorder, journal)()
    async def original(ctx, args):
        await asyncio.sleep(0)
        actor = ctx.context.agent_role
        sink.emit_tool_call(agent_id=actor, agent_role=actor, tool_name="read", tool_args=json.loads(args), tool_result="done")
        return "done"
    tool = recorder.wrap(SimpleNamespace(name="read", on_invoke_tool=original))
    async def run():
        for stage, actor in [("s:first", "hub"), ("s:work", "worker"), ("s:final", "hub")]:
            with journal.scope(stage):
                await tool.on_invoke_tool(SimpleNamespace(context=SimpleNamespace(agent_role=actor), tool_call_id=stage), '{}')
        with journal.scope("s:first", "s:work"):
            sink.emit_communication(agent_id="hub", agent_role="hub", target_agent="worker", target_role="worker", content="report")
        with journal.scope("s:final"):
            sink.emit_communication(agent_id="hub", agent_role="hub", target_agent="user", target_role="user", content="final")
    asyncio.run(run())
    journal.emit("run_finished", {"status": "completed"})
    recorded = rows(journal.path)
    checked = verify_shared_journal(recorded, rows(tmp_path / "arrival_events.jsonl"), rows(tmp_path / "tool_lifecycle.jsonl"))
    assert checked["all_legacy_record_links_verified"]
    binding = StageBindings(w)
    for row in recorded:
        binding.observe(row)
    assert len(binding.invocations) == 3
    assert {r["payload"]["stage_id"] for r in recorded if r["kind"] == "tool_handler_started"} == set(w["nodes"])


def test_async_scopes_do_not_mix_same_actor_stages(tmp_path):
    journal = StageRuntimeJournal(tmp_path / "journal.jsonl", workflow())
    async def run(stage):
        with journal.scope(stage):
            await asyncio.sleep(0)
            journal.emit("tool_handler_started", {"actor": "hub", "invocation_id": stage})
    async def gather():
        await asyncio.gather(run("s:first"), run("s:final"))
    asyncio.run(gather())
    assert {r["payload"]["stage_id"] for r in rows(journal.path)} == {"s:first", "s:final"}


def test_prompt_limits_stage_work_without_importing_hidden_policy():
    prompt = stage_instructions({"role": "hub", "description": "Coordinate", "system_prompt": "Original duty"}, "s:review", "Review quote")
    assert "Original duty" in prompt and "Review quote" in prompt and "fresh context" in prompt
    assert "useful_tools" not in prompt and "access_rules" not in prompt


def test_stage_collector_imports_without_numpy_in_sdk_environment():
    python = ROOT / '.local/environments/collection/harnessaudit_oai_20260919/Scripts/python.exe'
    if not python.exists():
        pytest.skip("SDK runtime not installed")
    result = subprocess.run([str(python), "-X", "utf8", "-B", str(ROOT / 'code/commands/collect_stage_harnessaudit.py'), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
