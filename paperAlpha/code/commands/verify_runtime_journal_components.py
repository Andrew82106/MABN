"""Exercise real SDK FunctionTool/context + official sink/bank, without an LLM.

Invocation arguments and communications are manually scripted component data.
This is neither an API-generated MAS run nor an accuracy experiment.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import collect_harnessaudit_api_pilot as pilot


async def verify(output):
    if output.exists():
        raise FileExistsError("component verification needs a new directory")
    upstream = pilot.DEFAULT_UPSTREAM
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != pilot.EXPECTED_REVISION:
        raise ValueError("upstream revision differs from reviewed components")
    pilot.import_official_oai_components(upstream)
    from multi_agent.loader import load_tool_catalog
    from multi_agent.banks import get_bank
    from multi_agent.frameworks.oai.tools import build_domain_tools
    from multi_agent.frameworks.oai.adapter import MultiAgentContext
    from multi_agent.frameworks.core.action_sink import ActionSink
    from multi_agent.trace.writer import TraceWriter
    from agents.tool_context import ToolContext
    from agents.tracing import set_tracing_disabled
    set_tracing_disabled(True)
    from paperalpha_runtime.journal import RuntimeJournal, RuntimeEvidenceProjector

    catalog, bank = load_tool_catalog("daily_life"), get_bank("daily_life")
    output.mkdir(parents=True)
    journal = RuntimeJournal(output / "runtime_events.jsonl")
    journal.emit("run_started", {"agents": ["fixture-a", "fixture-b"], "external_endpoints": ["user"]})
    lifecycle = pilot.ToolLifecycleRecorder(output, journal)
    results = []
    try:
        with TraceWriter(output / "native_component_events.jsonl") as writer:
            sink = pilot.make_arrival_sink(ActionSink, output, lifecycle, journal)(run_id="component-fixture", trace_writer=writer)
            def communicate(actor, target, content):
                sink.emit_communication(agent_id=actor, agent_role=actor, target_agent=target, target_role=target, content=content)
            communicate("fixture-a", "fixture-b", "Manually scripted component request, not an LLM message.")
            selected = next(t for t in build_domain_tools(catalog) if t.name == "get_customer_context")
            tool = lifecycle.wrap(selected)
            async def invoke(actor):
                arguments = json.dumps({"customer_id": "DL-CUST-001", "context_scope": "profile"})
                ctx = ToolContext(context=MultiAgentContext(action_sink=sink, bank=bank, workspace=None, agent_role=actor),
                    tool_name=tool.name, tool_call_id=f"component-call-{actor}", tool_arguments=arguments)
                result = await tool.on_invoke_tool(ctx, arguments)
                communicate(actor, "fixture-b" if actor == "fixture-a" else "fixture-a", "Manually scripted post-handler message.")
                return result
            results = await asyncio.gather(invoke("fixture-a"), invoke("fixture-b"))
            communicate("fixture-a", "user", "Manually scripted final component message.")
            actions = sink.actions
    finally:
        bank.close()
    journal.emit("run_finished", {"status": "completed"})
    projector = RuntimeEvidenceProjector()
    model_events = []
    for line in journal.path.read_text(encoding="utf-8").splitlines():
        event = projector.push(json.loads(line))
        if event is not None:
            model_events.append(event)
    if projector.report()["open_invocations"] or lifecycle.started != 2 or lifecycle.returned != 2:
        raise AssertionError("SDK component lifecycle not fully bound")
    snapshot = output / "source_snapshot"
    local = [Path(__file__), Path(pilot.__file__), *(pilot.ROOT / 'code/src/paperalpha_runtime').glob("*.py")]
    source_files = []
    for path in local:
        target = snapshot / path.resolve().relative_to(pilot.ROOT.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_files.append(target)
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("multi_agent.") and filename and Path(filename).is_relative_to(upstream.resolve()):
            path = Path(filename)
            target = snapshot / "upstream" / path.resolve().relative_to(upstream.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            source_files.append(target)
    report = {"status": "passed_component_check", "model_api_calls": 0, "upstream_commit": revision,
        "sdk_versions": {n: importlib.metadata.version(n) for n in ("openai-agents", "openai")},
        "actual_sdk_tool_invocations": lifecycle.started, "original_handlers_returned": lifecycle.returned,
        "native_actions": len(actions), "journal": projector.report(), "projected_model_events": len(model_events),
        "scope": "manually scripted invocations/messages through real SDK and official bank/sink; not an API-generated task, accuracy result or online warning experiment",
        "tool_results": results, "source_sha256": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    pilot.write_json(output / "REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"tool_results", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(verify(parser.parse_args().out))
