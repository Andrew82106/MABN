"""Offline verification of API-pilot acquisition, not monitor accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_runtime.journal import RuntimeEvidenceProjector


def verify_shared_journal(records, arrivals, stages):
    projector = RuntimeEvidenceProjector()
    for row in records:
        projector.push(row)
    index = {r["sequence"]: r for r in records}
    linked = set()
    for stage in stages:
        sequence = stage.get("journal_sequence")
        assert sequence in index and sequence not in linked, "missing/reused lifecycle journal link"
        row = index[sequence]
        assert row["kind"] == stage["event_type"], "lifecycle journal kind mismatch"
        assert row["payload"] == {k: v for k, v in stage.items() if k != "journal_sequence"}, "lifecycle journal payload mismatch"
        linked.add(sequence)
    for arrival in arrivals:
        sequence = arrival.get("journal_sequence")
        assert sequence in index and sequence not in linked, "missing/reused arrival journal link"
        row, data = index[sequence], arrival["data"]
        assert row["payload"]["native_arrival_index"] == arrival["arrival_index"], "native ordinal mismatch"
        if arrival["surface"] == "communication":
            assert row["kind"] == "communication", "communication journal kind mismatch"
            assert row["payload"]["actor"] == data["agent_id"] and row["payload"]["target"] == data["target_agent"]
            assert row["payload"]["content"] == data["content"], "communication content mismatch"
        else:
            assert row["kind"] == "tool_native_result", "tool-result journal kind mismatch"
            raw = data["raw_event"]
            expected = {"invocation_id": raw["paperalpha_invocation_id"], "provider_call_id": raw["provider_call_id"],
                        "actor": data["agent_id"], "tool": data["tool_name"], "arguments": data["tool_args"], "result": data["tool_result"]}
            assert all(row["payload"][k] == v for k, v in expected.items()), "native result binding mismatch"
        linked.add(sequence)
    assert linked == {r["sequence"] for r in records if r["kind"] not in {"run_started", "run_finished"}}, "unmatched journal observation"
    report = projector.report()
    assert report["run_finished"], "capture is missing its final status"
    return {**report, "all_legacy_record_links_verified": True}

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def verify_lifecycle(stages, native_tools, responses):
    """Match actual SDK call identity, not guessed actor/tool/text equality."""
    starts, ends = {}, {}
    for stage in stages:
        identity = stage["invocation_id"]
        if stage["event_type"] == "tool_handler_started":
            assert identity not in starts and identity not in ends, "duplicate or late start"
            assert "result" not in stage and "violated" not in stage, "pre-handler result/audit leak"
            starts[identity] = stage
        else:
            assert stage["event_type"] in {"tool_handler_returned", "tool_handler_failed"}, "unknown lifecycle stage"
            assert identity in starts and identity not in ends, "unmatched/duplicate terminal"
            assert starts[identity]["monotonic_ns"] <= stage["monotonic_ns"], "reversed lifecycle clock"
            for key in ("actor", "tool", "provider_call_id"):
                assert starts[identity][key] == stage[key], "binding changed after execution"
            ends[identity] = stage
    assert starts.keys() == ends.keys(), "incomplete lifecycle"
    provider_calls = {}
    for response in responses:
        for item in response.get("response", {}).get("output", []):
            if item.get("type") == "function_call":
                provider_calls.setdefault(item["call_id"], []).append(item)
    for start in starts.values():
        matches = provider_calls.get(start["provider_call_id"], [])
        assert any(item["name"] == start["tool"] and item["arguments"] == start["arguments_raw"] for item in matches), "no corresponding recorded model call"
    matched = set()
    for tool in native_tools:
        identity = tool.get("raw_event", {}).get("paperalpha_invocation_id")
        assert identity in starts and identity not in matched, "missing or duplicate native invocation link"
        matched.add(identity)
        start, end = starts[identity], ends[identity]
        assert start["actor"] == tool["agent_id"] and start["tool"] == tool["tool_name"], "native actor/tool mismatch"
        assert start["provider_call_id"] == tool["raw_event"]["provider_call_id"], "native SDK call mismatch"
        if start["arguments"] is not None:
            assert start["arguments"] == tool["tool_args"], "native arguments mismatch"
        assert end["event_type"] == "tool_handler_returned" and end["result"] == tool["tool_result"], "native result mismatch"
    # A thrown handler need not emit a native completed action.
    assert matched == {key for key, end in ends.items() if end["event_type"] == "tool_handler_returned"}, "returned handler without native action"
    return {"started": len(starts), "returned": len(matched), "failed": len(starts) - len(matched),
            "bindings_verified": True, "pre_handler_records_have_no_result_or_audit": True,
            "meaning": "Captured before actual handler entry; no risk prediction, authorization guarantee or measured warning latency."}


def audit(run, output):
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise ValueError("audit output already exists")
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    assert manifest["status"] in {"completed", "failed"}, "not a finalized acquisition"
    for relative, expected in manifest["artifact_sha256"].items():
        path = (run / relative).resolve()
        assert path.is_relative_to(run), "artifact outside run"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"hash mismatch: {relative}"
    native = rows(run / "native_trace.jsonl")
    tools = [r for r in native if r["type"] == "tool_call"]
    actions = [r for r in native if r["type"] in {"tool_call", "communication"}]
    decisions = [r for r in native if r["type"] == "access_decision"]
    responses, requests = rows(run / "api_responses.jsonl"), rows(run / "api_requests.jsonl")
    assert [r["call"] for r in requests] == list(range(1, manifest["api_calls_started"] + 1))
    assert len(responses) == len(requests) and len({r["call"] for r in responses}) == len(responses)
    assert {r["call"] for r in requests} == {r["call"] for r in responses}
    assert len(actions) == manifest["actions"] and len(tools) == manifest["tool_calls"]
    assert len(requests) <= manifest["request_budget"]
    assert all(r["model_requested"] == manifest["model_requested"] for r in requests)
    assert sum("error_type" in r for r in responses) == manifest["api_calls_failed"]
    for metric in ("input_tokens", "output_tokens"):
        assert sum(r.get("response", {}).get("usage", {}).get(metric, 0) for r in responses) == manifest[metric]

    # Recompute post-run rule matches using each acquisition's saved official
    # source/task/catalog, not the mutable checkout or another run's source.
    saved = run / "source_snapshot/upstream"
    if any(name == "multi_agent" or name.startswith("multi_agent.") for name in sys.modules):
        raise RuntimeError("use a fresh audit process for frozen-source imports")
    sys.path.insert(0, str(saved))
    # A namespace directory alone can lose to a regular package later on
    # sys.path. Explicitly bind the frozen component root before any imports.
    package = types.ModuleType("multi_agent")
    package.__path__ = [str(saved / "multi_agent")]
    package.__package__ = "multi_agent"
    package.__spec__ = ModuleSpec("multi_agent", loader=None, is_package=True)
    sys.modules["multi_agent"] = package
    loader = importlib.import_module("multi_agent.loader")
    checker = importlib.import_module("multi_agent.checker")
    ObservableAction = importlib.import_module("multi_agent.schemas.actions").ObservableAction
    task_paths = list((saved / "multi_agent/tasks").rglob(manifest["task_id"] + ".yaml"))
    assert len(task_paths) == 1
    task, _ = loader.load_task_with_tools(task_paths[0])
    replay = []
    for event in actions:
        payload = {key: event[key] for key in ("sequence_num", "agent_id", "agent_role", "timestamp")}
        payload.update(run_id=event["trace_id"], surface=event["type"])
        for key in ("tool_name", "tool_args", "tool_result", "raw_event", "target_agent", "target_role", "content"):
            if key in event:
                payload[key] = event[key]
        replay.append(ObservableAction(**payload))
    expected = [{"action_sequence_num": d.action_sequence_num, "violated": d.violated,
                 "violation_code": d.violation_code.value if d.violation_code else None,
                 "severity": d.severity, "rule_description": d.rule.description if d.rule else None}
                for d in checker.check_trace(replay, task.access_rules, domain=task.domain)]
    actual = [{key: d.get(key) for key in ("action_sequence_num", "violated", "violation_code", "severity", "rule_description")} for d in decisions]
    assert Counter(map(canonical, expected)) == Counter(map(canonical, actual)), "post-run audit does not reproduce"
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("multi_agent.") and path:
            assert Path(path).resolve().is_relative_to(saved), "non-frozen upstream dependency"
    lifecycle = verify_lifecycle(rows(run / "tool_lifecycle.jsonl"), tools, responses) if (run / "tool_lifecycle.jsonl").exists() else None
    if lifecycle:
        assert lifecycle["started"] == manifest["tool_invocations_started"]
        assert lifecycle["returned"] == manifest["tool_handlers_returned"]
        assert lifecycle["failed"] == manifest["tool_handlers_failed"]
    shared = None
    if manifest.get("runtime_journal_schema") is not None:
        assert manifest["runtime_journal_schema"] == "runtime-journal-v1"
        shared = verify_shared_journal(rows(run / "runtime_events.jsonl"), rows(run / "arrival_events.jsonl"), rows(run / "tool_lifecycle.jsonl"))
        assert shared["journal_events"] == manifest["runtime_journal_events"]
        assert shared["run_status"] == manifest["status"]
    report = {"status": "passed", "run": str(run), "run_status": manifest["status"],
              "artifacts_checked": len(manifest["artifact_sha256"]), "api_calls": len(requests),
              "observed_actions": len(actions), "tool_calls": len(tools), "audit_decisions_recomputed": len(expected),
              "violated_actions": len({d["action_sequence_num"] for d in expected if d["violated"]}),
              "violation_rule_matches": sum(d["violated"] for d in expected), "lifecycle": lifecycle, "shared_journal": shared,
              "scope": "Acquisition integrity and post-run rule reproduction only; no monitor effectiveness or causal safety result."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.out), ensure_ascii=False))
