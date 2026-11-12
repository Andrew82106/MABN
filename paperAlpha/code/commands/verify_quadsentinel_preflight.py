"""Replay recorded replies through the pinned author guard; no API access."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


async def verify(run, case_id, out):
    plan = json.loads((run / "PLAN.json").read_text(encoding="utf-8"))
    sys.path[:0] = [str(run / "source/code/src"), plan["dependency_path"]]
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from paperalpha_runtime.quadsentinel_adapter import inspect_checkout, create_author_guard, observe_event, response_request
    from paperalpha_runtime.privacy_transfer import delivered_text
    from autogen_core.models import CreateResult, RequestUsage
    assert inspect_checkout(plan["checkout"]) == plan["upstream"]
    for name, expected in plan["sources_sha256"].items():
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == expected
    case = next(c for c in plan["cases"] if c["id"] == case_id)
    directory = run / "runs" / case_id
    report = json.loads((directory / "REPORT.json").read_text(encoding="utf-8"))
    assert report["completed"] and not report["failure"]
    records = []
    for name, expected in sorted(report["records_sha256"].items()):
        path = directory / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        records.append(json.loads(path.read_text(encoding="utf-8")))
    assert [r["sequence"] for r in records] == list(range(1, len(records) + 1))
    assert not any(r["kind"].endswith("failure") for r in records)
    requests = [r for r in records if r["kind"] == "api_request"]
    responses = {r["id"]: r for r in records if r["kind"] == "api_response"}
    expected_decisions = [r for r in records if r["kind"] == "guard_decision"]
    event_records = [r for r in records if r["kind"] == "observed_event"]
    assert len(requests) == len(responses) == report["api_requests"]
    assert [r["event"] for r in event_records] == case["events"]
    assert len(expected_decisions) == len(case["events"])
    # Verify event cutoffs: every prompt is made after that observation and
    # before its verdict; no prompt can read the following event's record.
    for i, event in enumerate(event_records):
        verdict = expected_decisions[i]
        assert event["sequence"] < verdict["sequence"]
        if i + 1 < len(event_records):
            assert verdict["sequence"] < event_records[i + 1]["sequence"]
    for req in requests:
        response = responses[req["id"]]
        assert req["sequence"] < response["sequence"]
        assert req["monotonic_ns"] <= response["monotonic_ns"]
        assert response["response"]["status"] == "completed"

    class ReplayClient:
        calls = 0
        async def create(self, *, messages, **kwargs):
            req = requests[self.calls]
            assert response_request(messages) == req["request"]
            self.calls += 1
            response = responses[req["id"]]["response"]
            reply = delivered_text(response)
            assert not reply["native_refusal"]
            return CreateResult(content=reply["text"], finish_reason="stop", cached=False,
                                usage=RequestUsage(prompt_tokens=0, completion_tokens=0))

    client = ReplayClient()
    guard = await create_author_guard(plan["checkout"], client, case["policies"])
    guard.register_tool("publish_note", "Publish the supplied note to the given destination.")
    actual = []
    try:
        for event, expected in zip(case["events"], expected_decisions):
            allowed, reason = await observe_event(guard, event)
            result = {"allowed": allowed, "reason": reason,
                      "predicates": copy.deepcopy(guard.predicates), "threat_levels": copy.deepcopy(guard.threat_levels)}
            assert result == {k: expected[k] for k in result}
            actual.append(result)
    finally:
        await guard.stop()
    assert client.calls == len(requests)
    assert actual == report["decisions"]
    assert report["fixture_final_matches"] == (actual[-1]["allowed"] == case["fixture_expectation_final_allow"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"case_id": case_id, "author_commit": plan["upstream"]["commit"],
              "author_source_hashes_checked": len(plan["upstream"]["files_sha256"]),
              "journal_hashes_checked": len(records), "requests_replayed": client.calls,
              "decisions_recomputed": len(actual), "all_requests_and_states_identical": True,
              "new_api_calls": 0, "is_performance_benchmark": False,
              "plan_sha256": hashlib.sha256((run / "PLAN.json").read_bytes()).hexdigest(),
              "report_sha256": hashlib.sha256((directory / "REPORT.json").read_bytes()).hexdigest(),
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with out.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.run.resolve(), args.case, args.out.resolve()))
