"""Protocol fixtures only: preserve and audit partial execution, never label it."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_crossed_purchase import run_flow
from collect_exclusive_purchase import read, write, sha
from paperalpha_runtime import crossed_purchase as study
from score_crossed_purchase import score
from verify_crossed_purchase import verify_flow
from test_crossed_purchase import models, reply


def partial_run(out, failure="transport", *, first_abstains=False):
    case = study.cases()[0]; root = models(out)
    async def api(request):
        payload = json.loads(request["input"][0]["content"])
        if "assignment" not in payload:
            return reply(json.dumps({"priority": "balanced", "work_order": "Unit-test protocol fixture."}))
        assignment = payload["assignment"]
        if assignment["actor"] == case["branches"][1]["actor"]:
            if failure == "transport": raise RuntimeError("unit-test unavailable endpoint")
            if failure == "incomplete": return {"status": "incomplete", "output": []}
            if failure == "refusal": return {"status": "completed", "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "refusal", "refusal": "Unit-test refusal."}]}]}
            if failure == "malformed": return reply("not JSON")
            if failure == "unknown_sku": return reply(json.dumps({"decision": "propose", "sku": "not-listed", "note": "Unit-test malformed choice."}))
        if first_abstains and assignment["actor"] == case["branches"][0]["actor"]:
            return reply(json.dumps({"decision": "abstain", "sku": None, "note": "Unit-test abstention."}))
        return reply(json.dumps({"decision": "propose", "sku": assignment["catalogue"][0]["sku"], "note": "Unit-test choice."}))
    async def forecast(run, directory, plan):
        write(directory / "FORECAST.json", score(read(directory / "PREFIX.json")["value"], root))
    report = asyncio.run(run_flow(case, out, {}, api, forecast))
    directory = out / "runs" / case["id"]
    return case, directory, report


@pytest.mark.parametrize("failure", ["transport", "incomplete", "refusal", "malformed", "unknown_sku"])
@pytest.mark.parametrize("first_abstains", [False, True])
def test_second_buyer_failure_preserves_verified_first_event_without_label(tmp_path, failure, first_abstains):
    case, directory, report = partial_run(tmp_path, failure, first_abstains=first_abstains)
    original = {p.name: sha(p) for p in directory.glob("*.json")}
    row = verify_flow(case, tmp_path, sha(directory / "REPORT.json"))
    assert not row["complete"] and not report["complete"] and report["api_calls"] == 3
    assert "label" not in row and not (directory / "OUTCOME.json").exists()
    assert row["forecast"] is not None
    assert row["protocol_audit"]["requests"] == 3
    assert row["protocol_audit"]["responses"] == (2 if failure == "transport" else 3)
    assert row["protocol_audit"]["parsed_proposals"] == row["protocol_audit"]["runtime_events"] == 1
    assert len(row["verified_runtime_events"]) == 1
    assert row["verified_runtime_events"][0]["actor"] == case["branches"][0]["actor"]
    assert row["verified_runtime_events"][0]["kind"] == ("abstention" if first_abstains else "simulated_dispatch")
    assert original == {p.name: sha(p) for p in directory.glob("*.json")}


def replace_records(directory, records, report):
    """Re-sign test-owned artifacts so the semantic audit, not hashes, is tested."""
    for i, record in enumerate(records, 1):
        record["sequence"] = i; record["monotonic_ns"] = i
        (directory / f"record_{i:04d}.json").write_text(json.dumps(record), encoding="utf-8")
    report["files_sha256"] = {p.name: sha(p) for p in directory.glob("*.json") if p.name != "REPORT.json"}
    (directory / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    return sha(directory / "REPORT.json")


@pytest.mark.parametrize("tamper", ["response_choice", "response_refusal", "parsed_command", "execution_amount",
    "execution_actor", "buyer_request", "next_request_before_event", "failure_not_last", "duplicate_response",
    "extra_dispatch", "wrong_failure_type", "wrong_forecast_flag"])
def test_partial_execution_tampering_rejected_even_with_updated_hashes(tmp_path, tamper):
    case, directory, report = partial_run(tmp_path)
    records = [read(p) for p in sorted(directory.glob("record_*.json"))]
    first = case["branches"][0]
    response = next(r for r in records if r["kind"] == "response" and r["stage"] == first["actor"])
    parsed = next(r for r in records if r["kind"] == "parsed_proposal")
    event = next(r for r in records if r["kind"] == "runtime_event")
    request2 = next(r for r in records if r["kind"] == "request" and r["stage"] == case["branches"][1]["actor"])
    if tamper == "response_choice":
        response["response"] = reply(json.dumps({"decision": "propose", "sku": first["catalogue"][1]["sku"], "note": "Changed choice."}))
    if tamper == "response_refusal":
        response["response"] = {"status": "completed", "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "refusal", "refusal": "Changed refusal."}]}]}
    if tamper == "parsed_command": parsed["command"] = None
    if tamper == "execution_amount": event["event"]["tool_result"]["charge_units"] = "999"
    if tamper == "execution_actor": event["event"]["actor"] = case["branches"][1]["actor"]
    if tamper == "buyer_request": request2["request"]["instructions"] += " Changed instructions."
    if tamper == "next_request_before_event":
        i, j = records.index(event), records.index(request2); records[i], records[j] = records[j], records[i]
    if tamper == "failure_not_last": records[-1], records[-2] = records[-2], records[-1]
    if tamper == "duplicate_response": records[records.index(parsed)] = copy.deepcopy(response)
    if tamper == "extra_dispatch": records.append(copy.deepcopy(event))
    if tamper == "wrong_failure_type": report["failure_type"] = "ValueError"
    if tamper == "wrong_forecast_flag": report["forecast_made"] = False
    expected = replace_records(directory, records, report)
    with pytest.raises(ValueError): verify_flow(case, tmp_path, expected)


def test_manifest_must_include_every_record(tmp_path):
    case, directory, report = partial_run(tmp_path)
    name = next(n for n in report["files_sha256"] if n.startswith("record_"))
    del report["files_sha256"][name]
    (directory / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        verify_flow(case, tmp_path, sha(directory / "REPORT.json"))
