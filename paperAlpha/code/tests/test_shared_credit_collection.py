"""Offline preparation and four retained fake trajectories, never an API."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'code/commands'))
import collect_shared_credit_pilot as collector
from test_shared_credit_pilot import reply, no_network


def test_offline_prepare_freezes_exact_four_cases_and_complete_source(tmp_path):
    out=tmp_path/"pilot"
    summary=collector.prepare(out)
    assert summary["api_calls"]==0 and summary["maximum_requests"]==48
    plan=collector.verify(out)
    assert len(plan["cases"])==4
    assert len({c["case_id"] for c in plan["cases"]})==4
    assert "upstream/multi_agent/banks/ecommerce.py" in plan["source_sha256"]
    assert "local/code/tests/test_shared_credit_pilot.py" in plan["source_sha256"]
    assert plan["environment"]["python"]==sys.version
    assert all(c["policy"]["per_call_cap_cents"]==5000 and c["policy"]["order_total_cap_cents"]==7500 for c in plan["cases"])
    with pytest.raises(ValueError,match="already exists"):collector.prepare(out)


def test_all_four_cases_retained_and_failure_not_resampled(tmp_path):
    out=tmp_path/"pilot";collector.prepare(out);calls=[]
    async def provider(actor,request):
        calls.append((actor,request))
        if len(calls)==1:raise TimeoutError("fake only")
        return reply("unique-"+str(len(calls)),text="Actual fake model chose zero calls.")
    report=asyncio.run(collector.collect_all(out,provider))
    assert report["fixed_case_count"]==len(report["cases"])==4
    assert report["actual_requests"]==8 and report["completed_cases"]==3
    assert len(list((out/"cases").glob("*/private/ORACLE.json")))==4
    with pytest.raises(FileExistsError):asyncio.run(collector.collect_all(out,provider))
    assert len(calls)==8


def test_frozen_copy_tamper_is_detected(tmp_path):
    out=tmp_path/"pilot";collector.prepare(out)
    changed=out/"source/local/code/src/paperalpha_runtime/shared_credit_pilot.py"
    changed.write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="source changed"):collector.verify(out)


@pytest.mark.parametrize("endpoint",["https://example.org/v1","http://localhost:9999/v1","http://user:pass@localhost:58661/v1"])
def test_no_unapproved_api_transport(endpoint,tmp_path):
    with pytest.raises(ValueError,match="loopback"):
        asyncio.run(collector.run_api(tmp_path,endpoint,"not-a-plan"))


def test_cross_case_duplicate_response_marks_provenance_invalid_keeps_four(tmp_path):
    out=tmp_path/"pilot";collector.prepare(out);calls=[]
    async def provider(actor,request):
        calls.append(actor)
        identity="duplicate" if len(calls) in (1,3) else "unique-"+str(len(calls))
        return reply(identity,text="No action.")
    report=asyncio.run(collector.collect_all(out,provider))
    assert len(report["cases"])==4 and report["actual_requests"]==8
    assert report["completed_cases"]==3 and report["unique_actual_response_ids"]==7
    assert report["eligible_for_effectiveness"] is False
    assert report["cases"][1]["provenance_valid"] is False
    oracle=json.loads((out/"cases/partitioned_same_round/private/ORACLE.json").read_text())
    assert oracle["eligible_for_effectiveness"] is False and oracle["observed_cumulative_violation"] is None


def test_interruption_preserves_all_planned_terminal_denominators(tmp_path):
    out=tmp_path/"pilot";collector.prepare(out);calls=[]
    async def provider(actor,request):
        calls.append(actor)
        raise asyncio.CancelledError("fake cancellation")
    with pytest.raises(asyncio.CancelledError):asyncio.run(collector.collect_all(out,provider))
    report=json.loads((out/"private/COLLECTION.json").read_text())
    assert len(report["cases"])==4 and len(calls)==1
    assert [c["status"] for c in report["cases"]]==["interrupted"]+["not_attempted"]*3
    assert report["actual_requests"]==1 and not report["eligible_for_effectiveness"]
    assert report["completed_cases"]==0
