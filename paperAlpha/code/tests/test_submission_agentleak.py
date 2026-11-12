"""Development collector tests use fake transport only; no API or old holdout."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import agentleak_submission as study
import collect_submission_agentleak as runner


@pytest.fixture
def cohort(tmp_path):
    out = tmp_path / "agentleak"
    plan = runner.prepare(out)
    return out, plan


class FakeTransport:
    def __init__(self, fail=False, invalid=False, duplicate=False, refuse=False):
        self.requests = []
        self.fail, self.invalid, self.duplicate, self.refuse = fail, invalid, duplicate, refuse
        self.active = self.peak = 0

    async def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        number = len(self.requests)
        self.active += 1
        self.peak = max(self.active, self.peak)
        await asyncio.sleep(0)
        self.active -= 1
        if self.fail:
            raise RuntimeError("SECRET-DO-NOT-PERSIST")
        content = {"type": "refusal", "refusal": "I cannot comply."} if self.refuse else {
            "type": "output_text", "text": f"completed fixture response {number}"}
        return {"id": "same" if self.duplicate else f"fake-{number}", "model": "gpt-5.5",
                "status": "incomplete" if self.invalid else "completed", "usage": {"output_tokens": 12},
                "output": [{"type": "reasoning", "summary": [{"text": "HIDDEN-DO-NOT-PERSIST"}],
                            "encrypted_content": "ENCRYPTED-DO-NOT-PERSIST"},
                           {"type": "message", "role": "assistant", "status": "completed", "content": [content]}]}


def test_prepare_covers_all_templates_and_complete_multirecord_values(cohort):
    out, plan = cohort
    assert len(plan["cases"]) == 16
    assert len({c["template_id"] for c in plan["cases"]}) == 16
    assert plan["maximum_api_calls"] == 48
    assert plan["upstream_commit"] == study.EXPECTED_UPSTREAM_COMMIT
    for case in plan["cases"]:
        assert len(case["records"]) == 3
        assert len({r["record_id"] for r in case["records"]}) == 3
        assert [r["fields"] for r in case["records"]] == [r["fields"] for r in case["author_scenario"]["private_vault"]["records"]]
        assert len(case["field_registry"]) == sum(len(r["fields"]) for r in case["records"])
        assert sum(f["is_canary"] for f in case["field_registry"]) > 5
        assert case["author_scenario"]["attack"]["enabled"] is False
    assert runner.load_plan(out) == plan
    assert not any("holdout" in name for name in plan["source_sha256"])
    with pytest.raises(ValueError, match="new cohort"):
        runner.prepare(out)


def test_actual_collection_is_48_once_and_concurrency_two(cohort):
    out, plan = cohort
    transport = FakeTransport()
    report = asyncio.run(runner.collect(out, transport))
    assert report["api_attempts"] == 48
    assert report["completed_cases"] == 16
    assert len(transport.requests) == 48 and transport.peak == 2
    assert all(r["model"] == "gpt-5.5" and r["max_output_tokens"] == 2400 and r["reasoning"] == {"effort": "low"} for r in transport.requests)
    rerun = FakeTransport()
    assert asyncio.run(runner.collect(out, rerun))["api_attempts"] == 48
    assert not rerun.requests
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in (out / "cases").rglob("*.json"))
    assert "HIDDEN-DO-NOT-PERSIST" not in persisted and "ENCRYPTED-DO-NOT-PERSIST" not in persisted
    assert '"output_tokens": 12' in persisted
    trace = runner.read(out / "cases" / plan["cases"][0]["case_id"] / "TRACE.json")
    assert [row["channel"] for row in trace["channels"]] == ["C2", "C2", "C5", "C1"]
    assert trace["channels"][2]["synthetic_cache"] is True
    view = study.monitor_prefix(plan["cases"][0], trace["stages"][:2])
    assert len(view["observations"]) == 2
    assert "labels" not in json.dumps(view)
    assert "author_scenario" not in view and "attack" not in view and "evaluation" not in view
    assert "case_id" not in view and "template_id" not in view
    assert trace["stages"][2]["delivered_text"] not in json.dumps(view)
    assert study.monitor_prefix(plan["cases"][0], [])["observations"] == []


def test_prefix_is_actual_past_and_policy_never_enters_request(cohort):
    _, plan = cohort
    case = copy.deepcopy(plan["cases"][0])
    case["normative_policy"]["allowed_fields"].append("EVALUATOR_ONLY_CANARY")
    case["author_scenario"]["evaluation"]["canaries"].append("EVALUATOR_ONLY_CANARY")
    messages = []
    for index in range(3):
        request = study.stage_request(case, index, messages)
        encoded = json.dumps(request)
        assert "EVALUATOR_ONLY_CANARY" not in encoded
        assert "normative_policy" not in encoded and "allowed_fields" not in encoded
        assert "FUTURE_STAGE_SECRET" not in encoded
        if index < 2:
            assert all(r["record_id"] in encoded for r in case["records"])
        else:
            assert "record-03" not in encoded
            assert messages[-1] in encoded and messages[0] not in encoded
        messages.append(f"ACTUAL_STAGE_{index}_UNIQUE")
    with pytest.raises(ValueError):
        study.stage_request(case, 1, ["past", "FUTURE_STAGE_SECRET"])


def test_all_failures_remain_unknown_never_safe_and_never_retry(cohort):
    out, _ = cohort
    transport = FakeTransport(fail=True)
    report = asyncio.run(runner.collect(out, transport))
    assert len(transport.requests) == 16
    assert report["api_attempts"] == 16 and report["completed_cases"] == 0
    assert all(c["final_label"] is None and c["status"] == "failed" for c in report["cases"])
    rerun = FakeTransport()
    asyncio.run(runner.collect(out, rerun))
    assert not rerun.requests
    persisted = "\n".join(p.read_text(encoding="utf-8") for p in (out / "cases").rglob("*.json"))
    assert "SECRET-DO-NOT-PERSIST" not in persisted


def test_incomplete_response_stops_case(cohort):
    out, _ = cohort
    report = asyncio.run(runner.collect(out, FakeTransport(invalid=True)))
    assert report["api_attempts"] == 16
    assert all(c["final_label"] is None for c in report["cases"])
    assert all(c["status"] == "invalid_response" for c in report["cases"])


def write_receipt(out, case, plan, index, messages, seq, response=None):
    request = study.stage_request(case, index, messages)
    before = study.prefix(case, messages)
    folder = runner.stage_folder(out, case, index)
    runner.durable_new(folder / "attempt.json", {"plan_sha256": plan["sha256"], "request": request,
        "prefix_before": before, "prefix_sha256": study.digest(before), "request_seq": seq})
    if response is not None:
        runner.durable_new(folder / "response.json", {"request_sha256": study.digest(request), "response": response})


def test_resume_uncertain_attempt_never_resends(cohort):
    out, plan = cohort
    case = plan["cases"][0]
    write_receipt(out, case, plan, 0, [], 1)
    transport = FakeTransport()
    report = asyncio.run(runner.collect(out, transport))
    assert report["api_attempts"] == 46 and len(transport.requests) == 45
    assert report["cases"][0]["status"] == "attempted_unknown"
    assert report["cases"][0]["final_label"] is None


def test_resume_completed_receipt_continues_next_stage_without_resend(cohort):
    out, plan = cohort
    case = plan["cases"][0]
    response = {"id": "prior-response", "model": "gpt-5.5", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Prior real message"}]}]}
    write_receipt(out, case, plan, 0, [], 1, response)
    transport = FakeTransport()
    report = asyncio.run(runner.collect(out, transport))
    assert report["api_attempts"] == 48 and len(transport.requests) == 47
    assert any("Prior real message" in r["input"][0]["content"] for r in transport.requests)


def test_failure_receipt_cannot_be_overwritten_by_success(cohort):
    out, plan = cohort
    case = plan["cases"][0]
    response = {"id": "prior-response", "model": "gpt-5.5", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Prior real message"}]}]}
    write_receipt(out, case, plan, 0, [], 1, response)
    runner.durable_new(runner.stage_folder(out, case, 0) / "failure.json", {"error_type": "ValueError", "retry_allowed": False})
    trace = runner.case_trace(out, case, plan)
    assert trace["status"] == "failed_after_response"
    assert trace["labels"]["final_literal_policy_violation"] is None
    assert trace["channels"] == []


def test_multirecord_labels_and_exemptions(cohort):
    _, plan = cohort
    case = plan["cases"][0]
    forbidden = [f for f in case["field_registry"] if f["field_name"] == "ssn"]
    assert len(forbidden) == 3
    text = forbidden[2]["value"]
    result = study.literal_labels(case, text)
    assert result["y"] == 1
    hits = [f for f in result["fields"] if f["violation"]]
    assert [(f["record_id"], f["field_name"]) for f in hits] == [("record-03", "ssn")]
    changed = copy.deepcopy(case)
    changed["normative_policy"]["allowed_fields"].append("ssn")
    assert study.literal_labels(changed, text)["y"] == 0
    changed = copy.deepcopy(case)
    changed["request"] += " " + text
    assert study.literal_labels(changed, text)["y"] == 0
    assert study.literal_labels(case, None) is None


def test_native_refusal_is_recorded_not_transport_failure(cohort):
    out, _ = cohort
    report = asyncio.run(runner.collect(out, FakeTransport(refuse=True)))
    assert report["completed_cases"] == 16
    assert report["api_attempts"] == 48


def test_duplicate_response_ids_are_disclosed(cohort):
    out, _ = cohort
    report = asyncio.run(runner.collect(out, FakeTransport(duplicate=True)))
    assert len(report["response_identity_exclusions"]) == 16


def test_modified_plan_and_snapshot_rejected(cohort):
    out, plan = cohort
    snapshot = out / "source" / next(iter(plan["source_sha256"]))
    snapshot.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.load_plan(out)


def test_endpoint_is_local_only():
    assert runner.validate_endpoint("http://localhost:58661/v1")
    for endpoint in ("https://example.com/v1", "http://user:secret@localhost:58661/v1", "http://localhost:1/v1"):
        with pytest.raises(ValueError):
            runner.validate_endpoint(endpoint)
