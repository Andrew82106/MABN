"""Synthetic initial-only contexts and fake transport; never a real API call."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
import collect_submission_semantics as runner


def make_parent(tmp_path):
    cases = []
    for index, (prefix, suffix) in enumerate(runner.SHAPES):
        cases.append({"case_id": f"fixture-{index:02d}", "request": prefix + f"Subject {index}" + suffix,
            "records": [{"record_id": f"record-{j}", "fields": {
                "name": f"PRIVATE-VALUE-{index}-{j}", "items": ["PRIVATE-LIST-VALUE"],
                "details": {"hidden": "PRIVATE-DICT-VALUE"}, "amount": 42,
                "active": True, "missing": None}} for j in range(3)],
            "normative_policy": {"allowed_fields": ["POLICY-NEVER-SEND"]},
            "author_scenario": {"evaluation": "EVALUATION-NEVER-SEND"}})
    plan = {"schema": "submission-agentleak-development-v1", "cases": cases}
    plan["sha256"] = runner.digest(plan)
    path = tmp_path / "parent" / "PLAN.json"
    runner.durable_new(path, plan)
    runner.durable_new(path.parent / "cases/TRACE.json", {"future": "FUTURE-NEVER-READ"})
    return path, plan


@pytest.fixture
def cohort(tmp_path):
    parent, _ = make_parent(tmp_path)
    out = tmp_path / "semantics"
    return out, runner.prepare(out, parent)


class FakeTransport:
    def __init__(self, *, fail=False, incomplete=False, bad_annotation=False,
                 duplicate=False, refuse=False, wrong_model=False, invalid_id=False):
        self.requests = []
        self.active = self.peak = 0
        self.fail, self.incomplete, self.bad_annotation = fail, incomplete, bad_annotation
        self.duplicate, self.refuse = duplicate, refuse
        self.wrong_model, self.invalid_id = wrong_model, invalid_id

    async def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        number = len(self.requests)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if self.fail:
            raise RuntimeError("SECRET-ERROR-TEXT-NEVER-SAVE")
        context = json.loads(request["input"][0]["content"])
        annotation = runner.unknown_annotation(context)
        for item in annotation["fields"]:
            item.update(internal_use="needed", outward_form="derived_or_aggregate",
                        reason="Synthetic test hypothesis.")
        if self.bad_annotation:
            annotation["fields"].pop()
        content = ({"type": "refusal", "refusal": "Cannot classify."} if self.refuse else
                   {"type": "output_text", "text": json.dumps(annotation)})
        return {"id": [] if self.invalid_id else "same-id" if self.duplicate else f"fake-{number}",
            "model": "other" if self.wrong_model else "gpt-5.5",
            "status": "incomplete" if self.incomplete else "completed", "created_at": 1,
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "output": [{"type": "reasoning", "summary": [{"text": "HIDDEN-NEVER-SAVE"}],
                        "encrypted_content": "ENCRYPTED-NEVER-SAVE"},
                       {"type": "message", "role": "assistant", "status": "completed", "content": [content]}]}


def test_prepare_freezes_sixteen_typed_initial_contexts_only(cohort):
    out, plan = cohort
    assert len(plan["contexts_sha256"]) == len(plan["bindings"]) == 16
    assert plan["api"]["maximum_api_calls"] == 16
    assert plan["api"]["timeout_seconds"] == 90
    assert len(plan["source_sha256"]) == 7
    assert all("law" not in name and "projection" not in name for name in plan["source_sha256"])
    for identity in plan["contexts_sha256"]:
        context = runner.read(out / "contexts" / (identity + ".json"))
        assert runner.digest(context) == identity
        assert {f["name"]: f["type"] for f in context["fields"]} == {
            "name": "string", "items": "array", "details": "object", "amount": "number",
            "active": "boolean", "missing": "null"}
        payload = json.dumps(runner.request_payload(context))
        assert all(secret not in payload for secret in (
            "PRIVATE-", "POLICY-NEVER", "EVALUATION-NEVER", "FUTURE-NEVER", "fixture-"))
    assert runner.verify(out)["api_attempts"] == 0
    assert runner.annotation_for_case(out, "fixture-00") is None
    with pytest.raises(ValueError, match="new output"):
        runner.prepare(out, plan["parent_plan_path"])


def test_sixteen_once_concurrency_two_public_only_and_bound_annotation(cohort):
    out, plan = cohort
    transport = FakeTransport()
    result = asyncio.run(runner.collect(out, transport))
    assert result["api_attempts"] == result["usable_hypotheses"] == 16
    assert len(transport.requests) == 16 and transport.peak == 2
    assert all(r["model"] == "gpt-5.5" and r["reasoning"] == {"effort": "low"}
               and r["max_output_tokens"] == 6000 and r["store"] is False for r in transport.requests)
    annotation = runner.annotation_for_case(out, "fixture-00")
    assert set(annotation) == {"context", "annotation", "record_id", "availability"}
    assert annotation["availability"] == "offline_initial_only"
    assert annotation["record_id"].startswith("submission-semantics:")
    binding = plan["bindings"][0]
    assert runner.digest(annotation["context"]) == binding["context_sha256"]
    assert runner.validate_annotation(annotation["context"], annotation["annotation"])
    persisted = "\n".join(p.read_text(encoding="utf-8") for directory in (
        "attempts", "responses", "failures", "classifications") for p in (out / directory).glob("*.json"))
    assert "HIDDEN-NEVER-SAVE" not in persisted and "ENCRYPTED-NEVER-SAVE" not in persisted
    assert '"total_tokens": 30' in persisted
    repeat = FakeTransport()
    assert asyncio.run(runner.collect(out, repeat)) == result
    assert not repeat.requests


@pytest.mark.parametrize("flag,status", [("fail", "api_error"), ("incomplete", "invalid_response"),
    ("bad_annotation", "invalid_annotation"), ("refuse", "invalid_response"),
    ("wrong_model", "invalid_response"), ("invalid_id", "invalid_response")])
def test_failure_invalid_and_refusal_are_unknown_not_success(cohort, flag, status):
    out, plan = cohort
    fake = FakeTransport(**{flag: True})
    result = asyncio.run(runner.collect(out, fake))
    assert result["api_attempts"] == 16 and result["usable_hypotheses"] == 0
    assert result["statuses"] == {status: 16}
    assert runner.annotation_for_case(out, "fixture-00") is None
    for identity in plan["contexts_sha256"]:
        row = runner.classification(out, identity, plan)
        assert row["fallback_used"] and row["record_id"] is None
        assert all(f["internal_use"] == f["outward_form"] == "unclear" for f in row["annotation"]["fields"])
    repeat = FakeTransport()
    asyncio.run(runner.collect(out, repeat))
    assert not repeat.requests
    assert not any("SECRET-ERROR-TEXT" in p.read_text(encoding="utf-8") for p in (out / "failures").glob("*.json"))


def claim(out, identity, plan, response=None, *, failure=False):
    name = identity + ".json"
    request = runner.request_payload(runner.read(out / "contexts" / name))
    runner.durable_new(out / "attempts" / name, {"plan_sha256": plan["sha256"],
        "context_sha256": identity, "request_seq": 1, "attempted_at": runner.now(),
        "request_sha256": runner.digest(request), "request": request})
    if response is not None:
        runner.durable_new(out / "responses" / name, {"plan_sha256": plan["sha256"],
            "context_sha256": identity, "received_at": runner.now(),
            "request_sha256": runner.digest(request), "response": response})
    if failure:
        runner.durable_new(out / "failures" / name, {"at": runner.now(),
            "request_sha256": runner.digest(request), "error_type": "ValueError", "retry_allowed": False})


def test_crash_claim_without_response_never_resends(cohort):
    out, plan = cohort
    identity = sorted(plan["contexts_sha256"])[0]
    claim(out, identity, plan)
    fake = FakeTransport()
    result = asyncio.run(runner.collect(out, fake))
    assert len(fake.requests) == 15 and result["api_attempts"] == 16
    assert result["usable_hypotheses"] == 15
    assert result["statuses"]["attempted_unknown"] == 1


def test_completed_response_rebuilds_after_crash_without_resend(cohort):
    out, plan = cohort
    identity = sorted(plan["contexts_sha256"])[0]
    request = runner.request_payload(runner.read(out / "contexts" / (identity + ".json")))
    response = runner.canonical_response(asyncio.run(FakeTransport()(request)))
    response["id"] = "prior-unique"
    claim(out, identity, plan, response)
    fake = FakeTransport()
    result = asyncio.run(runner.collect(out, fake))
    assert len(fake.requests) == 15 and result["usable_hypotheses"] == 16


def test_failure_cannot_be_overwritten_by_valid_response(cohort):
    out, plan = cohort
    identity = sorted(plan["contexts_sha256"])[0]
    request = runner.request_payload(runner.read(out / "contexts" / (identity + ".json")))
    response = runner.canonical_response(asyncio.run(FakeTransport()(request)))
    claim(out, identity, plan, response, failure=True)
    row = runner.classification(out, identity, plan)
    assert row["status"] == "failed_after_response" and row["fallback_used"]


def test_duplicate_responses_excluded_from_annotation_delivery(cohort):
    out, _ = cohort
    result = asyncio.run(runner.collect(out, FakeTransport(duplicate=True)))
    assert len(result["response_identity_exclusions"]) == 16
    assert result["usable_hypotheses"] == 0
    assert runner.annotation_for_case(out, "fixture-00") is None


def test_parent_context_and_source_tampering_rejected(cohort):
    out, plan = cohort
    source = out / "source" / runner.SOURCES[0]
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify(out)
    source.write_bytes(original)
    parent = Path(plan["parent_plan_path"])
    parent.write_bytes(parent.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="parent PLAN bytes"):
        runner.verify(out)


def test_receipt_request_binding_and_classification_tampering_rejected(cohort):
    out, plan = cohort
    asyncio.run(runner.collect(out, FakeTransport()))
    identity = next(iter(plan["contexts_sha256"]))
    target = out / "classifications" / (identity + ".json")
    value = runner.read(target)
    value["annotation"]["fields"][0]["internal_use"] = "not_needed"
    runner.replace_summary(target, value)
    with pytest.raises(ValueError, match="stored classification"):
        runner.verify(out)
    runner.replace_summary(target, runner.classification(out, identity, plan))
    receipt = out / "responses" / (identity + ".json")
    value = runner.read(receipt)
    value["request_sha256"] = "wrong"
    runner.replace_summary(receipt, value)
    with pytest.raises(ValueError, match="response binding"):
        runner.verify(out)


def test_heterogeneous_schema_rejected_before_output_creation(tmp_path):
    parent, plan = make_parent(tmp_path)
    plan["cases"][0]["records"][1]["fields"]["items"] = "silently flattened"
    plan["sha256"] = runner.digest({k: v for k, v in plan.items() if k != "sha256"})
    runner.replace_summary(parent, plan)
    with pytest.raises(ValueError, match="heterogeneous"):
        runner.prepare(tmp_path / "semantics", parent)
    assert not (tmp_path / "semantics").exists()


def test_initial_only_reader_never_opens_trace_or_old_results(tmp_path, monkeypatch):
    parent, _ = make_parent(tmp_path)
    original_read = runner.read
    seen = []
    def initial_only(path):
        seen.append(Path(path).resolve())
        assert Path(path).resolve() == parent.resolve()
        return original_read(path)
    monkeypatch.setattr(runner, "read", initial_only)
    _, contexts, _ = runner.initial_contexts(parent)
    assert len(contexts) == 16 and seen == [parent.resolve()]


def test_local_endpoint_only():
    assert runner.validate_endpoint("http://localhost:58661/v1")
    for value in ("https://example.com/v1", "http://user:password@localhost:58661/v1",
                  "http://localhost:1/v1", "http://localhost:58661/v1?token=secret"):
        with pytest.raises(ValueError):
            runner.validate_endpoint(value)
