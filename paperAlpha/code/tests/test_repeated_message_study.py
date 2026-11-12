"""Fake API protocol tests only: no model quality or real-API claims."""
import asyncio
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import repeated_message_study as study
from paperalpha_monitor.grouped_event_learning import GroupedEventTrace
import collect_repeated_message_study as runner


class FakeAPI:
    def __init__(self, *, fail_call=None, failure="transport", identity_issue=None, identity_calls=None):
        self.calls = []
        self.fail_call = fail_call
        self.failure = failure
        self.identity_issue = identity_issue
        self.identity_calls = identity_calls
        self.closed = False
        self.responses = SimpleNamespace(create=self.create)

    async def close(self):
        self.closed = True

    async def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        failed = len(self.calls) == self.fail_call
        if failed and self.failure == "transport":
            raise RuntimeError("fake network outage; sensitive exception body must not be persisted")
        payload = json.loads(request["input"][0]["content"])
        text = payload.get("source_record", payload.get("reviewed_packet"))
        response = {"id": "fixture-response-" + str(len(self.calls)), "model": "gpt-5.5",
                    "status": "incomplete" if failed and self.failure == "incomplete" else "completed",
                    "incomplete_details": {"reason": "max_output_tokens"} if failed and self.failure == "incomplete" else None,
                    "output": [{"type": "message", "role": "assistant", "status": "completed",
                                "content": [{"type": "refusal", "refusal": "Fixture refusal"}] if failed and self.failure == "refusal"
                                else [{"type": "output_text", "text": text}]}]}
        if self.identity_calls is None or len(self.calls) in self.identity_calls:
            if self.identity_issue == "missing":
                response["id"] = None
            elif self.identity_issue == "duplicate":
                response["id"] = "fixture-duplicate-response"
            elif self.identity_issue == "model":
                response["model"] = "gpt-5.5-undeclared-snapshot"
            elif self.identity_issue == "missing_model":
                response["model"] = None
        return SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(response))


def group_case(layout="shared"):
    return next(c for c in study.cases() if c["base_task"] == "equipment_collection" and c["repeat"] == 0 and c["layout"] == layout)


def collect_fixture(tmp_path, layout="shared", **kwargs):
    out = tmp_path / "cohort"
    runner.prepare(out)
    case = group_case(layout)
    api = FakeAPI(**kwargs)
    report = asyncio.run(runner.collect_group(out, case, api))
    verified = runner.verify_group(out, case)
    return out, case, api, report, verified


def test_fixed_budget_and_inputs_identical_across_repeats_layouts():
    cases = study.cases()
    assert len(cases) == 48 and len({c["id"] for c in cases}) == 48
    assert sum(study.planned_calls(c) for c in cases) == 264
    assert {c["split"] for c in cases} == {"development"}
    assert len({c["marker"] for c in cases}) == 4
    for task in study.base_tasks():
        subset = [c for c in cases if c["base_task"] == task["base_task"]]
        assert len(subset) == 12
        requests = [study.request(c, node, {"source": c["source"]}) for c in subset for node in study.reviewers(c)]
        assert all(r == requests[0] for r in requests)
        assert all(c["source"] == task["source"] for c in subset)
        for c in subset:
            assert c["id"] not in json.dumps(requests[0])


@pytest.mark.parametrize("layout,calls", [("shared", 5), ("separate", 6)])
def test_success_reconstructs_one_shared_label_set_and_two_continuations(tmp_path, layout, calls):
    out, case, api, report, group = collect_fixture(tmp_path, layout)
    assert len(api.calls) == calls == report["api_calls"]
    assert group["complete"] and group["eligible_for_grouped_fit"]
    assert group["eligible_for_statistics"] and group["response_identity_qualified"]
    assert group["response_identity_audit_scope"] == "within_group_only"
    assert len(group["shared_labels"]) == (1 if layout == "shared" else 2)
    assert len(group["continuations"]) == 2
    trace = GroupedEventTrace(group["dag"], group["shared_labels"], group["continuations"])
    assert trace is not None
    labels = [*group["shared_labels"].values(), *(v for c in group["continuations"] for v in c.values())]
    ids = [v["record_id"] for v in labels]
    assert len(ids) == len(set(ids))
    assert all(out.name + "/runs/" + case["id"] + "/record_" in identity for identity in ids)
    assert max(v["sequence"] for v in group["shared_labels"].values()) < min(v["sequence"] for c in group["continuations"] for v in c.values())
    for c in group["continuations"]:
        assert c["publish-a"]["sequence"] < c["publish-b"]["sequence"]
    # The exact same reviewer message is fed again; no earlier publisher reply.
    assert api.calls[-4] == api.calls[-2]
    assert api.calls[-3] == api.calls[-1]


@pytest.mark.parametrize("failure", ["transport", "incomplete", "refusal"])
def test_upstream_failure_is_not_resampled_or_negative_supervision(tmp_path, failure):
    out, case, api, report, group = collect_fixture(tmp_path, fail_call=1, failure=failure)
    assert len(api.calls) == 1
    assert not group["upstream_complete"] and not group["eligible_for_grouped_fit"]
    assert group["shared_labels"] == {} and group["continuations"] == []
    assert len(group["failures"]) == 1
    text = "".join(p.read_text(encoding="utf-8") for p in (out / "runs" / case["id"]).glob("*.json"))
    assert "sensitive exception body" not in text


@pytest.mark.parametrize("failure", ["transport", "incomplete", "refusal"])
def test_one_failed_publisher_keeps_other_actual_labels_and_next_pair(tmp_path, failure):
    out, case, api, report, group = collect_fixture(tmp_path, fail_call=2, failure=failure)
    assert len(api.calls) == 5
    assert group["upstream_complete"] and not group["complete"]
    assert len(group["continuations"]) == 1 and group["eligible_for_grouped_fit"]
    assert len(group["incomplete_continuations"]) == 1
    assert set(group["incomplete_continuations"][0]["labels"]) == {"publish-b"}
    assert GroupedEventTrace(group["dag"], group["shared_labels"], group["continuations"])


def test_prefixes_precede_calls_and_do_not_reuse_previous_continuation(tmp_path):
    out, case, api, report, group = collect_fixture(tmp_path)
    folder = out / "runs" / case["id"]
    records = [runner.read(p) for p in sorted(folder.glob("record_*.json"))]
    for i, record in enumerate(records):
        if record["kind"] == "request" and record["node"].startswith("publish-"):
            before = records[i - 1]
            assert before["kind"] == "prefix" and before["before_node"] == record["node"]
            dag = study.compile_dag(before["packet"])
            assert record["node"] not in dag.observations
            assert all(o["sequence"] < record["sequence"] for o in dag.observations.values())
            if record["node"] == "publish-a":
                assert not any(n.startswith("publish-") for n in dag.observations)


@pytest.mark.parametrize("tamper", ["response", "parent", "scope", "outcome"])
def test_replay_rejects_raw_or_rehashed_semantic_tampering(tmp_path, tamper):
    out, case, api, report, group = collect_fixture(tmp_path)
    folder = out / "runs" / case["id"]
    records = [runner.read(p) for p in sorted(folder.glob("record_*.json"))]
    target = next(r for r in records if r["kind"] == "request" and r["node"] == "publish-a")
    path = Path(target["record_id"])
    if tamper == "response":
        path = folder / "EARLY.json"
        value = runner.read(path)
        value["group_id"] = "wrong"
    elif tamper == "parent":
        value = target
        payload = json.loads(value["request"]["input"][0]["content"])
        payload["reviewed_packet"] = "Invented replacement parent"
        value["request"]["input"][0]["content"] = json.dumps(payload)
    elif tamper == "scope":
        value = target
        value["record_id"] = "record_0008.json"
    else:
        path = folder / "OUTCOME.json"
        value = runner.read(path)
        value["shared_labels"]["review-a"]["value"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")
    if tamper != "response":
        report["files_sha256"][path.name] = runner.sha(path)
        (folder / "REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.verify_group(out, case)


def test_existing_group_never_restarted(tmp_path):
    out, case, api, report, group = collect_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        asyncio.run(runner.collect_group(out, case, api))
    assert len(api.calls) == 5


def test_second_reviewer_failure_preserves_first_without_filling_second(tmp_path):
    out, case, api, report, group = collect_fixture(tmp_path, "separate", fail_call=2)
    assert len(api.calls) == 2
    assert set(group["shared_labels"]) == {"review-a"}
    assert not group["upstream_complete"] and not group["eligible_for_grouped_fit"]
    assert group["continuations"] == []


def test_entire_fake_collection_budget_read_only_replay_and_grouped_training_compatibility(tmp_path, monkeypatch):
    from paperalpha_runtime import quadsentinel_adapter
    out = tmp_path / "full_fake_protocol"
    plan = runner.prepare(out)
    prepare_environment = runner.read(out / "PREPARE_ENVIRONMENT.json")
    assert prepare_environment["stage"] == "prepare"
    assert plan["prepare_environment_sha256"] == runner.sha(out / "PREPARE_ENVIRONMENT.json")
    api = FakeAPI()
    monkeypatch.setattr(quadsentinel_adapter, "create_api_client", lambda endpoint, key: api)
    receipt = asyncio.run(runner.collect(out))
    assert api.closed and len(api.calls) == receipt["api_calls"] == 264
    collect_environment = runner.read(out / "COLLECT_ENVIRONMENT.json")
    assert collect_environment["stage"] == "collect" and collect_environment != prepare_environment
    assert receipt["collect_environment_sha256"] == runner.sha(out / "COLLECT_ENVIRONMENT.json")
    before = {str(p): runner.sha(p) for p in out.rglob("*.json")}
    result = runner.verify(out)
    assert result["groups"] == result["eligible_groups"] == result["complete_groups"] == 48
    assert result["complete_continuations"] == 96 and result["prospective_predictions"] == 0
    assert result["response_identity_audit"]["scope"] == "complete_collection"
    assert result["response_identity_audit"]["qualified_group_count"] == 48
    records = []
    for group in result["grouped_traces"]:
        assert GroupedEventTrace(group["dag"], group["shared_labels"], group["continuations"])
        records += [v["record_id"] for v in group["shared_labels"].values()]
        records += [v["record_id"] for c in group["continuations"] for v in c.values()]
    assert len(records) == len(set(records)) == 264
    assert before == {str(p): runner.sha(p) for p in out.rglob("*.json")}
    json.dumps(runner.serializable_verified(result))
    with pytest.raises(ValueError, match="not be restarted"):
        asyncio.run(runner.collect(out))
    assert len(api.calls) == 264


def test_plan_freezes_source_and_has_no_performance_claim(tmp_path):
    out = tmp_path / "freeze"
    plan = runner.prepare(out)
    assert runner.load_plan(out) == plan
    assert plan["predictions"] is None and plan["maximum_api_calls"] == 264
    assert 'code/src/paperalpha_monitor/grouped_event_learning.py' in plan["source_sha256"]
    assert 'code/tests/test_grouped_event_learning.py' in plan["source_sha256"]
    assert 'code/commands/diagnose_repeated_messages.py' in plan["source_sha256"]
    assert 'code/tests/test_repeated_message_diagnostics.py' in plan["source_sha256"]
    assert 'code/commands/evaluate_repeated_messages.py' in plan["source_sha256"]
    assert 'code/tests/test_repeated_message_evaluation.py' in plan["source_sha256"]
    frozen = out / "source/code/src/paperalpha_runtime/repeated_message_study.py"
    frozen.write_text(frozen.read_text(encoding="utf-8") + "\n# edited", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source"):
        runner.load_plan(out)


def test_environment_is_allowlisted_reports_missing_packages_and_hashes_actual_executable(monkeypatch):
    monkeypatch.setenv("DO_NOT_LOG_TEST_SECRET", "secret-canary-never-record")
    original = runner.metadata.version

    def versions(name):
        if name == "openai":
            raise runner.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(runner.metadata, "version", versions)
    evidence = runner.runtime_environment("prepare")
    assert evidence["sys_executable"] == str(Path(sys.executable).resolve())
    assert evidence["sys_executable_sha256"] == runner.sha(sys.executable)
    assert evidence["sys_version"] == sys.version
    assert evidence["packages"]["openai"] is None
    assert evidence["packages"]["numpy"] == original("numpy")
    assert "secret-canary-never-record" not in json.dumps(evidence)
    assert "DO_NOT_LOG_TEST_SECRET" not in json.dumps(evidence)
    assert set(evidence) == {"schema", "stage", "captured_at_utc", "sys_executable", "sys_executable_sha256", "sys_version", "platform", "packages"}


def test_environment_does_not_claim_unreadable_interpreter_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.sys, "executable", str(tmp_path / "not-present-python.exe"))
    evidence = runner.runtime_environment("collect")
    assert evidence["sys_executable"].endswith("not-present-python.exe")
    assert evidence["sys_executable_sha256"] is None


def test_preparation_environment_tampering_is_rejected(tmp_path):
    out = tmp_path / "environment_tamper"
    runner.prepare(out)
    path = out / "PREPARE_ENVIRONMENT.json"
    value = runner.read(path)
    value["packages"]["openai"] = "invented"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="environment record changed"):
        runner.load_plan(out)


@pytest.mark.parametrize("identity_issue", ["missing", "duplicate", "model", "missing_model"])
def test_single_group_identity_issues_preserve_labels_but_exclude_statistics_and_training(tmp_path, identity_issue):
    out, case, api, report, group = collect_fixture(tmp_path, identity_issue=identity_issue)
    assert group["complete"] and group["upstream_complete"]
    assert len(group["continuations"]) == 2 and len(group["shared_labels"]) == 1
    assert not group["response_identity_qualified"]
    assert not group["eligible_for_statistics"] and not group["eligible_for_grouped_fit"]
    assert group["response_identity_issues"]
    assert report["completed"] and len(group["completed_response_receipts"]) == 5


@pytest.mark.parametrize("identity_calls,excluded", [(None, 48), ({1, 25}, 2)])
def test_whole_queue_duplicate_response_ids_exclude_all_involved_groups(tmp_path, monkeypatch, identity_calls, excluded):
    from paperalpha_runtime import quadsentinel_adapter
    out = tmp_path / "duplicate_identity_protocol"
    runner.prepare(out)
    api = FakeAPI(identity_issue="duplicate", identity_calls=identity_calls)
    monkeypatch.setattr(quadsentinel_adapter, "create_api_client", lambda endpoint, key: api)
    receipt = asyncio.run(runner.collect(out))
    assert receipt["complete_groups"] == 48 and receipt["api_calls"] == 264
    before = {str(p): runner.sha(p) for p in out.rglob("*.json")}
    result = runner.verify(out)
    assert result["complete_groups"] == 48 and result["complete_continuations"] == 96
    assert len(result["response_identity_audit"]["excluded_group_ids"]) == excluded
    assert result["eligible_groups"] == 48 - excluded
    duplicate = result["response_identity_audit"]["duplicate_provider_ids"]
    assert len(duplicate) == 1
    assert len({v["group_id"] for v in duplicate[0]["records"]}) == excluded
    for group in result["grouped_traces"]:
        assert group["response_identity_audit_scope"] == "complete_collection"
        assert group["complete"] and len(group["continuations"]) == 2
        assert group["eligible_for_statistics"] == group["response_identity_qualified"] == group["eligible_for_grouped_fit"]
    assert before == {str(p): runner.sha(p) for p in out.rglob("*.json")}


def test_identity_audit_checks_only_completed_responses_and_keeps_failure_records(tmp_path):
    out, case, api, report, group = collect_fixture(tmp_path, fail_call=2, failure="incomplete", identity_issue="missing", identity_calls={2})
    assert len(group["completed_response_receipts"]) == 4
    assert len(group["failures"]) == 1 and len(group["incomplete_continuations"]) == 1
    assert group["response_identity_qualified"] and group["eligible_for_grouped_fit"]
    assert group["response_identity_issues"] == []
