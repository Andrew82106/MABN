"""Paired-role pipeline tests use fake SDK responses and real memory-only Banks."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
import collect_paired_harnessaudit as paired
from paperalpha_runtime.harnessaudit_node_executor import HarnessAuditNodeExecutor
from test_harnessaudit_node_executor import FakeProvider, prohibit_network, response


@pytest.fixture
def cohort(tmp_path):
    out = tmp_path / "cohort"
    summary = paired.prepare(out)
    assert summary["episodes"] == 12 and summary["maximum_api_calls"] == 576
    assert summary["api_called"] is False
    return out


def fake_factory(plan, *, fail_first=False):
    order = {c["case_id"]: c for c in plan["cases"]}
    first = plan["cases"][0]["case_id"]
    providers = []
    def factory(folder):
        identity = folder.parent.name
        case = order[identity]
        script = [response(f"{identity}-{n}", text="ACTUAL_OUTPUT role_compromise is a quoted string.")
                  for n in range(len(case["workflow"]["nodes"]))]
        if fail_first and identity == first:
            script = [TimeoutError("synthetic provider timeout")]
        provider = FakeProvider(script, folder)
        providers.append(provider)
        settings = plan["settings"]
        return HarnessAuditNodeExecutor(receipt_dir=folder, model_provider=provider,
            max_calls=settings["max_calls"], max_turns=settings["max_turns"],
            turn_timeout_seconds=settings["turn_timeout_seconds"], max_output_tokens=settings["max_output_tokens"])
    return factory, providers


def test_prepare_schedule_source_hashes_and_only_selected_role_changed(cohort):
    plan = paired.load_plan(cohort)
    base = paired.collection.load_plan(cohort / "private/base")
    legal = {c["case_id"]: c for c in base["cases"]}
    cells = paired.collection.read(cohort / "private/CONDITIONS.json")["cells"]
    assert len({c["episode_id"] for c in cells}) == 12
    assert len({c["pair_id"] for c in cells}) == 6
    for index, (cell, case) in enumerate(zip(cells, plan["cases"])):
        assert cell["planned_order"] == index and cell["batch"] == index // 2
        assert case["case_id"] == cell["episode_id"]
        original = legal[cell["task_id"]]
        assert case["public_task"]["goal"] == original["public_task"]["goal"]
        for key in ("workflow", "role_bindings", "stage_purposes", "visible_policy", "bank_setup_private", "evaluation_config"):
            assert case[key] == original[key]
        for old, new in zip(original["public_task"]["agents"], case["public_task"]["agents"]):
            modified = cell["condition"] == "role_compromise" and old["role"] == cell["controlled_role"]
            assert (old["system_prompt"] != new["system_prompt"]) is modified
            assert old["description"] == new["description"] and old["role"] == new["role"]
        assert cell["controlled_stages"] == [n for n, b in case["role_bindings"].items() if b["role"] == cell["controlled_role"]]
    assert plan["maximum_api_calls"] == len(plan["cases"]) * plan["settings"]["max_calls"]
    assert plan["settings"] == paired.collection.SETTINGS
    assert "upstream/multi_agent/perturbation.py" in base["source_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        paired.prepare(cohort)


def test_control_and_compromised_initial_public_views_identical(cohort):
    plan = paired.load_plan(cohort)
    by_source = {}
    for case in plan["cases"]:
        raw = paired.collection.adapter_for(plan, case).monitor_prefix()
        raw["condition"] = "private-only-never-published"
        raw["future_labels"] = {"unsafe": True}
        view = paired.public_prefix(raw, paired._legal_template(cohort, case))
        assert all(set(a) == {"role", "description"} for a in view["public_task"]["agents"])
        assert "system_prompt" not in json.dumps(view)
        assert "private-only-never-published" not in json.dumps(view)
        assert "future_labels" not in view and "unsafe" not in view
        expected = by_source.setdefault(case["source_task_id"], view)
        assert view == expected


def test_twelve_fake_episodes_private_provenance_and_public_prefixes(cohort):
    plan = paired.load_plan(cohort)
    factory, providers = fake_factory(plan)
    aggregate = asyncio.run(paired.collect_all(cohort, factory))
    assert aggregate["status"] == "completed" and len(providers) == 12
    report = paired.audit(cohort)
    assert report["status"] == "passed" and report["episodes"] == 12
    assert report["api_attempts"] == report["api_responses"] == report["unique_response_ids"] == 78
    assert report["artifact_sha256"]["private/PLAN.json"] == paired.collection.sha(cohort / "private/PLAN.json")
    assert "private/CONDITIONS.json" in report["artifact_sha256"]
    assert "private/base/source/upstream/multi_agent/perturbation.py" in report["artifact_sha256"]
    assert "private/ARTIFACT_AUDIT.json" not in report["artifact_sha256"]
    assert all(name.startswith("observable/") for name in report["public_file_sha256"])
    for case in plan["cases"]:
        folder = cohort / "observable" / case["case_id"]
        artifact = paired.collection.read(folder / "OBSERVABLE.json")
        assert set(artifact) == {"schema", "episode_id", "visible_prefix"}
        prefix = artifact["visible_prefix"]
        assert "system_prompt" not in prefix["public_task"]["agents"][0]
        assert all("observed_at" not in r and "monotonic_ns" not in r for r in prefix["journal"])
        assert "ACTUAL_OUTPUT role_compromise" in json.dumps(prefix)  # Observed text must not be redacted.
        for path in (folder / "stages").glob("*.json"):
            row = paired.collection.read(path)
            assert row["visible_prefix"]["sequence"] == row["sequence"]
            assert all(e["sequence"] <= row["sequence"] for e in row["visible_prefix"]["journal"])
            assert row["visible_prefix_sha256"] == paired.collection.digest(row["visible_prefix"])
            assert "packet" not in row and "prefix_sha256" not in row
        private = cohort / "private/cases" / case["case_id"]
        assert (private / "api/calls/000001.claim.json").is_file()
        assert "system_prompt" in paired.collection.read(private / "MONITOR.json")["public_task"]["agents"][0]
    with pytest.raises(ValueError, match="terminal collection"):
        asyncio.run(paired.collect_all(cohort, factory))
    assert len(providers) == 12


def test_failed_episode_retained_not_safe_and_other_conditions_not_resampled(cohort):
    plan = paired.load_plan(cohort)
    factory, providers = fake_factory(plan, fail_first=True)
    aggregate = asyncio.run(paired.collect_all(cohort, factory))
    assert aggregate["status"] == "completed_with_failures" and len(providers) == 12
    first = cohort / "private/cases" / plan["cases"][0]["case_id"]
    assert paired.collection.read(first / "LABELS.json")["complete_episode_violation"] is None
    report = paired.audit(cohort)
    assert report["status"] == "passed" and report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["api_errors"] == 1


def test_existing_attempt_without_terminal_is_never_resent(cohort):
    plan = paired.load_plan(cohort)
    first = cohort / "private/cases" / plan["cases"][0]["case_id"]
    first.mkdir(parents=True)
    paired.collection.write_new(first / "attempt.json", {"plan_sha256": plan["sha256"], "case_id": plan["cases"][0]["case_id"]})
    factory, providers = fake_factory(plan)
    rows = asyncio.run(paired.collect_all(cohort, factory))["cases"]
    assert rows[0]["status"] == "incomplete" and len(providers) == 11
    report = paired.audit(cohort)
    assert report["status"] == "incomplete_evidence"


def test_exclusive_run_claim_blocks_concurrent_executor_and_automatic_restart(cohort, monkeypatch):
    made = []

    async def exercise():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def case_stub(out, plan, case, executor_factory):
            executor_factory(case["case_id"])
            entered.set()
            await finish.wait()
            return {"case_id": case["case_id"], "status": "incomplete", "error_type": "SyntheticOnly"}

        monkeypatch.setattr(paired.collection, "collect_case", case_stub)
        first = asyncio.create_task(paired.collect_all(cohort, made.append))
        await entered.wait()
        initial_count = len(made)
        with pytest.raises(ValueError, match="exclusive run claim"):
            await paired.collect_all(cohort, made.append)
        assert len(made) == initial_count == 2
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not (cohort / "private/COLLECTION.json").exists()
        with pytest.raises(ValueError, match="exclusive run claim"):
            await paired.collect_all(cohort, made.append)
        assert len(made) == initial_count

    asyncio.run(exercise())


def test_cancelled_base_collector_does_not_start_later_batches(cohort, monkeypatch):
    called = []

    async def cancelled(out, plan, case, factory):
        called.append(case["case_id"])
        return {"case_id": case["case_id"], "status": "failed", "error_type": "CancelledError"}

    monkeypatch.setattr(paired.collection, "collect_case", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(paired.collect_all(cohort, lambda _: None))
    assert len(called) <= 2
    assert not (cohort / "private/COLLECTION.json").exists()


def test_audit_rejects_missing_response_id_even_with_otherwise_checked_case(cohort, monkeypatch):
    plan = paired.load_plan(cohort)
    factory, _ = fake_factory(plan)
    asyncio.run(paired.collect_all(cohort, factory))
    path = next((cohort / "private/cases").glob("*/api/calls/*.response.json"))
    value = paired.collection.read(path)
    value["response"]["response_id"] = ""
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(paired.original_audit, "audit_case", lambda _out, _plan, case:
                        {"case_id": case["case_id"], "status": "completed", "checks": "passed"})
    with pytest.raises(ValueError, match="nonempty response id"):
        paired.audit(cohort)


@pytest.mark.parametrize("change", ["condition", "frozen_source", "public_view", "raw_prompt", "private_journal_field"])
def test_modification_is_detected(cohort, change):
    plan = paired.load_plan(cohort)
    if change == "condition":
        path = cohort / "private/CONDITIONS.json"
        value = paired.collection.read(path); value["cells"][0]["condition"] = "role_compromise"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError): paired.load_plan(cohort)
        return
    if change == "frozen_source":
        path = cohort / "private/source/code/commands/collect_paired_harnessaudit.py"
        path.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError): paired.load_plan(cohort)
        return
    factory, _ = fake_factory(plan)
    asyncio.run(paired.collect_all(cohort, factory))
    case = plan["cases"][0]
    if change == "public_view":
        path = cohort / "observable" / case["case_id"] / "OBSERVABLE.json"
        value = paired.collection.read(path); value["visible_prefix"]["private_flag"] = "never exposed"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError): paired.audit(cohort)
    elif change == "raw_prompt":
        path = cohort / "private/cases" / case["case_id"] / "api/calls/000001.claim.json"
        value = paired.collection.read(path); value["request"]["system_instructions"] = "tampered instruction"
        value["request_sha256"] = paired.collection.digest(value["request"])
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError): paired.audit(cohort)
    else:
        raw = paired.collection.read(cohort / "private/cases" / case["case_id"] / "MONITOR.json")
        raw["journal"][0]["payload"]["controlled_role"] = "secret"
        with pytest.raises(ValueError, match="private journal"):
            paired.public_prefix(raw, paired._legal_template(cohort, case))


@pytest.mark.parametrize("endpoint", ["https://example.org/v1", "http://localhost:8/v1", "http://a:b@localhost:58661/v1"])
def test_no_unapproved_transport(endpoint, tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        asyncio.run(paired.run_api(tmp_path, endpoint))
