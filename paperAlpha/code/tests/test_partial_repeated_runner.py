"""Offline constructed failure cohorts; these are not API performance tests."""
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands'), str(ROOT / 'code/tests')]
import collect_repeated_message_study as original
import evaluate_partial_repeated_messages as runner
from paperalpha_runtime import repeated_message_study as study
from test_repeated_message_study import FakeAPI


class FailureFixtureAPI(FakeAPI):
    def __init__(self, case):
        super().__init__()
        self.case = case
        review_count = len(study.reviewers(case))
        self.failure_calls = ({review_count} if case["repeat"] == 0 else
                              {review_count+1, review_count+3} if case["repeat"] == 1 else
                              {review_count+1} if case["repeat"] == 2 else set())

    async def create(self, **request):
        next_call = len(self.calls)+1
        self.fail_call = next_call if next_call in self.failure_calls else None
        response = await super().create(**request)
        raw = response.model_dump()
        raw["id"] = self.case["id"] + ":" + raw["id"]
        return SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(raw))


def fake_collection(tmp_path):
    raw = tmp_path / "raw"
    original.prepare(raw)
    original.write(raw / "COLLECT_ENVIRONMENT.json", original.runtime_environment("collect"))
    reports = [asyncio.run(original.collect_group(raw, c, FailureFixtureAPI(c))) for c in study.cases()]
    original.write(raw / "COLLECTION.json", {
        "plan_sha256": original.sha(raw / "PLAN.json"),
        "collect_environment_sha256": original.sha(raw / "COLLECT_ENVIRONMENT.json"),
        "groups": len(reports), "api_calls": sum(r["api_calls"] for r in reports),
        "complete_groups": sum(r["completed"] for r in reports),
        "complete_continuations": sum(r["complete_continuations"] for r in reports),
        "reports_sha256": {r["id"]: original.sha(raw / "runs" / r["id"] / "REPORT.json") for r in reports},
        "development_only": True, "new_fits": 0, "prospective_predictions": 0})
    return raw


def test_prepare_binds_separate_protocol_without_touching_original(tmp_path):
    raw = tmp_path / "raw"
    original.prepare(raw)
    before = original.sha(raw / "PLAN.json")
    out = tmp_path / "partial"
    receipt = runner.prepare(raw, out)
    plan = runner.read(out / "PLAN.json")
    assert receipt["new_api_calls"] == 0 and plan["prepared_before_collection"]
    assert plan["raw_plan_sha256"] == before == original.sha(raw / "PLAN.json")
    assert plan["training_modes"] == ["complete_pairs", "all_observed"]
    assert not (raw / "COLLECTION.json").exists()
    with pytest.raises(ValueError, match="new extension"):
        runner.prepare(raw, out)
    with pytest.raises(ValueError, match="outside"):
        runner.prepare(raw, raw / "partial")
    with pytest.raises(ValueError, match="own frozen"):
        runner.evaluate(out)


def test_frozen_runner_preserves_raw_failures_and_runs_both_modes(tmp_path):
    raw = fake_collection(tmp_path)
    before = {p.relative_to(raw).as_posix(): original.sha(p) for p in raw.rglob("*") if p.is_file()}
    verified, fingerprint = runner.frozen_raw_verify(raw)
    assert len(verified["grouped_traces"]) == 48 and len(fingerprint) == 64
    assert any(not g["upstream_complete"] for g in verified["grouped_traces"])
    assert any(g["incomplete_continuations"] for g in verified["grouped_traces"])
    out = tmp_path / "partial"
    runner.prepare(raw, out)
    cmd = [sys.executable, "-X", "utf8", "-B", str(out / "source" / runner.SCRIPT), "evaluate", "--out", str(out)]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["new_api_calls"] == 0 and receipt["predictions"] > 0
    report = runner.read(out / "REPORT.json")
    assert report["execution"]["verified_collection_sha256"] == fingerprint
    assert before == {p.relative_to(raw).as_posix(): original.sha(p) for p in raw.rglob("*") if p.is_file()}
    again = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=30, check=False)
    assert again.returncode != 0 and "already started" in again.stderr


def test_original_frozen_verifier_detects_changed_raw_record(tmp_path):
    raw = fake_collection(tmp_path)
    record = next((raw / "runs").rglob("record_0001.json"))
    record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="verifier failed"):
        runner.frozen_raw_verify(raw)
