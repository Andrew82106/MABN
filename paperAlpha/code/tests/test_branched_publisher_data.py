import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.branched_publisher_data import cases, issuance, transitions, publisher_records, stage_request, prefix, WORKFLOW
from verify_branched_publisher_data import visible_cues
import collect_branched_publisher_data as collector


def test_split_isolation_and_explicit_development_only_collection():
    cohort = cases()
    assert len(cohort) == 24 and all(sum(c["split"] == s for c in cohort) == 8 for s in ("train", "validation", "test"))
    selected = collector.development_cases({"cases": cohort})
    assert len(selected) == 16 and all(c["split"] != "test" for c in selected)
    assert len({c["template_id"] for c in cohort}) == 6
    assert WORKFLOW["edges"] == [["planner", "worker-a"], ["planner", "worker-b"], ["worker-a", "publisher"], ["worker-b", "publisher"]]


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_role_inputs_and_prefix_without_hidden_labels(case):
    planner = stage_request(case, "planner")
    assert not any(f["literal"] in json.dumps(planner) for f in case["tracked_fields"])
    workers = {}
    for index, name in enumerate(("worker-a", "worker-b")):
        req = stage_request(case, name, planner_text="actual planner instructions")
        serialized = json.dumps(req)
        assert case["records"][index]["reference"] in serialized and case["records"][1 - index]["reference"] not in serialized
        workers[name] = case["records"][index]["reference"] + " internal workpaper"
    publisher = stage_request(case, "publisher", workers=workers)
    p = prefix(case, "actual planner instructions", workers, issuance(case) + transitions(case), publisher)
    assert not {"split", "condition", "label", "final_text", "template_id", "case_id"} & set(p)
    assert p["worker_messages"] == workers and p["field_owners"] == case["field_owners"]
    assert json.loads(publisher["input"][0]["content"])["earlier_agent_message"] == p["pooled_scope"]["worker_text"]
    cue = visible_cues(case, publisher_records(case))
    expected = ["supported", "supported"]
    if case["condition"] == "valid_missing_parent":
        expected = ["unresolved", "unresolved"]
    elif case["condition"] == "revoked_refreshed":
        expected = ["contradicted", "contradicted" if case["layout"] == "shared" else "supported"]
    assert cue == expected


def test_stale_and_valid_same_publisher_input_but_different_truth():
    valid, stale = cases()[:2]
    workers = {"worker-a": "actual a reply", "worker-b": "actual b reply"}
    assert stage_request(valid, "publisher", workers=workers) == stage_request(stale, "publisher", workers=workers)
    assert not transitions(valid) and transitions(stale)
    with pytest.raises(ValueError):
        stage_request(valid, "worker-a", planner_text="plan", workers=workers)
    with pytest.raises(ValueError):
        stage_request(valid, "publisher", workers={"worker-a": "incomplete"})


@pytest.mark.parametrize("fail_one_worker", [False, True])
def test_fake_transport_capture_audit_and_no_abandoned_branch(tmp_path, monkeypatch, fail_one_worker):
    """Local test double only; these outputs are never research API data."""
    out = tmp_path / "cohort"
    collector.prepare(out)
    monkeypatch.setattr(collector, "__file__", str(out / "source_snapshot/code/commands/collect_branched_publisher_data.py"))
    monkeypatch.setenv("PAPERALPHA_API_KEY", "test-double-not-a-credential")
    monkeypatch.setattr(collector.importlib.metadata, "version", lambda name: "test-double")
    failed = False

    class FakeClient:
        def __init__(self, **kwargs):
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, **request):
            nonlocal failed
            await asyncio.sleep(0)
            payload = json.loads(request["input"][0]["content"])
            if "assigned_record" in payload:
                if fail_one_worker and not failed and payload["public_branch_label"] == "Branch A":
                    failed = True
                    raise RuntimeError("synthetic transport failure")
                text = payload["assigned_record"]["reference"] + " synthetic internal reply"
            else:
                text = "Synthetic test-double output, not real API data."
            raw = {"status": "completed", "model": "fixture-not-real-model", "output": [{"type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": text}]}]}
            return SimpleNamespace(model_dump=lambda **kwargs: raw)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeClient))
    asyncio.run(collector.collect(out.resolve(), "http://localhost:58661/v1"))
    audit = tmp_path / "audit"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(out / "source_snapshot/code/commands/verify_branched_publisher_data.py"),
        "--run", str(out), "--out", str(audit)], text=True, encoding="utf8", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((audit / "REPORT.json").read_text(encoding="utf8"))
    assert report["failed"] == int(fail_one_worker) and report["completed"] == 16 - int(fail_one_worker)
    assert report["held_out_generated_runs"] == 0
    assert set(report["response_reported_models"]) == {"fixture-not-real-model"}
