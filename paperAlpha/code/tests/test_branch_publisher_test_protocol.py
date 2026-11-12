"""Test doubles below are temporary protocol fixtures, NEVER API research data."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
import collect_branch_publisher_test as collector
from collect_branched_publisher_data import prepare as prepare_data
from collect_privacy_transfer_api import sha, write
from monitor_branch_publisher import FrozenBranchPilot, EXPECTED_FAMILIES
from branch_publisher_fixtures import fixture
from paperalpha_monitor.branch_benchmark import field_features
from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
from paperalpha_monitor.publisher_view_bn import PublisherViewPrivacyBN
from paperalpha_monitor.privacy_model_independent import ModelIndependentPrivacyBN


def fake_parent(tmp_path):
    data, fit, audit = [tmp_path / name for name in ("data", "fit", "audit")]
    prepare_data(data)
    entries = {}
    for family in sorted(EXPECTED_FAMILIES):
        name = None
        if family in {"branch", "branch_uncoupled", "pooled", "frozen_source_pooled"}:
            cls = BranchPublisherPrivacyBN if family.startswith("branch") else PublisherViewPrivacyBN
            model = cls(ModelIndependentPrivacyBN(coupled=family != "branch_uncoupled"))
            name = "models/" + family + ".json"
            write(fit / name, model.parameters())
        elif family in {"logistic", "histgb"}:
            x = np.concatenate([field_features(fixture(visible=v)) for v in ("supported", "unresolved")])
            model = LogisticRegression() if family == "logistic" else HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=1)
            model.fit(x, [1, 1, 0, 0])
            name = "models/" + family + ".joblib"
            joblib.dump(model, fit / name)
        entries[family] = {"id": "test-double-" + family, "family": family, "file": name}
    selected = {family: entry["id"] for family, entry in entries.items()}
    write(fit / "PLAN.json", {"data_plan_sha256": sha(data / "PLAN.json"), "scope": "TEST DOUBLE ONLY"})
    write(fit / "SELECTED.json", {"selected": selected, "scope": "TEST DOUBLE ONLY"})
    files = ["PLAN.json", "SELECTED.json", *[e["file"] for e in entries.values() if e["file"]]]
    frozen = {"schema": "fixed-branch-pilot-monitor-v1", "created_at": "2099-01-01T00:00:00+00:00", "selected": selected, "entries": entries,
        "files_sha256": {s: sha(fit / s) for s in files}, "scope": "TEST DOUBLE, not research results"}
    # Timestamp from the actual temporary data plan keeps ordinary order checks valid.
    frozen["created_at"] = json.loads((data / "PLAN.json").read_text(encoding="utf-8"))["created_at"]
    write(fit / "FROZEN.json", frozen)
    sources = {p.relative_to(ROOT).as_posix(): p for p in (ROOT / 'code/src').rglob("*.py")}
    sources.update({s: ROOT / s for s in ('code/commands/collect_privacy_transfer_api.py', 'code/commands/scalar_binary_graph.py',
        'knowledge/templates/risk_templates_v1.json', 'knowledge/templates/joint_risk_templates_v1.json', 'knowledge/risks/risk_catalog.json')})
    for name, source in sources.items():
        target = fit / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write(fit / "REPORT.json", {"artifacts_sha256": {p.relative_to(fit).as_posix(): sha(p) for p in fit.rglob("*") if p.is_file()}, "scope": "TEST DOUBLE"})
    write(audit / "REPORT.json", {"report_sha256": sha(fit / "REPORT.json"), "selection_recomputed": True, "scope": "TEST DOUBLE"})
    out = tmp_path / "heldout"
    collector.prepare(data, fit, audit, out)
    return out


def test_frozen_scoring_rejects_labels_and_model_change(tmp_path):
    out = fake_parent(tmp_path)
    service = FrozenBranchPilot(out / "bundle")
    sample = fixture()
    message = {"observation": sample.observation, "prefix_sequence": 20}
    result = service.score(message)
    assert set(result["count_distributions"]) == EXPECTED_FAMILIES
    for key in ("outcome", "case_id", "final_text", "condition"):
        with pytest.raises(ValueError):
            service.score({**message, key: 1})
    with (out / "bundle/models/branch.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError):
        FrozenBranchPilot(out / "bundle")


@pytest.mark.parametrize("failure", [None, "worker", "incomplete", "publisher", "monitor"])
def test_mock_heldout_capture_and_independent_replay(tmp_path, monkeypatch, failure):
    out = fake_parent(tmp_path)
    monkeypatch.setattr(collector, "__file__", str(out / "source_snapshot/code/commands/collect_branch_publisher_test.py"))
    monkeypatch.setenv("PAPERALPHA_API_KEY", "test-double-not-a-credential")
    monkeypatch.setattr(collector.importlib.metadata, "version", lambda name: "test-double")
    failed = False
    real_monitor = collector.Monitor
    if failure == "monitor":
        class FailingMonitor(real_monitor):
            def score(self, message):
                nonlocal failed
                if not failed:
                    failed = True
                    raise RuntimeError("test-double-monitor-failure")
                return super().score(message)
        monkeypatch.setattr(collector, "Monitor", FailingMonitor)

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, **request):
            nonlocal failed
            await asyncio.sleep(0)
            payload = json.loads(request["input"][0]["content"])
            worker, publisher = "assigned_record" in payload, "cached_authority_records" in payload
            should_fail = (failure in {"worker", "incomplete"} and worker) or (failure == "publisher" and publisher)
            status = "completed"
            if should_fail and not failed:
                failed = True
                if failure == "incomplete":
                    status = "incomplete"
                else:
                    raise RuntimeError("test-double-transport-failure")
            text = payload["assigned_record"]["reference"] if worker else "Test-double text, not a real model reply."
            raw = {"status": status, "model": "fixture-not-real-model", "output": [{"type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": text}]}]}
            return SimpleNamespace(model_dump=lambda **kwargs: raw)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeClient))
    asyncio.run(collector.collect(out, "http://localhost:58661/v1"))
    audit = tmp_path / "heldout-audit"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(out / "source_snapshot/code/commands/verify_branch_publisher_test.py"),
        "--run", str(out), "--out", str(audit)], capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert result.returncode == 0, result.stderr
    report = json.loads((audit / "REPORT.json").read_text(encoding="utf-8"))
    assert report["planned_cases"] == 8 and report["failed"] == int(failure is not None)
    assert report["completed"] == 8 - int(failure is not None)
    assert set(report["response_reported_models"]) == {"fixture-not-real-model"}
    assert set(report["scores"]) == EXPECTED_FAMILIES
    assert report["prefixes_scored_before_publisher_call"] == (7 if failure in {"worker", "incomplete", "monitor"} else 8)
    with pytest.raises(FileExistsError):
        asyncio.run(collector.collect(out, "http://localhost:58661/v1"))
