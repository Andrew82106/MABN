import asyncio
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'code/src'),
               str(Path(__file__).resolve().parents[2] / 'code/commands')]
import score_contract_forecast_judge as module
import collect_contract_forecast_judge as collector
from score_contract_forecast_judge import expand_distribution, STATES
from paperalpha_monitor.contract_forecast_llm import parse_forecast, KEYS, STATES as FEASIBLE
from test_contract_forecast_judge_collection import response
from test_harnessaudit_node_executor import prohibit_network


def test_20_feasible_states_expand_to_32_without_reindexing():
    desired = (0, "TRIGGER", 1, 1)
    raw = {"probabilities": {k: int(s == desired) for k, s in zip(KEYS, FEASIBLE)}}
    p = expand_distribution(parse_forecast(json.dumps(raw), 0))
    assert len(p) == 32 and p.sum() == 1
    assert p[STATES.index(desired)] == 1
    for state, value in zip(STATES, p):
        if state not in FEASIBLE:
            assert value == 0


def test_duplicate_states_rejected_even_if_length_matches():
    raw = {"probabilities": {k: .05 for k in KEYS}}
    prediction = parse_forecast(json.dumps(raw), None)
    prediction["states"][1] = prediction["states"][0]
    with pytest.raises(ValueError, match="exactly once"):
        expand_distribution(prediction)


def rewrite(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def inventory(folder, excluded):
    return {p.relative_to(folder).as_posix(): module.sha(p) for p in folder.rglob("*")
            if p.is_file() and p.relative_to(folder).as_posix() not in excluded}


def build(tmp_path, monkeypatch, *, replies=None, unknown=False):
    """Handwritten labels/checkpoints, real collector control flow, fake transport.

    No fitting or original episodes are used. Checkpoint byte stubs exercise
    inventory binding only; the scorer does not claim model execution replay.
    """
    dataset, judge, numerical = (tmp_path / name for name in ("dataset", "judge", "numerical"))
    episodes = []
    for identity, task, contract in (("e-ec", "ec-t3", "routed_case_before_compensation"),
                                      ("e-fin", "fin-t1", "bound_risk_update_before_trade")):
        windows, labels = [], []
        for number in (1, 2):
            windows.append({"claim_number": number, "covered": True, "projection": {
                "schema": "paired-contract-forecast-observer-v1", "covered": True,
                "metadata": {"stage_id": "worker", "episode_id": identity}, "current_a": 0,
                "features": {"bias": 1.0}, "evidence_path": {}, "visible_prefix": {}}})
            target = {"A": 0, "C": "TRIGGER", "B": 1, "S": 1}
            labels.append({"claim_number": number, "labels": target if not unknown else
                {"A": 0, "C": None, "B": None, "S": None}, "risk_label": 1 if not unknown else None})
        windows.append({"claim_number": 3, "covered": False})
        labels.append({"claim_number": 3, "labels": {k: None for k in ("A", "C", "B", "S")}, "risk_label": None})
        episode = {"episode_id": identity, "task_id": task, "contract_id": contract, "windows": windows}
        episodes.append(episode)
        module.write_new(dataset / "public" / (identity + ".json"), episode)
        module.write_new(dataset / "private" / (identity + ".json"),
                         {"episode_id": identity, "task_id": task, "labels": labels})
    module.write_new(dataset / "PLAN.json", {"episode_ids": [e["episode_id"] for e in episodes],
        "contracts": {e["task_id"]: {"id": e["contract_id"]} for e in episodes}})
    module.write_new(dataset / "private/REPORT.json", {"status": "projected",
        "artifact_sha256": inventory(dataset, {"private/REPORT.json"})})
    collector.prepare(dataset, judge)
    scripted = replies if replies is not None else [response(f"response-{i}") for i in range(4)]
    class FakeAPI:
        def __init__(self): self.responses, self.count = self, 0
        async def create(self, **request):
            self.count += 1
            assert collector.read(judge / "runs" / f"{self.count:04d}" / "REQUEST.json")["request"] == request
            value = scripted[self.count - 1]
            if isinstance(value, Exception): raise value
            return SimpleNamespace(model_dump=lambda **_: copy.deepcopy(value))
        async def close(self): pass
    fake = FakeAPI()
    monkeypatch.setenv("PAPERALPHA_LOCAL_API_KEY", "fake-not-a-credential")
    monkeypatch.setattr(collector, "create_api_client", lambda *args: fake)
    asyncio.run(collector.run(judge))
    assert fake.count == 4
    sources = {name: module.sha(module.ROOT / name) for name in module.numeric.SOURCES}
    for name in sources:
        destination = numerical / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(module.ROOT / name, destination)
    folds = module.numeric.folds_for(episodes)
    module.write_new(numerical / "PLAN.json", {"dataset_report_sha256": module.sha(dataset / "private/REPORT.json"),
        "methods": list(module.METHODS), "folds": folds, "config": module.numeric.CONFIG, "source_sha256": sources})
    for fold in folds:
        for name in ("bn.json", "histgb.joblib"):
            path = numerical / "models" / fold["test_task"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic inventory-only checkpoint", encoding="utf-8")
    probabilities = [float(state == (0, "TRIGGER", 1, 1)) for state in STATES]
    controls = [{"method": method, "episode_id": e["episode_id"], "task_id": e["task_id"],
                 "claim_number": number, "probabilities": probabilities,
                 "scores": {"risk_brier": 98765, "do_not_use_cached_scores": True}}
                for method in module.METHODS for e in episodes for number in (1, 2)]
    module.write_new(numerical / "PREDICTIONS.json", controls)
    module.write_new(numerical / "REPORT.json", {"status": "evaluated",
        "plan_sha256": module.sha(numerical / "PLAN.json"), "artifact_sha256": inventory(numerical, {"REPORT.json"})})
    return judge, dataset, numerical


def test_receipt_bound_rescoring_uses_identical_windows_and_ignores_cached_scores(tmp_path, monkeypatch):
    judge, dataset, numerical = build(tmp_path, monkeypatch)
    assert module.run(judge, dataset, numerical) == {"scored": 4, "failed": 0, "publication_ready": False}
    report = module.read(judge / "SCORES.json")
    assert set(report["common_windows_per_method"].values()) == {4}
    assert len(report["common_window_keys"]) == 4
    assert report["numeric_cached_scores_used"] is False
    for method in module.METHODS:
        assert report["common_valid_window_scores"][method]["task_macro"]["risk_brier"] == 0
    assert report["common_valid_window_scores"]["full_public_prefix_llm"]["task_macro"]["risk_brier"] == 1
    with pytest.raises(ValueError, match="never overwrite"):
        module.run(judge, dataset, numerical)


def test_duplicate_and_transport_failures_are_not_zero_and_common_keys_match(tmp_path, monkeypatch):
    paths = build(tmp_path, monkeypatch, replies=[response("duplicate"), response("duplicate"),
                  TimeoutError("synthetic"), response("unique")])
    assert module.run(*paths)["failed"] == 3
    report = module.read(paths[0] / "SCORES.json")
    assert len(report["judge_predictions"]) == 1
    assert set(report["common_windows_per_method"].values()) == {1}
    assert [r["error_type"] for r in report["failures"]] == ["DuplicateResponseIdentity", "DuplicateResponseIdentity", "TimeoutError"]


def test_all_judge_failures_leave_no_common_scores(tmp_path, monkeypatch):
    paths = build(tmp_path, monkeypatch, replies=[TimeoutError("synthetic") for _ in range(4)])
    assert module.run(*paths)["scored"] == 0
    report = module.read(paths[0] / "SCORES.json")
    assert report["common_valid_window_scores"] == {} and report["judge_predictions"] == []
    assert set(report["common_windows_per_method"].values()) == {0}


def test_unknown_future_labels_do_not_turn_into_negative_scores(tmp_path, monkeypatch):
    paths = build(tmp_path, monkeypatch, unknown=True)
    module.run(*paths)
    report = module.read(paths[0] / "SCORES.json")
    for entry in report["common_valid_window_scores"].values():
        assert entry["task_macro"]["risk_brier"] is None
        assert entry["score_denominators"]["risk_brier"]["windows"] == 0


@pytest.mark.parametrize("tamper", ["prediction", "resurrect_duplicate", "status"])
def test_forged_judge_report_cannot_override_raw_receipts(tmp_path, monkeypatch, tamper):
    replies = [response("duplicate"), response("duplicate"), response("a"), response("b")]
    paths = build(tmp_path, monkeypatch, replies=replies)
    path = paths[0] / "REPORT.json"
    report = module.read(path)
    if tamper == "prediction": report["records"][2]["prediction"]["probabilities"] = [0.05] * 20
    elif tamper == "resurrect_duplicate": report["records"][0] = module.read(paths[0] / "runs/0001/RESULT.json")
    else: report["records"][2]["status"] = "failed"
    rewrite(path, report)
    with pytest.raises(ValueError, match="REPORT does not reproduce"):
        module.run(*paths)


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "task", "method", "support", "hard_a"])
def test_invalid_numerical_predictions_fail_even_with_write_time_manifest(tmp_path, monkeypatch, tamper):
    paths = build(tmp_path, monkeypatch)
    folder = paths[2]
    rows = module.read(folder / "PREDICTIONS.json")
    if tamper == "missing": rows.pop()
    elif tamper == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif tamper == "task": rows[0]["task_id"] = "different-task"
    elif tamper == "method": rows[0]["method"] = "unplanned-model"
    else:
        state = (0, "STOP", 1, 1) if tamper == "support" else (1, "TRIGGER", 1, 1)
        rows[0]["probabilities"] = [float(s == state) for s in STATES]
    rewrite(folder / "PREDICTIONS.json", rows)
    report = module.read(folder / "REPORT.json")
    report["artifact_sha256"] = inventory(folder, {"REPORT.json"})
    rewrite(folder / "REPORT.json", report)
    with pytest.raises(ValueError):
        module.run(*paths)


@pytest.mark.parametrize("tamper", ["numeric_prediction", "numeric_extra", "numeric_source", "private_target", "judge_response"])
def test_changed_or_added_artifacts_rejected(tmp_path, monkeypatch, tamper):
    paths = build(tmp_path, monkeypatch)
    judge, dataset, numerical = paths
    if tamper == "numeric_extra": (numerical / "unreviewed.json").write_text("{}", encoding="utf-8")
    else:
        path = {"numeric_prediction": numerical / "PREDICTIONS.json",
                "numeric_source": numerical / "source" / module.numeric.SOURCES[0],
                "private_target": dataset / "private/e-ec.json",
                "judge_response": judge / "runs/0001/RESPONSE.json"}[tamper]
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        module.run(*paths)
