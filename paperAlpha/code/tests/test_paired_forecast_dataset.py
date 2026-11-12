from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'code/src'),
               str(Path(__file__).resolve().parents[2] / 'code/commands')]
import project_paired_forecasts as module


def invoke():
    return module.project_claim({"stage": "raw"}, {"call": 1, "node_id": "stage"},
        [{"event": "observed"}], {"id": "policy"}, {"task": "legal"},
        "opaque", {"secret_state": 99}, {"future": 88}, None)


def test_private_or_future_channel_never_passed_to_observer(monkeypatch):
    def project(stage, claim, journal, contract, *, legitimate_initial_prefix, episode_id):
        assert stage == {"stage": "raw"}
        assert claim == {"call": 1, "node_id": "stage"}
        assert journal == [{"event": "observed"}]
        assert contract == {"id": "policy"}
        assert legitimate_initial_prefix == {"task": "legal"}
        assert episode_id == "opaque"
        return {"current_a": 0, "prefix_horizon": 3, "visible_prefix": {"sequence": 3}}
    def target(oracle, contract, **kwargs):
        assert oracle == {"secret_state": 99}
        assert kwargs["response"] == {"future": 88}
        assert kwargs["horizon"] == 3
        return {"labels": {"A": 0, "C": "STOP", "B": 0, "S": 0}, "risk_label": 0}
    monkeypatch.setattr(module, "project_paired_forecast_input", project)
    monkeypatch.setattr(module, "project_next_reply_target", target)
    public, private = invoke()
    assert public["covered"]
    assert "secret_state" not in str(public) and "future" not in str(public)
    assert private["labels"]["C"] == "STOP"


def test_uncovered_timing_is_retained_not_imputed_safe(monkeypatch):
    def uncovered(*args, **kwargs):
        raise module.ForecastProjectionUncovered("ambiguous")
    monkeypatch.setattr(module, "project_paired_forecast_input", uncovered)
    monkeypatch.setattr(module, "project_next_reply_target", lambda *a, **k: pytest.fail("uncertified horizon"))
    public, private = invoke()
    assert public["covered"] is False
    assert private["risk_label"] is None
    assert all(value is None for value in private["labels"].values())


def test_conflicting_independent_evidence_fails_explicitly(monkeypatch):
    monkeypatch.setattr(module, "project_paired_forecast_input", lambda *a, **k: {"current_a": 1, "prefix_horizon": 3})
    monkeypatch.setattr(module, "project_next_reply_target", lambda *a, **k: {"labels": {"A": 0}})
    with pytest.raises(ValueError, match="contradicts"):
        invoke()


def test_manifest_does_not_escape_or_accept_changed_acquisition(tmp_path):
    path = tmp_path / "claim.json"
    path.write_text("test receipt", encoding="utf-8")
    module.verify_manifest(tmp_path, {path.name: module.sha(path)})
    with pytest.raises(ValueError, match="changed"):
        module.verify_manifest(tmp_path, {path.name: "wrong"})
    with pytest.raises(ValueError, match="changed"):
        module.verify_manifest(tmp_path, {"../outside": "wrong"})
    with pytest.raises(ValueError, match="manifest"):
        module.verify_manifest(tmp_path, {})
    (tmp_path / "unreviewed.claim.json").write_text("new receipt", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        module.verify_manifest(tmp_path, {path.name: module.sha(path)})
