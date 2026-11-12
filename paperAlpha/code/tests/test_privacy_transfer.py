import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.privacy_transfer import STAGES, cases, stage_request, delivered_text, outcome
from paperalpha_monitor.privacy_flow_data import final_value_labels, protected_groups
from monitor_privacy_prefixes import observation_packet
from collect_privacy_transfer_api import Journal
from verify_privacy_transfer import summarize


def response(text="visible", phase="final_answer", kind="output_text"):
    return {"status": "completed", "output": [{"type": "message", "role": "assistant", "phase": phase,
        "status": "completed", "content": [{"type": kind, "text" if kind == "output_text" else "refusal": text}]}]}


def test_cases_are_paired_records_not_paired_outcome_labels():
    data = cases()
    assert len(data) == 16 and len({c["template_id"] for c in data}) == 8
    assert len({c["id"] for c in data}) == 16
    for a, b in zip(data[::2], data[1::2]):
        assert a["initial"] == b["initial"] and a["source_note"] != b["source_note"]
        assert len(protected_groups(a["initial"])) == 6
        assert "y" not in a and "y" not in b


def test_actual_previous_reply_only_no_publisher_vault():
    case = cases()[0]
    for i, stage in enumerate(STAGES):
        payload = json.loads(stage_request(case, stage, None if i == 0 else "ACTUAL REPLY")["input"][0]["content"])
        assert ("vault" in payload) == (i < 2)
        assert ("earlier_agent_message" in payload) == (i > 0)
        if i:
            assert payload["earlier_agent_message"] == "ACTUAL REPLY"
        assert "condition" not in payload and "template_id" not in payload


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
def test_incomplete_is_not_a_safe_outcome(status):
    value = response()
    value["status"] = status
    with pytest.raises(ValueError):
        delivered_text(value)


def test_final_phase_selected_not_reasoning_or_commentary():
    value = response("FINAL")
    value["output"] = [response("PREAMBLE", "commentary")["output"][0], {"type": "reasoning", "text": "SECRET"}, *value["output"]]
    assert delivered_text(value) == {"text": "FINAL", "native_refusal": False}
    with pytest.raises(ValueError):
        delivered_text(response("PREAMBLE", "commentary"))
    assert delivered_text(response("REFUSAL", None, "refusal")) == {"text": "REFUSAL", "native_refusal": True}


@pytest.mark.parametrize("extra", ["condition", "model", "final_text", "label", "id"])
def test_monitor_rejects_acquisition_or_future_fields(extra):
    c = cases()[0]
    obs = {"initial": c["initial"], "vertical": c["vertical"], "coordinator_text": "internal note", "worker_text": None}
    assert observation_packet(obs)["stage"] == 1
    obs[extra] = "not observable"
    with pytest.raises(ValueError):
        observation_packet(obs)


def test_prediction_cutoff_is_respected_and_independent_literal_label_matches():
    c = cases()[0]
    initial = c["initial"]
    private = initial["vault"]["internal_tracking_token"]
    obs = {"initial": initial, "vertical": c["vertical"], "coordinator_text": private, "worker_text": None}
    a = observation_packet(obs)
    assert all(g["worker_present"] is None for g in a["groups"])
    changed = {**obs, "worker_text": private}
    assert any(g["worker_present"] for g in observation_packet(changed)["groups"])
    for text in (private, "nothing private", initial["vault"]["notice_id"]):
        label = outcome(initial, text)
        assert label["final_value_labels"] == final_value_labels(initial, text)
    assert outcome(initial, initial["vault"]["notice_id"])["y"] == 0


def test_duplicate_values_count_once_and_exemptions_preserved():
    initial = copy.deepcopy(cases()[0]["initial"])
    initial["vault"].update(short="abc", duplicate=initial["vault"]["internal_record_ref"], request_echo=initial["request"])
    label = outcome(initial, initial["vault"]["internal_record_ref"])
    assert sum(label["final_value_labels"]) == 1
    assert set(label["leaked_fields"]) == {"internal_record_ref", "duplicate"}
    assert label["final_value_labels"] == final_value_labels(initial, initial["vault"]["internal_record_ref"])


def test_durable_journal_is_ordered_and_cannot_be_overwritten(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    a = journal.add("prediction", "case")
    b = journal.add("api_request", "case")
    journal.close()
    assert a["seq"] == 1 and b["seq"] == 2 and a["monotonic_ns"] <= b["monotonic_ns"]
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        Journal(path)


def test_one_class_metrics_do_not_claim_ranking_ability():
    rows = [{"outcome": {"y": 0}, "predictions": {s: {"scores": {"model": .6}} for s in (1, 2)}}]
    result = summarize(rows)["1"]["model"]
    assert result["auroc"] is None and result["average_precision"] is None
    assert result["false_alerts_at_0.5"] == 1 and result["brier"] == .36
