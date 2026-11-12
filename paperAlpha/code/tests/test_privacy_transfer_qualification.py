import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
from qualify_privacy_transfer import diagnose


def sample(c=0, w=0, f=0, name="a"):
    case = {"id": name, "template_id": "template", "condition": "ordinary"}
    g = {"field_names": ["private"], "kind": "text", "coordinator_present": bool(c), "worker_present": None}
    events = [{"kind": "prediction", "case_id": name, "packet": {"stage": 1, "groups": [g]}},
        {"kind": "prediction", "case_id": name, "packet": {"stage": 2, "groups": [{**g, "worker_present": bool(w)}]}}]
    return case, events, {"case_id": name, "outcome": {"y": f}}


def test_all_safe_no_internal_flow_is_not_detection_evidence():
    case, events, row = sample()
    d = diagnose({"cases": [case]}, events, [row])
    assert d["patterns"] == {"C0W0F0": 1}
    assert d["eligible_descriptive_checks"] == {"negative_case_false_alarms": True, "binary_ranking": False,
        "positive_case_sensitivity": False, "observed_internal_transfer_examples": False}
    assert not d["general_MAS_superiority_established"]


def test_both_outcomes_and_transfer_allow_checks_not_superiority_claim():
    a, ea, ra = sample(1, 1, 1, "a")
    b, eb, rb = sample(1, 1, 0, "b")
    d = diagnose({"cases": [a, b]}, ea + eb, [ra, rb])
    assert all(d["eligible_descriptive_checks"].values())
    assert not d["general_MAS_superiority_established"]
    assert d["counts"]["templates"] == 1


def test_final_disclosure_without_internal_presence_is_possible():
    case, events, row = sample(0, 0, 1)
    d = diagnose({"cases": [case]}, events, [row])
    assert d["patterns"] == {"C0W0F1": 1}
    assert not d["eligible_descriptive_checks"]["observed_internal_transfer_examples"]
    assert d["eligible_descriptive_checks"]["positive_case_sensitivity"]


def test_rejects_changed_earlier_evidence():
    case, events, row = sample()
    events = copy.deepcopy(events)
    events[1]["packet"]["groups"][0]["coordinator_present"] = True
    with pytest.raises(ValueError, match="earlier"):
        diagnose({"cases": [case]}, events, [row])
