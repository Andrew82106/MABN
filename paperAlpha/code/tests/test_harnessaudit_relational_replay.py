"""Replay report bookkeeping; real outputs are not test fixtures."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
from replay_harnessaudit_relations import compare_actions, run


def evidence(a, b):
    return {"actions": [{"invocation_id": "x", "attempt": {"violation": a}, "effect": {"violation": b}}]}


def labels(a, b):
    return {"events": [{"invocation_id": "x", "attempt_violation": a, "effect_violation": b}]}


def test_unknown_is_not_negative_or_false_agreement():
    checks = compare_actions(evidence(None, False), labels(True, False))[0]["checks"]
    assert checks["attempt"] == {"monitor": None, "oracle": True, "known_both": False, "agree": None}
    assert checks["effect"]["agree"] is True


def test_disagreement_is_retained():
    assert compare_actions(evidence(False, True), labels(True, True))[0]["checks"]["attempt"]["agree"] is False


def test_absent_target_action_is_not_an_extra_correct_prediction():
    assert compare_actions({"actions": []}, {"events": []}) == []


def test_cannot_drop_or_duplicate_triggers():
    with pytest.raises(ValueError, match="inventories"):
        compare_actions(evidence(False, False), {"events": []})
    data = evidence(False, False)
    data["actions"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        compare_actions(data, labels(False, False))


def test_existing_report_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        run(tmp_path / "missing", tmp_path / "missing.json", tmp_path)
