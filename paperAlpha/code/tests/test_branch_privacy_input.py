from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.branched_publisher_data import cases, issuance, transitions, stage_request, prefix
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput


def branch_input(case=None, workers=None):
    case = case or cases()[0]
    workers = workers or {"worker-a": case["records"][0]["reference"], "worker-b": case["records"][1]["reference"]}
    req = stage_request(case, "publisher", workers=workers)
    raw = prefix(case, "actual shared plan", workers, issuance(case) + transitions(case), req)
    return BranchPrivacyInput(raw, available_journal_sequence=20, prefix_journal_sequence=20)


def test_branch_matrix_not_silently_pooled_and_cross_worker_occurrences_retained():
    sample = branch_input()
    assert sample.owner_indices == (0, 1) and sample.worker_presence.tolist() == [[True, False], [False, True]]
    assert sample.release.visible_features.tolist() == [[1., 0.], [1., 0.]]
    case = cases()[0]
    sample = branch_input(case, {"worker-a": "No references", "worker-b": " ".join(f["literal"] for f in case["tracked_fields"])})
    assert sample.worker_presence.tolist() == [[False, True], [False, True]]
    assert sample.owner_indices == (0, 1)


@pytest.mark.parametrize("change", ["cycle", "owner", "publisher", "pooled", "future", "label"])
def test_branch_contract_rejects_inconsistent_or_future_views(change):
    sample = branch_input()
    raw = sample.observation
    if change == "cycle":
        raw["workflow"]["edges"].append(["publisher", "planner"])
    elif change == "owner":
        raw["field_owners"]["reference_a"] = "unknown-worker"
    elif change == "publisher":
        raw["publisher_payload"]["earlier_agent_message"] = "different"
    elif change == "pooled":
        raw["pooled_scope"]["worker_text"] = "different"
    elif change == "label":
        raw["final_label"] = 1
    with pytest.raises(ValueError):
        BranchPrivacyInput(raw, available_journal_sequence=21 if change == "future" else 20, prefix_journal_sequence=20)


def test_actual_message_occurrence_not_json_escape_artifact():
    case = cases()[0]
    value = 'Record "alpha" \\ restricted'
    case["tracked_fields"][0]["literal"] = value
    case["records"][0]["reference"] = value
    sample = branch_input(case)
    assert sample.worker_presence[0, 0] and sample.release.packet["groups"][0]["worker_present"]
    assert not sample.legacy_pool_agrees
    sample.worker_presence[0, 0] = False
    with pytest.raises(ValueError):
        sample.validate()
