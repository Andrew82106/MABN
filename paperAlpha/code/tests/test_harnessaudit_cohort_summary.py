import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'code/commands'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("ha_cohort_summary_test", SCRIPTS / "summarize_harnessaudit_cohort.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.mark.parametrize("label", [0, 1])
def test_only_completed_independently_audited_outcomes_get_labels(label):
    evaluation = {"target": "any_recorded_access_rule_violation", "label": label,
                  "eligible": True, "ineligibility_reasons": [], "meaning": "not actual harm"}
    assert summary.qualified_label(evaluation, "completed", True)["label"] == label
    for status, audit in [("failed", True), ("completed", False), ("failed", False)]:
        result = summary.qualified_label(evaluation, status, audit)
        assert result["label"] is None and not result["eligible"]


def test_missing_audit_coverage_stays_unknown_even_if_run_finished():
    evaluation = {"target": "any_recorded_access_rule_violation", "label": None,
                  "eligible": False, "ineligibility_reasons": ["actions_without_audit_decisions"], "meaning": "not actual harm"}
    assert summary.qualified_label(evaluation, "completed", True)["label"] is None


@pytest.mark.parametrize("labels", [[1, 1, None], [0, 0, None], [None, None]])
def test_single_class_and_failures_cannot_manufacture_discrimination(labels):
    result = summary.class_readiness([{"label": y} for y in labels])
    assert not result["both_outcome_classes_present"]
    assert result["label_counts"]["null"] == labels.count(None)
    assert not result["final_test_eligible"]


def test_two_classes_do_not_automatically_make_a_final_test():
    result = summary.class_readiness([{"label": 0}, {"label": 1}])
    assert result["both_outcome_classes_present"]
    assert not result["independent_performance_claim_eligible"]


def test_rule_matches_are_not_unique_actions_and_ungraded_is_not_low():
    evaluation = {"decisions": [
        {"action_sequence_num": 1, "violated": True, "violation_code": "V-OT", "severity": "low"},
        {"action_sequence_num": 1, "violated": True, "violation_code": "V-OT", "severity": "low"},
        {"action_sequence_num": 2, "violated": True, "violation_code": "V-OR", "severity": None},
        {"action_sequence_num": 3, "violated": False},
    ]}
    assert summary.rule_match_profile(evaluation) == {
        "V-OT:low": {"rule_matches": 2, "distinct_actions": 1},
        "V-OR:ungraded": {"rule_matches": 1, "distinct_actions": 1},
    }


def run_binding_fixture(tmp_path):
    task_path = "multi_agent/tasks/daily_life/dining/dl-t18.yaml"
    task_source = tmp_path / "source_snapshot/upstream" / task_path
    pilot_source = tmp_path / "source_snapshot/collect_harnessaudit_api_pilot.py"
    summary.write_new(task_source, {"fixture": "task"})
    summary.write_new(pilot_source, {"fixture": "pilot"})
    plan = {"plan_sha256": "frozen-plan", "upstream_commit": "commit", "created_at": "2026-09-19T00:00:00+00:00",
            "settings": {"model": "gpt-5.5", "request_budget": 32, "max_turns": 8, "timeout": 300},
            "source_sha256": {'code/commands/collect_harnessaudit_api_pilot.py': summary.sha(pilot_source)}}
    task = {"task_id": "dl-t18", "task_path": task_path, "task_sha256": summary.sha(task_source)}
    manifest = {"task_id": "dl-t18", "upstream_commit": "commit", "model_requested": "gpt-5.5",
                "request_budget": 32, "max_hub_turns": 8, "agent_timeout_seconds": 300,
                "status": "completed", "started_at": "2026-09-19T00:01:00+00:00"}
    summary.write_new(tmp_path / "RUN.json", manifest)
    receipt = {"plan_sha256": "frozen-plan", "task_id": "dl-t18", "run_manifest_sha256": summary.sha(tmp_path / "RUN.json")}
    return plan, task, manifest, receipt


def test_run_is_bound_to_frozen_task_settings_source_and_receipt(tmp_path):
    plan, task, manifest, receipt = run_binding_fixture(tmp_path)
    summary.validate_run(plan, task, manifest, receipt, tmp_path)
    receipt["run_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="manifest changed"):
        summary.validate_run(plan, task, manifest, receipt, tmp_path)


@pytest.mark.parametrize("key,value", [("model_requested", "other"), ("request_budget", 64),
                                      ("status", "running"), ("started_at", "2026-09-18T00:00:00+00:00")])
def test_incompatible_or_preexisting_runs_cannot_enter_cohort(tmp_path, key, value):
    plan, task, manifest, receipt = run_binding_fixture(tmp_path)
    manifest[key] = value
    with pytest.raises(ValueError):
        summary.validate_run(plan, task, manifest, receipt, tmp_path)
