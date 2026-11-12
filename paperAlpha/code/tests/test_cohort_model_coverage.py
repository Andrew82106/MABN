import importlib.util
from pathlib import Path
import sys

from test_scoped_budget_bn import scenario

SCRIPTS = Path(__file__).resolve().parents[2] / 'code/commands'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("cohort_coverage_test", SCRIPTS / "inspect_cohort_model_coverage.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def test_replay_reports_missing_evidence_instead_of_creating_labels_or_context():
    report = coverage.inspect_rows(scenario())
    assert report["actions"] == report["actions_with_unknown_role"] == report["actions_with_unknown_permission"] == 2
    assert report["actions_with_unknown_capability"] == 0
    assert report["text_records"] == report["text_extraction_unavailable"] == 3
    assert report["instantiated_factors"] == {"authorization": 2, "declared_bound": 2}
    assert "label" not in report and "risk_probability" not in report


def test_failed_run_is_replayed_but_not_relabelled_successful():
    rows = scenario()
    rows[-1]["payload"]["status"] = "failed"
    report = coverage.inspect_rows(rows)
    assert report["model_events"] == 5
    assert "cannot make its endpoint label eligible" in report["interpretation"]
