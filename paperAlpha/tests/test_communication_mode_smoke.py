import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_communication_mode_smoke_v1 import _parent_context  # noqa: E402


def _parents():
    return {
        "run:a": {
            "node": "a",
            "status": "completed",
            "response": {"text": '{"decision":"deny","finding":"scope","handoff":"review"}'},
        },
        "run:b": {"node": "b", "status": "failed"},
    }


def test_direct_retains_parent_report_without_private_evaluator_fields():
    context = _parent_context("direct", _parents())
    assert context[0]["response"].startswith('{"decision"')
    assert "expected_policy_intent" not in str(context)


def test_summary_projects_status_and_decision_only():
    context = _parent_context("summary", _parents())
    assert context["completed"] == 1
    assert context["failed"] == 1
    assert context["reports"][0]["decision"] == "deny"
    assert "finding" not in str(context)


def test_vote_counts_unknown_failed_parent():
    context = _parent_context("vote", _parents())
    assert context["vote_counts"] == {"allow": 0, "deny": 1, "escalate": 0, "unknown": 1}
