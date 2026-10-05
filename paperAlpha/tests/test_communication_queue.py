import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import collect_frozen_communication_queue_v1 as queue_script  # noqa: E402


def test_order_covers_four_topologies_and_three_modes():
    order = queue_script.build_order(1)
    assert len(order) == 12
    assert {(row["topology"], row["communication_mode"]) for row in order} == {
        (topology, mode)
        for topology in queue_script.TOPOLOGIES
        for mode in queue_script.COMMUNICATION_MODES
    }
    assert [row["episode_id"] for row in order] == [
        f"{topology}-{mode}-000"
        for topology in queue_script.TOPOLOGIES
        for mode in queue_script.COMMUNICATION_MODES
    ]


def test_communication_context_has_distinct_observable_contracts():
    parents = {
        "p1": {"node": "worker_a", "status": "completed",
               "response": {"text": '{"decision":"deny","finding":"scope"}'}},
        "p2": {"node": "worker_b", "status": "failed",
               "error_message": "provider unavailable"},
    }
    direct = queue_script.communication_context("direct", parents)
    summary = queue_script.communication_context("summary", parents)
    vote = queue_script.communication_context("vote", parents)
    assert direct["parents"][0]["response"].startswith("{")
    assert "response" not in summary["parents"][0]
    assert summary["parents"][0]["summary_chars"] == len(summary["parents"][0]["summary"])
    assert vote["parents"][0]["vote"] == "deny"
    assert vote["parents"][1]["vote"] == "unknown"
    assert vote["vote_counts"] == {"allow": 0, "deny": 1, "unknown": 1}


def test_run_episode_keeps_failures_and_separates_evaluator(monkeypatch, tmp_path):
    calls = []

    def fake_api(endpoint, key, model, prompt, timeout, max_tokens):
        calls.append(prompt)
        if 'Node: authorize' in prompt:
            raise RuntimeError("fixture provider failure")
        return {
            "choices": [{"message": {"content":
                                      '{"decision":"deny","finding":"scope","handoff":"review"}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }, {"status": 200, "elapsed_seconds": 0.001}

    monkeypatch.setattr(queue_script, "api_call", fake_api)
    result = queue_script.run_episode(
        "http://localhost:58661/v1", "fixture-key", "fixture-model", tmp_path,
        "chain", "vote", 0, "violation", "fixture task", 1, 20,
    )
    assert result["stats"]["requests"] == 3
    assert result["stats"]["failed"] == 1
    assert result["monitor_failed_records"] == 1
    episode = tmp_path / "episodes" / "chain-vote-000"
    monitor = json.loads((episode / "monitor.json").read_text(encoding="utf-8"))
    evaluator = json.loads((episode / "evaluator.json").read_text(encoding="utf-8"))
    assert monitor["failed_request_ids"] == ["chain-vote-000:authorize"]
    assert "expected_policy_intent" not in json.dumps(monitor)
    assert evaluator["expected_policy_intent"] == 1
    assert evaluator["communication_mode"] == "vote"
    assert len(calls) == 3


def test_plan_marks_policy_labels_as_evaluator_boundary():
    plan = queue_script.make_plan("http://localhost:58661/v1", "fixture", 1, 7)
    assert plan["total_episodes"] == 12
    assert plan["communication_modes"] == ["direct", "summary", "vote"]
    assert "evaluator-only" in plan["monitor_boundary"]


def test_invalid_episode_count_rejected():
    with pytest.raises(ValueError):
        queue_script.build_order(0)
