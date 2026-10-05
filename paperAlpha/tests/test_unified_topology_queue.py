import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from paperalpha_runtime.unified_topology_queue import (  # noqa: E402
    UnifiedTopologyQueue,
    run_topology,
    topology_edges,
)


@pytest.mark.parametrize(
    ("topology", "nodes", "edges"),
    [
        ("chain", ["a", "b", "c"], [("a", "b"), ("b", "c")]),
        ("fork", ["root", "a", "b"], [("root", "a"), ("root", "b")]),
        ("join", ["a", "b", "join"], [("a", "join"), ("b", "join")]),
        ("review", ["work", "review"], [("work", "review")]),
    ],
)
def test_edges_are_explicit_and_deterministic(topology, nodes, edges):
    assert topology_edges(topology, nodes) == tuple(edges)


def test_failed_call_is_retained_and_evaluator_is_not_public():
    def handler(request, parents):
        if request.node == "bad":
            raise LookupError("fixture failure")
        return {"ok": True, "label": "should-not-be-public", "parents": list(parents)}

    def evaluator(private):
        return {"label": int(any(row["status"] == "failed" for row in private["records"]))}

    result = run_topology("fork", ["root", "bad", "good"], handler,
                          {"root": {}, "bad": {}, "good": {}}, evaluator=evaluator,
                          run_id="t1")
    monitor = result.monitor_projection()
    assert [row["status"] for row in monitor["records"]] == ["completed", "failed", "completed"]
    assert monitor["failed_request_ids"] == ["t1:bad"]
    assert "label" not in str(monitor)
    assert result.evaluation_view() == {"label": 1}
    assert "evaluation" not in monitor


def test_submit_order_does_not_erase_join_dependencies():
    queue = UnifiedTopologyQueue("join", ["left", "right", "join"], run_id="ordered")
    queue.submit("join", {})
    queue.submit("right", {})
    queue.submit("left", {})

    seen = []

    def handler(request, parents):
        seen.append((request.node, tuple(parents)))
        return {"node": request.node}

    result = queue.run(handler)
    assert seen[-1][0] == "join"
    assert len(seen[-1][1]) == 2
    assert result.records[-1]["parent_ids"] == ["ordered:right", "ordered:left"]


def test_missing_or_duplicate_nodes_rejected():
    queue = UnifiedTopologyQueue("chain", ["a", "b"], run_id="r")
    queue.submit("a")
    with pytest.raises(ValueError):
        queue.submit("a")
    with pytest.raises(ValueError):
        queue.run(lambda request, parents: {})

