import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_judge_baseline import batches, public_only, select_records


def test_public_only_preserves_trace_fields_and_rejects_labels():
    row = {"episode_id": "e1", "events": [{"target": "a2", "data_class": "clean"}]}
    assert public_only(row) == row
    try:
        public_only({"episode_id": "e1", "risk_label": 1})
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("label field must be rejected")


def test_sampling_is_stable_and_sorted():
    rows = [{"episode_id": f"e{i}"} for i in range(10, -1, -1)]
    a = select_records(rows, 5, 123)
    b = select_records(rows, 5, 123)
    assert a == b
    assert [r["episode_id"] for r in a] == sorted(r["episode_id"] for r in a)


def test_batching_has_fixed_order():
    rows = [{"episode_id": str(i)} for i in range(5)]
    assert [[x["episode_id"] for x in g] for g in batches(rows, 2)] == [["0", "1"], ["2", "3"], ["4"]]
