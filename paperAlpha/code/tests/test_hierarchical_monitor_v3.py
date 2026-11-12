from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/evaluate_hierarchical_monitor_v3.py'
SPEC = importlib.util.spec_from_file_location("hierarchical_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tool_side_effect_is_forbidden_in_runtime_events():
    trace = {"episode_id": "ep-1", "events": [{"actor": "a0", "tool_side_effect": True}]}
    try:
        MODULE.validate_public_trace(trace)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("terminal effect fields must not enter workflow-known evaluator")


def test_workflow_features_distinguish_missing_runtime_edge():
    base = {
        "agents": ["a0", "a1"],
        "workflow_dag": {"nodes": ["a0", "a1"], "edges": [["a0", "a1"]]},
        "events": [{"actor": "a0", "event_type": "input_received", "data_class": "untrusted"}],
    }
    missing = MODULE.workflow_features(base)
    observed = {**base, "events": base["events"] + [{"actor": "a0", "target": "a1", "event_type": "message_sent", "data_class": "clean", "edge_confidence": 1.0}]}
    present = MODULE.workflow_features(observed)
    assert missing[4] > present[4]  # missing_expected_edges
    assert present[3] > missing[3]  # expected/runtime overlap
