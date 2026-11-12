from pathlib import Path

import importlib.util


SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/evaluate_independent_fair_baselines_v1.py'
SPEC = importlib.util.spec_from_file_location("fair_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_public_episode_rejects_hidden_fields():
    labels = {"ep-1": 1}
    try:
        MODULE.public_episode({"episode_id": "ep-1", "hidden_fields": {}}, labels)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("hidden fields must not enter the public evaluator")


def test_group_split_is_deterministic_and_disjoint():
    eps = [
        {"episode_id": f"ep-{i}", "scenario_family": family, "label": i % 2}
        for i, family in enumerate(["b", "a", "c", "a", "b", "d", "e", "c", "d", "e"])
    ]
    train_a, test_a, split_a = MODULE.group_split(eps, 0.4)
    train_b, test_b, split_b = MODULE.group_split(eps, 0.4)
    assert train_a.tolist() == train_b.tolist()
    assert test_a.tolist() == test_b.tolist()
    assert set(train_a).isdisjoint(set(test_a))
    assert split_a["held_out_groups"] == split_b["held_out_groups"]


def test_feature_extractors_do_not_change_with_evaluation_label():
    ep = {
        "episode_id": "ep-1",
        "agents": ["a0", "a1"],
        "scenario_family": "family-only-for-split",
        "events": [
            {"sequence": 0, "actor": "a0", "event_type": "input_received", "data_class": "untrusted"},
            {"sequence": 1, "actor": "a0", "event_type": "message_sent", "target": "a1", "data_class": "untrusted", "edge_kind": "workflow", "edge_confidence": 0.8},
            {"sequence": 2, "actor": "a1", "event_type": "tool_call", "tool": "shell.exec", "data_class": "clean"},
        ],
    }
    before = (
        MODULE.aggregate_features(ep),
        MODULE.local_features(ep),
        MODULE.graph_features(ep),
    )
    after_ep = {**ep, "label": 1}
    after = (
        MODULE.aggregate_features(after_ep),
        MODULE.local_features(after_ep),
        MODULE.graph_features(after_ep),
    )
    for left, right in zip(before, after):
        assert (left == right).all()
