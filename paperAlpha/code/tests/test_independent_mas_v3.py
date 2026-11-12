from pathlib import Path
import importlib.util
import random


SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/generate_independent_mas_benchmark_v3.py'
SPEC = importlib.util.spec_from_file_location("independent_mas_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_events_have_no_terminal_label_fields():
    episode = MODULE.generate(0, random.Random(11))
    forbidden = {"label", "external_effect", "hidden_fields", "label_source", "mechanism"}
    assert all(not forbidden.intersection(e) for e in episode["events"])


def test_message_order_respects_forward_chain():
    for i in range(20):
        episode = MODULE.generate(i, random.Random(100 + i))
        messages = [e for e in episode["events"] if e["event_type"] == "message_sent"]
        for first in messages:
            for second in messages:
                if first["target"] == second["actor"]:
                    assert first["sequence"] < second["sequence"]


def test_runtime_graph_is_not_always_identical_to_workflow():
    differs = 0
    for i in range(30):
        episode = MODULE.generate(i, random.Random(200 + i))
        workflow = {tuple(e) for e in episode["workflow_dag"]["edges"]}
        runtime = {(e["actor"], e["target"]) for e in episode["events"] if e["event_type"] == "message_sent"}
        differs += int(workflow != runtime)
    assert differs > 0
