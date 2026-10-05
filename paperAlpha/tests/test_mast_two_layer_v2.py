import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mast_two_layer_v2_test", ROOT / "scripts" / "evaluate_mast_two_layer_v2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mast_projection_strips_terminal_evaluation_and_labels():
    row = {
        "mas_name": "ToyMAS",
        "benchmark_name": "Toy",
        "trace_id": 1,
        "trace": (
            "Message to Worker Agent\n"
            "Response from Worker Agent\n"
            "\nEvaluation\n"
            '{"success": false, "failure": "delete audit records"}'
        ),
        "mast_annotation": {"4.2": True},
    }
    episode = MODULE.row_to_episode(row, include_text=True)
    assert "Evaluation" not in episode["task_text"]
    assert "delete audit records" not in episode["task_text"]
    assert all("mast_annotation" not in event for event in episode["events"])
    assert episode["label"] == 1  # joined only after the public projection


def test_mast_projection_keeps_public_message_edges():
    row = {
        "mas_name": "ToyMAS",
        "benchmark_name": "Toy",
        "trace_id": 2,
        "trace": "Message to Worker Agent\nResponse from Worker Agent\n",
        "mast_annotation": {},
    }
    episode = MODULE.row_to_episode(row, include_text=False)
    assert {event["actor"] for event in episode["events"]} == {"supervisor", "worker"}
    assert all(event["event_type"] == "message_sent" for event in episode["events"])
