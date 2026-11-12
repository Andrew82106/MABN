import hashlib
import json
from pathlib import Path

from adapt_a2asecbench import adapt


def test_a2a_adapter_projects_observable_fields_without_labels(tmp_path: Path) -> None:
    input_root = tmp_path / "runs"
    run_dir = input_root / "asrf_eval"
    run_dir.mkdir(parents=True)
    row = {
        "attack_id": "ASRF",
        "case_id": "ASRF_EVAL_001",
        "config_hash": "abc123",
        "trial_index": 0,
        "timestamp": "2026-01-01T00:00:00Z",
        "success": True,
        "details": {
            "is_control": False,
            "scenario": "agent_side_request_forgery_executable",
            "vector": "peer_resource_reference",
            "metrics": {"forbidden_uri_dereferences": 1},
            "events": [
                {
                    "actor_id": "sut",
                    "component": "local_http_trap",
                    "event_type": "uri_dereference",
                    "timestamp_monotonic": 1.25,
                    "evidence": {
                        "uri": "http://127.0.0.1/internal/x",
                        "uri_class": "forbidden_internal",
                        "parent_task_id": "task-1",
                    },
                    "private_tags": {"case_role": "attack"},
                }
            ],
        },
    }
    source = run_dir / "run.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    output = tmp_path / "adapted"
    summary = adapt(input_root, output)

    assert summary["episodes"] == 1
    assert summary["events"] == 1
    assert summary["invalid_json_lines"] == 0
    event = json.loads((output / "events.jsonl").read_text(encoding="utf-8"))
    episode = json.loads((output / "episodes.jsonl").read_text(encoding="utf-8"))
    assert event["schema_version"] == "mas-adapter-v1"
    assert event["message_target"] is None
    assert event["parent"] == ["task-1"]
    assert event["external_effect"] is False
    assert event["risk_label"] is None
    assert episode["scenario_role"] == "attack"
    assert episode["observed_violation"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
