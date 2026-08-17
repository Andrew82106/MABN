"""Offline ledger replay; it has no transport imports and performs no I/O beyond artifacts."""

from __future__ import annotations

from typing import Any

from .common import append_jsonl, artifact_paths, load_json, read_jsonl, write_text_once
from .validation import render_delivery, validate_run


def replay_run(run_id: str, write_audit: bool = True) -> dict[str, Any]:
    validation = validate_run(run_id, write_audit=False)
    if not validation.get("passed"):
        result = {
            "passed": False,
            "run_id": run_id,
            "errors": ["validation_failed_before_replay", *validation.get("errors", [])],
            "consumed_event_count": 0,
            "consumed_transcript_count": 0,
            "replay_model_calls": 0,
            "replay_network_calls": 0,
        }
        try:
            append_jsonl(artifact_paths(run_id)["replay"], {"record_type": "replay", **result})
        except Exception:
            pass
        return result
    paths = artifact_paths(run_id)
    events = read_jsonl(paths["events"])
    transcripts = read_jsonl(paths["transcripts"])
    terminals = [item for item in events if item.get("record_type") == "model_terminal"]
    errors: list[str] = []
    if len(terminals) != len(transcripts):
        errors.append("terminal_transcript_count_mismatch")
    terminal_ids = [item.get("request_id") for item in terminals]
    transcript_ids = [item.get("request_id") for item in transcripts]
    if terminal_ids != transcript_ids:
        errors.append("ledger_order_or_identity_mismatch")
    result: dict[str, Any] = {
        "passed": not errors,
        "run_id": run_id,
        "errors": errors,
        "consumed_event_count": len(events),
        "consumed_transcript_count": len(transcripts),
        "replay_model_calls": 0,
        "replay_network_calls": 0,
    }
    if write_audit:
        append_jsonl(paths["replay"], {"record_type": "replay", **result})
    if result["passed"] and not paths["delivery"].exists():
        outcome = load_json(paths["outcome"])
        manifest = load_json(paths["manifest"])
        write_text_once(paths["delivery"], render_delivery(run_id, outcome, manifest))
    return result
