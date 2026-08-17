"""Offline protocol runner.  It only accepts injected transports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

from .config import (
    CARDS,
    MAX_CONTENT_LENGTH,
    PER_CALL_TIMEOUT_SECONDS,
    SCHEMA_VERSION,
    TARGET_MODEL,
    canonical_json,
    profile_commitment,
    public_profile_snapshot,
    schema_version_for_run_kind,
    sha256_json,
    sha256_text,
)
from .evidence import (
    artifact_manifest,
    build_ledger,
    derive_outcome,
    derive_report,
    local_delivery_text,
    run_directory,
    transcript_semantic_view,
    write_json,
    write_text,
)
from .fake_scenarios import successful_completion_steps, successful_metadata_step
from .fake_transport import FakeStep, FakeTransport, ProtocolTransportError, ManualClock
from .historical_integrity import historical_tree_snapshot


FATAL_OUTCOMES = {
    "deadline_exhausted",
    "transport_error",
    "timeout",
    "http_error",
    "connection_error",
    "redirect",
    "metadata_malformed",
    "metadata_target_absent",
    "model_mismatch",
    "identity_missing",
    "unsafe_provider_identity",
    "malformed_envelope",
}


class DeadlineGuard:
    """A monotonic total-budget guard shared by metadata and all completions."""

    def __init__(self, total_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.start = clock()
        self.total_seconds = float(total_seconds)

    def remaining_seconds(self) -> float:
        return self.total_seconds - (self.clock() - self.start)

    def may_start_io(self) -> bool:
        return self.remaining_seconds() > 0.0

    def timeout_seconds(self) -> float:
        return max(0.0, min(PER_CALL_TIMEOUT_SECONDS, self.remaining_seconds()))

    def exhausted(self) -> bool:
        return self.remaining_seconds() <= 0.0


def _safe_transport_category(error: ProtocolTransportError) -> str:
    code = error.code
    if code == "timeout":
        return "timeout"
    if code == "connection_error":
        return "connection_error"
    if code == "redirect":
        return "redirect"
    if isinstance(code, str) and code.startswith("http_") and code[5:].isdigit():
        return "http_error"
    return "transport_error"


def _unstarted_record(sequence: int, card: Any, reason: str) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "card_id": card.card_id,
        "profile": card.profile,
        "request_sha256": card.request_sha256,
        "started": False,
        "outcome_category": "unstarted",
        "expected_fields_satisfied": False,
        "content_length": None,
        "content_sha256": None,
        "provider_identity": None,
        "timeout_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "stop_reason": reason,
    }


def _normalise_models(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        return None, "metadata_malformed"
    normalized: set[str] = set()
    for item in raw["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return None, "metadata_malformed"
        model_id = item["id"]
        if not model_id or len(model_id) > 160 or any(ord(char) < 32 for char in model_id):
            return None, "metadata_malformed"
        normalized.add(model_id)
    ordered = sorted(normalized)
    return {
        "status": "accepted" if TARGET_MODEL in normalized else "metadata_target_absent",
        "target_model_present": TARGET_MODEL in normalized,
        "model_count": len(ordered),
        "model_list_sha256": sha256_json(ordered),
        "stop_reason": None if TARGET_MODEL in normalized else "metadata_target_absent",
    }, None


def _normal_content_record(
    card: Any, content: str, identity: dict[str, str | None], base: dict[str, Any]
) -> dict[str, Any]:
    if not content:
        return {**base, "outcome_category": "empty_content", "provider_identity": identity}
    if len(content) > MAX_CONTENT_LENGTH:
        return {**base, "outcome_category": "content_too_long", "provider_identity": identity}
    enriched = {
        **base,
        "provider_identity": identity,
        "content_length": len(content),
        "content_sha256": sha256_text(content),
    }
    if card.profile == "plain_text":
        return {
            **enriched,
            "outcome_category": "content_match"
            if content == card.expected_answer()
            else "response_contract_mismatch",
            "expected_fields_satisfied": content == card.expected_answer(),
        }
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {**enriched, "outcome_category": "non_json"}
    if not isinstance(parsed, dict) or set(parsed) != {"probe_id", "answer"}:
        return {**enriched, "outcome_category": "field_mismatch"}
    if not all(isinstance(parsed[key], str) for key in ("probe_id", "answer")):
        return {**enriched, "outcome_category": "field_mismatch"}
    matches = (
        parsed["probe_id"] == card.expected_probe_id()
        and parsed["answer"] == card.expected_answer()
    )
    return {
        **enriched,
        "outcome_category": "content_match" if matches else "field_mismatch",
        "expected_fields_satisfied": matches,
    }


def classify_completion(card: Any, envelope: Any, base: dict[str, Any]) -> dict[str, Any]:
    """Classify an in-memory response without retaining its raw content."""
    if not isinstance(envelope, dict):
        return {**base, "outcome_category": "malformed_envelope"}
    identity_raw = {
        "declared_model": envelope.get("model"),
        "request_id": envelope.get("id"),
        "system_fingerprint": envelope.get("system_fingerprint"),
        "provider_version": envelope.get("provider_version"),
    }
    identity: dict[str, str | None] = {}
    for key, value in identity_raw.items():
        if value is None:
            identity[key] = None
        elif not isinstance(value, str) or not value or len(value) > 160 or any(
            ord(char) < 32 or ord(char) == 127 for char in value
        ):
            return {**base, "outcome_category": "unsafe_provider_identity"}
        else:
            identity[key] = value
    if identity["declared_model"] is None:
        return {**base, "outcome_category": "identity_missing", "provider_identity": identity}
    if identity["declared_model"] != TARGET_MODEL:
        return {**base, "outcome_category": "model_mismatch", "provider_identity": identity}
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return {**base, "outcome_category": "malformed_envelope", "provider_identity": identity}
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return {**base, "outcome_category": "malformed_envelope", "provider_identity": identity}
    allowed_message_fields = {
        "role",
        "content",
        "tool_calls",
        "function_call",
        "refusal",
        "reasoning",
        "reasoning_content",
    }
    if not set(message).issubset(allowed_message_fields):
        return {**base, "outcome_category": "malformed_envelope", "provider_identity": identity}
    if message.get("tool_calls") is not None:
        return {**base, "outcome_category": "tool_calls", "provider_identity": identity}
    if message.get("function_call") is not None:
        return {**base, "outcome_category": "function_call", "provider_identity": identity}
    if message.get("reasoning") is not None or message.get("reasoning_content") is not None:
        return {**base, "outcome_category": "reasoning_detected", "provider_identity": identity}
    refusal = message.get("refusal")
    if refusal is not None:
        if not isinstance(refusal, str):
            return {**base, "outcome_category": "malformed_envelope", "provider_identity": identity}
        return {**base, "outcome_category": "refusal", "provider_identity": identity}
    content = message.get("content")
    if not isinstance(content, str):
        return {**base, "outcome_category": "malformed_envelope", "provider_identity": identity}
    if "<think>" in content.lower():
        return {**base, "outcome_category": "reasoning_detected", "provider_identity": identity}
    return _normal_content_record(card, content, identity, base)


def _append_unstarted(records: list[dict[str, Any]], reason: str) -> None:
    for sequence in range(len(records) + 1, len(CARDS) + 1):
        records.append(_unstarted_record(sequence, CARDS[sequence - 1], reason))


def execute_protocol(
    transport: Any,
    *,
    run_id: str,
    run_kind: str = "offline_fake",
    total_deadline_seconds: float = 150.0,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run metadata plus four frozen cards through an injected transport only."""
    guard = DeadlineGuard(total_deadline_seconds, clock)
    records: list[dict[str, Any]] = []
    stop_reason: str | None = None
    metadata: dict[str, Any]
    if not guard.may_start_io():
        metadata = {
            "status": "unstarted",
            "target_model_present": None,
            "model_count": None,
            "model_list_sha256": None,
            "stop_reason": "deadline_exhausted_preflight",
            "timeout_seconds": 0.0,
            "elapsed_seconds": 0.0,
        }
        stop_reason = "deadline_exhausted_preflight"
    else:
        metadata_timeout = guard.timeout_seconds()
        metadata_start = clock()
        try:
            raw_metadata = transport.get_models(metadata_timeout)
        except ProtocolTransportError as error:
            category = _safe_transport_category(error)
            metadata = {
                "status": category,
                "target_model_present": None,
                "model_count": None,
                "model_list_sha256": None,
                "stop_reason": category,
                "timeout_seconds": metadata_timeout,
                "elapsed_seconds": max(0.0, clock() - metadata_start),
            }
            stop_reason = category
        else:
            if guard.exhausted():
                metadata = {
                    "status": "deadline_exhausted",
                    "target_model_present": None,
                    "model_count": None,
                    "model_list_sha256": None,
                    "stop_reason": "deadline_exhausted",
                    "timeout_seconds": metadata_timeout,
                    "elapsed_seconds": max(0.0, clock() - metadata_start),
                }
                stop_reason = "deadline_exhausted"
            else:
                summary, issue = _normalise_models(raw_metadata)
                if issue is not None:
                    metadata = {
                        "status": issue,
                        "target_model_present": None,
                        "model_count": None,
                        "model_list_sha256": None,
                        "stop_reason": issue,
                        "timeout_seconds": metadata_timeout,
                        "elapsed_seconds": max(0.0, clock() - metadata_start),
                    }
                    stop_reason = issue
                else:
                    metadata = {
                        **summary,
                        "timeout_seconds": metadata_timeout,
                        "elapsed_seconds": max(0.0, clock() - metadata_start),
                    }
                    if summary["status"] != "accepted":
                        stop_reason = str(summary["status"])
    for sequence, card in enumerate(CARDS, start=1):
        if stop_reason is not None:
            records.append(_unstarted_record(sequence, card, stop_reason))
            continue
        if not guard.may_start_io():
            stop_reason = "deadline_exhausted_preflight"
            records.append(_unstarted_record(sequence, card, stop_reason))
            continue
        timeout_seconds = guard.timeout_seconds()
        call_start = clock()
        base = {
            "sequence": sequence,
            "card_id": card.card_id,
            "profile": card.profile,
            "request_sha256": card.request_sha256,
            "started": True,
            "expected_fields_satisfied": False,
            "content_length": None,
            "content_sha256": None,
            "provider_identity": None,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": 0.0,
            "stop_reason": None,
        }
        try:
            envelope = transport.complete(card.request_payload(), timeout_seconds)
        except ProtocolTransportError as error:
            category = _safe_transport_category(error)
            record = {
                **base,
                "outcome_category": category,
                "elapsed_seconds": max(0.0, clock() - call_start),
                "stop_reason": category,
            }
            records.append(record)
            stop_reason = category
            continue
        if guard.exhausted():
            record = {
                **base,
                "outcome_category": "deadline_exhausted",
                "elapsed_seconds": max(0.0, clock() - call_start),
                "stop_reason": "deadline_exhausted",
            }
            records.append(record)
            stop_reason = "deadline_exhausted"
            continue
        record = classify_completion(card, envelope, base)
        record["elapsed_seconds"] = max(0.0, clock() - call_start)
        if record["outcome_category"] in FATAL_OUTCOMES:
            record["stop_reason"] = record["outcome_category"]
            stop_reason = record["outcome_category"]
        records.append(record)
    if len(records) != len(CARDS):
        _append_unstarted(records, stop_reason or "protocol_incomplete")
    return {
        "schema_version": schema_version_for_run_kind(run_kind),
        "run_id": run_id,
        "run_kind": run_kind,
        "profile_commitment": profile_commitment(run_kind),
        "metadata": metadata,
        "records": records,
        "stop_reason": stop_reason,
        "transport_call_count": len(getattr(transport, "calls", [])),
        "total_deadline_seconds": float(total_deadline_seconds),
    }


def expected_fake_semantic_view() -> dict[str, Any]:
    """A source-frozen oracle for the deterministic fake run, with no raw reply."""
    clock = ManualClock()
    transport = FakeTransport(
        successful_metadata_step(), successful_completion_steps(), clock
    )
    transcript = execute_protocol(
        transport,
        run_id="P1V2Q2C-FAKE-ORACLE",
        total_deadline_seconds=150.0,
        clock=clock,
    )
    return transcript_semantic_view(transcript)


def make_fake_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"P1V2Q2C-FAKE-{timestamp}"


def persist_offline_transcript(
    data_root: Path,
    transcript: dict[str, Any],
    history_before: dict[str, Any],
    history_after: dict[str, Any],
) -> Path:
    """Persist only a pre-redacted fake transcript and its local evidence bundle."""
    actual_run_id = transcript.get("run_id")
    if not isinstance(actual_run_id, str):
        raise ValueError("missing_run_id")
    run_dir = run_directory(data_root, actual_run_id)
    if run_dir.exists():
        raise FileExistsError("run_id_already_exists")
    if history_before != history_after:
        raise RuntimeError("historical_tree_changed")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_snapshot = {
        "schema_version": SCHEMA_VERSION,
        "offline_only": True,
        "profile_commitment": profile_commitment(),
        "public_profile": public_profile_snapshot(),
    }
    history = {"before": history_before, "after": history_after, "unchanged": True}
    ledger = build_ledger(transcript)
    outcome = derive_outcome(transcript)
    report = derive_report(transcript, outcome)
    write_json(run_dir / "config_snapshot.json", config_snapshot)
    write_json(run_dir / "transcript.json", transcript)
    write_json(run_dir / "ledger.json", ledger)
    write_json(run_dir / "outcome.json", outcome)
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "historical_integrity.json", history)
    write_text(run_dir / "reports" / "DELIVERY.md", local_delivery_text(actual_run_id, outcome))
    write_json(run_dir / "run_manifest.json", artifact_manifest(run_dir, actual_run_id, "offline_fake"))
    return run_dir


def create_offline_fake_run(data_root: Path, run_id: str | None = None) -> tuple[str, Path]:
    """Create one self-contained fake run; this function never creates I/O outside data_root."""
    actual_run_id = run_id or make_fake_run_id()
    before = historical_tree_snapshot()
    clock = ManualClock()
    transport = FakeTransport(
        successful_metadata_step(), successful_completion_steps(), clock
    )
    transcript = execute_protocol(
        transport,
        run_id=actual_run_id,
        total_deadline_seconds=150.0,
        clock=clock,
    )
    after = historical_tree_snapshot()
    run_dir = persist_offline_transcript(data_root, transcript, before, after)
    return actual_run_id, run_dir
