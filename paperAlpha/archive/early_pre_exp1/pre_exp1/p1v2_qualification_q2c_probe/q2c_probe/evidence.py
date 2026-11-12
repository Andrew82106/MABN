"""Safe artifact construction, hashing, and evidence-chain utilities."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .config import (
    canonical_json,
    profile_commitment,
    schema_version_for_run_kind,
    sha256_json,
)


GENESIS_EVENT_HASH = "0" * 64
REQUIRED_ARTIFACTS = (
    "config_snapshot.json",
    "transcript.json",
    "ledger.json",
    "outcome.json",
    "report.json",
    "historical_integrity.json",
    "reports/DELIVERY.md",
)
LIVE_REQUIRED_ARTIFACTS = REQUIRED_ARTIFACTS + ("live_probe_once_guard.json",)
FORBIDDEN_ARTIFACT_FRAGMENTS = (
    "authorization",
    "api_key",
    "fake_secret",
    "raw_error_body",
    "tool_arguments",
    "q2c_ok",
    "<think>",
)
RUN_ID_PATTERN = re.compile(r"^P1V2Q2C-(?:FAKE|PROBE)-[A-Z0-9-]+$")
LIVE_DELIVERY_TITLE = "# Q2-C Live Probe Delivery"
OFFLINE_DELIVERY_TITLE = "# Q2-C Offline Fake Run Delivery"
LIVE_DELIVERY_SCOPE = "one constrained Q2-C live probe; credentials and raw responses are absent."
OFFLINE_DELIVERY_SCOPE = "offline fake transport only; no network and no credential access."
DELIVERY_EVIDENCE_BOUNDARY = (
    "replay validates safe summaries and commitments, not discarded raw responses."
)
LIVE_DELIVERY_INTEGRITY_BOUNDARY = (
    "coordinated full-disk rewriting cannot be externally disproven after raw responses are discarded."
)


def is_safe_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.fullmatch(run_id))


def expected_run_kind_for_id(run_id: str) -> str:
    if not is_safe_run_id(run_id):
        raise ValueError("unsafe_run_id")
    if run_id.startswith("P1V2Q2C-FAKE-"):
        return "offline_fake"
    if run_id.startswith("P1V2Q2C-PROBE-"):
        return "live_probe"
    raise ValueError("unrecognized_run_id_prefix")


def run_directory(data_root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not is_safe_run_id(run_id):
        raise ValueError("unsafe_run_id")
    root = data_root.resolve()
    candidate = (root / "runs" / run_id).resolve()
    if candidate.parent != (root / "runs").resolve():
        raise ValueError("run_path_escape")
    return candidate


def required_artifacts_for_kind(run_kind: str) -> tuple[str, ...]:
    if run_kind == "offline_fake":
        return REQUIRED_ARTIFACTS
    if run_kind == "live_probe":
        return LIVE_REQUIRED_ARTIFACTS
    raise ValueError("unsupported_run_kind")


def _artifact_text_is_safe(text: str) -> bool:
    lowered = text.lower()
    return not any(fragment in lowered for fragment in FORBIDDEN_ARTIFACT_FRAGMENTS)


def assert_artifact_text_safe(text: str) -> None:
    if not _artifact_text_is_safe(text):
        raise ValueError("unsafe_artifact_text")


def canonical_file_hash(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return sha256(data).hexdigest(), len(data)


def write_json(path: Path, value: Any) -> None:
    serialized = canonical_json(value) + "\n"
    assert_artifact_text_safe(serialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    assert_artifact_text_safe(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bounded_identity_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 160:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    return value


def extract_safe_identity(envelope: Any) -> tuple[dict[str, str | None], str | None]:
    """Return only bounded identity facts, or a reason for an unsafe identity."""
    if not isinstance(envelope, dict):
        return {}, "malformed_envelope"
    raw = {
        "declared_model": envelope.get("model"),
        "request_id": envelope.get("id"),
        "system_fingerprint": envelope.get("system_fingerprint"),
        "provider_version": envelope.get("provider_version"),
    }
    identity = {key: bounded_identity_value(value) for key, value in raw.items()}
    for key, value in raw.items():
        if value is not None and identity[key] is None:
            return identity, "unsafe_provider_identity"
    return identity, None


def record_semantic_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence": record.get("sequence"),
        "card_id": record.get("card_id"),
        "profile": record.get("profile"),
        "request_sha256": record.get("request_sha256"),
        "started": record.get("started"),
        "outcome_category": record.get("outcome_category"),
        "expected_fields_satisfied": record.get("expected_fields_satisfied"),
        "content_length": record.get("content_length"),
        "content_sha256": record.get("content_sha256"),
        "provider_identity": record.get("provider_identity"),
        "stop_reason": record.get("stop_reason"),
    }


def transcript_semantic_view(transcript: dict[str, Any]) -> dict[str, Any]:
    metadata = transcript.get("metadata", {})
    base = {
        "schema_version": transcript.get("schema_version"),
        "run_kind": transcript.get("run_kind"),
        "profile_commitment": transcript.get("profile_commitment"),
        "metadata": {
            "status": metadata.get("status"),
            "target_model_present": metadata.get("target_model_present"),
            "model_count": metadata.get("model_count"),
            "model_list_sha256": metadata.get("model_list_sha256"),
            "stop_reason": metadata.get("stop_reason"),
        },
        "records": [record_semantic_view(record) for record in transcript.get("records", [])],
        "stop_reason": transcript.get("stop_reason"),
    }
    if transcript.get("run_kind") == "live_probe":
        base["live_execution"] = {
            "live_policy_commitment": transcript.get("live_policy_commitment"),
            "live_runner_source_sha256": transcript.get("live_runner_source_sha256"),
            "live_transport_source_sha256": transcript.get("live_transport_source_sha256"),
            "metadata_call_count": transcript.get("metadata_call_count"),
            "completion_call_count": transcript.get("completion_call_count"),
        }
    return base


def build_ledger(transcript: dict[str, Any]) -> dict[str, Any]:
    run_kind = transcript.get("run_kind")
    schema_version = schema_version_for_run_kind(run_kind)
    semantic = transcript_semantic_view(transcript)
    payloads: list[tuple[str, dict[str, Any]]] = [
        (
            "run_started",
            {
                "schema_version": schema_version,
                "profile_commitment": profile_commitment(run_kind),
                "run_kind": run_kind,
            },
        ),
        ("metadata_recorded", semantic["metadata"]),
    ]
    payloads.extend(
        ("completion_recorded", record) for record in semantic["records"]
    )
    payloads.append(
        (
            "run_finished",
            {
                "record_count": len(semantic["records"]),
                "stop_reason": semantic["stop_reason"],
                "semantic_sha256": sha256_json(semantic),
            },
        )
    )
    previous_hash = GENESIS_EVENT_HASH
    events: list[dict[str, Any]] = []
    for sequence, (event_type, payload) in enumerate(payloads, start=1):
        core = {
            "sequence": sequence,
            "event_type": event_type,
            "payload": payload,
            "previous_event_sha256": previous_hash,
            "payload_sha256": sha256_json(payload),
        }
        event_hash = sha256_json(core)
        events.append({**core, "event_sha256": event_hash})
        previous_hash = event_hash
    return {
        "schema_version": schema_version,
        "run_id": transcript["run_id"],
        "events": events,
        "terminal_event_sha256": previous_hash,
    }


def verify_ledger(transcript: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    expected = build_ledger(transcript)
    if ledger != expected:
        return ["ledger_semantic_mismatch"]
    return []


def derive_outcome(transcript: dict[str, Any]) -> dict[str, Any]:
    run_kind = transcript.get("run_kind")
    if run_kind not in {"offline_fake", "live_probe"}:
        raise ValueError("unsupported_transcript_run_kind")
    if expected_run_kind_for_id(transcript["run_id"]) != run_kind:
        raise ValueError("run_id_kind_mismatch")
    schema_version = schema_version_for_run_kind(run_kind)
    records = transcript.get("records", [])
    profile_rows = [
        {
            "profile": record.get("profile"),
            "started": record.get("started"),
            "outcome_category": record.get("outcome_category"),
            "expected_fields_satisfied": record.get("expected_fields_satisfied"),
        }
        for record in records
    ]
    all_matched = bool(profile_rows) and all(
        row["started"] and row["outcome_category"] == "content_match"
        for row in profile_rows
    )
    if run_kind == "live_probe":
        metadata_status = transcript.get("metadata", {}).get("status")
        categories = {row["outcome_category"] for row in profile_rows}
        infrastructure_failure = bool(
            categories
            & {
                "transport_error",
                "timeout",
                "http_error",
                "connection_error",
                "redirect",
                "deadline_exhausted",
                "model_mismatch",
                "identity_missing",
                "unsafe_provider_identity",
                "malformed_envelope",
            }
        )
        if metadata_status == "accepted" and not infrastructure_failure:
            conclusion = "observed"
        elif metadata_status == "metadata_target_absent":
            conclusion = "not_observed"
        else:
            conclusion = "rework"
        return {
            "schema_version": schema_version,
            "run_id": transcript["run_id"],
            "run_kind": "live_probe",
            "live_conclusion": conclusion,
            "metadata_status": metadata_status,
            "record_count": len(records),
            "metadata_call_count": transcript.get("metadata_call_count"),
            "completion_call_count": transcript.get("completion_call_count"),
            "profile_outcomes": profile_rows,
            "transcript_semantic_sha256": sha256_json(transcript_semantic_view(transcript)),
        }
    return {
        "schema_version": schema_version,
        "run_id": transcript["run_id"],
        "run_kind": transcript.get("run_kind"),
        "offline_status": "offline_fake_passed" if all_matched else "offline_fake_exercised",
        "metadata_status": transcript.get("metadata", {}).get("status"),
        "record_count": len(records),
        "profile_outcomes": profile_rows,
        "transcript_semantic_sha256": sha256_json(transcript_semantic_view(transcript)),
    }


def derive_report(transcript: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    run_kind = transcript.get("run_kind")
    if run_kind not in {"offline_fake", "live_probe"} or outcome.get("run_kind") != run_kind:
        raise ValueError("report_run_kind_mismatch")
    schema_version = schema_version_for_run_kind(run_kind)
    if run_kind == "live_probe":
        return {
            "schema_version": schema_version,
            "run_id": transcript["run_id"],
            "report_kind": "live_probe_safe_summary",
            "offline_only": False,
            "scope": "one_constrained_q2c_live_probe",
            "live_conclusion": outcome.get("live_conclusion"),
            "metadata_status": outcome.get("metadata_status"),
            "record_count": outcome.get("record_count"),
            "metadata_call_count": outcome.get("metadata_call_count"),
            "completion_call_count": outcome.get("completion_call_count"),
            "profile_outcomes": outcome.get("profile_outcomes"),
            "replay_limit": "safe_summary_only",
            "coordinated_tamper_limit": "discarded_raw_content_prevents_external_authenticity_proof",
        }
    return {
        "schema_version": schema_version,
        "run_id": transcript["run_id"],
        "report_kind": "offline_fake_transport_report",
        "offline_only": True,
        "metadata_status": outcome.get("metadata_status"),
        "record_count": outcome.get("record_count"),
        "profile_outcomes": outcome.get("profile_outcomes"),
        "replay_limit": "safe_summary_only",
    }


def delivery_semantic_view(run_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
    """Return the small, reconstructible claim set allowed in DELIVERY.md."""
    if not isinstance(outcome, dict):
        raise ValueError("delivery_outcome_shape")
    run_kind = outcome.get("run_kind")
    if run_kind not in {"offline_fake", "live_probe"} or expected_run_kind_for_id(run_id) != run_kind:
        raise ValueError("delivery_run_kind_mismatch")
    if run_kind == "live_probe":
        conclusion = outcome.get("live_conclusion")
        metadata_calls = outcome.get("metadata_call_count")
        completion_calls = outcome.get("completion_call_count")
        if conclusion not in {"observed", "not_observed", "rework"}:
            raise ValueError("delivery_live_conclusion")
        if (
            type(metadata_calls) is not int
            or metadata_calls not in {0, 1}
            or type(completion_calls) is not int
            or not 0 <= completion_calls <= 4
        ):
            raise ValueError("delivery_live_counts")
        return {
            "run_kind": "live_probe",
            "run_id": run_id,
            "scope": LIVE_DELIVERY_SCOPE,
            "live_conclusion": conclusion,
            "metadata_call_count": metadata_calls,
            "completion_call_count": completion_calls,
            "evidence_boundary": DELIVERY_EVIDENCE_BOUNDARY,
            "integrity_boundary": LIVE_DELIVERY_INTEGRITY_BOUNDARY,
        }
    metadata_status = outcome.get("metadata_status")
    record_count = outcome.get("record_count")
    offline_status = outcome.get("offline_status")
    if not isinstance(metadata_status, str) or not metadata_status:
        raise ValueError("delivery_offline_metadata")
    if not isinstance(record_count, int) or record_count < 0:
        raise ValueError("delivery_offline_count")
    if offline_status not in {"offline_fake_passed", "offline_fake_exercised"}:
        raise ValueError("delivery_offline_status")
    return {
        "run_kind": "offline_fake",
        "run_id": run_id,
        "scope": OFFLINE_DELIVERY_SCOPE,
        "metadata_status": metadata_status,
        "record_count": record_count,
        "offline_status": offline_status,
        "evidence_boundary": DELIVERY_EVIDENCE_BOUNDARY,
    }


def local_delivery_text(run_id: str, outcome: dict[str, Any]) -> str:
    """Render the canonical, deliberately small local delivery summary."""
    fields = delivery_semantic_view(run_id, outcome)
    if fields["run_kind"] == "live_probe":
        return "\n".join(
            (
                LIVE_DELIVERY_TITLE,
                "",
                f"- Run ID: `{fields['run_id']}`",
                f"- Scope: {fields['scope']}",
                f"- Conclusion: `{fields['live_conclusion']}`",
                f"- Metadata calls: `{fields['metadata_call_count']}`",
                f"- Completion calls: `{fields['completion_call_count']}`",
                f"- Evidence boundary: {fields['evidence_boundary']}",
                f"- Integrity boundary: {fields['integrity_boundary']}",
                "",
            )
        )
    return "\n".join(
        (
            OFFLINE_DELIVERY_TITLE,
            "",
            f"- Run ID: `{fields['run_id']}`",
            f"- Scope: {fields['scope']}",
            f"- Metadata status: `{fields['metadata_status']}`",
            f"- Completion records: `{fields['record_count']}`",
            f"- Offline status: `{fields['offline_status']}`",
            f"- Evidence boundary: {fields['evidence_boundary']}",
            "",
        )
    )


def parse_local_delivery(text: str, run_kind: str) -> dict[str, Any] | None:
    """Parse only the canonical DELIVERY grammar; extra prose fails closed."""
    if not isinstance(text, str) or "\r" in text:
        return None
    if run_kind == "live_probe":
        pattern = re.compile(
            r"\A"
            + re.escape(LIVE_DELIVERY_TITLE)
            + r"\n\n- Run ID: `(?P<run_id>P1V2Q2C-PROBE-[A-Z0-9-]+)`"
            + r"\n- Scope: " + re.escape(LIVE_DELIVERY_SCOPE)
            + r"\n- Conclusion: `(?P<live_conclusion>observed|not_observed|rework)`"
            + r"\n- Metadata calls: `(?P<metadata_call_count>[0-9]+)`"
            + r"\n- Completion calls: `(?P<completion_call_count>[0-9]+)`"
            + r"\n- Evidence boundary: " + re.escape(DELIVERY_EVIDENCE_BOUNDARY)
            + r"\n- Integrity boundary: " + re.escape(LIVE_DELIVERY_INTEGRITY_BOUNDARY)
            + r"\n\Z"
        )
        matched = pattern.fullmatch(text)
        if matched is None:
            return None
        return {
            "run_kind": "live_probe",
            "run_id": matched["run_id"],
            "scope": LIVE_DELIVERY_SCOPE,
            "live_conclusion": matched["live_conclusion"],
            "metadata_call_count": int(matched["metadata_call_count"]),
            "completion_call_count": int(matched["completion_call_count"]),
            "evidence_boundary": DELIVERY_EVIDENCE_BOUNDARY,
            "integrity_boundary": LIVE_DELIVERY_INTEGRITY_BOUNDARY,
        }
    if run_kind == "offline_fake":
        pattern = re.compile(
            r"\A"
            + re.escape(OFFLINE_DELIVERY_TITLE)
            + r"\n\n- Run ID: `(?P<run_id>P1V2Q2C-FAKE-[A-Z0-9-]+)`"
            + r"\n- Scope: " + re.escape(OFFLINE_DELIVERY_SCOPE)
            + r"\n- Metadata status: `(?P<metadata_status>[a-z_]+)`"
            + r"\n- Completion records: `(?P<record_count>[0-9]+)`"
            + r"\n- Offline status: `(?P<offline_status>offline_fake_passed|offline_fake_exercised)`"
            + r"\n- Evidence boundary: " + re.escape(DELIVERY_EVIDENCE_BOUNDARY)
            + r"\n\Z"
        )
        matched = pattern.fullmatch(text)
        if matched is None:
            return None
        return {
            "run_kind": "offline_fake",
            "run_id": matched["run_id"],
            "scope": OFFLINE_DELIVERY_SCOPE,
            "metadata_status": matched["metadata_status"],
            "record_count": int(matched["record_count"]),
            "offline_status": matched["offline_status"],
            "evidence_boundary": DELIVERY_EVIDENCE_BOUNDARY,
        }
    return None


def validate_local_delivery_text(text: str, run_id: str, outcome: dict[str, Any]) -> list[str]:
    """Check kind-specific claims and the canonical field-level reconstruction."""
    try:
        expected = delivery_semantic_view(run_id, outcome)
        expected_text = local_delivery_text(run_id, outcome)
    except (KeyError, TypeError, ValueError):
        return ["delivery_expected"]
    run_kind = expected["run_kind"]
    lowered = text.lower() if isinstance(text, str) else ""
    forbidden = (
        ("offline", "no network", "offline_only", "offline-only")
        if run_kind == "live_probe"
        else ("live probe", "live_only", "live-only")
    )
    errors: list[str] = []
    if any(fragment in lowered for fragment in forbidden):
        errors.append("delivery_forbidden_kind_claim")
    parsed = parse_local_delivery(text, run_kind)
    if parsed is None:
        errors.append("delivery_parse")
    elif parsed != expected:
        errors.append("delivery_fields")
    if text != expected_text:
        errors.append("delivery_canonical")
    return errors


def artifact_manifest(
    run_dir: Path,
    run_id: str,
    run_kind: str,
    extra_core: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if expected_run_kind_for_id(run_id) != run_kind:
        raise ValueError("manifest_run_kind_mismatch")
    artifacts: dict[str, dict[str, Any]] = {}
    for relative_name in required_artifacts_for_kind(run_kind):
        candidate = run_dir / relative_name
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError("missing_or_unsafe_artifact")
        digest, size = canonical_file_hash(candidate)
        artifacts[relative_name] = {"sha256": digest, "size_bytes": size}
    core: dict[str, Any] = {
        "schema_version": schema_version_for_run_kind(run_kind),
        "run_id": run_id,
        "run_kind": run_kind,
        "profile_commitment": profile_commitment(run_kind),
        "artifacts": artifacts,
    }
    if extra_core:
        if set(extra_core) & set(core):
            raise ValueError("manifest_extra_core_collision")
        core.update(extra_core)
    return {**core, "manifest_core_sha256": sha256_json(core)}


def verify_manifest(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest_not_object"]
    core = {key: value for key, value in manifest.items() if key != "manifest_core_sha256"}
    if manifest.get("manifest_core_sha256") != sha256_json(core):
        errors.append("manifest_core_hash")
    try:
        required_artifacts = required_artifacts_for_kind(manifest.get("run_kind"))
    except ValueError:
        return errors + ["manifest_run_kind"]
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(required_artifacts):
        return errors + ["manifest_artifact_set"]
    for relative_name in required_artifacts:
        if Path(relative_name).is_absolute() or ".." in Path(relative_name).parts:
            errors.append("manifest_artifact_path")
            continue
        candidate = run_dir / relative_name
        try:
            if candidate.resolve().parent != (run_dir / relative_name).resolve().parent:
                errors.append("manifest_artifact_escape")
                continue
        except OSError:
            errors.append("manifest_artifact_resolve")
            continue
        if not candidate.is_file() or candidate.is_symlink():
            errors.append("manifest_artifact_missing")
            continue
        digest, size = canonical_file_hash(candidate)
        entry = artifacts.get(relative_name)
        if not isinstance(entry, dict) or entry.get("sha256") != digest or entry.get("size_bytes") != size:
            errors.append("manifest_artifact_hash")
    return errors


def scan_artifacts_for_forbidden_text(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            errors.append("artifact_symlink_or_missing")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append("artifact_not_utf8")
            continue
        if not _artifact_text_is_safe(text):
            errors.append("artifact_sensitive_text")
    return errors
