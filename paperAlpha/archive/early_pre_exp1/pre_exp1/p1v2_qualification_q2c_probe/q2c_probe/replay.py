"""Independent offline replay of the Q2-C safe-summary evidence chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    CARDS,
    TARGET_MODEL,
    profile_commitment,
    public_profile_snapshot,
    schema_version_for_run_kind,
    sha256_json,
    sha256_text,
)
from .evidence import (
    DELIVERY_EVIDENCE_BOUNDARY,
    LIVE_DELIVERY_INTEGRITY_BOUNDARY,
    LIVE_DELIVERY_SCOPE,
    LIVE_DELIVERY_TITLE,
    OFFLINE_DELIVERY_SCOPE,
    OFFLINE_DELIVERY_TITLE,
    canonical_file_hash,
    parse_local_delivery,
    expected_run_kind_for_id,
    read_json,
    required_artifacts_for_kind,
    run_directory,
    scan_artifacts_for_forbidden_text,
)
from .fake_scenarios import FAKE_MODELS
from .historical_integrity import historical_tree_snapshot
from .live_runner import (
    LIVE_GUARD_NAME,
    live_config_snapshot,
    live_execution_source_hashes,
    live_policy_commitment,
    live_runner_source_sha256,
    live_source_bundle_sha256,
    live_transport_source_sha256,
)


REPLAY_GENESIS_HASH = "0" * 64


def _expected_fake_semantic() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for sequence, card in enumerate(CARDS, start=1):
        if card.profile == "plain_text":
            content = card.expected_answer()
        else:
            content = (
                "{\"answer\":\"" + card.expected_answer() + "\",\"probe_id\":\""
                + card.expected_probe_id() + "\"}"
            )
        records.append(
            {
                "sequence": sequence,
                "card_id": card.card_id,
                "profile": card.profile,
                "request_sha256": card.request_sha256,
                "started": True,
                "outcome_category": "content_match",
                "expected_fields_satisfied": True,
                "content_length": len(content),
                "content_sha256": sha256_text(content),
                "provider_identity": {
                    "declared_model": TARGET_MODEL,
                    "request_id": f"fake-request-{sequence}",
                    "system_fingerprint": "fake-fingerprint-v1",
                    "provider_version": "fake-provider-v1",
                },
                "stop_reason": None,
            }
        )
    semantic = {
        "schema_version": schema_version_for_run_kind("offline_fake"),
        "run_kind": "offline_fake",
        "profile_commitment": profile_commitment("offline_fake"),
        "metadata": {
            "status": "accepted",
            "target_model_present": True,
            "model_count": len(FAKE_MODELS),
            "model_list_sha256": sha256_json(sorted(FAKE_MODELS)),
            "stop_reason": None,
        },
        "records": records,
        "stop_reason": None,
    }
    return semantic


def _replay_semantic(transcript: Any) -> dict[str, Any] | None:
    if not isinstance(transcript, dict) or not isinstance(transcript.get("metadata"), dict):
        return None
    raw_records = transcript.get("records")
    if not isinstance(raw_records, list):
        return None
    records: list[dict[str, Any]] = []
    for record in raw_records:
        if not isinstance(record, dict):
            return None
        records.append(
            {
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
        )
    metadata = transcript["metadata"]
    semantic = {
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
        "records": records,
        "stop_reason": transcript.get("stop_reason"),
    }
    if transcript.get("run_kind") == "live_probe":
        semantic["live_execution"] = {
            "live_policy_commitment": transcript.get("live_policy_commitment"),
            "live_runner_source_sha256": transcript.get("live_runner_source_sha256"),
            "live_transport_source_sha256": transcript.get("live_transport_source_sha256"),
            "metadata_call_count": transcript.get("metadata_call_count"),
            "completion_call_count": transcript.get("completion_call_count"),
        }
    return semantic


def _rebuild_ledger(transcript: dict[str, Any]) -> dict[str, Any] | None:
    semantic = _replay_semantic(transcript)
    if semantic is None:
        return None
    try:
        run_kind = transcript.get("run_kind")
        schema_version = schema_version_for_run_kind(run_kind)
    except ValueError:
        return None
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
    for record in semantic["records"]:
        payloads.append(("completion_recorded", record))
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
    prior = REPLAY_GENESIS_HASH
    events: list[dict[str, Any]] = []
    for sequence, (event_type, payload) in enumerate(payloads, start=1):
        core = {
            "sequence": sequence,
            "event_type": event_type,
            "payload": payload,
            "previous_event_sha256": prior,
            "payload_sha256": sha256_json(payload),
        }
        event_hash = sha256_json(core)
        events.append({**core, "event_sha256": event_hash})
        prior = event_hash
    return {
        "schema_version": schema_version,
        "run_id": transcript.get("run_id"),
        "events": events,
        "terminal_event_sha256": prior,
    }


def _replay_outcome(transcript: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "profile": record["profile"],
            "started": record["started"],
            "outcome_category": record["outcome_category"],
            "expected_fields_satisfied": record["expected_fields_satisfied"],
        }
        for record in semantic["records"]
    ]
    all_matched = bool(rows) and all(
        row["started"] and row["outcome_category"] == "content_match" for row in rows
    )
    if transcript.get("run_kind") == "live_probe":
        metadata_status = semantic["metadata"]["status"]
        categories = {row["outcome_category"] for row in rows}
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
            "schema_version": schema_version_for_run_kind("live_probe"),
            "run_id": transcript["run_id"],
            "run_kind": "live_probe",
            "live_conclusion": conclusion,
            "metadata_status": metadata_status,
            "record_count": len(rows),
            "metadata_call_count": transcript.get("metadata_call_count"),
            "completion_call_count": transcript.get("completion_call_count"),
            "profile_outcomes": rows,
            "transcript_semantic_sha256": sha256_json(semantic),
        }
    return {
        "schema_version": schema_version_for_run_kind("offline_fake"),
        "run_id": transcript["run_id"],
        "run_kind": transcript.get("run_kind"),
        "offline_status": "offline_fake_passed" if all_matched else "offline_fake_exercised",
        "metadata_status": semantic["metadata"]["status"],
        "record_count": len(rows),
        "profile_outcomes": rows,
        "transcript_semantic_sha256": sha256_json(semantic),
    }


def _replay_report(transcript: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    if transcript.get("run_kind") == "live_probe":
        return {
            "schema_version": schema_version_for_run_kind("live_probe"),
            "run_id": transcript["run_id"],
            "report_kind": "live_probe_safe_summary",
            "offline_only": False,
            "scope": "one_constrained_q2c_live_probe",
            "live_conclusion": outcome["live_conclusion"],
            "metadata_status": outcome["metadata_status"],
            "record_count": outcome["record_count"],
            "metadata_call_count": outcome["metadata_call_count"],
            "completion_call_count": outcome["completion_call_count"],
            "profile_outcomes": outcome["profile_outcomes"],
            "replay_limit": "safe_summary_only",
            "coordinated_tamper_limit": "discarded_raw_content_prevents_external_authenticity_proof",
        }
    return {
        "schema_version": schema_version_for_run_kind("offline_fake"),
        "run_id": transcript["run_id"],
        "report_kind": "offline_fake_transport_report",
        "offline_only": True,
        "metadata_status": outcome["metadata_status"],
        "record_count": outcome["record_count"],
        "profile_outcomes": outcome["profile_outcomes"],
        "replay_limit": "safe_summary_only",
    }


def _replay_delivery_fields(
    run_id: str, run_kind: str, outcome: dict[str, Any]
) -> dict[str, Any] | None:
    """Independently rebuild the only claims a delivery document may make."""
    try:
        if expected_run_kind_for_id(run_id) != run_kind or outcome.get("run_kind") != run_kind:
            return None
    except ValueError:
        return None
    if run_kind == "live_probe":
        conclusion = outcome.get("live_conclusion")
        metadata_calls = outcome.get("metadata_call_count")
        completion_calls = outcome.get("completion_call_count")
        if conclusion not in {"observed", "not_observed", "rework"}:
            return None
        if (
            type(metadata_calls) is not int
            or metadata_calls not in {0, 1}
            or type(completion_calls) is not int
            or not 0 <= completion_calls <= 4
        ):
            return None
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
    if run_kind == "offline_fake":
        metadata_status = outcome.get("metadata_status")
        record_count = outcome.get("record_count")
        offline_status = outcome.get("offline_status")
        if not isinstance(metadata_status, str) or not metadata_status:
            return None
        if not isinstance(record_count, int) or record_count < 0:
            return None
        if offline_status not in {"offline_fake_passed", "offline_fake_exercised"}:
            return None
        return {
            "run_kind": "offline_fake",
            "run_id": run_id,
            "scope": OFFLINE_DELIVERY_SCOPE,
            "metadata_status": metadata_status,
            "record_count": record_count,
            "offline_status": offline_status,
            "evidence_boundary": DELIVERY_EVIDENCE_BOUNDARY,
        }
    return None


def _replay_delivery_text(fields: dict[str, Any]) -> str:
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


def _validate_delivery_independently(
    text: str, run_id: str, run_kind: str, rebuilt_outcome: dict[str, Any]
) -> list[str]:
    fields = _replay_delivery_fields(run_id, run_kind, rebuilt_outcome)
    if fields is None:
        return ["replay_delivery_expected"]
    lowered = text.lower() if isinstance(text, str) else ""
    forbidden = (
        ("offline", "no network", "offline_only", "offline-only")
        if run_kind == "live_probe"
        else ("live probe", "live_only", "live-only")
    )
    errors: list[str] = []
    if any(fragment in lowered for fragment in forbidden):
        errors.append("replay_delivery_forbidden_kind_claim")
    parsed = parse_local_delivery(text, run_kind)
    if parsed is None:
        errors.append("replay_delivery_parse")
    elif parsed != fields:
        errors.append("replay_delivery_fields")
    if text != _replay_delivery_text(fields):
        errors.append("replay_delivery_canonical")
    return errors


def _verify_manifest_independently(run_dir: Path, manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return ["replay_manifest_not_object"]
    core = {key: value for key, value in manifest.items() if key != "manifest_core_sha256"}
    errors: list[str] = []
    if manifest.get("manifest_core_sha256") != sha256_json(core):
        errors.append("replay_manifest_core")
    try:
        required_artifacts = required_artifacts_for_kind(manifest.get("run_kind"))
    except ValueError:
        return errors + ["replay_manifest_run_kind"]
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict) or set(entries) != set(required_artifacts):
        return errors + ["replay_manifest_set"]
    for relative_name in required_artifacts:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append("replay_manifest_path")
            continue
        candidate = run_dir / relative
        if not candidate.is_file() or candidate.is_symlink():
            errors.append("replay_manifest_artifact")
            continue
        actual_digest, actual_size = canonical_file_hash(candidate)
        entry = entries[relative_name]
        if not isinstance(entry, dict) or entry.get("sha256") != actual_digest or entry.get("size_bytes") != actual_size:
            errors.append("replay_manifest_hash")
    return errors


def _replay_live_protocol(transcript: dict[str, Any], semantic: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if semantic.get("profile_commitment") != profile_commitment("live_probe"):
        errors.append("replay_live_profile")
    records = semantic.get("records")
    if not isinstance(records, list) or len(records) != len(CARDS):
        return errors + ["replay_live_records"]
    for sequence, (card, record) in enumerate(zip(CARDS, records), start=1):
        if (
            record.get("sequence") != sequence
            or record.get("card_id") != card.card_id
            or record.get("profile") != card.profile
            or record.get("request_sha256") != card.request_sha256
        ):
            errors.append("replay_live_frozen_card")
    live = semantic.get("live_execution")
    if not isinstance(live, dict):
        return errors + ["replay_live_execution"]
    if live.get("live_policy_commitment") != live_policy_commitment():
        errors.append("replay_live_policy")
    if live.get("live_runner_source_sha256") != live_runner_source_sha256():
        errors.append("replay_live_runner_source")
    if live.get("live_transport_source_sha256") != live_transport_source_sha256():
        errors.append("replay_live_transport_source")
    if type(live.get("metadata_call_count")) is not int or live.get("metadata_call_count") not in {0, 1}:
        errors.append("replay_live_metadata_count")
    if type(live.get("completion_call_count")) is not int or not 0 <= live["completion_call_count"] <= 4:
        errors.append("replay_live_completion_count")
    if live.get("metadata_call_count", -1) + live.get("completion_call_count", -1) != transcript.get("transport_call_count"):
        errors.append("replay_live_transport_counts")
    return errors


def replay_run(data_root: Path, run_id: str) -> list[str]:
    """Reconstruct the safe evidence semantics independently of validator.py."""
    try:
        run_dir = run_directory(data_root, run_id)
    except (OSError, ValueError):
        return ["replay_run_path"]
    if not run_dir.is_dir() or run_dir.is_symlink():
        return ["replay_run_directory"]
    try:
        manifest = read_json(run_dir / "run_manifest.json")
        config = read_json(run_dir / "config_snapshot.json")
        transcript = read_json(run_dir / "transcript.json")
        ledger = read_json(run_dir / "ledger.json")
        outcome = read_json(run_dir / "outcome.json")
        report = read_json(run_dir / "report.json")
        history = read_json(run_dir / "historical_integrity.json")
    except (OSError, ValueError):
        return ["replay_artifact_parse"]
    run_kind = transcript.get("run_kind") if isinstance(transcript, dict) else None
    if run_kind not in {"offline_fake", "live_probe"}:
        return ["replay_transcript_run_kind"]
    try:
        if expected_run_kind_for_id(run_id) != run_kind:
            return ["replay_run_id_kind"]
    except ValueError:
        return ["replay_run_id_kind"]
    errors = _verify_manifest_independently(run_dir, manifest)
    errors.extend(
        scan_artifacts_for_forbidden_text(
            run_dir / name for name in required_artifacts_for_kind(run_kind)
        )
    )
    if manifest.get("run_id") != run_id or manifest.get("run_kind") != run_kind:
        errors.append("replay_manifest_identity")
    if manifest.get("schema_version") != schema_version_for_run_kind(run_kind):
        errors.append("replay_manifest_schema")
    if manifest.get("profile_commitment") != profile_commitment(run_kind):
        errors.append("replay_profile_commitment")
    expected_config = (
        live_config_snapshot()
        if run_kind == "live_probe"
        else {
            "schema_version": schema_version_for_run_kind("offline_fake"),
            "offline_only": True,
            "profile_commitment": profile_commitment("offline_fake"),
            "public_profile": public_profile_snapshot("offline_fake"),
        }
    )
    if config != expected_config:
        errors.append("replay_config")
    semantic = _replay_semantic(transcript)
    if semantic is None:
        return sorted(set(errors + ["replay_transcript_shape"]))
    if run_kind == "offline_fake" and semantic != _expected_fake_semantic():
        errors.append("replay_fake_oracle")
    if run_kind == "live_probe":
        errors.extend(_replay_live_protocol(transcript, semantic))
    rebuilt_ledger = _rebuild_ledger(transcript)
    if ledger != rebuilt_ledger:
        errors.append("replay_ledger")
    rebuilt_outcome: dict[str, Any] | None = None
    try:
        rebuilt_outcome = _replay_outcome(transcript, semantic)
        rebuilt_report = _replay_report(transcript, rebuilt_outcome)
    except (KeyError, TypeError, ValueError):
        errors.append("replay_run_kind_artifact_semantics")
    else:
        if outcome != rebuilt_outcome:
            errors.append("replay_outcome")
        if report != rebuilt_report:
            errors.append("replay_report")
    expected_report_kind = "offline_fake_transport_report" if run_kind == "offline_fake" else "live_probe_safe_summary"
    expected_offline_only = run_kind == "offline_fake"
    if (
        not isinstance(outcome, dict)
        or outcome.get("run_kind") != run_kind
        or not isinstance(report, dict)
        or report.get("report_kind") != expected_report_kind
        or report.get("offline_only") is not expected_offline_only
    ):
        errors.append("replay_run_kind_artifact_identity")
    if not isinstance(history, dict) or history.get("unchanged") is not True or history.get("before") != history.get("after"):
        errors.append("replay_historical_record")
    else:
        try:
            if historical_tree_snapshot() != history["before"]:
                errors.append("replay_historical_current")
        except (OSError, ValueError):
            errors.append("replay_historical_read")
    if run_kind == "live_probe":
        expected_manifest = {
            "live_policy_commitment": live_policy_commitment(),
            "live_runner_source_sha256": live_runner_source_sha256(),
            "live_transport_source_sha256": live_transport_source_sha256(),
            "live_source_bundle_sha256": live_source_bundle_sha256(),
            "live_execution_source_hashes": live_execution_source_hashes(),
            "metadata_call_count": transcript.get("metadata_call_count"),
            "completion_call_count": transcript.get("completion_call_count"),
        }
        for key, value in expected_manifest.items():
            if manifest.get(key) != value:
                errors.append("replay_live_manifest_binding")
        try:
            local_guard = read_json(run_dir / LIVE_GUARD_NAME)
            root_guard = read_json(data_root / LIVE_GUARD_NAME)
        except (OSError, ValueError):
            errors.append("replay_live_guard_read")
        else:
            expected_guard = {
                "schema_version": schema_version_for_run_kind("live_probe"),
                "run_id": run_id,
                "state": "reserved",
                "live_policy_commitment": live_policy_commitment(),
            }
            if local_guard != expected_guard or root_guard != expected_guard:
                errors.append("replay_live_guard")
    try:
        delivery = (run_dir / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
    except OSError:
        errors.append("replay_delivery_read")
    else:
        if rebuilt_outcome is None:
            errors.append("replay_delivery_expected")
        else:
            errors.extend(
                _validate_delivery_independently(
                    delivery, run_id, run_kind, rebuilt_outcome
                )
            )
    return sorted(set(errors))
