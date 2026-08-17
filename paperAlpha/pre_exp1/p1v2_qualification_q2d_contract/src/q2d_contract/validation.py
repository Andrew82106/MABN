"""Fail-closed validator for Q2-D offline fake artifacts."""

from __future__ import annotations

import json
from typing import Any

from .artifacts import FORBIDDEN_PERSISTED_TOKENS, artifact_paths, read_json, safe_write_json, sha256_file, source_provenance
from .contract import CARDS, CARD_BY_ID, CONTRACT_VERSION, PROFILE, _PUBLISHER_CASES


def _base(run_id: str) -> dict[str, Any]:
    return {"schema_version": "1.0.0", "run_id": run_id, "phase": "Q2D_OFFLINE", "checks": {}, "errors": [], "passed": False}


def _put(result: dict[str, Any], key: str, value: bool, error: str | None = None) -> None:
    result["checks"][key] = bool(value)
    if not value and error:
        result["errors"].append(error)


def _safe_file(path) -> bool:
    try:
        text = path.read_text(encoding="utf-8").lower()
    except Exception:
        return False
    return not any(token in text for token in FORBIDDEN_PERSISTED_TOKENS)


def validate_run(run_id: str, *, write_outputs: bool = True) -> dict[str, Any]:
    result = _base(run_id)
    paths = artifact_paths(run_id)
    try:
        manifest = read_json(paths["manifest"])
        ledger = read_json(paths["ledger"])
        replay = read_json(paths["replay"]) if paths["replay"].is_file() else None
        if not isinstance(manifest, dict) or not isinstance(ledger, dict):
            raise ValueError("artifact roots must be objects")
        _put(result, "manifest_identity", manifest.get("run_id") == run_id and manifest.get("contract_version") == CONTRACT_VERSION and manifest.get("artifact_kind") == "q2d_offline_fake_run" and manifest.get("execution_mode") == "offline_fake_transport", "manifest identity drift")
        _put(result, "profile_frozen", manifest.get("profile") == PROFILE, "strict profile drift")
        expected_ids = [card["card_id"] for card in CARDS]
        _put(result, "card_coverage", manifest.get("card_ids") == expected_ids and manifest.get("card_count") == 18 and manifest.get("screen_card_count") == 9 and manifest.get("confirmation_card_count") == 9, "18-card coverage drift")
        _put(result, "schema_contracts", manifest.get("schema_hashes") == {card["card_id"]: card["schema_hash"] for card in CARDS} and manifest.get("expected_action_hashes") == {card["card_id"]: card["expected_action_hash"] for card in CARDS}, "schema or expected-action hash drift")
        _put(result, "source_provenance", manifest.get("source_provenance") == source_provenance(), "source provenance drift")
        _put(result, "publisher_coverage", manifest.get("publisher_case_coverage") == _PUBLISHER_CASES, "publisher case coverage drift")
        records = ledger.get("records")
        _put(result, "ledger_shape", isinstance(records, list) and len(records) == 18, "ledger does not contain exactly 18 records")
        if isinstance(records, list):
            ids = [row.get("card_id") for row in records if isinstance(row, dict)]
            phases = [row.get("phase") for row in records if isinstance(row, dict)]
            _put(result, "ledger_order", ids == expected_ids and phases[:9] == ["screen"] * 9 and phases[9:] == ["confirmation"] * 9, "ledger card order or phase boundary drift")
            rows_ok = all(isinstance(row, dict) and row.get("status") == "pass" and row.get("result_category") == "accepted" and row.get("action_hash") == row.get("expected_action_hash") and row.get("provider_identity") == manifest.get("model_identity") and row.get("metadata_present") is True and row.get("sensitive_content_detected") is False for row in records)
            _put(result, "all_cards_pass", rows_ok, "a card result is not an accepted exact action")
        else:
            _put(result, "ledger_order", False, "ledger records malformed")
            _put(result, "all_cards_pass", False, "ledger records malformed")
        _put(result, "screen_gate", ledger.get("screen_passed") is True and ledger.get("confirmation_blocked") is False and manifest.get("screen_passed") is True and manifest.get("confirmation_blocked") is False, "screen gate or confirmation state drift")
        _put(result, "zero_side_effects", ledger.get("tool_execution_calls") == 0 and ledger.get("network_calls") == 0 and ledger.get("env_reads") == 0 and manifest.get("tool_execution_calls") == 0 and manifest.get("network_calls") == 0 and manifest.get("env_reads") == 0, "nonzero side-effect counter")
        _put(result, "replay_present", isinstance(replay, dict) and replay.get("passed") is True and replay.get("matched_cards") == 18, "replay is missing or failed")
        safe = all(_safe_file(paths[key]) for key in ("manifest", "ledger", "replay", "run_report") if paths[key].is_file())
        _put(result, "safe_persisted_artifacts", safe, "forbidden response or credential material in persisted artifact")
        recorded_hashes = manifest.get("artifact_hashes")
        hash_ok = isinstance(recorded_hashes, dict) and all(key in recorded_hashes and paths[key].is_file() and sha256_file(paths[key]) == recorded_hashes[key] for key in ("ledger", "replay", "run_report"))
        _put(result, "artifact_hashes", hash_ok, "artifact hash mismatch")
        result.update({"cards": 18, "screen_cards": 9, "confirmation_cards": 9, "tool_execution_calls": 0, "network_calls": 0, "env_reads": 0})
        result["passed"] = not result["errors"]
    except Exception:
        result["checks"]["untrusted_artifact_structure_safe"] = False
        result["errors"].append("artifact structure is malformed")
        result["passed"] = False
    if write_outputs:
        safe_write_json(paths["validation"], result)
    return result


__all__ = ["validate_run"]
