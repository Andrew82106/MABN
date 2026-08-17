"""Deterministic replay of the safe Q2-D ledger."""

from __future__ import annotations

from typing import Any

from .artifacts import artifact_paths, read_json, safe_write_json, source_provenance
from .contract import CARDS, CONTRACT_VERSION, PROFILE, _PUBLISHER_CASES


def replay_run(run_id: str, *, write_output: bool = True) -> dict[str, Any]:
    paths = artifact_paths(run_id)
    result: dict[str, Any] = {"schema_version": "1.0.0", "run_id": run_id, "phase": "Q2D_OFFLINE", "replay_method": "ledger_contract_replay", "matched_cards": 0, "errors": [], "passed": False, "tool_execution_calls": 0, "network_calls": 0, "env_reads": 0}
    try:
        manifest = read_json(paths["manifest"])
        ledger = read_json(paths["ledger"])
        records = ledger.get("records")
        expected_ids = [card["card_id"] for card in CARDS]
        if manifest.get("run_id") != run_id or manifest.get("contract_version") != CONTRACT_VERSION or manifest.get("profile") != PROFILE:
            result["errors"].append("contract or profile drift")
        if manifest.get("source_provenance") != source_provenance():
            result["errors"].append("source provenance drift")
        if manifest.get("publisher_case_coverage") != _PUBLISHER_CASES:
            result["errors"].append("publisher case coverage drift")
        if not isinstance(records, list) or len(records) != 18:
            result["errors"].append("ledger card count drift")
        else:
            for card, row in zip(CARDS, records):
                if not isinstance(row, dict) or row.get("card_id") != card["card_id"] or row.get("phase") != card["phase"]:
                    result["errors"].append("ledger identity drift")
                    continue
                if row.get("status") != "pass" or row.get("result_category") != "accepted" or row.get("action_hash") != card["expected_action_hash"] or row.get("expected_action_hash") != card["expected_action_hash"]:
                    result["errors"].append("action contract mismatch")
                    continue
                result["matched_cards"] += 1
        result["passed"] = not result["errors"] and result["matched_cards"] == 18 and ledger.get("tool_execution_calls") == 0 and ledger.get("network_calls") == 0 and ledger.get("env_reads") == 0
    except Exception:
        result["errors"].append("replay artifact structure is malformed")
        result["passed"] = False
    if write_output:
        safe_write_json(paths["replay"], result)
    return result


__all__ = ["replay_run"]
