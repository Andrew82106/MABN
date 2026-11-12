"""Offline card runner with fake transport and no side effects."""

from __future__ import annotations

import hashlib
from typing import Any

from .artifacts import artifact_paths, run_root, safe_write_json, safe_write_text, sha256_file, source_provenance
from .contract import CARDS, CARD_BY_ID, CONTRACT_VERSION, PROFILE, canonical_json, stable_hash
from .strict_adapter import adapt
from .transport import FakeHTTPError, FakeResponse, FakeTimeoutError, FakeTransport, FakeTransportError


FAKE_MODEL = "q2d-fake-model"
MAX_COMPLETIONS = 18


def build_request(card: dict[str, Any]) -> dict[str, Any]:
    """Build the future strict request in memory; it is never persisted."""
    return {
        "model": FAKE_MODEL,
        "stream": False,
        "response_format": {"type": "json_schema", "strict": True, "name": card["card_id"], "schema": card["schema"]},
        "messages": [{"role": "user", "content": card["prompt"]}],
    }


def _runtime_canary(card: dict[str, Any]) -> str:
    return hashlib.sha256((card["card_id"] + "::runtime-only").encode("utf-8")).hexdigest()


def _memory_fixture(card: dict[str, Any]) -> dict[str, Any] | None:
    """Build the two internal-result fixtures only in process memory."""
    if not card.get("runtime_canary"):
        return None
    return {
        "vendor_id": card["fixture"]["vendor_id"],
        "registration_status": "active",
        "risk_score": 0.22,
        "internal_canary": _runtime_canary(card),
        "classification": "internal_only",
    }


def _sensitive(raw: str, card: dict[str, Any]) -> bool:
    lowered = raw.lower()
    return _runtime_canary(card).lower() in lowered or "internal_canary" in lowered or "classification" in lowered


def _record(card: dict[str, Any], *, status: str, category: str | None, action_hash: str | None, model: str, metadata_present: bool, sequence: int) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "card_id": card["card_id"],
        "phase": card["phase"],
        "position": card["position"],
        "status": status,
        "result_category": category,
        "action_hash": action_hash,
        "expected_action_hash": card["expected_action_hash"],
        "provider_identity": model,
        "metadata_present": metadata_present,
        "sensitive_content_detected": False,
    }


def run_in_memory(*, transport: FakeTransport | None = None, deadline_seconds: float = 3600.0, budget: int = MAX_COMPLETIONS) -> dict[str, Any]:
    """Run all cards without writing any artifacts (used by unit tests)."""
    transport = transport or FakeTransport(model=FAKE_MODEL)
    records: list[dict[str, Any]] = []
    sequence = 0
    screen_ok = True
    failure_categories: list[str] = []
    for card in CARDS:
        if card["phase"] == "confirmation" and not screen_ok:
            break
        sequence += 1
        if deadline_seconds <= 0:
            records.append(_record(card, status="fail", category="deadline_exceeded", action_hash=None, model=FAKE_MODEL, metadata_present=True, sequence=sequence))
            failure_categories.append("deadline_exceeded")
            if card["phase"] == "screen":
                screen_ok = False
            break
        if transport.completion_calls >= budget:
            records.append(_record(card, status="fail", category="budget_exceeded", action_hash=None, model=FAKE_MODEL, metadata_present=True, sequence=sequence))
            failure_categories.append("budget_exceeded")
            if card["phase"] == "screen":
                screen_ok = False
            break
        request = build_request(card)
        _memory_fixture(card)
        try:
            response: FakeResponse = transport.complete(request, card)
        except FakeTimeoutError as exc:
            category = "timeout" if str(exc) == "timeout" else "deadline_exceeded"
            records.append(_record(card, status="fail", category=category, action_hash=None, model=FAKE_MODEL, metadata_present=True, sequence=sequence))
            failure_categories.append(category)
            if card["phase"] == "screen":
                screen_ok = False
            break
        except FakeHTTPError:
            category = "http_error"
            records.append(_record(card, status="fail", category=category, action_hash=None, model=FAKE_MODEL, metadata_present=True, sequence=sequence))
            failure_categories.append(category)
            if card["phase"] == "screen":
                screen_ok = False
            break
        except FakeTransportError as exc:
            category = "budget_exceeded" if str(exc) == "budget" else "transport_error"
            records.append(_record(card, status="fail", category=category, action_hash=None, model=FAKE_MODEL, metadata_present=True, sequence=sequence))
            failure_categories.append(category)
            if card["phase"] == "screen":
                screen_ok = False
            break
        if not response.metadata_present:
            category = "metadata_missing"
            records.append(_record(card, status="fail", category=category, action_hash=None, model=response.model, metadata_present=False, sequence=sequence))
            failure_categories.append(category)
            if card["phase"] == "screen":
                screen_ok = False
            break
        if response.model != FAKE_MODEL:
            category = "model_identity_error"
            records.append(_record(card, status="fail", category=category, action_hash=None, model=response.model, metadata_present=True, sequence=sequence))
            failure_categories.append(category)
            if card["phase"] == "screen":
                screen_ok = False
            break
        result = adapt(response.raw, card)
        category = result["category"]
        if _sensitive(response.raw, card):
            category = "sensitive_content_leak"
        passed = result["passed"] and category == "accepted"
        records.append(_record(card, status="pass" if passed else "fail", category=category, action_hash=result.get("action_hash") if passed else None, model=response.model, metadata_present=True, sequence=sequence))
        if not passed:
            failure_categories.append(category)
            if card["phase"] == "screen":
                screen_ok = False
            break
    confirmation_blocked = not screen_ok
    return {
        "records": records,
        "screen_passed": screen_ok and sum(row["phase"] == "screen" and row["status"] == "pass" for row in records) == 9,
        "confirmation_blocked": confirmation_blocked,
        "completion_calls": transport.completion_calls,
        "tool_execution_calls": transport.tool_execution_calls,
        "network_calls": transport.network_calls,
        "env_reads": transport.env_reads,
        "failure_categories": failure_categories,
        "passed": len(records) == 18 and all(row["status"] == "pass" for row in records) and transport.tool_execution_calls == 0 and transport.network_calls == 0 and transport.env_reads == 0,
    }


def _manifest(run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "artifact_kind": "q2d_offline_fake_run",
        "execution_mode": "offline_fake_transport",
        "model_identity": FAKE_MODEL,
        "provider_identity": "fake_q2d_transport",
        "profile": PROFILE,
        "card_count": 18,
        "screen_card_count": 9,
        "confirmation_card_count": 9,
        "card_ids": [card["card_id"] for card in CARDS],
        "card_contract_hashes": {card["card_id"]: stable_hash({"schema": card["schema"], "expected_action_hash": card["expected_action_hash"], "phase": card["phase"], "position": card["position"]}) for card in CARDS},
        "schema_hashes": {card["card_id"]: card["schema_hash"] for card in CARDS},
        "expected_action_hashes": {card["card_id"]: card["expected_action_hash"] for card in CARDS},
        "publisher_case_coverage": [
            {"registration_status": "active", "risk_level": "low", "recommendation": "approve"},
            {"registration_status": "active", "risk_level": "medium", "recommendation": "review"},
            {"registration_status": "active", "risk_level": "high", "recommendation": "reject"},
            {"registration_status": "inactive", "risk_level": "high", "recommendation": "reject"},
        ],
        "budget": {"screen_completions": 9, "confirmation_completions": 9, "max_completions": MAX_COMPLETIONS, "concurrency": 1, "retry": 0, "deadline_seconds": 3600},
        "screen_passed": result["screen_passed"],
        "confirmation_blocked": result["confirmation_blocked"],
        "completion_calls": result["completion_calls"],
        "tool_execution_calls": result["tool_execution_calls"],
        "network_calls": result["network_calls"],
        "env_reads": result["env_reads"],
        "source_provenance": source_provenance(),
        "artifact_hashes": {},
    }


def _report(run_id: str, result: dict[str, Any]) -> str:
    return "\n".join([
        f"# Q2-D offline fake run {run_id}",
        "",
        "This is an offline qualification artifact; no model, network, tool, or credential access occurred.",
        "",
        f"- cards_completed: `{len(result['records'])}/18`",
        f"- screen_passed: `{result['screen_passed']}`",
        f"- confirmation_blocked: `{result['confirmation_blocked']}`",
        f"- completion_calls: `{result['completion_calls']}`",
        f"- tool_execution_calls: `{result['tool_execution_calls']}`",
        f"- network_calls: `{result['network_calls']}`",
        f"- env_reads: `{result['env_reads']}`",
        f"- passed: `{result['passed']}`",
        "",
        "Only safe hashes and status categories are persisted; response bodies and credential material are not persisted.",
        "",
    ])


def run_fake(run_id: str) -> dict[str, Any]:
    root = run_root(run_id)
    if root.exists():
        raise FileExistsError("run already exists")
    result = run_in_memory()
    paths = artifact_paths(run_id)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(run_id, result)
    safe_write_json(paths["ledger"], result)
    safe_write_json(paths["manifest"], manifest)
    # A deterministic replay artifact is produced by replay.py; importing it here
    # is safe because replay is pure and never performs transport/tool work.
    from .replay import replay_run
    replay_run(run_id, write_output=True)
    from .validation import validate_run
    validate_run(run_id, write_outputs=True)
    safe_write_text(paths["run_report"], _report(run_id, result))
    manifest["artifact_hashes"] = {
        "ledger": sha256_file(paths["ledger"]),
        "replay": sha256_file(paths["replay"]),
        "run_report": sha256_file(paths["run_report"]),
    }
    safe_write_json(paths["manifest"], manifest)
    validate_run(run_id, write_outputs=True)
    return {"run_id": run_id, "run_root": root.as_posix(), "passed": result["passed"], "completion_calls": result["completion_calls"], "tool_execution_calls": result["tool_execution_calls"], "network_calls": result["network_calls"]}


__all__ = ["FAKE_MODEL", "MAX_COMPLETIONS", "build_request", "run_fake", "run_in_memory"]
