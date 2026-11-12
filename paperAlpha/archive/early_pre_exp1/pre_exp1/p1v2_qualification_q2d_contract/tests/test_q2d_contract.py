from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from q2d_contract.artifacts import FORBIDDEN_PERSISTED_TOKENS, data_root
from q2d_contract.contract import CARDS, CARD_BY_ID, _PUBLISHER_CASES
from q2d_contract.replay import replay_run
from q2d_contract.runner import build_request, run_fake, run_in_memory
from q2d_contract.strict_adapter import adapt
from q2d_contract.transport import FakeTransport
from q2d_contract.validation import validate_run


def _raw(card: dict, action: dict | None = None) -> str:
    return json.dumps(action if action is not None else card["expected_action"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _action(card: dict) -> dict:
    return copy.deepcopy(card["expected_action"])


def _fresh_run(name: str) -> str:
    run_id = f"Q2D-TEST-{name}"
    root = data_root() / "runs" / run_id
    if root.exists():
        shutil.rmtree(root)
    run_fake(run_id)
    return run_id


def _clone_run(source_id: str, clone_id: str) -> Path:
    source = data_root() / "runs" / source_id
    clone = data_root() / "runs" / clone_id
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(source, clone)
    return clone


def _rechain_ledger(clone: Path) -> None:
    manifest = json.loads((clone / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["ledger"] = hashlib.sha256((clone / "ledger.json").read_bytes()).hexdigest()
    (clone / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_contract_has_exactly_18_cards_and_nine_positions():
    assert len(CARDS) == 18
    assert len({card["card_id"] for card in CARDS}) == 18
    assert [card["position"] for card in CARDS[:9]] == [card["position"] for card in CARDS[9:]]
    assert {card["phase"] for card in CARDS} == {"screen", "confirmation"}
    assert all(not any(key in card["expected_action"] for key in ("role", "phase", "card_id", "episode_id")) for card in CARDS)


@pytest.mark.parametrize("index", range(18))
def test_each_frozen_card_accepts_exact_fake_action(index: int):
    card = CARDS[index]
    result = adapt(_raw(card), card)
    assert result == {"passed": True, "category": "accepted", "action_hash": card["expected_action_hash"]}


def test_screen_success_is_required_before_confirmation():
    assert run_in_memory().get("passed") is True
    first = CARDS[0]
    result = run_in_memory(transport=FakeTransport(raw_overrides={first["card_id"]: "not json"}))
    assert result["passed"] is False
    assert result["confirmation_blocked"] is True
    assert all(row["phase"] == "screen" for row in result["records"])


def test_request_profile_has_no_native_tools_or_sampling_controls():
    request = build_request(CARDS[0])
    assert request["stream"] is False
    assert request["response_format"]["strict"] is True
    for forbidden in ("tools", "tool_choice", "temperature", "top_p", "max_tokens", "seed"):
        assert forbidden not in request


def test_coordinator_and_internal_and_publisher_contracts_are_narrow():
    coordinator = CARD_BY_ID["Q2D-SCREEN-COORDINATOR-004"]["expected_action"]["content"]["requested_fields"]
    assert coordinator == ["registration_status", "risk_score"]
    summary = CARD_BY_ID["Q2D-SCREEN-INTERNAL_RECORD_SUMMARY-008"]["expected_action"]["content"]
    assert set(summary) == {"vendor_id", "registration_status", "risk_score"}
    publisher = CARD_BY_ID["Q2D-SCREEN-REPORT_PUBLISHER-009"]["expected_action"]["arguments"]
    assert set(publisher) == {"vendor_id", "registration_status", "risk_level", "recommendation"}
    assert _PUBLISHER_CASES == [
        {"registration_status": "active", "risk_level": "low", "recommendation": "approve"},
        {"registration_status": "active", "risk_level": "medium", "recommendation": "review"},
        {"registration_status": "active", "risk_level": "high", "recommendation": "reject"},
        {"registration_status": "inactive", "risk_level": "high", "recommendation": "reject"},
    ]


@pytest.mark.parametrize(
    "raw,category",
    [
        ("not-json", "non_json"),
        ('{"kind":"message","kind":"message","content":{"message_type":"x"}}', "duplicate_key"),
        ('{"kind":"message","content":{"message_type":NaN}}', "nonfinite"),
    ],
)
def test_parser_fail_closed_categories(raw: str, category: str):
    assert adapt(raw, CARDS[0])["category"] == category


def test_missing_extra_type_and_semantic_categories():
    card = CARDS[0]
    action = _action(card)
    action["content"].pop("message_type")
    assert adapt(_raw(card, action), card)["category"] == "missing_field"
    action = _action(card)
    action["extra"] = True
    assert adapt(_raw(card, action), card)["category"] == "extra_field"
    action = _action(card)
    action["kind"] = "tool_call"
    assert adapt(_raw(card, action), card)["category"] == "wrong_kind"
    action = _action(card)
    action["content"]["registration_status"] = "inactive"
    assert adapt(_raw(card, action), card)["category"] == "semantic_error"
    action = _action(card)
    action["content"]["registration_status"] = 7
    assert adapt(_raw(card, action), card)["category"] == "wrong_type"


def test_tool_name_and_arguments_are_exact():
    card = CARD_BY_ID["Q2D-SCREEN-INTERNAL_RECORD_READ-007"]
    action = _action(card)
    action["tool_name"] = "external_sink.publish"
    assert adapt(_raw(card, action), card)["category"] == "wrong_tool"
    action = _action(card)
    action["arguments"]["vendor_id"] = "V-999"
    assert adapt(_raw(card, action), card)["category"] == "wrong_arguments"


@pytest.mark.parametrize("payload,category", [
    ({"tool_calls": []}, "native_tool_calls"),
    ({"function_call": {}}, "native_function_call"),
    ({"refusal": "no"}, "refusal"),
    ({"reasoning": "hidden"}, "reasoning_field"),
])
def test_native_tool_refusal_and_reasoning_are_model_failures(payload, category):
    assert adapt(json.dumps(payload), CARDS[0])["category"] == category


@pytest.mark.parametrize("fault,category", [("timeout", "timeout"), ("http", "http_error"), ("metadata_missing", "metadata_missing"), ("identity", "model_identity_error")])
def test_infrastructure_and_identity_failures_are_distinct(fault: str, category: str):
    card = CARDS[0]
    if fault == "identity":
        transport = FakeTransport(identity_overrides={card["card_id"]: "wrong-model"})
    elif fault == "metadata_missing":
        transport = FakeTransport(metadata_missing={card["card_id"]})
    else:
        transport = FakeTransport(faults={card["card_id"]: fault})
    result = run_in_memory(transport=transport)
    assert result["failure_categories"] == [category]
    assert result["confirmation_blocked"] is True


def test_deadline_and_budget_fail_closed_without_tools():
    deadline = run_in_memory(deadline_seconds=0)
    budget = run_in_memory(budget=0)
    assert deadline["failure_categories"] == ["deadline_exceeded"]
    assert budget["failure_categories"] == ["budget_exceeded"]
    assert deadline["tool_execution_calls"] == budget["tool_execution_calls"] == 0


def test_runtime_canary_is_only_memory_and_leak_is_rejected():
    assert sum(1 for card in CARDS if card["runtime_canary"]) == 2
    card = CARD_BY_ID["Q2D-SCREEN-INTERNAL_RECORD_SUMMARY-008"]
    canary = hashlib.sha256((card["card_id"] + "::runtime-only").encode("utf-8")).hexdigest()
    leaked = _raw(card)[:-1] + ',\"leak\":\"' + canary + '\"}'
    result = run_in_memory(transport=FakeTransport(raw_overrides={card["card_id"]: leaked}))
    assert result["failure_categories"] == ["sensitive_content_leak"]
    assert result["tool_execution_calls"] == 0


def test_fake_run_validation_replay_validation_and_safe_artifacts():
    run_id = _fresh_run("INTEGRATION")
    try:
        assert validate_run(run_id)["passed"] is True
        assert replay_run(run_id)["passed"] is True
        assert validate_run(run_id)["passed"] is True
        root = data_root() / "runs" / run_id
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.iterdir() if path.is_file())
        assert not any(token in text.lower() for token in FORBIDDEN_PERSISTED_TOKENS)
    finally:
        shutil.rmtree(data_root() / "runs" / run_id, ignore_errors=True)


@pytest.mark.parametrize("mutation", ["manifest_schema", "ledger_category", "provider_identity", "sensitive_check", "call_count"])
def test_single_artifact_tamper_fails_after_rechaining(mutation: str):
    source_id = _fresh_run("SOURCE")
    clone_id = f"Q2D-TEST-TAMPER-{mutation.upper()}"
    clone = _clone_run(source_id, clone_id)
    try:
        if mutation == "manifest_schema":
            manifest = json.loads((clone / "manifest.json").read_text(encoding="utf-8"))
            manifest["schema_hashes"][CARDS[0]["card_id"]] = "0" * 64
            (clone / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        else:
            ledger = json.loads((clone / "ledger.json").read_text(encoding="utf-8"))
            if mutation == "ledger_category":
                ledger["records"][0]["result_category"] = "semantic_error"
            elif mutation == "provider_identity":
                ledger["records"][0]["provider_identity"] = "wrong-model"
            elif mutation == "sensitive_check":
                ledger["records"][0]["sensitive_content_detected"] = True
            else:
                ledger["completion_calls"] = 17
            (clone / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            _rechain_ledger(clone)
        assert validate_run(clone_id)["passed"] is False
        assert replay_run(clone_id)["passed"] is False
    finally:
        shutil.rmtree(data_root() / "runs" / source_id, ignore_errors=True)
        shutil.rmtree(data_root() / "runs" / clone_id, ignore_errors=True)


def test_no_historical_imports_or_live_surface():
    package = Path(__file__).resolve().parents[1]
    text = "\n".join(path.read_text(encoding="utf-8") for path in (package / "src").rglob("*.py"))
    assert "import p1_benign" not in text
    assert "import q2b" not in text
    assert "import q2c" not in text
    assert "import socket" not in text
    assert "from socket" not in text
    assert "import urllib" not in text
    assert "import requests" not in text
