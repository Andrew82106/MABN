from __future__ import annotations

import json
from pathlib import Path

import pytest

from p1v2_benign.contract import ContractPolicy, validate_response
from p1v2_benign.core import StrictJsonError, strict_json_loads
from p1v2_benign.runner import build_dry_records
from p1v2_benign.workflow import MockPublicSink, evaluate_response


CODE_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FIELDS = ("vendor_id", "registration_status", "risk_level", "recommendation")
POLICY = ContractPolicy(
    allowed_roles=("coordinator", "publisher"),
    required_public_fields=PUBLIC_FIELDS,
    allow_continue_value="allow_continue",
    reject_value="reject",
)
VALID_RESPONSE = json.dumps(
    {
        "role": "coordinator",
        "episode_id": "P1V2-EP-001",
        "decision": "allow_continue",
        "public_report": {
            "vendor_id": "V-017",
            "registration_status": "active",
            "risk_level": "low",
            "recommendation": "approve",
        },
    }
)


def test_independent_literal_config_fixture_and_prompt_oracle() -> None:
    config = json.loads((CODE_ROOT / "configs" / "readiness.json").read_text(encoding="utf-8"))
    fixture = json.loads((CODE_ROOT / "fixtures" / "response_cases.json").read_text(encoding="utf-8"))
    prompt = (CODE_ROOT / "prompts" / "public_response_contract.txt").read_text(encoding="utf-8")
    assert config["allow_continue_value"] == "allow_continue"
    assert config["reject_value"] == "reject"
    assert config["allowed_roles"] == ["coordinator", "publisher"]
    assert config["required_public_fields"] == list(PUBLIC_FIELDS)
    assert config["expected_case_ids"] == [
        "valid_allow",
        "non_json",
        "duplicate_key",
        "nan_value",
        "infinity_value",
        "empty_input",
        "missing_public_field",
        "extra_internal_field",
        "extra_fixture_secret",
        "nested_extra_field",
        "role_mismatch",
        "episode_mismatch",
    ]
    assert [row["case_id"] for row in fixture] == config["expected_case_ids"]
    assert "never sent to a model" in prompt


def test_schema_oracle_requires_exclusive_conditional_fields() -> None:
    schema = json.loads((CODE_ROOT / "schemas" / "response_contract.schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert len(schema["oneOf"]) == 2
    branches = {branch["properties"]["decision"]["const"]: branch for branch in schema["oneOf"]}
    assert set(branches) == {"allow_continue", "reject"}
    allow = branches["allow_continue"]
    reject = branches["reject"]
    assert allow["required"] == ["role", "episode_id", "decision", "public_report"]
    assert reject["required"] == ["role", "episode_id", "decision", "rejection_reason"]
    assert allow["additionalProperties"] is False
    assert reject["additionalProperties"] is False
    assert reject["properties"]["rejection_reason"] == {
        "minLength": 1,
        "pattern": ".*\\S.*",
        "type": "string",
    }


@pytest.mark.parametrize(
    ("raw", "valid", "code"),
    [
        (VALID_RESPONSE, True, None),
        ('{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"allow_continue"}', False, "unexpected_or_missing_response_field"),
        ('{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"reject"}', False, "unexpected_or_missing_response_field"),
        ('{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"reject","rejection_reason":"fixture","public_report":{}}', False, "unexpected_or_missing_response_field"),
        ('{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"reject","rejection_reason":"   "}', False, "invalid_rejection_reason"),
        ('{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"reject","rejection_reason":"fixture"}', True, "declared_reject"),
    ],
)
def test_schema_execution_parity_matrix(raw: str, valid: bool, code: str | None) -> None:
    result = validate_response(raw, "coordinator", "P1V2-EP-001", POLICY)
    assert result["contract_valid"] is valid
    assert result["rejection_code"] == code


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("not-json", "invalid_json"),
        ("   ", "empty_json"),
        ('{"role":"coordinator","role":"publisher"}', "duplicate_json_key"),
        ("{\"role\":NaN}", "nonstandard_json_number"),
        ("{\"role\":Infinity}", "nonstandard_json_number"),
        ("1e999", "nonfinite_json_number"),
        (
            '{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"allow_continue","public_report":{"vendor_id":"V-017","registration_status":"active","risk_level":"low"}}',
            "unexpected_or_missing_public_field",
        ),
        (
            '{"role":"coordinator","episode_id":"P1V2-EP-001","decision":"allow_continue","public_report":{"vendor_id":"V-017","registration_status":"active","risk_level":"low","recommendation":"approve","internal_record":"x"}}',
            "unexpected_or_missing_public_field",
        ),
        (
            '{"role":"publisher","episode_id":"P1V2-EP-001","decision":"allow_continue","public_report":{"vendor_id":"V-017","registration_status":"active","risk_level":"low","recommendation":"approve"}}',
            "role_mismatch",
        ),
        (
            '{"role":"coordinator","episode_id":"P1V2-EP-999","decision":"allow_continue","public_report":{"vendor_id":"V-017","registration_status":"active","risk_level":"low","recommendation":"approve"}}',
            "episode_mismatch",
        ),
    ],
)
def test_literal_bad_responses_fail_closed(raw: str, code: str) -> None:
    result = validate_response(raw, "coordinator", "P1V2-EP-001", POLICY)
    assert result["contract_valid"] is False
    assert result["decision"] == "reject"
    assert result["rejection_code"] == code


def test_strict_json_rejects_nonfinite_exponent_before_contract_logic() -> None:
    with pytest.raises(StrictJsonError, match="nonfinite_json_number"):
        strict_json_loads("1e999")


def test_bad_cases_never_reach_mock_public_sink() -> None:
    sink = MockPublicSink()
    allowed = evaluate_response(
        raw_response=VALID_RESPONSE,
        expected_role="coordinator",
        expected_episode_id="P1V2-EP-001",
        policy=POLICY,
        sink=sink,
    )
    bad = evaluate_response(
        raw_response="not-json",
        expected_role="coordinator",
        expected_episode_id="P1V2-EP-001",
        policy=POLICY,
        sink=sink,
    )
    assert allowed["public_sink_published"] is True
    assert bad["public_sink_published"] is False
    assert len(sink.published_reports) == 1


def test_frozen_fixture_has_one_allowed_public_record_and_all_required_negative_types() -> None:
    _, _, _, outcomes, sink = build_dry_records()
    expected_rejected = {
        "non_json",
        "duplicate_key",
        "nan_value",
        "infinity_value",
        "empty_input",
        "missing_public_field",
        "extra_internal_field",
        "extra_fixture_secret",
        "nested_extra_field",
        "role_mismatch",
        "episode_mismatch",
    }
    assert [row["case_id"] for row in outcomes if row["public_sink_published"]] == ["valid_allow"]
    assert {row["case_id"] for row in outcomes if not row["public_sink_published"]} == expected_rejected
    assert len(sink.published_reports) == 1


def test_schema_is_an_executable_not_only_hashed_input() -> None:
    schema_path = CODE_ROOT / "schemas" / "response_contract.schema.json"
    original = schema_path.read_bytes()
    changed = json.loads(original.decode("utf-8"))
    for branch in changed["oneOf"]:
        branch["properties"]["role"]["enum"] = ["publisher"]
    schema_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="config_schema_mismatch"):
            build_dry_records()
    finally:
        schema_path.write_bytes(original)
