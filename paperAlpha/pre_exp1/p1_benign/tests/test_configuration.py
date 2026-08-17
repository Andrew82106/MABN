from __future__ import annotations

import json

import pytest

from p1_benign.configuration import load_static_inputs
from p1_benign.core import (
    DEFAULT_CONTEXT,
    EXPECTED_EDGES,
    EXPECTED_ROLE_IDS,
    REQUIRED_PUBLIC_FIELDS,
    expected_report_from_internal,
    recommendation_for,
    risk_level_for_score,
)
from p1_benign.event_contracts import EVENT_PAYLOAD_SCHEMAS


def test_exactly_eight_frozen_roles() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    assert tuple(
        agent["role_id"] for agent in static.agents["agents"]
    ) == EXPECTED_ROLE_IDS


def test_fixed_graph_edges_and_order() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    assert tuple(
        (edge["source"], edge["target"])
        for edge in static.graph["edges"]
    ) == EXPECTED_EDGES
    assert tuple(static.graph["execution_order"]) == EXPECTED_ROLE_IDS


def test_graph_disables_gate_shared_memory_and_hidden_channels() -> None:
    graph = load_static_inputs(DEFAULT_CONTEXT).graph
    assert graph["message_gating"] is False
    assert graph["shared_memory"] is False
    assert graph["hidden_channels"] is False


def test_twenty_unique_deterministic_tasks() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    assert len(static.task_ids) == 20
    assert len(
        {item["vendor_id"] for item in static.materials.values()}
    ) == 20


def test_canaries_are_unique() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    canaries = [
        item["internal_canary"]
        for item in static.internal_records.values()
    ]
    assert len(canaries) == 20
    assert len(set(canaries)) == 20


def test_canary_values_exist_only_in_internal_database_fixture() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    non_internal_text = "\n".join(
        [
            json.dumps(
                list(static.materials.values()),
                ensure_ascii=False,
            ),
            json.dumps(
                list(static.expected_reports.values()),
                ensure_ascii=False,
            ),
            json.dumps(static.task_template, ensure_ascii=False),
            *static.prompts.values(),
        ]
    )
    for record in static.internal_records.values():
        assert record["internal_canary"] not in non_internal_text


def test_benign_material_has_no_instruction_or_internal_field() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    for material in static.materials.values():
        assert "instruction" not in material
        assert "internal_canary" not in material
        assert "classification" not in material
        assert "risk_score" not in material


def test_expected_reports_have_only_public_fields() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    for item in static.expected_reports.values():
        assert set(item["report"]) == set(REQUIRED_PUBLIC_FIELDS)


def test_expected_reports_follow_frozen_rule() -> None:
    static = load_static_inputs(DEFAULT_CONTEXT)
    for task_id in static.task_ids:
        assert static.expected_reports[task_id]["report"] == (
            expected_report_from_internal(
                static.internal_records[task_id]
            )
        )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "low"),
        (3, "low"),
        (4, "medium"),
        (6, "medium"),
        (7, "high"),
        (10, "high"),
    ],
)
def test_risk_boundaries(score: int, expected: str) -> None:
    assert risk_level_for_score(score) == expected


@pytest.mark.parametrize("score", [-1, 11, 1.5, True])
def test_invalid_risk_score_rejected(score) -> None:
    with pytest.raises(ValueError):
        risk_level_for_score(score)


@pytest.mark.parametrize(
    ("status", "level", "expected"),
    [
        ("active", "low", "approve"),
        ("active", "medium", "manual_review"),
        ("active", "high", "reject"),
        ("suspended", "low", "reject"),
        ("pending", "medium", "reject"),
    ],
)
def test_recommendation_rule(
    status: str,
    level: str,
    expected: str,
) -> None:
    assert recommendation_for(status, level) == expected


def test_all_role_prompts_are_present_and_nonempty() -> None:
    prompts = load_static_inputs(DEFAULT_CONTEXT).prompts
    assert set(prompts) == set(EXPECTED_ROLE_IDS)
    assert all(prompt.strip() for prompt in prompts.values())


def test_condition_is_unintervened_benign_baseline() -> None:
    condition = load_static_inputs(DEFAULT_CONTEXT).experiment["condition"]
    assert condition == {
        "condition": "benign_baseline",
        "risk_seed_present": False,
        "intervention_applied": False,
    }


def test_xfyun_catalog_is_usage_restricted() -> None:
    providers = load_static_inputs(DEFAULT_CONTEXT).providers["providers"]
    xfyun = next(
        item
        for item in providers
        if item["provider_id"] == "xfyun_astron_coding_plan"
    )
    assert xfyun["usage_scope"] == "interactive_coding_only"
    assert xfyun["allowed_for_automated_experiment"] is False
    assert xfyun["credential_env"] == "XFYUN_ASTRON_CODING_API_KEY"
    assert xfyun["executable_by_p1"] is False


def test_xfyun_model_whitelist_is_frozen() -> None:
    providers = load_static_inputs(DEFAULT_CONTEXT).providers["providers"]
    xfyun = next(
        item
        for item in providers
        if item["provider_id"] == "xfyun_astron_coding_plan"
    )
    assert xfyun["allowed_models"] == [
        "xsparkx2agent",
        "xopglm5",
        "xopglm52",
        "xopdeepseekv4pro",
        "xopdeepseekv4flash",
        "xopkimik26",
        "xopglm51",
        "xopqwen36v35b",
        "xopqwen35397b",
    ]


def test_four_p1_schemas_exist() -> None:
    schema_root = DEFAULT_CONTEXT.static_root / "schemas"
    assert {
        path.name for path in schema_root.glob("*.schema.json")
    } == {
        "event.schema.json",
        "manifest.schema.json",
        "outcome.schema.json",
        "replay.schema.json",
    }


def test_every_event_type_has_one_strict_payload_contract() -> None:
    event_schema = json.loads(
        (
            DEFAULT_CONTEXT.static_root
            / "schemas"
            / "event.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert set(EVENT_PAYLOAD_SCHEMAS) == set(
        event_schema["properties"]["event_type"]["enum"]
    )
