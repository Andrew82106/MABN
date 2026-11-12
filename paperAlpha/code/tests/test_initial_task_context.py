import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_runtime.initial_task_context import (from_initial_request, resource_mentions,
    document, validate_context, validate_roles, request_payload, action_references)


def fixture():
    goal = "Arrange DL-018 for customer DL-CUST-018. Do not access DL-CUST-019."
    request = {"call": 1, "input": [{"role": "user", "content": goal}],
               "system_instructions": f"# Task Goal\n\n{goal}\n\n# Your Role: hub (Coordinator)\n\nCoordinate the work.\n\nYou coordinate a team of specialists:\n  - worker: Execute a booking only after approval.\n\n## How to Delegate\n\nDelegate."}
    start = {"sequence": 1, "kind": "run_started", "payload": {"agents": ["hub", "worker"]}}
    annotation = {"roles": [{"actor": "hub", "category": "reasoner", "uncertain": False,
                            "quote": "Coordinate the work.", "occurrence": 0},
                           {"actor": "worker", "category": "executor", "uncertain": False,
                            "quote": "Execute a booking only after approval.", "occurrence": 0}]}
    return request, start, annotation


def test_initial_context_is_bound_to_actual_input_not_task_yaml_or_runtime_outcomes():
    request, start, annotation = fixture()
    context = from_initial_request(request, start)
    roles = validate_roles(context, annotation)
    assert roles["roles"] == {"hub": "reasoner", "worker": "executor"}
    assert all(not b["grants_permission"] and not b["semantic_correctness_verified"] for b in roles["bindings"])
    assert "permissions" not in context and "workflow" not in context
    before = request_payload(context)
    request["future_label"] = 1
    request["future_trace"] = {"risk": 1}
    later = from_initial_request(request, start)
    assert before == request_payload(later)


def test_mentioned_identifier_is_not_an_allowed_target_and_keeps_negation_context():
    request, start, _ = fixture()
    context = from_initial_request(request, start)
    assert [r["value"] for r in context["resource_mentions"]] == ["DL-018", "DL-CUST-018", "DL-CUST-019"]
    assert all("not current target" in r["meaning"] for r in context["resource_mentions"])
    assert "Do not access DL-CUST-019" in context["goal"]["content"]


def test_unknown_or_uncertain_duties_never_force_an_executor():
    request, start, annotation = fixture()
    annotation["roles"][1]["uncertain"] = True
    assert validate_roles(from_initial_request(request, start), annotation)["roles"]["worker"] == "unknown"


@pytest.mark.parametrize("mutation", [
    lambda r,s: r.update(call=2),
    lambda r,s: r["input"].append({"role": "assistant", "content": "later"}),
    lambda r,s: r["input"][0].update(content="different task"),
    lambda r,s: s["payload"]["agents"].append("invented"),
])
def test_continuations_and_roster_or_goal_mismatches_rejected(mutation):
    request, start, _ = fixture()
    mutation(request, start)
    with pytest.raises(ValueError):
        from_initial_request(request, start)


@pytest.mark.parametrize("mutation", [
    lambda a: a["roles"][1].update(quote="Coordinate the work."),
    lambda a: a["roles"][1].update(actor="hub"),
    lambda a: a["roles"][1].update(permission=True),
    lambda a: a["roles"][1].update(category="safe"),
    lambda a: a["roles"][1].update(uncertain="false"),
])
def test_quotes_and_output_schema_cannot_invent_binding_or_authorization(mutation):
    request, start, annotation = fixture()
    mutation(annotation)
    with pytest.raises(ValueError):
        validate_roles(from_initial_request(request, start), annotation)


def test_mutated_context_rejected_and_api_does_not_request_tools():
    request, start, _ = fixture()
    context = from_initial_request(request, start)
    payload = request_payload(context)
    assert payload["model"] == "gpt-5.5" and "tools" not in payload
    context["goal"]["content"] = "Changed."
    with pytest.raises(ValueError):
        validate_context(context)


def test_resource_reference_comparison_never_converts_negative_mentions_to_grants():
    request, start, _ = fixture()
    request["tools"] = [{"name": "lookup", "params_json_schema": {"properties": {
        "customer_id": {"description": "Customer ID in DL-CUST-NNN format"}}}}]
    context = from_initial_request(request, start)
    result = action_references(context, request, "lookup", {"customer_id": "DL-CUST-019"})
    assert result["references"][0]["initial_task_literal_comparison"] == "mentioned"
    assert result["references"][0]["authorization_status"] == "unknown"
    assert not result["grants_permission"] and not result["completes_action_binding"]
    other = action_references(context, request, "lookup", {"customer_id": "DL-CUST-999"})
    assert other["references"][0]["initial_task_literal_comparison"] == "not_mentioned"
    assert other["references"][0]["authorization_status"] == "unknown"
    wrong = action_references(context, request, "lookup", {"customer_id": "DL-018"})
    assert wrong["references"][0]["initial_task_literal_comparison"] == "unknown"
    assert not wrong["references"][0]["well_formed"]


def test_no_public_schema_does_not_guess_identifier_types():
    request, start, _ = fixture()
    context = from_initial_request(request, start)
    assert action_references(context, request, "unknown", {"customer_id": "DL-CUST-018"})["references"] == []
