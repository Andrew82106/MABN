import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_runtime.semantic_evidence import (TextEvidenceWindow, validate_annotation,
    quote_span, request_payload, response_annotation, digest)


def rows():
    binding = {"invocation_id": "call1", "provider_call_id": "sdk1", "actor": "a", "tool": "read_context"}
    content = "Budget cap is $400. Approval is not recorded."
    data = [
        ("run_started", {"agents": ["a", "b", "c"], "external_endpoints": ["user"]}),
        ("communication", {"actor": "c", "target": "b", "content": "Private message for b only."}),
        ("tool_handler_started", {**binding, "arguments": {}}),
        ("tool_native_result", {**binding, "arguments": {}, "result": content}),
        ("tool_handler_returned", {**binding, "result": content}),
        ("communication", {"actor": "a", "target": "b", "content": "Please keep payment below $400."}),
        ("communication", {"actor": "b", "target": "a", "content": "I have approval. No receipt exists."}),
        ("communication", {"actor": "a", "target": "user", "content": "We have not paid."}),
        ("run_finished", {"status": "completed"}),
    ]
    return [{"schema_version": "runtime-journal-v1", "sequence": i, "kind": kind, "payload": payload}
            for i, (kind, payload) in enumerate(data, 1)]


def packets(records=None, **kwargs):
    window = TextEvidenceWindow(**kwargs)
    return [p for row in (rows() if records is None else records) if (p := window.push(row)) is not None]


def annotation(quote="Please keep payment below $400."):
    return {"status": "observations", "more_possible": False, "observations": [
        {"kind": "constraint", "presentation": "direct", "polarity": "affirmed", "quote": quote, "occurrence": 0,
         "references": [{"record_id": "journal-5", "quote": "Budget cap is $400.", "occurrence": 0, "relation": "supports"}]}]}


def test_tool_returns_are_selected_once_and_at_their_actual_time():
    values = packets()
    assert [p["current"]["sequence"] for p in values] == [2, 5, 6, 7, 8]
    assert values[1]["current"]["content"] == "Budget cap is $400. Approval is not recorded."
    assert values[1]["current"]["invocation_id"] == "call1"
    assert not values[1]["prior_records"]  # c->b is not accessible to a.


def test_context_contains_only_recorded_access_not_all_observations():
    values = packets()
    assert [d["record_id"] for d in values[2]["prior_records"]] == ["journal-5"]
    assert [d["record_id"] for d in values[3]["prior_records"]] == ["journal-2", "journal-6"]
    assert "journal-5" not in [d["record_id"] for d in values[3]["prior_records"]]


def test_future_content_and_final_status_do_not_change_prefix_packets():
    original = rows()
    modified = copy.deepcopy(original)
    modified[6]["payload"]["content"] = "Entirely different future communication."
    modified[-1]["payload"]["status"] = "failed"
    assert packets(original)[:3] == packets(modified)[:3]
    assert packets(original[:6]) == packets(original)[:3]


def test_context_bounds_omit_whole_records_and_report_gaps():
    value = packets(max_prior_records=1)[-1]
    assert [d["record_id"] for d in value["prior_records"]] == ["journal-7"]
    assert value["context"]["omitted_prior_records"] == 2
    tiny = packets(max_context_characters=3)[0]
    assert tiny["context"]["current_over_budget"]
    with pytest.raises(ValueError, match="budget"):
        request_payload(tiny)


def test_quoted_claim_and_prior_reference_are_grounded_but_not_certified_true():
    packet = packets()[2]
    result = validate_annotation(packet, annotation())
    obs = result["observations"][0]
    assert packet["current"]["content"][obs["anchor"]["start"]:obs["anchor"]["end"]] == obs["anchor"]["quote"]
    assert not obs["classification_verified"]
    assert not obs["references"][0]["semantic_relation_verified"]
    assert not result["authorization_established"] and not result["safety_established"]
    assert result["input_packet_sha256"] == digest(packet)


@pytest.mark.parametrize("change", ["quote", "occurrence", "future", "inaccessible", "extra", "permission", "duplicate", "category", "status"])
def test_invalid_or_unavailable_evidence_is_not_silently_repaired(change):
    value = annotation()
    item = value["observations"][0]
    if change == "quote":
        item["quote"] = "Payment has been approved."
    elif change == "occurrence":
        item["occurrence"] = 1
    elif change == "future":
        item["references"][0]["record_id"] = "journal-7"
    elif change == "inaccessible":
        item["references"][0].update(record_id="journal-2", quote="Private message for b only.")
    elif change == "extra":
        value["risk_probability"] = 0.1
    elif change == "permission":
        item["permission_status"] = "approved"
    elif change == "duplicate":
        value["observations"].append(copy.deepcopy(item))
    elif change == "category":
        item["kind"] = "verified_authorization"
    else:
        value["status"] = "no_supported_claims"
    with pytest.raises(ValueError):
        validate_annotation(packets()[2], value)


def test_unicode_offsets_and_repeated_exact_quotes():
    doc = {"record_id": "x", "sha256": "test", "content": "🙂 已批准。已批准。"}
    assert quote_span(doc, "已批准", 1)["start"] == 6
    with pytest.raises(ValueError):
        quote_span(doc, "已 批准", 0)


def test_no_supported_claims_is_not_safe_and_has_no_permission_event():
    result = validate_annotation(packets()[0], {"status": "no_supported_claims", "more_possible": False, "observations": []})
    assert not result["safety_established"] and not result["authorization_established"]
    assert "event_type" not in result and "data_class" not in result


def test_request_uses_fixed_model_structured_output_no_tools_or_future_status():
    request = request_payload(packets()[2])
    assert request["model"] == "gpt-5.5" and request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema" and request["text"]["format"]["strict"]
    assert "tools" not in request
    visible = json.loads(request["input"][1]["content"])
    assert "run_status" not in visible and "label" not in visible
    assert visible["current"]["sequence"] == 6


@pytest.mark.parametrize("status,content", [
    ("incomplete", [{"type": "output_text", "text": "{}"}]),
    ("completed", [{"type": "refusal", "refusal": "no"}]),
    ("completed", [{"type": "output_text", "text": "not json"}]),
    ("completed", []),
])
def test_invalid_api_outputs_never_become_empty_successful_extractions(status, content):
    with pytest.raises(ValueError):
        response_annotation({"status": status, "output": [{"type": "message", "content": content}]})


def test_complete_response_is_parsed_without_markdown_repair():
    result = {"status": "no_supported_claims", "more_possible": False, "observations": []}
    response = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(result)}]}]}
    assert response_annotation(response) == result
