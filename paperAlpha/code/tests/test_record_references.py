import copy

import pytest

from paperalpha_runtime.record_references import bind_record_references


def records():
    return [{"record_id": "r1", "fields": {"name": "Alice Green", "department": "Sales", "salary": 10000}},
            {"record_id": "r2", "fields": {"name": "Bob Brown", "department": "Sales", "salary": 20000}}]


def test_individual_reference_does_not_flatten_three_objects_or_grant_permission():
    source = records() + [{"record_id": "r3", "fields": {"name": "Carol Black"}}]
    before = copy.deepcopy(source)
    result = bind_record_references("Prepare a summary for Alice Green.", source, {"name": "individual"})
    assert result["features"]["r1"]["individual_reference"] == 1
    assert result["features"]["r2"]["no_literal_reference"] == 1
    assert result["features"]["r3"]["no_literal_reference"] == 1
    assert not result["grants_permission"]
    assert not result["semantic_identity_verified"]
    assert not result["absence_means_irrelevance"]
    assert source == before


def test_multiple_field_aliases_are_not_extra_evidence():
    source = records()
    source[0]["fields"]["patient_name"] = "Alice Green"
    result = bind_record_references("Alice Green needs a follow-up.", source,
                                    {"name": "individual", "patient_name": "individual"})
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["records"][0]["fields"] == ["name", "patient_name"]


def test_shared_name_is_ambiguous_not_unique_identity():
    source = records()
    source[1]["fields"]["name"] = source[0]["fields"]["name"]
    result = bind_record_references("Alice Green needs a follow-up.", source, {"name": "individual"})
    assert all(row["ambiguous_individual_reference"] == 1 for row in result["features"].values())
    assert all(row["individual_reference"] == 0 for row in result["features"].values())


def test_group_reference_keeps_every_matching_record():
    result = bind_record_references("Generate a headcount report for Sales department", records(),
                                    {"department": "group"})
    assert all(row["group_reference"] == 1 for row in result["features"].values())
    assert not result["evidence"][0]["ambiguous_individual_reference"]


def test_quotes_and_unicode_offsets_preserve_original_text():
    request = "说明：给 alice green 和 ALICE GREEN 整理资料。"
    result = bind_record_references(request, records(), {"name": "individual"})
    assert len(result["evidence"]) == 2
    for item in result["evidence"]:
        assert request[item["start"]:item["end"]] == item["quote"]


def test_boundary_short_nonstring_and_undeclared_fields_are_not_matches():
    source = [{"record_id": "r", "fields": {"name": "Alice", "id": "ABC", "number": 10000, "secret": "FORBIDDEN"}}]
    result = bind_record_references("AliceLong ABC 10000 FORBIDDEN", source,
                                    {"name": "individual", "id": "individual", "number": "individual"})
    assert result["evidence"] == []
    assert result["features"]["r"]["no_literal_reference"] == 1
    assert len(result["unmatchable_fields"]) == 2


def test_no_reference_declaration_does_not_guess_identity_fields():
    result = bind_record_references("Alice Green", records(), {})
    assert result["evidence"] == []
    assert all(row["no_literal_reference"] for row in result["features"].values())


@pytest.mark.parametrize("task_text,source,fields", [
    ("", records(), {}),
    ("task", [], {}),
    ("task", records(), {"name": "permission"}),
    ("task", records() + [records()[0]], {}),
    ("task", [{"record_id": "r", "fields": []}], {}),
])
def test_invalid_bindings_fail_explicitly(task_text, source, fields):
    with pytest.raises(ValueError):
        bind_record_references(task_text, source, fields)
