import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_runtime import semantic_evidence as v1
from paperalpha_runtime import semantic_evidence_v2 as v2


def packet(text="May pay up to $400 only after confirmation.", prior="Do not pay yet."):
    def document(identity, sequence, content):
        value = {"record_id": identity, "sequence": sequence, "kind": "communication", "actor": "a", "target": "b",
                 "tool": None, "invocation_id": None, "content": content}
        value["sha256"] = v1.digest(value)
        return value
    return {"schema_version": "text-evidence-input-v1", "current": document("current", 2, text),
            "prior_records": [document("earlier", 1, prior)], "context": {"current_over_budget": False}}


def annotation():
    return {"status": "observations", "more_possible": False, "observations": [{
        "predicate": "permission_claimed_conditional", "presentation": "direct",
        "quote": "May pay up to $400 only after confirmation.", "occurrence": 0,
        "condition_quotes": [{"quote": "only after confirmation", "occurrence": 0}],
        "quantity_quote": {"quote": "$400", "occurrence": 0}, "references": [{
            "record_id": "earlier", "relation": "stage_update", "quote": "Do not pay yet.", "occurrence": 0,
            "object_match": "same", "time_match": "updated", "basis_match": "unknown"}]}]}


def test_typed_targets_do_not_use_global_polarity_or_prove_permission():
    result = v2.validate_annotation(packet(), annotation())
    obs = result["observations"][0]
    assert (obs["kind"], obs["state"]) == ("permission", "claimed_conditional")
    assert "polarity" not in obs and obs["conditions_satisfied"] is None
    assert not result["authorization_established"] and not obs["classification_verified"]
    assert obs["condition_anchors"][0]["quote"] == "only after confirmation"
    assert obs["quantity_anchor"]["quote"] == "$400"


@pytest.mark.parametrize("predicate", list(v2.PREDICATES))
def test_each_predicate_has_an_explicit_distinct_target(predicate):
    value = annotation()
    value["observations"][0]["predicate"] = predicate
    result = v2.validate_annotation(packet(), value)
    obs = result["observations"][0]
    assert (obs["kind"], obs["state"]) == v2.PREDICATES[predicate]
    # The validator checks representation, not whether this predicate fits the text.
    assert obs["classification_verified"] is False


def test_missing_evidence_not_denied_permission_and_estimate_not_limit():
    assert v2.PREDICATES["evidence_claimed_missing"] != v2.PREDICATES["permission_claimed_denied"]
    assert v2.PREDICATES["reported_estimate"] != v2.PREDICATES["normative_upper_bound"]
    assert v2.PREDICATES["execution_claimed_not_occurred"] != v2.PREDICATES["action_prohibited"]


@pytest.mark.parametrize("field,state", [("object_match", "different"), ("time_match", "updated"), ("basis_match", "unknown")])
def test_proposed_conflict_with_incomplete_comparability_is_not_certified(field, state):
    value = annotation()
    ref = value["observations"][0]["references"][0]
    ref.update(relation="conflict", object_match="same", time_match="same", basis_match="same")
    ref[field] = state
    relation = v2.validate_annotation(packet(), value)["observations"][0]["references"][0]
    assert relation["proposed_relation"] == "conflict"
    assert relation["relation_hypothesis"] == "unresolved_comparison"
    assert not relation["semantic_relation_verified"] and not relation["comparability_verified"]


def test_even_fully_matching_scope_judgments_remain_a_candidate():
    value = annotation()
    value["observations"][0]["references"][0].update(relation="conflict", object_match="same", time_match="same", basis_match="same")
    relation = v2.validate_annotation(packet(), value)["observations"][0]["references"][0]
    assert relation["relation_hypothesis"] == "scope_matched_conflict_candidate"
    assert not relation["semantic_relation_verified"]


@pytest.mark.parametrize("change", ["missing_condition", "wrong_condition", "outside_condition", "missing_quantity", "old_polarity", "future", "duplicate", "bad_match"])
def test_bad_typed_evidence_is_rejected(change):
    value, data = annotation(), packet()
    item = value["observations"][0]
    if change == "missing_condition":
        item["condition_quotes"] = []
    elif change == "wrong_condition":
        item["condition_quotes"][0]["quote"] = "without confirmation"
    elif change == "outside_condition":
        data = packet("May pay up to $400 only after confirmation. Separate unrelated condition.")
        item["condition_quotes"][0]["quote"] = "Separate unrelated condition."
    elif change == "missing_quantity":
        item.update(predicate="normative_upper_bound", quantity_quote=None)
    elif change == "old_polarity":
        item["polarity"] = "affirmed"
    elif change == "future":
        item["references"][0]["record_id"] = "later"
    elif change == "duplicate":
        value["observations"].append(copy.deepcopy(item))
    else:
        item["references"][0]["time_match"] = "verified"
    with pytest.raises(ValueError):
        v2.validate_annotation(data, value)


def test_subquote_occurrence_is_local_but_returned_offsets_are_global():
    data = packet("Earlier estimate $400. May pay up to $400 only after confirmation.")
    value = annotation()
    result = v2.validate_annotation(data, value)["observations"][0]
    part = result["quantity_anchor"]
    assert part["start"] == data["current"]["content"].rfind("$400")
    assert data["current"]["content"][part["start"]:part["end"]] == "$400"


def test_v2_requests_have_new_schema_without_mutating_v1():
    before = v1.request_payload(packet())
    upgraded = v2.request_payload(packet())
    assert upgraded["model"] == before["model"] == "gpt-5.5"
    assert upgraded["max_output_tokens"] == 3200 and before["max_output_tokens"] == 2200
    assert upgraded["text"]["format"]["schema"] != before["text"]["format"]["schema"]
    assert before == v1.request_payload(packet())
