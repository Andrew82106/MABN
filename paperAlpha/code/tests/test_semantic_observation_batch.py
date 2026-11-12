import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_runtime.semantic_observation_batch import validate_items
from paperalpha_runtime.semantic_evidence_v2 import validate_annotation
from test_semantic_evidence_v2 import packet, annotation


def test_unrelated_valid_claim_survives_without_repair_of_missing_condition():
    data, proposed = packet(), annotation()
    broken = copy.deepcopy(proposed["observations"][0])
    proposed["observations"][0]["predicate"] = "normative_upper_bound"
    broken["condition_quotes"] = []
    proposed["observations"].append(broken)
    saved = copy.deepcopy(proposed)
    with pytest.raises(ValueError, match="condition"):
        validate_annotation(data, proposed)
    result = validate_items(data, proposed)
    assert proposed == saved
    assert (result["accepted_items"], result["rejected_items"], result["acceptance_status"]) == (1, 1, "partial")
    assert result["items"][1]["reason"] == "invalid_item"
    assert not result["authorization_established"] and not result["independent_evidence"]
    assert not result["exhaustive_coverage_established"]
    assert result["items"][0]["observation"]["conditions_satisfied"] is None


def test_every_duplicate_identity_is_rejected_not_arbitrarily_first_selected():
    proposed = annotation()
    other = copy.deepcopy(proposed["observations"][0])
    other["references"] = []  # Adding/removing references cannot create another proposition.
    proposed["observations"].append(other)
    result = validate_items(packet(), proposed)
    assert result["accepted_items"] == 0 and result["acceptance_status"] == "rejected"
    assert all(d["duplicate_item_indices"] == [0, 1] for d in result["items"])
    assert all("observation" not in d for d in result["items"])


def test_same_quote_different_predicate_is_not_independent_evidence():
    proposed = annotation()
    other = copy.deepcopy(proposed["observations"][0])
    other["predicate"] = "normative_upper_bound"
    proposed["observations"].append(other)
    result = validate_items(packet(), proposed)
    assert result["accepted_items"] == 2 and not result["independent_evidence"]
    assert result["observation_group_id"] == packet()["current"]["sha256"]


def test_same_sentence_two_caps_have_distinct_subpropositions_but_one_group():
    data = packet("Direct-pay cap $400; reimbursement cap $650.")
    first = {"predicate": "normative_upper_bound", "presentation": "direct", "occurrence": 0,
             "quote": data["current"]["content"], "condition_quotes": [],
             "quantity_quote": {"quote": "$400", "occurrence": 0}, "references": []}
    second = copy.deepcopy(first)
    second["quantity_quote"]["quote"] = "$650"
    value = {"status": "observations", "more_possible": False, "observations": [first, second]}
    # The frozen whole-record v2 policy was too coarse for this legitimate case.
    with pytest.raises(ValueError, match="duplicate"):
        validate_annotation(data, value)
    result = validate_items(data, value)
    assert result["accepted_items"] == 2 and not result["independent_evidence"]
    assert result["schema_version"] == "text-evidence-item-batch-v2"


def test_same_sentence_distinct_qualifications_remain_separate_unverified_candidates():
    data = packet("May pay if approved, may submit if confirmed.")
    item = {"predicate": "permission_claimed_conditional", "presentation": "direct", "occurrence": 0,
            "quote": data["current"]["content"], "condition_quotes": [{"quote": "if approved", "occurrence": 0}],
            "quantity_quote": None, "references": []}
    second = copy.deepcopy(item)
    second["condition_quotes"][0]["quote"] = "if confirmed"
    result = validate_items(data, {"status": "observations", "more_possible": False, "observations": [item, second]})
    assert result["accepted_items"] == 2
    assert all(i["observation"]["conditions_satisfied"] is None for i in result["items"])


@pytest.mark.parametrize("mutation", ["extra", "wrong_status", "too_many", "wrong_more", "no_list"])
def test_invalid_envelopes_are_not_salvaged(mutation):
    proposed = annotation()
    if mutation == "extra":
        proposed["safe"] = True
    elif mutation == "wrong_status":
        proposed["status"] = "no_supported_claims"
    elif mutation == "too_many":
        proposed["observations"] *= 9
    elif mutation == "wrong_more":
        proposed["more_possible"] = 1
    else:
        proposed["observations"] = {}
    with pytest.raises(ValueError):
        validate_items(packet(), proposed)


@pytest.mark.parametrize("status", ["uncertain", "no_supported_claims"])
def test_empty_is_distinct_from_successful_semantic_or_safety_estimation(status):
    result = validate_items(packet(), {"status": status, "more_possible": False, "observations": []})
    assert result["acceptance_status"] == "empty" and result["extraction_status"] == status
    assert not result["safety_established"] and not result["exhaustive_coverage_established"]


def test_frozen_validator_can_be_supplied_and_is_called_per_item():
    calls = []
    def frozen(data, value):
        calls.append(copy.deepcopy(value))
        return validate_annotation(data, value)
    result = validate_items(packet(), annotation(), strict_validator=frozen)
    assert len(calls) == 1 and result["accepted_items"] == 1


def test_invalid_item_type_and_bad_quote_do_not_erase_other_items():
    proposed = annotation()
    broken = copy.deepcopy(proposed["observations"][0])
    broken["quote"] = "Something never said."
    proposed["observations"].extend([None, broken])
    result = validate_items(packet(), proposed)
    assert (result["accepted_items"], result["rejected_items"]) == (1, 2)


def test_full_acceptance_matches_strict_validator_without_changing_semantic_claims():
    data, proposed = packet(), annotation()
    result = validate_items(data, proposed)
    strict = validate_annotation(data, proposed)
    assert result["acceptance_status"] == "all_items_accepted"
    assert [d["observation"] for d in result["items"]] == strict["observations"]
