"""Constructed protocol tests; fixtures are not real API study data."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import mas_purchase_study as study
from paperalpha_runtime.quadsentinel_adapter import validate_policy_book
from scalar_binary_graph import evaluate


def proposal(case, actor):
    branch = next(b for b in case["branches"] if b["actor"] == actor)
    item = branch["catalogue"][0]
    return json.dumps({"decision": "propose", "sku": item["sku"], "amount": item["price"], "currency": "USD", "note": "Protocol fixture."})


def packet(case):
    return study.packet(case, [study.parse_proposal(case, b["actor"], proposal(case, b["actor"])) for b in case["branches"]])


@pytest.mark.parametrize("case", study.cases(), ids=lambda c: c["id"])
def test_independent_reference_and_bn_graph_agree(case):
    value = packet(case)
    rule, result = study.reference_verdict(value), study.score_bn(value)
    assert rule["allowed"] == result["allowed"]
    assert rule["violations"] == result["violations"]
    assert result["unknown_authority_facts"] == 0
    assert result["trained"] is False
    for key, graph in result["graphs"].items():
        assert evaluate(graph) == pytest.approx(result["probabilities"][key], abs=1e-12)
    assert rule["allowed"] == (case["setting"] in {"shared_relaxed", "separate_pools"})
    local = study.reference_verdict(value, local_only=True)
    assert local["allowed"] == (case["setting"] != "revoked_parent")


def test_identical_amounts_different_pool_relations_change_verdict():
    a, b = [next(c for c in study.cases() if c["id"] == "office-" + s) for s in ("shared_tight", "separate_pools")]
    x, y = packet(a), packet(b)
    assert x["actions"] == y["actions"]
    assert not study.score_bn(x)["allowed"]
    assert study.score_bn(y)["allowed"]


def test_shared_parent_revocation_changes_both_children():
    case = next(c for c in study.cases() if c["id"] == "office-revoked_parent")
    result = study.score_bn(packet(case))
    assert result["violations"] == ["authorization:purchase-a", "authorization:purchase-b"]


def test_prompts_and_monitors_do_not_receive_study_labels():
    case = study.cases()[1]
    req = study.request(case, "buyer-a", "Actual prior reply")
    data = json.loads(req["input"][0]["content"])
    assert data["work_order"] == "Actual prior reply"
    assert "buyer-b" not in req["input"][0]["content"]
    encoded = json.dumps([study.request(case, "planner"), req, packet(case), study.guard_policies()])
    for forbidden in (case["id"], case["setting"], "expected_label", "fixture_expectation", "bn_score"):
        assert forbidden not in encoded
    assert req["model"] == "gpt-5.5" and req["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("mutation", ["amount", "sku", "schema", "currency", "fence"])
def test_invalid_model_command_not_repaired(mutation):
    case = study.cases()[0]
    obj = json.loads(proposal(case, "buyer-a"))
    if mutation == "amount": obj["amount"] = "1"
    elif mutation == "sku": obj["sku"] = "invented"
    elif mutation == "schema": obj["safe"] = True
    elif mutation == "currency": obj["currency"] = "EUR"
    text = json.dumps(obj)
    if mutation == "fence": text = "```json\n" + text + "\n```"
    with pytest.raises(ValueError):
        study.parse_proposal(case, "buyer-a", text)


def test_abstention_and_incompatible_purchase_are_not_silently_dropped():
    case = study.cases()[0]
    abstain = '{"decision":"abstain","sku":null,"amount":null,"currency":"USD","note":"Unable to proceed"}'
    assert study.parse_proposal(case, "buyer-a", abstain) is None
    obj = json.loads(proposal(case, "buyer-a"))
    obj.update(sku="a-incompatible", amount="15")
    assert study.parse_proposal(case, "buyer-a", json.dumps(obj))["arguments"]["sku"] == "a-incompatible"
    assert study.score_bn(study.packet(case, []))["allowed"]


def test_policy_book_is_shared_and_author_compatible():
    book = study.guard_policies()
    assert validate_policy_book(book) == book
    assert len(book) == 3
    events = study.guard_events("planner", {"buyer-a": "a", "buyer-b": "b"}, packet(study.cases()[0]))
    assert len(events) == 4 and events[-1]["tool"] == study.TOOL
    assert "current_purchase_batch" not in json.dumps(events)


def test_scope_mismatched_decoy_budget_does_not_change_decision():
    value = packet(study.cases()[0])
    extra = copy.deepcopy(value["contracts"][0])
    extra.update(id="unrelated", task_id="other-task", prior_requested="999", limit="1")
    value["contracts"].append(extra)
    assert study.reference_verdict(value)["allowed"]
    assert study.score_bn(value)["allowed"]


def test_standalone_snapshot_includes_transitive_configuration(tmp_path, monkeypatch):
    import run_mas_purchase_study as runner
    # No source clone and no API call: qualify the real numerical subprocess
    # against a temporary snapshot, including __init__'s transitive registry.
    monkeypatch.setattr(runner, "inspect_checkout", lambda path: {"commit": "unit-test-only", "files_sha256": {}})
    destination = tmp_path / "frozen"
    runner.prepare(destination, tmp_path / "unused-checkout", tmp_path / "unused-deps", Path(sys.executable))
    plan = json.loads((destination / "PLAN.json").read_text(encoding="utf-8"))
    check = json.loads((destination / "NUMERICAL_IMPORT_PREFLIGHT.json").read_text(encoding="utf-8"))
    assert check["exit_code"] == 0 and check["stdout"].strip() == "12" and check["new_api_calls"] == 0
    assert 'knowledge/templates/risk_templates_v1.json' in plan["sources_sha256"]
    assert 'knowledge/templates/workflow_risk_fragments_v1.json' in plan["sources_sha256"]
    assert plan["reuse_target_collection"] is None
