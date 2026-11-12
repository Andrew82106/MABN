"""New-cohort coverage, no-fit scoring, and actual forecast-before-call order."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import crossed_purchase as study
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input
from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN, attribute_rows
from collect_exclusive_purchase import read, write, sha
from collect_crossed_purchase import run_flow, MODEL_NAMES
from score_crossed_purchase import score
from verify_crossed_purchase import verify_flow, metrics


def models(tmp):
    out = tmp / "models"
    for name in MODEL_NAMES:
        model = AttributeChoiceWorkflowRiskBN(regularization=float(name.removesuffix(".json").rsplit("_", 1)[1]))
        model.theta[:10] = np.random.default_rng(337).normal(0, .3, 10)
        data = model.parameters() if name.startswith("attribute") else {"theta": model.theta[:10].tolist()}
        write(out / name, data)
    return out


def reply(text):
    return {"id": "unit-test-not-api", "model": "unit-test", "status": "completed", "incomplete_details": None,
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}


def test_predeclared_grid_breaks_old_price_warranty_confounding_without_labels():
    cases = study.cases(); assert len(cases) == 24 and len({c["id"] for c in cases}) == 24
    assert cases == study.cases()
    contrasts = []
    for case in cases:
        assert case["split"] == "prospective_transfer" and case["pools"][0]["limit"] == "110"
        assert all("basic" not in item["sku"] and "enhanced" not in item["sku"] for b in case["branches"] for item in b["catalogue"])
        for priority in ("economy", "balanced", "quality"):
            pre = study.prefix(case, json.dumps({"priority": priority, "work_order": "Constructed design check, not an API reply."}))
            s = compile_attribute_input(pre); x = np.asarray(study.independent_features(pre, s))
            assert np.array_equal(x, attribute_rows(s)[0])
            for g in s.choice_groups:
                ix = [s.action_ids.index(a) for a in g["action_ids"]]; contrasts.append(x[ix[1]] - x[ix[0]])
    assert np.linalg.matrix_rank(np.asarray(contrasts)) == 6
    assert all(sum(c["profile"] == p for c in cases) == 6 for p in study.PROFILES)


def test_frozen_scorer_has_eleven_methods_and_no_global_policy_in_choice(tmp_path):
    root = models(tmp_path); case = study.cases()[0]
    pre = study.prefix(case, json.dumps({"priority": "balanced", "work_order": "Constructed check"}))
    a = score(pre, root); other = copy.deepcopy(pre); other["contracts"][0]["limit"] = "170"
    b = score(other, root)
    assert len(a["methods"]) == 11 and a["numerical_audit"]["max_graph_error"] < 1e-12
    for name in a["methods"]:
        assert a["methods"][name]["slots"] == b["methods"][name]["slots"]
        assert b["methods"][name]["risk_probability"] == 0


@pytest.mark.parametrize("failure", [None, "planner", "buyer", "refusal", "incomplete"])
def test_forecast_committed_before_buyers_and_failed_flows_not_negative(tmp_path, failure):
    case = study.cases()[0]; root = models(tmp_path); directory = tmp_path / "runs" / case["id"]
    calls = []
    async def api(request):
        calls.append(request); payload = json.loads(request["input"][0]["content"])
        if "assignment" not in payload:
            if failure == "planner": raise RuntimeError("unit failure")
            return reply(json.dumps({"priority": "balanced", "work_order": "Choose appropriately."}))
        assert (directory / "FORECAST.json").exists()
        if failure == "buyer": raise RuntimeError("unit failure")
        if failure == "refusal": return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "unit-test refusal"}]}]}
        if failure == "incomplete": return {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}
        return reply(json.dumps({"decision": "propose", "sku": payload["assignment"]["catalogue"][0]["sku"], "note": "Unit test choice."}))
    async def forecast(out, d, plan): write(d / "FORECAST.json", score(read(d / "PREFIX.json")["value"], root))
    report = asyncio.run(run_flow(case, tmp_path, {}, api, forecast))
    reconstructed = verify_flow(case, tmp_path, sha(directory / "REPORT.json"))
    assert reconstructed["complete"] == (failure is None)
    assert report["forecast_made"] == (failure != "planner")
    if failure:
        assert "label" not in reconstructed and not (directory / "OUTCOME.json").exists()
    else:
        method = next(iter(reconstructed["forecast"]["methods"]))
        summary = metrics([reconstructed], method)
        assert summary["flows"] == 1 and summary["slots"] == 2
        changed = read(directory / "FORECAST.json"); changed["target"] = "changed"
        # Test immutability through a separate file update owned by this test.
        (directory / "FORECAST.json").write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="raw record/forecast changed"):
            verify_flow(case, tmp_path, sha(directory / "REPORT.json"))
