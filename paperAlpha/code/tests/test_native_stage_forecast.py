import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/external/HarnessAudit'))
from multi_agent.recognizer import contains
from paperalpha_monitor.native_stage_forecast import NAMES, NativePolicy, NativeStageForecastBN, independent_same_marginals, public_features


def test_independent_distribution_is_normalized():
    joint = np.full((2, 16), 1 / 16)
    result = independent_same_marginals(joint)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_chain_distribution_is_normalized():
    model = NativeStageForecastBN()
    x = np.zeros((12, len(NAMES)))
    bits = np.zeros((12, 4), dtype=int)
    bits[:6, 0] = 1
    bits[::3, 1] = 1
    bits[::4, 2] = 1
    bits[::5, 3] = 1
    model.fit(x, bits, np.ones(len(x)))
    assert np.allclose(model.joint(x).sum(axis=1), 1.0)


def test_prefix_rejects_future_event():
    run = ROOT / 'experiments/model_mechanisms/runs/submission__api_recovery_confirm_20261001'
    plan = __import__("json").loads((run / "PLAN.json").read_text(encoding="utf-8"))
    case = plan["cases"][0]
    stage = __import__("json").loads(next((run / "cases" / case["case_id"] / "stages").glob("*.json")).read_text(encoding="utf-8"))
    policy = NativePolicy(case["visible_policy"]["rules"], case["domain"], contains)
    bad = copy.deepcopy(stage)
    bad["prefix_before"]["events"].append({"sequence": bad["prefix_before"]["sequence"] + 1, "event_type": "message_sent",
        "actor": case["workflow"]["nodes"][0], "target": case["workflow"]["nodes"][-1], "content": "future"})
    try:
        public_features(case, bad, policy)
    except ValueError as exc:
        assert "future event" in str(exc)
    else:
        raise AssertionError("future event was accepted")
