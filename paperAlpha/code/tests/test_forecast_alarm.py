from pathlib import Path
import copy
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'code/src'))
from paperalpha_monitor.forecast_alarm import (ForecastAlarmStream,calibrate_forecast,fixed_policy,replay_forecast,forecast_alarm_metrics)
from paperalpha_monitor.alarm_policy import ALL_NEGATIVE


def episode(i,values=(.2,.8),phases=("early","late")):
    return {"episode_id":str(i),"model_fingerprint":"fixed-model","target_id":"fixed-final-risk",
        "prefixes":[{"phase":phase,"sequence":3+2*j,"probability":float(value),"input_sha256":"a"*64} for j,(phase,value) in enumerate(zip(phases,values))]}


def policy(threshold=.5):
    return fixed_policy(["early","late"],dict.fromkeys(["early","late"],threshold),"fixed-model","fixed-final-risk")


def test_first_alert_is_latched_and_prefix_only():
    e=episode(0,(.7,.1)); p=policy(); stream=ForecastAlarmStream(p,"0")
    p["thresholds"]["early"]=.99
    result=replay_forecast(e,policy())
    assert result[0]["alert_issued"] and not result[1]["alert_issued"] and result[1]["ever_alerted"]
    assert result[1]["first_alert"]["sequence"]==3
    for j,prefix in enumerate(e["prefixes"]):
        assert stream.push({**prefix,**{k:e[k] for k in ("episode_id","model_fingerprint","target_id")}})==result[j]
    assert replay_forecast({**e,"prefixes":e["prefixes"][:1]},policy())==result[:1]


@pytest.mark.parametrize("change",[{"target_id":"different"},{"model_fingerprint":"refitted"},{"episode_id":"other"},
    {"probability":float("nan")},{"input_sha256":"bad"},{"y":1},{"phase":"future"},{"sequence":-1}])
def test_invalid_record_does_not_advance_stream(change):
    e=episode(0); stream=ForecastAlarmStream(policy(),"0")
    good={**e["prefixes"][0],**{k:e[k] for k in ("episode_id","model_fingerprint","target_id")}}
    with pytest.raises(ValueError): stream.push({**good,**change})
    assert stream.push(good)["sequence"]==3


def test_time_and_phase_reversal_rejected():
    e=episode(0); stream=ForecastAlarmStream(policy(),"0")
    base={k:e[k] for k in ("episode_id","model_fingerprint","target_id")}
    stream.push({**base,**e["prefixes"][1]})
    with pytest.raises(ValueError): stream.push({**base,**e["prefixes"][1]})
    with pytest.raises(ValueError): stream.push({**base,**e["prefixes"][0],"sequence":7})


@pytest.mark.parametrize("mode",["episode_max","fixed_phase_budget"])
def test_calibration_counts_episodes_not_prefixes_and_handles_ties(mode):
    rng=np.random.default_rng(41)
    items=[episode(i,rng.random(2)) for i in range(101)]
    items[0]["prefixes"]=[]
    # Extra checks in one phase must not count as independent negative flows.
    items[1]=episode(1,[.95]*30,["early"]*30)
    kwargs={"phase_budgets":{"early":.05,"late":.05}} if mode=="fixed_phase_budget" else {}
    result=calibrate_forecast(items,phases=["early","late"],model_fingerprint="fixed-model",target_id="fixed-final-risk",total_budget=.1,mode=mode,**kwargs)
    metrics,_=forecast_alarm_metrics(items,{str(i):0 for i in range(101)},result["policy"])
    assert metrics["false_positive"]==result["actual_any_alarm_n"]<=10
    tied=calibrate_forecast([episode(i,(1,1)) for i in range(10)],phases=["early","late"],model_fingerprint="fixed-model",target_id="fixed-final-risk",total_budget=.1,mode=mode,**kwargs)
    assert all(t==ALL_NEGATIVE for t in tied["policy"]["thresholds"].values())


def test_missing_early_opportunities_remain_in_recall_denominator():
    rows=[episode(0,[.9],["late"]),episode(1,[.9,.1])]
    result,_=forecast_alarm_metrics(rows,{"0":1,"1":1},policy())
    assert result["first_phase_recall"]==.5 and result["any_alarm_recall"]==1
    assert result["wall_clock_lead_time"] is None
    with pytest.raises(ValueError): forecast_alarm_metrics(rows,{"0":1},policy())


def test_budget_overrun_and_duplicate_calibration_rejected():
    kwargs={"phases":["early","late"],"model_fingerprint":"fixed-model","target_id":"fixed-final-risk","total_budget":.1,"mode":"fixed_phase_budget","phase_budgets":{"early":.1,"late":.1}}
    with pytest.raises(ValueError): calibrate_forecast([episode(0)],**kwargs)
    kwargs["phase_budgets"]={"early":.05,"late":.05}
    with pytest.raises(ValueError): calibrate_forecast([episode(0),episode(0)],**kwargs)


def test_empty_episode_identity_and_extra_prefix_binding_rejected():
    e=episode(0); e["prefixes"]=[]; e["target_id"]="another-target"
    with pytest.raises(ValueError): replay_forecast(e,policy())
    e=episode(0); e["prefixes"][0]["model_fingerprint"]="another-model"
    with pytest.raises(ValueError): replay_forecast(e,policy())
