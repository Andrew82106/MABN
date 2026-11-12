"""Bound, first-alert runtime decisions for a fixed future-risk target.

Reuses the existing tie-safe empirical threshold calculation. Named phases
come from trusted runtime adapters, not Agent prose. No outcome labels enter
the streaming interface. Calibration controls its sample only, not deployment.
"""
from __future__ import annotations

import copy
import math

from paperalpha_runtime.canonical import digest
from .alarm_policy import ALL_NEGATIVE, calibrate_phase


def _text(value):
    if not isinstance(value,str) or not value:
        raise ValueError("nonempty target/model/episode/phase identifier required")
    return value


def _phases(phases):
    if not isinstance(phases,(list,tuple)) or not phases or len(set(phases))!=len(phases):
        raise ValueError("ordered unique phase schedule required")
    return tuple(_text(p) for p in phases)


def _probability(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError("finite probability in [0,1] required")
    return float(value)


def _validate_policy(policy):
    if set(policy)!={"schema","phases","thresholds","model_fingerprint","target_id","scope"} or policy["schema"]!="bound-forecast-alarm-v1":
        raise ValueError("explicit fixed-target forecast alarm policy required")
    phases=_phases(policy["phases"])
    _text(policy["model_fingerprint"]); _text(policy["target_id"])
    if set(policy["thresholds"])!=set(phases) or any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t)
        or not 0<=t<=ALL_NEGATIVE for t in policy["thresholds"].values()):
        raise ValueError("one finite threshold per phase required")
    return phases


def fixed_policy(phases,thresholds,model_fingerprint,target_id):
    policy={"schema":"bound-forecast-alarm-v1","phases":list(phases),"thresholds":dict(thresholds),
        "model_fingerprint":model_fingerprint,"target_id":target_id,
        "scope":"Fixed future target and phase schedule; empirical calibration is not a population false-alarm guarantee."}
    _validate_policy(policy)
    return policy


class ForecastAlarmStream:
    def __init__(self,policy,episode_id):
        self._phases=_validate_policy(policy)
        self._policy=copy.deepcopy(policy); self._policy_sha256=digest(policy)
        self._episode_id=_text(episode_id)
        self._last_sequence=-1; self._phase_index=-1; self._first=None

    def push(self,record):
        if not isinstance(record,dict) or set(record)!={"episode_id","phase","sequence","probability","input_sha256","model_fingerprint","target_id"}:
            raise ValueError("only prefix forecast fields allowed, never outcome labels")
        if (record["episode_id"]!=self._episode_id or record["target_id"]!=self._policy["target_id"]
            or record["model_fingerprint"]!=self._policy["model_fingerprint"]):
            raise ValueError("episode, future target and fitted model must stay fixed")
        if digest(self._policy)!=self._policy_sha256: raise ValueError("alarm policy changed")
        if record["phase"] not in self._phases: raise ValueError("phase not in frozen schedule")
        index=self._phases.index(record["phase"])
        sequence=record["sequence"]
        if type(sequence) is not int or sequence<=self._last_sequence or index<self._phase_index:
            raise ValueError("accepted prefix time and phase cannot go backwards or repeat sequence")
        probability=_probability(record["probability"])
        if not isinstance(record["input_sha256"],str) or len(record["input_sha256"])!=64 or any(c not in "0123456789abcdef" for c in record["input_sha256"]):
            raise ValueError("canonical bound prefix hash required")
        threshold=self._policy["thresholds"][record["phase"]]
        exceeded=probability>=threshold
        emitted=exceeded and self._first is None
        if emitted:
            self._first={"phase":record["phase"],"sequence":sequence,"input_sha256":record["input_sha256"],"probability":probability}
        self._phase_index=index; self._last_sequence=sequence
        return {"phase":record["phase"],"sequence":sequence,"probability":probability,"threshold":threshold,
            "threshold_exceeded":exceeded,"alert_issued":emitted,"ever_alerted":self._first is not None,
            "first_alert":copy.deepcopy(self._first),"input_sha256":record["input_sha256"],"policy_sha256":self._policy_sha256}


def replay_forecast(episode,policy):
    if set(episode)!={"episode_id","model_fingerprint","target_id","prefixes"} or not isinstance(episode["prefixes"],list):
        raise ValueError("label-free bound episode required")
    if episode["model_fingerprint"]!=policy["model_fingerprint"] or episode["target_id"]!=policy["target_id"]:
        raise ValueError("episode target/model differs, including empty episodes")
    if any(not isinstance(p,dict) or set(p)!={"phase","sequence","probability","input_sha256"} for p in episode["prefixes"]):
        raise ValueError("strict prefix-only score fields required")
    stream=ForecastAlarmStream(policy,episode["episode_id"])
    return [stream.push({**p,**{k:episode[k] for k in ("episode_id","model_fingerprint","target_id")}}) for p in episode["prefixes"]]


def calibrate_forecast(negative_episodes,*,phases,model_fingerprint,target_id,total_budget,mode,phase_budgets=None):
    """Use independent calibration-negative episodes; do not fit on evaluation."""
    phases=_phases(phases)
    if not negative_episodes or len({r["episode_id"] for r in negative_episodes})!=len(negative_episodes):
        raise ValueError("unique nonempty calibration episodes required")
    if not isinstance(total_budget,(int,float)) or not math.isfinite(total_budget) or not 0<=total_budget<=1:
        raise ValueError("total empirical budget in [0,1] required")
    validation=fixed_policy(phases,dict.fromkeys(phases,ALL_NEGATIVE),model_fingerprint,target_id)
    maxima=[]
    for episode in negative_episodes:
        replay_forecast(episode,validation)
        maxima.append({phase:max((p["probability"] for p in episode["prefixes"] if p["phase"]==phase),default=None) for phase in phases})
    if mode=="episode_max":
        if phase_budgets is not None: raise ValueError("episode-max has one budget, not phase budgets")
        selected=calibrate_phase([max((v for v in row.values() if v is not None),default=None) for row in maxima],total_budget)
        thresholds=dict.fromkeys(phases,selected["threshold"])
        details={"episode_max":selected}
    elif mode=="fixed_phase_budget":
        if not isinstance(phase_budgets,dict) or set(phase_budgets)!=set(phases) or any(not math.isfinite(x) or not 0<=x<=1 for x in phase_budgets.values()):
            raise ValueError("predeclared per-phase budgets required")
        if sum(phase_budgets.values())>total_budget+1e-15: raise ValueError("phase spending exceeds episode budget")
        details={phase:calibrate_phase([r[phase] for r in maxima],phase_budgets[phase]) for phase in phases}
        thresholds={p:details[p]["threshold"] for p in phases}
    else: raise ValueError("unknown fixed calibration rule")
    policy=fixed_policy(phases,thresholds,model_fingerprint,target_id)
    alarms=sum(any(v is not None and v>=thresholds[p] for p,v in row.items()) for row in maxima)
    if alarms>math.floor(total_budget*len(maxima)): raise AssertionError("calibration episode budget exceeded")
    return {"policy":policy,"mode":mode,"calibration_negative_ids":[r["episode_id"] for r in negative_episodes],
        "negative_episode_n":len(maxima),"actual_any_alarm_n":alarms,"empirical_any_alarm_fpr":alarms/len(maxima),
        "budget":total_budget,"details":details,"guarantee":"Calibration sample only; repeated prefixes are not independent examples."}


def forecast_alarm_metrics(episodes,labels,policy):
    """Offline outcome scoring, separate from label-free streaming decisions."""
    if not episodes or len({e["episode_id"] for e in episodes})!=len(episodes) or set(labels)!={e["episode_id"] for e in episodes}:
        raise ValueError("one offline outcome per unique episode required")
    if any(type(y) is not int or y not in (0,1) for y in labels.values()): raise ValueError("binary outcomes required")
    decisions=[replay_forecast(e,policy) for e in episodes]
    first=[next((d for d in ds if d["alert_issued"]),None) for ds in decisions]
    negative=[i for i,e in enumerate(episodes) if labels[e["episode_id"]]==0]
    positive=[i for i,e in enumerate(episodes) if labels[e["episode_id"]]==1]
    tp=sum(first[i] is not None for i in positive); fp=sum(first[i] is not None for i in negative)
    return {"positive_episode_n":len(positive),"negative_episode_n":len(negative),"true_positive":tp,"false_positive":fp,
        "any_alarm_recall":tp/len(positive) if positive else None,"any_alarm_fpr":fp/len(negative) if negative else None,
        "precision":tp/(tp+fp) if tp+fp else None,"f1":2*tp/(len(positive)+tp+fp) if len(positive)+tp+fp else None,
        "first_phase_recall":sum(first[i] is not None and first[i]["phase"]==policy["phases"][0] for i in positive)/len(positive) if positive else None,
        "first_alert_phase_counts":{p:sum(d is not None and d["phase"]==p for d in first) for p in policy["phases"]},
        "observed_phase_episode_counts":{p:sum(any(r["phase"]==p for r in e["prefixes"]) for e in episodes) for p in policy["phases"]},
        "wall_clock_lead_time":None}, decisions
