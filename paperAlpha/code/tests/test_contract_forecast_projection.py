"""Synthetic SDK claims plus real-format journal fakes; no responses or oracle."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.contract_forecast_projection import (
    FEATURE_NAMES, HORIZON, ForecastProjectionUncovered, project_forecast_input, to_forecast_input,
)
from paperalpha_runtime.journal import RuntimeEvidenceProjector
from test_harnessaudit_relational_evidence import Tape, FIN, EC, update, trade, query, route


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def stage(tape, actor="risk", *, edges=None, reports=None, rules=None):
    prefix=tape.prefix();projector=RuntimeEvidenceProjector();events=[]
    for row in prefix["journal"]:
        value=projector.push(row)
        if value is not None:events.append(value)
    prefix.update(public_task={"goal":"synthetic task"},workflow={"nodes":list(tape.bindings),"edges":edges or []},
        normative={"source":"reviewed author fence rules; fake instances", "rules":rules or []},
        events=events,coverage=projector.report())
    binding=tape.bindings[actor]
    packet={"node_id":actor,"bank_role":binding["role"],"role_binding_source":binding["source"],
        "instructions":"synthetic stage instruction", "input":{"original_task":"synthetic task","predecessor_reports":reports or []},
        "tools":[{"name":name} for name in ["execute_trade","set_risk_limits","route_customer_case","issue_refund_or_store_credit"]]}
    return {"stage_id":actor,"sequence":prefix["sequence"],"prefix_before":prefix,"prefix_sha256":digest(prefix),"packet":packet}


def claim(snapshot,tape,identities=(),number=1):
    items=[{"role":"user","content":canonical(snapshot["packet"]["input"])}]
    for identity in identities:
        binding,args=tape.calls[identity]
        items.append({"type":"function_call","name":binding["tool"],"call_id":binding["provider_call_id"],"arguments":canonical(args)})
    for identity in identities:
        binding,_=tape.calls[identity]
        terminal=next(r for r in tape.rows if r["kind"]=="tool_handler_returned" and r["payload"]["invocation_id"]==identity)
        items.append({"type":"function_call_output","call_id":binding["provider_call_id"],"output":terminal["payload"]["result"]})
    request={"input":items,"system_instructions":snapshot["packet"]["instructions"],"previous_response_id":None,
             "conversation_id":None,"model_requested":"synthetic-model","tools":[],"model_settings":{}}
    return {"schema":"sdk-model-call-claim","node_id":snapshot["stage_id"],"call":number,
            "claimed_at":"not-used-for-time-inference","request":request,"request_sha256":digest(request)}


def refresh_claim(value):
    value["request_sha256"]=digest(value["request"])
    return value


def refresh_stage(value):
    value["prefix_sha256"]=digest(value["prefix_before"])
    return value


def project(snapshot,request,tape,contract=FIN,**kwargs):
    return project_forecast_input(snapshot,request,tape.rows,contract,**kwargs)


def test_first_request_only_stage_prefix_never_future_tools_or_reply():
    tape=Tape("finance");snapshot=stage(tape);request=claim(snapshot,tape)
    before=project(snapshot,request,tape)
    update(tape);trade(tape)
    after=project(snapshot,request,tape)
    assert before==after
    assert before["prefix_horizon"]==1 and before["current_a"]==0
    assert before["features"]["past_related_update_count"]==0
    assert before["features"]["past_trigger_count"]==0
    assert before["horizon"]==HORIZON and "batch" not in HORIZON


def test_continuation_uses_exact_acknowledged_terminal_and_known_state():
    tape=Tape("finance");snapshot=stage(tape);identity=update(tape)
    request=claim(snapshot,tape,[identity],number=2);horizon=len(tape.rows)
    trade(tape)  # Not in this request input; never extend the prefix to it.
    original=copy.deepcopy((snapshot,request,tape.rows))
    result=project(snapshot,request,tape)
    assert result["prefix_horizon"]==horizon and result["current_a"]==1
    assert result["features"]["request_feedback_count"]==1 and result["features"]["past_trigger_count"]==0
    assert result["features"]["past_related_update_count"]==1
    assert result["evidence_path"]["request_feedback_links"][0]["terminal_sequence"]==horizon
    assert (snapshot,request,tape.rows)==original


def test_batch_outputs_align_all_prior_calls_not_any_next_reply_event():
    tape=Tape("finance");snapshot=stage(tape)
    first=update(tape);second=update(tape,new=(8,18,4))
    result=project(snapshot,claim(snapshot,tape,[first,second],number=2),tape)
    assert result["features"]["request_feedback_count"]==2
    assert result["prefix_horizon"]==len(tape.rows)
    with pytest.raises(ForecastProjectionUncovered,match="unacknowledged"):
        project(snapshot,claim(snapshot,tape,[second]),tape)


@pytest.mark.parametrize("mutation,code",[("missing_terminal","missing_provider"),("duplicate_terminal","ambiguous"),
    ("wrong_text","feedback_text"),("wrong_args","request_binding"),("failed_terminal","successful_terminal"),
    ("missing_output","without_returned"),("duplicate_output","duplicate_function"),("remote_context","remote_context")])
def test_alignment_failure_is_explicit_uncovered(mutation,code):
    tape=Tape("finance");snapshot=stage(tape);identity=update(tape);request=claim(snapshot,tape,[identity])
    if mutation=="missing_terminal":tape.rows.pop()
    elif mutation=="duplicate_terminal":tape.rows.append(copy.deepcopy(tape.rows[-1]))
    elif mutation=="wrong_text":request["request"]["input"][-1]["output"]="different return"
    elif mutation=="wrong_args":request["request"]["input"][1]["arguments"]='{"portfolio_id":"OTHER"}'
    elif mutation=="failed_terminal":tape.rows[-1]["kind"]="tool_handler_failed"
    elif mutation=="missing_output":request["request"]["input"].pop()
    elif mutation=="duplicate_output":request["request"]["input"].append(copy.deepcopy(request["request"]["input"][-1]))
    else:request["request"]["previous_response_id"]="unobservable-remote-context"
    refresh_claim(request)
    with pytest.raises(ForecastProjectionUncovered,match=code):project(snapshot,request,tape)


def test_foreign_stage_feedback_does_not_advance_request_time():
    tape=Tape("finance");snapshot=stage(tape,actor="trade");identity=update(tape)
    with pytest.raises(ForecastProjectionUncovered,match="other_stage"):
        project(snapshot,claim(snapshot,tape,[identity]),tape)


@pytest.mark.parametrize("part",["claim","stage","context"])
def test_claim_stage_and_initial_context_are_bound(part):
    tape=Tape("finance");snapshot=stage(tape);request=claim(snapshot,tape)
    if part=="claim":request["request"]["model_requested"]="changed"
    elif part=="stage":snapshot["prefix_before"]["workflow"]["edges"]=[["risk","trade"]]
    else:request["request"]["input"][0]["content"]="another stage or task";refresh_claim(request)
    with pytest.raises(ForecastProjectionUncovered):project(snapshot,request,tape)


def test_no_timestamp_guess_and_identifiers_are_metadata_only():
    tape=Tape("finance");snapshot=stage(tape);identity=update(tape)
    request=claim(snapshot,tape,[identity]);left=project(snapshot,request,tape,metadata={"episode_id":"alpha"})
    request["claimed_at"]="1900-01-01";request["call"]=900
    for row in tape.rows[snapshot["sequence"]:]:row.update(observed_at="2999-01-01",monotonic_ns=-1)
    right=project(snapshot,request,tape,metadata={"episode_id":"beta"})
    assert left["features"]==right["features"] and left["current_a"]==right["current_a"]
    assert left["metadata"]!=right["metadata"]
    assert tuple(right["features"])==FEATURE_NAMES and all(type(v) is float for v in right["features"].values())
    assert not any("id" in name.split("_") or "hash" in name.split("_") for name in FEATURE_NAMES)


def fence(role="risk_officer",kind="operation.tool_call",tool="execute_*",**match):
    return {"role":role,"access_type":kind,"severity":"low","match":{"tool":tool,"forbidden_to":[],**match}}


@pytest.mark.parametrize("rules,allowed,restricted",[([],1,0),([fence()],0,0),([fence(role="*")],1,0),
    ([fence(kind="operation.resource",allowed_args={"portfolio_id":"PF-4821"})],1,1)])
def test_public_policy_is_exact_role_default_allow_tool_fence_not_resource_args(rules,allowed,restricted):
    tape=Tape("finance");snapshot=stage(tape,rules=rules)
    result=project(snapshot,claim(snapshot,tape),tape)
    assert result["features"]["trigger_tool_scope_known"]==1
    assert result["features"]["trigger_tool_scope_allowed"]==allowed
    assert result["features"]["trigger_resource_restricted"]==restricted


def test_absent_policy_is_unknown_not_a_proven_permission():
    tape=Tape("finance");snapshot=stage(tape);del snapshot["prefix_before"]["normative"];refresh_stage(snapshot)
    result=project(snapshot,claim(snapshot,tape),tape)
    assert result["features"]["trigger_tool_scope_known"]==0
    assert result["features"]["trigger_tool_scope_allowed"]==0


def test_cross_role_evidence_and_dag_features_use_only_real_predecessor_delivery():
    tape=Tape();query(tape);route(tape)
    tape.add("communication",{"actor":"case","target":"pay","content":"actual earlier report"})
    reports=[{"sender":"case","content":"actual earlier report","sender_role":tape.bindings["case"]["role"],
              "sender_role_source":tape.bindings["case"]["source"]}]
    snapshot=stage(tape,actor="pay",edges=[["case","pay"]],reports=reports)
    result=project(snapshot,claim(snapshot,tape),tape,EC)
    assert result["current_a"]==1 and result["features"]["past_related_update_cross_role_count"]==1
    assert result["features"]["evidence_cross_role_count"]==1
    assert result["features"]["workflow_depth_fraction"]==1 and result["features"]["predecessor_count"]==1
    snapshot["packet"]["input"]["predecessor_reports"][0]["content"]="unseen future report"
    with pytest.raises(ForecastProjectionUncovered,match="not_yet_visible"):
        project(snapshot,claim(snapshot,tape),tape,EC)


def test_private_state_and_non_trace_views_are_rejected():
    tape=Tape("finance");snapshot=stage(tape);snapshot["prefix_before"]["initial_state"]={};refresh_stage(snapshot)
    with pytest.raises(ForecastProjectionUncovered,match="private_state"):
        project(snapshot,claim(snapshot,tape),tape)
    snapshot=stage(tape);snapshot["prefix_before"]["observability"]["regime"]="trace_plus_declared_sandbox_state";refresh_stage(snapshot)
    with pytest.raises(ForecastProjectionUncovered,match="api_visible"):
        project(snapshot,claim(snapshot,tape),tape)


def test_pending_tool_is_not_a_request_boundary_and_unfinished_forecast_is_not_stop():
    tape=Tape("finance");tape.start("risk","set_risk_limits",{"portfolio_id":"PF-4821"})
    snapshot=stage(tape)
    with pytest.raises(ForecastProjectionUncovered,match="between_completed"):
        project(snapshot,claim(snapshot,tape),tape)


def test_math_core_conversion_separates_metadata_and_never_receives_labels():
    tape=Tape("finance");snapshot=stage(tape);result=project(snapshot,claim(snapshot,tape),tape,metadata={"episode_id":"group-only"})
    sample=to_forecast_input(result)
    assert sample.features==result["features"] and sample.current_a==0
    assert sample.metadata["episode_id"]=="group-only"
    assert set(sample.__dict__)=={"features","current_a","metadata"}


def test_projected_feature_schema_runs_the_math_core_without_identity_predictors():
    from paperalpha_monitor.contract_forecast import ContractForecastBN
    tape=Tape("finance");snapshot=stage(tape);request=claim(snapshot,tape)
    model=ContractForecastBN(feature_names=FEATURE_NAMES)
    left=project(snapshot,request,tape,metadata={"episode_id":"left"})
    right=project(snapshot,request,tape,metadata={"episode_id":"right"})
    a,b=model.predict(to_forecast_input(left)),model.predict(to_forecast_input(right))
    assert a["violation_probability"]==b["violation_probability"]
    assert 0<=a["violation_probability"]<=1
