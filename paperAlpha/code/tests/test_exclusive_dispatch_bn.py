"""Categorical support, existing policy composition and runtime contracts."""
import copy
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]

from paperalpha_runtime.exclusive_purchase import cases,prefix,compile_input,parse_buyer,SimulatedDispatchSlots,outcome
from paperalpha_monitor.exclusive_dispatch_bn import ExclusiveDispatchInput,ExclusiveDispatchWorkflowRiskBN
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN,WorkflowRiskInput
from scalar_binary_graph import evaluate


def sample(priority="balanced",complete=True,revoked=False):
    case=cases()[0]; case["revoke_parent_after_proposals"]=revoked
    value=prefix(case,json.dumps({"priority":priority,"work_order":"Actual constructed instruction"}))
    return compile_input(value,authority_complete=complete)


def plain(s):
    return WorkflowRiskInput(workflow=s.workflow,authority=s.authority,proposals=s.proposals,contracts=s.contracts,
        combinations=s.combinations,cutoff=s.cutoff,horizon=s.horizon)


def test_exclusive_choice_support_and_budget_at_initial_parameters():
    s=sample(); model=ExclusiveDispatchWorkflowRiskBN(); result=model.predict(s)
    assert result["probabilities"]["channel:cumulative_budget"]==pytest.approx(1/9)
    states,logs,_=model._occurrence_worlds(s,model.theta)
    assert np.isfinite(logs).sum()==9
    for group in s.choice_groups:
        ix=[s.action_ids.index(a) for a in group["action_ids"]]
        assert np.exp(logs)[states[:,ix].sum(axis=1)>1].sum()==0
    with pytest.raises(ValueError,match="exclusive choices"): WorkflowRiskBN().predict(s)
    assert WorkflowRiskBN().predict(plain(s))["probabilities"]["system:any"]>result["probabilities"]["system:any"]


@pytest.mark.parametrize("complete,revoked",[(True,False),(True,True),(False,False)])
def test_every_policy_query_and_conditional_gradient_matches_binary_graph(complete,revoked):
    s=sample("quality",complete,revoked); model=ExclusiveDispatchWorkflowRiskBN()
    model.theta=np.random.default_rng(173).normal(0,.4,len(model.theta))
    for conditions in (None,{s.action_ids[1]:1},{s.action_ids[0]:0,s.action_ids[2]:0}):
        result=model.predict(s,derivatives=True,event_conditions=conditions)
        for name,p in result["probabilities"].items():
            assert evaluate(model.probability_graph(s,output=name,event_conditions=conditions))==pytest.approx(p,abs=1e-12)
        gradient=approx_derivative(lambda t:model.predict(s,theta=t,event_conditions=conditions)["probabilities"]["system:any"],model.theta).ravel()
        assert np.allclose(gradient,result["gradients"]["system:any"],atol=1e-8)


def test_impossible_conditions_and_labels_are_rejected_not_renormalized():
    s=sample(); model=ExclusiveDispatchWorkflowRiskBN()
    impossible={s.action_ids[0]:1,s.action_ids[1]:1}
    with pytest.raises(ValueError,match="zero model probability"): model.predict(s,event_conditions=impossible)
    with pytest.raises(ValueError,match="zero probability"): model.loss_gradient(model.theta,[s],[{"dispatch":impossible}])


def test_singleton_slots_reduce_to_old_bernoulli_law():
    s=sample(); groups=[{**copy.deepcopy(s.choice_groups[i//2]),"id":"single:"+a,"action_ids":[a]} for i,a in enumerate(s.action_ids)]
    one=ExclusiveDispatchInput.from_input(plain(s),groups)
    exclusive=ExclusiveDispatchWorkflowRiskBN(); old=WorkflowRiskBN()
    exclusive.theta=old.theta=np.random.default_rng(87).normal(0,.3,len(old.theta))
    assert exclusive.predict(one)["probabilities"]==pytest.approx(old.predict(plain(s))["probabilities"],abs=1e-12)


def test_slot_alternative_order_does_not_change_prediction():
    s=sample("quality",False); groups=copy.deepcopy(s.choice_groups)
    for g in groups: g["action_ids"].reverse()
    other=ExclusiveDispatchInput.from_input(plain(s),groups)
    model=ExclusiveDispatchWorkflowRiskBN(); model.theta=np.random.default_rng(41).normal(0,.4,len(model.theta))
    assert model.predict(other)["probabilities"]==pytest.approx(model.predict(s)["probabilities"],abs=1e-12)
    assert evaluate(model.probability_graph(other))==pytest.approx(model.predict(s)["probabilities"]["system:any"],abs=1e-12)


def test_full_and_partial_supervision_gradient_and_restore():
    samples=[sample(p,False) for p in ("economy","quality","balanced")]
    labels=[{"dispatch":{s.action_ids[i%2]:1,s.action_ids[2+i%2]:1}} for i,s in enumerate(samples)]
    model=ExclusiveDispatchWorkflowRiskBN(regularization=.02); theta=np.random.default_rng(91).normal(0,.2,len(model.theta))
    loss,grad=model.loss_gradient(theta,samples,labels)
    numeric=approx_derivative(lambda t:model.loss_gradient(t,samples,labels)[0],theta).ravel()
    assert np.isfinite(loss) and np.allclose(grad,numeric,atol=1e-8)
    model.fit(samples,labels,max_iter=150)
    restored=ExclusiveDispatchWorkflowRiskBN.restore(model.parameters())
    assert restored.predict(samples[0])==model.predict(samples[0])


@pytest.mark.parametrize("mutation",["duplicate","missing","cross_actor","unknown_rule","future","text_semantics","mutate_after_binding"])
def test_bad_runtime_contracts_rejected(mutation):
    s=sample(); groups=copy.deepcopy(s.choice_groups)
    if mutation=="mutate_after_binding":
        s.choice_groups[0]["evidence"]["text"]="changed"
        with pytest.raises(ValueError): s.validate()
        return
    if mutation=="duplicate": groups[1]["action_ids"].append(s.action_ids[0])
    elif mutation=="missing": groups[0]["action_ids"].pop()
    elif mutation=="cross_actor": groups=[{**groups[0],"action_ids":list(s.action_ids)}]
    elif mutation=="unknown_rule": groups[0]["rule"]="agent_promises_not_to"
    elif mutation=="future": groups[0]["evidence"]["sequence"]=s.cutoff+1
    elif mutation=="text_semantics": s.event_semantics[s.action_ids[0]]="literal_appearance"
    with pytest.raises(ValueError): ExclusiveDispatchInput.from_input(s,groups)


def test_actual_simulation_enforces_one_command_and_distinguishes_abstention():
    case=cases()[0]; engine=SimulatedDispatchSlots(case)
    first=parse_buyer(case,"buyer-a",json.dumps({"decision":"propose","sku":"a-enhanced","note":"Chosen"}))
    records=[engine.execute("buyer-a",first),engine.execute("buyer-b",None)]
    assert records[0]["kind"]=="simulated_dispatch" and records[1]["kind"]=="abstention"
    with pytest.raises(ValueError): engine.execute("buyer-a",first)
    result=outcome(case,records)
    assert sum(result["dispatch"].values())==1 and result["risk"]==0
    with pytest.raises(ValueError): parse_buyer(case,"buyer-a",json.dumps({"decision":"propose","sku":["a-basic","a-enhanced"],"note":"both"}))


def test_declared_amounts_and_planner_cue_must_match_source():
    value=prefix(cases()[0],json.dumps({"priority":"economy","work_order":"Keep costs low"}))
    changed=copy.deepcopy(value); changed["priority"]="quality"
    with pytest.raises(ValueError): compile_input(changed)
    changed=copy.deepcopy(value); changed["candidate_actions"][0]["arguments"]["amount"]="1"
    with pytest.raises(ValueError): compile_input(changed)


def test_experiment_scalar_reference_and_standalone_objective():
    from evaluate_exclusive_purchase import scalar_choice_worlds,standalone_softmax_objective
    train=[]
    for i,priority in enumerate(("economy","quality","balanced")):
        s=sample(priority); selected={s.action_ids[i%2],s.action_ids[2+i%2]}
        train.append({"sample":s,"label":{"dispatch":{a:int(a in selected) for a in s.action_ids}}})
    model=ExclusiveDispatchWorkflowRiskBN(regularization=.02); model.theta[:3]=[.2,-.3,.4]
    loss,gradient=model.loss_gradient(model.theta,[r["sample"] for r in train],[{"dispatch":r["label"]["dispatch"]} for r in train])
    expected,grad=standalone_softmax_objective(model.theta[:3],train,.02)
    assert loss==pytest.approx(expected,abs=1e-12) and np.allclose(gradient[:3],grad,atol=1e-12)
    for row in train:
        states,p=scalar_choice_worlds(row["sample"],model.theta[:3])
        actual,logs,_=model._occurrence_worlds(row["sample"],model.theta)
        assert np.array_equal(states,actual) and np.allclose(p,np.exp(logs),atol=1e-12)


def test_journal_sequence_orders_events_with_equal_timer_ticks():
    from evaluate_exclusive_purchase import validate_record_order
    assert validate_record_order([{"sequence":1,"monotonic_ns":100},{"sequence":2,"monotonic_ns":100},
                                  {"sequence":3,"monotonic_ns":200}])==1


@pytest.mark.parametrize("rows",[[{"sequence":1,"monotonic_ns":100},{"sequence":2,"monotonic_ns":99}],
                                 [{"sequence":1,"monotonic_ns":100},{"sequence":3,"monotonic_ns":101}]])
def test_journal_still_rejects_clock_reversal_or_missing_sequence(rows):
    from evaluate_exclusive_purchase import validate_record_order
    with pytest.raises(ValueError): validate_record_order(rows)
