import copy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.message_hurdle import (hurdle_kernel, MessageHurdleLaw, MessageHurdleWorkflowRiskBN,
    compile_message_training, message_objective)
from paperalpha_monitor.agentleak_event_projection import project, later_labels
from scalar_binary_graph import evaluate


def fixture(n=2, late=False):
    values = [f"protected-canary-{j:02d}" for j in range(n)]
    initial = {"vault":{f"field-{j}":s for j,s in enumerate(values)},"request":"Write a public summary", "allowed_set":{"fields":[]}}
    c,w,f = " ".join(values[::2]) or "none", " ".join(values[1:]) or "none", values[-1]
    args={"coordinator_text":c,"coordinator_record_id":"actual-coordinator-message"}
    if late: args.update(worker_text=w,worker_record_id="actual-worker-message")
    sample=project(initial,**args)
    labels=later_labels(initial,worker_text=w,worker_record_id="actual-worker-message",final_text=f,final_record_id="actual-final-message")
    if late: labels={k:v for k,v in labels.items() if k.endswith(":publish")}
    return sample,labels


@pytest.mark.parametrize("n", [1,2,5])
def test_kernel_normalization_observable_gate_and_derivatives(n):
    bits=np.asarray(list(product((0.,1.), repeat=n)))
    z=np.linspace(-1.3,.8,n); g=-.7
    logp,dz,dg=hurdle_kernel(np.tile(z,(len(bits),1)),np.full(len(bits),g),bits)
    p=np.exp(logp)
    assert p.sum()==pytest.approx(1.,abs=1e-13)
    assert p[1:].sum()==pytest.approx(expit(g),abs=1e-13)
    assert np.max(np.abs(p@dz))<1e-13 and abs(p@dg)<1e-13
    for k in range(n+1):
        zp,zm=z.copy(),z.copy(); gp,gm=g,g
        if k<n: zp[k]+=1e-5; zm[k]-=1e-5
        else: gp+=1e-5; gm-=1e-5
        plus=hurdle_kernel(np.tile(zp,(len(bits),1)),np.full(len(bits),gp),bits)[0]
        minus=hurdle_kernel(np.tile(zm,(len(bits),1)),np.full(len(bits),gm),bits)[0]
        assert np.allclose((plus-minus)/2e-5,dz[:,k] if k<n else dg,atol=1e-9)
    reversed_mass=hurdle_kernel(np.tile(z[::-1],(len(bits),1)),np.full(len(bits),g),bits[:,::-1])[0]
    assert np.allclose(logp,reversed_mass,atol=1e-13)


@pytest.mark.parametrize("n,late", [(1,False),(2,False),(3,False),(3,True)])
def test_group_ve_matches_explicit_bn_and_gradients(n,late):
    sample,labels=fixture(n,late)
    model=MessageHurdleWorkflowRiskBN()
    model.theta[:model.event_count]=np.random.default_rng(43).normal(0,.3,model.event_count)
    prediction=model.predict(sample,derivatives=True)
    fast=model.predict_known_authorization_union(sample,derivatives=True)
    assert prediction["probabilities"]["system:any"]==pytest.approx(fast["probabilities"]["system:any"],abs=1e-12)
    assert np.allclose(prediction["gradients"]["system:any"],fast["gradients"]["system:any"],atol=1e-12)
    graph=model.probability_graph(sample,output="system:any")
    assert evaluate(graph,max_free_nodes=18)==pytest.approx(fast["probabilities"]["system:any"],abs=1e-12)
    assert any(n.get("event_semantics")=="any_named_literal_in_this_message" for n in graph["nodes"])
    targets=[q["event_id"] for q in sample.queries.values()]
    joint=model.event_law.conditional(sample.event_dag,targets,model.theta[:45])
    assert np.exp(joint["log_probabilities"]).sum()==pytest.approx(1.,abs=1e-12)
    for k in (5,8,15,18,29,30,31,32,37,38,39,40):
        old=model.theta[k]
        model.theta[k]=old+1e-5; plus=model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
        model.theta[k]=old-1e-5; minus=model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
        model.theta[k]=old
        assert (plus-minus)/2e-5==pytest.approx(fast["gradients"]["system:any"][k],abs=2e-9)
    restored=MessageHurdleWorkflowRiskBN.restore(model.parameters())
    assert restored.predict(sample)["probabilities"]==pytest.approx(prediction["probabilities"])


def test_complete_message_objective_matches_marginal_likelihood_and_fit():
    sample,labels=fixture(3)
    model=MessageHurdleWorkflowRiskBN()
    theta=np.random.default_rng(3).normal(0,.2,45)
    compiled=compile_message_training(model.event_law,[sample.event_dag],[labels])
    loss,grad=message_objective(theta,compiled,.001)
    direct=model.event_law.label_log_likelihood(sample.event_dag,{k:v["value"] for k,v in labels.items()},theta)
    assert loss==pytest.approx(-direct["log_probability"]+.001*(theta@theta),abs=1e-12)
    assert np.allclose(grad,-direct["score"]+.002*theta,atol=1e-12)
    for k in range(45):
        plus,minus=theta.copy(),theta.copy(); plus[k]+=1e-5; minus[k]-=1e-5
        assert (message_objective(plus,compiled,.001)[0]-message_objective(minus,compiled,.001)[0])/2e-5==pytest.approx(grad[k],abs=1e-8)
    model.fit_messages([sample.event_dag],[labels],max_iter=200)
    assert model.optimizer["success"] and model.optimizer["message_count"]==2
    assert model.optimizer["label_count"]==6
    broken=copy.deepcopy(labels); broken["value-0:publish"]["record_id"]="different-actual-message"
    with pytest.raises(ValueError,match="same actual message"):
        compile_message_training(model.event_law,[sample.event_dag],[broken])


def test_13_field_union_keeps_all_events_without_independence_fallback():
    sample,_=fixture(13)
    model=MessageHurdleWorkflowRiskBN()
    result=model.predict_known_authorization_union(sample)
    # At zero coefficients the final observable OR gate is exactly .5,
    # regardless of worker output; no need to enumerate a 26-variable factor.
    assert result["probabilities"]["system:any"]==pytest.approx(.5,abs=1e-11)
    assert result["inference"]["labelled"]["peak_factor_variables"]==13
    with pytest.raises(ValueError,match="factor width"):
        model.predict(sample)


def test_semantic_spec_and_source_changes_rejected():
    sample,_=fixture(2,True)
    sample.event_dag.observations["value-0:work"]["record_id"]="other-message"
    # Reconstructing detects whole-message inconsistency independently of a hash.
    from paperalpha_monitor.observable_event_dag import ObservableEventDAG
    rebuilt=ObservableEventDAG(**sample.event_dag.specification())
    with pytest.raises(ValueError,match="one actual completed message"):
        MessageHurdleLaw.groups(rebuilt)
    saved=MessageHurdleWorkflowRiskBN().parameters()
    saved["event_law"]["message_hurdle"]["gate_semantics"]="hidden intent"
    with pytest.raises(ValueError,match="message semantics"):
        MessageHurdleWorkflowRiskBN.restore(saved)


def test_summary_matches_joint_and_independent_training_rejected():
    sample,labels=fixture(3,True)
    model=MessageHurdleWorkflowRiskBN()
    model.theta[:45]=np.random.default_rng(6).normal(0,.3,45)
    members=[q["event_id"] for q in sample.queries.values()]
    summary=model.event_law.message_summary(sample.event_dag,members,model.theta[:45])
    full=model.predict(sample)
    assert summary["any_appearance_probability"]==pytest.approx(full["probabilities"]["system:any"],abs=1e-12)
    assert list(summary["appearance_probabilities"].values())==pytest.approx(list(full["appearance_probabilities"].values()),abs=1e-12)
    from paperalpha_monitor.observable_event_learning import complete_trace_design, fit_event_traces
    with pytest.raises(ValueError,match="joint message"):
        complete_trace_design(model.event_law,[sample.event_dag],[labels])
    with pytest.raises(ValueError,match="joint message"):
        fit_event_traces([sample.event_dag],[labels],model_class=MessageHurdleWorkflowRiskBN)
