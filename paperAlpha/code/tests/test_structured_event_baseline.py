"""Constructed contract/equivalence tests, not security detection evidence."""
import copy

import numpy as np
import pytest

from test_message_workflow_compiler import specification,ev
from paperalpha_monitor.message_workflow_compiler import MessageWorkflowCompiler
from paperalpha_monitor.structured_event_baseline import StructuredEventBaseline,event_features,training_table,glm_objective
from paperalpha_monitor.observable_event_dag import ObservableEventLaw
from paperalpha_monitor.observable_event_learning import fit_event_traces,complete_trace_design
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from verify_matched_event_controls import reference_joint


def training_examples():
    dags=[]; labels=[]
    for i in range(10):
        d,_=MessageWorkflowCompiler().compile(**specification("fork" if i%2 else "chain"))
        outcomes={}
        for j,n in enumerate(k for k in d.order if k not in d.observations):
            value=int((i+j)%3==0)
            outcomes[n]={**ev(f"later:{i}:{n}","EXACT-SECRET" if value else "removed",3+j),"kind":"complete_message","value":value}
        dags.append(d); labels.append(outcomes)
    return dags,labels


@pytest.mark.parametrize("penalty",[.001,.03])
def test_same_training_information_glm_objective_and_exact_joint(penalty):
    dags,labels=training_examples()
    baseline=StructuredEventBaseline(penalty=penalty).fit(dags,labels)
    model=fit_event_traces(dags,labels,regularization=penalty,model_class=EliminatingDAGWorkflowRiskBN)
    x,y,_=training_table(dags,labels); a,b=complete_trace_design(model.event_law,dags,labels)
    assert np.array_equal(x,a) and np.array_equal(y,b)
    assert np.max(abs(model.theta[:25]-baseline.coef))<1e-7
    assert baseline.fit_report["label_count"]==model.optimizer["label_count"]==40
    for d in dags[:2]:
        targets=["secret:review","secret:publish"]
        expected=np.exp(model.event_law.conditional(d,targets,model.theta[:25])["log_probabilities"])
        assert np.allclose(baseline.predict_joint(d,targets),expected,atol=1e-10)


def test_shared_future_parent_is_integrated_not_replaced_by_its_mean():
    spec=specification(); edges=[["source","review"],["review","left"],["review","right"],["left","publish"],["right","publish"]]
    spec["workflow"]["edges"]=edges
    spec["deliveries"]=[{"source":a,"target":b,"subjects":["secret"],"basis":"declared_delivery","evidence":ev(a+b)} for a,b in edges]
    d,_=MessageWorkflowCompiler().compile(**spec)
    model=StructuredEventBaseline(); model.coef=np.zeros(25); model.coef[5]=-2; model.coef[8]=4
    targets=["secret:left","secret:right"]
    exact=model.predict_joint(d,targets); mean=model.predict_joint(d,targets,inference="mean_field")
    assert exact[3]>.39 and mean[3]==pytest.approx(.25)
    for mode in ("exact","mean_field","same_marginals_independent"):
        expected=reference_joint(model.metadata(),d,targets,mode)
        assert np.allclose(model.predict_joint(d,targets,inference=mode),expected,atol=1e-12)


@pytest.mark.parametrize("kind",["glm","histgb"])
@pytest.mark.parametrize("relations",[True,False])
def test_classifier_control_roundtrip_and_independent_formula(kind,relations):
    dags,labels=training_examples(); model=StructuredEventBaseline(kind=kind,relations=relations).fit(dags,labels)
    restored=StructuredEventBaseline.restore(model.metadata(),tree=model.tree)
    for mode in ("exact","mean_field","same_marginals_independent"):
        d=dags[1]; targets=["secret:review","secret:publish"]
        actual=model.predict_joint(d,targets,inference=mode)
        assert np.array_equal(actual,restored.predict_joint(d,targets,inference=mode))
        assert np.allclose(actual,reference_joint(model.metadata(),d,targets,mode,model.tree),atol=1e-12)


def test_gradient_and_no_parent_control_do_not_change_labels():
    dags,labels=training_examples(); x,y,_=training_table(dags,labels); z,other,_=training_table(dags,labels,relations=False)
    assert np.array_equal(y,other) and not np.any(z[:,[5*j+k for j in range(5) for k in (3,4)]])
    theta=np.random.default_rng(44).normal(0,.3,25); gradient=glm_objective(theta,x,y,len(dags),.03)[1]
    for i in range(25):
        h=np.zeros(25); h[i]=1e-5
        numeric=(glm_objective(theta+h,x,y,len(dags),.03)[0]-glm_objective(theta-h,x,y,len(dags),.03)[0])/2e-5
        assert numeric==pytest.approx(gradient[i],abs=1e-8)


@pytest.mark.parametrize("mutation",["missing_label","false_label","future_evidence","missing_history","duplicate_target","observed_target","capacity"])
def test_incomplete_or_future_leaked_inputs_are_rejected(mutation):
    dags,labels=training_examples(); d=dags[0]; targets=["secret:publish"]
    model=StructuredEventBaseline(); model.coef=np.zeros(25)
    if mutation in {"missing_label","false_label","future_evidence"}:
        outcomes=copy.deepcopy(labels[0]); first=next(iter(outcomes))
        if mutation=="missing_label": outcomes.pop(first)
        elif mutation=="false_label": outcomes[first]["value"]=1-outcomes[first]["value"]
        else: outcomes[first]["sequence"]=d.prefix_sequence
        with pytest.raises(ValueError): model.fit([d],[outcomes])
    elif mutation=="missing_history":
        spec=specification(); spec["records"]["left"]={"kind":"missing","evidence":ev("lost","Missing response",2)}
        missing,_=MessageWorkflowCompiler().compile(**spec)
        with pytest.raises(ValueError): model.predict_joint(missing,targets)
    else:
        if mutation=="duplicate_target": targets*=2
        elif mutation=="observed_target": targets=["secret:source"]
        with pytest.raises(ValueError): model.predict_joint(d,targets,max_free=1 if mutation=="capacity" else 18)


def test_features_ignore_names_and_reject_hard_fractional_parent_states():
    e={"id":"A","family":"review","cues":[1,0]}
    assert np.array_equal(event_features(e,[1,0]),event_features({**e,"id":"unrelated"},[1,0]))
    assert np.array_equal(event_features(e,[1,0]),ObservableEventLaw().design(e,[1,0]))
    with pytest.raises(ValueError): event_features(e,[.2,.7])
    assert event_features(e,[.2,.7],soft=True)[14]==pytest.approx(.14)
