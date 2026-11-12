"""New message factors must retain existing shared-authority/budget semantics."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from test_observable_event_dag import coupled_tool_fixture
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN
from scalar_binary_graph import evaluate


@pytest.mark.parametrize("layout,authority",[("fork","root"),("fork","known"),("chain","delegation"),("chain","invalid")])
def test_joint_message_tool_and_shared_policy_composition(layout,authority):
    sample,old=coupled_tool_fixture(n=3,layout=layout,authority=authority)
    old.theta+=np.random.default_rng(7).normal(0,.2,len(old.theta))
    model=MessageHurdleWorkflowRiskBN()
    model.theta[:25]=old.theta[:25]; model.theta[-3:]=old.theta[-3:]
    # Singleton plan-message event equals its OR; match the original plan law.
    model.theta[25]=old.theta[0]
    conditions={sample.action_ids[0]:1}
    expected=old.predict(sample,event_conditions=conditions)
    result=model.predict(sample,event_conditions=conditions,derivatives=True)
    assert result["probabilities"]==pytest.approx(expected["probabilities"],abs=1e-12)
    for query,p in result["probabilities"].items():
        assert evaluate(model.probability_graph(sample,output=query,event_conditions=conditions))==pytest.approx(p,abs=1e-12)
    for i in range(len(model.theta)):
        old_value=model.theta[i]
        model.theta[i]=old_value+1e-5
        plus=model.predict(sample,event_conditions=conditions)["probabilities"]["system:any"]
        model.theta[i]=old_value-1e-5
        minus=model.predict(sample,event_conditions=conditions)["probabilities"]["system:any"]
        model.theta[i]=old_value
        assert (plus-minus)/2e-5==pytest.approx(result["gradients"]["system:any"][i],abs=1e-8)
