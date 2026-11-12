import copy

import numpy as np
import pytest

from test_branched_message_compiler import inputs
from paperalpha_monitor.branched_message_compiler import compile_branch_prefix
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.task_cue_message import SemanticArrivalWorkflowRiskBN
from run_compiled_multiworkflow_study import features,policy_truth,rule_joint,make_record,risk_metrics
from verify_compiled_multiworkflow_study import reference


@pytest.mark.parametrize("cls",[EliminatingDAGWorkflowRiskBN,SemanticArrivalWorkflowRiskBN])
def test_independent_joint_worlds_match_compiled_source_and_policy_graph(cls):
    branch,calls,publisher=inputs(); sample,_=compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    model=cls(); model.theta+=np.random.default_rng(12).normal(0,.3,len(model.theta))
    expected,_=reference(model.parameters(),sample)
    targets=[sample.queries[a]["event_id"] for a in sample.action_ids]
    actual=np.exp(model.event_law.conditional(sample.event_dag,targets,model.theta[:model.event_count])["log_probabilities"])
    assert np.allclose(expected,actual,atol=1e-12)
    assert expected@policy_truth(sample)==pytest.approx(model.predict(sample)["probabilities"]["system:any"],abs=1e-12)


def test_full_graph_features_and_capacity_do_not_use_outcomes():
    branch,calls,publisher=inputs(); sample,_=compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    table=policy_truth(sample); x=features(sample,table)
    assert len(x)==949 and np.isfinite(x).all()
    assert np.array_equal(x[-4:],table)
    assert features(sample,table).tolist()==features(copy.deepcopy(sample),table).tolist()
    with pytest.raises(ValueError): features(sample,table,capacity=5)


def test_descriptive_rule_and_scores_keep_policy_separate_from_appearance():
    branch,calls,publisher=inputs(current="valid"); sample,_=compile_branch_prefix(branch,calls=calls,publisher_request=publisher)
    assert policy_truth(sample)==[0,0,0,0]
    assert rule_joint(sample,"visible_support_only")[3]==1
    row={"id":"constructed","cohort":"branch","split":"validation","template_id":"constructed","samples":{"late":sample},"bits":[1,1],"risk":0}
    record=make_record(row,"late",[.1,.2,.3,.4],policy_truth(sample))
    assert record["risk_probability"]==0 and record["joint"][3]==.4
    metric=risk_metrics([record]); assert metric["recall_at_half"] is None and metric["brier"]==0
