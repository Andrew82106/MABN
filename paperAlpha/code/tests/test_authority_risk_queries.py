from pathlib import Path
import sys

import pytest
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/tests')]
from paperalpha_monitor.authority_risk_queries import query_reference_risk, qualify_two_action_endpoint
from paperalpha_monitor.shared_authority_graph import AuthorityProofGraph, AuthorityWindow
from test_shared_authority_bn import bundle


def test_shared_query_has_explicit_observed_scope_and_nonstatistical_bounds():
    graph = bundle(True).graph
    result = query_reference_risk(graph, [logit(.2), 0., 0.], at_least=2)
    assert result["model_probability"] == pytest.approx(.2)
    assert (result["logical_lower"], result["logical_upper"]) == (0, 1)
    assert result["prior_dependent"] and not result["evidence_decides_target"]
    assert result["target"]["action_count"] == 2 and not result["future_forecast"]


def test_insufficient_observed_actions_cannot_have_two_failures_but_not_safe():
    original = bundle(True).graph
    graph = AuthorityProofGraph(original.window)
    from paperalpha_monitor.delegation_evidence import DelegationSnapshot
    proof = next(iter(original.proofs.values()))
    e = proof.evidence
    graph.add(DelegationSnapshot(proof.sequence, proof.action_id, proof.binding_sha256,
        tuple(e["features"][k] for k in e["feature_names"]), e))
    row = query_reference_risk(graph, [logit(.2), 0., 0.], at_least=2)
    assert row["model_probability"] == 0 and row["logical_upper"] == 0
    assert not row["proves_system_safety"] and row["uncertain_fact_count"] == 1
    assert query_reference_risk(graph, [logit(.2), 0., 0.])["model_probability"] == pytest.approx(.2)


def test_empty_observed_prefix_is_not_a_safe_task_label():
    result = query_reference_risk(AuthorityProofGraph(AuthorityWindow("empty", 0, 100)), [0., 0., 0.])
    assert result["model_probability"] == 0 and result["target"]["action_count"] == 0
    assert not result["proves_system_safety"] and not result["future_forecast"]


def test_known_evidence_decides_the_reference_query_without_prior_calibration():
    row = query_reference_risk(bundle(True, complete=True).graph, [8., 8., 8.])
    assert row["evidence_decides_target"] and not row["prior_dependent"]
    assert row["model_probability"] == 0 and not row["calibrated"]


@pytest.mark.parametrize("count,completed,accepted", [(0, True, False), (1, True, False), (2, False, False),
    (2, True, True), (3, True, False)])
def test_both_learned_heads_are_endpoint_scoped_without_changing_raw_scores(count, completed, accepted):
    row = qualify_two_action_endpoint(count, .47, .52, run_completed=completed)
    assert row["applicable"] is accepted
    assert row["raw_diagnostic_scores"] == {"system_bn": .47, "same_information_logistic": .52}
    assert (row["scores"] is not None) is accepted
    assert not row["future_forecast"] and not row["calibrated"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_raw_scores_cannot_pass_gate(value):
    with pytest.raises(ValueError):
        qualify_two_action_endpoint(2, value, .5, run_completed=True)
