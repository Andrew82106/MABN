import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.scoped_privacy_api import cases, issuance, authority_transition, stage_request, observation, outcome
from paperalpha_runtime.privacy_transfer import STAGES
from paperalpha_monitor.scoped_privacy_monitor import compile_observation, gated_distribution, ScopedPrivacyMonitor
from paperalpha_monitor.policy_gated_privacy import PolicyGatedPrivacyBN
from scalar_binary_graph import evaluate
from verify_scoped_privacy_api import same_json
from diagnose_scoped_privacy_coverage import information_witness
import paperalpha_runtime.scoped_privacy_api as scoped_runtime


def observed(case, view="full"):
    text = " ".join(f["literal"] for f in case["tracked_fields"])
    return observation(case, text, text, issuance(case) + authority_transition(case), view)


def test_design_and_no_treatment_leak_in_requests():
    cohort = cases()
    assert len(cohort) == 16 and len({c["template_id"] for c in cohort}) == 4
    for normal, stale in zip(cohort[::2], cohort[1::2]):
        for stage in STAGES:
            previous = None if stage == STAGES[0] else "actual earlier reply"
            assert stage_request(normal, stage, previous) == stage_request(stale, stage, previous)
        assert observed(normal, "root_updates_missing") == observed(stale, "root_updates_missing")


def test_publisher_can_read_bound_arguments_without_computing_hashes():
    case = cases()[0]
    payload = json.loads(stage_request(case, STAGES[2], "actual workpaper")["input"][0]["content"])
    grants = {r["payload"]["grant_id"]: r["payload"] for r in payload["cached_authority_records"]}
    for binding in payload["release_bindings"]:
        assert binding["arguments_sha256"] == grants[binding["grant_id"]]["scope"]["arguments_sha256"]
        assert binding["arguments"]["recipient"] == case["context"]["recipient"]


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_policy_truth_and_executable_graph(case):
    o = observed(case)
    sample = compile_observation(o)
    label = outcome(case, issuance(case) + authority_transition(case), o["worker_text"])
    expected = [0, 0] if case["treatment"] == "valid" else [1, int(case["layout"] == "shared")]
    assert label["authority_failure"] == expected and label["appearance"] == [1, 1]
    model = PolicyGatedPrivacyBN()
    assert model.predict(sample)["reference_failure_marginals"] == expected
    for k in (1, 2):
        assert model.predict(sample, at_least=k)["model_probability"] == pytest.approx(evaluate(model.probability_graph(sample, at_least=k)), abs=1e-12)
    # Authorized tracked values are present in the packet, not dropped as in
    # the old allowed_set extractor.
    assert len(sample.packet["groups"]) == 2 and all(g["worker_present"] for g in sample.packet["groups"])


def test_unknown_shared_prior_not_independent_product():
    shared = compile_observation(observed(cases()[0], "root_updates_missing"))
    separate = compile_observation(observed(cases()[2], "root_updates_missing"))
    assert len(shared.graph.facts) == 1 and len(separate.graph.facts) == 2
    assert gated_distribution(shared, [.5, .5], .2) == pytest.approx([.85, .10, .05])
    assert gated_distribution(separate, [.5, .5], .2) == pytest.approx([.81, .18, .01])


@pytest.mark.parametrize("extra", ["final_text", "label", "treatment", "model", "case_id"])
def test_future_and_hidden_fields_rejected(extra):
    o = observed(cases()[0])
    o[extra] = "not observable"
    with pytest.raises(ValueError):
        compile_observation(o)


def test_future_record_and_duplicate_literal_rejected():
    o = observed(cases()[0])
    future = copy.deepcopy(o["authority_records"][-1])
    future.update(sequence=13, event_id="future")
    future["payload"]["grant_id"] = "future-child"
    o["authority_records"].append(future)
    with pytest.raises(ValueError):
        compile_observation(o)
    o = observed(cases()[0])
    o["tracked_fields"][1]["literal"] = o["tracked_fields"][0]["literal"]
    with pytest.raises(ValueError):
        compile_observation(o)


def test_scope_binding_and_no_actual_appearance_no_violation():
    case = cases()[1]
    assert outcome(case, issuance(case) + authority_transition(case), "References withheld.")["count"] == 0
    o = observed(cases()[0])
    o["context"]["recipient"] = "unmatched-recipient"
    assert PolicyGatedPrivacyBN().predict(compile_observation(o))["reference_failure_marginals"] == [1, 1]


def test_frozen_monitor_same_gate_and_declared_untrained_scope():
    folder = ROOT / 'experiments/privacy_flow/runs/privacy_identity_free_fit_20260919'
    if not folder.exists():
        pytest.skip("optional locally frozen training bundle unavailable")
    monitor = ScopedPrivacyMonitor(folder)
    supported = monitor.score(observed(cases()[0]))
    for prior in ("0.1", "0.5", "0.9"):
        assert set(supported["results_by_assumed_prior"][prior]["scores"]["1"].values()) == {0.}
    assert supported["calibrated"] is False and not supported["joint_model_fitted_on_authority_api_data"]
    assert supported["authority_emission_coefficient"] == 0.
    stale = monitor.score(observed(cases()[1]))
    assert stale["results_by_assumed_prior"]["0.5"]["scores"]["1"]["current_presence_shared_gate"] == 1.
    assert all(np.isclose(sum(d), 1) for d in stale["results_by_assumed_prior"]["0.5"]["count_distributions"].values())


def test_cli_prior_only_packet_roundtrip():
    folder = ROOT / 'experiments/privacy_flow/runs/privacy_identity_free_fit_20260919'
    if not folder.exists():
        pytest.skip("optional locally frozen training bundle unavailable")
    data = json.dumps(observed(cases()[1])) + "\n"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / 'code/commands/monitor_scoped_privacy.py'), "--models", str(folder)],
        input=data, text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0 and not result.stderr.strip()
    rows = [json.loads(s) for s in result.stdout.splitlines()]
    assert rows[0]["ready"] and rows[1]["ok"]


def test_audit_json_normalization_does_not_hide_numerical_changes():
    assert same_json({"bound_fields": ("a", "b"), "risk": .123}, {"bound_fields": ["a", "b"], "risk": .123})
    assert not same_json({"risk": .123}, {"risk": .123 + 1e-15})


@pytest.mark.parametrize("index,changes", [(0, {"1": True, "2": True}), (2, {"1": True, "2": False})])
def test_hidden_authority_information_witness(index, changes):
    case = cases()[index]
    text = " ".join(f["literal"] for f in case["tracked_fields"])
    result = information_witness(case, text, text, text, scoped_runtime)
    assert result["missing_view_identical"] and result["publisher_request_identical"]
    assert result["targets_change"] == changes


def test_witness_does_not_force_a_violation_when_reference_withheld():
    case = cases()[0]
    result = information_witness(case, "internal refs", "internal refs", "References withheld", scoped_runtime)
    assert result["targets_change"] == {"1": False, "2": False}
