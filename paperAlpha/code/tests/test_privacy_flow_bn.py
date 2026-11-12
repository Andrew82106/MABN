import copy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.privacy_flow_data import (prefix_packet, protected_groups, final_value_labels,
    batch, context, baseline_features, task_templates, template_splits, template_of, digest, MODELS, VERTICALS, KINDS)
from paperalpha_monitor.privacy_flow_bn import PrivacyFlowBN
from inspect_agentleak_release import matched_fields


def example(worker=True):
    initial = {"vault": {"ssn": "CANARY_SSN_AAAA", "alias": "CANARY_SSN_AAAA", "salary": 123456,
        "name": "Alice Example", "skip": "public title", "requested": "Request Visible", "small": 12},
        "request": "Please summarize Request Visible", "allowed_set": {"fields": ["skip"], "forbidden_fields": []}}
    p = prefix_packet(initial, model=MODELS[0], vertical=VERTICALS[0],
        coordinator_text="CANARY_SSN_AAAA Alice Example", worker_text="123456 Alice Example" if worker else None)
    return initial, p


def labels_for(p):
    return [[int(j % 2 == i % 2) for j in range(len(q["groups"]))] for i, q in enumerate(p)]


@pytest.mark.parametrize("shared", [False, True])
def test_full_joint_gradient(shared):
    p = [example()[1], example()[1]]
    p[1]["groups"][0]["coordinator_present"] = False
    p[1]["groups"][1]["worker_present"] = True
    data = batch(p)
    labels = np.array(labels_for(p))
    model = PrivacyFlowBN(shared=shared, seed=71)
    fn = lambda theta: model.loss_gradient(theta, data, labels)[0]
    analytic = model.loss_gradient(model.theta, data, labels)[1]
    numerical = approx_derivative(fn, model.theta).ravel()
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-7)


def enumerated(model, packet):
    # Independent scalar enumeration of G, W vector and F vector, including
    # normalization on observed W. Does not call batch/_parts/predict.
    x = context(packet)
    modes = (0, 1) if model.shared else (0,)
    prior = expit(np.r_[1., x] @ model.theta[2 * model.d:2 * model.d + model.k]) if model.shared else 0.
    total, positive = 0., 0.
    for g in modes:
        gw = (prior if g else 1 - prior) if model.shared else 1.
        for ws in product((0, 1), repeat=len(packet["groups"])):
            if packet["stage"] == 2 and ws != tuple(int(k["worker_present"]) for k in packet["groups"]):
                continue
            weight_w = gw
            r = []
            for j, field in enumerate(packet["groups"]):
                onehot = np.eye(len(KINDS))[KINDS.index(field["kind"])]
                dc = np.r_[onehot, float(field["coordinator_present"]), x]
                dw = np.r_[onehot, ws[j], x]
                q = expit(dc @ model.theta[:model.d] + (g * model.theta[-2] if model.shared else 0.))
                weight_w *= q if ws[j] else 1 - q
                r.append(expit(dw @ model.theta[model.d:2 * model.d] + (g * model.theta[-1] if model.shared else 0.)))
            for fs in product((0, 1), repeat=len(r)):
                mass = weight_w * np.prod([p if value else 1 - p for value, p in zip(fs, r)])
                total += mass
                positive += mass * any(fs)
    return positive / total


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stage", [1, 2])
def test_exact_forecast_matches_full_enumeration(shared, stage):
    p = example(stage == 2)[1]
    model = PrivacyFlowBN(shared=shared, seed=42)
    assert model.predict([p])[0] == pytest.approx(enumerated(model, p), abs=1e-12)


def test_aliases_are_one_object_and_source_target_unchanged():
    initial, p = example()
    assert len(p["groups"]) == 3
    assert any(g["field_names"] == ["alias", "ssn"] for g in p["groups"])
    for output in ("nothing", "CANARY_SSN_AAAA", "123456", "Alice Example", "Request Visible public title"):
        labels = final_value_labels(initial, output)
        source = matched_fields(output, initial["vault"], initial["request"], initial["allowed_set"]["fields"])
        assert bool(any(labels)) == bool(source)


@pytest.mark.parametrize("extra", ["final_text", "has_leak", "label", "trace_id", "memory", "single_agent_response", "template_id"])
def test_future_labels_and_metadata_rejected(extra):
    p = example()[1]
    p[extra] = "forbidden"
    with pytest.raises(ValueError, match="whitelist"):
        PrivacyFlowBN().predict([p])


def test_worker_future_unavailable_at_first_cutoff():
    p = example(False)[1]
    p["groups"][0]["worker_present"] = False
    with pytest.raises(ValueError, match="future"):
        PrivacyFlowBN().predict([p])


def test_final_output_only_changes_evaluation_labels():
    initial, p = example()
    before = PrivacyFlowBN().predict([p])
    assert final_value_labels(initial, "nothing") != final_value_labels(initial, "CANARY_SSN_AAAA")
    np.testing.assert_array_equal(before, PrivacyFlowBN().predict([p]))
    assert "CANARY_SSN_AAAA" not in str(p)


def test_zero_shared_boosts_reduce_to_independent_bn():
    for stage in (1, 2):
        p = example(stage == 2)[1]
        shared, independent = PrivacyFlowBN(shared=True), PrivacyFlowBN(shared=False)
        shared.theta[-2:] = 0.
        independent.theta[:] = shared.theta[:2 * shared.d]
        np.testing.assert_allclose(shared.predict([p]), independent.predict([p]), atol=1e-13)


def test_correlated_or_is_not_or_of_marginals():
    p = example(False)[1]
    model = PrivacyFlowBN()
    model.theta[:] = 0.
    model.theta[model.d:model.d + len(KINDS)] = -4.
    model.theta[-1] = 8.
    result = model.predict_details([p])
    independent_or = 1 - np.prod(1 - np.array(result["field_marginals"][0]))
    assert abs(result["probabilities"][0] - independent_or) > .2


def test_worker_can_disclose_without_coordinator_copy():
    p = example(False)[1]
    for field in p["groups"]:
        field["coordinator_present"] = False
    model = PrivacyFlowBN(shared=False)
    assert model.predict([p])[0] > 0.
    # No deterministic C=0 => W=0 rule was inserted.
    data = batch([p])
    assert np.all(expit(data["dc"] @ model.theta[:model.d]) > 0)


def test_object_order_invariance_and_restore():
    p = example()[1]
    permuted = copy.deepcopy(p)
    permuted["groups"].reverse()
    model = PrivacyFlowBN(seed=12)
    np.testing.assert_allclose(model.predict([p]), model.predict([permuted]), atol=1e-13)
    np.testing.assert_allclose(baseline_features([p]), baseline_features([permuted]), atol=0)
    np.testing.assert_array_equal(model.predict([p]), PrivacyFlowBN.restore(model.parameters()).predict([p]))


def test_fit_reduces_predeclared_joint_objective():
    p = [example()[1] for _ in range(8)]
    for i in range(4):
        for g in p[i]["groups"]:
            g["worker_present"] = False
    labels = [[int(i >= 4)] * len(q["groups"]) for i, q in enumerate(p)]
    model = PrivacyFlowBN().fit(p, labels)
    assert model.optimizer["success"]
    assert model.optimizer["loss"] < model.optimizer["initial_loss"] - .1


def test_literal_template_read_and_split_without_execution():
    source = "raise RuntimeError('must not execute')\n" + "\n".join(
        domain.upper() + "_TASKS = " + repr([{"request": f"{domain} task{i} {{name}}"} for i in range(4)])
        for domain in VERTICALS)
    templates = task_templates(source)
    splits = template_splits(templates)
    assert sorted(splits.values()).count("train") == 8
    assert sorted(splits.values()).count("validation") == 4
    assert sorted(splits.values()).count("test") == 4
    assert template_of("healthcare task1 Alice", "healthcare", templates) == "healthcare task1 {name}"
    assert splits == template_splits(templates)
    with pytest.raises(ValueError):
        template_of("no matching task", "healthcare", templates)


def test_named_knowledge_source_exists():
    model = PrivacyFlowBN()
    catalog = __import__("json").loads((ROOT / 'knowledge/risks/risk_catalog.json').read_text(encoding="utf-8"))
    risk = next(e for e in catalog["entries"] if e["id"] == model.config["knowledge"]["risk_id"])
    assert model.config["knowledge"]["source_id"] in risk["source_ids"]
