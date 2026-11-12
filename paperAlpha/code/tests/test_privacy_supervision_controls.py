from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.privacy_flow_data import batch
from paperalpha_monitor.privacy_flow_bn import PrivacyFlowBN
from paperalpha_monitor.privacy_supervision_controls import (DirectObjectRisk, SeparatedTransitionGLMs,
    direct_design, outcome_matrix, bernoulli_glm_loss, object_predictions)
from test_privacy_flow_bn import example


def cases(stage):
    packets = [example(stage == 2)[1] for _ in range(4)]
    packets[1]["groups"].pop()
    packets[2]["groups"][0]["coordinator_present"] = False
    if stage == 2:
        packets[2]["groups"][0]["worker_present"] = True
    labels = [[int(j == 0 and i >= 2) for j in range(len(p["groups"]))] for i, p in enumerate(packets)]
    return packets, labels


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("supervision", ["endpoint", "objects"])
def test_both_objectives_have_correct_gradients(stage, supervision):
    packets, labels = cases(stage)
    model = DirectObjectRisk(stage=stage, supervision=supervision, seed=47)
    data = direct_design(packets)
    targets = outcome_matrix(data, labels) if supervision == "objects" else np.array([any(v) for v in labels])
    fn = lambda t: model.loss_gradient(t, data, targets)[0]
    analytic = model.loss_gradient(model.theta, data, targets)[1]
    np.testing.assert_allclose(analytic, approx_derivative(fn, model.theta).ravel(), atol=2e-8, rtol=1e-5)


@pytest.mark.parametrize("stage", [1, 2])
def test_changing_supervision_does_not_change_prediction_architecture(stage):
    packets, _ = cases(stage)
    endpoint = DirectObjectRisk(stage=stage, supervision="endpoint", seed=42)
    objects = DirectObjectRisk(stage=stage, supervision="objects", seed=42)
    assert endpoint.names == objects.names
    np.testing.assert_array_equal(endpoint.predict_details(packets)["probabilities"], objects.predict_details(packets)["probabilities"])
    np.testing.assert_allclose(endpoint.predict_details(packets)["field_marginals"][0], objects.predict_details(packets)["field_marginals"][0])


def test_endpoint_loss_rejects_object_labels():
    packets, labels = cases(1)
    model = DirectObjectRisk(stage=1, supervision="endpoint")
    with pytest.raises(ValueError):
        model.fit(packets, labels)


@pytest.mark.parametrize("supervision", ["endpoint", "objects"])
def test_fit_and_serialized_restore(supervision):
    packets, labels = cases(2)
    targets = labels if supervision == "objects" else [int(any(v)) for v in labels]
    model = DirectObjectRisk(stage=2, supervision=supervision).fit(packets, targets)
    assert model.optimizer["success"]
    assert model.optimizer["loss"] < model.optimizer["initial_loss"]
    restored = DirectObjectRisk.restore(model.parameters())
    np.testing.assert_array_equal(model.predict_details(packets)["probabilities"], restored.predict_details(packets)["probabilities"])
    with pytest.raises(ValueError, match="cutoff"):
        restored.predict_details(cases(1)[0])


@pytest.mark.parametrize("seed", [0, 31, 81])
def test_bn_objective_is_exact_sum_of_two_glms(seed):
    packets, labels = cases(2)
    model = PrivacyFlowBN(shared=False, seed=seed, regularization=.013)
    data = batch(packets)
    f = outcome_matrix(data, labels)
    lq, gq = bernoulli_glm_loss(model.theta[:model.d], data["dc"], data["w"], data["mask"], model.regularization)
    lr, gr = bernoulli_glm_loss(model.theta[model.d:], data["dw"], f, data["mask"], model.regularization)
    actual, gradient = model.loss_gradient(model.theta, data, f)
    assert actual == pytest.approx(lq + lr, abs=1e-12)
    np.testing.assert_allclose(gradient, np.r_[gq, gr], atol=1e-12)
    separate = SeparatedTransitionGLMs(regularization=model.regularization)
    separate.worker, separate.final = model.theta[:model.d].copy(), model.theta[model.d:].copy()
    for stage in (1, 2):
        p = cases(stage)[0]
        np.testing.assert_allclose(model.predict(p), separate.predict_details(p)["probabilities"], atol=1e-12)


def test_joint_label_permutation_keeps_endpoint_target_but_changes_object_target():
    packets, labels = cases(2)
    data = direct_design(packets)
    f = outcome_matrix(data, labels)
    alternative = f.copy()
    alternative[3, 0], alternative[3, 1] = alternative[3, 1], alternative[3, 0]
    assert np.array_equal(f.any(axis=1), alternative.any(axis=1))
    endpoint = DirectObjectRisk(stage=2, supervision="endpoint", seed=43)
    obj = DirectObjectRisk(stage=2, supervision="objects", seed=43)
    assert endpoint.loss_gradient(endpoint.theta, data, f.any(axis=1))[0] == endpoint.loss_gradient(endpoint.theta, data, alternative.any(axis=1))[0]
    assert obj.loss_gradient(obj.theta, data, f)[0] != pytest.approx(obj.loss_gradient(obj.theta, data, alternative)[0])


def test_or_handles_masking_and_exact_zero_one():
    q = np.array([[0., 0., 1.], [1., .2, .3], [.2, .3, .9]])
    mask = np.array([[True, True, False], [True, True, False], [True, True, False]])
    np.testing.assert_allclose(object_predictions(q, mask), [0., 1., .44])


def test_development_process_denies_old_test_and_result_files(tmp_path):
    import subprocess
    original = tmp_path / "old"
    original.mkdir()
    (original / "train.json").write_text("[]", encoding="utf-8")
    for name in ("test.json", "REPORT.json", "predictions.json", "explanations.json"):
        (original / name).write_text("must-not-be-read", encoding="utf-8")
    program = "\n".join([
        "import sys", "from pathlib import Path", f"sys.path.insert(0, {str(ROOT / 'code/commands')!r})",
        "from evaluate_privacy_supervision_controls import guard_old_test", f"root = Path({str(original)!r}).resolve()",
        "attempts = guard_old_test(root)", "assert (root / 'train.json').read_text() == '[]'",
        "for name in ('test.json', 'REPORT.json', 'predictions.json', 'explanations.json'):",
        "    try:", "        (root / name).read_text()", "    except PermissionError:", "        pass",
        "    else:", "        raise AssertionError('boundary not enforced')", "assert len(attempts) == 4",
        "print('guard passed')"])
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "guard passed"
