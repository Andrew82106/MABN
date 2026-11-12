"""Attribution controls, not a new claimed MAS monitoring algorithm.

Direct per-object probabilities and a fixed OR readout can be trained with
either endpoint or object labels while leaving the prediction architecture
unchanged. Separate transition GLMs expose the independent BN factorization.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .privacy_flow_data import batch, KINDS, CONTEXT_NAMES


def direct_design(packets):
    data = batch(packets)
    # Same visible packet as the BN. At cutoff 2 this stronger control also
    # retains individual C instead of discarding it after observing W.
    data["design"] = (data["dc"] if data["stage"] == 1 else
        np.concatenate((data["dc"], data["w"][:, :, None]), axis=2))
    return data


def outcome_matrix(data, labels):
    if len(labels) != len(data["mask"]):
        raise ValueError("one label vector per trace required")
    result = np.zeros(data["mask"].shape)
    for i, values in enumerate(labels):
        if len(values) != int(data["mask"][i].sum()) or not np.isin(values, [0, 1]).all():
            raise ValueError("aligned binary object outcomes required")
        result[i, :len(values)] = values
    return result


def object_predictions(probabilities, mask):
    if probabilities.shape != mask.shape or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("valid per-object probabilities required")
    # Handle exact one without clipping the model's deterministic prediction.
    with np.errstate(divide="ignore"):
        log_safe = np.sum(np.where(mask, np.log1p(-probabilities), 0.), axis=1)
    return -np.expm1(log_safe)


def _solve(fn, initial, max_iter, bounds=None):
    bounds = [(-12., 12.)] * len(initial) if bounds is None else bounds
    result = minimize(fn, initial, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7})
    gradient = result.jac.copy()
    for i, (lo, hi) in enumerate(bounds):
        if (result.x[i] <= lo + 1e-8 and gradient[i] > 0) or (result.x[i] >= hi - 1e-8 and gradient[i] < 0):
            gradient[i] = 0.
    return result.x, {"success": bool(result.success), "iterations": int(result.nit), "loss": float(result.fun),
        "message": str(result.message), "projected_gradient_max": float(np.max(np.abs(gradient))),
        "initial_loss": float(fn(initial)[0])}


class DirectObjectRisk:
    """Same sigmoid-object + OR architecture, two alternative loss functions."""
    def __init__(self, *, stage, supervision, regularization=.001, seed=0):
        if type(stage) is not int or stage not in (1, 2) or supervision not in ("endpoint", "objects"):
            raise ValueError("explicit cutoff and supervision required")
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("nonnegative regularization required")
        self.stage, self.supervision, self.regularization = stage, supervision, regularization
        self.names = [*("kind:" + k for k in KINDS), "coordinator_present", *CONTEXT_NAMES]
        if stage == 2:
            self.names.append("worker_present")
        self.theta = np.zeros(len(self.names))
        self.theta[:len(KINDS)] = -3.
        if seed:
            self.theta += np.random.default_rng(seed).normal(0, .2, len(self.theta))
        self.optimizer = None

    def loss_gradient(self, theta, data, targets):
        if data["stage"] != self.stage or np.shape(theta) != self.theta.shape or not np.isfinite(theta).all():
            raise ValueError("compatible finite model and cutoff required")
        x, mask = data["design"], data["mask"]
        logits = x @ theta
        p = expit(logits)
        targets = np.asarray(targets)
        if not np.isin(targets, [0, 1]).all():
            raise ValueError("binary outcomes required")
        if self.supervision == "objects":
            if targets.shape != mask.shape:
                raise ValueError("object supervision needs an aligned matrix")
            loss = np.sum((np.logaddexp(0., logits) - targets * logits) * mask) / len(mask)
            error = (p - targets) * mask
        else:
            if targets.shape != (len(mask),):
                raise ValueError("endpoint supervision accepts ONLY one label per trace")
            log_safe = -np.sum(np.logaddexp(0., logits) * mask, axis=1)
            log_risk = np.log(-np.expm1(log_safe))
            loss = -np.mean(np.where(targets == 1, log_risk, log_safe))
            multiplier = np.where(targets == 1, -np.exp(log_safe - log_risk), 1.)
            error = multiplier[:, None] * p * mask
        gradient = np.einsum("ij,ijd->d", error, x) / len(mask)
        return float(loss + self.regularization * (theta @ theta)), gradient + 2 * self.regularization * theta

    def fit(self, packets, targets, *, max_iter=1000):
        data = direct_design(packets)
        # Separate public APIs prevent the endpoint arm receiving object labels.
        target_array = outcome_matrix(data, targets) if self.supervision == "objects" else np.asarray(targets)
        self.theta, self.optimizer = _solve(lambda t: self.loss_gradient(t, data, target_array), self.theta, max_iter)
        return self

    def predict_details(self, packets):
        data = direct_design(packets)
        if data["stage"] != self.stage:
            raise ValueError("model cannot silently switch prediction cutoff")
        q = expit(data["design"] @ self.theta)
        return {"probabilities": object_predictions(q, data["mask"]),
            "field_marginals": [q[i, data["mask"][i]].tolist() for i in range(len(packets))],
            "stage": self.stage, "supervision": self.supervision, "trained": self.optimizer is not None,
            "calibrated": False,
            "interpretation": "Endpoint-only training does not identify these per-object probabilities as faithful object-level explanations."}

    def parameters(self):
        return {"model": "DirectObjectRisk-control-v1", "stage": self.stage, "supervision": self.supervision,
            "regularization": self.regularization, "names": self.names, "theta": self.theta.tolist(), "optimizer": self.optimizer}

    @classmethod
    def restore(cls, saved):
        model = cls(stage=saved["stage"], supervision=saved["supervision"], regularization=saved["regularization"])
        if model.names != saved["names"]:
            raise ValueError("saved parameter specification differs")
        model.theta = np.asarray(saved["theta"], dtype=float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite saved parameters required")
        model.optimizer = saved["optimizer"]
        return model


def bernoulli_glm_loss(theta, x, y, mask, regularization):
    """A single ordinary supervised GLM block, normalized by trace count."""
    logits = x @ theta
    loss = np.sum((np.logaddexp(0., logits) - y * logits) * mask) / len(mask)
    gradient = np.einsum("ij,ijd->d", (expit(logits) - y) * mask, x) / len(mask)
    return float(loss + regularization * (theta @ theta)), gradient + 2 * regularization * theta


class SeparatedTransitionGLMs:
    """Equivalent parametrization of independent PrivacyFlowBN, no BN engine."""
    def __init__(self, *, regularization=.001):
        self.regularization = regularization
        self.width = len(KINDS) + 1 + len(CONTEXT_NAMES)
        self.worker = np.zeros(self.width)
        self.worker[:len(KINDS)], self.worker[len(KINDS)] = -2., 2.
        self.final = self.worker.copy()
        self.optimizers = {}

    def fit(self, packets, final_labels):
        data = batch(packets)
        if data["stage"] != 2:
            raise ValueError("observed worker and separate final labels required")
        f = outcome_matrix(data, final_labels)
        for attr, x, y in (("worker", data["dc"], data["w"]), ("final", data["dw"], f)):
            theta, result = _solve(lambda t: bernoulli_glm_loss(t, x, y, data["mask"], self.regularization), getattr(self, attr), 1000)
            setattr(self, attr, theta)
            self.optimizers[attr] = result
        return self

    def predict_details(self, packets):
        data = batch(packets)
        if data["stage"] == 1:
            q = expit(data["dc"] @ self.worker)
            logits0 = data["dw"] @ self.final
            r = (1 - q) * expit(logits0) + q * expit(logits0 + self.final[len(KINDS)])
        else:
            r = expit(data["dw"] @ self.final)
        return {"probabilities": object_predictions(r, data["mask"]),
            "field_marginals": [r[i, data["mask"][i]].tolist() for i in range(len(packets))]}

    def parameters(self):
        return {"model": "SeparatedTransitionGLMs-equivalence-control", "regularization": self.regularization,
            "worker": self.worker.tolist(), "final": self.final.tolist(), "optimizers": self.optimizers}
