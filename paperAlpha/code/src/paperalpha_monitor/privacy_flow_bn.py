"""Small exact BN for prospective literal disclosure across MAS stages.

Conditional on binary G, distinct literal values have independent W/F chains.
Integrating G preserves co-disclosure dependence. This is a model assumption,
not causal discovery, semantic privacy understanding or calibrated certainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .privacy_flow_data import KINDS, CONTEXT_NAMES, batch, digest

from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]


class PrivacyFlowBN:
    def __init__(self, *, shared=True, regularization=.01, seed=0, config=None):
        self.config = config or json.loads(static_resource('knowledge/templates/privacy_flow_bn_v1.json').read_text(encoding="utf-8"))
        if (self.config["schema"] != "privacy-flow-bn-v1" or self.config["field_kinds"] != list(KINDS)
                or self.config["functional_stages"] != ["coordinator_prepare", "worker", "coordinator_finalize"]
                or not self.config["fragment"]["worker_direct_vault_access"]
                or self.config["fragment"]["derived_memory_is_independent_evidence"]):
            raise ValueError("unsupported knowledge fragment")
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("nonnegative finite regularization required")
        self.shared, self.regularization = bool(shared), regularization
        self.d, self.k = len(KINDS) + 1 + len(CONTEXT_NAMES), len(CONTEXT_NAMES) + 1
        local = (*("kind:" + k for k in KINDS), "previous_stage_present", *CONTEXT_NAMES)
        self.names = [stage + ":" + name for stage in ("worker", "final") for name in local]
        if self.shared:
            self.names += ["shared_prior:" + n for n in ("bias", *CONTEXT_NAMES)] + ["shared_boost:worker", "shared_boost:final"]
        self.theta = np.zeros(len(self.names))
        for offset in (0, self.d):
            self.theta[offset:offset + len(KINDS)] = -2.
            self.theta[offset + len(KINDS)] = 2.
        if self.shared:
            self.theta[2 * self.d] = -1.
            self.theta[-2:] = 1.
        if seed:
            self.theta += np.random.default_rng(seed).normal(0., .2, len(self.theta))
            if self.shared:
                self.theta[-2:] = np.maximum(.1, self.theta[-2:])
        self.optimizer = None

    def _parts(self, data, theta):
        if np.shape(theta) != self.theta.shape or not np.isfinite(theta).all():
            raise ValueError("finite compatible parameter vector required")
        modes = np.arange(2 if self.shared else 1)
        z = np.c_[np.ones(len(data["context"])), data["context"]]
        if self.shared:
            prior_logit = z @ theta[2 * self.d:2 * self.d + self.k]
            logprior = np.c_[-np.logaddexp(0., prior_logit), -np.logaddexp(0., -prior_logit)]
            boosts = theta[-2:]
        else:
            logprior, boosts = np.zeros((len(z), 1)), np.zeros(2)
        qlogit = (data["dc"] @ theta[:self.d])[:, :, None] + modes * boosts[0]
        rlogit = (data["dw"] @ theta[self.d:2 * self.d])[:, :, None] + modes * boosts[1]
        return z, logprior, qlogit, rlogit

    @staticmethod
    def _bernoulli_sum(logits, values, mask):
        return np.sum((values[:, :, None] * logits - np.logaddexp(0., logits)) * mask[:, :, None], axis=1)

    def loss_gradient(self, theta, data, final_labels):
        if data["stage"] != 2:
            raise ValueError("training requires observed worker and separate future labels")
        f = np.asarray(final_labels)
        if f.shape != data["mask"].shape or not np.isin(f, [0, 1]).all():
            raise ValueError("binary field-object outcomes matching this batch required")
        z, logprior, qlogit, rlogit = self._parts(data, theta)
        logjoint = logprior + self._bernoulli_sum(qlogit, data["w"], data["mask"]) + self._bernoulli_sum(rlogit, f, data["mask"])
        loglik = logsumexp(logjoint, axis=1)
        responsibility = np.exp(logjoint - loglik[:, None])
        qerr = (data["w"][:, :, None] - expit(qlogit)) * data["mask"][:, :, None] * responsibility[:, None, :]
        rerr = (f[:, :, None] - expit(rlogit)) * data["mask"][:, :, None] * responsibility[:, None, :]
        n = len(z)
        gradient = np.zeros_like(theta)
        gradient[:self.d] = -np.einsum("ijg,ijd->d", qerr, data["dc"]) / n
        gradient[self.d:2 * self.d] = -np.einsum("ijg,ijd->d", rerr, data["dw"]) / n
        if self.shared:
            gradient[2 * self.d:2 * self.d + self.k] = -z.T @ (responsibility[:, 1] - np.exp(logprior[:, 1])) / n
            gradient[-2:] = [-qerr[:, :, 1].sum() / n, -rerr[:, :, 1].sum() / n]
        return float(-loglik.mean() + self.regularization * (theta @ theta)), gradient + 2 * self.regularization * theta

    def fit(self, packets, final_labels, *, max_iter=500):
        data = batch(packets)
        f = np.zeros(data["mask"].shape)
        if len(final_labels) != len(packets):
            raise ValueError("one final label vector per trace required")
        for i, values in enumerate(final_labels):
            if len(values) != len(packets[i]["groups"]):
                raise ValueError("final label object count mismatch")
            f[i, :len(values)] = values
        fn = lambda theta: self.loss_gradient(theta, data, f)
        bounds = [tuple(self.config["parameter_bounds"])] * len(self.theta)
        if self.shared:
            bounds[-2:] = [(0., self.config["parameter_bounds"][1])] * 2
        initial = fn(self.theta)[0]
        result = minimize(fn, self.theta, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6})
        self.theta = result.x
        projected = result.jac.copy()
        for i, (low, high) in enumerate(bounds):
            if (result.x[i] <= low + 1e-8 and projected[i] > 0) or (result.x[i] >= high - 1e-8 and projected[i] < 0):
                projected[i] = 0.
        self.optimizer = {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
            "initial_loss": initial, "loss": float(result.fun), "projected_gradient_max": float(np.max(np.abs(projected)))}
        return self

    def predict_details(self, packets):
        data = batch(packets)
        _, logprior, qlogit, rlogit = self._parts(data, self.theta)
        if data["stage"] == 1:
            # Worker presence is not observed. Integrate it, including W=1
            # when C=0 because the worker also receives the full vault.
            q = expit(qlogit)
            r0 = expit(rlogit)  # batch W is zero only as a computation anchor
            r1 = expit(rlogit + self.theta[self.d + len(KINDS)])
            field_prob = (1 - q) * r0 + q * r1
            weights = np.exp(logprior)
        else:
            logposterior = logprior + self._bernoulli_sum(qlogit, data["w"], data["mask"])
            weights = np.exp(logposterior - logsumexp(logposterior, axis=1)[:, None])
            field_prob = expit(rlogit)
        # OR within each common mode, THEN mix: not OR of field marginals.
        log_safe = np.sum(np.log1p(-np.minimum(field_prob, 1 - 1e-15)) * data["mask"][:, :, None], axis=1)
        mode_risk = -np.expm1(log_safe)
        probability = np.sum(weights * mode_risk, axis=1)
        marginals = np.sum(field_prob * weights[:, None, :], axis=2)
        return {"probabilities": probability, "mode_weights": weights, "mode_risk": mode_risk,
            "field_marginals": [marginals[i, :len(p["groups"])].tolist() for i, p in enumerate(packets)],
            "target": self.config["target"], "cutoff": data["stage"], "calibrated": False,
            "interpretation": "Statistical literal-disclosure forecast conditional on configured policy/vault, not a causal or general-safety claim."}

    def predict(self, packets):
        return self.predict_details(packets)["probabilities"]

    def parameters(self):
        return {"model": "PrivacyFlowBN-v1", "shared": self.shared, "regularization": self.regularization,
            "parameter_names": self.names, "theta": self.theta.tolist(), "optimizer": self.optimizer,
            "config": self.config, "config_sha256": digest(self.config),
            "training_target": "Conditional joint likelihood of worker and final per-value disclosures, not direct endpoint BCE"}

    @classmethod
    def restore(cls, saved):
        model = cls(shared=saved["shared"], regularization=saved["regularization"], config=saved["config"])
        if saved["parameter_names"] != model.names or saved["config_sha256"] != digest(model.config):
            raise ValueError("saved specification mismatch")
        model.theta = np.asarray(saved["theta"], dtype=float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("invalid saved parameters")
        model.optimizer = saved["optimizer"]
        return model
