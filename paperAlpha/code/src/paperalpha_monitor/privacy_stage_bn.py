"""Role-stage disclosure states with an optional signed probabilistic link.

U is high worker disclosure propensity, V is high final-stage propensity.
Their local emission boosts are nonnegative, but the U->V statistical association
may be negative. This avoids forcing both different roles into the same bit.
Not an identified causal relation or an observed internal model state.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from .privacy_flow_data import batch, KINDS, CONTEXT_NAMES
from .privacy_supervision_controls import _solve, outcome_matrix


class PrivacyStageBN:
    def __init__(self, *, coupled=True, regularization=.001, seed=0):
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("nonnegative finite regularization required")
        self.coupled, self.regularization = bool(coupled), regularization
        self.d, self.k = len(KINDS) + 1 + len(CONTEXT_NAMES), 1 + len(CONTEXT_NAMES)
        local = (*("kind:" + k for k in KINDS), "previous_stage_present", *CONTEXT_NAMES)
        self.names = [f"{stage}:{n}" for stage in ("worker", "final") for n in local]
        self.names += [f"{stage}_prior:{n}" for stage in ("worker", "final") for n in ("bias", *CONTEXT_NAMES)]
        self.names += ["worker_state_boost", "final_state_boost"]
        if self.coupled:
            self.names.append("worker_to_final_state_log_odds")
        self.theta = np.zeros(len(self.names))
        for offset in (0, self.d):
            self.theta[offset:offset + len(KINDS)] = -2.
            self.theta[offset + len(KINDS)] = 2.
        self.theta[2 * self.d] = self.theta[2 * self.d + self.k] = -1.
        self.boost_start = 2 * self.d + 2 * self.k
        self.theta[self.boost_start:self.boost_start + 2] = 1.
        if seed:
            self.theta += np.random.default_rng(seed).normal(0., .2, len(self.theta))
            self.theta[self.boost_start:self.boost_start + 2] = np.maximum(.1, self.theta[self.boost_start:self.boost_start + 2])
        self.optimizer = None

    def _parts(self, data, theta):
        if np.shape(theta) != self.theta.shape or not np.isfinite(theta).all():
            raise ValueError("finite compatible parameter vector required")
        u, v = np.array([0., 0., 1., 1.]), np.array([0., 1., 0., 1.])
        x = np.c_[np.ones(len(data["context"])), data["context"]]
        logit_u = x @ theta[2 * self.d:2 * self.d + self.k]
        logit_v = (x @ theta[2 * self.d + self.k:2 * self.d + 2 * self.k])[:, None]
        logit_v = logit_v + u * (theta[-1] if self.coupled else 0.)
        log_prior = u * logit_u[:, None] - np.logaddexp(0., logit_u[:, None])
        log_prior += v * logit_v - np.logaddexp(0., logit_v)
        boost = theta[self.boost_start:self.boost_start + 2]
        qw = (data["dc"] @ theta[:self.d])[:, :, None] + u * boost[0]
        rf = (data["dw"] @ theta[self.d:2 * self.d])[:, :, None] + v * boost[1]
        return x, u, v, logit_u, logit_v, log_prior, qw, rf

    @staticmethod
    def _evidence(logit, y, mask):
        return np.sum((y[:, :, None] * logit - np.logaddexp(0., logit)) * mask[:, :, None], axis=1)

    def loss_gradient(self, theta, data, f):
        if data["stage"] != 2 or f.shape != data["mask"].shape or not np.isin(f, [0, 1]).all():
            raise ValueError("observed worker and aligned binary final labels required")
        x, u, v, lu, lv, prior, qw, rf = self._parts(data, theta)
        joint = prior + self._evidence(qw, data["w"], data["mask"]) + self._evidence(rf, f, data["mask"])
        loglik = logsumexp(joint, axis=1)
        post = np.exp(joint - loglik[:, None])
        ew = (data["w"][:, :, None] - expit(qw)) * data["mask"][:, :, None] * post[:, None, :]
        ef = (f[:, :, None] - expit(rf)) * data["mask"][:, :, None] * post[:, None, :]
        gradient, n = np.zeros_like(theta), len(x)
        gradient[:self.d] = -np.einsum("ijg,ijd->d", ew, data["dc"]) / n
        gradient[self.d:2 * self.d] = -np.einsum("ijg,ijd->d", ef, data["dw"]) / n
        gradient[2 * self.d:2 * self.d + self.k] = -x.T @ (post @ u - expit(lu)) / n
        residual_v = post * (v - expit(lv))
        gradient[2 * self.d + self.k:2 * self.d + 2 * self.k] = -x.T @ residual_v.sum(axis=1) / n
        gradient[self.boost_start:self.boost_start + 2] = np.array([-np.sum(ew * u), -np.sum(ef * v)]) / n
        if self.coupled:
            gradient[-1] = -np.sum(residual_v * u) / n
        return float(-loglik.mean() + self.regularization * (theta @ theta)), gradient + 2 * self.regularization * theta

    def fit(self, packets, final_labels, *, max_iter=1000):
        data = batch(packets)
        f = outcome_matrix(data, final_labels)
        bounds = [(-12., 12.)] * len(self.theta)
        bounds[self.boost_start:self.boost_start + 2] = [(0., 12.)] * 2
        self.theta, self.optimizer = _solve(lambda t: self.loss_gradient(t, data, f), self.theta, max_iter, bounds=bounds)
        return self

    def predict_details(self, packets):
        data = batch(packets)
        return self.predict_compiled(data)

    def predict_compiled(self, data):
        """Numerical core; input adapters must enforce their observation schema."""
        _, u, v, _, _, prior, qw, rf = self._parts(data, self.theta)
        if data["stage"] == 1:
            q = expit(qw)
            r = (1 - q) * expit(rf) + q * expit(rf + self.theta[self.d + len(KINDS)])
            weights = np.exp(prior)
        else:
            log_post = prior + self._evidence(qw, data["w"], data["mask"])
            weights = np.exp(log_post - logsumexp(log_post, axis=1)[:, None])
            r = expit(rf)
        with np.errstate(divide="ignore"):
            log_safe = np.sum(np.where(data["mask"][:, :, None], np.log1p(-r), 0.), axis=1)
        mode_risk = -np.expm1(log_safe)
        marginal = np.sum(weights[:, None, :] * r, axis=2)
        return {"probabilities": np.sum(weights * mode_risk, axis=1),
            "field_marginals": [marginal[i, data["mask"][i]].tolist() for i in range(len(data["mask"]))],
            "state_pairs": [[int(a), int(b)] for a, b in zip(u, v)], "state_weights": weights,
            "state_conditional_risk": mode_risk, "stage": data["stage"], "trained": self.optimizer is not None,
            "calibrated": False, "causal": False,
            "meaning": "Two different stage-local disclosure propensities with optional statistical coupling; same observation contract as PrivacyFlowBN-v1."}

    def parameters(self):
        return {"model": "PrivacyStageBN-v1", "coupled": self.coupled, "regularization": self.regularization,
            "names": self.names, "theta": self.theta.tolist(), "optimizer": self.optimizer,
            "state_link_log_odds": float(self.theta[-1]) if self.coupled else 0.,
            "local_boosts": self.theta[self.boost_start:self.boost_start + 2].tolist(),
            "scope": "Development-only privacy fragment, not general DAG discovery or identified causal/semantic internal states"}

    @classmethod
    def restore(cls, saved):
        model = cls(coupled=saved["coupled"], regularization=saved["regularization"])
        if model.names != saved["names"]:
            raise ValueError("saved parameter definition differs")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite saved vector required")
        model.optimizer = saved["optimizer"]
        return model
