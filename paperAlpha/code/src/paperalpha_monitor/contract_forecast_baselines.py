"""Same-information nonlinear factors on the fixed 32-world contract graph.

Only the four conditional probability families change: each is a HistGB
classifier over the SAME numeric features and named parent states used by the
linear core. Hard A evidence and the fixed support/policy gate remain identical.
Partial A/C/B/S labels receive exact posterior weights, not zero imputation.

Training is a trace-grouped conditional composite score, not a cross-window
generative likelihood. Weighted boosting is an approximate M step: actual
observed-label marginal loss is recorded each round, and a worse candidate is
rejected using TRAINING loss only. Neither monotone raw boosting updates nor a
global EM optimum is claimed. The caller owns family-held-out splits/tuning.

Declared Dirichlet smoothing is a prior, not fabricated observed examples.
Single-class/empty heads use that constant prior and explicitly report it.
The core schema/support helpers are shared; no fitted linear factors are used.
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .contract_forecast import (ContractForecastBN, CALL_TYPES, VARIABLES,
                                DEFAULT_FEATURE_NAMES, RISK_GATE, SUPPORT, FORECAST_SCOPE)


DEFAULT_TREE_CONFIG = {"max_leaf_nodes": 7, "max_iter": 100, "min_samples_leaf": 3,
                       "learning_rate": .1, "l2_regularization": .1,
                       "random_state": 17, "early_stopping": False}
CLASSES = {"A": 2, "C": 4, "B": 2, "S": 2}


class ContractFactorHistGB:
    """Four weighted classifiers, exact enumeration and a fixed policy gate."""
    SCHEMA = "contract-next-call-histgb-factors"

    def __init__(self, feature_names=DEFAULT_FEATURE_NAMES, *, tree_config=None,
                 max_rounds=5, prior_pseudocount=.5, prior_mix=.01):
        self._schema = ContractForecastBN(feature_names)
        self.feature_names, self.states = self._schema.feature_names, self._schema.states
        self.tree_config = {**DEFAULT_TREE_CONFIG, **(tree_config or {})}
        if self.tree_config.get("early_stopping") is not False:
            raise ValueError("row-random early stopping is forbidden; tune on training families")
        if type(self.tree_config.get("random_state")) is not int:
            raise ValueError("an explicit fixed integer tree seed is required")
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("positive fixed EM round budget required")
        if not np.isscalar(prior_pseudocount) or not np.isfinite(prior_pseudocount) or prior_pseudocount <= 0:
            raise ValueError("positive finite declared prior pseudocount required")
        if not np.isscalar(prior_mix) or not np.isfinite(prior_mix) or not 0 < prior_mix < 1:
            raise ValueError("explicit prior mixture strictly between zero and one required")
        self.max_rounds, self.prior_pseudocount, self.prior_mix = max_rounds, float(prior_pseudocount), float(prior_mix)
        self.heads = {name: self._prior_head(name, np.zeros(count)) for name, count in CLASSES.items()}
        self.fit_report = None

    def _prior_head(self, name, counts, *, rows=0):
        counts = np.asarray(counts, float)
        prior = (counts + self.prior_pseudocount) / (counts.sum() + len(counts)*self.prior_pseudocount)
        return {"kind": "constant_prior", "model": None, "prior": prior,
                "posterior_class_weight": counts, "positive_weight_rows": int(rows),
                "reason": "no_training_rows" if not rows else "single_posterior_class"}

    def _probabilities(self, name, design, heads=None):
        head = (self.heads if heads is None else heads)[name]
        x = np.asarray(design, float)
        if head["model"] is None:
            return np.tile(head["prior"], (len(x), 1))
        probabilities = np.zeros((len(x), CLASSES[name]))
        estimator = head["model"]
        probabilities[:, estimator.classes_.astype(int)] = estimator.predict_proba(x)
        return (1-self.prior_mix)*probabilities + self.prior_mix*head["prior"]

    def joint(self, sample, *, _heads=None):
        """Exact same support and hard evidence as the linear contract core."""
        x = self._schema._input(sample)
        with threadpool_limits(limits=1):
            pa = self._probabilities("A", [x], _heads)[0]
            pc = self._probabilities("C", [np.r_[x, a] for a in (0, 1)], _heads)
            pb = self._probabilities("B", [np.r_[x, a, c] for a, c in
                ((0, 0), (0, 1), (1, 0), (1, 1))], _heads)
            ps = self._probabilities("S", [np.r_[x, a, b, c] for a in (0, 1)
                for b in (0, 1) for c in (0, 1)], _heads)
        factors = np.zeros((32, 4))
        for i, (a, c, b, s) in enumerate(self.states):
            qa = pa[a] if sample.current_a is None else float(a == sample.current_a)
            active, trigger = c in {"UPDATE", "TRIGGER"}, int(c == "TRIGGER")
            factors[i] = (qa, pc[a, CALL_TYPES.index(c)],
                          pb[2*a+trigger, b] if active else float(b == 0),
                          ps[4*a+2*b+trigger, s] if active else float(s == 0))
        mass = factors.prod(axis=1)
        if not np.isfinite(mass).all() or abs(mass.sum()-1) > 1e-10:
            raise ValueError("conditional tree factors failed exact normalization")
        with np.errstate(divide="ignore"):
            logs = np.log(mass)
        return {"states": self.states, "probabilities": mass, "log_probabilities": logs,
                "cpd_probabilities": factors}

    def label_log_likelihood(self, sample, labels, *, _heads=None):
        selected = self._schema._label_mask(labels)
        distribution = self.joint(sample, _heads=_heads)
        value = float(logsumexp(distribution["log_probabilities"][selected]))
        if not np.isfinite(value):
            raise ValueError("labels contradict hard evidence or declared feasible states")
        posterior = np.zeros(32)
        posterior[selected] = np.exp(distribution["log_probabilities"][selected]-value)
        return {"log_probability": value, "posterior": posterior}

    @staticmethod
    def _aligned(traces, targets):
        if not isinstance(traces, (list, tuple)) or not traces or len(traces) != len(targets):
            raise ValueError("aligned nonempty independent trace groups required")
        for trace, labels in zip(traces, targets):
            if not isinstance(trace, (list, tuple)) or not trace or len(trace) != len(labels):
                raise ValueError("aligned nonempty next-call inputs and separate labels required")

    def composite_loss(self, traces, targets, *, _heads=None):
        self._aligned(traces, targets)
        return float(-sum(self.label_log_likelihood(sample, label, _heads=_heads)["log_probability"]
                          for trace, labels in zip(traces, targets) for sample, label in zip(trace, labels)) / len(traces))

    def _posterior_rows(self, traces, targets):
        """One unit of expected mass per window before trace normalization.

        Duplicate design/target rows are combined, not counted as independent
        observations. Proven A and structurally fixed B/S never train heads.
        Completely uninformative labels contribute zero objective and no rows.
        """
        rows = {name: {} for name in VARIABLES}
        diagnostic = {"informative_windows": 0, "zero_information_windows": 0,
                      "direct_label_fields": dict.fromkeys(VARIABLES, 0), "window_posterior_mass": []}
        for trace, labels in zip(traces, targets):
            for sample, label in zip(trace, labels):
                posterior = self.label_log_likelihood(sample, label)["posterior"]
                reference = self._schema.label_log_likelihood(sample, label)["log_probability"]
                if reference > -1e-12:
                    diagnostic["zero_information_windows"] += 1
                    continue
                diagnostic["informative_windows"] += 1
                diagnostic["window_posterior_mass"].append(float(posterior.sum()))
                for name in VARIABLES:
                    diagnostic["direct_label_fields"][name] += int(label.get(name) is not None)
                x = self._schema._input(sample)
                for state, weight in zip(self.states, posterior):
                    if not weight:
                        continue
                    a, c, b, s = state
                    active, trigger = c in {"UPDATE", "TRIGGER"}, int(c == "TRIGGER")
                    designs = {"A": x, "C": np.r_[x, a], "B": np.r_[x, a, trigger],
                               "S": np.r_[x, a, b, trigger]}
                    values = {"A": a, "C": CALL_TYPES.index(c), "B": b, "S": s}
                    for name in VARIABLES:
                        if name == "A" and sample.current_a is not None or name in {"B", "S"} and not active:
                            continue
                        key = (tuple(designs[name]), values[name])
                        rows[name][key] = rows[name].get(key, 0.) + float(weight)/len(traces)
        return rows, diagnostic

    def _fit_heads(self, rows):
        heads = {}
        for name, grouped in rows.items():
            design = np.asarray([key[0] for key in grouped], float)
            y = np.asarray([key[1] for key in grouped], int)
            weights = np.asarray(list(grouped.values()), float)
            counts = np.bincount(y, weights=weights, minlength=CLASSES[name])
            head = self._prior_head(name, counts, rows=len(y))
            if len(np.unique(y)) > 1:
                with threadpool_limits(limits=1):
                    estimator = HistGradientBoostingClassifier(**self.tree_config).fit(design, y, sample_weight=weights)
                head.update(kind="histgb", model=estimator, reason=None)
            heads[name] = head
        return heads

    def fit(self, traces, targets):
        """Use only training labels; retain a candidate only if its loss improves.

        HistGB L2/row-loss scaling differs from linear parameter L2. Hyperparameter
        values must be selected on development/training groups, not assumed equal
        merely because two libraries expose similarly named regularizers.
        """
        self._aligned(traces, targets)
        original = copy.deepcopy([(x.features, x.current_a, x.metadata) for trace in traces for x in trace])
        rows, diagnostic = self._posterior_rows(traces, targets)
        if not diagnostic["informative_windows"]:
            raise ValueError("training needs informative actually observed labels")
        history = []
        current = self.composite_loss(traces, targets)
        initial_loss = current
        for step in range(self.max_rounds):
            rows, diagnostic = self._posterior_rows(traces, targets)
            candidate = self._fit_heads(rows)
            loss = self.composite_loss(traces, targets, _heads=candidate)
            accepted = bool(np.isfinite(loss) and loss <= current + 1e-12)
            history.append({"round": step+1, "loss_before": current, "candidate_composite_loss": loss,
                            "accepted": accepted, "reason": "training_score_not_worse" if accepted else "retained_previous_training_candidate",
                            "candidate_head_kinds": {name: head["kind"] for name, head in candidate.items()},
                            "candidate_posterior_class_weights": {name: head["posterior_class_weight"].tolist()
                                for name, head in candidate.items()}})
            if not accepted:
                break
            self.heads, improvement = candidate, current-loss
            current = loss
            if improvement < 1e-8:
                break
        self.fit_report = {"trace_count": len(traces), "next_call_windows": sum(map(len, traces)),
            "initial_composite_loss": initial_loss, "final_composite_loss": current, "rounds": history,
            "accepted_rounds": sum(int(row["accepted"]) for row in history),
            "retained_initial_prior": not any(row["accepted"] for row in history),
            "posterior_training": diagnostic, "objective": "trace-grouped per-window conditional composite negative log score",
            "cross_window_state_model": False, "global_em_optimum_claimed": False,
            "boosting_m_step_guaranteed_monotone": False, "test_labels_used": False,
            "risk_head_fitted": False, "calibration_evaluated": False,
            "tree_row_semantics": "weighted distinct conditional design/target rows; min_samples_leaf is not a count of independent episodes",
            "head_summary": {name: {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in head.items() if key != "model"} for name, head in self.heads.items()}}
        if original != [(x.features, x.current_a, x.metadata) for trace in traces for x in trace]:
            raise ValueError("fitting mutated predictor inputs")
        return self

    def predict(self, sample, *, explain=False):
        result = self.joint(sample)
        output = {"model": self.SCHEMA, "scope": FORECAST_SCOPE, "support": SUPPORT,
                  "risk_gate": RISK_GATE, "metadata": copy.deepcopy(sample.metadata),
                  "feature_names": list(self.feature_names), "current_A_evidence": sample.current_a,
                  "A_belief_is_authorization": False, "future_labels_used": False,
                  "whole_episode_safety": "not_evaluated", **self._schema._summary(result["probabilities"])}
        if explain:
            output["world_contributions"] = [{**dict(zip(VARIABLES, state)),
                "probability": float(mass), "violation_contribution": float(mass)*self._schema._derived(state)[3],
                "cpd_probabilities": dict(zip(VARIABLES, result["cpd_probabilities"][i].tolist()))}
                for i, (state, mass) in enumerate(zip(self.states, result["probabilities"]))]
        return output

    def metadata(self):
        return {"schema": self.SCHEMA, "feature_names": list(self.feature_names),
                "tree_config": copy.deepcopy(self.tree_config), "max_rounds": self.max_rounds,
                "prior_pseudocount": self.prior_pseudocount, "prior_mix": self.prior_mix,
                "prior_meaning": "declared regularizing pseudo-counts, never observed label rows",
                "support": SUPPORT, "risk_gate": RISK_GATE, "forecast_scope": FORECAST_SCOPE,
                "fit_report": copy.deepcopy(self.fit_report)}

    def save(self, path):
        """Trusted local sklearn artifact only; never load an untrusted pickle."""
        joblib.dump({"metadata": self.metadata(), "heads": self.heads}, Path(path))

    @classmethod
    def load(cls, path):
        """Load only an artifact produced locally by trusted code (joblib pickle)."""
        saved = joblib.load(Path(path))
        meta = saved["metadata"]
        if (meta.get("schema") != cls.SCHEMA or meta.get("support") != SUPPORT
                or meta.get("risk_gate") != RISK_GATE or meta.get("forecast_scope") != FORECAST_SCOPE):
            raise ValueError("tree checkpoint changes contract semantics")
        model = cls(meta["feature_names"], tree_config=meta["tree_config"], max_rounds=meta["max_rounds"],
                    prior_pseudocount=meta["prior_pseudocount"], prior_mix=meta["prior_mix"])
        if set(saved["heads"]) != set(VARIABLES):
            raise ValueError("tree checkpoint must contain all four conditional factors")
        model.heads, model.fit_report = saved["heads"], meta["fit_report"]
        return model
