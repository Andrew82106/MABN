"""Small, inspectable next-call distribution; not an episode safety verdict.

The 32 enumerated worlds are A x C x B x S. A is the pre-call prerequisite
state, C is UPDATE/TRIGGER/OTHER/STOP, B is this call's declared contract binding,
and S is its contract-relevant simulator effect. OTHER/STOP have canonical B=S=0
(not applicable), not fabricated failed authorizations. Wrong-object UPDATE and
TRIGGER worlds remain possible. The immutable policy gate is
V = (C == TRIGGER) and not (A and B) and S.

STOP requires a successful terminal supported nonempty text reply with no tool
calls. API failure, empty or unsupported replies are unknown observations for
the adapter to handle, not STOP. Each input window estimates its own A; this
core does not implement shared latent states or transitions across windows.

An observed A is hard evidence. Unknown A has a learned predictive belief, never
a proven authorization. Numerical features must come from the supplied prefix;
this core neither extracts text nor receives future labels at prediction time.
Its factorization is exactly a collection of conditional GLMs, not a competing
algorithm that should outperform the same fitted factors under another name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import copy
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np
from scipy.special import expit, logsumexp


CALL_TYPES = ("UPDATE", "TRIGGER", "OTHER", "STOP")
VARIABLES = ("A", "C", "B", "S")
DEFAULT_FEATURE_NAMES = ("bias",)
STATE_SEMANTICS = {
    "A": "prerequisite state immediately before this call",
    "C": "next call type; STOP requires a successful terminal supported nonempty text reply without tool calls",
    "B": "object/parameter/role binding meets the explicit contract",
    "S": "contract-relevant simulator effect, not external harm",
}
SUPPORT = "OTHER/STOP force B=S=0; UPDATE/TRIGGER retain every B/S pair"
RISK_GATE = "(C == TRIGGER) and not (A and B) and S"
FACTORIZATION = "qA(x) qC(A,x) qB(C,A,x) qS(C,B,A,x)"
FORECAST_SCOPE = ("first tool call in the next API reply; STOP requires a successful terminal "
                  "supported nonempty text reply without tool calls; API failure, empty or "
                  "unsupported replies are unknown and handled by the adapter")


def _bit(value, description):
    if value is not None and (type(value) is not int or value not in (0, 1)):
        raise ValueError(description + " must be integer 0/1 or None")


def _names(names):
    names = tuple(names)
    if (not names or len(set(names)) != len(names) or any(not isinstance(n, str)
            or not n or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:.-]*", n) for n in names)):
        raise ValueError("distinct explicit feature names required")
    for name in names:
        tokens = re.split(r"[_:.-]+", name.lower())
        if ("id" in tokens or "hash" in tokens or
                any(t in tokens for t in ("taskid", "actorid", "recordid", "episodeid"))):
            raise ValueError("raw task/actor/record identifiers are not model features")
    return names


@dataclass(frozen=True)
class ContractForecastInput:
    """Only already-visible numeric features and optional established A.

    Feature naming checks are not a proof against an adapter encoding IDs under
    arbitrary names. Adapters remain responsible for visible, non-ID semantics.
    Evidence provenance belongs in the adapter, not in model coefficients.
    """
    features: dict
    current_a: int | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.features, dict):
            raise ValueError("explicit numeric feature mapping required")
        object.__setattr__(self, "features", copy.deepcopy(self.features))
        if not isinstance(self.metadata, dict):
            raise ValueError("optional metadata must be a mapping, never numeric features")
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
        _bit(self.current_a, "current A")


class ContractForecastBN:
    """Exactly enumerated conditional GLMs with a non-learned contract gate.

    STOP is the zero-logit reference for C. UPDATE is the reference call type
    for the active B/S logits. No risk classifier or learned policy threshold
    exists. No state transition or whole-episode prediction is claimed here.
    """
    SCHEMA = "contract-next-call-bn"

    def __init__(self, feature_names=DEFAULT_FEATURE_NAMES, *, theta=None):
        self.feature_names = _names(feature_names)
        self.design_names = {
            "A": self.feature_names,
            "C": self.feature_names + ("parent:A",),
            "B": self.feature_names + ("parent:A", "parent:C=TRIGGER"),
            "S": self.feature_names + ("parent:A", "parent:B", "parent:C=TRIGGER"),
        }
        names, slices = [], {}
        for factor in VARIABLES:
            categories = CALL_TYPES[:-1] if factor == "C" else (None,)
            for category in categories:
                key = factor if category is None else factor + "=" + category
                start = len(names)
                names.extend(key + ":" + name for name in self.design_names[factor])
                slices[key] = slice(start, len(names))
        self.parameter_names, self._slices = tuple(names), slices
        self.theta = np.zeros(len(names)) if theta is None else self._theta(theta).copy()
        self.optimizer = None
        self.states = tuple(product((0, 1), CALL_TYPES, (0, 1), (0, 1)))

    def _theta(self, theta):
        value = np.asarray(theta, dtype=float)
        if value.shape != (len(self.parameter_names),) or not np.isfinite(value).all():
            raise ValueError("finite compatible parameter vector required")
        return value

    def _input(self, sample):
        if not isinstance(sample, ContractForecastInput) or set(sample.features) != set(self.feature_names):
            raise ValueError("input must contain exactly the declared prefix feature names")
        _bit(sample.current_a, "current A")
        values = [sample.features[name] for name in self.feature_names]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v) for v in values):
            raise ValueError("features must be finite numeric values, never text or identifiers")
        if "bias" in sample.features and sample.features["bias"] != 1:
            raise ValueError("declared bias feature must equal one")
        return np.asarray(values, dtype=float)

    def _local(self, factor, value, design, theta):
        """Log probability, score, and named logit contributions of one CPD."""
        gradient = np.zeros(len(theta))
        contributions = {}
        if factor == "C":
            logits = []
            for category in CALL_TYPES[:-1]:
                key = "C=" + category
                terms = design * theta[self._slices[key]]
                logits.append(float(terms.sum()))
                contributions[key] = dict(zip(self.design_names[factor], terms.tolist()))
            logits = np.asarray([*logits, 0.])
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite conditional logits")
            log_probs = logits - logsumexp(logits)
            probs = np.exp(log_probs)
            for j, category in enumerate(CALL_TYPES[:-1]):
                gradient[self._slices["C=" + category]] = ((value == category) - probs[j]) * design
            return float(log_probs[CALL_TYPES.index(value)]), gradient, contributions
        terms = design * theta[self._slices[factor]]
        logit = float(terms.sum())
        if not np.isfinite(logit):
            raise ValueError("nonfinite conditional logit")
        gradient[self._slices[factor]] = (value - expit(logit)) * design
        contributions[factor] = dict(zip(self.design_names[factor], terms.tolist()))
        return float(value * logit - np.logaddexp(0., logit)), gradient, contributions

    def joint(self, sample, *, theta=None):
        """All 32 worlds; impossible worlds have zero mass and zero score.

        Scores are derivatives of normalized log mass with respect to the stable
        parameter vector. Hard evidence contributes no A-training score.
        """
        x = self._input(sample)
        theta = self._theta(self.theta if theta is None else theta)
        logs = np.full(32, -np.inf)
        scores = np.zeros((32, len(theta)))
        factor_logs = np.full((32, 4), -np.inf)
        logit_terms = []
        for index, (a, c, b, s) in enumerate(self.states):
            active = c in {"UPDATE", "TRIGGER"}
            possible = (sample.current_a is None or a == sample.current_a) and (active or not (b or s))
            local, score, terms = [], np.zeros(len(theta)), {}
            designs = {"A": x, "C": np.r_[x, a], "B": np.r_[x, a, int(c == "TRIGGER")],
                       "S": np.r_[x, a, b, int(c == "TRIGGER")]}
            for factor, value in zip(VARIABLES, (a, c, b, s)):
                if factor == "A" and sample.current_a is not None:
                    log_p, grad, detail = (0. if a == sample.current_a else -np.inf), np.zeros(len(theta)), {}
                elif factor in {"B", "S"} and not active:
                    log_p, grad, detail = (0. if value == 0 else -np.inf), np.zeros(len(theta)), {}
                else:
                    log_p, grad, detail = self._local(factor, value, designs[factor], theta)
                local.append(log_p); score += grad; terms.update(detail)
            factor_logs[index] = local
            if possible:
                logs[index], scores[index] = sum(local), score
            logit_terms.append(terms)
        normalizer = float(logsumexp(logs))
        mass = np.exp(logs - normalizer)
        normalizer_score = mass @ scores
        scores[np.isfinite(logs)] -= normalizer_score
        return {"states": self.states, "probabilities": mass, "log_probabilities": logs - normalizer,
                "scores": scores, "factor_log_probabilities": factor_logs,
                "logit_terms": logit_terms, "log_normalizer": normalizer}

    def _label_mask(self, labels):
        if not isinstance(labels, dict) or set(labels) - set(VARIABLES):
            raise ValueError("separate A/C/B/S supervision only; never train a risk head")
        for key, value in labels.items():
            if key == "C":
                if value is not None and value not in CALL_TYPES:
                    raise ValueError("known call type or None required")
            else:
                _bit(value, "label " + key)
        return np.asarray([all(value is None or state[VARIABLES.index(key)] == value
                               for key, value in labels.items()) for state in self.states])

    def label_log_likelihood(self, sample, labels, *, theta=None):
        """Exact marginal likelihood; absent/None fields stay unobserved."""
        selected = self._label_mask(labels)
        result = self.joint(sample, theta=theta)
        value = float(logsumexp(result["log_probabilities"][selected]))
        if not np.isfinite(value):
            raise ValueError("labels contradict hard evidence or declared feasible states")
        posterior = np.exp(result["log_probabilities"][selected] - value)
        return {"log_probability": value, "score": posterior @ result["scores"][selected],
                "observed_variables": [name for name in VARIABLES if labels.get(name) is not None]}

    @staticmethod
    def _derived(state):
        a, c, b, s = state
        trigger = int(c == "TRIGGER")
        prerequisite = int(bool(a and b))
        effect = int(bool(trigger and s))
        return trigger, prerequisite, effect, int(bool(trigger and not prerequisite and s))

    def _summary(self, mass):
        marginals = {name: {str(value): float(sum(p for state, p in zip(self.states, mass)
                                      if state[i] == value))
                           for value in (CALL_TYPES if name == "C" else (0, 1))}
                     for i, name in enumerate(VARIABLES)}
        joint_tpe = {f"{t},{p},{e}": 0. for t, p, e in product((0, 1), repeat=3)}
        risk = 0.
        for state, probability in zip(self.states, mass):
            t, p, e, v = self._derived(state)
            joint_tpe[f"{t},{p},{e}"] += float(probability)
            risk += float(probability) * v
        return {"marginals": marginals, "joint_T_P_E": joint_tpe, "violation_probability": risk}

    def predict(self, sample, *, explain=False, derivatives=False):
        result = self.joint(sample)
        output = {"model": self.SCHEMA, "scope": FORECAST_SCOPE,
                  "metadata": copy.deepcopy(sample.metadata),
                  "current_A_evidence": sample.current_a, "feature_names": list(self.feature_names),
                  "risk_gate": RISK_GATE, "support": SUPPORT, **self._summary(result["probabilities"]),
                  "A_belief_is_authorization": False,
                  "P_applicability": "A&B is computed for every world; interpret P only where T=1",
                  "future_labels_used": False, "whole_episode_safety": "not_evaluated"}
        if derivatives:
            gate = np.asarray([self._derived(state)[3] for state in self.states])
            output["risk_gradient"] = dict(zip(self.parameter_names,
                ((result["probabilities"] * gate) @ result["scores"]).tolist()))
        if explain:
            output["features"] = copy.deepcopy(sample.features)
            output["parameter_table"] = self.parameter_table()
            output["world_contributions"] = []
            for i, state in enumerate(self.states):
                t, p, e, v = self._derived(state)
                output["world_contributions"].append({**dict(zip(VARIABLES, state)),
                    "T": t, "P": p, "E": e, "V": v,
                    "probability": float(result["probabilities"][i]),
                    "violation_contribution": float(result["probabilities"][i]) * v,
                    "cpd_probabilities": dict(zip(VARIABLES, np.exp(result["factor_log_probabilities"][i]).tolist())),
                    "logit_contributions": result["logit_terms"][i]})
        return output

    def independent_control(self, sample):
        """Same marginals, A/B/S independent GIVEN C; retain feasibility.

        p(C)p(A|C)p(B|C)p(S|C) breaks A--B and A/B--S dependence within call
        type. It preserves A--C association and STOP/OTHER structural zeros.
        Fully independent singleton products cannot generally preserve those
        zeros while retaining all singleton marginals. This is a dependency
        ablation from the fitted joint, not a separately trained baseline.
        """
        original = self.joint(sample)["probabilities"]
        mass = np.zeros(32)
        for call in CALL_TYPES:
            selected = np.asarray([state[1] == call for state in self.states])
            total = float(original[selected].sum())
            if not total:
                continue
            marginals = {(i, value): float(sum(p for state, p in zip(self.states, original)
                            if state[1] == call and state[i] == value)) / total
                         for i in (0, 2, 3) for value in (0, 1)}
            for j, state in enumerate(self.states):
                if state[1] == call:
                    mass[j] = total * np.prod([marginals[i, state[i]] for i in (0, 2, 3)])
        return {"control": "conditional_independence_of_A_B_S_given_C",
                "broken_dependence": ["A--B given C", "A--S given C", "B--S given C"],
                "retained": "all singleton marginals, A--C, feasible support and fixed risk gate",
                "trained_baseline": False, "probabilities": mass.tolist(), **self._summary(mass)}

    def parameter_table(self):
        theta = self._theta(self.theta)
        return [{"name": name, "coefficient": float(value)} for name, value in zip(self.parameter_names, theta)]

    def parameters(self):
        return {"schema": self.SCHEMA, "feature_names": list(self.feature_names),
                "parameter_names": list(self.parameter_names), "theta": self._theta(self.theta).tolist(),
                "call_types": list(CALL_TYPES), "state_semantics": copy.deepcopy(STATE_SEMANTICS),
                "factorization": FACTORIZATION, "support": SUPPORT, "risk_gate": RISK_GATE,
                "forecast_scope": FORECAST_SCOPE,
                "optimizer": copy.deepcopy(self.optimizer)}

    @classmethod
    def restore(cls, saved):
        if not isinstance(saved, dict) or set(saved) != {"schema", "feature_names", "parameter_names", "theta",
                "call_types", "state_semantics", "factorization", "support", "risk_gate", "forecast_scope", "optimizer"}:
            raise ValueError("explicit contract forecast checkpoint required")
        if (saved["schema"] != cls.SCHEMA or saved["call_types"] != list(CALL_TYPES)
                or saved["state_semantics"] != STATE_SEMANTICS or saved["factorization"] != FACTORIZATION
                or saved["forecast_scope"] != FORECAST_SCOPE
                or saved["support"] != SUPPORT or saved["risk_gate"] != RISK_GATE):
            raise ValueError("checkpoint changes declared model semantics or fixed policy")
        model = cls(saved["feature_names"], theta=saved["theta"])
        if saved["parameter_names"] != list(model.parameter_names):
            raise ValueError("checkpoint parameter identities differ")
        model.optimizer = copy.deepcopy(saved["optimizer"])
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.parameters(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.restore(json.loads(Path(path).read_text(encoding="utf-8")))
