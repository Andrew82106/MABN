"""Model-identity-free privacy packets and stage BN, without fake model IDs.

No identity is accepted by the prediction schema. The original generator model
may be retained in acquisition metadata, outside this model's input contract.
"""
from __future__ import annotations

import copy

import numpy as np

from .privacy_flow_data import protected_groups, validate_packet, KINDS, VERTICALS
from .privacy_stage_bn import PrivacyStageBN
from .privacy_supervision_controls import _solve, outcome_matrix

CONTEXT = ("coordinator_present_fraction", "log_coordinator_chars_over_8", "log_group_count_over_3",
           *("vertical:" + v for v in VERTICALS[1:]))
SCHEMA = "privacy-prefix-without-model-identity-v1"


def validate(packet):
    if set(packet) != {"schema", "stage", "vertical", "coordinator_chars", "groups"}:
        raise ValueError("identity-free prediction fields strictly whitelisted")
    if (packet["schema"] != SCHEMA or type(packet["stage"]) is not int or packet["stage"] not in (1, 2)
            or packet["vertical"] not in VERTICALS or type(packet["coordinator_chars"]) is not int
            or packet["coordinator_chars"] < 0 or not isinstance(packet["groups"], list) or not packet["groups"]):
        raise ValueError("supported nonempty prefix required")
    fields = []
    for g in packet["groups"]:
        if set(g) != {"kind", "field_names", "coordinator_present", "worker_present"}:
            raise ValueError("object fields strictly whitelisted")
        if (g["kind"] not in KINDS or type(g["coordinator_present"]) is not bool or not isinstance(g["field_names"], list)
                or not g["field_names"] or any(not isinstance(f, str) or not f for f in g["field_names"])):
            raise ValueError("typed objects required")
        if (packet["stage"] == 1 and g["worker_present"] is not None) or (packet["stage"] == 2 and type(g["worker_present"]) is not bool):
            raise ValueError("only observed stage evidence permitted")
        fields.extend(g["field_names"])
    if len(fields) != len(set(fields)):
        raise ValueError("duplicate field membership")


def from_legacy(packet):
    validate_packet(packet)
    result = copy.deepcopy({k: v for k, v in packet.items() if k != "model"})
    result["schema"] = SCHEMA
    validate(result)
    return result


def make_packet(initial, vertical, coordinator_text, worker_text=None):
    if not isinstance(coordinator_text, str) or (worker_text is not None and not isinstance(worker_text, str)):
        raise ValueError("completed earlier text required")
    groups = [{"kind": g["kind"], "field_names": g["field_names"],
        "coordinator_present": g["literal"] in coordinator_text.lower(),
        "worker_present": None if worker_text is None else g["literal"] in worker_text.lower()}
        for g in protected_groups(initial)]
    result = {"schema": SCHEMA, "stage": 1 if worker_text is None else 2, "vertical": vertical,
        "coordinator_chars": len(coordinator_text), "groups": groups}
    validate(result)
    return result


def compile_packets(packets):
    if not packets or len({p["stage"] for p in packets}) != 1:
        raise ValueError("one nonempty cutoff batch required")
    for p in packets:
        validate(p)
    n, m = len(packets), max(len(p["groups"]) for p in packets)
    mask, kinds, c, w = np.zeros((n, m), bool), np.zeros((n, m), int), np.zeros((n, m)), np.zeros((n, m))
    contexts = []
    for i, p in enumerate(packets):
        groups = p["groups"]
        contexts.append([np.mean([g["coordinator_present"] for g in groups]), np.log1p(p["coordinator_chars"]) / 8.,
            np.log1p(len(groups)) / 3., *(float(p["vertical"] == v) for v in VERTICALS[1:])])
        for j, g in enumerate(groups):
            mask[i, j], kinds[i, j], c[i, j], w[i, j] = True, KINDS.index(g["kind"]), g["coordinator_present"], bool(g["worker_present"])
    contexts = np.asarray(contexts)
    common = np.broadcast_to(contexts[:, None, :], (n, m, len(CONTEXT)))
    types = np.eye(len(KINDS))[kinds]
    return {"stage": packets[0]["stage"], "context": contexts, "mask": mask, "kinds": kinds, "c": c, "w": w,
        "dc": np.concatenate((types, c[:, :, None], common), axis=2),
        "dw": np.concatenate((types, w[:, :, None], common), axis=2)}


BASELINE_NAMES = (*CONTEXT, *(f"{k}:{stat}" for k in KINDS for stat in ("count", "C", "W", "C_and_W")))


def graph_features(packets):
    data = compile_packets(packets)
    parts = [data["context"]]
    for i in range(len(KINDS)):
        applicable = data["mask"] & (data["kinds"] == i)
        parts.extend(np.sum(applicable * value, axis=1)[:, None] for value in (1., data["c"], data["w"], data["c"] * data["w"]))
    return np.concatenate(parts, axis=1)


def direct_features(packets):
    data = compile_packets(packets)
    data["design"] = data["dc"] if data["stage"] == 1 else np.concatenate((data["dc"], data["w"][:, :, None]), axis=2)
    return data


class ModelIndependentPrivacyBN(PrivacyStageBN):
    def __init__(self, *, coupled=True, regularization=.001, seed=0):
        super().__init__(coupled=coupled, regularization=regularization, seed=seed)
        keep = [i for i, name in enumerate(self.names) if ":model:" not in name]
        self.names = [self.names[i] for i in keep]
        self.theta = self.theta[keep]
        self.d, self.k = len(KINDS) + 1 + len(CONTEXT), 1 + len(CONTEXT)
        self.boost_start = 2 * self.d + 2 * self.k
        if len(self.theta) != 2 * self.d + 2 * self.k + 2 + int(self.coupled):
            raise ValueError("identity-free parameter layout mismatch")

    def fit(self, packets, final_labels, *, max_iter=1000):
        data = compile_packets(packets)
        labels = outcome_matrix(data, final_labels)
        bounds = [(-12., 12.)] * len(self.theta)
        bounds[self.boost_start:self.boost_start + 2] = [(0., 12.)] * 2
        self.theta, self.optimizer = _solve(lambda t: self.loss_gradient(t, data, labels), self.theta, max_iter, bounds=bounds)
        return self

    def predict_details(self, packets):
        result = self.predict_compiled(compile_packets(packets))
        result["model_identity_used"] = False
        result["meaning"] = "Model-identity-free statistical privacy forecast, not evidence of calibration on unseen generators."
        return result

    def parameters(self):
        saved = super().parameters()
        saved.update(model="ModelIndependentPrivacyBN-v1", observation_schema=SCHEMA, context_names=list(CONTEXT),
            scope="No generator-model identity features; source distribution and fixed privacy workflow assumptions still apply.")
        return saved

    @classmethod
    def restore(cls, saved):
        if saved["model"] != "ModelIndependentPrivacyBN-v1" or saved["observation_schema"] != SCHEMA or saved["context_names"] != list(CONTEXT):
            raise ValueError("explicit identity-free model required")
        model = cls(coupled=saved["coupled"], regularization=saved["regularization"])
        if model.names != saved["names"]:
            raise ValueError("saved model layout mismatch")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite saved coefficients required")
        model.optimizer = saved["optimizer"]
        return model
