"""Message and local-event cues share one normalized probability law.

Cues are classifier hypotheses about task relevance, NOT permissions or
observations of future outputs. Zero cues reduce to the old count-only law.
"""
import copy

import numpy as np

from .message_hurdle import GATE_FEATURES, HURDLE_SPEC, MessageHurdleLaw, MessageHurdleWorkflowRiskBN


class TaskCueMessageLaw(MessageHurdleLaw):
    gate_features = GATE_FEATURES + ("cue_support_fraction", "cue_opposition_fraction")
    message_spec = {**copy.deepcopy(HURDLE_SPEC), "schema": "task-cue-message-hurdle-v1",
        "gate_features": list(gate_features),
        "cue_meaning": "Noisy initial-task relevance hypotheses; do not grant authority or establish semantic truth."}

    def gate_design(self, dag, group, values):
        base = super().gate_design(dag, group, values)
        cues = np.asarray([dag.lookup[n]["cues"] for n in group], float).mean(axis=0)
        return np.column_stack((base, np.tile(cues, (len(base), 1))))


class TaskCueMessageWorkflowRiskBN(MessageHurdleWorkflowRiskBN):
    MODEL = "TaskCueMessageWorkflowRiskBN-v1"
    law_class = TaskCueMessageLaw


class SemanticArrivalMessageLaw(TaskCueMessageLaw):
    """Task relevance is paired with the matching field's upstream state.

    A functional edge alone is not arrival. Each child's declared input events
    provide binary states; unknown future states stay inside joint inference.
    The two new features average cue_j * mean(parent_states_j) over fields.
    """
    gate_features = TaskCueMessageLaw.gate_features + ("supported_arrival_fraction", "opposed_arrival_fraction")
    message_spec = {**copy.deepcopy(TaskCueMessageLaw.message_spec), "schema": "semantic-arrival-message-hurdle-v1",
        "gate_features": list(gate_features),
        "arrival_meaning": "For each named field, pair its downstream task cue with the mean of its own input event states; average those products across message fields. Not receipt attestation, adoption, authority or causal identification."}

    def gate_design(self, dag, group, values):
        base = super().gate_design(dag, group, values)
        paired = np.zeros((len(base), 2))
        for identity in group:
            parents = dag.parents[identity]
            mean = np.asarray([values[p] for p in parents]).mean(axis=0) if parents else np.zeros(len(base))
            paired += mean[:, None] * np.asarray(dag.lookup[identity]["cues"])[None, :]
        return np.column_stack((base, paired / len(group)))


class SemanticArrivalWorkflowRiskBN(MessageHurdleWorkflowRiskBN):
    MODEL = "SemanticArrivalWorkflowRiskBN-v1"
    law_class = SemanticArrivalMessageLaw
