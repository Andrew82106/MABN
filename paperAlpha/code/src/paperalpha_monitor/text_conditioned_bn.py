"""Text-conditioned process-state DBN, an opt-in v5 research model.

This changes the local state meaning: C is a learned process-concern hypothesis,
not recovered adoption/compromise truth. One grouped text observation conditions
a reversible binary transition. It is not a product of independent claim
likelihoods. Existing action-scope checks and exact history elimination remain.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from paperalpha_runtime.journal import RuntimeEvidenceProjector
from paperalpha_runtime.semantic_evidence import TextEvidenceWindow, PRESENTATIONS, digest
from paperalpha_runtime.semantic_evidence_v2 import PREDICATES
from paperalpha_runtime.semantic_observation_batch import validate_items
from .compact_action_bn import CompactActionCompiler, CompactCapabilityBN, CompactFilter
from .capability_action_bn import PARAMETER_NAMES as BASE_PARAMETERS
from .dynamic_joint_bn import JointTrace, state_bits


FEATURE_NAMES = tuple(f"claim:{predicate}:{presentation}" for predicate in PREDICATES for presentation in PRESENTATIONS) + (
    "has_condition_anchor", "has_quantity_anchor", "has_relation_proposal", "has_unresolved_comparison",
    "some_items_rejected", "extractor_more_possible", "no_supported_claims", "extractor_uncertain",
    "extraction_unavailable", "prior_context_omitted")
CHANNELS = ("communication", "tool_handler_returned")
BASE_SIZE = len(BASE_PARAMETERS)
CHANNEL_WIDTH = 2 + len(FEATURE_NAMES)
PARAMETER_NAMES = (*BASE_PARAMETERS, *(f"text:{channel}:{name}" for channel in CHANNELS
    for name in ("bias", "previous_state", *FEATURE_NAMES)))
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
MODEL_ID = "TextConditionedCapabilityBN-v5"
MODEL_SPEC = {"id": MODEL_ID, "features": FEATURE_NAMES, "channels": CHANNELS,
    "parameters": PARAMETER_NAMES, "item_protocol": "text-evidence-item-batch-v2",
    "transition": "sigmoid(bias + previous_state_weight*C_previous + weights @ grouped_features)",
    "order": "revise observed actor, then actual communication, then action dispatch, then global readout",
    "state_meaning": "process-concern hypothesis, not verified information adoption or compromise",
    "observation_model": "conditional transition; no independent-feature likelihood or calibrated semantic truth"}
MODEL_SHA256 = digest(MODEL_SPEC)


@dataclass(frozen=True)
class TextRevision:
    node: int
    channel: int
    features: np.ndarray
    evidence: dict


@dataclass(frozen=True)
class TextStep:
    base: object
    revision: TextRevision | None

    @property
    def sequence(self):
        return self.base.sequence


def grouped_features(packet, annotation):
    """Binary presence, never item counts, risk labels or trusted permissions."""
    features = np.zeros(len(FEATURE_NAMES))
    features[FEATURE_INDEX["prior_context_omitted"]] = bool(packet["context"]["omitted_prior_records"])
    batch, error = None, None
    if annotation is not None:
        try:
            batch = validate_items(packet, annotation)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
    if batch is None:
        features[FEATURE_INDEX["extraction_unavailable"]] = 1
    else:
        features[FEATURE_INDEX["some_items_rejected"]] = batch["rejected_items"] > 0
        features[FEATURE_INDEX["extractor_more_possible"]] = batch["extractor_more_possible"]
        features[FEATURE_INDEX["no_supported_claims"]] = batch["extraction_status"] == "no_supported_claims"
        features[FEATURE_INDEX["extractor_uncertain"]] = batch["extraction_status"] == "uncertain"
        for decision in batch["items"]:
            if decision["status"] != "accepted":
                continue
            obs = decision["observation"]
            features[FEATURE_INDEX[f"claim:{obs['predicate']}:{obs['presentation']}"]] = 1
            flags = {"has_condition_anchor": bool(obs["condition_anchors"]),
                "has_quantity_anchor": obs["quantity_anchor"] is not None,
                "has_relation_proposal": bool(obs["references"]),
                "has_unresolved_comparison": any(r["relation_hypothesis"] == "unresolved_comparison" for r in obs["references"])}
            for name, value in flags.items():
                features[FEATURE_INDEX[name]] = max(features[FEATURE_INDEX[name]], value)
    current = packet["current"]
    evidence = {"record_id": current["record_id"], "record_sha256": current["sha256"],
        "input_packet_sha256": digest(packet), "batch": batch, "invalid_envelope_reason": error,
        "feature_names": [name for name, value in zip(FEATURE_NAMES, features) if value],
        "authorization_established": False, "safety_established": False,
        "meaning": "Grouped noisy textual covariates, not independent evidence or state labels."}
    return features, evidence


class JournalTextCompiler:
    """Only a runtime prefix can construct text context and actor bindings.

    Missing/invalid annotations remain explicit covariates. This synchronous
    research interface does not account for extraction latency; offline replay
    must not be reported as real-time early warning.
    """
    def __init__(self, **context):
        self.projector, self.window = RuntimeEvidenceProjector(), TextEvidenceWindow()
        self.base = CompactActionCompiler(**context)
        self.last_step = None

    @property
    def variables(self):
        return self.base.variables

    @property
    def workflow(self):
        return self.base.workflow

    def push(self, row, annotation=None):
        # Atomic failure, including window/history and the underlying compiler.
        # Metadata copying is not a scalability claim.
        candidate = copy.deepcopy(self)
        result = candidate._push(row, annotation)
        self.__dict__.update(candidate.__dict__)
        return result

    def _push(self, row, annotation):
        event, packet = self.projector.push(row), self.window.push(row)
        if packet is None and annotation is not None:
            raise ValueError("text annotation cannot be attached to a non-text runtime record")
        if event is None:
            return None
        base_step = self.base.push(event)
        revision = None
        if packet is not None:
            features, evidence = grouped_features(packet, annotation)
            actor = packet["current"]["actor"]
            node = (self.workflow or {}).get("agent_to_node", {}).get(actor, actor)
            index = next(i for i, v in enumerate(self.variables) if v["kind"] == "information_state" and v["id"] == node)
            revision = TextRevision(index, CHANNELS.index(packet["current"]["kind"]), features, evidence)
        self.last_step = TextStep(base_step, revision)
        return copy.deepcopy(self.last_step)


def compile_text_journal(rows, annotations=None, **context):
    compiler, steps = JournalTextCompiler(**context), []
    annotations = {} if annotations is None else annotations
    seen = set()
    for row in rows:
        identity = f"journal-{row['sequence']}"
        if identity in annotations:
            seen.add(identity)
        step = compiler.push(row, annotations.get(identity))
        if step is not None:
            steps.append(step)
    if set(annotations) != seen:
        raise ValueError("annotations outside this journal prefix are not accepted")
    if not steps:
        raise ValueError("a nonempty model-event prefix is required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), tuple(steps), compiler.workflow)


class TextConditionedFilter(CompactFilter):
    parameter_count, step_type = len(PARAMETER_NAMES), TextStep

    def __init__(self, theta, *, use_text=True, mask_text_features=False, **options):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.parameter_count,) or not np.isfinite(theta).all():
            raise ValueError("finite text-v5 parameter vector required")
        super().__init__(theta[:BASE_SIZE], **options)
        self.theta = theta.copy()
        if self.derivatives:
            self.jacobian = np.zeros((1, 3, len(theta)))
        self.use_text, self.mask_text_features = use_text, mask_text_features
        self.last_revision = None

    def _revise(self, revision):
        start = BASE_SIZE + revision.channel * CHANNEL_WIDTH
        x = np.zeros_like(revision.features) if self.mask_text_features else revision.features
        logit = self.theta[start] + self.theta[start + 2:start + CHANNEL_WIDTH] @ x
        probabilities = expit([logit, logit + self.theta[start + 1]])
        q0, q1 = probabilities
        states = np.arange(len(self.distribution))
        low = states[(states & (1 << revision.node)) == 0]
        high = low | (1 << revision.node)
        a, b = self.distribution[low].copy(), self.distribution[high].copy()
        self.distribution[low] = (1 - q0) * a + (1 - q1) * b
        self.distribution[high] = q0 * a + q1 * b
        if self.derivatives:
            d0, d1 = np.zeros(len(self.theta)), np.zeros(len(self.theta))
            d0[start] = d1[start] = 1
            d1[start + 1] = 1
            d0[start + 2:start + CHANNEL_WIDTH] = d1[start + 2:start + CHANNEL_WIDTH] = x
            d0 *= q0 * (1 - q0)
            d1 *= q1 * (1 - q1)
            ja, jb = self.jacobian[low].copy(), self.jacobian[high].copy()
            change = a[:, :, None] * d0 + b[:, :, None] * d1
            self.jacobian[low] = (1 - q0) * ja + (1 - q1) * jb - change
            self.jacobian[high] = q0 * ja + q1 * jb + change
        self.last_revision = {**copy.deepcopy(revision.evidence), "node_index": revision.node,
            "channel": CHANNELS[revision.channel], "q_if_previous_zero": float(q0), "q_if_previous_one": float(q1),
            "feature_mask_applied": self.mask_text_features, "probabilities_calibrated": False}

    def apply(self, step, variables):
        if type(step) is not self.step_type:
            raise ValueError("text-v5 requires its own compiled step")
        self._expand(len(variables))
        self.last_revision = None
        if self.use_text and step.revision is not None:
            self._revise(step.revision)
        # Base conditional head consumes only its 23 coefficients; extra text
        # coefficients stay in the Jacobian, not in the readout feature matrix.
        probability = super().apply(step.base, variables)
        return probability

    def explain(self, step, variables):
        result = super().explain(step.base, variables)
        for variable in result["variables"]:
            if variable["kind"] == "information_state":
                variable.update(kind="process_concern_hypothesis", risk_channel="text_conditioned_process")
        for action in result["actions"]:
            action["process_concern_at_dispatch_hypothesis"] = action.pop("information_at_dispatch_hypothesis")
        result.update(text_revision=copy.deepcopy(self.last_revision),
            state_interpretation=MODEL_SPEC["state_meaning"], historical_field_note=
            "Inherited information/overlap summary names denote the v5 process-concern bit, not proven contamination.")
        return result


class TextConditionedBN(CompactCapabilityBN):
    model_name, step_type, parameter_names = MODEL_ID, TextStep, PARAMETER_NAMES

    def __init__(self, *, use_text=True, mask_text_features=False, **options):
        self.use_text, self.mask_text_features = use_text, mask_text_features
        super().__init__(**options)

    def initial_parameters(self):
        base = super().initial_parameters()
        channel = np.r_[-4., 8., np.zeros(len(FEATURE_NAMES))]
        return np.r_[base, channel, channel]

    def _engine(self, theta, *, derivatives=False):
        return TextConditionedFilter(theta, use_text=self.use_text, mask_text_features=self.mask_text_features,
            use_messages=self.use_messages, use_latent_head=self.use_latent_head, derivatives=derivatives)

    def parameters(self):
        return {"name": MODEL_ID, "specification_sha256": MODEL_SHA256, "specification": MODEL_SPEC,
            "parameter_names": list(PARAMETER_NAMES), "theta": self.theta.tolist(), "optimizer": copy.deepcopy(self.optimizer),
            "regularization": self.regularization, "max_iter": self.max_iter,
            "use_text": self.use_text, "mask_text_features": self.mask_text_features,
            "use_messages": self.use_messages, "use_latent_head": self.use_latent_head,
            "training": "episode-balanced prefix BCE, analytic gradients through conditional transitions and exact action elimination",
            "limitations": ["No independent semantic calibration or real MAS accuracy evaluation.",
                "Process-state transitions are conditional modeling assumptions, not causal or compromise truth.",
                "No cross-record source identity, numeric-scope comparison or role-specific risk subgraph compiler.",
                "Repeated rewordings may still influence state; grouped features are not a source de-correlation solution.",
                "Synchronous offline text availability excludes API extraction delay; no real-time early-warning claim.",
                "Exact state still exponential in information/source nodes; metadata copying grows with history."]}
