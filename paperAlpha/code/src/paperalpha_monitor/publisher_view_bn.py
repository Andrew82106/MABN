"""Separate observable publisher authority cues from current policy truth.

Visible cues describe a captured request, NOT the model's internal beliefs and
NOT current validity. Missing capture is different from a captured revocation.
The latent truth graph still controls H=F AND reference_failure.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from paperalpha_runtime.canonical import digest
from .delegation_evidence import DelegationLedger
from .policy_gated_privacy import ScopedRelease, PolicyGatedPrivacyBN


@dataclass(frozen=True)
class PublisherInput:
    payload: dict
    payload_sha256: str
    transport_observer: str
    captured_journal_sequence: int
    prefix_journal_sequence: int

    def validate(self):
        if not isinstance(self.payload, dict) or digest(self.payload) != self.payload_sha256:
            raise ValueError("captured input payload changed")
        if not isinstance(self.transport_observer, str) or not self.transport_observer:
            raise ValueError("explicit transport observer required")
        if (type(self.captured_journal_sequence) is not int or type(self.prefix_journal_sequence) is not int
                or not 0 < self.captured_journal_sequence <= self.prefix_journal_sequence):
            raise ValueError("publisher input must be available by this journal prefix")

    @classmethod
    def capture(cls, payload, *, transport_observer, captured_journal_sequence, prefix_journal_sequence):
        value = copy.deepcopy(payload)
        result = cls(value, digest(value), transport_observer, captured_journal_sequence, prefix_journal_sequence)
        result.validate()
        return result


class ViewSeparatedRelease(ScopedRelease):
    """One release with two deliberately distinct evidence sources and clocks.

    Authority-record sequences retain the release graph's logical clock. Input
    capture/prefix sequences use a separate transport journal. A trusted adapter
    must supply both; this class is neither a transport nor a signature verifier.
    """
    PAYLOAD_FIELDS = {"user_request", "task_id", "release_context", "cached_authority_records", "release_bindings", "earlier_agent_message"}

    def __init__(self, release, publisher_input, *, trusted_transport_observers):
        if not isinstance(release, ScopedRelease):
            raise ValueError("existing scoped truth query required")
        ScopedRelease.__init__(self, release.packet, release.graph, release.context, release.bindings)
        self.publisher_input = copy.deepcopy(publisher_input)
        self.trusted_transport_observers = tuple(trusted_transport_observers)
        if not self.trusted_transport_observers or any(not isinstance(v, str) or not v for v in self.trusted_transport_observers):
            raise ValueError("explicit trusted transport adapters required")
        self.visible_features, self.visible_evidence = self._derive_visible()

    def _derive_visible(self):
        if self.publisher_input is None:
            return np.zeros((len(self.action_ids), 2)), {"capture": "missing", "states": ["unobserved_input"] * len(self.action_ids)}
        source = self.publisher_input
        if not isinstance(source, PublisherInput):
            raise ValueError("typed captured publisher request or explicit None required")
        if self.packet["stage"] != 2:
            raise ValueError("a final request containing the worker reply is unavailable at the pre-worker cutoff")
        source.validate()
        if source.transport_observer not in self.trusted_transport_observers:
            raise ValueError("untrusted request capture")
        p = source.payload
        if set(p) != self.PAYLOAD_FIELDS:
            raise ValueError("strict prior-input payload required; no labels or final response")
        if not isinstance(p["user_request"], str) or not isinstance(p["earlier_agent_message"], str):
            raise ValueError("actual input strings required")
        expected_context = {k: getattr(self.context, k) for k in ("actor", "principal", "recipient", "purpose")}
        if p["task_id"] != self.context.task_id or p["release_context"] != expected_context:
            raise ValueError("publisher input belongs to a different task/actor/recipient/purpose")
        if not isinstance(p["cached_authority_records"], list) or not isinstance(p["release_bindings"], list):
            raise ValueError("explicit captured authority records and bindings required")
        # Only the records actually delivered to this actor are processed. Never
        # fill a cache with revocations from the monitor's truth graph.
        first = self.graph.proofs[self.action_ids[0]].evidence
        ledger = DelegationLedger(root_principals=first["root_principals"], trusted_observers=first["trusted_observers"], coverage_complete=False)
        for event in p["cached_authority_records"]:
            ledger.observe(event)
        mapping = {}
        for binding in p["release_bindings"]:
            if not isinstance(binding, dict) or set(binding) != {"field_names", "resource", "grant_id", "arguments", "arguments_sha256"}:
                raise ValueError("captured exact argument bindings required")
            fields = tuple(sorted(binding["field_names"]))
            if fields in mapping or not fields or len(set(fields)) != len(fields):
                raise ValueError("one visible binding per alias group required")
            mapping[fields] = binding
        if set(mapping) != {tuple(sorted(g["field_names"])) for g in self.packet["groups"]}:
            raise ValueError("visible bindings must cover exactly the release query")
        states, rows = [], []
        for action_id, group in zip(self.action_ids, self.packet["groups"]):
            proof = self.graph.proofs[action_id]
            action = proof.evidence["action_binding"]
            binding = mapping[tuple(sorted(group["field_names"]))]
            if (binding["resource"] != action["resource"] or binding["arguments"] != action["arguments"]
                    or binding["arguments_sha256"] != digest(action["arguments"])):
                raise ValueError("publisher argument binding differs from query")
            snapshot = ledger.assess(action, sequence=proof.sequence, task_id=self.context.task_id, grant_ids=[binding["grant_id"]])
            status = snapshot.evidence["reference_status"]
            states.append(status)
            rows.append([float(status == "supported_reference"), float(status in {"invalid_references", "no_reference"})])
        return np.asarray(rows), {"capture": "observed", "states": states, "payload_sha256": source.payload_sha256,
            "transport_observer": source.transport_observer, "captured_journal_sequence": source.captured_journal_sequence,
            "prefix_journal_sequence": source.prefix_journal_sequence,
            "records_promoted_to_current_truth": False,
            "meaning": "Apparent reference support in captured input. Transport capture does not authenticate embedded authority claims; neither current validity nor identified internal belief."}

    def validate(self):
        ScopedRelease.validate(self)
        features, evidence = self._derive_visible()
        if not np.array_equal(features, self.visible_features) or evidence != self.visible_evidence:
            raise ValueError("visible evidence changed after binding")

    def specification(self):
        return {**ScopedRelease.specification(self), "publisher_input": copy.deepcopy(self.visible_evidence),
            "visible_feature_names": ["apparent_support_in_publisher_input", "apparent_contradiction_in_publisher_input"],
            "visible_features": self.visible_features.tolist(),
            "separation": "Publisher input conditions F; current reference truth conditions H. Unknown input is the zero-feature reference category."}


class PublisherViewPrivacyBN(PolicyGatedPrivacyBN):
    MODEL_NAME = "PublisherViewPrivacyBN-v1"
    EMISSION_NAMES = ("publisher_supported_emission", "publisher_contradicted_emission")

    def _emission_features(self, sample, failed):
        if not isinstance(sample, ViewSeparatedRelease):
            raise ValueError("publisher-visible and current truth evidence must be separate")
        return np.broadcast_to(sample.visible_features[:, None, :], (*failed.shape, 2))

    def _emission_graph(self, sample, index, nodes):
        if not isinstance(sample, ViewSeparatedRelease):
            raise ValueError("view-separated release required")
        ids = [f"publisher_supported:{index}", f"publisher_contradicted:{index}"]
        for identity, value in zip(ids, sample.visible_features[index]):
            nodes.append({"id": identity, "parents": [], "kind": "binary_cpt", "probability_one": [float(value)],
                "observed": int(value), "meaning": sample.visible_evidence})
        supported, contradicted = self.theta[self.n + 3:]
        return ids, [0., float(contradicted), float(supported), float(supported + contradicted)]

    def predict(self, sample, *, at_least=1, theta=None):
        result = super().predict(sample, at_least=at_least, theta=theta)
        result["emission_uses_current_authority_truth"] = False
        result["publisher_input_is_internal_belief"] = False
        result["publisher_input_available"] = sample.publisher_input is not None
        return result

    def parameters(self):
        return {**super().parameters(), "scope": "Two evidence views, fixed final message/static current truth window; visible-input cues do not identify beliefs or causal effects"}
