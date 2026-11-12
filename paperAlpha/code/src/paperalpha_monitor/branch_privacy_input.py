"""Branch-preserving observation contract for the fixed parallel-worker DAG.

Not an automatic DAG learner and not itself a branch-aware probability model.
Ownership is explicit host metadata; occurrences come from each actual message.
"""
from __future__ import annotations

import copy

import numpy as np

from paperalpha_runtime.branched_publisher_data import WORKFLOW, joined_workers
from .scoped_privacy_monitor import compile_observation
from .policy_gated_privacy import ScopedRelease
from .publisher_view_bn import PublisherInput, ViewSeparatedRelease


class BranchPrivacyInput:
    FIELDS = {"schema", "workflow", "field_owners", "worker_messages", "publisher_payload", "pooled_scope", "projection_scope"}

    def __init__(self, observation, *, available_journal_sequence, prefix_journal_sequence):
        self.observation = copy.deepcopy(observation)
        self.available_journal_sequence = available_journal_sequence
        self.prefix_journal_sequence = prefix_journal_sequence
        o = self.observation
        if set(o) != self.FIELDS or o["schema"] != "branched-publisher-prefix-v1" or o["workflow"] != WORKFLOW:
            raise ValueError("declared fixed four-role DAG and strict prefix fields required")
        self.workers = tuple(WORKFLOW["worker_order"])
        if set(o["worker_messages"]) != set(self.workers):
            raise ValueError("one actual completed message for every worker required")
        merged = joined_workers(o["worker_messages"])
        if o["pooled_scope"]["worker_text"] != merged or o["publisher_payload"]["earlier_agent_message"] != merged:
            raise ValueError("pooled and publisher views must bind the same actual branch messages")
        source = PublisherInput.capture(o["publisher_payload"], transport_observer="api-request-journal",
            captured_journal_sequence=available_journal_sequence, prefix_journal_sequence=prefix_journal_sequence)
        original = compile_observation(o["pooled_scope"])
        registry = o["pooled_scope"]["tracked_fields"]
        all_names = {name for field in registry for name in field["field_names"]}
        if not isinstance(o["field_owners"], dict) or set(o["field_owners"]) != all_names:
            raise ValueError("host ownership must cover exactly the tracked aliases")
        owners, matrix = [], []
        for field in registry:
            declared = {o["field_owners"][name] for name in field["field_names"]}
            if len(declared) != 1 or not declared <= set(self.workers):
                raise ValueError("one declared worker owner per distinct literal group")
            owners.append(self.workers.index(next(iter(declared))))
            literal = field["literal"].lower()
            matrix.append([literal in o["worker_messages"][name].lower() for name in self.workers])
        self.owner_indices = tuple(owners)
        self.worker_presence = np.asarray(matrix, dtype=bool)
        self.declared_owner_mask = np.eye(len(self.workers), dtype=bool)[list(owners)]
        # Match actual per-worker text, not the serialized container. Values
        # containing quotes/backslashes can change form in JSON serialization.
        packet = copy.deepcopy(original.packet)
        union = self.worker_presence.any(axis=1)
        self.legacy_pool_agrees = all(g["worker_present"] == bool(p) for g, p in zip(packet["groups"], union))
        for group, present in zip(packet["groups"], union):
            group["worker_present"] = bool(present)
        scoped = ScopedRelease(packet, original.graph, original.context, original.bindings)
        self.release = ViewSeparatedRelease(scoped, source, trusted_transport_observers=["api-request-journal"])

    def validate(self):
        rebuilt = BranchPrivacyInput(self.observation, available_journal_sequence=self.available_journal_sequence,
            prefix_journal_sequence=self.prefix_journal_sequence)
        if (rebuilt.workers != self.workers or rebuilt.legacy_pool_agrees != self.legacy_pool_agrees
                or rebuilt.owner_indices != self.owner_indices or not np.array_equal(rebuilt.worker_presence, self.worker_presence)
                or not np.array_equal(rebuilt.declared_owner_mask, self.declared_owner_mask)):
            raise ValueError("bound branch evidence changed")
        self.release.validate()
        if rebuilt.release.packet != self.release.packet or rebuilt.release.specification() != self.release.specification():
            raise ValueError("derived release evidence changed")

    def specification(self):
        return {"workflow": copy.deepcopy(WORKFLOW), "workers": list(self.workers), "owner_indices": list(self.owner_indices),
            "worker_presence": self.worker_presence.astype(int).tolist(), "declared_owner_mask": self.declared_owner_mask.astype(int).tolist(),
            "legacy_serialized_pool_agrees": self.legacy_pool_agrees, "release": self.release.specification(),
            "scope": "Bound multi-worker evidence; not yet a model with independent per-worker states. Ownership is host-declared, not inferred capability."}
