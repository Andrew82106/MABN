"""Label-free, prefix-only inputs for evaluating a particular tool dispatch.

This is a data interface, not a new trained model or a future-outcome target.
Evaluation code joins retrospective labels separately by invocation identity.
"""
from __future__ import annotations

import copy

from .initial_task_context import from_initial_request
from .journal import RuntimeEvidenceProjector
from .semantic_evidence import digest


def dispatch_inputs(rows, initial_request):
    if not rows:
        raise ValueError("initial runtime record required")
    context = from_initial_request(initial_request, rows[0])
    # Only content which actually existed at run start; no later classification.
    visible_context = {"goal": context["goal"], "actors": context["actors"]}
    projector, events, samples = RuntimeEvidenceProjector(), [], []
    for row in rows:
        event = projector.push(row)
        if event is not None:
            events.append(event)
        if row["kind"] == "tool_handler_started":
            samples.append({"sample_id": row["payload"]["invocation_id"],
                            "journal_sequence": row["sequence"],
                            "observation_count": len(events),
                            "structured_arguments": event["event_type"] == "tool_call"})
    return {"schema": "dispatch-prefix-inputs-v1", "context": visible_context,
            "events": events, "samples": samples,
            "meaning": "Slice with materialize_prefix; never give a scorer the full future-containing document."}


def materialize_prefix(inputs, sample_id):
    matches = [s for s in inputs["samples"] if s["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError("unique dispatch sample required")
    sample = matches[0]
    count = sample["observation_count"]
    if type(count) is not int or not 0 < count <= len(inputs["events"]):
        raise ValueError("invalid prefix length")
    events = inputs["events"][:count]
    if (events[-1]["sequence"] != sample["journal_sequence"]
            or events[-1].get("action_id") != sample_id
            or events[-1]["event_type"] not in {"tool_call", "unparsed_tool_entry"}
            or any(e["sequence"] > sample["journal_sequence"] for e in events)):
        raise ValueError("prefix must end at the matching pre-handler observation")
    return copy.deepcopy({"context": inputs["context"], "events": events})


def prefix_digest(inputs, sample_id):
    return digest(materialize_prefix(inputs, sample_id))
