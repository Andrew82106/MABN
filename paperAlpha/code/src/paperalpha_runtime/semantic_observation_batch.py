"""Item-wise structural acceptance for typed, still-unverified text claims.

One invalid proposal does not erase unrelated well-anchored proposals. The
strict v2 item validator is unchanged: nothing is repaired, reclassified, or
promoted to a verified fact. Conflicting duplicate identities are not resolved
by picking whichever proposal happened to appear first.
"""
from __future__ import annotations

from collections import defaultdict

from .semantic_evidence import digest, exact_keys, STATUSES
from .semantic_evidence_v2 import MAX_OBSERVATIONS, validate_annotation


def validate_items(packet, value, *, strict_validator=validate_annotation):
    """Return all item decisions; malformed top-level envelopes still fail.

    ``strict_validator`` lets an offline derivation use the original run's
    frozen v2 validator rather than silently upgrading historical semantics.
    Items from one record remain one correlated observation group, not votes.
    """
    exact_keys(value, ("status", "more_possible", "observations"), "annotation-v2")
    status, items = value["status"], value["observations"]
    if status not in STATUSES or type(value["more_possible"]) is not bool:
        raise ValueError("invalid status/coverage indicator")
    if not isinstance(items, list) or len(items) > MAX_OBSERVATIONS or bool(items) != (status == "observations"):
        raise ValueError("status and observation list must agree")
    decisions, identities = [], defaultdict(list)
    for index, item in enumerate(items):
        decision = {"item_index": index, "proposal_sha256": digest(item)}
        try:
            observation = strict_validator(packet, {"status": "observations",
                "more_possible": value["more_possible"], "observations": [item]})["observations"][0]
        except (ValueError, KeyError, TypeError) as exc:
            decision.update(status="rejected", reason="invalid_item", detail=str(exc))
        else:
            decision.update(status="accepted", observation=observation)
            anchor = observation["anchor"]
            # One sentence can legitimately contain two caps or two conditions.
            # Identity therefore includes the exact subordinate proposition,
            # but not references (adding a reference cannot create a new vote).
            quantity = observation["quantity_anchor"]
            quantity_span = None if quantity is None else (quantity["start"], quantity["end"])
            conditions = tuple(sorted((part["start"], part["end"]) for part in observation["condition_anchors"]))
            identity = (observation["predicate"], observation["presentation"], anchor["start"], anchor["end"],
                        quantity_span, conditions)
            identities[identity].append(index)
        decisions.append(decision)
    for duplicates in identities.values():
        if len(duplicates) > 1:
            for index in duplicates:
                decisions[index].pop("observation")
                decisions[index].update(status="rejected", reason="duplicate_identity",
                    detail="All proposals with this predicate/presentation/main-and-subspan identity are excluded; no first-item selection.",
                    duplicate_item_indices=list(duplicates))
    accepted = sum(d["status"] == "accepted" for d in decisions)
    current = packet["current"]
    return {"schema_version": "text-evidence-item-batch-v2", "extraction_protocol": "text-evidence-v2",
        "record_id": current["record_id"], "record_sha256": current["sha256"], "sequence": current["sequence"],
        "input_packet_sha256": digest(packet), "proposal_sha256": digest(value),
        "extraction_status": status, "extractor_more_possible": value["more_possible"],
        "acceptance_status": "partial" if 0 < accepted < len(items) else
            "rejected" if items and not accepted else "all_items_accepted" if items else "empty",
        "proposed_items": len(items), "accepted_items": accepted, "rejected_items": len(items) - accepted,
        "items": decisions, "observation_group_id": current["sha256"],
        "identity_contract": "predicate, presentation, main span, quantity span, sorted condition spans; references excluded",
        "independent_evidence": False, "exhaustive_coverage_established": False,
        "authorization_established": False, "safety_established": False,
        "meaning": "Item-level schema and anchor checks only. All retained classifications, conditions and relations remain hypotheses."}
