"""Literal task-to-record references, not permission or semantic identity truth.

The adapter declares which public schema fields identify individuals or groups.
Only original task text and initial records are inspected. Names of cases,
domains, templates, attacks, outcomes and allowed disclosure sets are not inputs.
No match means no observed literal reference, not that a record is irrelevant.
"""
from __future__ import annotations

import copy
import re

from .canonical import digest


FEATURES = ("individual_reference", "ambiguous_individual_reference",
            "group_reference", "no_literal_reference")


def bind_record_references(request, records, reference_fields):
    """Bind exact bounded references with original-text spans and alias dedup.

    `reference_fields` maps a schema field to `individual` or `group`. A name
    shared by two records is an ambiguous individual reference; departments
    remain group references. Neither event authorizes any disclosure.
    Matching ignores case but preserves original offsets (Unicode code points).
    Non-string, empty and shorter-than-four-character values are not matched.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("nonempty original request required")
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty initial record list required")
    if not isinstance(reference_fields, dict) or any(
            not isinstance(k, str) or not k or v not in {"individual", "group"}
            for k, v in reference_fields.items()):
        raise ValueError("explicit individual/group schema declarations required")
    identities = []
    candidates = {}
    unmatchable = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("fields"), dict):
            raise ValueError("record fields must be a named mapping")
        identity = record.get("record_id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("distinct nonempty record IDs required")
        identities.append(identity)
        for field, kind in reference_fields.items():
            value = record["fields"].get(field)
            if not isinstance(value, str) or len(value.strip()) < 4:
                unmatchable.append({"record_id": identity, "field_name": field,
                                    "reason": "absent_nonstring_or_short"})
                continue
            pattern = (r"(?<!\w)" if value[0].isalnum() else "") + re.escape(value)
            pattern += r"(?!\w)" if value[-1].isalnum() else ""
            for match in re.finditer(pattern, request, flags=re.IGNORECASE):
                # One actual span/kind is one reference, even under field aliases.
                key = (kind, match.start(), match.end())
                candidates.setdefault(key, {}).setdefault(identity, []).append(field)
    evidence = []
    features = {identity: dict.fromkeys(FEATURES, 0) for identity in identities}
    for (kind, start, end), matches in sorted(candidates.items()):
        ambiguous = kind == "individual" and len(matches) > 1
        column = ("ambiguous_individual_reference" if ambiguous else
                  "individual_reference" if kind == "individual" else "group_reference")
        for identity in matches:
            features[identity][column] = 1
        evidence.append({"kind": kind, "start": start, "end": end,
                         "quote": request[start:end],
                         "records": [{"record_id": identity, "fields": sorted(set(fields))}
                                     for identity, fields in sorted(matches.items())],
                         "ambiguous_individual_reference": ambiguous})
    for row in features.values():
        row["no_literal_reference"] = int(not any(row.values()))
    return {"schema": "initial-record-reference-binding",
            "initial_sha256": digest({"request": request, "records": records,
                                      "reference_fields": reference_fields}),
            "reference_fields": copy.deepcopy(reference_fields),
            "features": features, "evidence": evidence,
            "unmatchable_fields": unmatchable,
            "grants_permission": False, "semantic_identity_verified": False,
            "absence_means_irrelevance": False,
            "scope": "Declared-schema literal references in initial task text only; no future outputs or privacy policy."}
