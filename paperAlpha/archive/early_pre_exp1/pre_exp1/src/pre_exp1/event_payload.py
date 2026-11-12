"""Canonical event payload hashing shared by writers and validators."""

from __future__ import annotations

import copy
from typing import Any

from .state import stable_hash


CANONICAL_PAYLOAD_FIELD = "canonical_payload"


def canonical_event_payload(payload: Any) -> Any:
    """Return the JSON-compatible payload whose hash is stored on an event."""

    return copy.deepcopy(payload)


def event_payload_hash(payload: Any) -> str:
    return stable_hash(canonical_event_payload(payload))

