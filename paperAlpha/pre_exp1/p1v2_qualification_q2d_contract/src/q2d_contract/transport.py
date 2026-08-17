"""Deterministic in-memory transport used only by the offline Q2-D runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .contract import canonical_json


class FakeTransportError(RuntimeError):
    """Safe infrastructure category; contains no server response body."""


class FakeTimeoutError(FakeTransportError):
    pass


class FakeHTTPError(FakeTransportError):
    pass


@dataclass(frozen=True)
class FakeResponse:
    model: str
    raw: str
    metadata_present: bool = True


class FakeTransport:
    """An in-memory completion transport with explicit zero side effects."""

    def __init__(self, *, model: str = "q2d-fake-model", raw_overrides: dict[str, Any] | None = None, faults: dict[str, str] | None = None, identity_overrides: dict[str, str] | None = None, metadata_missing: set[str] | None = None) -> None:
        self.model = model
        self.raw_overrides = dict(raw_overrides or {})
        self.faults = dict(faults or {})
        self.identity_overrides = dict(identity_overrides or {})
        self.metadata_missing = set(metadata_missing or set())
        self.completion_calls = 0
        self.tool_execution_calls = 0
        self.network_calls = 0
        self.env_reads = 0

    def metadata(self) -> dict[str, Any]:
        return {"transport": "fake", "model": self.model, "network_calls": self.network_calls}

    def complete(self, request: dict[str, Any], card: dict[str, Any]) -> FakeResponse:
        self.completion_calls += 1
        card_id = card["card_id"]
        fault = self.faults.get(card_id)
        if fault == "timeout":
            raise FakeTimeoutError("timeout")
        if fault == "http":
            raise FakeHTTPError("http")
        if fault == "deadline":
            raise FakeTimeoutError("deadline")
        if fault == "budget":
            raise FakeTransportError("budget")
        if fault == "metadata_missing":
            return FakeResponse(model=self.model, raw=canonical_json(card["expected_action"]), metadata_present=False)
        model = self.identity_overrides.get(card_id, self.model)
        raw = self.raw_overrides.get(card_id, canonical_json(card["expected_action"]))
        if isinstance(raw, dict):
            raw = json.dumps(raw, ensure_ascii=False, allow_nan=False)
        return FakeResponse(model=model, raw=str(raw), metadata_present=card_id not in self.metadata_missing)

    def execute_tool(self, *_args: Any, **_kwargs: Any) -> None:
        self.tool_execution_calls += 1
        raise AssertionError("Q2-D fake transport must never execute tools")


__all__ = ["FakeTransport", "FakeResponse", "FakeTransportError", "FakeTimeoutError", "FakeHTTPError"]

