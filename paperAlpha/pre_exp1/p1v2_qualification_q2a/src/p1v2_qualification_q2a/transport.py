"""In-process mock transport only.

This module intentionally contains no HTTP, socket, DNS, subprocess, or environment
access.  A production transport is outside Q2-A and must be separately approved.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import MockTransportError


class MockOpenAITransport:
    """Deterministic injected transport used exclusively by offline tests/readiness."""

    def __init__(
        self,
        scripted_responses: list[str | MockTransportError],
        fake_key: str,
        identity_before: dict[str, Any],
        identity_after: dict[str, Any] | None = None,
    ) -> None:
        self._scripted_responses = list(scripted_responses)
        self._fake_key = fake_key
        self._identity_before = deepcopy(identity_before)
        self._identity_after = deepcopy(identity_after if identity_after is not None else identity_before)
        self._cursor = 0
        self.counts = {
            "mock_transport_calls": 0,
            "mock_inference_calls": 0,
            "mock_identity_checks": 0,
            "real_network_calls": 0,
            "real_model_calls": 0,
            "real_credential_reads": 0,
            "real_environment_reads": 0,
        }

    def _authorization_value(self) -> str:
        """The authorization value exists transiently only within this mock object."""
        return "Bearer " + self._fake_key

    def _internal_auth_check(self) -> None:
        authorization_value = self._authorization_value()
        if not authorization_value.startswith("Bearer "):
            raise MockTransportError("mock_authorization_error")

    def preflight_identity(self) -> dict[str, Any]:
        self._internal_auth_check()
        self.counts["mock_transport_calls"] += 1
        self.counts["mock_identity_checks"] += 1
        return deepcopy(self._identity_before)

    def postflight_identity(self) -> dict[str, Any]:
        self._internal_auth_check()
        self.counts["mock_transport_calls"] += 1
        self.counts["mock_identity_checks"] += 1
        return deepcopy(self._identity_after)

    def chat(self, request: dict[str, Any]) -> str:
        self._internal_auth_check()
        if not isinstance(request, dict):
            raise MockTransportError("mock_request_type")
        self.counts["mock_transport_calls"] += 1
        self.counts["mock_inference_calls"] += 1
        if self._cursor >= len(self._scripted_responses):
            raise MockTransportError("mock_script_exhausted")
        response = self._scripted_responses[self._cursor]
        self._cursor += 1
        if isinstance(response, MockTransportError):
            raise response
        return response
