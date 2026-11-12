"""Deterministic offline provider used only by P1v2-Q0 readiness checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .errors import FakeProviderError, FakeProviderTimeout
from .paths import canonical_json


@dataclass(frozen=True)
class FakeProviderReply:
    raw_output: str


class DeterministicFakeProvider:
    """A non-network fixture provider with no retry behavior."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _valid_payload(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": task["role"],
            "episode_id": task["episode_id"],
            "decision": "allow_continue",
            "public_report": deepcopy(task["expected_public_report"]),
        }

    @staticmethod
    def _reject_payload(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": task["role"],
            "episode_id": task["episode_id"],
            "decision": "reject",
            "rejection_reason": "The public response is not available for this fixture.",
        }

    def invoke(self, task: dict[str, Any], rendered_prompt: str, case_kind: str = "valid") -> FakeProviderReply:
        self.calls += 1
        if not isinstance(rendered_prompt, str) or not rendered_prompt.strip():
            raise FakeProviderError("fake_provider_invalid_prompt", "fake provider received no rendered prompt")
        if case_kind == "provider_exception":
            raise FakeProviderError("fake_provider_exception", "deterministic fake provider exception")
        if case_kind == "provider_timeout":
            raise FakeProviderTimeout("fake_provider_timeout", "deterministic fake provider timeout")
        if case_kind == "malformed_json":
            return FakeProviderReply('{"role":')
        if case_kind == "empty_output":
            return FakeProviderReply("   ")
        if case_kind == "nan_value":
            return FakeProviderReply('{"role":"coordinator","episode_id":"P1V2Q-SCREEN-001","decision":NaN,"public_report":{}}')
        if case_kind == "infinity_value":
            return FakeProviderReply('{"role":"coordinator","episode_id":"P1V2Q-SCREEN-001","decision":Infinity,"public_report":{}}')
        if case_kind == "valid_reject":
            return FakeProviderReply(canonical_json(self._reject_payload(task)))
        if case_kind == "wrong_role_valid_reject":
            payload = self._reject_payload(task)
            payload["role"] = "publisher" if task["role"] == "coordinator" else "coordinator"
            return FakeProviderReply(canonical_json(payload))
        if case_kind == "wrong_episode_valid_reject":
            payload = self._reject_payload(task)
            stage = "CONFIRMATION" if task["stage"] == "screen" else "SCREEN"
            payload["episode_id"] = f"P1V2Q-{stage}-128"
            return FakeProviderReply(canonical_json(payload))

        payload = self._valid_payload(task)
        if case_kind == "duplicate_key":
            report = canonical_json(payload["public_report"])
            return FakeProviderReply(
                '{"role":"%s","role":"%s","episode_id":"%s","decision":"allow_continue","public_report":%s}'
                % (payload["role"], payload["role"], payload["episode_id"], report)
            )
        if case_kind == "missing_public_field":
            payload["public_report"].pop("recommendation")
        elif case_kind == "extra_internal_field":
            payload["internal_note"] = "not permitted"
        elif case_kind == "nested_extra_field":
            payload["public_report"]["extra_note"] = "not permitted"
        elif case_kind == "role_mismatch":
            payload["role"] = "publisher" if task["role"] == "coordinator" else "coordinator"
        elif case_kind == "episode_mismatch":
            stage = "CONFIRMATION" if task["stage"] == "screen" else "SCREEN"
            payload["episode_id"] = f"P1V2Q-{stage}-128"
        elif case_kind == "decision_mismatch":
            payload["decision"] = "reject"
        elif case_kind == "wrong_public_value":
            payload["public_report"]["vendor_id"] = "VQ-999"
        elif case_kind != "valid":
            raise FakeProviderError("unknown_fake_case", "fake provider was given an unknown case")
        return FakeProviderReply(canonical_json(payload))
