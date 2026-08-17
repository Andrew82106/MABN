"""Deterministic in-memory inputs for the one offline fake evidence run."""

from __future__ import annotations

from typing import Any

from .config import CARDS, TARGET_MODEL
from .fake_transport import FakeStep


FAKE_MODELS = (TARGET_MODEL, "fake-secondary-model")


def successful_metadata_step(duration_seconds: float = 0.25) -> FakeStep:
    return FakeStep(
        kind="response",
        duration_seconds=duration_seconds,
        payload={"data": [{"id": model_id} for model_id in FAKE_MODELS]},
    )


def successful_completion_step(index: int, duration_seconds: float = 0.25) -> FakeStep:
    card = CARDS[index]
    if card.profile == "plain_text":
        content = card.expected_answer()
    else:
        content = (
            "{\"answer\":\"" + card.expected_answer() + "\",\"probe_id\":\""
            + card.expected_probe_id() + "\"}"
        )
    payload: dict[str, Any] = {
        "id": f"fake-request-{index + 1}",
        "model": TARGET_MODEL,
        "system_fingerprint": "fake-fingerprint-v1",
        "provider_version": "fake-provider-v1",
        "choices": [{"message": {"content": content}}],
    }
    return FakeStep(kind="response", payload=payload, duration_seconds=duration_seconds)


def successful_completion_steps(duration_seconds: float = 0.25) -> list[FakeStep]:
    return [successful_completion_step(index, duration_seconds) for index in range(len(CARDS))]
