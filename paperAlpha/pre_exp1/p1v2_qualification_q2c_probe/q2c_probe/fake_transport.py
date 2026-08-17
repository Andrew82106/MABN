"""Dependency-injected, in-memory transport used only for offline tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


class ProtocolTransportError(RuntimeError):
    """A deliberately body-free transport failure shared by injected transports."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FakeTransportError(ProtocolTransportError):
    """A body-free transport failure emitted by the in-memory fake transport."""


@dataclass(frozen=True)
class FakeStep:
    kind: str
    payload: Any = None
    duration_seconds: float = 0.0


class ManualClock:
    def __init__(self, start_seconds: float = 0.0) -> None:
        self.value = float(start_seconds)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class FakeTransport:
    """Fake metadata/completion transport with an inspectable I/O call log."""

    def __init__(
        self,
        metadata_step: FakeStep,
        completion_steps: Iterable[FakeStep],
        clock: ManualClock,
    ) -> None:
        self.metadata_step = metadata_step
        self.completion_steps = list(completion_steps)
        self.clock = clock
        self.calls: list[tuple[str, float]] = []

    def _perform(self, label: str, step: FakeStep, timeout_seconds: float) -> Any:
        self.calls.append((label, float(timeout_seconds)))
        self.clock.advance(step.duration_seconds)
        if step.kind == "error":
            raise FakeTransportError(str(step.payload))
        if step.kind != "response":
            raise FakeTransportError("invalid_fake_step")
        return step.payload

    def get_models(self, timeout_seconds: float) -> Any:
        return self._perform("metadata", self.metadata_step, timeout_seconds)

    def complete(self, request_payload: dict[str, Any], timeout_seconds: float) -> Any:
        index = sum(1 for label, _ in self.calls if label.startswith("completion"))
        if index >= len(self.completion_steps):
            raise FakeTransportError("fake_completion_plan_exhausted")
        return self._perform(
            f"completion-{index + 1}", self.completion_steps[index], timeout_seconds
        )
