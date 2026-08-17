"""Pure in-memory staged runner used to prove deadline-gated I/O ordering.

This module deliberately has no live transport, environment, or network dependency.
Every fake call is made only through :class:`BatchDeadline` and is recorded as
minimal timing evidence.  Returned values remain in memory and never enter the
public execution evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .deadline import BatchDeadline
from .errors import DeadlineExceeded


STAGES = ("metadata_start", "completion", "metadata_end")


class InMemoryClock:
    """Deterministic clock for offline tests and the repair-native audit."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass(frozen=True)
class FakeCall:
    stage: str
    timeout_seconds: float
    remaining_before_seconds: float
    deadline_at_seconds: float

    def public_dict(self) -> dict[str, float | str]:
        return {
            "deadline_at_seconds": self.deadline_at_seconds,
            "remaining_before_seconds": self.remaining_before_seconds,
            "stage": self.stage,
            "timeout_seconds": self.timeout_seconds,
        }


class FakeStagedTransport:
    """A deterministic local stand-in whose call list is the I/O witness."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        advance: Callable[[float], None],
        values: Mapping[str, Any],
        delays_seconds: Mapping[str, float] | None = None,
    ) -> None:
        if set(values) != set(STAGES):
            raise ValueError("fake_transport_values")
        self._clock = clock
        self._advance = advance
        self._values = dict(values)
        self._delays = dict(delays_seconds or {})
        if set(self._delays) - set(STAGES) or any(value < 0 for value in self._delays.values()):
            raise ValueError("fake_transport_delays")
        self.calls: list[FakeCall] = []

    def call(self, *, stage: str, timeout_seconds: float, deadline_at_seconds: float) -> Any:
        if stage not in STAGES:
            raise ValueError("fake_transport_stage")
        self.calls.append(
            FakeCall(
                stage=stage,
                timeout_seconds=timeout_seconds,
                remaining_before_seconds=deadline_at_seconds - self._clock(),
                deadline_at_seconds=deadline_at_seconds,
            )
        )
        self._advance(self._delays.get(stage, 0.0))
        return self._values[stage]


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    reason: str | None
    timeout_seconds: float | None
    io_attempted: bool
    value: Any = field(repr=False, compare=False, default=None)

    def public_dict(self) -> dict[str, bool | float | str | None]:
        return {
            "io_attempted": self.io_attempted,
            "reason": self.reason,
            "stage": self.stage,
            "status": self.status,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class StagedRun:
    flow_status: str
    termination_reason: str | None
    stages: tuple[StageResult, ...]
    calls: tuple[FakeCall, ...]

    def stage_value(self, stage: str) -> Any:
        for result in self.stages:
            if result.stage == stage:
                return result.value
        raise KeyError(stage)

    def public_evidence(self) -> dict[str, Any]:
        """Return only call ordering/timing; never return fake response values."""
        return {
            "calls": [call.public_dict() for call in self.calls],
            "flow_status": self.flow_status,
            "runner_kind": "offline_fake_staged_deadline_guard",
            "stages": [result.public_dict() for result in self.stages],
            "termination_reason": self.termination_reason,
        }


class StagedDeadlineRunner:
    """Run start-metadata → completion → end-metadata under one deadline."""

    def __init__(
        self,
        *,
        deadline: BatchDeadline,
        transport: FakeStagedTransport,
        before_stage: Callable[[str], None] | None = None,
    ) -> None:
        self._deadline = deadline
        self._transport = transport
        self._before_stage = before_stage

    @property
    def deadline(self) -> BatchDeadline:
        """Expose the injected guard for deterministic integration fixtures."""
        return self._deadline

    def run(self) -> StagedRun:
        results: list[StageResult] = []
        for index, stage in enumerate(STAGES):
            if self._before_stage is not None:
                self._before_stage(stage)
            try:
                call = self._deadline.invoke(
                    stage,
                    lambda timeout, current_stage=stage: self._transport.call(
                        stage=current_stage,
                        timeout_seconds=timeout,
                        deadline_at_seconds=self._deadline.deadline_at_seconds,
                    ),
                )
            except DeadlineExceeded as exc:
                status = "unstarted" if exc.code.startswith("deadline_before_") else "failed"
                results.append(StageResult(stage, status, exc.code, None, False if status == "unstarted" else True))
                for blocked_stage in STAGES[index + 1 :]:
                    results.append(
                        StageResult(
                            blocked_stage,
                            "not_attempted",
                            "blocked_after_" + stage,
                            None,
                            False,
                        )
                    )
                return StagedRun(
                    flow_status="engineering_failed",
                    termination_reason=exc.code,
                    stages=tuple(results),
                    calls=tuple(self._transport.calls),
                )
            results.append(StageResult(stage, "completed", None, call.timeout_seconds, True, call.value))
        return StagedRun(
            flow_status="completed",
            termination_reason=None,
            stages=tuple(results),
            calls=tuple(self._transport.calls),
        )
