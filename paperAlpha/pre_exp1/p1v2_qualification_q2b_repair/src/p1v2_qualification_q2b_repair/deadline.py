"""Explicit batch-deadline guard usable with injected fake calls only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .errors import DeadlineExceeded


T = TypeVar("T")


@dataclass(frozen=True)
class DeadlineCall(Generic[T]):
    label: str
    timeout_seconds: float
    value: T


class BatchDeadline:
    def __init__(self, *, clock: Callable[[], float], budget_seconds: float, started_at: float | None = None) -> None:
        self._clock = clock
        self._started_at = clock() if started_at is None else started_at
        self._deadline = self._started_at + budget_seconds

    def remaining_seconds(self) -> float:
        return self._deadline - self._clock()

    @property
    def deadline_at_seconds(self) -> float:
        """Absolute deadline on the injected clock, exposed to fake transports only."""
        return self._deadline

    def invoke(self, label: str, operation: Callable[[float], T]) -> DeadlineCall[T]:
        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise DeadlineExceeded("deadline_before_" + label)
        timeout = min(60.0, remaining)
        value = operation(timeout)
        if self.remaining_seconds() <= 0:
            raise DeadlineExceeded("deadline_during_" + label)
        return DeadlineCall(label=label, timeout_seconds=timeout, value=value)


def guarded_card_record(deadline: BatchDeadline, label: str, operation: Callable[[float], T], task_id: str) -> dict[str, str | float]:
    """Map pre-I/O expiry to an unstarted card and in-I/O expiry to a failed card."""
    try:
        call = deadline.invoke(label, operation)
    except DeadlineExceeded as exc:
        status = "unstarted" if exc.code.startswith("deadline_before_") else "failed"
        return {"task_id": task_id, "status": status, "reason": exc.code}
    return {"task_id": task_id, "status": "completed", "timeout_seconds": call.timeout_seconds}
