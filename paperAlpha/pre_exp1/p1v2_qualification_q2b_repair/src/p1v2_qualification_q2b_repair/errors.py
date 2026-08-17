"""Fixed-code exceptions; no raw response or credential text is carried."""

from __future__ import annotations


class RepairError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DeadlineExceeded(RepairError):
    pass
