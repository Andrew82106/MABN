"""Fixed-category errors used by the offline mock transport and adapter."""

from __future__ import annotations


class ProtocolViolation(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class MockTransportError(RuntimeError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)
