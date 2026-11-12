"""Fixed-category errors.  They intentionally carry no request, header, or body text."""

from __future__ import annotations


class ConfigError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class TransportError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ContractViolation(ValueError):
    def __init__(self, code: str, disposition: str) -> None:
        self.code = code
        self.disposition = disposition
        super().__init__(code)
