"""Narrow exception taxonomy used by the Q1 qualification implementation."""


class Q1Error(Exception):
    """Base class for expected, structured Q1 failures."""


class PreflightError(Q1Error):
    """An environment, provenance, or protocol fault detected before inference."""


class TransportError(Q1Error):
    """A loopback transport fault; this always makes the batch rework."""


class ContractError(Q1Error):
    """A strict public-response contract fault."""


class ValidationError(Q1Error):
    """An artifact or provenance integrity fault."""
