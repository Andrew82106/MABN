"""Strict, live-only HTTP transport for Q2-C.

This module contains no automatic credential loading and opens no connection at
import time.  Tests inject an in-memory connection factory; the default factory
is reachable only from the explicitly released live CLI path.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
from typing import Any, Callable, Protocol

from .config import (
    BASE_URL_IDENTITY,
    CARDS,
    TARGET_MODEL,
    canonical_json,
)
from .fake_transport import ProtocolTransportError


LIVE_HOST = "127.0.0.1"
LIVE_PORT = 58661
LIVE_MODELS_PATH = "/v1/models"
LIVE_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True)
class LiveConfiguration:
    base_url: str
    api_key: str
    model: str


class HTTPResponseProtocol(Protocol):
    status: int

    def read(self, amt: int | None = None) -> bytes: ...


class HTTPConnectionProtocol(Protocol):
    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None: ...

    def getresponse(self) -> HTTPResponseProtocol: ...

    def close(self) -> None: ...


ConnectionFactory = Callable[[str, int, float], HTTPConnectionProtocol]


def validate_live_configuration(configuration: LiveConfiguration) -> None:
    """Reject every endpoint/model variation before a connection can be created."""
    if configuration.base_url != BASE_URL_IDENTITY:
        raise ValueError("live_base_url_policy")
    if configuration.model != TARGET_MODEL:
        raise ValueError("live_model_policy")
    if (
        not isinstance(configuration.api_key, str)
        or not configuration.api_key
        or len(configuration.api_key) > 8192
        or any(ord(character) < 32 or ord(character) == 127 for character in configuration.api_key)
    ):
        raise ValueError("live_api_key_policy")


def default_connection_factory(host: str, port: int, timeout_seconds: float) -> HTTPConnectionProtocol:
    """Direct numeric-IP connection; no redirect or proxy support is used."""
    if host != LIVE_HOST or port != LIVE_PORT or timeout_seconds <= 0.0:
        raise ValueError("live_connection_factory_policy")
    return http.client.HTTPConnection(host, port, timeout=timeout_seconds)


class LiveHTTPTransport:
    """A direct HTTP client with strict routes and a transient Authorization header."""

    def __init__(
        self,
        configuration: LiveConfiguration,
        connection_factory: ConnectionFactory = default_connection_factory,
    ) -> None:
        validate_live_configuration(configuration)
        self._api_key: str | None = configuration.api_key
        self._connection_factory = connection_factory
        self.calls: list[dict[str, Any]] = []

    @property
    def secret_cleared(self) -> bool:
        return self._api_key is None

    def clear_secret(self) -> None:
        self._api_key = None

    def _request(self, method: str, path: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
        if timeout_seconds <= 0.0:
            raise ProtocolTransportError("timeout")
        if method == "GET" and path != LIVE_MODELS_PATH:
            raise ProtocolTransportError("policy_error")
        if method == "POST" and path != LIVE_CHAT_COMPLETIONS_PATH:
            raise ProtocolTransportError("policy_error")
        if method not in {"GET", "POST"}:
            raise ProtocolTransportError("policy_error")
        if self._api_key is None:
            raise ProtocolTransportError("credential_error")
        body = None if payload is None else canonical_json(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + self._api_key,
        }
        if method == "POST":
            headers["Content-Type"] = "application/json"
        connection: HTTPConnectionProtocol | None = None
        self.calls.append(
            {
                "kind": "metadata" if method == "GET" else "completion",
                "method": method,
                "path": path,
                "timeout_seconds": float(timeout_seconds),
            }
        )
        try:
            connection = self._connection_factory(LIVE_HOST, LIVE_PORT, float(timeout_seconds))
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            status = getattr(response, "status", None)
            if not isinstance(status, int):
                raise ProtocolTransportError("transport_error")
            if 300 <= status < 400:
                raise ProtocolTransportError("redirect")
            if status < 200 or status >= 300:
                raise ProtocolTransportError(f"http_{status}")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
                raise ProtocolTransportError("transport_error")
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ProtocolTransportError("transport_error") from None
            if not isinstance(decoded, dict):
                raise ProtocolTransportError("transport_error")
            return decoded
        except ProtocolTransportError:
            raise
        except TimeoutError:
            raise ProtocolTransportError("timeout") from None
        except OSError:
            raise ProtocolTransportError("connection_error") from None
        except (TypeError, ValueError):
            raise ProtocolTransportError("transport_error") from None
        finally:
            headers.clear()
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def get_models(self, timeout_seconds: float) -> dict[str, Any]:
        return self._request("GET", LIVE_MODELS_PATH, None, timeout_seconds)

    def complete(self, request_payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        canonical_payload = canonical_json(request_payload)
        allowed_payloads = {canonical_json(card.request_payload()) for card in CARDS}
        if canonical_payload not in allowed_payloads:
            raise ProtocolTransportError("policy_error")
        return self._request("POST", LIVE_CHAT_COMPLETIONS_PATH, request_payload, timeout_seconds)
