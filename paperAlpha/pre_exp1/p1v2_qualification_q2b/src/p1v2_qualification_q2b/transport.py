"""Restricted live HTTP transport with an injectable offline connection factory."""

from __future__ import annotations

import http.client
import socket
from dataclasses import dataclass
from typing import Any, Callable

from .env_config import FROZEN_BASE_URL, FROZEN_MODEL, LiveConfig
from .errors import ConfigError, TransportError


@dataclass(frozen=True)
class WireResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class _DirectLoopbackConnection(http.client.HTTPConnection):
    """A numeric AF_INET socket only; it does not resolve names or use proxies."""

    def __init__(self) -> None:
        super().__init__("127.0.0.1", port=58661, timeout=60)

    def connect(self) -> None:
        direct_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        direct_socket.settimeout(60)
        direct_socket.connect(("127.0.0.1", 58661))
        self.sock = direct_socket


def _live_connection_factory() -> _DirectLoopbackConnection:
    return _DirectLoopbackConnection()


class RestrictedGatewayTransport:
    """The only Q2-B client.  It permits exactly two fixed endpoint paths."""

    def __init__(self, config: LiveConfig, connection_factory: Callable[[], Any] | None = None) -> None:
        if config.base_url != FROZEN_BASE_URL or config.model != FROZEN_MODEL:
            raise ConfigError("transport_config_not_frozen")
        self._config = config
        self._connection_factory = connection_factory or _live_connection_factory
        self.counts = {
            "metadata_calls": 0,
            "completion_calls": 0,
            "retry_count": 0,
            "redirect_count": 0,
        }

    def _perform(self, method: str, path: str, body: bytes | None) -> WireResponse:
        if (method, path) not in {("GET", "/v1/models"), ("POST", "/v1/chat/completions")}:
            raise TransportError("endpoint_not_allowed")
        connection: Any | None = None
        try:
            connection = self._connection_factory()
            headers = {"Accept": "application/json", "Authorization": "Bearer " + self._config.api_key}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            status = getattr(response, "status", None)
            if not isinstance(status, int):
                raise TransportError("http_protocol_error")
            raw_body = response.read()
            if 300 <= status < 400:
                self.counts["redirect_count"] += 1
                raise TransportError("http_redirect")
            if status >= 400:
                if status in {401, 403, 429}:
                    raise TransportError("http_" + str(status))
                if 500 <= status <= 599:
                    raise TransportError("http_5xx")
                raise TransportError("http_4xx")
            if status < 200 or status >= 300:
                raise TransportError("http_unexpected_status")
            safe_headers: dict[str, str] = {}
            for name, value in response.getheaders():
                normalized = str(name).lower()
                if normalized in {"x-request-id", "request-id"} and isinstance(value, str) and value:
                    safe_headers[normalized] = value
            if not isinstance(raw_body, bytes):
                raise TransportError("http_body_type")
            return WireResponse(status=status, headers=safe_headers, body=raw_body)
        except TransportError:
            raise
        except (socket.timeout, TimeoutError) as exc:
            raise TransportError("timeout") from exc
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise TransportError("connection_error") from exc
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def get_models(self) -> WireResponse:
        if self.counts["metadata_calls"] >= 2:
            raise TransportError("metadata_budget_exceeded")
        self.counts["metadata_calls"] += 1
        return self._perform("GET", "/v1/models", None)

    def complete(self, request: dict[str, Any]) -> WireResponse:
        if self.counts["completion_calls"] >= 128:
            raise TransportError("completion_budget_exceeded")
        self.counts["completion_calls"] += 1
        try:
            body = __import__("json").dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TransportError("request_serialization") from exc
        return self._perform("POST", "/v1/chat/completions", body)
