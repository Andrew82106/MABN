"""The deliberately tiny, standard-library-only loopback Ollama adapter."""

from __future__ import annotations

import http.client
import json
import subprocess
from dataclasses import dataclass
from typing import Any

from .common import canonical_json_bytes, sha256_bytes, strict_json_loads
from .errors import TransportError


LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = 11434
TAGS_PATH = "/api/tags"
CHAT_PATH = "/api/chat"
MODEL_TAG = "qwen3:8b"


@dataclass(frozen=True)
class ModelFingerprint:
    name: str
    digest: str
    size: int
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "digest": self.digest, "size": self.size, "details": self.details}

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_dict()))


@dataclass(frozen=True)
class VersionEvidence:
    sha256: str
    length: int
    returncode: int

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "length": self.length, "returncode": self.returncode}


@dataclass(frozen=True)
class ChatResult:
    content: str
    response_sha256: str
    response_length: int
    thinking_present: bool
    thinking_sha256: str | None
    thinking_length: int


def assert_loopback_constants() -> None:
    if (LOOPBACK_HOST, LOOPBACK_PORT, TAGS_PATH, CHAT_PATH, MODEL_TAG) != (
        "127.0.0.1",
        11434,
        "/api/tags",
        "/api/chat",
        "qwen3:8b",
    ):
        raise TransportError("transport_constants_modified")


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    return value is not None and value is not False and value != ""


class OllamaLoopbackTransport:
    """No endpoint parameter, proxy support, redirection, or retry mechanism exists."""

    def _request(self, method: str, path: str, body: bytes | None, timeout: float) -> bytes:
        assert_loopback_constants()
        if method not in {"GET", "POST"} or path not in {TAGS_PATH, CHAT_PATH}:
            raise TransportError("illegal_loopback_request")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0 or timeout > 60:
            raise TransportError("invalid_timeout")
        connection = http.client.HTTPConnection(LOOPBACK_HOST, LOOPBACK_PORT, timeout=float(timeout))
        try:
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            if 300 <= response.status < 400:
                raise TransportError("redirect_rejected")
            if response.status < 200 or response.status >= 300:
                raise TransportError(f"http_status_{response.status}")
            return raw
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            raise TransportError(f"loopback_transport_{type(exc).__name__}") from exc
        finally:
            connection.close()

    @staticmethod
    def _selected_fingerprint(raw: bytes) -> ModelFingerprint:
        try:
            decoded = raw.decode("utf-8")
            value = strict_json_loads(decoded)
        except Exception as exc:
            raise TransportError("invalid_tags_json") from exc
        if not isinstance(value, dict) or not isinstance(value.get("models"), list):
            raise TransportError("invalid_tags_shape")
        matches = [model for model in value["models"] if isinstance(model, dict) and model.get("name") == MODEL_TAG]
        if len(matches) != 1:
            raise TransportError("required_model_tag_missing_or_ambiguous")
        model = matches[0]
        digest, size, details = model.get("digest"), model.get("size"), model.get("details")
        if not isinstance(digest, str) or not digest or not isinstance(size, int) or isinstance(size, bool) or size <= 0 or not isinstance(details, dict):
            raise TransportError("incomplete_model_fingerprint")
        return ModelFingerprint(name=MODEL_TAG, digest=digest, size=size, details=details)

    def get_tags(self, timeout: float = 10.0) -> ModelFingerprint:
        raw = self._request("GET", TAGS_PATH, None, timeout)
        return self._selected_fingerprint(raw)

    def chat(self, payload: dict[str, Any], timeout: float) -> ChatResult:
        assert_loopback_constants()
        expected_keys = {"model", "messages", "stream", "format", "think", "options"}
        if set(payload) != expected_keys or payload.get("model") != MODEL_TAG:
            raise TransportError("chat_payload_not_frozen")
        body = canonical_json_bytes(payload)
        raw = self._request("POST", CHAT_PATH, body, timeout)
        raw_sha256 = sha256_bytes(raw)
        try:
            root = strict_json_loads(raw.decode("utf-8"))
        except Exception as exc:
            raise TransportError("invalid_chat_transport_json") from exc
        if not isinstance(root, dict) or not isinstance(root.get("message"), dict):
            raise TransportError("invalid_chat_transport_shape")
        message = root["message"]
        content = message.get("content")
        if not isinstance(content, str):
            raise TransportError("missing_message_content")
        thinking_values = [root.get("thinking"), message.get("thinking")]
        thinking_present = any(_nonempty(value) for value in thinking_values)
        thinking_material = [value for value in thinking_values if _nonempty(value)]
        thinking_evidence = canonical_json_bytes(thinking_material) if thinking_material else b""
        return ChatResult(
            content=content,
            response_sha256=raw_sha256,
            response_length=len(raw),
            thinking_present=thinking_present,
            thinking_sha256=sha256_bytes(thinking_evidence) if thinking_material else None,
            thinking_length=len(thinking_evidence),
        )

    def ollama_version(self) -> VersionEvidence:
        try:
            completed = subprocess.run(
                ["ollama", "--version"],
                check=False,
                shell=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TransportError(f"ollama_version_{type(exc).__name__}") from exc
        material = completed.stdout + completed.stderr
        if completed.returncode != 0 or not material.strip():
            raise TransportError("ollama_version_failed")
        return VersionEvidence(sha256=sha256_bytes(material), length=len(material), returncode=completed.returncode)


class MockTransport:
    """Offline test double.  It cannot open sockets or invoke Ollama."""

    def __init__(
        self,
        fingerprints: list[ModelFingerprint | Exception],
        chat_results: list[ChatResult | Exception],
        versions: list[VersionEvidence | Exception] | None = None,
    ) -> None:
        self.fingerprints = list(fingerprints)
        self.chat_results = list(chat_results)
        self.versions = list(versions or [VersionEvidence("mock-version", 12, 0), VersionEvidence("mock-version", 12, 0)])
        self.tag_calls = 0
        self.chat_calls = 0
        self.version_calls = 0
        self.timeouts: list[float] = []
        self.payloads: list[dict[str, Any]] = []

    @staticmethod
    def _next(items: list[Any], code: str) -> Any:
        if not items:
            raise TransportError(code)
        value = items.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def get_tags(self, timeout: float = 10.0) -> ModelFingerprint:
        self.tag_calls += 1
        self.timeouts.append(timeout)
        return self._next(self.fingerprints, "mock_tags_exhausted")

    def chat(self, payload: dict[str, Any], timeout: float) -> ChatResult:
        self.chat_calls += 1
        self.timeouts.append(timeout)
        self.payloads.append(payload)
        return self._next(self.chat_results, "mock_chat_exhausted")

    def ollama_version(self) -> VersionEvidence:
        self.version_calls += 1
        return self._next(self.versions, "mock_version_exhausted")
