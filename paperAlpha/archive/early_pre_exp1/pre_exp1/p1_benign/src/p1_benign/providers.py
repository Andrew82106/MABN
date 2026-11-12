"""Minimal provider abstraction with deterministic, local, and remote adapters."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from .core import (
    REQUIRED_PUBLIC_FIELDS,
    canonical_json,
    recommendation_for,
    risk_level_for_score,
    sanitized_error,
)


class ProviderError(RuntimeError):
    """Base provider failure with a deliberately non-secret message."""


class ProviderParseError(ProviderError):
    """Provider returned no valid structured action."""


class ProviderUnavailableError(ProviderError):
    """Provider endpoint or model was unavailable."""


REASONING_FIELD_NAMES = frozenset(
    {
        "analysis",
        "internalanalysis",
        "reasoning",
        "reasoningcontent",
        "reasoningdetails",
        "reasoningtrace",
        "modelreasoning",
        "hiddenreasoning",
        "privatereasoning",
        "internalmonologue",
        "chainofthought",
        "thoughtprocess",
        "thought",
        "thoughts",
        "thinking",
        "scratchpad",
        "rationale",
        "deliberation",
        "cot",
    }
)


def _normalized_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _is_reasoning_field_name(value: Any) -> bool:
    normalized = _normalized_field_name(value)
    return (
        normalized in REASONING_FIELD_NAMES
        or "reasoning" in normalized
        or "scratchpad" in normalized
        or "rationale" in normalized
        or "internalmonologue" in normalized
        or "thoughtprocess" in normalized
        or "chainofthought" in normalized
        or "deliberation" in normalized
    )


def sanitize_observable_response(
    value: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Recursively discard hidden-reasoning fields before logging/hashing."""

    discarded = False

    def visit(item: Any) -> Any:
        nonlocal discarded
        if isinstance(item, dict):
            cleaned: dict[str, Any] = {}
            for key, child in item.items():
                if _is_reasoning_field_name(key):
                    discarded = True
                    continue
                cleaned[key] = visit(child)
            return cleaned
        if isinstance(item, list):
            return [visit(child) for child in item]
        return copy.deepcopy(item)

    result = visit(value)
    if not isinstance(result, dict):
        raise ProviderParseError("Provider response was not an object")
    return result, discarded


_FROZEN_DECODING_KEYS = frozenset(
    {
        "temperature",
        "sampling_seed",
        "max_output_tokens",
        "thinking",
        "frozen",
    }
)


def contains_reasoning_field(
    value: Any,
    *,
    allow_disabled_thinking_control: bool = False,
    allow_public_evidence_controls: bool = False,
) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            folded_key = _normalized_field_name(key)
            is_frozen_decoding_control = (
                allow_disabled_thinking_control
                and folded_key == "thinking"
                and child is False
                and frozenset(str(name) for name in value)
                == _FROZEN_DECODING_KEYS
            )
            is_public_evidence_control = (
                allow_public_evidence_controls
                and (
                    (
                        folded_key
                        in {
                            "hiddenreasoningrequested",
                            "hiddenreasoninglogged",
                        }
                        and child is False
                    )
                    or (
                        folded_key == "reasoningdiscarded"
                        and isinstance(child, bool)
                    )
                )
            )
            if (
                _is_reasoning_field_name(key)
                and not is_frozen_decoding_control
                and not is_public_evidence_control
            ):
                return True
            if contains_reasoning_field(
                child,
                allow_disabled_thinking_control=(
                    allow_disabled_thinking_control
                ),
                allow_public_evidence_controls=(
                    allow_public_evidence_controls
                ),
            ):
                return True
        return False
    if isinstance(value, list):
        return any(
            contains_reasoning_field(
                child,
                allow_disabled_thinking_control=(
                    allow_disabled_thinking_control
                ),
                allow_public_evidence_controls=(
                    allow_public_evidence_controls
                ),
            )
            for child in value
        )
    return False


@dataclass(frozen=True)
class ModelRequest:
    run_id: str
    episode_id: str
    role_id: str
    role_prompt: str
    phase: str
    call_index: int
    visible_messages: list[dict[str, Any]]
    response_contract: dict[str, Any]


@dataclass(frozen=True)
class ProviderResult:
    response: dict[str, Any]
    latency_ms: float
    token_usage: dict[str, int | None]
    finish_reason: str | None
    reasoning_discarded: bool
    safe_response_hash_source: str


class ModelProvider(ABC):
    provider_id: str
    model_id: str
    endpoint_identifier: str
    decoding: dict[str, Any]
    concurrency: int = 1
    is_live_provider: bool = False
    is_replay_provider: bool = False

    @abstractmethod
    def generate(self, request: ModelRequest) -> ProviderResult:
        raise NotImplementedError

    def manifest_summary(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "endpoint_identifier": self.endpoint_identifier,
            "decoding": copy.deepcopy(self.decoding),
            "concurrency": self.concurrency,
            "credential_env": None,
        }


def _message_contents(
    request: ModelRequest,
    source_agent: str | None = None,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for message in request.visible_messages:
        if source_agent is None or message.get("source_agent") == source_agent:
            content = message.get("content")
            if isinstance(content, dict):
                values.append(content)
    return values


class DeterministicTestProvider(ModelProvider):
    """Frozen, network-free structured outputs for tests and dry-runs."""

    provider_id = "deterministic_test"
    model_id = "deterministic-p1-v1"
    endpoint_identifier = "none"
    decoding = {
        "temperature": 0,
        "sampling_seed": 20260731,
        "max_output_tokens": 256,
        "thinking": False,
        "frozen": True,
    }
    concurrency = 1
    is_live_provider = False

    def generate(self, request: ModelRequest) -> ProviderResult:
        started = time.perf_counter()
        response = self._response_for(request)
        return ProviderResult(
            response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
            token_usage={
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            finish_reason="deterministic",
            reasoning_discarded=False,
            safe_response_hash_source=canonical_json(response),
        )

    def _response_for(self, request: ModelRequest) -> dict[str, Any]:
        role_id = request.role_id
        contents = _message_contents(request)
        if role_id == "intake":
            material = contents[0]
            return {
                "kind": "message",
                "content": {
                    "vendor_id": material["vendor_id"],
                    "vendor_name": material["vendor_name"],
                    "declared_registration_status": material[
                        "declared_registration_status"
                    ],
                    "public_risk_score": material["public_risk_score"],
                    "evidence_reference": material["evidence_reference"],
                },
            }
        if role_id == "dossier_extractor":
            material = contents[0]
            return {
                "kind": "message",
                "content": {
                    "vendor_id": material["vendor_id"],
                    "registration_status": material[
                        "declared_registration_status"
                    ],
                    "public_risk_score": material["public_risk_score"],
                },
            }
        if role_id == "independent_verifier":
            material = contents[0]
            return {
                "kind": "message",
                "content": {
                    "vendor_id": material["vendor_id"],
                    "registration_status": material[
                        "declared_registration_status"
                    ],
                    "evidence_verified": True,
                    "evidence_reference": material["evidence_reference"],
                },
            }
        if role_id == "coordinator":
            extractor = _message_contents(
                request,
                "dossier_extractor",
            )[0]
            verifier = _message_contents(
                request,
                "independent_verifier",
            )[0]
            return {
                "kind": "message",
                "content": {
                    "vendor_id": extractor["vendor_id"],
                    "registration_status": extractor[
                        "registration_status"
                    ],
                    "public_risk_score": extractor["public_risk_score"],
                    "evidence_verified": verifier["evidence_verified"],
                    "requested_internal_fields": [
                        "registration_status",
                        "risk_score",
                    ],
                },
            }
        if role_id == "risk_analyst":
            coordination = contents[0]
            risk_level = risk_level_for_score(
                coordination["public_risk_score"]
            )
            return {
                "kind": "message",
                "content": {
                    "vendor_id": coordination["vendor_id"],
                    "risk_level": risk_level,
                    "recommendation": recommendation_for(
                        coordination["registration_status"],
                        risk_level,
                    ),
                },
            }
        if role_id == "policy_reviewer":
            coordination = contents[0]
            return {
                "kind": "message",
                "content": {
                    "vendor_id": coordination["vendor_id"],
                    "policy_compliant": True,
                    "allowed_public_fields": list(REQUIRED_PUBLIC_FIELDS),
                },
            }
        if role_id == "internal_record_agent":
            if request.phase == "request_tool":
                coordination = contents[0]
                return {
                    "kind": "tool_call",
                    "tool_name": "internal_db.read",
                    "arguments": {
                        "vendor_id": coordination["vendor_id"],
                    },
                }
            tool_result = _message_contents(
                request,
                "internal_db.read",
            )[0]
            return {
                "kind": "message",
                "content": {
                    "vendor_id": tool_result["vendor_id"],
                    "registration_status": tool_result[
                        "registration_status"
                    ],
                    "risk_score": tool_result["risk_score"],
                },
            }
        if role_id == "report_publisher":
            internal = _message_contents(
                request,
                "internal_record_agent",
            )[0]
            risk_level = risk_level_for_score(internal["risk_score"])
            report = {
                "vendor_id": internal["vendor_id"],
                "registration_status": internal["registration_status"],
                "risk_level": risk_level,
                "recommendation": recommendation_for(
                    internal["registration_status"],
                    risk_level,
                ),
            }
            return {
                "kind": "tool_call",
                "tool_name": "external_sink.publish",
                "arguments": report,
            }
        raise ProviderParseError("Unsupported deterministic role")


def _validate_local_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http":
        raise ValueError("Ollama endpoint must use http")
    if parsed.username or parsed.password:
        raise ValueError("Ollama endpoint must not contain credentials")
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama endpoint must be loopback-only")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama endpoint must be an origin URL")
    port = parsed.port or 11434
    return f"http://{host if host != '::1' else '[::1]'}:{port}"


def ollama_model_is_installed(
    model_id: str = "qwen3:8b",
    *,
    timeout_seconds: float = 15.0,
) -> bool:
    """Use only `ollama list`; never pull, create, or download a model."""

    if model_id != "qwen3:8b":
        return False
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    names = {
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.split()
    }
    return model_id in names


_THINK_PATTERN = re.compile(
    r"<think>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _safe_json_from_model_content(content: str) -> tuple[dict[str, Any], bool]:
    reasoning_discarded = bool(_THINK_PATTERN.search(content))
    safe_content = _THINK_PATTERN.sub("", content).strip()
    candidates = [safe_content]
    fenced = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        safe_content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if fenced != safe_content:
        candidates.append(fenced)
    first = safe_content.find("{")
    last = safe_content.rfind("}")
    if first >= 0 and last > first:
        candidates.append(safe_content[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, reasoning_discarded
    raise ProviderParseError("Provider response was not a JSON object")


def _validate_structured_action(response: dict[str, Any]) -> None:
    kind = response.get("kind")
    if kind == "message":
        if not isinstance(response.get("content"), dict):
            raise ProviderParseError("Message content must be an object")
        return
    if kind == "tool_call":
        if not isinstance(response.get("tool_name"), str):
            raise ProviderParseError("Tool call requires a tool_name")
        if not isinstance(response.get("arguments"), dict):
            raise ProviderParseError("Tool call arguments must be an object")
        return
    raise ProviderParseError("Structured response has an unsupported kind")


class OllamaLocalProvider(ModelProvider):
    """Loopback-only Ollama adapter; never performs model installation."""

    provider_id = "ollama_local"
    concurrency = 1
    is_live_provider = False

    def __init__(
        self,
        *,
        model_id: str = "qwen3:8b",
        endpoint: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 90.0,
        max_output_tokens: int = 256,
    ) -> None:
        if model_id != "qwen3:8b":
            raise ValueError("Phase A local shakedown is fixed to qwen3:8b")
        if not 1 <= max_output_tokens <= 512:
            raise ValueError("Ollama output-token cap must be 1..512")
        self.model_id = model_id
        self.endpoint = _validate_local_endpoint(endpoint)
        self.endpoint_identifier = "loopback_ollama"
        self.timeout_seconds = timeout_seconds
        self.decoding = {
            "temperature": 0,
            "sampling_seed": 20260731,
            "max_output_tokens": max_output_tokens,
            "thinking": False,
            "frozen": True,
        }

    def manifest_summary(self) -> dict[str, Any]:
        summary = super().manifest_summary()
        summary.update(
            {
                "timeout_seconds": self.timeout_seconds,
                "automatic_model_pull": False,
                "model_availability_check": "ollama list",
            }
        )
        return summary

    def generate(self, request: ModelRequest) -> ProviderResult:
        system = (
            request.role_prompt.strip()
            + "\n/no_think\n"
            + "Do not reveal hidden reasoning. Return only one JSON object "
            + "matching this response contract:\n"
            + canonical_json(request.response_contract)
        )
        user = canonical_json(
            {
                "phase": request.phase,
                "visible_messages": request.visible_messages,
            }
        )
        payload = {
            "model": self.model_id,
            "stream": False,
            "think": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0,
                "seed": self.decoding["sampling_seed"],
                "num_predict": self.decoding["max_output_tokens"],
            },
        }
        request_body = canonical_json(payload).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.endpoint}/api/chat",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                body = response.read()
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as exc:
            raise ProviderUnavailableError(sanitized_error(exc)) from None
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            document = json.loads(body)
            message = document["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ProviderParseError(
                "Ollama response did not contain message.content"
            ) from None
        parsed, content_reasoning_discarded = _safe_json_from_model_content(
            content
        )
        parsed, embedded_reasoning_discarded = (
            sanitize_observable_response(parsed)
        )
        _validate_structured_action(parsed)
        reasoning_discarded = (
            content_reasoning_discarded
            or embedded_reasoning_discarded
            or bool(message.get("thinking"))
            or "thinking" in document
        )
        return ProviderResult(
            response=parsed,
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": document.get("prompt_eval_count"),
                "completion_tokens": document.get("eval_count"),
                "total_tokens": (
                    (
                        document.get("prompt_eval_count", 0)
                        + document.get("eval_count", 0)
                    )
                    if isinstance(document.get("prompt_eval_count"), int)
                    and isinstance(document.get("eval_count"), int)
                    else None
                ),
            },
            finish_reason=document.get("done_reason"),
            reasoning_discarded=reasoning_discarded,
            safe_response_hash_source=canonical_json(parsed),
        )


class OpenAICompatibleProvider(ModelProvider):
    """Small remote-ready adapter; construction does not read credentials."""

    concurrency = 1
    is_live_provider = True

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        endpoint: str,
        credential_env: str,
        decoding: dict[str, Any],
        timeout_seconds: float = 90.0,
    ) -> None:
        if not provider_id or not model_id:
            raise ValueError("Remote provider and model IDs are required")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Remote endpoint must be an HTTP(S) origin")
        if parsed.username or parsed.password:
            raise ValueError("Remote endpoint must not embed credentials")
        self.provider_id = provider_id
        self.model_id = model_id
        self.endpoint = endpoint.rstrip("/")
        self.endpoint_identifier = (
            f"{parsed.scheme}://{parsed.hostname}:{parsed.port or ''}"
        )
        self.credential_env = credential_env
        self.decoding = copy.deepcopy(decoding)
        self.timeout_seconds = timeout_seconds

    def manifest_summary(self) -> dict[str, Any]:
        summary = super().manifest_summary()
        summary["credential_env"] = self.credential_env
        return summary

    def generate(self, request: ModelRequest) -> ProviderResult:
        try:
            api_key = os.environ[self.credential_env]
        except KeyError:
            raise ProviderUnavailableError(
                "Configured credential environment variable is absent"
            ) from None
        system = (
            request.role_prompt.strip()
            + "\nDo not provide hidden reasoning. Return only JSON matching: "
            + canonical_json(request.response_contract)
        )
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": canonical_json(
                        {
                            "phase": request.phase,
                            "visible_messages": request.visible_messages,
                        }
                    ),
                },
            ],
            "temperature": self.decoding["temperature"],
            "max_tokens": self.decoding["max_output_tokens"],
            "response_format": {"type": "json_object"},
        }
        http_request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=canonical_json(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                body = response.read()
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as exc:
            raise ProviderUnavailableError(sanitized_error(exc)) from None
        finally:
            api_key = ""
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            document = json.loads(body)
            choice = document["choices"][0]
            message = choice["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            raise ProviderParseError(
                "Remote response did not contain message content"
            ) from None
        parsed, content_reasoning_discarded = _safe_json_from_model_content(
            content
        )
        parsed, embedded_reasoning_discarded = (
            sanitize_observable_response(parsed)
        )
        _validate_structured_action(parsed)
        usage = document.get("usage", {})
        return ProviderResult(
            response=parsed,
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            finish_reason=choice.get("finish_reason"),
            reasoning_discarded=(
                content_reasoning_discarded
                or embedded_reasoning_discarded
                or "reasoning" in message
                or "reasoning_content" in message
            ),
            safe_response_hash_source=canonical_json(parsed),
        )


class TranscriptProvider(ModelProvider):
    """Replays recorded observable outputs without any provider call."""

    is_live_provider = False
    is_replay_provider = True
    concurrency = 1

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        decoding: dict[str, Any],
        actions: Sequence[dict[str, Any]],
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.endpoint_identifier = "transcript_replay"
        self.decoding = copy.deepcopy(decoding)
        self._actions = [copy.deepcopy(action) for action in actions]
        self._position = 0

    @property
    def exhausted(self) -> bool:
        return self._position == len(self._actions)

    def generate(self, request: ModelRequest) -> ProviderResult:
        if self._position >= len(self._actions):
            raise ProviderParseError("Transcript has no remaining action")
        action = self._actions[self._position]
        self._position += 1
        if (
            action.get("role_id") != request.role_id
            or action.get("phase") != request.phase
        ):
            raise ProviderParseError("Transcript call order mismatch")
        if action.get("failed"):
            error_type = action.get("error_type")
            if error_type == "ProviderParseError":
                raise ProviderParseError("Replayed parse failure")
            if error_type == "ProviderUnavailableError":
                raise ProviderUnavailableError(
                    "Replayed provider failure"
                )
            if error_type == "UnexpectedProviderError":
                # This must bypass the ProviderError handler so workflow
                # records the same unexpected-failure class.
                raise RuntimeError("Replayed unexpected provider failure")
            raise ProviderParseError(
                "Transcript provider failure type is invalid"
            )
        response = action.get("response")
        if not isinstance(response, dict):
            raise ProviderParseError("Transcript response is invalid")
        return ProviderResult(
            response=copy.deepcopy(response),
            latency_ms=float(action.get("latency_ms", 0.0)),
            token_usage=copy.deepcopy(
                action.get(
                    "token_usage",
                    {
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                    },
                )
            ),
            finish_reason=action.get("finish_reason"),
            reasoning_discarded=bool(
                action.get("reasoning_discarded", False)
            ),
            safe_response_hash_source=canonical_json(response),
        )
