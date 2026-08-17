"""Loopback-only Ollama provider and read-only local model inspection."""

from __future__ import annotations

import copy
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from p1_benign.providers import (
    DeterministicTestProvider,
    ModelProvider,
    ModelRequest,
    ProviderError,
    ProviderParseError,
    ProviderResult,
    ProviderUnavailableError,
    _safe_json_from_model_content,
    _validate_structured_action,
    sanitize_observable_response,
)

from .core import CANDIDATE_MODELS, canonical_json, safe_error, stable_hash


FORBIDDEN_OLLAMA_OPERATIONS = frozenset({"pull", "create", "copy", "rm"})


def validate_loopback_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise ValueError("Ollama endpoint must be a credential-free http origin")
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama endpoint must be loopback-only")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama endpoint must not include a path or query")
    port = parsed.port or 11434
    return f"http://{host if host != '::1' else '[::1]'}:{port}"


def _run_ollama(command: list[str], timeout_seconds: float = 20.0) -> str:
    if not command or command[0] != "ollama":
        raise ValueError("only local ollama commands are permitted")
    if any(token in FORBIDDEN_OLLAMA_OPERATIONS for token in command[1:]):
        raise PermissionError("model download or mutation commands are unreachable")
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderUnavailableError(safe_error(exc)) from None
    return completed.stdout


def inspect_local_models(
    models: tuple[str, ...] = CANDIDATE_MODELS,
) -> dict[str, Any]:
    """Read inventory/version/metadata only; never installs or alters a model."""

    if tuple(models) != CANDIDATE_MODELS:
        raise ValueError("qualification model set must be exactly the two frozen tags")
    version = _run_ollama(["ollama", "--version"])
    listing = _run_ollama(["ollama", "list"])
    installed = {
        line.split()[0]
        for line in listing.splitlines()[1:]
        if line.split()
    }
    missing = [model for model in models if model not in installed]
    metadata: dict[str, dict[str, str]] = {}
    for model in models:
        if model in installed:
            show = _run_ollama(["ollama", "show", model, "--verbose"])
            metadata[model] = {
                "metadata_hash": stable_hash(show),
                "metadata_available": "true",
            }
    return {
        "ollama_version": version.strip(),
        "inventory_hash": stable_hash(listing),
        "installed_models": sorted(installed),
        "missing_models": missing,
        "models": metadata,
        "download_attempted": False,
        "mutation_command_attempted": False,
    }


class QualificationOllamaProvider(ModelProvider):
    """Strict local adapter for exactly the two approval candidates."""

    provider_id = "ollama_local"
    endpoint_identifier = "loopback_ollama"
    concurrency = 1
    is_live_provider = False

    def __init__(
        self,
        *,
        model_id: str,
        endpoint: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 90.0,
        max_output_tokens: int = 256,
    ) -> None:
        if model_id not in CANDIDATE_MODELS:
            raise ValueError("model tag is not in the frozen qualification candidate set")
        if timeout_seconds != 90.0 or max_output_tokens != 256:
            raise ValueError("provider timeout or token cap differs from the frozen budget")
        self.model_id = model_id
        self.endpoint = validate_loopback_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self.decoding = {
            "temperature": 0,
            "sampling_seed": 20260731,
            "max_output_tokens": 256,
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
                "remote_calls_allowed": False,
            }
        )
        return summary

    def generate(self, request: ModelRequest) -> ProviderResult:
        system = (
            request.role_prompt.strip()
            + "\n/no_think\n"
            + "Do not reveal hidden reasoning. Return only one JSON object matching this response contract:\n"
            + canonical_json(request.response_contract)
        )
        payload = {
            "model": self.model_id,
            "stream": False,
            "think": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": canonical_json(
                        {"phase": request.phase, "visible_messages": request.visible_messages}
                    ),
                },
            ],
            "options": {
                "temperature": 0,
                "seed": 20260731,
                "num_predict": 256,
            },
        }
        http_request = urllib.request.Request(
            f"{self.endpoint}/api/chat",
            data=canonical_json(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                document = json.loads(response.read())
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError(safe_error(exc)) from None
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            message = document["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
        except (KeyError, TypeError):
            raise ProviderParseError("Ollama response did not contain message.content") from None
        parsed, text_reasoning_discarded = _safe_json_from_model_content(content)
        parsed, embedded_reasoning_discarded = sanitize_observable_response(parsed)
        _validate_structured_action(parsed)
        prompt_tokens = document.get("prompt_eval_count")
        completion_tokens = document.get("eval_count")
        total_tokens = (
            prompt_tokens + completion_tokens
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
            else None
        )
        return ProviderResult(
            response=copy.deepcopy(parsed),
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            finish_reason=document.get("done_reason"),
            reasoning_discarded=(
                text_reasoning_discarded
                or embedded_reasoning_discarded
                or bool(message.get("thinking"))
                or "thinking" in document
            ),
            safe_response_hash_source=canonical_json(parsed),
        )


def coding_plan_readiness() -> dict[str, Any]:
    """A policy gate that deliberately never inspects environment credentials."""

    return {
        "provider_id": "xfyun_astron_coding_plan",
        "usage_scope": "interactive_coding_only",
        "allowed_for_automated_experiment": False,
        "credential_checked": False,
        "network_request_attempted": False,
        "ready": False,
        "reason": "automated_experiment_prohibited",
    }


__all__ = [
    "DeterministicTestProvider",
    "QualificationOllamaProvider",
    "ProviderError",
    "ProviderParseError",
    "ProviderUnavailableError",
    "coding_plan_readiness",
    "inspect_local_models",
    "validate_loopback_endpoint",
]
