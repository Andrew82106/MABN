"""Scientific-layer provider and frozen qwen3:8b preflight."""

from __future__ import annotations

import subprocess
from typing import Any

from p1_benign.providers import DeterministicTestProvider
from p1_model_qualification.providers import QualificationOllamaProvider

from .core import (
    ENDPOINT,
    ENDPOINT_IDENTIFIER,
    ENDPOINT_ORIGIN_HASH,
    MODEL_DECODING,
    MODEL_ID,
    OLLAMA_VERSION,
    QUALIFICATION_METADATA_HASH,
    QUALIFICATION_RUN_ID,
    stable_hash,
)


FORBIDDEN_COMMANDS = frozenset({"pull", "create", "copy", "rm"})


class ScientificOllamaProvider(QualificationOllamaProvider):
    """The qualification adapter, narrowed to the frozen scientific model."""

    def __init__(self, *, endpoint: str = ENDPOINT) -> None:
        super().__init__(model_id=MODEL_ID, endpoint=endpoint, timeout_seconds=90.0, max_output_tokens=256)

    def manifest_summary(self) -> dict[str, Any]:
        summary = super().manifest_summary()
        summary.update({"scientific_model_frozen": True, "qualification_run_id": QUALIFICATION_RUN_ID, "remote_calls_allowed": False})
        return summary


def _run_readonly(command: list[str]) -> str:
    if command[:1] != ["ollama"] or any(token in FORBIDDEN_COMMANDS for token in command[1:]):
        raise PermissionError("Ollama mutation commands are unreachable")
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return completed.stdout


def inspect_qwen3() -> dict[str, Any]:
    """Run exactly the three permitted local read-only Ollama commands."""
    version = _run_readonly(["ollama", "--version"]).strip()
    listing = _run_readonly(["ollama", "list"])
    installed = [line.split()[0] for line in listing.splitlines()[1:] if line.split()]
    if MODEL_ID not in installed:
        raise RuntimeError("qwen3:8b is not installed; no pull is permitted")
    metadata_text = _run_readonly(["ollama", "show", MODEL_ID, "--verbose"])
    return {
        "ollama_version": version,
        "installed_models": sorted(set(installed)),
        "inventory_hash": stable_hash(listing),
        "model_id": MODEL_ID,
        "metadata_hash": stable_hash(metadata_text),
        "metadata_available": True,
        "endpoint": ENDPOINT,
        "endpoint_identifier": ENDPOINT_IDENTIFIER,
        "endpoint_origin_hash": ENDPOINT_ORIGIN_HASH,
        "download_attempted": False,
        "mutation_command_attempted": False,
        "commands": ["ollama --version", "ollama list", "ollama show qwen3:8b --verbose"],
    }


def validate_frozen_preflight(observed: dict[str, Any]) -> None:
    expected = {
        "ollama_version": OLLAMA_VERSION,
        "model_id": MODEL_ID,
        "metadata_hash": QUALIFICATION_METADATA_HASH,
        "endpoint": ENDPOINT,
        "endpoint_identifier": ENDPOINT_IDENTIFIER,
        "endpoint_origin_hash": ENDPOINT_ORIGIN_HASH,
        "download_attempted": False,
        "mutation_command_attempted": False,
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"frozen qwen preflight mismatch: {key}")
    if observed.get("commands") != ["ollama --version", "ollama list", "ollama show qwen3:8b --verbose"]:
        raise ValueError("preflight command set is not frozen")


def frozen_provider_identity(provider: ScientificOllamaProvider) -> dict[str, Any]:
    if provider.model_id != MODEL_ID or provider.endpoint != ENDPOINT or provider.decoding != MODEL_DECODING or provider.concurrency != 1 or provider.timeout_seconds != 90.0:
        raise ValueError("scientific provider identity drift")
    return {"provider_id": "ollama_local", "model_id": MODEL_ID, "endpoint_identifier": ENDPOINT_IDENTIFIER, "endpoint_origin_hash": ENDPOINT_ORIGIN_HASH, "decoding": MODEL_DECODING, "timeout_seconds": 90.0, "concurrency": 1, "retry_per_call": 0, "remote_calls_allowed": False, "automatic_model_pull": False, "qualification_run_id": QUALIFICATION_RUN_ID, "qualification_metadata_hash": QUALIFICATION_METADATA_HASH}


__all__ = ["DeterministicTestProvider", "ScientificOllamaProvider", "inspect_qwen3", "validate_frozen_preflight", "frozen_provider_identity"]
