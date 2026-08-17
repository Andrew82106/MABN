"""Frozen, public Q2-C protocol definitions.

Prompts and expected answer values exist only while a request is constructed in
memory.  Artifacts receive the resulting canonical request digest, never the
request body or content.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any


# The original offline value is deliberately retained as the default so that
# the already-persisted fake qualification run remains independently
# verifiable.  Live evidence has a distinct, non-offline schema identity.
OFFLINE_SCHEMA_VERSION = "q2c-probe-offline-v1"
LIVE_SCHEMA_VERSION = "q2c-probe-live-v1"
SCHEMA_VERSION = OFFLINE_SCHEMA_VERSION
TARGET_MODEL = "gpt-5.3-codex-spark"
BASE_URL_IDENTITY = "http://127.0.0.1:58661/v1"
MODELS_ENDPOINT = "/models"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
TOTAL_DEADLINE_SECONDS = 150.0
PER_CALL_TIMEOUT_SECONDS = 30.0
MAX_IDENTITY_LENGTH = 160
MAX_CONTENT_LENGTH = 16_384


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


@dataclass(frozen=True)
class ProbeCard:
    card_id: str
    profile: str
    response_format_kind: str
    prompt: str

    def request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": TARGET_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": self.prompt}],
        }
        if self.response_format_kind == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.response_format_kind == "strict_json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "q2c_minimal_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["probe_id", "answer"],
                        "properties": {
                            "probe_id": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                    },
                },
            }
        return payload

    @property
    def request_sha256(self) -> str:
        return sha256_json(self.request_payload())

    def expected_probe_id(self) -> str:
        return f"Q2C-{self.profile}"

    def expected_answer(self) -> str:
        return "Q2C_OK"

    def public_snapshot(self) -> dict[str, Any]:
        """The artifact-safe profile representation, with no prompt or answer."""
        return {
            "card_id": self.card_id,
            "profile": self.profile,
            "response_format_kind": self.response_format_kind,
            "request_sha256": self.request_sha256,
            "expected_field_names": ["probe_id", "answer"]
            if self.profile != "plain_text"
            else [],
        }


CARDS: tuple[ProbeCard, ...] = (
    ProbeCard(
        card_id="Q2C-CARD-01-PLAIN",
        profile="plain_text",
        response_format_kind="none",
        prompt="Return exactly the short text Q2C_OK. Use ordinary assistant content and do not call tools.",
    ),
    ProbeCard(
        card_id="Q2C-CARD-02-PROMPT-JSON",
        profile="prompt_only_json",
        response_format_kind="none",
        prompt=(
            "Return only a JSON object with probe_id set to Q2C-prompt_only_json "
            "and answer set to Q2C_OK. Use ordinary assistant content and do not call tools."
        ),
    ),
    ProbeCard(
        card_id="Q2C-CARD-03-JSON-OBJECT",
        profile="json_object",
        response_format_kind="json_object",
        prompt=(
            "Return only a JSON object with probe_id set to Q2C-json_object "
            "and answer set to Q2C_OK. Use ordinary assistant content and do not call tools."
        ),
    ),
    ProbeCard(
        card_id="Q2C-CARD-04-STRICT-SCHEMA",
        profile="strict_json_schema",
        response_format_kind="strict_json_schema",
        prompt=(
            "Return only a JSON object with probe_id set to Q2C-strict_json_schema "
            "and answer set to Q2C_OK. Use ordinary assistant content and do not call tools."
        ),
    ),
)


def schema_version_for_run_kind(run_kind: str) -> str:
    """Return the artifact schema bound to one supported execution kind."""
    if run_kind == "offline_fake":
        return OFFLINE_SCHEMA_VERSION
    if run_kind == "live_probe":
        return LIVE_SCHEMA_VERSION
    raise ValueError("unsupported_run_kind")


def public_profile_snapshot(run_kind: str = "offline_fake") -> dict[str, Any]:
    """Return the public profile commitment for one explicitly named run kind."""
    return {
        "schema_version": schema_version_for_run_kind(run_kind),
        "target_model": TARGET_MODEL,
        "base_url_identity": BASE_URL_IDENTITY,
        "models_endpoint": MODELS_ENDPOINT,
        "chat_completions_endpoint": CHAT_COMPLETIONS_ENDPOINT,
        "stream": False,
        "cards": [card.public_snapshot() for card in CARDS],
    }


def profile_commitment(run_kind: str = "offline_fake") -> str:
    return sha256_json(public_profile_snapshot(run_kind))


def card_for_profile(profile: str) -> ProbeCard:
    for card in CARDS:
        if card.profile == profile:
            return card
    raise KeyError(profile)
