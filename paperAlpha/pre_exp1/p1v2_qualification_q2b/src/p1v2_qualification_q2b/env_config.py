"""The live-only .env reader and frozen configuration validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .common import PACKAGE_ROOT
from .errors import ConfigError


REQUIRED_NAMES = ("P1V2_Q2_BASE_URL", "P1V2_Q2_API_KEY", "P1V2_Q2_MODEL")
FROZEN_BASE_URL = "http://127.0.0.1:58661/v1"
FROZEN_MODEL = "gpt-5.3-codex-spark"
LIVE_ENV_PATH = PACKAGE_ROOT.parents[1] / ".env"


@dataclass(frozen=True)
class LiveConfig:
    base_url: str
    api_key: str = field(repr=False, compare=False)
    model: str


def parse_dotenv_lines(lines: object) -> dict[str, str]:
    """Parse only the three permitted names from injected lines; never print values."""
    values: dict[str, str] = {}
    for raw_line in lines:  # type: ignore[union-attr]
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if name not in REQUIRED_NAMES:
            continue
        if name in values:
            raise ConfigError("duplicate_live_variable")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            raise ConfigError("empty_live_variable")
        values[name] = value
    if set(values) != set(REQUIRED_NAMES):
        raise ConfigError("missing_live_variable")
    return values


def parse_dotenv_text(text: str) -> dict[str, str]:
    """Offline-test convenience wrapper for injected fictional text."""
    return parse_dotenv_lines(text.splitlines())


def validate_live_values(values: dict[str, str]) -> LiveConfig:
    if values.get("P1V2_Q2_BASE_URL") != FROZEN_BASE_URL:
        raise ConfigError("base_url_not_frozen")
    if values.get("P1V2_Q2_MODEL") != FROZEN_MODEL:
        raise ConfigError("model_not_frozen")
    key = values.get("P1V2_Q2_API_KEY")
    if not isinstance(key, str) or not key:
        raise ConfigError("invalid_live_key")
    return LiveConfig(base_url=FROZEN_BASE_URL, api_key=key, model=FROZEN_MODEL)


def load_live_config(env_path: Path = LIVE_ENV_PATH) -> LiveConfig:
    """Read the real file only when the explicitly authorized live path invokes this function."""
    try:
        with env_path.open("r", encoding="utf-8") as handle:
            values = parse_dotenv_lines(handle)
    except OSError as exc:
        raise ConfigError("live_env_unreadable") from exc
    return validate_live_values(values)
