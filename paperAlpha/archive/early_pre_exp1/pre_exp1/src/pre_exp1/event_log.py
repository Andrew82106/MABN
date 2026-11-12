"""Append-only JSONL event logging."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .event_payload import CANONICAL_PAYLOAD_FIELD, event_payload_hash
from .models import Event
from .paths import DATA_ROOT, require_data_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AppendOnlyEventLog:
    """Write one complete JSON object per append operation."""

    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        data_root: Path = DATA_ROOT,
    ) -> None:
        self.path = require_data_path(path, data_root)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        episode_id: str,
        event_type: str,
        agent_id: str | None,
        edge_id: str | None,
        treatment: str | None,
        payload: Any,
        state_before_hash: str,
        state_after_hash: str,
        details: dict[str, Any] | None = None,
    ) -> Event:
        event_details = dict(details or {})
        if CANONICAL_PAYLOAD_FIELD in event_details:
            raise ValueError(
                f"{CANONICAL_PAYLOAD_FIELD} is reserved by AppendOnlyEventLog"
            )
        event_details[CANONICAL_PAYLOAD_FIELD] = payload
        event = Event(
            run_id=self.run_id,
            episode_id=episode_id,
            event_id=f"EVT-{self.run_id}-{uuid.uuid4().hex}",
            event_type=event_type,
            timestamp=utc_now(),
            agent_id=agent_id,
            edge_id=edge_id,
            treatment=treatment,
            payload_hash=event_payload_hash(payload),
            state_before_hash=state_before_hash,
            state_after_hash=state_after_hash,
            details=event_details,
        )
        encoded = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
        return event

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def read_events(
    path: Path,
    *,
    data_root: Path = DATA_ROOT,
) -> list[dict[str, Any]]:
    resolved = require_data_path(path, data_root)
    with resolved.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
