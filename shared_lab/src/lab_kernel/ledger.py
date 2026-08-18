"""Append-only, hash-chained JSON Lines ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from .canonical import content_hash, file_hash


GENESIS_HASH = "0" * 64


class LedgerIntegrityError(RuntimeError):
    """Raised when any ledger invariant fails."""


class AppendOnlyLedger:
    """Expose append operations only; existing ledgers are never reopened for writing."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path.resolve()
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to append to existing ledger: {self.path}") from exc
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self._sequence = 0
        self._head = GENESIS_HASH
        self._closed = False

    @property
    def head_hash(self) -> str:
        return self._head

    @property
    def count(self) -> int:
        return self._sequence

    def append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Ledger is closed")
        base = {
            "run_id": self.run_id,
            "sequence": self._sequence,
            "event_type": event_type,
            "previous_hash": self._head,
            "payload": dict(payload),
        }
        event = {**base, "event_hash": content_hash(base)}
        self._handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._head = event["event_hash"]
        self._sequence += 1
        return event

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "AppendOnlyLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise LedgerIntegrityError(f"Blank ledger line at {number}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(f"Invalid JSON at line {number}") from exc
            if not isinstance(event, dict):
                raise LedgerIntegrityError(f"Non-object event at line {number}")
            yield event


def verify_ledger(
    path: Path,
    *,
    expected_run_id: str | None = None,
    expected_head: str | None = None,
    expected_file_hash: str | None = None,
) -> dict[str, Any]:
    if expected_file_hash is not None and file_hash(path) != expected_file_hash:
        raise LedgerIntegrityError("Ledger file hash does not match trusted receipt")
    previous = GENESIS_HASH
    count = 0
    run_id = expected_run_id
    for event in iter_events(path):
        required = {"run_id", "sequence", "event_type", "previous_hash", "payload", "event_hash"}
        if set(event) != required:
            raise LedgerIntegrityError(f"Unexpected event fields at sequence {count}")
        if event["sequence"] != count or event["previous_hash"] != previous:
            raise LedgerIntegrityError(f"Broken ordering or hash chain at sequence {count}")
        if run_id is None:
            run_id = event["run_id"]
        if event["run_id"] != run_id:
            raise LedgerIntegrityError(f"Run identifier changed at sequence {count}")
        claimed = event.pop("event_hash")
        actual = content_hash(event)
        event["event_hash"] = claimed
        if claimed != actual:
            raise LedgerIntegrityError(f"Invalid event hash at sequence {count}")
        previous = claimed
        count += 1
    if not count:
        raise LedgerIntegrityError("Ledger is empty")
    if expected_head is not None and previous != expected_head:
        raise LedgerIntegrityError("Ledger head does not match trusted receipt")
    return {"run_id": run_id, "event_count": count, "head_hash": previous, "file_hash": file_hash(path)}
