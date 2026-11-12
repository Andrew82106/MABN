"""One process-local observation order, shared by messages and tool hooks."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time


KINDS = frozenset({"run_started", "communication", "tool_handler_started", "tool_native_result",
                   "tool_handler_returned", "tool_handler_failed", "run_finished"})


class RuntimeJournal:
    def __init__(self, path):
        self.path = Path(path)
        # No append-to-old-run ambiguity; callers must create a new run.
        with self.path.open("x", encoding="utf-8"):
            pass
        self._sequence, self._lock = 0, threading.Lock()

    @property
    def count(self):
        return self._sequence

    def emit(self, kind, payload):
        if kind not in KINDS or not isinstance(payload, dict):
            raise ValueError("supported journal kind and object payload required")
        # Freeze external objects and validate before assigning a sequence.
        frozen = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        with self._lock:
            row = {"schema_version": "runtime-journal-v1", "sequence": self._sequence + 1,
                   "observed_at": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
                   "kind": kind, "payload": frozen}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._sequence += 1
        return copy.deepcopy(row)


def nonempty(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {field} required")
    return value


class RuntimeEvidenceProjector:
    """Project one freshly observed journal event; no look-ahead or sorting.

    External endpoints are not silently promoted to MAS agents. Raw content
    stays opaque; this module does not discover provenance or authorization.
    """
    def __init__(self):
        self.sequence, self.agents, self.external = 0, None, set()
        self.started, self.ended, self.native = {}, {}, set()
        self.finished = None
        self.unmodeled = []
        self.counts = {kind: 0 for kind in KINDS}

    def push(self, row):
        if row.get("schema_version") != "runtime-journal-v1":
            raise ValueError("a shared runtime journal is required; old logs cannot be silently upgraded")
        sequence, kind, data = row.get("sequence"), row.get("kind"), row.get("payload")
        if type(sequence) is not int or sequence != self.sequence + 1:
            raise ValueError("journal sequence must be contiguous in recorded order; do not sort")
        if kind not in KINDS or not isinstance(data, dict) or self.finished is not None:
            raise ValueError("unsupported record or event after run completion")
        event = None
        base = {"sequence": sequence, "event_id": f"journal-{sequence}"}
        if kind == "run_started":
            agents, external = data.get("agents"), data.get("external_endpoints")
            if self.agents is not None or not isinstance(agents, list) or not agents or not isinstance(external, list):
                raise ValueError("one run start with explicit agent and external endpoint lists required")
            if any(not isinstance(v, str) or not v for v in [*agents, *external]) or len(set(agents)) != len(agents) or set(agents).intersection(external):
                raise ValueError("invalid or overlapping participants")
            self.agents, self.external = set(agents), set(external)
        else:
            if self.agents is None:
                raise ValueError("run_started must precede observations")
            if kind == "run_finished":
                if data.get("status") not in {"completed", "failed"}:
                    raise ValueError("explicit run completion status required")
                self.finished = copy.deepcopy(data)
            elif kind == "communication":
                actor, target = data.get("actor"), data.get("target")
                if actor not in self.agents or target not in self.agents.union(self.external) or not isinstance(data.get("content"), str):
                    raise ValueError("communication participants/content do not match the declared run")
                event = {**base, "event_type": "message_sent" if target in self.agents else "external_message_observed",
                         "actor": actor, "target": target, "content": data["content"], "data_class": "unknown"}
            else:
                identity = nonempty(data.get("invocation_id"), "invocation_id")
                actor, tool = data.get("actor"), data.get("tool")
                if actor not in self.agents:
                    raise ValueError("tool actor is not a declared agent")
                nonempty(tool, "tool")
                if kind == "tool_handler_started":
                    if identity in self.started:
                        raise ValueError("duplicate actual tool invocation")
                    if isinstance(data.get("arguments"), dict):
                        event = {**base, "event_type": "tool_call", "actor": actor, "action_id": identity,
                                 "tool": tool, "arguments": copy.deepcopy(data["arguments"]), "principal": None, "resource": None}
                    else:
                        event = {**base, "event_type": "unparsed_tool_entry", "actor": actor, "action_id": identity}
                        self.unmodeled.append({"sequence": sequence, "action_id": identity, "reason": "arguments_not_a_structured_object"})
                    self.started[identity] = copy.deepcopy(data)
                else:
                    start = self.started.get(identity)
                    if start is None or any(start.get(k) != data.get(k) for k in ("actor", "tool", "provider_call_id")):
                        raise ValueError("tool stage is not bound to its actual entry")
                    if identity in self.ended:
                        raise ValueError("tool evidence after terminal handler record")
                    if kind == "tool_native_result":
                        if identity in self.native:
                            raise ValueError("duplicate native tool result")
                        if isinstance(start.get("arguments"), dict) and start["arguments"] != data.get("arguments"):
                            raise ValueError("native tool arguments differ from handler entry")
                        self.native.add(identity)  # Audit link only; never a second action.
                    else:
                        event = {**base, "actor": actor, "action_id": identity,
                            "event_type": "tool_return_observed" if kind == "tool_handler_returned" else "tool_failure_observed"}
                        if kind == "tool_handler_returned":
                            event["content"] = json.dumps(data.get("result"), ensure_ascii=False, allow_nan=False)
                        else:
                            event["error_type"] = nonempty(data.get("error_type"), "error_type")
                        self.ended[identity] = kind
        self.sequence = sequence
        self.counts[kind] += 1
        return event

    def report(self):
        return {"journal_events": self.sequence, "counts": dict(self.counts),
            "run_finished": self.finished is not None,
            "run_status": None if self.finished is None else self.finished["status"],
            "open_invocations": sorted(set(self.started) - set(self.ended)), "unmodeled": copy.deepcopy(self.unmodeled),
            "information_semantics_available": False, "authorization_scope_available": False,
            "order": "single-process observation order, not inferred distributed causality",
            "boundary": "content/results preserved but not semantically classified; no label or post-run audit input"}
