"""Minimal, deterministic API queue shared by the four MAS topologies.

This module is deliberately smaller than any of the paper runners.  It is a
transport contract: topology and observed API outcomes are public, while an
optional evaluator callback produces a separate, evaluator-only label.  A
failed call is a normal terminal record and is never removed or retried.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
import json
from typing import Any, Callable, Iterable, Mapping
import uuid


SCHEMA = "paperalpha-unified-topology-queue-v1"
TOPOLOGIES = frozenset({"chain", "fork", "join", "review"})
_PRIVATE_KEYS = frozenset({
    "label", "labels", "expected", "expected_label", "ground_truth",
    "truth", "oracle", "evaluation", "evaluator", "score",
})


def _json_copy(value: Any, field_name: str) -> Any:
    """Copy and validate JSON-native data before it enters a record."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc


def _name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {field_name} required")
    return value


def topology_edges(topology: str, nodes: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return one deterministic edge list for a named topology.

    ``chain`` is a linear pipeline, ``fork`` has one root and independent
    leaves, ``join`` has independent inputs and one sink, and ``review`` is a
    work node followed by a reviewer (additional named stages are allowed as a
    linear chain).  Names are caller supplied; no hidden graph is inferred.
    """
    topology = _name(topology, "topology").lower()
    if topology not in TOPOLOGIES:
        raise ValueError(f"unsupported topology: {topology}")
    values = tuple(_name(node, "node") for node in nodes)
    if not values or len(set(values)) != len(values):
        raise ValueError("one or more unique nodes required")
    if topology == "chain" or topology == "review":
        if topology == "review" and len(values) < 2:
            raise ValueError("review requires a work node and a reviewer")
        return tuple(zip(values, values[1:]))
    if len(values) < 2:
        raise ValueError(f"{topology} requires at least two nodes")
    if topology == "fork":
        return tuple((values[0], node) for node in values[1:])
    return tuple((node, values[-1]) for node in values[:-1])


def _public(value: Any) -> Any:
    """Project untrusted values without evaluator-only fields.

    The allowlist is intentionally key based and recursive.  Evaluator data is
    never needed by the monitor, so private-looking fields are omitted rather
    than guessed or transformed into a monitor feature.
    """
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()
                if str(key).lower() not in _PRIVATE_KEYS}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, tuple):
        return [_public(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class QueueRequest:
    request_id: str
    topology: str
    node: str
    payload: Any
    parent_ids: tuple[str, ...] = ()


@dataclass
class QueueRun:
    run_id: str
    topology: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    records: list[dict[str, Any]] = field(default_factory=list)
    _evaluation: Any = field(default=None, repr=False)

    def monitor_projection(self) -> dict[str, Any]:
        """Return only deployment-visible observations.

        This method builds a fresh object each time; adding an evaluator result
        later cannot mutate a previously returned monitor snapshot.
        """
        records = []
        for record in self.records:
            item = {"sequence": record["sequence"], "request_id": record["request_id"],
                    "node": record["node"], "topology": record["topology"],
                    "parent_ids": list(record["parent_ids"]), "status": record["status"],
                    "request": _public(record["request"])}
            if "response" in record:
                item["response"] = _public(record["response"])
            if "error_type" in record:
                item["error_type"] = record["error_type"]
            if "error_message" in record:
                item["error_message"] = record["error_message"]
            records.append(item)
        return {"schema": SCHEMA, "run_id": self.run_id, "topology": self.topology,
                "nodes": list(self.nodes), "edges": [list(edge) for edge in self.edges],
                "records": records,
                "failed_request_ids": [r["request_id"] for r in records if r["status"] == "failed"]}

    def evaluation_view(self) -> Any:
        """Return evaluator output only; never include it in monitor data."""
        return copy.deepcopy(self._evaluation)

    def snapshot(self) -> dict[str, Any]:
        result = {"monitor": self.monitor_projection()}
        if self._evaluation is not None:
            result["evaluation"] = self.evaluation_view()
        return result


class UnifiedTopologyQueue:
    """One-shot FIFO queue for chain/fork/join/review API calls."""

    def __init__(self, topology: str, nodes: Iterable[str], *, run_id: str | None = None):
        self.topology = _name(topology, "topology").lower()
        self.nodes = tuple(_name(node, "node") for node in nodes)
        self.edges = topology_edges(self.topology, self.nodes)
        self.run_id = run_id or uuid.uuid4().hex
        self._requests: list[QueueRequest] = []
        self._request_ids_by_node: dict[str, str] = {}
        self._used_nodes: set[str] = set()
        self._run: QueueRun | None = None

    def submit(self, node: str, payload: Any = None, *, request_id: str | None = None) -> QueueRequest:
        """Queue one node call; each node is submitted at most once."""
        if self._run is not None:
            raise RuntimeError("queue already executed")
        node = _name(node, "node")
        if node not in self.nodes:
            raise ValueError(f"unknown node: {node}")
        if node in self._used_nodes:
            raise ValueError(f"node already submitted: {node}")
        identity = _name(request_id, "request_id") if request_id is not None else f"{self.run_id}:{node}"
        if any(r.request_id == identity for r in self._requests):
            raise ValueError(f"duplicate request_id: {identity}")
        parents = tuple(r.request_id for r in self._requests if (r.node, node) in self.edges)
        request = QueueRequest(identity, self.topology, node, _json_copy(payload, "payload"), parents)
        self._requests.append(request)
        self._request_ids_by_node[node] = identity
        self._used_nodes.add(node)
        return request

    def submit_all(self, payloads: Mapping[str, Any] | None = None) -> tuple[QueueRequest, ...]:
        payloads = payloads or {}
        if not isinstance(payloads, Mapping):
            raise ValueError("payloads must be a mapping")
        return tuple(self.submit(node, payloads.get(node)) for node in self.nodes)

    def run(self, handler: Callable[[QueueRequest, Mapping[str, Any]], Any], *,
            evaluator: Callable[[Mapping[str, Any]], Any] | None = None) -> QueueRun:
        """Execute queued calls once, retaining successes and failures.

        ``handler`` receives the request and a mapping of parent request IDs to
        terminal records.  A handler exception becomes a failed record and does
        not cancel independent nodes or erase a later join/review call.
        """
        if self._run is not None:
            raise RuntimeError("queue already executed")
        if not callable(handler):
            raise ValueError("callable handler required")
        missing = [node for node in self.nodes if node not in self._used_nodes]
        if missing:
            raise ValueError(f"all topology nodes must be submitted; missing {missing}")
        result = QueueRun(self.run_id, self.topology, self.nodes, self.edges)
        by_id: dict[str, dict[str, Any]] = {}
        # Submission follows topology order in normal use.  The scheduler also
        # handles a caller-supplied order by waiting until all parents exist.
        pending = list(self._requests)
        sequence = 0
        while pending:
            # Recompute dependencies from the explicit topology so submission
            # order cannot accidentally erase an information edge.
            effective = {
                r.request_id: replace(r, parent_ids=tuple(
                    self._request_ids_by_node[item.node]
                    for item in self._requests
                    if (item.node, r.node) in self.edges
                ))
                for r in pending
            }
            ready = next((r for r in pending if all(parent in by_id for parent in effective[r.request_id].parent_ids)), None)
            if ready is None:
                raise RuntimeError("queued requests do not form a schedulable topology")
            pending.remove(ready)
            ready = effective[ready.request_id]
            parent_records = {parent: copy.deepcopy(by_id[parent]) for parent in ready.parent_ids}
            sequence += 1
            base = {"sequence": sequence, "request_id": ready.request_id, "node": ready.node,
                    "topology": ready.topology, "parent_ids": list(ready.parent_ids),
                    "request": _json_copy(ready.payload, "payload")}
            try:
                response = handler(ready, parent_records)
                response = _json_copy(response, "handler response")
                record = {**base, "status": "completed", "response": response}
            except Exception as exc:  # failure is data, not a reason to drop a row
                record = {**base, "status": "failed", "error_type": type(exc).__name__,
                          "error_message": str(exc) or type(exc).__name__}
            result.records.append(record)
            by_id[ready.request_id] = record
        self._run = result
        if evaluator is not None:
            if not callable(evaluator):
                raise ValueError("evaluator must be callable")
            private = {"run_id": result.run_id, "topology": result.topology,
                       "nodes": list(result.nodes), "edges": [list(edge) for edge in result.edges],
                       "records": copy.deepcopy(result.records)}
            result._evaluation = _json_copy(evaluator(private), "evaluation")
        return result


def run_topology(topology: str, nodes: Iterable[str], handler: Callable[[QueueRequest, Mapping[str, Any]], Any],
                 payloads: Mapping[str, Any] | None = None, *, evaluator: Callable[[Mapping[str, Any]], Any] | None = None,
                 run_id: str | None = None) -> QueueRun:
    """Convenience API used by small scripts and fixtures."""
    queue = UnifiedTopologyQueue(topology, nodes, run_id=run_id)
    queue.submit_all(payloads)
    return queue.run(handler, evaluator=evaluator)

