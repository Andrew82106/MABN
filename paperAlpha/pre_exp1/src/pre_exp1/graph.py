"""Validated graph configuration that drives P0 message execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import GraphEdge, Message


class GraphConfigError(ValueError):
    pass


class MessageRouteError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphConfig:
    version: str
    nodes: tuple[str, ...]
    edges: dict[str, GraphEdge]
    gated_edge_id: str
    execution_order: tuple[str, ...]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "GraphConfig":
        raw_nodes = config.get("nodes", [])
        if not raw_nodes or len(raw_nodes) != len(set(raw_nodes)):
            raise GraphConfigError("Graph nodes must be non-empty and unique")
        raw_edges = config.get("edges", [])
        edge_ids = [edge.get("edge_id") for edge in raw_edges]
        if not raw_edges or len(edge_ids) != len(set(edge_ids)):
            raise GraphConfigError("Graph edge IDs must be non-empty and unique")
        edges = {
            edge["edge_id"]: GraphEdge(
                edge_id=edge["edge_id"],
                source=edge["source"],
                target=edge["target"],
                allowed=bool(edge["allowed"]),
                gated=bool(edge.get("gated", False)),
            )
            for edge in raw_edges
        }
        graph = cls(
            version=str(config["version"]),
            nodes=tuple(raw_nodes),
            edges=edges,
            gated_edge_id=str(config.get("gated_edge", "")),
            execution_order=tuple(config.get("execution_order", [])),
        )
        graph.validate()
        return graph

    @classmethod
    def from_file(cls, path: Path) -> "GraphConfig":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @property
    def gated_edge(self) -> GraphEdge:
        return self.edges[self.gated_edge_id]

    def validate(self) -> None:
        node_set = set(self.nodes)
        for edge in self.edges.values():
            if edge.source not in node_set or edge.target not in node_set:
                raise GraphConfigError(
                    f"Edge {edge.edge_id!r} references an unknown endpoint"
                )
            if edge.source == edge.target:
                raise GraphConfigError(
                    f"Self-loop is not allowed for edge {edge.edge_id!r}"
                )
        if (
            len(self.execution_order) != len(self.nodes)
            or set(self.execution_order) != node_set
        ):
            raise GraphConfigError(
                "execution_order must cover every configured node exactly once"
            )
        position = {
            node_id: index for index, node_id in enumerate(self.execution_order)
        }
        for edge in self.edges.values():
            if position[edge.source] >= position[edge.target]:
                raise GraphConfigError(
                    "execution_order violates DAG topology at "
                    f"edge {edge.edge_id!r}"
                )
        if self.gated_edge_id not in self.edges:
            raise GraphConfigError(
                f"Configured gated edge {self.gated_edge_id!r} does not exist"
            )
        if not self.gated_edge.allowed:
            raise GraphConfigError("The configured gated edge must be allowed")
        flagged = {edge.edge_id for edge in self.edges.values() if edge.gated}
        if flagged != {self.gated_edge_id}:
            raise GraphConfigError(
                "Exactly the configured gated_edge must have gated=true"
            )

    def edge_between(self, source: str, target: str) -> GraphEdge:
        matches = [
            edge
            for edge in self.edges.values()
            if edge.source == source and edge.target == target
        ]
        if len(matches) != 1:
            raise MessageRouteError(
                f"Expected one edge from {source!r} to {target!r}; "
                f"found {len(matches)}"
            )
        return matches[0]

    def validate_message(self, message: Message) -> GraphEdge:
        edge = self.edges.get(message.edge_id)
        if edge is None:
            raise MessageRouteError(
                f"Message references unknown edge {message.edge_id!r}"
            )
        if (
            edge.source != message.sender_id
            or edge.target != message.recipient_id
        ):
            raise MessageRouteError(
                f"Message endpoints {message.sender_id!r}->{message.recipient_id!r} "
                f"do not match edge {edge.edge_id!r} "
                f"({edge.source!r}->{edge.target!r})"
            )
        if not edge.allowed:
            raise MessageRouteError(
                f"Communication is disabled on edge {edge.edge_id!r}"
            )
        return edge

    def consecutive_edges(self) -> tuple[GraphEdge, ...]:
        return tuple(
            self.edge_between(source, target)
            for source, target in zip(
                self.execution_order,
                self.execution_order[1:],
            )
        )
