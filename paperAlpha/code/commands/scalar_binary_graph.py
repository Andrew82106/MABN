"""Small-graph audit interpreter, independent of production BN numerics."""
from __future__ import annotations

import math


def evaluate(graph, *, max_free_nodes=18):
    nodes = graph["nodes"]
    seen = set()
    free = 0
    for node in nodes:
        if node["id"] in seen or not set(node["parents"]) <= seen:
            raise ValueError("unique topologically ordered graph required")
        seen.add(node["id"])
        if node["kind"] == "binary_cpt":
            probs = node["probability_one"]
            if len(probs) != 2 ** len(node["parents"]) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
                raise ValueError("proper binary conditional table required")
            free += "observed" not in node
    if free > max_free_nodes or graph["output_id"] not in seen:
        raise ValueError("small explicit output graph required")
    positive, total = 0., 0.

    def visit(index, assignment, weight):
        nonlocal positive, total
        if index == len(nodes):
            total += weight
            positive += weight * assignment[graph["output_id"]]
            return
        node = nodes[index]
        parents = [assignment[p] for p in node["parents"]]
        if node["kind"] == "binary_cpt":
            position = 0
            for p in parents:
                position = 2 * position + p
            q = node["probability_one"][position]
            for value in [node["observed"]] if "observed" in node else (0, 1):
                chance = q if value else 1 - q
                if chance:
                    assignment[node["id"]] = value
                    visit(index + 1, assignment, weight * chance)
            return
        if node["kind"] == "and":
            value = int(all(parents))
        elif node["kind"] == "count_at_least":
            value = int(sum(parents) >= node["threshold"])
        elif node["kind"] == "all_paths_failed":
            value = int(not any(all(assignment[p] == 0 for p in path) for path in node["support_paths"]))
        else:
            raise ValueError("unknown deterministic operator")
        assignment[node["id"]] = value
        visit(index + 1, assignment, weight)

    visit(0, {}, 1.)
    if total <= 0:
        raise ValueError("evidence has zero probability")
    return positive / total
