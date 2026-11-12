"""Scalar joint-evidence audit, independent of vector BN likelihood/gradients."""
import math


def evidence_probability(graph, observations):
    nodes, seen, free = graph["nodes"], set(), 0
    if any(type(v) is not int or v not in (0, 1) for v in observations.values()):
        raise ValueError("binary observations required")
    for node in nodes:
        if node["id"] in seen or not set(node["parents"]) <= seen:
            raise ValueError("topologically ordered graph required")
        seen.add(node["id"])
        if node["kind"] == "binary_cpt":
            probabilities = node["probability_one"]
            if len(probabilities) != 2 ** len(node["parents"]) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError("valid CPT required")
            free += "observed" not in node and node["id"] not in observations
    if free > 18 or not set(observations) <= seen:
        raise ValueError("small graph and known observations required")

    def visit(index, assignment):
        if index == len(nodes):
            return 1.
        node = nodes[index]
        identity = node["id"]
        parents = [assignment[p] for p in node["parents"]]
        if "observed" in node and identity in observations and node["observed"] != observations[identity]:
            return 0.
        expected = observations.get(identity, node.get("observed"))
        if node["kind"] == "binary_cpt":
            position = 0
            for bit in parents:
                position = 2 * position + bit
            q, result = node["probability_one"][position], 0.
            for bit in (0, 1) if expected is None else (expected,):
                chance = q if bit else 1 - q
                if chance:
                    assignment[identity] = bit
                    result += chance * visit(index + 1, assignment)
            return result
        if node["kind"] == "and":
            bit = int(all(parents))
        elif node["kind"] == "count_at_least":
            bit = int(sum(parents) >= node["threshold"])
        elif node["kind"] == "all_paths_failed":
            bit = int(not any(all(assignment[p] == 0 for p in path) for path in node["support_paths"]))
        else:
            raise ValueError("unsupported factor")
        if expected is not None and bit != expected:
            return 0.
        assignment[identity] = bit
        return visit(index + 1, assignment)

    return visit(0, {})
