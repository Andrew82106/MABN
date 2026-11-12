"""Operational output gate for the fork monitor, leaving frozen captures intact."""
from __future__ import annotations

from .authority_risk_queries import query_reference_risk, qualify_two_action_endpoint


class QualifiedBranchedStream:
    def __init__(self, monitor):
        self.monitor, self.last, self.closed = monitor, {}, False

    def push(self, record):
        if self.closed:
            raise ValueError("completed observation stream cannot accept more events")
        raw = self.monitor.push(record)
        result = {"sequence": raw["sequence"], "views": {}, "automatic_blocking": False,
            "future_forecast": False, "proves_system_safety": False}
        for view, content in raw["views"].items():
            dispatches = []
            for row in content["dispatches"]:
                self.last[view] = row
                graph = self.monitor.graphs[view]
                priors = self.monitor.reference.theta[self.monitor.reference.prior_slice]
                dispatches.append({"action_id": row["action_id"], "evidence_insufficient": row["evidence_insufficient"],
                    "reference_any_failure": query_reference_risk(graph, priors),
                    "reference_two_failures": query_reference_risk(graph, priors, at_least=2),
                    "endpoint_heads": qualify_two_action_endpoint(len(graph.proofs), row["model_point_probability"],
                        row["graph_logistic_two_failures"], run_completed=False),
                    "proof_graph": row["graph"]})
            result["views"][view] = {"observed": content["observed"], "dispatches": dispatches}
        return result

    def finish(self, *, completed):
        if self.closed or type(completed) is not bool:
            raise ValueError("single explicit completion status required")
        self.closed = True
        result = {"completed": completed, "views": {}, "future_forecast": False, "proves_system_safety": False}
        for view in self.monitor.bases:
            if view not in self.last:
                result["views"][view] = {"action_count": 0, "endpoint_heads": None,
                    "reason": "No observed actions; not a safe-task or successful-monitoring label."}
                continue
            row = self.last[view]
            graph = self.monitor.graphs[view]
            result["views"][view] = {"action_count": len(graph.proofs), "endpoint_heads": qualify_two_action_endpoint(
                len(graph.proofs), row["model_point_probability"], row["graph_logistic_two_failures"], run_completed=completed)}
        return result
