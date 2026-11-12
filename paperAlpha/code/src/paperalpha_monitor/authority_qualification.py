"""Evidence-qualified streaming output, without changing frozen BN scores.

A visible grant chain cannot establish current authority when revocations may
be missing. Qualification is an observation-contract check, not calibration,
an inferred safe/unsafe label, or a probabilistic coverage guarantee.
"""
from __future__ import annotations

import copy


def qualify(evidence, point_probability):
    features, status = evidence["features"], evidence["reference_status"]
    reasons = []
    if status in {"no_reference", "unresolved_reference"}:
        reasons.append(status)
    if features["binding_incomplete"]:
        reasons.append("incomplete_action_binding")
    if features["coverage_incomplete"]:
        reasons.append("authority_updates_may_be_missing")
    return {"model_point_probability": float(point_probability), "reference_status": status,
        "evidence_insufficient": bool(reasons), "qualification_reasons": reasons,
        "current_authority_assessment": "unresolved" if reasons else "referenced_chain_evaluable_under_observation_contract",
        "calibrated": False, "proves_safety": False, "features": copy.deepcopy(features),
        "candidates": copy.deepcopy(evidence["candidates"])}


class QualifiedDelegationStream:
    def __init__(self, model, compiler):
        if "episode_outcome" not in model.fitted_targets:
            raise ValueError("a separately supervised endpoint head is required")
        self.model, self.compiler, self.engine = model, compiler, model.engine()

    def push(self, record):
        step = self.compiler.push(record)
        if step is not None:
            self.engine.apply(step, self.compiler.compact.variables)
        reports = []
        if step is not None:
            for d in step.base.dispatches:
                reports.append({"action_id": d.action_id, "sequence": step.sequence,
                    **qualify(self.compiler.snapshots[d.action_id].evidence, self.engine.probability)})
        return {"sequence": record["sequence"], "model_step": step is not None, "dispatches": reports}
