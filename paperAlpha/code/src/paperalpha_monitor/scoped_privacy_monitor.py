"""Same-evidence frozen presence models composed with scoped authority worlds.

This is an uncalibrated transport experiment: older models learned on prohibited
values, not this newly tracked authorized universe. Unknown-authority priors and
the zero authority/emission coefficient are declared assumptions, not fitted.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from .privacy_frozen_monitor import FrozenPrivacyMonitor
from .privacy_model_independent import SCHEMA, direct_features
from .policy_gated_privacy import PolicyGatedPrivacyBN, ScopedRelease, ReleaseBinding, ReleaseContext, count_distribution
from .delegation_evidence import DelegationLedger
from .shared_authority_graph import AuthorityProofGraph, AuthorityWindow

OBSERVATION_FIELDS = {"schema", "vertical", "coordinator_text", "worker_text", "tracked_fields", "context",
    "authority_records", "root_principals", "trusted_observers", "coverage_complete", "complete_grant_ids", "window", "proposal_origin"}


def compile_observation(observation):
    o = observation
    if set(o) != OBSERVATION_FIELDS or o["schema"] != "scoped-privacy-observation-v1":
        raise ValueError("strict prior-only observation required")
    if any(not isinstance(o[k], str) for k in ("coordinator_text", "worker_text")):
        raise ValueError("actual completed prior messages required")
    if not o["tracked_fields"]:
        raise ValueError("explicit tracked universe required")
    literals, groups, bindings = set(), [], []
    context = ReleaseContext(**o["context"])
    ledger = DelegationLedger(root_principals=o["root_principals"], trusted_observers=o["trusted_observers"], coverage_complete=o["coverage_complete"])
    for record in o["authority_records"]:
        ledger.observe(record)
    graph = AuthorityProofGraph(AuthorityWindow(**o["window"], complete_grant_ids=tuple(o["complete_grant_ids"])))
    for j, field in enumerate(o["tracked_fields"]):
        if set(field) != {"literal", "kind", "field_names", "resource", "grant_id"}:
            raise ValueError("explicit literal/proposal registry required")
        literal = field["literal"]
        if not isinstance(literal, str) or len(literal) <= 3 or literal.lower() in literals:
            raise ValueError("one explicit nontrivial literal per alias group required")
        literals.add(literal.lower())
        groups.append({"kind": field["kind"], "field_names": field["field_names"],
            "coordinator_present": literal.lower() in o["coordinator_text"].lower(), "worker_present": literal.lower() in o["worker_text"].lower()})
        action_id = f"release-{j}"
        action = {"action_id": action_id, "actor": context.actor, "principal": context.principal, "tool": "release_literal",
            "resource": field["resource"], "arguments": {"recipient": context.recipient, "purpose": context.purpose, "field_names": sorted(field["field_names"])}}
        graph.add(ledger.assess(action, sequence=10 + j, task_id=context.task_id, grant_ids=[field["grant_id"]]))
        bindings.append(ReleaseBinding(tuple(field["field_names"]), action_id, field["resource"]))
    packet = {"schema": SCHEMA, "stage": 2, "vertical": o["vertical"], "coordinator_chars": len(o["coordinator_text"]), "groups": groups}
    return ScopedRelease(packet, graph, context, bindings)


def gated_distribution(sample, probabilities, prior):
    """Give simple baselines the SAME shared worlds, not independent marginals."""
    _, _, mass, _, failures = sample.graph.worlds(np.repeat(logit(prior), 3))
    local = np.asarray(probabilities)[:, None] * np.asarray([failures[a] for a in sample.action_ids], float)
    return mass @ count_distribution(local.T)


class ScopedPrivacyMonitor:
    def __init__(self, folder):
        self.frozen = FrozenPrivacyMonitor(folder)

    def models(self, prior):
        return {name: PolicyGatedPrivacyBN(presence, prior_logits=np.repeat(logit(prior), 3))
                for name, presence in self.frozen.bn.items()}

    def score(self, observation):
        sample = compile_observation(observation)
        data = direct_features([sample.packet])
        estimator = self.frozen.classifiers["2"]["field_histgb"]
        q = estimator.predict_proba(data["design"][data["mask"]])[:, 1]
        current = np.asarray([g["worker_present"] for g in sample.packet["groups"]], float)
        results = {}
        for prior in (.1, .5, .9):
            distributions, details = {}, {}
            for name, model in self.models(prior).items():
                detail = model.predict(sample)
                distributions[name] = detail["violation_count_probabilities"]
                details[name] = detail
            distributions["field_histgb_shared_gate"] = gated_distribution(sample, q, prior).tolist()
            distributions["current_presence_shared_gate"] = gated_distribution(sample, current, prior).tolist()
            # Clear ablation, not a fair headline baseline: deliberately lose shared dependence.
            marginals = np.array(details["bn_coupled"]["violation_marginals"])
            distributions["independent_marginals_ablation"] = count_distribution(marginals).tolist()
            results[str(prior)] = {"count_distributions": distributions,
                "scores": {str(k): {name: float(sum(dist[k:])) for name, dist in distributions.items()} for k in (1, 2)},
                "bn_details": details}
        return {"packet": sample.packet, "results_by_assumed_prior": results, "primary_assumed_prior": .5,
            "field_histgb_appearance": q.tolist(), "calibrated": False, "joint_model_fitted_on_authority_api_data": False,
            "authority_emission_coefficient": 0., "source_scope_shift": "Source trained on prohibited values; current explicit universe includes authorized values",
            "target": "Fixed final message violates specified referenced-authority policy for at least k distinct tracked values",
            "automatic_blocking": False, "specification": sample.specification()}
