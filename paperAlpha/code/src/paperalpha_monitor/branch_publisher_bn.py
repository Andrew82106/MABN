"""Branch-local disclosure states, publisher aggregation and shared authority.

This fixed-DAG component consumes actual completed branch messages. It does not
infer model internals, discover a workflow, or guarantee safety. Worker states
are conditionally independent given the declared context; emissions share
parameters, not latent bits. Current authority never conditions appearance.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.special import expit, logsumexp

from .branch_privacy_input import BranchPrivacyInput
from .policy_gated_privacy import PolicyGatedPrivacyBN, count_distribution
from .privacy_flow_data import KINDS
from .privacy_model_independent import compile_packets


class BranchPublisherPrivacyBN(PolicyGatedPrivacyBN):
    MODEL_NAME = "BranchPublisherPrivacyBN-v1"
    EMISSION_NAMES = ("publisher_supported_emission", "publisher_contradicted_emission",
                      "nonowner_emission_shift")

    def _components(self, sample, theta):
        if not isinstance(sample, BranchPrivacyInput):
            raise ValueError("branch-preserving input required, not a pooled release")
        sample.validate()
        t = np.asarray(theta, float)
        if t.shape != self.theta.shape or not np.isfinite(t).all():
            raise ValueError("compatible finite branch parameter vector required")
        data = compile_packets([sample.release.packet])
        if data["stage"] != 2:
            raise ValueError("completed worker messages required at this cutoff")
        m = len(sample.workers)
        bits = np.asarray(list(product((0., 1.), repeat=m + 1)))
        u, v = bits[:, :m], bits[:, m]
        mean_u = u.mean(axis=1)
        x = np.r_[1., data["context"][0]]
        d, k, boost = self.presence.d, self.presence.k, self.presence.boost_start
        lu = float(x @ t[2 * d:2 * d + k])
        lv = x @ t[2 * d + k:2 * d + 2 * k] + mean_u * (t[self.n - 1] if self.presence.coupled else 0.)
        prior = u.sum(axis=1) * lu - m * np.logaddexp(0., lu)
        prior += v * lv - np.logaddexp(0., lv)
        nonowner = 1. - sample.declared_owner_mask.astype(float)
        qw = (data["dc"][0] @ t[:d])[:, None, None] + u.T[None] * t[boost]
        qw = qw + nonowner[:, :, None] * t[self.n + 5]
        rf = (data["dw"][0] @ t[d:2 * d])[:, None] + v[None] * t[boost + 1]
        rf += (sample.release.visible_features @ t[self.n + 3:self.n + 5])[:, None]
        w = sample.worker_presence.astype(float)
        log_w = np.sum(w[:, :, None] * qw - np.logaddexp(0., qw), axis=(0, 1))
        _, _, mass, score, failures = sample.release.graph.worlds(t[self.n:self.n + 3])
        failed = np.asarray([failures[a] for a in sample.release.action_ids], float)
        return dict(data=data, x=x, u=u, v=v, mean_u=mean_u, lu=lu, lv=lv, prior=prior,
                    qw=qw, rf=rf, w=w, nonowner=nonowner, log_w=log_w,
                    mass=mass, score=score, failed=failed)

    def predict(self, sample, *, at_least=1, theta=None):
        if type(at_least) is not int or at_least < 1:
            raise ValueError("positive integer violation count required")
        p = self._components(sample, self.theta if theta is None else theta)
        log_post = p["prior"] + p["log_w"]
        states = np.exp(log_post - logsumexp(log_post))
        r = expit(p["rf"])
        joint = states[:, None] * p["mass"][None]
        local = r[:, :, None] * p["failed"][:, None, :]
        counts = count_distribution(local.transpose(1, 2, 0))
        distribution = np.einsum("sb,sbc->c", joint, counts)
        conditional = counts[..., at_least:].sum(axis=-1)
        per_world = states @ conditional
        trained = self.optimizer is not None
        return {"model": self.MODEL_NAME, "model_probability": float(distribution[at_least:].sum()),
            "violation_count_probabilities": distribution.tolist(),
            "appearance_marginals": (r @ states).tolist(),
            "violation_marginals": np.einsum("sb,jsb->j", joint, local).tolist(),
            "reference_failure_marginals": (p["failed"] @ p["mass"]).tolist(),
            "state_labels": [*["U:" + w for w in sample.workers], "V:publisher"],
            "state_assignments": np.c_[p["u"], p["v"]].astype(int).tolist(),
            "stage_state_weights": states.tolist(), "worker_state_marginals": (states @ p["u"]).tolist(),
            "publisher_state_marginal": float(states @ p["v"]), "authority_world_count": len(p["mass"]),
            "authority_state_conditional_range": [float(per_world.min()), float(per_world.max())],
            "range_meaning": "Prediction range across compatible authority worlds; not confidence or safety bounds",
            "emission_uses_current_authority_truth": False, "publisher_input_is_internal_belief": False,
            "at_least": at_least, "phase": "after all worker replies, before final release",
            "joint_likelihood_fitted": trained,
            "jointly_trained": trained and set(self.optimizer["trainable_names"]) == set(self.names),
            "optimized_parameter_count": len(self.optimizer["trainable_names"]) if trained else 0,
            "calibrated": False, "proves_system_safety": False, "fixed_window_assumption": True,
            "scope": sample.specification(), "model_scope": self.parameters()["scope"]}

    def joint_appearance_law(self, sample, *, theta=None, max_fields=10):
        """Exact P(F-vector | observed W), with analytic log-probability scores.

        Summing the existing joint W/F likelihood over every F vector provides
        the W evidence normalizer. No independent-marginal approximation and
        no new fitted parameters. Authority labels are deliberately absent:
        policy truth is combined exactly once by the downstream policy graph.
        This conditional objective differs from the original joint W/F fit.
        """
        t = np.asarray(self.theta if theta is None else theta, float)
        if not isinstance(sample, BranchPrivacyInput):
            raise ValueError("branch-preserving prefix required")
        count = len(sample.release.action_ids)
        if type(max_fields) is not int or not 1 <= max_fields <= 12 or count > max_fields:
            raise ValueError("bounded explicit joint appearance table required")
        bits = np.asarray(list(product((0, 1), repeat=count)), int)
        results = [self._likelihood(sample, row, [None] * count, t) for row in bits]
        logs = -np.asarray([r[0] for r in results])
        scores = -np.asarray([r[1] for r in results])
        log_evidence = float(logsumexp(logs))
        probabilities = np.exp(logs - log_evidence)
        evidence_score = probabilities @ scores
        return {"action_ids": sample.release.action_ids, "states": bits,
            "log_probabilities": logs - log_evidence, "probabilities": probabilities,
            "log_probability_gradients": scores - evidence_score,
            "log_worker_evidence": log_evidence, "log_worker_evidence_gradient": evidence_score,
            "event_semantics": "Exact tracked literal occurrence in the next bound publisher reply; not tool dispatch"}

    def _likelihood(self, sample, f, authority_labels, theta, violation_label=None, at_least=1):
        p = self._components(sample, theta)
        f = np.asarray(f)
        if f.shape != (p["w"].shape[0],) or not np.isin(f, [0, 1]).all():
            raise ValueError("one binary final-appearance label per literal required")
        if len(authority_labels) != len(f) or any(a is not None and (type(a) is not int or a not in (0, 1)) for a in authority_labels):
            raise ValueError("aligned authority-failure labels, or explicit None, required")
        compatible = np.ones(len(p["mass"]), bool)
        for j, label in enumerate(authority_labels):
            if label is not None:
                compatible &= p["failed"][j] == label
        if violation_label is not None:
            if type(violation_label) is not int or violation_label not in (0, 1):
                raise ValueError("optional binary system label required")
            compatible &= (np.sum(f[:, None] * p["failed"], axis=0) >= at_least) == violation_label
        if not compatible.any():
            raise ValueError("labels contradict shared authority or system target")
        log_f = np.sum(f[:, None] * p["rf"] - np.logaddexp(0., p["rf"]), axis=0)
        joint = (p["prior"] + p["log_w"] + log_f)[:, None] + np.log(p["mass"])[None]
        joint[:, ~compatible] = -np.inf
        log_likelihood = logsumexp(joint)
        post = np.exp(joint - log_likelihood)
        state_post = post.sum(axis=1)
        ew = (p["w"][:, :, None] - expit(p["qw"])) * state_post[None, None]
        ef = (f[:, None] - expit(p["rf"])) * state_post[None]
        d, k, boost = self.presence.d, self.presence.k, self.presence.boost_start
        grad = np.zeros_like(theta)
        grad[:d] = -np.einsum("jis,jd->d", ew, p["data"]["dc"][0])
        grad[d:2 * d] = -np.einsum("js,jd->d", ef, p["data"]["dw"][0])
        grad[2 * d:2 * d + k] = -p["x"] * (state_post @ p["u"].sum(axis=1) - len(sample.workers) * expit(p["lu"]))
        rv = state_post * (p["v"] - expit(p["lv"]))
        grad[2 * d + k:2 * d + 2 * k] = -p["x"] * rv.sum()
        grad[boost] = -np.sum(ew * p["u"].T[None])
        grad[boost + 1] = -np.sum(ef * p["v"][None])
        if self.presence.coupled:
            grad[self.n - 1] = -rv @ p["mean_u"]
        grad[self.n:self.n + 3] = -(post.sum(axis=0) @ p["score"])
        grad[self.n + 3:self.n + 5] = -np.einsum("js,je->e", ef, sample.release.visible_features)
        grad[self.n + 5] = -np.sum(ew * p["nonowner"][:, :, None])
        return float(-log_likelihood), grad

    def probability_graph(self, sample, *, at_least=1):
        """Export an executable BN, retaining every observed worker likelihood."""
        if type(at_least) is not int or at_least < 1:
            raise ValueError("positive integer violation count required")
        p = self._components(sample, self.theta)
        t, base, release, nodes = self.theta, self.presence, sample.release, []
        for fact in release.graph.facts.values():
            nodes.append({"id": "fact:" + fact.identity, "parents": [], "kind": "binary_cpt",
                "probability_one": [float(expit(t[self.n + fact.kind]))], "meaning": fact.provenance})
        uids = ["U:" + name for name in sample.workers]
        for uid in uids:
            nodes.append({"id": uid, "parents": [], "kind": "binary_cpt", "probability_one": [float(expit(p["lu"]))],
                "meaning": "Worker-specific latent disclosure tendency; shared coefficients, separate state"})
        vbase = p["x"] @ t[2 * base.d + base.k:2 * base.d + 2 * base.k]
        vprobs = [float(expit(vbase + np.mean(bits) * t[self.n - 1]))
                  for bits in product((0, 1), repeat=len(uids))] if base.coupled else [float(expit(vbase))]
        nodes.append({"id": "V", "parents": uids if base.coupled else [], "kind": "binary_cpt", "probability_one": vprobs,
            "aggregation": "Mean worker state with learned signed association; not causal identification"})
        targets = []
        for j, group in enumerate(release.packet["groups"]):
            cid, oid, aid, fid, hid = [f"{name}:{j}" for name in ("C", "O", "A", "F", "H")]
            proof = release.graph.proofs[release.action_ids[j]]
            paths = [["fact:" + f for f in path] for path in proof.paths]
            nodes.append({"id": aid, "parents": sorted({f for path in paths for f in path}),
                "kind": "all_paths_failed", "support_paths": paths, "binding_sha256": proof.binding_sha256})
            nodes.append({"id": cid, "parents": [], "kind": "binary_cpt", "probability_one": [float(group["coordinator_present"])],
                "observed": int(group["coordinator_present"])})
            wids = []
            for i, worker in enumerate(sample.workers):
                mid, wid = f"declared_owner:{j}:{worker}", f"W:{j}:{worker}"
                nodes.append({"id": mid, "parents": [], "kind": "binary_cpt", "probability_one": [float(sample.declared_owner_mask[j, i])],
                    "observed": int(sample.declared_owner_mask[j, i]), "meaning": "Host-declared assignment, not a permission or certified capability"})
                probs = []
                for u, c, owner in product((0, 1), repeat=3):
                    design = p["data"]["dc"][0, j].copy()
                    design[len(KINDS)] = c
                    probs.append(float(expit(design @ t[:base.d] + u * t[base.boost_start] + (1 - owner) * t[self.n + 5])))
                nodes.append({"id": wid, "parents": [uids[i], cid, mid], "kind": "binary_cpt", "probability_one": probs,
                    "observed": int(sample.worker_presence[j, i])})
                wids.append(wid)
            nodes.append({"id": oid, "parents": wids, "kind": "count_at_least", "threshold": 1,
                "meaning": "Actual occurrence in at least one worker message; does not replace branch evidence"})
            eids = [f"publisher_supported:{j}", f"publisher_contradicted:{j}"]
            for eid, value in zip(eids, release.visible_features[j]):
                nodes.append({"id": eid, "parents": [], "kind": "binary_cpt", "probability_one": [float(value)],
                    "observed": int(value), "meaning": release.visible_evidence})
            probs = []
            for v, w, support, contradict in product((0, 1), repeat=4):
                design = p["data"]["dw"][0, j].copy()
                design[len(KINDS)] = w
                probs.append(float(expit(design @ t[base.d:2 * base.d] + v * t[base.boost_start + 1]
                    + support * t[self.n + 3] + contradict * t[self.n + 4])))
            nodes.append({"id": fid, "parents": ["V", oid, *eids], "kind": "binary_cpt", "probability_one": probs,
                "field_names": list(group["field_names"])})
            nodes.append({"id": hid, "parents": [fid, aid], "kind": "and"})
            targets.append(hid)
        nodes.append({"id": "system_violation", "parents": targets, "kind": "count_at_least", "threshold": at_least})
        return {"schema": "executable-branched-privacy-graph-v1", "model": self.MODEL_NAME,
            "nodes": nodes, "output_id": "system_violation", "conditional_context": p["data"]["context"][0].tolist(),
            "scope": sample.specification(), "semantics": "Listed-parent big-endian CPTs; condition on observed messages, not interventions"}

    def parameters(self):
        return {**super().parameters(), "scope": "Fixed completed-worker DAG: separate conditionally independent U_i, shared coefficients, mean-U publisher association, actual branch emissions, declared ownership, dual authority views; not arbitrary workflows or calibrated safety"}
