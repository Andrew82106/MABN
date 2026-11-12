"""Connect the existing branch appearance BN to typed knowledge-policy graphs.

Appearance events keep their FULL conditional joint distribution. Optional
pending tool dispatches are a separate event type with a separate independent
predictor; textual numbers never become monetary effects. One authority graph
and one parameter slice supply all current-policy facts. No new causal claim.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np
from scipy.special import expit

from paperalpha_runtime.canonical import digest
from .branch_privacy_input import BranchPrivacyInput
from .branch_publisher_bn import BranchPublisherPrivacyBN
from .workflow_risk_bn import WorkflowRiskInput, WorkflowRiskBN, WorkflowFragmentRegistry


def branch_reference(branch):
    return {"branch_observation_sha256": digest(branch.observation),
            "available_journal_sequence": branch.available_journal_sequence,
            "prefix_journal_sequence": branch.prefix_journal_sequence}


class BranchWorkflowInput(WorkflowRiskInput):
    def __init__(self, *, branch, **policy):
        if not isinstance(branch, BranchPrivacyInput):
            raise ValueError("existing raw branch observation contract required")
        self.branch = copy.deepcopy(branch)
        self.branch.validate()
        super().__init__(**policy)
        self._check_link()

    @classmethod
    def from_branch(cls, branch, *, combinations=(), horizon=None):
        branch.validate()
        release = branch.release
        original = branch.observation["workflow"]
        workflow = {"nodes": copy.deepcopy(original["nodes"]), "edges": copy.deepcopy(original["edges"]),
                    "actors": {node: [node] for node in original["nodes"]}}
        proposals = [{"action_id": aid, "node": "publisher", "binding": copy.deepcopy(release.graph.proofs[aid].evidence["action_binding"]),
            "family": "release", "visible": None, "evidence": branch_reference(branch), "quantity": None}
            for aid in release.action_ids]
        return cls(branch=branch, workflow=workflow, authority=release.graph, proposals=proposals, combinations=combinations,
                   cutoff=release.context.observed_through_sequence,
                   horizon=release.graph.window.last_sequence if horizon is None else horizon,
                   event_semantics={aid: "literal_appearance" for aid in release.action_ids})

    def _check_link(self):
        self.branch.validate()
        b = self.branch
        literal_ids = {aid for aid, kind in self.event_semantics.items() if kind == "literal_appearance"}
        if literal_ids != set(b.release.action_ids):
            raise ValueError("each branch field must connect to exactly one literal event")
        if self.authority.window != b.release.graph.window or self.cutoff < b.release.context.observed_through_sequence:
            raise ValueError("branch and policy authority windows/cutoffs must agree")
        actual = b.observation["workflow"]
        if not set(actual["nodes"]) <= set(self.workflow["nodes"]) or not {tuple(e) for e in actual["edges"]} <= {tuple(e) for e in self.workflow["edges"]}:
            raise ValueError("this appearance provider requires its actual fixed branch subworkflow")
        for name in actual["nodes"]:
            if name not in self.workflow["actors"][name]:
                raise ValueError("actual branch actor must own its declared stage")
        for p in self.proposals:
            aid = p["action_id"]
            if aid in literal_ids:
                if p["node"] != "publisher" or p["evidence"] != branch_reference(b):
                    raise ValueError("literal event must bind the actual publisher prefix without clock conversion")
                if self.authority.proofs[aid] != b.release.graph.proofs[aid]:
                    raise ValueError("appearance and policy components must use identical authority evidence")
        for identity, fact in b.release.graph.facts.items():
            if self.authority.facts.get(identity) != fact:
                raise ValueError("shared authority identity cannot change at the interface")

    def specification(self):
        return {**super().specification(), "appearance_provider": "BranchPublisherPrivacyBN-v1",
                "branch_reference": branch_reference(self.branch)}

    def validate(self):
        super().validate()
        self._check_link()


class BranchWorkflowRiskBN(WorkflowRiskBN):
    MODEL = "BranchWorkflowRiskBN-v1"

    def __init__(self, *, branch_model=None, **options):
        super().__init__(**options)
        self.branch_model = copy.deepcopy(branch_model or BranchPublisherPrivacyBN())
        if type(self.branch_model) is not BranchPublisherPrivacyBN:
            raise ValueError("the verified branch appearance implementation is required")
        self.branch_count = len(self.branch_model.theta)
        self.names = tuple("branch:" + name for name in self.branch_model.names) + tuple(self.registry.names[:9])
        self.theta = np.r_[self.branch_model.theta, np.zeros(9)]

    def _authority_indices(self):
        return np.arange(self.branch_model.n, self.branch_model.n + 3)

    def _parameter_bounds(self):
        bounds = super()._parameter_bounds()
        start = self.branch_model.presence.boost_start
        bounds[start:start + 2] = [(0., 12.), (0., 12.)]
        return bounds

    def _design(self, sample):
        # Only actual tool events receive the dispatch predictor. Appearance
        # fields use their original multi-worker model, with no placeholder cues.
        x = np.zeros((len(sample.proposals), len(self.names)))
        for i, p in enumerate(sample.proposals):
            if sample.event_semantics[p["action_id"]] != "tool_dispatch":
                continue
            start = self.branch_count + self.registry.specification["dispatch_families"].index(p["family"]) * 3
            x[i, start:start + 3] = (1., p["visible"] == "support", p["visible"] == "opposition")
        return x

    def _occurrence_worlds(self, sample, theta):
        if not isinstance(sample, BranchWorkflowInput):
            raise ValueError("typed linked branch/policy input required")
        sample._check_link()
        # Include marginalized latent states in the capacity contract; the
        # output table alone does not describe the producer's full exact cost.
        if len(sample.proposals) + len(sample.authority.facts) + len(sample.branch.workers) + 1 > self.max_binary_variables:
            raise ValueError("joint appearance/policy latent capacity exceeded")
        law = self.branch_model.joint_appearance_law(sample.branch, theta=theta[:self.branch_count])
        states = np.asarray(list(product((0., 1.), repeat=len(sample.proposals))))
        positions = [sample.action_ids.index(aid) for aid in law["action_ids"]]
        indices = (states[:, positions] @ (2 ** np.arange(len(positions) - 1, -1, -1))).astype(int)
        log_p = law["log_probabilities"][indices].copy()
        score = np.zeros((len(states), len(self.names)))
        score[:, :self.branch_count] = law["log_probability_gradients"][indices]
        tool = [i for i, aid in enumerate(sample.action_ids) if sample.event_semantics[aid] == "tool_dispatch"]
        if tool:
            x = self._design(sample)[tool]
            logits = x @ theta
            log_p += states[:, tool] @ logits - np.logaddexp(0., logits).sum()
            score += (states[:, tool] - expit(logits)) @ x
        return states, log_p, score

    def _event_node_id(self, action_id):
        return "event:" + action_id

    def _occurrence_graph_nodes(self, sample, probabilities):
        model = copy.deepcopy(self.branch_model)
        model.theta = self.theta[:self.branch_count].copy()
        graph = model.probability_graph(sample.branch)
        mapping = {f"F:{i}": self._event_node_id(aid) for i, aid in enumerate(sample.branch.release.action_ids)}
        retained = [n for n in graph["nodes"] if not n["id"].startswith(("fact:", "A:", "H:")) and n["id"] != "system_violation"]
        for node in retained:
            mapping.setdefault(node["id"], "appearance:" + node["id"])
        nodes = []
        for node in retained:
            item = copy.deepcopy(node)
            item["id"] = mapping[node["id"]]
            item["parents"] = [mapping[p] for p in node["parents"]]
            if item["id"].startswith("event:"):
                item["event_semantics"] = "literal_appearance"
            nodes.append(item)
        for i, proposal in enumerate(sample.proposals):
            aid = proposal["action_id"]
            if sample.event_semantics[aid] == "tool_dispatch":
                nodes.append({"id": self._event_node_id(aid), "parents": [], "kind": "binary_cpt",
                    "probability_one": [float(probabilities["dispatch"][i])], "event_semantics": "tool_dispatch", "workflow_node": proposal["node"]})
        return nodes

    def predict(self, sample, *, theta=None, derivatives=False, event_conditions=None):
        result = super().predict(sample, theta=theta, derivatives=derivatives, event_conditions=event_conditions)
        all_events = result.pop("dispatch_probabilities")
        result["event_probabilities"] = all_events
        result["dispatch_probabilities"] = {k: v for k, v in all_events.items() if sample.event_semantics[k] == "tool_dispatch"}
        result["appearance_probabilities"] = {k: v for k, v in all_events.items() if sample.event_semantics[k] == "literal_appearance"}
        result["event_semantics"] = copy.deepcopy(sample.event_semantics)
        result["branch_reference"] = branch_reference(sample.branch)
        result["target"] = "Configured joint policy events over exact next-reply literal appearances and optional once-dispatched tool actions within a fixed authority window"
        result["limitations"] = [s for s in result["limitations"] if not s.startswith("Dispatches conditionally")]
        result["limitations"] += ["Branch appearance uses the existing fixed two-worker model, not arbitrary-DAG information propagation.",
            "Appearance law is joint; optional tool dispatches are conditionally independent of it and of each other.",
            "Appearance provider still concerns exact tracked literals, not semantic paraphrases or physical tool effects.",
            "Conditional outcome training differs from the original joint worker/final-message objective; reuse is not proof of recalibration."]
        for fragment in result["compiled_fragments"]:
            fragment["event_semantics"] = {a: sample.event_semantics[a] for a in fragment["actions"]}
        return result

    def parameters(self):
        result = super().parameters()
        branch = copy.deepcopy(self.branch_model)
        branch.theta = self.theta[:self.branch_count].copy()
        # The composite's optimizer is the authoritative fit metadata. Do not
        # inherit a historical component optimizer as if the new graph was fit.
        branch.optimizer = None
        result["branch_definition"] = branch.parameters()
        result["training_target"] = "Conditional event-vector/fact/policy-label likelihood given observed worker messages"
        return result

    @classmethod
    def restore(cls, saved):
        if saved["model"] != cls.MODEL:
            raise ValueError("wrong composite model")
        registry = WorkflowFragmentRegistry(saved["registry"])
        branch = BranchPublisherPrivacyBN.restore(saved["branch_definition"])
        model = cls(branch_model=branch, registry=registry, regularization=saved["regularization"], max_binary_variables=saved["max_binary_variables"])
        if registry.sha256 != saved["registry_sha256"] or list(model.names) != saved["names"]:
            raise ValueError("composite parameter/knowledge identity mismatch")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite compatible composite parameters required")
        if not np.array_equal(model.theta[:model.branch_count], branch.theta):
            raise ValueError("branch definition and top-level parameters differ")
        model.optimizer = copy.deepcopy(saved["optimizer"])
        return model
