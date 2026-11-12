"""Learnable observable-event DAG law, conditioned on actual prefix evidence.

No latent 'agent intent' is required. Unobserved intermediate variables denote
specific future observable events. Given prefix evidence, integrate them before
passing the FULL joint target distribution into the existing policy compiler.
Declared delivery is an architecture assumption, not causal discovery.
"""
from __future__ import annotations

import copy
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from paperalpha_runtime.canonical import digest
from paperalpha_runtime.fixed_workflow import topological_order
from .workflow_risk_bn import WorkflowRiskInput, WorkflowRiskBN, WorkflowFragmentRegistry

from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("prepare", "work", "review", "publish", "dispatch")
FEATURES = ("bias", "visible_support", "visible_opposition", "parent_mean", "multiple_parents_all")


class ObservableEventDAG:
    """Evidence-bound declared event topology. Does not infer roles or edges.

    Raw literal observations are checked against their actual text; 0 requires a
    complete observed message, not a missing log. Cues/delivery attestations are
    adapter-provided and must be independently validated by a runtime adapter.
    All times here use the transport prefix clock, not authority ledger time.
    """
    def __init__(self, *, workflow, events, edges, observations, prefix_sequence, missing_past=(), relational_context=None):
        self.workflow, self.events, self.edges, self.observations = map(copy.deepcopy, (workflow, events, edges, observations))
        self.prefix_sequence = prefix_sequence
        if type(prefix_sequence) is not int or prefix_sequence < 1:
            raise ValueError("explicit transport prefix required")
        order = topological_order(workflow)
        if set(workflow) != {"nodes", "edges", "actors"} or set(workflow["actors"]) != set(order):
            raise ValueError("functional DAG with explicit actor ownership required")
        for actors in workflow["actors"].values():
            if not isinstance(actors, list) or not actors or len(set(actors)) != len(actors) or any(not isinstance(a, str) or not a for a in actors):
                raise ValueError("unique nonempty actor ownership list required")
        if not isinstance(events, list) or not 1 <= len(events) <= 256:
            raise ValueError("one to 256 explicit events required; inference has separate capacity checks")
        ids = []
        for event in events:
            if set(event) != {"id", "node", "actor", "family", "semantics", "subject", "cues", "cue_evidence"}:
                raise ValueError("strict typed event fields required")
            if any(not isinstance(event[k], str) or not event[k] for k in ("id", "node", "actor", "subject")):
                raise ValueError("nonempty event identity and subject required")
            if event["id"] in ids or event["node"] not in order or event["actor"] not in workflow["actors"][event["node"]]:
                raise ValueError("unique event and functional actor binding required")
            if event["family"] not in FAMILIES or event["semantics"] not in {"literal_appearance", "tool_dispatch"}:
                raise ValueError("known probability family and observable event type required")
            if not isinstance(event["cues"], list) or len(event["cues"]) != 2 or any(type(x) is not int or x not in (0, 1) for x in event["cues"]) or sum(event["cues"]) > 1:
                raise ValueError("support/opposition/unknown must be explicit prefix cues")
            self._evidence(event["cue_evidence"])
            ids.append(event["id"])
        self.ids = tuple(ids)
        self.lookup = {e["id"]: e for e in self.events}
        if not isinstance(observations, dict) or not set(observations) <= set(ids):
            raise ValueError("only named prefix events may be observed")
        if not isinstance(missing_past, (list, tuple)) or len(set(missing_past)) != len(missing_past) or not set(missing_past) <= set(ids) - set(observations):
            raise ValueError("missing past content must be distinct known unobserved events")
        self.missing_past = tuple(missing_past)
        past = set(observations) | set(missing_past)
        for identity, observation in observations.items():
            if set(observation) != {"value", "record_id", "sequence", "source", "text", "kind"}:
                raise ValueError("observation must retain source text and event type")
            self._evidence({k: observation[k] for k in ("record_id", "sequence", "source", "text")})
            if type(observation["value"]) is not int or observation["value"] not in (0, 1):
                raise ValueError("explicit binary observation required")
            event = self.lookup[identity]
            if event["semantics"] == "literal_appearance":
                if observation["kind"] != "complete_message" or observation["value"] != int(event["subject"].casefold() in observation["text"].casefold()):
                    raise ValueError("literal observation must match a completed source message")
            elif observation["kind"] not in {"dispatch", "cancelled"} or observation["value"] != int(observation["kind"] == "dispatch"):
                raise ValueError("tool evidence must explicitly be dispatch or cancellation, not text mention")
        seen = set()
        parents = {identity: [] for identity in ids}
        for edge in edges:
            if set(edge) != {"source", "target", "basis", "evidence"}:
                raise ValueError("delivery links require explicit source, target and evidence")
            u, v = edge["source"], edge["target"]
            if u not in self.lookup or v not in self.lookup or (u, v) in seen or u == v:
                raise ValueError("unique known non-self event link required")
            a, b = self.lookup[u], self.lookup[v]
            if [a["node"], b["node"]] not in workflow["edges"]:
                raise ValueError("event delivery requires a declared direct functional edge")
            if b["semantics"] == "literal_appearance" and (a["semantics"] != "literal_appearance" or a["subject"].casefold() != b["subject"].casefold()):
                raise ValueError("literal transfer links must track the same literal")
            if edge["basis"] not in {"observed_delivery", "declared_delivery"}:
                raise ValueError("explicit observed or planned delivery basis required")
            self._evidence(edge["evidence"])
            if edge["basis"] == "observed_delivery" and u not in past:
                raise ValueError("future source output cannot be an observed delivery")
            if v in past and u not in past:
                raise ValueError("a past event cannot depend on a future event; declare missing past content explicitly")
            if u in observations and v in observations and observations[u]["sequence"] > observations[v]["sequence"]:
                raise ValueError("message propagation cannot reverse observed time")
            parents[v].append(u)
            seen.add((u, v))
        self.order = tuple(sorted(ids, key=lambda identity: (order.index(self.lookup[identity]["node"]), ids.index(identity))))
        self.parents = {k: tuple(v) for k, v in parents.items()}
        self.relational_context = copy.deepcopy(relational_context)
        self.relational_features = None
        if self.relational_context is not None:
            from .relational_event_law import validate_relational_context
            self.relational_features = validate_relational_context(self, self.relational_context)
        self.sha256 = digest(self.specification())

    def _evidence(self, value):
        if not isinstance(value, dict) or set(value) != {"record_id", "sequence", "source", "text"}:
            raise ValueError("bound prefix evidence required")
        if type(value["sequence"]) is not int or not 0 <= value["sequence"] <= self.prefix_sequence:
            raise ValueError("future evidence is forbidden")
        if any(not isinstance(value[k], str) or not value[k] for k in ("record_id", "source", "text")):
            raise ValueError("evidence source and nonempty text required")

    def specification(self):
        result = {"workflow": copy.deepcopy(self.workflow), "events": copy.deepcopy(self.events), "edges": copy.deepcopy(self.edges),
                  "observations": copy.deepcopy(self.observations), "prefix_sequence": self.prefix_sequence}
        # Preserve the just-frozen ancestor-complete development inputs exactly.
        if self.missing_past:
            result["missing_past"] = list(self.missing_past)
        if self.relational_context is not None:
            result["relational_context"] = copy.deepcopy(self.relational_context)
        return result

    def validate(self):
        if digest(self.specification()) != self.sha256:
            raise ValueError("event DAG evidence changed")
        rebuilt = ObservableEventDAG(**self.specification())
        if self.ids != rebuilt.ids or self.order != rebuilt.order or self.parents != rebuilt.parents or self.lookup != rebuilt.lookup or self.missing_past != rebuilt.missing_past or self.relational_features != rebuilt.relational_features:
            raise ValueError("derived event topology changed")


class ObservableEventLaw:
    def __init__(self, specification=None):
        self.specification = json.loads(static_resource('knowledge/templates/observable_event_dag_v1.json').read_text(encoding="utf-8")) if specification is None else copy.deepcopy(specification)
        s = self.specification
        if s.get("schema") != "observable-event-dag-v1" or s.get("families") != list(FAMILIES) or s.get("features") != list(FEATURES):
            raise ValueError("supported shared probability-template definition required")
        catalog = json.loads(static_resource('knowledge/risks/risk_catalog.json').read_text(encoding="utf-8"))
        risks = {r["id"]: r for r in catalog["entries"]}
        knowledge = s["knowledge"]
        if not knowledge["risk_ids"] or not set(knowledge["risk_ids"]) <= set(risks) or not knowledge["source_ids"] or not set(knowledge["source_ids"]) <= {i for r in knowledge["risk_ids"] for i in risks[r]["source_ids"]}:
            raise ValueError("event template knowledge must resolve in the risk catalogue")
        self.names = tuple(f"event:{family}:{feature}" for family in FAMILIES for feature in FEATURES)
        self.sha256 = digest(s)

    def design(self, event, parents):
        x = np.zeros(len(self.names))
        start = FAMILIES.index(event["family"]) * len(FEATURES)
        x[start:start + 5] = [1., *event["cues"], float(np.mean(parents)) if len(parents) else 0., float(len(parents) > 1 and all(parents))]
        return x

    def node_design(self, dag, identity, parents):
        """Context-aware hook; existing laws keep their original public design."""
        return self.design(dag.lookup[identity], parents)

    def conditional(self, dag, targets, theta, *, max_free=16):
        dag.validate()
        if digest(self.specification) != self.sha256:
            raise ValueError("event probability templates changed")
        if len(set(targets)) != len(targets) or not targets or not set(targets) <= set(dag.ids) - set(dag.observations) - set(dag.missing_past):
            raise ValueError("distinct unobserved future targets required")
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.names),) or not np.isfinite(theta).all():
            raise ValueError("finite compatible event coefficients required")
        free = [identity for identity in dag.order if identity not in dag.observations]
        if type(max_free) is not int or not 1 <= max_free <= 18 or len(free) > max_free:
            raise ValueError("exact future-event capacity exceeded; no independence fallback")
        free_states = np.asarray(list(product((0., 1.), repeat=len(free))))
        values = {identity: free_states[:, free.index(identity)] if identity in free else np.full(len(free_states), dag.observations[identity]["value"], float) for identity in dag.order}
        logs, scores = np.zeros(len(free_states)), np.zeros((len(free_states), len(theta)))
        for identity in dag.order:
            event = dag.lookup[identity]
            parents = np.asarray([values[p] for p in dag.parents[identity]]).T if dag.parents[identity] else np.empty((len(free_states), 0))
            design = np.asarray([self.node_design(dag, identity, row) for row in parents])
            logits, y = design @ theta, values[identity]
            logs += y * logits - np.logaddexp(0., logits)
            scores += (y - expit(logits))[:, None] * design
        log_evidence = float(logsumexp(logs))
        weights = np.exp(logs - log_evidence)
        evidence_score = weights @ scores
        states = np.asarray(list(product((0., 1.), repeat=len(targets))))
        indices = np.column_stack([values[t] for t in targets]) @ (2 ** np.arange(len(targets) - 1, -1, -1))
        joint_logs, joint_scores = [], []
        for index in range(len(states)):
            selected = indices == index
            z = logsumexp(logs[selected])
            w = np.exp(logs[selected] - z)
            joint_logs.append(z - log_evidence)
            joint_scores.append(w @ scores[selected] - evidence_score)
        return {"states": states, "log_probabilities": np.asarray(joint_logs), "scores": np.asarray(joint_scores),
                "log_evidence": log_evidence, "free_events": len(free), "worlds_enumerated": len(free_states)}

    def graph_nodes(self, dag, theta, mapping):
        nodes = []
        for identity in dag.order:
            event, parents = dag.lookup[identity], dag.parents[identity]
            if len(parents) > 18:
                raise ValueError("unreduced conditional-table export exceeds its parent capacity")
            probs = [float(expit(self.node_design(dag, identity, row) @ theta)) for row in product((0, 1), repeat=len(parents))]
            node = {"id": mapping[identity], "parents": [mapping[p] for p in parents], "kind": "binary_cpt", "probability_one": probs,
                "workflow_node": event["node"], "event_semantics": event["semantics"], "subject": event["subject"],
                "probability_family": event["family"], "knowledge": copy.deepcopy(self.specification["knowledge"])}
            if identity in dag.observations:
                node["observed"] = dag.observations[identity]["value"]
                node["evidence"] = copy.deepcopy(dag.observations[identity])
            nodes.append(node)
        return nodes


class DAGWorkflowInput(WorkflowRiskInput):
    def __init__(self, *, event_dag, queries, **policy):
        if not isinstance(event_dag, ObservableEventDAG):
            raise ValueError("externally observable event DAG required")
        self.event_dag, self.queries = copy.deepcopy(event_dag), copy.deepcopy(queries)
        super().__init__(**policy)
        self._check_link()

    def _check_link(self):
        self.event_dag.validate()
        dag = self.event_dag
        if dag.workflow != self.workflow or set(self.queries) != set(self.action_ids):
            raise ValueError("event and policy workflows and complete target bindings must agree")
        used = set()
        for proposal in self.proposals:
            aid, q = proposal["action_id"], self.queries[proposal["action_id"]]
            if set(q) != {"event_id", "binding_sha256", "field_literals"} or q["event_id"] in used:
                raise ValueError("one explicit observable variable per policy event required")
            event = dag.lookup.get(q["event_id"])
            if event is None or q["event_id"] in dag.observations or q["event_id"] in dag.missing_past or event["node"] != proposal["node"] or event["actor"] != proposal["binding"]["actor"] or event["semantics"] != self.event_semantics[aid] or q["binding_sha256"] != digest(proposal["binding"]):
                raise ValueError("future event must match the exact policy action binding")
            if event["semantics"] == "literal_appearance":
                fields = proposal["binding"]["arguments"]["field_names"]
                if not isinstance(q["field_literals"], dict) or set(q["field_literals"]) != set(fields) or set(q["field_literals"].values()) != {event["subject"]}:
                    raise ValueError("literal identity and release aliases must agree")
                expected = {"event_dag_sha256": dag.sha256, "event_id": event["id"], "available_journal_sequence": dag.prefix_sequence, "prefix_journal_sequence": dag.prefix_sequence}
                if proposal["evidence"] != expected:
                    raise ValueError("literal query must reference its actual event DAG and transport clock")
            elif q["field_literals"] is not None or event["subject"] != digest(proposal["binding"]):
                raise ValueError("tool event requires exact action identity, not a textual amount")
            used.add(q["event_id"])

    def specification(self):
        return {**super().specification(), "event_dag": self.event_dag.specification(), "queries": copy.deepcopy(self.queries)}

    def validate(self):
        super().validate()
        self._check_link()


class DAGWorkflowRiskBN(WorkflowRiskBN):
    MODEL = "ObservableDAGWorkflowRiskBN-v1"

    def __init__(self, *, event_law=None, **options):
        super().__init__(**options)
        self.event_law = copy.deepcopy(event_law or ObservableEventLaw())
        self.event_count = len(self.event_law.names)
        self.names = self.event_law.names + self.registry.names[-3:]
        self.theta = np.zeros(len(self.names))

    def _occurrence_worlds(self, sample, theta):
        if not isinstance(sample, DAGWorkflowInput):
            raise ValueError("linked observable-event and knowledge graph required")
        free = len(sample.event_dag.ids) - len(sample.event_dag.observations)
        if free + len(sample.authority.facts) > self.max_binary_variables:
            raise ValueError("all unobserved intermediate events and authority facts count toward exact capacity")
        law = self.event_law.conditional(sample.event_dag, [sample.queries[a]["event_id"] for a in sample.action_ids], theta[:self.event_count], max_free=self.max_binary_variables)
        scores = np.zeros((len(law["states"]), len(self.names)))
        scores[:, :self.event_count] = law["scores"]
        return law["states"], law["log_probabilities"], scores

    def _event_node_id(self, action_id):
        return "event:" + action_id

    def _occurrence_graph_nodes(self, sample, probabilities):
        mapping = {q["event_id"]: self._event_node_id(a) for a, q in sample.queries.items()}
        for identity in sample.event_dag.ids:
            mapping.setdefault(identity, "trace:" + identity)
        return self.event_law.graph_nodes(sample.event_dag, self.theta[:self.event_count], mapping)

    def predict(self, sample, *, theta=None, derivatives=False, event_conditions=None):
        result = super().predict(sample, theta=theta, derivatives=derivatives, event_conditions=event_conditions)
        probabilities = result.pop("dispatch_probabilities")
        result["event_probabilities"] = probabilities
        result["appearance_probabilities"] = {a: p for a, p in probabilities.items() if sample.event_semantics[a] == "literal_appearance"}
        result["dispatch_probabilities"] = {a: p for a, p in probabilities.items() if sample.event_semantics[a] == "tool_dispatch"}
        result["event_dag_sha256"] = sample.event_dag.sha256
        result["enumerated_future_event_worlds"] = 2 ** (len(sample.event_dag.ids) - len(sample.event_dag.observations) - len(sample.event_dag.missing_past))
        if sample.event_dag.missing_past:
            result["enumerated_unobserved_event_worlds"] = 2 ** (len(sample.event_dag.ids) - len(sample.event_dag.observations))
            result["missing_past_event_ids"] = list(sample.event_dag.missing_past)
        result["target"] = "Configured joint policy risk over bound future literal appearances and once-dispatched tool events, conditioned on actual prefix messages"
        result["learned_parameters"] = self.optimizer is not None
        result["limitations"] = ["Supplied acyclic delivery topology and externally bound cues, not automatic role/DAG extraction or causal discovery.",
            "Up to 12 functional nodes; bounded exact future-event enumeration, not large-system inference.",
            "Literal tracking is not semantic leakage or instruction adoption. Missing completed messages are not observed absences.",
            "Past missing content must be declared separately from future events; neither is an observed absence. No future-dependent evidence or dynamic authority window.",
            "Families share logistic coefficients; unobserved events are conditionally independent given listed parents and prefix cues. Unlisted common causes remain possible.",
            "No claim of real-data calibration or improvements solely from changing graph size."]
        return result

    def parameters(self):
        result = {**super().parameters(), "event_law": copy.deepcopy(self.event_law.specification), "event_law_sha256": self.event_law.sha256,
                "training_target": "Conditional target-event/fact/policy likelihood given actual prefix events; unobserved intermediate events marginalized"}
        if hasattr(self.event_law, "optimizer_summary"):
            result["optimizer"] = self.event_law.optimizer_summary(self.optimizer)
        return result

    @classmethod
    def restore(cls, saved):
        if saved["model"] != cls.MODEL:
            raise ValueError("wrong event DAG model")
        registry = WorkflowFragmentRegistry(saved["registry"])
        law = restore_event_law(saved["event_law"])
        model = cls(event_law=law, registry=registry, regularization=saved["regularization"], max_binary_variables=saved["max_binary_variables"])
        if registry.sha256 != saved["registry_sha256"] or law.sha256 != saved["event_law_sha256"] or list(model.names) != saved["names"]:
            raise ValueError("model/template identity mismatch")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("compatible finite model required")
        model.optimizer = copy.deepcopy(saved["optimizer"])
        return model


def restore_event_law(specification, *, max_factor_variables=None):
    """Explicit schema dispatch, never a pickled model or a case registry."""
    if specification.get("schema") == "relational-observable-event-law":
        from .relational_event_law import RelationalEventLaw
        return RelationalEventLaw(specification, max_factor_variables=16 if max_factor_variables is None else max_factor_variables)
    if max_factor_variables is not None:
        from .event_elimination import EliminatingEventLaw
        return EliminatingEventLaw(specification, max_factor_variables=max_factor_variables)
    return ObservableEventLaw(specification)
