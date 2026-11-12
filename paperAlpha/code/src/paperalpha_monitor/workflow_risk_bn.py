"""Knowledge-selected action fragments sharing dispatch and authority variables.

This complements, not replaces, the existing branch disclosure model. Workflow
nodes own fragments; shared policy scopes/facts create actual cross-node risk
edges. A normative workflow edge alone does NOT assert stochastic dependence.
Dispatches are conditionally independent given their prefix cues, independent
of hidden authority facts. These are explicit unvalidated forecasting choices.
"""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from paperalpha_runtime.canonical import digest
from paperalpha_runtime.fixed_workflow import topological_order
from .privacy_supervision_controls import _solve
from .scoped_quantities import decimal_amount, currency_code
from .shared_authority_graph import AuthorityProofGraph, PRIOR_NAMES

from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]
SPEC = static_resource('knowledge/templates/workflow_risk_fragments_v1.json')
SUPPORTED = {
    "authorization": ("every_proposal", "dispatch_and_failed_reference"),
    "single_bound": ("payment_with_declared_bound", "dispatch_and_amount_exceeds"),
    "cumulative_budget": ("matching_budget_contract", "requested_charge_sum_exceeds"),
    "prohibited_combination": ("declared_bound_action_set", "all_selected_dispatches"),
}


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty identity/source required")
    return value


def _sequence(value, cutoff):
    if type(value) is not int or not 0 <= value <= cutoff:
        raise ValueError("only records at or before the prediction cutoff are permitted")


class WorkflowFragmentRegistry:
    def __init__(self, specification=None, catalog=None):
        spec = json.loads(SPEC.read_text(encoding="utf-8")) if specification is None else copy.deepcopy(specification)
        catalog = json.loads(static_resource('knowledge/risks/risk_catalog.json').read_text(encoding="utf-8")) if catalog is None else catalog
        if spec.get("schema") != "workflow-risk-fragments-v1" or spec.get("dispatch_features") != ["bias", "visible_support", "visible_opposition"]:
            raise ValueError("supported typed probability specification required")
        if spec.get("dispatch_families") != ["payment", "release", "other"]:
            raise ValueError("supported dispatch families required")
        risks = {e["id"]: e for e in catalog["entries"]}
        sources = {s["id"] for s in catalog["sources"]}
        self.templates = {}
        for item in spec.get("fragments", []):
            name = item.get("id")
            if name in self.templates or name not in SUPPORTED or (item.get("selector"), item.get("operator")) != SUPPORTED[name]:
                raise ValueError("unique supported fragment selector/operator required")
            rids, sids = item.get("risk_ids", []), item.get("source_ids", [])
            if not rids or not sids or not set(rids) <= set(risks) or not set(sids) <= sources:
                raise ValueError("fragment must cite existing risk/source knowledge")
            if not set(sids) <= {s for r in rids for s in risks[r]["source_ids"]}:
                raise ValueError("fragment source not linked to its cited risk")
            _text(item.get("meaning"))
            self.templates[name] = copy.deepcopy(item)
        if not self.templates:
            raise ValueError("at least one declared mechanism family required")
        self.specification, self.sha256 = spec, digest(spec)
        self.names = tuple(f"dispatch:{family}:{feature}" for family in spec["dispatch_families"]
                           for feature in spec["dispatch_features"]) + PRIOR_NAMES

    def validate(self):
        expected = {f["id"]: f for f in self.specification["fragments"]}
        if digest(self.specification) != self.sha256 or self.templates != expected:
            raise ValueError("knowledge configuration changed after compilation")


class WorkflowRiskInput:
    """Strict adapter contract; external declarations are NOT authenticated here.

    `authority` is independently rebuilt by the existing delegation verifier.
    All proposals share a task/principal and have not dispatched at cutoff.
    The candidate set and authority must remain fixed until horizon; additional
    actions, retries, grant changes or missing history invalidate this query.
    """
    def __init__(self, *, workflow, authority, proposals, contracts=(), combinations=(), cutoff, horizon, event_semantics=None):
        if type(cutoff) is not int or type(horizon) is not int or not 0 <= cutoff < horizon:
            raise ValueError("explicit increasing forecast window required")
        if not isinstance(authority, AuthorityProofGraph):
            raise ValueError("validated existing authority proof graph required")
        self.workflow = copy.deepcopy(workflow)
        if set(workflow) != {"nodes", "edges", "actors"}:
            raise ValueError("declared workflow nodes, edges and actor ownership required")
        topological_order(workflow)
        actors = workflow["actors"]
        if not isinstance(actors, dict) or set(actors) != set(workflow["nodes"]):
            raise ValueError("one explicit actor list per functional node required")
        for names in actors.values():
            if not isinstance(names, list) or not names or len(names) != len(set(names)):
                raise ValueError("unique nonempty actor ownership lists required")
            for name in names:
                _text(name)
        self.authority = copy.deepcopy(authority)
        self.authority.validate()
        if not authority.window.first_sequence <= cutoff < horizon <= authority.window.last_sequence:
            raise ValueError("forecast must fit the fixed authority window")
        self.proposals, self.contracts, self.combinations = map(copy.deepcopy, (list(proposals), list(contracts), list(combinations)))
        self.cutoff, self.horizon = cutoff, horizon
        if not self.proposals:
            raise ValueError("nonempty explicit pending proposal set required")
        self.event_semantics = ({p["action_id"]: "tool_dispatch" for p in self.proposals}
                                if event_semantics is None else copy.deepcopy(event_semantics))
        if (not isinstance(self.event_semantics, dict) or set(self.event_semantics) != {p["action_id"] for p in self.proposals}
                or not set(self.event_semantics.values()) <= {"tool_dispatch", "literal_appearance"}):
            raise ValueError("one explicit supported event semantics per candidate required")
        ids, scope = [], set()
        for p in self.proposals:
            if set(p) != {"action_id", "node", "binding", "family", "visible", "evidence", "quantity"}:
                raise ValueError("strict prefix proposal fields required; no outcomes or labels")
            aid, node = _text(p["action_id"]), p["node"]
            proof = authority.proofs.get(aid)
            if aid in ids or proof is None or p["family"] not in {"payment", "release", "other"}:
                raise ValueError("unique action, existing proof and known family required")
            if node not in actors or p["binding"].get("actor") not in actors[node]:
                raise ValueError("proposal actor must belong to its functional node")
            if proof.binding_sha256 != digest(p["binding"]) or proof.evidence["action_binding"] != p["binding"]:
                raise ValueError("proposal must use the exact assessed action binding")
            _sequence(proof.sequence, cutoff)
            scope.add((proof.evidence["task_id"], p["binding"]["principal"]))
            e = p["evidence"]
            if self.event_semantics[aid] == "literal_appearance":
                if p["family"] != "release" or p["binding"]["tool"] != "release_literal" or p["visible"] is not None:
                    raise ValueError("literal event requires the bound appearance provider, not dispatch cues")
                branch_fields = {"branch_observation_sha256", "available_journal_sequence", "prefix_journal_sequence"}
                dag_fields = {"event_dag_sha256", "event_id", "available_journal_sequence", "prefix_journal_sequence"}
                if not isinstance(e, dict) or set(e) not in (branch_fields, dag_fields):
                    raise ValueError("literal event needs a separately clocked bound appearance-provider reference")
                if set(e) == branch_fields:
                    _text(e["branch_observation_sha256"])
                else:
                    _text(e["event_dag_sha256"])
                    _text(e["event_id"])
                if type(e["prefix_journal_sequence"]) is not int or type(e["available_journal_sequence"]) is not int or not 0 < e["available_journal_sequence"] <= e["prefix_journal_sequence"]:
                    raise ValueError("branch observation must be available in its transport prefix")
            else:
                if p["visible"] not in {"support", "opposition", "unknown"}:
                    raise ValueError("explicit visible cue category required")
                if not isinstance(e, dict) or set(e) != {"record_id", "sequence", "source", "text"}:
                    raise ValueError("bound prefix cue evidence required")
                _sequence(e["sequence"], cutoff)
                for field in ("record_id", "source", "text"):
                    _text(e[field])
            q = p["quantity"]
            if p["family"] == "payment":
                if not isinstance(q, dict) or set(q) != {"amount_field", "currency_field", "single_limit", "limit_source", "limit_sequence"}:
                    raise ValueError("payment needs a public quantity profile and explicit limit coverage")
                decimal_amount(p["binding"]["arguments"].get(q["amount_field"]))
                if currency_code(p["binding"]["arguments"].get(q["currency_field"])) is None:
                    raise ValueError("known unambiguous payment currency required")
                if q["single_limit"] is not None:
                    decimal_amount(q["single_limit"])
                    _text(q["limit_source"])
                    _sequence(q["limit_sequence"], cutoff)
                elif q["limit_source"] is not None or q["limit_sequence"] is not None:
                    raise ValueError("missing limit is unknown, not a sourced zero limit")
            elif q is not None:
                raise ValueError("nonpayment cannot instantiate a quantity fragment")
            ids.append(aid)
        if len(scope) != 1 or set(ids) != set(authority.proofs):
            raise ValueError("one task/principal and exact proposal/proof coverage required")
        # Every future candidate must be judged against the same current record
        # prefix. A mixture of stale and fresh action snapshots is not a forecast.
        prefixes = {digest({k: proof.evidence[k] for k in ("source_records", "trusted_observers", "root_principals")})
                    for proof in authority.proofs.values()}
        if len(prefixes) != 1:
            raise ValueError("all proposals need the same current authority record prefix")
        for proof in authority.proofs.values():
            for record in proof.evidence["source_records"]:
                if record["kind"] == "grant" and record["payload"]["valid_until_sequence"] < horizon:
                    raise ValueError("known grant expiry inside forecast requires a temporal model")
        self.task_id, self.principal = next(iter(scope))
        self.action_ids = tuple(ids)
        self._validate_policies()
        self.sha256 = digest(self.specification())

    def quantity(self, proposal):
        q, args = proposal["quantity"], proposal["binding"]["arguments"]
        return decimal_amount(args[q["amount_field"]]), args[q["currency_field"]]

    def _validate_policies(self):
        seen = set()
        for c in self.contracts:
            if set(c) != {"id", "task_id", "principal", "resources", "currency", "limit", "prior_requested", "source", "sequence", "history_complete"}:
                raise ValueError("strict requested-charge contract fields required")
            if c["history_complete"] is not True:
                raise ValueError("incomplete prior requested-charge history is unsupported, not zero")
            if not isinstance(c["resources"], list) or not c["resources"] or len(set(c["resources"])) != len(c["resources"]):
                raise ValueError("unique exact contract resource set required")
            for resource in c["resources"]:
                _text(resource)
            if currency_code(c["currency"]) is None:
                raise ValueError("explicit currency required")
            decimal_amount(c["limit"])
            decimal_amount(c["prior_requested"])
        for c in self.combinations:
            if set(c) != {"id", "task_id", "principal", "bindings", "source", "sequence"}:
                raise ValueError("strict unordered exact-action conjunction required")
            if not isinstance(c["bindings"], dict) or not 2 <= len(c["bindings"]) <= len(self.proposals):
                raise ValueError("combination needs at least two exact action bindings")
            lookup = {p["action_id"]: digest(p["binding"]) for p in self.proposals}
            if any(lookup.get(a) != h for a, h in c["bindings"].items()):
                raise ValueError("combination must bind exact observed action parameters")
        for c in (*self.contracts, *self.combinations):
            if c["task_id"] != self.task_id or c["principal"] != self.principal:
                raise ValueError("policy task/principal mismatch")
            identity = _text(c["id"])
            if identity in seen:
                raise ValueError("unique policy identities required")
            seen.add(identity)
            _text(c["source"])
            _sequence(c["sequence"], self.cutoff)

    def specification(self):
        result = {"workflow": copy.deepcopy(self.workflow), "authority": self.authority.specification(),
                "proposals": copy.deepcopy(self.proposals), "contracts": copy.deepcopy(self.contracts),
                "combinations": copy.deepcopy(self.combinations), "cutoff": self.cutoff, "horizon": self.horizon}
        # Preserve v1 dispatch-only input fingerprints. Mixed typed inputs have
        # an explicit marker; no silent reinterpretation of old dispatch labels.
        if any(v != "tool_dispatch" for v in self.event_semantics.values()):
            result["event_semantics"] = copy.deepcopy(self.event_semantics)
        return result

    def validate(self):
        self.authority.validate()
        if digest(self.specification()) != self.sha256:
            raise ValueError("prefix/policy input changed after binding")
        if self.action_ids != tuple(p["action_id"] for p in self.proposals):
            raise ValueError("derived action order changed")
        if set(self.event_semantics) != set(self.action_ids) or not set(self.event_semantics.values()) <= {"tool_dispatch", "literal_appearance"}:
            raise ValueError("event semantics changed")


class WorkflowRiskBN:
    MODEL = "WorkflowRiskBN-v1"
    SUPPORTS_EXCLUSIVE_CHOICES = False

    def __init__(self, *, registry=None, regularization=.001, max_binary_variables=16):
        self.registry = copy.deepcopy(registry or WorkflowFragmentRegistry())
        self.names = self.registry.names
        self.theta = np.zeros(len(self.names))
        if not np.isfinite(regularization) or regularization < 0 or type(max_binary_variables) is not int or not 1 <= max_binary_variables <= 18:
            raise ValueError("nonnegative regularization and bounded exact capacity required")
        self.regularization, self.max_binary_variables = regularization, max_binary_variables
        self.optimizer = None

    def _design(self, sample):
        if hasattr(sample, "choice_groups") and not self.SUPPORTS_EXCLUSIVE_CHOICES:
            raise ValueError("independent dispatch law cannot ignore explicit exclusive choices")
        if any(v != "tool_dispatch" for v in sample.event_semantics.values()):
            raise ValueError("independent dispatch model cannot consume literal-appearance events")
        x = np.zeros((len(sample.proposals), len(self.names)))
        for i, p in enumerate(sample.proposals):
            start = self.registry.specification["dispatch_families"].index(p["family"]) * 3
            x[i, start:start + 3] = (1., p["visible"] == "support", p["visible"] == "opposition")
        return x

    def _occurrence_worlds(self, sample, theta):
        x = self._design(sample)
        states = np.asarray(list(product((0., 1.), repeat=len(sample.proposals))))
        logits = x @ theta
        return states, states @ logits - np.logaddexp(0., logits).sum(), (states - expit(logits)) @ x

    def _authority_indices(self):
        return np.arange(len(self.names) - 3, len(self.names))

    def _parameter_bounds(self):
        return [(-12., 12.)] * len(self.theta)

    def _mechanisms(self, sample, states):
        """Compile deterministic policy fragments over SHARED dispatch bits."""
        terms, records = {}, []
        ids = sample.action_ids
        def record(template, identity, indices, **extra):
            records.append({"id": identity, "template": template,
                "actions": [ids[i] for i in indices],
                "workflow_nodes": sorted({sample.proposals[i]["node"] for i in indices}),
                "knowledge": copy.deepcopy(self.registry.templates[template]), **extra})
        for i, p in enumerate(sample.proposals):
            if "authorization" in self.registry.templates:
                record("authorization", "authorization:" + ids[i], [i])
            if "single_bound" in self.registry.templates and p["family"] == "payment" and p["quantity"]["single_limit"] is not None:
                amount, _ = sample.quantity(p)
                hit = amount > decimal_amount(p["quantity"]["single_limit"])
                identity = "single_bound:" + ids[i]
                terms[identity] = states[:, i].astype(bool) & hit
                record("single_bound", identity, [i], amount=str(amount), limit=p["quantity"]["single_limit"])
        with localcontext() as context:
            context.prec = 512
            for c in sample.contracts if "cumulative_budget" in self.registry.templates else ():
                indices, amounts = [], []
                for i, p in enumerate(sample.proposals):
                    if p["family"] != "payment" or p["binding"]["resource"] not in c["resources"]:
                        continue
                    amount, currency = sample.quantity(p)
                    if currency == c["currency"]:
                        indices.append(i)
                        amounts.append(amount)
                limit, prior = decimal_amount(c["limit"]), decimal_amount(c["prior_requested"])
                identity = "cumulative_budget:" + c["id"]
                terms[identity] = np.asarray([prior + sum((a * int(s[i]) for i, a in zip(indices, amounts)), Decimal(0)) > limit for s in states])
                record("cumulative_budget", identity, indices, amounts=[str(a) for a in amounts], prior_requested=str(prior), limit=str(limit), policy=copy.deepcopy(c))
        for c in sample.combinations if "prohibited_combination" in self.registry.templates else ():
            indices = [i for i, aid in enumerate(ids) if aid in c["bindings"]]
            identity = "prohibited_combination:" + c["id"]
            terms[identity] = np.all(states[:, indices] == 1, axis=1)
            record("prohibited_combination", identity, indices, policy=copy.deepcopy(c))
        return terms, records

    def _worlds(self, sample, theta):
        if not isinstance(sample, WorkflowRiskInput):
            raise ValueError("compiled prefix contract required")
        sample.validate()
        self.registry.validate()
        t = np.asarray(theta, float)
        if t.shape != self.theta.shape or not np.isfinite(t).all():
            raise ValueError("compatible finite parameters required")
        n = len(sample.proposals)
        if n + len(sample.authority.facts) > self.max_binary_variables:
            raise ValueError("exact joint capacity exceeded; no independent fallback")
        states, log_d, score_d = self._occurrence_worlds(sample, t)
        fact_ids, fact_states, mass_a, authority_scores, failed = sample.authority.worlds(t[self._authority_indices()])
        score_a = np.zeros((len(mass_a), len(self.names)))
        score_a[:, self._authority_indices()] = authority_scores
        log_mass = log_d[:, None] + np.log(mass_a)[None]
        terms, records = self._mechanisms(sample, states)
        events = {k: np.broadcast_to(v[:, None], log_mass.shape) for k, v in terms.items()}
        if "authorization" in self.registry.templates:
            for i, aid in enumerate(sample.action_ids):
                events["authorization:" + aid] = states[:, i, None].astype(bool) & failed[aid][None]
        channels = {}
        for name in self.registry.templates:
            applicable = [v for k, v in events.items() if k.startswith(name + ":")]
            channels["channel:" + name] = np.logical_or.reduce(applicable) if applicable else np.zeros(log_mass.shape, bool)
        events.update(channels)
        events["system:any"] = np.logical_or.reduce(list(channels.values()))
        for node in sample.workflow["nodes"]:
            # Queries whose contributors all belong to this functional node;
            # membership is not causal attribution or blame.
            applicable = [events[r["id"]] for r in records if len(r["workflow_nodes"]) == 1 and r["workflow_nodes"][0] == node]
            events["node:" + node] = np.logical_or.reduce(applicable) if applicable else np.zeros(log_mass.shape, bool)
        return dict(states=states, fact_ids=fact_ids, fact_states=fact_states, log_mass=log_mass,
                    score_d=score_d, score_a=score_a, events=events, records=records, dispatch=np.exp(log_d) @ states)

    def _condition_events(self, sample, p, event_conditions):
        if event_conditions is None:
            event_conditions = {}
        if not isinstance(event_conditions, dict) or not set(event_conditions) <= set(sample.action_ids):
            raise ValueError("conditions must name existing bound events, not policy labels")
        if any(type(v) is not int or v not in (0, 1) for v in event_conditions.values()):
            raise ValueError("explicit integer 0/1 event conditions required")
        conditions = dict(sorted(event_conditions.items()))
        mask = np.ones(len(p["states"]), bool)
        for aid, value in conditions.items():
            mask &= p["states"][:, sample.action_ids.index(aid)] == value
        log_evidence = float(logsumexp(p["log_mass"][mask]))
        if not np.isfinite(log_evidence):
            raise ValueError("event conditions have zero model probability")
        if not conditions:
            # Preserve all historical unconditional numerics and output keys.
            return conditions, np.exp(p["log_mass"]), log_evidence, None
        mass = np.zeros_like(p["log_mass"])
        mass[mask] = np.exp(p["log_mass"][mask] - log_evidence)
        mean_score = mass.sum(axis=1) @ p["score_d"] + mass.sum(axis=0) @ p["score_a"]
        return conditions, mass, log_evidence, mean_score

    def predict(self, sample, *, theta=None, derivatives=False, event_conditions=None):
        p = self._worlds(sample, self.theta if theta is None else theta)
        conditions, mass, log_evidence, mean_score = self._condition_events(sample, p, event_conditions)
        probabilities, gradients = {}, {}
        for key, event in p["events"].items():
            probabilities[key] = float(np.sum(mass * event))
            if derivatives:
                w = mass * event
                g = w.sum(axis=1) @ p["score_d"]
                g += w.sum(axis=0) @ p["score_a"]
                if conditions:
                    g -= probabilities[key] * mean_score
                gradients[key] = g
        instantiated = {r["template"] for r in p["records"]}
        occurrence = mass.sum(axis=1) @ p["states"] if conditions else p["dispatch"]
        result = {"model": self.MODEL, "probabilities": probabilities, "gradients": gradients,
            "dispatch_probabilities": dict(zip(sample.action_ids, occurrence.tolist())),
            "compiled_fragments": p["records"], "coverage": {k: k in instantiated for k in SUPPORTED},
            "disabled_templates": sorted(set(SUPPORTED) - set(self.registry.templates)),
            "joint_worlds": int(mass.size), "mass_sum": float(mass.sum()), "input_sha256": sample.sha256,
            "registry_sha256": self.registry.sha256, "calibrated": False, "proves_safety": False,
            "target": "Any configured policy violated by fixed pending actions dispatched at most once before horizon; not actual harm or settled spend",
            "limitations": ["No automatic DAG or semantic cue extraction.", "Normative workflow edges do not imply observed communication or dependence.",
                "Dispatches conditionally independent given visible categories; authority uncertainty independent of dispatch.",
                "No unlisted future actions, retries, ordered combination history or changing authority within horizon.",
                "Absent policy coverage is not proof of safety; sum of marginal risk types is not a system probability."]}
        if conditions:
            result["event_conditions"] = conditions
            result["condition_log_probability"] = log_evidence
            result["query_mode"] = "conditional_event_scenario"
            result["conditioning_semantics"] = "P(configured policy violation | declared event values, prefix); an explicit scenario, not proof of occurrence, not a do-intervention or a newly calibrated forecast. Unconditioned future events remain uncertain."
        return result

    def loss_gradient(self, theta, samples, labels, *, weights=None):
        if not samples or len(samples) != len(labels):
            raise ValueError("aligned nonempty samples and labels required")
        weights = np.ones(len(samples)) if weights is None else np.asarray(weights, float)
        if weights.shape != (len(samples),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("positive finite sample/frequency weights required")
        loss, grad = 0., np.zeros(len(self.names))
        worlds_cache = {}  # Only within this fixed-parameter objective call.
        for sample, label, weight in zip(samples, labels, weights / weights.sum()):
            if id(sample) not in worlds_cache:
                worlds_cache[id(sample)] = self._worlds(sample, theta)
            p = worlds_cache[id(sample)]
            if not isinstance(label, dict) or not label or not set(label) <= {"dispatch", "events", "facts", "risks"}:
                raise ValueError("explicit typed event/fact/risk labels required")
            compatible = np.ones(p["log_mass"].shape, bool)
            seen = 0
            for kind, values in label.items():
                if not isinstance(values, dict):
                    raise ValueError("label mappings required")
                for key, value in values.items():
                    if type(value) is not int or value not in (0, 1):
                        raise ValueError("binary labels required; omit unavailable labels")
                    if kind in {"dispatch", "events"} and key in sample.action_ids:
                        if kind == "dispatch" and sample.event_semantics[key] != "tool_dispatch":
                            raise ValueError("literal appearance label is not a tool-dispatch label")
                        observed = p["states"][:, sample.action_ids.index(key), None]
                    elif kind == "facts" and key in p["fact_ids"]:
                        observed = p["fact_states"][None, :, p["fact_ids"].index(key)]
                    elif kind == "risks" and key in p["events"]:
                        observed = p["events"][key]
                    else:
                        raise ValueError("label refers to an unknown variable/query")
                    compatible &= observed == value
                    seen += 1
            if not seen or not compatible.any():
                raise ValueError("empty or structurally contradictory labels")
            ll = logsumexp(np.where(compatible, p["log_mass"], -np.inf))
            if not np.isfinite(ll):
                raise ValueError("labels have zero probability under the declared structural support")
            post = np.exp(np.where(compatible, p["log_mass"] - ll, -np.inf))
            g = -(post.sum(axis=1) @ p["score_d"])
            g -= post.sum(axis=0) @ p["score_a"]
            loss -= weight * ll
            grad += weight * g
        t = np.asarray(theta)
        return float(loss + self.regularization * (t @ t)), grad + 2 * self.regularization * t

    def fit(self, samples, labels, *, weights=None, max_iter=500):
        self.theta, self.optimizer = _solve(lambda t: self.loss_gradient(t, samples, labels, weights=weights), self.theta, max_iter, bounds=self._parameter_bounds())
        return self

    def _event_node_id(self, action_id):
        return "dispatch:" + action_id

    def _occurrence_graph_nodes(self, sample, probabilities):
        return [{"id": self._event_node_id(aid), "parents": [], "kind": "binary_cpt",
                 "probability_one": [float(probabilities["dispatch"][i])], "workflow_node": sample.proposals[i]["node"]}
                for i, aid in enumerate(sample.action_ids)]

    def probability_graph(self, sample, *, output="system:any", event_conditions=None):
        p = self._worlds(sample, self.theta)
        conditions, _, _, _ = self._condition_events(sample, p, event_conditions)
        if output not in p["events"]:
            raise ValueError("unknown risk query")
        nodes = []
        for f in sample.authority.facts.values():
            nodes.append({"id": "fact:" + f.identity, "parents": [], "kind": "binary_cpt", "probability_one": [float(expit(self.theta[self._authority_indices()[f.kind]]))]})
        nodes.extend(self._occurrence_graph_nodes(sample, p))
        for node in nodes:
            for aid, value in conditions.items():
                if node["id"] == self._event_node_id(aid):
                    # Observations retain the original CPT, including its
                    # likelihood on latent parents. Do not replace with do().
                    node["observed"] = value
        for i, aid in enumerate(sample.action_ids):
            if "authorization" in self.registry.templates:
                proof = sample.authority.proofs[aid]
                paths = [["fact:" + f for f in path] for path in proof.paths]
                nodes.append({"id": "failed:" + aid, "parents": sorted({f for path in paths for f in path}), "kind": "all_paths_failed", "support_paths": paths})
                nodes.append({"id": "authorization:" + aid, "parents": [self._event_node_id(aid), "failed:" + aid], "kind": "and"})
        for record in p["records"]:
            if record["template"] == "authorization":
                continue
            parents = [self._event_node_id(aid) for aid in record["actions"]]
            # Export exact deterministic CPTs, compatible with the independent
            # existing scalar interpreter (no new auditor operator needed).
            values = []
            for bits in product((0, 1), repeat=len(parents)):
                whole = [0] * len(sample.action_ids)
                for aid, value in zip(record["actions"], bits):
                    whole[sample.action_ids.index(aid)] = value
                index = 0
                for value in whole:
                    index = 2 * index + value
                values.append(float(p["events"][record["id"]][index, 0]))
            nodes.append({"id": record["id"], "parents": parents, "kind": "binary_cpt", "probability_one": values, "knowledge": record["knowledge"]})
        for name in self.registry.templates:
            nodes.append({"id": "channel:" + name, "parents": [r["id"] for r in p["records"] if r["template"] == name], "kind": "count_at_least", "threshold": 1})
        nodes.append({"id": "system:any", "parents": ["channel:" + name for name in self.registry.templates], "kind": "count_at_least", "threshold": 1})
        for node in sample.workflow["nodes"]:
            nodes.append({"id": "node:" + node, "parents": [r["id"] for r in p["records"] if r["workflow_nodes"] == [node]], "kind": "count_at_least", "threshold": 1})
        result = {"schema": "executable-workflow-policy-graph-v1", "nodes": nodes, "output_id": output,
                  "input_sha256": sample.sha256, "registry_sha256": self.registry.sha256}
        if conditions:
            result["event_conditions"] = conditions
            result["query_mode"] = "conditional_event_scenario"
        return result

    def parameters(self):
        return {"model": self.MODEL, "names": list(self.names), "theta": self.theta.tolist(),
                "regularization": self.regularization, "max_binary_variables": self.max_binary_variables,
                "registry": copy.deepcopy(self.registry.specification), "registry_sha256": self.registry.sha256,
                "optimizer": copy.deepcopy(self.optimizer), "calibrated": False}

    @classmethod
    def restore(cls, saved):
        if saved["model"] != cls.MODEL:
            raise ValueError("wrong model")
        registry = WorkflowFragmentRegistry(saved["registry"])
        if registry.sha256 != saved["registry_sha256"] or list(registry.names) != saved["names"]:
            raise ValueError("parameter/knowledge identity mismatch")
        model = cls(registry=registry, regularization=saved["regularization"], max_binary_variables=saved["max_binary_variables"])
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite compatible saved parameters required")
        model.optimizer = copy.deepcopy(saved["optimizer"])
        return model
