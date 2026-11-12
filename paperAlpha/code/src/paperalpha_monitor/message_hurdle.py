"""Observable message-level hurdle factors in the existing event/policy DAG.

G = any tracked literal actually present in a message. It is NOT an inferred
intent, internal state or hidden propensity. G=0 forces every literal absent;
G=1 has a zero-truncated product Bernoulli law. The gate is integrated in VE,
and exported explicitly with normalized sequential binary CPTs.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from paperalpha_runtime.canonical import digest
from .event_elimination import EliminatingEventLaw, EliminatingDAGWorkflowRiskBN, LogFactor
from .observable_event_dag import FAMILIES, ObservableEventLaw
from .observable_event_learning import validated_labels
from .workflow_risk_bn import WorkflowFragmentRegistry

GATE_FEATURES = ("bias", "log_group_size", "parent_fraction", "any_parent")
HURDLE_SPEC = {"schema": "observable-message-hurdle-v1",
    "grouping": "literal events sharing functional node, actor and probability family",
    "gate_semantics": "at least one of this fixed, named literal set appears in the same completed message",
    "gate_features": list(GATE_FEATURES),
    "conditional_positive_law": "independent Bernoulli product conditioned on nonzero vector",
    "references": ["https://mc-stan.org/docs/2_32/stan-users-guide/zero-inflated.html"],
    "claim": "Standard hurdle construction adapted to observable messages; not novel distribution, causal identification or model internal intent."}


def hurdle_kernel(logits, gate, bits):
    """Batch log mass and derivatives w.r.t. field logits and gate logit."""
    logits, gate, bits = np.asarray(logits, float), np.asarray(gate, float), np.asarray(bits, float)
    if logits.ndim != 2 or logits.shape != bits.shape or gate.shape != (len(logits),) or logits.shape[1] < 1:
        raise ValueError("aligned nonempty binary vectors and gate logits required")
    if not np.isfinite(logits).all() or not np.isfinite(gate).all() or not np.isin(bits, [0, 1]).all():
        raise ValueError("finite logits and binary observations required")
    positive = np.any(bits, axis=1)
    base = np.logaddexp(0., logits)
    normalizer = -np.expm1(-base.sum(axis=1))
    if np.any(normalizer <= 0):
        raise ValueError("positive-vector mass below numerical range")
    mass = positive * gate - np.logaddexp(0., gate)
    mass += np.where(positive, (bits*logits-base).sum(axis=1)-np.log(normalizer), 0.)
    local_score = np.where(positive[:, None], bits-expit(logits)/normalizer[:, None], 0.)
    gate_score = positive-expit(gate)
    return mass, local_score, gate_score


class MessageHurdleLaw(EliminatingEventLaw):
    factorizes_into_independent_events = False
    gate_features = GATE_FEATURES
    message_spec = HURDLE_SPEC

    def __init__(self, specification=None, *, max_factor_variables=16):
        if specification is None:
            super().__init__(max_factor_variables=max_factor_variables)
            self.specification["message_hurdle"] = copy.deepcopy(self.message_spec)
        else:
            if specification.get("message_hurdle") != self.message_spec:
                raise ValueError("fixed observable message semantics required")
            base = copy.deepcopy(specification); base.pop("message_hurdle")
            super().__init__(base, max_factor_variables=max_factor_variables)
            self.specification = copy.deepcopy(specification)
        self.names += tuple(f"message:{family}:{feature}" for family in FAMILIES for feature in self.gate_features)
        self.sha256 = digest(self.specification)

    def design(self, event, parents):
        # Base implementation allocates len(self.names), leaving gate slots zero.
        return ObservableEventLaw.design(self, event, parents)

    @staticmethod
    def groups(dag):
        groups = {}
        for n in dag.order:
            e = dag.lookup[n]
            key = (e["node"], e["actor"], e["family"]) if e["semantics"] == "literal_appearance" else (n,)
            groups.setdefault(key, []).append(n)
        for group in groups.values():
            if dag.lookup[group[0]]["semantics"] == "literal_appearance":
                subjects = [dag.lookup[n]["subject"].casefold() for n in group]
                if len(set(subjects)) != len(subjects):
                    raise ValueError("duplicate literal is one event, not independent field aliases")
                observations = [dag.observations[n] for n in group if n in dag.observations]
                if len({(o["record_id"], o["text"], o["sequence"]) for o in observations}) > 1:
                    raise ValueError("message group must refer to one actual completed message")
        return list(groups.values())

    def gate_design(self, dag, group, values):
        parents = list(dict.fromkeys(p for n in group for p in dag.parents[n]))
        width = len(next(iter(values.values())))
        parent = np.asarray([values[n] for n in parents]).T if parents else np.zeros((width, 0))
        return np.column_stack((np.ones(width), np.full(width, np.log1p(len(group))),
            parent.mean(axis=1) if parents else np.zeros(width),
            parent.any(axis=1).astype(float) if parents else np.zeros(width)))

    def local_designs(self, dag, group, values):
        width = len(next(iter(values.values())))
        x = np.zeros((width, len(group), 25))
        for j, n in enumerate(group):
            e = dag.lookup[n]; offset = FAMILIES.index(e["family"])*5
            parents = np.asarray([values[p] for p in dag.parents[n]]).T if dag.parents[n] else np.zeros((width, 0))
            x[:, j, offset:offset+3] = [1., *e["cues"]]
            if len(dag.parents[n]):
                x[:, j, offset+3] = parents.mean(axis=1)
            if len(dag.parents[n]) > 1:
                x[:, j, offset+4] = parents.all(axis=1)
        return x

    def group_parts(self, dag, group, values, theta):
        x = self.local_designs(dag, group, values)
        g = self.gate_design(dag, group, values)
        width = len(self.gate_features)
        if g.shape[1] != width:
            raise ValueError("message feature width differs from specification")
        offset = 25+width*FAMILIES.index(dag.lookup[group[0]]["family"])
        return x, g, offset, x@theta[:25], g@theta[offset:offset+width]

    def _group_factor(self, dag, group, theta, observed):
        if dag.lookup[group[0]]["semantics"] != "literal_appearance":
            return self._node_factor(dag, group[0], theta, observed)
        full = set(group) | {p for n in group for p in dag.parents[n]}
        scope = tuple(n for n in dag.order if n in full and n not in observed)
        self._capacity(scope)
        states = np.asarray(list(product((0., 1.), repeat=len(scope)))).reshape(-1, len(scope)) if scope else np.empty((1, 0))
        values = {n: np.full(len(states), observed[n], float) if n in observed else states[:, scope.index(n)] for n in full}
        x,g,offset,z,h = self.group_parts(dag, group, values, theta)
        y = np.asarray([values[n] for n in group]).T
        mass, score_z, score_g = hurdle_kernel(z, h, y)
        score = np.zeros((len(states), len(self.names)))
        score[:, :25] = np.einsum("ij,ijd->id", score_z, x)
        score[:, offset:offset+g.shape[1]] = score_g[:, None]*g
        shape = (2,)*len(scope)
        return LogFactor(scope, mass.reshape(shape), score.reshape((*shape, len(theta))))

    def _initial_factors(self, dag, theta, observed, retained):
        groups = self.groups(dag)
        membership = {n: group for group in groups for n in group}
        active = set(observed) | set(retained); pending = list(active)
        while pending:
            n = pending.pop()
            for ancestor in (*membership[n], *dag.parents[n]):
                if ancestor not in active:
                    active.add(ancestor); pending.append(ancestor)
        return active, [self._group_factor(dag, g, theta, observed) for g in groups if g[0] in active]

    def message_summary(self, dag, members, theta):
        """Moments of one future message with fully observed input parents."""
        theta = self._check(dag, theta)
        groups = [g for g in self.groups(dag) if set(g)==set(members)]
        if len(groups)!=1 or not members or set(members)&(set(dag.observations)|set(dag.missing_past)):
            raise ValueError("one complete future message group required")
        group=groups[0]
        if dag.lookup[group[0]]["semantics"]!="literal_appearance": raise ValueError("literal message required")
        parents={p for n in group for p in dag.parents[n]}
        if not parents<=set(dag.observations): raise ValueError("summary requires all actual parent observations")
        values={n:np.array([o["value"]],float) for n,o in dag.observations.items()}
        if not values: values={group[0]:np.zeros(1)}
        _,_,_,z,h=self.group_parts(dag,group,values,theta)
        normalizer=-np.expm1(-np.logaddexp(0.,z[0]).sum())
        q=float(expit(h[0])); marginal=q*expit(z[0])/normalizer
        return {"members":group,"any_appearance_probability":q,"appearance_probabilities":dict(zip(group,map(float,marginal))),
            "expected_appearance_count":float(marginal.sum()),"semantics":self.message_spec["gate_semantics"],
            "input_sha256":dag.sha256,"approximation":False}

    def graph_nodes(self, dag, theta, mapping):
        """Explicit G then sequential fields; every G=1 world has some X=1."""
        theta = self._check(dag, theta); nodes = []
        for group in self.groups(dag):
            if dag.lookup[group[0]]["semantics"] != "literal_appearance":
                # Tool events remain ordinary observable-event logistic nodes.
                n = group[0]; parents = dag.parents[n]
                if len(parents)>18: raise ValueError("CPT export width exceeded")
                node = {"id": mapping[n], "parents": [mapping[p] for p in parents], "kind": "binary_cpt",
                    "probability_one": [float(expit(self.design(dag.lookup[n], v)@theta)) for v in product((0,1), repeat=len(parents))]}
                if n in dag.observations: node["observed"] = dag.observations[n]["value"]
                nodes.append(node); continue
            parents = tuple(n for n in dag.order if n in {p for a in group for p in dag.parents[a]})
            if len(parents)+len(group)>18: raise ValueError("explicit message CPT export capacity exceeded")
            gate_id = "message-gate:"+digest([mapping[n] for n in group])[:20]
            probability, parameters = [], {}
            for state in product((0,1), repeat=len(parents)):
                values = {n: np.array([value]) for n,value in zip(parents,state)}
                if not values: values = {group[0]: np.zeros(1)}
                _,_,_,z,g = self.group_parts(dag,group,values,theta)
                probability.append(float(expit(g[0]))); parameters[state] = expit(z[0])
            nodes.append({"id":gate_id, "parents":[mapping[p] for p in parents], "kind":"binary_cpt", "probability_one":probability,
                "event_semantics":"any_named_literal_in_this_message", "member_events":[mapping[n] for n in group],
                "meaning": "Observable aggregate, not agent intent or a causal claim; unobserved aggregate is marginalized."})
            for j,n in enumerate(group):
                probs = []
                for state in product((0,1), repeat=len(parents)+1+j):
                    parent_state, gate, previous = state[:len(parents)], state[len(parents)], state[len(parents)+1:]
                    p = parameters[parent_state]
                    value = 0. if not gate else p[j] if any(previous) else p[j]/(-np.expm1(np.log1p(-p[j:]).sum()))
                    probs.append(float(np.clip(value, 0, 1)))
                node = {"id":mapping[n], "parents":[*[mapping[p] for p in parents], gate_id, *[mapping[a] for a in group[:j]]],
                    "kind":"binary_cpt", "probability_one":probs, "event_semantics":"literal_appearance",
                    "subject":dag.lookup[n]["subject"], "meaning":"Sequential factorization of a permutation-invariant joint law; order is not causal."}
                if n in dag.observations:
                    node["observed"] = dag.observations[n]["value"]; node["evidence"] = copy.deepcopy(dag.observations[n])
                nodes.append(node)
        return nodes


def compile_message_training(law, dags, labels):
    """One complete future message factor per workflow; no duplicated OR label."""
    batches = {}
    if not dags or len(dags)!=len(labels): raise ValueError("aligned complete traces required")
    for dag, outcomes in zip(dags, labels):
        names = validated_labels(dag, outcomes)
        if dag.missing_past or set(names)!=set(dag.ids)-set(dag.observations):
            raise ValueError("complete actual future labels required; partial data needs marginal learning")
        if any(not set(dag.parents[n])<=set(dag.observations) for n in dag.observations):
            raise ValueError("ancestor-complete prefix required")
        observed = {**dag.observations, **outcomes}
        values = {n: np.array([o["value"]],float) for n,o in observed.items()}
        for group in law.groups(dag):
            if all(n in dag.observations for n in group): continue
            if not set(group)<=set(outcomes): raise ValueError("entire future message supervision required")
            literal = dag.lookup[group[0]]["semantics"] == "literal_appearance"
            if literal and len({(observed[n]["record_id"], observed[n]["text"], observed[n]["sequence"]) for n in group})!=1:
                raise ValueError("future group labels must come from the same actual message")
            x,g,offset,_,_ = law.group_parts(dag, group, values, np.zeros(len(law.names)))
            key = literal, offset, len(group)
            b = batches.setdefault(key, {"x":[], "g":[], "y":[], "literal":literal, "offset":offset})
            b["x"].append(x[0]); b["g"].append(g[0]); b["y"].append([outcomes[n]["value"] for n in group])
    for b in batches.values():
        for key in ("x","g","y"): b[key] = np.asarray(b[key],float)
    return list(batches.values()), len(dags)


def message_objective(theta, compiled, regularization):
    batches, count = compiled; total, grad = 0., np.zeros_like(theta)
    for b in batches:
        x,g,y,offset = b["x"],b["g"],b["y"],b["offset"]
        z = x@theta[:25]
        if b["literal"]:
            mass, dz, dh = hurdle_kernel(z, g@theta[offset:offset+g.shape[1]], y)
            grad[offset:offset+g.shape[1]] -= g.T@dh
        else:
            mass = (y*z-np.logaddexp(0.,z)).sum(axis=1); dz=y-expit(z)
        total -= mass.sum(); grad[:25] -= np.einsum("ij,ijd->d",dz,x)
    return float(total/count+regularization*(theta@theta)), grad/count+2*regularization*theta


class MessageHurdleWorkflowRiskBN(EliminatingDAGWorkflowRiskBN):
    MODEL = "ObservableMessageHurdleWorkflowRiskBN-v1"
    law_class = MessageHurdleLaw

    def __init__(self, *, event_law=None, max_factor_variables=16, **options):
        super().__init__(event_law=event_law or self.law_class(max_factor_variables=max_factor_variables), **options)
        if not isinstance(self.event_law, MessageHurdleLaw): raise ValueError("observable message law required")

    def predict_known_authorization_union(self, sample, *, derivatives=False):
        result = super().predict_known_authorization_union(sample, derivatives=derivatives)
        # A fully observed-parent, complete message OR has probability exactly
        # q. Re-normalizing its whole history can introduce ~1e-15 differences
        # between identical gate inputs, spuriously breaking ranking/alarm ties.
        # Keep the superclass scope/evidence checks and use q only when this
        # identity applies; partial groups or unknown parents retain joint VE.
        targets = set(result["invalid_authorization_events"])
        dag = sample.event_dag
        groups = [g for g in self.event_law.groups(dag) if set(g) == targets]
        if len(groups) != 1 or not targets:
            return result
        group = groups[0]
        if dag.lookup[group[0]]["semantics"] != "literal_appearance":
            return result
        if not {p for n in group for p in dag.parents[n]} <= set(dag.observations):
            return result
        values = {n: np.array([o["value"]], float) for n, o in dag.observations.items()}
        if not values:
            values = {group[0]: np.zeros(1)}
        _, features, offset, _, logit = self.event_law.group_parts(dag, group, values, self.theta[:self.event_count])
        probability = float(expit(logit[0]))
        result["probabilities"]["system:any"] = probability
        result["inference"]["exact_message_gate_identity"] = True
        if derivatives:
            gradient = np.zeros(len(self.theta))
            gradient[offset:offset+features.shape[1]] = probability*(1-probability)*features[0]
            result["gradients"]["system:any"] = gradient
        return result

    def fit_messages(self, dags, labels, *, max_iter=500):
        before = [d.sha256 for d in dags]
        compiled = compile_message_training(self.event_law, dags, labels)
        result = minimize(lambda t: message_objective(t,compiled,self.regularization), np.zeros(self.event_count),
            jac=True, method="L-BFGS-B", bounds=[(-20.,20.)]*self.event_count,
            options={"maxiter":max_iter,"ftol":1e-12,"gtol":1e-8})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all(): raise ValueError("nonfinite message fit")
        self.theta[:self.event_count] = result.x
        self.optimizer = {"success":bool(result.success),"message":str(result.message),"iterations":int(result.nit),
            "loss":float(result.fun),"max_gradient":float(np.max(np.abs(result.jac))),"trace_count":len(dags),
            "label_count":sum(len(x) for x in labels),"message_count":sum(len(b["y"]) for b in compiled[0]),
            "supervision":"one joint future-message likelihood per workflow; OR derived, not counted again",
            "authority_priors_fitted":False,"bounds":[-20.,20.]}
        if before!=[d.sha256 for d in dags]: raise ValueError("fit changed prefix")
        return self

    def predict(self, sample, **kwargs):
        result = super().predict(sample,**kwargs)
        result["limitations"][4] = "Literal fields within one message have an observable OR gate and a zero-truncated product law; unlisted between-message dependencies remain possible."
        result["inference"]["message_law"] = self.event_law.message_spec["schema"]
        return result

    @classmethod
    def restore(cls, saved):
        if saved["model"]!=cls.MODEL: raise ValueError("wrong message model")
        registry = WorkflowFragmentRegistry(saved["registry"])
        law = cls.law_class(saved["event_law"],max_factor_variables=saved["max_factor_variables"])
        model = cls(event_law=law,registry=registry,regularization=saved["regularization"],max_binary_variables=saved["max_binary_variables"])
        if list(model.names)!=saved["names"] or law.sha256!=saved["event_law_sha256"] or registry.sha256!=saved["registry_sha256"]:
            raise ValueError("saved model semantics differ")
        model.theta=np.asarray(saved["theta"],float)
        if model.theta.shape!=(len(model.names),) or not np.isfinite(model.theta).all(): raise ValueError("invalid saved coefficients")
        model.optimizer=copy.deepcopy(saved["optimizer"])
        return model
