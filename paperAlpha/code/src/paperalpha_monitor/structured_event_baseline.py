"""Same-supervision conditional classifiers with explicit DAG composition.

This is an attribution control, not an original native-MAS baseline. A product
of conditional classifiers on a DAG is itself a Bayesian-network factorization.
The GLM uses the event BN's effective features, but independently constructs and
fits them. HistGB replaces the local conditional law, not the available evidence.
No production event design, training objective or inference engine is called.
"""
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

from .observable_event_learning import validated_labels

FAMILIES = ("prepare", "work", "review", "publish", "dispatch")
FEATURES = ("bias", "visible_support", "visible_opposition", "parent_mean", "multiple_parents_all")
TREE_CONFIG = dict(max_leaf_nodes=7, max_iter=100, min_samples_leaf=3,
                   l2_regularization=.1, random_state=17)


def event_features(event, parents, *, relations=True, soft=False):
    """Same 25 effective GLM columns; IDs, future labels and policy are excluded."""
    p = np.asarray(parents, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite parent probabilities or binary states required")
    if not soft and not np.isin(p, [0, 1]).all():
        raise ValueError("hard conditional features require binary parents")
    x = np.zeros(25)
    start = 5 * FAMILIES.index(event["family"])
    x[start:start+3] = [1, *event["cues"]]
    if relations and len(p):
        x[start+3] = p.mean()
        if len(p) > 1:
            x[start+4] = p.prod() if soft else float(p.all())
    return x


def validate_prefix(dag):
    dag.validate()
    if dag.missing_past:
        raise ValueError("this control requires complete historical content")
    if any(not set(dag.parents[n]) <= set(dag.observations) for n in dag.observations):
        raise ValueError("this control requires an ancestor-complete prefix")


def training_table(dags, labels, *, relations=True):
    if not dags or len(dags) != len(labels):
        raise ValueError("aligned nonempty training flows required")
    x, y, families = [], [], []
    for dag, outcomes in zip(dags, labels):
        validate_prefix(dag)
        identities = validated_labels(dag, outcomes)
        if set(identities) != set(dag.ids)-set(dag.observations):
            raise ValueError("every future event needs actual separate supervision")
        values = {k: v["value"] for k, v in {**dag.observations, **outcomes}.items()}
        for identity in identities:
            event = dag.lookup[identity]
            x.append(event_features(event, [values[p] for p in dag.parents[identity]], relations=relations))
            y.append(values[identity]); families.append(event["family"])
    return np.asarray(x), np.asarray(y), families


def glm_objective(theta, x, y, flow_count, penalty):
    if flow_count < 1 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("positive flow count and finite nonnegative penalty required")
    z = x @ theta
    loss = (np.logaddexp(0, z)-y*z).sum()/flow_count + penalty*np.square(theta).sum()
    gradient = x.T @ (expit(z)-y)/flow_count + 2*penalty*theta
    return float(loss), gradient


def rank_report(x, y, families):
    report = {"rows":len(y), "positive_events":int(y.sum()), "columns":x.shape[1],
              "rank":int(np.linalg.matrix_rank(x)), "families":{}}
    for family in FAMILIES:
        chosen = np.asarray([f == family for f in families]); a = x[chosen]
        report["families"][family] = {"rows":len(a), "positives":int(y[chosen].sum()),
            "rank":int(np.linalg.matrix_rank(a)) if len(a) else 0}
    return report


class StructuredEventBaseline:
    SCHEMA = "same-supervision-conditional-classifiers-v1"

    def __init__(self, *, kind="glm", relations=True, penalty=.001, tree_config=None):
        if kind not in {"glm", "histgb"} or type(relations) is not bool:
            raise ValueError("explicit GLM/HistGB and relation setting required")
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("finite nonnegative penalty required")
        self.kind, self.relations, self.penalty = kind, relations, penalty
        self.tree_config = dict(TREE_CONFIG if tree_config is None else tree_config)
        self.coef = None; self.tree = None; self.fit_report = None

    def fit(self, dags, labels, *, max_iter=500):
        before = [d.sha256 for d in dags]
        x, y, families = training_table(dags, labels, relations=self.relations)
        report = {"flows":len(dags), "label_count":len(y), "design":rank_report(x,y,families),
            "supervision":"One complete future-event likelihood per earliest training prefix; actual parent labels used only during training.",
            "authority_priors_fitted":False}
        if self.kind == "glm":
            result = minimize(lambda t:glm_objective(t,x,y,len(dags),self.penalty), np.zeros(25),
                jac=True, method="L-BFGS-B", options={"maxiter":max_iter,"ftol":1e-12,"gtol":1e-8})
            if not result.success or not np.isfinite(result.x).all():
                raise ValueError("conditional GLM did not converge: "+str(result.message))
            self.coef = result.x
            report.update(loss=float(result.fun), max_gradient=float(abs(result.jac).max()),
                          iterations=int(result.nit), success=bool(result.success))
        else:
            self.tree = HistGradientBoostingClassifier(**self.tree_config).fit(x,y)
            report.update(success=True, classes=self.tree.classes_.tolist(), iterations=int(self.tree.n_iter_),
                weighting="One row per same future event as BN; HistGB uses its own mean-row loss and L2, not numerically matched BN regularization.")
        if before != [d.sha256 for d in dags]:
            raise ValueError("training changed evidence")
        self.fit_report = report
        return self

    def probability(self, x):
        x = np.asarray(x,float)
        if x.ndim != 2 or x.shape[1] != 25 or not np.isfinite(x).all():
            raise ValueError("finite 25-column effective features required")
        if self.kind == "glm":
            if self.coef is None: raise ValueError("unfitted GLM")
            p = expit(x @ self.coef)
        else:
            if self.tree is None: raise ValueError("unfitted conditional tree")
            q = self.tree.predict_proba(x)
            p = q[:,int(np.flatnonzero(self.tree.classes_==1)[0])] if 1 in self.tree.classes_ else np.zeros(len(x))
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError("invalid classifier probabilities")
        return p

    @staticmethod
    def independent_joint(joint, target_count):
        worlds = np.asarray(list(product((0,1),repeat=target_count)))
        marginals = np.asarray(joint) @ worlds
        return np.prod(np.where(worlds,marginals,1-marginals),axis=1)

    def predict_joint(self, dag, targets, *, inference="exact", max_free=18):
        validate_prefix(dag)
        observed = {k:v["value"] for k,v in dag.observations.items()}
        free = [n for n in dag.order if n not in observed]
        if not targets or len(set(targets))!=len(targets) or not set(targets)<=set(free):
            raise ValueError("distinct future targets required")
        if type(max_free) is not int or not 1<=max_free<=18 or len(free)>max_free:
            raise ValueError("explicit exhaustive control capacity exceeded")
        if inference not in {"exact","mean_field","same_marginals_independent"}:
            raise ValueError("known inference control required")
        if inference=="mean_field":
            values = dict(observed)
            for n in free:
                x = event_features(dag.lookup[n],[values[p] for p in dag.parents[n]],relations=self.relations,soft=True)
                values[n] = float(self.probability(x[None])[0])
            worlds = np.asarray(list(product((0,1),repeat=len(targets))))
            p = np.asarray([values[t] for t in targets])
            return np.prod(np.where(worlds,p,1-p),axis=1)
        worlds = np.asarray(list(product((0,1),repeat=len(free))))
        values = {**{k:np.full(len(worlds),v) for k,v in observed.items()},
                  **{n:worlds[:,i] for i,n in enumerate(free)}}
        log_mass = np.zeros(len(worlds))
        for n in free:
            parents = dag.parents[n]
            x = np.asarray([event_features(dag.lookup[n],[values[p][i] for p in parents],relations=self.relations)
                            for i in range(len(worlds))])
            p = self.probability(x)
            with np.errstate(divide="ignore"):
                log_mass += np.log(np.where(values[n],p,1-p))
        z = float(logsumexp(log_mass))
        if not np.isfinite(z) or abs(z)>1e-10:
            raise ValueError("conditional classifiers do not form a normalized joint")
        weights = np.exp(log_mass-z)
        indices = sum(values[n]*2**(len(targets)-i-1) for i,n in enumerate(targets))
        result = np.bincount(indices,weights=weights,minlength=2**len(targets))
        return self.independent_joint(result,len(targets)) if inference=="same_marginals_independent" else result

    def metadata(self):
        return {"schema":self.SCHEMA,"kind":self.kind,"relations":self.relations,"penalty":self.penalty,
            "coef":self.coef.tolist() if self.coef is not None else None,"tree_config":self.tree_config,
            "fit_report":self.fit_report,"features":list(FEATURES),"families":list(FAMILIES),
            "meaning":"DAG-composed conditional classifier; itself a BN factorization, not a new native MAS monitor."}

    @classmethod
    def restore(cls, metadata, *, tree=None):
        if metadata["schema"]!=cls.SCHEMA or metadata["features"]!=list(FEATURES) or metadata["families"]!=list(FAMILIES):
            raise ValueError("compatible conditional baseline metadata required")
        model=cls(kind=metadata["kind"],relations=metadata["relations"],penalty=metadata["penalty"],tree_config=metadata["tree_config"])
        model.fit_report=metadata["fit_report"]
        if model.kind=="glm":
            model.coef=np.asarray(metadata["coef"],float)
            if model.coef.shape!=(25,) or not np.isfinite(model.coef).all(): raise ValueError("finite GLM coefficients required")
        else:
            if tree is None or tree.n_features_in_!=25: raise ValueError("saved conditional tree required")
            model.tree=tree
        return model
