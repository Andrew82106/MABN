"""Same-information development controls, never a claim of a new MAS method.

Conditional GLM/HistGB factors receive the exact declared event feature
function and complete later event labels. GLM uses the existing per-task joint
likelihood objective. Inference is independent component-wise enumeration,
not the production inference engine or an independent-marginal fallback.
Only one explicitly declared earliest prefix per task may enter factor fitting.
"""
from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

from .observable_event_learning import complete_trace_objective, validated_labels


DEFAULT_TREE_CONFIG = {
    "max_leaf_nodes": 7, "max_iter": 100, "min_samples_leaf": 3,
    "l2_regularization": .1, "random_state": 17, "early_stopping": False,
}


def _names(values):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError("fixed unique nonempty feature names required")
    return values


def _groups(task_ids, group_ids=None):
    tasks = tuple(task_ids)
    groups = tasks if group_ids is None else tuple(group_ids)
    if not tasks or len(tasks) != len(groups) or any(not isinstance(x, str) or not x for x in (*tasks, *groups)):
        raise ValueError("aligned nonempty task and template-group identifiers required")
    task_groups = {}
    for task, group in zip(tasks, groups):
        if task in task_groups and task_groups[task] != group:
            raise ValueError("one task cannot cross template groups")
        task_groups[task] = group
    return tasks, groups


def equal_group_task_weights(task_ids, group_ids=None, group_weights=None):
    """Each group has its declared total weight, shared among distinct tasks."""
    tasks, groups = _groups(task_ids, group_ids)
    if len(set(tasks)) != len(tasks):
        raise ValueError("repeated task/prefix; factor fitting permits one earliest prefix per task")
    counts = Counter(groups)
    weights = dict.fromkeys(counts, 1.) if group_weights is None else dict(group_weights)
    if set(weights) != set(counts) or any(isinstance(w, bool) or not isinstance(w, (int, float))
                                        or not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("one finite positive weight per template group required")
    return np.asarray([weights[group] / counts[group] for group in groups], dtype=float)


def template_group_folds(task_ids, group_ids, *, n_splits=4, seed=17):
    """Label-blind folds. All objects, fields, continuations and prefixes stay together."""
    tasks, groups = _groups(task_ids, group_ids)
    distinct = sorted(set(groups))
    if type(n_splits) is not int or not 2 <= n_splits <= len(distinct):
        raise ValueError("at least two groups and no more folds than groups required")
    shuffled = np.asarray(distinct, dtype=object)[np.random.default_rng(seed).permutation(len(distinct))]
    result = []
    for fold, held in enumerate(np.array_split(shuffled, n_splits)):
        held = set(held.tolist())
        test = [i for i, group in enumerate(groups) if group in held]
        train = [i for i, group in enumerate(groups) if group not in held]
        result.append({"fold": fold, "train_indices": train, "test_indices": test,
                       "train_groups": sorted(set(groups)-held), "test_groups": sorted(held)})
    return result


def weighted_complete_objective(theta, x, y, *, row_weights, task_weight_sum, regularization):
    """Weighted sum of complete event factors divided by total task weight.

    With one equally weighted task per group this is exactly the existing
    complete_trace_objective, including its L2 scaling (not mean-event loss).
    """
    theta, x, y, weights = map(lambda a: np.asarray(a, dtype=float), (theta, x, y, row_weights))
    if (x.ndim != 2 or theta.shape != (x.shape[1],) or y.shape != (len(x),)
            or weights.shape != y.shape or not np.isfinite(x).all()
            or not np.isfinite(theta).all() or not np.isin(y, [0, 1]).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isfinite(task_weight_sum) or task_weight_sum <= 0
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError("finite aligned event design, positive task weights and nonnegative penalty required")
    if np.all(weights == 1.):
        return complete_trace_objective(theta, x, y, task_weight_sum, regularization)
    logits = x @ theta
    residual = weights * (expit(logits)-y)
    return (float(weights @ (np.logaddexp(0., logits)-y*logits) / task_weight_sum
                  + regularization*(theta @ theta)),
            x.T @ residual / task_weight_sum + 2*regularization*theta)


def weak_components(dag):
    """Declared event edges, not workflow names or learned score similarity."""
    dag.validate()
    adjacent = {identity: set() for identity in dag.ids}
    for identity in dag.ids:
        for parent in dag.parents[identity]:
            adjacent[identity].add(parent)
            adjacent[parent].add(identity)
    remaining = set(dag.ids)
    components = []
    for first in dag.order:
        if first not in remaining:
            continue
        pending, members = [first], set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(adjacent[current]-members)
        remaining -= members
        components.append(tuple(n for n in dag.order if n in members))
    return tuple(components)


class ConditionalFactorBaseline:
    """Exact same-feature conditional classifiers and independently coded inference.

    ``feature_function(dag, identity, parent_bits)`` is supplied by the shared
    observable feature contract, e.g. law.node_design with law.names. It must
    depend only on that node, declared context and the supplied parent states.
    Caller-provided labels never enter a prediction feature function.
    """

    def __init__(self, feature_names, feature_function, *, kind="glm", regularization=.001,
                 tree_config=None):
        self.feature_names = _names(feature_names)
        if not callable(feature_function) or kind not in {"glm", "histgb"}:
            raise ValueError("explicit feature function and GLM/HistGB kind required")
        if isinstance(regularization, bool) or not np.isfinite(regularization) or regularization < 0:
            raise ValueError("finite nonnegative regularization required")
        self.feature_function = feature_function
        self.kind, self.regularization = kind, float(regularization)
        self.tree_config = {**DEFAULT_TREE_CONFIG, **(tree_config or {})}
        if self.tree_config.get("early_stopping") is not False:
            raise ValueError("row-random internal early stopping is forbidden; use held template groups")
        self.coef, self.tree, self.fit_report = None, None, None

    def _design(self, dag, identity, parents):
        row = np.asarray(self.feature_function(dag, identity, tuple(parents)), dtype=float)
        if row.shape != (len(self.feature_names),) or not np.isfinite(row).all():
            raise ValueError("feature function changed its fixed finite dimension")
        return row

    def training_design(self, dags, labels, *, task_ids, earliest_prefix_sequences,
                        group_ids=None, group_weights=None):
        tasks, groups = _groups(task_ids, group_ids)
        if not dags or len(dags) != len(labels) or len(dags) != len(tasks):
            raise ValueError("aligned complete earliest-prefix training tasks required")
        task_weights = equal_group_task_weights(tasks, groups, group_weights)
        if not isinstance(earliest_prefix_sequences, dict) or set(earliest_prefix_sequences) != set(tasks):
            raise ValueError("explicit frozen earliest prefix sequence for every task required")
        x, y, row_weights, counts = [], [], [], []
        for index, (dag, outcomes) in enumerate(zip(dags, labels)):
            earliest = earliest_prefix_sequences[tasks[index]]
            if type(earliest) is not int or dag.prefix_sequence != earliest:
                raise ValueError("factor fit accepts the declared earliest prefix only")
            identities = validated_labels(dag, outcomes)
            if dag.missing_past or set(identities) != set(dag.ids)-set(dag.observations):
                raise ValueError("every future event needs actual supervision; no future-parent zero imputation")
            if any(not set(dag.parents[n]) <= set(dag.observations) for n in dag.observations):
                raise ValueError("complete factor training requires an ancestor-complete prefix")
            values = {n: observation["value"] for n, observation in {**dag.observations, **outcomes}.items()}
            for identity in identities:
                x.append(self._design(dag, identity, [values[p] for p in dag.parents[identity]]))
                y.append(values[identity])
                row_weights.append(task_weights[index])
            counts.append(len(identities))
        return {
            "x": np.asarray(x), "y": np.asarray(y), "row_weights": np.asarray(row_weights),
            "task_weight_sum": float(task_weights.sum()), "task_weights": task_weights,
            "task_count": len(tasks), "group_count": len(set(groups)), "event_counts_by_task": counts,
        }

    def fit(self, dags, labels, *, task_ids, earliest_prefix_sequences, group_ids=None,
            group_weights=None, max_iter=500):
        before = [dag.sha256 for dag in dags]
        design = self.training_design(dags, labels, task_ids=task_ids,
            earliest_prefix_sequences=earliest_prefix_sequences, group_ids=group_ids,
            group_weights=group_weights)
        x, y = design["x"], design["y"]
        report = {"kind": self.kind, "task_count": design["task_count"],
                  "group_count": design["group_count"], "event_label_count": len(y),
                  "task_weight_sum": design["task_weight_sum"],
                  "training_prefixes_per_task": 1, "authority_priors_fitted": False,
                  "supervision": "Same complete later event labels, not only final workflow labels.",
                  "feature_names": list(self.feature_names),
                  "design_rank": int(np.linalg.matrix_rank(x)),
                  "feature_nonzero_rows": np.count_nonzero(x, axis=0).tolist()}
        if self.kind == "glm":
            objective = lambda theta: weighted_complete_objective(theta, x, y,
                row_weights=design["row_weights"], task_weight_sum=design["task_weight_sum"],
                regularization=self.regularization)
            optimized = minimize(objective, np.zeros(len(self.feature_names)), method="L-BFGS-B", jac=True,
                                 options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
            if not optimized.success or not np.isfinite(optimized.x).all() or not np.isfinite(optimized.fun):
                raise ValueError("conditional GLM failed to converge: " + str(optimized.message))
            self.coef = optimized.x
            report.update(loss=float(optimized.fun), iterations=int(optimized.nit),
                          max_gradient=float(np.max(np.abs(optimized.jac))), success=True,
                          objective="Weighted task joint log loss plus L2; equal-weight case exactly matches complete_trace_objective.")
        else:
            self.tree = HistGradientBoostingClassifier(**self.tree_config).fit(
                x, y, sample_weight=design["row_weights"])
            report.update(success=True, iterations=int(self.tree.n_iter_), classes=self.tree.classes_.tolist(),
                objective="Same event rows, labels and task weights. HistGB has its own weighted row-loss/L2 convention, not numerically identical GLM regularization.")
        if before != [dag.sha256 for dag in dags]:
            raise ValueError("training mutated an observation prefix")
        for dag in dags:
            dag.validate()
        self.fit_report = report
        return self

    def set_coefficients(self, coefficients):
        """Explicit frozen GLM parameters for equivalence checks or restoration."""
        values = np.asarray(coefficients, dtype=float)
        if self.kind != "glm" or values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError("finite GLM coefficient vector matching the feature names required")
        self.coef = values.copy()
        return self

    def conditional_log_probabilities(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError("finite matrix matching the frozen feature names required")
        if self.kind == "glm":
            if self.coef is None:
                raise ValueError("unfitted conditional GLM")
            logits = x @ self.coef
            return np.column_stack([-np.logaddexp(0., logits), -np.logaddexp(0., -logits)])
        if self.tree is None:
            raise ValueError("unfitted conditional HistGB")
        probabilities = np.zeros((len(x), 2))
        if len(self.tree.classes_) == 1:
            probabilities[:, int(self.tree.classes_[0])] = 1.
        else:
            probabilities[:, self.tree.classes_.astype(int)] = self.tree.predict_proba(x)
        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    def _enumerate_component(self, dag, members, targets, max_free):
        observed = {n: dag.observations[n]["value"] for n in members if n in dag.observations}
        free = tuple(n for n in members if n not in observed)
        if len(free) > max_free:
            raise ValueError("exact component free-variable capacity exceeded; no independence fallback")
        worlds = (np.asarray(list(product((0, 1), repeat=len(free))), dtype=np.int8)
                  if free else np.empty((1, 0), dtype=np.int8))
        values = {**{n: np.full(len(worlds), bit, dtype=np.int8) for n, bit in observed.items()},
                  **{n: worlds[:, index] for index, n in enumerate(free)}}
        log_mass = np.zeros(len(worlds))
        for identity in members:
            parent_names = dag.parents[identity]
            # Prefix facts are conditioned-on inputs, not training targets.
            # Their fully observed-parent factors cancel analytically. Do not
            # ask a classifier trained only on future messages to invent a
            # distribution for the configured vault/source. An observed node
            # with a missing parent still contributes its likelihood below.
            if identity in observed and all(parent in observed for parent in parent_names):
                continue
            if parent_names:
                parent_rows = np.column_stack([values[p] for p in parent_names])
                configurations, inverse = np.unique(parent_rows, axis=0, return_inverse=True)
            else:
                configurations, inverse = np.empty((1, 0)), np.zeros(len(worlds), dtype=int)
            x = np.asarray([self._design(dag, identity, row) for row in configurations])
            log_probabilities = self.conditional_log_probabilities(x)
            log_mass += log_probabilities[inverse, values[identity]]
        log_evidence = float(logsumexp(log_mass))
        if not np.isfinite(log_evidence):
            raise ValueError("observed evidence has zero or invalid probability under the fitted factors")
        log_mass -= log_evidence
        indices = np.zeros(len(worlds), dtype=np.int64)
        for target in targets:
            indices = 2*indices + values[target]
        log_joint = np.full(2**len(targets), -np.inf)
        np.logaddexp.at(log_joint, indices, log_mass)
        if abs(float(logsumexp(log_joint))) > 1e-9:
            raise ValueError("independent enumeration failed normalization")
        return {"targets": list(targets), "log_probabilities": log_joint.tolist(),
                "node_count": len(members), "free_count": len(free), "world_count": len(worlds)}

    def predict(self, dag, targets, *, max_free=18, include_dense_joint=False, max_dense_targets=18):
        """Return union, marginals and exact component target distributions.

        No outcome argument enters inference. ``target_vector_log_probability``
        below scores later outcomes against this already-computed prediction.
        Products across components follow the declared graph at fixed fitted
        parameters; shared unobserved parents stay inside their component.
        """
        dag.validate()
        targets = tuple(targets)
        if len(set(targets)) != len(targets) or not set(targets) <= set(dag.ids)-set(dag.observations):
            raise ValueError("distinct unobserved future targets required")
        if set(targets) & set(dag.missing_past):
            raise ValueError("missing historical content is not a future forecast target")
        if type(max_free) is not int or not 1 <= max_free <= 18:
            raise ValueError("exact component capacity must be explicitly between one and eighteen")
        if type(max_dense_targets) is not int or not 1 <= max_dense_targets <= 18:
            raise ValueError("dense target output capacity must be between one and eighteen")
        if include_dense_joint and len(targets) > max_dense_targets:
            raise ValueError("dense target-vector capacity exceeded; use exact component tables")
        tables, marginals, log_none = [], {}, 0.
        for members in weak_components(dag):
            selected = tuple(n for n in targets if n in members)
            if not selected:
                continue
            table = self._enumerate_component(dag, members, selected, max_free)
            log_joint = np.asarray(table["log_probabilities"])
            states = np.asarray(list(product((0, 1), repeat=len(selected))))
            for index, target in enumerate(selected):
                marginals[target] = float(np.exp(logsumexp(log_joint[states[:, index] == 1])))
            log_none += log_joint[0]
            tables.append(table)
        result = {
            "targets": list(targets), "union_probability": float(np.clip(-np.expm1(log_none), 0., 1.)),
            "target_marginals": {n: marginals[n] for n in targets}, "components": tables,
            "component_count": len(tables), "max_component_free": max((t["free_count"] for t in tables), default=0),
            "inference": "exact enumeration within declared weak components; exact factorization across disconnected components",
            "independence_fallback": False,
        }
        if include_dense_joint:
            result["joint_log_probabilities"] = [target_vector_log_probability(result, dict(zip(targets, bits)))
                for bits in product((0, 1), repeat=len(targets))]
        return result

    def metadata(self):
        return {"schema": "same-information-conditional-factor-baseline", "kind": self.kind,
                "feature_names": list(self.feature_names), "regularization": self.regularization,
                "coef": None if self.coef is None else self.coef.tolist(),
                "tree_config": dict(self.tree_config), "fit_report": self.fit_report,
                "meaning": "A same-supervision BN factorization control, not a distinct new MAS security method."}


def target_vector_log_probability(prediction, outcomes):
    """Offline score of a later observed target vector; never predictor input."""
    if not isinstance(outcomes, dict) or set(outcomes) != set(prediction["targets"]):
        raise ValueError("one later binary outcome per named prediction target required")
    if any(type(value) is not int or value not in (0, 1) for value in outcomes.values()):
        raise ValueError("binary target outcomes required")
    total = 0.
    seen = set()
    for table in prediction["components"]:
        if seen & set(table["targets"]):
            raise ValueError("overlapping component target tables")
        seen.update(table["targets"])
        index = 0
        for identity in table["targets"]:
            index = 2*index + outcomes[identity]
        total += table["log_probabilities"][index]
    if seen != set(prediction["targets"]):
        raise ValueError("component tables do not cover the prediction targets")
    return float(total)


def grouped_proper_scores(probabilities, labels, task_ids, group_ids, *, epsilon=1e-15):
    """Average prefixes within tasks, tasks within templates, then templates.

    Proper binary scores only; no threshold is learned here. Existing
    forecast_alarm.calibrate_forecast must receive independent calibration
    groups when alarm metrics are evaluated. No tiny-sample guarantee is made.
    """
    tasks, groups = _groups(task_ids, group_ids)
    p, y = np.asarray(probabilities, float), np.asarray(labels, float)
    if (p.shape != (len(tasks),) or y.shape != p.shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all()
            or not 0 < epsilon < .5):
        raise ValueError("aligned binary labels, finite probabilities and explicit clipping required")
    q = np.clip(p, epsilon, 1-epsilon)
    task_scores = {}
    for task in sorted(set(tasks)):
        selected = np.asarray([t == task for t in tasks])
        if len(np.unique(y[selected])) != 1:
            raise ValueError("one fixed final target label must persist across a task's prefixes")
        task_scores[task] = {"brier": float(np.square(p[selected]-y[selected]).mean()),
            "log_loss": float(-(y[selected]*np.log(q[selected])+(1-y[selected])*np.log1p(-q[selected])).mean())}
    task_group = dict(zip(tasks, groups))
    group_scores = {group: {metric: float(np.mean([values[metric] for task, values in task_scores.items()
        if task_group[task] == group])) for metric in ("brier", "log_loss")} for group in sorted(set(groups))}
    return {**{metric: float(np.mean([values[metric] for values in group_scores.values()]))
               for metric in ("brier", "log_loss")},
            "task_count": len(task_scores), "group_count": len(group_scores), "prefix_count": len(p),
            "probability_clipping_epsilon": epsilon,
            "independent_unit": "template/task group, not fields, API stages or prefixes"}
