"""Offline, task-held-out development comparison for repeated real messages.

No API access or test-set selection. A continuation pair is an outcome, not an
independent upstream sample. This module consumes the collector's verified
cohort; the CLI always obtains that object by replaying the original collection.
"""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from paperalpha_runtime import repeated_message_study as study
from .grouped_event_learning import GroupedEventTrace, fit_grouped_event_traces
from .observable_event_dag import ObservableEventDAG, ObservableEventLaw, DAGWorkflowRiskBN
from .structured_event_baseline import (StructuredEventBaseline, event_features,
                                        glm_objective, rank_report)

PENALTIES = (.001, .03)
TARGETS = ("publish-a", "publish-b")
METHODS = ("exact_bn", "same_marginals_independent", "no_parent_retrained", "matched_glm")
PHASES = ("early", "after_review")
STATES = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)))
PARENT_COLUMNS = tuple(5 * family + feature for family in range(5) for feature in (3, 4))
METRICS = ("joint_brier", "joint_nll", "marginal_brier", "risk_brier")
NUMERIC_TOLERANCES = {"bn_glm_joint": 1e-7, "declared_independence": 1e-10,
                      "same_marginals": 1e-12, "no_parent_restore": 1e-12}


class _NoParentLaw(ObservableEventLaw):
    def design(self, event, parents):
        x = super().design(event, parents)
        x[list(PARENT_COLUMNS)] = 0.
        return x


class _NoParentBN(DAGWorkflowRiskBN):
    def __init__(self, **kwargs):
        super().__init__(event_law=_NoParentLaw(), **kwargs)


def independent_grouped_table(traces, *, relations=True):
    """Construct GLM features independently; do not call the BN design/table."""
    if not traces:
        raise ValueError("nonempty grouped training data required")
    x, y, families = [], [], []
    seen = set()
    for trace in traces:
        trace.validate()
        records = {v["record_id"] for block in [trace.shared_labels, *trace.continuations] for v in block.values()}
        if records & seen:
            raise ValueError("actual response records repeated across training groups")
        seen.update(records)
        observed = {n: v["value"] for n, v in trace.dag.observations.items()}
        shared = {n: v["value"] for n, v in trace.shared_labels.items()}
        blocks = [(trace.shared_labels, {**observed, **shared})]
        blocks += [(c, {**observed, **shared, **{n: v["value"] for n, v in c.items()}}) for c in trace.continuations]
        for labels, values in blocks:
            for identity in trace.dag.order:
                if identity not in labels:
                    continue
                event = trace.dag.lookup[identity]
                x.append(event_features(event, [values[p] for p in trace.dag.parents[identity]], relations=relations))
                y.append(labels[identity]["value"])
                families.append(event["family"])
    return np.asarray(x), np.asarray(y), families


def fit_matched_glm(traces, *, penalty, max_iter=500):
    x, y, families = independent_grouped_table(traces)
    result = minimize(lambda t: glm_objective(t, x, y, len(traces), penalty), np.zeros(25),
                      jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("grouped independent GLM failed to converge: " + str(result.message))
    model = StructuredEventBaseline(kind="glm", penalty=penalty)
    model.coef = result.x
    model.fit_report = {"groups": len(traces), "design": rank_report(x, y, families),
        "loss": float(result.fun), "max_gradient": float(abs(result.jac).max()),
        "iterations": int(result.nit), "success": bool(result.success),
        "normalization": "shared factors once plus every actual continuation factor, divided by upstream groups",
        "authority_priors_fitted": False, "calibrated": False}
    return model


def after_review_dag(trace):
    """Actual reviewer records become observations; no publisher is exposed."""
    trace.validate()
    spec = trace.dag.specification()
    spec["observations"].update(copy.deepcopy(trace.shared_labels))
    spec["prefix_sequence"] = max(v["sequence"] for v in trace.shared_labels.values())
    # Existing declared routes are retained: observing a response is not proof
    # that a publisher has already received it. No new delivery is fabricated.
    return ObservableEventDAG(**spec)


def bn_joint(model, dag):
    law = model.event_law.conditional(dag, list(TARGETS), model.theta[:model.event_count])
    return np.exp(law["log_probabilities"])


def independent_joint(joint):
    marginal = np.asarray(joint, float) @ STATES
    return np.prod(np.where(STATES, marginal, 1-marginal), axis=1)


def outcome_scores(joint, bits):
    p, bits = np.asarray(joint, float), np.asarray(bits)
    if p.shape != (4,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-10, rtol=0):
        raise ValueError("normalized finite four-state joint required")
    if bits.shape != (2,) or not np.isin(bits, [0, 1]).all():
        raise ValueError("two actual binary outcomes required")
    index = int(2*bits[0] + bits[1])
    truth = np.eye(4)[index]
    zero = bool(p[index] == 0)
    return {"joint_brier": float(np.square(p-truth).sum()),
            "joint_nll": None if zero else float(-np.log(p[index])),
            "joint_nll_infinite": zero,
            "marginal_brier": float(np.square(p @ STATES-bits).mean()),
            "risk_brier": float((1-p[0]-int(bits.any()))**2)}


def mean_scores(scores):
    if not scores:
        return None
    result = {key: None if any(v[key] is None for v in scores) else float(np.mean([v[key] for v in scores])) for key in METRICS}
    result["joint_nll_infinite"] = any(v["joint_nll_infinite"] for v in scores)
    return result


def group_scores(joint, continuations):
    """One equal-weight score per upstream group, whatever its valid count."""
    if not continuations:
        raise ValueError("at least one complete continuation required for scoring")
    return mean_scores([outcome_scores(joint, [c[n]["value"] for n in TARGETS]) for c in continuations])


def coverage(groups):
    complete = sum(len(g["continuations"]) for g in groups)
    incomplete = sum(len(g["incomplete_continuations"]) for g in groups)
    return {"planned_groups": len(groups), "planned_continuations": 2*len(groups),
            "upstream_complete_groups": sum(g["upstream_complete"] for g in groups),
            "complete_groups": sum(g["complete"] for g in groups),
            "complete_continuations": complete, "incomplete_continuations": incomplete,
            "unattempted_continuations": 2*len(groups)-complete-incomplete,
            "eligible_training_groups": sum(g["eligible_for_grouped_fit"] for g in groups),
            "eligible_scoring_groups": sum(g["eligible_for_statistics"] for g in groups),
            "scored_continuations": sum(len(g["continuations"]) for g in groups if g["eligible_for_statistics"]),
            "identity_excluded_groups": sum(not g["response_identity_qualified"] for g in groups),
            "api_calls": sum(g["api_calls"] for g in groups),
            "failure_records": sum(len(g["failures"]) for g in groups)}


def checked_groups(verified):
    if verified.get("schema") != "verified-repeated-message-development-v1" or not verified.get("development_only"):
        raise ValueError("collector-verified development collection required")
    groups = verified["grouped_traces"]
    cases = {c["id"]: c for c in study.cases()}
    if len(groups) != 48 or len({g["id"] for g in groups}) != 48 or {g["id"] for g in groups} != set(cases):
        raise ValueError("all 48 planned groups, including failures, must be retained")
    traces = {}
    for group in groups:
        case = cases[group["id"]]
        if any(group[k] != case[k] for k in ("base_task", "layout", "repeat")):
            raise ValueError("fixed task/layout/repetition identity mismatch")
        for flag in ("eligible_for_grouped_fit", "eligible_for_statistics", "response_identity_qualified"):
            if type(group.get(flag)) is not bool:
                raise ValueError("explicit collector identity and eligibility verdict required")
        qualified = group["upstream_complete"] and bool(group["continuations"]) and group["response_identity_qualified"]
        if (group["eligible_for_grouped_fit"] or group["eligible_for_statistics"]) and not qualified:
            raise ValueError("unqualified raw records cannot enter training or scoring")
        if len(group["continuations"])+len(group["incomplete_continuations"]) > 2:
            raise ValueError("more continuations than the fixed collection protocol")
        if group["eligible_for_grouped_fit"] or group["eligible_for_statistics"]:
            traces[group["id"]] = GroupedEventTrace(group["dag"], group["shared_labels"], group["continuations"])
    return sorted(groups, key=lambda g: g["id"]), traces


def fixed_task_folds(groups):
    tasks = sorted(t["base_task"] for t in study.base_tasks())
    return [{"held_out_task": task,
             "train_ids": sorted(g["id"] for g in groups if g["base_task"] != task and g["eligible_for_grouped_fit"]),
             "score_ids": sorted(g["id"] for g in groups if g["base_task"] == task and g["eligible_for_statistics"])} for task in tasks]


def cue_coverage(train, score):
    def tokens(traces):
        return {(e["family"], tuple(e["cues"])) for t in traces for e in t.dag.events if e["id"] not in t.dag.observations}
    fitted, tested = tokens(train), tokens(score)
    return {"training_family_cues": [[f, list(c)] for f, c in sorted(fitted)],
            "unseen_test_family_cues": [[f, list(c)] for f, c in sorted(tested-fitted)],
            "scope": "Cue availability only; parent-state coverage requires the separately reported training design."}


def aggregate_predictions(rows):
    """Continuation mean -> upstream-group mean -> layout mean -> task mean."""
    task_layout = defaultdict(list)
    for row in rows:
        task_layout[row["base_task"], row["layout"]].append(row)
    blocks, task_metrics = [], defaultdict(list)
    for (task, layout), values in sorted(task_layout.items()):
        metrics = mean_scores([v["scores"] for v in values])
        task_metrics[task].append(metrics)
        blocks.append({"base_task": task, "layout": layout, "groups": len(values),
                       "continuations": sum(v["continuations"] for v in values), "scores": metrics})
    tasks = [{"base_task": task, "available_layouts": len(values), "scores": mean_scores(values)} for task, values in sorted(task_metrics.items())]
    return {"groups": len(rows), "continuations": sum(r["continuations"] for r in rows),
            "task_layout": blocks, "task_macro": tasks,
            "overall": mean_scores([v["scores"] for v in tasks]),
            "complete_task_layout_coverage": len(blocks) == 8,
            "weighting": "equal valid continuations within group; equal groups within task/layout; equal available layouts within task; equal available tasks",
            "missing_strata": [[t["base_task"], l] for t in study.base_tasks() for l in ("shared", "separate") if (t["base_task"], l) not in task_layout]}


def prediction_audit(predictions, restored_no_parent_joint, *, phase, layout):
    exact = predictions["exact_bn"]
    checks = {"bn_glm_joint": float(np.max(abs(exact-predictions["matched_glm"]))),
              "same_marginals": float(np.max(abs(exact @ STATES-predictions["same_marginals_independent"] @ STATES))),
              "no_parent_restore": float(np.max(abs(predictions["no_parent_retrained"]-restored_no_parent_joint)))}
    if phase == "after_review" or layout == "separate":
        checks["declared_independence"] = float(np.max(abs(exact-predictions["same_marginals_independent"])))
    for name, difference in checks.items():
        if not np.isfinite(difference) or difference > NUMERIC_TOLERANCES[name]:
            raise ValueError("numerical comparison audit failed: " + name)
    return checks


def exclusion_causes(group):
    result = []
    if not group["upstream_complete"]: result.append("upstream_incomplete")
    if not group["continuations"]: result.append("no_complete_continuation")
    if not group["response_identity_qualified"]: result.append("response_identity_unqualified")
    if not group["eligible_for_grouped_fit"]: result.append("collector_training_excluded")
    if not group["eligible_for_statistics"]: result.append("collector_statistics_excluded")
    return result


def evaluate_collection(verified, *, max_iter=500):
    groups, traces = checked_groups(verified)
    by_id = {g["id"]: g for g in groups}
    folds, rows, model_records, audits = fixed_task_folds(groups), [], [], []
    for fold in folds:
        training = [traces[n] for n in fold["train_ids"]]
        scoring = [traces[n] for n in fold["score_ids"]]
        fold["training_tasks"] = sorted({by_id[n]["base_task"] for n in fold["train_ids"]})
        fold["training_coverage"] = coverage([g for g in groups if g["base_task"] != fold["held_out_task"]])
        fold["held_out_coverage"] = coverage([g for g in groups if g["base_task"] == fold["held_out_task"]])
        fold["cue_coverage"] = cue_coverage(training, scoring)
        if not training or not scoring:
            fold["status"] = "not_evaluable_empty_training_or_scoring"
            continue
        x, y, families = independent_grouped_table(training)
        fold["design"] = rank_report(x, y, families)
        no_parent_x, _, _ = independent_grouped_table(training, relations=False)
        fold["no_parent_design"] = rank_report(no_parent_x, y, families)
        fold["status"] = "evaluated"
        for penalty in PENALTIES:
            exact = fit_grouped_event_traces(training, regularization=penalty, max_iter=max_iter)
            no_parent = fit_grouped_event_traces(training, regularization=penalty, max_iter=max_iter, model_class=_NoParentBN)
            glm = fit_matched_glm(training, penalty=penalty, max_iter=max_iter)
            if not exact.optimizer["success"] or not no_parent.optimizer["success"]:
                raise ValueError("a grouped BN did not converge; no partial leaderboard")
            if np.any(no_parent.theta[list(PARENT_COLUMNS)] != 0):
                raise ValueError("no-parent fit unexpectedly learned parent coefficients")
            restored_no_parent = DAGWorkflowRiskBN.restore(no_parent.parameters())
            model_records.append({"held_out_task": fold["held_out_task"], "penalty": penalty,
                "exact_bn": exact.parameters(), "no_parent_retrained": no_parent.parameters(),
                "matched_glm": glm.metadata(),
                "no_parent_definition": "all parent_mean/multiple_parents_all design columns zeroed before fitting; zero coefficients remain exactly zero",
                "bn_glm_max_coefficient_difference": float(np.max(abs(exact.theta[:25]-glm.coef)))})
            for identity in fold["score_ids"]:
                group, trace = by_id[identity], traces[identity]
                for phase, dag in (("early", trace.dag), ("after_review", after_review_dag(trace))):
                    joint = bn_joint(exact, dag)
                    predictions = {"exact_bn": joint, "same_marginals_independent": independent_joint(joint),
                                   "no_parent_retrained": bn_joint(no_parent, dag), "matched_glm": glm.predict_joint(dag, list(TARGETS))}
                    audits.append({"group_id": identity, "phase": phase, "penalty": penalty,
                                   "checks": prediction_audit(predictions, bn_joint(restored_no_parent, dag), phase=phase, layout=group["layout"])})
                    for method, p in predictions.items():
                        marginal = p @ STATES
                        rows.append({"group_id": identity, "base_task": group["base_task"], "layout": group["layout"],
                            "held_out_task": fold["held_out_task"], "penalty": penalty, "method": method, "phase": phase,
                            "input_sha256": dag.sha256, "continuations": len(trace.continuations),
                            "joint": p.tolist(), "marginals": marginal.tolist(), "risk_probability": float(1-p[0]),
                            "modeled_dependence": float(p[3]-np.prod(marginal)),
                            "scores": group_scores(p, trace.continuations),
                            "actual_outcomes": [[c[n]["value"] for n in TARGETS] for c in trace.continuations],
                            "actual_response_records": [[c[n]["record_id"] for n in TARGETS] for c in trace.continuations]})
    summaries = [{"penalty": p, "method": m, "phase": phase,
                  **aggregate_predictions([r for r in rows if (r["penalty"], r["method"], r["phase"]) == (p, m, phase)])}
                 for p in PENALTIES for m in METHODS for phase in PHASES]
    contrasts = []
    indexed = {(r["group_id"], r["penalty"], r["phase"], r["method"]): r for r in rows}
    for row in rows:
        if row["method"] == "exact_bn":
            for control in METHODS[1:]:
                other = indexed[row["group_id"], row["penalty"], row["phase"], control]
                contrasts.append({"group_id": row["group_id"], "base_task": row["base_task"], "layout": row["layout"],
                    "penalty": row["penalty"], "phase": row["phase"], "control": control,
                    "delta_bn_minus_control": {k: None if row["scores"][k] is None or other["scores"][k] is None
                                               else row["scores"][k]-other["scores"][k] for k in METRICS}})
    return {"schema": "repeated-message-development-loto-v1", "development_only": True,
        "prospective_predictions": 0, "old_test_read": False, "calibrated": False,
        "penalties": list(PENALTIES), "selected_penalty": None, "target_states": STATES.tolist(),
        "target": "Actual literal appearance in two publisher messages; OR is violation of the study's fixed no-release contract, not semantic leakage or actual harm.",
        "coverage": coverage(groups), "response_identity_audit": copy.deepcopy(verified.get("response_identity_audit")),
        "group_eligibility": [{k: copy.deepcopy(g[k]) for k in ("id", "base_task", "layout", "repeat", "complete", "upstream_complete",
                              "eligible_for_grouped_fit", "eligible_for_statistics", "response_identity_qualified", "failures")}
                              | {"response_identity_issues": copy.deepcopy(g.get("response_identity_issues", [])),
                                 "exclusion_causes": exclusion_causes(g), "complete_continuations": len(g["continuations"]),
                                 "incomplete_continuations": len(g["incomplete_continuations"])} for g in groups],
        "folds": folds, "models": model_records, "predictions": rows, "summaries": summaries, "paired_group_contrasts": contrasts,
        "numerical_audit": {"status": "passed" if audits else "not_evaluable", "tolerances": NUMERIC_TOLERANCES,
                            "rows": audits, "maximum_difference": {name: max((a["checks"][name] for a in audits if name in a["checks"]), default=None)
                                                                     for name in NUMERIC_TOLERANCES},
                            "comparison_counts": {name: sum(name in a["checks"] for a in audits) for name in NUMERIC_TOLERANCES}},
        "limitations": ["Four task-held-out DEVELOPMENT folds; not an independent test, deployment estimate, or causal comparison.",
            "Two continuations share upstream responses and never count as independent upstream groups.",
            "A product of matched conditional GLMs is itself a BN factorization; equality is expected, not evidence of BN superiority.",
            "The current graph makes publishers independent after all reviewer bits are observed. The same-marginal control should then equal exact BN.",
            "Complete-pair and response-identity qualification may select data; all failures and excluded-group denominators are retained.",
            "Unknown common factors and wording within a fixed literal state are not learned by these event features.",
            "No significance, confidence interval, uncertainty calibration or favorable positive-class coverage is asserted."]}
