"""Optional all-observed training comparison; no new model or raw-data edits.

The original complete-pair scoring contract is unchanged. Missing, refused and
unattempted messages are not negative labels. Both supervision modes divide the
loss by all 36 planned training groups, including empty and identity-excluded
groups, so adding observed labels does not silently change the L2 scale.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize

from paperalpha_runtime import repeated_message_study as study
from .partial_grouped_event_learning import (PartialGroupedEventTrace,
                                            fit_partial_grouped_event_traces)
from .observable_event_dag import DAGWorkflowRiskBN
from .structured_event_baseline import StructuredEventBaseline, event_features, glm_objective, rank_report
from .repeated_message_evaluation import (PENALTIES, METHODS, PHASES, TARGETS, STATES,
    METRICS, PARENT_COLUMNS, NUMERIC_TOLERANCES, _NoParentBN, checked_groups, coverage,
    after_review_dag, bn_joint, independent_joint, group_scores, aggregate_predictions,
    prediction_audit, exclusion_causes, cue_coverage)

SUPERVISION_MODES = ("complete_pairs", "all_observed")
NORMALIZATION_GROUPS = 36


def continuation_slots(group):
    """Recover original indices, not complete-list + incomplete-list order."""
    complete, incomplete = group["continuations"], group["incomplete_continuations"]
    if not group["upstream_complete"]:
        if complete or incomplete:
            raise ValueError("upstream failure cannot have attempted continuations in this protocol")
        return [{}, {}]
    if len(complete) + len(incomplete) != study.CONTINUATIONS:
        raise ValueError("completed upstream requires both planned continuation outcomes")
    slots = [None] * study.CONTINUATIONS
    for record in incomplete:
        index = record.get("index")
        labels = record.get("labels")
        if (type(index) is not int or not 0 <= index < study.CONTINUATIONS
                or slots[index] is not None or record.get("completed") is not False
                or not isinstance(labels, dict) or not set(labels) < set(TARGETS)):
            raise ValueError("unique indexed incomplete continuation with partial labels required")
        slots[index] = copy.deepcopy(labels)
    available = [i for i, labels in enumerate(slots) if labels is None]
    for index, labels in zip(available, complete):
        if not isinstance(labels, dict) or set(labels) != set(TARGETS):
            raise ValueError("complete continuation must contain both actual publisher labels")
        slots[index] = copy.deepcopy(labels)
    return slots


def partial_trace(group, *, mode="all_observed"):
    """Use protocol-declared shared IDs even when their responses are missing."""
    if mode not in SUPERVISION_MODES:
        raise ValueError("known supervision mode required")
    shared_ids = study.reviewers(group)
    slots = continuation_slots(group)
    shared = group["shared_labels"]
    if mode == "complete_pairs":
        # Match the old admitted evidence, retaining empty slots and groups.
        if not group["eligible_for_grouped_fit"]:
            shared, slots = {}, [{}, {}]
        else:
            slots = [labels if set(labels) == set(TARGETS) else {} for labels in slots]
    return PartialGroupedEventTrace(group["dag"], shared_ids, shared, slots)


def independent_partial_table(traces, *, relations=True):
    """Independent GLM feature construction; never calls the BN design/table."""
    x, y, families, seen = [], [], [], set()
    for trace in traces:
        trace.validate()
        records = {v["record_id"] for b in [trace.shared_labels, *trace.continuations] for v in b.values()}
        if records & seen:
            raise ValueError("actual response records repeated across training groups")
        seen.update(records)
        prefix = {n: v["value"] for n, v in trace.dag.observations.items()}
        shared = {n: v["value"] for n, v in trace.shared_labels.items()}
        blocks = [(trace.shared_labels, {**prefix, **shared})]
        blocks += [(b, {**prefix, **shared, **{n: v["value"] for n, v in b.items()}})
                   for b in trace.continuations]
        for labels, values in blocks:
            for identity in trace.dag.order:
                if identity not in labels:
                    continue
                parents = trace.dag.parents[identity]
                if not set(parents) <= set(values):
                    raise ValueError("all observed training labels need observed ancestors")
                event = trace.dag.lookup[identity]
                x.append(event_features(event, [values[p] for p in parents], relations=relations))
                y.append(labels[identity]["value"])
                families.append(event["family"])
    return np.asarray(x, float).reshape((-1, 25)), np.asarray(y, float), families


def fit_partial_matched_glm(traces, *, penalty, normalization_groups=NORMALIZATION_GROUPS, max_iter=500):
    if type(normalization_groups) is not int or normalization_groups < max(1, len(traces)):
        raise ValueError("normalization must retain every planned training group")
    x, y, families = independent_partial_table(traces)
    if not len(y):
        raise ValueError("no observed labels: partial fit is not evaluable")
    result = minimize(lambda t: glm_objective(t, x, y, normalization_groups, penalty), np.zeros(25),
                      jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("independent partial GLM failed to converge: " + str(result.message))
    model = StructuredEventBaseline(kind="glm", penalty=penalty)
    model.coef = result.x
    model.fit_report = {"groups": len(traces), "normalization_groups": normalization_groups,
        "design": rank_report(x, y, families), "loss": float(result.fun),
        "max_gradient": float(abs(result.jac).max()), "iterations": int(result.nit),
        "success": bool(result.success), "authority_priors_fitted": False, "calibrated": False,
        "normalization": "observed shared factors once plus observed continuation factors, divided by planned groups"}
    return model


def supervision_coverage(groups, traces):
    """Original admission and new information counts are separate contracts."""
    chosen = [g for g in groups if g["id"] in traces]
    shared = sum(len(traces[g["id"]].shared_labels) for g in chosen)
    publishers = sum(len(c) for g in chosen for c in traces[g["id"]].continuations)
    partial_publishers = sum(len(c) for g in chosen for c in traces[g["id"]].continuations
                             if set(c) != set(TARGETS))
    counts = {g["id"]: len(traces[g["id"]].shared_labels)
              + sum(len(c) for c in traces[g["id"]].continuations) for g in chosen}
    labels = [v for g in chosen for b in [traces[g["id"]].shared_labels, *traces[g["id"]].continuations]
              for v in b.values()]
    return {**coverage(groups), "identity_qualified_input_groups": len(chosen),
        "informative_training_groups": sum(n > 0 for n in counts.values()),
        "empty_training_groups": sum(n == 0 for n in counts.values()),
        "informative_not_originally_eligible_groups": sum(counts[g["id"]] > 0 and not g["eligible_for_grouped_fit"] for g in chosen),
        "shared_only_groups": sum(bool(traces[g["id"]].shared_labels)
            and not any(traces[g["id"]].continuations) for g in chosen),
        "effective_labels": {"shared": shared, "publisher_complete_pairs": publishers-partial_publishers,
            "publisher_incomplete_pairs": partial_publishers, "publisher_total": publishers,
            "total": shared+publishers, "positive": sum(v["value"] for v in labels),
            "negative": sum(not v["value"] for v in labels),
            "unique_response_records": len({v["record_id"] for v in labels})},
        "original_eligibility_unchanged": True}


def _difference(left, right):
    return {k: None if left[k] is None or right[k] is None else left[k]-right[k] for k in METRICS}


def evaluate_partial_collection(verified, *, max_iter=500):
    groups, scoring_traces = checked_groups(verified)
    by_id = {g["id"]: g for g in groups}
    # Validate all available labels before the complete-pair mode removes any.
    all_traces = {g["id"]: partial_trace(g) for g in groups if g["response_identity_qualified"]}
    modes = {"all_observed": all_traces,
             "complete_pairs": {g["id"]: partial_trace(g, mode="complete_pairs")
                                for g in groups if g["response_identity_qualified"]}}
    folds, rows, models, audits = [], [], [], []
    for task in sorted(t["base_task"] for t in study.base_tasks()):
        planned_train = [g for g in groups if g["base_task"] != task]
        held_out = [g for g in groups if g["base_task"] == task]
        if len(planned_train) != NORMALIZATION_GROUPS:
            raise ValueError("fixed LOTO normalization requires 36 planned training groups")
        score_ids = [g["id"] for g in held_out if g["eligible_for_statistics"]]
        fold = {"held_out_task": task, "planned_train_ids": [g["id"] for g in planned_train],
            "score_ids": score_ids, "normalization_groups": NORMALIZATION_GROUPS,
            "training_tasks": sorted({g["base_task"] for g in planned_train}),
            "training_coverage": coverage(planned_train), "held_out_coverage": coverage(held_out),
            "supervision_modes": {}}
        folds.append(fold)
        for mode in SUPERVISION_MODES:
            training = [modes[mode][g["id"]] for g in planned_train if g["id"] in modes[mode]]
            mode_report = {"train_ids": [g["id"] for g in planned_train if g["id"] in modes[mode]],
                "normalization_groups": NORMALIZATION_GROUPS,
                "training_coverage": supervision_coverage(planned_train, modes[mode]),
                "cue_coverage": cue_coverage(training, [scoring_traces[n] for n in score_ids])}
            fold["supervision_modes"][mode] = mode_report
            x, y, families = independent_partial_table(training)
            if not len(y) or not score_ids:
                mode_report["status"] = "not_evaluable_no_information" if not len(y) else "not_evaluable_no_scoring_groups"
                continue
            mode_report["status"] = "evaluated"
            mode_report["design"] = rank_report(x, y, families)
            no_x, _, _ = independent_partial_table(training, relations=False)
            mode_report["no_parent_design"] = rank_report(no_x, y, families)
            for penalty in PENALTIES:
                options = dict(regularization=penalty, max_iter=max_iter, normalization_groups=NORMALIZATION_GROUPS)
                exact = fit_partial_grouped_event_traces(training, **options)
                no_parent = fit_partial_grouped_event_traces(training, model_class=_NoParentBN, **options)
                glm = fit_partial_matched_glm(training, penalty=penalty, max_iter=max_iter,
                                             normalization_groups=NORMALIZATION_GROUPS)
                if not exact.optimizer["success"] or not no_parent.optimizer["success"]:
                    raise ValueError("partial BN did not converge; no partial leaderboard")
                if np.any(no_parent.theta[list(PARENT_COLUMNS)] != 0):
                    raise ValueError("no-parent fit unexpectedly learned parent coefficients")
                restored = DAGWorkflowRiskBN.restore(no_parent.parameters())
                models.append({"held_out_task": task, "supervision_mode": mode, "penalty": penalty,
                    "normalization_groups": NORMALIZATION_GROUPS, "exact_bn": exact.parameters(),
                    "no_parent_retrained": no_parent.parameters(), "matched_glm": glm.metadata(),
                    "bn_glm_max_coefficient_difference": float(np.max(abs(exact.theta[:25]-glm.coef)))})
                for identity in score_ids:
                    group, trace = by_id[identity], scoring_traces[identity]
                    for phase, dag in (("early", trace.dag), ("after_review", after_review_dag(trace))):
                        joint = bn_joint(exact, dag)
                        predictions = {"exact_bn": joint, "same_marginals_independent": independent_joint(joint),
                            "no_parent_retrained": bn_joint(no_parent, dag), "matched_glm": glm.predict_joint(dag, list(TARGETS))}
                        audits.append({"group_id": identity, "phase": phase, "penalty": penalty, "supervision_mode": mode,
                            "checks": prediction_audit(predictions, bn_joint(restored, dag), phase=phase, layout=group["layout"])})
                        for method, p in predictions.items():
                            marginal = p @ STATES
                            rows.append({"group_id": identity, "base_task": group["base_task"], "layout": group["layout"],
                                "held_out_task": task, "supervision_mode": mode, "penalty": penalty, "method": method, "phase": phase,
                                "input_sha256": dag.sha256, "continuations": len(trace.continuations),
                                "joint": p.tolist(), "marginals": marginal.tolist(), "risk_probability": float(1-p[0]),
                                "modeled_dependence": float(p[3]-np.prod(marginal)), "scores": group_scores(p, trace.continuations),
                                "actual_outcomes": [[c[n]["value"] for n in TARGETS] for c in trace.continuations],
                                "actual_response_records": [[c[n]["record_id"] for n in TARGETS] for c in trace.continuations]})
    summaries = [{"supervision_mode": mode, "penalty": p, "method": method, "phase": phase,
                  **aggregate_predictions([r for r in rows if (r["supervision_mode"], r["penalty"], r["method"], r["phase"])
                                           == (mode, p, method, phase)])}
                 for mode in SUPERVISION_MODES for p in PENALTIES for method in METHODS for phase in PHASES]
    indexed = {(r["group_id"], r["supervision_mode"], r["penalty"], r["phase"], r["method"]): r for r in rows}
    method_contrasts, supervision_contrasts = [], []
    for row in rows:
        common = {k: row[k] for k in ("group_id", "base_task", "layout", "penalty", "phase")}
        if row["method"] == "exact_bn":
            for control in METHODS[1:]:
                other = indexed[row["group_id"], row["supervision_mode"], row["penalty"], row["phase"], control]
                method_contrasts.append({**common, "supervision_mode": row["supervision_mode"], "control": control,
                                        "delta_bn_minus_control": _difference(row["scores"], other["scores"])})
        if row["supervision_mode"] == "all_observed":
            other = indexed.get((row["group_id"], "complete_pairs", row["penalty"], row["phase"], row["method"]))
            if other is not None:
                if (row["input_sha256"], row["actual_response_records"], row["actual_outcomes"]) != (
                        other["input_sha256"], other["actual_response_records"], other["actual_outcomes"]):
                    raise ValueError("supervision comparisons must share predictions' inputs and actual complete-pair outcomes")
                supervision_contrasts.append({**common, "method": row["method"],
                    "delta_all_observed_minus_complete_pairs": _difference(row["scores"], other["scores"])})
    paired_summaries, comparison_coverage = [], []
    for penalty in PENALTIES:
        for method in METHODS:
            for phase in PHASES:
                selected = {mode: [r for r in rows if (r["supervision_mode"], r["penalty"], r["method"], r["phase"])
                                  == (mode, penalty, method, phase)] for mode in SUPERVISION_MODES}
                ids = {mode: {r["group_id"] for r in selected[mode]} for mode in SUPERVISION_MODES}
                common_ids = set.intersection(*ids.values())
                same = ids["complete_pairs"] == ids["all_observed"]
                dimensions = {"penalty": penalty, "method": method, "phase": phase}
                comparison_coverage.append({**dimensions, "same_scored_group_ids": same,
                    "direct_summary_comparison_valid": same and bool(common_ids),
                    "scored_groups_by_mode": {m: len(v) for m, v in ids.items()},
                    "common_score_ids": sorted(common_ids),
                    "unpaired_score_ids": {m: sorted(v-common_ids) for m, v in ids.items()}})
                for mode in SUPERVISION_MODES:
                    paired_summaries.append({**dimensions, "supervision_mode": mode,
                        **aggregate_predictions([r for r in selected[mode] if r["group_id"] in common_ids])})
    return {"schema": "partial-repeated-message-development-loto-v1", "development_only": True,
        "prospective_predictions": 0, "old_test_read": False, "calibrated": False,
        "supervision_modes": list(SUPERVISION_MODES), "normalization_groups_per_fold": NORMALIZATION_GROUPS,
        "penalties": list(PENALTIES), "selected_penalty": None, "selected_supervision_mode": None,
        "target_states": STATES.tolist(), "model_expressiveness_changed": False,
        "target": "Compose locally fitted literal-appearance laws and score on the collector's complete-pair selected sample; OR denotes the study's fixed no-release contract, not semantic leakage or unconditional runtime risk.",
        "coverage": coverage(groups), "supervision_coverage": {m: supervision_coverage(groups, modes[m]) for m in SUPERVISION_MODES},
        "response_identity_audit": copy.deepcopy(verified.get("response_identity_audit")),
        "group_eligibility": [{k: copy.deepcopy(g[k]) for k in ("id", "base_task", "layout", "repeat", "complete", "upstream_complete",
            "eligible_for_grouped_fit", "eligible_for_statistics", "response_identity_qualified", "failures")}
            | {"response_identity_issues": copy.deepcopy(g.get("response_identity_issues", [])),
               "exclusion_causes": exclusion_causes(g),
               "partial_training_informative": g["id"] in all_traces and bool(all_traces[g["id"]].shared_labels
                    or any(all_traces[g["id"]].continuations))} for g in groups],
        "folds": folds, "models": models, "predictions": rows, "summaries": summaries,
        "paired_group_contrasts": method_contrasts, "paired_supervision_contrasts": supervision_contrasts,
        "supervision_comparison_coverage": comparison_coverage,
        "paired_supervision_summaries": paired_summaries,
        "numerical_audit": {"status": "passed" if audits else "not_evaluable", "tolerances": NUMERIC_TOLERANCES,
            "rows": audits, "maximum_difference": {n: max((a["checks"][n] for a in audits if n in a["checks"]), default=None)
                                                         for n in NUMERIC_TOLERANCES},
            "comparison_counts": {n: sum(n in a["checks"] for a in audits) for n in NUMERIC_TOLERANCES}},
        "limitations": ["Four task-held-out DEVELOPMENT folds, not an independent test, prospective performance or deployment estimate.",
            "Both supervision modes use the same 36 planned-group denominator, not the number of surviving groups or labels.",
            "The complete_pairs mode is a newly denominator-matched control, not a numerical replay of the frozen evaluator that divided by surviving eligible groups.",
            "A mode with no training information is not fitted. Unpaired summaries may then cover different groups; compare supervision modes only on the explicitly reported common-group paired summaries.",
            "Only complete publisher pairs in originally statistics-qualified groups are scored; partial pairs never supply invented truth.",
            "Local factors fit usable reply content. Their composed joint is not automatically P(A,B | entire workflow completed), because completion may depend on reviewer content; nor is it unconditional runtime risk. Such interpretations require additional ignorable-selection and correctly specified local-model assumptions.",
            "Reported errors are development errors on the original complete-pair selected sample. Refusal, transport failure and unattempted calls are not zero labels or a learned completion model.",
            "Ignoring missing factors assumes missingness is ignorable for content parameters conditional on modeled parents/prefix. Content-dependent refusal can violate this; shared observability alone does not remove selection bias.",
            "Only ancestor-closed observed factors are accepted; an observed child with an unobserved parent is rejected, not silently imputed.",
            "Shared factors count once; each actual continuation factor counts once. Unique response IDs do not establish statistical independence.",
            "More observed supervision does not enlarge the 25-coefficient event model or establish improvement. Unknown common wording within a literal state is still omitted.",
            "Matched GLM has the same conditional law; equality is expected, not BN superiority. Publishers become independent after reviewer bits are observed under the declared graph.",
            "No significance, confidence interval, calibration or overall superiority is asserted."]}
