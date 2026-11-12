"""Post-fit diagnostic: equal terminal predictions can conceal flipped meanings.

This targeted symmetry only applies to this pilot's review->publish topology.
It changes a diagnostic parameter copy, never a frozen model or runtime event.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def flip_review_representation(model, dags):
    for dag in dags:
        for node in dag.order:
            event = dag.lookup[node]
            if event["family"] == "review":
                if node in dag.observations or any(dag.lookup[p]["family"] == "review" for p in dag.parents[node]):
                    raise ValueError("symmetry requires unobserved single-stage reviewers")
            elif node not in dag.observations:
                if event["family"] != "publish" or not dag.parents[node] or any(dag.lookup[p]["family"] != "review" for p in dag.parents[node]):
                    raise ValueError("only future publication children of reviewers supported")
    other = copy.deepcopy(model)
    names = other.names
    if abs(other.theta[names.index("event:publish:multiple_parents_all")]) > 1e-14:
        raise ValueError("nonlinear all-parent term is outside this exact transformation")
    for i, name in enumerate(names):
        if name.startswith("event:review:"):
            other.theta[i] *= -1
    weight = names.index("event:publish:parent_mean")
    other.theta[names.index("event:publish:bias")] += other.theta[weight]
    other.theta[weight] *= -1
    return other


def run(out, destination):
    live_root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(out / "source/code/src"), str(out / "source/code/commands")]
    from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN
    from paperalpha_monitor.observable_event_learning import trace_loss_gradient
    # This diagnostic helper is new after collection; record its exact bytes
    # independently. All model/data code continues to come from the old freeze.
    helper_path = Path(__file__).resolve().parent / "QUALIFICATION_HELPER.py"
    if not helper_path.exists():
        helper_path = live_root / 'code/src/paperalpha_monitor/observable_event_qualification.py'
    spec = importlib.util.spec_from_file_location("paperalpha_monitor.event_meaning_audit_helper", helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    spec = importlib.util.spec_from_file_location("frozen_relation_study", out / "source/code/commands/run_relational_forecast_study.py")
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    data, verification = frozen.verified_data(out)
    fit_report = frozen.read(out / "fit/REPORT.json")
    result = {"scope": "Post-fit development diagnostic; not a new candidate selection or independent test.",
        "coverage": {}, "models": {}, "representation_flip": {}, "collection": verification}
    all_early = [row["samples"]["early"].event_dag for row in data]
    for split in ("train", "validation"):
        rows = [row for row in data if row["case"]["split"] == split]
        result["coverage"][split] = helper.binary_target_coverage([int(any(row["labels"][n]["value"] for n in ("publish-a", "publish-b"))) for row in rows])
    for name, expected in fit_report["models_sha256"].items():
        if not name.startswith("dag_"):
            continue
        path = out / "fit/models" / name
        assert frozen.sha(path) == expected
        model = DAGWorkflowRiskBN.restore(frozen.read(path))
        flipped = flip_review_representation(model, all_early)
        name = name[:-5]
        result["models"][name] = {}
        for split in ("train", "validation"):
            rows = [row for row in data if row["case"]["split"] == split]
            dags = [row["samples"]["early"].event_dag for row in rows]
            labels = [row["labels"] for row in rows]
            result["models"][name][split] = helper.intermediate_semantics(model, dags, labels, [["publish-a", "publish-b"]] * len(rows))
        maximum_terminal_error, maximum_complement_error, late_change = 0., 0., 0.
        for row in data:
            dag = row["samples"]["early"].event_dag
            def probs(m, d, targets):
                return np.exp(m.event_law.conditional(d, targets, m.theta[:25])["log_probabilities"])
            maximum_terminal_error = max(maximum_terminal_error, float(np.max(np.abs(probs(model, dag, ["publish-a", "publish-b"]) - probs(flipped, dag, ["publish-a", "publish-b"])))))
            for node in dag.ids:
                if dag.lookup[node]["family"] == "review":
                    maximum_complement_error = max(maximum_complement_error, abs(float(probs(model, dag, [node])[1] + probs(flipped, dag, [node])[1] - 1)))
            late = row["samples"]["late"].event_dag
            late_change = max(late_change, float(np.max(np.abs(probs(model, late, ["publish-a", "publish-b"]) - probs(flipped, late, ["publish-a", "publish-b"])))))
        train = [row for row in data if row["case"]["split"] == "train"]
        dags = [row["samples"]["early"].event_dag for row in train]
        terminal = [{n: row["labels"][n] for n in ("publish-a", "publish-b")} for row in train]
        complete = [row["labels"] for row in train]
        losses = {kind: [trace_loss_gradient(m.event_law, m.theta[:25], dags, labels, 0.)[0] for m in (model, flipped)]
            for kind, labels in (("terminal_unregularized_nll", terminal), ("complete_trace_unregularized_nll", complete))}
        assert maximum_terminal_error < 1e-12 and maximum_complement_error < 1e-12
        assert abs(losses["terminal_unregularized_nll"][0] - losses["terminal_unregularized_nll"][1]) < 1e-12
        result["representation_flip"][name] = {"maximum_terminal_probability_change_early": maximum_terminal_error,
            "maximum_review_complement_error": maximum_complement_error, "maximum_terminal_probability_change_late": late_change,
            **losses, "original_parameters": model.theta.tolist(), "diagnostic_flipped_parameters": flipped.theta.tolist(),
            "interpretation": "Unregularized terminal likelihood cannot distinguish X from 1-X with the adjusted child CPT. L2 penalties may differ and choose an orientation without proving its observable meaning. Actual intermediate labels distinguish them."}
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), destination / "AUDITOR.py")
    shutil.copyfile(helper_path, destination / "QUALIFICATION_HELPER.py")
    result.update({"new_api_calls": 0, "new_fit": False, "frozen_models_changed": False,
        "source_sha256": {n: frozen.sha(destination / n) for n in ("AUDITOR.py", "QUALIFICATION_HELPER.py")},
        "plan_sha256": frozen.sha(out / "PLAN.json"), "fit_report_sha256": frozen.sha(out / "fit/REPORT.json")})
    frozen.write(destination / "REPORT.json", result)
    print(json.dumps({"coverage": result["coverage"], "intermediate_scores": {n: {s: v["by_family"] for s, v in r.items()} for n, r in result["models"].items()},
        "early_equivalence_maximum_error": max(v["maximum_terminal_probability_change_early"] for v in result["representation_flip"].values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve(), args.destination.resolve())
