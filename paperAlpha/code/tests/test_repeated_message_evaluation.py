"""Constructed offline protocol tests; no API data or quality claims."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import repeated_message_study as study
from paperalpha_monitor.grouped_event_learning import (GroupedEventTrace, grouped_complete_design,
                                                      grouped_loss_gradient, fit_grouped_event_traces)
from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN
from paperalpha_monitor.repeated_message_evaluation import (PENALTIES, PARENT_COLUMNS,
    independent_grouped_table, fit_matched_glm, after_review_dag, bn_joint, independent_joint,
    group_scores, outcome_scores, aggregate_predictions, checked_groups, fixed_task_folds, evaluate_collection, prediction_audit)
from paperalpha_monitor.structured_event_baseline import glm_objective
import evaluate_repeated_messages as runner


def cohort():
    groups = []
    for case in study.cases():
        source = {"record_id": case["id"]+":source", "sequence": 1, "text": case["source"]}
        packet = study.prefix(case, {"source": source}, 2)
        dag = study.compile_dag(packet)
        sequence = 2

        def label(name, value):
            nonlocal sequence
            sequence += 2
            return {"record_id": case["id"]+":"+name, "sequence": sequence,
                    "text": "Fixture " + (case["marker"] if value else "reference removed"),
                    "source": "constructed-not-api", "kind": "complete_message", "value": int(value)}

        shared = {n: label(n, (case["repeat"]+i) % 2) for i, n in enumerate(study.reviewers(case))}
        continuations = [{n: label(str(k)+":"+n, (case["repeat"]+k+i) % 2)
                          for i, n in enumerate(("publish-a", "publish-b"))} for k in range(2)]
        groups.append({"id": case["id"], "base_task": case["base_task"], "layout": case["layout"], "repeat": case["repeat"],
            "dag": dag, "shared_labels": shared, "continuations": continuations, "complete": True,
            "upstream_complete": True, "api_calls": study.planned_calls(case),
            "eligible_for_grouped_fit": True, "eligible_for_statistics": True,
            "response_identity_qualified": True, "response_identity_issues": [],
            "failures": [], "incomplete_continuations": []})
    return {"schema": "verified-repeated-message-development-v1", "development_only": True,
            "grouped_traces": groups, "response_identity_audit": {"fixture_only": True}}


def traces_for_test():
    groups, traces = checked_groups(cohort())
    return [traces[g["id"]] for g in groups[:6]]


def test_early_shared_correlation_and_observed_review_independence():
    groups, traces = checked_groups(cohort())
    model = DAGWorkflowRiskBN()
    model.theta[15] = -3
    model.theta[18] = 6
    for layout in ("shared", "separate"):
        group = next(g for g in groups if g["layout"] == layout)
        trace = traces[group["id"]]
        early = bn_joint(model, trace.dag)
        if layout == "shared":
            assert early[3]-(early @ np.asarray(((0,0),(0,1),(1,0),(1,1)))).prod() > .1
        else:
            assert np.allclose(early, independent_joint(early), atol=1e-12)
        late_dag = after_review_dag(trace)
        assert not any(n.startswith("publish-") for n in late_dag.observations)
        assert late_dag.observations["source"] == trace.dag.observations["source"]
        for n, observation in trace.shared_labels.items():
            assert late_dag.observations[n] == observation
        late = bn_joint(model, late_dag)
        assert np.allclose(late, independent_joint(late), atol=1e-12)


@pytest.mark.parametrize("penalty", PENALTIES)
def test_grouped_glm_independently_matches_bn_and_normalization(penalty):
    traces = traces_for_test()
    x, y, _ = independent_grouped_table(traces)
    law = DAGWorkflowRiskBN().event_law
    a, b, audit = grouped_complete_design(law, traces)
    assert np.array_equal(x, a) and np.array_equal(y, b)
    theta = np.random.default_rng(292).normal(0, .2, 25)
    actual = glm_objective(theta, x, y, len(traces), penalty)
    expected = grouped_loss_gradient(law, theta, traces, penalty)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert np.allclose(actual[1], expected[1], atol=1e-12)
    zero = glm_objective(np.zeros(25), x, y, len(traces), penalty)[0]
    assert zero == pytest.approx(len(y)*np.log(2)/len(traces))
    # Rows or continuation count would use a materially different L2 scale.
    assert zero != pytest.approx(np.log(2))
    assert audit["continuations"] == 2*len(traces)
    bn = fit_grouped_event_traces(traces, regularization=penalty)
    glm = fit_matched_glm(traces, penalty=penalty)
    assert np.max(abs(bn.theta[:25]-glm.coef)) < 1e-7
    for trace in traces:
        for dag in (trace.dag, after_review_dag(trace)):
            assert np.allclose(bn_joint(bn, dag), glm.predict_joint(dag, ["publish-a", "publish-b"]), atol=1e-9)


def test_four_folds_keep_all_layouts_repetitions_and_continuations_together():
    groups, _ = checked_groups(cohort())
    folds = fixed_task_folds(groups)
    assert len(folds) == 4
    seen = []
    for fold in folds:
        assert len(fold["train_ids"]) == 36 and len(fold["score_ids"]) == 12
        assert not set(fold["train_ids"]) & set(fold["score_ids"])
        held = [g for g in groups if g["id"] in fold["score_ids"]]
        assert {g["base_task"] for g in held} == {fold["held_out_task"]}
        assert {g["repeat"] for g in held} == set(range(6))
        assert {g["layout"] for g in held} == {"shared", "separate"}
        seen.extend(fold["score_ids"])
    assert len(seen) == len(set(seen)) == 48


def test_repeated_continuations_do_not_double_group_score_or_weight():
    trace = traces_for_test()[0]
    joint = np.asarray((.1, .2, .3, .4))
    assert group_scores(joint, trace.continuations) == group_scores(joint, trace.continuations*2)
    low = outcome_scores((1., 0., 0., 0.), (0, 0))
    high = outcome_scores((1., 0., 0., 0.), (1, 1))
    rows = [{"base_task": "equipment_collection", "layout": "shared", "continuations": 1, "scores": low},
            {"base_task": "equipment_collection", "layout": "shared", "continuations": 2, "scores": high}]
    result = aggregate_predictions(rows)
    assert result["overall"]["risk_brier"] == .5
    assert result["overall"]["joint_nll"] is None and result["overall"]["joint_nll_infinite"]
    assert result["groups"] == 2 and result["continuations"] == 3
    assert not result["complete_task_layout_coverage"]


def with_failures():
    result = cohort()
    groups = result["grouped_traces"]
    first = groups[0]
    first.update(upstream_complete=False, complete=False, shared_labels={}, continuations=[],
                 failures=["fixture-transport-failure"], api_calls=1,
                 eligible_for_grouped_fit=False, eligible_for_statistics=False)
    second = groups[1]
    missed = second["continuations"].pop(0)
    second.update(complete=False, failures=["fixture-publisher-failure"],
                  incomplete_continuations=[{"index": 0, "completed": False, "labels": {"publish-b": missed["publish-b"]}}])
    groups[2].update(response_identity_qualified=False, response_identity_issues=["fixture-duplicate-provider-id"],
                     eligible_for_grouped_fit=False, eligible_for_statistics=False)
    return result


@pytest.fixture(scope="module")
def evaluated():
    return evaluate_collection(with_failures())


def test_complete_development_evaluation_retains_failure_denominators(evaluated):
    report = evaluated
    assert report["development_only"] and report["selected_penalty"] is None
    assert report["prospective_predictions"] == 0 and not report["calibrated"]
    assert len(report["models"]) == 8 and len(report["summaries"]) == 16
    c = report["coverage"]
    assert c["planned_groups"] == 48 and c["planned_continuations"] == 96
    assert c["complete_continuations"] == 93 and c["incomplete_continuations"] == 1
    assert c["unattempted_continuations"] == 2 and c["identity_excluded_groups"] == 1
    assert c["eligible_training_groups"] == c["eligible_scoring_groups"] == 46
    assert c["scored_continuations"] == 91
    assert len(report["predictions"]) == 46*2*2*4
    assert len(report["paired_group_contrasts"]) == 46*2*2*3
    assert any(f["cue_coverage"]["unseen_test_family_cues"] for f in report["folds"])
    assert all(f["design"]["rank"] <= 25 for f in report["folds"])
    for record in report["models"]:
        assert record["bn_glm_max_coefficient_difference"] < 1e-7
        assert not np.any(np.asarray(record["no_parent_retrained"]["theta"])[list(PARENT_COLUMNS)])


def test_evaluation_cannot_leak_between_tasks_and_expected_controls_equal(evaluated):
    report = evaluated
    for fold in report["folds"]:
        assert fold["held_out_task"] not in fold["training_tasks"]
        assert not set(fold["train_ids"]) & set(fold["score_ids"])
    for contrast in report["paired_group_contrasts"]:
        if contrast["control"] == "matched_glm" or (contrast["phase"] == "after_review" and contrast["control"] == "same_marginals_independent"):
            assert all(abs(v) < 1e-8 for v in contrast["delta_bn_minus_control"].values() if v is not None)
    assert all(s["groups"] == 46 and s["continuations"] == 91 for s in report["summaries"])
    assert report["numerical_audit"]["status"] == "passed"
    assert report["numerical_audit"]["comparison_counts"]["bn_glm_joint"] == 46*2*2
    assert all(v < 1e-8 for v in report["numerical_audit"]["maximum_difference"].values())
    assert any("response_identity_unqualified" in row["exclusion_causes"] for row in report["group_eligibility"])
    json.dumps(report, allow_nan=False)


def test_no_parent_export_restores_equivalent_original_law(evaluated):
    _, traces = checked_groups(with_failures())
    row = next(r for r in evaluated["predictions"] if r["method"] == "no_parent_retrained")
    saved = next(m["no_parent_retrained"] for m in evaluated["models"]
                 if m["held_out_task"] == row["held_out_task"] and m["penalty"] == row["penalty"])
    restored = DAGWorkflowRiskBN.restore(saved)
    trace = traces[row["group_id"]]
    assert not np.any(restored.theta[list(PARENT_COLUMNS)])
    assert np.allclose(bn_joint(restored, trace.dag), row["joint"], atol=1e-12)


@pytest.mark.parametrize("case", ["glm", "late_independence", "separate_independence", "restore"])
def test_numerical_control_disagreement_fails_not_just_a_warning(case):
    correlated = np.asarray((.4, .1, .1, .4))
    predictions = {"exact_bn": correlated, "same_marginals_independent": np.full(4, .25),
                   "matched_glm": correlated.copy(), "no_parent_retrained": np.full(4, .25)}
    restored = np.full(4, .25)
    phase, layout = "early", "shared"
    if case == "glm": predictions["matched_glm"] = np.full(4, .25)
    elif case == "late_independence": phase = "after_review"
    elif case == "separate_independence": layout = "separate"
    else: restored = correlated
    with pytest.raises(ValueError, match="audit failed"):
        prediction_audit(predictions, restored, phase=phase, layout=layout)


@pytest.mark.parametrize("mutation", ["missing_group", "duplicate_id", "missing_eligibility", "unqualified_fit", "unverified"])
def test_incomplete_unqualified_or_unverified_cohort_rejected(mutation):
    value = cohort()
    if mutation == "missing_group": value["grouped_traces"].pop()
    elif mutation == "duplicate_id": value["grouped_traces"][0] = copy.deepcopy(value["grouped_traces"][1])
    elif mutation == "missing_eligibility": value["grouped_traces"][0].pop("eligible_for_statistics")
    elif mutation == "unqualified_fit": value["grouped_traces"][0]["response_identity_qualified"] = False
    else: value["schema"] = "unverified"
    with pytest.raises(ValueError): checked_groups(value)


def test_all_failed_keeps_fixed_folds_without_fabricating_scores():
    value = cohort()
    for group in value["grouped_traces"]:
        group.update(upstream_complete=False, complete=False, shared_labels={}, continuations=[],
                     eligible_for_grouped_fit=False, eligible_for_statistics=False, api_calls=1, failures=["fixture"])
    report = evaluate_collection(value)
    assert len(report["folds"]) == 4 and not report["models"] and not report["predictions"]
    assert all(s["overall"] is None for s in report["summaries"])
    assert report["coverage"]["unattempted_continuations"] == 96


def frozen_runner_fixture(tmp_path):
    raw = tmp_path/"raw"
    frozen = raw/"source"
    files = [*sorted((ROOT / 'code/src').rglob("*.py")), ROOT / 'code/commands/collect_repeated_message_study.py', ROOT / 'code/commands/evaluate_repeated_messages.py']
    hashes = {}
    for file in files:
        name = file.relative_to(ROOT)
        target = frozen/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        hashes[name.as_posix()] = runner.sha(target)
    (raw/"PLAN.json").write_text(json.dumps({"source_sha256": hashes}), encoding="utf-8")
    (raw/"COLLECTION.json").write_text('{"fixture":true}', encoding="utf-8")
    return raw, frozen


def test_cli_uses_verifier_twice_preserves_inputs_and_refuses_overwrite(tmp_path, monkeypatch, evaluated):
    raw, frozen = frozen_runner_fixture(tmp_path)
    monkeypatch.setattr(runner, "ROOT", frozen)
    before = {p.relative_to(raw).as_posix(): p.read_bytes() for p in raw.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(runner, "verify", lambda path: calls.append(path) or with_failures())
    monkeypatch.setattr(runner, "evaluate_collection", lambda value, **kw: copy.deepcopy(evaluated))
    out = tmp_path/"new-evaluation"
    report = runner.evaluate(raw, out)
    assert len(calls) == 2 and report["development_only"]
    assert {p.relative_to(raw).as_posix(): p.read_bytes() for p in raw.rglob("*") if p.is_file()} == before
    saved = json.loads((out/"REPORT.json").read_text(encoding="utf-8"))
    assert saved["execution"]["raw_collection"] == str(raw.resolve())
    environment = saved["execution"]["fit_environment"]
    assert environment["sys_version"] == sys.version
    assert environment["sys_executable_sha256"] == runner.sha(sys.executable)
    assert set(environment["packages"]) == {"numpy", "scipy", "scikit-learn"}
    assert saved["execution"]["source_before"]["source_sha256"] == saved["execution"]["source_after"]["source_sha256"]
    with pytest.raises(ValueError, match="new directory"): runner.evaluate(raw, out)
    with pytest.raises(ValueError, match="separate"): runner.evaluate(raw, raw/"forbidden-output")


@pytest.mark.parametrize("mutation", ["outside_runner", "changed_source", "missing_script", "extra_source"])
def test_actual_source_must_match_declared_snapshot(tmp_path, monkeypatch, mutation):
    raw, frozen = frozen_runner_fixture(tmp_path)
    if mutation != "outside_runner": monkeypatch.setattr(runner, "ROOT", frozen)
    if mutation == "changed_source":
        path = frozen/'code/src/paperalpha_monitor/repeated_message_evaluation.py'
        path.write_text(path.read_text(encoding="utf-8")+"\n# changed\n", encoding="utf-8")
    elif mutation == "missing_script":
        plan = json.loads((raw/"PLAN.json").read_text(encoding="utf-8"))
        plan["source_sha256"].pop('code/commands/evaluate_repeated_messages.py')
        (raw/"PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    elif mutation == "extra_source":
        (frozen/'code/src/unplanned.py').write_text("# fixture\n", encoding="utf-8")
    with pytest.raises(ValueError): runner.source_evidence(raw)
