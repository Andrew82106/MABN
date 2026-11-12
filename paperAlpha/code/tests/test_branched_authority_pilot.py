from collections import defaultdict
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/commands'), str(ROOT / 'code/src')]
from paperalpha_runtime.branched_authority_pilot import (
    ACTORS, TREATMENTS, configuration, monitor_configuration, tool_definitions, BranchedAuthorityEnvironment, stage_input)
from paperalpha_monitor.branched_authority_monitor import BranchedAuthorityMonitor, FEATURE_NAMES, visible_features
from paperalpha_monitor.shared_authority_joint_bn import SharedAuthorityJointBN
from collect_branched_authority_api import instructions


def models():
    # Explicit test coefficients/qualification, NOT fitted performance.
    model = SharedAuthorityJointBN()
    model.theta[model.prior_slice] = [logit(.2), logit(.9), logit(.5)]
    model.theta[-2:] = [1., 2.]
    reference, system = model.parameters(), model.parameters()
    reference["fitted_targets"], system["fitted_targets"] = ["reference_failure_count:1"], ["system_outcome"]
    return reference, system, {"feature_names": FEATURE_NAMES, "theta": [0.] * len(FEATURE_NAMES)}


def setup(layout="shared", treatment="normal", *, broken=False):
    config, saved = configuration("test", layout, treatment), defaultdict(list)
    monitor = BranchedAuthorityMonitor(monitor_configuration(config), *models())
    def observe(record):
        if broken:
            raise RuntimeError("test observer failure")
        return monitor.push(record)
    env = BranchedAuthorityEnvironment(config, observe=observe, save=lambda n, v: saved[n].append(v))
    return env, monitor, saved


def plan(env):
    children = {}
    env.invoke("planner", "read_authority", {}, "read-planner")
    for actor, assignment in env.config["assignments"].items():
        result = env.invoke("planner", "issue_delegation", {"recipient": actor,
            "parent_grant_id": assignment["parent"], "resource": assignment["resource"]}, "delegate-" + actor)
        children[actor] = result["record"]["payload"]["grant_id"]
    env.after_planner("Delegated both orders. Workers must verify current authority.")
    return children


def execute(env):
    children = plan(env)
    for actor, assignment in env.config["assignments"].items():
        env.invoke(actor, "read_authority", {}, "read-" + actor)
        env.invoke(actor, "simulate_payment", {"resource": assignment["resource"], "amount": "50", "grant_ids": [children[actor]]}, "pay-" + actor)


@pytest.mark.parametrize("layout,treatment", [(a, b) for a in ("shared", "separate") for b in TREATMENTS])
def test_paired_observation_views_and_independent_labels(layout, treatment):
    env, monitor, saved = setup(layout, treatment)
    execute(env)
    assert not env.observation_failures
    expected = 0 if treatment == "normal" else (2 if layout == "shared" or treatment == "revoke_both_stale" else 1)
    assert sum(x["reference_policy_violation"] for x in env.payments) == expected
    last = saved["online_predictions"][-1]["prediction"]["views"]
    full, partial = (last[k]["dispatches"][0] for k in ("complete", "root_updates_missing"))
    assert full["reference_two_failures"] == int(expected >= 2)
    assert full["reference_any_failure"] == int(expected > 0)
    assert full["visible_rule_two_failures"] == int(expected >= 2)
    assert partial["reference_two_failures"] == pytest.approx(.2 if layout == "shared" else .04)
    assert partial["reference_any_failure"] == pytest.approx(.2 if layout == "shared" else .36)
    assert partial["independent_action_two_failures"] == pytest.approx(.04)
    assert partial["shared_fact_count"] == (1 if layout == "shared" else 2)
    assert partial["evidence_insufficient"] and partial["visible_rule_two_failures"] is None
    assert all(v == pytest.approx(.2) for v in partial["action_marginals"].values())
    before = {r["sequence"]: r for r in saved["online_predictions"]}
    for outcome in env.payments:
        score = before[outcome["sequence"]]
        assert score["record_order"] < outcome["record_order"]
        assert score["recorded_monotonic_ns"] <= outcome["recorded_monotonic_ns"]
    assert "reference_policy_violation" not in json.dumps(saved["monitor_events"])


def test_hidden_treatment_is_not_monitor_configuration_and_no_label_enters_tools():
    a, b = configuration("same", "shared", "normal"), configuration("same", "shared", "revoke_a_stale")
    assert monitor_configuration(a) == monitor_configuration(b)
    with pytest.raises(ValueError, match="public fields"):
        BranchedAuthorityMonitor(a, *models())
    env, _, saved = setup(treatment="revoke_a_stale")
    execute(env)
    assert len(saved["injections"]) == 2
    assert all(any(r["kind"] == "revoke" for r in i["actual_records"]) for i in saved["injections"])
    assert all(not any(r["kind"] == "revoke" for r in i["delivered_records"]) for i in saved["injections"])


def test_graph_uses_actual_delegation_not_requested_layout():
    env, _, saved = setup(layout="shared")
    env.config["assignments"]["worker-b"]["parent"] = "root-b"
    execute(env)
    row = saved["online_predictions"][-1]["prediction"]["views"]["root_updates_missing"]["dispatches"][0]
    assert row["shared_fact_count"] == 2 and row["reference_two_failures"] == pytest.approx(.04)


def test_replay_exactly_matches_prefixes_without_effect_labels():
    env, _, saved = setup(treatment="revoke_a_stale")
    execute(env)
    monitor = BranchedAuthorityMonitor(monitor_configuration(env.config), *models())
    for event, prediction in zip(saved["monitor_events"], saved["online_predictions"]):
        assert monitor.push(event) == prediction["prediction"]


def test_omitted_truth_does_not_change_any_partial_score():
    rows = []
    for treatment in ("normal", "revoke_a_stale", "revoke_both_stale"):
        env, _, saved = setup(treatment=treatment)
        execute(env)
        rows.append(saved["online_predictions"][-1]["prediction"]["views"]["root_updates_missing"]["dispatches"][0])
    for key in ("model_point_probability", "graph_logistic_two_failures", "reference_any_failure", "reference_two_failures",
                "independent_action_any_failure", "independent_action_two_failures", "same_information_features"):
        assert rows[0][key] == rows[1][key] == rows[2][key]


def test_verifier_checks_raw_sdk_user_message_not_just_prompt_substring():
    from verify_branched_authority_api import initial_input_matches
    assert initial_input_matches("goal", "goal")
    assert initial_input_matches([{"role": "user", "content": "goal"}], "goal")
    assert not initial_input_matches([{"role": "user", "content": "goal secret label"}], "goal")
    assert not initial_input_matches([{"role": "assistant", "content": "goal"}], "goal")


def test_failed_monitor_is_passive_and_does_not_suppress_actions():
    env, _, _ = setup(broken=True)
    execute(env)
    assert env.observation_failures and len(env.payments) == 2


@pytest.mark.parametrize("actor,tool,args", [("worker-a", "issue_delegation", {}),
    ("planner", "simulate_payment", {}), ("planner", "read_authority", {"actor": "user"})])
def test_actor_and_tool_capabilities_cannot_be_forged(actor, tool, args):
    env, _, _ = setup()
    with pytest.raises(ValueError):
        env.invoke(actor, tool, args, "bad")
    assert len(env.world) == 2 and not env.payments


def test_static_window_is_opened_after_changes_and_rejects_later_mutation():
    env, monitor, _ = setup(treatment="revoke_both_stale")
    plan(env)
    for graph in monitor.graphs.values():
        assert graph.window.first_sequence > max(r["sequence"] for r in env.world)
    with pytest.raises(ValueError, match="static"):
        monitor.push({"kind": "revoke", "sequence": env.sequence + 1, "payload": {"grant_id": "root-a"}})


def test_fork_contexts_receive_only_public_task_and_actual_planner_text():
    config = configuration("context", "separate", "revoke_a_stale")
    for actor in ACTORS:
        text = stage_input(config, actor, "actual report")
        assert "revoke_a_stale" not in text and "revoke_a_stale" not in instructions(config, actor)
        if actor != "planner":
            assert "actual report" in text and "You are " + actor in text
    assert config["workflow"]["edges"] == [["planner", "worker-a"], ["planner", "worker-b"]]
    for actor in ACTORS:
        for definition in tool_definitions(actor):
            schema = definition["params_json_schema"]
            assert not schema["additionalProperties"] and set(schema["required"]) == set(schema["properties"])


def test_graph_baseline_features_equal_frozen_study_definition():
    from verify_shared_authority_bn import visible_features as original
    env, monitor, _ = setup()
    execute(env)
    for graph in monitor.graphs.values():
        np.testing.assert_array_equal(visible_features(graph), original(graph))


def test_monitor_child_accepts_only_public_configuration_and_matches_in_process(tmp_path):
    env, _, saved = setup()
    execute(env)
    (tmp_path / "config.json").write_text(json.dumps(monitor_configuration(env.config)), encoding="utf-8")
    for name, value in zip(("reference_fit.json", "system_fit.json", "logistic.json"), models()):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    run = subprocess.run([sys.executable, "-B", str(ROOT / 'code/commands/monitor_branched_authority.py'),
        "--config", str(tmp_path / "config.json"), "--models", str(tmp_path)],
        input="".join(json.dumps(r) + "\n" for r in saved["monitor_events"]), capture_output=True, text=True, timeout=40)
    assert run.returncode == 0 and not run.stderr
    output = [json.loads(line) for line in run.stdout.splitlines()]
    assert output[0]["ready"]
    assert [r["result"] for r in output[1:]] == [r["prediction"] for r in saved["online_predictions"]]
