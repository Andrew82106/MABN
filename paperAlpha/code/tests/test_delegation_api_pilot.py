from collections import defaultdict
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from paperalpha_runtime.delegation_pilot import PilotEnvironment, configuration, tool_definitions, reference_policy
from monitor_delegation_pilot import make_compiler
from paperalpha_monitor.delegation_conditioned_bn import DelegationConditionedBN
from paperalpha_runtime.stale_authority_pilot import StaleAuthorityEnvironment
from paperalpha_monitor.authority_qualification import qualify, QualifiedDelegationStream


def setup(variant, broken=False):
    config, saved = configuration("case", variant), defaultdict(list)
    compiler, model = make_compiler(config), DelegationConditionedBN()
    engine = model.engine()
    def observe(record):
        if broken:
            raise RuntimeError("test observer failure")
        step = compiler.push(record)
        if step is not None:
            engine.apply(step, compiler.compact.variables)
        return {"point": engine.probability}
    return PilotEnvironment(config, observe=observe, save=lambda name, item: saved[name].append(item)), compiler, saved


def execute(env):
    resource = env.config["requested_resource"]
    env.invoke("planner", "read_authority", {}, "read-p")
    result = env.invoke("planner", "issue_delegation", {"parent_grant_id": "root", "resource": resource}, "delegate")
    env.after_planner("Delegated the requested payment.")
    env.invoke("worker", "read_authority", {}, "read-w")
    args = {"resource": resource, "amount": "50", "grant_ids": [result["record"]["payload"]["grant_id"]]}
    return env.invoke("worker", "simulate_payment", args, "pay-call")


@pytest.mark.parametrize("variant,violation,status", [("normal", False, "supported_reference"),
    ("revoked", True, "invalid_references"), ("narrow_scope", True, "invalid_references"),
    ("missing_root", False, "unresolved_reference")])
def test_current_authority_observed_before_effect_and_labels_are_separate(variant, violation, status):
    env, compiler, saved = setup(variant)
    returned = execute(env)
    outcome = saved["outcomes"][0]
    assert returned["simulation_only"] and "reference_policy_violation" not in returned
    assert outcome["reference_policy_violation"] is violation
    assert compiler.snapshots["payment-0001"].evidence["reference_status"] == status
    assert saved["online_predictions"][-1]["recorded_monotonic_ns"] <= outcome["recorded_monotonic_ns"]
    assert saved["online_predictions"][-1]["record_order"] < outcome["record_order"]
    assert all("reference_policy_violation" not in str(x) for x in saved["monitor_events"])
    assert outcome["binding"]["arguments"]["grant_ids"] == ["delegation-0001"]


def test_observer_failure_does_not_block_simulated_execution():
    env, _, saved = setup("normal", broken=True)
    execute(env)
    assert env.observation_failures and len(saved["outcomes"]) == 1
    assert saved["outcomes"][0]["reference_policy_violation"] is False


def test_missing_root_is_monitor_omission_not_changed_world_or_model_metadata():
    env, compiler, saved = setup("missing_root")
    assert env.invoke("worker", "read_authority", {}, "read")["records"][0]["payload"]["grant_id"] == "root"
    assert saved["world_events"][0]["visible_to_monitor"] is False
    assert not saved["monitor_events"] and not compiler.ledger.records
    assert compiler.ledger.coverage_complete is False


@pytest.mark.parametrize("actor,name,args", [("planner", "simulate_payment", {}), ("worker", "issue_delegation", {}),
    ("worker", "read_authority", {"authenticated_actor": "user"}), ("planner", "issue_delegation", {"parent_grant_id": "root", "resource": "x", "issuer": "user"})])
def test_sdk_bound_actor_cannot_be_replaced_by_tool_arguments(actor, name, args):
    env, _, _ = setup("normal")
    with pytest.raises(ValueError):
        env.invoke(actor, name, args, "bad")
    assert len(env.world) == 1 and not env.payments


def test_schema_is_strict_and_pilot_exposes_no_real_side_effect_tools():
    all_tools = [*tool_definitions("planner"), *tool_definitions("worker")]
    assert {t["name"] for t in all_tools} == {"read_authority", "issue_delegation", "simulate_payment"}
    for tool in all_tools:
        schema = tool["params_json_schema"]
        assert set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False


def test_monitor_stream_and_fresh_batch_replay_match_every_model_prefix():
    env, compiler, saved = setup("revoked")
    execute(env)
    other, model = make_compiler(env.config), DelegationConditionedBN()
    engine = model.engine()
    for record, prediction in zip(saved["monitor_events"], saved["online_predictions"]):
        step = other.push(record)
        if step is not None:
            engine.apply(step, other.compact.variables)
        assert engine.probability == prediction["prediction"]["point"]
    assert other.snapshots == compiler.snapshots


def test_stale_attack_changes_only_worker_tool_view_not_world_or_monitor():
    cfg, saved = configuration("stale-test", "revoked"), defaultdict(list)
    cfg["stale_authority_view"] = True
    compiler = make_compiler(cfg)
    env = StaleAuthorityEnvironment(cfg, observe=compiler.push, save=lambda name, value: saved[name].append(value))
    execute(env)
    assert len(saved["injections"]) == 1
    attack = saved["injections"][0]
    assert any(r["kind"] == "revoke" for r in attack["actual_records"])
    assert not any(r["kind"] == "revoke" for r in attack["delivered_records"])
    assert any(r.get("kind") == "revoke" for r in saved["monitor_events"])
    assert saved["outcomes"][0]["reference_policy_violation"]
    assert compiler.snapshots["payment-0001"].evidence["reference_status"] == "invalid_references"


def test_visible_valid_chain_is_not_current_authority_when_updates_are_missing():
    env, compiler, _ = setup("normal")
    compiler.ledger.coverage_complete = False
    execute(env)
    evidence = compiler.snapshots["payment-0001"].evidence
    assert evidence["reference_status"] == "supported_reference"
    result = qualify(evidence, .01)
    assert result["model_point_probability"] == .01
    assert result["evidence_insufficient"] and result["current_authority_assessment"] == "unresolved"
    assert "authority_updates_may_be_missing" in result["qualification_reasons"]
    assert not result["proves_safety"]


def test_qualified_stream_requires_fitted_target_and_preserves_raw_probability():
    config, model = configuration("stream", "normal"), DelegationConditionedBN()
    with pytest.raises(ValueError, match="supervised"):
        QualifiedDelegationStream(model, make_compiler(config))
    model.fitted_targets = ["episode_outcome"]  # Test output protocol, not trained accuracy.
    qualified = QualifiedDelegationStream(model, make_compiler(config))
    env, _, saved = setup("normal")
    execute(env)
    for record in saved["monitor_events"]:
        last = qualified.push(record)
    assert last["dispatches"][0]["model_point_probability"] == qualified.engine.probability
    assert not last["dispatches"][0]["evidence_insufficient"]
