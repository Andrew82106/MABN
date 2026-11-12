"""Only synthetic fixtures and static author schemas; no collected outputs/API."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/commands'), str(ROOT / 'code/src')]
import evaluate_paired_policy_executor as policy
from paperalpha_runtime.harnessaudit_guard_policy import compile_guard_policy, read_public_recognizers
from test_replay_paired_quadsentinel import public_artifact, compile_case, no_network


def bundle(kind, match):
    return compile_guard_policy([{"role": "worker", "access_type": kind, "match": match}],
        domain="ecommerce", recognizer_patterns={"email": [r"[a-z]+@example\.org"]},
        source={"rules": "synthetic", "recognizers": "synthetic"})


@pytest.mark.parametrize("expression", ["__import__('os').getcwd()", "a[0]", "a == b", "True", "a if b else a", "a AND missing"])
def test_only_boolean_ast_no_execution_or_short_circuit_bypass(expression):
    with pytest.raises((ValueError, SyntaxError)):
        policy.boolean_logic(expression, {"a": False, "b": True})


def test_boolean_ast_truth_and_unknown_values():
    assert policy.boolean_logic("NOT (a AND b) OR c", {"a": True, "b": True, "c": False}) is False
    assert policy.boolean_logic("NOT (a AND b) OR c", {"a": True, "b": False, "c": False}) is True
    with pytest.raises(ValueError): policy.boolean_logic("a", {"a": 1})


@pytest.mark.parametrize("allowed,args,expected", [
    (None, {}, False), ({}, {"id": "wrong"}, True), ({"id": "A"}, {}, True),
    ({"id": "A"}, {"id": None}, True), ({"id": "A"}, {"id": "A"}, True),
    ({"id": "A"}, {"id": ""}, False), ({"id": "A"}, {"id": "B"}, False),
    ({"id": "0"}, {"id": 0}, True), ({"id": "False"}, {"id": False}, True),
    ({"id": "PF-*"}, {"id": "PF-1"}, True), ({"path": 'code/src/*'}, {"path": 'code/src/a/b'}, True),
    ({"path": 'code/src/*'}, {"path": "src-other/a"}, False)])
def test_protected_argument_semantics_and_field_evidence(allowed, args, expected):
    compiled = bundle("operation.resource", {"tool": "pay", "allowed_args": allowed})
    result = policy.decide(compiled, {"agent_role": "worker", "surface": "tool_call", "tool_name": "pay", "tool_args": args})
    assert result["allowed"] is expected
    if not expected:
        assert result["matched_rules"][0]["violation_code"] == "V-OR"
        assert result["matched_rules"][0]["raw_rule_sha256"] == compiled["rule_map"][0]["raw_rule_sha256"]
        assert result["matched_rules"][0]["primitive_evidence"]


def test_tool_role_communication_and_current_payload_are_distinct():
    compiled = bundle("operation.tool_call", {"tool": "pay"})
    event = {"agent_role": "worker", "surface": "tool_call", "tool_name": "pay"}
    assert not policy.decide(compiled, event)["allowed"]
    assert policy.decide(compiled, {**event, "agent_role": "other"})["allowed"]
    communication = bundle("information.communication", {"recipient": "hub*"})
    assert not policy.decide(communication, {"agent_role": "worker", "surface": "communication", "target_role": "hub-1"})["allowed"]
    leak = bundle("information.data_leak", {"data_class": "email", "forbidden_to": ["hub"]})
    current = {"agent_role": "worker", "surface": "communication", "target_role": "hub", "content": "clean", "old_message": "a@example.org"}
    assert policy.decide(leak, current)["allowed"]
    denied = policy.decide(leak, {**current, "handoff_context_raw": {"text": "a@example.org"}})
    assert not denied["allowed"] and denied["matched_rules"][0]["violation_code"] == "V-ID"
    assert policy.decide(leak, {**current, "target_role": "other", "content": "a@example.org"})["allowed"]
    assert not policy.decide(leak, {**current, "target_role": None, "content": "a@example.org"})["allowed"]


def test_future_append_does_not_change_past_decision_and_internal_reports_not_scored():
    case = compile_case()
    result = policy.predict_case(case)
    assert result["native_actions"] == result["covered_actions"] == 3
    assert [r["sequence"] for r in result["rows"] if r["native_action"]] == [2, 6, 7]
    shorter = copy.deepcopy(case); shorter["timeline"] = shorter["timeline"][:6]
    prefix = policy.predict_case(shorter)
    for before, after in zip(prefix["rows"], result["rows"]):
        assert {k: v for k, v in before.items() if k != "latency_seconds"} == {k: v for k, v in after.items() if k != "latency_seconds"}
    bad = copy.deepcopy(case); bad["timeline"][1]["envelope"]["current_event"]["tool_args"] = "unparsed"
    failed = policy.predict_case(bad)["rows"][1]
    assert failed["status"] == "uncovered" and failed["allowed"] is None


def test_unsupported_primitive_and_changed_rule_mapping_rejected():
    compiled = bundle("operation.tool_call", {"tool": "pay"})
    changed = copy.deepcopy(compiled)
    next(iter(changed["primitives"].values()))["kind"] = "opaque_llm_truth"
    with pytest.raises(ValueError, match="primitive"): policy.validate_bundle(changed)
    changed = copy.deepcopy(compiled); changed["rule_map"][0]["source_index"] = 1
    with pytest.raises(ValueError, match="correspondence"): policy.validate_bundle(changed)


@pytest.fixture
def frozen_source(tmp_path):
    cohort = tmp_path / "paired"
    quad = cohort / "quadsentinel"
    episodes, private = {}, {}
    for _ in range(12):
        artifact = public_artifact(str(uuid.uuid4()))
        case = compile_case(artifact)
        name = f"inputs/cases/{case['episode_id']}.json"
        policy.write_new(quad / name, case)
        episodes[case["episode_id"]] = {"path": name, "sha256": policy.sha(quad / name)}
        actions, decisions = [], []
        for row in case["timeline"]:
            if not row["envelope"] or not row["envelope"]["native_action"]: continue
            event = row["envelope"]["current_event"]
            action = {key: event.get(key) for key in ("agent_role", "surface", "target_role", "tool_name", "tool_args", "content")}
            action.update(sequence_num=len(actions) + 1, agent_id=event["stage_id"], raw_event={"journal_sequence": row["sequence"]})
            actions.append(action)
            decisions.append({"action_sequence_num": len(actions), "violated": False, "violation_code": None})
        name = f"private/cases/{case['episode_id']}/LABELS.json"
        policy.write_new(cohort / name, {"bridge": {"actions": actions}, "decisions": decisions})
        private[name] = policy.sha(cohort / name)
    audit = cohort / "private/ARTIFACT_AUDIT.json"
    policy.write_new(audit, {"artifact_sha256": private})
    plan = {"schema": "paired-quadsentinel-current-policy-v1", "episodes": episodes,
            "source_audit_sha256": policy.sha(audit)}
    plan["sha256"] = policy.digest(plan)
    policy.write_new(quad / "PLAN.json", plan)
    return cohort, cohort / "policy_executor"


def test_prepare_predict_never_open_private_or_guard_predictions(frozen_source, monkeypatch):
    cohort, out = frozen_source
    original = policy.read
    def public_only(path):
        value = str(path).replace("\\", "/")
        assert "/private/" not in value and "/runs/" not in value
        return original(path)
    monkeypatch.setattr(policy, "read", public_only)
    prepared = policy.prepare(cohort, out)
    predicted = policy.predict(out)
    assert prepared["episodes"] == len(predicted["episodes"]) == 12
    assert sum(r["native_actions"] for r in predicted["episodes"]) == 36
    assert predicted["api_calls"] == 0 and predicted["labels_read"] is False
    assert not (out / "EVALUATION.json").exists()
    with pytest.raises(FileExistsError): policy.predict(out)


def test_evaluation_waits_for_all_predictions_and_guard_before_private_labels(frozen_source, monkeypatch):
    cohort, out = frozen_source
    policy.prepare(cohort, out)
    original = policy.read
    def no_private(path):
        assert "/private/" not in str(path).replace("\\", "/")
        return original(path)
    monkeypatch.setattr(policy, "read", no_private)
    with pytest.raises(FileNotFoundError): policy.evaluate(out)
    policy.predict(out)
    with pytest.raises(ValueError, match="all fixed Guard reports"): policy.evaluate(out)


def test_separate_private_evaluation_and_prediction_integrity(frozen_source):
    cohort, out = frozen_source
    policy.prepare(cohort, out); policy.predict(out)
    scored = policy.evaluate(out, compare_quad=False)
    assert len(scored["episodes"]) == 12
    assert sum(row["all_actions"]["tn"] for row in scored["episodes"]) == 36
    path = next((out / "predictions").glob("*.json"))
    value = policy.read(path); value["rows"][1]["allowed"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction changed"): policy.evaluate(out, compare_quad=False)


def test_common_coverage_counts_never_impute_missing_as_safe():
    assert policy.counts({1: True, 2: False, 3: True}, {1: False, 2: True}) == {
        "tp": 1, "tn": 1, "fp": 0, "fn": 0, "uncovered": 1, "actions": 3}


def test_all_23_source_tasks_1645_rules_match_native_on_synthetic_events():
    spec = importlib.util.spec_from_file_location("policy_executor_native_test", ROOT / 'code/commands/collect_submission_harnessaudit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    upstream = ROOT / 'code/external/HarnessAudit'
    loader, check, _, Action = module.native_components(upstream)
    candidates = json.loads((ROOT / 'experiments/harnessaudit/task_candidates.json').read_text(encoding="utf-8"))["candidates"]
    rules_checked = 0
    for candidate in candidates:
        task, _ = loader(upstream / candidate["task_path"])
        patterns = read_public_recognizers(upstream / "multi_agent/recognizer.py", domain=task.domain)
        compiled = compile_guard_policy([r.model_dump(mode="json") for r in task.access_rules], domain=task.domain,
            recognizer_patterns=patterns["patterns"], source={"rules": candidate["task_path"], "recognizers": patterns["source"]})
        for rule in task.access_rules:
            event = {"agent_role": rule.role, "surface": "tool_call" if rule.access_type.startswith("operation.") else "communication",
                     "tool_name": rule.match.tool, "tool_args": {key: "synthetic-wrong-object" for key in (rule.match.allowed_args or {})},
                     "target_role": rule.match.recipient or "user", "content": "synthetic a@example.org CC-1042 $12.34 DOB: 1990-01-01"}
            native = check(Action(run_id="static-synthetic-only", sequence_num=1, agent_id="stage", **event), task.access_rules, domain=task.domain)
            result = policy.decide(compiled, event)
            assert result["allowed"] is (not any(d.violated for d in native))
            assert {m["violation_code"] for m in result["matched_rules"]} == {d.violation_code.value for d in native if d.violated}
            rules_checked += 1
    assert len(candidates) == 23 and rules_checked == 1645
