"""Synthetic policy tests. No API calls or collected episode outputs are read."""
import ast
import copy
from fnmatch import fnmatch
import importlib.util
import json
from pathlib import Path
import re

import pytest

from paperalpha_runtime.harnessaudit_guard_policy import compile_guard_policy, read_public_recognizers
from paperalpha_runtime.quadsentinel_adapter import validate_policy_book


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'code/external/HarnessAudit'
SOURCE = {"rules": {"path": "synthetic-public-rules", "sha256": "a" * 64},
          "recognizers": {"path": "synthetic-public-registry", "sha256": "b" * 64}}
PATTERNS = {"customer_email": [r"[a-z]+@example\.org"]}


def rule(kind, match, **kwargs):
    return dict(role="worker", access_type=kind, match=match, **kwargs)


def compile_rules(rules):
    return compile_guard_policy(rules, domain="ecommerce", recognizer_patterns=PATTERNS, source=SOURCE)


def primitive_values(bundle, event):
    """Synthetic facts only: these are never injected into a live guard."""
    result = {}
    for name, p in bundle["primitives"].items():
        kind = p["kind"]
        if kind == "agent_role_is":
            value = event.get("agent_role") == p["role"]
        elif kind == "surface_is":
            value = event.get("surface") == p["surface"]
        elif kind == "tool_name_matches":
            value = event.get("tool_name") is not None and fnmatch(event["tool_name"], p["pattern"])
        elif kind == "argument_present_nonnull":
            value = (event.get("tool_args") or {}).get(p["key"]) is not None
        elif kind == "argument_matches_pattern":
            actual = (event.get("tool_args") or {}).get(p["key"])
            pattern = p["pattern"]
            value = actual is not None and (str(actual).startswith(pattern[:-1]) if pattern.endswith("/*") else fnmatch(str(actual), pattern))
        elif kind == "target_role_matches":
            value = event.get("target_role") is not None and fnmatch(event["target_role"], p["pattern"])
        elif kind == "recipient_forbidden":
            target = event.get("target_role") or ""
            value = not target or "*" in p["forbidden_to"] or target in p["forbidden_to"]
        elif kind == "payload_contains_class":
            parts = [event["content"]] if event.get("content") else []
            if event.get("handoff_context_raw") is not None:
                parts.append(json.dumps(event["handoff_context_raw"], ensure_ascii=False))
            value = any(re.search(pattern, "\n".join(parts)) for pattern in p["regexes"])
        else:
            raise AssertionError(kind)
        result[name] = bool(value)
    return result


def evaluate(bundle, event):
    values = primitive_values(bundle, event)
    # Execute only admitted propositional syntax in synthetic tests.
    validate_policy_book(bundle["policy_book"])
    return all(eval(re.sub(r"\b(NOT|AND|OR)\b", lambda m: m[0].lower(), row["logic"]),
                    {"__builtins__": {}}, values) for row in bundle["policy_book"])


def test_true_means_allowed_and_predicates_start_unobserved():
    bundle = compile_rules([rule("operation.tool_call", {"tool": "pay"})])
    event = {"agent_role": "worker", "surface": "tool_call", "tool_name": "pay"}
    assert not evaluate(bundle, event)
    assert evaluate(bundle, {**event, "agent_role": "other"})
    assert all(definition[3] is False for row in bundle["policy_book"] for definition in row["predicates"])
    assert "not a safe verdict" in bundle["input_contract"]["updates"]


@pytest.mark.parametrize("allowed,args,expected", [
    (None, {}, False), ({}, {"id": "wrong"}, True),
    ({"id": "A"}, {}, True), ({"id": "A"}, {"id": None}, True),
    ({"id": "A"}, {"id": "A"}, True), ({"id": "A"}, {"id": ""}, False),
    ({"id": "A"}, {"id": "B"}, False), ({"id": "0"}, {"id": 0}, True),
    ({"id": "False"}, {"id": False}, True),
    ({"id": "PF-*"}, {"id": "PF-4821"}, True),
    ({"path": 'code/src/*'}, {"path": 'code/src/a/b.py'}, True),
    ({"path": 'code/src/*'}, {"path": "other/a.py"}, False),
    ({"id": "A", "kind": "X"}, {"id": "A", "kind": "Y"}, False),
])
def test_resource_protected_arguments_match_native(allowed, args, expected):
    bundle = compile_rules([rule("operation.resource", {"tool": "pay", "allowed_args": allowed})])
    assert evaluate(bundle, {"agent_role": "worker", "surface": "tool_call", "tool_name": "pay", "tool_args": args}) is expected


def test_absent_allowed_args_is_full_prohibition():
    bundle = compile_rules([rule("operation.resource", {"tool": "pay"})])
    assert not evaluate(bundle, {"agent_role": "worker", "surface": "tool_call", "tool_name": "pay"})


def test_overlapping_rules_are_conjunctive_not_unioned():
    bundle = compile_rules([rule("operation.resource", {"tool": "pay", "allowed_args": {"id": x}}) for x in ("A", "B")])
    for value in ("A", "B", "C"):
        assert not evaluate(bundle, {"agent_role": "worker", "surface": "tool_call", "tool_name": "pay", "tool_args": {"id": value}})


@pytest.mark.parametrize("recipient,expected", [("hub", False), ("other", True), (None, True), ("", True)])
def test_communication_is_role_scoped(recipient, expected):
    bundle = compile_rules([rule("information.communication", {"recipient": "hub"})])
    assert evaluate(bundle, {"agent_role": "worker", "surface": "communication", "target_role": recipient}) is expected


def test_communication_null_and_empty_differ_for_wildcard():
    bundle = compile_rules([rule("information.communication", {"recipient": "*"})])
    base = {"agent_role": "worker", "surface": "communication"}
    assert evaluate(bundle, base)
    assert not evaluate(bundle, {**base, "target_role": ""})


def test_data_leak_current_payload_and_recipient_semantics():
    bundle = compile_rules([rule("information.data_leak", {"data_class": "customer_email", "forbidden_to": ["hub"]})])
    base = {"agent_role": "worker", "surface": "communication", "target_role": "hub"}
    assert evaluate(bundle, {**base, "content": "safe", "older_message": "a@example.org"})
    assert not evaluate(bundle, {**base, "content": "a@example.org"})
    assert not evaluate(bundle, {**base, "handoff_context_raw": {"tool_reply": "a@example.org"}})
    assert evaluate(bundle, {**base, "target_role": "other", "content": "a@example.org"})
    assert not evaluate(bundle, {**base, "target_role": None, "content": "a@example.org"})
    assert evaluate(bundle, {**base, "surface": "tool_call", "content": "a@example.org"})
    empty = compile_rules([rule("information.data_leak", {"data_class": "customer_email", "forbidden_to": []})])
    assert not evaluate(empty, {**base, "target_role": "", "content": "a@example.org"})
    assert evaluate(empty, {**base, "content": "a@example.org"})
    glob = compile_rules([rule("information.data_leak", {"data_class": "customer_email", "forbidden_to": ["hu*"]})])
    assert evaluate(glob, {**base, "content": "a@example.org"})


@pytest.mark.parametrize("bad", [
    rule("unknown", {"tool": "pay"}),
    rule("operation.tool_call", {"tool": "pay", "hidden_extra": 1}),
    rule("operation.resource", {"tool": "pay", "allowed_args": {"id": ["A", "B"]}}),
    rule("operation.resource", {"tool": "pay", "allowed_args": {"id": None}}),
    rule("information.data_leak", {"data_class": "not_registered"}),
    rule("information.communication", {"recipient": None}),
    {**rule("operation.tool_call", {"tool": "pay"}), "violated": True},
])
def test_unsupported_input_fails_whole_compilation(bad):
    with pytest.raises(ValueError):
        compile_rules([rule("operation.tool_call", {"tool": "ok"}), bad])


def test_raw_provenance_inactive_fields_and_no_event_answers():
    raw = rule("operation.resource", {"tool": "pay", "allowed_args": {}, "recipient": None,
        "data_class": None, "forbidden_to": ["*"]}, severity=None, synthesized=True, description="Do not pay.")
    original = copy.deepcopy(raw)
    bundle = compile_rules([raw, raw])
    assert bundle == compile_rules([raw, raw])
    assert len(bundle["rule_map"]) == 2
    assert bundle["rule_map"][0]["raw_rule"] == original
    assert bundle["rule_map"][0]["native_inactive_match_fields"] == ["data_class", "forbidden_to", "recipient"]
    assert "no-op" in bundle["rule_map"][0]["note"]
    assert bundle["rule_map"][0]["native_evidence"]["matcher"].endswith("_matches/_arg_matches_allowed")
    raw["match"]["allowed_args"]["later"] = "not-shared"
    assert bundle["rule_map"][0]["raw_rule"] == original
    assert "Exact public native rule" in bundle["natural_language_policy"]
    assert "violated" not in bundle


def test_registry_is_literal_and_never_executed(tmp_path):
    path = tmp_path / "registry.py"
    path.write_text("raise AssertionError('must not execute')\n_register('ecommerce', {'x': ['abc']})\n", encoding="utf-8")
    result = read_public_recognizers(path, domain="ecommerce")
    assert result["patterns"] == {"x": ["abc"]}
    assert len(result["source"]["sha256"]) == 64
    path.write_text("_register('ecommerce', build_patterns())\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_public_recognizers(path, domain="ecommerce")


def test_all_23_static_tasks_compile_and_match_native_synthetic_events():
    candidate_path = ROOT / 'experiments/harnessaudit/task_candidates.json'
    if not candidate_path.exists() or not UPSTREAM.exists():
        pytest.skip("optional pinned author sources not installed")
    spec = importlib.util.spec_from_file_location("guard_policy_native_import", ROOT / 'code/commands/collect_submission_harnessaudit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader, check, _, Action = module.native_components(UPSTREAM)
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))["candidates"]
    count = 0
    for candidate in candidates:
        task, _ = loader(UPSTREAM / candidate["task_path"])
        registry = read_public_recognizers(UPSTREAM / "multi_agent/recognizer.py", domain=task.domain)
        rules = [r.model_dump(mode="json") for r in task.access_rules]
        compiled = compile_guard_policy(rules, domain=task.domain, recognizer_patterns=registry["patterns"],
            source={"rules": {"path": candidate["task_path"], "sha256": candidate["task_sha256"]}, "recognizers": registry["source"]})
        validate_policy_book(compiled["policy_book"])
        assert len(compiled["rule_map"]) == len(rules)
        assert [entry["raw_rule"] for entry in compiled["rule_map"]] == rules
        count += len(rules)
        for r in task.access_rules:
            event = {"agent_role": r.role, "surface": "tool_call" if r.access_type.startswith("operation.") else "communication",
                     "tool_name": r.match.tool, "tool_args": {k: "synthetic-wrong-object" for k in (r.match.allowed_args or {})},
                     "target_role": r.match.recipient or "user", "content": "synthetic a@example.org CC-1042 $12.34 DOB: 1990-01-01"}
            action = Action(run_id="synthetic-policy-test", sequence_num=1, agent_id="stage", **event)
            assert evaluate(compiled, event) is (not any(d.violated for d in check(action, task.access_rules, domain=task.domain)))
    assert len(candidates) == 23
    assert count == 1645


def test_author_propositional_evaluator_without_guard_dependencies():
    """Read and run only the author's pure boolean method, not a fake Guard."""
    path = ROOT / 'code/external/QuadSentinel/src/quadsentinel/agents/verifier.py'
    if not path.exists():
        pytest.skip("optional author source absent")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PolicyVerifier")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_evaluate_logic")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    bundle = compile_rules([rule("operation.tool_call", {"tool": "pay"})])
    for tool, expected in [("pay", False), ("read", True)]:
        values = primitive_values(bundle, {"agent_role": "worker", "surface": "tool_call", "tool_name": tool})
        values = dict(sorted(values.items(), key=lambda item: len(item[0]), reverse=True))
        assert namespace["_evaluate_logic"](None, bundle["policy_book"][0]["logic"], values) is expected
