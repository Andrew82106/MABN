import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.contextual_knowledge_input import ContextualKnowledgeCompiler, compile_contextual_journal
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN, compile_knowledge_journal
from paperalpha_monitor.tool_capabilities import load_tool_registry
from test_initial_task_context import fixture


def example():
    request, start, annotation = fixture()
    start = {"schema_version": "runtime-journal-v1", **start}
    start["payload"]["external_endpoints"] = ["user"]
    rows = [start,
            {"schema_version": "runtime-journal-v1", "sequence": 2, "kind": "communication",
             "payload": {"actor": "hub", "target": "worker", "content": "Book after approval."}},
            {"schema_version": "runtime-journal-v1", "sequence": 3, "kind": "tool_handler_started",
             "payload": {"actor": "worker", "tool": "submit_service_request", "invocation_id": "a1", "provider_call_id": "p1",
                         "arguments": {"request_id": "DL-018"}}}]
    context = {"initial_request": request, "run_start": start, "role_annotation": annotation, "tool_registry": load_tool_registry()}
    return rows, context


def test_unknown_mode_is_exact_old_v7_and_predicted_roles_change_real_cpt_selection():
    rows, context = example()
    old = compile_knowledge_journal(rows, tool_registry=load_tool_registry())
    unknown, _ = compile_contextual_journal(rows, role_mode="unknown", **context)
    predicted, evidence = compile_contextual_journal(rows, role_mode="predicted", **context)
    model = KnowledgeScopedBN()
    np.testing.assert_array_equal(model.predict([old]), model.predict([unknown]))
    a, b = old.steps[-1].mechanism_factors[0], predicted.steps[-1].mechanism_factors[0]
    assert a.parameter_indices != b.parameter_indices
    assert predicted.steps[-1].base.actions[0]["role"] == "executor"
    assert predicted.steps[-1].base.actions[0]["permission_status"] == "unknown"
    assert predicted.steps[-1].base.actions[0]["binding"]["resource"] is None
    assert evidence["effective_node_roles"] == {"hub": "reasoner", "worker": "executor"}
    # Identical initialized role tables imply identical scores, not improvement.
    np.testing.assert_array_equal(model.predict([unknown]), model.predict([predicted]))
    model.theta[list(b.parameter_indices)] += 1.0
    model.theta[model.parameter_names.index("head:dispatched_authorization_any")] = 1.0
    assert model.predict([predicted])[0][-1] != model.predict([unknown])[0][-1]


def test_mixed_roles_in_one_functional_node_stay_unknown():
    rows, context = example()
    context["workflow"] = {"nodes": ["group"], "edges": [], "agent_to_node": {"hub": "group", "worker": "group"}}
    _, explanation = compile_contextual_journal(rows, role_mode="predicted", **context)
    assert explanation["effective_node_roles"] == {"group": "unknown"}
    assert explanation["grouped_role_ambiguities"] == ["group"]


def test_future_rows_cannot_change_earlier_scores_and_bad_start_is_atomic():
    rows, context = example()
    compiler = ContextualKnowledgeCompiler(role_mode="predicted", **context)
    wrong = copy.deepcopy(rows[0])
    wrong["payload"]["agents"].append("other")
    with pytest.raises(ValueError, match="run-start"):
        compiler.push(wrong)
    assert compiler.projector.sequence == 0
    short, _ = compile_contextual_journal(rows[:2], role_mode="predicted", **context)
    longer, _ = compile_contextual_journal(rows, role_mode="predicted", **context)
    model = KnowledgeScopedBN()
    np.testing.assert_array_equal(model.predict([short])[0], model.predict([longer])[0][:1])


def test_conflicting_external_roles_or_missing_classifier_rejected():
    _, context = example()
    with pytest.raises(ValueError, match="override"):
        ContextualKnowledgeCompiler(**context, roles={"worker": "reader"})
    context.pop("role_annotation")
    with pytest.raises(ValueError, match="requires"):
        ContextualKnowledgeCompiler(**context, role_mode="predicted")
