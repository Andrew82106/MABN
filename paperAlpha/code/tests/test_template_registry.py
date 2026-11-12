import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.streaming_bn import (  # noqa: E402
    CELL_NAMES, N_CELLS, OFFSETS, TEMPLATES, EvidenceStream, Snapshot, TemplateBN,
)
from paperalpha_monitor.template_registry import (  # noqa: E402
    DEFAULT_CATALOG, DEFAULT_CONFIG, TemplateRegistry, compile_snapshot, load_registry,
)


def raw():
    return (json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")), json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8")))


def chain():
    stream = EvidenceStream(workflow={"nodes": ["a", "b"], "edges": []})
    snapshots = [stream.push({"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted", "source_id": "doc1"})]
    snapshots.append(stream.push({"sequence": 1, "actor": "a", "event_type": "message_sent", "target": "b", "data_class": "untrusted"}))
    snapshots.append(stream.push({"sequence": 2, "actor": "b", "event_type": "tool_call", "tool": "shell.exec", "action_id": "run", "permission_denied": True}))
    return stream, snapshots


def test_all_49_cells_exactly_match_existing_model():
    registry = load_registry()
    assert registry.specs == TEMPLATES
    assert registry.n_cells == N_CELLS == 49
    assert registry.offsets == OFFSETS
    assert registry.cell_names == CELL_NAMES
    visited = []
    for name, dims, _ in TEMPLATES:
        for values in itertools.product(*(range(d) for d in dims)):
            actual = registry.cell_index(name, values)
            assert actual == OFFSETS[name] + int(np.ravel_multi_index(values, dims))
            visited.append(actual)
    assert visited == list(range(49))


@pytest.mark.parametrize("mutation", ["bad_dimension", "duplicate_template", "duplicate_parent", "bad_state", "bad_total"])
def test_invalid_specs_rejected(mutation):
    config, catalog = raw()
    if mutation == "bad_dimension":
        config["templates"][0]["dims"][0] = 2
    elif mutation == "duplicate_template":
        config["templates"].append(copy.deepcopy(config["templates"][0]))
    elif mutation == "duplicate_parent":
        config["templates"][0]["parents"][1]["name"] = "permission_status"
    elif mutation == "bad_state":
        config["templates"][0]["parents"][0]["states"][0]["value"] = 3
    else:
        config["expected_total_cells"] = 50
    with pytest.raises(ValueError):
        TemplateRegistry(config, catalog)


@pytest.mark.parametrize("source", [6, 999])
def test_source_must_belong_to_that_specific_risk(source):
    config, catalog = raw()
    config["templates"][0]["catalog_links"][0]["source_ids"] = [source]
    with pytest.raises(ValueError, match="does not belong"):
        TemplateRegistry(config, catalog)


def test_metadata_has_real_sources_and_explicit_coverage_gaps():
    registry = load_registry()
    metadata = registry.semantic_metadata("information_flow")["templates"][0]
    assert metadata["catalog_links"][1]["risk_id"] == "RB026"
    assert metadata["catalog_links"][1]["sources"][0]["id"] == 22
    assert "No semantic" in metadata["catalog_links"][1]["coverage"]
    assert "independence" in metadata["parents"][1]["states"][1]["meaning"]
    assert registry.semantic_metadata("global_context")["templates"][0]["catalog_links"] == []
    metadata["id"] = "mutated"
    assert registry.semantic_metadata("information_flow")["templates"][0]["id"] == "information_flow"


def test_compiled_graph_is_a_real_parameterized_conditional_dag():
    _, snapshots = chain()
    snapshot = snapshots[-1]
    graph = compile_snapshot(snapshot)
    nodes = {node["id"]: node for node in graph.nodes}
    # A three-level graph cannot cycle: observed roots -> risk -> OR.
    levels = {"observed": 0, "risk": 1, "outcome": 2}
    assert all(levels[nodes[left]["kind"]] < levels[nodes[right]["kind"]] for left, right in graph.edges)
    risks = [node for node in graph.nodes if node["kind"] == "risk"]
    assert len(risks) == len(snapshot.factors) == 4
    for factor, risk in zip(snapshot.factors, risks):
        assert risk["cpt"]["selected_cell"] == factor["cell"]
        assert risk["cpt"]["cell_name"] == factor["cell_name"]
        assert [nodes[parent]["variable"] for parent in risk["parents"]] == graph.shared_cpts[factor["template"]]["parent_names"]
        assert risk["sequence"] == factor["sequence"]
        assert risk["observed_paths"] == factor.get("observed_paths", {})
        if risk["scope"] == "action":
            assert risk["actor"] == "b" and risk["action_id"] == "run"
            assert risk["observed_paths"] == {"source:doc1": [0, 1]}
    permission_nodes = [node for node in graph.nodes if node.get("variable") == "permission_status"]
    assert len(permission_nodes) == 1  # Same observed action parent is shared.
    assert permission_nodes[0]["state_name"] == "observed_mismatch"
    model = TemplateBN()
    model.theta = np.linspace(-6, 2, N_CELLS)
    assert graph.risk_probability(model.theta) == pytest.approx(model.predict([snapshot])[0])
    json.dumps(graph.to_dict())


def test_unknown_states_and_unavailable_workflow_are_explicit():
    stream = EvidenceStream()
    snapshot = stream.push({"sequence": 0, "actor": "b", "event_type": "tool_call", "tool": "shell.exec"})
    graph = compile_snapshot(snapshot)
    observed = {node["variable"]: node for node in graph.nodes if node["kind"] == "observed"}
    assert observed["permission_status"]["knowledge_status"] == "unknown"
    assert observed["capability_status"]["knowledge_status"] == "unknown"
    assert observed["unexpected_messages"]["knowledge_status"] == "unavailable"
    assert observed["unexpected_messages"]["observed_state"] == 0


def test_future_or_unrelated_metadata_is_not_read_or_exported():
    stream, snapshots = chain()
    before = compile_snapshot(snapshots[1]).to_dict()
    stream.push({"sequence": 3, "actor": "future_agent", "event_type": "tool_call", "tool": "payments.write"})
    assert compile_snapshot(snapshots[1]).to_dict() == before
    changed = copy.deepcopy(snapshots[1].named_evidence)
    changed["future_events"] = [{"actor": "secret_future", "label": 1, "sequence": 500}]
    changed["workflow_dag"] = {"nodes": ["secret_future"], "edges": []}
    assert compile_snapshot(Snapshot(snapshots[1].sequence, snapshots[1].factors, changed)).to_dict() == before
    assert "secret_future" not in json.dumps(before)
    assert not any(node.get("scope") == "action" for node in before["nodes"])


def test_normative_edge_cannot_invent_a_runtime_path():
    stream = EvidenceStream(workflow={"nodes": ["a", "b"], "edges": [["a", "b"]]})
    stream.push({"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted"})
    snapshot = stream.push({"sequence": 1, "actor": "b", "event_type": "tool_call", "tool": "shell.exec"})
    graph = compile_snapshot(snapshot)
    assert not any(node.get("template") == "information_flow" for node in graph.nodes)


@pytest.mark.parametrize("mutation", ["cell", "parent", "future_factor", "future_path", "duplicate_factor"])
def test_inconsistent_or_future_snapshot_factors_rejected(mutation):
    _, snapshots = chain()
    original = snapshots[-1]
    factors = copy.deepcopy(list(original.factors))
    if mutation == "cell":
        factors[1]["cell"] += 1
    elif mutation == "parent":
        factors[1]["parents"][0] = 3
    elif mutation == "future_factor":
        factors[1]["sequence"] = 50
    elif mutation == "future_path":
        factors[1]["observed_paths"]["source:doc1"].append(50)
    else:
        factors.append(copy.deepcopy(factors[1]))
    with pytest.raises(ValueError):
        compile_snapshot(Snapshot(original.sequence, tuple(factors), original.named_evidence))


def test_shared_cpt_indices_used_for_different_actions():
    stream = EvidenceStream()
    stream.push({"sequence": 0, "actor": "a", "event_type": "tool_call", "tool": "shell.exec", "action_id": "a"})
    snapshot = stream.push({"sequence": 1, "actor": "b", "event_type": "tool_call", "tool": "shell.exec", "action_id": "b"})
    graph = compile_snapshot(snapshot)
    risks = [node for node in graph.nodes if node.get("template") == "authorization"]
    assert len(risks) == 2 and risks[0]["id"] != risks[1]["id"]
    assert risks[0]["cpt"]["selected_cell"] == risks[1]["cpt"]["selected_cell"] == 0
    with pytest.raises(ValueError, match="49 finite"):
        graph.risk_probability(np.ones(48))
