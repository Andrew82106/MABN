from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_two_layer_v2.py"
SPEC = importlib.util.spec_from_file_location("two_layer_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_terminal_event_field_cannot_create_local_risk_feature():
    clean = {"agents": ["a0"], "events": [{"actor": "a0", "event_type": "tool_call", "tool": "search.read"}]}
    tampered = {"agents": ["a0"], "events": [{"actor": "a0", "event_type": "tool_call", "tool": "search.read", "external_effect": True, "tool_side_effect": True}]}
    left = MODULE.local_vector(MODULE.event_features(clean))
    right = MODULE.local_vector(MODULE.event_features(tampered))
    assert (left == right).all()


def test_local_cpt_uses_correct_semantic_name():
    episode = {"agents": ["a0"], "events": []}
    bn = MODULE.fit_bn([episode], [0])
    assert "same_agent_source_sink" in bn["feature_names"]


def _minimal_bn():
    return {
        "prior": 0.5,
        "feature_names": ["untrusted_input", "privileged_tool", "same_agent_source_sink", "permission_mismatch"],
        "cpt": {name: 0.5 for name in ["untrusted_input", "privileged_tool", "same_agent_source_sink", "permission_mismatch"]},
    }


def _one_edge(confidence=1.0):
    return {
        "agents": ["a0", "a1"],
        "events": [
            {"actor": "a0", "event_type": "input_received", "data_class": "untrusted", "sequence": 0},
            {"actor": "a0", "target": "a1", "event_type": "message_sent", "data_class": "untrusted", "edge_confidence": confidence, "sequence": 1},
            {"actor": "a1", "event_type": "tool_call", "tool": "shell.exec", "sequence": 2},
        ],
    }


def test_runtime_edge_is_transmitted_once_not_n_times():
    parsed = MODULE.event_features(_one_edge())
    theta = [0.0] * 13  # sigmoid(0) = 0.5 for the single edge gate
    _, explanation = MODULE._propagate(theta, parsed, _minimal_bn(), explain=True)
    assert abs(explanation["raw_path_evidence"] - 0.5) < 1e-9


def test_edge_confidence_is_reliability_not_gate_and_reliability_twice():
    theta = [0.0] * 13
    theta[3] = 8.0  # must not alter the gate after confidence is separated
    _, full = MODULE._propagate(theta, MODULE.event_features(_one_edge(1.0)), _minimal_bn(), explain=True)
    _, half = MODULE._propagate(theta, MODULE.event_features(_one_edge(0.5)), _minimal_bn(), explain=True)
    assert abs(full["raw_path_evidence"] - 0.5) < 1e-9
    assert abs(half["raw_path_evidence"] - 0.25) < 1e-9
