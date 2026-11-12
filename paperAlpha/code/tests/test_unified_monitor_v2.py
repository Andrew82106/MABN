import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_unified_monitor_v2 import encode, forward


def test_unified_monitor_ignores_label_evidence():
    ep = {
        "events": [
            {"actor": "ingress", "event_type": "input_received", "data_class": "untrusted"},
            {"actor": "ingress", "event_type": "message_sent", "target": "executor", "data_class": "untrusted", "edge_kind": "unexpected"},
            {"actor": "executor", "event_type": "tool_call", "tool": "shell.exec", "data_class": "untrusted"},
        ],
        "label": 1,
        "label_evidence": {"source_agents": ["fake"], "sink_agents": ["fake"]},
    }
    theta = [0.0] * 13
    p1, _ = forward(theta, ep, True)
    ep2 = dict(ep)
    ep2["label_evidence"] = {"source_agents": [], "sink_agents": []}
    p2, _ = forward(theta, ep2, True)
    assert p1 == p2


def test_encode_exposes_named_local_features_and_edges():
    ep = {"events": [
        {"actor": "a", "event_type": "input_received", "data_class": "untrusted"},
        {"actor": "a", "event_type": "message_sent", "target": "b", "data_class": "untrusted", "edge_kind": "workflow"},
        {"actor": "b", "event_type": "tool_call", "tool": "payments.write", "data_class": "untrusted"},
    ]}
    names, X, edges = encode(ep)
    assert names == ["a", "b"]
    assert X.shape == (2, 6)
    assert len(edges) == 1
    assert edges[0][3]["source"] == "a"


def test_public_independent_trace_has_no_hidden_label_fields():
    public = Path(__file__).resolve().parents[2] / 'experiments/synthetic_mas/runs/independent_mas_v2/traces_public.jsonl'
    if not public.exists():
        return
    row = json.loads(public.read_text(encoding="utf-8").splitlines()[0])
    assert not {"label", "external_effect", "hidden_fields", "label_source", "scenario_family"}.intersection(row)
