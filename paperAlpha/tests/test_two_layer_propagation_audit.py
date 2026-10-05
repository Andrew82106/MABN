from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_two_layer_v2.py"
    spec = importlib.util.spec_from_file_location("paperalpha_two_layer_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_edge_is_processed_once_not_persistently_repropagated():
    mod = _module()
    info = {"source": "source", "target": "sink", "sequence": 0}
    edge_features = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    parsed = {
        "names": ["source", "sink"],
        "source": np.array([1.0, 0.0]),
        "sink": np.array([0.0, 1.0]),
        "edges": [(0, 1, edge_features, info)],
        "local": {"source": set(), "sink": set()},
        "aggregate": {},
        "semantic": {"score": 0.0, "hits": [], "safe_cue": False},
    }
    names = ["untrusted_input", "privileged_tool", "same_agent_source_sink", "permission_mismatch"]
    bn = {"feature_names": names, "cpt": {name: 0.5 for name in names}, "prior": 0.1}
    theta = np.zeros(13)
    _, explanation = mod._propagate(theta, parsed, bn, explain=True)
    # sigmoid(0)=.5 and one observed edge with confidence 1 gives exactly .5.
    # Repeating the old persistent update twice would incorrectly produce .75.
    assert explanation["raw_path_evidence"] == 0.5

    uncertain = dict(parsed)
    uncertain["edges"] = [(0, 1, edge_features.copy(), info.copy())]
    uncertain["edges"][0][2][3] = 0.5
    _, uncertain_explanation = mod._propagate(theta, uncertain, bn, explain=True)
    # The same confidence is applied once as observation reliability.
    assert uncertain_explanation["raw_path_evidence"] == 0.25


def test_unsequenced_dag_edges_are_topologically_ordered_once():
    mod = _module()
    edges = [
        (1, 2, np.zeros(7), {"sequence": None}),
        (0, 1, np.zeros(7), {"sequence": None}),
    ]
    ordered = mod._ordered_runtime_edges(edges, 3)
    assert [(edge[0], edge[1]) for edge in ordered] == [(0, 1), (1, 2)]
