from pathlib import Path
import importlib.util
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_local_seeded_relation_v1.py"
SPEC = importlib.util.spec_from_file_location("local_seeded_relation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Two:
    @staticmethod
    def layer1_score(parsed, bn):
        return 0.2, {"a0": 0.9, "a1": 0.0}


def test_layer_one_node_score_seeds_dynamic_state(monkeypatch):
    parsed = {
        "names": ["a0", "a1"], "source": np.zeros(2), "sink": np.array([0., 1.]),
        "edges": [(0, 1, np.array([1., 0., 0., 1., 0., 0., 1.]), {"source": "a0", "target": "a1"})],
    }
    theta = np.zeros(13)
    # Gate ~0.5 and positive path/head weights; only a Layer-1 seed can make
    # the downstream sink non-zero in this fixture.
    theta[7:13] = [-1., 4., 0., 0., 1., 0.]
    score = MODULE.seeded_propagate(_Two, theta, parsed, {})
    assert score > 0.5

