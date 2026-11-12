"""Author-path numerical/contract tests; full text encoding is an opt-in test."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]

from paperalpha_runtime.blindguard_adapter import (
    BlindGuardMonitor, CHECKPOINT, SOURCE_BLOBS, default_repository,
    inspect_checkout, inspect_embedding_snapshot, validate_graph,
)


@pytest.fixture(scope="module")
def monitor():
    for dependency in ["torch", "torch_geometric", "torch_sparse", "torch_scatter"]:
        pytest.importorskip(dependency)
    if not (default_repository() / CHECKPOINT).is_file():
        pytest.skip("Selected author checkout/checkpoint is not locally available")
    return BlindGuardMonitor()


def graph():
    return [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]


def embeddings():
    return np.random.default_rng(72).normal(size=(4, 384)).astype(np.float32)


@pytest.mark.parametrize("matrix,n", [([[0, 1]], 2), ([[0, 0], [0, 0]], 2),
                                      ([[0, -1], [1, 0]], 2),
                                      ([[0, float("nan")], [1, 0]], 2),
                                      ([[0, 1.5], [1, 0]], 2), ([[0]], 1)])
def test_invalid_graph_rejected(matrix, n):
    with pytest.raises(ValueError):
        validate_graph(matrix, n)


def test_original_recipient_projection_preserved(monitor):
    import torch

    features = embeddings()
    x, edge_index, edge_attr = monitor.project_embeddings(features, graph())
    assert tuple(x.shape) == (4, 384)
    assert tuple(edge_attr.shape) == (4, 1536)
    assert torch.equal(x[0], torch.zeros(384))
    np.testing.assert_allclose(x[1:].numpy(), features[1:], atol=0, rtol=0)
    np.testing.assert_array_equal(edge_index.numpy(), np.array(graph()).nonzero())


def test_author_encoder_matches_three_part_formula(monitor):
    import torch

    x, edge_index, _ = monitor.project_embeddings(embeddings(), graph())
    adjacency = torch.tensor(graph(), dtype=torch.float32)
    degrees = adjacency.sum(1)
    inverse = degrees.pow(-0.5)
    inverse[torch.isinf(inverse)] = 0
    neighborhood = (inverse[:, None] * adjacency * inverse[None, :]) @ x
    global_mean = x.mean(0).expand(len(x), -1)
    with torch.inference_mode():
        reference = monitor.model.fc(torch.cat([x, neighborhood, global_mean], dim=1))
        actual = monitor.model.encode(x, edge_index)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-6)


def test_scores_match_original_cosine_and_topk(monitor):
    import torch

    features = embeddings()
    x, edge_index, _ = monitor.project_embeddings(features, graph())
    with torch.inference_mode():
        z = torch.nn.functional.normalize(monitor.model.encode(x, edge_index), dim=-1)
        expected = -(z @ z.T).sum(1)
    result = monitor.score_embeddings(features, graph(), topk=3)
    np.testing.assert_allclose(result["node_anomaly_scores"], expected.numpy(), rtol=0, atol=1e-6)
    assert result["topk_node_indices"] == torch.topk(expected, 3).indices.tolist()
    assert result["is_probability"] is False
    assert result["used_round_index"] == 0
    assert result["monitor_only"] is True


@pytest.mark.parametrize("topk", [0, 5, True, 1.5])
def test_topk_not_silently_clipped(monitor, topk):
    with pytest.raises(ValueError):
        monitor.score_embeddings(embeddings(), graph(), topk=topk)


def test_rejects_nonoriginal_embedding_dimension(monitor):
    with pytest.raises(ValueError):
        monitor.score_embeddings(np.zeros((4, 768)), graph())


def test_rejects_nonfinite_embedding(monitor):
    features = embeddings()
    features[0, 0] = np.nan
    with pytest.raises(ValueError):
        monitor.score_embeddings(features, graph())


def test_metadata_not_accepted_as_message(monitor):
    with pytest.raises(TypeError):
        monitor.encode_messages([{"text": "visible", "is_attack": True}, {"text": "other"}])


def test_later_rounds_do_not_enter_author_profile(monkeypatch, monitor):
    seen = []
    def score_first(messages, adjacency, *, topk):
        seen.append(messages)
        return {"topk_node_indices": [2, 1, 0], "is_probability": False}
    monkeypatch.setattr(monitor, "score_initial_round", score_first)
    first = ["a", "b", "c", "d"]
    result = monitor.score_visible_prefix([first, ["later"] * 4], graph())
    assert seen == [first]
    assert result["later_rounds_used"] is False
    assert result["visible_round_count"] == 2


def test_mutated_source_fails_before_import(tmp_path, monkeypatch):
    from paperalpha_runtime import blindguard_adapter as adapter

    for relative in SOURCE_BLOBS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = default_repository() / relative
        if not source.is_file():
            pytest.skip("Author checkout not available")
        shutil.copy2(source, target)
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *a, **k: adapter.UPSTREAM_COMMIT)
    with (tmp_path / "PI" / "TAM.py").open("ab") as stream:
        stream.write(b"\n# deliberate test mutation\n")
    with pytest.raises(ValueError, match="Author source was changed"):
        inspect_checkout(tmp_path)


def test_missing_cache_fails_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_embedding_snapshot(tmp_path)


@pytest.mark.skipif(os.environ.get("PAPERALPHA_BLINDGUARD_SMOKE") != "1",
                    reason="Explicit local-cache integration opt-in; no downloads")
def test_original_embedding_and_checkpoint_smoke():
    from smoke_blindguard import smoke

    report = smoke()
    assert report["status"] == "author_weight_inference_smoke_passed"
    assert report["benchmark_performance_measured"] is False
    assert report["api_calls"] == 0
    assert report["checks"]["embedding_shape"] == [4, 384]
    assert report["checks"]["finite_scores"] is True
