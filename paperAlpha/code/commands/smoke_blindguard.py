'Offline author-weight BlindGuard smoke test; prints JSON, never API calls.\n\nSynthetic, nonsensitive messages test execution only. No attack labels or\nbenchmark performance metrics are produced. The report can be embedded in\nexperiments/baselines.json after review; this script does not mutate catalogs.\n'
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))

from paperalpha_runtime.blindguard_adapter import BlindGuardMonitor


def synthetic_case():
    return {
        "messages": [
            "The planning agent asks for a short summary of a public garden schedule.",
            "The research agent reports that the public garden opens at nine in the morning.",
            "The review agent checks that the summary uses only the provided public schedule.",
            "The writing agent prepares a two-sentence summary of the garden opening hours.",
        ],
        "adjacency": [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
    }


def smoke(*, repository=None, embedding_snapshot=None, device="cpu"):
    import numpy as np
    import torch

    started = time.perf_counter()
    monitor = BlindGuardMonitor(repository, device=device)
    monitor.load_embedding(embedding_snapshot)
    case = synthetic_case()
    embeddings = monitor.encode_messages(case["messages"])
    # PI/utils.py calls encode on one string at a time. Confirm that caching
    # and batching preserve its numerical output with this exact local model.
    single_embeddings = np.asarray([
        monitor._sentence_encoder.encode(message, show_progress_bar=False)
        for message in case["messages"]
    ])
    embedding_error = float(np.max(np.abs(embeddings - single_embeddings)))
    result = monitor.score_embeddings(embeddings, case["adjacency"], topk=3)
    repeat = monitor.score_embeddings(embeddings, case["adjacency"], topk=3)
    x, edge_index, _ = monitor.project_embeddings(embeddings, case["adjacency"])
    with torch.inference_mode():
        z = monitor.model.encode(x, edge_index)
        normalized = torch.nn.functional.normalize(z, p=2, dim=-1)
        independent_reference = -(normalized @ normalized.T).sum(dim=1).cpu().numpy()
    scores = np.asarray(result["node_anomaly_scores"])
    exact_repeat = np.array_equal(scores, repeat["node_anomaly_scores"])
    max_error = float(np.max(np.abs(scores - independent_reference)))
    if not np.isfinite(scores).all() or not exact_repeat or max_error > 1e-6 or embedding_error > 1e-5:
        raise AssertionError("Author inference parity/finite-score smoke check failed")
    return {
        "status": "author_weight_inference_smoke_passed",
        "benchmark_performance_measured": False,
        "input_kind": "four-agent synthetic nonsensitive directed acyclic communication graph",
        "task_boundary": "Not a privacy or attack-detection evaluation; no positive/negative labels exist for this smoke input.",
        "api_calls": 0, "network_model_downloads": 0,
        "checks": {"finite_scores": True, "repeat_identical": exact_repeat,
                   "independent_cosine_score_max_abs_error": max_error,
                   "batch_vs_author_single_encode_max_abs_error": embedding_error,
                   "strict_author_state_load": True, "weights_only_load": True,
                   "embedding_shape": list(embeddings.shape),
                   "embedding_norms": np.linalg.norm(embeddings, axis=1).tolist(),
                   "source_only_node_feature_zero_preserved": bool(torch.equal(x[0], torch.zeros_like(x[0])))},
        "result": result, "provenance": monitor.provenance,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--embedding-snapshot", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(smoke(repository=args.repository, embedding_snapshot=args.embedding_snapshot,
                           device=args.device), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
