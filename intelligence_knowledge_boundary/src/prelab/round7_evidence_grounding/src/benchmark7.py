"""Post-test CPU scoring benchmark; no training, retuning, labels or GPU calls.

Requires the completed, frozen unified test. Model/feature I/O and matrix assembly
are timed separately. Five full-cohort batches follow one warm-up per method.
The per-item figure is amortized batch time, not online or end-to-end latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import statistics
import time

from evaluate7 import (ROOT, METHODS, EXTERNAL_SPLIT, Bank, data_hashes,
                       metadata, np, save, score_method, sha, threadpool_limits,
                       torch)

REPEATS = 5


class PreparedBank:
    """Exact Bank matrices assembled before the scoring timer starts."""
    def __init__(self, bank, items, models):
        self.item_ids = tuple(item["item_id"] for item in items)
        keys = set()
        for model in models.values():
            if model is None or model["kind"] == "constant":
                continue
            if model["kind"] == "redeep":
                keys.update(("redeep_ecs", "redeep_pks"))
            else:
                keys.add(model["feature"])
        self.matrices = {key: bank.matrix(items, key) for key in sorted(keys)}

    def matrix(self, items, key):
        # The same immutable cohort list is used in every measured invocation.
        return self.matrices[key]


def benchmark(root):
    out = root/"results"
    complete_path = out/"test_complete.json"
    assert complete_path.exists(), "Benchmark is permitted only after unified test completion"
    target = out/"cpu_scoring_benchmark.json"
    assert not target.exists(), "Preserve the first prespecified benchmark run"
    started = time.perf_counter()
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    frozen = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert complete["all_methods_and_both_cohorts_evaluated"]
    assert complete["freeze_sha256"] == sha(out/"freeze.json")
    assert set(frozen["method_names"]) == set(METHODS)
    assert frozen["evaluator_sha256"] == sha(Path(__file__).with_name("evaluate7.py"))
    assert frozen["frozen_model_sha256"] == sha(out/"frozen_models.pkl")
    assert frozen["selection_sha256"] == sha(out/"selection.json")
    assert frozen["data_sha256"] == data_hashes(root)
    integrity_seconds = time.perf_counter()-started

    started = time.perf_counter()
    models = pickle.loads((out/"frozen_models.pkl").read_bytes())
    model_load_seconds = time.perf_counter()-started
    started = time.perf_counter()
    _, _, all_items = metadata(root)
    metadata_load_seconds = time.perf_counter()-started
    cohorts = {}
    with threadpool_limits(limits=4):
        for split, name in (("test", "main"), (EXTERNAL_SPLIT, "external")):
            items = [item for item in all_items if item["split"] == split]
            assert items and [i["item_id"] for i in items] == frozen[
                "test_item_ids" if split == "test" else "external_item_ids"]
            started = time.perf_counter()
            bank = Bank(root, items)
            feature_load_seconds = time.perf_counter()-started
            started = time.perf_counter()
            prepared = PreparedBank(bank, items, models)
            matrix_prepare_seconds = time.perf_counter()-started
            measurements = {}
            for method in METHODS:
                started = time.perf_counter()
                warm = score_method(models[method], prepared, items)
                warmup_seconds = time.perf_counter()-started
                assert np.asarray(warm).shape == (len(items),)
                samples = []
                identical = True
                for _ in range(REPEATS):
                    started = time.perf_counter()
                    values = score_method(models[method], prepared, items)
                    samples.append(time.perf_counter()-started)
                    identical = identical and bool(np.allclose(values, warm, rtol=0, atol=0, equal_nan=True))
                median = statistics.median(samples)
                measurements[method] = {
                    "items": len(items), "finite_scores": int(np.isfinite(warm).sum()),
                    "warmup_batches": 1, "warmup_seconds": warmup_seconds,
                    "measured_batches": REPEATS, "batch_seconds": samples,
                    "median_batch_seconds": median,
                    "amortized_median_seconds_per_item": median/len(items),
                    "identical_scores_all_repeats": identical,
                    "score_sha256": hashlib.sha256(np.asarray(warm, dtype="<f8").tobytes()).hexdigest(),
                    "raw_or_constant": models[method] is not None and models[method]["kind"] in {"raw", "constant"},
                    "mlp_network_reconstruction_included": models[method] is not None and models[method]["kind"] == "mlp",
                }
            cohorts[name] = {
                "split": split, "items": len(items),
                "item_ids": [item["item_id"] for item in items],
                "feature_bank_load_seconds": feature_load_seconds,
                "matrix_prepare_seconds": matrix_prepare_seconds,
                "prepared_matrix_bytes": sum(x.nbytes for x in prepared.matrices.values()),
                "methods": measurements,
            }
            del bank, prepared
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha(out/"freeze.json"), "test_complete_sha256": sha(complete_path),
        "benchmark_script_sha256": sha(Path(__file__)),
        "evaluator_sha256": frozen["evaluator_sha256"],
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__, "numpy": np.__version__, "cpu_threads_max": 4,
        "integrity_check_seconds": integrity_seconds, "model_load_seconds": model_load_seconds,
        "metadata_load_seconds": metadata_load_seconds,
        "cohorts": cohorts,
        "labels_read": False, "training_or_threshold_changes": False, "gpu_calls": 0,
        "scope": "CPU score_method only, with loaded models and preassembled feature matrices; includes classifier scaling and internal scoring allocations, excludes file I/O, matrix assembly, thresholding and report metrics",
        "limitations": [
            "Amortized full-cohort batch time is not single-request online latency or end-to-end detector cost.",
            "Raw methods only read already-computed signals here; Qwen generation/replay/self-confidence/direct-check costs must come from GPU-stage records.",
            "The unchanged evaluator rebuilds the small MLP and copies its loaded state_dict inside each mlp_scores call; this CPU setup is included, so MLP time is actual scorer-call time rather than pure forward-pass time.",
            "One warm-up and five measured batches in fixed method order; actual machine load and filesystem cache are not controlled.",
            "Prepared matrix bytes are payload size, not peak CPU process memory. GPU allocated peaks in stage records are not reserved VRAM or whole-device occupancy.",
            "No labels or test metrics are read and no predictions or thresholds are modified.",
        ],
    }
    save(target, result)
    print("CPU_SCORING_BENCHMARK_COMPLETE", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    benchmark(parser.parse_args().root.resolve())
