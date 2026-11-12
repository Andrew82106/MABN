"""Recompute saved streaming-v1 predictions against its exact source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/streaming_template_bn_20260919')
    args = parser.parse_args()
    manifest = load_json(args.run / "manifest.json")
    for relative, expected in manifest["input_sha256"].items():
        check(sha(ROOT / relative) == expected, "changed input: " + relative)
    for relative, expected in manifest["code_sha256"].items():
        check(sha(args.run / "source_snapshot" / relative) == expected, "changed frozen source: " + relative)
    package_root = args.run / "source_snapshot/code/src/paperalpha_monitor"
    package = types.ModuleType("frozen_monitor")
    package.__path__ = [str(package_root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "frozen_monitor.streaming_bn", package_root / "streaming_bn.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    data = ROOT / manifest["config"]["data"]
    traces = jsonl(data / "traces_public.jsonl")
    labels = {row["episode_id"]: row["label"] for row in jsonl(data / "labels.jsonl")}
    ids = [trace["episode_id"] for trace in traces]
    positions = {identity: i for i, identity in enumerate(ids)}
    groups = {trace["episode_id"]: core.topology_group(trace["workflow_dag"]) for trace in traces}
    y = np.asarray([labels[identity] for identity in ids])
    for seed, split in manifest["splits"].items():
        parts = [set(split[name]["episode_ids"]) for name in ("fit", "validation", "test")]
        check(set.union(*parts) == set(ids), "partition coverage " + seed)
        for left, right in ((0, 1), (0, 2), (1, 2)):
            check(not parts[left] & parts[right], "episode overlap " + seed)
            check(not {groups[i] for i in parts[left]} & {groups[i] for i in parts[right]}, "topology overlap " + seed)
    result = load_json(args.run / "metrics.json")
    diagnostics = {}
    checked_predictions = checked_prefixes = 0
    for setting in manifest["config"]["settings"]:
        key = setting["workflow"] + "__" + setting["permission"]
        trajectories = [core.replay(
            trace, workflow=trace["workflow_dag"] if setting["workflow"] == "workflow_known" else None,
            permission_mode=setting["permission"]) for trace in traces]
        final = [trajectory[-1] for trajectory in trajectories]
        matrix = np.asarray([snapshot.vector() for snapshot in final])
        active = np.any(matrix != 0, axis=0)
        prefix_active = np.any(np.asarray([s.vector() for trajectory in trajectories for s in trajectory]) != 0, axis=0)
        diagnostics[key] = {
            "terminal_active_cells": int(active.sum()), "terminal_design_rank": int(np.linalg.matrix_rank(matrix)),
            "prefix_only_active_cells": [core.CELL_NAMES[i] for i in np.where(prefix_active & ~active)[0]],
            "warning": "rank deficiency means local factor risks need not be uniquely identifiable",
        }
        for run in (r for r in result["runs"] if r["setting"] == key):
            directory = args.run / key / str(run["seed"])
            predictions = jsonl(directory / "predictions.jsonl")
            split = manifest["splits"][str(run["seed"])]
            check([r["episode_id"] for r in predictions] == split["test"]["episode_ids"], "wrong test IDs")
            test = [positions[row["episode_id"]] for row in predictions]
            val = [positions[identity] for identity in split["validation"]["episode_ids"]]
            params = load_json(directory / "model_parameters.json")
            for method, metrics in run["metrics"].items():
                probabilities = np.asarray([row["scores"][method] for row in predictions])
                threshold = metrics["threshold"]
                predicted = probabilities >= threshold
                values = {
                    "f1": f1_score(y[test], predicted, zero_division=0),
                    "precision": precision_score(y[test], predicted, zero_division=0),
                    "recall": recall_score(y[test], predicted, zero_division=0),
                    "auroc": roc_auc_score(y[test], probabilities),
                    "auprc": average_precision_score(y[test], probabilities),
                    "brier": brier_score_loss(y[test], probabilities),
                }
                for name, value in values.items():
                    check(abs(value - metrics[name]) < 1e-10, f"metric mismatch {key}/{method}/{name}")
                checked_predictions += len(test)
                if method not in params:
                    continue  # Baseline metric check, not a second training reproduction.
                model = core.TemplateBN(exclude=frozenset(params[method]["exclude"]))
                model.theta = np.asarray(params[method]["theta"])
                np.testing.assert_allclose(model.predict([final[i] for i in test]), probabilities, atol=1e-12, rtol=1e-12)
                p_val = model.predict([final[i] for i in val])
                candidates = np.unique(np.r_[p_val, 0, np.nextafter(1., 2.)])
                chosen = max(candidates, key=lambda t: (f1_score(y[val], p_val >= t, zero_division=0), t))
                check(chosen == threshold, "BN validation threshold mismatch")
                if method == "template_bn":
                    for row, i in zip(predictions, test):
                        check(row["label"] == labels[row["episode_id"]], "label mismatch")
                        check(row["group"] == groups[row["episode_id"]], "topology group mismatch")
                        np.testing.assert_allclose(model.predict(trajectories[i]), row["prefix_scores"], atol=1e-12, rtol=1e-12)
                        checked_prefixes += len(trajectories[i])
    audit = {
        "status": "passed", "source": "exact source_snapshot; original manifest hashes verified",
        "runs": len(result["runs"]), "checked_method_episode_predictions": checked_predictions,
        "checked_full_bn_prefixes": checked_prefixes, "diagnostics": diagnostics,
        "scope": "artifact integrity, score/threshold/metric recomputation; not external validity or a full refit",
        "limitations": ["baseline saved-score metrics verified but estimators not refitted",
                        "terminal-trained prefix scores remain diagnostics, not calibrated forecasts"],
    }
    (args.run / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
