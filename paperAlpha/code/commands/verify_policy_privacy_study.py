"""Audit frozen numerical code; normalize only JSON tuples/arrays in records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder, out):
    if out.exists():
        raise ValueError("new audit output required")
    report = json.loads((folder / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("artifact changed: " + name)
    snapshot = folder / "source_snapshot"
    sys.path[:0] = [str(snapshot / 'code/src'), str(snapshot / 'code/commands')]
    import paperalpha_monitor.policy_gated_privacy as numerical
    from study_policy_gated_privacy import dataset
    from policy_privacy_fixtures import fixture
    from scalar_binary_graph import evaluate
    if Path(numerical.__file__).resolve() != snapshot / 'code/src/paperalpha_monitor/policy_gated_privacy.py':
        raise ValueError("fresh process must import frozen numerical code")
    model_class = numerical.PolicyGatedPrivacyBN
    losses, gradients, errors = [], [], []
    for name, datafile in (("population", "population_frequency_rows.json"), ("joint", "joint_control_rows.json")):
        model = model_class.restore(json.loads((folder / f"{name}_fit.json").read_text(encoding="utf-8")))
        samples, f, a, weights = dataset(json.loads((folder / datafile).read_text(encoding="utf-8")))
        loss, grad = model.loss_gradient(model.theta, samples, f, a, weights=weights)
        losses.append(abs(loss - model.optimizer["loss"]))
        assert losses[-1] < 1e-10
        for seed in range(3):
            direction = np.random.default_rng(seed).normal(size=len(model.theta))
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            plus = model.loss_gradient(model.theta + eps * direction, samples, f, a, weights=weights)[0]
            minus = model.loss_gradient(model.theta - eps * direction, samples, f, a, weights=weights)[0]
            gradients.append(abs((plus - minus) / (2 * eps) - grad @ direction))
            assert gradients[-1] < 1e-7
    population = model_class.restore(json.loads((folder / "population_fit.json").read_text(encoding="utf-8")))
    predictions = json.loads((folder / "graph_examples.json").read_text(encoding="utf-8"))
    for saved in predictions:
        sample = fixture(**saved["fixture"])
        result = population.predict(sample, at_least=saved["at_least"])
        # No tolerance or dropped fields here: only tuple->JSON array conversion.
        assert json.loads(json.dumps(result)) == saved["result"]
        graph = population.probability_graph(sample, at_least=saved["at_least"])
        assert json.loads(json.dumps(graph)) == saved["graph"]
        scalar = evaluate(saved["graph"])
        errors.append(abs(scalar - result["model_probability"]))
        assert errors[-1] < 1e-12
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / "auditor_source.py")
    audit = {"parent_report_sha256": sha(folder / "REPORT.json"), "auditor_sha256": sha(out / "auditor_source.py"),
        "artifact_hashes_checked": len(report["artifacts_sha256"]), "max_objective_error": max(losses),
        "max_directional_gradient_error": max(gradients), "graphs_recomputed": len(errors),
        "max_executable_graph_error": max(errors), "saved_predictions_exact_after_json_normalization": True,
        "scope": "Frozen synthetic mechanism study, not detection performance; no refit or new API calls."}
    with (out / "REPORT.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
