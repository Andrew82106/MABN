"""Freeze/recompute conditional scenario queries; no API data or accuracy fit."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN
from paperalpha_monitor.branch_workflow_risk import BranchWorkflowRiskBN
from workflow_risk_fixtures import fixture as action_fixture
from branch_workflow_fixtures import fixture as branch_fixture
from scalar_binary_graph import evaluate


def write(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ("code/src", "knowledge"):
        shutil.copytree(ROOT / name, out / "source" / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    scripts = ["audit_event_conditions.py", "workflow_risk_fixtures.py", "branch_workflow_fixtures.py",
               "branch_publisher_fixtures.py", "publisher_view_fixtures.py", "policy_privacy_fixtures.py", "scalar_binary_graph.py"]
    for name in scripts:
        target = out / "source/code/commands" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'code/commands' / name, target)
    write(out / "PLAN.json", {"scope": "conditional query math and semantics, not real performance",
          "sources_sha256": {str(p.relative_to(out / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((out / "source").rglob("*")) if p.is_file()},
          "new_api_calls": 0, "read_old_test": False, "fit_real_model": False})
    print(json.dumps({"prepared": str(out)}))


def run(out):
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    assert Path(__file__).resolve() == (out / "source/code/commands/audit_event_conditions.py").resolve()
    for name, digest in plan["sources_sha256"].items():
        assert hashlib.sha256((out / "source" / name).read_bytes()).hexdigest() == digest
    model = BranchWorkflowRiskBN(regularization=0.)
    model.theta += np.random.default_rng(919).normal(0, .2, len(model.theta))
    start = model.branch_model.presence.boost_start
    model.theta[start:start + 2] = [2., 3.]
    model.theta[model.branch_model.n - 1] = 1.3
    rows, max_graph, max_gradient = [], 0., 0.
    for kind in ("known", "invalid", "root", "delegation", "opaque"):
        sample = action_fixture(authority=kind, combination=True)
        action_model = WorkflowRiskBN()
        action_model.theta = np.random.default_rng(18).normal(0, .5, len(action_model.theta))
        for bits in product((0, 1), repeat=2):
            conditions = dict(zip(sample.action_ids, bits))
            result = action_model.predict(sample, event_conditions=conditions)
            for query, expected in result["probabilities"].items():
                actual = evaluate(action_model.probability_graph(sample, output=query, event_conditions=conditions))
                max_graph = max(max_graph, abs(actual - expected))
                rows.append({"family": "actions", "authority": kind, "conditions": conditions, "query": query, "production": expected, "scalar": actual})
    sample = branch_fixture(current="root_unknown", payment=True)
    for conditions in ({"release-0": 1}, {"release-1": 0, "payment": 1}, {"release-0": 1, "release-1": 1, "payment": 0}):
        result = model.predict(sample, event_conditions=conditions, derivatives=True)
        for query, expected in result["probabilities"].items():
            actual = evaluate(model.probability_graph(sample, output=query, event_conditions=conditions))
            max_graph = max(max_graph, abs(actual - expected))
            rows.append({"family": "branch", "conditions": conditions, "query": query, "production": expected, "scalar": actual})
        numeric = approx_derivative(lambda t: model.predict(sample, theta=t, event_conditions=conditions)["probabilities"]["system:any"], model.theta).ravel()
        max_gradient = max(max_gradient, float(np.max(np.abs(numeric - result["gradients"]["system:any"]))))
    assert max_graph < 1e-12 and max_gradient < 1e-7
    write(out / "queries.json", rows)
    write(out / "constructed_model.json", model.parameters())
    report = {"conditional_graph_queries": len(rows), "max_graph_error": max_graph,
              "gradient_parameter_count": len(model.theta), "gradient_directions": 3 * len(model.theta),
              "max_gradient_error": max_gradient, "new_accuracy_evaluation": False, "new_api_calls": 0,
              "source_files_checked": len(plan["sources_sha256"]),
              "interpretation": "Exact conditional scenario queries preserve parent likelihoods and subtract normalizer gradients. Not interventions, observed facts, or improved predictions."}
    write(out / "REPORT.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.out.resolve())
