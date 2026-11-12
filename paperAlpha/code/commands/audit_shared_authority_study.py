"""Independent saved-fit loss/gradient reconstruction for the shared BN study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from harnessaudit_cohort import ROOT, write_new
from paperalpha_runtime.workspace import resolve_historical_path
from collect_delegation_api_pilot import load
from verify_shared_authority_bn import known_bad, visible_features
from test_shared_authority_bn import bundle
from paperalpha_monitor.shared_authority_joint_bn import SharedAuthorityJointBN


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, out):
    if out.exists():
        raise ValueError("new audit output required")
    plan, report = [json.loads((run / name).read_text(encoding="utf-8")) for name in ("PLAN.json", "REPORT.json")]
    for relative, expected in report["artifacts_sha256"].items():
        assert hash_file(run / relative) == expected
    for relative, expected in plan["source_sha256"].items():
        assert hash_file(ROOT / relative) == expected
    cases = {"shared": bundle(True), "independent": bundle(False), "known_good": bundle(True, complete=True), "known_bad": known_bad()}
    assert {k: v.graph.specification() for k, v in cases.items()} == json.loads((run / "graphs.json").read_text(encoding="utf-8"))
    saved = json.loads((run / "system_fit.json").read_text(encoding="utf-8"))
    model = SharedAuthorityJointBN(regularization=plan["system_regularization"], max_iter=plan["system_max_iter"])
    assert saved["parameter_names"] == list(model.parameter_names) and saved["registry_sha256"] == model.registry.sha256
    model.theta = np.asarray(saved["theta"])
    table = plan["system_table"]
    traces, labels, weights = [cases[r["case"]] for r in table], [r["label"] for r in table], [r["count"] for r in table]
    loss, gradient = model.loss_gradient(model.theta, traces, labels, weights=weights)
    assert abs(loss - saved["optimizer"]["loss"]) < 1e-12
    logistic = json.loads((run / "logistic.json").read_text(encoding="utf-8"))
    expected = json.loads((run / "training_predictions.json").read_text(encoding="utf-8"))
    maximum = 0.
    by_case = {}
    for row in expected:
        trace = cases[row["case"]]
        p = float(model.forward(trace)["probabilities"][-1])
        q = float(expit(visible_features(trace.graph) @ np.asarray(logistic["theta"])))
        maximum = max(maximum, abs(p - row["system_bn"]), abs(q - row["same_information_logistic"]))
        by_case[row["case"]] = (p, q)
    assert maximum < 1e-12
    w = np.asarray(weights, dtype=float) / sum(weights)
    raw_nll, briers = {}, {}
    for name, position in (("system_bn", 0), ("same_information_logistic", 1)):
        p = np.asarray([by_case[r["case"]][position] for r in table])
        y = np.asarray(labels)
        raw_nll[name] = float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))))
        briers[name] = float(w @ (p - y) ** 2)
    assert abs(raw_nll["system_bn"] + model.regularization * np.sum((model.theta - model.initial) ** 2) - loss) < 1e-12
    errors = []
    for seed in range(1920, 1926):
        direction = np.random.default_rng(seed).normal(0, 1, len(model.theta))
        direction /= np.linalg.norm(direction)
        high = model.loss_gradient(model.theta + direction * 1e-5, traces, labels, weights=weights)[0]
        low = model.loss_gradient(model.theta - direction * 1e-5, traces, labels, weights=weights)[0]
        errors.append(abs((high - low) / 2e-5 - gradient @ direction))
    assert max(errors) < 1e-7
    parents = []
    for name in ("delegation_api_pilot_20260919", "stale_authority_api_pilot_20260919"):
        directory = resolve_historical_path("results/" + name)
        api_plan = load(directory / "PLAN.json")
        for config in api_plan["cases"]:
            case = directory / "runs" / config["id"]
            for file, h in json.loads((case / "ARTIFACTS.json").read_text(encoding="utf-8")).items():
                assert hash_file(case / file) == h
        parents.append(name)
    out.mkdir(parents=True)
    with (out / "audit_source.py").open("xb") as f:
        f.write(Path(__file__).read_bytes())
    result = {"study_report_sha256": hash_file(run / "REPORT.json"), "study_artifacts_checked": len(report["artifacts_sha256"]),
        "prediction_max_error": maximum, "fit_loss_max_error": abs(loss - saved["optimizer"]["loss"]),
        "directional_gradient_max_error": max(errors), "training_frequency_table_nll": raw_nll,
        "training_frequency_table_brier": briers, "unchanged_original_API_cohorts": parents,
        "qualification": "Reproduction/gradient evidence and descriptive fitting errors only; no independent test performance.",
        "audit_source_sha256": hash_file(out / "audit_source.py")}
    write_new(out / "REPORT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.out.resolve())
