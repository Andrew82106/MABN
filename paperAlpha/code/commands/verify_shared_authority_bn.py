"""Frozen engineering study of shared authority; not a new MAS benchmark.

Two explicitly constructed frequency tables validate learnability. They are
aggregated hypothetical worlds, NOT 35/55 independent tasks or API episodes.
No held-out performance or superiority claim is authorized by this study.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logit

from harnessaudit_cohort import ROOT, sha, write_new, now
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/tests')]
from audit_delegation_joint_context import shared_root_fixture
from monitor_delegation_pilot import make_compiler
from paperalpha_runtime.workspace import resolve_historical_path
from paperalpha_monitor.delegation_conditioned_bn import DelegationEvidenceCompiler
from paperalpha_monitor.shared_authority_graph import AuthorityWindow, AuthorityProofGraph, SharedAuthorityTrace, SharedAuthorityCompiler
from paperalpha_monitor.shared_authority_joint_bn import SharedAuthorityJointBN
from paperalpha_monitor.typed_action_supervision import bce
from test_shared_authority_bn import bundle, active_bundle


def known_bad():
    old, records = shared_root_fixture(True)
    base = DelegationEvidenceCompiler(duties=copy.deepcopy(old.duties), task_id=old.task_id,
        root_principals=list(old.ledger.roots), trusted_observers=list(old.ledger.observers), coverage_complete=True,
        tool_registry=old.compact._logical._registry, workflow=old.compact.workflow)
    compiler = SharedAuthorityCompiler(base, AuthorityWindow("known-bad", 0, 100))
    for event in records:
        event = copy.deepcopy(event)
        if event.get("kind") == "grant" and event["payload"]["parent_id"] is None:
            event["payload"]["can_delegate"] = False
        compiler.push(event)
    return compiler.trace()


def visible_features(graph):
    # Same graph evidence as the BN, including shared identities. No prior
    # probabilities, world assignments, hidden outcomes or generated labels.
    counts = [sum(k in path for proof in graph.proofs.values() for path in proof.paths) for k in graph.facts]
    return np.asarray([1., len(graph.proofs), sum(not p.paths for p in graph.proofs.values()),
        sum(f.kind == 0 for f in graph.facts.values()), sum(f.kind == 1 for f in graph.facts.values()),
        sum(f.kind == 2 for f in graph.facts.values()), sum(len(p.paths) for p in graph.proofs.values()),
        max(counts, default=0), sum(counts)], dtype=float)


def weighted_logistic(cases, table, regularization=.0005):
    x = np.asarray([visible_features(cases[r["case"]].graph) for r in table])
    y, w = np.asarray([r["label"] for r in table]), np.asarray([r["count"] for r in table], dtype=float)
    w /= w.sum()
    def objective(theta):
        p = expit(x @ theta)
        loss = float(np.sum(w * (np.logaddexp(0., x @ theta) - y * (x @ theta))) + regularization * (theta @ theta))
        gradient = x.T @ (w * (p - y)) + 2 * regularization * theta
        return loss, gradient
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 200, "gtol": 1e-7, "ftol": 1e-11})
    return {"model": "same-observation graph-statistic logistic", "theta": result.x.tolist(),
        "optimizer": {"success": bool(result.success), "loss": float(result.fun), "iterations": int(result.nit)},
        "feature_names": ["bias", "action_count", "known_unsupported_actions", "uncertain_roots", "uncertain_delegations",
            "opaque_references", "candidate_path_count", "maximum_fact_reuse", "total_fact_uses"]}


def run(output):
    if output.exists():
        raise ValueError("new frozen study directory required")
    output.mkdir(parents=True)
    hashes = {}
    files = [Path(__file__), ROOT / 'code/commands/audit_delegation_joint_context.py', ROOT / 'code/commands/monitor_delegation_pilot.py',
        ROOT / 'code/commands/collect_delegation_api_pilot.py', ROOT / 'code/commands/collect_harnessaudit_api_pilot.py', ROOT / 'code/commands/harnessaudit_cohort.py',
        ROOT / 'code/tests/test_shared_authority_bn.py', ROOT / 'code/tests/test_delegation_evidence.py', ROOT / 'code/tests/test_qualified_delegation_cli.py',
        *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
        *(ROOT / 'knowledge/templates').glob("*.json"), ROOT / 'knowledge/risks/risk_catalog.json']
    for path in files:
        relative = path.relative_to(ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            f.write(path.read_bytes())
        hashes[relative.as_posix()] = sha(path)
    reference_table = [{"case": "shared", "label": 1, "count": 2}, {"case": "shared", "label": 0, "count": 8},
        {"case": "independent", "label": 1, "count": 9}, {"case": "independent", "label": 0, "count": 16}]
    system_table = [{"case": "known_good", "label": 0, "count": 10}, {"case": "known_bad", "label": 1, "count": 10},
        {"case": "shared", "label": 1, "count": 2}, {"case": "shared", "label": 0, "count": 8},
        {"case": "independent", "label": 1, "count": 1}, {"case": "independent", "label": 0, "count": 24}]
    plan = {"created_at": now(), "source_sha256": hashes, "reference_table": reference_table, "system_table": system_table,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "reference_target": "At least one action violates the reference-required policy.",
        "system_target": "Toy cascade outcome: two actions violate the reference-required policy.",
        "construction": "Frequency tables chosen from a declared static 0.2 per-root failure law; not collected or independent tasks.",
        "system_regularization": .0005, "system_max_iter": 150,
        "protocol": "Fit demonstration only. No test split, no API calls, no benchmark ranking. Information-channel checks use hand-authored input, not new real text."}
    write_new(output / "PLAN.json", plan)
    cases = {"shared": bundle(True), "independent": bundle(False), "known_good": bundle(True, complete=True), "known_bad": known_bad()}
    write_new(output / "graphs.json", {k: v.graph.specification() for k, v in cases.items()})
    analytic = {}
    for identity in ("shared", "independent"):
        graph = cases[identity].graph
        analytic[identity] = {"at_least_one": graph.query([logit(.2), -3., 0.], at_least=1)[0],
            "at_least_two": graph.query([logit(.2), -3., 0.], at_least=2)[0]}
    write_new(output / "analytic_structure_check.json", analytic)
    reference = SharedAuthorityJointBN(regularization=0)
    reference.fit([cases[r["case"]] for r in reference_table], [r["label"] for r in reference_table],
        weights=[r["count"] for r in reference_table], target="reference_failure_count")
    write_new(output / "reference_fit.json", reference.parameters())
    system = SharedAuthorityJointBN(regularization=plan["system_regularization"], max_iter=plan["system_max_iter"])
    system.fit([cases[r["case"]] for r in system_table], [r["label"] for r in system_table], weights=[r["count"] for r in system_table])
    write_new(output / "system_fit.json", system.parameters())
    logistic = weighted_logistic(cases, system_table, plan["system_regularization"])
    write_new(output / "logistic.json", logistic)
    predictions = []
    for name, trace in cases.items():
        _, _, _, _, failures = trace.graph.worlds([0., 0., 0.])
        any_label = (np.sum(list(failures.values()), axis=0) >= 2)
        rule = int(any_label[0]) if np.all(any_label == any_label[0]) else None
        predictions.append({"case": name, "system_bn": float(system.forward(trace)["probabilities"][-1]),
            "same_information_logistic": float(expit(visible_features(trace.graph) @ np.asarray(logistic["theta"]))),
            "visible_rule": rule, "rule_meaning": "Null means this observation contract cannot determine the cascade outcome."})
    write_new(output / "training_predictions.json", predictions)
    initial = SharedAuthorityJointBN()
    dynamic = initial.forward(active_bundle(), derivatives=True)
    transmission = [i for i, name in enumerate(initial.parameter_names) if name.startswith("transmission:")]
    actual = []
    for run_name in ("delegation_api_pilot_20260919", "stale_authority_api_pilot_20260919"):
        directory = resolve_historical_path("results/" + run_name)
        plan_api = json.loads((directory / "PLAN.json").read_text(encoding="utf-8"))
        for config in plan_api["cases"]:
            records = [json.loads(line) for line in (directory / "runs" / config["id"] / "monitor_events.jsonl").read_text(encoding="utf-8").splitlines()]
            compiler = SharedAuthorityCompiler(make_compiler(config), AuthorityWindow("diagnostic-" + config["id"], 0, max(r["sequence"] for r in records)))
            for record in records:
                step = compiler.push(record)
                if step is not None and step.base.dispatches:
                    actual.append({"case": config["id"], "source_observations_sha256": sha(directory / "runs" / config["id"] / "monitor_events.jsonl"),
                        "reference_any_failure": compiler.graph.query(reference.theta[reference.prior_slice])[0],
                        "facts": len(compiler.graph.facts), "point_calibrated": False,
                        "scope": "Post-hoc structural replay. No new live system-head prediction or accuracy claim."})
    write_new(output / "real_observation_structural_replay.json", actual)
    for relative, expected in hashes.items():
        assert sha(ROOT / relative) == expected and sha(output / "source_snapshot" / relative) == expected
    report = {"parameters": len(system.theta), "reference_optimizer": reference.optimizer,
        "reference_fitted_root_probability": float(expit(reference.theta[reference.base_count])),
        "system_optimizer": system.optimizer, "same_information_logistic_optimizer": logistic["optimizer"],
        "system_changed_parameter_count": int(np.sum(np.abs(system.theta - system.initial) > 1e-7)),
        "system_prior_logits": system.theta[system.prior_slice].tolist(), "system_joint_head": system.theta[-2:].tolist(),
        "active_manual_trace_transmission_gradient_max": float(np.max(np.abs(dynamic["gradient"][-1, transmission]))),
        "active_manual_trace_shared_prior_gradient": float(dynamic["gradient"][-1, initial.base_count]),
        "new_API_calls": 0, "independent_performance_evidence": False,
        "limitations": ["Static uncertainty window; no hidden revocation hazard model.", "Endpoint-only controlled fits; not real MAS generalization.",
            "Same-source roots assumed independent unless they are exactly the same primitive.", "Old local authorization/duty weights are bypassed by explicit proofs.",
            "Exact enumeration and full-prefix replay are limited to small graphs."],
        "artifacts_sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}}
    write_new(output / "REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts_sha256"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
