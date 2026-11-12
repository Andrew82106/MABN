"""Channel activation and a shared-root representation diagnostic, not accuracy.

The synthetic root comparison changes only a child's parent reference. Both
worlds have identical resolved local evidence and action bindings. A separate
declared hypothetical root-failure law illustrates what information the current
feature-pooled model cannot represent; it is NOT a label or fitted prior.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from collect_delegation_api_pilot import load
from harnessaudit_cohort import ROOT, sha, write_new
from monitor_delegation_pilot import make_compiler, restore
from paperalpha_monitor.delegation_conditioned_bn import DelegationEvidenceCompiler
from paperalpha_monitor.duty_conditioned_bn import DutyConditioner
from paperalpha_monitor.dynamic_joint_bn import state_bits
from paperalpha_runtime.duty_graph import public_packet
from paperalpha_runtime.delegation_pilot import configuration


def replay(model, config, records, *, messages=True, relations=True):
    compiler = make_compiler(config)
    local = copy.deepcopy(model)
    local.use_messages = messages
    if not relations:
        local.theta[list(local.registry.relation_indices)] = 0.
    engine, rows = local.engine(derivatives=True), []
    indices = [i for i, name in enumerate(local.parameter_names)
        if name.startswith(("source:", "adoption:", "transmission:", "text:"))]
    for record in records:
        step = compiler.push(record)
        if step is None:
            continue
        engine.apply(step, compiler.compact.variables)
        bits = state_bits(len(compiler.compact.variables))
        marginal = engine.distribution.sum(axis=1) @ bits
        for dispatch in step.base.dispatches:
            rows.append({"action_id": dispatch.action_id, "sequence": step.sequence,
                "probability": engine.probability,
                "maximum_process_or_source_marginal": float(max(marginal, default=0.)),
                "information_parameter_count": len(indices),
                "maximum_information_parameter_derivative": float(np.max(np.abs(engine.gradient[indices]))),
                "active_mechanisms": sorted({f.template_id for s in compiler.steps for f in s.mechanism_factors}),
                "text_revisions": sum(s.revision is not None for s in compiler.steps)})
    return rows


def shared_root_fixture(shared):
    config = configuration("shared-root-diagnostic", "normal")
    public = config["public_configuration"]
    worker = public["agents"].pop()
    for identity in ("worker-a", "worker-b"):
        item = copy.deepcopy(worker)
        item["role"] = identity
        public["agents"].append(item)
    packet, aliases = public_packet(public)
    annotation = {"duties": [], "dependencies": [], "requirements": [], "unresolved_actors": [], "more_possible": False}
    for i, actor in enumerate(public["agents"]):
        annotation["duties"].append({"id": "d" + str(i + 1), "actor": actor["role"],
            "kind": "coordination" if i == 0 else "payment_execution", "uncertain": False,
            "record_id": "role:" + actor["role"] + ":description", "quote": actor["description"], "occurrence": 0})
    registry = make_compiler(configuration("registry", "normal")).compact._logical._registry
    workflow = {"nodes": ["planner", "worker-a", "worker-b"],
        "edges": [["planner", "worker-a"], ["planner", "worker-b"]],
        "roles": {"planner": "reasoner", "worker-a": "executor", "worker-b": "executor"}}
    compiler = DelegationEvidenceCompiler(duties=DutyConditioner(packet, annotation, aliases),
        task_id=config["task_id"], root_principals=["user"], trusted_observers=[config["observer"]],
        coverage_complete=False, tool_registry=registry, workflow=workflow)
    scope = {"task_id": config["task_id"], "principal": "user", "tools": ["simulate_payment"],
        "resources": [config["requested_resource"]], "arguments_sha256": None}
    records = []
    for seq, identity, parent, issuer, recipient in (
        (1, "root-a", None, "user", "planner"), (2, "root-b", None, "user", "planner"),
        (3, "child-a", "root-a", "planner", "worker-a"),
        (4, "child-b", "root-a" if shared else "root-b", "planner", "worker-b")):
        records.append({"event_id": "g-" + identity, "sequence": seq, "kind": "grant", "observer": config["observer"],
            "authenticated_actor": issuer, "payload": {"grant_id": identity, "parent_id": parent, "issuer": issuer,
                "recipient": recipient, "scope": copy.deepcopy(scope), "can_delegate": parent is None, "valid_until_sequence": 100}})
    for seq, who, child in ((5, "worker-a", "child-a"), (6, "worker-b", "child-b")):
        records.append({"event_id": "payment-" + who, "sequence": seq, "event_type": "tool_call",
            "action_id": "action-" + who, "actor": who, "principal": "user", "tool": "simulate_payment",
            "resource": config["requested_resource"], "arguments": {"amount": "50"}, "grant_ids": [child]})
    for record in records:
        compiler.push(record)
    return compiler, records


def root_diagnostic(model):
    left, left_records = shared_root_fixture(True)
    right, right_records = shared_root_fixture(False)
    assert left_records[:3] == right_records[:3] and left_records[4:] == right_records[4:]
    assert left.steps[-1].base.actions == right.steps[-1].base.actions
    for identity in left.snapshots:
        assert left.snapshots[identity].features == right.snapshots[identity].features
    pairs = []
    # A fixed parameter stress check, not fitted models or selected seeds.
    for seed in (None, 1901, 1902, 1903):
        candidate = copy.deepcopy(model)
        if seed is not None:
            candidate.theta += np.random.default_rng(seed).normal(0, .3, len(candidate.theta))
        a, b = candidate.predict_episode(left.trace()), candidate.predict_episode(right.trace())
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=0.)
        pairs.append({"perturbation_seed": seed, "shared_root_scores": a, "separate_root_scores": b,
            "maximum_difference": float(np.max(np.abs(np.asarray(a) - b)))})
    p = .2
    return {"kind": "hand-authored structural diagnostic, not benchmark labels", "parameter_checks": pairs,
        "left_records": left_records, "right_records": right_records,
        "left_paths": {k: v.evidence["candidates"] for k, v in left.snapshots.items()},
        "right_paths": {k: v.evidence["candidates"] for k, v in right.snapshots.items()},
        "hypothetical_static_root_failure_law": {"assumed_per_root_failure": p,
            "shared_root_any_failure": p, "independent_roots_any_failure": 1 - (1 - p) ** 2,
            "assumptions": "A root has one fixed hidden status throughout this window; distinct roots independent; no other failures.",
            "empirically_estimated": False},
        "conclusion": "Root identities/paths survive in explanations but not joint uncertainty factors. Equal scores here are current factorization, not proven correct or wrong real-world probabilities."}


def audit(runs, output):
    if output.exists():
        raise ValueError("new audit output required")
    all_rows, manifests = [], []
    for run in runs:
        plan = load(run / "PLAN.json")
        model = restore(run / "model.json")
        manifests.append({"run": str(run), "plan_sha256": sha(run / "PLAN.json"), "model_sha256": sha(run / "model.json")})
        for config in plan["cases"]:
            path = run / "runs" / config["id"] / "monitor_events.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            original, no_messages, no_relations = (replay(model, config, records, **options)
                for options in ({}, {"messages": False}, {"relations": False}))
            for a, b, c in zip(original, no_messages, no_relations):
                all_rows.append({"case_id": config["id"], **a,
                    "message_toggle_absolute_difference": abs(a["probability"] - b["probability"]),
                    "zero_relation_weights_probability": c["probability"],
                    "note": "Fixed-parameter sensitivity, not retrained ablation or accuracy."})
    comparison = root_diagnostic(model)
    output.mkdir(parents=True)
    write_new(output / "channels.json", all_rows)
    write_new(output / "shared_roots.json", comparison)
    with (output / "audit_source.py").open("xb") as f:
        f.write(Path(__file__).read_bytes())
    report = {"source_runs": manifests, "actual_payment_actions": len(all_rows),
        "maximum_message_toggle_difference": max(r["message_toggle_absolute_difference"] for r in all_rows),
        "maximum_information_parameter_derivative": max(r["maximum_information_parameter_derivative"] for r in all_rows),
        "maximum_process_or_source_marginal": max(r["maximum_process_or_source_marginal"] for r in all_rows),
        "interpretation": "These pilots exercise authorization-chain observations, not active text/message uncertainty propagation. Shared-root uncertainty remains an explicit design gap.",
        "artifacts_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_new(output / "REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit([p.resolve() for p in args.run], args.out.resolve())
