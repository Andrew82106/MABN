"""Untrained real-journal validation of knowledge-selected local BN fragments."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN, JournalKnowledgeCompiler, compile_knowledge_journal
from paperalpha_monitor.text_conditioned_bn import TextConditionedBN, TextStep
from paperalpha_monitor.dynamic_joint_bn import JointTrace
from paperalpha_monitor.tool_capabilities import load_tool_registry
from paperalpha_monitor.scoped_quantities import load_quantity_profiles
from audit_semantic_extraction import audit, read_rows, sha
from verify_scoped_budget_runtime import check_public_profile, save


def verify(run, tool_requests, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("v7 verification requires a new directory")
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    if manifest.get("semantic_protocol") != "v2":
        raise ValueError("requires original typed extraction v2")
    for name in ("semantic_evidence.py", "semantic_evidence_v2.py"):
        if sha(ROOT / 'code/src/paperalpha_runtime' / name) != sha(run / "source_snapshot/code/src/paperalpha_runtime" / name):
            raise ValueError("extractor changed; explicit migration required")
    profile = load_quantity_profiles()
    profile_check = check_public_profile(profile, tool_requests)
    output.mkdir(parents=True)
    source_audit = audit(run, output / "SOURCE_AUDIT.json")
    frozen = importlib.import_module("paperalpha_semantics_audit_frozen.semantic_evidence_v2")
    annotations, failures = {}, {}
    for row in read_rows(run / "api_responses.jsonl"):
        try:
            annotations[row["record_id"]] = frozen.response_annotation(row["response"])
        except (ValueError, KeyError, TypeError) as exc:
            failures[row["record_id"]] = type(exc).__name__
    rows = read_rows(run / "runtime_events.jsonl")
    context = {"tool_registry": load_tool_registry(), "quantity_profiles": profile}
    trace = compile_knowledge_journal(rows, annotations, **context)
    legacy = JointTrace(trace.variables, tuple(TextStep(s.base, s.revision) for s in trace.steps), trace.workflow)
    model, old = KnowledgeScopedBN(), TextConditionedBN()
    rng = np.random.default_rng(20260919)
    probes = [model.theta.copy(), *(model.theta + rng.normal(0, .5, len(model.theta)) for _ in range(8))]
    equivalence_error, authority_error, effects = 0., 0., []
    for n, theta in enumerate(probes):
        with_nodes = model._engine(theta, derivatives=True)
        masked = theta.copy()
        for indices in model.registry.head_indices.values():
            masked[list(indices)] = 0.
        no_numeric_readout = model._engine(masked)
        old.theta = theta[:326].copy()
        old_engine = old._engine(old.theta)
        changed, gradient_flags = [], np.zeros(len(theta), dtype=bool)
        for step, old_step in zip(trace.steps, legacy.steps):
            active = with_nodes.apply(step, trace.variables)
            base = no_numeric_readout.apply(step, trace.variables)
            expected = old_engine.apply(old_step, trace.variables)
            equivalence_error = max(equivalence_error, abs(base - expected))
            marginalized = with_nodes.distribution.reshape(-1, 3, 3, 3).sum(axis=(2, 3))
            authority_error = max(authority_error, float(np.max(np.abs(marginalized - old_engine.distribution))))
            gradient_flags |= np.abs(with_nodes.gradient) > 1e-12
            changed.append(abs(active - base))
        effects.append({"probe": n, "max_numeric_readout_effect": max(changed),
            "active_new_gradient_parameters": [name for name, active in zip(model.parameter_names[326:], gradient_flags[326:]) if active]})
    assert equivalence_error < 3e-14 and authority_error < 3e-14
    theta = probes[1]
    engine = model._engine(theta, derivatives=True)
    for step in trace.steps:
        engine.apply(step, trace.variables)
    numeric = approx_derivative(lambda t: model._forward(t, trace)[0][-1], theta).ravel()
    gradient_error = float(np.max(np.abs(numeric - engine.gradient)))
    np.testing.assert_allclose(numeric, engine.gradient, rtol=3e-5, atol=5e-8)
    compiler, online, scores = JournalKnowledgeCompiler(**context), model._engine(theta), []
    for row in rows:
        step = compiler.push(row, annotations.get(f"journal-{row['sequence']}"))
        if step is not None:
            scores.append({"sequence": step.sequence, "parameter_probe": 1, "score": online.apply(step, compiler.variables)})
    online_error = float(np.max(np.abs(model._forward(theta, trace)[0] - [x["score"] for x in scores])))
    assert online_error < 3e-14
    counts = Counter(f.template_id for step in trace.steps for f in step.mechanism_factors)
    explanation = model.explain(trace)
    assert all(a["permission_status"] == "unknown" and a["accepted_check"] is None for a in explanation["actions"])
    assert explanation["mechanisms"]["cumulative_budget"]["concern_probability"] is None
    snapshots = {}
    for source in [Path(__file__), ROOT / 'code/commands/verify_scoped_budget_runtime.py', ROOT / 'code/commands/audit_semantic_extraction.py',
                   *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
                   *(ROOT / 'knowledge/templates').glob("*.json"), ROOT / 'knowledge/risks/risk_catalog.json']:
        destination = output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshots[source.relative_to(ROOT).as_posix()] = sha(destination)
    for name, value in {"model_parameters": model.parameters(), "parameter_probes": [x.tolist() for x in probes],
                        "sensitivity": effects, "prefix_predictions": scores, "final_explanation": explanation,
                        "profile_check": profile_check}.items():
        save(output / (name + ".json"), value)
    report = {"status": "passed", "source_run": str(run), "source_manifest_sha256": sha(run / "RUN.json"),
        "source_audit": source_audit["status"], "registry_sha256": model.registry.sha256, "model_parameters": len(model.theta),
        "model_events": len(trace.steps), "actions": len(explanation["actions"]), "instantiated_factors": dict(counts),
        "configured_global_contracts": 0, "retained_joint_states": explanation["retained_joint_states"],
        "no_numeric_readout_vs_v5_error": equivalence_error, "authorization_process_marginal_vs_v5_error": authority_error,
        "default_numeric_readout_effect": effects[0]["max_numeric_readout_effect"],
        "nondefault_max_numeric_readout_effect": max(e["max_numeric_readout_effect"] for e in effects[1:]),
        "terminal_score_gradient_max_error": gradient_error, "incremental_vs_batch_error": online_error,
        "parse_failures": failures, "new_api_calls": 0,
        "artifact_parameter_probes": {"prefix_predictions.json": 1, "final_explanation.json": 0, "model_parameters.json": 0},
        "scope": "Untrained structural/gradient replay, not independent semantic validation, accuracy, early-warning or baseline superiority.",
        "source_snapshot": snapshots}
    report["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "REPORT.json", report)
    return {k: v for k, v in report.items() if k not in {"source_snapshot", "artifact_sha256"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tool-request-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.tool_request_log, args.out), ensure_ascii=False, indent=2))
