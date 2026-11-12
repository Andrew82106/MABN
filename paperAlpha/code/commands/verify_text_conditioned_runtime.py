"""Replay saved typed extraction through v5; mechanism tests, not accuracy."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.text_conditioned_bn import (TextConditionedBN, compile_text_journal,
    PARAMETER_NAMES, BASE_SIZE, FEATURE_NAMES, CHANNEL_WIDTH, MODEL_SHA256)
from paperalpha_monitor.compact_action_bn import CompactCapabilityBN
from paperalpha_monitor.dynamic_joint_bn import JointTrace
from paperalpha_monitor.tool_capabilities import load_tool_registry
from audit_semantic_extraction import audit, read_rows, sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("text-model verification requires a new output directory")
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    if manifest.get("semantic_protocol") != "v2":
        raise ValueError("typed v2 extraction required")
    # Reusing an extractor means reusing its exact semantics, not merely a name.
    for name in ("semantic_evidence.py", "semantic_evidence_v2.py"):
        if sha(ROOT / 'code/src/paperalpha_runtime' / name) != sha(run / "source_snapshot/code/src/paperalpha_runtime" / name):
            raise ValueError("current and frozen extractor semantics differ; explicit migration needed")
    output.mkdir(parents=True)
    source_audit = audit(run, output / "SOURCE_AUDIT.json")
    frozen = importlib.import_module("paperalpha_semantics_audit_frozen.semantic_evidence_v2")
    annotations, parse_failures = {}, {}
    for row in read_rows(run / "api_responses.jsonl"):
        try:
            annotations[row["record_id"]] = frozen.response_annotation(row["response"])
        except (ValueError, TypeError, KeyError) as exc:
            parse_failures[row["record_id"]] = type(exc).__name__
    rows = read_rows(run / "runtime_events.jsonl")
    compiled = compile_text_journal(rows, annotations, tool_registry=load_tool_registry())
    packets = {p["current"]["record_id"]: p for p in read_rows(run / "input_packets.jsonl")}
    for step in compiled.steps:
        if step.revision and step.revision.evidence["record_id"] in packets:
            assert frozen.digest(packets[step.revision.evidence["record_id"]]) == step.revision.evidence["input_packet_sha256"]
    model = TextConditionedBN()
    rng = np.random.default_rng(20260919)
    vectors = [model.theta.copy(), *(model.theta + rng.normal(0, .5, len(model.theta)) for _ in range(8))]
    scores, diagnostics, max_legacy_error = [], [], 0.
    legacy_trace = JointTrace(compiled.variables, tuple(s.base for s in compiled.steps), compiled.workflow)
    for index, theta in enumerate(vectors):
        modes = {"full": {}, "masked_features": {"mask_text_features": True},
                 "no_messages": {"use_messages": False}, "no_text": {"use_text": False}}
        values, active_gradient = {}, np.zeros(len(theta), dtype=bool)
        for name, options in modes.items():
            current = TextConditionedBN(**options)
            engine = current._engine(theta, derivatives=name == "full")
            values[name] = []
            for step in compiled.steps:
                values[name].append(engine.apply(step, compiled.variables))
                if name == "full":
                    active_gradient |= np.abs(engine.gradient) > 1e-12
                    if index == 0:
                        scores.append({"sequence": step.sequence, "score": engine.probability,
                            "state_mass_sum": float(engine.distribution.sum()),
                            "retained_joint_states": int(engine.distribution.size),
                            "text_evidence": None if step.revision is None else step.revision.evidence,
                            "text_transition": copy.deepcopy(engine.last_revision)})
            values[name] = np.asarray(values[name])
        legacy = CompactCapabilityBN()
        legacy.theta = theta[:BASE_SIZE].copy()
        error = float(np.max(np.abs(legacy.predict([legacy_trace])[0] - values["no_text"])))
        max_legacy_error = max(max_legacy_error, error)
        diagnostics.append({"parameter_probe": index, "untrained_default": index == 0,
            "max_score_change_masking_text_features": float(np.max(np.abs(values["full"] - values["masked_features"]))),
            "max_score_change_disabling_messages": float(np.max(np.abs(values["full"] - values["no_messages"]))),
            "max_score_change_disabling_text_transitions": float(np.max(np.abs(values["full"] - values["no_text"]))),
            "active_score_derivatives_total": int(active_gradient.sum()),
            "active_score_derivatives_text": int(active_gradient[BASE_SIZE:].sum()),
            "active_parameter_names": [n for n, a in zip(PARAMETER_NAMES, active_gradient) if a]})
    # All-parameter finite differences on one predeclared non-default vector;
    # score derivatives only: this real task has no invented training label.
    theta = vectors[1]
    engine = model._engine(theta, derivatives=True)
    for step in compiled.steps:
        engine.apply(step, compiled.variables)
    numeric = approx_derivative(lambda t: model._forward(t, compiled)[0][-1], theta, method="3-point").ravel()
    derivative_error = float(np.max(np.abs(numeric - engine.gradient)))
    np.testing.assert_allclose(engine.gradient, numeric, rtol=3e-5, atol=4e-8)
    assert max_legacy_error < 2e-14
    # Batch-vs-incremental compiler/filter prefix agreement, with no future rows
    # or future annotations supplied to the prefix computation.
    from paperalpha_monitor.text_conditioned_bn import JournalTextCompiler
    compiler, online = JournalTextCompiler(tool_registry=load_tool_registry()), model._engine(theta)
    prefix_scores = []
    for row in rows:
        step = compiler.push(row, annotations.get(f"journal-{row['sequence']}"))
        if step is not None:
            prefix_scores.append(online.apply(step, compiler.variables))
    prefix_error = float(np.max(np.abs(np.asarray(prefix_scores) - model._forward(theta, compiled)[0])))
    assert prefix_error < 2e-14
    revisions = [s.revision for s in compiled.steps if s.revision is not None]
    accepted = sum(r.evidence["batch"]["accepted_items"] for r in revisions if r.evidence["batch"] is not None)
    final = model.explain(compiled)
    assert all(a["permission_status"] == "unknown" and a["accepted_check"] is None for a in final["actions"])
    snapshots = {}
    for path in [Path(__file__), ROOT / 'code/commands/audit_semantic_extraction.py',
                 *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
                 *(ROOT / 'knowledge/templates').glob("*.json")]:
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        snapshots[path.relative_to(ROOT).as_posix()] = sha(target)
    save(output / "prefix_predictions.json", scores)
    save(output / "parameter_probes.json", [t.tolist() for t in vectors])
    save(output / "sensitivity.json", diagnostics)
    save(output / "model_parameters.json", model.parameters())
    save(output / "final_explanation.json", final)
    report = {"status": "passed", "source_run": str(run), "source_manifest_sha256": sha(run / "RUN.json"),
        "source_audit": source_audit["status"], "model_specification_sha256": MODEL_SHA256,
        "model_parameters": len(PARAMETER_NAMES), "grouped_text_features": len(FEATURE_NAMES),
        "new_api_calls": 0, "model_events": len(compiled.steps), "text_revisions": len(revisions),
        "accepted_candidate_items": accepted, "parse_failures": parse_failures,
        "information_source_variables": len(compiled.variables), "retained_joint_states": final["retained_joint_states"],
        "actions": len(final["actions"]), "trusted_permission_checks": 0,
        "default_feature_mask_score_change": diagnostics[0]["max_score_change_masking_text_features"],
        "nondefault_max_feature_mask_score_change": max(d["max_score_change_masking_text_features"] for d in diagnostics[1:]),
        "nondefault_max_no_messages_score_change": max(d["max_score_change_disabling_messages"] for d in diagnostics[1:]),
        "max_no_text_vs_v4_error": max_legacy_error,
        "all_parameter_terminal_score_gradient_max_error": derivative_error,
        "incremental_vs_batch_prefix_max_error": prefix_error,
        "source_snapshot": snapshots,
        "scope": "Untrained mechanism/gradient diagnostics on an already-used real runtime. No risk labels, fitting, calibrated probabilities, accuracy, novelty, real-time warning or MAS-baseline superiority."}
    report["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "REPORT.json", report)
    return {k: v for k, v in report.items() if k not in {"source_snapshot", "artifact_sha256"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.out), ensure_ascii=False, indent=2))
