"""Verify initial-role conditioning and literal references on frozen real runs.

No labels, fitting, risk accuracy, selection by score or new model API calls.
Random probes are fixed in advance and measure sensitivity, never improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from harnessaudit_cohort import ROOT, load_plan, sha, write_new
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.initial_task_context import from_initial_request, validate_roles, request_payload
from paperalpha_runtime.semantic_evidence import digest, response_annotation
from paperalpha_monitor.contextual_knowledge_input import ContextualKnowledgeCompiler, compile_contextual_journal
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN, compile_knowledge_journal
from paperalpha_monitor.tool_capabilities import load_tool_registry


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(plan_path, extraction, output):
    plan_path, extraction, output = plan_path.resolve(), extraction.resolve(), output.resolve()
    plan, extracted = load_plan(plan_path), read(extraction / "REPORT.json")
    if output.exists():
        raise ValueError("new replay directory required")
    if plan["plan_sha256"] != extracted["plan_sha256"]:
        raise ValueError("classification cohort differs")
    for name, expected in extracted["artifact_sha256"].items():
        if sha(extraction / name) != expected:
            raise ValueError(f"classification artifact changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    registry = load_tool_registry()
    model = KnowledgeScopedBN()
    rng = np.random.default_rng(20260919)
    probes = [model.theta.copy(), *(model.theta + rng.normal(0, .5, len(model.theta)) for _ in range(4))]
    write_new(output / "parameter_probes.json", {"seed": 20260919, "selection": "default plus four fixed Normal(0,0.5) perturbations; no fitting or score selection",
                                               "names": list(model.parameter_names), "vectors": [t.tolist() for t in probes]})
    results, token_totals = [], Counter()
    for task in plan["tasks"]:
        identity = task["task_id"]
        source = read(extraction / "inputs" / f"{identity}.json")
        classification = read(extraction / "classifications" / f"{identity}.json")
        response = read(extraction / "responses" / f"{identity}.json")
        context = from_initial_request(source["initial_request"], source["run_start"])
        annotation = response_annotation(response)
        assert context == source["context"], "initial context does not reconstruct"
        assert annotation == classification["annotation"], "annotation differs from actual model output"
        assert validate_roles(context, annotation) == classification["evidence"], "role evidence does not reconstruct"
        assert request_payload(context) == read(extraction / "requests" / f"{identity}.json"), "role request does not reconstruct"
        for key in ("input_tokens", "output_tokens"):
            token_totals[key] += response.get("usage", {}).get(key, 0)
        run = plan_path.parent / "runs" / identity
        request_path, journal_path = run / "api_requests.jsonl", run / "runtime_events.jsonl"
        manifest = read(run / "RUN.json")
        assert sha(request_path) == manifest["artifact_sha256"]["api_requests.jsonl"]
        assert sha(journal_path) == manifest["artifact_sha256"]["runtime_events.jsonl"]
        with request_path.open(encoding="utf-8") as handle:
            assert json.loads(next(handle)) == source["initial_request"], "not the actual first request"
        rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0] == source["run_start"], "different runtime roster"
        common = {"initial_request": source["initial_request"], "run_start": source["run_start"],
                  "role_annotation": annotation, "tool_registry": registry}
        predicted, evidence = compile_contextual_journal(rows, role_mode="predicted", **common)
        unknown, _ = compile_contextual_journal(rows, role_mode="unknown", **common)
        old = compile_knowledge_journal(rows, tool_registry=registry)
        old_error, role_effects = [], []
        for theta in probes:
            model.theta = theta.copy()
            a, b, c = model.predict([predicted, unknown, old])
            old_error.append(float(np.max(np.abs(b - c))))
            role_effects.append(float(np.max(np.abs(a - b))))
        assert max(old_error) == 0., "unknown mode changed v7"
        assert role_effects[0] == 0., "initial tied role tables should be equivalent"
        model.theta = probes[1].copy()
        compiler, engine, online = ContextualKnowledgeCompiler(role_mode="predicted", **common), model._engine(model.theta), []
        for row in rows:
            step = compiler.push(row)
            if step is not None:
                online.append(engine.apply(step, compiler.variables))
        error = float(np.max(np.abs(np.asarray(online) - model.predict([predicted])[0])))
        assert error < 1e-14, "incremental/batch difference"
        explanation = engine.explain(compiler.last_step, compiler.variables)
        original_actions = unknown.steps[-1].base.actions
        actions = predicted.steps[-1].base.actions
        assert len(actions) == len(original_actions)
        for left, right in zip(actions, original_actions):
            assert left["binding"] == right["binding"] and left["permission_status"] == right["permission_status"]
            assert left["accepted_check"] == right["accepted_check"]
        references = [r for v in evidence["action_resource_references"].values() for r in v["references"]]
        changed = sum(x.parameter_indices != y.parameter_indices for a, b in zip(predicted.steps, unknown.steps)
                      for x, y in zip(a.mechanism_factors, b.mechanism_factors) if x.template_id == "authorization")
        result = {"task_id": identity, "model_events": len(predicted.steps), "actions": len(actions),
                  "effective_roles": evidence["effective_node_roles"], "authorization_factors_selecting_different_role_cells": changed,
                  "permissions_and_authenticated_bindings_unchanged": True,
                  "typed_reference_observations": len(references),
                  "reference_literal_comparisons": dict(Counter(r["initial_task_literal_comparison"] for r in references)),
                  "unknown_mode_v7_score_error": max(old_error), "default_role_score_effect": role_effects[0],
                  "fixed_probe_role_score_effects": role_effects[1:], "incremental_batch_error": error,
                  "input_journal_sha256": sha(journal_path), "context_sha256": context["context_sha256"],
                  "scope": "Input conditioning and sensitivity, not role accuracy, safe permissions or detection improvement."}
        write_new(output / "explanations" / f"{identity}.json", {"probe_index": 1, "model": explanation, "input_context": evidence})
        results.append(result)
    assert all(token_totals[key] == extracted[key] for key in token_totals), "API usage does not reproduce"
    sources = [Path(__file__), ROOT / 'code/commands/harnessaudit_cohort.py', *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"),
               *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"), *(ROOT / 'knowledge/templates').glob("*.json"), ROOT / 'knowledge/risks/risk_catalog.json']
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        hashes[relative.as_posix()] = sha(target)
    report = {"status": "passed", "model": "KnowledgeScopedBN-v7 with initial-role-classifier inputs", "parameter_count": 345,
              "plan_sha256": plan["plan_sha256"], "extraction_report_sha256": sha(extraction / "REPORT.json"),
              "extraction_artifacts_verified": len(extracted["artifact_sha256"]), "runs": results,
              "new_api_calls_in_verification": 0, "fitted": False, "source_sha256": hashes,
              "qualification": "Offline initial-context conditioning. Only roles affect existing CPT selection; task-resource mentions are not scoring features or grants. No new efficacy conclusion."}
    report["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.plan, args.extraction, args.out)
    print(json.dumps(report["runs"], ensure_ascii=False, indent=2))
