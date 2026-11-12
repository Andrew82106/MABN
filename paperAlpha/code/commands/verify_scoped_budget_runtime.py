"""Public amount-profile verification and v6 real-prefix mechanism replay."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.scoped_budget_bn import (ScopedBudgetBN, compile_budget_journal, JournalBudgetCompiler,
    PARAMETER_NAMES, QUANTITY_OFFSET, MODEL_SHA256)
from paperalpha_monitor.scoped_quantities import load_quantity_profiles, DEFAULT_PROFILE
from paperalpha_monitor.text_conditioned_bn import TextConditionedBN, TextStep
from paperalpha_monitor.dynamic_joint_bn import JointTrace
from paperalpha_monitor.tool_capabilities import load_tool_registry
from audit_semantic_extraction import audit, sha, read_rows


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_public_profile(profile, api_requests):
    # Only names and public parameter descriptions from the first saved MAS
    # request are used; task prompt, arguments, responses and labels are ignored.
    with Path(api_requests).open(encoding="utf-8") as stream:
        first = json.loads(next(line for line in stream if line.strip()))
    tools = {tool["name"]: tool for tool in first["tools"]}
    checked = []
    for name, entry in profile.tools.items():
        properties = tools[name]["params_json_schema"]["properties"]
        expected = {f["name"]: f["description"] for f in entry["amount_fields"]}
        expected[entry["currency_field"]] = entry["currency_description"]
        expected[entry["limit_field"]] = entry["limit_description"]
        for field, description in expected.items():
            if properties[field]["description"] != description:
                raise ValueError(f"public parameter description mismatch: {name}.{field}")
            checked.append(f"{name}.{field}")
        if not all(field in properties for field in entry["scope_fields"]):
            raise ValueError("scope mapping is absent from public tool schema")
        if f"(default {entry['default_currency']})" not in entry["currency_description"]:
            raise ValueError("currency default lacks its public description")
    return {"matched_descriptions": checked, "source_sha256": sha(api_requests),
        "profile_sha256": profile.sha256, "actual_effect_or_permission_verified": False}


def verify(run, tool_requests, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("v6 verification requires a new output directory")
    profile = load_quantity_profiles()
    profile_check = check_public_profile(profile, tool_requests)
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    if manifest.get("semantic_protocol") != "v2":
        raise ValueError("typed v2 extraction required")
    for name in ("semantic_evidence.py", "semantic_evidence_v2.py"):
        if sha(ROOT / 'code/src/paperalpha_runtime' / name) != sha(run / "source_snapshot/code/src/paperalpha_runtime" / name):
            raise ValueError("extractor semantics changed; migration needed")
    output.mkdir(parents=True)
    source_audit = audit(run, output / "SOURCE_AUDIT.json")
    frozen = importlib.import_module("paperalpha_semantics_audit_frozen.semantic_evidence_v2")
    annotations, failures = {}, {}
    for row in read_rows(run / "api_responses.jsonl"):
        try:
            annotations[row["record_id"]] = frozen.response_annotation(row["response"])
        except (ValueError, TypeError, KeyError) as exc:
            failures[row["record_id"]] = type(exc).__name__
    rows = read_rows(run / "runtime_events.jsonl")
    context = {"tool_registry": load_tool_registry(), "quantity_profiles": profile}
    # No global contract is fabricated from the agent's own amount_limit or
    # imported from hidden benchmark policy. This real replay has none.
    compiled = compile_budget_journal(rows, annotations, **context)
    legacy = JointTrace(compiled.variables, tuple(TextStep(s.base, s.revision) for s in compiled.steps), compiled.workflow)
    model, old = ScopedBudgetBN(), TextConditionedBN()
    rng = np.random.default_rng(20260919)
    vectors = [model.theta.copy(), *(model.theta + rng.normal(0, .5, len(model.theta)) for _ in range(8))]
    diagnostics, max_equivalence = [], 0.
    for index, theta in enumerate(vectors):
        active = model._engine(theta, derivatives=True)
        disabled = ScopedBudgetBN(use_quantities=False)._engine(theta)
        active_scores, disabled_scores, gradient_flags = [], [], np.zeros(len(theta), dtype=bool)
        for step in compiled.steps:
            active_scores.append(active.apply(step, compiled.variables))
            disabled_scores.append(disabled.apply(step, compiled.variables))
            gradient_flags |= np.abs(active.gradient) > 1e-12
        old.theta = theta[:QUANTITY_OFFSET].copy()
        error = float(np.max(np.abs(old.predict([legacy])[0] - np.asarray(disabled_scores))))
        max_equivalence = max(max_equivalence, error)
        diagnostics.append({"probe": index, "default": index == 0,
            "max_score_change_disabling_numeric_factors": float(np.max(np.abs(np.asarray(active_scores) - disabled_scores))),
            "numeric_score_gradient_active": [name for name, flag in zip(PARAMETER_NAMES[QUANTITY_OFFSET:], gradient_flags[QUANTITY_OFFSET:]) if flag]})
    theta = vectors[1]
    engine = model._engine(theta, derivatives=True)
    for step in compiled.steps:
        engine.apply(step, compiled.variables)
    numeric = approx_derivative(lambda t: model._forward(t, compiled)[0][-1], theta, method="3-point").ravel()
    np.testing.assert_allclose(numeric, engine.gradient, rtol=3e-5, atol=4e-8)
    compiler, incremental = JournalBudgetCompiler(**context), model._engine(theta)
    scores = []
    for row in rows:
        step = compiler.push(row, annotations.get(f"journal-{row['sequence']}"))
        if step is not None:
            scores.append({"sequence": step.sequence, "parameter_probe": 1, "score": incremental.apply(step, compiler.variables)})
    prefix_error = float(np.max(np.abs(model._forward(theta, compiled)[0] - [s["score"] for s in scores])))
    assert prefix_error < 2e-14 and max_equivalence < 2e-14
    observed = compiled.steps[-1].quantity_observations
    actions = model.explain(compiled)["actions"]
    assert all(a["accepted_check"] is None and a["permission_status"] == "unknown" for a in actions)
    snapshots = {}
    for path in [Path(__file__), ROOT / 'code/commands/audit_semantic_extraction.py',
                 *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
                 *(ROOT / 'knowledge/templates').glob("*.json")]:
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshots[path.relative_to(ROOT).as_posix()] = sha(destination)
    for name, data in {"profile_check": profile_check, "quantities": observed, "sensitivity": diagnostics,
                       "parameter_probes": [t.tolist() for t in vectors], "model_parameters": model.parameters(),
                       "prefix_predictions": scores, "final_explanation": model.explain(compiled)}.items():
        save(output / f"{name}.json", data)
    report = {"status": "passed", "source_run": str(run), "source_manifest_sha256": sha(run / "RUN.json"),
        "source_audit": source_audit["status"], "model_specification_sha256": MODEL_SHA256,
        "source_tool_request_sha256": sha(tool_requests), "profile_sha256": profile.sha256,
        "artifact_parameter_probes": {"prefix_predictions.json": 1, "final_explanation.json": 0, "model_parameters.json": 0},
        "model_parameters": len(PARAMETER_NAMES), "model_events": len(compiled.steps),
        "numeric_profiled_actions": len(observed), "configured_global_contracts": 0,
        "self_comparisons": {k: v["self_declared_bound_comparison"] for k, v in observed.items()},
        "default_numeric_mask_score_difference": diagnostics[0]["max_score_change_disabling_numeric_factors"],
        "nondefault_max_numeric_mask_score_difference": max(d["max_score_change_disabling_numeric_factors"] for d in diagnostics[1:]),
        "numeric_active_score_parameters": diagnostics[0]["numeric_score_gradient_active"],
        "no_quantities_vs_v5_max_error": max_equivalence, "incremental_vs_batch_max_error": prefix_error,
        "terminal_score_gradient_max_error": float(np.max(np.abs(numeric - engine.gradient))),
        "parse_failures": failures, "new_api_calls": 0, "source_snapshot": snapshots,
        "scope": "Untrained declared-quantity mechanism replay, not actual spending, authorization truth, global-risk validation, accuracy or baseline superiority."}
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
