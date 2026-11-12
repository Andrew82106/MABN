"""Versioned, no-provider repair audit for an existing scientific run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .core import (
    CONDITION,
    PHASE,
    DATA_ROLE_DRY,
    DATA_ROLE_PILOT,
    DEFAULT_CONTEXT,
    ScientificContext,
    capture_provenance,
    labels,
    read_json,
    read_jsonl,
    require_generated_path,
    sha256_file,
    stable_hash,
    write_json,
    write_text,
)
from .reporting import aggregate_metrics, render_gate_report, render_run_report, threshold_rows, candidate_conclusion


REPAIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _validation_report(result: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [f"# P1 scientific benign validation repair audit — {result.get('run_id')}", "", "本文件是版本化派生审计，不覆盖原 pilot validation。", "", *[f"- {key}: `{manifest.get(key)}`" for key in ("phase", "condition", "risk_seed_present", "intervention_applied", "data_role", "eligible_for_scientific_analysis", "scientific_analysis_scope", "eligible_for_p1_gate_analysis", "eligible_for_confirmatory_analysis", "eligible_for_causal_effect_analysis", "scientific_gate")], f"- passed: `{result.get('passed')}`", "", "| check | result |", "|---|:---:|"]
    lines.extend(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in result.get("checks", {}).items())
    if result.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in result["errors"]]])
    return "\n".join(lines) + "\n"


def write_repair_audit(run_id: str, *, context: ScientificContext = DEFAULT_CONTEXT, repair_id: str, validation: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    if not REPAIR_ID_PATTERN.fullmatch(repair_id):
        raise ValueError("repair ID must be an alphanumeric, underscore or hyphen token")
    source_paths = context.outputs.artifact_paths(run_id)
    source_manifest = read_json(source_paths["manifest"])
    events = read_jsonl(source_paths["events"])
    outcomes = read_jsonl(source_paths["outcomes"])
    if not isinstance(source_manifest, dict):
        raise ValueError("source manifest is malformed")
    data_role = source_manifest.get("data_role")
    if data_role not in {DATA_ROLE_DRY, DATA_ROLE_PILOT}:
        raise ValueError("source manifest data role is malformed")
    static = __import__("p1_scientific_benign.core", fromlist=["load_static_inputs"]).load_static_inputs(context)
    root = context.outputs.root / "acceptance_repair" / repair_id
    require_generated_path(root, context.outputs.data_root)
    root.mkdir(parents=True, exist_ok=False)
    source_hashes = {key: sha256_file(path) for key, path in source_paths.items() if path.exists()}
    metrics = aggregate_metrics(outcomes, events=events, static=static)
    rows = threshold_rows(metrics, validation, replay)
    provenance = capture_provenance(context)
    source_execution_mode = source_manifest.get("execution_mode")
    audit_manifest = {
        "schema_version": "1.0.0",
        "artifact_kind": "derived_repair_audit",
        "is_derived_audit": True,
        "is_new_p1_run": False,
        "run_id": f"REPAIR-{repair_id}-{run_id}",
        "audit_run_id": f"REPAIR-{repair_id}-{run_id}",
        "derived_from_run_id": run_id,
        "source_run_id": run_id,
        "repair_id": repair_id,
        "phase": PHASE,
        "condition": CONDITION,
        "risk_seed_present": False,
        "intervention_applied": False,
        "source_execution_mode": source_execution_mode,
        "execution_mode": "derived_repair_audit",
        "audit_mode": "no_provider_reinterpretation",
        **labels(data_role),
        "p1_go": False,
        "candidate_conclusion": candidate_conclusion(rows, repair_audit=True),
        "no_provider_calls": True,
        "network_calls": 0,
        "source_artifact_hashes": source_hashes,
        "source_manifest_recorded_artifact_hashes": source_manifest.get("artifact_hashes", {}),
        "source_manifest_provenance_schema_present": isinstance(source_manifest.get("provenance", {}).get("shared_schema_files"), dict),
        "source_manifest_provenance_historical_fact": "original manifest predates shared schema provenance field; original manifest was not rewritten",
        "repair_tool_provenance": provenance,
        "metrics": metrics,
        "thresholds": rows,
        "validation": validation,
        "replay": {"passed": replay.get("passed"), "episodes": len(replay.get("episodes", [])), "live_provider_calls": replay.get("live_provider_calls", 0), "network_calls": replay.get("network_calls", 0)},
    }
    output = {
        "validation": root / "validation.json",
        "replay": root / "replay.json",
        "run_report": root / "run_report.md",
        "p1_gate_report": root / "p1_gate_report.md",
        "source_hashes": root / "source_hashes.json",
        "derived_audit_identity": root / "derived_audit_identity.json",
        "derived_audit_note": root / "DERIVED_AUDIT.md",
        "manifest": root / "manifest.json",
    }
    for path in output.values():
        require_generated_path(path, context.outputs.data_root)
    write_json(output["validation"], validation, context.outputs.data_root)
    write_json(output["replay"], replay, context.outputs.data_root)
    identity = {
        "artifact_kind": "derived_repair_audit",
        "is_derived_audit": True,
        "is_new_p1_run": False,
        "source_run_id": run_id,
        "source_execution_mode": source_execution_mode,
        "execution_mode": "derived_repair_audit",
        "audit_mode": "no_provider_reinterpretation",
        "repair_id": repair_id,
        "no_provider_calls": True,
        "network_calls": 0,
    }
    write_json(output["source_hashes"], {**identity, "derived_from_run_id": run_id, "source_artifact_hashes": source_hashes}, context.outputs.data_root)
    write_json(output["derived_audit_identity"], identity, context.outputs.data_root)
    write_text(output["derived_audit_note"], "# DERIVED AUDIT - NOT A NEW P1 RUN\n\nThis directory is a derived repair audit of an existing source pilot. It does not create a new P1 run and does not modify the source run.\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in identity.items()) + "\n", context.outputs.data_root)
    report_manifest = {**source_manifest, **audit_manifest}
    write_text(output["run_report"], f"# DERIVED REPAIR AUDIT - NOT A NEW P1 RUN\n\nsource_run_id={run_id}\nThis report is a derived audit of source run `{run_id}`; it is not a new P1 run.\n\n" + render_run_report(report_manifest, outcomes, replay, events=events, static=static), context.outputs.data_root)
    write_text(output["p1_gate_report"], f"# DERIVED REPAIR AUDIT GATE REPORT - NOT A NEW P1 RUN\n\nsource_run_id={run_id}\nThis report is a derived audit of source run `{run_id}`; it is not a new P1 run.\n\n" + render_gate_report(report_manifest, outcomes, replay, validation, events=events, static=static, repair_audit=True), context.outputs.data_root)
    audit_manifest["output_files"] = {key: path.relative_to(context.outputs.root.parent).as_posix() for key, path in output.items()}
    audit_manifest["artifact_hashes"] = {key: sha256_file(path) for key, path in output.items() if key != "manifest"}
    write_json(output["manifest"], audit_manifest, context.outputs.data_root)
    return {"repair_id": repair_id, "repair_root": root.as_posix(), "manifest": audit_manifest, "output_files": {key: path.as_posix() for key, path in output.items()}, "metrics": metrics, "thresholds": rows, "validation": validation, "replay": replay}


__all__ = ["REPAIR_ID_PATTERN", "write_repair_audit"]
