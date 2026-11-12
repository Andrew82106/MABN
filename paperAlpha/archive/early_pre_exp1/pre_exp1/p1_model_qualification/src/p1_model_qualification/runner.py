"""Manifest-first deterministic and real local qualification execution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from p1_benign.providers import DeterministicTestProvider, ModelProvider
from p1_benign.workflow import execute_episode

from .core import (
    CANDIDATE_MODELS,
    CONDITION,
    DEFAULT_CONTEXT,
    EXECUTION_MODE,
    PHASE,
    QualificationContext,
    QualificationEventRecorder,
    SCIENTIFIC_GATE,
    TASK_ORDER,
    TEST_DOUBLE_MODE,
    capture_provenance,
    load_static_inputs,
    new_run_id,
    normalize_outcome,
    read_json,
    sha256_file,
    stable_hash,
    utc_now,
    validate_run_id,
    write_json,
    write_jsonl,
    write_text,
)
from .providers import QualificationOllamaProvider, inspect_local_models


RUN_PREFIX = {
    TEST_DOUBLE_MODE: "P1-BENIGN-QUAL-DRY-",
    EXECUTION_MODE: {
        "qwen3:8b": "P1-BENIGN-QUAL-QWEN3-",
        "ministral-3:8b": "P1-BENIGN-QUAL-MINISTRAL3-",
    },
}


def _relative_output(path: Path, context: QualificationContext) -> str:
    return path.resolve().relative_to(context.outputs.data_root.parent.resolve()).as_posix()


def _render_report(manifest: dict[str, Any], outcomes: list[dict[str, Any]], validation_passed: bool | None) -> str:
    count = len(outcomes)
    successes = sum(row["task_success"] for row in outcomes)
    fields = sum(row["required_field_accuracy"] * 4 for row in outcomes)
    calls = sum(row["provider_call_count"] for row in outcomes)
    structured = sum(row["structured_output_count"] for row in outcomes)
    provider_duration = sum(row["total_provider_duration_ms"] for row in outcomes)
    wall_duration = sum(row["episode_wall_duration_ms"] for row in outcomes)
    return "\n".join(
        [
            f"# P1 local-model qualification run: {manifest['run_id']}",
            "",
            "This is a non-scientific model qualification record, not a P1 scientific result.",
            "",
            f"- phase: `{PHASE}`",
            f"- scientific gate: `{SCIENTIFIC_GATE}`",
            "- eligible_for_scientific_analysis: `false`",
            f"- model: `{manifest['provider']['model_id']}`",
            f"- episodes: `{count}`",
            f"- task successes: `{successes}/{count}`",
            f"- required fields correct: `{int(fields)}/{count * 4}`",
            f"- provider calls: `{calls}`",
            f"- structured outputs: `{structured}`",
            f"- provider duration ms: `{provider_duration:.3f}`",
            f"- wall duration ms: `{wall_duration:.3f}`",
            f"- validation: `{'PASS' if validation_passed else 'PENDING'}`",
            "- remote calls: `0`",
            "- model downloads: `0`",
            "",
        ]
    )


def _has_existing_real_run(model_id: str, context: QualificationContext) -> bool:
    if not context.outputs.manifests.exists():
        return False
    for path in context.outputs.manifests.glob("manifest_P1-BENIGN-QUAL-*.json"):
        try:
            manifest = read_json(path)
        except Exception:
            continue
        if (
            isinstance(manifest, dict)
            and manifest.get("execution_mode") == EXECUTION_MODE
            and isinstance(manifest.get("provider"), dict)
            and manifest["provider"].get("model_id") == model_id
        ):
            return True
    return False


def _assert_empty(paths: dict[str, Path]) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("run ID already owns one or more qualification artifacts")


def run_qualification(
    *,
    context: QualificationContext = DEFAULT_CONTEXT,
    provider: ModelProvider,
    execution_mode: str,
    episode_count: int = 3,
    run_id: str | None = None,
    allow_local_qualification: bool = False,
) -> dict[str, Any]:
    static = load_static_inputs(context)
    if episode_count != 3:
        raise ValueError("qualification must run exactly three episodes")
    if execution_mode not in {TEST_DOUBLE_MODE, EXECUTION_MODE}:
        raise ValueError("unsupported qualification execution mode")
    local_evidence: dict[str, Any] | None = None
    if execution_mode == TEST_DOUBLE_MODE:
        if type(provider) is not DeterministicTestProvider:
            raise ValueError("test-double mode requires the deterministic provider")
        prefix = RUN_PREFIX[TEST_DOUBLE_MODE]
    else:
        if type(provider) is not QualificationOllamaProvider:
            raise ValueError("local qualification requires the loopback Ollama adapter")
        if provider.model_id not in CANDIDATE_MODELS or allow_local_qualification is not True:
            raise PermissionError("explicit local qualification authorization is required")
        if provider.concurrency != 1 or provider.timeout_seconds != 90.0 or provider.decoding != {
            "temperature": 0,
            "sampling_seed": 20260731,
            "max_output_tokens": 256,
            "thinking": False,
            "frozen": True,
        }:
            raise ValueError("provider differs from the frozen local qualification configuration")
        if _has_existing_real_run(provider.model_id, context):
            raise FileExistsError("a real qualification run already exists for this candidate model")
        local_evidence = inspect_local_models()
        if local_evidence["missing_models"]:
            raise RuntimeError("candidate model inventory is incomplete; no model call was made")
        prefix = RUN_PREFIX[EXECUTION_MODE][provider.model_id]
    actual_run_id = run_id or new_run_id(prefix)
    validate_run_id(actual_run_id)
    if not actual_run_id.startswith(prefix):
        raise ValueError("run ID prefix does not match the selected execution mode/model")
    context.outputs.ensure()
    paths = context.outputs.artifact_paths(actual_run_id)
    _assert_empty(paths)
    provider_summary = provider.manifest_summary()
    provider_summary["endpoint_origin_hash"] = stable_hash(getattr(provider, "endpoint", None))
    provider_summary["credential_value_recorded"] = False
    provider_summary["hidden_reasoning_requested"] = False
    provider_summary["hidden_reasoning_logged"] = False
    provider_summary["model_inspection"] = local_evidence
    manifest = {
        "schema_version": "1.0.0",
        "run_id": actual_run_id,
        "phase": PHASE,
        "condition": CONDITION,
        "risk_seed_present": False,
        "intervention_applied": False,
        "execution_mode": execution_mode,
        "eligible_for_scientific_analysis": False,
        "scientific_gate": SCIENTIFIC_GATE,
        "status": "running",
        "failure_reason": None,
        "started_at": utc_now(),
        "completed_at": None,
        "task_instance_ids": list(TASK_ORDER),
        "provider": provider_summary,
        "budget": copy.deepcopy(static.qualification["budget"]),
        "output_files": {name: _relative_output(path, context) for name, path in paths.items()},
        "artifact_hashes": {},
        "provenance": capture_provenance(context),
        "validation": None,
    }
    write_json(paths["manifest"], manifest, context.outputs.data_root)
    recorder = QualificationEventRecorder(
        run_id=actual_run_id,
        execution_mode=execution_mode,
        path=paths["events"],
        data_root=context.outputs.data_root,
    )
    outcomes: list[dict[str, Any]] = []
    try:
        for episode_number, task_id in enumerate(TASK_ORDER, start=1):
            outcome = execute_episode(
                run_id=actual_run_id,
                episode_number=episode_number,
                task_instance_id=task_id,
                material=static.materials[task_id],
                internal_record=static.internal_records[task_id],
                expected_report=static.expected_reports[task_id]["report"],
                static=static.workflow,
                provider=provider,
                recorder=recorder,
            )
            outcomes.append(normalize_outcome(outcome, execution_mode))
        write_jsonl(paths["outcomes"], outcomes, context.outputs.data_root)
        from .replay import replay_run
        from .validation import validate_run

        replay = replay_run(actual_run_id, context=context, write_output=True)
        write_text(paths["run_report"], _render_report(manifest, outcomes, None), context.outputs.data_root)
        manifest["artifact_hashes"] = {
            name: sha256_file(paths[name]) for name in ("events", "outcomes", "replay", "run_report")
        }
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["replay_summary"] = {
            "passed": replay["passed"],
            "episodes": len(replay["episodes"]),
            "network_calls": replay["network_calls"],
        }
        write_json(paths["manifest"], manifest, context.outputs.data_root)
        validation = validate_run(actual_run_id, context=context, write_outputs=True)
        manifest["validation"] = {"passed": validation["passed"], "check_count": len(validation["checks"])}
        write_json(paths["manifest"], manifest, context.outputs.data_root)
        return {
            "run_id": actual_run_id,
            "manifest": manifest,
            "outcomes": outcomes,
            "replay": replay,
            "validation": validation,
        }
    except Exception as exc:
        manifest["status"] = "failed_implementation"
        manifest["failure_reason"] = type(exc).__name__
        manifest["completed_at"] = utc_now()
        write_json(paths["manifest"], manifest, context.outputs.data_root)
        raise
