"""P1 run orchestration with manifest-first writes and fixed budgets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .configuration import (
    frozen_budget_for_mode,
    load_static_inputs,
    provider_catalog_entry,
)
from .core import (
    CONDITION,
    INTERVENTION_APPLIED,
    LIVE_MODEL,
    LOCAL_MODEL_SHAKEDOWN,
    NON_SCIENTIFIC_MODES,
    PHASE,
    RISK_SEED_PRESENT,
    SCHEMA_VERSION,
    SCIENTIFIC_GATE,
    TEST_DOUBLE,
    EventRecorder,
    ProjectContext,
    new_run_id,
    sha256_file,
    stable_hash,
    utc_now,
    validate_run_id,
    write_json,
    write_jsonl,
    write_text,
)
from .provenance import capture_provenance
from .providers import (
    DeterministicTestProvider,
    ModelProvider,
    OllamaLocalProvider,
    ollama_model_is_installed,
)
from .reporting import aggregate_metrics, render_run_report
from .replay import replay_run
from .validation import _validate_run_preliminary, validate_run
from .workflow import execute_episode


RUN_PREFIX = {
    TEST_DOUBLE: "P1-BENIGN-DRY-",
    LOCAL_MODEL_SHAKEDOWN: "P1-BENIGN-LOCAL-",
    LIVE_MODEL: "P1-BENIGN-LIVE-",
}


def _relative_output(path: Path, context: ProjectContext) -> str:
    paper_root = context.outputs.data_root.parent
    return path.resolve().relative_to(paper_root.resolve()).as_posix()


def _mode_provider_valid(
    execution_mode: str,
    provider: ModelProvider,
    catalog: dict[str, Any] | None,
    expected_decoding: dict[str, Any],
) -> bool:
    if catalog is None:
        return False
    if execution_mode == TEST_DOUBLE:
        return (
            type(provider) is DeterministicTestProvider
            and catalog.get("provider_type") == "test_double"
            and provider.provider_id == "deterministic_test"
            and provider.model_id == "deterministic-p1-v1"
            and provider.endpoint_identifier == "none"
            and provider.concurrency == 1
            and provider.is_live_provider is False
            and provider.decoding == expected_decoding
        )
    if execution_mode == LOCAL_MODEL_SHAKEDOWN:
        return (
            type(provider) is OllamaLocalProvider
            and catalog.get("provider_type") == "local"
            and provider.provider_id == "ollama_local"
            and provider.model_id == "qwen3:8b"
            and provider.endpoint_identifier == "loopback_ollama"
            and provider.endpoint == catalog.get("endpoint")
            and provider.concurrency == 1
            and provider.is_live_provider is False
            and provider.timeout_seconds == 90.0
            and provider.decoding == expected_decoding
        )
    return False


def run_p1(
    *,
    context: ProjectContext,
    provider: ModelProvider,
    execution_mode: str,
    episode_count: int,
    run_id: str | None = None,
    task_instance_ids: Sequence[str] | None = None,
    eligible_for_scientific_analysis: bool = False,
    root_seed: int = 20260731,
    allow_local_shakedown: bool = False,
) -> dict[str, Any]:
    if execution_mode not in RUN_PREFIX:
        raise ValueError(f"Unsupported P1 execution mode: {execution_mode}")
    if execution_mode == LIVE_MODEL:
        raise PermissionError(
            "Phase A live execution is disabled at the runner boundary"
        )
    if (
        execution_mode in NON_SCIENTIFIC_MODES
        and eligible_for_scientific_analysis
    ):
        raise ValueError(
            f"{execution_mode} cannot be scientifically eligible"
        )
    if episode_count < 1:
        raise ValueError("At least one P1 episode is required")
    if execution_mode == LOCAL_MODEL_SHAKEDOWN and episode_count != 1:
        raise ValueError("Local shakedown is capped at exactly one episode")

    static = load_static_inputs(context)
    if root_seed != static.experiment["root_seed"]:
        raise ValueError("P1 root seed must match the frozen experiment")
    catalog = provider_catalog_entry(static, provider.provider_id)
    if not _mode_provider_valid(
        execution_mode,
        provider,
        catalog,
        static.experiment["phase_a_decoding"],
    ):
        raise ValueError(
            "Provider is not authorized for the requested execution mode"
        )
    if execution_mode == LOCAL_MODEL_SHAKEDOWN:
        if allow_local_shakedown is not True:
            raise PermissionError(
                "Explicit local shakedown authorization is required"
            )
        if not ollama_model_is_installed(provider.model_id):
            raise RuntimeError(
                "qwen3:8b is not present in the local Ollama inventory"
            )
    selected = (
        list(task_instance_ids)
        if task_instance_ids is not None
        else list(static.task_ids[:episode_count])
    )
    if len(selected) != episode_count:
        raise ValueError("Task count and episode count differ")
    if len(selected) != len(set(selected)):
        raise ValueError("P1 tasks must be unique within a run")
    if any(task_id not in static.materials for task_id in selected):
        raise ValueError("Unknown P1 task instance")

    actual_run_id = run_id or new_run_id(RUN_PREFIX[execution_mode])
    validate_run_id(actual_run_id)
    if not actual_run_id.startswith(RUN_PREFIX[execution_mode]):
        raise ValueError("Run ID prefix does not match execution mode")
    context.outputs.ensure()
    artifact_paths = context.outputs.artifact_paths(actual_run_id)
    existing = [
        path for path in artifact_paths.values() if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "P1 run ID is not unique; one or more artifacts already exist"
        )

    budget = frozen_budget_for_mode(
        execution_mode,
        episode_count,
        static.experiment,
    )
    if episode_count > budget["max_episodes"]:
        raise ValueError("Episode request exceeds the frozen budget")
    provenance = capture_provenance(context)
    provider_summary = provider.manifest_summary()
    provider_summary["endpoint_origin_hash"] = stable_hash(
        getattr(provider, "endpoint", None)
    )
    provider_summary["provider_config_hash"] = provenance[
        "provider_config_hash"
    ]
    provider_summary["credential_value_recorded"] = False
    provider_summary["hidden_reasoning_requested"] = False
    provider_summary["hidden_reasoning_logged"] = False
    provider_summary["local_model_installed_verified"] = (
        execution_mode == LOCAL_MODEL_SHAKEDOWN
    )
    output_files = {
        name: _relative_output(path, context)
        for name, path in artifact_paths.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": actual_run_id,
        "phase": PHASE,
        "condition": CONDITION,
        "risk_seed_present": RISK_SEED_PRESENT,
        "intervention_applied": INTERVENTION_APPLIED,
        "execution_mode": execution_mode,
        "eligible_for_scientific_analysis": (
            eligible_for_scientific_analysis
        ),
        "scientific_gate": SCIENTIFIC_GATE,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "root_seed": root_seed,
        "task_instance_ids": selected,
        "provider": provider_summary,
        "budget": budget,
        "output_files": output_files,
        "artifact_hashes": {},
        "provenance": provenance,
        "validation": None,
    }
    # Safety invariant: manifest with budget/model evidence exists first.
    write_json(artifact_paths["manifest"], manifest)

    recorder = EventRecorder(
        run_id=actual_run_id,
        execution_mode=execution_mode,
        eligible_for_scientific_analysis=(
            eligible_for_scientific_analysis
        ),
        path=artifact_paths["events"],
        data_root=context.outputs.data_root,
    )
    outcomes: list[dict[str, Any]] = []
    try:
        for episode_number, task_id in enumerate(selected, start=1):
            outcomes.append(
                execute_episode(
                    run_id=actual_run_id,
                    episode_number=episode_number,
                    task_instance_id=task_id,
                    material=static.materials[task_id],
                    internal_record=static.internal_records[task_id],
                    expected_report=static.expected_reports[task_id][
                        "report"
                    ],
                    static=static,
                    provider=provider,
                    recorder=recorder,
                )
            )
        write_jsonl(artifact_paths["outcomes"], outcomes)
        replay = replay_run(
            actual_run_id,
            context=context,
            write_output=True,
        )
        write_text(
            artifact_paths["run_report"],
            render_run_report(
                manifest=manifest,
                outcomes=outcomes,
                validation_passed=None,
            ),
        )
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["artifact_hashes"] = {
            name: sha256_file(artifact_paths[name])
            for name in (
                "events",
                "outcomes",
                "replay",
                "run_report",
            )
        }
        write_json(artifact_paths["manifest"], manifest)

        preliminary = _validate_run_preliminary(
            actual_run_id,
            context=context,
        )
        write_text(
            artifact_paths["run_report"],
            render_run_report(
                manifest=manifest,
                outcomes=outcomes,
                validation_passed=preliminary["passed"],
            ),
        )
        manifest["artifact_hashes"]["run_report"] = sha256_file(
            artifact_paths["run_report"]
        )
        manifest["validation"] = {
            "passed": preliminary["passed"],
            "check_count": len(preliminary["checks"]),
            "error_count": len(preliminary["errors"]),
        }
        write_json(artifact_paths["manifest"], manifest)
        validation = validate_run(
            actual_run_id,
            context=context,
            write_outputs=True,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["failure"] = {"error_type": type(exc).__name__}
        write_json(artifact_paths["manifest"], manifest)
        raise

    metrics = aggregate_metrics(outcomes)
    return {
        "passed": validation["passed"] and replay["passed"],
        "run_id": actual_run_id,
        "execution_mode": execution_mode,
        "eligible_for_scientific_analysis": (
            eligible_for_scientific_analysis
        ),
        "scientific_gate": SCIENTIFIC_GATE,
        "validation_passed": validation["passed"],
        "replay_passed": replay["passed"],
        "live_provider_calls": replay["live_provider_calls"],
        "metrics": metrics,
        "artifact_paths": output_files,
    }
