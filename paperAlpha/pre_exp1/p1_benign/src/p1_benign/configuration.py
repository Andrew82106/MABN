"""Load and validate the frozen Phase A graph, tasks, and provider catalog."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    CONDITION,
    EXPECTED_EDGES,
    EXPECTED_ROLE_IDS,
    FORBIDDEN_INTERNAL_FIELDS,
    INTERVENTION_APPLIED,
    LIVE_MODEL,
    LOCAL_MODEL_SHAKEDOWN,
    REQUIRED_PUBLIC_FIELDS,
    RISK_SEED_PRESENT,
    TEST_DOUBLE,
    ProjectContext,
    expected_report_from_internal,
    read_json,
)


@dataclass(frozen=True)
class StaticInputs:
    agents: dict[str, Any]
    graph: dict[str, Any]
    experiment: dict[str, Any]
    providers: dict[str, Any]
    task_template: dict[str, Any]
    materials: dict[str, dict[str, Any]]
    internal_records: dict[str, dict[str, Any]]
    expected_reports: dict[str, dict[str, Any]]
    prompts: dict[str, str]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.materials))


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    document = read_json(path)
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a records array")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_instance_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path} has an invalid task_instance_id")
        if task_id in result:
            raise ValueError(f"Duplicate task_instance_id: {task_id}")
        result[task_id] = record
    return result


def _validate_graph(
    agents: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    configured_roles = tuple(
        agent["role_id"] for agent in agents.get("agents", [])
    )
    if configured_roles != EXPECTED_ROLE_IDS:
        raise ValueError(
            "P1 agents must exactly match the frozen eight-role order"
        )
    nodes = tuple(graph.get("nodes", []))
    if nodes != EXPECTED_ROLE_IDS:
        raise ValueError("P1 graph nodes do not match the eight roles")
    edges = tuple(
        (edge["source"], edge["target"])
        for edge in graph.get("edges", [])
    )
    if edges != EXPECTED_EDGES:
        raise ValueError("P1 graph edges differ from the frozen DAG")
    order = tuple(graph.get("execution_order", []))
    if order != EXPECTED_ROLE_IDS:
        raise ValueError("P1 execution order differs from the frozen order")
    position = {role_id: index for index, role_id in enumerate(order)}
    if any(position[source] >= position[target] for source, target in edges):
        raise ValueError("P1 graph is not acyclic in execution order")


def _validate_permissions(agents: dict[str, Any]) -> None:
    permission_map = {
        agent["role_id"]: set(agent.get("allowed_tools", []))
        for agent in agents["agents"]
    }
    for role_id, tools in permission_map.items():
        expected = set()
        if role_id == "internal_record_agent":
            expected = {"internal_db.read"}
        elif role_id == "report_publisher":
            expected = {"external_sink.publish"}
        if tools != expected:
            raise ValueError(
                f"Unexpected tool permissions for {role_id}: {sorted(tools)}"
            )


def _validate_tasks(
    materials: dict[str, dict[str, Any]],
    internals: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> None:
    task_ids = set(materials)
    if len(task_ids) != 20:
        raise ValueError("P1 requires exactly 20 deterministic tasks")
    if task_ids != set(internals) or task_ids != set(expected):
        raise ValueError("P1 material/internal/expected task IDs differ")
    vendor_ids: set[str] = set()
    canaries: set[str] = set()
    for task_id in sorted(task_ids):
        material = materials[task_id]
        internal = internals[task_id]
        expected_report = expected[task_id]["report"]
        vendor_id = material["vendor_id"]
        if internal["vendor_id"] != vendor_id:
            raise ValueError(f"Vendor ID mismatch for {task_id}")
        if expected_report["vendor_id"] != vendor_id:
            raise ValueError(f"Expected report vendor mismatch for {task_id}")
        if vendor_id in vendor_ids:
            raise ValueError(f"Duplicate vendor ID: {vendor_id}")
        vendor_ids.add(vendor_id)
        canary = internal["internal_canary"]
        if canary in canaries:
            raise ValueError(f"Duplicate internal Canary: {canary}")
        canaries.add(canary)
        if set(expected_report) != set(REQUIRED_PUBLIC_FIELDS):
            raise ValueError(
                f"Expected report fields are not frozen for {task_id}"
            )
        computed = expected_report_from_internal(internal)
        if expected_report != computed:
            raise ValueError(
                f"Expected report violates frozen rule for {task_id}"
            )
        if any(field in material for field in FORBIDDEN_INTERNAL_FIELDS):
            raise ValueError(
                f"Internal field appears in public material for {task_id}"
            )


def load_static_inputs(context: ProjectContext) -> StaticInputs:
    agents = read_json(context.config_root / "agents.json")
    graph = read_json(context.config_root / "graph.json")
    experiment = read_json(context.config_root / "experiment.json")
    providers = read_json(context.config_root / "providers.json")
    task_template = read_json(context.static_root / "task_template.json")
    materials = _load_records(context.static_root / "vendor_materials.json")
    internals = _load_records(context.static_root / "internal_records.json")
    expected = _load_records(context.static_root / "expected_reports.json")
    prompts = {
        role_id: (context.prompt_root / f"{role_id}.txt").read_text(
            encoding="utf-8"
        )
        for role_id in EXPECTED_ROLE_IDS
    }

    _validate_graph(agents, graph)
    _validate_permissions(agents)
    _validate_tasks(materials, internals, expected)
    condition = experiment.get("condition", {})
    if condition != {
        "condition": CONDITION,
        "risk_seed_present": RISK_SEED_PRESENT,
        "intervention_applied": INTERVENTION_APPLIED,
    }:
        raise ValueError("P1 condition must be the benign baseline")
    if tuple(experiment.get("required_public_fields", [])) != (
        REQUIRED_PUBLIC_FIELDS
    ):
        raise ValueError("Required public fields are not frozen")
    if set(experiment.get("forbidden_internal_fields", [])) != set(
        FORBIDDEN_INTERNAL_FIELDS
    ):
        raise ValueError("Forbidden internal fields are not frozen")

    return StaticInputs(
        agents=agents,
        graph=graph,
        experiment=experiment,
        providers=providers,
        task_template=task_template,
        materials=materials,
        internal_records=internals,
        expected_reports=expected,
        prompts=prompts,
    )


def provider_catalog_entry(
    static: StaticInputs,
    provider_id: str,
) -> dict[str, Any] | None:
    for entry in static.providers.get("providers", []):
        if entry.get("provider_id") == provider_id:
            return entry
    return None


def frozen_budget_for_mode(
    execution_mode: str,
    episode_count: int,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    """Build the sole permitted budget record for one execution mode."""

    if execution_mode not in {
        TEST_DOUBLE,
        LOCAL_MODEL_SHAKEDOWN,
        LIVE_MODEL,
    }:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    if episode_count < 1:
        raise ValueError("At least one episode is required")
    if execution_mode == LIVE_MODEL:
        budget = copy.deepcopy(experiment["live_budget"])
        budget["requested_episodes"] = episode_count
        return budget
    max_calls = 9
    max_episodes = (
        1 if execution_mode == LOCAL_MODEL_SHAKEDOWN else 20
    )
    return {
        "max_episodes": max_episodes,
        "requested_episodes": episode_count,
        "max_model_calls_per_episode": max_calls,
        "max_retries_per_call": 0,
        "max_total_model_calls": max_calls * episode_count,
        "max_output_tokens_per_call": 256,
        "max_total_output_tokens": 256 * max_calls * episode_count,
        "resume_policy": (
            "no_in_place_resume; rerun with a new unique run_id; "
            "tool idempotency keys are episode-scoped"
        ),
    }
