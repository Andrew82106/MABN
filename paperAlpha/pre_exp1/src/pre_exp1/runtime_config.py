"""Single validated runtime view over experiment, graph, Agent, and template inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import GraphConfig, GraphConfigError
from .models import TreatmentName
from .paths import DEFAULT_PATHS, ProjectPaths
from .templates import TemplateConfigError, TemplateRegistry


EXPECTED_P0_CONDITIONS = (
    ("normal_safe", "original", "normal_original"),
    ("dangerous_original", "original", "dangerous_original"),
    ("dangerous_safe", "safe", "dangerous_original"),
    ("dangerous_drop", "drop", "dangerous_original"),
)


class RuntimeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SmokeCondition:
    scenario: str
    treatment: TreatmentName
    input_template_id: str
    expected: dict[str, Any]


@dataclass(frozen=True)
class RuntimeConfig:
    paths: ProjectPaths
    experiment: dict[str, Any]
    graph: GraphConfig
    templates: TemplateRegistry
    treatments: dict[str, Any]
    conditions: tuple[SmokeCondition, ...]
    agents: dict[str, dict[str, Any]]
    reader_agent_id: str
    publisher_agent_id: str
    experiment_path: Path
    agents_path: Path
    graph_path: Path
    treatments_path: Path
    vendor_fixture_path: Path
    task_template_path: Path
    schema_paths: dict[str, Path]

    @classmethod
    def load(
        cls,
        paths: ProjectPaths = DEFAULT_PATHS,
    ) -> "RuntimeConfig":
        experiment_path = paths.config_dir / "experiment.json"
        experiment = _load_json(experiment_path)
        graph_path = _resolve_config_reference(
            paths.config_dir, experiment["graph_config"]
        )
        treatments_path = _resolve_config_reference(
            paths.config_dir, experiment["treatments"]
        )
        agents_path = _resolve_config_reference(
            paths.config_dir, experiment["agents_config"]
        )
        graph = GraphConfig.from_file(graph_path)
        treatments = _load_json(treatments_path)
        if treatments.get("gated_edge") != graph.gated_edge_id:
            raise RuntimeConfigError(
                "treatments.json gated_edge must match graph.json gated_edge"
            )
        template_path = _resolve_project_reference(
            paths.paper_alpha_root,
            experiment["message_templates"],
        )
        treatment_template_path = _resolve_config_reference(
            treatments_path.parent,
            treatments["template_source"],
        )
        if template_path != treatment_template_path:
            raise RuntimeConfigError(
                "experiment and treatment config must reference one template source"
            )
        templates = TemplateRegistry.from_file(template_path)
        for treatment_name in ("safe", "drop"):
            template_id = treatments["treatments"][treatment_name]["template_id"]
            if template_id != treatment_name:
                raise RuntimeConfigError(
                    f"{treatment_name} must reference template {treatment_name!r}"
                )
            templates.get(template_id)
        original = treatments["treatments"]["original"]
        if original.get("mode") != "passthrough":
            raise RuntimeConfigError("Original treatment must be passthrough")
        conditions = tuple(
            SmokeCondition(
                scenario=item["scenario"],
                treatment=item["treatment"],
                input_template_id=item["input_template_id"],
                expected=dict(item["expected"]),
            )
            for item in experiment["smoke_conditions"]
        )
        actual_conditions = tuple(
            (
                condition.scenario,
                condition.treatment,
                condition.input_template_id,
            )
            for condition in conditions
        )
        if actual_conditions != EXPECTED_P0_CONDITIONS:
            raise RuntimeConfigError(
                "experiment.json must define exactly the four fixed P0 conditions"
            )
        for condition in conditions:
            templates.get(condition.input_template_id)
        agents_config = _load_json(agents_path)
        agent_items = agents_config["agents"]
        agent_ids = [item["agent_id"] for item in agent_items]
        if len(agent_ids) != len(set(agent_ids)):
            raise RuntimeConfigError("Agent IDs must be unique")
        agents = {item["agent_id"]: item for item in agent_items}
        if set(agents) != set(graph.nodes):
            raise RuntimeConfigError(
                "agents.json and graph.json must define the same node IDs"
            )
        readers = [
            agent_id
            for agent_id, item in agents.items()
            if "get_internal_vendor_record" in item.get("allowed_tools", [])
        ]
        publishers = [
            agent_id
            for agent_id, item in agents.items()
            if "publish_public_report" in item.get("allowed_tools", [])
        ]
        if len(readers) != 1 or len(publishers) != 1:
            raise RuntimeConfigError(
                "Exactly one reader and one publisher Agent are required"
            )
        reader_index = graph.execution_order.index(readers[0])
        publisher_index = graph.execution_order.index(publishers[0])
        if reader_index == 0 or reader_index >= publisher_index:
            raise RuntimeConfigError(
                "The reader Agent must follow an upstream node and precede "
                "the publisher Agent"
            )
        gated_target_index = graph.execution_order.index(graph.gated_edge.target)
        if gated_target_index > reader_index:
            raise RuntimeConfigError(
                "The P0 gated edge must be on the request path to the reader"
            )
        graph.consecutive_edges()
        shared = experiment["shared_inputs"]
        vendor_fixture_path = _resolve_project_reference(
            paths.paper_alpha_root, shared["vendor_fixture"]
        )
        task_template_path = _resolve_project_reference(
            paths.paper_alpha_root, experiment["task_template"]
        )
        schema_paths = {
            name: _resolve_project_reference(paths.paper_alpha_root, relative)
            for name, relative in shared["schemas"].items()
        }
        expected_schemas = {"event", "outcome", "manifest", "replay"}
        if set(schema_paths) != expected_schemas:
            raise RuntimeConfigError(
                f"Expected schema entries {sorted(expected_schemas)}"
            )
        return cls(
            paths=paths,
            experiment=experiment,
            graph=graph,
            templates=templates,
            treatments=treatments,
            conditions=conditions,
            agents=agents,
            reader_agent_id=readers[0],
            publisher_agent_id=publishers[0],
            experiment_path=experiment_path,
            agents_path=agents_path,
            graph_path=graph_path,
            treatments_path=treatments_path,
            vendor_fixture_path=vendor_fixture_path,
            task_template_path=task_template_path,
            schema_paths=schema_paths,
        )

    @property
    def config_paths(self) -> tuple[Path, ...]:
        return (
            self.experiment_path,
            self.agents_path,
            self.graph_path,
            self.treatments_path,
        )

    @property
    def shared_input_paths(self) -> tuple[Path, ...]:
        return (
            self.vendor_fixture_path,
            self.task_template_path,
            self.templates.source_path,
            *(self.schema_paths[key] for key in sorted(self.schema_paths)),
        )

    def condition(self, scenario: str) -> SmokeCondition:
        matches = [item for item in self.conditions if item.scenario == scenario]
        if len(matches) != 1:
            raise RuntimeConfigError(f"Unknown P0 scenario {scenario!r}")
        return matches[0]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeConfigError(f"Unable to load runtime config {path}") from exc


def _resolve_config_reference(base: Path, reference: str) -> Path:
    return (base / reference).resolve()


def _resolve_project_reference(project_root: Path, reference: str) -> Path:
    path = (project_root / reference).resolve()
    root = project_root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeConfigError(f"Input path escapes paperAlpha: {reference}")
    return path
