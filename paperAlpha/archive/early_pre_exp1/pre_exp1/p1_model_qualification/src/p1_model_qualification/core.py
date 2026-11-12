"""Frozen inputs, append-only evidence, paths, and provenance helpers."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from p1_benign.configuration import StaticInputs
from p1_benign.core import (
    EXPECTED_EDGES,
    EXPECTED_ROLE_IDS,
    FORBIDDEN_INTERNAL_FIELDS,
    REQUIRED_PUBLIC_FIELDS,
    expected_report_from_internal,
)
from p1_benign.event_contracts import validate_event_payload_contract
from pre_exp1.state import stable_hash as _p0_stable_hash


SCHEMA_VERSION = "1.0.0"
PHASE = "P1_MODEL_QUALIFICATION"
EXECUTION_MODE = "local_model_qualification"
TEST_DOUBLE_MODE = "test_double"
SCIENTIFIC_GATE = "NOT_STARTED"
CONDITION = "benign_baseline"
RISK_SEED_PRESENT = False
INTERVENTION_APPLIED = False
ELIGIBLE_FOR_SCIENTIFIC_ANALYSIS = False
CANDIDATE_MODELS = ("qwen3:8b", "ministral-3:8b")
TASK_ORDER = ("P1-TASK-901", "P1-TASK-902", "P1-TASK-903")
RUN_ID_PATTERN = re.compile(
    r"^P1-BENIGN-QUAL-(?:DRY|QWEN3|MINISTRAL3)-[A-Za-z0-9_.:-]+$"
)

PACKAGE_DIR = Path(__file__).resolve().parent
QUALIFICATION_ROOT = PACKAGE_DIR.parents[1]
PRE_EXP_ROOT = PACKAGE_DIR.parents[2]
PAPER_ALPHA_ROOT = PACKAGE_DIR.parents[3]
DATA_ROOT = PAPER_ALPHA_ROOT / "data"
STATIC_ROOT = DATA_ROOT / "shared" / "p1_model_qualification"
OUTPUT_ROOT = DATA_ROOT / "pre_exp1" / "p1_model_qualification"
PHASE_A_ROOT = PRE_EXP_ROOT / "p1_benign"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    canonical_json(value)
    return _p0_stable_hash(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id(prefix: str) -> str:
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("qualification run ID has an invalid prefix or suffix")
    return run_id


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_json_constant,
            parse_float=_strict_float,
            object_pairs_hook=_strict_object,
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                parse_constant=_reject_json_constant,
                parse_float=_strict_float,
                object_pairs_hook=_strict_object,
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            result.append(value)
    return result


def require_generated_path(path: Path, data_root: Path) -> Path:
    resolved = path.resolve()
    root = data_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("generated path escapes its data root") from exc
    return resolved


def _atomic_write(path: Path, text: str, data_root: Path) -> None:
    require_generated_path(path, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any, data_root: Path) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        data_root,
    )


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
    data_root: Path,
) -> None:
    _atomic_write(path, "".join(canonical_json(row) + "\n" for row in records), data_root)


def write_text(path: Path, text: str, data_root: Path) -> None:
    _atomic_write(path, text, data_root)


@dataclass(frozen=True)
class OutputPaths:
    data_root: Path
    root: Path
    raw: Path
    interim: Path
    processed: Path
    manifests: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path, *, data_root: Path) -> "OutputPaths":
        root = require_generated_path(root, data_root)
        return cls(
            data_root=data_root.resolve(),
            root=root,
            raw=root / "raw",
            interim=root / "interim",
            processed=root / "processed",
            manifests=root / "manifests",
            reports=root / "reports",
        )

    def ensure(self) -> None:
        for directory in (
            self.raw,
            self.interim,
            self.processed,
            self.manifests,
            self.reports,
        ):
            require_generated_path(directory, self.data_root)
            directory.mkdir(parents=True, exist_ok=True)

    def artifact_paths(self, run_id: str) -> dict[str, Path]:
        validate_run_id(run_id)
        paths = {
            "events": self.raw / f"events_{run_id}.jsonl",
            "outcomes": self.processed / f"outcomes_{run_id}.jsonl",
            "manifest": self.manifests / f"manifest_{run_id}.json",
            "replay": self.interim / f"replay_{run_id}.json",
            "validation": self.interim / f"validation_{run_id}.json",
            "run_report": self.reports / f"run_{run_id}.md",
            "validation_report": self.reports / f"validation_{run_id}.md",
        }
        for path in paths.values():
            require_generated_path(path, self.data_root)
        return paths


@dataclass(frozen=True)
class QualificationContext:
    paper_alpha_root: Path
    qualification_root: Path
    source_root: Path
    config_root: Path
    phase_a_root: Path
    static_root: Path
    outputs: OutputPaths

    @classmethod
    def discover(cls) -> "QualificationContext":
        return cls(
            paper_alpha_root=PAPER_ALPHA_ROOT,
            qualification_root=QUALIFICATION_ROOT,
            source_root=QUALIFICATION_ROOT / "src",
            config_root=QUALIFICATION_ROOT / "configs",
            phase_a_root=PHASE_A_ROOT,
            static_root=STATIC_ROOT,
            outputs=OutputPaths.from_root(OUTPUT_ROOT, data_root=DATA_ROOT),
        )

    def with_test_output(self, paper_alpha_root: Path) -> "QualificationContext":
        data_root = paper_alpha_root.resolve() / "data"
        return replace(
            self,
            outputs=OutputPaths.from_root(
                data_root / "pre_exp1" / "p1_model_qualification",
                data_root=data_root,
            ),
        )


DEFAULT_CONTEXT = QualificationContext.discover()


@dataclass(frozen=True)
class QualificationStatic:
    workflow: StaticInputs
    qualification: dict[str, Any]
    materials: dict[str, dict[str, Any]]
    internal_records: dict[str, dict[str, Any]]
    expected_reports: dict[str, dict[str, Any]]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self.qualification["task_order"])


def _records(path: Path) -> dict[str, dict[str, Any]]:
    document = read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError(f"{path.name} must contain a records array")
    result: dict[str, dict[str, Any]] = {}
    for record in document["records"]:
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} contains a non-object record")
        task_id = record.get("task_instance_id")
        if not isinstance(task_id, str) or task_id in result:
            raise ValueError(f"{path.name} has an invalid or duplicate task ID")
        result[task_id] = record
    return result


def _contains(value: Any, needle: str) -> bool:
    if isinstance(value, dict):
        return any(_contains(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, needle) for item in value)
    return value == needle


def validate_frozen_config(config: dict[str, Any]) -> None:
    expected = {
        "phase": PHASE,
        "condition": CONDITION,
        "risk_seed_present": False,
        "intervention_applied": False,
        "execution_mode": EXECUTION_MODE,
        "eligible_for_scientific_analysis": False,
        "scientific_gate": SCIENTIFIC_GATE,
        "candidate_models": list(CANDIDATE_MODELS),
        "task_order": list(TASK_ORDER),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen qualification config mismatch: {key}")
    budget = config.get("budget")
    if not isinstance(budget, dict) or budget != {
        "episodes_per_model": 3,
        "maximum_role_calls_per_episode": 9,
        "maximum_calls_per_model": 27,
        "maximum_calls_total": 54,
        "concurrency": 1,
        "retry_per_call": 0,
        "temperature": 0,
        "sampling_seed": 20260731,
        "max_output_tokens": 256,
        "thinking": False,
        "per_call_timeout_seconds": 90,
        "maximum_wall_time_per_model_minutes": 15,
        "maximum_wall_time_total_minutes": 30,
    }:
        raise ValueError("qualification budget differs from the frozen budget")
    if config.get("endpoint") != "http://127.0.0.1:11434":
        raise ValueError("qualification endpoint is not the frozen loopback origin")


def load_static_inputs(context: QualificationContext = DEFAULT_CONTEXT) -> QualificationStatic:
    qualification = read_json(context.config_root / "qualification.json")
    if not isinstance(qualification, dict):
        raise ValueError("qualification config must be an object")
    validate_frozen_config(qualification)
    materials = _records(context.static_root / "vendor_materials.json")
    internal_records = _records(context.static_root / "internal_records.json")
    expected_reports = _records(context.static_root / "expected_reports.json")
    if tuple(qualification["task_order"]) != TASK_ORDER:
        raise ValueError("qualification task order is not frozen")
    if set(materials) != set(TASK_ORDER) or set(internal_records) != set(TASK_ORDER) or set(expected_reports) != set(TASK_ORDER):
        raise ValueError("qualification static records do not exactly match the three tasks")
    phase_a_materials = _records(
        context.phase_a_root.parents[1] / "data" / "shared" / "p1_benign" / "vendor_materials.json"
    )
    phase_a_ids = set(phase_a_materials)
    phase_a_vendors = {row["vendor_id"] for row in phase_a_materials.values()}
    phase_a_names = {row["vendor_name"] for row in phase_a_materials.values()}
    canaries: set[str] = set()
    vendors: set[str] = set()
    for task_id in TASK_ORDER:
        material = materials[task_id]
        internal = internal_records[task_id]
        expected = expected_reports[task_id]
        if material.get("task_instance_id") != task_id or internal.get("task_instance_id") != task_id or expected.get("task_instance_id") != task_id:
            raise ValueError("qualification task ID mismatch")
        vendor_id = material.get("vendor_id")
        if not isinstance(vendor_id, str) or vendor_id in vendors or vendor_id in phase_a_vendors:
            raise ValueError("qualification vendor ID is invalid or overlaps P1 Phase A")
        if task_id in phase_a_ids or material.get("vendor_name") in phase_a_names:
            raise ValueError("qualification task overlaps a P1 scientific candidate")
        vendors.add(vendor_id)
        if internal.get("vendor_id") != vendor_id:
            raise ValueError("qualification internal vendor ID differs from public material")
        canary = internal.get("internal_canary")
        if not isinstance(canary, str) or canary in canaries:
            raise ValueError("qualification canaries must be unique")
        canaries.add(canary)
        report = expected.get("report")
        if not isinstance(report, dict) or set(report) != set(REQUIRED_PUBLIC_FIELDS):
            raise ValueError("qualification expected report violates the public allowlist")
        if report != expected_report_from_internal(internal):
            raise ValueError("qualification expected report violates the frozen rule")
        if any(field in material for field in FORBIDDEN_INTERNAL_FIELDS):
            raise ValueError("internal field appears in qualification public material")
        if _contains(material, canary) or _contains(expected, canary):
            raise ValueError("qualification canary escaped its internal fixture")
    agents = read_json(context.phase_a_root / "configs" / "agents.json")
    graph = read_json(context.phase_a_root / "configs" / "graph.json")
    roles = tuple(agent.get("role_id") for agent in agents.get("agents", []))
    if roles != EXPECTED_ROLE_IDS or tuple(graph.get("nodes", [])) != EXPECTED_ROLE_IDS:
        raise ValueError("reused eight-role workflow changed")
    edges = tuple((edge.get("source"), edge.get("target")) for edge in graph.get("edges", []))
    if edges != EXPECTED_EDGES or tuple(graph.get("execution_order", [])) != EXPECTED_ROLE_IDS:
        raise ValueError("reused workflow graph changed")
    prompts = {
        role: (context.phase_a_root / "prompts" / f"{role}.txt").read_text(encoding="utf-8")
        for role in EXPECTED_ROLE_IDS
    }
    if not all(text.strip() for text in prompts.values()):
        raise ValueError("a reused role prompt is empty")
    for canary in canaries:
        if any(canary in prompt for prompt in prompts.values()):
            raise ValueError("qualification canary appears in a reused prompt")
    workflow = StaticInputs(
        agents=agents,
        graph=graph,
        experiment={},
        providers={},
        task_template={},
        materials=materials,
        internal_records=internal_records,
        expected_reports=expected_reports,
        prompts=prompts,
    )
    return QualificationStatic(
        workflow=workflow,
        qualification=qualification,
        materials=materials,
        internal_records=internal_records,
        expected_reports=expected_reports,
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _hash_map(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {_relative(path, root): sha256_file(path) for path in sorted(paths)}


def _tree_hash(mapping: dict[str, str]) -> str:
    return stable_hash(mapping)


def own_execution_paths(context: QualificationContext) -> tuple[Path, ...]:
    paths = list((context.qualification_root / "src").rglob("*.py"))
    paths.extend((context.qualification_root / "scripts").glob("*.py"))
    paths.extend((context.qualification_root / "configs").glob("*.json"))
    return tuple(sorted(paths))


def reused_phase_a_paths(context: QualificationContext) -> tuple[Path, ...]:
    paths = list((context.phase_a_root / "src" / "p1_benign").glob("*.py"))
    paths.extend((context.phase_a_root / "configs").glob("*.json"))
    paths.extend((context.phase_a_root / "prompts").glob("*.txt"))
    return tuple(sorted(paths))


def qualification_static_paths(context: QualificationContext) -> tuple[Path, ...]:
    return tuple(sorted(context.static_root.glob("*.json")))


def capture_environment() -> dict[str, Any]:
    installed: list[str] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed.append(f"{name}=={distribution.version}")
    return {
        "sys_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pytest_version": metadata.version("pytest"),
        "installed_distribution": {
            "name": "paperalpha-pre-exp1",
            "version": metadata.version("paperalpha-pre-exp1"),
        },
        "dependency_inventory": sorted(set(installed), key=str.lower),
    }


def capture_provenance(context: QualificationContext = DEFAULT_CONTEXT) -> dict[str, Any]:
    own = _hash_map(own_execution_paths(context), context.paper_alpha_root)
    reused = _hash_map(reused_phase_a_paths(context), context.paper_alpha_root)
    static = _hash_map(qualification_static_paths(context), context.paper_alpha_root)
    return {
        "environment": capture_environment(),
        "qualification_execution_files": own,
        "qualification_execution_tree_hash": _tree_hash(own),
        "reused_phase_a_files": reused,
        "reused_phase_a_tree_hash": _tree_hash(reused),
        "qualification_static_files": static,
        "qualification_static_tree_hash": _tree_hash(static),
    }


class QualificationEventRecorder:
    """Append-only event recorder compatible with the validated Phase A contracts."""

    def __init__(
        self,
        *,
        run_id: str,
        execution_mode: str,
        path: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        validate_run_id(run_id)
        if execution_mode not in {TEST_DOUBLE_MODE, EXECUTION_MODE}:
            raise ValueError("unsupported qualification execution mode")
        self.run_id = run_id
        self.execution_mode = execution_mode
        self.eligible = False
        self.path = path
        self.data_root = data_root
        self.events: list[dict[str, Any]] = []
        self._sequence = 0
        if path is not None:
            if data_root is None:
                raise ValueError("data root is required for file recording")
            require_generated_path(path, data_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n"):
                pass

    def emit(
        self,
        *,
        episode_id: str,
        task_instance_id: str,
        event_type: str,
        role_id: str | None = None,
        source_agent: str | None = None,
        target_agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = copy.deepcopy(payload or {})
        errors = validate_event_payload_contract(
            event_type,
            body,
            role_id=role_id,
            source_agent=source_agent,
            target_agent=target_agent,
        )
        if errors:
            raise ValueError("invalid event payload: " + "; ".join(errors))
        # Phase A's event contract validates the shared workflow outcome before
        # its local phase label is adapted to this new qualification namespace.
        if event_type == "episode_finished" and isinstance(body.get("outcome"), dict):
            body["outcome"]["phase"] = PHASE
        self._sequence += 1
        identity = {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "sequence": self._sequence,
            "event_type": event_type,
            "payload_hash": stable_hash(body),
        }
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"P1Q-EVT-{stable_hash(identity)[:24]}",
            "run_id": self.run_id,
            "episode_id": episode_id,
            "task_instance_id": task_instance_id,
            "sequence": self._sequence,
            "recorded_at": utc_now(),
            "phase": PHASE,
            "condition": CONDITION,
            "risk_seed_present": False,
            "intervention_applied": False,
            "execution_mode": self.execution_mode,
            "eligible_for_scientific_analysis": False,
            "event_type": event_type,
            "role_id": role_id,
            "source_agent": source_agent,
            "target_agent": target_agent,
            "payload": body,
            "payload_hash": stable_hash(body),
        }
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as out:
                out.write(canonical_json(event) + "\n")
                out.flush()
                os.fsync(out.fileno())
        return copy.deepcopy(event)


def normalize_outcome(outcome: dict[str, Any], execution_mode: str) -> dict[str, Any]:
    value = copy.deepcopy(outcome)
    value["phase"] = PHASE
    value["execution_mode"] = execution_mode
    value["eligible_for_scientific_analysis"] = False
    value["condition"] = CONDITION
    value["risk_seed_present"] = False
    value["intervention_applied"] = False
    return value


def safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: operation failed"
