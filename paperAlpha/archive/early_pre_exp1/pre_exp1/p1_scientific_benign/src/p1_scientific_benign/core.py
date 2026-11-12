"""Frozen P1 pilot inputs, paths, labels, evidence and provenance."""

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

from p1_benign.configuration import StaticInputs, load_static_inputs as load_phase_a_inputs
from p1_benign.core import (
    EXPECTED_EDGES,
    EXPECTED_ROLE_IDS,
    FORBIDDEN_INTERNAL_FIELDS,
    REQUIRED_PUBLIC_FIELDS,
    expected_report_from_internal,
)
from p1_benign.event_contracts import validate_event_payload_contract
from pre_exp1.state import stable_hash as _stable_hash


SCHEMA_VERSION = "1.0.0"
PHASE = "P1_SCIENTIFIC_BENIGN"
CONDITION = "benign_baseline"
DATA_ROLE_DRY = "engineering_dry_run"
DATA_ROLE_PILOT = "p1_gate_pilot"
SCIENTIFIC_GATE_DRY = "NOT_STARTED"
SCIENTIFIC_GATE_PILOT = "P1_AWAITING_MAIN_AGENT_ACCEPTANCE"
SCIENTIFIC_SCOPE_DRY = "ENGINEERING_DRY_RUN_ONLY"
SCIENTIFIC_SCOPE_PILOT = "P1_GATE_PILOT_ONLY"
EXECUTION_DRY = "test_double"
EXECUTION_REAL = "local_model_pilot"
MODEL_ID = "qwen3:8b"
PROVIDER_ID = "ollama_local"
ENDPOINT = "http://127.0.0.1:11434"
ENDPOINT_IDENTIFIER = "loopback_ollama"
TASK_ORDER = tuple(f"P1-TASK-{i:03d}" for i in range(1, 21))
REQUIRED_FIELDS = tuple(REQUIRED_PUBLIC_FIELDS)
MODEL_DECODING = {
    "temperature": 0,
    "sampling_seed": 20260731,
    "max_output_tokens": 256,
    "thinking": False,
    "frozen": True,
}
BUDGET = {
    "episodes": 20,
    "maximum_role_calls_per_episode": 9,
    "maximum_model_calls": 180,
    "concurrency": 1,
    "retry_per_call": 0,
    "per_call_timeout_seconds": 90,
    "maximum_wall_time_minutes": 60,
    "remote_calls": 0,
    "model_downloads": 0,
    "extra_warmup_calls": 0,
}
QUALIFICATION_RUN_ID = "P1-BENIGN-QUAL-QWEN3-20260731T121840716022Z"
QUALIFICATION_METADATA_HASH = "4d0f241b66d0e67f72e6ffbae82796ad7441c5df87bd27a2dee28c1ff480108f"
OLLAMA_VERSION = "ollama version is 0.24.0"
ENDPOINT_ORIGIN_HASH = _stable_hash(ENDPOINT)

RUN_ID_PATTERN = re.compile(r"^P1-BENIGN-PILOT-(?:DRY|QWEN3)-[A-Za-z0-9_.:-]+$")
PACKAGE_DIR = Path(__file__).resolve().parent
SCIENTIFIC_ROOT = PACKAGE_DIR.parents[1]
PRE_EXP_ROOT = PACKAGE_DIR.parents[2]
PAPER_ALPHA_ROOT = PACKAGE_DIR.parents[3]
DATA_ROOT = PAPER_ALPHA_ROOT / "data"
STATIC_ROOT = DATA_ROOT / "shared" / "p1_benign"
OUTPUT_ROOT = DATA_ROOT / "pre_exp1" / "p1_scientific_benign"
PHASE_A_ROOT = PRE_EXP_ROOT / "p1_benign"
QUALIFICATION_ROOT = PRE_EXP_ROOT / "p1_model_qualification"
QUALIFICATION_MANIFEST = DATA_ROOT / "pre_exp1" / "p1_model_qualification" / "manifests" / f"manifest_{QUALIFICATION_RUN_ID}.json"
SHARED_SCHEMA_RELATIVE_PATHS = (
    "data/shared/p1_benign/schemas/event.schema.json",
    "data/shared/p1_benign/schemas/manifest.schema.json",
    "data/shared/p1_benign/schemas/outcome.schema.json",
    "data/shared/p1_benign/schemas/replay.schema.json",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    canonical_json(value)
    return _stable_hash(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id(prefix: str) -> str:
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("scientific pilot run ID has an invalid prefix")
    return run_id


def _reject_constant(value: str) -> None:
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
        return json.load(handle, parse_constant=_reject_constant, parse_float=_strict_float, object_pairs_hook=_strict_object)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_constant, parse_float=_strict_float, object_pairs_hook=_strict_object)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            records.append(value)
    return records


def require_generated_path(path: Path, data_root: Path) -> Path:
    resolved, root = path.resolve(), data_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("generated path escapes scientific data root") from exc
    return resolved


def _atomic_write(path: Path, text: str, data_root: Path) -> None:
    require_generated_path(path, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any, data_root: Path) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", data_root)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], data_root: Path) -> None:
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
        return cls(data_root.resolve(), root, root / "raw", root / "interim", root / "processed", root / "manifests", root / "reports")

    def ensure(self) -> None:
        for path in (self.raw, self.interim, self.processed, self.manifests, self.reports):
            require_generated_path(path, self.data_root)
            path.mkdir(parents=True, exist_ok=True)

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
            "preflight_report": self.reports / f"preflight_{run_id}.json",
            "p1_gate_report": self.reports / f"p1_gate_{run_id}.md",
        }
        for path in paths.values():
            require_generated_path(path, self.data_root)
        return paths


@dataclass(frozen=True)
class ScientificContext:
    paper_alpha_root: Path
    scientific_root: Path
    source_root: Path
    config_root: Path
    phase_a_root: Path
    static_root: Path
    outputs: OutputPaths

    @classmethod
    def discover(cls) -> "ScientificContext":
        return cls(PAPER_ALPHA_ROOT, SCIENTIFIC_ROOT, SCIENTIFIC_ROOT / "src", SCIENTIFIC_ROOT / "configs", PHASE_A_ROOT, STATIC_ROOT, OutputPaths.from_root(OUTPUT_ROOT, data_root=DATA_ROOT))

    def with_test_output(self, root: Path) -> "ScientificContext":
        data_root = root.resolve() / "data"
        return replace(self, outputs=OutputPaths.from_root(data_root / "pre_exp1" / "p1_scientific_benign" / "engineering_dry", data_root=data_root))

    def for_data_role(self, data_role: str) -> "ScientificContext":
        if data_role not in {DATA_ROLE_DRY, DATA_ROLE_PILOT}:
            raise ValueError("unknown scientific data role")
        root = self.outputs.data_root / "pre_exp1" / "p1_scientific_benign" / ("engineering_dry" if data_role == DATA_ROLE_DRY else "pilot")
        return replace(self, outputs=OutputPaths.from_root(root, data_root=self.outputs.data_root))


DEFAULT_CONTEXT = ScientificContext.discover()


def _contains(value: Any, needle: str) -> bool:
    if isinstance(value, dict):
        return any(_contains(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, needle) for item in value)
    return value == needle


def load_static_inputs(context: ScientificContext = DEFAULT_CONTEXT) -> StaticInputs:
    """Read the frozen Phase A fixture tree without writing to it."""
    phase_context = __import__("p1_benign.core", fromlist=["DEFAULT_CONTEXT"]).DEFAULT_CONTEXT
    static = load_phase_a_inputs(phase_context)
    if static.task_ids != TASK_ORDER:
        raise ValueError("scientific P1 requires exactly P1-TASK-001..020 in order")
    canaries = {row["internal_canary"] for row in static.internal_records.values()}
    if len(canaries) != 20:
        raise ValueError("scientific P1 requires twenty unique canaries")
    for task_id in TASK_ORDER:
        material = static.materials[task_id]
        expected = static.expected_reports[task_id]["report"]
        canary = static.internal_records[task_id]["internal_canary"]
        if _contains(material, canary) or _contains(expected, canary):
            raise ValueError("canary appears in public fixture")
        if expected != expected_report_from_internal(static.internal_records[task_id]):
            raise ValueError("expected report does not match internal fixture")
        if set(expected) != set(REQUIRED_PUBLIC_FIELDS):
            raise ValueError("public field allowlist drift")
        if any(field in material for field in FORBIDDEN_INTERNAL_FIELDS):
            raise ValueError("internal field leaked into public fixture")
    return static


def labels(data_role: str) -> dict[str, Any]:
    if data_role == DATA_ROLE_DRY:
        return {"data_role": data_role, "eligible_for_scientific_analysis": False, "scientific_analysis_scope": SCIENTIFIC_SCOPE_DRY, "eligible_for_p1_gate_analysis": False, "eligible_for_confirmatory_analysis": False, "eligible_for_causal_effect_analysis": False, "scientific_gate": SCIENTIFIC_GATE_DRY}
    if data_role == DATA_ROLE_PILOT:
        return {"data_role": data_role, "eligible_for_scientific_analysis": True, "scientific_analysis_scope": SCIENTIFIC_SCOPE_PILOT, "eligible_for_p1_gate_analysis": True, "eligible_for_confirmatory_analysis": False, "eligible_for_causal_effect_analysis": False, "scientific_gate": SCIENTIFIC_GATE_PILOT}
    raise ValueError("unknown scientific data role")


class ScientificEventRecorder:
    """Append-only recorder with explicit scientific-layer labels."""

    def __init__(self, *, run_id: str, execution_mode: str, data_role: str, path: Path | None = None, data_root: Path | None = None) -> None:
        validate_run_id(run_id)
        if execution_mode not in {EXECUTION_DRY, EXECUTION_REAL}:
            raise ValueError("invalid scientific execution mode")
        self.run_id, self.execution_mode, self.data_role = run_id, execution_mode, data_role
        self.eligible = data_role == DATA_ROLE_PILOT
        self.path, self.data_root, self.events, self._sequence = path, data_root, [], 0
        if path is not None:
            if data_root is None:
                raise ValueError("data root required")
            require_generated_path(path, data_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("x", encoding="utf-8", newline="").close()

    def emit(self, *, episode_id: str, task_instance_id: str, event_type: str, role_id: str | None = None, source_agent: str | None = None, target_agent: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = copy.deepcopy(payload or {})
        errors = validate_event_payload_contract(event_type, body, role_id=role_id, source_agent=source_agent, target_agent=target_agent)
        if errors:
            raise ValueError("invalid event payload: " + "; ".join(errors))
        if event_type == "episode_finished" and isinstance(body.get("outcome"), dict):
            body["outcome"].update({"phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, **labels(self.data_role)})
        self._sequence += 1
        event = {"schema_version": SCHEMA_VERSION, "event_id": f"P1S-EVT-{stable_hash([self.run_id, self._sequence, event_type, stable_hash(body)])[:24]}", "run_id": self.run_id, "episode_id": episode_id, "task_instance_id": task_instance_id, "sequence": self._sequence, "recorded_at": utc_now(), "phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, "execution_mode": self.execution_mode, **labels(self.data_role), "event_type": event_type, "role_id": role_id, "source_agent": source_agent, "target_agent": target_agent, "payload": body, "payload_hash": stable_hash(body)}
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return copy.deepcopy(event)


def normalize_outcome(outcome: dict[str, Any], execution_mode: str, data_role: str) -> dict[str, Any]:
    value = copy.deepcopy(outcome)
    value.update({"phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, "execution_mode": execution_mode, **labels(data_role)})
    return value


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _hash_map(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {_relative(path, root): sha256_file(path) for path in sorted(paths)}


def _tree_hash(mapping: dict[str, str]) -> str:
    return stable_hash(mapping)


def shared_schema_provenance(context: ScientificContext = DEFAULT_CONTEXT) -> dict[str, Any]:
    """Return the exact four shared schema files and their derived tree hash."""
    paths = [context.paper_alpha_root / Path(relative) for relative in SHARED_SCHEMA_RELATIVE_PATHS]
    if any(not path.is_file() for path in paths):
        raise ValueError("required shared schema file is missing")
    mapping = _hash_map(paths, context.paper_alpha_root)
    if tuple(mapping) != tuple(sorted(SHARED_SCHEMA_RELATIVE_PATHS)):
        raise ValueError("shared schema provenance set drifted")
    return {"shared_schema_files": mapping, "shared_schema_tree_hash": _tree_hash(mapping)}


def validate_shared_schema_provenance(provenance: dict[str, Any], context: ScientificContext = DEFAULT_CONTEXT) -> None:
    """Fail closed unless provenance names exactly the four frozen schema files."""
    expected = shared_schema_provenance(context)
    observed = provenance.get("shared_schema_files") if isinstance(provenance, dict) else None
    if not isinstance(observed, dict) or set(observed) != set(expected["shared_schema_files"]):
        raise ValueError("shared schema provenance set is not the required four-file set")
    if observed != expected["shared_schema_files"] or provenance.get("shared_schema_tree_hash") != expected["shared_schema_tree_hash"]:
        raise ValueError("shared schema provenance hash or tree hash mismatch")


def _env() -> dict[str, Any]:
    packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(f"{name}=={distribution.version}")
    return {"sys_executable": sys.executable, "python_version": sys.version.split()[0], "pytest_version": metadata.version("pytest"), "installed_distribution": {"name": "paperalpha-pre-exp1", "version": metadata.version("paperalpha-pre-exp1")}, "dependency_inventory": sorted(set(packages), key=str.lower)}


def capture_provenance(context: ScientificContext = DEFAULT_CONTEXT) -> dict[str, Any]:
    own = list((context.scientific_root / "src").rglob("*.py")) + list((context.scientific_root / "scripts").glob("*.py")) + list((context.scientific_root / "configs").glob("*.json"))
    reuse = list((context.phase_a_root / "src" / "p1_benign").glob("*.py")) + list((context.phase_a_root / "configs").glob("*.json")) + list((context.phase_a_root / "prompts").glob("*.txt"))
    reuse += list((PRE_EXP_ROOT / "src").glob("*.py")) + list((QUALIFICATION_ROOT / "src" / "p1_model_qualification").glob("*.py"))
    shared = list(context.static_root.glob("*.json"))
    schema_provenance = shared_schema_provenance(context)
    own_map = _hash_map(own, context.paper_alpha_root)
    reuse_map = _hash_map(reuse, context.paper_alpha_root)
    static_map = _hash_map(shared, context.paper_alpha_root)
    return {"environment": _env(), "scientific_execution_files": own_map, "scientific_execution_tree_hash": _tree_hash(own_map), "reused_upstream_files": reuse_map, "reused_upstream_tree_hash": _tree_hash(reuse_map), "p1_benign_static_files": static_map, "p1_benign_static_tree_hash": _tree_hash(static_map), **schema_provenance}


def route_context(run_id: str, *, context: ScientificContext = DEFAULT_CONTEXT) -> ScientificContext:
    """Resolve the only permitted source directory from the run ID, then verify its manifest."""
    validate_run_id(run_id)
    if run_id.startswith("P1-BENIGN-PILOT-QWEN3-"):
        data_role, execution_mode = DATA_ROLE_PILOT, EXECUTION_REAL
    elif run_id.startswith("P1-BENIGN-PILOT-DRY-"):
        data_role, execution_mode = DATA_ROLE_DRY, EXECUTION_DRY
    else:
        raise ValueError("unsupported scientific run prefix")
    routed = context.for_data_role(data_role)
    manifest_path = routed.outputs.artifact_paths(run_id)["manifest"]
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise ValueError("run manifest does not match routed run ID")
    if manifest.get("data_role") != data_role or manifest.get("execution_mode") != execution_mode:
        raise ValueError("run manifest data role or execution mode contradicts run prefix")
    if not _labels_match(manifest, data_role):
        raise ValueError("run manifest labels contradict routed data role")
    return routed


def _labels_match(value: dict[str, Any], data_role: str) -> bool:
    expected = {"phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, **labels(data_role)}
    return all(value.get(key) == expected_value for key, expected_value in expected.items())


def safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: operation failed"
