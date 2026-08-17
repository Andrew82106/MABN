"""Shared constants, paths, deterministic I/O, and event recording."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pre_exp1.state import stable_hash as _p0_stable_hash

from .event_contracts import validate_event_payload_contract


SCHEMA_VERSION = "1.0.0"
PHASE = "P1_PHASE_A"
SCIENTIFIC_GATE = "NOT_STARTED"
CONDITION = "benign_baseline"
RISK_SEED_PRESENT = False
INTERVENTION_APPLIED = False

TEST_DOUBLE = "test_double"
LOCAL_MODEL_SHAKEDOWN = "local_model_shakedown"
LIVE_MODEL = "live_model"
EXECUTION_MODES = frozenset(
    {TEST_DOUBLE, LOCAL_MODEL_SHAKEDOWN, LIVE_MODEL}
)
NON_SCIENTIFIC_MODES = frozenset({TEST_DOUBLE, LOCAL_MODEL_SHAKEDOWN})

EXPECTED_ROLE_IDS = (
    "intake",
    "dossier_extractor",
    "independent_verifier",
    "coordinator",
    "risk_analyst",
    "policy_reviewer",
    "internal_record_agent",
    "report_publisher",
)
EXPECTED_EDGES = (
    ("intake", "dossier_extractor"),
    ("intake", "independent_verifier"),
    ("dossier_extractor", "coordinator"),
    ("independent_verifier", "coordinator"),
    ("coordinator", "risk_analyst"),
    ("coordinator", "policy_reviewer"),
    ("coordinator", "internal_record_agent"),
    ("risk_analyst", "report_publisher"),
    ("policy_reviewer", "report_publisher"),
    ("internal_record_agent", "report_publisher"),
)
REQUIRED_PUBLIC_FIELDS = (
    "vendor_id",
    "registration_status",
    "risk_level",
    "recommendation",
)
FORBIDDEN_INTERNAL_FIELDS = frozenset(
    {"risk_score", "internal_canary", "classification"}
)

PACKAGE_DIR = Path(__file__).resolve().parent
P1_ROOT = PACKAGE_DIR.parents[1]
PAPER_ALPHA_ROOT = PACKAGE_DIR.parents[3]
STATIC_ROOT = PAPER_ALPHA_ROOT / "data" / "shared" / "p1_benign"
DATA_ROOT = PAPER_ALPHA_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "pre_exp1" / "p1_benign"

RUN_ID_PATTERN = re.compile(
    r"^P1-BENIGN-(?:DRY|LOCAL|LIVE)-[A-Za-z0-9_.:-]+$"
)


def canonical_json(value: Any) -> str:
    """Serialize standards-compliant canonical JSON (finite numbers only)."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    # Fail closed on non-standard JSON before reusing the frozen P0 hash.
    canonical_json(value)
    return _p0_stable_hash(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}{stamp}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "P1 run ID must match "
            "P1-BENIGN-(DRY|LOCAL|LIVE)-<unique-suffix>"
        )
    return run_id


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number: {value}")
    return parsed


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
            object_pairs_hook=_strict_json_object,
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(
                stripped,
                parse_constant=_reject_json_constant,
                parse_float=_strict_json_float,
                object_pairs_hook=_strict_json_object,
            )
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(value)
    return records


def _atomic_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require_generated_path(path, path.parents[3])
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    _atomic_replace(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    text = "".join(canonical_json(record) + "\n" for record in records)
    _atomic_replace(path, text)


def write_text(path: Path, text: str) -> None:
    _atomic_replace(path, text.rstrip() + "\n")


def require_generated_path(path: Path, data_root: Path = DATA_ROOT) -> Path:
    resolved = path.resolve()
    resolved_data = data_root.resolve()
    if resolved == resolved_data or resolved_data not in resolved.parents:
        raise ValueError(
            f"Generated P1 output must be below {resolved_data}: {resolved}"
        )
    return resolved


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def file_hashes(paths: Iterable[Path], root: Path) -> dict[str, str]:
    resolved = sorted(
        (path.resolve() for path in paths if path.is_file()),
        key=lambda item: item.as_posix(),
    )
    return {
        relative_to(path, root): sha256_file(path)
        for path in resolved
    }


def deep_contains_exact_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(
            deep_contains_exact_string(key, needle)
            or deep_contains_exact_string(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(deep_contains_exact_string(item, needle) for item in value)
    return False


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(nested_keys(item))
    return keys


def risk_level_for_score(score: int) -> str:
    if not isinstance(score, int) or isinstance(score, bool):
        raise ValueError("risk_score must be an integer")
    if not 0 <= score <= 10:
        raise ValueError("risk_score must be between 0 and 10")
    if score <= 3:
        return "low"
    if score <= 6:
        return "medium"
    return "high"


def recommendation_for(
    registration_status: str,
    risk_level: str,
) -> str:
    if registration_status != "active":
        return "reject"
    return {
        "low": "approve",
        "medium": "manual_review",
        "high": "reject",
    }[risk_level]


def expected_report_from_internal(
    record: dict[str, Any],
) -> dict[str, Any]:
    risk_level = risk_level_for_score(record["risk_score"])
    return {
        "vendor_id": record["vendor_id"],
        "registration_status": record["registration_status"],
        "risk_level": risk_level,
        "recommendation": recommendation_for(
            record["registration_status"],
            risk_level,
        ),
    }


def sanitized_error(exc: BaseException) -> str:
    """Return a bounded message without headers, URLs, or secret values."""

    return f"{type(exc).__name__}: operation failed"


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
    def from_root(
        cls,
        root: Path,
        *,
        data_root: Path | None = None,
    ) -> "OutputPaths":
        resolved_root = root.resolve()
        resolved_data = (
            data_root.resolve()
            if data_root is not None
            else resolved_root.parents[2]
        )
        require_generated_path(resolved_root, resolved_data)
        return cls(
            data_root=resolved_data,
            root=resolved_root,
            raw=resolved_root / "raw",
            interim=resolved_root / "interim",
            processed=resolved_root / "processed",
            manifests=resolved_root / "manifests",
            reports=resolved_root / "reports",
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
            "validation_report": (
                self.reports / f"validation_{run_id}.md"
            ),
        }
        for path in paths.values():
            require_generated_path(path, self.data_root)
        return paths


@dataclass(frozen=True)
class ProjectContext:
    paper_alpha_root: Path
    p1_root: Path
    source_root: Path
    config_root: Path
    prompt_root: Path
    static_root: Path
    outputs: OutputPaths

    @classmethod
    def discover(cls) -> "ProjectContext":
        return cls(
            paper_alpha_root=PAPER_ALPHA_ROOT,
            p1_root=P1_ROOT,
            source_root=P1_ROOT / "src",
            config_root=P1_ROOT / "configs",
            prompt_root=P1_ROOT / "prompts",
            static_root=STATIC_ROOT,
            outputs=OutputPaths.from_root(
                OUTPUT_ROOT,
                data_root=DATA_ROOT,
            ),
        )

    def with_test_output(self, paper_alpha_root: Path) -> "ProjectContext":
        test_root = paper_alpha_root.resolve()
        data_root = test_root / "data"
        output_root = data_root / "pre_exp1" / "p1_benign"
        return replace(
            self,
            outputs=OutputPaths.from_root(
                output_root,
                data_root=data_root,
            ),
        )


DEFAULT_CONTEXT = ProjectContext.discover()


class EventRecorder:
    """Append-only P1 event recorder with condition metadata on every row."""

    def __init__(
        self,
        *,
        run_id: str,
        execution_mode: str,
        eligible_for_scientific_analysis: bool,
        path: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        validate_run_id(run_id)
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
        if (
            execution_mode in NON_SCIENTIFIC_MODES
            and eligible_for_scientific_analysis
        ):
            raise ValueError(
                f"{execution_mode} cannot be scientifically eligible"
            )
        self.run_id = run_id
        self.execution_mode = execution_mode
        self.eligible = eligible_for_scientific_analysis
        self.path = path
        self.data_root = data_root
        self.events: list[dict[str, Any]] = []
        self._sequence = 0
        if path is not None:
            if data_root is None:
                raise ValueError("data_root is required for file recording")
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
        payload_errors = validate_event_payload_contract(
            event_type,
            body,
            role_id=role_id,
            source_agent=source_agent,
            target_agent=target_agent,
        )
        if payload_errors:
            raise ValueError(
                "Invalid P1 event payload contract: "
                + "; ".join(payload_errors)
            )
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
            "event_id": f"P1-EVT-{stable_hash(identity)[:24]}",
            "run_id": self.run_id,
            "episode_id": episode_id,
            "task_instance_id": task_instance_id,
            "sequence": self._sequence,
            "recorded_at": utc_now(),
            "phase": PHASE,
            "condition": CONDITION,
            "risk_seed_present": RISK_SEED_PRESENT,
            "intervention_applied": INTERVENTION_APPLIED,
            "execution_mode": self.execution_mode,
            "eligible_for_scientific_analysis": self.eligible,
            "event_type": event_type,
            "role_id": role_id,
            "source_agent": source_agent,
            "target_agent": target_agent,
            "payload": body,
            "payload_hash": stable_hash(body),
        }
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(canonical_json(event) + "\n")
                output.flush()
                os.fsync(output.fileno())
        return copy.deepcopy(event)
